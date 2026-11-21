"""Fetch + build llama.cpp for a chosen backend, and download model files with resume.
Everything reports progress through callbacks so the GUI can show it."""
import glob, os, re, shutil, subprocess, threading, time, urllib.request
from pathlib import Path

from . import hardware

LLAMA_GIT = "https://github.com/ggml-org/llama.cpp"
# Tested combination: llama.cpp master at this commit + PR #32740 (Inkling 2 compact flash-attention
# bias; cuts compute buffers from ~12 GB to ~0.7 GB at 128K context).
PINNED_BASE = "e6d624b9a4b95c6b52f5ca384d70b75581f9821a"
FIX_PR = 32740
FIX_PR_COMMIT = "ff91e4f84de442a954e2c73178b78630997df9f0"
GIT_ID = ["-c", "user.name=Inkwell", "-c", "user.email=inkwell@localhost"]


class Cancelled(Exception):
    pass


class Runner:
    """Runs commands, streams their output to on_log, and supports cancel."""

    def __init__(self, on_log):
        self.on_log = on_log
        self.cancel = threading.Event()
        self.proc = None

    def run(self, cmd, cwd=None, env=None, on_line=None, check=True):
        if self.cancel.is_set():
            raise Cancelled()
        self.on_log(f"$ {' '.join(cmd)}\n")
        self.proc = subprocess.Popen(cmd, cwd=cwd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                     stdin=subprocess.DEVNULL, text=True, bufsize=1, start_new_session=True)
        tail = []
        for line in self.proc.stdout:
            self.on_log(line)
            tail = (tail + [line])[-40:]
            if on_line:
                on_line(line)
            if self.cancel.is_set():
                self.proc.terminate()
        rc = self.proc.wait()
        if self.cancel.is_set():
            raise Cancelled()
        if check and rc != 0:
            raise RuntimeError(f"{cmd[0]} failed (exit {rc}):\n" + "".join(tail[-12:]))
        return rc

    def out(self, cmd, cwd=None):
        return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True).stdout.strip()

    def stop(self):
        self.cancel.set()
        if self.proc and self.proc.poll() is None:
            try:
                os.killpg(self.proc.pid, 15)
            except ProcessLookupError:
                pass


# ---------------- llama.cpp ----------------
def prepare_source(r, src, mode, step):
    """mode: 'recommended' (pinned + fix) or 'latest' (master + fix if still needed)."""
    src = Path(src)
    if not (src / ".git").exists():
        step("Downloading llama.cpp source…", None)
        src.parent.mkdir(parents=True, exist_ok=True)
        if src.exists() and any(src.iterdir()):
            raise RuntimeError(f"{src} exists and isn't a llama.cpp checkout. Pick an empty folder.")
        r.run(["git", "clone", "--filter=blob:none", "--progress", LLAMA_GIT, str(src)])
    step("Updating llama.cpp…", None)
    r.run(["git", "fetch", "--filter=blob:none", "origin", "master"], cwd=src)
    base = PINNED_BASE if mode == "recommended" else "FETCH_HEAD"
    if mode == "recommended":
        r.run(["git", "fetch", "--filter=blob:none", "origin", PINNED_BASE], cwd=src, check=False)
    r.run(["git", "checkout", "-f", "-B", "inkwell", base], cwd=src)
    step("Adding the Inkling VRAM fix…", None)
    rc = r.run(["git", "fetch", "--filter=blob:none", "origin", f"pull/{FIX_PR}/head"], cwd=src, check=False)
    target = "FETCH_HEAD"
    if mode == "recommended":
        if r.run(["git", "fetch", "--filter=blob:none", "origin", FIX_PR_COMMIT], cwd=src, check=False) == 0:
            target = FIX_PR_COMMIT
    if rc != 0 and target == "FETCH_HEAD":
        r.on_log("! Couldn't fetch the VRAM fix; building without it.\n")
        return "no-fix"
    if r.run(["git", "merge-base", "--is-ancestor", target, "HEAD"], cwd=src, check=False) == 0:
        return "fix-already-included"
    if r.run(["git", *GIT_ID, "merge", "--no-edit", target], cwd=src, check=False) != 0:
        r.run(["git", "merge", "--abort"], cwd=src, check=False)
        r.on_log("! The VRAM fix no longer merges cleanly; building without it.\n")
        return "no-fix"
    return "fix-merged"


def build_env(backend):
    env = dict(os.environ)
    if backend == "cuda":
        root = hardware.cuda_root()
        if root:
            env["PATH"] = f"{root}/bin:" + env["PATH"]
            env["CUDACXX"] = f"{root}/bin/nvcc"
    if backend == "rocm":
        env["PATH"] = "/opt/rocm/bin:" + env["PATH"]
        hipconfig = shutil.which("hipconfig", path=env["PATH"])
        if hipconfig:
            llvm = subprocess.run([hipconfig, "-l"], capture_output=True, text=True).stdout.strip()
            if llvm and Path(llvm, "clang").exists():
                env["HIPCXX"] = f"{llvm}/clang"
    return env


def cmake_flags(backend, sysinfo):
    f = ["-DCMAKE_BUILD_TYPE=Release", "-DLLAMA_CURL=OFF", "-DLLAMA_BUILD_TESTS=OFF",
         "-DLLAMA_BUILD_EXAMPLES=OFF", "-DGGML_NATIVE=ON"]
    if backend == "cuda":
        f += ["-DGGML_CUDA=ON", "-DCMAKE_CUDA_ARCHITECTURES=native"]
    elif backend == "vulkan":
        f += ["-DGGML_VULKAN=ON"]
    elif backend == "rocm":
        gfx = next((g.gfx for g in sysinfo.gpus if g.vendor == "amd" and g.gfx), "")
        f += ["-DGGML_HIP=ON"] + ([f"-DAMDGPU_TARGETS={gfx}", f"-DGPU_TARGETS={gfx}"] if gfx else [])
    return f


def build(r, src, backend, sysinfo, step, progress):
    src = Path(src)
    bdir = src / f"build-{backend}"
    env = build_env(backend)
    gen = ["-G", "Ninja"] if shutil.which("ninja") else []
    step("Configuring the build…", None)
    r.run(["cmake", "-S", str(src), "-B", str(bdir), *gen, *cmake_flags(backend, sysinfo)], env=env)
    ram_gb = max(1, sysinfo.ram_mb // 1024)
    jobs = max(1, min(os.cpu_count() or 2, ram_gb // 2))
    step(f"Compiling ({jobs} at a time; CUDA can take 10–20 minutes)…", 0.0)

    def on_line(line):
        m = re.match(r"\[(\d+)/(\d+)\]", line) or re.match(r"\[\s*(\d+)%\]", line)
        if m:
            frac = int(m.group(1)) / int(m.group(2)) if m.lastindex == 2 else int(m.group(1)) / 100
            progress(frac)
    r.run(["cmake", "--build", str(bdir), "-j", str(jobs), "--target", "llama-server"], env=env, on_line=on_line)
    exe = bdir / "bin" / "llama-server"
    if not exe.exists():
        raise RuntimeError("Build finished but llama-server wasn't produced.")
    return str(exe)


def binary_works(exe):
    try:
        out = subprocess.run([exe, "--version"], capture_output=True, text=True, timeout=20)
        return out.returncode == 0, (out.stdout + out.stderr).strip().splitlines()[-2:]
    except Exception as e:
        return False, [str(e)]


# ---------------- downloads ----------------
def download(r, url, dest, size, progress):
    """Resumable download to dest (via dest.part). progress(done, total, bytes_per_s)."""
    dest = Path(dest)
    if dest.exists() and (not size or dest.stat().st_size == size):
        progress(size or dest.stat().st_size, size or dest.stat().st_size, 0)
        return str(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    for attempt in range(6):
        have = part.stat().st_size if part.exists() else 0
        if size and have >= size:
            break
        req = urllib.request.Request(url, headers={"User-Agent": "Inkwell/0.1",
                                                   **({"Range": f"bytes={have}-"} if have else {})})
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                if have and resp.status != 206:
                    have = 0  # server ignored Range; start over
                total = size or (have + int(resp.headers.get("Content-Length") or 0))
                t0, done0 = time.time(), have
                with open(part, "ab" if have else "wb") as f:
                    while True:
                        if r.cancel.is_set():
                            raise Cancelled()
                        chunk = resp.read(1 << 20)
                        if not chunk:
                            break
                        f.write(chunk)
                        have += len(chunk)
                        dt = time.time() - t0
                        progress(have, total, (have - done0) / dt if dt > 0 else 0)
            if not size or have >= size:
                break
        except Cancelled:
            raise
        except Exception as e:
            r.on_log(f"! Download interrupted ({e}); retrying in {2 ** attempt}s…\n")
            time.sleep(2 ** attempt)
    if size and part.stat().st_size != size:
        raise RuntimeError(f"Download of {dest.name} incomplete ({part.stat().st_size} of {size} bytes).")
    part.replace(dest)
    return str(dest)


# ---------------- dependency installer (Debian/Ubuntu) ----------------
HELPER = Path(__file__).resolve().parent.parent / "helpers" / "install-deps.sh"


def deps_command(backend, sysinfo):
    """Command that installs missing build dependencies with a graphical password prompt."""
    cuda = ""
    if backend == "cuda":
        g = next((g for g in sysinfo.gpus if g.vendor == "nvidia"), None)
        cuda = g.cuda_max if g else ""
    return ["pkexec", str(HELPER), backend, cuda or "-", sysinfo.distro_id or "-", sysinfo.distro_version or "-"]
