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
    # Already at the tested version with the fix? Leave the files alone so the rebuild is incremental.
    if mode == "recommended" and r.out(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=src) == "inkwell":
        has = lambda c: r.run(["git", "merge-base", "--is-ancestor", c, "HEAD"], cwd=src, check=False) == 0
        clean = r.out(["git", "status", "--porcelain", "--untracked-files=no"], cwd=src) == ""
        if clean and r.out(["git", "rev-parse", "HEAD^1"], cwd=src) == PINNED_BASE and has(FIX_PR_COMMIT):
            step("llama.cpp source is up to date", None)
            return "fix-merged"
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
        root = rocm_root()
        env["PATH"] = "/opt/rocm/bin:" + (f"{root}/bin:" if root else "") + env["PATH"]
        if root:
            # llama.cpp's documented recipe: HIPCXX=<rocm clang>, HIP_PATH=<rocm root>
            env["HIP_PATH"] = env["ROCM_PATH"] = root
        clang = rocm_clang(root, env)
        if clang:
            env["HIPCXX"] = clang
        _, dl = rocm_hip_flags(root, clang)
        if dl:
            env["HIP_DEVICE_LIB_PATH"] = dl   # clang reads this too
    return env


ROCM_GLOB = "/opt/rocm*"
HIP_LANG = ["lib/cmake/hip-lang/hip-lang-config.cmake", "lib64/cmake/hip-lang/hip-lang-config.cmake",
            "lib/x86_64-linux-gnu/cmake/hip-lang/hip-lang-config.cmake"]


def rocm_root():
    """The ROCm prefix that actually holds hip-lang's CMake package. Newer ROCm releases split
    things into /opt/rocm/core-X.Y etc., where the compiler's folder is not where hip-lang lives."""
    cands = []
    hipconfig = shutil.which("hipconfig") or ("/opt/rocm/bin/hipconfig" if Path("/opt/rocm/bin/hipconfig").exists() else None)
    if hipconfig:
        out = subprocess.run([hipconfig, "-R"], capture_output=True, text=True).stdout.strip()
        if out:
            cands.append(out)
    cands += [os.environ.get("ROCM_PATH", ""), os.environ.get("HIP_PATH", ""), "/opt/rocm", "/usr"]
    cands += sorted(glob.glob(f"{ROCM_GLOB}"), reverse=True) + sorted(glob.glob(f"{ROCM_GLOB}/core-*"), reverse=True)
    for c in cands:
        if c and any(Path(c, h).exists() for h in HIP_LANG):
            return str(Path(c).resolve()) if c != "/usr" else "/usr"
    for hit in sorted(glob.glob(f"{ROCM_GLOB}/**/hip-lang-config.cmake", recursive=True)):
        return str(Path(hit).parents[3] if "x86_64" not in hit else Path(hit).parents[4])
    return None


def rocm_device_libs(root, clang):
    """Folder with ROCm's GPU bitcode (ocml.bc, ockl.bc…). Newer split installs keep it away from clang,
    which then fails with 'cannot find ROCm device library'."""
    cands = []
    if clang:
        llvm = Path(clang).parent.parent                      # …/lib/llvm
        cands += [llvm / "amdgcn/bitcode", llvm.parent / "amdgcn/bitcode", llvm.parent.parent / "amdgcn/bitcode"]
        cands += sorted(llvm.glob("lib/clang/*/lib/amdgcn/bitcode"), reverse=True)
    if root:
        cands += [Path(root, "amdgcn/bitcode"), Path(root, "lib/amdgcn/bitcode"), Path(root, "lib/llvm/amdgcn/bitcode")]
    for c in cands:
        if (c / "ocml.bc").exists():
            return str(c)
    hits = sorted(glob.glob(f"{ROCM_GLOB}/**/amdgcn/bitcode/ocml.bc", recursive=True), reverse=True)
    hits += sorted(glob.glob("/usr/lib/llvm-*/lib/clang/*/lib/amdgcn/bitcode/ocml.bc"), reverse=True)
    hits += sorted(glob.glob("/usr/lib/x86_64-linux-gnu/amdgcn/bitcode/ocml.bc"))
    return str(Path(hits[0]).parent) if hits else None


def rocm_hip_flags(root, clang):
    flags = []
    if root:
        flags.append(f"--rocm-path={root}")
    dl = rocm_device_libs(root, clang)
    if dl:
        flags.append(f"--rocm-device-lib-path={dl}")
    return flags, dl


def rocm_clang(root, env):
    cands = []
    if root:
        cands += [f"{root}/llvm/bin/clang", f"{root}/lib/llvm/bin/clang"]
    hipconfig = shutil.which("hipconfig", path=env["PATH"])
    if hipconfig:
        llvm = subprocess.run([hipconfig, "-l"], capture_output=True, text=True).stdout.strip()
        if llvm:
            cands.append(f"{llvm}/clang")
    cands += sorted(glob.glob(f"{ROCM_GLOB}/llvm/bin/clang") + glob.glob(f"{ROCM_GLOB}/*/lib/llvm/bin/clang")
                    + glob.glob(f"{ROCM_GLOB}/lib/llvm/bin/clang"), reverse=True)
    return next((c for c in cands if Path(c).exists()), None)


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
        root = rocm_root()
        if root:  # tell CMake's HIP language where hip-lang lives, independent of the compiler's folder
            f += [f"-DCMAKE_HIP_COMPILER_ROCM_ROOT={root}", f"-DCMAKE_PREFIX_PATH={root}", f"-DROCM_PATH={root}"]
        hip_flags, _ = rocm_hip_flags(root, rocm_clang(root, {"PATH": "/opt/rocm/bin:" + os.environ.get("PATH", "")}))
        if hip_flags:
            f.append("-DCMAKE_HIP_FLAGS=" + " ".join(hip_flags))
    return f


def build(r, src, backend, sysinfo, step, progress):
    src = Path(src)
    bdir = src / f"build-{backend}"
    env = build_env(backend)
    gen = ["-G", "Ninja"] if shutil.which("ninja") else []
    step("Configuring the build…", None)
    cfg = ["cmake", "-S", str(src), "-B", str(bdir), *gen, *cmake_flags(backend, sysinfo)]
    if r.run(cfg, env=env, check=False) != 0:
        # A failed configure caches the bad compiler detection; wipe the cache (not the objects) and retry once.
        r.on_log("! Configure failed; clearing the CMake cache and trying again…\n")
        (bdir / "CMakeCache.txt").unlink(missing_ok=True)
        shutil.rmtree(bdir / "CMakeFiles", ignore_errors=True)
        r.run(cfg, env=env)
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
