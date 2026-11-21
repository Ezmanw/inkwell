"""Detect GPUs, RAM, distro and installed build tools, and recommend a backend."""
import glob, os, re, shutil, subprocess
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class GPU:
    vendor: str          # nvidia | amd | intel | other
    name: str
    vram_mb: int = 0     # 0 = unknown / shared memory
    gfx: str = ""        # AMD gfx target, e.g. gfx1100 (needs rocminfo)
    cuda_max: str = ""   # NVIDIA: highest CUDA version the driver supports, e.g. "13.2"


@dataclass
class System:
    gpus: list = field(default_factory=list)
    ram_mb: int = 0
    distro_id: str = ""
    distro_like: str = ""
    distro_version: str = ""
    is_debian_like: bool = False

    @property
    def best_gpu(self):
        return max(self.gpus, key=lambda g: g.vram_mb, default=None)


def _run(cmd, timeout=5):
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout).stdout
    except Exception:
        return ""


def _nvidia():
    out = _run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"])
    cuda = re.search(r"CUDA Version:\s*([\d.]+)", _run(["nvidia-smi"]))
    gpus = []
    for line in out.strip().splitlines():
        try:
            name, mem = [x.strip() for x in line.rsplit(",", 1)]
            gpus.append(GPU("nvidia", name, int(float(mem)), cuda_max=cuda.group(1) if cuda else ""))
        except ValueError:
            pass
    return gpus


def _drm_gpus():
    """AMD/Intel (and NVIDIA without driver tools) from sysfs."""
    gpus = []
    names = {}
    for line in _run(["lspci", "-mm", "-nn"]).splitlines():
        m = re.match(r'(\S+) "(?:VGA|3D|Display)[^"]*" "([^"]*)" "([^"]*)"', line)
        if m:
            names[m.group(1)] = re.sub(r"\s*\[[0-9a-f]{4}\]", "", f"{m.group(2)} {m.group(3)}")
    for dev in sorted(glob.glob("/sys/class/drm/card[0-9]*/device")):
        if "-" in Path(dev).parent.name:
            continue
        try:
            vendor = Path(dev, "vendor").read_text().strip()
        except OSError:
            continue
        v = {"0x1002": "amd", "0x8086": "intel", "0x10de": "nvidia"}.get(vendor, "other")
        vram = 0
        try:
            vram = int(Path(dev, "mem_info_vram_total").read_text()) // (1024 * 1024)
        except (OSError, ValueError):
            pass
        slot = os.path.basename(os.path.realpath(dev))
        short = slot.split(":", 1)[1] if slot.count(":") == 2 else slot
        gpus.append(GPU(v, names.get(short, f"{v.upper()} GPU"), vram))
    return gpus


def _amd_gfx():
    out = _run(["rocminfo"], timeout=10)
    return sorted(set(re.findall(r"\b(gfx\d{3,4}[a-z]?)\b", out)))


def detect():
    s = System()
    gpus = _nvidia()
    for g in _drm_gpus():
        if g.vendor == "nvidia" and gpus:
            continue  # nvidia-smi already gave better info
        if g.vendor == "intel" and g.vram_mb == 0 and any(x.vendor in ("nvidia", "amd") for x in gpus + _drm_gpus()):
            continue  # skip iGPU when a real GPU exists
        gpus.append(g)
    gfx = _amd_gfx() if any(g.vendor == "amd" for g in gpus) else []
    for g in gpus:
        if g.vendor == "amd" and gfx:
            g.gfx = gfx[0]
    s.gpus = gpus
    try:
        m = re.search(r"MemTotal:\s+(\d+)", Path("/proc/meminfo").read_text())
        s.ram_mb = int(m.group(1)) // 1024
    except Exception:
        pass
    try:
        osr = dict(re.findall(r'^(\w+)="?([^"\n]*)"?$', Path("/etc/os-release").read_text(), re.M))
        s.distro_id, s.distro_like, s.distro_version = osr.get("ID", ""), osr.get("ID_LIKE", ""), osr.get("VERSION_ID", "")
    except Exception:
        pass
    s.is_debian_like = "debian" in (s.distro_id + " " + s.distro_like) or "ubuntu" in s.distro_like
    return s


# ---------------- backends ----------------
BACKENDS = {
    "cuda":   {"title": "CUDA", "for": "NVIDIA graphics cards", "desc": "Fastest on NVIDIA."},
    "vulkan": {"title": "Vulkan", "for": "Any graphics card (NVIDIA, AMD, Intel)",
               "desc": "Works almost everywhere. The safe pick for AMD and Intel."},
    "rocm":   {"title": "ROCm (HIP)", "for": "AMD graphics cards",
               "desc": "Can be faster than Vulkan on AMD, but trickier to set up. Experimental."},
    "cpu":    {"title": "CPU only", "for": "No usable graphics card", "desc": "Works on anything, but slow."},
}


def recommend_backend(s):
    g = s.best_gpu
    if not g:
        return "cpu"
    if g.vendor == "nvidia":
        return "cuda"
    return "vulkan"


def cuda_root():
    """Find an installed CUDA toolkit (nvcc)."""
    for p in [os.environ.get("CUDA_HOME", ""), "/usr/local/cuda"] + sorted(glob.glob("/usr/local/cuda-*"), reverse=True):
        if p and Path(p, "bin", "nvcc").exists():
            return p
    nv = shutil.which("nvcc")
    return str(Path(nv).resolve().parent.parent) if nv else None


def _have_header(name):
    return any(Path(d, name).exists() for d in ["/usr/include", "/usr/local/include"]) or bool(
        glob.glob(f"/usr/local/cuda*/targets/*/include/{name}"))


def missing_tools(backend):
    """Return list of (what, why) that are missing for this backend."""
    miss = []
    for exe, why in [("git", "download llama.cpp"), ("cmake", "configure the build"),
                     ("g++", "compile C++"), ("make", "build tools")]:
        if not shutil.which(exe):
            miss.append((exe, why))
    if backend == "cuda":
        root = cuda_root()
        if not root:
            miss.append(("nvcc", "CUDA compiler"))
        else:
            for h, why in [("cublas_v2.h", "cuBLAS headers"), ("nvml.h", "NVML headers")]:
                if not (glob.glob(f"{root}/include/{h}") or glob.glob(f"{root}/targets/*/include/{h}")):
                    miss.append((h, why))
    elif backend == "vulkan":
        if not shutil.which("glslc"):
            miss.append(("glslc", "Vulkan shader compiler"))
        if not _have_header("vulkan/vulkan.h"):
            miss.append(("vulkan headers", "Vulkan development files"))
        if not _have_header("spirv/unified1/spirv.hpp"):
            miss.append(("spirv headers", "SPIR-V headers"))
    elif backend == "rocm":
        if not shutil.which("hipcc") and not Path("/opt/rocm/bin/hipcc").exists():
            miss.append(("hipcc", "ROCm compiler"))
        if not shutil.which("rocminfo") and not Path("/opt/rocm/bin/rocminfo").exists():
            miss.append(("rocminfo", "detect your AMD GPU model"))
        if not (_have_header("hipblas/hipblas.h") or Path("/opt/rocm/include/hipblas/hipblas.h").exists()):
            miss.append(("hipBLAS", "ROCm math library"))
        from . import builder  # local import: builder imports this module
        root = builder.rocm_root()
        if not builder.rocm_device_libs(root, builder.rocm_clang(root, {"PATH": "/opt/rocm/bin:" + os.environ.get("PATH", "")})):
            miss.append(("ROCm device libraries", "GPU math bitcode (ocml.bc), package rocm-device-libs"))
    return miss
