"""Inkling 2 Mini GGUF catalog (unsloth, mainline-llama.cpp layout) and VRAM fit maths."""
import json, math, urllib.request

REPO = "unsloth/Inkling-2-Mini-GGUF"
FOLDER = "Mainline_Support"
HF = "https://huggingface.co"

# Fallback if Hugging Face can't be reached for the live list (sizes in bytes).
FALLBACK = {
    "UD-IQ1_S": 7764293056, "UD-IQ1_M": 8024442304, "UD-IQ2_XXS": 8486237632, "UD-Q2_K_XL": 9138716608,
    "UD-IQ3_XXS": 11663310784, "UD-Q3_K_XL": 13631779296, "UD-IQ4_XS": 15733756512,
    "UD-Q4_K_M": 16610544608, "UD-Q4_K_XL": 17629760736, "UD-Q8_K_XL": 22867452896,
}
MMPROJ = ("mmproj-F16.gguf", 639348160)

QUALITY = {
    "IQ1_S": "Rough", "IQ1_M": "Rough", "IQ2_XXS": "Basic", "Q2_K_XL": "Decent",
    "IQ3_XXS": "Good", "Q3_K_XL": "Good", "IQ4_XS": "Very good", "Q4_K_M": "Very good",
    "Q4_K_XL": "Very good", "Q8_K_XL": "Near-perfect",
}

# Layers 6-17 run three times per token ("loop" layers); experts of the other layers are the
# cheapest to move to system RAM. (Moving loop-layer experts to CPU also triggers llama.cpp bug #32755.)
NONLOOP_LAYERS = [23, 22, 21, 20, 19, 18, 0, 1, 2, 3, 4, 5]
EXPERT_SHARE_PER_LAYER = 0.032   # one layer's experts ~= 3.2% of the file (measured on Q2_K_XL)
OVERHEAD_MB = 2300               # context cache + compute buffers at 32K, q8_0, -ub 128, plus desktop headroom


class Quant:
    def __init__(self, name, size):
        self.name, self.size = name, size
        self.short = name.replace("UD-", "")
        self.quality = QUALITY.get(self.short, "")

    @property
    def filename(self):
        return f"Inkling-2-Mini-{self.name}.gguf"

    @property
    def url(self):
        return f"{HF}/{REPO}/resolve/main/{FOLDER}/Inkling-2-Mini-{self.name}.gguf_file"

    @property
    def gb(self):
        return self.size / 1e9


def fetch_quants(timeout=8):
    try:
        with urllib.request.urlopen(f"{HF}/api/models/{REPO}/tree/main/{FOLDER}", timeout=timeout) as r:
            items = json.load(r)
        qs = []
        for it in items:
            p = it.get("path", "")
            if p.endswith(".gguf_file"):
                name = p.split("/")[-1].replace("Inkling-2-Mini-", "").replace(".gguf_file", "")
                qs.append(Quant(name, int(it.get("size") or 0)))
        if qs:
            return sorted(qs, key=lambda q: q.size), True
    except Exception:
        pass
    return sorted((Quant(n, s) for n, s in FALLBACK.items()), key=lambda q: q.size), False


def mmproj_url():
    return f"{HF}/{REPO}/resolve/main/{MMPROJ[0]}"


def plan_fit(quant_size, vram_mb, ram_mb, vision_on_gpu=False):
    """How a model file would run here.
    status: gpu (all on the card) | split (some experts in RAM) | slow (whole layers on CPU) |
            ram_tight | cpu (no GPU). Also returns cpu_layers and gpu_layers to use."""
    size_mb = quant_size / 2**20
    if vram_mb <= 0:
        return {"status": "cpu", "cpu_layers": 0, "gpu_layers": 0}
    # measured: Q2_K_XL (8.7 GB) used 10.4 GB at 32K incl. vision; keep ~0.7 GB for the desktop
    budget = vram_mb - OVERHEAD_MB - 0.04 * size_mb - (650 if vision_on_gpu else 0)
    if size_mb <= budget:
        return {"status": "gpu", "cpu_layers": 0, "gpu_layers": 99}
    if size_mb + 2500 > ram_mb * 0.8 + max(0, budget):
        return {"status": "ram_tight", "cpu_layers": len(NONLOOP_LAYERS), "gpu_layers": 99}
    per_layer = size_mb * EXPERT_SHARE_PER_LAYER
    n = math.ceil((size_mb - budget) / per_layer)
    if n <= MAX_SPLIT_LAYERS:
        return {"status": "split", "cpu_layers": n, "gpu_layers": 99}
    gl = max(0, int(24 * max(0, budget) / size_mb))
    return {"status": "slow", "cpu_layers": 0, "gpu_layers": gl}


MAX_SPLIT_LAYERS = 10


def recommend(quants, vram_mb, ram_mb):
    """Best quality that runs fully on the GPU; else best quality with a modest expert split;
    else the smallest file."""
    fits = [q for q in quants if plan_fit(q.size, vram_mb, ram_mb)["status"] == "gpu"]
    split = [q for q in quants if plan_fit(q.size, vram_mb, ram_mb)["status"] == "split"]
    if fits and (fits[-1].quality != "Rough" or not split):
        return fits[-1]
    # a 1-bit file is too rough; a better file with some experts in RAM beats it
    return split[-1] if split else quants[0]


def offload_args(cpu_layers):
    if cpu_layers <= 0:
        return []
    layers = "|".join(str(n) for n in sorted(NONLOOP_LAYERS[:cpu_layers]))
    return ["-ot", rf"blk\.({layers})\.ffn_(gate_up|down)_exps\.=CPU"]
