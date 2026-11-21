<p align="center"><img src="data/io.github.Ezmanw.Inkwell.svg" width="96" alt=""></p>

<h1 align="center">Inkwell</h1>
<p align="center"><b>Run Inkling 2 Mini on your own Linux PC.</b> No cloud, no account, no web UI.</p>

Inkwell is a native GTK4/libadwaita app that sets everything up for you:

- **Builds llama.cpp for your graphics card.** Pick CUDA (NVIDIA), Vulkan (any GPU), ROCm (AMD) or CPU.
  It installs the build tools for you on Debian/Ubuntu (one password prompt).
- **Includes the Inkling VRAM fix.** Mainline llama.cpp currently builds a huge attention-bias table for
  Inkling 2 (about 12 GB at 128K context). Inkwell applies
  [llama.cpp PR #32740](https://github.com/ggml-org/llama.cpp/pull/32740), which brings that down to about 0.7 GB.
- **Picks the right model size.** It reads your VRAM and RAM and recommends a GGUF from
  [unsloth/Inkling-2-Mini-GGUF](https://huggingface.co/unsloth/Inkling-2-Mini-GGUF). If the best one doesn't quite
  fit, it moves a few layers' experts to system RAM instead of dropping to a worse file.
- **Chat** with thinking you can expand, effort control, and **image input** (attach or drag and drop).
- **Phone & other devices:** share it on your Wi-Fi as an **OpenAI-compatible API** with a key. Paste the
  URL and key into any phone chat app that supports custom OpenAI providers (QR codes included).
- **dsh (DeepSeek Harness):** one button adds Inkling to a dsh profile. Your old config is backed up first.
- **Tuning without flags:** context size, batch size, cache quality, where the vision part runs
  (GPU/CPU/off), thinking effort, thinking limit, and live VRAM and speed readouts.

## Install

Needs **Ubuntu 24.04+, Pop!_OS 24.04, Linux Mint 22, Debian 13** or newer (GTK 4.12+, libadwaita 1.5+).

1. Download `inkwell_<version>_all.deb` from [Releases](https://github.com/Ezmanw/inkwell/releases).
2. Install it:
   ```bash
   sudo apt install ./inkwell_0.1.0_all.deb
   ```
3. Open **Inkwell** from your app menu and follow the setup. Plan on 10–30 minutes (mostly compiling and
   downloading) and 10–25 GB of disk.

To run from source instead: `./run.sh` (same dependencies as the .deb).

## What it measured on an RTX 3060 12 GB

| Model | Context | Images | VRAM | Speed |
|---|---|---|---|---|
| UD-Q2_K_XL (9.1 GB) | 32K | on GPU | 11.1 GB | ~38 tok/s |
| UD-Q2_K_XL | 32K | on CPU | 9.6 GB | ~38 tok/s |
| UD-Q2_K_XL | 64K | on GPU | 11.8 GB | ~24 tok/s with a 35K-token prompt |

## Where things go

| What | Where |
|---|---|
| Settings, chats, server log | `~/.config/inkwell/` |
| llama.cpp source + build, models | `~/.local/share/inkwell/` (you can pick other folders during setup) |
| Model API (this PC only, no key) | `http://127.0.0.1:8686/v1`, model `inkling-2-mini` |
| Wi-Fi API (only while sharing is on) | `http://<your-ip>:8691/v1`, Bearer key shown in the app |

CUDA: if the toolkit is missing, Inkwell adds NVIDIA's apt repository **at low priority (100)**, so it can install
`cuda-nvcc` and friends but never replaces your distro's graphics driver. It picks the newest toolkit your driver
supports.

## Known issues

- **ROCm** is experimental. Vulkan is the safer choice on AMD. ROCm builds of `inkling2_loop` abort on some RDNA2
  cards ([#32732](https://github.com/ggml-org/llama.cpp/issues/32732)).
- `--n-cpu-moe` gives garbage on long prompts with this model
  ([#32755](https://github.com/ggml-org/llama.cpp/issues/32755)). Inkwell only ever moves experts from the
  non-looped layers (0–5, 18–23) with `-ot`, which avoids it.
- Even at the lowest thinking effort, Inkling still thinks for a line or two. The model can't fully skip it.
- Once PR #32740 is merged upstream, choose **Latest** under *Model → Setup → Rebuild* to drop the pinned version.

## Uninstall

```bash
sudo apt remove inkwell
rm -rf ~/.config/inkwell ~/.local/share/inkwell   # settings, chats, build and models
```

## Credits

- [llama.cpp](https://github.com/ggml-org/llama.cpp) (MIT) and the authors of the Inkling 2 support and the compact
  flash-attention bias PR.
- [Unsloth](https://huggingface.co/unsloth) for the Inkling 2 Mini GGUFs.
- Built with the help of Claude (Anthropic).

MIT licensed. See [LICENSE](LICENSE).
