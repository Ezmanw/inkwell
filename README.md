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
- **See what it's doing:** the header shows live speed for *any* app using the model (dsh, your phone, the
  chat), e.g. "Writing · 37 tok/s". The **Activity** button opens a plain-English feed ("An app sent a
  request", "Wrote 412 tokens at 36 tok/s", "Context full…") or the raw server log. **Pop out** puts it in its
  own window to keep next to dsh.
- **Several engines, one click to switch:** build llama.cpp with CUDA, Vulkan, ROCm and/or CPU side by side
  (*Model → Setup → Build another…*), then hot-switch under *Engine*. The model restarts on the new build.
- **Tuning without flags:** context size, batch size, cache quality, where the vision part runs
  (GPU/CPU/off), thinking effort, thinking limit, and live VRAM and speed readouts.

## Install

Needs **Ubuntu 24.04+, Pop!_OS 24.04, Linux Mint 22, Debian 13** or newer (GTK 4.12+, libadwaita 1.5+).

1. Download `inkwell_<version>_all.deb` from [Releases](https://github.com/Ezmanw/inkwell/releases).
2. Install it:
   ```bash
   sudo apt install ./inkwell_0.2.3_all.deb
   ```
3. Open **Inkwell** from your app menu and follow the setup. Plan on 10–30 minutes (mostly compiling and
   downloading) and 10–25 GB of disk.

**Already downloaded the model?** On the *Storage* step, set **Models** to the folder that has your
`Inkling-2-Mini-*.gguf` (and `mmproj-F16.gguf`), then pick the same size on the *Model* step. Files that are
already there and complete are skipped. A half-finished download (`.part`) picks up where it stopped. You can
also point *llama.cpp* at an existing checkout to reuse it.

To run from source instead: `./run.sh` (same dependencies as the .deb).

## What it measured on an RTX 3060 12 GB

| Model | Context | Images | VRAM | Speed |
|---|---|---|---|---|
| UD-Q2_K_XL (9.1 GB) | 32K | on GPU | 11.1 GB | ~38 tok/s |
| UD-Q2_K_XL | 32K | on CPU | 9.6 GB | ~38 tok/s |
| UD-Q2_K_XL | 64K | on GPU | 11.8 GB | ~24 tok/s with a 35K-token prompt |
| UD-Q2_K_XL, 6 layers' experts in RAM | 16K | off | 7.4 GB | ~26 tok/s |
| UD-Q2_K_XL, **Vulkan** instead of CUDA | 32K | on CPU | 10.0 GB | ~16 tok/s |

## Speed: what actually helps

Measured on an RTX 3060 12 GB with UD-Q2_K_XL (writing = generation speed, reading = prompt processing):

| Setup | Reading | Writing | VRAM |
|---|---|---|---|
| 128K context, images on GPU, 10 layers' experts in RAM, batch 256 | 379 tok/s | 21.9 tok/s | 9.7 GB |
| 64K, images on CPU, all on GPU, batch 256 | 484 tok/s | 38.2 tok/s | 10.4 GB |
| 64K, images on CPU, all on GPU, **batch 512** (Inkwell's default now) | 540 tok/s | 37.9–40.9 tok/s | 10.6 GB |
| 64K, images on CPU, all on GPU, batch 1024 | 566 tok/s | 38.9 tok/s | 11.0 GB |

- **Keep every layer on the GPU** if you can. Moving experts to RAM roughly halves writing speed, so give up some
  context first.
- **Bigger batch = faster prompt reading**, at the cost of some VRAM.
- Apps that re-send a growing conversation (dsh) only pay for the new part: the rest is reused automatically
  (12.5K-token history: 24.8 s the first time, 0.37 s on the next turn).
- **AMD/Vulkan:** prompt reading is limited by the Inkling attention-bias path, which runs Vulkan's basic
  flash-attention kernel (no matrix cores) until a faster version lands upstream. If VRAM is nearly full, AMD's
  driver silently spills to system RAM and everything gets several times slower, so watch the VRAM bar.

## Speed-up drafter (DSpark)

Inkwell's recommended build includes [llama.cpp PR #32218](https://github.com/ggml-org/llama.cpp/pull/32218), so
it can run Unsloth's 0.4 GB DSpark drafter (*Model → Speed-up drafter*). A small helper model guesses a few tokens
ahead and the big model checks them in one go. Output is identical, just faster when the guesses are good.

- On an **RX 9070 XT 16 GB (Vulkan)**: 47.6 → **80 tok/s on code** with draft length 4, but slower on prose
  ([#32773](https://github.com/ggml-org/llama.cpp/pull/32773)).
- On an **RTX 3060 12 GB** with the 2-bit model it **didn't help**: it only fit at 48K context, code stayed at
  ~40 tok/s (39% of guesses accepted) and prose dropped to 28 tok/s. A 2-bit model drifts further from the
  drafter's guesses than a 3–4-bit one.
- So setup switches it on by default only for **16 GB+ cards**. It needs ~0.75 GB extra VRAM; Inkwell keeps its
  cache at q4_0 (with the default f16 it can overflow a 16 GB card and slow everything down).
- Already installed Inkwell? Use *Model → Setup → Build another…* to rebuild with the drafter support, then turn on
  *Speed-up drafter* (it downloads the drafter by itself).

## Manual mode (bring your own flags)

*Model → Manual mode* runs `llama-server` with **exactly** the flags you type and nothing else: no sampling
defaults, no thinking level, no context/batch settings (those controls grey out). It starts pre-filled with
Inkwell's own command, one flag per line, and `#` starts a comment. It warns about characters that sneak in when copying flags from a web page or chat (hidden spaces, zero-width characters, curly quotes and dashes) and cleans them up on Apply. Inkwell follows your `--host`/`--port`
(llama-server's default is 8080), so the chat, Activity view, phone API and dsh button keep working.

## dsh thinking levels

*Connect dsh* adds Inkling with dsh's thinking picker wired to what the model supports:
**minimal / low / medium / high / xhigh** (Inkling's template maps them to effort 0.1 / 0.2 / 0.7 / 0.9 / 0.99).
dsh's *off* maps to *minimal*, because Inkling can't fully skip thinking and llama.cpp's `none` is currently
buggy ([#32753](https://github.com/ggml-org/llama.cpp/issues/32753)). Updating from ≤ 0.1.6? Press *Connect dsh*
again: older versions left out the placeholder auth header dsh needs for keyless local servers.

## Extra flags on top of Inkwell's

*Model → Advanced → Extra llama-server flags* adds anything you type to the end of the `llama-server` command,
so your values override Inkwell's for most settings (e.g. `--top-k 40 --repeat-penalty 1.05`). Quotes work for
values with spaces. Don't change `--host`/`--port`, because the app, dsh and the phone API connect to
`127.0.0.1:8686`. The exact command is shown under *Server log*.

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

- **ROCm** is experimental. Vulkan is the safer choice on AMD. Inkwell finds ROCm's HIP CMake files even in
  newer split layouts (`/opt/rocm/core-X.Y`), and if a build fails, *Change settings* lets you switch to Vulkan
  without re-downloading. ROCm builds of `inkling2_loop` abort on some RDNA2
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
