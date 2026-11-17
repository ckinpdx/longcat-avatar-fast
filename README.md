# longcat-avatar-fast

A single-GPU launcher that runs [LongCat-Video-Avatar 1.6](https://huggingface.co/meituan-longcat/LongCat-Video-Avatar-1.6) about **3x faster at the same 8 steps, and 3.8x faster at its default 6 steps**, on one 32 GB card under Windows. It patches Meituan's own demo script from the outside; their repo stays unmodified.

Measured on an RTX 5090, 480p bucket, one 3.2 s continuation segment:

| Setup | Per step | Per 3.2 s segment | Slower than real time |
|---|---|---|---|
| Their demo (their INT8 + FlashAttention 2), 8 steps | 13.6 s | 115 s | 36x |
| + SageAttention 2.2 | 10.8 s | 93 s | 29x |
| + ConvRot INT8 linears | 7.6 s | 65 s | 20x |
| + fused LoRA add | 6.75 s | 59 s | 18x |
| + `torch.compile` + LightVAE-pro decoder | 4.55 s | 38.9 s | 12x |
| **6 steps (default)** | | **30.0 s** | **9.4x** |
| 4 steps | | 20.7 s | 6.5x |

It's an offline renderer, not a real-time one.

## Samples

The same image, audio and seed at each step count: 13.3 s of video at 640x608 (4 segments). Click a frame to download the clip (with sound).

| 4 steps | 6 steps (default) | 8 steps |
|---|---|---|
| [![4 steps](assets/teal_4step.jpg)](https://github.com/ckinpdx/longcat-avatar-fast/raw/main/assets/teal_4step.mp4) | [![6 steps](assets/teal_6step.jpg)](https://github.com/ckinpdx/longcat-avatar-fast/raw/main/assets/teal_6step.mp4) | [![8 steps](assets/teal_8step.jpg)](https://github.com/ckinpdx/longcat-avatar-fast/raw/main/assets/teal_8step.mp4) |
| **168.8 s** total | **203.6 s** total | **227.1 s** total |
| 111.2 s generating | 145.0 s generating | 170.0 s generating |

"Total" is the whole process, cold. About 57 s of it is loading the weights and encoding the prompt and audio. Segment 1 also carries the one-time `torch.compile` (~25 s) and the LightVAE warm-up.

Six steps is the default because it's hard to tell apart from 8: the two clips differ by 30.7 dB PSNR, which is a smaller gap than between two 8-step runs that differ only in their INT8 kernels (27.4 dB). Four steps loses a little detail and is still good.

## What it changes

| Change | What it does | Effect |
|---|---|---|
| **ConvRot INT8** (`longcat_convrot.py`) | Every Linear in the 48 DiT blocks (624 layers) runs as W8A8 on INT8 tensor cores through [comfy-kitchen](https://github.com/Comfy-Org/comfy-kitchen)'s `int8_linear`. A group-256 Hadamard rotation keeps the activation outliers quantizable. Their own INT8 checkpoint only stores INT8: it dequantizes to bf16 before every matmul, so it's no faster than bf16. | 3.4x on the linears; min layer cosine 0.99984 vs bf16 |
| **SageAttention 2.2** | Self-attention in both their `Attention` classes. The avatar DiT has its own class, so patching only the shared one does nothing. Cross-attention stays on FlashAttention 2. | 2.6x on self-attention, cosine 0.9992 |
| **Fused LoRA add** | The DMD/OPSD LoRAs run as a side path on every Linear. Their forward makes 3-4 full-size passes over each output; this scales the rank-128 activation instead and accumulates into the output with `addmm_`. | about 0.85 s per step |
| **`torch.compile`** | Their `--compile_dit`, made usable: the LoRA closures no longer graph-break, so dynamo stays under its recompile limit. | about 2 s per step |
| **LightVAE-pro** ([ModelTC/LightVAE](https://github.com/ModelTC/LightVAE), Apache-2.0) | Decodes the Wan2.1 latents instead of the Wan VAE. The launcher re-normalises the latents, because LongCat hands de-normalised ones to the VAE. | 1.5 s vs 3.5 s per segment, 35.9 dB vs the Wan VAE |
| **Fewer steps** | Picks 6 (or 4) of the 8 trained distillation sigmas, and sets `--mix_steps` to half the step count. | |
| **32 GB fit** | UMT5 encodes on the CPU, once per prompt. Whisper is built encoder-only (2.4 GB instead of 5.7 GB; their loader builds the decoder too, in fp32) and visits the GPU only to encode the audio. PyTorch's cache is emptied after every DiT step, since Windows has no `expandable_segments`. VRAM is capped (3 GB headroom) so the Windows driver raises out-of-memory instead of silently spilling into system RAM and crawling. | Peak 24.2 GB allocated |
| **Windows** | Swaps NCCL for gloo (world size 1). | |

## Setup

Tested on Windows 11, an RTX 5090, Python 3.11, torch 2.11 + CUDA 13.0. Linux should work but hasn't been tried.

1. **Clone their repo and install its requirements**, following their README:

   ```
   git clone https://github.com/meituan-longcat/LongCat-Video
   ```

   Put it next to this repo as `LongCat-Video`, or point `LONGCAT_REPO` at it.

2. **Install the extras** into the same environment:

   ```
   pip install comfy-kitchen==0.2.41 "git+https://github.com/ModelTC/LightVAE@38b0587"
   ```

   You also need:
   - **SageAttention 2.2.** On Windows, use a [woct0rdho wheel](https://github.com/woct0rdho/SageAttention/releases) matching your torch and CUDA.
   - **Triton.** On Windows, that's `triton-windows`.
   - **FlashAttention 2**, which their code uses for cross-attention.

3. **Download the weights**, keeping both folders side by side because their demo reads the base model from `<avatar>/../LongCat-Video`:

   ```
   weights/
     LongCat-Video/              tokenizer, text_encoder, vae, scheduler + configs (the dit folder isn't needed)
     LongCat-Video-Avatar-1.6/   base_model, loraA, loraB, whisper-large-v3, vocal_separator, scheduler + configs
   ```

   - The base model comes from `meituan-longcat/LongCat-Video`; the avatar folder is `meituan-longcat/LongCat-Video-Avatar-1.6`.
   - Their `base_model_int8` isn't needed.
   - Get the LightVAE decoder from `lightx2v/LightVAE`:

     ```
     hf download lightx2v/LightVAE lightvae-pro-wan21-decoder.safetensors --local-dir weights/LongCat-Video-Avatar-1.6/lightvae
     ```

4. **Convert the DiT to ConvRot once** (about 35 s on a 5090):

   ```
   python convert_convrot.py weights/LongCat-Video-Avatar-1.6
   ```

   This writes `base_model_convrot` (15 GB). After that, the bf16 `base_model` (27 GB) can be deleted.

## Run

Write an input JSON. Relative paths in it are read relative to the JSON file:

```json
{
  "prompt": "A woman with long dark hair ... talking to the camera in a relaxed, warm way ... The camera does not move.",
  "cond_image": "portrait.png",
  "cond_audio": {"person1": "speech.wav"}
}
```

Then run it:

```
run_fast.bat -- --checkpoint_dir=weights/LongCat-Video-Avatar-1.6 --input_json=my/talk.json --num_segments=4
```

`run_fast.bat` loads the MSVC environment that `torch.compile` needs on Windows and runs the `python` on your PATH (or `LONGCAT_PYTHON`). On Linux, or with `LONGCAT_COMPILE=0`, run `python run_fast.py -- ...` directly.

- **Segments:** the first segment is 93 frames (3.7 s at 25 fps); each `--num_segments` beyond 1 adds 80 frames (3.2 s). Size the audio to match.
- **Output:** clips go to `./outputs` unless you pass `--output_dir`.
- **Other flags:** everything after `--` is their demo's own flags. Flags you leave out get the tested defaults:

  ```
  --model_type=avatar-v1.6 --stage_1=ai2v --resolution=480p --use_distill --use_int8 --compile_dit
  --mix_steps=<steps/2> --mix_alpha=1.0 --ref_img_index=10 --mask_frame_range=0
  ```

  `--use_int8` is what routes their loader to the ConvRot DiT.

| Environment variable | Default | |
|---|---|---|
| `LONGCAT_STEPS` | `6` | `4`, `6` or `8` (8 is their schedule) |
| `LONGCAT_SIGMAS` | | Comma list overriding the sigmas picked for `LONGCAT_STEPS` |
| `LONGCAT_LIGHTVAE` | `pro` | `pro`, `lite` or `off` (Wan VAE) |
| `LONGCAT_LIGHTVAE_DIR` | `<checkpoint_dir>/lightvae` | |
| `LONGCAT_LIGHTVAE_CHECK` | `0` | `1` also decodes segment 1 with the Wan VAE and prints the PSNR |
| `LONGCAT_COMPILE` | `1` | `0` drops `--compile_dit` (slower, no MSVC needed) |
| `LONGCAT_VRAM_HEADROOM_GB` | `3` | VRAM left for the desktop and browser |
| `LONGCAT_SAGE`, `LONGCAT_CONVROT`, `LONGCAT_FAST_LORA` | `1` | `0` turns one optimisation off for A/B runs |
| `LONGCAT_PROFILE_CALL` | `0` | `N` profiles DiT call N and prints where its GPU time went |
| `LONGCAT_REPO` | `../LongCat-Video` | Their repo |

`tools/` has the micro-benchmarks behind the numbers above (attention, linears) and a check of the fused LoRA add against their formula.

## Notes

- Only their single-audio demo (`run_demo_avatar_single_audio_to_video.py`) is wired up.
- The first segment compiles for its shape (~25 s). The first continuation recompiles once for the continuation shape; after that, segments run warm.
- Each run reloads everything (~57 s). Rendering many clips from one process would save that, but this launcher doesn't do it.

## Licence

MIT, for the code in this repo. LongCat-Video and its weights are MIT (Meituan). LightVAE is Apache-2.0 (ModelTC). comfy-kitchen and SageAttention are Apache-2.0. No weights are distributed here.
