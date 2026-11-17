"""Fast single-GPU launcher for LongCat-Video-Avatar 1.6 (their run_demo_avatar_single_audio_to_video.py, patched).

What it changes, all from outside their code:
  - ConvRot INT8 DiT linears on comfy-kitchen kernels (convert once with convert_convrot.py)
  - SageAttention 2.2 for self-attention
  - fused LoRA add for the DMD/OPSD LoRA side path
  - 6 distilled steps by default (LONGCAT_STEPS=4|6|8; 8 is their schedule)
  - LightVAE-pro decoder instead of the Wan VAE decoder
  - fits a 32 GB card: UMT5 encodes on the CPU, Whisper visits the GPU only to encode, VRAM is capped so the
    Windows driver raises OOM instead of silently spilling into system RAM
  - gloo instead of NCCL, so it runs on Windows

Everything after "--" goes to their demo. Defaults for the tested setup are filled in for flags you leave out.

    python run_fast.py -- --checkpoint_dir=D:/weights/LongCat-Video-Avatar-1.6 --input_json=my/talk.json --num_segments=4

Environment:
  LONGCAT_REPO          their LongCat-Video clone (default: ../LongCat-Video next to this folder)
  LONGCAT_TEXT_ENCODER  a ComfyUI UMT5 file (e.g. umt5_xxl_fp8_e4m3fn_scaled.safetensors) instead of their 22 GB fp32 one
  LONGCAT_STEPS         4, 6 (default) or 8
  LONGCAT_SIGMAS        comma list overriding the sigma pick for LONGCAT_STEPS
  LONGCAT_LIGHTVAE      pro (default), lite, or off (Wan VAE)
  LONGCAT_LIGHTVAE_DIR  folder holding lightvae-<pro|lite>-wan21-decoder.safetensors (default <checkpoint_dir>/lightvae)
  LONGCAT_LIGHTVAE_CHECK=1  also decode segment 1 with the Wan VAE and print the PSNR between the two
  LONGCAT_COMPILE       1 (default) adds --compile_dit; 0 leaves it off. On Windows run through run_fast.bat (needs MSVC)
  LONGCAT_VRAM_HEADROOM_GB  VRAM left for the desktop/browser (default 3)
  LONGCAT_SAGE, LONGCAT_CONVROT, LONGCAT_FAST_LORA  set to 0 to turn one optimisation off for A/B runs
  LONGCAT_PROFILE_CALL=N    profile DiT call N and print where its GPU time went
"""
import inspect
import json
import os
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.environ.get("LONGCAT_REPO", os.path.join(HERE, "..", "LongCat-Video")))
if not os.path.isfile(os.path.join(REPO, "run_demo_avatar_single_audio_to_video.py")):
    sys.exit(f"LongCat-Video clone not found at {REPO}; set LONGCAT_REPO to your clone of "
             "https://github.com/meituan-longcat/LongCat-Video")
if "--" not in sys.argv:
    sys.exit(__doc__)

os.environ.update(RANK="0", WORLD_SIZE="1", LOCAL_RANK="0", USE_LIBUV="0")
os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
os.environ.setdefault("MASTER_PORT", "29531")
if os.name != "nt":   # not supported on Windows (warns and is ignored)
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

# ---- their demo's arguments: fill in the tested defaults, make paths absolute (we chdir into their repo) ----
STEPS = int(os.environ.get("LONGCAT_STEPS", "6"))
DEMO_ARGS = sys.argv[sys.argv.index("--") + 1:]


def _given(flag):
    return any(a == flag or a.startswith(flag + "=") for a in DEMO_ARGS)


def _value(flag):
    for i, a in enumerate(DEMO_ARGS):
        if a.startswith(flag + "="):
            return a.split("=", 1)[1], i, True
        if a == flag and i + 1 < len(DEMO_ARGS):
            return DEMO_ARGS[i + 1], i + 1, False
    return None, None, None


DEFAULTS = {"--model_type": "avatar-v1.6", "--stage_1": "ai2v", "--resolution": "480p",
            "--mix_steps": str(STEPS // 2), "--mix_alpha": "1.0", "--ref_img_index": "10", "--mask_frame_range": "0",
            "--output_dir": os.path.join(os.getcwd(), "outputs")}
for flag, val in DEFAULTS.items():
    if not _given(flag):
        DEMO_ARGS.append(f"{flag}={val}")
for flag in ("--use_distill", "--use_int8"):   # --use_int8 is what routes their loader to the ConvRot DiT
    if flag not in DEMO_ARGS:
        DEMO_ARGS.append(flag)
if os.environ.get("LONGCAT_COMPILE", "1") == "1" and "--compile_dit" not in DEMO_ARGS:
    DEMO_ARGS.append("--compile_dit")

for flag in ("--checkpoint_dir", "--output_dir", "--input_json"):
    val, i, joined = _value(flag)
    if val is not None:
        DEMO_ARGS[i] = (f"{flag}={os.path.abspath(val)}" if joined else os.path.abspath(val))
CHECKPOINT_DIR, _, _ = _value("--checkpoint_dir")
if CHECKPOINT_DIR is None:
    sys.exit("pass --checkpoint_dir=<your LongCat-Video-Avatar-1.6 folder>")

# Paths inside the input JSON are relative to the JSON file; their demo would read them relative to its own repo.
_json, _ji, _jj = _value("--input_json")
if _json:
    with open(_json, encoding="utf-8") as f:
        case = json.load(f)
    base = os.path.dirname(_json)
    absolute = lambda p: p if os.path.isabs(p) else os.path.normpath(os.path.join(base, p))   # noqa: E731
    if "cond_image" in case:
        case["cond_image"] = absolute(case["cond_image"])
    case["cond_audio"] = {k: absolute(v) for k, v in case.get("cond_audio", {}).items()}
    tmp = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8")
    json.dump(case, tmp)
    tmp.close()
    import atexit
    atexit.register(os.remove, tmp.name)
    DEMO_ARGS[_ji] = f"--input_json={tmp.name}" if _jj else tmp.name

import torch  # noqa: E402
import torch.distributed as dist  # noqa: E402

# Windows' driver silently spills past VRAM into system RAM and crawls; cap PyTorch so it raises OOM instead.
_total = torch.cuda.get_device_properties(0).total_memory / 2 ** 30
_headroom = float(os.environ.get("LONGCAT_VRAM_HEADROOM_GB", "3"))
torch.cuda.set_per_process_memory_fraction(max(0.1, (_total - _headroom) / _total), 0)

_init = dist.init_process_group
dist.init_process_group = lambda backend=None, **kw: _init(backend="gloo", **kw)

os.chdir(REPO)
sys.path.insert(0, REPO)
sys.path.insert(0, HERE)
import longcat_video.pipeline_longcat_video_avatar as P  # noqa: E402

TIMES = []


def timed(name):
    original = getattr(P.LongCatVideoAvatarPipeline, name)

    def wrapper(self, *args, **kwargs):
        torch.cuda.synchronize()
        t = time.perf_counter()
        out = original(self, *args, **kwargs)
        torch.cuda.synchronize()
        TIMES.append({"call": name, "seconds": round(time.perf_counter() - t, 2),
                      "frames": kwargs.get("num_frames"), "steps": kwargs.get("num_inference_steps"),
                      "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 2 ** 30, 1)})
        print("[timing]", json.dumps(TIMES[-1]), flush=True)
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()   # the cache otherwise grows segment by segment
        return out
    setattr(P.LongCatVideoAvatarPipeline, name, wrapper)


for name in ("generate_ai2v", "generate_at2v", "generate_avc"):
    timed(name)

# Fewer distilled steps. Their demo forces 8 with --use_distill, and the v1.5/1.6 distill schedule is a fixed 8 sigmas
# (1.0 .875 .75 .625 .5 .375 .25 .125). LONGCAT_STEPS=N picks N of those trained sigmas (LONGCAT_SIGMAS overrides
# the pick); --mix_steps defaults to N/2 so the DMD/OPSD LoRA split stays at the midpoint.
STEP_PICKS = {8: [1.0, 0.875, 0.75, 0.625, 0.5, 0.375, 0.25, 0.125],
              6: [1.0, 0.875, 0.75, 0.5, 0.25, 0.125],
              4: [1.0, 0.75, 0.5, 0.25]}
if os.environ.get("LONGCAT_SIGMAS"):
    SIGMAS = [float(s) for s in os.environ["LONGCAT_SIGMAS"].split(",")]
elif STEPS in STEP_PICKS:
    SIGMAS = STEP_PICKS[STEPS]
else:
    sys.exit(f"LONGCAT_STEPS={STEPS}: pick 4, 6 or 8, or give LONGCAT_SIGMAS")
assert len(SIGMAS) == STEPS, f"LONGCAT_SIGMAS has {len(SIGMAS)} values for {STEPS} steps"
_sigmas = P.LongCatVideoAvatarPipeline.get_timesteps_sigmas


def _picked_sigmas(self, sampling_steps, use_distill=False):
    if not use_distill:
        return _sigmas(self, sampling_steps, use_distill)
    return torch.tensor(SIGMAS, dtype=torch.float32)


P.LongCatVideoAvatarPipeline.get_timesteps_sigmas = _picked_sigmas
for name in ("generate_ai2v", "generate_at2v", "generate_avc"):
    _gen = getattr(P.LongCatVideoAvatarPipeline, name)

    def _gen_steps(self, *args, _gen=_gen, **kwargs):
        kwargs["num_inference_steps"] = STEPS
        return _gen(self, *args, **kwargs)
    setattr(P.LongCatVideoAvatarPipeline, name, _gen_steps)
print(f"[timing] distilled steps: {STEPS}, sigmas {SIGMAS}", flush=True)

# Their demo puts every model on the GPU at once (DiT + 11 GB UMT5 + Whisper + VAE), which overflows 32 GB.
# Keep the text encoder off the GPU during .to(); it encodes on the CPU below.
_to = P.LongCatVideoAvatarPipeline.to


def _to_keep_te_on_cpu(self, device):
    te, self.text_encoder = self.text_encoder, None   # their .to() skips a None text encoder
    out = _to(self, device)
    self.text_encoder = te
    torch.cuda.empty_cache()
    gb = 2 ** 30
    print(f"[timing] after pipe.to: allocated {torch.cuda.memory_allocated() / gb:.1f} GB, "
          f"reserved {torch.cuda.memory_reserved() / gb:.1f} GB", flush=True)
    for name in ("dit", "vae", "audio_encoder"):
        m = getattr(self, name, None)
        if m is not None:
            size = sum(t.numel() * t.element_size() for t in list(m.parameters()) + list(m.buffers()))
            print(f"[timing]   {name}: {size / gb:.1f} GB", flush=True)

    calls = {"n": 0}
    prof_call = int(os.environ.get("LONGCAT_PROFILE_CALL", "0"))   # profile this DiT call (1-based); 0 = off

    def pre(mod, args, kwargs):
        torch.cuda.synchronize()
        if calls["n"] + 1 == prof_call:
            calls["prof"] = torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA])
            calls["prof"].__enter__()
        calls["t"] = time.perf_counter()

    def post(mod, args, kwargs, output):
        torch.cuda.synchronize()
        if "prof" in calls:
            prof = calls.pop("prof")
            prof.__exit__(None, None, None)
            _report_profile(prof, time.perf_counter() - calls["t"])
        torch.cuda.empty_cache()   # Windows has no expandable_segments; fragmentation OOM'd a continuation
        calls["n"] += 1
        print(f"[timing] dit call {calls['n']}: {time.perf_counter() - calls['t']:.2f} s, "
              f"peak {torch.cuda.max_memory_allocated() / gb:.1f} GB", flush=True)

    self.dit.register_forward_pre_hook(pre, with_kwargs=True)
    self.dit.register_forward_hook(post, with_kwargs=True)
    return out


P.LongCatVideoAvatarPipeline.to = _to_keep_te_on_cpu


def _report_profile(prof, wall):
    """Print where one DiT step's GPU time went: our labelled regions, then the top kernels."""
    events = prof.key_averages()
    dev = lambda e: getattr(e, "device_time_total", getattr(e, "cuda_time_total", 0)) / 1e6   # noqa: E731
    selfdev = lambda e: getattr(e, "self_device_time_total", getattr(e, "self_cuda_time_total", 0)) / 1e6  # noqa: E731
    total = sum(selfdev(e) for e in events)
    print(f"[profile] wall {wall:.2f} s, summed GPU kernel time {total:.2f} s", flush=True)
    for e in sorted((e for e in events if e.key.startswith("ck::")), key=dev, reverse=True):
        print(f"[profile] region {e.key:28s} {dev(e):6.2f} s  ({e.count} calls)", flush=True)
    for e in sorted(events, key=selfdev, reverse=True)[:25]:
        if selfdev(e) > 0.01:
            print(f"[profile] op     {e.key[:60]:60s} {selfdev(e):6.2f} s  ({e.count} calls)", flush=True)


import longcat_video.modules.avatar.longcat_video_dit_avatar as DA  # noqa: E402


def lora_add_(out, x, lora):
    """out += lora(x) * multiplier * alpha_scale, without full-size temporaries.

    Their forward makes 3-4 passes over the [tokens, out_features] output (cast, x multiplier, x alpha_scale, add).
    Here the scale goes on the rank-128 activation and lora_up accumulates straight into `out` (addmm_, beta=1).
    """
    # _ck_alpha is set once when the LoRAs are enabled (a python float), so nothing here syncs or graph-breaks
    lx = lora.lora_down(x.to(lora.lora_down.weight.dtype)) * (lora.multiplier * lora._ck_alpha)
    lx = lx.to(out.dtype).reshape(-1, lx.shape[-1])
    out2 = out.view(-1, out.shape[-1])
    up = lora.lora_up
    if hasattr(up, "blocks"):   # LoRAUPParallel: block i maps lx chunk i to output column chunk i (qkv)
        n = len(up.blocks)
        for blk, lx_i, out_i in zip(up.blocks, lx.chunk(n, -1), out2.chunk(n, -1)):
            out_i.addmm_(lx_i, blk.weight.to(out.dtype).t())
    else:
        out2.addmm_(lx, up.weight.to(out.dtype).t())
    return out


FAST_LORA = os.environ.get("LONGCAT_FAST_LORA", "1") == "1"
PROFILING = os.environ.get("LONGCAT_PROFILE_CALL", "0") != "0"


class _NoLabel:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _multi_lora_forward(self, module, loras):
    for lora in loras:
        lora._ck_alpha = float(lora.alpha_scale)   # once, here, outside anything torch.compile traces

    def multi_lora_forward(x, *args, **kwargs):
        out = module.org_forward(x, *args, **kwargs)
        if not any(lora.use_lora for lora in loras):
            return out
        # record_function only when profiling: under torch.compile it is just noise
        with (torch.profiler.record_function("ck::lora_side_path") if PROFILING else _NoLabel()):
            if not FAST_LORA or not out.is_contiguous():
                total = 0
                for lora in loras:
                    if lora.use_lora:
                        lx = lora.lora_up(lora.lora_down(x.to(lora.lora_down.weight.dtype)))
                        total += lx.to(x.dtype) * lora.multiplier * lora.alpha_scale
                return out + total
            for lora in loras:
                if lora.use_lora:
                    lora_add_(out, x, lora)
            return out
    return multi_lora_forward


DA.LongCatVideoAvatarTransformer3DModel._create_multi_lora_forward = _multi_lora_forward

from longcat_video.modules.autoencoder_kl_wan import AutoencoderKLWan  # noqa: E402

for _vae_fn in ("encode", "decode"):
    def _vae_timed(self, *args, _orig=getattr(AutoencoderKLWan, _vae_fn), _name=_vae_fn, **kwargs):
        torch.cuda.synchronize()
        t = time.perf_counter()
        out = _orig(self, *args, **kwargs)
        torch.cuda.synchronize()
        print(f"[timing] vae {_name}: {time.perf_counter() - t:.2f} s", flush=True)
        return out
    setattr(AutoencoderKLWan, _vae_fn, _vae_timed)

_t5 = P.LongCatVideoAvatarPipeline._get_t5_prompt_embeds
_T5_CACHE = {}


def _t5_cached(self, *args, **kwargs):
    key = repr((args, sorted((k, v) for k, v in kwargs.items() if not isinstance(v, torch.Tensor))))
    if key not in _T5_CACHE:
        # No room for 11 GB of UMT5 beside the DiT, so encode the prompt on the CPU, once per prompt.
        bound = inspect.signature(_t5).bind(self, *args, **kwargs)
        bound.apply_defaults()
        device = bound.arguments["device"]
        bound.arguments["device"] = "cpu"
        t = time.perf_counter()
        embeds, mask = _t5(*bound.args, **bound.kwargs)
        _T5_CACHE[key] = (embeds.to(device), mask.to(device))
        print(f"[timing] text encoder ran once on CPU: {time.perf_counter() - t:.1f} s", flush=True)
    return _T5_CACHE[key]


P.LongCatVideoAvatarPipeline._get_t5_prompt_embeds = _t5_cached

# Whisper: the pipeline only calls audio_encoder.encoder(...), but their loader builds the whole WhisperModel (decoder
# included) in fp32, 5.7 GB. Build just the encoder (2.5 GB fp32, same weights, same features) from their checkpoint.
import longcat_video.audio_process as AP  # noqa: E402

_get_audio_encoder = AP.get_audio_encoder


class _WhisperEncoderOnly(torch.nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder

    @property
    def dtype(self):
        return next(self.parameters()).dtype


def _encoder_only(checkpoint_path, model_type="avatar-v1.0"):
    if model_type != "avatar-v1.5":
        return _get_audio_encoder(checkpoint_path, model_type)
    import glob
    from safetensors import safe_open
    from transformers import WhisperConfig
    from transformers.models.whisper.modeling_whisper import WhisperEncoder

    with torch.device("meta"):   # no random init; the checkpoint fills every tensor
        encoder = WhisperEncoder(WhisperConfig.from_pretrained(checkpoint_path))
    prefix = "model.encoder."
    state = {}
    for f in glob.glob(os.path.join(checkpoint_path, "*.safetensors")):
        with safe_open(f, "pt") as h:
            state.update({k[len(prefix):]: h.get_tensor(k).float() for k in h.keys() if k.startswith(prefix)})
    encoder.load_state_dict(state, strict=True, assign=True)
    return _WhisperEncoderOnly(encoder).eval().requires_grad_(False)


AP.get_audio_encoder = _encoder_only

# Whisper encodes the whole clip's audio once before denoising, then idles. Visit the GPU only for that.
_audio = P.LongCatVideoAvatarPipeline.get_audio_embedding


def _audio_then_offload(self, *args, **kwargs):
    self.audio_encoder.to("cuda")
    out = _audio(self, *args, **kwargs)
    self.audio_encoder.to("cpu")
    torch.cuda.empty_cache()
    return out


P.LongCatVideoAvatarPipeline.get_audio_embedding = _audio_then_offload

# SageAttention for the DiT's self-attention (2.6x FA2 at 38k tokens, cos 0.9992). Cross-attention stays on FA2.
if os.environ.get("LONGCAT_SAGE", "1") == "1":
    import longcat_video.modules.attention as A
    import longcat_video.modules.avatar.attention as AA   # the avatar DiT has its own Attention class
    from sageattention import sageattn

    def _sage_for(cls):
        original = cls._process_attn

        def _process_attn_sage(self, q, k, v, shape):
            if self.enable_bsa and shape[0] > 1:
                return original(self, q, k, v, shape)
            if PROFILING:
                with torch.profiler.record_function("ck::self_attention_sage"):
                    return sageattn(q, k, v, tensor_layout="HND", is_causal=False, sm_scale=self.scale)
            return sageattn(q, k, v, tensor_layout="HND", is_causal=False, sm_scale=self.scale)   # B H S D

        cls._process_attn = _process_attn_sage

    for cls in (A.Attention, AA.Attention):
        _sage_for(cls)
    print("[timing] self-attention: SageAttention", flush=True)

# ConvRot INT8 DiT. Their demo imports load_quantized_dit by name, so swap it before importing the demo;
# --use_int8 then loads <checkpoint_dir>/base_model_convrot instead of their base_model_int8.
if os.environ.get("LONGCAT_CONVROT", "1") == "1":
    import longcat_video.modules.quantization as Q
    from longcat_convrot import SUBFOLDER, load_convrot_dit

    def _load_dit(checkpoint_dir, subfolder="base_model_int8", **kwargs):
        return load_convrot_dit(checkpoint_dir, SUBFOLDER, **kwargs)

    Q.load_quantized_dit = _load_dit
    print("[timing] DiT linears: ConvRot INT8 (comfy-kitchen)", flush=True)

# LightVAE (ModelTC, Apache-2.0) as the decoder: same Wan2.1 latent space, a fraction of the FLOPs. Encoding the
# conditioning frames stays on the Wan VAE.
LIGHTVAE = os.environ.get("LONGCAT_LIGHTVAE", "pro")
if LIGHTVAE in ("pro", "lite"):
    from importlib import resources
    from lightvae.wan21_lightvae_decoder import Decoder as LightDecoder

    LIGHTVAE_DIR = os.environ.get("LONGCAT_LIGHTVAE_DIR", os.path.join(CHECKPOINT_DIR, "lightvae"))
    _LIGHT_CKPT = os.path.join(LIGHTVAE_DIR, f"lightvae-{LIGHTVAE}-wan21-decoder.safetensors")
    if not os.path.isfile(_LIGHT_CKPT):
        sys.exit(f"{_LIGHT_CKPT} not found: download it (see README) or set LONGCAT_LIGHTVAE=off")
    _LIGHT = {}
    _decode_wan = P.LongCatVideoAvatarPipeline._decode_video_latents

    def _decode_video_latents_light(self, latents):
        if "model" not in _LIGHT:
            cfg = json.loads(resources.files("lightvae").joinpath("config", f"lightvae-{LIGHTVAE}-wan21.json").read_text())
            _LIGHT["model"] = LightDecoder(config=cfg, checkpoint=_LIGHT_CKPT).to(latents.device, torch.bfloat16).eval()
        # LongCat hands the Wan VAE de-normalised latents; LightVAE wants them normalised (it de-normalises itself)
        mean = torch.tensor(self.vae.config.latents_mean, device=latents.device).view(1, -1, 1, 1, 1)
        std = torch.tensor(self.vae.config.latents_std, device=latents.device).view(1, -1, 1, 1, 1)
        z = ((latents.float() - mean) / std).to(torch.bfloat16)
        torch.cuda.synchronize()
        t = time.perf_counter()
        video = _LIGHT["model"](z).clamp_(-1, 1)
        torch.cuda.synchronize()
        print(f"[timing] lightvae-{LIGHTVAE} decode: {time.perf_counter() - t:.2f} s", flush=True)
        if os.environ.get("LONGCAT_LIGHTVAE_CHECK") == "1" and "checked" not in _LIGHT:
            _LIGHT["checked"] = True
            ref = _decode_wan(self, latents).float()
            got = video.float()[:, :, : ref.shape[2]]
            mse = ((ref - got) ** 2).mean().item()   # both in [-1, 1]: peak-to-peak 2
            print(f"[check] lightvae-{LIGHTVAE} vs Wan VAE: PSNR {10 * torch.log10(torch.tensor(4.0 / mse)).item():.1f} dB",
                  flush=True)
        return video.to(latents.dtype)

    P.LongCatVideoAvatarPipeline._decode_video_latents = _decode_video_latents_light
    print(f"[timing] VAE decode: LightVAE-{LIGHTVAE}", flush=True)
elif LIGHTVAE != "off":
    sys.exit(f"LONGCAT_LIGHTVAE={LIGHTVAE}: use pro, lite or off")

import run_demo_avatar_single_audio_to_video as demo  # noqa: E402

# UMT5 from a ComfyUI text-encoder file (e.g. Wan's umt5_xxl_fp8_e4m3fn_scaled.safetensors, 6.7 GB) instead of their
# 22 GB fp32 checkpoint, which takes ~30 s to load. Same google UMT5-XXL; fp8 weights are dequantized to bf16.
TEXT_ENCODER_FILE = os.environ.get("LONGCAT_TEXT_ENCODER")
if TEXT_ENCODER_FILE:
    from safetensors import safe_open
    from transformers import UMT5Config

    class _ComfyUMT5:
        @staticmethod
        def from_pretrained(path, subfolder=None, torch_dtype=torch.bfloat16, **kwargs):
            t = time.perf_counter()
            with torch.device("meta"):
                model = demo.transformers_UMT5EncoderModel(UMT5Config.from_pretrained(path, subfolder=subfolder))
            state = {}
            with safe_open(TEXT_ENCODER_FILE, "pt") as f:
                keys = set(f.keys())
                for k in keys:
                    if k.endswith((".scale_weight", ".scale_input")) or k in ("scaled_fp8", "spiece_model"):
                        continue
                    w = f.get_tensor(k)
                    scale = k[: -len("weight")] + "scale_weight"
                    w = w.float() * f.get_tensor(scale).float() if scale in keys else w
                    state[k] = w.to(torch_dtype)
            emb = state.get("shared.weight", state.get("encoder.embed_tokens.weight"))   # tied pair; files carry one
            state["shared.weight"] = state["encoder.embed_tokens.weight"] = emb
            model.load_state_dict(state, strict=True, assign=True)
            model.encoder.embed_tokens = model.shared
            print(f"[timing] text encoder from {os.path.basename(TEXT_ENCODER_FILE)}: {time.perf_counter() - t:.1f} s",
                  flush=True)
            return model.eval().requires_grad_(False)

    demo.transformers_UMT5EncoderModel = demo.UMT5EncoderModel
    demo.UMT5EncoderModel = _ComfyUMT5

sys.argv = [demo.__file__] + DEMO_ARGS
print("[timing] demo args:", " ".join(DEMO_ARGS), flush=True)
t0 = time.perf_counter()
demo.generate(demo._parse_args())
print("[timing] total", round(time.perf_counter() - t0, 1), "s", flush=True)
