"""Convert a UMT5-XXL encoder (Wan's and LongCat's text encoder) to INT8 ConvRot, in ComfyUI's quantized format.

    python convert_umt5_convrot.py <LongCat-Video>/text_encoder umt5_xxl_int8_convrot.safetensors [--spiece <spiece.model>]

The source is a Hugging Face UMT5 folder (LongCat ships google/umt5-xxl in fp32) or a single fp16/bf16 safetensors.
- The 168 attention and FFN Linears become int8_tensorwise with ConvRot (group 256) and per-row scales. They run on
  comfy-kitchen's INT8 kernels.
- The 256k-token embedding becomes per-row INT8 (no rotation); lookups dequantize the selected rows.
- Layer norms and the relative attention biases stay in bf16.
- spiece_model (the SentencePiece tokenizer as bytes) is embedded the way ComfyUI's Wan UMT5 files carry it, so the
  file also loads in ComfyUI's CLIPLoader (type wan). ComfyUI currently runs it less accurately than run_fast.py does.

Prints the output cosine vs the source weights for a sample of layers on random activations.
"""
import argparse
import glob
import json
import os
import re
import sys
import time

import torch
import torch.nn.functional as F
from safetensors import safe_open
from safetensors.torch import save_file

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from longcat_convrot import GROUP_SIZE, comfy_quant_tensor, quantize_weight  # noqa: E402

LINEAR = re.compile(r"^encoder\.block\.\d+\.layer\.\d+\.(SelfAttention\.(q|k|v|o)|DenseReluDense\.(wi_0|wi_1|wo))\.weight$")
EMBED = "shared.weight"

ap = argparse.ArgumentParser()
ap.add_argument("src")
ap.add_argument("dst")
ap.add_argument("--spiece", help="spiece.model to embed (default: <src>/../tokenizer/spiece.model if present)")
args = ap.parse_args()

files = glob.glob(os.path.join(args.src, "*.safetensors")) if os.path.isdir(args.src) else [args.src]
spiece = args.spiece or os.path.join(os.path.dirname(os.path.normpath(args.src)), "tokenizer", "spiece.model")

out, cosines, n_lin, t0 = {}, [], 0, time.perf_counter()
for path in files:
    with safe_open(path, "pt", device="cpu") as f:
        for key in f.keys():
            if key in ("spiece_model", "scaled_fp8") or key.endswith((".scale_weight", ".scale_input")):
                continue
            t = f.get_tensor(key)
            if LINEAR.match(key):
                if t.shape[1] % GROUP_SIZE:
                    raise ValueError(f"{key}: in_features {t.shape[1]} not a multiple of {GROUP_SIZE}")
                w = t.to("cuda", torch.bfloat16)
                q, scale = quantize_weight(w)
                name = key[: -len(".weight")]
                out[key], out[f"{name}.weight_scale"] = q.cpu().contiguous(), scale.cpu().contiguous()
                out[f"{name}.comfy_quant"] = comfy_quant_tensor()
                if n_lin % 12 == 0:
                    x = torch.randn(512, w.shape[1], device="cuda", dtype=torch.bfloat16)
                    ref = F.linear(x, w)
                    got = torch.ops.comfy_kitchen.int8_linear(x, q, scale, None, 2, True, GROUP_SIZE)
                    cosines.append((name, F.cosine_similarity(ref.float().flatten(), got.float().flatten(), 0).item()))
                n_lin += 1
            elif key in (EMBED, "encoder.embed_tokens.weight"):
                if EMBED in out:
                    continue   # tied pair: keep one
                from comfy_kitchen.tensor.int8 import TensorWiseINT8Layout
                q, params = TensorWiseINT8Layout.quantize(t.to("cuda", torch.bfloat16), is_weight=True, per_channel=True)
                out[EMBED] = q.cpu().contiguous()
                out["shared.weight_scale"] = params.scale.float().reshape(-1, 1).cpu().contiguous()
                out["shared.comfy_quant"] = torch.tensor(list(json.dumps({"format": "int8_tensorwise"}).encode()),
                                                         dtype=torch.uint8)
            else:
                out[key] = t.to(torch.bfloat16).contiguous()
    print(f"[convert] {os.path.basename(path)}: {n_lin} linears so far, {time.perf_counter() - t0:.0f} s", flush=True)

if os.path.isfile(spiece):
    with open(spiece, "rb") as f:
        out["spiece_model"] = torch.frombuffer(bytearray(f.read()), dtype=torch.uint8)
else:
    print(f"[convert] no spiece.model at {spiece}: the file works here but not in ComfyUI's CLIPLoader")

save_file(out, args.dst, metadata={"format": "pt"})
print(f"[convert] {n_lin} ConvRot linears + INT8 embedding -> {args.dst} ({os.path.getsize(args.dst) / 2 ** 30:.2f} GB)")
for name, c in cosines:
    print(f"[check] {name}: cos {c:.5f}")
print(f"[check] min cos {min(c for _, c in cosines):.5f} over {len(cosines)} sampled layers")
