"""Convert LongCat-Video-Avatar 1.6's bf16 base DiT to ConvRot INT8 (group 256), shard by shard.

    python convert_convrot.py <path>/LongCat-Video-Avatar-1.6

Reads <ckpt>/base_model, writes <ckpt>/base_model_convrot. Block Linears (2-D weights under blocks.N.) are rotated
and quantized; everything else is copied as-is. Prints the output cosine vs bf16 for a sample of layers on
random activations, as a quantization sanity check.
"""
import json
import os
import shutil
import sys
import time

import torch
import torch.nn.functional as F
from safetensors import safe_open
from safetensors.torch import save_file

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from longcat_convrot import GROUP_SIZE, QUANT_LINEAR_PATTERN, SUBFOLDER, comfy_quant_tensor, quantize_weight  # noqa: E402

ckpt = sys.argv[1]
src, dst = os.path.join(ckpt, "base_model"), os.path.join(ckpt, SUBFOLDER)
os.makedirs(dst, exist_ok=True)
shutil.copy(os.path.join(src, "config.json"), dst)
with open(os.path.join(src, "diffusion_pytorch_model.safetensors.index.json")) as f:
    shards = sorted(set(json.load(f)["weight_map"].values()))

weight_map, n_quant, cosines, t0 = {}, 0, [], time.perf_counter()
for i, shard in enumerate(shards):
    out = {}
    with safe_open(os.path.join(src, shard), "pt", device="cpu") as f:
        for key in f.keys():
            t = f.get_tensor(key)
            name = key[: -len(".weight")] if key.endswith(".weight") else None
            if name and t.ndim == 2 and QUANT_LINEAR_PATTERN.match(name) and t.shape[1] % GROUP_SIZE == 0:
                w = t.to("cuda", torch.bfloat16)
                q, scale = quantize_weight(w)
                out[f"{name}.weight"] = q.cpu().contiguous()
                out[f"{name}.weight_scale"] = scale.cpu().contiguous()
                out[f"{name}.comfy_quant"] = comfy_quant_tensor()
                if n_quant % 37 == 0:   # spot-check ~1 in 37 layers on random activations
                    x = torch.randn(512, w.shape[1], device="cuda", dtype=torch.bfloat16)
                    ref = F.linear(x, w)
                    got = torch.ops.comfy_kitchen.int8_linear(x, q, scale, None, 2, True, GROUP_SIZE)
                    cosines.append((name, F.cosine_similarity(ref.float().flatten(), got.float().flatten(), 0).item()))
                n_quant += 1
            else:
                out[key] = t.contiguous()
    name = f"diffusion_pytorch_model-{i + 1:05d}-of-{len(shards):05d}.safetensors"
    save_file(out, os.path.join(dst, name), metadata={"format": "pt"})
    weight_map.update({k: name for k in out})
    print(f"[convert] {shard} -> {name}: {n_quant} linears so far, {time.perf_counter() - t0:.0f} s", flush=True)

with open(os.path.join(dst, "diffusion_pytorch_model.safetensors.index.json"), "w") as f:
    json.dump({"metadata": {"quantization": "int8 convrot group 256 (comfy-kitchen)"}, "weight_map": weight_map}, f, indent=1)
print(f"[convert] done: {n_quant} ConvRot linears")
for name, c in cosines:
    print(f"[check] {name}: cos {c:.5f}")
print(f"[check] min cos {min(c for _, c in cosines):.5f} over {len(cosines)} sampled layers")
