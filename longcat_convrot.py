"""INT8 ConvRot linears for the LongCat-Video-Avatar DiT, on comfy-kitchen kernels (the same layer I wrote for
SoulX-FlashHead's INT8 path).

ConvRot rotates each Linear's input dimension with a group-wise Hadamard transform before row-wise INT8
quantization. The weight is rotated offline (at conversion); the activation is rotated and quantized on the fly
inside comfy-kitchen's int8_linear, so the matmul runs on INT8 tensor cores (3.4x bf16 on a 5090 at 38k tokens).

Checkpoint layout per quantized Linear (ComfyUI's native layout):
    <name>.weight        int8  [N, K]  rotated, row-wise quantized
    <name>.weight_scale  fp32  [N, 1]  per-row scale
    <name>.bias          bf16  [N]
    <name>.comfy_quant   uint8 JSON {"format": "int8_tensorwise", "convrot": true, "convrot_groupsize": 256}
"""
import json
import os
import re

import torch
import torch.nn as nn
from safetensors.torch import load_file

import comfy_kitchen  # noqa: F401  registers torch.ops.comfy_kitchen.*

QUANT_FORMAT = "int8_tensorwise"
GROUP_SIZE = 256
SUBFOLDER = "base_model_convrot"
# Every Linear inside the 48 DiT blocks (attention, both cross-attentions, FFN, adaLN). All in_features
# (512, 768, 4096, 11008) divide by 256. Top-level embedders and final_layer stay bf16.
QUANT_LINEAR_PATTERN = re.compile(r"^blocks\.\d+\.")
_DTYPE_TO_CODE = {torch.float32: 0, torch.float16: 1, torch.bfloat16: 2}


def comfy_quant_tensor(group_size=GROUP_SIZE):
    conf = {"format": QUANT_FORMAT, "convrot": True, "convrot_groupsize": group_size}
    return torch.tensor(list(json.dumps(conf).encode("utf-8")), dtype=torch.uint8)


def quantize_weight(weight, group_size=GROUP_SIZE):
    """Rotate and row-wise quantize one [N, K] weight. Returns (int8 [N, K], fp32 [N, 1] scale)."""
    from comfy_kitchen.tensor.int8 import TensorWiseINT8Layout
    qdata, params = TensorWiseINT8Layout.quantize(
        weight, is_weight=True, per_channel=True, convrot=True, convrot_groupsize=group_size
    )
    return qdata, params.scale.float().reshape(-1, 1)


class QuantizedLinear(nn.Module):
    """Named QuantizedLinear on purpose: LongCat's LoRA code only attaches to classes named Linear or
    QuantizedLinear, and reads in_features/out_features. The LoRAs then run as a bf16 side path on top."""

    def __init__(self, in_features, out_features, bias=True, group_size=GROUP_SIZE, device="meta"):
        super().__init__()
        if in_features % group_size != 0:
            raise ValueError(f"in_features {in_features} is not a multiple of convrot group size {group_size}")
        self.in_features = in_features
        self.out_features = out_features
        self.group_size = group_size
        self.register_buffer("weight", torch.empty(out_features, in_features, dtype=torch.int8, device=device))
        self.register_buffer("weight_scale", torch.empty(out_features, 1, dtype=torch.float32, device=device))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, device=device, dtype=torch.bfloat16), requires_grad=False)
        else:
            self.register_parameter("bias", None)

    def _apply(self, fn, recurse=True):
        # Module.to(dtype=...) casts every floating tensor; the per-row scale must stay fp32.
        scale = self._buffers.pop("weight_scale")
        super()._apply(fn, recurse)
        self._buffers["weight_scale"] = scale.to(self.weight.device)
        return self

    def forward(self, x):
        return torch.ops.comfy_kitchen.int8_linear(
            x, self.weight, self.weight_scale, self.bias, _DTYPE_TO_CODE[x.dtype], True, self.group_size
        )

    def extra_repr(self):
        return (f"in_features={self.in_features}, out_features={self.out_features}, "
                f"bias={self.bias is not None}, convrot_groupsize={self.group_size}")


def _set_module(model, name, module):
    parent = model
    *path, last = name.split(".")
    for part in path:
        parent = getattr(parent, part)
    setattr(parent, last, module)


def load_convrot_dit(checkpoint_dir, subfolder=SUBFOLDER, **kwargs):
    """Build the avatar DiT on the meta device, swap block Linears for ConvRot ones, and load the converted
    checkpoint straight into it (no 54 GB fp32 CPU initialisation like the stock INT8 loader)."""
    from longcat_video.modules.avatar.longcat_video_dit_avatar import LongCatVideoAvatarTransformer3DModel

    model_dir = os.path.join(checkpoint_dir, subfolder)
    with open(os.path.join(model_dir, "config.json")) as f:
        config = json.load(f)
    for key in ("_class_name", "architectures", "_diffusers_version", "model_max_length"):
        config.pop(key, None)
    config.update(kwargs)

    with torch.device("meta"):
        model = LongCatVideoAvatarTransformer3DModel(**config)

    state_dict = {}
    with open(os.path.join(model_dir, "diffusion_pytorch_model.safetensors.index.json")) as f:
        for shard in sorted(set(json.load(f)["weight_map"].values())):
            state_dict.update(load_file(os.path.join(model_dir, shard), device="cpu"))

    quantized = {k[: -len(".comfy_quant")] for k in state_dict if k.endswith(".comfy_quant")}
    for name in quantized:
        conf = json.loads(bytes(state_dict.pop(f"{name}.comfy_quant").tolist()).decode("utf-8"))
        old = model.get_submodule(name)
        _set_module(model, name, QuantizedLinear(old.in_features, old.out_features, bias=old.bias is not None,
                                                 group_size=int(conf.get("convrot_groupsize", GROUP_SIZE))))

    model.load_state_dict(state_dict, strict=True, assign=True)
    for module in model.modules():
        if isinstance(module, QuantizedLinear):
            continue
        for pname, param in module.named_parameters(recurse=False):
            if param.dtype == torch.float32:
                setattr(module, pname, nn.Parameter(param.data.to(torch.bfloat16), requires_grad=False))

    leftovers = [n for n, t in list(model.named_parameters()) + list(model.named_buffers()) if t.is_meta]
    leftovers += [f"{mn}.{an}" for mn, m in model.named_modules() for an, v in vars(m).items()
                  if isinstance(v, torch.Tensor) and v.is_meta]
    if leftovers:
        raise RuntimeError(f"tensors still on meta after load: {leftovers[:10]}")
    print(f"[convrot] loaded {len(quantized)} ConvRot INT8 linears from {model_dir}", flush=True)
    return model.eval()
