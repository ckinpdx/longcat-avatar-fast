"""One LongCat DiT block's big linears at 38k tokens: their INT8 (dequant + bf16 matmul), plain bf16, ConvRot W8A8."""
import time, torch, torch.nn.functional as F
import comfy_kitchen  # noqa: F401
from comfy_kitchen.tensor.int8 import TensorWiseINT8Layout

T = 38000
SHAPES = {"attn.qkv": (4096, 12288), "attn.proj": (4096, 4096), "cross_attn.q": (4096, 4096),
          "cross_attn.proj": (4096, 4096), "audio_cross.q": (4096, 4096), "audio_cross.proj": (4096, 4096),
          "ffn.w1": (4096, 11008), "ffn.w3": (4096, 11008), "ffn.w2": (11008, 4096)}

def bench(fn, n=10):
    fn(); torch.cuda.synchronize(); t = time.perf_counter()
    for _ in range(n): fn()
    torch.cuda.synchronize(); return (time.perf_counter() - t) / n

tot = {"their_int8": 0, "bf16": 0, "convrot": 0}
for name, (K, N) in SHAPES.items():
    x = torch.randn(T, K, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(N, K, device="cuda", dtype=torch.bfloat16) * 0.02
    b = torch.zeros(N, device="cuda", dtype=torch.bfloat16)
    s = w.abs().amax(1).clamp(min=1e-8) / 127.0
    wi8 = (w / s[:, None]).round().clamp(-128, 127).to(torch.int8)
    their = lambda: F.linear(x, wi8.to(torch.bfloat16) * s.to(torch.bfloat16)[:, None], b)
    plain = lambda: F.linear(x, w, b)
    q, p = TensorWiseINT8Layout.quantize(w, is_weight=True, per_channel=True, convrot=True, convrot_groupsize=256)
    sc = p.scale.float()
    cr = lambda: torch.ops.comfy_kitchen.int8_linear(x, q, sc, b, 2, True, 256)
    ref = plain()
    cos = F.cosine_similarity(ref.float().flatten(), cr().float().flatten(), dim=0).item()
    r = {"their_int8": bench(their), "bf16": bench(plain), "convrot": bench(cr)}
    for k in r: tot[k] += r[k]
    print(f"{name:16s} their {r['their_int8']*1e3:6.1f} ms  bf16 {r['bf16']*1e3:6.1f} ms  convrot {r['convrot']*1e3:6.1f} ms  cos {cos:.5f}", flush=True)
print("per block  :", {k: f"{v*1e3:.1f} ms" for k, v in tot.items()})
print("x48 blocks :", {k: f"{v*48:.2f} s" for k, v in tot.items()})
