"""FA2 vs SageAttention at LongCat's self-attention shape (B=1, H=32, D=128, S tokens). Times one call, x48 blocks."""
import sys, time, torch
from flash_attn import flash_attn_func
from sageattention import sageattn

def bench(fn, n=5):
    fn(); torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(n): fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t) / n

for S in (25 * 1520, 25 * 1024):   # 608x640 bucket now; ~512x512
    q, k, v = (torch.randn(1, 32, S, 128, device="cuda", dtype=torch.bfloat16) for _ in range(3))
    qs, ks, vs = (t.transpose(1, 2).contiguous() for t in (q, k, v))       # FA2 wants B S H D
    ref = flash_attn_func(qs, ks, vs).transpose(1, 2)
    out = sageattn(q, k, v, tensor_layout="HND", is_causal=False)
    cos = torch.nn.functional.cosine_similarity(ref.float().flatten(), out.float().flatten(), dim=0).item()
    fa = bench(lambda: flash_attn_func(qs, ks, vs))
    sa = bench(lambda: sageattn(q, k, v, tensor_layout="HND", is_causal=False))
    print(f"S={S}: FA2 {fa*1e3:.0f} ms, sage {sa*1e3:.0f} ms ({fa/sa:.2f}x) | x48 blocks: FA2 {fa*48:.1f} s, sage {sa*48:.1f} s | cos {cos:.5f}", flush=True)
