"""Fused LoRA add (lora_add_) vs LongCat's original formula, for plain and parallel (qkv) lora_up."""
import os, sys, types, torch
sys.path.insert(0, os.environ.get("LONGCAT_REPO", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "LongCat-Video")))
from longcat_video.modules.lora_utils import LoRAUPParallel

src = open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "run_fast.py")).read()
start = src.index("def lora_add_"); end = src.index("def _multi_lora_forward")
ns = {"torch": torch, "os": os}; exec(src[start:end], ns); lora_add_ = ns["lora_add_"]

torch.manual_seed(0)
for n_sep, N in ((1, 4096), (3, 12288)):
    K, r, T = 4096, 128, 4000
    lora = types.SimpleNamespace(multiplier=0.7, alpha_scale=torch.tensor(64 / 128, device="cuda"), use_lora=True)
    lora._ck_alpha = float(lora.alpha_scale)   # run_fast sets this when the LoRAs are enabled
    lora.lora_down = torch.nn.Linear(K, n_sep * r, bias=False).cuda().bfloat16()
    lora.lora_up = (LoRAUPParallel([torch.nn.Linear(r, N // n_sep, bias=False) for _ in range(n_sep)]) if n_sep > 1
                    else torch.nn.Linear(r, N, bias=False)).cuda().bfloat16()
    x = torch.randn(1, T, K, device="cuda", dtype=torch.bfloat16)
    base = torch.randn(1, T, N, device="cuda", dtype=torch.bfloat16)
    with torch.no_grad():
        ref = base + lora.lora_up(lora.lora_down(x)).to(x.dtype) * lora.multiplier * lora.alpha_scale
        got = lora_add_(base.clone(), x, lora)
        delta_ref, delta_got = (ref - base).float(), (got - base).float()
    cos = torch.nn.functional.cosine_similarity(delta_ref.flatten(), delta_got.flatten(), 0).item()
    print(f"n_seperate={n_sep}: max |diff| {(ref - got).abs().max().item():.4f}, LoRA-delta cos {cos:.6f}")
