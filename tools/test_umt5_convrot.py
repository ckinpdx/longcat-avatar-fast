"""INT8 ConvRot UMT5 (run_fast's loader) vs the bf16 reference encoder: prompt-embedding error and timing.

    python tools/test_umt5_convrot.py <LongCat-Video dir> <umt5_xxl_int8_convrot.safetensors> [fp8 file to compare too]
"""
import json, os, sys, time, types
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, UMT5EncoderModel

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
base, files = sys.argv[1], sys.argv[2:]
src = open(os.path.join(HERE, "run_fast.py"), encoding="utf-8").read()
block = src[src.index("TEXT_ENCODER_FILE = os.environ"):src.index("    demo.transformers_UMT5EncoderModel = demo.UMT5EncoderModel")]
block = block.replace("if TEXT_ENCODER_FILE:\n", "if True:\n")

PROMPTS = [
    "A woman with long dark hair, pink eye makeup and glossy lips, wearing a black jacket and silver chain necklaces, "
    "stands under teal stage lights against a hazy blue backdrop. She is talking to the camera in a relaxed, warm way, "
    "speaking with small natural head movements, blinks and the occasional smile. The camera does not move.",
    "An elderly fisherman in a yellow raincoat sits on a wooden dock at dawn, laughing as he tells a story, gulls "
    "circling over grey water behind him.",
]
tok = AutoTokenizer.from_pretrained(base, subfolder="tokenizer")
inp = tok(PROMPTS, padding="max_length", max_length=512, truncation=True, return_tensors="pt")
n_valid = inp.attention_mask.sum(1).tolist()

def encode(model, device):
    with torch.no_grad():
        return model.to(device)(inp.input_ids.to(device), inp.attention_mask.to(device)).last_hidden_state.float().cpu()

def report(name, ref, out):
    for i, n in enumerate(n_valid):
        cos = F.cosine_similarity(ref[i, :n], out[i, :n], dim=-1)
        rel = ((ref[i, :n] - out[i, :n]).norm() / ref[i, :n].norm()).item()
        print(f"{name} prompt {i} ({n} tokens): rel err {rel:.4f}, token cosine min {cos.min():.5f} mean {cos.mean():.5f}")

ref_m = UMT5EncoderModel.from_pretrained(base, subfolder="text_encoder", torch_dtype=torch.bfloat16).eval()
ref = encode(ref_m, "cuda"); del ref_m; torch.cuda.empty_cache()
for path in files:
    ns = {"os": os, "torch": torch, "F": F, "json": json, "time": time, "TEXT_ENCODER_FILE": path,
          "demo": types.SimpleNamespace(transformers_UMT5EncoderModel=UMT5EncoderModel)}
    exec(block.replace('TEXT_ENCODER_FILE = os.environ.get("LONGCAT_TEXT_ENCODER")\n', ""), ns)
    m = ns["_ComfyUMT5"].from_pretrained(base, subfolder="text_encoder")
    dev = "cuda" if m._ck_gpu_encode else "cpu"
    torch.cuda.synchronize(); t = time.perf_counter()
    out = encode(m, dev)
    torch.cuda.synchronize(); print(f"{os.path.basename(path)}: encode on {dev} incl. transfer {time.perf_counter() - t:.2f} s")
    report(os.path.basename(path)[:28], ref, out)
    del m; torch.cuda.empty_cache()
