import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL = "Qwen/Qwen3-0.6B"
tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to("cuda").eval()

# ---- 1. how we judge quality
TEXT = """The city library opened its doors in the early morning, and the first visitors were students
looking for a quiet place to study before their exams. On the second floor, an old man read the
newspaper while a young mother helped her daughter choose picture books. The librarian, who had worked
there for more than twenty years, knew most of the regular visitors by name. She often said that a
library is not only a building full of books, but a place where people of every age can learn, rest,
and meet each other. In the afternoon, a group of children arrived for a reading class. They sat in a
circle on the carpet and listened to a story about a fox who wanted to cross a wide river. When the
story ended, the children asked many questions, and the teacher answered each one with patience. By
evening, the rooms were calm again, the lights were turned down, and the books waited for the next day."""
PROMPT = "The three primary colors are"

ids = tok(TEXT, return_tensors="pt").input_ids.cuda()

@torch.no_grad()
def perplexity():                         # how "surprised" the model is by normal text. lower = better
    return torch.exp(model(ids, labels=ids).loss).item()

@torch.no_grad()
def write():                              # greedy answer to a fixed prompt
    x = tok(PROMPT, return_tensors="pt").to("cuda")
    out = model.generate(**x, max_new_tokens=20, do_sample=False)
    return tok.decode(out[0][x.input_ids.shape[1]:], skip_special_tokens=True).replace("\n", " ")


# ---- 2. the quantizers (quantize → then turn back into BF16 so we can see what was lost)
def quant_int(W, bits, block):
    qmax = 2 ** (bits - 1) - 1                              # INT4 → 7, INT8 → 127
    g = W.float().reshape(1, -1) if block is None else W.float().reshape(-1, block)
    scale = (g.abs().amax(dim=1, keepdim=True) / qmax).clamp(min=1e-12)   # biggest number just fits
    q = (g / scale).round().clamp(-qmax - 1, qmax)          # the small integers we would store
    return (q * scale).reshape(W.shape).to(W.dtype)          # read back

def quant_fp8(W, block):
    g = W.float().reshape(1, -1) if block is None else W.float().reshape(-1, block)
    scale = (g.abs().amax(dim=1, keepdim=True) / 448).clamp(min=1e-12)    # 448 = FP8 max
    q = (g / scale).to(torch.float8_e4m3fn).float()
    return (q * scale).reshape(W.shape).to(W.dtype)


# ---- 3. every weight matrix inside the 28 blocks (embedding / lm_head stay BF16)
linears = [m for m in model.model.layers.modules() if isinstance(m, torch.nn.Linear)]
original = [m.weight.data.clone() for m in linears]
n_params = sum(w.numel() for w in original)
print(f"{len(linears)} weight matrices, {n_params / 1e6:.0f}M numbers")

# ---- 4. outliers: look at one real matrix
W = original[-1].float()                                    # last block's MLP down_proj
print(f"one matrix: typical |w| = {W.abs().mean():.4f}   biggest |w| = {W.abs().max():.4f}"
      f"   → biggest is {W.abs().max() / W.abs().mean():.0f}× the typical\n")


# ---- 5. run every setting, one change at a time
runs = [
    ("BF16 original",   16, None, None),
    ("FP8  per-tensor",  8, None, lambda W: quant_fp8(W, None)),
    ("INT8 per-tensor",  8, None, lambda W: quant_int(W, 8, None)),
    ("INT4 per-tensor",  4, None, lambda W: quant_int(W, 4, None)),
    ("INT4 block 32",    4, 32,   lambda W: quant_int(W, 4, 32)),
    ("INT3 block 32",    3, 32,   lambda W: quant_int(W, 3, 32)),
    ("INT2 block 32",    2, 32,   lambda W: quant_int(W, 2, 32)),
]

print(f"{'setting':16} {'size MB':>8} {'zeros %':>8} {'perplexity':>11}   answer")
for name, bits, block, fn in runs:
    zeros = 0
    for m, w in zip(linears, original):
        m.weight.data = w.clone() if fn is None else fn(w)
        zeros += (m.weight.data == 0).sum().item()
    scales = 0 if block is None else n_params / block * 2   # one 2-byte scale per block
    size_mb = (n_params * bits / 8 + scales) / 1e6
    print(f"{name:16} {size_mb:8.0f} {100 * zeros / n_params:8.1f} {perplexity():11.2f}   {write()}")

for m, w in zip(linears, original):                          # put the real weights back
    m.weight.data = w
