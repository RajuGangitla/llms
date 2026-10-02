import argparse
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

p = argparse.ArgumentParser()
p.add_argument("--prompt", default="Explain why the sky is blue.")
p.add_argument("--long", type=int, default=1, help="repeat the prompt N times to make prefill bigger")
p.add_argument("--steps", type=int, default=15)
p.add_argument("--temp", type=float, default=0.7)
p.add_argument("--top-k", type=int, default=20)
p.add_argument("--top-p", type=float, default=0.95)
args = p.parse_args()

MODEL = "Qwen/Qwen3-0.6B"
DEV = "cuda"
LAPTOP_BANDWIDTH_GBS = 256


def section(title):
    print("\n" + "=" * 70 + f"\n{title}\n" + "=" * 70)


def timed(fn):
    torch.cuda.synchronize()
    t = time.time()
    with torch.no_grad():
        out = fn()
    torch.cuda.synchronize()
    return out, time.time() - t


def sample(logits, temp, top_k, top_p):
    if temp == 0:
        return int(logits.argmax())
    probs = torch.softmax(logits / temp, -1)
    vals, idx = probs.sort(descending=True)
    vals, idx = vals[:top_k], idx[:top_k]
    keep = (vals.cumsum(0) - vals) < top_p
    vals, idx = vals[keep], idx[keep]
    return int(idx[torch.multinomial(vals / vals.sum(), 1)])


# ---------------------------------------------------------------- 1. CONFIG
section("1. CONFIG  (the model's config.json)")
tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to(DEV).eval()
c = model.config
head_dim = getattr(c, "head_dim", None) or c.hidden_size // c.num_attention_heads
params = sum(t.numel() for t in model.parameters())
weight_bytes = sum(t.numel() * t.element_size() for t in model.parameters())
kv_per_token = 2 * c.num_hidden_layers * c.num_key_value_heads * head_dim * 2

print(f"architecture     {c.architectures[0]}")
print(f"parameters       {params/1e9:.2f} B  ({weight_bytes/1e9:.2f} GB in bf16)")
print(f"layers (blocks)  {c.num_hidden_layers}")
print(f"hidden size      {c.hidden_size}   (each token = {c.hidden_size} numbers)")
print(f"query heads      {c.num_attention_heads}")
print(f"KV heads         {c.num_key_value_heads}   (GQA: {c.num_attention_heads // c.num_key_value_heads} query heads share 1 KV head)")
print(f"head dim         {head_dim}")
print(f"MLP size         {c.intermediate_size}")
print(f"vocab size       {c.vocab_size}")
print(f"KV cache/token   2 x {c.num_hidden_layers} layers x {c.num_key_value_heads} heads x {head_dim} x 2 bytes "
      f"= {kv_per_token/1024:.0f} KB")

# ---------------------------------------------------------------- 2. CHAT TEMPLATE
section("2. CHAT TEMPLATE + TOKENS  (what the model actually sees)")
prompt = " ".join([args.prompt] * args.long)
text = tok.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False,
                               add_generation_prompt=True, enable_thinking=False)
ids = tok(text, return_tensors="pt").input_ids.to(DEV)
N = ids.shape[1]
print(f"template text:\n{text[:400]}{' ...' if len(text) > 400 else ''}")
print(f"\n{N} tokens. First 12:")
for i, t in zip(ids[0, :12].tolist(), tok.convert_ids_to_tokens(ids[0, :12])):
    print(f"  {i:>7}  {t!r}")

# hooks: record the hidden-state shape coming out of every layer
shapes = []
model.model.embed_tokens.register_forward_hook(lambda m, i, o: shapes.append(("embedding", tuple(o.shape))))
for n, layer in enumerate(model.model.layers):
    layer.register_forward_hook(
        lambda m, i, o, n=n: shapes.append((f"block {n:2d}", tuple((o[0] if isinstance(o, tuple) else o).shape))))
model.lm_head.register_forward_hook(lambda m, i, o: shapes.append(("LM head", tuple(o.shape))))

timed(lambda: model(ids[:, :4]))  # warm-up so CUDA startup isn't counted

# ---------------------------------------------------------------- 3. PREFILL
section(f"3. PREFILL  (all {N} prompt tokens in ONE pass -> builds the KV cache)")
shapes.clear()
out, prefill_s = timed(lambda: model(ids, use_cache=True))
for name, shape in shapes:
    print(f"  {name:10}  shape {shape}")
cache = out.past_key_values
print(f"\nprefill time     {prefill_s*1000:.1f} ms  ->  {N/prefill_s:.0f} tokens/s  (= TTFT on the GPU)")
print(f"compute done     2 x {params/1e9:.2f}B params x {N} tokens = {2*params*N/1e12:.3f} TFLOP "
      f"-> {2*params*N/prefill_s/1e12:.1f} TFLOPS achieved")
print(f"KV cache         {cache.get_seq_length()} tokens x {kv_per_token/1024:.0f} KB = "
      f"{cache.get_seq_length()*kv_per_token/1e6:.2f} MB")

# ---------------------------------------------------------------- 4. SAMPLING
section("4. SAMPLING  (logits for the next token -> how temperature changes the odds)")
logits = out.logits[0, -1].float()
print(f"logits vector: {logits.shape[0]} scores, one per vocab token\n")
for T in [0.0, 0.5, 1.0, 1.5]:
    probs = torch.softmax(logits / T, -1) if T else torch.nn.functional.one_hot(logits.argmax(), logits.shape[0]).float()
    top = probs.topk(5)
    row = "  ".join(f"{tok.decode([i])!r} {v*100:4.1f}%" for v, i in zip(top.values.tolist(), top.indices.tolist()))
    print(f"T={T:<4} {row}")
probs = torch.softmax(logits / args.temp, -1).sort(descending=True).values
after_k = probs[:args.top_k]
after_p = int(((after_k.cumsum(0) - after_k) < args.top_p).sum())
print(f"\nT={args.temp}: top-k={args.top_k} keeps {len(after_k)} candidates, then top-p={args.top_p} keeps {after_p}")

# ---------------------------------------------------------------- 5. DECODE
section(f"5. DECODE  (one token per pass, KV cache grows by 1 each step)")
next_id = sample(logits, args.temp, args.top_k, args.top_p)
generated = [next_id]
print(f"prefill already produced token 1: {tok.decode([next_id])!r}\n")
step_times = []
for step in range(args.steps):
    shapes.clear()
    o, dt = timed(lambda: model(torch.tensor([[next_id]], device=DEV), past_key_values=cache, use_cache=True))
    cache = o.past_key_values
    next_id = sample(o.logits[0, -1].float(), args.temp, args.top_k, args.top_p)
    generated.append(next_id)
    step_times.append(dt)
    if step == 0:
        print(f"  every block's shape this step: {shapes[1][1]}   <- just 1 token, vs {N} in prefill\n")
    kv = cache.get_seq_length()
    print(f"  step {step+1:2d}  {tok.decode([next_id])!r:14} {dt*1000:5.1f} ms   "
          f"KV cache {kv} tokens = {kv*kv_per_token/1e6:.2f} MB")
    if next_id == tok.eos_token_id:
        break

# ---------------------------------------------------------------- 6. MEMORY MOVING
section("6. MEMORY MOVING  (is decode memory-bound?)")
avg = sum(step_times[1:]) / max(1, len(step_times) - 1)
print(f"avg decode step  {avg*1000:.1f} ms  ->  {1/avg:.0f} tokens/s")
print(f"weights read     {weight_bytes/1e9:.2f} GB per step")
print(f"bandwidth used   {weight_bytes/avg/1e9:.0f} GB/s   (laptop GPU max ~{LAPTOP_BANDWIDTH_GBS} GB/s)")
print(f"ideal decode     {LAPTOP_BANDWIDTH_GBS/(weight_bytes/1e9):.0f} tokens/s if memory was the only limit")
print(f"prefill vs decode: {N/prefill_s:.0f} tok/s vs {1/avg:.0f} tok/s")

section("OUTPUT")
print(args.prompt if args.long == 1 else f"[prompt x{args.long}]", "->", tok.decode(generated, skip_special_tokens=True))
