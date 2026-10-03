import torch 
from transformers import AutoTokenizer, AutoModelForCausalLM
import  time

MODEL = "Qwen/Qwen3-0.6B"

# disk -> RAM -> VRAM
tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to("cuda")


def sample(logits, temp, top_k, top_p):
    if temp == 0:
        return int(logits.argmax())                  # greedy: just the best

    probs = torch.softmax(logits / temp, dim=-1)
    vals, idx = probs.sort(descending=True)          # best first

    vals, idx = vals[:top_k], idx[:top_k]            # top-k: keep k best

    keep = (vals.cumsum(0) - vals) < top_p           # top-p: keep until we pass p
    vals, idx = vals[keep], idx[keep]

    vals = vals / vals.sum()                         # renormalize to 100%
    choice = torch.multinomial(vals, 1)              # random pick, weighted by probability
    return int(idx[choice])

c = model.config

layers   = c.num_hidden_layers        # how many transformer blocks
hidden   = c.hidden_size              # each token = this many numbers
q_heads  = c.num_attention_heads      # query heads
kv_heads = c.num_key_value_heads      # KV heads (fewer = GQA)
head_dim = c.head_dim                 # numbers per head
vocab    = c.vocab_size               # how many tokens the model knows

# total parameters and their size in memory
params = sum(p.numel() for p in model.parameters())
weight_bytes = sum(p.numel() * p.element_size() for p in model.parameters())

# KV cache per token: K and V (2) x every layer x every KV head x head_dim x 2 bytes (bf16)
kv_per_token = 2 * layers * kv_heads * head_dim * 2

print("parameters  ", round(params / 1e9, 2), "B")
print("weights size", round(weight_bytes / 1e9, 2), "GB")
print("layers      ", layers)
print("hidden size ", hidden)
print("q heads     ", q_heads)
print("kv heads    ", kv_heads, "->", q_heads // kv_heads, "query heads share 1 KV head")
print("head_dim    ", head_dim)
print("vocab       ", vocab)
print("KV per token", kv_per_token / 1024, "KB")

# ---- Part 2: chat template + tokens
messages = [{"role": "user", "content": "Give me one random name for a cat."}]

# wrap the message in the model's special chat format
text = tok.apply_chat_template(messages, tokenize=False,
                               add_generation_prompt=True, enable_thinking=False)
print(text)


# text -> token IDs, as a tensor on the GPU
ids = tok(text, return_tensors="pt").input_ids.to("cuda")
print("shape:", ids.shape)          # [batch, number of tokens]

# show each ID next to the piece of text it stands for
for i in ids[0].tolist():
    print(i, repr(tok.decode([i])))


# ---- Part 3: prefill

# hooks = small functions that run when a layer finishes, so we can see its output shape


def show(name):
    def hook(module, inp, out):
        o = out[0] if isinstance(out, tuple) else out
        print(f"{name:12} {tuple(o.shape)}")
    return hook

hooks = []
hooks.append(model.model.embed_tokens.register_forward_hook(show("embedding")))
hooks.append(model.model.layers[0].register_forward_hook(show("block 0")))
hooks.append(model.model.layers[-1].register_forward_hook(show("block 27")))
hooks.append(model.lm_head.register_forward_hook(show("LM head")))


with torch.no_grad():                       # inference only: don't store gradients
    model(ids[:, :4])                       # warm-up: first GPU call is slow (startup)

    torch.cuda.synchronize()                # wait until the GPU is idle
    t = time.time()
    out = model(ids, use_cache=True)        # PREFILL: all 20 tokens at once
    torch.cuda.synchronize()                # wait until the GPU has really finished
    prefill = time.time() - t

N = ids.shape[1]
cache = out.past_key_values
print("logits shape :", tuple(out.logits.shape))
print("prefill time :", round(prefill * 1000, 1), "ms")
print("tokens/sec   :", round(N / prefill))
print("KV cache     :", cache.get_seq_length(), "tokens =",
      round(cache.get_seq_length() * kv_per_token / 1e6, 2), "MB")


# ---- Part 4: sampling

logits = out.logits[0, -1].float()     # last position only -> 151936 scores
print("logits:", logits.shape)

for T in [0.1, 0.5, 1.0, 1.5]:
    probs = torch.softmax(logits / T, dim=-1)    # temperature divides the scores, then softmax
    top = probs.topk(5)                          # the 5 most likely tokens
    row = ""
    for p, i in zip(top.values.tolist(), top.indices.tolist()):
        row += f"{tok.decode([i])!r} {p*100:.1f}%   "
    print(f"T={T}:", row)

# ---- Part 5: decode

next_id = sample(logits, 0.7, 20, 0.95)        # token 1 came from PREFILL's logits
generated = [next_id]
times = []

for step in range(15):
    if step == 1:                              # show shapes for the first decode step only
        for h in hooks:
            h.remove()

    new = torch.tensor([[next_id]], device="cuda")    # just ONE token, shape (1, 1)

    torch.cuda.synchronize()
    t = time.time()
    with torch.no_grad():
        o = model(new, past_key_values=cache, use_cache=True)   # reuse the cache
    torch.cuda.synchronize()
    dt = time.time() - t
    times.append(dt)

    cache = o.past_key_values
    next_id = sample(o.logits[0, -1].float(), 0.7, 20, 0.95)
    generated.append(next_id)

    kv = cache.get_seq_length()
    print(f"step {step+1:2}  {tok.decode([next_id])!r:12} {dt*1000:5.1f} ms   KV {kv} tokens = {kv*kv_per_token/1e6:.2f} MB")

    if next_id == tok.eos_token_id:
        break

print("\nanswer:", tok.decode(generated, skip_special_tokens=True))



# ---- Part 6: memory moving

avg = sum(times[1:]) / len(times[1:])          # skip step 1 (slower, includes hook printing)
bandwidth = weight_bytes / avg / 1e9           # GB read per second

print("\navg decode step :", round(avg * 1000, 1), "ms ->", round(1 / avg), "tok/s")
print("weights read    :", round(weight_bytes / 1e9, 2), "GB every step")
print("bandwidth used  :", round(bandwidth), "GB/s   (laptop max ~256 GB/s)")
print("ideal decode    :", round(256 / (weight_bytes / 1e9)), "tok/s if only memory mattered")
print("prefill vs decode:", round(N / prefill), "tok/s vs", round(1 / avg), "tok/s")
