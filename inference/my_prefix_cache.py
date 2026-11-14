
import torch, time
from transformers import AutoTokenizer, AutoModelForCausalLM
import copy 

tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B", dtype=torch.bfloat16).to("cuda")

# a long system prompt (~3,000 tokens), like a real chatbot's instructions
rules = "You are a support assistant for an online bookstore. Always be polite, answer briefly, and never share customer data. "
system = rules * 120
question = " Question: How do I return a book?"

sys_ids = tok(system, return_tensors="pt").input_ids.cuda()
q_ids   = tok(question, return_tensors="pt").input_ids.cuda()
full    = torch.cat([sys_ids, q_ids], dim=1)          # system + question, as token IDs
print("system tokens:", sys_ids.shape[1], " question tokens:", q_ids.shape[1])

@torch.no_grad()
def timed(fn):
    torch.cuda.synchronize(); t = time.time()
    out = fn()
    torch.cuda.synchronize()
    return out, (time.time() - t) * 1000                # milliseconds

timed(lambda: model(full))                             # warm-up (first GPU run is always slow)

# A: no cache, prefill everything
out_a, ms_a = timed(lambda: model(full))
print(f"A no cache   : {ms_a:6.1f} ms")


# B: prefix cache, step 1: prefill ONLY the system prompt, ONCE, and keep its KV cache
sys_out, ms_build = timed(lambda: model(sys_ids, use_cache=True))
cache = sys_out.past_key_values
print(f"build cache  : {ms_build:6.1f} ms   (paid once, for all users)")

# B: prefix cache, step 2: a new request only prefills the QUESTION, on top of the saved cache
def with_cache():
    c = copy.deepcopy(cache)                       # copy, so the saved cache stays clean for the next user
    return model(q_ids, past_key_values=c)

timed(with_cache)                                  # warm-up
out_b, ms_b = timed(with_cache)
print(f"B with cache : {ms_b:6.1f} ms")
print(f"speedup      : {ms_a / ms_b:.1f}x")

# C: lossless check: is the first generated token the same?
tok_a = out_a.logits[0, -1].argmax().item()
tok_b = out_b.logits[0, -1].argmax().item()
print("same first token?", tok_a == tok_b, repr(tok.decode(tok_a)))



# D: break it: timestamp at the START (time changes every request)
def shared_prefix(text1, text2):
    a = tok(text1).input_ids
    b = tok(text2).input_ids
    n = 0
    while n < min(len(a), len(b)) and a[n] == b[n]:   # count matching tokens from the start
        n += 1
    return n, len(b)

n, total = shared_prefix("Today is 2026-11-14 10:32:05. " + system,
                         "Today is 2026-11-14 10:32:09. " + system)
print(f"timestamp FIRST: {n} of {total} tokens reusable → {total - n} must be prefilled again")

# fix: timestamp at the END
n, total = shared_prefix(system + " Today is 2026-11-14 10:32:05.",
                         system + " Today is 2026-11-14 10:32:09.")
print(f"timestamp LAST : {n} of {total} tokens reusable → {total - n} must be prefilled again")