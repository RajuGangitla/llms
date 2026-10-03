import torch

token = torch.randn(5120)              # " sky": 5120 numbers (from lookup)
W_q   = torch.randn(5120, 6144)        # weight matrix (learned in training)
W_k   = torch.randn(5120, 1024)        # weight matrix (learned in training)
W_v   = torch.randn(5120, 1024)        # weight matrix (learned in training)

Q     = token @ W_q                    # multiply
K     = token @ W_k                    # multiply
V     = token @ W_v                    # multiply

print("Q:", Q.shape)
print("K:", K.shape)
print("V:", V.shape)

Q_heads = Q.view(24, 256)              # cut into 24 heads of 256
K_heads = K.view(4, 256)              # cut into 4 heads of 256
V_heads = V.view(4, 256)              # cut into 4 heads of 256

print("Q heads:", Q_heads.shape)
print("K heads:", K_heads.shape)
print("V heads:", V_heads.shape)

K_shared = K_heads.repeat_interleave(6, dim=0)
V_shared = V_heads.repeat_interleave(6, dim=0)
print("K shared:", K_shared.shape)
print("same?", torch.equal(K_shared[6], K_shared[11]))   # Q heads 6 and 11 → same KV head?


# ---- Step 4: 10 tokens at once (prefill)
tokens = torch.randn(10, 5120)          # 10 tokens, each 5120 numbers

Q_all = tokens @ W_q                    # same W_q for every token
K_all = tokens @ W_k
V_all = tokens @ W_v
print("Q_all:", Q_all.shape)
print("K_all:", K_all.shape)

K_all = K_all.view(10, 4, 256)          # 10 tokens, 4 heads, 256 each
V_all = V_all.view(10, 4, 256)
print("K_all heads:", K_all.shape)


# ---- Step 5: attention for the LAST token, head 0 only
q = Q_all[-1].view(24, 256)[0]       # last token's Q, head 0      → 256 numbers
k = K_all[:, 0, :]                    # all 10 tokens' K, KV head 0 → (10, 256)
v = V_all[:, 0, :]                    # all 10 tokens' V, KV head 0 → (10, 256)

scores  = k @ q / 256 ** 0.5          # q·k for every token, ÷ √head_dim → 10 scores
weights = torch.softmax(scores, dim=0)   # → 10 probabilities that add up to 1
out     = weights @ v                 # weighted mix of the 10 V's → 256 numbers

print("scores :", scores.shape)
print("weights:", weights.shape, "sum =", weights.sum().item())
print("out    :", out.shape)


# ---- Step 6: decode 2000 tokens — no cache vs KV cache
import time

dev = "cuda"
N = 2000
W_q = torch.randn(5120, 6144, device=dev) * 0.02     # same weights, now on GPU
W_k = torch.randn(5120, 1024, device=dev) * 0.02
W_v = torch.randn(5120, 1024, device=dev) * 0.02
stream = torch.randn(N, 5120, device=dev)            # pretend: the 2000 tokens that arrive one by one

def attend(q, K, V):                       # step 5, made into a function (head 0)
    q = q.view(24, 256)[0]
    k = K.view(-1, 4, 256)[:, 0, :]
    v = V.view(-1, 4, 256)[:, 0, :]
    w = torch.softmax(k @ q / 256 ** 0.5, dim=0)
    return w @ v
# A: NO cache — recompute K, V for ALL tokens every step
torch.cuda.synchronize(); start = time.time()
for t in range(1, N + 1):
    K = stream[:t] @ W_k                   # ALL t tokens, again
    V = stream[:t] @ W_v
    q = stream[t-1] @ W_q                  # Q only for the newest token
    out_a = attend(q, K, V)
    if t in (100, 1000, 2000):
        torch.cuda.synchronize(); print("A no cache, step", t, round(time.time() - start, 2), "s")

# B: KV CACHE — compute K, V for the NEW token only, append
K_cache = torch.empty(0, 1024, device=dev)   # starts empty: 0 tokens
V_cache = torch.empty(0, 1024, device=dev)
torch.cuda.synchronize(); start = time.time()
for t in range(1, N + 1):
    x = stream[t-1:t]                      # ONLY the new token
    K_cache = torch.cat([K_cache, x @ W_k])    # grow by 1
    V_cache = torch.cat([V_cache, x @ W_v])
    q = stream[t-1] @ W_q
    out_b = attend(q, K_cache, V_cache)
    if t in (100, 1000, 2000):
        torch.cuda.synchronize(); print("B cache,    step", t, round(time.time() - start, 2), "s")

print("same answer?", torch.allclose(out_a, out_b, atol=1e-3))
print("cache shape:", K_cache.shape, " size:", K_cache.numel() * 4 / 1e6, "MB (K only, fp32, 1 block)")