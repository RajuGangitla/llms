# Inference Engineering Notes

## Ch 0: Inference
- 3 layers: **Runtime** (make one model on one GPU fast) → **Infrastructure** (scale across GPUs/regions/clouds) → **Tooling** (make it easy for engineers to control)
- Remember: **Fast → Scale → Easy**
- 6 runtime techniques:
  - **Batching**: many users on one GPU at once
  - **Caching**: reuse KV cache for shared prompt starts
  - **Quantization**: smaller numbers → less memory, more speed
  - **Speculation**: small model guesses, big model verifies
  - **Parallelism**: split one model across GPUs
  - **Disaggregation**: prefill and decode on separate machines

### Ch 0 insights (from quiz)
- **Batching:** decode's slow part is reading all weights from memory. Batch shares that read → each user a bit slower, total throughput ~N× higher. Bad for latency-critical single-user apps, and too-big batches run out of VRAM (each user needs own KV cache).
- **Speculation:** checking k draft tokens costs ~same as 1 token (same weight read). Always ≥1 token per pass. Hurts when acceptance is low, and helps less under heavy batching (spare compute already used).
- **Prefix caching:** reuse KV cache of identical prefix. Must match token-for-token from the start → put fixed content first, changing content last (no timestamp at top!).

### Ch 0 lab: batching (`lab_batching.py`, RTX 4060 laptop, qwen3:4b Q4, 200 tokens)
| 4 users | no batching (queue) | batching (NUM_PARALLEL=4) |
|---|---|---|
| total throughput | 66 tok/s | 137 tok/s (2.1×) |
| worst wait | 12.1 s | 5.8 s |
| per-user gen speed | 69 tok/s | 37 tok/s |

- Without batching = a queue: users finish at ~3, 6, 9, 12 s. "Gen speed" looked fine but hid the waiting → always measure end-to-end wait.
- 2 users: per-user only ~12% slower (memory-bound, weight read shared). 4 users: ~45% slower → small model hit the **knee** where compute becomes the limit. Past the knee, extra users just split the speed.
- Setting `OLLAMA_NUM_PARALLEL` in my shell did nothing — it must be set on the **server** process.

### Ch 0 deeper
- **Why decode is slow:** weights live in VRAM but must stream through the tiny compute cores for EVERY generated token (all layers, all params; only embedding is a lookup). Memory trip = bottleneck ("memory-bound").
- **Prefill vs decode:** prompt tokens are all known → processed in ONE pass (one weight trip). Answer tokens come one at a time → one trip per token.
- **3 fixes for the same waste:** batching (share trip across users), speculation (share trip across guessed tokens), quantization (shorter trip). Spec + batching compete for the same spare compute → dynamic speculation.
- **Static vs continuous batching:** static waits for the longest request; continuous refills a seat every step (Orca). llama.cpp/Ollama `parallel` = number of seats, continuous by default.
- **Knee** = point where adding users stops being free. Bigger model → longer memory trip → knee at MORE users. But bigger model leaves less VRAM for KV cache → may run out of memory at FEWER users. Real max = whichever limit hits first.
- **Dense vs MoE:** same total size → dense smarter; same compute per token → MoE smarter. MoE must still store all experts in memory.
- **Batching as a general systems idea (LLM vs ClickHouse):** ask 3 questions — (1) what's the expensive fixed cost? (2) is it paid per step, per batch, or per request? (3) how long can the user wait? Per-step cost + impatient users → continuous batching (LLMs). Per-batch cost + tolerant writers → wait-to-fill with timeout (ClickHouse `async_insert`, avoids "Too many parts"). Batching always trades waiting for efficiency.
- GPUs aren't ideal for single-user decode (memory wall) but win on flexibility + ecosystem; Groq/Cerebras put memory on-chip (fast, expensive).

## Ch 1: Prerequisites
- Core trade-off: **latency vs throughput vs quality** — optimize for your use case, not one number (NFL player analogy).
- **Shared (API, pay per token)** first. Switch to **dedicated (own GPUs)** for: **Scale** (per-GPU-hour cheaper than per-token), **Specialization** (fine-tuned/custom model, strict latency/uptime), **Orchestration** (multi-model pipelines).
  - Rough math: $2/hr GPU ≈ $1,440/mo ≈ 2.9B tokens at $0.50/M. One GPU at 1,000 tok/s 24/7 ≈ 2.6B tokens/mo → self-hosting only wins with lots of steady traffic.
- **Online** (chat, voice, code completion) → optimize latency. **Offline** (batch transcription, doc processing) → optimize throughput, big batches. Same model can have two deployments.
- **Consumer** → cost + flexibility. **B2B** → latency + uptime. Compliance (data region, privacy) limits infra choices.
- **Model selection is the biggest optimization:** smallest model that passes YOUR evals. Public benchmarks get gamed (Goodhart's law). Fine-tuning = adapt weights to a domain (text-to-SQL: few-B model can match huge ones). Distillation = small student copies big teacher's probability outputs.
- **Metrics:** TTFT (prefill, compute-bound), TPS (decode, bandwidth-bound), ITL (10 ms ITL = 100 tok/s). "TPS" is ambiguous: per-user (latency) vs total (throughput) — my lab showed both.
- Use **percentiles** (P50/P90/P99), not averages — latency is right-skewed. Measure **end-to-end** (incl. queue + network), not just GPU time. Fast GPU time + slow end-to-end → fix infrastructure, not the model.
- TTFT fixes: long prompt → prefix caching / shorter prompt; queueing → capacity/batching. Weight-only quantization barely helps prefill (compute-bound).

## Ch 2: Models
- Two model styles: **autoregressive** (LLMs, one token per pass → memory-bound decode) vs **iterative denoising** (diffusion images/video, whole image per step, fixed steps → compute-bound; speed-up = fewer steps).

### 2.1 Neural networks
- Neuron → layer (neurons side by side, same input) → network (layers chained). Input / hidden / output layers. Hidden state = token's vector between layers; dimensionality = its size (text goes UP: 1 id → 4096 numbers; images go DOWN).
- **Encoder** = network that understands input (embedding models, BERT). **Decoder** = network that generates (LLMs). Whisper = encoder+decoder. Modern LLMs are decoder-only: ONE network does both prefill and decode. (DeepSeek V4.1 Flash, Sep 2026: causal encoder-decoder — 8B active input / 16B active output.)
- **Linear layer = matmul:** `y = xW + b`. W = grid of weights (one column per neuron). Compute per token ≈ **2 × parameters** (multiply + add per weight).
- **Shape of x decides the bottleneck:** decode 1 user = x is 1 row → each weight read used once → memory-bound. Prefill / batching = many rows → each weight read reused many times → compute-bound. Reuse ratio = **arithmetic intensity** (2.4.1).
- **Activation functions:** stacked linear layers collapse into one (x·2·3 = x·6). Non-linear ReLU/SiLU/SwiGLU between layers prevents collapse → depth means something. Cheap vs matmul.

### 2.2 LLM inference mechanics
- Loop: chat template + tokenize → **prefill** (process input, BUILD KV cache) → **decode** repeat: forward pass → logits (1 score per vocab token, 100K+) → softmax → sampling → stop token or max_tokens.
- 3 sequences share the context window: input, reasoning (thinking), output. Wrong chat template = dumber model even with perfect weights.
- **Thinking tokens are decode tokens** (generated one by one). 1,000 thinking tokens at 69 tok/s ≈ 15 s before the real answer → TTFT lies for reasoning models; measure time to first answer token. Reasoning = expensive (slow phase + bigger KV cache).
- **Temperature** divides the logits (scores, not token IDs) before softmax: low → top token dominates, high → flatter/random, 0 → always top. Top-k = keep k best; top-p = smallest set summing to p. Structured output = block invalid tokens (logit biasing). Sampling is cheap, but high temperature lowers speculative-decoding acceptance.

### 2.2.1 Architecture (config.json)
- `config.json` = model shape (layers, hidden size, heads, vocab, context). `ollama show qwen3:4b`: arch qwen3, 4.0B params, context 262144, hidden 2560, Q4_K_M, temp 0.6 / top_k 20 / top_p 0.95, stop `<|im_end|>`.
- Name `Qwen3MoeForCausalLM` = family + version + MoE + causal LM (predicts next token looking backward; vs masked LM like BERT).
- Sizes, base/instruct, LoRA fine-tunes share one architecture → engine optimizations transfer for free. LoRA changes weights, not architecture.
- Config context length = what the model CAN handle, not what VRAM can HOLD.

### 2.2.2 Transformer blocks
- Embedding layer → transformer blocks ×N (attention + MLP + norm) → LM head (output layer) → logits.
- **MLP = most weights (~⅔)** → main quantization target. **Attention = most complexity** (KV cache). Norm/activations = rounding error.
- LM head = hidden × vocab matmul (2560 × ~152K ≈ 390M weights ≈ 10% of 4B), read every decode step. Embedding is a cheap lookup.

### 2.2.3 Attention + KV cache
- LLMs use **self-attention** (Q,K,V from same sequence, causal mask). Cross-attention = Q from one input, K/V from another (image gen, Whisper decoder).
- Q, K, V = token vector × learned W_Q, W_K, W_V — for every token, at every layer.
- **Without KV cache:** recompute K,V of all past tokens every step → quadratic. **With KV cache:** compute K,V once per token, store, reuse → linear.
- Only **K and V** cached: Q is only needed for the newest token, once. K/V are read by every future token.
- Cache layout: [layer] × [K/V] × [KV head] × [position] × [head dim]. qwen3:4b ≈ 2 × 36 × 8 × 128 × 2 bytes ≈ **144 KB per token per user** → 4K tokens ≈ 590 MB, 262K ≈ 37 GB.
- Prefill builds the cache; decode reads it + appends 1 entry per step. Grows per token AND per user → VRAM limit.

### 2.2.4 MoE
- Qwen3-235B-A22B: 235B total, 22B active; 128 experts/layer, router picks 8 per layer per token.
- 1 user → reads only active experts → fast. **Batching kills much of the advantage:** different users pick different experts → nearly all experts read each step. Fix = expert parallelism (spread experts across GPUs, ch 5).
- Strata-style offload (hot experts on GPU, rest in RAM) works for 1 user, breaks with many users (rare experts needed constantly).
- Dense usually better under ~32B; MoE pays off at 100B+.

(2.3 image generation — skipped, not my goal.)

### 2.4 Calculating bottlenecks (diagnosis)
- GPU has 2 speeds: **compute** (ops/s) and **memory bandwidth** (bytes/s). Find which one limits you; optimizing the other does nothing.
- Prefill → compute-bound. Decode → memory-bound. Image/video gen → compute-bound. Batching makes decode less memory-bound (more math per byte moved).
- **ops:byte ratio** (GPU) = compute ÷ bandwidth. H100 FP16: 989 TFLOPS ÷ 3.35 TB/s ≈ **295**. **Arithmetic intensity** (work) = ops ÷ bytes moved. Intensity < ratio → memory-bound; > ratio → compute-bound. Roofline chart = slope (memory limit) then flat roof (compute limit).
- Decode weights FP16: 2 ops per weight ÷ 2 bytes ≈ 1 op/byte per user → batch ~295 users on H100 before compute-bound. Book's standard-attention example (N=4096, d=128) ≈ 62 ops/byte < 295 → memory-bound. Exact calc is academic; the intuition matters.

### 2.5 Optimizing attention (treatment)
- Problem: N×N score grid (4096² ≈ 32 MB), quadratic. Naive attention writes S and P to VRAM and reads them back.
- **Implementation fixes (lossless, runtime = my job):** **FlashAttention** — tiles fit in fast SRAM, fused score→softmax→×V, never writes the big grid; GPU-specific kernels. **PagedAttention** — KV cache in pages via lookup table → less fragmentation, more users.
- **Algorithm fixes (baked in training):** sliding window (Muse Glimmer), gated (Muse Glimmer), linear (Qwen 3.8 Gated DeltaNet), compressed (DeepSeek V4), MLA (DeepSeek/Kimi/GLM), Mamba/state-space hybrids (Nemotron). Intuition: nearby tokens matter more.

### End-of-chapter-2 LAB checklist
- [x] Install torch + transformers (CUDA), model Qwen3-0.6B
- [x] Print config (layers, hidden, heads, KV heads)
- [x] Chat template + token IDs
- [x] Prefill: hidden-state shape at every layer, time, KV cache size
- [x] Decode: shape per step, KV cache growing, ms per token
- [x] Sampling: top-5 probs under different temperature / top-k / top-p
- [x] Memory moving: weight bytes per step ÷ step time = real bandwidth vs 256 GB/s
- [ ] Build my own KV cache in attention.py (Q used once, K/V appended)

### Memory-bound vs compute-bound
- Each step = **moving time** (weights VRAM → cores, limited by memory bandwidth) + **math time** (limited by compute). Longer one = bottleneck.
- Memory-bound = cores waiting for weights. Compute-bound = cores busy doing math. Same weights moved either way; only the amount of math per weight changes.
- **Decode speed ≈ memory bandwidth ÷ model size.** Check: RTX 4060 laptop ~256 GB/s ÷ 2.5 GB (qwen3:4b Q4) ≈ 100 tok/s ceiling; measured 69 tok/s → single-user decode is memory-bound.

### Chapter 2 LAB results (my_lab.py, Qwen3-0.6B bf16, RTX 4060 laptop)
- Config: 0.6B params, 1.19 GB, 28 layers, hidden 1024, 16 Q heads / 8 KV heads (GQA 2:1), head_dim 128, vocab 151,936. KV = 112 KB/token → 40K tokens ≈ 4.5 GB, ~4x the weights.
- Tokens: 7-word prompt became 20 tokens; the chat template adds ~11 tokens every turn. Special tokens have the highest IDs (added after BPE). Space is part of the token (' why' ≠ 'why').
- Model = embedding (lookup table) → 28 blocks (attention + MLP) → LM head. Shape stays (1, N, 1024) through all blocks; the LM head makes (1, N, 151936).
- Prefill: (1, 21, 1024) in ~26 ms. Decode: (1, 1, 1024) in ~14.6 ms. **One token costs about as much as 21**, because both read all the weights → decode is memory/overhead-bound.
- Sampling: confident prompt → T barely matters ("The" 99.7% even at T=1.5). Open prompt → T=1.5 spreads it out (top-5 only 51%, the rest is junk in the tail → top-k/top-p cut it). T=0.1 showed IDs 0–3 at 0.0%: underflow ties, not real choices → engines use argmax for T=0.
- Timing varies run to run (46.8 ms vs 26 ms prefill) → benchmark many runs, report P50/P99.
- Bandwidth: 1.19 GB ÷ 14.9 ms = **80 GB/s, only 31% of 256**. Expected 4.6 ms; the other ~10 ms is **launch overhead**: ~400 small kernels per token launched one by one from Python, with the GPU idle in between.
- Proof: Ollama (llama.cpp, C++) runs qwen3:4b (2.5 GB) at the same 69 tok/s ≈ 170 GB/s. Same GPU, same speed, 4x the model → the gap is software. Fixes: CUDA graphs, kernel fusion, no Python in the hot loop.
