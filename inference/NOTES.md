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
- [x] Build my own KV cache (inference/kv_cache.py, Qwen3.8-27B sizes)

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

### My own KV cache (kv_cache.py, Qwen3.8-27B attention sizes, 1 head, GPU)
- Token 5120 → W_q (5120×6144) → Q = 24 heads × 256; W_k/W_v (5120×1024) → K, V = 4 heads × 256. Matmul rule: (a,b)@(b,c) → (a,c); output size = W's columns.
- GQA: 24 Q heads ÷ 4 KV heads = 6 Q heads read the same KV head, all at the same time (reading isn't exclusive). Fewer KV heads → smaller cache AND fewer bytes read per decode step.
- 2000 decode steps: no cache 7.2 s (5.1 ms/step and rising, N²) vs cache 1.6 s (0.8 ms/step, N). Same answer (allclose True). Q is never cached.
- The cache still slows a little: attention reads all cached K/V, and torch.cat copies the whole cache every step → PagedAttention uses fixed pages instead.
- Cache match: within a request by POSITION (row i = position i; same token ID at a different position has different K, because of context + RoPE). Across requests (prefix caching) by the exact token-ID sequence from the start, hashed in blocks of 16 with the parent hash.
- Size check: 2000 tokens × 1024 K numbers × 2 bytes (bf16) × 2 (K+V) × 16 full-attention blocks = 131 MB = 64 KB/token ✓.

### Reading: How Transformers Work — Attention Explained (prompt20 blog)
https://blog.prompt20.com/posts/how-transformers-work-attention-explained/
Best single article on the full transformer block. Re-read for the parts I haven't built yet.
- **Block** = norm → attention → **add back** → norm → feed-forward → **add back**.
- **Residual stream**: one vector per token flows through every block; each block *adds* to it ("a shared notepad"), it doesn't replace it.
- **Feed-forward (MLP)**: expand → nonlinearity → shrink, per token, no mixing between tokens. Holds most of the parameters.
- **Multi-head**: heads split a fixed budget; outputs concatenated → **W_O** → back to hidden size.
- **Position**: sinusoidal (2017) → learned → **RoPE** (modern; rotates Q and K so attention depends on the distance i−j).
- **Cost**: attention is quadratic in length; the KV cache grows linearly and can exceed the weights on long chats.
- **Misconceptions**: attention is a soft blend (not a hard pick); heads often multitask; long context ≠ evenly used context.
- **Gaps still to build**: W_O, residual + RMSNorm, the MLP at real size (5120 → 17408 → 5120).
- **Questions to answer from memory**:
  1. Delete block 15 of a trained model — garbage, or slightly worse? Why (residual)?
  2. Qwen3.8-27B: attention weights ≈ 5120×(6144+1024+1024+6144) vs MLP ≈ 3×5120×17408 — which is bigger? What does that mean for quantization and decode speed?
  3. Qwen3-0.6B: hidden 1024 but Q = 16×128 = 2048 — does that break "heads split a fixed budget"? What does it cost?
  4. A 262K-context model that mostly ignores the middle — what are you paying for in the KV cache?
- **Section 1 — why RNNs hit a wall** (done):
  - RNN = one fixed-size notebook rewritten each word → forgets; sequential → slow training; word 1→50 = 49 hops (vanishing gradient).
  - Attention fixes each one, and each fix has a "but": keeps every token's K,V (→ KV cache grows); parallel in training + prefill (→ decode is still sequential, one token per pass); 1 hop between any two words.
  - Trade-off: 100 users × 100K ctx, pure attention 27B-size = 25.6 GB KV/user = 2.5 TB total; linear/RNN-style state ≈ 150 MB/user fixed. → Hybrid models (Qwen3.8: 16 attention + 48 linear layers) keep exact recall and cut memory ~4×.
  - Lesson: never answer "of course X" — do the math for the workload.

## Ch 3: Hardware
- Modes: cloud, on-premise (my team: prod 2× 5090, machine 191 Qualcomm), air-gapped.

### 3.1 GPU architecture
- GPU = throughput machine: one simple op on thousands of numbers at once. GPU → many **SMs** (workshops); each SM has 4 **Tensor Cores** (matmul, MMA: D = A×B + C), 128 CUDA cores (single numbers: adds, norms), SFUs (exp/sin/log for softmax), and its own L1 cache. RTX 4060 laptop: 24 SMs / 96 Tensor Cores. RTX 5090: 170 / 680. A *tensor* is data; a *Tensor Core* is hardware.
- FLOPS: compare **dense**, at the **same precision**. Compute limits prefill and image/video.
- Memory: **SRAM** on-chip caches (L0, L1 per SM, L2 shared; KB–MB, fast) vs **VRAM** = DRAM off-chip (GB, slower). VRAM bandwidth limits decode at low/medium batch.
- Sizing rule: VRAM ≥ weights + at least 50% headroom for KV cache, or OOM.
- **My tests (ch3_cpu_vs_gpu.py, ch3_memory.py):** 4096² matmul: CPU 1 thread 1243 ms, CPU all threads 231 ms, GPU FP32 16 ms (8.6 TFLOPS), GPU FP16 5 ms (~27 TFLOPS; 3.2×, book says ~2×). All threads only 5.4× (10 real cores, 4 small, shared power and memory).
- Read speed: 8–16 MB blocks 712–943 GB/s (cache) vs 64 MB+ 153–173 GB/s (VRAM; ~230 in an earlier run, rated ~256). Cache is 5–6× faster; the cliff is between 16 and 64 MB.
- Correction: my lab decode split (4.6 ms memory + 10 ms overhead) assumed the rated 256 GB/s. With measured speed it's more like 5–8 ms memory + 7–10 ms overhead.

### 3.2 GPU generations
- Name = letter (generation) + number (model; bigger = bigger GPU). H100 → H200 same gen; B200 next gen. New generation every 1–2 years.
- Each generation adds: smaller formats, faster memory, better links. **Hopper** (H100 80 GB 3.35 TB/s, H200 141 GB 4.8 TB/s): FP8 (2× FP16), FlashAttention 3. **Ada** (L4 24 GB 300 GB/s, L40 48 GB 864 GB/s): FP8, graphics-focused, **no NVLink** → only cheap small models. **Blackwell** (B200 192 GB, B300 288 GB, up to 8 TB/s, ~5 PFLOPS FP8): **FP4** + MXFP8/MXFP4/NVFP4, FlashAttention 4 = new gold standard. **Rubin** (2026): **HBM4** helps decode + **CPX** chip for prefill = disaggregation in hardware. Feynman 2028.
- Grace/Vera CPUs: NVLink chip-to-chip up to 900 GB/s CPU↔GPU (several × PCIe) → fast KV/LoRA offload to RAM.
- New generations take ~1 year for software to catch up → benchmark before trusting specs.
- My machines: laptop RTX 4060 + llm.remote RTX 5000 = Ada; prod 2× RTX 5090 = Blackwell (consumer, no NVLink → PCIe only).
- Prices (Nov 2026, estimates): B200 ~$30–50K to buy, ~$3.50–8.60/GPU-hour to rent; B300 ~$50–60K, ~$7–18/hr. One B200 ≈ 15–25 RTX 5090s. DGX = NVIDIA's full 8-GPU server (HGX = just the board, others build the server). DGX B200 ≈ 1.5 TB GPU memory → weights up to ~1 TB with KV headroom. Kimi-K3.1-Pro (1,561 GB, 4-bit) doesn't fit one; GLM-5.3 (756 GB) and DeepSeek-V4.1-Flash (510 GB) do.

### 3.3 Instances
- Instance = GPUs + CPUs + RAM + storage + network + interconnect. Any can be the bottleneck. Same GPU can differ (A100 SXM vs PCIe).
- Node = 8 GPUs. Inside: NVLink (1,800 GB/s Blackwell, 900 Hopper) + NVSwitch (all-to-all). Between nodes: InfiniBand up to 400 Gb/s = 50 GB/s (Gb ≠ GB; ~36× slower than NVLink). NVL72 = 72 Blackwell GPUs in one NVLink rack. Keep heavy syncing inside a node.
- PCIe 5.0 ×16 ≈ 64 GB/s (prod 5090s); Qualcomm cards also PCIe → why multi-chip sync is costly on my machines.
- **MIG**: built-in hardware feature on A100/H100/H200/B200 (not 4060/5090/RTX 5000). Splits one GPU into up to 7 slices, each with its own SMs, VRAM, L2 and bandwidth → no interference. H100 132 SMs → 7 compute slices, 8 memory slices. Fixed profiles (1g.10gb, 3g.40gb...), admin command `nvidia-smi mig`. Good for small models (3B TTS). vs llm.remote: 5 processes share one GPU without MIG → they compete.

### 3.4 Other accelerators
- AMD MI350, Google TPU, AWS Inferentia/Trainium, Groq LPU (SRAM-only), Cerebras WSE-3 (wafer-scale), Qualcomm Cloud AI 100 Ultra, Etched Sohu (transformer ASIC), Furiosa RNGD, SambaNova RDU.
- Bets: memory bandwidth (Groq, Cerebras), power efficiency (Qualcomm, Furiosa), cloud integration (Google, AWS). Shared problems: software (no CUDA = rebuild the stack), manufacturing, distribution. CUDA = NVIDIA's moat.
- SRAM-only (Groq) removes the memory road but needs ~600 chips for a 70B model. Qualcomm = middle: more SRAM than GPUs, cheap slow LPDDR (~550 GB/s per Ultra card, from memory), ~150 W. My estimate: Qwen3.8-27B at 50 TPS for one user needs ~4 cards (range 3–8), limited by chip-to-chip sync; 5090 does ~110 TPS ceiling alone.

### 3.5 Local inference
- Pros: no network latency, offline, privacy, free for the developer. Cons: weak hardware, thermals, fragmented devices, battery.
- Desktop: RTX 5090 32 GB 1,792 GB/s ~$5K vs Apple M3 Ultra 512 GB 819 GB/s ~$10K (capacity vs speed). MoE helps local (few active weights). Ollama/llama.cpp run quantized 100B+ models.
- Mobile: >1–2B params is hard; good for speech, transcription, translation. Future = local + cloud together.

### Kernels (side discussion)
- Kernel = one function launched on thousands of GPU threads. One decode step ≈ 400 kernels → launch gaps (my 80 GB/s). Fixes: fused kernels → CUDA Graphs → megakernel (whole forward pass in one kernel, overlaps weight loading; near bandwidth limit; very hard).
- Plan: finish the book first, then CUDA basics → Triton → fused kernels → CUDA Graphs on my_lab decode → megakernels.

## Ch 4: Software
Ladder (closest to GPU → farthest): CUDA kernels → PyTorch → inference engines → Dynamo.
Hardware changes every 1–2 years, software every week → most inference work is at the engine level.

### 4.1 CUDA
- CUDA = how you write code for NVIDIA GPUs (C++ compiled by nvcc). Parts: kernel, CUDA Graph, driver, runtime.
- Kernel = a function run by thousands of threads at once (each thread finds its own index i).
- Fast kernel = fewer trips to VRAM (cook analogy: VRAM = supermarket, SRAM = counter, Tensor Cores = hands).
- 4.1.1 Don't start from zero: BLAS → cuBLAS (GEMM = every linear layer), cuDNN, FlashInfer (paged attention,
  fused sampling), CUTLASS/CuTe (building blocks for custom kernels; FlashAttention 3 uses CUTLASS).
- Prefill GEMM ≈ 1,170 FLOPs/byte (compute-bound); decode batch-1 GEMM ≈ 1 FLOP/byte (memory-bound). 4060 ridge ≈ 100.
- 4.1.2 Kernel selection: kernels are tied to one GPU (SRAM size, Tensor Cores hard-coded). Usually automatic;
  sometimes swap by hand (DeepGEMM for FP8 on Hopper).
- 4.1.3 Fusion: merge kernels → skip the write+read round trip (×2 then ×3 = ×6). Matters most in decode.
  FlashAttention = scores+softmax+×V in one kernel, scores matrix never hits VRAM.
- Custom kernels are written for: new GPUs, new math (MLA → FlashMLA), fusion. Workflow: profile → replace slowest → re-measure.

### 4.2 Frameworks
- PyTorch won (TensorFlow faded, JAX = sharp edges). torch.compile = auto kernel selection + fusion,
  but can't fuse plugin kernels (FlashAttention, DeepGEMM).
- safetensors: weights only, no code (safe, unlike pickle .bin), mmap loading, split into many files.
  ONNX: weights + graph (portable). GGUF (llama.cpp/Ollama): weights + config + tokenizer, often quantized.
- ONNX Runtime (any hardware) vs TensorRT (NVIDIA only, compiled engine per GPU type, fastest).
  Export breaks on new ops (MLA). For LLMs the trend is safetensors → engine directly.
- transformers/diffusers = reference code + config.json + downloads. Not production.

### 4.3 Inference engines
| | vLLM | SGLang | TensorRT-LLM |
|---|---|---|---|
| speed | good | good | best (NVIDIA kernels, some closed) |
| ease | easy | easy | hard (Docker + config.yaml) |
| strength | widest model + hardware support, day zero | big MoE multi-node, close to DeepSeek/Qwen/Kimi | Hopper/Blackwell, NVFP4 |
- All have continuous batching, prefix caching, quantization, speculation, parallelism, disaggregation.
- Principle: more constraints → more performance.
- TRT-LLM V0 = TensorRT plugin; V1 (2025) = PyTorch-based. Check the version.

### 4.4 Dynamo
- Manager above engines: KV-aware routing, disaggregation (prefill/decode workers), multi-node parallelism,
  SLA planner auto-scaling. Only worth it at big scale (not for 1–2 GPU boxes).

### 4.5 Benchmarking
- Best = shadow production traffic. Else match ISL/OSL, traffic pattern (jitter), real prompts, real params.
- Tools: genai-bench, GenAI-Perf, Locust, `vllm bench serve`. Report p50/p90 TTFT, TPOT, throughput.
- Eval datasets (MMLU, gsm8k, HumanEval) = realistic prompts + quality check after optimizing.
- Baseline first, change one thing at a time, repeat runs. Optimizations can fight (speculation + big batch).
- Profiling = why: PyTorch Profiler → Nsight Systems (whole system) → Nsight Compute (inside one kernel).

### Machines found (Ch4 side)
- 191: Ollama on CPU — Qwen2.5-7B Q4_K_M (4.7 GB), Qwen2.5-3B Q4_K_M, ~ALLaM-7B Q8_0; cohere ASR on QAIC cards.
- 122: k8s/ClickHouse/MSSQL node, no accelerators, no LLMs.
- Rent plan (Nov 2026, RunPod community): 5090 $0.69, H100 PCIe $1.99, H100 SXM $2.69, B200 $5.98 → ~$150–200 total.

## Ch 5: Techniques
### 5.1 Quantization — theory so far
- Fewer bits → prefill faster (if math runs in low precision: FP8 W8A8) and decode faster (fewer bytes). ~30–50% per step down.
- Weight-only (Q4, AWQ, GPTQ) → decode faster only. Weights + activations (FP8) → both.
- 8 bits = 256 patterns (−128..127), 4 bits = 16, 2 bits = 4.
- Float = sign + exponent (range) + mantissa (precision), like scientific notation in base 2.
  FP16 1-5-10 max 65,504; BF16 1-8-7 (FP32 range, Google); FP8 E4M3 1-4-3 max 448 (1.75 × 2^8); E5M2 more range.
- Hardware: memory/decode wins work anywhere with an unpack kernel; math wins need a circuit (FP4 only on Blackwell).
- Symmetric quant = coins: coin = max|w| ÷ 127, store round(w ÷ coin), read back = stored × coin.
  Asymmetric = shift the ruler (+ zero point) for one-sided data. Clipping = cap outliers.
- Granularity: per-tensor (one coin for millions) vs block 32 (one coin per 32) → outliers only hurt their block.
- MXFP4/MXFP8 = block 32; NVFP4 = block 16 + FP32 global scale. Production → FP8. Local → GGUF dynamic.

### 5.1 LAB: ch5_quantize.py (Qwen3-0.6B, RTN fake-quant of all 196 linears, 440M numbers)
One matrix: typical |w| 0.0209, biggest 0.5547 → outlier 26× typical.
| setting | size MB | zeros % | perplexity | answer |
|---|---|---|---|---|
| BF16 | 881 | 0 | 13.32 | red, blue, yellow ✅ |
| FP8 per-tensor | 440 | 0 | 13.67 (+2.6%) | ✅ |
| INT8 per-tensor | 440 | 5.7 | 13.71 (+2.9%) | ✅ |
| INT4 per-tensor | 220 | 72.8 | 64,347,636 💥 | garbage |
| INT4 block 32 | 248 | 14.1 | 18.36 (+38%) | ✅ but dumber |
| INT3 block 32 | 193 | 31.8 | 49.20 | loops "the three of the three" |
| INT2 block 32 | 138 | 75.9 | 38,545,576 💥 | garbage |
Lessons: granularity > bits (INT3 block works, INT4 per-tensor dies); outliers crush small weights
(coin 0.079 vs typical 0.021 → 0); scales cost memory (248 vs 220 MB); FP8 beats INT8 at the same size.
Would not ship INT4 RTN (+38%) → use FP8, or AWQ/GPTQ + evals (5.1.2, 5.1.3).

### 5.1 my own quantizer (my_quant.py, my_quant_real.py)
- 6 numbers: INT8 scale 0.085, errors ≤ 0.04; INT4 scale 1.54 → 0.25 became 0; outlier 100 → INT8 as bad as INT4;
  2 blocks of 3 → the block without the outlier got scale 0.024 and error ~0.
- Real Qwen down_proj (block 27, 1024×3072 = 3.1M): mean |w| 0.0209, absmax 0.5547 (26×).
  INT8 per-tensor 7.1% zeros err 0.00109 | INT4 per-tensor 86.5% zeros err 0.01738 | INT4 block 32 14.8% err 0.00227.
- Block sweep INT4 err (real bits): 8 0.00151 (6.0) | 16 0.00191 (5.0) | 32 0.00227 (4.5) | 64 0.00260 (4.25)
  | 128 0.00291 (4.125) | 1024 0.00382 (4.016) | per-tensor 0.01738 (4.0). Smooth until 1024, then a jump → why MX=32, NVFP4=16.
- 5.1.2: I can only do PTQ (QAT = labs: GPT-OSS MXFP4, Kimi K2 INT4). ModelOpt / llm-compressor.
  Sensitivity: weights < activations < KV cache (compounds over tokens) < attention/softmax. Keep first/last layers.
- 5.1.3: perplexity + benchmarks (MMLU, gsm8k) + custom evals; loss must be noise-level. Quantization is a scale.

### 5.2 Speculative decoding
- Speculator drafts N tokens, target verifies in ONE pass → N+1 tokens per pass. Lossless, TPS only (not TTFT).
- Factors: draft cost, draft length, acceptance rate. Bad at high temperature and big batches (turn off).
- Draft-target (≥10× smaller, same tokenizer) → Medusa (extra heads) → EAGLE-3 (reads 3 hidden states, ≤8 drafts)
  → n-gram/SuffixDecoding (copy from input; code edits) → MTP (built into DeepSeek/Qwen3.5+/Gemma 4)
  → DFlash (Z-Lab 2026, block-diffusion drafter, whole block in one pass) → DSpark.
- vLLM AMD benchmarks (Aug 2026): DFlash up to 2.87×, Gemma4 MTP 2.74×, EAGLE-3 ~2.2× (but <1× on Qwen3-8B MATH500).

### 5.3 Caching
- Prefix caching: reuse KV across requests, stops at the FIRST different token → stable stuff first, variable stuff last.
- KV storage G1 VRAM → G2 CPU RAM → G3 SSD → G4 network; loading 10K tokens from RAM ~10 ms vs re-prefill ~1–2 s.
- Cache-aware routing (send user back to the replica with their cache, balance load); global KV cache on G4.
- Long context: 27B at 128K = 8 GB KV per user. FlashAttention, PagedAttention, chunked prefill.

### 5.3 LAB: my_prefix_cache.py (Qwen3-0.6B, 2,761-token system prompt + 9-token question, 4060)
| test | TTFT |
|---|---|
| A no cache | ~220 ms |
| build cache (once) | ~221 ms |
| B with prefix cache | ~23 ms (9.5×), same first token ' Please' ✅ lossless |
- 23 ms = one memory-bound pass over 1.2 GB weights + deepcopy of 309 MB cache (real engines share pages).
- Timestamp FIRST: only 21 of 2,784 tokens reusable (digits tokenized one by one, breaks at 5 vs 9).
  Timestamp LAST: 2,782 of 2,784 reusable. One line moved = 9× TTFT.
