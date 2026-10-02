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
