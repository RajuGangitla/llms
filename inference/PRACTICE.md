# Chapter 2 practice — real model, real measurements

Each one: predict first → write code → run → compare with prediction → explain the gap.
Model: Qwen3-0.6B (bf16) on RTX 4060 laptop, unless noted.

## Level 1 — measure what you already built
- [ ] **P1. Prefill scaling.** Prompt lengths 10, 100, 500, 1000, 4000 tokens → prefill ms and tok/s.
      Where does tok/s stop rising? (that's where prefill becomes compute-bound)
- [ ] **P2. Decode vs context length.** Decode ms/token with KV at 100, 1K, 8K, 16K tokens.
      Predict: at how many tokens does the KV cache weigh as much as the weights (1.19 GB ÷ 112 KB)?
- [ ] **P3. Greedy check.** Your own decode loop at T=0 vs `model.generate(do_sample=False)`.
      Must be the exact same tokens. If not — why?

## Level 2 — batching on the real model
- [ ] **P4. Batch decode.** Batch sizes 1, 2, 4, 8, 16, 32, 64 → ms/step, total tok/s, per-user tok/s.
      Find the knee. Compare with the Ollama lab from chapter 0.
- [ ] **P5. Out of memory.** Predict the max (batch × context) that fits in 8 GB, then find it for real.

## Level 3 — improve your own KV cache (kv_cache.py)
- [ ] **P6. All heads.** Attention for all 24 Q heads at once, GQA without repeat_interleave
      (hint: view Q as (4 groups, 6, 256)).
- [ ] **P7. Preallocated cache.** Allocate (N, 1024) once, write row t each step instead of torch.cat.
      Measure the speedup vs cat.
- [ ] **P8. Paged cache.** Pages of 16 tokens + a page table (list of page numbers). Same answer as before.

## Level 4 — inference tricks
- [ ] **P9. Prefix caching.** Two prompts share a long system prompt. Prefill it once, reuse the cache
      for both. Measure TTFT saved.
- [ ] **P10. KV quantization.** Store K/V in int8 (scale + round). Measure the error and the memory saved.

## Daily drills — rebuild from memory (no looking), until each takes < 10 min
- [ ] D1. Blank file → load Qwen3-0.6B, print config, compute KV bytes/token by hand first.
- [ ] D2. Blank file → prefill + decode loop with the cache, print ms/token.
- [ ] D3. Blank file → kv_cache.py: Q/K/V + heads + GQA + attention + cache vs no cache.
- [ ] D4. Paper, no code: draw the flow text → tokens → embedding → 28 blocks (attention + MLP) → LM head → sample, with shapes.
- [ ] D5. Mental math: for any model config, compute weights GB, KV/token, KV at 32K, decode tok/s ceiling on a given GPU.

## Resources (read → then rebuild the idea in code)
- Transformer Inference Arithmetic — kipp.ly blog (the memory/compute math, KV size, why decode is memory-bound)
- Making Deep Learning Go Brrrr From First Principles — Horace He (compute vs memory vs overhead — explains the 80 GB/s result)
- LLM Inference Series — Pierre Lienhart, Medium (KV cache parts are excellent)
- Let's build GPT from scratch — Andrej Karpathy, YouTube + nanoGPT repo
- Transformer Explainer — poloclub.github.io/transformer-explainer (visual)
- vLLM PagedAttention paper: "Efficient Memory Management for LLM Serving with PagedAttention" (before P8)
- nano-vllm — github.com/GeeeekExplorer/nano-vllm (a ~1,200-line vLLM; read after P8)

### Attention / KV-cache article path (from my list, in order)
1. Understanding and Coding the KV Cache in LLMs from Scratch — Sebastian Raschka (compare with my kv_cache.py)
2. Understanding KV Cache: The Hidden Memory Cost of Serving LLMs (MHA/MQA/GQA/MLA + cache formula) → then do P6
3. MQA vs GQA vs MLA: Three Ways to Shrink the KV Cache (memory numbers + quality tradeoff)
4. MLA: Redefining KV-Cache Through Low-Rank Projections (DeepSeek; explains GLM-5.3's kv_lora_rank 512)
5. Attention Variants Explained: MHA, GQA, MQA, MLA, SWA, DSA (reference card)
6. Flash Attention Explained — AFTER chapter 3 (needs SRAM vs HBM)
- Refresher only if stuck: The Illustrated Transformer (Alammar), Transformers Explained Visually parts 3/4
