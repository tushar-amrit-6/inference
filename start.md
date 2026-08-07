# LLM Inference: A Study Guide

**Goal:** research-level understanding of how LLM inference works and why it's hard.
**Assumed starting point:** comfortable with Python and basic ML; transformers not yet solid.
**Pacing:** ~11 modules. At a few hours a week, roughly 3 months. Modules 0–3 are the foundation — don't rush them.

---

## The one idea that organizes everything

Generating a token requires reading *every* model weight from memory. On modern hardware, moving those bytes takes far longer than the arithmetic performed on them. So decoding a single sequence leaves the GPU's compute almost entirely idle.

Nearly every technique in this guide is an answer to one question: **how do we do more useful work per byte moved from memory?** Batching, quantization, GQA, speculative decoding, PagedAttention — different attacks on that same problem. Keep this in view and the field stops looking like a pile of tricks.

---

## Module 0 — Transformer fundamentals

You can't reason about inference without a mechanical picture of the forward pass.

**Learn**
- Tokenization; embeddings; what a "hidden state" actually is
- Self-attention: Q, K, V; the QK^T → softmax → weighted-V pipeline
- Causal masking, and why it makes cached generation possible
- Multi-head attention, MLP block, residual stream, layer norm
- Shapes at every step — this matters more than you'd expect
- Training vs. inference: teacher forcing vs. the autoregressive loop

**Read**
- *Attention Is All You Need* (Vaswani et al., 2017)
- *The Illustrated Transformer* (Alammar) — for intuition
- Karpathy's **nanoGPT** / "Let's build GPT" video — build one end to end

**Exercise:** implement single-head attention in NumPy. No frameworks. Print shapes at each step.

**Checkpoint:** you can hand-draw the data flow for one transformer layer and state the tensor shape at every arrow.

---

## Module 1 — The autoregressive loop

**Learn**
- Why generation is inherently sequential
- **Prefill** (process the whole prompt at once) vs. **decode** (one token at a time)
- Why these are two different computational workloads, not two phases of one
- The naive-recomputation problem: O(n²) waste without caching

**Exercise:** write a generation loop with no KV cache. Time it as sequence length grows. Plot it.

---

## Module 2 — The KV cache

The single most important object in inference.

**Learn**
- What's stored and why keys/values (not queries) are cached
- The size formula: `2 × layers × kv_heads × head_dim × seq_len × batch × dtype_bytes`
- Why KV cache, not weights, is usually what limits your batch size
- Fragmentation and over-allocation in naive implementations

**Exercise:** compute the KV cache footprint for Llama-3-70B at 8k context, batch 32, fp16. Compare it to the weights. Then re-run Module 1's loop *with* a cache and re-plot.

**Checkpoint:** you can explain, without notes, why long context is expensive at inference in a way that's unrelated to attention's O(n²) compute cost.

---

## Module 3 — Decoding and sampling

**Learn**
- Greedy, beam search, and why beam search largely lost for open-ended generation
- Temperature, top-k, top-p (nucleus), min-p; what each distorts about the distribution
- Repetition penalties and their failure modes
- Constrained/structured decoding: grammars, FSMs, JSON schemas
- The determinism question: why identical inputs can yield different outputs

**Read**
- *The Curious Case of Neural Text Degeneration* (Holtzman et al., 2019) — the nucleus sampling paper

**Exercise:** implement top-k, top-p, and min-p from scratch. Sample the same prompt across a temperature sweep and characterize how the outputs degrade at each end.

---

## Module 4 — Performance: metrics and the roofline

The analytical core. Everything after this is applied roofline reasoning.

**Learn**
- **TTFT** (time to first token), **TPOT/ITL** (per-output-token), end-to-end latency, throughput
- Arithmetic intensity: FLOPs per byte moved
- The roofline model; memory-bound vs. compute-bound regions
- Why **prefill is compute-bound and decode is memory-bound** — the central asymmetry of the field
- Reading a GPU spec sheet: HBM bandwidth vs. peak FLOPs, and the ratio between them
- Latency–throughput tension, and why "fast" is meaningless without saying for whom

**Read**
- Anything on the roofline model; then a modern LLM-inference-arithmetic writeup (Kipply's "Transformer Inference Arithmetic" is the standard reference)

**Exercise:** hand-derive the theoretical minimum TPOT for a 7B model in fp16 on a given GPU using only bandwidth and parameter count. Then measure the real thing and account for the gap.

**Checkpoint:** given a model, a GPU, and a batch size, you can predict roughly where you sit on the roofline and what the binding constraint is.

---

## Module 5 — Batching

**Learn**
- Static batching and why it wastes enormous capacity when sequences finish at different times
- Dynamic batching
- **Continuous / in-flight batching** — evict finished sequences, admit new ones each step
- **Chunked prefill** and prefill/decode interference: how one long prompt stalls everyone else's decode
- Scheduling policies and fairness

**Read**
- *Orca: A Distributed Serving System for Transformer-Based Generative Models* (OSDI '22) — introduced continuous batching

**Exercise:** simulate (no GPU needed) static vs. continuous batching over a realistic distribution of sequence lengths. Measure GPU utilization and mean latency under each.

---

## Module 6 — Memory optimization

**Learn**
- **PagedAttention:** virtual-memory paging applied to the KV cache; near-zero fragmentation
- **Prefix caching / radix trees:** reuse across requests sharing a prompt prefix
- Architectural KV reduction: **MQA → GQA → MLA**, and what each trades away
- Quantization: weight-only (GPTQ, AWQ), weight+activation (SmoothQuant), and **KV cache quantization**
- Where quality actually degrades, and how to measure it honestly

**Read**
- *Efficient Memory Management for LLM Serving with PagedAttention* (vLLM, SOSP '23)
- *GQA: Training Generalized Multi-Query Transformer Models* (Ainslie et al., 2023)
- *GPTQ* and *AWQ*
- DeepSeek-V2 paper for **MLA** (multi-head latent attention)

**Exercise:** recompute your Module 2 KV cache number under MQA, GQA-8, and fp8 KV quantization. Report the batch-size increase each buys.

---

## Module 7 — Attention kernels

**Learn**
- Why the GPU memory hierarchy (HBM ↔ SRAM) is where attention performance is decided
- **FlashAttention:** tiling and online softmax, avoiding materialization of the N×N matrix
- Why FlashAttention is exact, not an approximation — and why that's the point
- **FlashDecoding** and why the decode phase needs a different kernel than prefill
- Kernel fusion and the cost of memory round-trips
- Where the efficient-attention research line went: linear attention, SSMs/Mamba, hybrids

**Read**
- *FlashAttention* (Dao et al., 2022), then FlashAttention-2
- Optionally: *Mamba* — to see the alternative bet on the recurrence side

**Exercise:** you don't need to write CUDA. Do trace through the FlashAttention tiling algorithm on paper for a small matrix and convince yourself the online softmax is exact.

---

## Module 8 — Speculative decoding

The cleverest idea in the field: buy back the wasted compute in memory-bound decoding.

**Learn**
- Draft-then-verify; why verification is a parallel (compute-bound) operation
- The rejection-sampling proof that output distribution is **preserved exactly**
- Acceptance rate as the governing metric; when speculation is a net loss
- Self-speculation: **Medusa** heads, **EAGLE**, n-gram/prompt lookup
- Tree attention and multi-candidate verification

**Read**
- *Fast Inference from Transformers via Speculative Decoding* (Leviathan et al., 2023)
- *Accelerating LLM Decoding with Speculative Sampling* (Chen et al., 2023)
- *Medusa*, then *EAGLE*

**Exercise:** derive the expected speedup as a function of acceptance rate, draft length, and the draft/target cost ratio. Find the break-even point where speculation starts hurting.

**Checkpoint:** you can explain why speculative decoding is free performance rather than a quality/speed tradeoff.

---

## Module 9 — Distributed inference

**Learn**
- **Tensor parallelism:** splitting within a layer; the all-reduce cost per layer
- **Pipeline parallelism:** splitting across layers; bubbles
- **Expert parallelism** for MoE; routing, load imbalance, capacity factors
- **Prefill/decode disaggregation:** run the two phases on separate hardware pools
- Interconnect as the real constraint (NVLink vs. PCIe vs. Ethernet)

**Read**
- *Megatron-LM* (for TP)
- *DistServe* or *Splitwise* (for disaggregation)
- A Mixtral or DeepSeek-MoE paper for the sparse-model view

**Exercise:** for a 70B model, work out the minimum GPU count under fp16 and under int8, and explain what TP degree does to TTFT versus what it does to throughput.

---

## Module 10 — Systems, evaluation, and the frontier

**Learn**
- How vLLM, SGLang, and TensorRT-LLM differ in design philosophy
- Benchmarking that isn't misleading: request distributions, warmup, percentiles over means
- Evaluating optimized models — perplexity is not sufficient
- Open frontier problems: long-context inference, KV compression/eviction, test-time compute and reasoning-model serving, agentic/multi-turn workloads

**Exercise:** reproduce a published inference benchmark and try to explain any discrepancy you find. This is genuinely instructive — the discrepancies are where the real understanding is.

---

## How we'll work through it

For each module:

1. **Concept pass** — I explain it, targeting intuition first, mechanism second.
2. **Math by hand** — you work the numbers. This is non-negotiable; the arithmetic is where the understanding lives.
3. **Code lab** — small, self-contained, runnable on CPU or a free Colab GPU.
4. **Paper reading** — I give you a reading frame first (what to extract, what to skim), then we discuss it.
5. **Teach-back** — you explain the module to me. I probe the soft spots.
6. **Checkpoint** — I tell you honestly whether you're ready to move on.

Say "Module 0" (or wherever you'd like to start) and we begin.

---

## A note on currency

The foundational papers above are stable, but inference is a fast-moving area and there will be work newer than my training data. When we reach the later modules — especially 8 and 10 — ask me to search for recent developments rather than trusting my recall.