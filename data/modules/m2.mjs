export default {
  n: 2,
  slug: 'kv-cache',
  title: 'THE KV CACHE',
  tagline: 'The single most important object in inference. Everything downstream is memory management.',
  hours: '6–8 hours',
  prereqs: ['Module 0', 'Module 1'],

  bigIdea: `Module 1 established that decode is memory-bound and that batching is the way out.
This module explains why you cannot simply crank the batch size to 295 and go home.

Every sequence in flight owns a private, growing block of GPU memory: its KV cache. The weights
are shared by all sequences — load them once, use them for everyone. The KV cache is not shared.
Each sequence pays for its own, and the bill grows with every token generated.

So your batch size is not a tuning parameter you choose. It is a **quotient**:

\`\`\`
max_batch = (GPU_memory - weights - workspace) / kv_bytes_per_sequence
\`\`\`

That is the master equation of LLM serving. Weights are fixed the moment you pick a model.
Workspace is roughly fixed. Everything else — every technique in Modules 5 through 9 — is an
attempt to make the numerator bigger or the denominator smaller, so that more sequences fit and
each weight load does more useful work.

Learn to compute \`kv_bytes_per_sequence\` in your head. It is the most useful piece of mental
arithmetic in the field.`,

  concepts: [
    {
      name: 'What is stored, and deriving the size formula',
      keyPoint: 'kv_bytes = 2 × layers × kv_heads × head_dim × seq_len × batch × dtype_bytes, and every term is there for a reason you can name.',
      body: `Build the formula up rather than memorising it. For **one token, at one layer, in one
attention head**, you store:

- a key vector of \`head_dim\` numbers,
- a value vector of \`head_dim\` numbers.

That is \`2 × head_dim\` numbers. The leading **2** is K and V — nothing more mysterious than that.

Now multiply out the dimensions you have:

\`\`\`
x kv_heads      each KV head has its own K and V (this is n_kv_heads, NOT n_heads)
x layers        every layer attends independently and keeps its own cache
x seq_len       one entry per position, and it only ever grows
x batch         no sharing between sequences
x dtype_bytes   2 for fp16/bf16, 1 for fp8/int8
\`\`\`

Giving:

\`\`\`
kv_bytes = 2 * n_layers * n_kv_heads * head_dim * seq_len * batch * dtype_bytes
\`\`\`

The term people get wrong is \`n_kv_heads\`. On a GQA model this is much smaller than \`n_heads\`,
and using the wrong one inflates your estimate by the GQA ratio — 4× on Llama-3-8B, 8× on
Llama-3-70B. Check the config, not the head count.

The most useful form is **bytes per token per sequence**, because it strips out the two things
that vary at runtime:

\`\`\`
bytes_per_token = 2 * n_layers * n_kv_heads * head_dim * dtype_bytes
\`\`\`

For Llama-3-8B at fp16: \`2 × 32 × 8 × 128 × 2 = 131,072 bytes = 128 KiB per token\`.
For Llama-3-70B at fp16: \`2 × 80 × 8 × 128 × 2 = 327,680 bytes = 320 KiB per token\`.

Commit those two numbers. From them everything else is one multiplication: an 8k-context 70B
sequence is \`320 KiB × 8192 = 2.5 GiB\`, and thirty-two of them is 80 GiB — an entire H100,
holding nothing but cache.

Note what is **not** in the formula: \`n_heads\`, \`d_model\`, and \`vocab_size\` are all absent.
The KV cache is governed by the *KV* side of attention alone, which is precisely why
architectural attacks on it (MQA, GQA, MLA in Module 6) are so effective — they change one term
in a product.`,
      ascii: `  per token, per layer:

    K  [n_kv_heads=8, head_dim=128]  ─┐
                                      ├─ 2 x 8 x 128 x 2 bytes = 4 KiB
    V  [n_kv_heads=8, head_dim=128]  ─┘

  x 80 layers                     = 320 KiB   per token   (Llama-3-70B)
  x 8192 tokens                   = 2.5 GiB   per sequence
  x 32 sequences                  = 80 GiB    <- one entire H100`,
    },
    {
      name: 'Why the KV cache, not the weights, caps your batch size',
      keyPoint: 'Weights are a one-time fixed cost shared by every sequence; KV cache is a per-sequence cost that grows without bound.',
      body: `Weights and KV cache behave completely differently as you scale, and confusing them
is the most common planning error in LLM serving.

**Weights are fixed and shared.** A 70B model at fp16 is 140 GB whether you serve one request or
a thousand. Load once, use for everyone.

**KV cache is per-sequence and grows.** Every admitted request adds its own, and every token
generated grows it.

Put real numbers on it. Serving **Llama-3-70B on 4× H100 80GB** (320 GB total):

\`\`\`
total memory                        320 GB
- weights (fp16)                    140 GB
- activations, workspace, frag       ~20 GB
                                    -------
available for KV cache              ~160 GB
\`\`\`

At 320 KiB per token, that budget buys you \`160e9 / 327680 ≈ 488,000 tokens\` of cache — total,
across all sequences. How that divides up depends entirely on context length:

| context per request | KV per sequence | concurrent sequences |
|---|---|---|
| 1,024 | 320 MiB | ~477 |
| 4,096 | 1.25 GiB | ~119 |
| 8,192 | 2.5 GiB | ~59 |
| 32,768 | 10 GiB | ~14 |
| 131,072 | 40 GiB | ~3 |

Read the last row. On a **$120,000 machine**, a 70B model at 128k context can hold **three**
concurrent requests. Not three hundred. Three.

Now connect it to Module 1. Decode arithmetic intensity equals batch size, and an H100's ridge
point is 295 FLOP/byte. At 8k context you can reach batch 59 — roughly 20% of the way to the
ridge, so you are still 5× off saturating the machine's arithmetic. At 128k context you reach
batch 3, which is 1% of the ridge. **The KV cache is what stands between you and the throughput
the hardware is capable of.**

This is why memory optimization (Module 6) is not a side quest. Halving the KV cache does not
just save memory; it doubles the batch size, which doubles arithmetic intensity, which
approximately doubles throughput. Memory work *is* throughput work.`,
      ascii: '',
    },
    {
      name: 'The GQA multiplier, and what it bought',
      keyPoint: 'Grouped-query attention divides the KV cache by n_heads/n_kv_heads, and it is the reason long context is affordable at all.',
      body: `Run the formula with the KV-head count as the only variable, on Llama-3-70B
(\`n_layers=80\`, \`head_dim=128\`, fp16, 8k context, batch 32):

| scheme | kv_heads | bytes/token | per 8k seq | batch 32 |
|---|---|---|---|---|
| MHA (64 q heads, 64 kv) | 64 | 2.5 MiB | 20 GiB | **640 GiB** |
| GQA-8 (what Llama 3 ships) | 8 | 320 KiB | 2.5 GiB | **80 GiB** |
| MQA (single kv head) | 1 | 40 KiB | 320 MiB | **10 GiB** |

The MHA row is the one to sit with. A hypothetical multi-head Llama-3-70B would need **640 GiB
of KV cache** to serve 32 sequences at 8k — eight H100s of pure cache, on top of the two you need
for weights. Ten GPUs to serve 32 users. That configuration is not expensive, it is unshippable.

GQA-8 turns it into 80 GiB, which fits alongside the weights on a 4-GPU node with room to spare.
**Grouped-query attention is not a minor efficiency tweak. It is the architectural decision that
made long-context serving economically possible**, and it is why essentially every model released
since 2023 uses it.

The trade is quality, and the honest framing is that it is small but real. Ainslie et al. found
GQA lands close to MHA quality while training much faster than converting to MQA, and MQA —
sharing a single KV head across all queries — degrades noticeably more. The sweet spot the field
converged on is 4 to 8 KV heads, and it was found empirically rather than derived.

There is a second, subtler benefit that matters in Module 9. Tensor parallelism splits attention
heads across GPUs, and **you cannot split a KV head across two devices without replicating it**.
With 8 KV heads you can run TP-8 cleanly, one KV head per GPU. Beyond TP-8 you must duplicate KV
heads across devices, and the cache stops shrinking as you add GPUs. The KV head count silently
sets a ceiling on how far you can usefully parallelize.`,
      ascii: '',
    },
    {
      name: 'Fragmentation: the waste nobody budgets for',
      keyPoint: 'Naive allocators reserve max_seq_len per request, so a system that looks full is typically 60–80% empty.',
      body: `The arithmetic above assumes you use every byte you allocate. Pre-vLLM systems did
not come close.

The problem is that you do not know how long a sequence will be until it finishes. A request
arrives, you must give it cache memory, and its final length is unknown. The obvious safe answer
is to allocate for the worst case: reserve \`max_seq_len\` up front, contiguously.

If \`max_seq_len\` is 8,192 and the request generates 200 tokens, you reserved 2.5 GiB and used
61 MiB. **97.6% waste.** And because you took a contiguous block, the unused portion cannot be
lent to anyone else.

The vLLM paper breaks the waste into three kinds, and the distinction is worth keeping:

- **Internal fragmentation** — space reserved inside a sequence's own allocation that it never
  fills. The dominant term, and the one above.
- **External fragmentation** — free memory that exists but is split into chunks too small to
  satisfy a contiguous request. The classic malloc problem.
- **Reservation waste** — memory held for future tokens of a sequence that is currently running.
  Legitimately needed eventually, but idle right now.

Their measurement is the memorable one: in existing systems, **only 20.4% to 38.2% of allocated
KV cache memory held actual token state.** Between 60% and 80% was waste. Meaning that a system
reporting "GPU memory full, cannot admit requests" was, most of the time, two-thirds empty.

There is a subtler cost too. Because you must reserve for the worst case, \`max_seq_len\` becomes
a *global* tax. Raising your advertised context window from 8k to 32k shrinks your maximum batch
size by 4× **even if no request ever uses more than 2k**, because every request still reserves
32k. Operators discovered this the hard way.

PagedAttention (Module 6) fixes this by borrowing the idea operating systems settled on in the
1960s: allocate in small fixed-size blocks, keep a per-sequence block table, and let physical
memory be non-contiguous. Waste drops below 4%. It is a genuinely lovely result — a hard systems
problem solved by noticing it was a solved problem in a different field.`,
      ascii: `  NAIVE: reserve max_seq_len contiguously

  seq A  [####································]  200/8192 used
  seq B  [#########···························]  900/8192 used
  seq C  [##··································]  150/8192 used
         ^^^^^^^^^^ used     ^^^^^^^^^^^^^^^^^^ reserved, wasted, unlendable

  PAGED: 16-token blocks, allocated on demand, non-contiguous

  seq A  [#][#][#][#]                     ->  free pool: [ ][ ][ ][ ][ ][ ]...
  seq B  [#][#][#][#][#][#][#][#]             any sequence can take any block
  seq C  [#][#][#]`,
    },
    {
      name: 'The two-regime memory budget',
      keyPoint: 'Below a threshold the weights dominate your bytes-per-step; above it the KV cache does, and your optimization targets change completely.',
      body: `A decode step reads two things: all the weights, and all the KV cache of every
sequence in the batch. Which dominates determines what you should be optimizing.

\`\`\`
bytes_per_step = weight_bytes  +  batch * seq_len * kv_bytes_per_token
                 (constant)       (grows with load and context)
\`\`\`

Llama-3-8B at fp16: weights ≈ 15.0 GB, KV ≈ 128 KiB/token. Setting the terms equal:

\`\`\`
batch * seq_len = 15.0e9 / 131072 = 114,441 tokens
\`\`\`

So the crossover is at about **114,000 cached tokens in flight**. Below that you are
weight-dominated; above it, cache-dominated.

| situation | tokens in flight | regime |
|---|---|---|
| batch 1, 2k context | 2,048 | weights, overwhelmingly (0.02×) |
| batch 32, 2k context | 65,536 | weights still (0.57×) |
| batch 32, 8k context | 262,144 | **cache dominates** (2.3×) |
| batch 8, 128k context | 1,048,576 | cache, crushingly (9.2×) |

This table should change how you read optimization advice, because the two regimes want opposite
things:

**Weight-dominated** (small batch, short context — interactive single-user, local inference).
Quantize the weights. int4 weight-only quantization nearly quadruples decode speed here because
weight bytes *are* your step time. KV optimizations do almost nothing. This is why llama.cpp is
obsessed with weight quantization: it is the correct obsession for its regime.

**Cache-dominated** (large batch, long context — production serving). Weight quantization helps
much less. You want KV quantization, GQA/MLA, PagedAttention, prefix caching, eviction. This is
why vLLM and SGLang are built the way they are.

Most arguments about "the best inference optimization" are really two people in different regimes
talking past each other. Ask which side of the crossover they are on and the disagreement usually
dissolves.`,
      ascii: '',
    },
    {
      name: 'Cache reuse: the free win hiding in chat',
      keyPoint: 'Multi-turn conversation re-sends a prefix that has not changed, so the KV cache for it can be kept rather than recomputed.',
      body: `Consider a chat application. Turn 3 of a conversation sends:

\`\`\`
[system prompt: 500 tokens]
[user turn 1] [assistant turn 1]
[user turn 2] [assistant turn 2]
[user turn 3]                        <- only this is new
\`\`\`

A stateless server prefills all of it — say 3,000 tokens — to produce one new token. But the KV
entries for the first 2,900 tokens are *identical* to what it computed last turn. Causality
guarantees it: appending tokens cannot change the K or V of anything earlier.

If you keep that cache, turn 3's prefill is 100 tokens instead of 3,000. A **30× reduction in
TTFT**, for free, with no approximation.

The same idea applies across *different* requests that share a prefix. A thousand users hitting
the same 800-token system prompt share those 800 tokens' worth of KV state exactly. Compute it
once, point every sequence at it. This is **prefix caching**, and with a radix tree to find the
longest matching prefix (SGLang's RadixAttention) it generalises to arbitrary sharing patterns —
few-shot examples, document QA over a shared document, agent loops that replay a growing
trajectory.

Agentic workloads make this decisive rather than merely nice. An agent that takes 20 steps
re-sends its entire growing history every step. Without prefix caching, total prefill work is
quadratic in the number of steps. With it, each step prefills only the newest action and
observation.

Three things make this harder than it sounds, and all three are real bugs people ship:

1. **RoPE position indices.** Cached keys were rotated by their absolute position. Reuse a prefix
   at a different offset without handling this and you get silently wrong attention.
2. **Eviction policy.** Cache memory is the same memory sequences need. Keeping prefixes for
   possible future hits costs you batch size *now*. LRU is the usual answer; the right answer
   depends on your traffic.
3. **Isolation.** Sharing cache across users is sharing derived state across users. The state is
   keyed by exact token prefix, so a hit requires an exact match — but it is worth knowing this
   is a cross-tenant surface, and that hit/miss timing can in principle leak whether someone else
   has sent a given prefix.`,
      ascii: '',
    },
  ],

  mathLab: {
    prompt: `The canonical calculation. Do it on paper before touching a calculator.

**Model: Llama-3-70B.** Config: \`n_layers = 80\`, \`n_heads = 64\`, \`n_kv_heads = 8\`,
\`head_dim = 128\`, \`d_model = 8192\`, 70.6 B parameters.

1. KV cache bytes **per token per sequence** at fp16.
2. KV cache for **one sequence at 8,192 context**. In GiB.
3. KV cache for **batch 32 at 8,192 context**. In GiB.
4. Model weights at fp16, in GB. Compare to (3). Which is bigger?
5. You have **4× H100 80GB** = 320 GB, and you reserve 20 GB for activations and workspace. How
   many 8,192-token sequences fit concurrently?
6. Redo (5) at **128k context**. How many sequences now?
7. Now the payoff. Recompute (3) under: **(a)** MQA (1 KV head), **(b)** fp8 KV cache, **(c)**
   both together. For each, state the batch-size multiplier it buys.
8. Given decode arithmetic intensity ≈ batch size and an H100 ridge point of 295 FLOP/byte, which
   of these configurations gets you closest to saturating the GPU's arithmetic?`,

    solution: `**1. Bytes per token**

\`\`\`
2 (K and V) x 80 layers x 8 kv_heads x 128 head_dim x 2 bytes
  = 2 x 80 x 8 x 128 x 2
  = 327,680 bytes
  = 320 KiB per token
\`\`\`

**2. One sequence at 8,192 tokens**

\`\`\`
327,680 x 8,192 = 2,684,354,560 bytes
                = 2,621,440 KiB = 2,560 MiB = 2.5 GiB
\`\`\`

**3. Batch 32**

\`\`\`
2.5 GiB x 32 = 80 GiB   (85.9 GB)
\`\`\`

**4. Weights**

\`\`\`
70.6e9 x 2 bytes = 141.2 GB = 131.5 GiB
\`\`\`

Weights (131.5 GiB) still exceed the cache (80 GiB) at this batch and context — but only by
1.6×. Push to batch 64 and the cache wins. The two are the same order of magnitude, which is the
regime change that makes KV memory a first-class concern.

**5. Concurrent sequences at 8k**

\`\`\`
320 GB total - 141.2 GB weights - 20 GB workspace = 158.8 GB for KV
158.8e9 / 2.684e9 per sequence = 59.2  ->  59 sequences
\`\`\`

**6. At 128k context**

\`\`\`
per sequence: 327,680 x 131,072 = 42.95e9 bytes = 40 GiB
158.8e9 / 42.95e9 = 3.7  ->  3 sequences
\`\`\`

Three. On four H100s. This is the long-context serving problem in one number.

**7. The optimizations**, all at 8k context, batch 32:

**(a) MQA, 1 KV head:**

\`\`\`
2 x 80 x 1 x 128 x 2 = 40,960 bytes/token  (40 KiB, an 8x reduction)
batch 32 at 8k: 40,960 x 8192 x 32 = 10.7e9 = 10 GiB
new capacity: 158.8e9 / (40,960 x 8192) = 473 sequences
multiplier vs GQA-8: 8x
\`\`\`

**(b) fp8 KV cache, keeping GQA-8:**

\`\`\`
2 x 80 x 8 x 128 x 1 = 163,840 bytes/token  (160 KiB, a 2x reduction)
batch 32 at 8k: 40 GiB
new capacity: 118 sequences
multiplier: 2x
\`\`\`

**(c) Both:**

\`\`\`
2 x 80 x 1 x 128 x 1 = 20,480 bytes/token  (20 KiB, a 16x reduction)
batch 32 at 8k: 5 GiB
new capacity: 946 sequences
multiplier: 16x
\`\`\`

**8. Which saturates the GPU?**

Decode arithmetic intensity ≈ batch size; the H100 ridge is 295 FLOP/byte.

\`\`\`
GQA-8, fp16:      59 sequences   ->  intensity ~59    20% of ridge
fp8 KV:          118 sequences   ->  intensity ~118   40% of ridge
MQA, fp16:       473 sequences   ->  past the ridge -- now COMPUTE-bound
MQA + fp8:       946 sequences   ->  well past the ridge
\`\`\`

The baseline configuration reaches only about 20% of the ridge point: even with memory perfectly
packed, four H100s spend most of every decode step waiting on HBM. Combining KV optimizations
pushes past the ridge and **changes the bottleneck from memory to compute** — at which point
memory work stops paying and you switch to Modules 7 and 8.

That is the real lesson. The goal of KV optimization is not "use less memory." It is to buy
enough batch size to cross the ridge point, after which the same effort buys you nothing and you
should be doing something else.

*(Caveat for honesty: real systems do not hit these numbers. MQA costs quality, fp8 KV costs a
little more, and at batch 473 you are contending with scheduling overhead, attention kernels that
scale with total cached tokens, and the fact that 946 concurrent users have wildly heterogeneous
lengths. Treat these as upper bounds that tell you where the ceiling is, not throughput
predictions.)*`,
  },

  codeLab: {
    goal: `Two parts.

**Part A** — write a KV cache calculator you will actually keep. Feed it a model config and get
back bytes per token, per-sequence cost at a given context, and how many sequences fit in a given
memory budget. Then use it to reproduce the math lab.

**Part B** — go back to Module 1's generation loop and instrument the real cache, so you can see
your formula predict PyTorch's actual allocation to the byte.`,
    code: `"""
Part A: the KV cache calculator. Keep this file.
Part B: verify the formula against a real model's actual cache.
"""
from dataclasses import dataclass

GiB = 1024 ** 3
GB = 10 ** 9


@dataclass
class ModelConfig:
    name: str
    n_layers: int
    n_heads: int
    n_kv_heads: int
    head_dim: int
    n_params: float           # total parameters

    def kv_bytes_per_token(self, dtype_bytes=2):
        # 2 = one K vector and one V vector.
        # NOTE: n_kv_heads, not n_heads. This is the term people get wrong.
        return 2 * self.n_layers * self.n_kv_heads * self.head_dim * dtype_bytes

    def kv_bytes(self, seq_len, batch=1, dtype_bytes=2):
        return self.kv_bytes_per_token(dtype_bytes) * seq_len * batch

    def weight_bytes(self, dtype_bytes=2):
        return int(self.n_params * dtype_bytes)

    def max_batch(self, total_mem_bytes, seq_len,
                  weight_dtype=2, kv_dtype=2, workspace_bytes=20 * GB):
        budget = total_mem_bytes - self.weight_bytes(weight_dtype) - workspace_bytes
        if budget <= 0:
            return 0
        return int(budget // self.kv_bytes(seq_len, 1, kv_dtype))


LLAMA3_8B = ModelConfig("Llama-3-8B", 32, 32, 8, 128, 8.03e9)
LLAMA3_70B = ModelConfig("Llama-3-70B", 80, 64, 8, 128, 70.6e9)
MISTRAL_7B = ModelConfig("Mistral-7B", 32, 32, 8, 128, 7.24e9)


def report(cfg, seq_len, batch, total_mem, kv_dtype=2):
    per_tok = cfg.kv_bytes_per_token(kv_dtype)
    print(f"\\n=== {cfg.name} @ {seq_len} ctx, batch {batch}, kv={kv_dtype}B ===")
    print(f"  kv per token       {per_tok:>14,} B  ({per_tok/1024:.0f} KiB)")
    print(f"  kv per sequence    {cfg.kv_bytes(seq_len)/GiB:>14.2f} GiB")
    print(f"  kv for batch       {cfg.kv_bytes(seq_len, batch, kv_dtype)/GiB:>14.2f} GiB")
    print(f"  weights (fp16)     {cfg.weight_bytes()/GB:>14.1f} GB")
    print(f"  max concurrent     {cfg.max_batch(total_mem, seq_len, kv_dtype=kv_dtype):>14,}")


# --- reproduce the math lab ---
FOUR_H100 = 4 * 80 * GB
report(LLAMA3_70B, 8192, 32, FOUR_H100)
report(LLAMA3_70B, 131072, 32, FOUR_H100)
report(LLAMA3_70B, 8192, 32, FOUR_H100, kv_dtype=1)      # fp8 KV

# --- what GQA actually bought us ---
print("\\n=== Llama-3-70B @ 8k, batch 32: KV scheme comparison ===")
print(f"  {'scheme':<12} {'kv/token':>12} {'batch-32 KV':>14} {'max seqs':>10}")
for label, kvh in [("MHA (64)", 64), ("GQA-8", 8), ("MQA (1)", 1)]:
    c = ModelConfig("x", 80, 64, kvh, 128, 70.6e9)
    print(f"  {label:<12} {c.kv_bytes_per_token()/1024:>9.0f} KiB"
          f" {c.kv_bytes(8192, 32)/GiB:>11.1f} GiB"
          f" {c.max_batch(FOUR_H100, 8192):>10,}")

# --- the two-regime crossover ---
print("\\n=== where does KV overtake weights? (Llama-3-8B) ===")
crossover = LLAMA3_8B.weight_bytes() / LLAMA3_8B.kv_bytes_per_token()
print(f"  crossover at {crossover:,.0f} cached tokens in flight")
for b, s in [(1, 2048), (32, 2048), (32, 8192), (8, 131072)]:
    tot = b * s
    ratio = LLAMA3_8B.kv_bytes(s, b) / LLAMA3_8B.weight_bytes()
    regime = "CACHE-dominated" if ratio > 1 else "weight-dominated"
    print(f"  batch {b:>3}, ctx {s:>6}: {tot:>9,} tokens  kv/weights={ratio:>5.2f}  {regime}")


# ============================================================
# Part B: check the formula against a real model
# ============================================================
def verify():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    name = "gpt2"
    model = AutoModelForCausalLM.from_pretrained(name)
    tok = AutoTokenizer.from_pretrained(name)
    c = model.config

    # GPT-2 is plain MHA: n_kv_heads == n_head
    cfg = ModelConfig(name, c.n_layer, c.n_head, c.n_head,
                      c.n_embd // c.n_head, 124e6)

    ids = tok("Attention is all you need, but memory bandwidth is what you get.",
              return_tensors="pt").input_ids
    seq = ids.shape[1]

    with torch.no_grad():
        out = model(ids, use_cache=True)

    # count real bytes in past_key_values
    actual = 0
    for layer in out.past_key_values:
        for t in layer:                      # (key, value)
            actual += t.numel() * t.element_size()

    dtype_bytes = out.past_key_values[0][0].element_size()
    predicted = cfg.kv_bytes(seq, 1, dtype_bytes)

    print(f"\\n=== Part B: formula vs reality ({name}, seq={seq}) ===")
    print(f"  layers={cfg.n_layers} kv_heads={cfg.n_kv_heads} "
          f"head_dim={cfg.head_dim} dtype={dtype_bytes}B")
    print(f"  predicted  {predicted:>12,} bytes")
    print(f"  actual     {actual:>12,} bytes")
    print(f"  match: {predicted == actual}")


if __name__ == "__main__":
    try:
        verify()
    except ImportError:
        print("\\n(pip install torch transformers to run Part B)")

# --- TODO for you ---
#   1. Add an mla_bytes_per_token() for DeepSeek-style latent attention:
#      per layer it stores ONE latent vector of kv_lora_rank (512) plus a
#      rope part (64), so 576 values -- not 2 x heads x head_dim. Compare.
#   2. Add a method that answers: "given N GPUs of M GB, what is the longest
#      context I can serve at batch B?" Invert the equation.
`,
    expect: `Part A reproduces the math lab exactly: 320 KiB per token for the 70B, 2.5 GiB per
8k sequence, 80 GiB at batch 32, ~59 concurrent sequences on 4×H100, and 3 at 128k context. The
GQA table shows MHA needing 640 GiB and MQA needing 10 GiB for the same workload.

The crossover section prints ~114,441 tokens, and correctly labels batch 32 at 2k context as
weight-dominated but batch 32 at 8k as cache-dominated.

Part B prints \`match: True\`. GPT-2 small has 12 layers, 12 heads, head_dim 64, fp32, so at a
16-token sequence: \`2 × 12 × 12 × 64 × 4 × 16 = 1,179,648 bytes\`. Seeing your formula predict
PyTorch's allocation to the byte is the point of the exercise — after this you will trust the
arithmetic.

If it does not match, the usual causes are: the model has GQA and you passed \`n_head\` for
\`n_kv_heads\`; or the cache dtype is not what you assumed.`,
    stretch: `Return to Module 1's timing loop and plot the *cached* line against your predicted
KV cache size at each step. The residual slope of that line is the cache being read. Then extend
the calculator to model **PagedAttention**: given a block size of 16 tokens, compute the internal
fragmentation for a realistic distribution of sequence lengths (try a lognormal with median 300
and a long tail) and compare against the naive reserve-\`max_seq_len\` allocator. You should
reproduce something close to the vLLM paper's finding that naive allocation wastes 60–80%.`,
  },

  papers: [
    {
      title: 'Fast Transformer Decoding: One Write-Head is All You Need',
      by: 'Noam Shazeer, 2019',
      url: 'https://arxiv.org/abs/1911.02150',
      why: 'The MQA paper — the first to name the KV cache as *the* inference bottleneck and attack it architecturally. Four years ahead of the field.',
      frame: `Six pages. Read **Section 2** for the incremental-decoding cost model, which is
essentially this module's argument written in 2019. **Section 3** defines MQA in a paragraph. The
memory-bandwidth analysis at the start is the historically important part: Shazeer identified the
problem before hardware trends made it universally painful.`,
    },
    {
      title: 'GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints',
      by: 'Ainslie et al., 2023',
      url: 'https://arxiv.org/abs/2305.13245',
      why: 'GQA, which every current open-weight model uses. Also shows how to convert an existing MHA checkpoint rather than retraining.',
      frame: `**Section 2.2** defines GQA and **Section 2.1** describes the uptraining procedure —
mean-pool the KV heads within each group, then fine-tune on ~5% of the original compute. Figure 1
is the picture worth remembering. **Figure 5** has the quality-versus-speed curve that justifies
8 groups. Skim the experiments.`,
    },
    {
      title: 'Efficient Memory Management for Large Language Model Serving with PagedAttention',
      by: 'Kwon et al., SOSP 2023',
      url: 'https://arxiv.org/abs/2309.06180',
      why: 'The vLLM paper. Read the first half now for the fragmentation analysis; the full treatment is Module 6.',
      frame: `For this module read **Section 3** only — the memory-waste breakdown and Figure 2,
which is where the "20.4–38.2% utilization" figure comes from. Stop before Section 4. You will
come back for PagedAttention itself.`,
    },
    {
      title: 'Efficiently Scaling Transformer Inference',
      by: 'Pope et al., 2022',
      url: 'https://arxiv.org/abs/2211.05102',
      why: 'Has the cleanest treatment of how KV cache size interacts with parallelism and batch size.',
      frame: 'Revisit **Section 2** with the KV formula in hand. Their cost model separates weight traffic from KV traffic explicitly, and the two-regime split of this module falls straight out of their equations.',
    },
  ],

  checkpoint: {
    claim: `You can compute a KV cache footprint in your head from a config file, and you can
explain why long context is expensive at inference time in a way that has nothing to do with
attention's O(N²) compute cost.`,
    questions: [
      {
        q: 'Explain why 128k context is expensive at inference, without mentioning the O(N²) cost of attention.',
        a: `Because the KV cache is linear in context length and must be held in GPU memory for
the entire lifetime of the request. Llama-3-70B stores 320 KiB per token, so a 128k-token
sequence needs 40 GiB of cache — half an H100 for one user. On a 4×H100 node with 141 GB of
weights and 20 GB of workspace, you have about 159 GB left, which fits **three** such sequences.
Batch size collapses to 3, arithmetic intensity collapses with it, and you are running a
$120,000 machine at roughly 1% of its arithmetic capability. The compute cost of attention is a
separate problem that mostly bites during prefill. The memory cost is what stops you serving
anyone.`,
      },
      {
        q: 'A model has 48 layers, 40 attention heads, 8 KV heads, head_dim 128. What is its KV cache per token at fp16, and at fp8?',
        a: `\`2 × 48 × 8 × 128 × 2 = 196,608 bytes = 192 KiB\` per token at fp16, and half that at
fp8: \`98,304 bytes = 96 KiB\`. The 40 query heads do not appear anywhere in the calculation — only
\`n_kv_heads\` matters. Using 40 instead of 8 would give 960 KiB and overstate the cache by 5×,
which is the single most common error in these estimates.`,
      },
      {
        q: 'Your serving system reports "out of memory" at batch 12 when your arithmetic says batch 40 should fit. What is the most likely cause?',
        a: `Fragmentation from a naive allocator that reserves \`max_seq_len\` contiguously for
every request. If \`max_seq_len\` is 8,192 and typical requests use 2,000 tokens, you are wasting
about 75% of allocated cache — turning a theoretical batch of 40 into a real batch of 10–12,
which matches. The vLLM paper measured exactly this: only 20.4–38.2% of allocated KV memory held
real tokens. The fix is paged allocation in small blocks. Worth ruling out first, though: check
whether you also under-budgeted activation workspace, and whether CUDA graph capture is holding
extra memory.`,
      },
      {
        q: 'Why does halving your KV cache do more than just save memory?',
        a: `Because it converts directly into throughput. Halving the per-sequence cache doubles
how many sequences fit, which doubles the batch size. Decode arithmetic intensity equals batch
size, so intensity doubles too — you now do twice as much useful arithmetic per byte of weights
streamed. Since decode is memory-bound, throughput roughly doubles. Memory work is throughput
work. The caveat is that this holds only until you reach the ridge point (≈295 FLOP/byte on
H100); past it you are compute-bound and further KV savings buy nothing.`,
      },
      {
        q: 'When would you prefer weight quantization over KV cache quantization, and vice versa?',
        a: `It depends on which side of the crossover you are on. Bytes per step are
\`weight_bytes + batch × seq_len × kv_bytes_per_token\`. For Llama-3-8B those terms are equal at
about 114,000 cached tokens in flight. Below that — small batch, short context, single-user local
inference — weights dominate and int4 weight quantization gives you close to a linear speedup;
KV quantization does nearly nothing. Above it — large batch, long context, production serving —
the cache dominates, weight quantization gives diminishing returns, and fp8/int8 KV is what buys
you batch size. Most disagreements about "the best optimization" are two people in different
regimes.`,
      },
      {
        q: 'Why does the number of KV heads limit how far you can scale tensor parallelism?',
        a: `Tensor parallelism splits attention heads across GPUs, and a KV head cannot be split
across two devices without replicating its cache on both. With 8 KV heads you can run TP-8 with
exactly one KV head per GPU and the per-GPU cache shrinks 8×. At TP-16 you must duplicate each KV
head across two GPUs, so per-GPU cache stops shrinking even though you added hardware — you are
paying for GPUs that add bandwidth and compute but no cache capacity. This is a real constraint
on large-model deployment and one reason KV head count is chosen with parallelism in mind, not
just quality.`,
      },
    ],
  },

  pitfalls: [
    {
      wrong: 'Using n_heads instead of n_kv_heads in the KV cache formula.',
      right: `The most common error in this arithmetic, and it inflates your estimate by the GQA
ratio — 4× on Llama-3-8B, 8× on Llama-3-70B. Only the K and V projections contribute to the
cache, and on a GQA model there are fewer of those than query heads. Always read \`n_kv_heads\`
(or \`num_key_value_heads\`) out of the config; never infer it from the head count.`,
    },
    {
      wrong: 'The KV cache is small compared to the model weights.',
      right: `True at small batch and short context, false exactly where production lives. For
Llama-3-8B the two are equal at about 114,000 cached tokens in flight — batch 32 at 8k context is
already past it, at 2.3× the weight bytes. At batch 8 with 128k context the cache is 9× the
weights. Whether it is "small" is a question about your operating point, not a property of the
model.`,
    },
    {
      wrong: 'Raising max_seq_len is free if nobody uses long contexts.',
      right: `Not with a naive allocator. If you reserve \`max_seq_len\` per request up front,
raising the advertised limit from 8k to 32k cuts your maximum batch size by 4× even when every
real request is 500 tokens — the reservation is charged whether or not it is used. This is the
single most expensive configuration mistake in LLM serving. With PagedAttention the problem
mostly goes away, because blocks are allocated on demand.`,
    },
    {
      wrong: 'Prefix caching is an approximation that might change the output.',
      right: `It is exact. Causal masking guarantees the K and V of a prefix are bit-identical
regardless of what follows, so reusing them gives the same result as recomputing. The failure
modes are implementation bugs, not approximation — most commonly mishandling RoPE position
indices when a prefix is reused at a different offset, which corrupts attention silently. Done
correctly, prefix caching is free performance with no quality cost.`,
    },
  ],

  glossary: [
    { term: 'KV cache', def: 'The stored key and value tensors for all previous positions, kept so they are not recomputed each step. Per-sequence, and it grows with every token.' },
    { term: 'bytes per token', def: '2 x n_layers x n_kv_heads x head_dim x dtype_bytes. The most useful derived number in inference planning.' },
    { term: 'internal fragmentation', def: 'Memory reserved inside a sequence allocation that the sequence never uses. The dominant waste in naive allocators.' },
    { term: 'external fragmentation', def: 'Free memory that exists but is broken into pieces too small to satisfy a contiguous allocation.' },
    { term: 'prefix caching', def: 'Reusing cached KV state across requests that share a leading token sequence. Exact, not approximate.' },
    { term: 'RadixAttention', def: "SGLang's radix-tree index over cached prefixes, enabling automatic longest-prefix reuse across arbitrary sharing patterns." },
    { term: 'crossover point', def: 'The number of cached tokens in flight at which KV traffic overtakes weight traffic. About 114k tokens for Llama-3-8B at fp16.' },
    { term: 'MQA', def: 'Multi-query attention: a single KV head shared by all query heads. Maximum cache savings, largest quality cost.' },
    { term: 'workspace', def: 'GPU memory reserved for activations, temporary buffers, and communication. Typically 10-20 GB, and easy to forget when budgeting.' },
  ],
};
