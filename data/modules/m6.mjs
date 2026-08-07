export default {
  n: 6,
  slug: 'memory-optimization',
  title: 'MEMORY OPTIMIZATION',
  tagline: 'Paging, prefix reuse, architectural KV reduction, and quantization — four attacks on the same denominator.',
  hours: '8–10 hours',
  prereqs: ['Module 2', 'Module 4', 'Module 5'],

  bigIdea: `Module 2 gave you the master equation:

\`\`\`
max_batch = (GPU_memory - weights - workspace) / kv_bytes_per_sequence
\`\`\`

Module 4 established that batch size *is* your arithmetic intensity, and Module 5 gave you the
scheduler that keeps the batch full. This module is about the equation's two variable terms —
making the numerator bigger and the denominator smaller — and it is where most of the practical
engineering in LLM serving lives.

There are exactly four ways to attack it, and they compose:

1. **Stop wasting what you allocate.** PagedAttention. Recovers 60–80% of memory that naive
   allocators throw away. Free — no quality cost at all.
2. **Stop storing the same thing twice.** Prefix caching. Also free and also exact.
3. **Store less per token, by design.** MQA, GQA, MLA. Requires training the model that way.
4. **Store the same thing in fewer bits.** Quantization of weights and of the cache. Cheap, and
   the only one of the four that costs quality.

The ordering is deliberate: do the free ones first. A surprising amount of production tuning
consists of reaching for quantization — which has a quality cost you then have to measure and
defend — when the system is still throwing away two thirds of its cache to fragmentation.`,

  concepts: [
    {
      name: 'PagedAttention: virtual memory for the KV cache',
      keyPoint: 'Allocate the cache in small fixed-size blocks with a per-sequence block table, so physical memory need not be contiguous and nothing is reserved for tokens that may never exist.',
      body: `Module 2 established the waste: a naive allocator reserves \`max_seq_len\` contiguously
per sequence, and the vLLM authors measured that only **20.4% to 38.2%** of allocated KV memory
actually held tokens. Between 60% and 80% was fragmentation.

Operating systems solved exactly this problem in the 1960s, and the solution transfers almost
without modification.

**The mechanism.** Divide the KV cache into fixed-size **blocks** — typically 16 tokens' worth.
Each sequence gets a **block table** mapping its logical positions to physical block numbers.
Blocks are allocated on demand as the sequence grows, from a shared pool, and returned when it
finishes. Physical blocks belonging to one sequence need not be adjacent.

\`\`\`
sequence A, 35 tokens, block size 16:
  block table: [ 7, 2, 19 ]          3 blocks = 48 slots, 35 used

  physical pool:
    block 0  [ free    ]
    block 1  [ seq B   ]
    block 2  [ seq A   ]  <- logical positions 16..31
    block 3  [ seq C   ]
    ...
    block 7  [ seq A   ]  <- logical positions 0..15
    ...
    block 19 [ seq A   ]  <- logical positions 32..34, 13 slots spare
\`\`\`

**Waste drops to at most one partial block per sequence** — on average half a block, so 8 tokens.
Against 128 KiB/token on Llama-3-8B that is about 1 MiB per sequence, versus gigabytes under
reservation. vLLM reports waste below 4%.

The attention kernel has to change to match: instead of reading a contiguous \`[seq_len,
head_dim]\` tensor, it gathers through the block table. That indirection costs a little, and the
gain vastly outweighs it.

**Copy-on-write falls out for free**, and this is the elegant part. Two sequences sharing a prefix
can point their block tables at the *same physical blocks*. Nothing is copied. When one of them
writes to a shared block — which only happens at the boundary block where they diverge — that
block is copied and the writer's table updated. Parallel sampling with \`n=4\` from one prompt
therefore costs one copy of the prompt's cache, not four. Beam search gets the same treatment.

**Choosing the block size** is a real trade-off. Small blocks (8) minimize internal fragmentation
but mean more block-table entries and more indirection per attention call. Large blocks (32+)
reduce overhead but waste more on partial blocks and make prefix sharing coarser — two sequences
must share a whole block to share anything. 16 is the common default and is a reasonable
compromise rather than a derived optimum.`,
      ascii: `  NAIVE                              PAGED
  reserve max_seq_len contiguously   16-token blocks from a shared pool

  A [####·······················]    A -> [7][2][19]
  B [#########··················]    B -> [1][4][11][3]
  C [##·························]    C -> [0]
    ^ used  ^ reserved, unlendable
                                     free pool: [5][6][8][9][10][12]...
  utilization 20-38%                 utilization > 96%
                                     and A,B can SHARE a block if they
                                     share a prefix -- no copy`,
    },
    {
      name: 'Prefix caching and radix trees',
      keyPoint: 'Requests that share a leading token sequence share its KV state exactly, so it should be computed once and pointed at, not recomputed.',
      body: `Module 2 introduced the idea; here is how it is actually built.

The naive version handles one fixed system prompt: hash it, cache its KV blocks, reuse them.
Useful, but it only catches the case you anticipated.

**RadixAttention** — SGLang's contribution — generalises it. Maintain a radix tree (a compressed
trie) whose edges are token sequences and whose nodes hold the corresponding KV blocks. For each
incoming request, walk the tree to find the **longest matching prefix**, reuse those blocks, and
prefill only the remainder. Insert the new suffix as you go. Evict by LRU when memory is short.

This catches sharing you did not design for:

| workload | what is shared |
|---|---|
| chat, turn N | everything before the newest user message |
| few-shot prompting | the entire example block |
| RAG over one document | the document, across every question about it |
| agent loop, step N | the whole trajectory so far |
| parallel sampling | the prompt, across all \`n\` samples |
| tree search / self-consistency | every common ancestor in the tree |

The agentic case is the one that changes the economics. An agent taking 20 steps re-sends its
growing history every step, so without prefix caching the total prefill work is quadratic in step
count. With it, each step prefills only the newest action and observation. For long-running
agents this is not an optimization, it is the difference between viable and not.

Three implementation realities:

**RoPE positions must be handled.** Cached keys were rotated by their absolute position. Reuse a
prefix at a different offset without accounting for this and attention is silently wrong — no
crash, just degraded output. This is the most common prefix-caching bug.

**Block granularity limits matching.** With 16-token blocks you can only share whole blocks, so
two prompts differing at token 5 share nothing even though the first 5 tokens match. Prefixes
that matter should be block-aligned, which is one reason system prompts are often padded.

**Eviction is a genuine trade-off, not a detail.** Cached prefixes occupy memory that running
sequences need. Keeping a prefix costs you batch size *now* against a possible hit *later*. LRU is
the usual answer and it is a heuristic; the right policy depends on your traffic's reuse
distribution, which most operators have never measured.

There is also a security consideration worth naming: cache is shared derived state across users.
Hits require an exact token-prefix match, so content does not leak directly, but hit/miss timing
can in principle reveal whether another user has submitted a given prefix. For most deployments
this is acceptable; for multi-tenant systems with sensitive prompts it is worth an explicit
decision rather than a default.`,
      ascii: `  radix tree over cached prefixes

              [ "You are a helpful assistant." ]        <- 1 copy, shared by all
                     /                    \\
        [ "Summarize:" ]              [ "Translate to French:" ]
           /        \\                        |
    [ doc A ]   [ doc B ]              [ sentence 1 ]
        |
   [ "What is the main point?" ]   <- only this is prefilled on a new request`,
    },
    {
      name: 'MQA to GQA to MLA: attacking the formula itself',
      keyPoint: 'The cache formula has n_kv_heads and head_dim as factors, so architectures that shrink either one shrink the cache proportionally.',
      body: `Paging and prefix caching stop you wasting memory. Architecture changes how much you
need in the first place.

Recall the formula: \`2 × n_layers × n_kv_heads × head_dim × dtype_bytes\`. Two of those terms are
architectural choices.

**MHA → MQA → GQA** attack \`n_kv_heads\`, and Module 2 has the numbers: on Llama-3-70B, MHA needs
2.5 MiB/token, GQA-8 needs 320 KiB, MQA needs 40 KiB. GQA at 4–8 groups is where the field
settled, because MQA's single shared KV head costs measurably more quality than the memory is
worth for most models.

**MLA** — multi-head latent attention, from DeepSeek-V2 — attacks the problem differently and more
aggressively. Instead of reducing the *number* of KV heads, it compresses K and V jointly into a
single low-rank **latent vector** per token, and caches only that.

\`\`\`
standard GQA, per token per layer:
    K: n_kv_heads x head_dim  = 8 x 128 = 1024 values
    V: n_kv_heads x head_dim  = 8 x 128 = 1024 values
                                          ---- 2048 values

MLA, per token per layer:
    c_kv: kv_lora_rank        = 512 values      (compressed K and V together)
    k_rope: rope dim          =  64 values      (decoupled positional part)
                                 ---- 576 values
\`\`\`

At inference, the per-head K and V are reconstructed from the latent by up-projection. The trick
that makes this cheap is **absorption**: the up-projection matrices can be folded into the
adjacent query and output projections algebraically, so you never actually materialize the full K
and V — you compute attention directly against the latent.

The decoupled RoPE part exists because rotation does not commute with the low-rank projection.
Applying RoPE to a compressed latent and then decompressing does not give the same answer as
compressing rotated keys, so MLA keeps a small separate rotary component. It is an
implementation-forced wrinkle rather than an elegant piece of design, and it is where most of the
complexity of MLA lives.

DeepSeek report a KV reduction of roughly **93% versus MHA**, and — the part that made people pay
attention — *better* benchmark quality than the MHA baseline, not worse. The plausible explanation
is that the low-rank bottleneck acts as a regularizer. Whether that generalises beyond their
models is not yet established.

The catch for all three: **these are training-time decisions.** You cannot convert a deployed MHA
model to MLA. GQA has an uptraining recipe (mean-pool the KV heads within a group, fine-tune on
~5% of original compute), but MLA requires training from scratch. If you are choosing a model to
serve, its attention architecture is one of the most consequential facts about its inference cost,
and it is fixed before you get involved.`,
      ascii: '',
    },
    {
      name: 'Weight quantization: where the bytes actually are',
      keyPoint: 'At small batch and short context, weight bytes are essentially the whole decode cost, so 4-bit weights approach a 4x speedup.',
      body: `Recall the two regimes from Module 2. In the **weight-dominated** regime — small
batch, short context, single-user local inference — weight bytes are your step time, so halving
them roughly halves your latency.

**Weight-only quantization** stores weights at low precision and dequantizes to fp16 on the fly
for the matmul. The arithmetic still happens in fp16; the saving is purely in bytes moved, which
is exactly what you want when you are memory-bound.

**GPTQ** (Frantar et al., 2022) quantizes layer by layer, using approximate second-order
information from a small calibration set to choose quantization points that minimize the
layer's output error. It works down to 3–4 bits on large models with modest perplexity loss.

**AWQ** (Lin et al., 2023) starts from the observation that not all weights matter equally: a
small fraction of channels — around 1% — are salient, identifiable from *activation* magnitude
rather than weight magnitude. Scale those channels up before quantizing so they land on finer
grid points. Simple, no backpropagation, and generally more robust than GPTQ at 4 bits.

**Weight+activation quantization** is a different game. **SmoothQuant** (Xiao et al., 2022)
addresses the fact that activations have severe outliers that make them hard to quantize, by
migrating the difficulty into the weights via a per-channel scaling that cancels out
mathematically. This enables int8 for both operands, which speeds up the *arithmetic* too — worth
it only when you are compute-bound, which for decode you are not.

**FP8** on Hopper and later is the pragmatic modern default for datacenter serving: hardware
support, roughly half the bytes, and quality loss small enough that many deployments accept it
without extensive validation. **FP4** on Blackwell extends the same idea further.

A rough guide:

| precision | size (8B model) | typical quality cost |
|---|---|---|
| fp16 / bf16 | 16.1 GB | baseline |
| fp8 | 8.0 GB | small, often negligible |
| int4 (GPTQ/AWQ) | ~4.5 GB with scales | small but real; task-dependent |
| int3 | ~3.5 GB | noticeable |
| int2 | ~2.5 GB | severe without special methods |

Two cautions. First, **the speedup is not the compression ratio** — you also move activations and
KV cache, and dequantization costs some arithmetic. 4-bit weights typically give 2.5–3.5×, not 4×.
Second, and more important: **in the cache-dominated regime, weight quantization does much less
than you expect.** At batch 32 with 8k context on Llama-3-8B, the KV cache is 2.3× the weights, so
even eliminating weight bytes entirely would cut your step time by only about 30%. Check which
regime you are in before choosing.`,
      ascii: '',
    },
    {
      name: 'KV cache quantization, and why K and V want different treatment',
      keyPoint: 'Keys have per-channel outliers and values do not, so quantizing keys per-channel and values per-token works far better than treating them alike.',
      body: `In the cache-dominated regime, quantizing the cache is the lever that matters. It
halves (fp8/int8) or quarters (int4) the denominator of the batch equation directly.

The naive approach — quantize everything per-tensor — degrades quality more than you would expect
from the bit count. The reason is a real structural asymmetry, documented in the KIVI work and
others:

**Keys have strong per-channel outliers.** Certain channels of the key vectors consistently carry
much larger magnitudes than others, across tokens. Quantizing a key tensor per-tensor means those
outlier channels dominate the scale and every other channel is crushed into a few quantization
levels.

**Values do not show this pattern.** Value distributions are comparatively uniform across
channels.

So the correct treatment is asymmetric:

\`\`\`
K: quantize PER CHANNEL   -- each channel gets its own scale, so an outlier
                             channel does not ruin the others

V: quantize PER TOKEN     -- each token's value vector gets its own scale
\`\`\`

Per-channel quantization of K has an awkward consequence: the cache grows one token at a time, so
a per-channel scale computed over the tokens seen so far changes as you append. Implementations
handle this by quantizing in groups and keeping a small recent window in full precision.

Practical guidance:

- **fp8 KV is close to free** on hardware with fp8 support. It is a reasonable default for
  production serving and doubles your batch capacity.
- **int8 KV** with the per-channel/per-token split is also reliable.
- **int4 KV** is where quality starts to show, and it shows *specifically* on long-context tasks —
  which is a problem, because long context is exactly when you wanted the savings. Test on
  retrieval and long-context benchmarks, not perplexity.

There is a related family that reduces the cache without changing its precision: **eviction and
compression**.

- **StreamingLLM** observed that models allocate large attention weight to the first few tokens
  regardless of content — "attention sinks" — and that keeping those plus a sliding window lets a
  model stream indefinitely without collapse. Cheap and effective for streaming; it does discard
  the middle, so it is not suitable when you need recall over the full context.
- **H2O** keeps "heavy hitter" tokens identified by accumulated attention score, plus recent
  tokens. Better recall than a pure window.
- **SnapKV** compresses the prompt's cache at the end of prefill by observing which positions the
  final query attends to.

All of these are **lossy in a way that is hard to bound**. You cannot know at eviction time which
token some future query will need. They work well on benchmarks that resemble their design
assumptions and can fail sharply outside them. Treat them as workload-specific tools, not
defaults.`,
      ascii: '',
    },
    {
      name: 'Measuring quality honestly',
      keyPoint: 'Perplexity is not sufficient: quantization damage concentrates in long context, rare tokens, and multi-step reasoning — exactly what perplexity averages away.',
      body: `Every technique in this module except paging and prefix caching costs something. The
discipline that separates good work from bad here is measurement, and the field is not good at it.

**Why perplexity is insufficient.** Perplexity is the average negative log-likelihood over a
corpus. It is dominated by common tokens in ordinary contexts, and a quantized model reproduces
those nearly perfectly. Damage concentrates in the tail: rare tokens, unusual contexts, long
dependencies, precise arithmetic. A 4-bit model can match fp16 perplexity to two decimal places
and fail visibly at 20-step reasoning.

There is a second problem: perplexity is measured on short sequences, usually 2k–4k tokens. If
your concern is a quantized KV cache, that is exactly the regime where it looks fine.

**What to measure instead**, roughly in order of value per unit effort:

1. **Long-context retrieval.** Needle-in-a-haystack and, better, RULER — which tests multi-needle,
   aggregation and tracing rather than a single lookup. This is where KV quantization damage shows
   first and clearest.
2. **Multi-step reasoning.** GSM8K, MATH, or your own chain-of-thought tasks. Errors compound
   across steps, so small per-token degradation becomes visible.
3. **Your actual task.** A few hundred real prompts with a human or model-graded comparison
   against the fp16 baseline. Unglamorous and the most informative thing on this list.
4. **Output distribution divergence.** KL divergence between quantized and baseline logits on a
   fixed prompt set. Sensitive, cheap, and catches problems before they show in task metrics.
5. **Standard benchmark suites.** MMLU and friends. Useful as a regression check, weak as a
   quality signal — they are multiple-choice and forgiving.

**The comparison must be controlled.** Same prompts, same sampling parameters, same seed where
possible. It is remarkably common to see quantization "evaluated" against a baseline that used
different sampling settings, which measures nothing.

**Test at the operating point.** If you serve at 32k context, evaluating at 2k tells you almost
nothing about KV quantization. If you serve at batch 64, evaluate under batching — batch-dependent
kernel selection can itself shift outputs, as Module 3 discussed.

The honest position on most published quantization results is that they demonstrate the method
does not catastrophically fail, not that it is free. Whether it is free *for your workload* is a
question only your evaluation can answer.`,
      ascii: '',
    },
  ],

  mathLab: {
    prompt: `**Llama-3-70B**: 80 layers, 64 query heads, 8 KV heads, head_dim 128, 70.6 B
parameters. Serving on **4× H100 80GB** = 320 GB, with 20 GB reserved for workspace. Context
8,192 tokens throughout.

**Part 1 — the baseline.** Recompute from Module 2: KV bytes per token at fp16, per-sequence cost
at 8k, weight bytes, and how many sequences fit.

**Part 2 — stack the optimizations.** For each configuration, compute KV bytes per token,
sequences that fit, and the multiplier versus baseline:

  a) baseline: GQA-8, fp16 weights, fp16 KV
  b) + fp8 KV cache
  c) + fp8 weights as well
  d) + int4 weights (assume 4.5 bits effective including scales), fp8 KV
  e) hypothetical MLA: 576 values per token per layer at fp16, with fp8 weights

**Part 3 — fragmentation.** Your allocator reserves \`max_seq_len = 32768\` contiguously per
request, but actual sequences average 8,192 tokens. Under configuration (a):

  a) How much memory does each request actually reserve?
  b) How many sequences fit now?
  c) What fraction of allocated KV memory holds real tokens?
  d) PagedAttention with 16-token blocks: what is the average waste per sequence, and how many
     sequences fit?

**Part 4 — turn it into throughput.** Decode arithmetic intensity equals batch size and the H100
ridge point is 295. For configurations (a) through (e), state the intensity reached and whether
you cross the ridge. Which single change buys the most?

**Part 5 — the ordering question.** You have time to implement exactly one thing. Your system
currently uses naive contiguous allocation, GQA-8, fp16 everything. What do you do first, and
why?`,

    solution: `**Part 1 — baseline**

\`\`\`
kv/token = 2 x 80 x 8 x 128 x 2      = 327,680 B = 320 KiB
per 8k sequence = 327,680 x 8,192    = 2.684 GB  (2.5 GiB)
weights = 70.6e9 x 2                 = 141.2 GB
available = 320 - 141.2 - 20         = 158.8 GB
sequences = 158.8 / 2.684            = 59
\`\`\`

**Part 2 — stacking**

**(a) baseline:** 320 KiB/token, **59 sequences**, 1.0×

**(b) + fp8 KV:**
\`\`\`
kv/token = 2 x 80 x 8 x 128 x 1 = 163,840 B = 160 KiB
per sequence = 1.342 GB
available unchanged at 158.8 GB
sequences = 158.8 / 1.342 = 118        (2.0x)
\`\`\`

**(c) + fp8 weights:**
\`\`\`
weights = 70.6e9 x 1 = 70.6 GB
available = 320 - 70.6 - 20 = 229.4 GB
sequences = 229.4 / 1.342 = 170        (2.9x)
\`\`\`

**(d) int4 weights (4.5 bits effective), fp8 KV:**
\`\`\`
weights = 70.6e9 x 4.5/8 = 39.7 GB
available = 320 - 39.7 - 20 = 260.3 GB
sequences = 260.3 / 1.342 = 193        (3.3x)
\`\`\`

**(e) MLA, fp16 latent, fp8 weights:**
\`\`\`
kv/token = 80 layers x 576 values x 2 bytes = 92,160 B = 90 KiB
   (note: no factor of 2 -- the latent already encodes both K and V)
per sequence = 92,160 x 8,192 = 0.755 GB
available = 229.4 GB
sequences = 229.4 / 0.755 = 303        (5.1x)
\`\`\`

MLA versus GQA-8 on the cache alone: \`320 / 90 = 3.6×\`. Against a hypothetical MHA baseline of
2.5 MiB/token it is \`2560 / 90 = 28×\`, consistent with DeepSeek's ~93% reduction claim.

**Part 3 — fragmentation**

a) Each request reserves \`327,680 × 32,768 = 10.74 GB\`, and uses 2.684 GB.

b) \`158.8 / 10.74 = 14 sequences\`. Down from 59 — a **4.2× loss** to reservation alone.

c) \`2.684 / 10.74 = 25.0%\` of allocated KV memory holds real tokens. This sits squarely inside
the 20.4–38.2% range vLLM measured in the wild, which is a good sign the model is realistic.

d) With 16-token blocks, waste is at most one partial block per sequence, averaging 8 tokens:
\`\`\`
waste = 8 x 327,680 = 2.62 MB per sequence
per sequence = 2.684 GB + 0.0026 GB = 2.687 GB
sequences = 158.8 / 2.687 = 59
utilization = 2.684 / 2.687 = 99.9%
\`\`\`

**Paging alone takes you from 14 sequences to 59 — a 4.2× improvement — at zero quality cost.**
That is the single largest win available, and it is free.

**Part 4 — throughput**

\`\`\`
config                        batch   intensity   vs ridge (295)
(a) baseline paged              59        59        20%   memory-bound
(b) + fp8 KV                   118       118        40%   memory-bound
(c) + fp8 weights              170       170        58%   memory-bound
(d) + int4 weights             193       193        65%   memory-bound
(e) MLA + fp8 weights          303       303       103%   AT THE RIDGE
\`\`\`

Marginal gains per step:

\`\`\`
naive -> paged:        14 -> 59     +45 sequences   (free)
paged -> fp8 KV:       59 -> 118    +59 sequences   (small quality cost)
fp8 KV -> fp8 weights:118 -> 170    +52 sequences   (small quality cost)
fp8 w -> int4 w:      170 -> 193    +23 sequences   (real quality cost)
GQA -> MLA:           170 -> 303   +133 sequences   (requires a different model)
\`\`\`

The best single change you can *make* is fp8 KV (+59). The best change overall is architectural
(MLA, +133) but you cannot apply it to an existing checkpoint. Note that int4 weights buy the
least (+23) while costing the most quality — because at batch 170 you are firmly cache-dominated,
so weight bytes are no longer where your problem is. This is the two-regime rule biting.

**Part 5 — what to do first**

**PagedAttention, unambiguously.** It is 4.2× — larger than any other single change available —
and it costs nothing in quality. Every other option on the list trades accuracy for memory, and
it would be perverse to spend accuracy while still discarding 75% of the cache to fragmentation.

The general principle: **do the free optimizations before the lossy ones.** Paging and prefix
caching are exact. Quantization is not. A system that has not exhausted the exact wins has no
business reaching for the approximate ones.

Second priority is prefix caching, also exact, with a payoff that depends entirely on your
traffic — negligible for one-shot independent prompts, transformative for chat and agents.

Only then fp8 KV, and only then weight quantization.`,
  },

  codeLab: {
    goal: `Two parts.

**Part A** — implement a block allocator and compare it against a naive reserving allocator over
a realistic distribution of sequence lengths. Reproduce the vLLM paper's utilization finding
yourself.

**Part B** — build the prefix-sharing structure. A radix tree over token sequences that reports,
for a stream of requests, how many prefill tokens you saved.`,
    code: `"""
Part A: paged vs naive KV allocation.
Part B: a radix tree for prefix cache reuse.

    pip install numpy
"""
import numpy as np

rng = np.random.default_rng(11)
GB = 10 ** 9


# ==========================================================================
# Part A -- allocators
# ==========================================================================
class NaiveAllocator:
    """Reserve max_seq_len contiguously, up front. The pre-vLLM approach."""

    def __init__(self, total_bytes, bytes_per_token, max_seq_len):
        self.total = total_bytes
        self.bpt = bytes_per_token
        self.max_seq = max_seq_len
        self.reserved = 0
        self.live = {}

    def admit(self, sid):
        need = self.bpt * self.max_seq
        if self.reserved + need > self.total:
            return False
        self.reserved += need
        self.live[sid] = 0
        return True

    def append(self, sid, n=1):
        self.live[sid] += n                     # already reserved; nothing to do

    def release(self, sid):
        self.reserved -= self.bpt * self.max_seq
        del self.live[sid]

    def utilization(self):
        used = sum(self.live.values()) * self.bpt
        return used / self.reserved if self.reserved else 0.0


class PagedAllocator:
    """Fixed-size blocks from a shared pool, allocated on demand."""

    def __init__(self, total_bytes, bytes_per_token, block_tokens=16):
        self.bpt = bytes_per_token
        self.block_tokens = block_tokens
        self.block_bytes = bytes_per_token * block_tokens
        self.n_blocks = int(total_bytes // self.block_bytes)
        self.free = list(range(self.n_blocks))
        self.tables = {}                        # sid -> [physical block ids]
        self.lengths = {}

    def _blocks_needed(self, n_tokens):
        return (n_tokens + self.block_tokens - 1) // self.block_tokens

    def admit(self, sid, initial_tokens=0):
        need = self._blocks_needed(initial_tokens)
        if need > len(self.free):
            return False
        self.tables[sid] = [self.free.pop() for _ in range(need)]
        self.lengths[sid] = initial_tokens
        return True

    def append(self, sid, n=1):
        self.lengths[sid] += n
        need = self._blocks_needed(self.lengths[sid])
        while len(self.tables[sid]) < need:
            if not self.free:
                return False                    # out of memory -- caller must preempt
            self.tables[sid].append(self.free.pop())
        return True

    def release(self, sid):
        self.free.extend(self.tables.pop(sid))
        del self.lengths[sid]

    def utilization(self):
        allocated = sum(len(t) for t in self.tables.values()) * self.block_tokens
        used = sum(self.lengths.values())
        return used / allocated if allocated else 0.0


def run(alloc, lengths, label):
    """Admit as many as fit, grow them to completion, report peak concurrency."""
    admitted, peak = [], 0
    for i, L in enumerate(lengths):
        ok = alloc.admit(i, 1) if isinstance(alloc, PagedAllocator) else alloc.admit(i)
        if not ok:
            break
        admitted.append((i, L))
        peak = max(peak, len(admitted))

    # grow everyone to their final length
    for sid, L in admitted:
        alloc.append(sid, L - 1)

    print(f"  {label:<34} concurrent {len(admitted):>4}   "
          f"utilization {alloc.utilization()*100:>5.1f}%")
    return len(admitted)


# --- Llama-3-70B on 4x H100, 20 GB workspace ---
KV_PER_TOKEN = 2 * 80 * 8 * 128 * 2            # 327,680 bytes = 320 KiB
AVAILABLE = 320 * GB - 141.2 * GB - 20 * GB
MAX_SEQ = 32768

lengths = rng.lognormal(mean=8.9, sigma=0.35, size=500).astype(int).clip(512, MAX_SEQ)
print(f"=== KV allocation: {len(lengths)} requests, mean length "
      f"{lengths.mean():.0f}, max {lengths.max()} ===")
print(f"    budget {AVAILABLE/GB:.1f} GB, {KV_PER_TOKEN/1024:.0f} KiB/token, "
      f"max_seq_len {MAX_SEQ}\\n")

n_naive = run(NaiveAllocator(AVAILABLE, KV_PER_TOKEN, MAX_SEQ), lengths,
              f"naive (reserve {MAX_SEQ})")
n_paged = run(PagedAllocator(AVAILABLE, KV_PER_TOKEN, 16), lengths,
              "paged (16-token blocks)")
print(f"\\n  paging improvement: {n_paged/max(n_naive,1):.1f}x concurrent sequences, "
      f"at zero quality cost")

print("\\n=== block size trade-off ===")
for bs in (1, 8, 16, 32, 64, 128):
    a = PagedAllocator(AVAILABLE, KV_PER_TOKEN, bs)
    n = run(a, lengths, f"block size {bs}")


# ==========================================================================
# Part B -- radix tree prefix cache
# ==========================================================================
class RadixNode:
    __slots__ = ("children", "n_tokens")

    def __init__(self):
        self.children = {}          # first token -> (edge_tokens tuple, RadixNode)
        self.n_tokens = 0           # tokens cached along the path to here


class PrefixCache:
    """Longest-prefix matching over cached token sequences."""

    def __init__(self):
        self.root = RadixNode()
        self.cached_tokens = 0

    def match_and_insert(self, tokens):
        """Returns how many leading tokens were already cached."""
        node, i = self.root, 0
        while i < len(tokens):
            key = tokens[i]
            if key not in node.children:
                break
            edge, child = node.children[key]
            # how far along this edge do we agree?
            j = 0
            while j < len(edge) and i + j < len(tokens) and edge[j] == tokens[i + j]:
                j += 1
            if j == len(edge):
                node, i = child, i + j          # consumed the whole edge, descend
                continue
            # partial match: split the edge
            mid = RadixNode()
            node.children[key] = (edge[:j], mid)
            mid.children[edge[j]] = (edge[j:], child)
            node, i = mid, i + j
            break

        matched = i
        if i < len(tokens):                     # insert the remaining suffix
            suffix = tuple(tokens[i:])
            node.children[suffix[0]] = (suffix, RadixNode())
            self.cached_tokens += len(suffix)
        return matched


def synth_chat(n_turns, sys_len=500):
    """A conversation: shared system prompt, growing history."""
    system = list(range(1000, 1000 + sys_len))
    history, out = list(system), []
    for t in range(n_turns):
        user = list(rng.integers(0, 50000, size=int(rng.integers(20, 120))))
        history = history + user
        out.append(list(history))
        history = history + list(rng.integers(0, 50000, size=int(rng.integers(80, 400))))
    return out


print("\\n=== prefix cache: a 10-turn conversation ===")
cache = PrefixCache()
total, saved = 0, 0
for turn, req in enumerate(synth_chat(10)):
    m = cache.match_and_insert(req)
    total += len(req)
    saved += m
    print(f"  turn {turn:>2}: {len(req):>6} tokens, {m:>6} cached, "
          f"{len(req)-m:>5} to prefill")
print(f"\\n  total prompt tokens  {total:>8,}")
print(f"  served from cache    {saved:>8,}  ({saved/total*100:.1f}%)")

print("\\n=== prefix cache: 200 requests sharing one system prompt ===")
cache = PrefixCache()
system = list(range(1000, 1500))
total, saved = 0, 0
for _ in range(200):
    req = system + list(rng.integers(0, 50000, size=int(rng.integers(50, 300))))
    m = cache.match_and_insert(req)
    total += len(req)
    saved += m
print(f"  total prompt tokens  {total:>8,}")
print(f"  served from cache    {saved:>8,}  ({saved/total*100:.1f}%)")

# --- TODO for you ---
#   1. Add LRU eviction to PrefixCache with a token budget. Measure hit rate
#      as the budget shrinks -- this is the real operational trade-off.
#   2. Make the radix tree block-aligned (16-token granularity) and see how
#      much the hit rate drops. That is the cost of paging the prefix cache.
#   3. Add copy-on-write to PagedAllocator: let two sequences share blocks
#      and only copy on divergence. Measure the saving for n=8 parallel sampling.
`,
    expect: `**Part A.** The naive allocator admits around **14 sequences** and reports
utilization near **25%** — right inside the 20.4–38.2% band the vLLM paper measured. The paged
allocator admits around **59** at over **99%** utilization. A **~4× improvement at zero quality
cost**, reproduced from first principles.

The block-size sweep shows the trade-off directly: block size 1 gives perfect utilization (and
would be unusable in practice — one table entry per token), while 128 wastes noticeably more.
Everything from 8 to 32 is within a percent or two, which is why 16 is a fine default and not a
finely-tuned optimum.

**Part B.** The conversation shows the hit rate climbing every turn — by turn 10, typically
**90%+** of the prompt is already cached, so you prefill only the newest message. This is the
mechanism that makes multi-turn chat affordable.

The shared-system-prompt case shows savings around **60–75%**, depending on the sampled user
message lengths. With a longer system prompt or shorter user messages it goes higher.

Note what these numbers mean in latency: a 90% cache hit on a 3,000-token prompt turns a 3,000-token
prefill into a 300-token prefill, roughly a 10× TTFT improvement, exactly and with no quality
cost.`,
    stretch: `Combine the two. Give \`PagedAllocator\` copy-on-write block sharing, back the
\`PrefixCache\` with real blocks, and add LRU eviction under a fixed memory budget. Then simulate a
mixed workload — 60% chat with long histories, 40% one-shot prompts — and plot the trade-off
curve: as you give more memory to the prefix cache, hit rate rises but concurrent batch size
falls. Find the point that maximizes total throughput. That curve is the actual operational
decision every serving deployment makes, usually without measuring it.`,
  },

  papers: [
    {
      title: 'Efficient Memory Management for LLM Serving with PagedAttention',
      by: 'Kwon et al., SOSP 2023',
      url: 'https://arxiv.org/abs/2309.06180',
      why: 'The vLLM paper. The most influential systems paper in LLM inference, and a model of how to transfer a solved idea from one field to another.',
      frame: `Now read it properly. **Section 3** is the fragmentation analysis (Figure 2 is the
20.4–38.2% figure). **Section 4** is PagedAttention and the block manager — pay attention to
**4.3 on copy-on-write sharing**, which is the part that makes parallel sampling and beam search
cheap and which most summaries omit. **Section 4.5** covers scheduling and preemption. Skim the
evaluation, but note the ablation on block size.`,
    },
    {
      title: 'SGLang: Efficient Execution of Structured Language Model Programs',
      by: 'Zheng et al., 2023',
      url: 'https://arxiv.org/abs/2312.07104',
      why: 'RadixAttention — generalizing prefix reuse from a special case to the default structure of the workload.',
      frame: 'The RadixAttention section is the payload: the radix tree, longest-prefix matching, and the LRU eviction policy. Read the frontend language section more quickly — it matters for how the system is used, less for the mechanism. Note the cache-aware scheduling discussion, which is where the real subtlety is.',
    },
    {
      title: 'DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model',
      by: 'DeepSeek-AI, 2024',
      url: 'https://arxiv.org/abs/2405.04434',
      why: 'Multi-head latent attention. The most aggressive architectural attack on the KV cache, and the one that claims to improve quality rather than trade it.',
      frame: `**Section 2.1** is MLA. Work through the low-rank compression and then the
**absorption** trick — folding the up-projections into the adjacent query and output matrices is
what makes it fast, and it is easy to miss. The decoupled RoPE component in 2.1.3 exists because
rotation does not commute with the low-rank projection; understand why before accepting it. The
MoE material is Module 9.`,
    },
    {
      title: 'AWQ: Activation-aware Weight Quantization',
      by: 'Lin et al., 2023',
      url: 'https://arxiv.org/abs/2306.00978',
      why: 'The most practical 4-bit weight quantization method. Simple, no backpropagation, and generally robust.',
      frame: 'Read **Section 3**: the observation that ~1% of channels are salient, that saliency is identifiable from *activation* magnitude rather than weight magnitude, and that scaling those channels before quantization protects them. The whole method is one good idea, clearly explained.',
    },
    {
      title: 'GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers',
      by: 'Frantar et al., 2022',
      url: 'https://arxiv.org/abs/2210.17323',
      why: 'The other standard 4-bit method, and the one that showed accurate one-shot quantization of very large models was possible at all.',
      frame: 'Read Sections 2 and 3 for the layer-wise objective and the approximate second-order solution. The algorithmic details of the Cholesky reformulation matter less than the framing: quantization as a per-layer reconstruction problem.',
    },
    {
      title: 'Efficient Streaming Language Models with Attention Sinks',
      by: 'Xiao et al., 2023',
      url: 'https://arxiv.org/abs/2309.17453',
      why: 'The attention-sink phenomenon: models allocate large attention weight to the first few tokens regardless of content, and keeping them plus a window enables indefinite streaming.',
      frame: 'The observation in Section 3 is the interesting part and is genuinely surprising — the sink tokens carry little semantic content but removing them collapses the model. Be clear-eyed that this method discards the middle of the context, so it enables streaming rather than long-context recall.',
    },
  ],

  checkpoint: {
    claim: `You can look at a serving deployment and say, in order, which memory optimizations
would pay and by how much — and you can justify why the exact ones come before the lossy ones.`,
    questions: [
      {
        q: 'Explain PagedAttention to someone who knows operating systems, in three sentences.',
        a: `It is demand paging applied to the KV cache: divide cache memory into fixed-size
blocks (typically 16 tokens), give each sequence a block table mapping logical positions to
physical blocks, and allocate blocks on demand from a shared pool as the sequence grows. This
eliminates the need for contiguous per-sequence allocation, so you no longer reserve
\`max_seq_len\` for a sequence that may generate 200 tokens — waste drops from 60–80% to under 4%.
As with OS paging, copy-on-write falls out for free: sequences sharing a prefix point at the same
physical blocks and only copy when one of them writes.`,
      },
      {
        q: 'Why does weight quantization help much less at batch 64 with 8k context than at batch 1?',
        a: `Because you have crossed into the cache-dominated regime. Bytes per step are
\`weight_bytes + batch × seq_len × kv_per_token\`. For Llama-3-8B at batch 1 with short context,
weights are essentially 100% of the traffic, so 4-bit weights approach a 4× speedup. At batch 64
with 8k context, the KV cache is \`64 × 8192 × 128 KiB = 67 GB\` against 16 GB of weights — the
cache is over 4× the weights. Eliminating weight bytes entirely would cut step time by under 20%.
In that regime you want KV quantization, not weight quantization. Which optimization is correct
is a property of your operating point, not of the technique.`,
      },
      {
        q: 'Why is quantizing keys per-channel but values per-token, rather than treating them alike?',
        a: `Because they have different distributional structure. Key vectors show strong
per-channel outliers — certain channels consistently carry much larger magnitudes across all
tokens — so a per-tensor or per-token scale is dominated by those outliers and crushes every other
channel into a handful of quantization levels. Giving each channel its own scale isolates the
outliers. Value vectors do not show that pattern; their magnitudes are comparatively uniform
across channels, so per-token scaling is both sufficient and simpler, and it has the practical
advantage of not needing to be revised as the cache grows. Applying the same scheme to both is a
common shortcut and it costs more quality than the bit count suggests.`,
      },
      {
        q: 'Your system uses naive allocation, GQA-8, fp16 everything. You have time for one change. What and why?',
        a: `PagedAttention. In the worked example it takes concurrent sequences from 14 to 59 —
about 4× — which is larger than any other single change on the table, and it costs **nothing** in
quality because it is purely an allocation change. Every alternative (fp8 KV, weight
quantization, eviction) trades accuracy for memory, and it would be perverse to spend accuracy
while still discarding 75% of your cache to fragmentation. The general rule: exhaust the exact
optimizations before reaching for the approximate ones. Prefix caching is the natural second, also
exact, with a payoff that depends on how much your traffic actually shares.`,
      },
      {
        q: 'Why is perplexity a poor way to evaluate a quantized model, and what would you measure instead?',
        a: `Perplexity averages negative log-likelihood over a corpus, so it is dominated by
common tokens in ordinary contexts — exactly the cases a quantized model reproduces almost
perfectly. Damage concentrates in the tail: rare tokens, long dependencies, precise arithmetic,
multi-step reasoning where errors compound. It is also typically measured at 2k–4k sequence
lengths, which is precisely the regime where KV quantization looks harmless. Better: long-context
retrieval suites like RULER, multi-step reasoning tasks, KL divergence against the fp16 baseline's
logits, and a few hundred prompts from your actual workload graded against the baseline. And
evaluate at your real operating point — context length and batch size both matter.`,
      },
      {
        q: 'What does MLA do that GQA cannot, and what is the catch?',
        a: `GQA reduces the *number* of KV heads, so it is bounded — at one KV head you are at MQA
and cannot go further without abandoning per-head structure. MLA instead compresses K and V
jointly into a single low-rank latent vector per token, caching 576 values per layer instead of
2,048 for GQA-8, and reconstructing per-head K and V by up-projection at inference. The
up-projections can be algebraically absorbed into the adjacent query and output matrices, so the
full K and V are never materialized. DeepSeek report ~93% KV reduction versus MHA with *better*
quality, plausibly because the bottleneck regularizes. The catch is twofold: it must be trained
in from the start, with no conversion path for existing checkpoints; and rotation does not commute
with the low-rank projection, so it needs a separate decoupled RoPE component, which is where most
of its implementation complexity lives.`,
      },
    ],
  },

  pitfalls: [
    {
      wrong: 'Reaching for quantization before fixing fragmentation.',
      right: `A naive allocator throws away 60–80% of your KV memory, which is a larger factor
than fp8 quantization buys and costs no quality. Paging is exact; quantization is not. Spending
accuracy to work around an allocation bug is the wrong trade, and it is common.`,
    },
    {
      wrong: 'Assuming 4-bit weights give a 4x speedup.',
      right: `Two reasons it does not. You still move activations and KV cache, and
dequantization costs arithmetic, so even in the weight-dominated regime you typically see
2.5–3.5×. And in the cache-dominated regime — large batch, long context — weights are a minority
of your bytes, so the speedup can be under 20%. Compute which regime you are in first.`,
    },
    {
      wrong: 'Treating KV eviction methods as safe defaults.',
      right: `H2O, SnapKV and sliding-window methods discard tokens based on what has mattered so
far, and you cannot know which token a future query will need. They perform well on benchmarks
resembling their design assumptions and can fail sharply outside them — particularly on retrieval
over the discarded region. They are workload-specific tools, and they need workload-specific
evaluation.`,
    },
    {
      wrong: 'Enabling prefix caching and assuming positions take care of themselves.',
      right: `Cached keys carry RoPE rotations applied at their original absolute positions.
Reusing a prefix at a different offset without handling this breaks the relative-distance property
that makes attention work, and it fails silently — no crash, no error, just quietly degraded
output. It is the most common prefix-caching bug and the hardest to notice in production.`,
    },
  ],

  glossary: [
    { term: 'PagedAttention', def: 'Fixed-size block allocation for the KV cache with per-sequence block tables, borrowed from OS virtual memory. Cuts fragmentation waste from 60-80% to under 4%.' },
    { term: 'block table', def: 'Per-sequence map from logical token positions to physical KV cache blocks. What allows non-contiguous allocation.' },
    { term: 'copy-on-write', def: 'Sequences sharing a prefix point at the same physical blocks; a block is duplicated only when one of them writes to it.' },
    { term: 'RadixAttention', def: 'A radix tree over cached token prefixes enabling automatic longest-prefix reuse across arbitrary request patterns.' },
    { term: 'MLA', def: 'Multi-head latent attention. Compresses K and V into a shared low-rank latent per token; the up-projections are absorbed into neighbouring matrices.' },
    { term: 'GPTQ', def: 'Post-training weight quantization using approximate second-order information, layer by layer. Works to 3-4 bits.' },
    { term: 'AWQ', def: 'Activation-aware weight quantization: scale up the ~1% of channels identified as salient from activation magnitude, then quantize.' },
    { term: 'SmoothQuant', def: 'Migrates activation outliers into the weights via per-channel scaling, enabling int8 for both operands.' },
    { term: 'attention sink', def: 'The first few tokens of a sequence, which receive large attention weight regardless of content. Keeping them enables indefinite streaming.' },
    { term: 'per-channel quantization', def: 'A separate scale for each channel of a tensor. Necessary for keys, which have strong per-channel outliers.' },
    { term: 'RULER', def: 'A long-context benchmark testing multi-needle retrieval, aggregation and tracing. Far more informative than perplexity for evaluating KV compression.' },
  ],
};
