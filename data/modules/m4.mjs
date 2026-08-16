export default {
  n: 4,
  slug: 'roofline',
  title: 'METRICS AND THE ROOFLINE',
  tagline: 'The analytical core. Everything after this module is applied roofline reasoning.',
  hours: '8–10 hours',
  prereqs: ['Module 1', 'Module 2'],

  bigIdea: `Up to now you have been told that decode is memory-bound. This module gives you the
tool to *derive* that, and to derive the answer for any model, any GPU, any batch size, any
context length — on paper, in about ninety seconds.

The tool is the **roofline model**, and it rests on one observation: a computation has two costs,
arithmetic and data movement, and they happen concurrently. Whichever takes longer is your time.
That gives you a single number to compute — **arithmetic intensity**, FLOPs performed per byte
moved — and a single number to compare it against — the machine's **ridge point**, its peak FLOP/s
divided by its peak bytes/s.

\`\`\`
intensity < ridge point   ->  memory-bound. Optimize bytes.
intensity > ridge point   ->  compute-bound. Optimize FLOPs.
\`\`\`

That is the whole model. Its power is that it tells you which optimizations *cannot possibly
help* before you spend a week implementing one. If you are memory-bound at intensity 4 on a
machine whose ridge is 295, then a kernel that halves your FLOP count buys exactly nothing, and
you should know that before you write it.

By the end of this module you should be unable to read a performance claim without immediately
asking: at what batch size, and on which side of the ridge?`,

  concepts: [
    {
      name: 'The four metrics, and why one number is never enough',
      keyPoint: 'TTFT, TPOT, end-to-end latency and throughput are separate quantities that trade against each other through batch size.',
      body: `**TTFT — time to first token.** Queue time plus prefill. Scales with prompt length
(roughly linearly, with a quadratic attention term at long context) and with how much other work
is ahead of you. This is what determines whether an interface feels responsive.

**TPOT / ITL — time per output token.** The steady-state gap between tokens once streaming has
started. Set by decode, so it is set by bandwidth. Roughly independent of context length, and
degrades gently with batch size. Human reading is around 5–8 tokens/second, so anything under
~150 ms/token feels adequate and anything under ~30 ms feels instant.

**End-to-end latency.** \`TTFT + TPOT × output_tokens\`. What actually matters to someone waiting
for a complete answer, and dominated by the second term for any substantial output.

**Throughput.** Total output tokens per second across all concurrent requests. This is what sets
your cost per million tokens, and it is the only one of the four that your finance team cares
about.

The reason a single number is meaningless is that these move in opposite directions:

\`\`\`
batch size  1 ->  256

throughput   ▲▲▲▲▲▲▲▲▲  up ~100x or more
TPOT         ▼          slightly worse
TTFT         ▼▼▼▼       much worse (queueing + prefill contention)
\`\`\`

A system advertising "20,000 tokens/second" at batch 256 may be delivering 10 tokens/second to
each user. Both numbers are honest; only one of them is relevant to any given question.

Two further points that separate real measurement from marketing:

**Percentiles, not means.** Latency distributions in serving are heavily right-skewed — a mean
TPOT of 30 ms is compatible with a p99 of 400 ms if a few requests get preempted or stuck behind
long prefills. Report p50, p95, p99. A mean alone hides exactly the behaviour users complain
about.

**Goodput, not throughput.** If you have an SLO — say, TTFT under 500 ms and TPOT under 50 ms —
then tokens delivered outside that SLO are worth nothing. **Goodput** counts only requests that
met their SLO. A system can have excellent throughput and terrible goodput by running at a batch
size that violates latency targets for everyone. Optimizing throughput without an SLO constraint
is optimizing the wrong thing, and it is a common way for benchmarks to mislead.`,
      ascii: '',
    },
    {
      name: 'Arithmetic intensity, and how to compute it',
      keyPoint: 'Intensity is FLOPs divided by bytes moved from DRAM — and for a decode-time matmul it comes out exactly equal to the batch size.',
      body: `Arithmetic intensity is the ratio you can compute from the operation alone, with no
knowledge of the hardware:

\`\`\`
I = FLOPs performed / bytes moved between DRAM and the chip
\`\`\`

"Moved from DRAM" is the important qualifier. Data already resident in cache or registers is
free by this accounting; the roofline is about traffic to *main* memory.

Work the fundamental case. A matmul \`Y = W @ X\` with \`W\` of shape \`[n, k]\` and \`X\` of shape
\`[k, B]\`:

\`\`\`
FLOPs  = 2 * n * k * B          (one multiply and one add per element pair)
bytes  = 2 * n * k              (read W once, fp16; X and Y are small by comparison)

I = 2nkB / 2nk = B
\`\`\`

**The arithmetic intensity of a weight-stationary matmul equals the batch size.** That is one of
the cleanest and most useful results in the field, and worth stopping on. It says the *only* way
to raise the intensity of a linear layer is to put more sequences through it. Not a better kernel,
not a faster GPU — more concurrent work per weight load.

Applying it to the transformer:

| operation | intensity | why |
|---|---|---|
| decode linear layers, batch B | **B** | each weight read once, used by B sequences |
| prefill linear layers, seq S | **≈ B × S** | each weight read once, used by every position |
| RMSNorm, residual add, RoPE | **≈ 1** | read, do a couple of FLOPs, write |
| attention during decode | **≈ 1** per KV byte | each cached K/V element used once |
| attention during prefill | **≈ S / const** | grows with sequence length |

Two consequences that people find surprising:

**Attention during decode is memory-bound no matter what you do to it.** Each cached key element
is read and used in a single multiply-accumulate. The intensity is O(1), independent of batch
size — because every sequence has its *own* cache, so batching does not amortize KV reads the way
it amortizes weight reads. This is why FlashDecoding exists as a separate kernel from
FlashAttention (Module 7), and why KV cache size directly and unavoidably costs you decode time.

**The elementwise operations are pure overhead.** RMSNorm, residual adds and RoPE do almost no
arithmetic per byte. Left unfused, each one is a full round trip to HBM for nothing. In a decode
step where you are already bandwidth-starved, they can account for 10–20% of the time. This is
the entire justification for kernel fusion.`,
      ascii: '',
    },
    {
      name: 'The roofline model and the ridge point',
      keyPoint: 'Attainable performance is min(peak FLOP/s, intensity × bandwidth), and the ridge point is where those two lines cross.',
      body: `Two hard limits bound any computation on a machine:

\`\`\`
compute ceiling:   P_peak                    FLOP/s
memory ceiling:    I * BW                    FLOP/s  (intensity x bandwidth)

attainable = min(P_peak, I * BW)
\`\`\`

Plot attainable performance against intensity on log-log axes and you get a roof: a diagonal line
of slope 1 (the bandwidth limit) meeting a horizontal line (the compute limit). The corner where
they meet is the **ridge point**:

\`\`\`
I_ridge = P_peak / BW
\`\`\`

Below it you are on the slanted roof, memory-bound, and performance rises linearly with intensity.
Above it you are on the flat roof, compute-bound, and more intensity buys nothing.

Here are the machines you are likely to care about. Compute figures are **dense** BF16 (not the
sparsity-doubled numbers vendors like to headline):

| GPU | memory | bandwidth | BF16 dense | ridge point |
|---|---|---|---|---|
| A100 80GB SXM | 80 GB HBM2e | 2.04 TB/s | 312 TFLOP/s | **153** |
| H100 SXM | 80 GB HBM3 | 3.35 TB/s | 989 TFLOP/s | **295** |
| H200 SXM | 141 GB HBM3e | 4.8 TB/s | 989 TFLOP/s | **206** |
| B200 | 192 GB HBM3e | ~8 TB/s | ~2.25 PFLOP/s | **~281** |
| L40S | 48 GB GDDR6 | 0.86 TB/s | 362 TFLOP/s | **419** |
| RTX 4090 | 24 GB GDDR6X | 1.01 TB/s | 165 TFLOP/s | **164** |
| MI300X | 192 GB HBM3 | 5.3 TB/s | ~1.31 PFLOP/s | **~247** |

*(Treat these as good working numbers rather than gospel — vendors quote compute under varying
assumptions, and you should confirm against the current datasheet for anything load-bearing.)*

Read the table for structure, not for individual entries.

**The H200 has a lower ridge point than the H100.** Same silicon compute, 43% more bandwidth. A
lower ridge is *better for decode*: you need less intensity — a smaller batch — to saturate the
machine. The H200 is a straightforwardly better decode part than the H100 while being an
identical prefill part. This is what it looks like when a vendor targets the memory wall directly.

**The L40S has a ridge point of 419.** GDDR6 instead of HBM, and a lot of compute. You would need
batch 419 to saturate it, which the 48 GB of memory will not let you reach for any large model.
It is a fine prefill and fine-tuning card and a poor decode card, and the ridge point tells you
that in one number.

**Every ridge point is in the 150–420 range.** Modern accelerators have added FLOPs far faster
than bandwidth for a decade, so the ridge keeps drifting right and decode keeps getting relatively
worse. This trend is why these chapters exist.`,
      ascii: `  attainable
  FLOP/s
    |                       ┌──────────────────  P_peak (989 TF)
    |                      /
    |                    /
    |                  /  slope = bandwidth
    |                /
    |              /
    |            /
    |          /
    +--------+-------------+-------------------> arithmetic intensity
             ^             ^
       decode, B=1     ridge point
       I ~= 1          I = 295

  You are here ^                          ...you want to be here ^`,
    },
    {
      name: 'Deriving TPOT from first principles',
      keyPoint: 'Minimum decode latency is bytes moved divided by bandwidth, and you can compute it before you have access to the machine.',
      body: `Since decode is memory-bound, its floor follows directly from bandwidth. The recipe:

\`\`\`
1.  bytes_per_step = weight_bytes + batch * seq_len * kv_bytes_per_token
2.  TPOT_min       = bytes_per_step / bandwidth
3.  throughput_max = batch / TPOT_min
\`\`\`

**Worked: Llama-3-8B, fp16, batch 1, 2k context, on an H100.**

\`\`\`
weights:      8.03e9 x 2       = 16.06 GB   (15.0 GB excluding the embedding lookup)
KV:           131072 x 2048    =  0.27 GB
                                  --------
                                   15.27 GB

TPOT_min = 15.27 / 3350 GB/s   = 4.56 ms
max rate = 1 / 0.00456         = 219 tokens/second
\`\`\`

That is the ceiling. No implementation, however good, beats 219 tok/s for this model on this GPU
at batch 1. If a vendor claims 400, they have changed something — quantization, speculative
decoding, or the definition of a token.

**Now the same model at batch 64:**

\`\`\`
weights:  15.0 GB      (unchanged -- shared)
KV:       131072 x 2048 x 64 = 17.2 GB
                               -------
                               32.2 GB

TPOT_min   = 32.2 / 3350        = 9.6 ms
throughput = 64 / 0.0096        = 6,667 tokens/second
\`\`\`

Per-user latency worsened by 2.1× while aggregate throughput improved by 30×. And notice *why*
TPOT got worse: not because of the weights, which are shared, but because the KV cache now
outweighs them. This is the two-regime crossover from Module 2 showing up directly in a latency
number.

**Now account for the gap to reality.** Measured TPOT is typically 1.5–3× the roofline floor. The
causes, roughly in order:

1. **You do not get peak bandwidth.** Real kernels achieve 70–85% of the spec figure. That alone
   is a 1.2–1.4× gap, and it is unavoidable.
2. **Kernel launch overhead.** Hundreds of tiny kernels per decode step at ~5 µs each is ~1 ms of
   CPU-side cost against a 4.5 ms budget. CUDA Graphs largely fix this; without them it is
   brutal.
3. **Unfused elementwise ops.** Every unfused RMSNorm or residual add is an extra HBM round trip.
4. **Attention kernel inefficiency.** Decode attention has query length 1, which is a genuinely
   awkward shape to parallelize.
5. **Sampling and detokenization.** Sorting 128k logits per sequence is not free at large batch.
6. **Python and scheduler overhead.** Real in naive implementations, largely engineered away in
   vLLM/TRT-LLM.
7. **Communication**, if you are running tensor-parallel (Module 9).

The discipline worth building: compute the roofline number first, measure second, and then
*account for the gap*. If you are within 1.5× your implementation is good. At 5× something
specific is wrong and the list above tells you where to look.`,
      ascii: '',
    },
    {
      name: 'Where the batch dial actually takes you',
      keyPoint: 'Throughput rises with batch size until either you hit the ridge point or you run out of KV cache — and for large models it is almost always the second.',
      body: `Module 2 said the KV cache caps batch size. This module lets you say what that costs
in throughput terms.

**Llama-3-8B, fp16, on one H100 80GB, 4k context:**

\`\`\`
weights 16.06 GB, workspace ~4 GB  ->  ~60 GB for KV
KV per sequence: 131072 x 4096 = 0.54 GB
max batch = 60 / 0.54 = 111
\`\`\`

| batch | bytes/step | TPOT | throughput | intensity | % of ridge |
|---|---|---|---|---|---|
| 1 | 15.5 GB | 4.6 ms | 217 tok/s | 1 | 0.3% |
| 8 | 19.3 GB | 5.8 ms | 1,390 tok/s | 8 | 2.7% |
| 32 | 32.2 GB | 9.6 ms | 3,330 tok/s | 32 | 11% |
| 64 | 49.4 GB | 14.7 ms | 4,340 tok/s | 64 | 22% |
| 111 | 74.9 GB | 22.4 ms | 4,960 tok/s | 111 | 38% |

Three things to take from this table.

**Throughput saturates well before the ridge point.** Going from batch 64 to 111 — nearly double
the sequences — buys only 14% more throughput. The reason is that KV traffic now dominates, and
unlike weight traffic, KV traffic *scales with batch*. Batching amortizes weights; it does not
amortize the cache. Once you are cache-dominated, additional batch adds proportional bytes and
buys proportionally little.

**You never reach the ridge.** At maximum batch you are at 38% of the intensity you would need to
saturate the H100's arithmetic. The machine is memory-bound at every operating point available to
you. This is the normal condition of LLM serving, not an edge case.

**Latency degrades 5× across the range.** TPOT goes from 4.6 ms to 22.4 ms. That is the price of
the 23× throughput gain, and whether it is acceptable is an SLO question, not a technical one.

Now the important corollary. Since KV traffic is what stops the table from continuing, **halving
the KV cache moves the whole table**. With fp8 KV you get to batch ~200 and past 6,000 tok/s. This
is the concrete mechanism by which Module 6's memory work turns into throughput, and it is why
"we reduced memory" and "we increased throughput" are frequently the same sentence.`,
      ascii: '',
    },
    {
      name: 'Reading a GPU spec sheet without being misled',
      keyPoint: 'For decode, read the bandwidth line first; the headline FLOP number is usually inflated by sparsity and rarely the binding constraint.',
      body: `Vendor marketing is optimized for a benchmark you are not running. A short field
guide.

**The FLOP number is probably doubled by sparsity.** NVIDIA quotes structured 2:4 sparsity
figures prominently — an H100 is listed at 1,979 TFLOP/s BF16 "with sparsity" and 989 dense.
Almost no production LLM uses 2:4 sparsity. **Always use the dense number.** If a spec sheet does
not say which it is, assume sparse and halve it.

**FP8 and FP4 numbers are real but conditional.** FP8 on Hopper genuinely doubles throughput over
BF16, and it is usable — fp8 inference is in production. FP4 on Blackwell likewise. But these
apply only to operations you have actually quantized, and they do nothing for a memory-bound
decode step except insofar as they shrink the bytes.

**Bandwidth is the number that matters for decode, and it is honest.** HBM bandwidth figures are
close to achievable — you will see 70–85% of spec in a good kernel. Unlike FLOPs, there is no
sparsity asterisk.

**Memory capacity determines what you can run at all.** 80 GB versus 141 GB is the difference
between a 70B model fitting on two GPUs or one, which changes the communication picture entirely
(Module 9).

**Interconnect matters the moment you need more than one GPU.** NVLink between H100s is around
900 GB/s per GPU; PCIe Gen5 x16 is about 64 GB/s bidirectional. That is a 14× difference, and
tensor parallelism does an all-reduce twice per layer. TP over PCIe is usually a mistake.

A quick decision procedure for a serving workload:

\`\`\`
1. Will the weights fit?           -> memory capacity
2. How fast can I decode?          -> bandwidth / weight_bytes
3. How many sequences fit?         -> (capacity - weights) / kv_per_seq
4. Will I ever be compute-bound?   -> compare max batch to the ridge point
5. If multi-GPU, what links?       -> NVLink vs PCIe
\`\`\`

Note that step 2 uses only bandwidth and step 4 usually answers "no". For decode-dominated
serving, **peak FLOPs is close to irrelevant** — which is a strange thing to say about a GPU
purchase, and precisely the point of this module.`,
      ascii: '',
    },
  ],

  mathLab: {
    prompt: `The signature calculation of these chapters. Do it on paper.

**Part 1 — derive the floor.**
A 7B model, fp16, batch 1, short context (ignore the KV cache). Compute the theoretical minimum
TPOT and maximum tokens/second on:

\`\`\`
a) A100 80GB SXM   2.04 TB/s
b) H100 SXM        3.35 TB/s
c) H200 SXM        4.80 TB/s
d) RTX 4090        1.01 TB/s
\`\`\`

**Part 2 — the gap.** You measure 61 tok/s on the H100 with a naive Hugging Face loop. Your
answer to 1(b) is much higher. Account for the gap: list the causes and give each a plausible
multiplier that multiplies out to what you observed.

**Part 3 — where is the ridge?** Compute the ridge point for each GPU above using dense BF16
figures: A100 312 TFLOP/s, H100 989, H200 989, 4090 165. What batch size would you need on each
to become compute-bound during decode?

**Part 4 — can you get there?** For the H100, with the 7B model at fp16, GQA-8 (32 layers, 8 kv
heads, head_dim 128) and 4k context: how many sequences fit in 80 GB? Is that above or below the
batch you computed in Part 3?

**Part 5 — the design question.** You must serve this model with a p95 TPOT budget of 40 ms. What
is the largest batch size you can run on an H100, and what throughput does it give?`,

    solution: `**Part 1 — the floor**

Weight bytes: \`7e9 × 2 = 14 GB\`.

\`\`\`
a) A100:  14 / 2040  = 6.86 ms  ->  146 tok/s
b) H100:  14 / 3350  = 4.18 ms  ->  239 tok/s
c) H200:  14 / 4800  = 2.92 ms  ->  343 tok/s
d) 4090:  14 / 1010  = 13.86 ms ->   72 tok/s
\`\`\`

Note the H100→H200 improvement is 43%, exactly the bandwidth ratio, with identical compute. And
the 4090 — a card with 165 TFLOP/s of perfectly good arithmetic — is 3.3× slower than the H200
purely on bandwidth. For decode, the compute spec is decoration.

**Part 2 — accounting for the gap**

Observed 61 tok/s against a floor of 239 tok/s is a **3.9× gap**. A plausible decomposition:

\`\`\`
achieved bandwidth ~80% of peak                      1.25x
no CUDA graphs: ~250 kernel launches x 5 us = 1.25ms
   against a 4.18 ms step                            1.30x
unfused RMSNorm / residual / RoPE round trips        1.15x
decode attention kernel inefficiency (q_len = 1)     1.10x
Python loop, sampling, detokenization per step       1.20x
HF generate() overhead: cache reallocation, logits
   processors, tensor churn                          1.35x
                                                     ------
1.25 x 1.30 x 1.15 x 1.10 x 1.20 x 1.35            = 3.33x
\`\`\`

\`239 / 3.33 = 72 tok/s\`, close to the observed 61 — the residual is unmodelled overhead. The
useful conclusion: **almost none of this gap is arithmetic.** It is bandwidth efficiency and CPU
overhead. Switching to vLLM or TensorRT-LLM typically recovers most of it and lands you in the
150–200 tok/s range, i.e. within 1.2–1.6× of the floor.

**Part 3 — ridge points**

\`\`\`
A100:  312e12 / 2.04e12  = 153 FLOP/byte
H100:  989e12 / 3.35e12  = 295
H200:  989e12 / 4.80e12  = 206
4090:  165e12 / 1.01e12  = 163
\`\`\`

Since decode intensity equals batch size, those numbers *are* the required batch sizes: 153, 295,
206 and 163 respectively.

The H200 needs a **smaller** batch to saturate than the H100 (206 vs 295) despite being the faster
card. More bandwidth relative to compute means an easier machine to keep busy. That is the shape
of a decode-optimized part.

**Part 4 — can you reach it?**

\`\`\`
kv per token = 2 x 32 x 8 x 128 x 2 = 131,072 bytes = 128 KiB
kv per 4k sequence = 131072 x 4096 = 0.537 GB

available = 80 GB - 14 GB weights - 4 GB workspace = 62 GB
max batch = 62 / 0.537 = 115 sequences
\`\`\`

**115 against a required 295.** You reach 39% of the ridge point. Even with memory perfectly
packed, a 7B model on a single H100 at 4k context is memory-bound at every achievable operating
point. There is no batch size that makes this workload compute-bound.

That is worth stating plainly: **you cannot saturate an H100's arithmetic with a 7B model at 4k
context on one GPU.** To get there you would need to shrink the KV cache — fp8 KV doubles you to
230, still short; fp8 KV plus fp8 weights gets you past it. Which is exactly the argument for
Module 6.

**Part 5 — the SLO-constrained design**

Solve for the batch where TPOT hits 40 ms.

\`\`\`
bytes_at_40ms = 0.040 s x 3350 GB/s = 134 GB
\`\`\`

That exceeds the 80 GB of physical memory, so at the roofline floor the memory capacity binds
before the latency target does — the whole feasible range is inside the SLO. Recompute using the
realistic 1.5× implementation factor:

\`\`\`
effective bandwidth ~ 3350 / 1.5 = 2233 GB/s
bytes_at_40ms = 0.040 x 2233 = 89.3 GB
89.3 = 14 (weights) + batch x 0.537
batch = (89.3 - 14) / 0.537 = 140
\`\`\`

Still above the 115 that memory allows. **So memory capacity is the binding constraint, not the
latency SLO**, and the answer is batch 115:

\`\`\`
bytes/step = 14 + 115 x 0.537 = 75.8 GB
TPOT       = 75.8 / 2233 = 34 ms      (inside the 40 ms budget)
throughput = 115 / 0.034 = 3,380 tok/s
\`\`\`

The design conclusion — and this is the habit worth forming — is that **you are memory-capacity
bound, not latency bound**. Every gram of effort should go into fitting more sequences in memory,
because you have 6 ms of latency headroom you cannot spend. Had the numbers come out the other
way, you would be tuning the scheduler instead. The roofline told you which problem you have
before you wrote a line of code.`,
  },

  codeLab: {
    goal: `Build a roofline calculator that takes a model config, a GPU, a batch size and a
context length, and reports where you sit, what your bottleneck is, and what your ceiling is.
Then measure a real model and account for the gap.

This is the tool you will reach for in every remaining module. Write it properly.`,
    code: `"""
A roofline calculator for LLM inference. Keep this one.
    pip install torch transformers    (only needed for the measurement section)
"""
from dataclasses import dataclass

GB = 10 ** 9
TF = 10 ** 12


@dataclass
class GPU:
    name: str
    memory_gb: float
    bandwidth_gbs: float        # GB/s
    dense_bf16_tflops: float    # DENSE, not the sparsity-doubled marketing number

    @property
    def ridge(self):
        """FLOP/byte at which the machine flips from memory- to compute-bound."""
        return (self.dense_bf16_tflops * TF) / (self.bandwidth_gbs * GB)


@dataclass
class Model:
    name: str
    n_params: float
    n_layers: int
    n_kv_heads: int
    head_dim: int

    def weight_bytes(self, dtype=2):
        return self.n_params * dtype

    def kv_per_token(self, dtype=2):
        return 2 * self.n_layers * self.n_kv_heads * self.head_dim * dtype


A100 = GPU("A100 80GB", 80, 2039, 312)
H100 = GPU("H100 SXM", 80, 3350, 989)
H200 = GPU("H200 SXM", 141, 4800, 989)
B200 = GPU("B200", 192, 8000, 2250)
L40S = GPU("L40S", 48, 864, 362)
RTX4090 = GPU("RTX 4090", 24, 1008, 165)

LLAMA3_8B = Model("Llama-3-8B", 8.03e9, 32, 8, 128)
LLAMA3_70B = Model("Llama-3-70B", 70.6e9, 80, 8, 128)
MISTRAL_7B = Model("Mistral-7B", 7.24e9, 32, 8, 128)


def analyze(model, gpu, batch=1, seq_len=2048, w_dtype=2, kv_dtype=2,
            n_gpus=1, efficiency=1.0):
    """efficiency < 1 models real achieved bandwidth (0.6-0.8 is typical)."""
    w = model.weight_bytes(w_dtype)
    kv = model.kv_per_token(kv_dtype) * seq_len * batch
    total_bytes = w + kv

    bw = gpu.bandwidth_gbs * GB * n_gpus * efficiency
    flops_per_step = 2 * model.n_params * batch

    tpot = total_bytes / bw
    intensity = flops_per_step / total_bytes

    return {
        "weight_gb": w / GB,
        "kv_gb": kv / GB,
        "total_gb": total_bytes / GB,
        "tpot_ms": tpot * 1000,
        "tok_per_s": batch / tpot,
        "intensity": intensity,
        "ridge": gpu.ridge,
        "bound": "COMPUTE" if intensity > gpu.ridge else "MEMORY",
        "pct_of_ridge": 100 * intensity / gpu.ridge,
        "fits": total_bytes < gpu.memory_gb * GB * n_gpus * 0.95,
    }


def max_batch(model, gpu, seq_len, w_dtype=2, kv_dtype=2, n_gpus=1, workspace_gb=4):
    avail = gpu.memory_gb * GB * n_gpus - model.weight_bytes(w_dtype) - workspace_gb * GB
    per_seq = model.kv_per_token(kv_dtype) * seq_len
    return max(0, int(avail // per_seq))


# ============================================================
print("=== Part 1: the floor for a 7B fp16 model at batch 1 ===")
print(f"{'GPU':<12} {'BW GB/s':>9} {'TPOT ms':>9} {'tok/s':>8} {'ridge':>7}")
for g in (A100, H100, H200, RTX4090):
    r = analyze(MISTRAL_7B, g, batch=1, seq_len=1)
    print(f"{g.name:<12} {g.bandwidth_gbs:>9.0f} {r['tpot_ms']:>9.2f} "
          f"{r['tok_per_s']:>8.0f} {g.ridge:>7.0f}")

print("\\n=== Part 3/4: can a 7B saturate an H100 at 4k context? ===")
mb = max_batch(MISTRAL_7B, H100, 4096)
print(f"  ridge point needs batch  {H100.ridge:.0f}")
print(f"  memory allows batch      {mb}")
print(f"  -> {'compute-bound reachable' if mb > H100.ridge else 'MEMORY-BOUND at every batch'}")

print("\\n=== the batch sweep (Llama-3-8B, H100, 4k ctx) ===")
print(f"{'batch':>6} {'bytes GB':>10} {'TPOT ms':>9} {'tok/s':>9} "
      f"{'intensity':>10} {'% ridge':>8} {'bound':>8}")
for b in (1, 8, 16, 32, 64, 111):
    r = analyze(LLAMA3_8B, H100, batch=b, seq_len=4096)
    flag = "" if r["fits"] else "  <- OOM"
    print(f"{b:>6} {r['total_gb']:>10.1f} {r['tpot_ms']:>9.1f} {r['tok_per_s']:>9.0f} "
          f"{r['intensity']:>10.0f} {r['pct_of_ridge']:>7.0f}% {r['bound']:>8}{flag}")

print("\\n=== what fp8 KV buys you (Llama-3-8B, H100, 4k ctx) ===")
for label, kvd in [("fp16 KV", 2), ("fp8 KV", 1)]:
    mb = max_batch(LLAMA3_8B, H100, 4096, kv_dtype=kvd)
    r = analyze(LLAMA3_8B, H100, batch=mb, seq_len=4096, kv_dtype=kvd)
    print(f"  {label:<9} max batch {mb:>4}  ->  {r['tok_per_s']:>6.0f} tok/s  "
          f"({r['pct_of_ridge']:.0f}% of ridge)")

print("\\n=== Llama-3-70B: how many GPUs, and how many users? ===")
for n in (2, 4, 8):
    w = LLAMA3_70B.weight_bytes() / GB
    cap = H100.memory_gb * n
    if w > cap * 0.9:
        print(f"  {n}x H100: weights {w:.0f} GB do not fit in {cap:.0f} GB")
        continue
    mb = max_batch(LLAMA3_70B, H100, 8192, n_gpus=n)
    r = analyze(LLAMA3_70B, H100, batch=max(mb, 1), seq_len=8192, n_gpus=n)
    print(f"  {n}x H100: max batch {mb:>3} @ 8k ctx  ->  {r['tok_per_s']:>6.0f} tok/s aggregate")


# ============================================================
# Measure a real model and account for the gap
# ============================================================
def measure():
    import time, torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not torch.cuda.is_available():
        print("\\n(no CUDA -- skipping measurement)")
        return

    name = "gpt2"
    model = AutoModelForCausalLM.from_pretrained(
        name, torch_dtype=torch.float16).cuda().eval()
    tok = AutoTokenizer.from_pretrained(name)
    ids = tok("Roofline analysis says", return_tensors="pt").input_ids.cuda()

    n_params = sum(p.numel() for p in model.parameters())
    props = torch.cuda.get_device_properties(0)
    print(f"\\n=== measuring {name} ({n_params/1e6:.0f}M params) on {props.name} ===")

    with torch.no_grad():
        out = model(ids, use_cache=True)
        past, nxt = out.past_key_values, out.logits[:, -1:].argmax(-1)
        for _ in range(10):                                  # warmup
            out = model(nxt, past_key_values=past, use_cache=True)
            past, nxt = out.past_key_values, out.logits[:, -1:].argmax(-1)
        torch.cuda.synchronize()

        t0 = time.perf_counter()
        N = 100
        for _ in range(N):
            out = model(nxt, past_key_values=past, use_cache=True)
            past, nxt = out.past_key_values, out.logits[:, -1:].argmax(-1)
        torch.cuda.synchronize()
        measured = (time.perf_counter() - t0) / N

    weight_bytes = n_params * 2
    # torch reports bandwidth in kHz * bus bits; convert to GB/s
    peak_bw = props.memory_clock_rate * 1000 * (props.memory_bus_width / 8) * 2 / GB
    floor = weight_bytes / (peak_bw * GB)

    print(f"  peak bandwidth (from device)  {peak_bw:>8.0f} GB/s")
    print(f"  roofline floor                {floor*1000:>8.2f} ms/token")
    print(f"  measured                      {measured*1000:>8.2f} ms/token")
    print(f"  gap                           {measured/floor:>8.2f}x")
    print("  -> small models are launch-overhead dominated; expect a large gap here.")


if __name__ == "__main__":
    try:
        measure()
    except ImportError:
        print("\\n(pip install torch transformers to run the measurement)")

# --- TODO for you ---
#   1. Add a prefill_analyze() that computes TTFT and reports whether prefill is
#      compute- or memory-bound. Find the sequence length at which the O(S^2)
#      attention term overtakes the 2*N*S matmul term.
#   2. Add speculative decoding: given acceptance rate alpha, draft length gamma,
#      and draft cost ratio c, compute effective tok/s. Come back after Module 8.
`,
    expect: `Part 1 reproduces the math lab: 146 / 239 / 343 / 72 tok/s and ridge points of
153 / 295 / 206 / 163.

The saturation check prints \`MEMORY-BOUND at every batch\` — memory allows batch ~115 against a
required 295.

The batch sweep shows throughput climbing from ~217 to ~4,900 tok/s while TPOT degrades from
4.6 ms to 22 ms, and \`% ridge\` never exceeding about 38%. The fp8 KV comparison roughly doubles
the achievable batch and pushes throughput past 6,000 tok/s — that single line is the argument
for Module 6 in numeric form.

The 70B section shows 2×H100 failing to fit 141 GB of weights, with 4× and 8× working.

The measurement section on GPT-2 will show a **large** gap — often 10–30×. That is expected and
instructive: GPT-2 is 124M parameters, so a decode step moves only ~250 MB and takes well under a
millisecond at the roofline, which means fixed per-step overhead (kernel launches, Python)
completely dominates. Small models are overhead-bound, not bandwidth-bound. Run it on a 7B model
if you have the memory and the gap drops to 1.5–3×.`,
    stretch: `Produce the actual roofline plot. Log-log axes, intensity on x and attainable
FLOP/s on y. Draw the two rooflines for an H100 and an H200. Then plot points for: decode at
batch 1, 8, 32, 128; prefill at 512, 2k, 8k tokens; and an RMSNorm. Seeing prefill and decode land
three orders of magnitude apart on the same chart is the single clearest picture of why this field
is shaped the way it is.`,
  },

  papers: [
    {
      title: 'Roofline: An Insightful Visual Performance Model for Multicore Architectures',
      by: 'Williams, Waterman & Patterson, 2009',
      url: 'https://dl.acm.org/doi/10.1145/1498765.1498785',
      why: 'The original roofline paper. Predates deep learning entirely, which is part of the point — this is a general model that happens to explain LLM inference perfectly.',
      frame: `Read **Sections 2 and 3** for the model and the ridge point. The multicore examples
are dated; the framework is not. Notice that they were already arguing in 2009 that memory
bandwidth, not FLOPs, is the binding constraint for most real kernels. Fifteen years later the
ratio has only got worse.`,
    },
    {
      title: 'Transformer Inference Arithmetic',
      by: 'Carol Chen (kipply), 2022',
      url: 'https://kipp.ly/transformer-inference-arithmetic/',
      why: 'The standard practitioner reference for exactly the arithmetic in this module.',
      frame: 'Read it end to end with a calculator. Re-derive every number rather than accepting it. The flops-versus-memory-bandwidth section is the heart. Some hardware figures have aged out; the method has not.',
    },
    {
      title: 'Efficiently Scaling Transformer Inference',
      by: 'Pope et al., 2022',
      url: 'https://arxiv.org/abs/2211.05102',
      why: 'Applies exactly this analysis at scale, and derives partitioning strategies from it. The best worked example of roofline reasoning in the LLM literature.',
      frame: `Now read **Section 2** properly — you have the background for it. Their cost model
separates weight traffic, KV traffic and communication, which is the decomposition you will need
in Module 9. **Figure 3** (the latency/throughput Pareto frontier) is the picture this module has
been building toward.`,
    },
    {
      title: 'DistServe: Disaggregating Prefill and Decoding for Goodput-optimized LLM Serving',
      by: 'Zhong et al., OSDI 2024',
      url: 'https://arxiv.org/abs/2401.09670',
      why: 'The clearest treatment of goodput as the right metric, and the strongest argument that prefill and decode should not share hardware.',
      frame: 'Read **Sections 2 and 3** now for the goodput definition and the interference analysis. The disaggregation mechanism itself is Module 9 material — come back to Sections 4–6 then.',
    },
  ],

  checkpoint: {
    claim: `Given a model, a GPU and a batch size, you can predict roughly where you sit on the
roofline, name the binding constraint, and state which optimizations could possibly help — before
running anything.`,
    questions: [
      {
        q: 'What is the arithmetic intensity of a decode-time linear layer, and why is that result so useful?',
        a: `It equals the batch size, exactly. For \`Y = W @ X\` with \`W\` of shape \`[n, k]\` and \`X\`
of shape \`[k, B]\`, the FLOPs are \`2nkB\` and the bytes are \`2nk\` (reading the fp16 weights once),
so intensity is \`B\`. It is useful because it collapses a complicated question into a
one-parameter comparison: put your batch size next to the machine's ridge point and you know your
regime immediately. Batch 32 on an H100 (ridge 295) is 11% of the way there — decisively
memory-bound, so any optimization that only reduces FLOPs is guaranteed to do nothing.`,
      },
      {
        q: 'The H200 has the same compute as the H100 but 43% more bandwidth. What happens to its ridge point, and is that good or bad?',
        a: `The ridge point *falls*, from 295 to 206 FLOP/byte, because ridge is peak FLOP/s
divided by bandwidth. A lower ridge is better for decode: you need less arithmetic intensity — a
smaller batch — to saturate the machine's compute. In practice you also just decode 43% faster at
any batch, since decode time is bytes over bandwidth. The H200 is a strictly better decode part
and an identical prefill part. The general lesson: for memory-bound work, read the bandwidth line
first, and be suspicious of ranking accelerators by FLOPs.`,
      },
      {
        q: 'You compute a floor of 239 tok/s and measure 61. Where did the factor of 4 go?',
        a: `Not into arithmetic — decode does about 0.016 ms of FLOPs against a 4.2 ms step. The
usual decomposition: achieved bandwidth is 70–85% of spec (~1.25×); several hundred unfused
kernel launches at ~5 µs each add ~1 ms of CPU-side cost to a 4.2 ms budget (~1.3×); unfused
RMSNorm and residual adds add HBM round trips (~1.15×); the decode attention kernel is awkward at
query length 1 (~1.1×); sampling and detokenization per step (~1.2×); and Hugging Face
\`generate()\` overhead like cache reallocation and logits processors (~1.35×). Those multiply to
roughly 3.3×. The fix is a real serving engine: CUDA graphs, fused kernels, and a C++ scheduler
typically recover most of it.`,
      },
      {
        q: 'Why does throughput stop improving with batch size well before the ridge point?',
        a: `Because batching amortizes weight traffic but not KV traffic. Weights are shared, so
their bytes are constant as batch grows. Each sequence brings its own KV cache, so KV bytes grow
linearly with batch. Once the cache dominates the byte count, adding a sequence adds
proportionally to both the numerator and denominator of throughput, and the curve flattens. For
Llama-3-8B on an H100 at 4k context, going from batch 64 to 111 adds 73% more sequences for 14%
more throughput. The way to move the whole curve is to shrink the per-sequence cache — which is
why memory optimization and throughput optimization are the same project.`,
      },
      {
        q: 'When would peak FLOPs actually matter for an LLM serving deployment?',
        a: `Three situations. First, prefill-heavy workloads — long prompts with short outputs,
like classification, RAG scoring, or embedding — where you are on the compute-bound side and
FLOPs directly set TTFT. Second, speculative decoding, which deliberately converts spare compute
into throughput by verifying several draft tokens per pass; the more headroom you have below the
ridge, the more speculation pays. Third, very large batch on a model whose KV cache has been
aggressively shrunk, where you finally cross the ridge. For ordinary decode-dominated chat
serving, peak FLOPs is close to irrelevant, which is the counterintuitive punchline of this
module.`,
      },
      {
        q: 'Why is goodput a better target than throughput, and how can they diverge?',
        a: `Goodput counts only requests that met their SLO; throughput counts every token
delivered. They diverge whenever you push batch size past the point where latency targets hold.
A system running batch 256 might report 20,000 tok/s while every request has a TPOT of 80 ms
against a 50 ms budget — throughput excellent, goodput zero. Because throughput rises
monotonically with batch and latency degrades with it, optimizing throughput alone reliably lands
you on a configuration that violates your SLO for everyone. Stating the SLO first and maximizing
goodput under it is the honest formulation, and it is the framing DistServe argues for.`,
      },
    ],
  },

  pitfalls: [
    {
      wrong: 'Using the headline TFLOPs from the spec sheet.',
      right: `That number is usually the 2:4-structured-sparsity figure, which is double the dense
number and does not apply to any production LLM. An H100 is 989 dense BF16 TFLOP/s, not 1,979.
Using the sparse figure inflates your ridge point by 2× and will tell you that you are
memory-bound when you are not — or, more often, exaggerate how far from the ridge you are.`,
    },
    {
      wrong: 'Optimizing FLOPs when you are memory-bound.',
      right: `At batch 32 on an H100 you are at 11% of the ridge point. A kernel that halves your
arithmetic changes your runtime by approximately zero, because the arithmetic was already hidden
behind memory traffic. Compute the intensity first; it tells you which entire *categories* of
optimization are dead before you invest in one.`,
    },
    {
      wrong: 'Treating the roofline floor as an achievable target.',
      right: `The floor assumes 100% of peak bandwidth, zero launch overhead, perfect fusion and
no sampling cost. Real systems land at 1.5–3× the floor, and a well-engineered one at 1.2–1.6×.
Use the floor to know whether your gap is 1.5× (fine, stop optimizing) or 8× (something specific
is broken), not as a number to chase.`,
    },
    {
      wrong: 'Quoting throughput without batch size, context length, and percentiles.',
      right: `Those three numbers determine the result more than the implementation does. "15,000
tok/s" at batch 256 with 512-token contexts and "1,200 tok/s" at batch 8 with 32k contexts can
come from the same system on the same GPU. Any benchmark missing them cannot be compared against
anything, and any benchmark reporting a mean rather than p95/p99 is hiding its worst behaviour.`,
    },
  ],

  glossary: [
    { term: 'roofline model', def: 'attainable = min(peak FLOP/s, intensity x bandwidth). A two-line model that identifies your bottleneck.' },
    { term: 'arithmetic intensity', def: 'FLOPs performed per byte moved from main memory. For a decode linear layer it equals the batch size.' },
    { term: 'ridge point', def: 'peak FLOP/s divided by bandwidth. The intensity at which a machine flips from memory- to compute-bound.' },
    { term: 'goodput', def: 'Throughput counting only requests that met their latency SLO. The metric that matches what you are actually selling.' },
    { term: 'TTFT', def: 'Time to first token: queue time plus prefill. Scales with prompt length.' },
    { term: 'TPOT / ITL', def: 'Time per output token. Set by decode, so set by bandwidth.' },
    { term: 'dense vs sparse FLOPs', def: 'Vendors quote 2:4 structured sparsity figures that are double the dense number. Production LLMs use dense.' },
    { term: 'achieved bandwidth', def: 'The fraction of peak HBM bandwidth a real kernel reaches. Typically 70-85%.' },
    { term: 'CUDA Graph', def: 'A captured, replayable launch sequence that removes per-kernel CPU overhead. Essential when a decode step is hundreds of tiny kernels.' },
  ],
};
