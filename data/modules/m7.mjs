export default {
  n: 7,
  slug: 'attention-kernels',
  title: 'ATTENTION KERNELS',
  tagline: 'Attention performance is decided by the memory hierarchy, not the FLOP count.',
  hours: '7–9 hours',
  prereqs: ['Module 0', 'Module 4'],

  bigIdea: `The roofline model treats memory as one thing. Inside the GPU it is a hierarchy, and
attention is where the difference between the tiers becomes decisive.

Naive attention computes \`S = QK^T\`, writes the \`[N, N]\` score matrix to HBM, reads it back to
softmax it, writes it again, reads it once more to multiply by V. For \`N = 8192\` that matrix is
128 MB per head in fp16, and you have moved it four times. The arithmetic is unavoidable; the
traffic is not.

**FlashAttention** is the observation that you can compute exact attention without ever
materializing that matrix, by tiling the computation so each tile fits in the SM's shared memory —
a tier with roughly 6× the bandwidth of HBM — and using an *online* softmax that updates a running
result as tiles stream past. Same arithmetic, same answer to the last bit, a fraction of the
traffic.

The second theme of this module is that **prefill and decode need different kernels**, for the
same reason they sit on opposite sides of the roofline. Prefill has thousands of queries to
parallelize over. Decode has one. A kernel tuned for the first is badly wrong for the second, and
FlashDecoding exists because of it.`,

  concepts: [
    {
      name: 'The memory hierarchy, and where attention actually runs',
      keyPoint: 'Shared memory has roughly 6x the bandwidth of HBM and 16x lower latency, so keeping a computation resident there is worth restructuring the algorithm for.',
      body: `The roofline's "bandwidth" is HBM bandwidth. On an H100 there are several tiers above
it:

| tier | capacity | bandwidth | latency |
|---|---|---|---|
| registers | 256 KB per SM | ~100 TB/s | ~1 cycle |
| shared memory / L1 | 228 KB per SM | ~20 TB/s | ~30 cycles |
| L2 | 50 MB | ~10 TB/s | ~200 cycles |
| HBM3 | 80 GB | 3.35 TB/s | ~500 cycles |

Shared memory is **about 6× the bandwidth of HBM and 16× lower latency**, but there is only
228 KB of it per SM. That constraint — fast but tiny — is what shapes every high-performance
attention kernel.

Now count what naive attention costs in HBM traffic. For one head, sequence length \`N\`, head
dimension \`d\`, in fp16:

\`\`\`
1. read Q, K                       2 * N * d * 2 bytes
2. write S = QK^T                  N^2 * 2 bytes
3. read S                          N^2 * 2 bytes
4. write P = softmax(S)            N^2 * 2 bytes
5. read P, read V                  N^2 * 2 + N * d * 2 bytes
6. write O                         N * d * 2 bytes
                                   ---------------------
                                   ~4 N^2 * 2 bytes of traffic for the score matrix
\`\`\`

At \`N = 8192, d = 128\`: the \`N × d\` terms are about 2 MB each, while each \`N²\` term is
**134 MB**. The score matrix dominates by roughly 60×, and you move it four times.

For a full model — 32 heads, 32 layers — that is \`4 × 134 MB × 32 × 32 = 549 GB\` of HBM traffic
for one 8k-token prefill, just shuttling intermediate scores. At 3.35 TB/s that is 164 ms of pure
memory movement, and none of it is arithmetic.

The FLOPs, meanwhile, are \`4 N² d\` per head — real work that must happen. **The problem is not the
arithmetic. It is that the intermediate never needed to exist in HBM.**

This is a general lesson worth extracting: the roofline says "minimize bytes from DRAM", and one
of the most powerful ways to do that is to restructure an algorithm so intermediates stay in a
faster tier. That is what kernel fusion is, and attention is its most valuable application.`,
      ascii: `  NAIVE                              FLASH
  HBM                                HBM
   |  Q,K  ->                         |  Q,K,V (tiles) ->
   |          [S = QK^T]              |
   |  <- S (134 MB)                   |    SRAM: [tile QK^T]
   |  S ->                            |          [online softmax]
   |          [softmax]               |          [accumulate O]
   |  <- P (134 MB)                   |    (never leaves SRAM)
   |  P,V ->                          |
   |          [PV]                    |  <- O only
   |  <- O                            |
                                      |
  4 x N^2 traffic                     O(N) traffic for the scores`,
    },
    {
      name: 'Online softmax: the algebra that makes tiling possible',
      keyPoint: 'A running maximum and a running sum let you rescale a partial softmax when a larger value appears, so the result is exact without ever seeing all the scores at once.',
      body: `The obstacle to tiling attention is softmax. It needs a denominator that sums over
*every* score in the row, so it appears to require the whole row before producing any output.

The resolution is to compute it incrementally and correct as you go. This is worth deriving,
because the exactness of FlashAttention rests entirely on it.

Standard safe softmax over a vector \`x\`:

\`\`\`
m = max(x)                       subtract the max so exp() cannot overflow
l = sum_i exp(x_i - m)
y_i = exp(x_i - m) / l
\`\`\`

Now suppose \`x\` arrives in two blocks, \`x1\` then \`x2\`. Process \`x1\`:

\`\`\`
m1 = max(x1)
l1 = sum exp(x1 - m1)
\`\`\`

Then \`x2\` arrives and may contain a larger value. The new global max is
\`m = max(m1, m2)\`, and the correction is one multiplication:

\`\`\`
l = l1 * exp(m1 - m)  +  l2 * exp(m2 - m)
\`\`\`

The factor \`exp(m1 - m)\` rescales the old partial sum from base \`m1\` to base \`m\`. Since
\`exp(a - m1) × exp(m1 - m) = exp(a - m)\` exactly, **nothing is approximated.** It is the same
number, re-expressed.

The same correction applies to the accumulated output. If you have a partial output \`O1\` computed
against the old normalization, rescale it the same way before adding the new block's
contribution:

\`\`\`
O = O1 * (l1 * exp(m1 - m) / l)  +  (new block's contribution)
\`\`\`

In FlashAttention this is arranged so the division by \`l\` happens once at the end, and the inner
loop just tracks unnormalized accumulators along with \`m\` and \`l\`.

**Why exactness matters so much.** There is a long line of approximate attention methods —
Linformer, Performer, Reformer, sparse patterns — that reduce complexity by changing what is
computed. They all face the same adoption problem: you cannot drop them into a trained model
without retraining, and you cannot be sure what capability you gave up. FlashAttention computes
**bit-comparable** attention (modulo floating-point reassociation) and is therefore a pure drop-in.
You can apply it to any existing checkpoint with no evaluation required.

That is the whole reason it won. Not the speedup — a 2–4× speedup with an asterisk about quality
is a much harder sell than a 2–4× speedup with no asterisk at all.`,
      ascii: `  process K/V in blocks, keeping (m, l, O) running:

  block 1:  scores [2.0, 1.0]     m=2.0   l=1+0.368=1.368       O = partial

  block 2:  scores [5.0, 3.0]     m2=5.0
            new global max        m=5.0
            rescale old:          l1' = 1.368 * exp(2.0-5.0) = 0.0681
            add new:              l  = 0.0681 + (1+0.135) = 1.203
            rescale O the same way, then add block 2's contribution

  exp(a-2.0) * exp(2.0-5.0) == exp(a-5.0)      <- exact, not approximate`,
    },
    {
      name: 'FlashAttention: the tiling algorithm',
      keyPoint: 'Loop over K/V blocks in the outer loop and Q blocks in the inner, keeping a tile of the computation in shared memory and never writing scores to HBM.',
      body: `Put the pieces together. The algorithm splits Q, K and V into blocks sized so a
working set fits in shared memory, then loops:

\`\`\`
for each block of K, V:                     # loaded into SRAM once
    for each block of Q:                    # loaded into SRAM
        S_ij  = Q_i @ K_j^T                 # in SRAM, small
        m_new = max(m_i, rowmax(S_ij))
        P_ij  = exp(S_ij - m_new)           # in SRAM
        l_new = l_i * exp(m_i - m_new) + rowsum(P_ij)
        O_i   = O_i * (l_i * exp(m_i - m_new) / l_new) + (P_ij @ V_j) / l_new
        m_i, l_i = m_new, l_new
    # O_i, m_i, l_i live in registers/SRAM across the inner loop
write O                                     # the only large HBM write
\`\`\`

The \`[N, N]\` score matrix is never assembled. Only \`[block, block]\` tiles exist, in shared memory,
and they are discarded as soon as they are consumed.

The traffic analysis: HBM accesses drop from \`O(N² + Nd)\` to \`O(N²d²/M)\`, where \`M\` is the SRAM
size. With \`d = 128\` and \`M ≈ 100 KB\`, the factor \`d²/M\` is well under 1, so the reduction is
substantial — the paper reports 9× fewer HBM accesses on GPT-2 and 2–4× end-to-end speedups.

Memory goes from **quadratic to linear** in sequence length, because you only ever hold blocks.
This is what made long-context training practical at all.

**The trade is recomputation.** During the backward pass you need the attention probabilities,
which you did not store. FlashAttention recomputes them from Q, K, V on the fly. That is *more*
FLOPs than the naive version — and it is still faster, because the FLOPs were never the
bottleneck. It is the clearest possible demonstration of this course's thesis: trading arithmetic
for bandwidth is a winning trade, even at a large arithmetic premium.

**FlashAttention-2** (2023) reworked the same algorithm around GPU scheduling rather than the
memory hierarchy: reduce non-matmul FLOPs (rescaling operations are much slower per FLOP than
tensor-core matmuls on modern GPUs), swap the loop order so the query block is outer, and
parallelize over sequence length as well as batch and heads. Roughly another 2×, reaching
50–73% of theoretical peak matmul throughput.

**FlashAttention-3** (2024) targets Hopper specifically: warp specialization (separate warps
produce and consume data), asynchronous TMA copies overlapped with computation, interleaving the
matmul and softmax so the non-tensor-core softmax work hides behind tensor-core work, and FP8
support. The trend across all three is instructive — the first was an algorithmic insight, the
next two were about exploiting specific hardware asynchrony. Extracting the last factor of two now
requires knowing the machine intimately.`,
      ascii: '',
    },
    {
      name: 'Why decode needs a different kernel',
      keyPoint: 'At decode the query length is 1, so there is nothing to parallelize over unless you split along the KV dimension instead.',
      body: `FlashAttention parallelizes over batch, heads, and **query blocks**. At decode time
the query length is 1, so the third source of parallelism disappears.

Concretely: batch 8, 32 heads, query length 1, KV length 8192. You have \`8 × 32 = 256\` independent
work units for a GPU with 132 SMs — and each unit must sequentially scan 8,192 keys. Occupancy is
poor and the long serial scan dominates. Meanwhile at batch 1 you have only 32 work units for 132
SMs: **the GPU is more than 75% idle.**

The fix is **FlashDecoding**, and it is a split-K reduction. Partition the KV sequence into
chunks, compute a partial attention result for each chunk in parallel, then combine.

\`\`\`
1. split the 8192 keys into, say, 16 chunks of 512
2. each chunk computes, independently and in parallel:
      - partial output O_c
      - its running max m_c and sum l_c
3. combine the 16 partials using exactly the online-softmax rescaling
   from earlier -- the same algebra, applied across chunks instead of
   within a loop
\`\`\`

Parallelism goes from \`batch × heads\` to \`batch × heads × chunks\` — a 16× increase in this
example, which is the difference between an idle GPU and a busy one. Reported speedups for
long-context decode are substantial, up to several times for large KV lengths.

Note how neatly the online softmax pays off twice. It was introduced to allow sequential tiling
within one kernel; the identical algebra allows *parallel* partial results to be combined
afterwards. That is a sign the abstraction was the right one.

Three further decode-specific realities:

**GQA changes the arithmetic.** With 8 KV heads serving 32 query heads, the 4 query heads sharing
a KV head can read it once from HBM and use it four times. A GQA-aware decode kernel gets 4× the
arithmetic intensity on the attention step. A kernel that naively replicates KV heads to match
query heads throws that away — and this is a real and common performance bug.

**Paged KV changes the memory access pattern.** Under PagedAttention the cache is not contiguous,
so the kernel must gather through a block table. vLLM's paged attention kernels handle this;
a stock FlashAttention kernel cannot read a paged cache directly. The two features have to be
co-designed, which is why they arrived together.

**Attention at decode is memory-bound no matter what.** Each cached key element is read and used
in one multiply-accumulate, so intensity is O(1) — and unlike weights, batching does not amortize
it, because every sequence has its own cache. No kernel fixes this. The only fixes are storing
less (Module 6) or reading it less often.`,
      ascii: `  PREFILL: parallelize over query blocks
    Q [2048, 128]  x  K [2048, 128]
    -> 2048/64 = 32 query blocks x 32 heads x batch = plenty of work

  DECODE, naive: query length 1
    Q [1, 128]  x  K [8192, 128]
    -> 1 x 32 heads x batch 8 = 256 units, each scanning 8192 keys
       132 SMs, poor occupancy, long serial scan

  DECODE, FlashDecoding: split along K
    chunk 0   chunk 1   chunk 2  ...  chunk 15    (parallel)
    (O_0,m,l) (O_1,m,l) (O_2,m,l)     (O_15,m,l)
         \\        |         /            /
          +-------+---------+-----------+
                  |
          online-softmax combine  ->  O
    -> 256 x 16 = 4096 units. GPU busy.`,
    },
    {
      name: 'Kernel fusion and the cost of round trips',
      keyPoint: 'Every unfused elementwise operation is a full HBM round trip for almost no arithmetic, and a decode step is full of them.',
      body: `FlashAttention is the most valuable instance of a general principle: **if two
operations are performed back to back, do not send the intermediate to HBM.**

Consider RMSNorm followed by a linear projection at decode time, batch 1, \`d_model = 4096\`, fp16:

\`\`\`
unfused:
  RMSNorm kernel:  read 8 KB, ~3 FLOPs/element, write 8 KB
  matmul kernel:   read 8 KB + weights, write output
                   -> the 8 KB intermediate makes a full HBM round trip

fused:
  one kernel:      read 8 KB, normalize in registers, matmul, write output
                   -> intermediate never leaves the chip
\`\`\`

8 KB sounds negligible, and per operation it is. The problem is the count. A decode step for a
32-layer model has, per layer: two RMSNorms, RoPE on Q and K, two residual adds, and a SiLU-and-
multiply. That is roughly 7 elementwise operations × 32 layers ≈ **224 kernel launches**, each
with a round trip and each with ~5 µs of launch overhead.

\`\`\`
launch overhead:  224 x 5 us          = 1.12 ms
against a theoretical step of         = 4.5 ms
                                        -> 25% of the budget
\`\`\`

Plus the memory traffic, which is small per operation but adds up, and — often worse — each tiny
kernel fails to saturate the memory system at all, so its effective bandwidth is far below peak.

The standard remedies:

**Fusion.** Combine adjacent operations into one kernel. Fuse the norm into the following matmul,
fuse SiLU and the elementwise multiply in SwiGLU, fuse the residual add into whatever precedes it.
Every serving engine ships hand-written fused kernels for the common patterns, and
\`torch.compile\` generates them automatically for many cases.

**CUDA Graphs.** Capture the whole sequence of launches once and replay it as a single unit. This
does not reduce memory traffic but eliminates nearly all the per-launch CPU cost. For decode —
where the shape of every step is identical — it is close to free and is standard practice. It is
frequently the single largest fix when a naive PyTorch loop is 3× off the roofline.

**Persistent kernels.** One kernel that loops internally over layers, avoiding launches entirely.
Highest performance, least flexibility, hardest to write.

The reason this module comes after Module 4 is that the roofline tells you *whether* fusion will
help. If you are compute-bound, an extra HBM round trip may hide behind arithmetic and fusion buys
little. At decode you are never compute-bound, so every round trip is exposed and every fusion
pays.`,
      ascii: '',
    },
    {
      name: 'Where the efficient-attention research line went',
      keyPoint: 'Sub-quadratic architectures trade exact recall for a constant-size state, and the field settled on hybrids rather than picking a side.',
      body: `FlashAttention makes exact attention fast, but it does not make it sub-quadratic — the
FLOPs are still \`O(N²d)\` and the KV cache still grows linearly. A parallel research line tried to
change the asymptotics.

**Linear attention.** Replace \`softmax(QK^T)V\` with a kernel feature map \`φ\` so that
\`φ(Q)(φ(K)^T V)\` can be computed by associativity in \`O(Nd²)\` instead of \`O(N²d)\`. The state is a
fixed \`[d, d]\` matrix — constant size, no growing cache. The cost is that softmax's sharp,
content-addressable selectivity is replaced by something smoother, and exact retrieval suffers.

**State space models.** Mamba (2023) is the most successful. It maintains a fixed-size recurrent
state updated per token, with input-dependent (selective) state transitions, and a parallel scan
that makes training efficient. **There is no KV cache at all** — decode memory is constant in
sequence length, which is an enormous inference advantage.

The trouble is a genuine capability gap. Attention can look up any previous token exactly; a
fixed-size state cannot store an arbitrary amount of detail. Empirically, pure SSMs
underperform on tasks requiring precise recall from long context — copying, retrieval, in-context
learning from many examples. This is not an implementation shortfall, it is information-theoretic:
a constant-size state cannot losslessly summarize an unbounded history.

**Hybrids won.** The field's answer was to interleave: mostly SSM or linear-attention layers, with
a few full-attention layers to provide exact recall. Jamba, Zamba and Samba are examples, and the
approach is now common in production models. You get most of the memory saving with most of the
recall.

**Sparse attention** is the other direction: keep softmax attention but restrict which positions
attend to which. Sliding-window attention (Mistral) bounds the cache at the window size. Global
plus local patterns give some long-range connectivity cheaply. DeepSeek's Native Sparse Attention
(2025) trains the sparsity pattern rather than fixing it in advance, which addresses the main
objection to earlier sparse methods — that a hand-designed pattern is a guess about what the model
needs.

The honest summary as of now: **full attention with FlashAttention kernels remains the default for
quality-critical work**, hybrids are a real and growing production option where long context
matters more than exact recall, and the sub-quadratic line has not displaced attention so much as
found a complementary role. Anyone claiming attention is obsolete is ahead of the evidence.`,
      ascii: '',
    },
  ],

  mathLab: {
    prompt: `**Part 1 — the cost of materializing scores.**
Llama-3-8B prefill: 32 layers, 32 heads, head_dim 128, fp16. Sequence length \`N = 8192\`, batch 1.

  a) Size in bytes of the \`[N, N]\` score matrix for **one head**.
  b) Naive attention touches that matrix 4 times (write S, read S, write P, read P). Total HBM
     traffic for the score matrix across all heads and all layers.
  c) At 3.35 TB/s, how long is that?
  d) The attention FLOPs are \`4 × N² × d\` per head. Total across the model, and time at
     989 TFLOP/s. Compare to (c) — which dominates?
  e) Repeat (a)–(c) for \`N = 32768\`. What happens to the ratio?

**Part 2 — trace the online softmax.**
Scores arrive in two blocks: \`x1 = [2.0, 1.0]\` then \`x2 = [5.0, 3.0]\`.

  a) Compute \`m1\` and \`l1\` for block 1 alone.
  b) Compute \`m2\` and \`l2\` for block 2 alone.
  c) Combine to get the global \`m\` and \`l\` using the rescaling rule.
  d) Compute the standard softmax over all four values directly and confirm the denominators
     match. State why the agreement is exact and not approximate.

**Part 3 — FlashDecoding parallelism.**
Decode on an H100 (132 SMs). Batch 8, 32 query heads, 8 KV heads, KV length 8192.

  a) Without split-K, how many independent work units does the attention kernel have? What
     fraction of the SMs can be occupied?
  b) With FlashDecoding splitting KV into chunks of 512, how many work units?
  c) Repeat both for batch 1. Which case does FlashDecoding help more, and why?
  d) A GQA-aware kernel reads each KV head once and uses it for 4 query heads. What does that do
     to the arithmetic intensity of the attention step?

**Part 4 — fusion overhead.**
A 32-layer model, decode, batch 1. Per layer: 2 RMSNorms, RoPE on Q and K (count as 2), 2 residual
adds, 1 SiLU-multiply.

  a) Unfused kernel launches per decode step, at 5 µs each. Total overhead.
  b) The roofline floor for this model is 4.5 ms/token. What fraction is launch overhead?
  c) CUDA graphs cut per-launch cost to ~0.5 µs. New overhead and new fraction.
  d) Perfect fusion reduces the count to ~3 launches per layer. Combined with CUDA graphs, what
     is the overhead now?`,

    solution: `**Part 1**

a) \`8192 × 8192 × 2 bytes = 134,217,728 = 134.2 MB\` per head.

b)
\`\`\`
4 touches x 134.2 MB x 32 heads x 32 layers = 549,755,813,888 bytes = 550 GB
\`\`\`

c) \`550 / 3350 = 0.164 s = 164 ms\` of pure HBM traffic for intermediates.

d)
\`\`\`
FLOPs per head = 4 x 8192^2 x 128 = 3.436e10
total = 3.436e10 x 32 heads x 32 layers = 3.518e13 = 35.2 TFLOP
time  = 35.2e12 / 989e12 = 35.6 ms
\`\`\`

**Memory traffic (164 ms) is 4.6× the arithmetic (35.6 ms).** Naive attention is memory-bound
because of an intermediate that did not need to exist. FlashAttention removes essentially all of
the 164 ms.

e) At \`N = 32768\`:
\`\`\`
score matrix = 32768^2 x 2 = 2.147 GB per head
traffic = 4 x 2.147 x 32 x 32 = 8,796 GB = 8.8 TB
time    = 8796 / 3350 = 2.63 s

FLOPs = 4 x 32768^2 x 128 x 32 x 32 = 5.63e14 = 563 TFLOP
time  = 563e12 / 989e12 = 0.569 s
\`\`\`

Ratio: \`2.63 / 0.569 = 4.6×\` — **unchanged**. Both terms are \`O(N²)\`, so the ratio is
scale-invariant; it is set by the constant \`4 touches × 2 bytes\` against \`4d\` FLOPs per element.
The absolute numbers get catastrophic (2.6 seconds of intermediate shuffling) but the *shape* of
the problem is the same at every length. This is why FlashAttention helps at all lengths, not just
long ones.

**Part 2**

a) \`m1 = 2.0\`; \`l1 = exp(0) + exp(-1) = 1 + 0.367879 = 1.367879\`

b) \`m2 = 5.0\`; \`l2 = exp(0) + exp(-2) = 1 + 0.135335 = 1.135335\`

c) Global max \`m = max(2.0, 5.0) = 5.0\`. Rescale:
\`\`\`
l = l1 * exp(m1 - m) + l2 * exp(m2 - m)
  = 1.367879 * exp(-3.0) + 1.135335 * exp(0)
  = 1.367879 * 0.0497871 + 1.135335
  = 0.0681015 + 1.135335
  = 1.2034365
\`\`\`

d) Direct softmax over \`[2.0, 1.0, 5.0, 3.0]\` with \`m = 5.0\`:
\`\`\`
exp(2-5) = 0.0497871
exp(1-5) = 0.0183156
exp(5-5) = 1.0
exp(3-5) = 0.1353353
                     sum = 1.2034380
\`\`\`

The two agree to seven significant figures (the residual is fp rounding in my arithmetic, not the
method). The agreement is **exact** because \`exp(a - m1) × exp(m1 - m) = exp(a - m1 + m1 - m) =
exp(a - m)\` is an algebraic identity. The rescaling re-expresses the same quantity in a different
base; it does not approximate it. This is the entire reason FlashAttention is a drop-in
replacement requiring no re-evaluation.

**Part 3**

a) Work units = \`batch × heads = 8 × 32 = 256\`. With 132 SMs that is about 1.9 units per SM —
occupancy is nominally fine, but each unit serially scans 8,192 keys, so the kernel is
latency-bound on a long dependent scan rather than throughput-bound. In practice utilization is
poor.

b) Chunks = \`8192 / 512 = 16\`. Work units = \`8 × 32 × 16 = 4,096\`, a **16× increase**, and each
unit now scans only 512 keys.

c) At batch 1:
\`\`\`
without split-K:  1 x 32 = 32 units for 132 SMs  ->  76% of the GPU is IDLE
with split-K:     1 x 32 x 16 = 512 units        ->  fully occupied
\`\`\`

**FlashDecoding helps far more at small batch.** At large batch you already have enough units to
fill the machine; at batch 1 you cannot fill it at all without splitting the KV dimension. Since
small batch is exactly the latency-sensitive regime people care about, this is where the technique
earns its place.

d) The attention step reads the KV cache. Without GQA awareness you read each KV head once per
query head that uses it — 4 redundant reads. With awareness you read once and reuse from
registers/SRAM:
\`\`\`
naive:      1 MAC per KV element read           I = ~1 FLOP/byte
GQA-aware:  4 MACs per KV element read          I = ~4 FLOP/byte
\`\`\`
A **4× improvement in the arithmetic intensity of the attention step**. Since attention at decode
is memory-bound, that translates fairly directly into a 4× faster attention step. A kernel that
materializes replicated KV heads to make the shapes match throws this away entirely — a real and
easily-made performance bug.

**Part 4**

a)
\`\`\`
per layer: 2 norms + 2 RoPE + 2 residual + 1 SiLU-mul = 7
total: 7 x 32 = 224 elementwise launches
overhead: 224 x 5 us = 1,120 us = 1.12 ms
\`\`\`
(Plus the matmul launches — 7 per layer — so the true count is higher; 224 is the elementwise
portion alone.)

b) \`1.12 / 4.5 = 24.9%\` of the theoretical step budget, spent launching work rather than doing it.

c) With CUDA graphs at 0.5 µs: \`224 × 0.5 = 112 µs = 0.112 ms\`, which is \`2.5%\`. **A 10×
reduction from one change that requires no kernel work at all** — which is why it is the first
thing to check when a naive loop is far off the roofline.

d) Fusion to ~3 launches per layer = 96 launches; with CUDA graphs, \`96 × 0.5 = 48 µs = 0.048 ms\`,
or **1.1%** of the budget. Overhead has gone from a quarter of your step time to a rounding error.

The lesson: **CUDA graphs are the cheap 10×, fusion is the expensive further 2×.** Do them in that
order.`,
  },

  codeLab: {
    goal: `You do not need to write CUDA. You need to convince yourself the online softmax is
exact, and to see the memory-traffic argument in numbers.

**Part A** — implement tiled attention with online softmax in NumPy and verify it matches naive
attention to floating-point precision.

**Part B** — count HBM traffic for both, and reproduce the ratio from the math lab.

**Part C** — implement the FlashDecoding split-K combine and confirm the parallel partials merge
exactly.`,
    code: `"""
FlashAttention on paper, in NumPy. The point is to prove exactness to yourself.

    pip install numpy
"""
import numpy as np

rng = np.random.default_rng(3)


# ==========================================================================
# Part A -- naive vs tiled
# ==========================================================================
def attention_naive(Q, K, V, causal=True):
    """Materializes the full [N, N] score matrix. The thing we are avoiding."""
    N, d = Q.shape
    S = (Q @ K.T) / np.sqrt(d)
    if causal:
        S = np.where(np.triu(np.ones((N, N), dtype=bool), 1), -np.inf, S)
    S = S - S.max(axis=-1, keepdims=True)
    P = np.exp(S)
    P = P / P.sum(axis=-1, keepdims=True)
    return P @ V


def attention_flash(Q, K, V, block_q=64, block_k=64, causal=True):
    """Tiled, with online softmax. Never builds the [N, N] matrix.

    Follows the FlashAttention-2 loop order: query blocks outer, key blocks inner.
    """
    N, d = Q.shape
    scale = 1.0 / np.sqrt(d)
    O = np.zeros_like(Q)

    for i0 in range(0, N, block_q):
        i1 = min(i0 + block_q, N)
        Qi = Q[i0:i1]                                    # -> SRAM

        # running state, held in registers across the inner loop
        m = np.full((i1 - i0, 1), -np.inf)               # running max
        l = np.zeros((i1 - i0, 1))                       # running sum
        acc = np.zeros((i1 - i0, d))                     # running UNNORMALIZED output

        for j0 in range(0, N, block_k):
            j1 = min(j0 + block_k, N)
            if causal and j0 > i1 - 1:
                break                                     # whole block is masked out
            Kj, Vj = K[j0:j1], V[j0:j1]                   # -> SRAM

            S = (Qi @ Kj.T) * scale                       # [bq, bk], stays in SRAM
            if causal:
                qi = np.arange(i0, i1)[:, None]
                kj = np.arange(j0, j1)[None, :]
                S = np.where(kj > qi, -np.inf, S)

            m_blk = S.max(axis=-1, keepdims=True)
            m_new = np.maximum(m, m_blk)
            # guard the all-masked row case: -inf - -inf is nan
            m_new = np.where(np.isneginf(m_new), 0.0, m_new)

            correction = np.exp(m - m_new)                # rescale factor for old state
            correction = np.where(np.isnan(correction), 0.0, correction)

            P = np.exp(S - m_new)                         # [bq, bk]
            P = np.where(np.isnan(P), 0.0, P)

            l = l * correction + P.sum(axis=-1, keepdims=True)
            acc = acc * correction + P @ Vj               # accumulate unnormalized
            m = m_new

        O[i0:i1] = acc / np.where(l == 0, 1, l)           # single divide at the end

    return O


print("=== Part A: is the tiled version exact? ===")
for N, d in [(128, 64), (512, 64), (1024, 128)]:
    Q = rng.standard_normal((N, d)).astype(np.float64)
    K = rng.standard_normal((N, d)).astype(np.float64)
    V = rng.standard_normal((N, d)).astype(np.float64)

    ref = attention_naive(Q, K, V)
    for bq, bk in [(32, 32), (64, 64), (128, 256)]:
        got = attention_flash(Q, K, V, bq, bk)
        err = np.abs(ref - got).max()
        print(f"  N={N:>5} d={d:>4} blocks {bq:>3}x{bk:<4} max abs error {err:.3e}"
              f"   {'EXACT (fp roundoff only)' if err < 1e-12 else 'MISMATCH'}")


# ==========================================================================
# Part B -- HBM traffic
# ==========================================================================
def traffic_naive(N, d, n_heads, n_layers, dtype_bytes=2):
    per_head = (
        2 * N * d * dtype_bytes          # read Q, K
        + N * N * dtype_bytes            # write S
        + N * N * dtype_bytes            # read S
        + N * N * dtype_bytes            # write P
        + N * N * dtype_bytes            # read P
        + N * d * dtype_bytes            # read V
        + N * d * dtype_bytes            # write O
    )
    return per_head * n_heads * n_layers


def traffic_flash(N, d, n_heads, n_layers, dtype_bytes=2, sram_blocks=1):
    # Q, K, V read (K and V re-read once per query block), O written.
    per_head = (N * d * dtype_bytes) * (2 + 2 * sram_blocks)
    return per_head * n_heads * n_layers


def attn_flops(N, d, n_heads, n_layers):
    return 4 * N * N * d * n_heads * n_layers


print("\\n=== Part B: HBM traffic, Llama-3-8B prefill (32 heads, 32 layers, d=128) ===")
print(f"  {'N':>7} {'naive GB':>10} {'flash GB':>10} {'ratio':>8} "
      f"{'naive ms':>9} {'flash ms':>9} {'flop ms':>8}")
BW, PEAK = 3350e9, 989e12
for N in (2048, 8192, 32768, 131072):
    tn = traffic_naive(N, 128, 32, 32)
    tf = traffic_flash(N, 128, 32, 32, sram_blocks=max(1, N // 4096))
    fl = attn_flops(N, 128, 32, 32)
    print(f"  {N:>7} {tn/1e9:>10.1f} {tf/1e9:>10.2f} {tn/tf:>7.0f}x "
          f"{tn/BW*1000:>9.1f} {tf/BW*1000:>9.1f} {fl/PEAK*1000:>8.1f}")

print("\\n  -> naive is memory-bound (traffic time >> flop time)")
print("     flash is compute-bound (flop time >> traffic time). Same arithmetic.")


# ==========================================================================
# Part C -- FlashDecoding split-K combine
# ==========================================================================
def decode_attention_split(q, K, V, n_splits=8):
    """One query vs a long KV cache, computed as parallel partials then merged.

    Each split is independent -- on a GPU these run on different SMs.
    """
    d = q.shape[0]
    scale = 1.0 / np.sqrt(d)
    n = K.shape[0]
    bounds = np.linspace(0, n, n_splits + 1).astype(int)

    partials = []
    for a, b in zip(bounds[:-1], bounds[1:]):
        if a == b:
            continue
        s = (q @ K[a:b].T) * scale
        m = s.max()
        p = np.exp(s - m)
        partials.append((m, p.sum(), p @ V[a:b]))       # (max, sum, unnormalized out)

    # combine with exactly the online-softmax rescaling
    m_all = max(p[0] for p in partials)
    l_all = sum(p[1] * np.exp(p[0] - m_all) for p in partials)
    o_all = sum(p[2] * np.exp(p[0] - m_all) for p in partials)
    return o_all / l_all


print("\\n=== Part C: FlashDecoding split-K combine ===")
d, n = 128, 8192
q = rng.standard_normal(d)
K = rng.standard_normal((n, d))
V = rng.standard_normal((n, d))

s = (q @ K.T) / np.sqrt(d)
p = np.exp(s - s.max())
ref = (p / p.sum()) @ V

for splits in (1, 2, 8, 16, 64, 256):
    got = decode_attention_split(q, K, V, splits)
    print(f"  {splits:>4} splits  max abs error {np.abs(ref-got).max():.3e}"
          f"   work units per (batch,head): {splits}")

print("\\n  -> the answer does not change with the split count. Parallelism is free.")

# --- TODO for you ---
#   1. Break the online softmax on purpose: drop the exp(m - m_new) correction
#      and see how wrong the answer gets. That term is the whole algorithm.
#   2. Add GQA to decode_attention_split: 8 KV heads, 32 query heads. Count how
#      many bytes of K you read with and without reusing a KV head across its
#      4 query heads.
#   3. Work out, for block_q=64, block_k=64, d=128, fp16, how many bytes one
#      tile's working set needs. Does it fit in 228 KB of shared memory?
`,
    expect: `**Part A** prints a max absolute error around \`1e-15\` to \`1e-16\` for every block
size — pure float64 rounding. Not "close", not "within tolerance for a fast approximation":
**identical to machine precision, at every block size**. Changing the tiling does not change the
answer. That is the property that made FlashAttention adoptable without re-evaluating any model.

**Part B** reproduces the math lab. At \`N = 8192\` naive attention moves ~550 GB against flash's
single-digit GB — a ratio in the hundreds. The three timing columns tell the real story: naive's
traffic time far exceeds its FLOP time (memory-bound on an intermediate that need not exist),
while flash's FLOP time far exceeds its traffic time (compute-bound, which is where you want to
be). Same arithmetic, opposite regime.

**Part C** shows the error staying at rounding level from 1 split to 256. The number of parallel
chunks has **no effect on the answer**, which is what makes FlashDecoding's parallelism free — you
can split as far as you need to fill the GPU without any accuracy consequence.

If TODO 1 is done, dropping the correction term produces errors of order 1.0 — completely wrong
output. Worth doing, because it makes clear that the rescaling is not a numerical nicety but the
load-bearing element of the whole algorithm.`,
    stretch: `Add a wall-clock comparison. Time \`attention_naive\` and \`attention_flash\` in NumPy
at \`N\` from 512 to 8192 and plot both. NumPy will not reproduce GPU behaviour — it has no shared
memory to exploit and the tiled version will likely be *slower* due to Python loop overhead — and
that is itself the lesson: FlashAttention is not an algorithmic-complexity win, it is a
memory-hierarchy win, and it only pays on hardware with the hierarchy it was designed for.

Then measure peak memory for both with \`tracemalloc\`. Naive should grow quadratically with \`N\`
while flash stays roughly flat. That part *does* transfer to any hardware, and it is why
long-context training became possible.`,
  },

  papers: [
    {
      title: 'FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness',
      by: 'Dao et al., 2022',
      url: 'https://arxiv.org/abs/2205.14135',
      why: 'The paper that reframed attention performance as an IO problem rather than a FLOP problem. Among the most impactful systems contributions in deep learning.',
      frame: `**Section 3.1** has the algorithm — work through Algorithm 1 line by line with the
online softmax derivation in hand. **Section 3.2** is the IO-complexity analysis giving
\`O(N²d²/M)\`; understand where the \`M\` comes from. Note that the word "exact" is in the title
deliberately: the contrast is with the approximate-attention literature the paper was pushing
back on. Skim the experiments.`,
    },
    {
      title: 'FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning',
      by: 'Tri Dao, 2023',
      url: 'https://arxiv.org/abs/2307.08691',
      why: 'Roughly 2x over the original, from scheduling rather than algorithmic changes. Instructive about what actually limits GPU kernels.',
      frame: `**Section 3.1** on reducing non-matmul FLOPs is the key insight: on an A100, non-matmul
throughput is roughly 16× lower than tensor-core matmul throughput, so the rescaling operations
matter far more than their FLOP count suggests. **Section 3.2** covers the loop-order swap and
parallelizing over sequence length. Read for the reasoning, not the numbers.`,
    },
    {
      title: 'FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision',
      by: 'Shah et al., 2024',
      url: 'https://arxiv.org/abs/2407.08608',
      why: 'Hopper-specific: warp specialization, async TMA, and FP8. Shows what extracting the last factor of two now requires.',
      frame: 'Read the introduction and Section 3 for the asynchrony techniques. The details are hardware-specific and will date; the durable lesson is that modern kernel performance comes from overlapping heterogeneous units rather than from better asymptotics.',
    },
    {
      title: 'Online normalizer calculation for softmax',
      by: 'Milakov & Gimelshein, 2018',
      url: 'https://arxiv.org/abs/1805.02867',
      why: 'The online softmax itself, four years before FlashAttention used it. Short, and the algebra is the foundation of everything in this module.',
      frame: 'Three pages. Read all of it. The rescaling identity in Section 3 is the single most important piece of algebra in modern attention kernels, and seeing it stated on its own — outside the attention context — makes it much clearer.',
    },
    {
      title: 'Mamba: Linear-Time Sequence Modeling with Selective State Spaces',
      by: 'Gu & Dao, 2023',
      url: 'https://arxiv.org/abs/2312.00752',
      why: 'The strongest version of the alternative bet: constant-size recurrent state, no KV cache at all.',
      frame: `Read **Section 3** for the selection mechanism — making the state transitions
input-dependent is what lifted SSMs to competitive quality. **Section 3.3** on the hardware-aware
parallel scan is a nice companion to FlashAttention: same philosophy of designing the algorithm
around the memory hierarchy. Read the results with a critical eye toward recall-heavy tasks, which
is where the constant-state limitation shows.`,
    },
  ],

  checkpoint: {
    claim: `You can trace the FlashAttention tiling on paper for a small matrix and convince
yourself the online softmax is exact — and you can explain why decode needs a different kernel
than prefill.`,
    questions: [
      {
        q: 'Why is FlashAttention faster when it does more arithmetic than naive attention?',
        a: `Because attention was never limited by arithmetic. Naive attention writes and reads
the \`[N, N]\` score matrix four times: at \`N = 8192\` with 32 heads and 32 layers that is about
550 GB of HBM traffic, roughly 164 ms at 3.35 TB/s, against only ~36 ms of actual arithmetic.
FlashAttention keeps tiles in shared memory — about 6× the bandwidth of HBM — and never writes the
score matrix out at all, so the 164 ms largely disappears. In the backward pass it *recomputes*
attention probabilities rather than storing them, spending extra FLOPs to save bandwidth, and it
is still faster. That trade only makes sense once you accept that bytes, not FLOPs, are the
currency.`,
      },
      {
        q: 'Show why the online softmax rescaling is exact rather than approximate.',
        a: `Suppose you have processed a block with max \`m1\` and partial sum
\`l1 = Σ exp(x - m1)\`, and a new block raises the max to \`m\`. Multiply the old sum by
\`exp(m1 - m)\`. For each term, \`exp(x - m1) × exp(m1 - m) = exp(x - m1 + m1 - m) = exp(x - m)\` —
which is exactly the term you would have computed had you known \`m\` from the start. The rescaling
re-expresses the same quantity in a different exponential base; nothing is dropped or
approximated. The same factor applies to the accumulated output. This is why FlashAttention is a
drop-in for any trained model with no re-evaluation needed, and it is the reason it displaced the
approximate-attention literature rather than joining it.`,
      },
      {
        q: 'Why does the decode phase need a different attention kernel than prefill?',
        a: `Because the parallelism disappears. FlashAttention parallelizes over batch, heads, and
query blocks — and at decode the query length is 1, so query-block parallelism is gone. At batch 1
with 32 heads you have 32 independent work units for a 132-SM H100: over 75% of the machine idle,
with each unit serially scanning the entire KV cache. FlashDecoding fixes this by splitting along
the *KV* dimension: compute partial results for chunks of the cache in parallel, then merge them
with the same online-softmax rescaling. Splitting 8,192 keys into 16 chunks gives 16× the work
units. Notice the algebra pays off twice — introduced for sequential tiling, reused for parallel
combination.`,
      },
      {
        q: 'What does a GQA-aware decode kernel do differently, and what does it buy?',
        a: `With 8 KV heads serving 32 query heads, four query heads share each KV head. A
GQA-aware kernel reads a KV head from HBM once and reuses it from registers or shared memory for
all four, raising the arithmetic intensity of the attention step from about 1 to about 4
MAC-per-byte. Since decode attention is memory-bound, that is close to a 4× faster attention step.
A kernel that instead materializes replicated KV heads to make the tensor shapes match reads the
same data four times and throws the entire benefit away — a real and easily-made bug, and one
reason GQA support has to be built into the kernel rather than handled by a reshape.`,
      },
      {
        q: 'Why are unfused elementwise operations disproportionately expensive at decode time?',
        a: `Because they have essentially zero arithmetic intensity and there is nothing to hide
them behind. An RMSNorm on a \`[1, 4096]\` vector reads 8 KB, does a few FLOPs per element, and
writes 8 KB — a full HBM round trip for almost no work. A 32-layer decode step has roughly seven
such operations per layer, so about 224 of them, each with ~5 µs of launch overhead: 1.12 ms
against a 4.5 ms theoretical step, or 25% of the budget spent launching rather than computing.
During prefill these hide behind compute-bound matmuls; at decode you are already memory-bound so
every round trip is fully exposed. CUDA graphs cut per-launch cost roughly 10× and are the cheap
fix; fusion is the further, more expensive one.`,
      },
      {
        q: 'Why did hybrid attention/SSM models win over pure state-space models?',
        a: `Because a constant-size recurrent state cannot losslessly summarize an unbounded
history — that is an information-theoretic limit, not an implementation shortfall. Pure SSMs like
Mamba get constant decode memory with no KV cache, which is an enormous inference advantage, but
they underperform on tasks needing precise recall from long context: copying, retrieval, in-context
learning from many examples. Attention can look up any previous token exactly because it keeps
every key. Hybrids interleave mostly SSM or linear-attention layers with a few full-attention
layers, so most of the memory saving is retained while the attention layers supply exact recall
where it is needed. Jamba, Zamba and Samba are examples. The general shape — most layers cheap,
a few layers exact — recurs throughout this field.`,
      },
    ],
  },

  pitfalls: [
    {
      wrong: 'FlashAttention is an approximation that trades accuracy for speed.',
      right: `It is exact. The online softmax rescaling is an algebraic identity, not a numerical
shortcut, so the output is identical to naive attention up to floating-point reassociation.
Exactness is the entire reason it was adoptable — you can apply it to any trained checkpoint with
no re-evaluation, which the approximate-attention methods it displaced could never claim.`,
    },
    {
      wrong: 'FlashAttention makes attention sub-quadratic.',
      right: `The FLOPs are still O(N²d) and the KV cache still grows linearly. What becomes
linear is the *memory* required for intermediates, because the score matrix is never
materialized. It removes a memory-traffic constant, not an asymptotic term. Sub-quadratic
attention is a different research line (linear attention, SSMs, sparse patterns) with different
trade-offs.`,
    },
    {
      wrong: 'Using a prefill-tuned attention kernel for decode.',
      right: `At query length 1 there are no query blocks to parallelize over, so a standard
FlashAttention kernel leaves most of the GPU idle — at batch 1 with 32 heads, over 75% of an
H100's SMs have nothing to do. You need a split-K decode kernel (FlashDecoding) that parallelizes
along the KV dimension instead. Serving engines ship separate kernels for the two phases for
exactly this reason.`,
    },
    {
      wrong: 'Reaching for kernel fusion before enabling CUDA graphs.',
      right: `Fusion is real work — writing or sourcing kernels, validating them. CUDA graphs are
a configuration change that cuts per-launch overhead roughly 10×, taking launch cost from ~25% of
a decode step to ~2.5%. For a decode loop, where every step has identical shapes, graph capture is
close to free. Do the configuration change first and re-measure before writing any kernels.`,
    },
  ],

  glossary: [
    { term: 'shared memory / SRAM', def: 'On-chip memory, ~228 KB per SM on Hopper, with roughly 6x the bandwidth of HBM. The tier FlashAttention keeps its tiles in.' },
    { term: 'online softmax', def: 'Computing softmax incrementally with a running max and sum, rescaling earlier partials when a larger value appears. Exact, not approximate.' },
    { term: 'FlashAttention', def: 'Tiled exact attention that never materializes the N x N score matrix in HBM. Cuts HBM accesses to O(N^2 d^2 / M).' },
    { term: 'FlashDecoding', def: 'Decode-phase attention that parallelizes along the KV dimension (split-K) because query-length-1 offers no query parallelism.' },
    { term: 'split-K', def: 'Partitioning a reduction across parallel workers and combining their partial results afterward.' },
    { term: 'kernel fusion', def: 'Combining adjacent operations into one kernel so intermediates never round-trip to HBM.' },
    { term: 'CUDA Graph', def: 'A captured, replayable sequence of kernel launches. Removes most per-launch CPU overhead; essential for decode loops.' },
    { term: 'warp specialization', def: 'Assigning different warps within a block to different roles (producer/consumer) so data movement overlaps computation. Central to FlashAttention-3.' },
    { term: 'linear attention', def: 'Replacing softmax with a kernel feature map so attention can be reassociated into O(N d^2). Constant state, weaker exact recall.' },
    { term: 'SSM', def: 'State space model. Maintains a fixed-size recurrent state, so there is no KV cache; Mamba is the leading example.' },
    { term: 'attention hybrid', def: 'A stack mixing mostly SSM or linear-attention layers with a few full-attention layers, trading some memory saving for exact recall.' },
  ],
};
