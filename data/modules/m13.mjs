export default {
  n: 13,
  slug: 'gpus-and-tpus',
  title: 'GPUS AND TPUS',
  tagline: 'Two chips built on opposite philosophies, bolted to the same kind of memory. What a systolic array buys, what a warp scheduler buys, and why decode barely notices either.',
  hours: '6–8 hours',
  prereqs: ['Module 4', 'Module 7', 'Module 9'],

  bigIdea: `Every roofline in these chapters so far has been drawn for one machine — an H100 — and
every conclusion has been about the *shape* of the roofline rather than the chip underneath it.
This level swaps the chip out and asks which of those conclusions survive.

The two designs could hardly have started further apart. The GPU is a graphics processor that grew
into a general-purpose parallel machine: thousands of hardware threads, a scheduler that picks
which of them runs each cycle, a cache hierarchy that guesses what you will need next, and matrix
units bolted on later as an accelerator inside an accelerator. The TPU was designed backwards from
one operation. Google built the first one because a projection said that if speech recognition
took off, they would need to double their datacenter footprint — so the chip is a matrix multiplier
with the smallest amount of other machinery that will keep it fed. No caches in the GPU sense. No
dynamic instruction scheduling. One enormous grid of multiply-accumulate cells with data flowing
through it.

And yet: **both are attached to HBM, and their bandwidths differ by less than a factor of two.**
That is the fact this whole level turns on. Generating one token at batch 1 means reading every
weight once, so decode time is \`weight_bytes / bandwidth\` on both machines, and every architectural
difference below — the systolic array, the warp scheduler, the scratchpad, the compiler — sits on
the side of the ledger that decode never reads. The differences are real and they are large, but
they show up in *prefill*, in *large-batch decode*, in *energy per token*, and in *what happens when
you need 64 chips instead of 8* — never in the single-stream latency number people usually quote
when comparing them.

By the end you should be able to look at any accelerator datasheet — including ones that do not
exist yet — and work out in about five minutes what it will and will not do for inference: what
its ridge point is, what batch size its arithmetic geometry demands, how far its fast interconnect
reaches, and which of those three will bite you first.`,

  concepts: [
    {
      name: 'Same wall, two floor plans',
      keyPoint: 'GPUs and TPUs are both HBM-attached and within ~2× on bandwidth, so at batch 1 they are the same machine; every difference below lives on the compute and network side of the roofline.',
      body: `Put the two families in one table and the first thing to notice is how *similar* the
column that matters is.

| chip | HBM | bandwidth | dense bf16 | ridge point |
|---|---|---|---|---|
| A100 80GB | 80 GB | 2.04 TB/s | 312 TFLOP/s | 153 |
| H100 SXM | 80 GB | 3.35 TB/s | 989 TFLOP/s | 295 |
| H200 SXM | 141 GB | 4.8 TB/s | 989 TFLOP/s | 206 |
| TPU v4 | 32 GB | 1.2 TB/s | 275 TFLOP/s | 229 |
| TPU v5e | 16 GB | 0.82 TB/s | 197 TFLOP/s | 241 |
| TPU v5p | 95 GB | 2.77 TB/s | 459 TFLOP/s | 166 |
| TPU v6e | 32 GB | 1.64 TB/s | ~918 TFLOP/s | ~560 |

Nothing in that table is a different *kind* of number. Both families buy the same HBM stacks from
the same two or three suppliers, and no amount of architectural cleverness on the compute side
changes what those stacks deliver. So the ratio that Module 4 taught you to compute first — the
ridge point, \`dense FLOP/s ÷ bandwidth\` — lands in the same band for both, roughly 150 to 300, and
its ordering does not track any intuition about which chip is "better." The v5p has the lowest
ridge point in the table and the H100 nearly the highest; that means the v5p needs a smaller batch
to saturate its arithmetic, which is a statement about the *balance* of the part, not its speed.
The v6e sits at ~560 for the opposite reason: its compute grew 4.7× over the v5e while its
bandwidth only doubled. Same memory wall, same direction of travel, on both sides of the aisle.

The consequence is worth being blunt about, because it is the conclusion most comparisons get
wrong. **At batch 1, decoding on a TPU and decoding on a GPU are the same operation with the same
bottleneck, and the faster chip is simply the one with more bandwidth.** Every structural
difference in the rest of this level — a systolic array versus tensor cores, a compiler-managed
scratchpad versus hardware caches, a torus versus an NVLink island — is invisible to that
measurement. They become visible the moment you leave it: at prefill, at large batch, at scale, and
on the power bill.

One clarification before going further, because the vocabulary is genuinely confusing. "TPU" here
means Google's Tensor Processing Unit, whose compute core is a systolic array. "Tensor core" means
the matrix unit inside an NVIDIA SM, which is *not* a systolic array in the same sense. The names
collide; the designs do not.`,
      ascii: `  ONE DECODE STEP, LLAMA-3-8B fp16, BATCH 1

  H100 SXM   weights  16.06 GB / 3.35 TB/s  ████████████████       4.79 ms
             math    16.06 GFLOP / 989 TF/s  ▏                     0.02 ms

  TPU v5p    weights  16.06 GB / 2.77 TB/s  ███████████████████    5.81 ms
             math    16.06 GFLOP / 459 TF/s  ▏                     0.03 ms

  The gap between the two machines is 1.21× — exactly their bandwidth ratio.
  A systolic array, a warp scheduler, a cache, a compiler: none of them
  contribute anything to these four rows. That is what this level is about.`,
    },
    {
      name: 'Inside a GPU: oversubscription as a design principle',
      keyPoint: 'A GPU hides memory latency by keeping far more threads resident than it can run, and a hardware scheduler picks a ready one every cycle; the matrix units are an accelerator inside that machine, not the machine itself.',
      body: `An H100 SXM is 132 **streaming multiprocessors**. Each SM is divided into four
processing blocks, and each block has its own warp scheduler, its own register file slice, and its
own tensor core. Around them sits 256 KB of registers and up to 228 KB of combined L1/shared memory
per SM, backed by a 50 MB L2 shared across the chip and then 80 GB of HBM3.

The unit of execution is the **warp**: 32 threads that issue the same instruction together. This is
SIMT — single instruction, multiple threads — and it is a compromise between the two obvious
designs. You write scalar code for one thread, and the hardware runs 32 of them in lockstep; when
they disagree at a branch, the hardware executes both sides with some lanes masked off, which is
correct but costs you the lanes. So control flow is *allowed* but *priced*.

The interesting design choice is what the GPU does about memory latency, which is roughly 500
cycles to HBM. It does not try to avoid the wait. It oversubscribes: an SM can hold up to 64 warps
resident simultaneously, each with its registers already allocated, and every cycle the scheduler
picks one that is not blocked and issues from it. **Latency is hidden by having somebody else to
run**, and the metric for whether you have enough somebodies is occupancy. This is the deep reason
GPUs need thousands of threads to go fast, and the reason a kernel that uses too many registers per
thread runs slowly even when it is doing the right arithmetic — fewer registers left over means
fewer resident warps means nothing to switch to when one stalls.

The tensor cores are a later addition, and it shows in how they are reached. They are per-processing
block instructions, not a separate chip: a warp (Volta through Ampere) or a **warpgroup** of four
warps (Hopper's \`wgmma\`) collectively issues a matrix-multiply-accumulate over operands staged in
registers and shared memory. Two numbers put them in perspective. First, they are where essentially
all the FLOPs are — the H100's non-tensor FP32 throughput is around 67 TFLOP/s against 989 TFLOP/s
dense BF16 on the tensor cores, so a kernel that does not use them is leaving roughly 93% of the
machine unused. Second, they are fast enough that feeding them is the entire problem, which is why
Hopper also added the **Tensor Memory Accelerator** — a DMA engine that copies tiles between global
and shared memory asynchronously, so the SM does not spend its warps on address arithmetic — and
thread-block clusters, which let neighbouring SMs read each other's shared memory.

Every one of these features is a way of getting bytes to the matrix units on a machine whose
organising principle is *generality*. That generality is not free, and the next concept is what a
chip looks like when it declines to pay for it.`,
      ascii: `  ONE SM (× 132 on an H100)

  ┌──────────────────────────────────────────────────────┐
  │  4 × [ warp scheduler │ regs 64 KB │ TENSOR CORE ]   │
  │                                                      │
  │  up to 64 resident warps — the scheduler issues      │
  │  (2048 threads)            from any warp that is     │
  │                            not stalled this cycle    │
  │                                                      │
  │  SHARED MEMORY / L1 : 228 KB   ← you manage this     │
  └────────────────────────────┬─────────────────────────┘
                               │
                      L2 CACHE 50 MB   ← hardware manages this
                               │
                      HBM3 80 GB @ 3.35 TB/s   ~500 cycles away`,
    },
    {
      name: 'Inside a TPU: a matmul built out of wires',
      keyPoint: 'The MXU is a 128×128 grid of multiply-accumulate cells with weights held stationary and activations flowing through; a value read once from memory feeds 128 multiplications, which is where the energy advantage comes from.',
      body: `A TPU chip contains one or two **TensorCores** (v5e has one, v4 and v5p have two). A
TensorCore is four **MXUs**, a vector unit, and a scalar unit, and almost all of the silicon and all
of the headline FLOPs are in the MXUs.

An MXU is a 128×128 grid of multiply-accumulate cells — 16,384 of them — wired only to their
immediate neighbours. In the weight-stationary dataflow TPUs use, a 128×128 tile of the weight
matrix is loaded into the grid and held there. Activation rows then enter from the left, one per
cycle, and partial sums flow downward, each cell adding its own product to the sum arriving from
above. After the wavefront has crossed the array, a finished row of outputs emerges from the bottom
every cycle.

The reason to build a matmul this way is not that it is faster per MAC. It is what does *not*
happen. In a conventional design, each multiply-accumulate reads two operands from a register file
and writes one back; the register file access costs several times more energy than the arithmetic
does. In a systolic array, the operand is handed directly from one cell to its neighbour along a
wire that is a few micrometres long. **One value fetched from memory feeds 128 multiplications on
its way through the array**, and the intermediate sums are never written anywhere addressable. That
is the whole trick, and it is a dataflow trick rather than a circuit one.

You can check the geometry against the datasheet, which is a good habit and the first thing the
math lab asks you to do. A v4 chip has 8 MXUs; each retires 128 × 128 MACs — 2 FLOP apiece — per
cycle, so peak equals \`8 × 16,384 × 2 × clock\`. Setting that to the published 275 TFLOP/s gives
1.05 GHz, which is exactly the v4's documented clock. The arithmetic closes, which means the mental
model is right.

Two things follow from the geometry, and both matter more for inference than the energy story.

**There is a pipeline to fill.** The first output does not appear until the wavefront has crossed
all 128 rows of the array. Streaming *m* rows through one weight tile therefore costs roughly
\`m + 128\` cycles to do *m* rows of useful work, so the array runs at \`m / (m + 128)\` of peak: 50%
at m = 128, 80% at m = 512, and 0.8% at m = 1. Real MXUs pipeline consecutive tiles to hide some of
this, so the true curve is better than the naive one — but the shape is right, and it means **the
batch size at which a TPU becomes efficient is set by its array geometry, not only by its ridge
point.**

**128 becomes a quantum.** Feeding a 128-wide array with a tensor whose dimension is not a multiple
of 128 means padding, and padded lanes compute zeros at full cost. This is one of the quieter ways
hardware has shaped models: it is not a coincidence that hidden sizes, head dimensions and
intermediate widths in modern architectures are almost always multiples of 128.

The rest of the TensorCore exists because a transformer is not only matmuls. The **VPU** is a
(8, 128)-shaped lane array that does the elementwise work — softmax's exponentials, RMSNorm,
SwiGLU's gating, the residual adds — and there is a scalar unit that runs the control flow and
issues the very wide VLIW instruction bundles the whole core executes in order. Nothing here
predicts branches, reorders instructions, or prefetches. What runs, and when, was decided by a
compiler before the program started.`,
      ascii: `  MXU: 128×128 WEIGHT-STATIONARY SYSTOLIC ARRAY

                               w00     w01     w0n     ← a 128×128 weight tile,
  row t   ─────────────►       ■ ────► ■ ────► ■       loaded once and held
  row t+1 ─────────────►       ■       ■       ■
  row t+2 ─────────────►       ■       ■       ■       activations enter left,
                               │       │       │       sums fall downward
                               ▼       ▼       ▼
                               out0    out1    outn    one finished row per cycle

  ONE activation read feeds 128 multiplies as it crosses ──►
  ONE weight load feeds every row that streams through   ──▼

  but: the wavefront needs 128 cycles to cross the array before
       the first result appears  ->  efficiency = m / (m + 128)`,
    },
    {
      name: 'Cache or scratchpad: who decides what stays on chip',
      keyPoint: 'A GPU gives you 228 KB of shared memory per SM behind a hardware-managed 50 MB L2; a TPU gives the compiler roughly 128 MiB of explicitly managed VMEM and no cache at all — about 9× more on-chip staging per unit of compute.',
      body: `Module 7 built FlashAttention out of one observation: the N×N score matrix never has to
exist in HBM if you tile the computation so each tile fits in the fast tier. That reasoning is
universal, but the tier it refers to is not the same object on the two machines.

On a GPU there are two kinds of on-chip memory and they answer to different masters. **Shared
memory** — up to 228 KB per SM, about 30 MB chip-wide — is a scratchpad you allocate and fill by
hand, and it is what FlashAttention's tiles live in. **L2**, 50 MB, is a cache: the hardware decides
what is in it, you influence it only indirectly through access patterns, and you find out whether
your kernel used it well by measuring. The split is deliberate. Caches make unpredictable, pointer-
chasing, branch-heavy code fast without anyone thinking about it, which is exactly what a general-
purpose parallel machine needs.

A TPU has no L1 or L2 in that sense. It has **VMEM**, a software-managed scratchpad of roughly
128 MiB on recent generations, and every byte that enters or leaves it does so because a compiler
emitted an instruction saying so. Nothing is speculative, nothing is evicted behind your back, and
there is no such thing as a cache miss — but equally, nothing is rescued at runtime if the schedule
was wrong.

Normalise those to compute and the difference stops looking like a detail. An H100 has about 31 KB
of programmer-visible scratchpad per TFLOP/s of arithmetic. A v5p has about 292 KB — roughly **9.4×
more staging area per unit of compute**. That is the single number that best explains why the two
programming models feel so different. On a GPU, a kernel author's central problem is that the fast
tier is tiny: everything is tiles, and FlashAttention is celebrated precisely because fitting
attention into 228 KB took real invention. On a TPU, whole activations for a layer can sit in VMEM
across several operations, so the equivalent optimisation is something the compiler does as a
matter of course when it fuses a subgraph — not a paper.

The trade is symmetric, and neither side is free. Hardware caches keep working when the access
pattern is data-dependent, which is what a paged KV cache with a per-sequence block table is: a
gather through an indirection computed at runtime. A statically scheduled scratchpad has to be told
where the bytes are, which means the *shape* of that gather must be known when the program is
compiled — even if the addresses are not. Which is the subject of the next concept, because it is
the place where the TPU's design has the most direct consequences for a serving stack.`,
      ascii: `  ON-CHIP STAGING, NORMALISED TO COMPUTE

  H100      SMEM 228 KB × 132 SMs = 30 MB  +  L2 50 MB (hardware-managed)
            ├── you manage ──┘                └── it manages ──┘
            31 KB of scratchpad per TFLOP/s

  TPU v5p   VMEM ~128 MiB, all compiler-managed, no cache
            └────────── XLA manages every byte ──────────┘
            292 KB of scratchpad per TFLOP/s   ~9.4× more`,
    },
    {
      name: 'Who schedules, and when: runtime hardware versus an ahead-of-time compiler',
      keyPoint: 'A GPU decides at runtime what runs where; a TPU program is scheduled by XLA before it starts, which is why static shapes, bucketing and recompilation are first-class concerns on TPU serving stacks and not on GPU ones.',
      body: `This is the difference that a serving engineer actually feels, and it is downstream of
everything above.

A CUDA kernel launch is a runtime event. You hand the driver a grid of blocks and a stream; the
hardware distributes blocks to SMs as they free up, warps stall and resume on their own schedule,
and the L2 fills with whatever was touched. Shapes can change between launches at no cost beyond
possibly picking a different kernel, control flow can depend on data, and the same binary handles a
batch of 1 and a batch of 137 without anyone recompiling anything. Eager-mode PyTorch is viable on
GPUs *because* the hardware absorbs that much dynamism.

A TPU program is compiled by **XLA** into a static schedule: which VLIW bundle issues on which
cycle, which DMA moves which tile into VMEM and when, laid out ahead of time against a machine
model. The compiler can do this only because it knows the shape of every tensor, which is why
**XLA specializes on shapes**: change a batch size or a sequence length and you get a different
program, compiled on first use, and cached under a key that includes those shapes.

For training this is nearly all upside — shapes are fixed, and a compiler that knows them can fuse
aggressively, schedule DMAs perfectly, and lay out the whole step with no runtime overhead at all.
For *serving*, whose defining property is that every request is a different length and the batch
changes on every iteration, it is a genuine tax. Everything in Modules 5, 6 and 12 — continuous
batching, paged blocks, admission and preemption every single step — is dynamism by construction.

The answer TPU serving stacks converge on is **bucketing**: quantize the dynamic dimensions to a
small set of values (batch 8/16/32/64, context 512/1024/2048/…), pad up to the nearest bucket, and
compile one program per bucket. That converts unbounded shape variety into a handful of cached
executables, at the cost of computing on padding — which is precisely the waste the systolic array's
128-quantum already taught you to expect, now applied at the level of the serving loop instead of
the matmul. The related discipline is keeping the *host* out of the inner loop: a scheduler that
makes a Python-level decision per step, on a machine whose whole advantage is a precomputed
schedule, will not go fast.

It is worth stating clearly what this is *not*, because the folklore overshoots. Paged attention,
continuous batching, and ragged inputs all exist on TPUs — vLLM has a TPU backend, and JAX/Pallas
lets you write the gathering kernels by hand when the compiler will not produce them. The cost is
not impossibility. It is that a dynamic-shape workload on a static-shape machine pays in
compilation time, in padded compute, and in engineering effort that on a GPU you would not have
spent — and conversely, the GPU pays for its dynamism continuously, in silicon area spent on
schedulers and caches and in an achieved fraction of peak that is rarely as high as a well-compiled
TPU program's.`,
      ascii: `  SAME REQUEST STREAM, TWO EXECUTION MODELS

  GPU     batch=1   batch=7   batch=23  batch=31   ← any shape, same binary
          └──────── one kernel, hardware schedules blocks ────────┘

  TPU     batch=1  ─┐
          batch=7  ─┼─► bucket 8   ──► program A   (compiled once, cached)
          batch=23 ─┤
          batch=31 ─┴─► bucket 32  ──► program B   (compiled once, cached)
                          ▲
                          └── 23 and 31 are padded up to 32, and the
                              padding is computed at full cost`,
    },
    {
      name: 'The fabric: an island with a cliff, or a torus with a diameter',
      keyPoint: 'NVLink gives a handful of GPUs enormous bandwidth and then falls off a cliff to the network; ICI gives every TPU in a pod the same moderate bandwidth to its neighbours, with no cliff but a hop count that grows with distance.',
      body: `Module 9 established that tensor parallelism belongs inside a node, because it does two
all-reduces per layer and NVLink is 18× faster than InfiniBand. That advice is correct — and it is
advice about a *topology*, not about parallelism in general. Change the topology and the advice
changes.

A GPU system is built as **islands**. Inside an NVLink domain — 8 GPUs on an H100 node, 72 on a
Blackwell NVL72 rack — every GPU talks to every other at around 900 GB/s through NVSwitch, which is
about a quarter of HBM bandwidth and fast enough that TP's all-reduces nearly disappear. Cross the
island boundary and you are on InfiniBand or Ethernet at roughly 50 GB/s per GPU: an 18× cliff, in
one hop. The entire parallelism playbook in Module 9 — TP inside the node, PP and DP across nodes —
is a map of where that cliff is.

A TPU pod has no cliff, because it has no switches in the middle. Each chip has **ICI** links wired
directly to its neighbours in a 2D (v5e) or 3D (v4, v5p) torus — six links at roughly 100 GB/s each
on a v5p — and that is the bandwidth between neighbouring chips whether the pod holds 64 of them or
8,960. What you get instead of a cliff is a **diameter**: two chips on opposite corners of a 3D
torus are many hops apart, so collectives are structured to move along torus axes, and the mapping
of a model onto the mesh is something you specify rather than something you hope for. TPU v4 adds
optical circuit switches between 4×4×4 cubes, which makes the pod's topology reconfigurable per job
and lets a failed cube be routed around instead of taking the pod down.

Run Module 9's own arithmetic across both shapes and the difference is stark. For Llama-3-70B
decoding at batch 32, tensor parallelism moves about 147 MB per token at TP-8 and 165 MB at TP-64,
by the ring all-reduce formula from that level:

\`\`\`
TP-8   over NVLink   900 GB/s  ->  0.16 ms   against a ~5 ms step: free
TP-64  over ICI      600 GB/s  ->  0.28 ms   against a ~5 ms step: still free
TP-64  over InfiniBand 50 GB/s ->  3.30 ms   against a ~5 ms step: catastrophic
\`\`\`

TP-64 is a perfectly reasonable configuration on a torus and an unusable one on a GPU cluster
without a large NVLink domain — not because the TPU's links are faster (they are not; NVLink is
faster than ICI per link) but because **the TPU's fast tier does not end at 8 chips.** That is the
single most useful thing to know about the fabric difference, and it is why the two ecosystems have
different instincts about how much of a model to shard and how far.

The same reasoning explains NVL72. Extending the NVLink domain from 8 GPUs to 72 does not make any
link faster; it moves the cliff. Which is a way of saying the two designs are converging on the
same insight from opposite ends.`,
      ascii: `  BANDWIDTH vs DISTANCE

  GPU     900 GB/s ████████████████  (within 8, or 72 on NVL72)
           50 GB/s █                 (everything beyond)  ← the cliff

  TPU     ~100 GB/s per link, 6 links per chip
          ███ ███ ███ ███ ███ ███ ███ ███ ...  out to thousands of chips
          no cliff — but chip (0,0,0) to chip (7,7,7) is 12 hops away`,
    },
  ],

  mathLab: {
    prompt: `Everything below reuses figures from elsewhere in these chapters — Llama-3-8B's 8.03e9
parameters, Llama-3-70B's 70.6e9 parameters with \`d_model\` 8192 and 80 layers, the H100's
3.35 TB/s and 989 TFLOP/s, and Module 9's ring all-reduce formula — plus the TPU datasheet figures
from the table in the first concept. The point of most of these is that you can derive a chip's
internals from two published numbers and a guess about its geometry.

\`\`\`
peak_flops   = n_mxu x dim x dim x 2 x clock      # a systolic array, from geometry
ridge        = dense_flops / bandwidth
array_eff(m) = m / (m + dim)                      # pipeline fill, per weight tile
allreduce_B  = 2 x batch x seq x d_model x 2 x (N-1)/N     # Module 9, per GPU
\`\`\`

1. A TPU v4 chip has 2 TensorCores of 4 MXUs each, and each MXU is 128×128. Given the published
   275 TFLOP/s of dense bf16, what clock does that imply? Do the same for v5e (1 TensorCore,
   4 MXUs, 197 TFLOP/s) and v5p (2 TensorCores, 459 TFLOP/s). Then try v6e, which publishes
   ~918 TFLOP/s — assume the same 8 × 128×128 geometry and say what the answer tells you.
2. Compute the ridge point for the H100, v4, v5e and v5p. Which part is best balanced for decode,
   and what does "best balanced" actually mean here?
3. Llama-3-70B in fp16 is 141.2 GB of weights. For an H100 (80 GB), a v5e (16 GB) and a v5p
   (95 GB): how many chips before the model fits at all, and what is the per-token decode floor at
   that minimum count? Then compute the floor at 8 chips of each.
4. Using \`array_eff(m) = m / (m + 128)\`: what fraction of peak does an MXU reach at m = 1, 8, 128,
   512 and 2048 tokens? What m do you need for 90%? Compare that number to the v5p's ridge point
   from question 2 and say which constraint binds first.
5. An H100 has 228 KB of shared memory per SM across 132 SMs; a v5p has roughly 128 MiB of VMEM.
   Normalise both to bytes of programmer-visible on-chip staging per TFLOP/s of dense bf16. What is
   the ratio, and what does it predict about how each machine's attention kernels are written?
6. Your server averages 257 tokens per step. On a machine whose GEMM tiles the batch dimension in
   units of 128, how much arithmetic do you actually pay for? Separately: GPT-2's vocabulary is
   50,257 — how much does padding it to a multiple of 128 add, and why is the measured speedup from
   doing so far larger than that percentage?
7. Llama-3-70B, decode, batch 32, \`d_model\` 8192, 80 layers, fp16. Using Module 9's formula,
   compute the total all-reduce bytes per token at TP-8 and TP-64, then the time over NVLink
   (900 GB/s), over aggregate ICI (600 GB/s), and over InfiniBand (50 GB/s). Compare each against
   the ~5.3 ms decode step from question 3. What does this say about where TP-64 is reasonable?
8. Which of the numbers above would change if the workload were prefill instead of decode, and
   which would not?`,

    solution: `**1. Backing the clock out of the geometry**

Each MXU retires \`128 × 128 = 16,384\` MACs per cycle, and a MAC is 2 FLOP:

\`\`\`
v4:   275e12 / (8 x 16384 x 2)  =  1.049 GHz
v5e:  197e12 / (4 x 16384 x 2)  =  1.503 GHz
v5p:  459e12 / (8 x 16384 x 2)  =  1.751 GHz
\`\`\`

The v4 answer is the interesting one: its documented clock is 1,050 MHz, so the model closes to
three digits. That is a real check, not a coincidence — it confirms the MXU count and array size.

\`\`\`
v6e:  918e12 / (8 x 16384 x 2)  =  3.502 GHz
\`\`\`

3.5 GHz is not a plausible clock for a datacenter accelerator, so the assumption must be wrong:
the v6e cannot have the same 8 × 128×128 geometry. Either it has more MXUs or larger ones. This is
the most useful thing back-of-envelope arithmetic does — not confirming what you believed, but
telling you precisely which belief is broken.

**2. Ridge points**

\`\`\`
H100:  989e12 / 3.35e12  =  295
v4:    275e12 / 1.20e12  =  229
v5e:   197e12 / 0.819e12 =  241
v5p:   459e12 / 2.765e12 =  166
\`\`\`

The v5p is the best-balanced part in this set: it needs an arithmetic intensity of 166 to saturate,
where the H100 needs 295. Since decode intensity equals batch size, that is a statement that the
v5p reaches its own ceiling at a batch of 166 while the H100 needs 295 to reach its. It says
nothing about which ceiling is higher — the H100's is 2.2× the v5p's. A low ridge point means
"easy to saturate," not "fast."

**3. How many chips before it runs, and how fast then**

\`\`\`
weights = 70.6e9 x 2 = 141.2 GB

H100 (80 GB):  ceil(141.2/80) = 2 chips   ->  70.6 GB per chip / 3.35 TB/s  = 21.07 ms  (47 tok/s)
v5e  (16 GB):  ceil(141.2/16) = 9 chips   ->  15.7 GB per chip / 0.819 TB/s = 19.16 ms  (52 tok/s)
v5p  (95 GB):  ceil(141.2/95) = 2 chips   ->  70.6 GB per chip / 2.765 TB/s = 25.53 ms  (39 tok/s)

at 8 chips:
H100:  17.65 GB / 3.35 TB/s  =  5.27 ms   (190 tok/s)
v5e:   17.65 GB / 0.819 TB/s = 21.55 ms   ( 46 tok/s)
v5p:   17.65 GB / 2.765 TB/s =  6.38 ms   (157 tok/s)
\`\`\`

Two things to notice. First, the minimum-fit configuration is nobody's answer — you shard past the
capacity requirement because sharding buys aggregate bandwidth, which is the actual currency of
decode. Second, the v5e needs 9 chips before it can run at all and is still slower on 8 than an
H100 is on 8: it is a small, cheap, throughput-oriented part, and a 70B model in fp16 is simply not
what it is for. That is a statement about part selection, not about TPUs.

**4. Array fill**

\`\`\`
m =    1:  1/129    =   0.8%
m =    8:  8/136    =   5.9%
m =  128:  128/256  =  50.0%
m =  512:  512/640  =  80.0%
m = 2048:  2048/2176 = 94.1%

90%:  m/(m+128) = 0.9  ->  m = 9 x 128 = 1,152 tokens
\`\`\`

Set that against the v5p's ridge point of 166. To be *memory-bound-free* you need a batch of 166;
to run the array near its peak you need a batch of 1,152. Under this (deliberately pessimistic,
un-pipelined) model the geometry binds long after the roofline does — so on a systolic machine the
question "what batch do I need?" has two answers and you must take the larger. Real MXUs overlap
the fill of one weight tile with the drain of the previous one, which pulls the second number down
substantially; the code lab leaves that as an exercise precisely because how much it helps is the
whole argument about how much the fill really costs.

**5. Scratchpad per unit of compute**

\`\`\`
H100:  228 KB x 132       =  30.8 MB  ->  30.8e6 / 989   =  31.1 KB per TFLOP/s
v5p:   128 MiB            = 134.2 MB  -> 134.2e6 / 459   = 292.4 KB per TFLOP/s
ratio: 9.4x
\`\`\`

The prediction is exactly what the two ecosystems look like. On a GPU, fitting attention into
228 KB per SM required an algorithm — tile, recompute the softmax online, never materialise the
N×N matrix — and that algorithm has a name and a citation. On a TPU, keeping a layer's activations
on chip across several fused operations is a scheduling decision the compiler makes, because there
is room. Same idea, one is a paper and one is a pass.

**6. Padding, twice**

\`\`\`
257 tokens -> ceil(257/128) x 128 = 384 tiles' worth of arithmetic
wasted: 127/384 = 33.1%
\`\`\`

A batch of 257 costs exactly what a batch of 384 costs. Sizing your server's step to land just past
a tile boundary is one of the easiest large wins available, and one of the easiest to miss because
nothing reports it.

\`\`\`
50,257 -> 50,304 (= 128 x 393), extra 47 columns = 0.093% more LM-head FLOPs
\`\`\`

The padding itself is negligible — yet padding the vocabulary is a well-known and *large* speedup.
The reason is that the extra 0.093% is not what you are buying. An unaligned dimension pushes the
matmul off the library's fast path: it selects a different, slower kernel, or falls back to one that
does not use the tensor cores efficiently at all. You are not saving the 0.093%; you are buying
back the fast kernel. Whenever a tiny alignment change produces a large speedup, that is the shape
of the explanation.

**7. Collectives at TP-8 and TP-64**

\`\`\`
per all-reduce, TP-8:   2 x 32 x 1 x 8192 x 2 x (7/8)   = 0.918 MB
per token:              x 2 all-reduces x 80 layers      = 146.8 MB

per all-reduce, TP-64:  2 x 32 x 1 x 8192 x 2 x (63/64)  = 1.032 MB
per token:                                                = 165.2 MB
\`\`\`

\`\`\`
TP-8,  NVLink   900 GB/s:  0.163 ms   ~3%  of a 5.27 ms step   free
TP-64, ICI      600 GB/s:  0.275 ms   ~5%                      free
TP-64, InfiniBand 50 GB/s: 3.30 ms    ~63%                     fatal
\`\`\`

TP-64 is unremarkable on a torus and unusable across a commodity network, and the reason is not
link speed — a single NVLink domain is faster per link than ICI. It is *reach*: the TPU's fast tier
extends to the whole pod, so "how many chips can I shard a layer across?" has a different answer on
each machine. Note also that this compares bandwidth only; ring all-reduce latency grows with hop
count, so the torus's diameter shows up as a per-collective fixed cost the bandwidth model here
ignores — one more reason the real crossover is workload-specific.

**8. What changes at prefill**

The ridge points (2) do not change — they are properties of the chip. The array-fill numbers (4)
change completely and in the TPU's favour: prefill processes thousands of tokens at once, so
\`m\` is large, \`array_eff\` is near 1, and the geometry penalty that dominates decode disappears.
The decode floors (3) are replaced by a compute-bound calculation entirely. The collective volumes
(7) scale with the number of tokens in the batch, so they grow by three orders of magnitude and
stop being free — which is why prefill is where interconnect actually gets tested. In short:
**decode measures your memory system, prefill measures your arithmetic and your network.** The two
chips are nearly identical under the first measurement and quite different under the second, which
is the entire content of this level in one sentence.`,
  },

  codeLab: {
    goal: `Build a two-machine roofline calculator that models the one thing GPUs and TPUs
genuinely differ on in the arithmetic path — how a matmul's shape maps onto the units — and use it
to reproduce this level's central claim: that in the decode regime the difference is invisible,
because both machines are waiting on HBM. No GPU, no TPU, no dependencies. Datasheet numbers in,
ridge points and step times out.`,
    code: `"""Two accelerators, one roofline: an SIMT GPU and a systolic-array TPU.

    python3 accelerators.py        # standard library only

The point is not to simulate silicon. It is to show that the two designs
differ in exactly one place that matters for inference -- how a matmul's
shape maps onto the arithmetic units -- and that in the decode regime these
chapters are about, that difference is invisible, because both machines
are waiting on HBM either way.

Peak FLOPs and bandwidth are datasheet figures (verify them). Everything
else is derived: the clock is backed out of the array geometry, and the
efficiency models are the two shape-quantization effects each design has.
"""

from math import ceil

# ---- the model: Llama-3-8B, the same one used in every chapter here ----
N_PARAMS = 8.03e9
DTYPE_B = 2
W_BYTES = N_PARAMS * DTYPE_B          # 16.06 GB read once per forward pass
FLOPS_PER_TOKEN = 2 * N_PARAMS        # 16.06 GFLOP
D_MODEL = 4096


class Machine:
    """peak: dense bf16 FLOP/s.  bw: HBM bytes/s.  hbm: bytes of HBM."""

    def __init__(self, name, peak, bw, hbm, kind, **kw):
        self.name, self.peak, self.bw, self.hbm, self.kind = name, peak, bw, hbm, kind
        self.__dict__.update(kw)

    @property
    def ridge(self):
        """FLOP per byte at which the machine stops being memory-bound."""
        return self.peak / self.bw

    def clock(self):
        """Back the clock out of the array geometry. A 128x128 MXU retires
        128*128 multiply-accumulates -- 2 FLOP each -- every cycle."""
        if self.kind != "systolic":
            return None
        return self.peak / (self.n_mxu * self.dim * self.dim * 2)

    def efficiency(self, m, n):
        """Fraction of peak a single [m, k] x [k, n] matmul can reach, from
        shape quantization alone. Both effects below are real and neither is
        a flaw: they are what you pay for the structure that makes the
        arithmetic cheap in the first place."""
        if self.kind == "systolic":
            # Weight-stationary array: rows stream in one per cycle, and the
            # first result leaves the array only after the wavefront has
            # crossed all \`dim\` of its rows. That fill is paid per weight
            # tile, and nothing about a small m makes it cheaper.
            return m / (m + self.dim)
        # SIMT: the output is cut into tiles handed to independent SMs. A
        # batch of m < tile_m pads its tile and the padding is computed for
        # nothing; the tile count itself is kept a multiple of the SM count
        # by splitting K, which is why there is no wave term here.
        return m / (ceil(m / self.tile) * self.tile)

    def decode_step(self, batch):
        """Time for one decode step: weights stream from HBM, arithmetic runs
        on whatever fraction of peak the shape allows. Whichever is slower is
        the step time -- that is the roofline, applied per machine."""
        t_mem = W_BYTES / self.bw
        eff = self.efficiency(batch, D_MODEL)
        t_math = FLOPS_PER_TOKEN * batch / (self.peak * eff)
        return max(t_mem, t_math), t_mem, t_math, eff


MACHINES = [
    Machine("H100 SXM", 989e12, 3.35e12, 80e9, "simt", units=132, tile=128),
    Machine("TPU v5e", 197e12, 0.819e12, 16e9, "systolic", n_mxu=4, dim=128),
    Machine("TPU v5p", 459e12, 2.765e12, 95e9, "systolic", n_mxu=8, dim=128),
]

if __name__ == "__main__":
    print("=== Part 1: the two numbers that decide everything ===\\n")
    print(f"  {'machine':<12} {'HBM':>7} {'bandwidth':>11} {'dense bf16':>12} {'ridge':>7} "
          f"{'units':>9} {'clock':>8}")
    for m in MACHINES:
        clk = m.clock()
        units = f"{m.units} SMs" if m.kind == "simt" else f"{m.n_mxu} MXUs"
        print(f"  {m.name:<12} {m.hbm/1e9:6.0f}G {m.bw/1e12:10.3f}T {m.peak/1e12:11.0f}T "
              f"{m.ridge:7.0f} {units:>9} {(f'{clk/1e9:.2f} GHz' if clk else '-'):>8}")
    print("\\n  ridge = dense FLOP/s / bandwidth = the arithmetic intensity, and so the")
    print("  batch size, at which the machine stops being memory-bound.")

    print("\\n=== Part 2: shape quantization -- what each design wastes ===\\n")
    print(f"  {'batch':>6} {'H100 eff':>10} {'v5p eff':>9}   what is being wasted")
    for b in (1, 8, 32, 64, 128, 256, 512, 1024, 2048):
        gpu = MACHINES[0].efficiency(b, D_MODEL)
        tpu = MACHINES[2].efficiency(b, D_MODEL)
        note = "padding / array fill" if b < 512 else "both converging"
        print(f"  {b:6d} {gpu*100:9.1f}% {tpu*100:8.1f}%   {note}")
    print("\\n  The GPU's loss is a step function -- it is zero once m is a multiple")
    print("  of 128. The array's is a hyperbola: it never quite reaches 100%.")

    print("\\n=== Part 3: one decode step, Llama-3-8B fp16, one chip ===\\n")
    for m in MACHINES:
        if m.hbm < W_BYTES:
            print(f"  {m.name}: 16.06 GB of weights do not fit in {m.hbm/1e9:.0f} GB "
                  f"-- needs {ceil(W_BYTES/(m.hbm*0.8))} chips before it can run at all\\n")
            continue
        print(f"  {m.name}")
        print(f"    {'batch':>6} {'t_mem':>9} {'t_math':>9} {'step':>9} {'tok/s':>9} {'bound by':>10}")
        for b in (1, 32, 128, 256, 512, 1024, 2048):
            t, tm, tc, eff = m.decode_step(b)
            print(f"    {b:6d} {tm*1e3:8.2f}m {tc*1e3:8.2f}m {t*1e3:8.2f}m "
                  f"{b/t:9.0f} {'HBM' if tm >= tc else 'arithmetic':>10}")
        print()

    print("=== Part 4: where each machine turns the corner ===\\n")
    for m in MACHINES:
        b = 1
        while b < 65536:
            t, tm, tc, eff = m.decode_step(b)
            if tc > tm:
                break
            b += 1
        eff_at = m.efficiency(b, D_MODEL)
        print(f"  {m.name:<12} compute-bound from batch {b:5d}  "
              f"(ideal ridge {m.ridge:5.0f}, shape efficiency there {eff_at*100:5.1f}%)")
    print("\\n  The gap between 'ideal ridge' and the batch you actually need is the")
    print("  price of shape quantization -- and it is paid in the same currency on")
    print("  both machines, just for different structural reasons.")

# --- TODO for you ----------------------------------------------------------
#   1. Add TPU v5e as a 4-chip tensor-parallel group: divide W_BYTES and
#      peak by 4, then add an all-reduce of [batch, 4096] bf16 per layer at
#      an ICI bandwidth of ~90 GB/s. At what batch does the collective stop
#      being free?
#   2. Model int8 weights: halve W_BYTES and double peak. Which machine
#      gains more, and why is the answer "the one whose ridge point moved"?
#   3. Replace the systolic efficiency model with a double-buffered one that
#      hides the fill of tile i+1 behind the drain of tile i. How much of
#      the batch-128 gap does that close?
`,
    expect: `Part 1 reproduces the table from the first concept and adds the derived clocks —
1.50 GHz for the v5e and 1.75 GHz for the v5p, backed out of nothing but the MXU count and the
published peak. Part 2 is the one to read carefully:

\`\`\`
   batch   H100 eff   v5p eff
       1       0.8%      0.8%
      32      25.0%     20.0%
     128     100.0%     50.0%
     512     100.0%     80.0%
    2048     100.0%     94.1%
\`\`\`

Two different curves from two different causes. The GPU's loss is padding, so it is a sawtooth that
hits exactly 100% whenever the batch is a multiple of the tile — and 0% waste is genuinely
reachable. The array's loss is pipeline fill, so it is a hyperbola that approaches 100% and never
arrives.

Part 3 is the level's thesis, printed:

\`\`\`
  H100 SXM
     batch     t_mem    t_math      step     tok/s   bound by
         1     4.79m     2.08m     4.79m       209        HBM
       128     4.79m     2.08m     4.79m     26700        HBM
       256     4.79m     4.16m     4.79m     53400        HBM
       512     4.79m     8.31m     8.31m     61582 arithmetic

  TPU v5e: 16.06 GB of weights do not fit in 16 GB -- needs 2 chips before it can run at all

  TPU v5p
         1     5.81m     4.51m     5.81m       172        HBM
        32     5.81m     5.60m     5.81m      5509        HBM
       128     5.81m     8.96m     8.96m     14290 arithmetic
\`\`\`

At batch 1 both machines are HBM-bound and the step time is simply \`16.06 GB / bandwidth\`: 4.79 ms
against 5.81 ms, a 1.21× gap that is exactly the bandwidth ratio and has nothing to do with
anything else in this level. Every architectural difference you just spent five concepts on
contributes precisely zero to that row.

They diverge as batch grows, and not in the direction the peak-FLOPs column suggests. The v5p turns
compute-bound at batch 128 while the H100 is still riding its memory ceiling at 256 — because the
array's fill penalty is a *tax on effective peak*, and a lower effective peak means an earlier
corner. Part 4 makes that explicit: the H100 turns at batch 257 against an ideal ridge of 295, the
v5p at batch 39 against an ideal ridge of 166. Turning early is not a virtue here; it means the
ceiling you hit is lower than the one on the datasheet.

Treat the v5p rows as a pessimistic bound rather than a prediction — TODO 3 is where you find out
how much of that gap double-buffering closes, and the honest answer is "most of it, for large
matmuls." If your numbers differ, check that \`efficiency\` divides by \`ceil(m/tile)*tile\` for the
SIMT machine and by \`(m + dim)\` for the systolic one; swapping those two models is the one mistake
that still produces plausible-looking output.`,
    stretch: `Add a fourth machine that does not exist: same 3.35 TB/s of bandwidth as an H100, but
ten times the peak FLOPs. Re-run every part. Watch the ridge point go to 2,950 and the decode rows
not move at all — that is the memory wall in one experiment, and it is why "10× the FLOPs" in a
launch keynote should be read as a claim about training.

Then do the reverse and the more useful one: hold peak fixed and double bandwidth, the H100→H200
change. Notice which parts of the output move. The gap between those two experiments is the whole
argument of these chapters, expressed as a diff.

Finally, implement TODO 1 and extend it: shard Llama-3-70B across 8 chips of each machine, with the
per-layer all-reduce priced at NVLink bandwidth for the GPU and ICI bandwidth for the TPU, and then
across 64. The 8-chip answers will be close. The 64-chip answers will not be, and the reason will
be the fabric rather than the chip — which is the one place in this level where the two designs
produce genuinely different serving architectures.`,
  },

  papers: [
    {
      title: 'In-Datacenter Performance Analysis of a Tensor Processing Unit',
      by: 'Jouppi et al., ISCA 2017',
      url: 'https://arxiv.org/abs/1704.04760',
      why: 'The original TPU paper, and still the clearest statement of why a systolic array is the right answer to a specific question.',
      frame: `Read Section 3 for the systolic array itself, then read the roofline analysis in
Section 6 as the ancestor of Module 4's — this is where the "your workload is memory-bound and
your accelerator is not the problem" argument was made for inference, years before LLMs. Note that
the TPU v1 was an *inference* chip with no training support at all, and that its authors spend
much of the paper arguing about memory bandwidth rather than FLOPs. The field rediscovered that
argument in 2022.`,
    },
    {
      title: 'TPU v4: An Optically Reconfigurable Supercomputer with Hardware Support for Embeddings',
      by: 'Jouppi et al., ISCA 2023',
      url: 'https://arxiv.org/abs/2304.01433',
      why: 'The fabric half of this level. Optical circuit switches, the 3D torus, and what "no bandwidth cliff" costs to build.',
      frame: `Read for the interconnect, not the chip. The argument to extract is why a
reconfigurable topology matters operationally — jobs get the shape of mesh they want, failed cubes
are routed around instead of downing a pod — and then compare that to the NVLink-domain model in
Module 9. Both are answers to "how far does my fast tier reach?", and reading them together is
more useful than reading either alone.`,
    },
    {
      title: 'Efficiently Scaling Transformer Inference',
      by: 'Pope et al., MLSys 2023',
      url: 'https://arxiv.org/abs/2211.05102',
      why: 'The best worked example of inference partitioning on TPUs, and the paper that makes the multi-dimensional sharding trade-offs concrete.',
      frame: `This is Module 9's material re-derived on a torus, and the contrast is the lesson:
because the fast fabric reaches the whole pod, the partitioning search space is larger and the
paper spends its effort choosing among layouts rather than avoiding a cliff. Their per-regime
analysis of which sharding wins at which batch size and context length is directly transferable to
GPUs — the arithmetic does not care which chip it is on, only which link.`,
    },
    {
      title: 'Benchmarking and Dissecting the Nvidia Hopper GPU Architecture',
      by: 'Luo et al., 2024',
      url: 'https://arxiv.org/abs/2402.13499',
      why: 'What an H100 actually does, measured rather than claimed — latencies, tensor core throughput, TMA and the memory hierarchy.',
      frame: `Read this as the empirical counterweight to the vendor whitepaper. The numbers to
carry away are the achieved latencies and bandwidths at each level of the hierarchy, since those
are what Module 7's kernels are written against. Notice how much of the paper is about
*asynchrony* — TMA, warp specialization, clusters — which is the modern GPU's answer to the same
problem the TPU solves by compiling a static schedule.`,
    },
    {
      title: 'GSPMD: General and Scalable Parallelization for ML Computation Graphs',
      by: 'Xu et al., 2021',
      url: 'https://arxiv.org/abs/2105.04663',
      why: 'The compiler half of this level: how XLA turns per-tensor sharding annotations into a partitioned program, and why static shapes are the price.',
      frame: `Read for the mental model of ahead-of-time partitioning — you annotate tensors, the
compiler infers the rest and inserts the collectives — and then ask what that model demands of the
serving loop. Every dynamic-shape workload in Modules 5, 6 and 12 is in tension with this design,
and the bucketing strategy in this level's fifth concept is the reconciliation. The paper will
also make it obvious why TPU parallelism is described in terms of mesh axes rather than TP and PP
degrees.`,
    },
  ],

  checkpoint: {
    claim: `You can read any accelerator datasheet and predict its inference behaviour: compute
the ridge point, work out the batch size the arithmetic geometry demands, identify how far the
fast interconnect reaches, and say which of those three binds first for a given workload — and you
can explain why almost none of it matters at batch 1.`,
    questions: [
      {
        q: 'A colleague benchmarks single-stream decode of an 8B model on an H100 and a TPU v5p, gets 4.8 ms and 5.8 ms per token, and concludes the GPU\'s tensor cores beat the systolic array. What is wrong with the conclusion?',
        a: `The measurement is right and the explanation is unrelated to it. At batch 1 both
machines read all 16.06 GB of weights once per token and do about 16 GFLOP with them — an
arithmetic intensity of 1, which is a hundred-plus times below either chip's ridge point. Both are
sitting on the memory-bound slope of their roofline, so the step time is \`weight_bytes /
bandwidth\` on each: 16.06/3.35 = 4.79 ms and 16.06/2.765 = 5.81 ms. The ratio of the two
measurements, 1.21, is exactly the ratio of the bandwidths. Neither machine's matrix units were
meaningfully involved in the result; you could have replaced either chip's arithmetic with
something 10× slower and measured the same numbers. The benchmark is a bandwidth test wearing a
compute test's clothes, and the conclusion attaches the outcome to the one component that had
nothing to do with it.`,
      },
      {
        q: 'A TPU\'s MXU is a 128×128 systolic array. Give two distinct consequences of that geometry for inference, one about batch size and one about model architecture.',
        a: `**Batch size:** the array has a pipeline to fill. Results only start emerging after the
wavefront has crossed all 128 rows, so streaming \`m\` rows through a weight tile costs roughly
\`m + 128\` cycles for \`m\` rows of work — 50% of peak at m = 128, 80% at 512, and under 1% at
m = 1. That gives a systolic machine a second, independent reason to want a large batch, on top of
the roofline reason every machine has. The batch you need is the larger of the two, not the ridge
point alone.

**Model architecture:** 128 becomes a quantum. A tensor dimension that is not a multiple of 128
gets padded, and the padded lanes compute zeros at full cost and full energy. This has quietly
shaped model design — hidden sizes, head dimensions, intermediate widths and vocabularies in modern
architectures are nearly always multiples of 128, and the well-known trick of padding GPT-2's
50,257-token vocabulary to 50,304 is the same effect on the GPU side, where the tensor-core tiling
imposes its own alignment. The hardware's shape has propagated into the models.`,
      },
      {
        q: 'Why does a TPU need roughly nine times as much on-chip scratchpad per unit of compute as a GPU, and what does that predict about how attention kernels get written on each?',
        a: `Because it has nothing else. A GPU has 228 KB of shared memory per SM plus a 50 MB
hardware-managed L2 that rescues access patterns nobody planned for; a TPU has ~128 MiB of VMEM,
no cache, and a compiler that must place every byte ahead of time. Normalised to compute that is
about 31 KB per TFLOP/s versus about 292 KB per TFLOP/s. A statically scheduled machine cannot rely
on a cache to cover a mistake, so it needs enough explicit staging area that the schedule does not
have to be tight.

The prediction is borne out by what each ecosystem produced. On a GPU, fitting attention into a
228 KB scratchpad required an algorithm — tiling plus online softmax so the N×N matrix never
materialises — which is FlashAttention, a paper with a lineage of successors. On a TPU, holding a
layer's activations across several fused operations is just what the compiler does when it fuses a
subgraph, because there is room. The same underlying idea shows up as a research contribution on
one machine and a compiler pass on the other.`,
      },
      {
        q: 'You want tensor parallelism across 64 chips. Explain why that is routine on a TPU pod and usually a mistake on a GPU cluster, without saying "TPU links are faster".',
        a: `Because they are not faster — NVLink is faster per link than ICI. The difference is
reach. A GPU's fast fabric is an island: 900 GB/s to every peer inside an 8-GPU NVLink domain (72
on an NVL72 rack), and then a cliff to roughly 50 GB/s per GPU on InfiniBand the moment you cross
the boundary. TP-64 on 8-GPU nodes puts most of the all-reduce traffic across that cliff. A TPU pod
has no cliff: each chip has direct ICI links to its neighbours in a torus at about 100 GB/s per
link, and that is the bandwidth whether the pod is 64 chips or 8,960.

The arithmetic makes it concrete. Llama-3-70B at batch 32 moves about 165 MB per token of
all-reduce traffic at TP-64: 0.28 ms at aggregate ICI bandwidth against a ~5 ms decode step, versus
3.30 ms over InfiniBand — about 63% of the step spent communicating. Same model, same parallelism
degree, same collective volume; the only variable is how far the fast tier extends. And note that
NVL72 is NVIDIA moving the cliff rather than removing it, which is the same insight approached from
the other side.`,
      },
      {
        q: 'Your team runs a continuous-batching server on GPUs and is evaluating a TPU port. What specifically gets harder, and what is the standard answer?',
        a: `Everything that makes continuous batching what it is. The batch composition changes on
every iteration, sequence lengths differ per request, and paged attention gathers KV blocks through
a table computed at runtime. A GPU absorbs all of that: kernel launches are runtime events, the
hardware distributes blocks to SMs as they free, and a batch of 1 and a batch of 137 use the same
binary. XLA compiles a static schedule against known shapes, so each distinct shape is a distinct
program, compiled on first use and cached under a key that includes those shapes. Left alone, a
continuous-batching server would trigger recompilation constantly.

The standard answer is **bucketing**: quantize the dynamic dimensions — batch to 8/16/32/64,
context to powers of two — pad up to the nearest bucket, and compile one program per bucket, so
unbounded shape variety becomes a handful of cached executables. The cost is computing on padding,
which is the same waste the 128-quantum imposes inside the matmul, now applied at the level of the
serving loop. The related discipline is keeping host-side per-step decisions out of the inner loop.
What is *not* true is that any of this is impossible: paged attention and continuous batching exist
on TPUs — vLLM has a TPU backend, and Pallas exists for kernels the compiler will not generate. The
cost is compilation time, padded compute and engineering effort, not capability.`,
      },
    ],
  },

  pitfalls: [
    {
      wrong: 'TPUs are faster than GPUs for inference (or the reverse).',
      right: `The question is malformed until you say which regime. At batch 1 the comparison is
purely a bandwidth comparison — both chips read every weight once per token and both sit on the
memory-bound slope, so the winner is whichever part has the higher HBM bandwidth, and the margin is
exactly the bandwidth ratio. At large batch the comparison becomes a compute comparison, where peak
FLOPs and shape quantization decide it. At scale it becomes an interconnect comparison, where reach
matters more than link speed. These three regimes can and do have different winners for the same
pair of chips, which is why a single "X is faster" claim is almost always a claim about one
unstated workload. Ask which regime, then compute the relevant ratio — it is one division either
way.`,
    },
    {
      wrong: 'A GPU\'s FLOPs come from its thousands of CUDA cores.',
      right: `They come from the tensor cores, and it is not close. An H100's non-tensor FP32
throughput is roughly 67 TFLOP/s against 989 TFLOP/s of dense BF16 on the tensor cores — so a
kernel that does its matmuls on the general-purpose lanes is using something like 7% of the
machine. This matters practically rather than trivially: it is why an unaligned tensor dimension
that pushes a matmul onto a fallback path can cost far more than the padding arithmetic suggests,
why "the GPU is at 100% utilization" (as reported by \`nvidia-smi\`, which measures whether *any*
kernel is resident) tells you nothing about whether the tensor cores did anything, and why
achieved-FLOPs is the only utilization number worth quoting. The CUDA cores are there for
everything that is not a matmul — and in a transformer, that is softmax, normalisation and
activations, not the bulk of the work.`,
    },
    {
      wrong: 'A systolic array can only do matmuls, so TPUs must be bad at softmax and normalization.',
      right: `The MXU only does matmuls; the TensorCore around it also has a VPU — a
(8, 128)-shaped vector lane array — and a scalar unit, precisely because a transformer is not only
matmuls. Softmax, RMSNorm, SwiGLU's gating and the residual adds all run there, and they run fine.
The real thing to notice is more interesting than the myth: because the MXU is so fast relative to
everything else, the *non-matmul* work is disproportionately likely to be what limits a kernel — the
same phenomenon Module 7 describes on GPUs, where attention's exponentials and reductions can cost
more than its matmuls at short sequence lengths. This is a general property of any accelerator with
a big matrix unit bolted to a modest vector unit, not a TPU quirk, and it is why operator fusion
matters so much on both machines.`,
    },
    {
      wrong: 'TPUs cannot do PagedAttention or continuous batching, because they need static shapes.',
      right: `They can, and both exist in production TPU serving stacks — vLLM has a TPU backend,
and JAX/Pallas exists for exactly the kernels XLA will not generate on its own. What static shapes
change is the *cost structure*, not the capability. Dynamic dimensions get bucketed and padded, each
bucket is a separately compiled program, and the padding is computed at full price; per-step
host-side decisions are more expensive than on a GPU because they interrupt a precomputed schedule.
So the honest statement is that a dynamic-shape workload on a static-shape machine pays in
compilation, padding and engineering effort. Overstating this into "impossible" leads teams to
dismiss a platform on architecture-astronomy grounds instead of measuring; understating it leads
them to port a GPU serving loop unchanged and wonder why it is slow.`,
    },
    {
      wrong: 'Compare accelerators by peak FLOPs.',
      right: `Peak FLOPs is the least informative number on the datasheet for inference, and the
most likely to be inflated. Vendors headline the 2:4 structured-sparsity figure, which is double
the dense number and does not apply to production LLMs; they quote FP8 or INT8 where the comparison
part is quoted in BF16; and they compare a full chip against a competitor's single die. Meanwhile
the number that actually sets your decode latency — HBM bandwidth — is stable, hard to game, and
usually printed two rows below. Compute the ridge point (\`dense FLOP/s ÷ bandwidth\`) and you have
collapsed both numbers into the one that says something: the batch size at which the machine stops
being memory-bound. A part whose ridge point went up between generations got *worse* at decode
relative to its own peak, however impressive the headline is — which is exactly what has happened
across most of the last decade on both sides.`,
    },
  ],

  glossary: [
    { term: 'SIMT', def: 'Single instruction, multiple threads — the GPU execution model where a warp of 32 threads issues one instruction together, with divergent branches serialized under lane masks.' },
    { term: 'streaming multiprocessor (SM)', def: 'The GPU\'s independent execution unit: warp schedulers, registers, shared memory and tensor cores. An H100 SXM has 132 of them.' },
    { term: 'warp', def: 'The 32-thread group a GPU schedules as a unit. Hopper adds the warpgroup — four warps issuing one wgmma matrix instruction together.' },
    { term: 'occupancy', def: 'How many warps are resident on an SM relative to the maximum. It is the GPU\'s mechanism for hiding memory latency: with nothing else resident, a stall is an idle cycle.' },
    { term: 'tensor core', def: 'The matrix-multiply unit inside an SM, where essentially all of a GPU\'s FLOPs live. Not a systolic array in the TPU sense, despite the name.' },
    { term: 'TMA (Tensor Memory Accelerator)', def: 'Hopper\'s asynchronous DMA engine for moving tiles between global and shared memory without spending warps on address arithmetic.' },
    { term: 'systolic array', def: 'A mesh of multiply-accumulate cells wired to their neighbours, through which operands flow and partial sums accumulate. One memory read feeds many multiplications, which is where its energy advantage comes from.' },
    { term: 'MXU', def: 'A TPU\'s Matrix Multiply Unit: a 128×128 weight-stationary systolic array. A v5p TensorCore has four; a chip has two TensorCores.' },
    { term: 'weight-stationary', def: 'A dataflow in which a tile of weights is loaded into the array and held while activations stream through, rather than both operands moving.' },
    { term: 'pipeline fill (array)', def: 'The cycles before a systolic array\'s first result emerges, while the wavefront crosses it. Costs a matmul of m rows roughly m/(m+128) of peak on a 128-wide array.' },
    { term: 'VPU', def: 'The TPU\'s vector unit, an (8, 128) lane array that handles the elementwise work — softmax, normalization, activations, residual adds — that the MXU cannot.' },
    { term: 'VMEM', def: 'A TPU\'s on-chip scratchpad, roughly 128 MiB on recent generations, entirely managed by the compiler. There is no cache behind it.' },
    { term: 'XLA', def: 'The compiler that turns a JAX or TensorFlow graph into a statically scheduled TPU program. It specializes on tensor shapes, which is why shape variety costs compilations.' },
    { term: 'shape bucketing', def: 'Quantizing dynamic dimensions (batch, context length) to a small set of values and padding up, so a serving loop needs only a few cached compiled programs instead of one per distinct shape.' },
    { term: 'ICI (Inter-Chip Interconnect)', def: 'The direct chip-to-chip links forming a TPU pod\'s 2D or 3D torus. Around 100 GB/s per link on a v5p, and the same bandwidth whether the pod has 64 chips or thousands.' },
    { term: 'optical circuit switch (OCS)', def: 'The reconfigurable optical fabric connecting TPU v4 cubes, allowing a job\'s topology to be chosen at schedule time and failed cubes to be routed around.' },
  ],
};
