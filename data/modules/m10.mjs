export default {
  n: 10,
  slug: 'systems-and-frontier',
  title: 'SYSTEMS AND THE FRONTIER',
  tagline: 'How the engines differ, how to benchmark without lying to yourself, and what is still unsolved.',
  hours: '6–8 hours',
  prereqs: ['All previous modules'],

  bigIdea: `You now have the analytical machinery. This module is about using it on real systems
and real measurements — and about knowing where the machinery runs out.

Three things.

**Engine design philosophy.** vLLM, SGLang, TensorRT-LLM and llama.cpp have converged on the same
feature list, which means features no longer distinguish them. What distinguishes them is what
each was built to optimize and what it gave up. Understanding that tells you which one fits your
problem better than any benchmark table will.

**Benchmarking honestly.** Almost every published LLM inference benchmark is misleading, usually
not deliberately. The failure modes are systematic and learnable: wrong load model, means instead
of percentiles, no SLO, unrepresentative length distributions, and comparing systems at different
operating points. You should be able to look at a benchmark and say what it does not tell you.

**The frontier.** What is genuinely open. Long context is still unaffordable, reasoning models
inverted the workload, speculation and batching fight each other, and we still cannot evaluate a
quantized model honestly. Knowing the open problems is how you tell a real advance from a
repackaged one.`,

  concepts: [
    {
      name: 'What actually distinguishes the serving engines',
      keyPoint: 'Every engine now implements the same techniques, so the real differences are in what each optimizes for and what it deliberately gives up.',
      body: `Start with the convergence, because it is the more informative fact. Every serious
engine now has: continuous batching, paged KV cache, prefix caching, chunked prefill, CUDA graphs,
fused kernels, FlashAttention-family kernels, weight quantization, increasingly KV quantization,
and speculative decoding.

**Those are table stakes, not differentiators.** Four independently designed systems converging on
the same list is good evidence the list is correct — and it means a feature-comparison table tells
you almost nothing.

What differs is the organizing bet.

**vLLM — memory management as the foundation.** The thesis is that if you fix KV fragmentation,
everything else follows: recovered memory becomes batch size, which becomes arithmetic intensity,
which becomes throughput. PagedAttention drove the whole architecture, and continuous batching
depends on it. The V1 engine rewrite moved scheduling out of the Python hot path, which was the
main remaining overhead. What it gives up: peak single-request latency, and some raw kernel
performance relative to a compiled engine, because it optimizes for generality across a very large
model zoo.

**SGLang — the workload is structured, not a stream of independent requests.** Real traffic shares
system prompts, few-shot examples, documents, and conversation history. RadixAttention makes
prefix reuse the default rather than a special case, and the frontend language lets you express
branching, parallel calls and constrained generation as a program. Structured output is
first-class, not a wrapper. What it gives up: simplicity, and some generality — the benefits
concentrate in workloads that genuinely share prefixes.

**TensorRT-LLM — decide everything at build time.** The bet is that the last 20–30% lives in kernel
selection, fusion and layout choices that are only correct for a specific model on specific
hardware at a specific batch size, so make them ahead of time with full knowledge of the
deployment. Typically the fastest option on NVIDIA hardware with the tightest tail latency. What it
gives up: a great deal of flexibility. The engine is tied to the GPU it was built for; changing
max batch size or context length means recompiling; new architectures land later.

**llama.cpp — the weight-dominated regime, taken seriously.** One user, short context, not enough
memory. That is Module 2's weight-dominated regime, where weight bytes are essentially the whole
decode cost — so aggressive weight quantization is not one optimization among many, it is *the*
optimization. Hence the k-quant family, CPU inference, partial GPU offload, and a zero-dependency
build. What it gives up: throughput at scale. Serving hundreds of concurrent users is not the
design centre.

The pattern worth extracting: **each engine is right about a different operating point.** Most
arguments about which is "best" are two people in different regimes talking past each other — the
same failure mode as the weight-versus-KV-quantization argument in Module 6. Ask what batch size
and context length someone is running before taking their recommendation.`,
      ascii: '',
    },
    {
      name: 'How to benchmark without misleading yourself',
      keyPoint: 'Closed-loop load testing, means instead of percentiles, and no SLO are the three errors that make most published numbers uninterpretable.',
      body: `**Error 1: closed-loop load generation.** The most common and most damaging. A
closed-loop harness keeps \`N\` requests in flight, sending a new one only when an old one finishes.
That is not how users behave, and it has a specific pathology: **it cannot show you overload.** If
the system slows down, the harness sends fewer requests, so the queue never grows and latency
degrades gracefully — hiding exactly the behaviour you needed to see.

Real traffic is **open-loop**: requests arrive on their own schedule, typically Poisson, whether
or not you are ready. Under open-loop load a system past its capacity shows queue growth and
latency blowup, which is the honest picture. Benchmark open-loop, sweep the arrival rate, and find
where it breaks.

**Error 2: means instead of percentiles.** Serving latency distributions are heavily right-skewed.
A mean TPOT of 30 ms is entirely compatible with a p99 of 400 ms — preempted requests, long
prefills, unlucky scheduling. Users experience the tail. Report p50, p95, p99, and report them for
TTFT and TPOT separately, since they have different causes.

**Error 3: no SLO, so no goodput.** Throughput without a latency constraint is maximized by an
absurd batch size that violates every latency target. Define the SLO first — "TTFT under 500 ms
and p95 TPOT under 50 ms" — then measure **goodput**: requests per second that met it. That number
is what you are actually selling. A system at 20,000 tok/s aggregate with 80 ms TPOT against a
50 ms budget has excellent throughput and zero goodput.

**Error 4: unrepresentative length distributions.** Results depend enormously on input and output
lengths, and fixed-length synthetic workloads (every request exactly 512 in, 128 out) produce
numbers that do not transfer. Real distributions are heavy-tailed, and the tail is what stresses
the scheduler. Use a real trace if you have one; failing that, a lognormal with a long tail.

**Error 5: no warmup.** First requests pay CUDA context creation, kernel autotuning, CUDA graph
capture, memory pool growth, and cold prefix caches. Discard the first 10–20% of measurements.

**Error 6: comparing at different operating points.** Engine A at batch 256 versus engine B at
batch 32 is not a comparison of engines. Fix the SLO, let each system choose its own best
configuration under that constraint, and compare goodput. That is the only comparison that
answers a real question.

A checklist worth keeping:

\`\`\`
[ ] open-loop arrivals, Poisson, sweep the rate
[ ] realistic length distribution, heavy-tailed
[ ] warmup discarded
[ ] p50 / p95 / p99 for TTFT and TPOT separately
[ ] an explicit SLO, and goodput measured against it
[ ] each system tuned for the SLO, not copied settings
[ ] the same model, precision, and hardware -- state all three
[ ] enough duration to reach steady state
\`\`\`

And when reading someone else's benchmark, the fastest tell: **if it does not state batch size,
context length distribution, and percentiles, it is not a measurement, it is a claim.**`,
      ascii: `  CLOSED-LOOP (misleading)        OPEN-LOOP (honest)

  keep 64 in flight               arrivals ~ Poisson(rate)
  finish one -> send one          send regardless of state

  system slows -> fewer sent      system slows -> queue GROWS
  -> latency degrades gently      -> latency blows up
  -> you never find the cliff     -> you find the cliff

  reported: "64 concurrent,       reported: "meets SLO up to
  p50 42ms, looks fine"           18 req/s, collapses at 22"`,
    },
    {
      name: 'Evaluating optimized models',
      keyPoint: 'Quantization damage concentrates in long context, rare tokens and multi-step reasoning — exactly what perplexity and multiple-choice benchmarks average away.',
      body: `Module 6 made this point; it deserves expanding, because it is the weakest link in
most production optimization work.

**Why perplexity fails.** It is the average negative log-likelihood over a corpus, dominated by
common tokens in ordinary contexts — precisely what a quantized model reproduces almost perfectly.
Damage concentrates in the tail. A 4-bit model can match fp16 perplexity to two decimals and fail
visibly at 20-step arithmetic. Worse, perplexity is usually measured at 2k–4k tokens, exactly the
regime where KV quantization looks harmless.

**Why MMLU-style benchmarks are weak too.** Multiple choice with four options is forgiving: the
model needs the right answer to be marginally more likely than three alternatives, not to be
confidently right. Degradation that would be obvious in open generation is invisible. Useful as a
regression check, weak as a quality signal.

**What to measure**, in rough order of information per unit effort:

1. **Long-context retrieval.** RULER is the good one — multi-needle, aggregation, variable
   tracking, at configurable lengths. Needle-in-a-haystack is easier and less informative but
   better than nothing. This is where KV quantization and eviction damage shows first.
2. **Multi-step reasoning.** GSM8K, MATH, or your own chain-of-thought tasks. Errors compound, so
   small per-token degradation becomes visible.
3. **Your actual task.** A few hundred real prompts, graded against the fp16 baseline by a human
   or a strong judge model. Unglamorous, and the most informative item here.
4. **Distributional divergence.** KL between the optimized model's logits and the baseline's on a
   fixed prompt set. Cheap, sensitive, and catches problems before task metrics move.
5. **Standard suites.** As a regression gate only.

**Control the comparison.** Same prompts, same sampling parameters, same seed. Evaluating a
quantized model at different settings than the baseline measures nothing, and it happens
constantly.

**Test at your operating point.** If you serve 32k contexts at batch 64, evaluating at 2k context
and batch 1 tells you very little. Batch composition itself can shift outputs (Module 3), so
evaluate under batching.

**Watch for the specific failure shapes.** Quantization damage is not uniform noise. Common
patterns: degradation that appears only past some context length; failures concentrated on rare
tokens, numbers, and non-English text; loss of instruction-following precision while fluency is
preserved; and reasoning that stays plausible while becoming wrong. Fluent-but-wrong is the
dangerous one, because it survives casual inspection.

The honest position on most published quantization results is that they show the method does not
catastrophically fail — not that it is free. Whether it is free *for your workload* is a question
only your own evaluation answers.`,
      ascii: '',
    },
    {
      name: 'Reasoning models changed the workload',
      keyPoint: 'Long chain-of-thought moves nearly all wall-clock into decode, holds batch slots far longer, and makes TTFT almost irrelevant.',
      body: `The largest recent shift in inference is not a technique, it is a change in what the
workload looks like.

A conventional request might be 2,000 prompt tokens and 200 output tokens. A reasoning request is
2,000 prompt tokens and 20,000 output tokens, most of which are thinking the user never sees.

Every consequence follows from Module 1's asymmetry:

**Almost all wall-clock becomes decode.** Prefill was already a small fraction; now it is
negligible. For Llama-3-8B on an H100, 2k prefill is ~35 ms and 20,000 decode tokens are ~90
seconds. **Prefill is 0.04% of the request.** All optimization effort belongs on the memory-bound
side.

**TTFT stops mattering.** The user waits 90 seconds regardless; whether the first thinking token
appears at 35 ms or 300 ms is irrelevant. This weakens the case for chunked prefill and
strengthens the case for large prefill batches — the opposite of the conventional-chat conclusion.

**Batch slots are held far longer.** A sequence occupying a slot for 20,000 steps instead of 200
means 100× less slot turnover. Continuous batching's advantage over static batching shrinks,
because there is less variance to exploit. Admission control matters more, because a bad admission
decision costs you for minutes.

**KV cache grows much larger per request.** 20,000 tokens of cache per sequence rather than 2,200.
At 128 KiB/token that is 2.5 GiB per sequence for an 8B model — so concurrency falls sharply, and
every technique in Module 6 becomes more valuable.

**Speculation becomes more attractive.** Thinking tokens are often highly predictable —
self-consistent reasoning has a lot of structure — and since the user never sees them, per-token
latency is irrelevant; only total time matters. That combination favours aggressive speculation
even at moderate batch, and n-gram lookup can do well on repetitive reasoning patterns.

**New questions with no settled answers.** Should thinking tokens use a cheaper decode path, or a
smaller model, or more aggressive quantization, given they are never displayed? Can you detect a
reasoning trace that is going nowhere and prune it mid-flight? Should reasoning requests be
scheduled on a separate pool given their radically different profile? Is there a principled way to
trade thinking length against answer quality at serve time?

The honest summary: the field's serving infrastructure was designed for a workload that no longer
describes a growing share of traffic, and the adaptation is very much in progress.`,
      ascii: `  CONVENTIONAL CHAT              REASONING
  prompt  2,000 tokens          prompt   2,000 tokens
  output    200 tokens          output  20,000 tokens (mostly hidden)

  prefill    35 ms  (15%)       prefill     35 ms  (0.04%)
  decode    900 ms  (85%)       decode  90,000 ms  (99.96%)

  slot held for 200 steps       slot held for 20,000 steps
  KV grows to 275 MB            KV grows to 2.5 GB
  TTFT matters                  TTFT is noise
  speculation marginal          speculation strongly favoured`,
    },
    {
      name: 'The open problems',
      keyPoint: 'Long context, evaluation, adaptive speculation, agentic cache policy and the widening memory wall are the areas where no approach has clearly won.',
      body: `Where the machinery in these chapters runs out.

**1. Long context is still fundamentally unaffordable.** The arithmetic does not improve with
cleverness: Llama-3-70B at 128k context needs 40 GiB of cache per sequence, which is three
concurrent users on four H100s. Every attack trades something real — eviction is lossy in ways you
cannot bound in advance, compression concentrates its damage on exactly the long-context tasks you
bought the context for, architectural fixes need training from scratch, and offload runs into a
50× bandwidth cliff. There is no approach that is simultaneously lossless,
architecture-agnostic and cheap.

**2. We cannot evaluate optimized models honestly.** Arguably the most important unglamorous
problem in the field. There is no standard, trusted suite that answers "is this quantized model
still the model I tested?", which means a great deal of production quantization is deployed on
faith and spot-checks.

**3. Speculation and batching fight each other.** Both spend the same idle arithmetic (Module 8),
so speculation stops paying around the batch size where you approach the ridge point. Adaptive
schemes — choosing draft length per request per step from current occupancy and observed
acceptance — are the obvious answer, and doing it well, with a scheduler that models the
interaction, is unsolved.

**4. Agentic workloads are a cache-policy problem nobody has solved.** An agent re-sends its whole
growing trajectory every step, so prefix caching turns quadratic prefill work into linear. That
makes the cache the working set of the system, and the hard questions become: what do you evict
when holding hundreds of partially-shared trajectories? Should you keep a trajectory warm while
its agent waits on a slow external tool call, paying memory for latency you do not control? How do
you schedule fairly when one agent's cache is worth 50× another's? Under-explored relative to its
importance.

**5. Disaggregation at scale is awkward in practice.** Clearly right in principle (Module 9), but
placement across heterogeneous pools, sizing the two pools as traffic mix shifts, and hiding the
KV transfer are all open engineering problems.

**6. The memory wall keeps widening.** Compute has grown faster than bandwidth for a decade — the
ridge point went from 153 on an A100 to 295 on an H100. Every generation makes decode relatively
worse. FP4 and FP8 help by shrinking bytes and larger HBM helps capacity, but the structural fix
would be a memory-centric architecture with far more bandwidth per FLOP, or compute placed nearer
memory. Several companies are betting on variants of this. Whether any displaces GPUs at scale is
genuinely uncertain, and it is the question that would most change everything else in these
chapters.

**7. The economics have no shared vocabulary.** Inference is now the dominant cost of operating a
model, and the field has no common language for cost-quality trade-offs. What is the right price
for a token that took 20,000 thinking tokens to produce? How do you expose the latency/throughput
dial to customers who do not know they want it? Not technical questions, but they increasingly
determine which technical work gets done.

*A note on currency: these chapters were written from the author's knowledge and not verified
against live sources at build time. Points 3, 4 and 6 move fastest. Before relying on any specific
claim, search for recent work — and if something here has been solved since, that is a good sign.*`,
      ascii: '',
    },
    {
      name: 'What to do next',
      keyPoint: 'Reproducing a published benchmark and explaining the discrepancy is the single most instructive exercise available.',
      body: `You have the framework. The way to convert it into understanding is to use it on
something real.

**Reproduce a published benchmark and account for the gap.** This is the highest-value exercise in
these chapters, and it is the one in the original outline for good reason. Pick a published number —
a vLLM blog post, an engine comparison, a model card's throughput claim. Reproduce it. You will
not match it. Then find out why.

The discrepancies are where the understanding is, and the list of causes is a summary of these
chapters in one list: different hardware SKU, different achieved bandwidth, different length
distribution, closed-loop versus open-loop load, mean versus percentile, different batch size,
warmup included, different precision, speculative decoding silently enabled, chunked prefill
settings, or a prefix-cache hit rate you did not have. Working through that list on a real
discrepancy will teach you more than re-reading any of these modules.

**Then, in rough order of value:**

**Instrument something you run.** Measure TTFT and TPOT percentiles on a real workload. Compute
the roofline floor. Calculate your gap. If it is over 2×, find out which item from Module 4's list
is responsible. This is a genuinely useful skill and few people have it.

**Read the code.** vLLM's block manager and scheduler are readable and are the clearest expression
of Modules 2, 5 and 6 in practice. Read \`v1/core/\` and follow one request through admission,
scheduling and eviction.

**Pick one module and go deeper than these chapters did.** Each of these has a substantial
literature behind it: KV compression, speculative decoding variants, MoE serving, long-context
attention, structured decoding. Depth in one is worth more than breadth across all.

**Build the small thing.** A KV cache calculator you actually use. A roofline planner. A batching
simulator. The labs in these chapters are starting points — the versions you extend for your own
hardware and models are the ones you will keep.

**Stay current deliberately.** This field moves fast enough that specific numbers age within
months. What does not age is the framework: bytes moved per unit of useful work, and which side of
the ridge point you are on. When you meet a new technique, ask what term of the cost model it
attacks and at what operating point. If the answer is not clear, the technique probably is not
either.`,
      ascii: '',
    },
  ],

  mathLab: {
    prompt: `This one is a design exercise rather than pure arithmetic. Show your reasoning with
numbers throughout.

**The brief.** You are asked to serve **Llama-3-70B** for two products on the same hardware
budget. You have **8× H100 80GB** in one NVLink node.

\`\`\`
Product A -- interactive chat
   prompt   ~1,500 tokens (lognormal, p99 ~8,000)
   output     ~300 tokens (lognormal, p99 ~1,500)
   SLO: TTFT p95 < 800 ms,  TPOT p95 < 40 ms
   load: 12 requests/second at peak

Product B -- document summarization
   prompt  ~24,000 tokens
   output     ~600 tokens
   SLO: end-to-end p95 < 60 s
   load: 0.4 requests/second
\`\`\`

**Part 1 — feasibility.** At fp16, weights are 141.2 GB against 640 GB of node memory. Compute
per-GPU weights, per-GPU KV per token, and available KV budget at TP-8. How many 1,500-token
sequences fit? How many 24,000-token ones?

**Part 2 — the latency budget.** At TP-8, aggregate bandwidth is 26.8 TB/s. Assume 70% achieved.
  a) TPOT floor at batch 32 with 2,000-token average context.
  b) Largest batch that keeps TPOT under the 40 ms SLO.
  c) TTFT for a 1,500-token prefill and for a 24,000-token prefill, at 989 TFLOP/s per GPU and 70%
     efficiency.
  d) Does the 24,000-token prefill fit inside Product A's 800 ms TTFT budget if it runs unchunked?
     What does that tell you?

**Part 3 — capacity.** Using Little's Law, what concurrency does Product A need at 12 req/s? What
does Product B need? Do they both fit in your answer to Part 1?

**Part 4 — the design.** Choose and justify, with numbers:
  a) One pool or two? Co-located or disaggregated?
  b) TP degree, and whether to replicate.
  c) Chunked prefill on or off, and at what token budget.
  d) KV precision.
  e) Speculative decoding on or off, and for which product.

**Part 5 — the benchmark.** Write the benchmark plan that would validate your design. What load
model, what distributions, what metrics, what would falsify it?`,

    solution: `**Part 1 — feasibility**

\`\`\`
TP-8, fp16:
  weights per GPU  = 141.2 / 8            = 17.65 GB
  KV per token per GPU (8 KV heads, TP-8, so 1 head each):
                   = 2 x 80 x 1 x 128 x 2 = 40,960 B = 40 KiB
  workspace        ~ 10 GB per GPU
  KV budget/GPU    = 80 x 0.95 - 17.65 - 10 = 48.35 GB
  node KV budget   = 48.35 x 8            = 386.8 GB
\`\`\`

Per sequence (using the full-model figure of 320 KiB/token, since the cache is split across GPUs
and the node total is what matters):

\`\`\`
1,500 tokens:  327,680 x 1,500  = 0.49 GB   ->  386.8 / 0.49  = 787 sequences
24,000 tokens: 327,680 x 24,000 = 7.86 GB   ->  386.8 / 7.86  =  49 sequences
\`\`\`

Plenty of headroom on paper. Memory is not the binding constraint here — which is unusual and
worth noticing.

**Part 2 — the latency budget**

Effective bandwidth: \`26,800 × 0.70 = 18,760 GB/s\`.

a) Batch 32, 2,000-token context:
\`\`\`
bytes = 141.2 GB (weights) + 327,680 x 2,000 x 32 = 141.2 + 21.0 = 162.2 GB
TPOT  = 162.2 / 18,760 = 8.6 ms
\`\`\`

Comfortably inside the 40 ms budget.

b) Solve for batch at 40 ms:
\`\`\`
bytes_at_40ms = 0.040 x 18,760 = 750.4 GB
750.4 = 141.2 + batch x 0.655 GB       (327,680 x 2,000 = 0.655 GB per sequence)
batch = (750.4 - 141.2) / 0.655 = 930
\`\`\`

But memory caps us at \`386.8 / 0.655 = 590\` sequences. **Memory binds before latency** — at
maximum batch you would be at \`(141.2 + 386.8)/18,760 = 28 ms\`, still inside the SLO.

c) Prefill, 8 GPUs at 989 TFLOP/s × 70% = 5,538 TFLOP/s aggregate:
\`\`\`
 1,500 tokens: 2 x 70.6e9 x 1,500  = 2.118e14 / 5.538e15 =  38 ms
24,000 tokens: 2 x 70.6e9 x 24,000 = 3.389e15 / 5.538e15 = 612 ms
\`\`\`

d) A 612 ms unchunked prefill against Product A's 800 ms TTFT budget: it *technically* fits for
the request itself. **But that is the wrong question.** The problem is what it does to everyone
else: for 612 ms, every decoding Product A sequence produces nothing. Their TPOT for that step is
612 ms against a 40 ms SLO — a **15× violation** — and at ~500 concurrent sequences that is
roughly 300 seconds of aggregate stall from a single Product B request.

At 0.4 req/s, a Product B request arrives every 2.5 seconds, so roughly **25% of all wall-clock
time** would be spent in a state where Product A is completely stalled. That is fatal, and it is
the central finding of this exercise.

**Part 3 — capacity via Little's Law**

\`\`\`
Product A: latency ~ TTFT + 300 x TPOT = 0.04 + 300 x 0.0086 = 2.62 s
           concurrency = 12 req/s x 2.62 s = 31 sequences

Product B: latency ~ 0.61 + 600 x 0.0086 = 5.8 s
           concurrency = 0.4 x 5.8 = 2.3 -> 3 sequences
\`\`\`

Total concurrency needed: about **34 sequences**, against capacity for hundreds. **You are not
capacity-constrained at all** — you are interference-constrained. That reframes the entire design
problem, and it is why doing Part 1 and Part 3 before Part 4 matters.

**Part 4 — the design**

**a) Two pools, disaggregated by product.** Not by prefill/decode — by *product*, which is the
sharper split here. Product B's 24k prefills are the entire problem and they must not touch
Product A's decode.

A reasonable allocation given the tiny concurrency requirements:

\`\`\`
6 GPUs -> Product A  (TP-6 is awkward; use 2 replicas of TP-4? see (b))
2 GPUs -> Product B
\`\`\`

But 2 GPUs cannot hold 141.2 GB of fp16 weights. So either quantize Product B's copy to fp8
(70.6 GB, fits in 2×80 GB with ~80 GB left for its 3-sequence KV requirement of 24 GB — workable),
or use a 4/4 split.

**Recommended: 4 GPUs each, TP-4, fp8 weights on both.** fp8 halves weights to 70.6 GB, so TP-4
gives 17.65 GB/GPU and leaves ample KV room. This also doubles decode speed.

**b) TP-4 per pool, two pools.** TP-8 in one pool would give lower latency per request, but we
established we are not latency-constrained — Product A's TPOT floor is 8.6 ms against a 40 ms
budget. Spending GPUs on latency we do not need, at the cost of leaving Product B co-located, is
the wrong trade. TP-4 also keeps both pools within NVLink and preserves the 8-KV-head split
cleanly.

**c) Chunked prefill: on for both, token budget 2,048.**

For Product A it bounds the p99 TTFT spike from the occasional 8,000-token prompt:
\`\`\`
8,000-token prefill unchunked at TP-4: 2 x 70.6e9 x 8000 / 2,769e12 = 408 ms
chunked at 2,048: 4 iterations, ~104 ms each -> worst ITL ~104 ms
\`\`\`
Still above the 40 ms TPOT SLO, so drop the budget to **1,024** for Product A: ~52 ms worst ITL,
close enough that with real overlap it lands inside budget. Measure and tune.

For Product B, chunking barely matters since it has its own pool, but leave it on at 2,048 to keep
its own concurrent requests from blocking each other.

**d) KV precision: fp16 for both.** We have enormous memory headroom — 386.8 GB against a need of
about 25 GB. **There is no reason to spend quality on memory we are not short of.** This is the
discipline from Module 6: do not reach for lossy optimizations to solve a problem you do not have.
Revisit if load grows 10×.

**e) Speculative decoding: on for Product A, off for Product B.**

Product A runs at concurrency ~31 per pool — well below the ridge point, so there is idle
arithmetic to spend. With EAGLE-style self-speculation (\`c ≈ 0.1\`, \`α ≈ 0.7\`) at \`γ = 4\`, expect
around 1.9×, taking TPOT from 8.6 ms to roughly 4.5 ms. Not needed for the SLO, but it buys
headroom for load growth.

Product B has concurrency 3 and is dominated by prefill; speculation would help its 600 decode
tokens marginally and adds complexity. Leave it off.

**Summary of the design:**

\`\`\`
Pool A: 4x H100, TP-4, fp8 weights, fp16 KV, chunked prefill @1024,
        EAGLE speculation, target batch ~32
Pool B: 4x H100, TP-4, fp8 weights, fp16 KV, chunked prefill @2048,
        no speculation, target batch ~3
\`\`\`

The single most important decision was **separating the pools**, and it came from Part 2(d) — not
from a memory calculation or a throughput calculation, but from noticing that one product's
prefill destroys the other product's SLO.

**Part 5 — the benchmark plan**

\`\`\`
LOAD MODEL
  open-loop, Poisson arrivals, two independent streams
  Product A: sweep 4 -> 24 req/s
  Product B: fixed 0.4 req/s, then sweep to 1.5 to find the breaking point
  run both streams SIMULTANEOUSLY -- testing them separately would
  miss the interference this design exists to prevent

DISTRIBUTIONS
  A: prompt lognormal median 1500 p99 8000; output lognormal median 300 p99 1500
  B: prompt normal around 24000 +/- 6000; output around 600
  ideally replayed from a real trace instead

PROCEDURE
  5 min warmup, discarded
  20 min steady state
  3 repetitions, report variance

METRICS
  per product: TTFT p50/p95/p99, TPOT p50/p95/p99, end-to-end p95
  GOODPUT: A-requests/s meeting (TTFT<800ms AND TPOT p95<40ms)
           B-requests/s meeting (e2e < 60 s)
  GPU utilization and KV occupancy per pool
  realized speculation acceptance rate on Pool A

WHAT WOULD FALSIFY THE DESIGN
  - A's TPOT p95 exceeds 40 ms at 12 req/s
      -> chunk budget too high, or batch too large; lower budget, re-measure
  - A's TPOT p99 spikes correlate in time with B's arrivals
      -> the pools are not actually isolated (shared NVLink? shared host?)
  - B's e2e p95 exceeds 60 s
      -> Pool B is under-provisioned; rebalance 5/3
  - KV occupancy above ~70% on either pool
      -> the fp16-KV decision was wrong; revisit fp8
  - speculation acceptance below 0.5 on real traffic
      -> speculation is near break-even; measure whether it still pays

CONTROL
  same model, same precision, same hardware, stated explicitly
  compare against a single-pool co-located baseline -- that comparison
  is the whole justification for the design
\`\`\`

That last line matters. A benchmark that only measures your chosen design tells you whether it
meets the SLO, not whether it was the right design. Always measure the alternative you rejected.`,
  },

  codeLab: {
    goal: `Build an open-loop load generator and a goodput measurement harness — the tool that
would let you actually run the benchmark plan from the math lab.

Then run it against the simulated serving model from Module 5 and reproduce the closed-loop
versus open-loop discrepancy for yourself. Seeing a closed-loop harness fail to detect overload
is the most convincing possible argument for open-loop testing.`,
    code: `"""
Open-loop load generation and goodput measurement.

The point: closed-loop harnesses cannot show you overload. This demonstrates it.

    pip install numpy
"""
import heapq
import numpy as np

rng = np.random.default_rng(23)

# --- a simple serving model (Llama-3-8B on one H100) ----------------------
WEIGHT_GB = 15.0
KV_GB_PER_TOKEN = 131072 / 1e9
BW = 3350 * 0.70
FLOPS = 989e12 * 0.70
N_PARAMS = 8.03e9
MAX_BATCH = 64
CHUNK_BUDGET = 2048


def decode_step_time(cached_tokens):
    return (WEIGHT_GB + cached_tokens * KV_GB_PER_TOKEN) / BW


def prefill_time(n_tokens):
    return (2 * N_PARAMS * n_tokens) / FLOPS


class Server:
    """Continuous batching with chunked prefill. Simplified but structurally honest."""

    def __init__(self, max_batch=MAX_BATCH, chunk=CHUNK_BUDGET):
        self.max_batch = max_batch
        self.chunk = chunk
        self.t = 0.0
        self.running = []          # decoding
        self.prefilling = []       # [tokens_left, req]
        self.queue = []            # admitted but not started
        self.done = []

    def submit(self, req):
        self.queue.append(req)

    def _admit(self):
        while (self.queue
               and len(self.running) + len(self.prefilling) < self.max_batch):
            r = self.queue.pop(0)
            self.prefilling.append([r["prompt"], r])

    def step(self):
        self._admit()
        if not self.running and not self.prefilling:
            return None

        decode_n = len(self.running)
        cached = sum(r["cached"] for r in self.running)
        step_t = decode_step_time(cached) if decode_n else 0.0

        if self.prefilling:
            take = min(self.prefilling[0][0], max(0, self.chunk - decode_n))
            if take > 0:
                step_t = max(step_t, prefill_time(take))
                self.prefilling[0][0] -= take
                if self.prefilling[0][0] <= 0:
                    _, r = self.prefilling.pop(0)
                    r["cached"] = r["prompt"]
                    r["ttft"] = self.t + step_t - r["arrival"]
                    self.running.append(r)

        if step_t <= 0:
            return None
        self.t += step_t

        for r in list(self.running):
            r["itls"].append(step_t)
            r["remaining"] -= 1
            r["cached"] += 1
            if r["remaining"] <= 0:
                r["finish"] = self.t
                self.done.append(r)
                self.running.remove(r)
        return step_t


def make_request(arrival, rng):
    return {
        "arrival": arrival,
        "prompt": int(np.clip(rng.lognormal(7.3, 0.9), 32, 32000)),
        "remaining": 0, "output": 0, "cached": 0,
        "ttft": None, "itls": [], "finish": None,
    }


def run_open_loop(rate, duration=120.0, seed=0):
    """Poisson arrivals at the given rate, regardless of server state."""
    r = np.random.default_rng(seed)
    srv = Server()

    # pre-generate the arrival schedule -- arrivals do NOT depend on the server
    arrivals, t = [], 0.0
    while t < duration:
        t += r.exponential(1.0 / rate)
        if t < duration:
            req = make_request(t, r)
            req["output"] = int(np.clip(r.lognormal(5.2, 1.0), 1, 3000))
            req["remaining"] = req["output"]
            arrivals.append(req)

    i = 0
    guard = 0
    while (i < len(arrivals) or srv.queue or srv.running or srv.prefilling):
        guard += 1
        if guard > 500_000:
            break
        while i < len(arrivals) and arrivals[i]["arrival"] <= srv.t:
            srv.submit(arrivals[i]); i += 1
        if srv.step() is None:
            if i < len(arrivals):
                srv.t = arrivals[i]["arrival"]
            else:
                break
    return srv.done, len(arrivals), srv.t


def run_closed_loop(concurrency, duration=120.0, seed=0):
    """Keep N requests in flight. Send a new one only on completion."""
    r = np.random.default_rng(seed)
    srv = Server()
    completed = []

    for _ in range(concurrency):
        req = make_request(0.0, r)
        req["output"] = int(np.clip(r.lognormal(5.2, 1.0), 1, 3000))
        req["remaining"] = req["output"]
        srv.submit(req)

    guard = 0
    while srv.t < duration:
        guard += 1
        if guard > 500_000:
            break
        n_before = len(srv.done)
        if srv.step() is None:
            break
        for r_done in srv.done[n_before:]:
            completed.append(r_done)
            new = make_request(srv.t, r)                 # replace immediately
            new["output"] = int(np.clip(r.lognormal(5.2, 1.0), 1, 3000))
            new["remaining"] = new["output"]
            srv.submit(new)
    return completed, srv.t


def report(done, wall, label, submitted=None,
           ttft_slo=1.0, tpot_slo=0.050):
    if not done:
        print(f"  {label}: no completions"); return
    ttft = np.array([r["ttft"] for r in done if r["ttft"] is not None])
    tpot = np.array([np.mean(r["itls"]) for r in done if r["itls"]])
    tpot_p95_each = np.array([np.percentile(r["itls"], 95) for r in done if r["itls"]])
    e2e = np.array([r["finish"] - r["arrival"] for r in done])
    out_tokens = sum(r["output"] for r in done)

    met = sum(1 for r in done
              if r["ttft"] is not None and r["ttft"] < ttft_slo
              and r["itls"] and np.percentile(r["itls"], 95) < tpot_slo)

    print(f"  {label}")
    print(f"    completed {len(done):>5}"
          + (f" / {submitted} submitted" if submitted else ""))
    print(f"    TTFT  p50 {np.percentile(ttft,50)*1000:>7.0f}ms  "
          f"p95 {np.percentile(ttft,95)*1000:>7.0f}ms  "
          f"p99 {np.percentile(ttft,99)*1000:>8.0f}ms")
    print(f"    TPOT  p50 {np.percentile(tpot,50)*1000:>7.1f}ms  "
          f"p95 {np.percentile(tpot_p95_each,95)*1000:>7.1f}ms")
    print(f"    e2e   p95 {np.percentile(e2e,95):>7.1f}s")
    print(f"    throughput {out_tokens/wall:>7.0f} tok/s   "
          f"GOODPUT {met/wall:>5.2f} req/s ({100*met/len(done):.0f}% met SLO)")


print("=" * 74)
print("OPEN-LOOP: sweep the arrival rate. Watch for the cliff.")
print("=" * 74)
for rate in (2, 4, 6, 8, 10, 14):
    done, submitted, wall = run_open_loop(rate, duration=90.0, seed=1)
    report(done, wall, f"arrival rate {rate} req/s", submitted)
    print()

print("=" * 74)
print("CLOSED-LOOP: fix concurrency. Note that latency degrades GENTLY")
print("and nothing ever looks broken -- the harness throttles itself.")
print("=" * 74)
for conc in (8, 16, 32, 64, 128):
    done, wall = run_closed_loop(conc, duration=90.0, seed=1)
    report(done, wall, f"concurrency {conc}")
    print()

print("=" * 74)
print("The open-loop sweep shows where the system BREAKS.")
print("The closed-loop sweep shows a smooth curve that never breaks,")
print("because it stops sending load exactly when the system slows down.")
print("Only one of these tells you your capacity.")
print("=" * 74)

# --- TODO for you ---
#   1. Add an SLO-aware admission controller: reject requests when the
#      predicted TTFT would violate the SLO. Does goodput improve?
#   2. Add a second traffic class with 24k prompts at 0.4 req/s and measure
#      what it does to the first class's TPOT p99. Then separate the pools
#      and measure again. That is the math lab's central finding, reproduced.
#   3. Report the FULL ITL distribution, not the mean per request. The tail
#      within a single request is what users perceive as stuttering.
`,
    expect: `The open-loop sweep should show a **cliff**. At low rates TTFT p95 is a few hundred
milliseconds and goodput tracks the arrival rate closely. Somewhere in the middle — the exact rate
depends on your sampled distributions — the queue starts growing faster than it drains, TTFT p95
jumps by an order of magnitude, and goodput *stops rising and then falls*. That inflection is your
capacity, and it is the single number the benchmark exists to find.

The closed-loop sweep shows something entirely different: a smooth, well-behaved curve. Latency
rises gradually with concurrency, throughput rises and flattens, and **nothing ever looks broken**
— because when the server slows down, the harness sends fewer requests. There is no cliff to find
because the load adapts to the system.

That contrast is the whole lesson. A closed-loop benchmark reporting "concurrency 128, p50 42 ms,
looks healthy" and an open-loop benchmark reporting "collapses above 9 req/s" can describe the
same system on the same hardware. Only the second answers "can I deploy this?"

Note also that closed-loop throughput numbers are often *higher* — which is why they get published.

The simulator is deliberately simplified: no KV memory limit, no preemption, an approximate
overlap model. It gets the shape right, not the absolute numbers.`,
    stretch: `Do TODO 2 — it reproduces the math lab's central finding empirically. Add a second
traffic class with 24,000-token prompts at 0.4 req/s and watch what it does to the first class's
TPOT p99. You should see the p99 spike badly even though the *mean* barely moves, which is exactly
why percentiles matter. Then run the two classes on separate simulated servers and measure again.
The difference between those two numbers is the entire justification for disaggregation, derived
from your own measurement rather than from a paper.`,
  },

  papers: [
    {
      title: 'DistServe: Disaggregating Prefill and Decoding for Goodput-optimized LLM Serving',
      by: 'Zhong et al., OSDI 2024',
      url: 'https://arxiv.org/abs/2401.09670',
      why: 'The clearest argument in the literature for goodput as the correct metric, with the measurement methodology to back it.',
      frame: 'Re-read **Section 2** specifically for the goodput definition and why SLO-constrained measurement changes conclusions. Their experimental setup in Section 6 is a good model for how to describe a benchmark honestly — note how much configuration detail they state.',
    },
    {
      title: 'Efficient Memory Management for LLM Serving with PagedAttention',
      by: 'Kwon et al., SOSP 2023',
      url: 'https://arxiv.org/abs/2309.06180',
      why: 'The vLLM paper. Worth a final re-read now that you have the full picture — the design decisions read differently once you understand what each is trading against.',
      frame: 'Read the whole thing again quickly. This time pay attention to what they chose *not* to do and why, and to the evaluation methodology: request rate sweeps, real trace-derived length distributions, and normalized latency. That is what a careful benchmark looks like.',
    },
    {
      title: 'SGLang: Efficient Execution of Structured Language Model Programs',
      by: 'Zheng et al., 2023',
      url: 'https://arxiv.org/abs/2312.07104',
      why: 'The strongest statement of the view that real workloads are structured and shared rather than independent — which is increasingly correct as agentic traffic grows.',
      frame: 'Focus on RadixAttention and the cache-aware scheduling policy. The scheduling interaction — that cache hit rate and batch size compete for the same memory — is the subtle part and the one most relevant to Module 6.',
    },
    {
      title: 'RULER: What\'s the Real Context Size of Your Long-Context Language Models?',
      by: 'Hsieh et al., 2024',
      url: 'https://arxiv.org/abs/2404.06654',
      why: 'The benchmark to use when evaluating anything that touches the KV cache. Shows that claimed context lengths substantially exceed effective ones.',
      frame: 'Read Sections 2 and 3 for the task categories — multi-needle retrieval, variable tracking, aggregation. The finding that models degrade well before their advertised context limit is directly relevant to evaluating KV quantization and eviction, and it is the benchmark to reach for instead of perplexity.',
    },
    {
      title: 'Taming Throughput-Latency Tradeoff in LLM Inference with Sarathi-Serve',
      by: 'Agrawal et al., OSDI 2024',
      url: 'https://arxiv.org/abs/2403.02310',
      why: 'Read alongside DistServe. Two careful papers, same diagnosed problem, opposite architectural conclusions — a good exercise in evaluating systems arguments.',
      frame: 'Compare the two directly: which workload assumptions favour chunked prefill over disaggregation, and vice versa? Neither is universally right, and working out the conditions under which each wins is more valuable than picking a side.',
    },
  ],

  checkpoint: {
    claim: `You can read a published inference benchmark and say what it does not tell you; you
can design a serving deployment from an SLO; and you can name what is genuinely unsolved rather
than merely unimplemented.`,
    questions: [
      {
        q: 'Why can a closed-loop benchmark never show you your capacity?',
        a: `Because the load adapts to the system. A closed-loop harness keeps \`N\` requests in
flight and only sends a new one when an old one completes — so if the server slows down, the
harness automatically sends fewer requests. The queue never grows, latency degrades smoothly, and
there is no cliff to find. Real traffic is open-loop: requests arrive on their own schedule
whether or not you are ready, so an overloaded system shows queue growth and latency blowup. The
same system can look "healthy at concurrency 128, p50 42 ms" under closed-loop testing and
"collapses above 9 req/s" under open-loop. Only the second answers whether you can deploy it.`,
      },
      {
        q: 'What is goodput, and why is optimizing throughput without it a mistake?',
        a: `Goodput counts only requests that met their latency SLO; throughput counts every token
delivered. They diverge because throughput rises monotonically with batch size while latency
degrades with it — so maximizing throughput alone reliably lands you at a configuration that
violates the SLO for everyone. A system reporting 20,000 tok/s at batch 256 with an 80 ms TPOT
against a 50 ms budget has excellent throughput and zero goodput: every token it produced is worth
nothing to the product. Stating the SLO first and maximizing goodput under it is the formulation
that matches what you are actually selling.`,
      },
      {
        q: 'Your quantized model matches fp16 perplexity to two decimals. What have you learned?',
        a: `Very little. Perplexity is the average negative log-likelihood over a corpus, dominated
by common tokens in ordinary contexts — exactly what a quantized model reproduces almost
perfectly. Damage concentrates in the tail: rare tokens, long dependencies, precise arithmetic,
multi-step reasoning where errors compound. It is also usually measured at 2k–4k sequence lengths,
which is precisely where KV quantization looks harmless. You would need long-context retrieval
(RULER), multi-step reasoning tasks, KL divergence against the baseline's logits, and a few
hundred prompts from your real workload graded against the fp16 baseline — all evaluated at your
actual serving context length and batch size.`,
      },
      {
        q: 'How do reasoning models change what you should optimize?',
        a: `They move essentially all wall-clock into decode. A 2,000-token prompt with 20,000
output tokens spends about 35 ms on prefill and 90 seconds on decode — prefill is 0.04% of the
request. Consequences: TTFT stops mattering, which weakens the case for chunked prefill; batch
slots are held 100× longer, so slot turnover falls and continuous batching's advantage over static
batching shrinks while admission control matters more; KV cache per sequence grows roughly 10×, so
concurrency collapses and every Module 6 technique becomes more valuable; and speculation becomes
more attractive, because thinking tokens are predictable and the user never sees them, so only
total time matters rather than per-token latency.`,
      },
      {
        q: 'A colleague says vLLM is better than TensorRT-LLM. What is the missing information?',
        a: `Their operating point, and what they are optimizing. The two are right about different
things. TensorRT-LLM compiles ahead of time for a specific model, GPU, precision and batch range,
so it typically wins on raw speed and tail-latency predictability on NVIDIA hardware — at the cost
of a compilation step, an engine tied to its GPU, and slower support for new architectures. vLLM
optimizes for generality and operational simplicity across a large model zoo, with memory
management as its organizing principle. If you serve one fixed model at scale and 25% matters,
TensorRT-LLM. If you serve many models, iterate quickly, or want to deploy today, vLLM. Since both
now implement the same technique list, a feature comparison answers nothing.`,
      },
      {
        q: 'Name three genuinely open problems and say why each is hard.',
        a: `**Long context affordability** — Llama-3-70B at 128k needs 40 GiB per sequence, and
every attack trades something real: eviction is lossy in ways you cannot bound because you cannot
know which token a future query needs; compression concentrates damage on the long-context tasks
you bought the context for; architectural fixes require training from scratch. **Evaluating
optimized models** — there is no trusted standard for "is this quantized model still the model I
tested?", because damage is non-uniform and concentrates in capabilities that are hardest to
measure, so much production quantization ships on faith. **Adaptive speculation under batching** —
speculation and batching spend the same idle arithmetic, so speculation stops paying near the
ridge point; choosing draft length per step from live occupancy and acceptance rate is the obvious
fix and nobody does it well, because it requires a scheduler that models the interaction.`,
      },
    ],
  },

  pitfalls: [
    {
      wrong: 'Benchmarking with a closed-loop load generator.',
      right: `It cannot show overload, because it throttles itself exactly when the system slows
down. Latency degrades smoothly, throughput flattens, and there is never a cliff — so you learn
nothing about capacity. Use open-loop Poisson arrivals and sweep the rate until goodput stops
rising. That inflection is the number you needed.`,
    },
    {
      wrong: 'Reporting mean latency.',
      right: `Serving latency distributions are heavily right-skewed: a 30 ms mean TPOT is
compatible with a 400 ms p99 caused by preemption, long prefills, or unlucky scheduling. Users
experience the tail, not the mean. Report p50, p95 and p99, separately for TTFT and TPOT since
they have different causes.`,
    },
    {
      wrong: 'Choosing an engine from a feature comparison table.',
      right: `Every serious engine now implements continuous batching, paged KV, prefix caching,
chunked prefill, CUDA graphs, quantization and speculation. Features no longer distinguish them.
What differs is the operating point each was designed for — and most disagreements about which is
best are two people in different regimes talking past each other.`,
    }, { wrong: 'Assuming a technique that helped someone else will help you.', right: `Almost every
optimization in these chapters is regime-dependent. Weight quantization is transformative at batch 1
and nearly useless at batch 64 with long context. Speculation is a 3× win at low concurrency and a
net loss at high. Chunked prefill matters for chat and barely matters for reasoning workloads.
Always ask what term of the cost model a technique attacks, and whether that term dominates *your*
deployment.`,
    },
  ],

  glossary: [
    { term: 'goodput', def: 'Requests per second that met their latency SLO. The metric that matches what a serving product actually delivers.' },
    { term: 'open-loop load', def: 'Arrivals follow an external schedule (usually Poisson) regardless of server state. The only way to observe overload.' },
    { term: 'closed-loop load', def: 'Fixed concurrency; a new request is sent only when one finishes. Self-throttling, so it hides the capacity cliff.' },
    { term: 'RULER', def: 'A long-context benchmark with multi-needle retrieval, variable tracking and aggregation. What to use instead of perplexity for KV compression.' },
    { term: 'vLLM', def: 'Serving engine organized around memory management: PagedAttention, continuous batching, broad model support.' },
    { term: 'SGLang', def: 'Serving engine organized around structured, prefix-sharing workloads: RadixAttention plus a frontend programming language.' },
    { term: 'TensorRT-LLM', def: 'NVIDIA serving engine that compiles a model into a hardware-specific engine ahead of time. Fastest and least flexible.' },
    { term: 'llama.cpp', def: 'Local-inference engine optimized for the weight-dominated regime: aggressive weight quantization, CPU and partial-offload support.' },
    { term: 'reasoning workload', def: 'Requests that generate thousands of hidden chain-of-thought tokens, shifting nearly all wall-clock into memory-bound decode.' },
  ],
};
