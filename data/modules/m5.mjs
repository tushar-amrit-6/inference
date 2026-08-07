export default {
  n: 5,
  slug: 'batching',
  title: 'BATCHING',
  tagline: 'The largest throughput lever in serving, and the scheduling problem it creates.',
  hours: '5–7 hours',
  prereqs: ['Module 1', 'Module 2', 'Module 4'],

  bigIdea: `Module 4 gave you the result that matters: **decode arithmetic intensity equals batch
size**. At batch 1 you are at 1 FLOP per byte on a machine that wants 295. At batch 64 you are at
64. The weights get read once and serve everyone.

So the throughput problem reduces to a scheduling problem: **keep the batch as full as possible,
at all times**. That sounds trivial and is not, because sequences have wildly different lengths
and nobody tells you in advance how long any of them will be. A batch assembled at time \`t\` is
mostly finished by time \`t + 200ms\`, and if you wait for the slowest member before starting
anything new, the GPU spends most of its life running a batch of two.

The fix — **continuous batching**, from the Orca paper — is one of those ideas that seems obvious
once stated and took years to arrive: make batching decisions at every *iteration* rather than
every *request*. A sequence that finishes leaves immediately; a waiting request takes its slot on
the next step.

The second half of this module is the complication that continuous batching creates. Prefill and
decode want the GPU for different reasons and cannot politely share it. One 32k-token prompt
arriving mid-stream will stall every other user's token stream unless you do something about it,
and **chunked prefill** is what you do.`,

  concepts: [
    {
      name: 'Static batching, and where the capacity goes',
      keyPoint: 'A static batch runs until its longest member finishes, so every shorter sequence leaves its slot idle for the remainder.',
      body: `The naive approach: collect \`B\` requests, run them together until all are done, then
collect the next \`B\`.

The problem is that generation lengths vary enormously and are not known in advance. Real traffic
has a heavy right tail — most responses are short, a few are very long. If your batch of 8 has
lengths \`[20, 35, 40, 60, 80, 120, 200, 900]\`, the batch runs for 900 steps. The sequence that
finished at step 20 occupies a slot doing nothing for the remaining 880.

Quantify it. Utilization is mean length over max length:

\`\`\`
total useful work  = 20+35+40+60+80+120+200+900 = 1455 sequence-steps
slots x steps      = 8 x 900                    = 7200 sequence-steps
utilization        = 1455 / 7200                = 20.2%
\`\`\`

**Four fifths of your paid-for capacity produced nothing.** And this is not a contrived example —
a lognormal length distribution with a realistic tail gives numbers in this range routinely. The
larger the batch, the worse it gets, because \`max\` over more samples reaches further into the
tail. Static batching is one of the few optimizations that gets *worse* as you scale it up.

There is a second cost that is easy to miss: **padding**. Sequences in a static batch must be
padded to a common length for the tensors to be rectangular. Attention over padding positions is
wasted arithmetic, and the padding also sits in the KV cache consuming memory that could have
held real sequences.

Orca's contribution begins with noticing that the padding problem and the early-finish problem
have the same root cause — treating a batch as a fixed rectangular object with a lifetime.`,
      ascii: `  STATIC BATCH, 8 slots, batch runs until the longest finishes

  slot 0  ####································································
  slot 1  ######······························································
  slot 2  #######·····························································
  slot 3  ##########··························································
  slot 4  #############·······················································
  slot 5  ####################································
  slot 6  #################################···································
  slot 7  ####################################################################

          ^ useful work        ^ idle slot, still allocated, still costing you

          utilization = 20.2%`,
    },
    {
      name: 'Continuous batching: decide every iteration, not every request',
      keyPoint: 'Evict finished sequences and admit waiting ones at every decode step, so the batch stays full regardless of length variance.',
      body: `Orca's insight is to move the scheduling granularity from the request to the
**iteration**. The loop becomes:

\`\`\`
loop forever:
    1. remove any sequence that emitted EOS or hit its length limit
    2. free its KV cache blocks
    3. admit waiting requests while memory allows
    4. run one forward step for the whole current batch
    5. sample one token for each sequence
\`\`\`

A sequence that finishes at step 20 releases its slot at step 20. Someone else takes it at step
21. The batch stays near capacity permanently, and utilization goes from ~20% to ~90%+ on the
same traffic.

Orca reported throughput improvements of up to **36.9×** over FasterTransformer at comparable
latency. That number is large partly because the baseline was weak, but the mechanism is real and
it is why every serious engine implements this.

There is a genuine technical obstacle, and Orca's solution to it is the part people skip.
**Different sequences in the batch are at different positions**, so their attention computations
have different shapes. You cannot batch them into one rectangular attention call the way you can
for a uniform batch.

Orca's answer is **selective batching**: batch the operations that permit it and split the one
that does not.

\`\`\`
QKV projections, MLP, LM head   ->  batch across sequences
                                     (these are position-independent; each
                                      token is an independent row of a matmul)

attention                       ->  handle per-sequence
                                     (each has a different KV length)
\`\`\`

This works because the position-independent operations are where nearly all the weight traffic
lives. Flatten all the batch's tokens into a single \`[total_tokens, d_model]\` matrix, do one big
matmul, and the weights are amortized across everyone. Attention is then handled with a kernel
that takes per-sequence lengths and offsets — which is exactly what PagedAttention and
FlashAttention's varlen variants provide.

Two costs worth knowing. First, admitting a request means running its **prefill**, which is a
different and much heavier operation than a decode step — the subject of the next two concepts.
Second, the scheduler now runs every iteration, so it must be fast; at 100 steps/second a
scheduler taking 2 ms is consuming 20% of your wall clock. This is why production schedulers are
C++ and why vLLM's V1 rewrite moved the scheduling loop out of the Python hot path.`,
      ascii: `  CONTINUOUS BATCHING, same 8 slots, same traffic

  slot 0  ####|######|##########|#############|####################|####
  slot 1  ######|#######|###################################|##########
  slot 2  #######|####################|#########|################|####
  slot 3  ##########|#############################|###############
  slot 4  #############|##########|#####################|############
  slot 5  ####################|###############|#####################
  slot 6  #################################|##########|##############
  slot 7  ####################################################################

          | = a sequence finished, its slot was refilled on the next step

          utilization > 90%`,
    },
    {
      name: 'Prefill/decode interference',
      keyPoint: 'One long prefill occupies the GPU for hundreds of milliseconds, during which every decoding sequence emits nothing.',
      body: `Continuous batching creates a problem it did not have to solve before: prefill and
decode now contend for the same GPU, iteration by iteration.

Numbers make it concrete. Llama-3-8B on an H100:

\`\`\`
one decode step, batch 32       ~10 ms
one prefill of 8,000 tokens    ~130 ms  (compute-bound: 2 x 8.03e9 x 8000 FLOP)
\`\`\`

If the scheduler runs that prefill as a single unit, then for 130 ms **every one of your 32
decoding sequences produces nothing**. Their inter-token latency for that step is 130 ms instead
of 10 ms — a 13× spike. To a user watching text stream, it is a visible stall.

Worse, this is not rare. Any traffic mix with both chat (short prompts, long outputs) and
document processing (long prompts, short outputs) produces it constantly. A single user pasting a
long document degrades the experience of everyone currently streaming.

The scheduling policies you can choose between all have a real cost:

**Prefill-priority.** Run prefills as soon as they arrive. Best TTFT, worst TPOT stability. This
is what naive continuous batching does by default.

**Decode-priority.** Never interrupt decoding; queue prefills until there is slack. Smooth token
streams, terrible TTFT under load, and requests can starve.

**Chunked prefill.** Split a long prefill into fixed-size chunks and process one chunk per
iteration alongside the ongoing decode. The next concept, and the answer most systems settled on.

**Disaggregation.** Put prefill and decode on separate hardware entirely so they cannot interfere.
The cleanest fix, and the subject of Module 9.

The underlying reason the conflict exists is the asymmetry from Module 1: prefill is
compute-bound and decode is memory-bound. They are not competing for the same resource, which
sounds like it should help, but they cannot run *simultaneously* on the same SMs — so in practice
they simply take turns, and the long one wins.`,
      ascii: `  UNCHUNKED: one 8k prefill blocks everyone

  time ->  |--10ms--|--10ms--|------------- 130ms -------------|--10ms--|
  decoders    tok      tok    (nothing for 130ms -- visible stall)  tok
  new req                     [======== prefill 8000 ========]

  CHUNKED: 512-token chunks share each iteration

  time ->  |-14ms-|-14ms-|-14ms-|-14ms-| ... |-14ms-|
  decoders   tok    tok    tok    tok         tok      (steady stream)
  new req    [c1]   [c2]   [c3]   [c4]  ...   [c16]    (TTFT slightly later)`,
    },
    {
      name: 'Chunked prefill and the token budget',
      keyPoint: 'Cap the tokens processed per iteration, then fill that budget with decode tokens first and prefill chunks second.',
      body: `Chunked prefill splits a long prompt into pieces and processes one piece per
iteration, mixed in with ongoing decode work. Since causal attention only looks backward, chunk
\`k\` can attend to all of chunks \`0..k-1\` from the cache — the result is identical to prefilling
the whole prompt at once.

The mechanism is a **token budget** per iteration. A scheduler with a budget of 2,048 tokens might
assemble:

\`\`\`
iteration budget: 2048 tokens

  32 decode sequences        ->    32 tokens
  1 prefill chunk            -> 2,016 tokens
                                 -------
                                 2,048
\`\`\`

Decode tokens are admitted first because they are latency-critical and cheap — 32 of them cost 32
of your 2,048. Prefill chunks fill the remainder. The result is that every iteration takes about
the same amount of time, so inter-token latency stays flat regardless of what prompts are
arriving.

The choice of budget is a real trade-off:

| budget | TTFT | TPOT stability | GPU efficiency |
|---|---|---|---|
| small (256) | worse — more iterations to finish a prefill | excellent | poor — too few tokens to saturate |
| medium (2048) | good | good | good |
| large (8192) | best | poor — long iterations | best |

There is also a subtle benefit that is easy to miss and is arguably the main one. A pure decode
iteration at batch 32 has arithmetic intensity 32 — deep in memory-bound territory, GPU mostly
idle. Adding 2,016 prefill tokens to that same iteration raises the intensity to roughly 2,048.
**The prefill work is nearly free**, because it uses the arithmetic capacity that the decode step
was wasting anyway.

That reframes chunked prefill entirely. It is not a scheduling compromise that trades TTFT for
TPOT. It is a way to run prefill in the compute the decode step was leaving on the floor — the
purest possible expression of this course's thesis. Both workloads finish faster than they would
if you ran them separately.

Sarathi-Serve, which introduced this technique as "stall-free batching", reported substantial
improvements in serving capacity under latency SLOs for exactly this reason.`,
      ascii: '',
    },
    {
      name: 'Scheduling policy: admission, preemption, and fairness',
      keyPoint: 'Because KV memory is finite and sequence lengths are unknown, the scheduler must be able to admit optimistically and take memory back.',
      body: `The scheduler decides, every iteration, who runs. It faces a problem with no clean
solution: **it does not know how long any sequence will be**, but it must commit memory to it
anyway.

**Admission control.** How many sequences to admit is a bet. Admit too few and you waste
throughput. Admit too many and you run out of KV cache mid-generation, because every admitted
sequence's cache grows every step. Some systems admit conservatively based on \`max_tokens\`;
others admit optimistically and deal with the consequences.

**Preemption** is how you deal with the consequences. When memory runs short, a running sequence
must give some back. Two options:

- **Swap** — copy its KV cache to CPU memory and restore it later. Costs a PCIe round trip:
  moving 2.5 GiB at ~50 GB/s is about 50 ms each way.
- **Recompute** — discard the cache and re-prefill from the tokens when the sequence resumes.
  Costs a prefill.

Which is cheaper depends on sequence length and interconnect, and vLLM implements both. The
crossover is roughly where prefill time exceeds transfer time — so short sequences favour
recomputation, long ones favour swapping.

**Queueing policy.** FCFS is the default and is fair in the obvious sense, but it lets one 100k
prompt delay everyone behind it. Shortest-job-first would be better for mean latency, but you
cannot see job length in advance — output length is unknown at admission time. Some systems
predict it; the predictions are not good.

**Priority and fairness.** A production system with multiple tenants needs to prevent one user
from occupying the whole batch. This is genuinely hard because the natural unit of fairness is
unclear: is it requests, tokens, or GPU-time? A user sending one 100k-token request and a user
sending a thousand 100-token requests consume comparable resources through very different shapes.

**The starvation trap.** Any policy that prioritizes decode over prefill can starve new requests
indefinitely under sustained load, because there is always more decode work. Real schedulers need
an aging mechanism — a request's priority rises with its wait time — or they will occasionally
never serve someone.

The honest summary is that scheduling here is a live research area with no settled answer, and
the defaults in every engine are heuristics tuned against a particular traffic assumption. If
your traffic looks different from that assumption, the defaults will be wrong for you, and
measuring is the only way to find out.`,
      ascii: '',
    },
    {
      name: 'How batching interacts with the roofline',
      keyPoint: 'Batching raises arithmetic intensity linearly until KV traffic takes over, and the knee of the throughput curve is where you should operate.',
      body: `Tie this module back to Module 4. Batching is the mechanism that moves you right
along the roofline, but the movement is bounded twice.

**Bound 1 — the ridge point.** Beyond intensity ≈ 295 on an H100, you are compute-bound and
further batching buys nothing per weight-byte. In practice you rarely get there.

**Bound 2 — the KV cache.** Every sequence brings its own cache, so batch is capped by memory —
and, more subtly, throughput saturates *before* the memory cap, because KV traffic grows with
batch while weight traffic does not.

The throughput curve therefore has a characteristic shape:

\`\`\`
throughput
    |                    ........................  <- flat: KV-traffic dominated
    |              .....
    |         ....
    |      ...
    |    ..                                        <- steep: weight-amortization region
    |  ..
    | .
    +--------------------------------------------> batch size
        ^                    ^
     the steep part      the knee -- operate here
\`\`\`

For Llama-3-8B on an H100 at 4k context (from Module 4's table), the knee is around batch 32–64:
going from 1 to 32 gains 15× throughput, and from 64 to 111 gains only 14%.

**Operating past the knee is usually a mistake.** You gain a few percent of throughput and pay
for it with substantially worse tail latency, more preemption, and less headroom for traffic
spikes. The right operating point is the knee, or wherever your latency SLO puts you if that is
tighter.

Two practical consequences:

**Measure the knee for your own workload.** It moves with model, context length, KV dtype and
hardware. A batch size copied from someone else's blog post is a guess.

**Anything that shrinks the KV cache moves the knee right.** fp8 KV, GQA, MLA, better paging —
each one lets the steep region continue further before KV traffic flattens it. This is why
Module 6 belongs immediately after this one: memory optimization is how you get more out of
batching, and batching is how memory optimization turns into throughput.`,
      ascii: '',
    },
  ],

  mathLab: {
    prompt: `**Part 1 — the cost of static batching.**
A batch of 8 requests generates \`[20, 35, 40, 60, 80, 120, 200, 900]\` tokens.

  a) Under static batching, how many iterations does the batch run for?
  b) How many sequence-steps of useful work are done?
  c) What is the slot utilization?
  d) Now suppose you use batch size 32 drawn from the same distribution, and the longest of the
     32 is 2,000 tokens with a mean of 180. What is the utilization? Which direction did it move,
     and why?

**Part 2 — continuous batching throughput.**
Llama-3-8B on an H100. A decode step at batch \`B\` takes \`(15.0 + B × 0.27) / 3350\` seconds,
where 15.0 GB is the weights and 0.27 GB is the KV cache for one sequence at 2k context.
Assume 70% achieved bandwidth.

  a) Compute TPOT and throughput at batch 1, 8, 32, 64, 128.
  b) Where is the knee — the batch beyond which you gain less than 10% throughput for a doubling?
  c) At the knee, what is the arithmetic intensity, and what fraction of the H100's ridge point
     (295) is that?

**Part 3 — chunked prefill.**
A request arrives with an 8,000-token prompt while 32 sequences are decoding.

  a) Unchunked: prefill costs \`2 × 8.03e9 × 8000\` FLOP at 989 TFLOP/s with 70% efficiency. How
     long, and what is the worst inter-token latency the 32 decoders see?
  b) Chunked at a 2,048-token budget: how many iterations does the prefill take? What is the
     per-iteration cost, and the worst inter-token latency now?
  c) Total time for the prefill to complete in each case. Which finishes the prompt sooner?
  d) Compute the arithmetic intensity of a pure decode iteration at batch 32, and of a mixed
     iteration with 32 decode tokens plus a 2,016-token prefill chunk. What does the comparison
     tell you?`,

    solution: `**Part 1**

a) The batch runs until the longest finishes: **900 iterations**.

b) \`20+35+40+60+80+120+200+900 = 1,455\` sequence-steps of useful work.

c)
\`\`\`
capacity = 8 slots x 900 iterations = 7,200 sequence-steps
utilization = 1455 / 7200 = 20.2%
\`\`\`

d)
\`\`\`
useful   = 32 x 180 = 5,760
capacity = 32 x 2000 = 64,000
utilization = 5760 / 64000 = 9.0%
\`\`\`

Utilization got **worse**, from 20.2% to 9.0%. The reason is that \`max\` over a larger sample
reaches further into the tail while the mean stays put, so the ratio mean/max falls as batch
grows. Static batching degrades as you scale it — which is close to the worst property an
optimization can have.

**Part 2**

Effective bandwidth: \`3350 × 0.70 = 2345 GB/s\`.

\`\`\`
B      bytes (GB)        TPOT (ms)   throughput (tok/s)   intensity
1      15.0 + 0.27 = 15.27   6.51            154              1
8      15.0 + 2.16 = 17.16   7.32          1,093              8
32     15.0 + 8.64 = 23.64  10.08          3,175             32
64     15.0 + 17.28 = 32.28 13.76          4,651             64
128    15.0 + 34.56 = 49.56 21.13          6,057            128
\`\`\`

Working one: at B=32, \`23.64 / 2345 = 0.01008 s = 10.08 ms\`, and \`32 / 0.01008 = 3,175 tok/s\`.

b) Gains per doubling:

\`\`\`
1  -> 8:     154 -> 1,093    +610%
8  -> 32:  1,093 -> 3,175    +190%
32 -> 64:  3,175 -> 4,651     +46%
64 -> 128: 4,651 -> 6,057     +30%
\`\`\`

No doubling in this range falls below 10%, so the strict knee is beyond 128 — but the returns are
clearly decaying, and the practical knee is around **64**, where you have captured most of the
available gain (77% of the throughput at batch 128) at 65% of the latency. Note also that batch
128 needs \`15.0 + 34.56 = 49.6\` GB, so on an 80 GB card you are approaching the memory limit as
well.

c) At batch 64: intensity **64**, which is \`64 / 295 = 22%\` of the ridge point. Even at the knee
you are using roughly a fifth of the machine's arithmetic capability. This is completely normal
and it is what Module 6 exists to improve.

**Part 3**

a) Unchunked:

\`\`\`
FLOPs   = 2 x 8.03e9 x 8000 = 1.285e14 = 128.5 TFLOP
rate    = 989e12 x 0.70 = 692 TFLOP/s
time    = 128.5 / 692 = 0.186 s = 186 ms
\`\`\`

The 32 decoders see an inter-token latency of **186 ms** for that step, against their usual
~10 ms. An **18.6× spike**, and a visible stall.

b) Chunked at 2,048:

\`\`\`
chunks = 8000 / 2048 = 3.9  ->  4 iterations
per-iteration prefill tokens ~ 2,016 (32 slots go to decode)
\`\`\`

Cost of one mixed iteration — take the larger of the compute and memory times:

\`\`\`
compute: 2 x 8.03e9 x 2048 = 3.29e13 FLOP / 692e12 = 47.5 ms
memory:  15.0 GB weights + KV / 2345 GB/s          = ~10 ms
-> compute-bound, ~47.5 ms per iteration
\`\`\`

Worst inter-token latency: **~48 ms**, versus 186 ms unchunked. Still a spike, but a 3.9×
improvement — and reducing the budget to 512 would bring it to ~13 ms at the cost of more
iterations.

c) Total prefill completion:

\`\`\`
unchunked:  186 ms
chunked:    4 x 47.5 = 190 ms
\`\`\`

Essentially identical — about 2% slower. **You get a 3.9× reduction in latency spike for a 2%
increase in TTFT.** That trade is why chunked prefill is on by default nearly everywhere.

d) Arithmetic intensity:

\`\`\`
pure decode, batch 32:
  FLOPs = 2 x 8.03e9 x 32 = 5.14e11
  bytes = 23.64e9
  I = 21.7 FLOP/byte          -> 7% of ridge, deeply memory-bound

mixed, 32 decode + 2016 prefill = 2048 tokens:
  FLOPs = 2 x 8.03e9 x 2048 = 3.29e13
  bytes = 23.64e9   (the same weights! prefill adds negligible bytes)
  I = 1,392 FLOP/byte         -> 4.7x PAST the ridge, compute-bound
\`\`\`

This is the result worth sitting with. The mixed iteration moves **the same bytes** as the pure
decode iteration — the weights are read once either way — but does **64× the arithmetic**. The
prefill work is riding along in compute capacity the decode step was already wasting.

So chunked prefill is not a compromise between TTFT and TPOT. It is a way to reclaim idle
arithmetic, and it makes both workloads faster than running them separately would. It is the
clearest example in the whole course of what "more useful work per byte moved" means in
practice.`,
  },

  codeLab: {
    goal: `Simulate static versus continuous batching over a realistic distribution of sequence
lengths. No GPU needed — this is a discrete-event simulation, and the point is to reproduce the
utilization numbers yourself rather than take them on faith.

Then add chunked prefill and measure what it does to the tail of the inter-token latency
distribution.`,
    code: `"""
Batching simulator: static vs continuous, then chunked prefill.
Pure Python + numpy. No GPU.

    pip install numpy matplotlib
"""
import heapq
import numpy as np

rng = np.random.default_rng(7)

# --- workload model -------------------------------------------------------
# Real traffic is heavy-tailed: most responses short, a few very long.
N_REQUESTS = 400
ARRIVAL_RATE = 12.0          # requests/second (Poisson)
MAX_BATCH = 32


def sample_workload(n):
    prompt = rng.lognormal(mean=6.2, sigma=1.1, size=n).astype(int).clip(16, 32000)
    output = rng.lognormal(mean=4.6, sigma=1.2, size=n).astype(int).clip(1, 4000)
    gaps = rng.exponential(1.0 / ARRIVAL_RATE, size=n)
    arrival = np.cumsum(gaps)
    return list(zip(arrival, prompt, output))


# --- cost model (Llama-3-8B on an H100, 70% achieved) ---------------------
WEIGHT_GB = 15.0
KV_GB_PER_TOKEN = 131072 / 1e9
BW = 3350 * 0.70                       # GB/s
FLOPS = 989e12 * 0.70                  # FLOP/s
N_PARAMS = 8.03e9


def decode_step_time(batch_tokens_cached):
    """Memory-bound: weights + all cached KV."""
    gb = WEIGHT_GB + batch_tokens_cached * KV_GB_PER_TOKEN
    return gb / BW


def prefill_time(n_tokens):
    """Compute-bound."""
    return (2 * N_PARAMS * n_tokens) / FLOPS


# ==========================================================================
def simulate_static(work, batch_size=MAX_BATCH):
    """Collect batch_size requests, run to completion, repeat."""
    t = 0.0
    i = 0
    useful = 0
    capacity = 0
    latencies = []

    while i < len(work):
        group = work[i:i + batch_size]
        i += len(group)
        t = max(t, group[-1][0])                       # wait for the last arrival

        for _, p, _o in group:                         # prefill each, unchunked
            t += prefill_time(p)

        longest = max(o for _, _, o in group)
        cached = sum(p for _, p, _ in group)
        for step in range(longest):
            alive = sum(1 for _, _, o in group if o > step)
            cached += alive
            t += decode_step_time(cached)
        for a, p, o in group:
            latencies.append(t - a)

        useful += sum(o for _, _, o in group)
        capacity += len(group) * longest

    return {"makespan": t, "utilization": useful / capacity,
            "throughput": sum(o for _, _, o in work) / t,
            "p50": np.percentile(latencies, 50), "p99": np.percentile(latencies, 99)}


# ==========================================================================
def simulate_continuous(work, max_batch=MAX_BATCH, chunk_budget=None):
    """Iteration-level scheduling. chunk_budget=None means unchunked prefill."""
    t = 0.0
    pending = list(work)
    running = []            # dicts: remaining output, cached tokens, arrival, ttft
    itl = []                # every inter-token gap seen by a decoding sequence
    done = []
    prefill_queue = []      # (tokens_left, request)

    total_out = sum(o for _, _, o in work)

    while pending or running or prefill_queue:
        # admit arrivals whose time has come, while slots remain
        while pending and pending[0][0] <= t and len(running) + len(prefill_queue) < max_batch:
            a, p, o = pending.pop(0)
            prefill_queue.append([p, {"arrival": a, "prompt": p, "out": o,
                                      "cached": 0, "ttft": None}])

        if not running and not prefill_queue:
            t = pending[0][0]                          # idle: jump to next arrival
            continue

        # --- assemble this iteration ---
        decode_tokens = len(running)
        cached = sum(r["cached"] for r in running)
        step_t = decode_step_time(cached) if running else 0.0

        prefill_tokens = 0
        if prefill_queue:
            want = prefill_queue[0][0]
            if chunk_budget is None:
                prefill_tokens = want                  # whole prompt, blocking
            else:
                prefill_tokens = min(want, max(0, chunk_budget - decode_tokens))
            if prefill_tokens > 0:
                # compute- and memory-bound paths run on the same SMs: take the max
                step_t = max(step_t, prefill_time(prefill_tokens))
                prefill_queue[0][0] -= prefill_tokens
                if prefill_queue[0][0] <= 0:
                    _, req = prefill_queue.pop(0)
                    req["cached"] = req["prompt"]
                    req["ttft"] = t + step_t - req["arrival"]
                    running.append(req)

        if step_t == 0:
            t += 1e-4
            continue

        t += step_t

        # --- advance every decoding sequence by one token ---
        for r in list(running):
            if r["ttft"] is None:
                continue
            itl.append(step_t)
            r["out"] -= 1
            r["cached"] += 1
            if r["out"] <= 0:
                r["finish"] = t
                done.append(r)
                running.remove(r)

    lat = [r["finish"] - r["arrival"] for r in done]
    return {"makespan": t, "throughput": total_out / t,
            "p50": np.percentile(lat, 50), "p99": np.percentile(lat, 99),
            "itl_p50": np.percentile(itl, 50) * 1000,
            "itl_p99": np.percentile(itl, 99) * 1000,
            "itl_max": max(itl) * 1000}


# ==========================================================================
work = sample_workload(N_REQUESTS)
lens = [o for _, _, o in work]
prompts = [p for _, p, _ in work]
print(f"workload: {N_REQUESTS} requests")
print(f"  prompt  tokens: mean {np.mean(prompts):.0f}  p99 {np.percentile(prompts,99):.0f}"
      f"  max {max(prompts)}")
print(f"  output  tokens: mean {np.mean(lens):.0f}  p99 {np.percentile(lens,99):.0f}"
      f"  max {max(lens)}")

print("\\n=== static batching ===")
s = simulate_static(work)
print(f"  slot utilization  {s['utilization']*100:>8.1f}%")
print(f"  throughput        {s['throughput']:>8.0f} tok/s")
print(f"  latency p50/p99   {s['p50']:>8.1f} / {s['p99']:.1f} s")

print("\\n=== continuous batching, unchunked prefill ===")
c = simulate_continuous(work, chunk_budget=None)
print(f"  throughput        {c['throughput']:>8.0f} tok/s"
      f"   ({c['throughput']/s['throughput']:.1f}x static)")
print(f"  latency p50/p99   {c['p50']:>8.1f} / {c['p99']:.1f} s")
print(f"  ITL p50/p99/max   {c['itl_p50']:>8.1f} / {c['itl_p99']:.1f} / {c['itl_max']:.1f} ms")

print("\\n=== continuous batching + chunked prefill ===")
print(f"  {'budget':>8} {'tok/s':>9} {'ITL p50':>9} {'ITL p99':>9} {'ITL max':>9}")
for budget in (256, 512, 1024, 2048, 8192):
    r = simulate_continuous(work, chunk_budget=budget)
    print(f"  {budget:>8} {r['throughput']:>9.0f} {r['itl_p50']:>9.1f} "
          f"{r['itl_p99']:>9.1f} {r['itl_max']:>9.1f}")

# --- TODO for you ---
#   1. Add KV memory as a hard constraint: cap total cached tokens at
#      (80 GB - 15 GB) / 128 KiB and preempt (recompute) when you exceed it.
#      Watch throughput fall as context lengths grow.
#   2. Add a priority queue with aging and measure whether long prompts starve
#      under a decode-priority policy.
#   3. Plot the throughput/latency Pareto frontier by sweeping MAX_BATCH.
`,
    expect: `Static batching lands somewhere around **10–25% slot utilization** — the exact figure
depends on the sampled tail, but it will be low, and it gets lower if you raise \`MAX_BATCH\`. That
is the headline result: the number is bad, and scaling makes it worse.

Continuous batching should show a **3–10× throughput improvement** over static on the same
workload. It will not reproduce Orca's 36.9× — that figure was against a weaker baseline and with
a different workload — but the direction and rough magnitude are right.

The chunked prefill table is the interesting one. As the budget falls from 8192 to 256:

- **ITL max drops sharply** — often by 10× or more. This is the stall being eliminated.
- **ITL p50 stays roughly flat** — typical steps were never the problem.
- **Throughput dips slightly** at very small budgets, because 256 tokens is not enough work to use
  the GPU well.

The sweet spot is usually in the 512–2048 range, which is what production defaults look like. If
your run shows ITL max at 8192 being 10–20× the p50 and at 512 being 2–3×, you have reproduced
the effect the Sarathi-Serve paper is about.

The simulator is deliberately simplified — no KV memory limit, no preemption, an approximate
overlap model for mixed iterations. It gets the shapes right, not the absolute numbers.`,
    stretch: `Add the KV memory constraint (TODO 1) and then sweep context length from 1k to 32k,
plotting achievable throughput. You should see throughput collapse as context grows, not because
of any compute effect but purely because fewer sequences fit — reproducing Module 2's table from
the scheduler's side. Then add fp8 KV (halve \`KV_GB_PER_TOKEN\`) and watch the whole curve lift.
That plot is the single best argument for Module 6.`,
  },

  papers: [
    {
      title: 'Orca: A Distributed Serving System for Transformer-Based Generative Models',
      by: 'Yu et al., OSDI 2022',
      url: 'https://www.usenix.org/conference/osdi22/presentation/yu',
      why: 'Introduced continuous (iteration-level) batching, which is the single largest throughput win in LLM serving and is now in every engine.',
      frame: `**Section 3** is the contribution: iteration-level scheduling. Pay particular
attention to **Section 3.2 on selective batching** — the observation that you can batch the
position-independent operations while handling attention per-sequence is what makes the whole
thing implementable, and it is the part most summaries omit. Figure 4 is the one to remember.
Skim the distributed execution details in Section 4.`,
    },
    {
      title: 'Taming Throughput-Latency Tradeoff in LLM Inference with Sarathi-Serve',
      by: 'Agrawal et al., OSDI 2024',
      url: 'https://arxiv.org/abs/2403.02310',
      why: 'Chunked prefill and "stall-free batching". The clearest analysis of prefill/decode interference and how to schedule around it.',
      frame: `**Section 3** quantifies the interference problem — read it before the solution so
the numbers land. **Section 4** is chunked prefill and the token-budget scheduler. The key insight
to extract is that mixing a prefill chunk into a decode iteration is close to free because the
decode iteration had idle arithmetic; that reframes the technique from a compromise into a win.`,
    },
    {
      title: 'Efficient Memory Management for LLM Serving with PagedAttention',
      by: 'Kwon et al., SOSP 2023',
      url: 'https://arxiv.org/abs/2309.06180',
      why: 'Continuous batching depends on being able to cheaply allocate and free per-sequence memory, which is exactly what paging provides. The two ideas are inseparable in practice.',
      frame: 'Read **Section 4.3** on scheduling and preemption — the swap-versus-recompute decision and how the block manager supports admission and eviction. The rest is Module 6.',
    },
    {
      title: 'DistServe: Disaggregating Prefill and Decoding',
      by: 'Zhong et al., OSDI 2024',
      url: 'https://arxiv.org/abs/2401.09670',
      why: 'The argument that prefill and decode should not share hardware at all — the alternative to chunked prefill rather than a complement to it.',
      frame: 'Read **Section 3** for the interference analysis and compare it against Sarathi-Serve. The two papers diagnose the same problem and reach opposite architectural conclusions, which makes reading them together unusually instructive.',
    },
  ],

  checkpoint: {
    claim: `You can explain why continuous batching is worth an order of magnitude, why prefill
and decode interfere, and why mixing them in one iteration turns out to be nearly free.`,
    questions: [
      {
        q: 'Why does static batching get worse as you increase the batch size?',
        a: `Because a static batch runs until its longest member finishes, and utilization is
mean length divided by max length. As you draw more samples from a heavy-tailed distribution, the
maximum reaches further into the tail while the mean stays roughly constant, so the ratio falls.
A batch of 8 with a mean of 180 and a max of 900 gives 20% utilization; a batch of 32 with the
same mean and a max of 2,000 gives 9%. It is one of the few optimizations that actively degrades
as you scale it, which is precisely why iteration-level scheduling was necessary rather than just
nice.`,
      },
      {
        q: 'What is selective batching, and why is it needed for continuous batching to work?',
        a: `In a continuous batch every sequence is at a different position, so their attention
computations have different KV lengths and cannot be packed into one rectangular call. Selective
batching — Orca's solution — splits the layer: the position-independent operations (QKV
projections, MLP, LM head) are batched by flattening all the batch's tokens into a single
\`[total_tokens, d_model]\` matmul, while attention is handled per-sequence with a kernel that
takes per-sequence lengths and offsets. This works because essentially all the weight traffic
lives in the batchable operations, so you still get the amortization that makes batching
worthwhile.`,
      },
      {
        q: 'A user pastes a 32k-token document and everyone else\'s token stream freezes. Explain the mechanism and give two fixes.',
        a: `The scheduler ran that prefill as a single unit. Prefill is compute-bound and scales
with prompt length — 32k tokens on Llama-3-8B is roughly 2 × 8.03e9 × 32000 = 514 TFLOP, about
740 ms at realistic H100 throughput. For that whole time the GPU is doing prefill, so every
decoding sequence produces nothing and sees an inter-token latency of 740 ms instead of ~10 ms.
Two fixes: **chunked prefill**, which splits the prompt into fixed-size pieces and processes one
per iteration alongside ongoing decode, capping the stall at the chunk cost; and
**disaggregation**, running prefill and decode on separate hardware pools so they cannot contend
at all.`,
      },
      {
        q: 'Why is adding a prefill chunk to a decode iteration close to free?',
        a: `Because they are bounded by different resources. A decode iteration at batch 32 moves
~23.6 GB of weights and KV and does only ~5.1e11 FLOP — arithmetic intensity around 22, deep in
memory-bound territory, with the tensor cores mostly idle. Adding 2,016 prefill tokens to that
iteration moves **the same bytes** (the weights are read once regardless) while doing 64× the
arithmetic, pushing intensity to roughly 1,400 — past the ridge point. The prefill runs in
compute capacity the decode step was already wasting. That is why chunked prefill improves both
TTFT stability and effective throughput rather than trading one for the other.`,
      },
      {
        q: 'Your scheduler must free memory. When would you swap a sequence to CPU rather than recompute its cache?',
        a: `Compare the two costs. Swapping means a PCIe round trip: a 2.5 GiB cache at ~50 GB/s
is roughly 50 ms out and 50 ms back. Recomputing means re-running prefill over the sequence's
tokens, which is compute-bound and scales with length. For short sequences prefill is cheap and
recomputation wins; for long ones prefill is expensive and swapping wins, provided you have host
memory and PCIe headroom. The crossover is roughly where prefill time exceeds transfer time. vLLM
implements both for exactly this reason. A third consideration: swapping consumes PCIe bandwidth
that may be needed for other things, so under heavy load recomputation can be preferable even
when it looks slower in isolation.`,
      },
      {
        q: 'Why should you usually operate at the knee of the throughput curve rather than at maximum batch size?',
        a: `Because past the knee you buy very little throughput for a lot of latency. Once KV
traffic dominates the byte count, each extra sequence adds proportionally to both work done and
bytes moved, so the curve flattens — for Llama-3-8B on an H100 at 4k context, going from batch 64
to 111 adds 73% more sequences for about 14% more throughput while TPOT rises substantially. You
also lose headroom: running at the memory ceiling means any traffic spike triggers preemption,
which is far more expensive than the throughput you gained. The knee is where you have captured
most of the available gain and still have slack.`,
      },
    ],
  },

  pitfalls: [
    {
      wrong: 'Bigger batch is always better for throughput.',
      right: `Only until KV traffic dominates. Batching amortizes weight bytes, which are fixed,
but each sequence brings its own cache, so KV bytes scale with batch. Once the cache dominates,
extra batch adds work and bytes in equal measure and throughput flattens — while latency keeps
degrading and preemption risk keeps rising. Find the knee and operate there.`,
    },
    {
      wrong: 'Continuous batching is just dynamic batching with a shorter timeout.',
      right: `Dynamic batching still runs a batch to completion; it only changes how the batch is
assembled. Continuous batching changes the *scheduling granularity* to the iteration, so a
sequence can join or leave the batch at any decode step. That is what eliminates the
wait-for-the-longest problem, and it requires selective batching and per-sequence KV management
to implement at all.`,
    },
    {
      wrong: 'Chunked prefill trades TTFT for TPOT.',
      right: `It costs a little TTFT — around 2% in the worked example — but it does not merely
redistribute a fixed amount of work. A mixed iteration moves the same bytes as a pure decode
iteration while doing far more arithmetic, so the prefill is riding in capacity that was being
wasted. Both phases come out ahead of running them separately.`,
    },
    {
      wrong: 'A decode-priority scheduler is the safe choice for smooth streaming.',
      right: `It gives smooth streams to sequences that are already running, and can starve new
requests indefinitely under sustained load, because there is always more decode work to do.
Any priority scheme needs an aging mechanism so that waiting requests eventually win, or some
user will simply never be served.`,
    },
  ],

  glossary: [
    { term: 'static batching', def: 'Fixed batch that runs to completion. Utilization collapses when sequence lengths vary, and gets worse as batch grows.' },
    { term: 'continuous batching', def: 'Iteration-level scheduling: evict finished sequences and admit waiting ones at every decode step. From Orca.' },
    { term: 'selective batching', def: 'Batch the position-independent operations across sequences while handling attention per-sequence. What makes continuous batching implementable.' },
    { term: 'chunked prefill', def: 'Splitting a long prefill into fixed-size pieces processed one per iteration, mixed with ongoing decode.' },
    { term: 'token budget', def: 'The cap on tokens processed per scheduler iteration. Decode tokens are admitted first, prefill chunks fill the rest.' },
    { term: 'prefill/decode interference', def: 'A long compute-bound prefill blocking every decoding sequence, spiking their inter-token latency.' },
    { term: 'preemption', def: 'Reclaiming a running sequence\'s KV memory, either by swapping it to host memory or discarding and recomputing it.' },
    { term: 'admission control', def: 'Deciding how many sequences to admit given that their eventual lengths, and therefore memory needs, are unknown.' },
    { term: 'knee', def: 'The batch size beyond which throughput gains become small relative to the latency cost. The right default operating point.' },
  ],
};
