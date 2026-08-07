export default {
  n: 1,
  slug: 'autoregressive-loop',
  title: 'THE AUTOREGRESSIVE LOOP',
  tagline: 'Prefill and decode are not two phases of one job. They are two different jobs.',
  hours: '4–6 hours',
  prereqs: ['Module 0', 'Comfort with big-O reasoning'],

  bigIdea: `A language model is a function from a sequence to a distribution over the next token.
To get a sequence out of it you have to call it repeatedly, feeding each output back in. That
loop is the whole of generation, and it is where all the difficulty lives.

The critical realisation is that the loop contains **two workloads that look identical in code
and behave nothing alike on hardware**. Processing the prompt — *prefill* — hands the GPU
thousands of positions at once and it happily saturates. Generating each subsequent token —
*decode* — hands it a single position and it starves, because you must still stream every weight
in the model to produce that one token.

Once you see them as separate workloads, a lot of confusing behaviour resolves. Why does TTFT
scale with prompt length but TPOT barely does? Why does batching help throughput enormously and
latency not at all? Why do serving systems increasingly run prefill and decode on *different
machines*? All of it follows from this split.`,

  concepts: [
    {
      name: 'Why the loop cannot be parallelized',
      keyPoint: 'Token n+1 is a function of token n, so the dependency chain has length N and no amount of hardware shortens it.',
      body: `Generation is a strict dependency chain. To sample token \`n+1\` you need the logits at
position \`n\`, which need the hidden state at position \`n\`, which needs token \`n\` to have already
been sampled. There is no reordering that breaks this.

The practical consequence is that **generation latency has a floor set by the number of
sequential model invocations**, not by total arithmetic. If one forward pass takes 5 ms, then 500
tokens take at least 2.5 seconds. Buying a GPU with twice the FLOPs does not help if each pass is
waiting on memory. Buying two GPUs does not help either, unless you split each pass across them
(Module 9) — you cannot run pass \`n\` and pass \`n+1\` concurrently.

This is what makes LLM serving strange compared to most inference workloads. An image classifier
is one forward pass; you scale it by adding replicas. A language model is *N* forward passes with
a serial dependency, and the only ways out are:

1. **Make each pass cheaper** — quantization, smaller models, better kernels (Modules 6, 7).
2. **Do more per pass** — batching, so one weight load serves many sequences (Module 5).
3. **Take more than one token per pass** — speculative decoding, the only technique that attacks
   the chain length itself (Module 8).

Category 3 is worth flagging now because it is the only one that shortens the dependency chain
rather than making each link cheaper. That is why it gets a whole module and why people find it
so satisfying.

One caveat on "inherently sequential": it is true for a fixed model and a fixed output. It is not
a law of nature. Diffusion language models and other non-autoregressive approaches sidestep it
entirely, at a quality cost that has so far kept them out of production for general text.`,
      ascii: `   prompt "the cat sat"
        |
        v
   [ PREFILL ]  3 positions in one pass  -->  " on"
        |
        v
   [ DECODE  ]  1 position  -->  " the"      pass 2
   [ DECODE  ]  1 position  -->  " mat"      pass 3
   [ DECODE  ]  1 position  -->  "."         pass 4
   [ DECODE  ]  1 position  -->  <eos>       pass 5

   4 sequential passes for 4 tokens. Unavoidable.`,
    },
    {
      name: 'Prefill: compute-bound, and it scales with the prompt',
      keyPoint: 'Prefill processes the whole prompt in one pass, so each weight load is amortized over thousands of positions and the GPU actually gets busy.',
      body: `When a request arrives you have the entire prompt already. There is no dependency
problem — every prompt token is known — so you run one forward pass over all of them at once,
exactly like a training step without the backward pass.

For a 2,000-token prompt, the \`W_gate\` matmul is \`[2000, 4096] @ [4096, 14336]\`. You read the
58.7 M-parameter weight matrix once and do 2,000 positions' worth of arithmetic with it. The
arithmetic-to-bytes ratio is excellent, the tensor cores are fed, and you are **compute-bound**.

The cost model for prefill is roughly:

\`\`\`
prefill_FLOPs ~= 2 * N_params * S     +     attention term
                 (the matmuls)              (grows as S^2)
\`\`\`

The \`2 * N_params\` per token is the standard estimate — every parameter participates in one
multiply and one add. For Llama-3-8B that is \`2 x 8.03e9 = 16.1 GFLOP per token\`. A 2,000-token
prompt costs about \`32.1 TFLOP\` in matmuls alone.

The attention term is separate and quadratic. Per layer, \`Q @ K.T\` and \`weights @ V\` together
cost about \`4 * S^2 * d_model\` FLOPs. Across 32 layers at \`S = 2000\`:
\`4 x 2000^2 x 4096 x 32 = 2.1 TFLOP\`. Small next to 32 TFLOP — but at \`S = 32000\` it becomes
\`4 x 32000^2 x 4096 x 32 = 537 TFLOP\` against \`514 TFLOP\` of matmul, and attention has taken
over. This crossover is why long-context prefill is its own engineering problem.

Because prefill is compute-bound, **TTFT scales roughly linearly with prompt length** (plus a
quadratic term that bites at long context). Doubling the prompt roughly doubles time-to-first-token.
And because it already saturates the GPU, batching prefill buys you very little — the machine was
already busy.`,
      ascii: '',
    },
    {
      name: 'Decode: memory-bound, and it barely notices the prompt',
      keyPoint: 'Every decode step re-reads every weight in the model to produce one token, so its cost is set by bandwidth and is nearly independent of sequence length.',
      body: `After prefill you have one token. Now you need the next, and you have exactly one new
position to process.

The same \`W_gate\` matmul is now \`[1, 4096] @ [4096, 14336]\`. You still read all 58.7 M
parameters. You do 1 position of arithmetic with them. The ratio has collapsed by 2,000×.

Per decode step, for Llama-3-8B at fp16:

\`\`\`
bytes moved:  ~15.0 GB   (every weight, once; embedding is a lookup)
FLOPs done:    ~16.1 GFLOP
intensity:      16.1e9 / 15.0e9  ~=  1.07 FLOP per byte
\`\`\`

An H100 can do about 295 FLOPs for every byte it moves. You are asking it to do 1. **You are
using roughly 0.4% of the machine's arithmetic capability.** The GPU spends essentially the whole
step waiting for HBM.

Two consequences follow, and both are counterintuitive until you see the numbers:

**Decode time is almost independent of context length.** The weights are the same 15 GB whether
your context is 100 tokens or 100,000. Only the attention step touches the KV cache, and that
cache is usually small relative to the weights. Going from 1k to 8k context might move TPOT by a
few percent — nothing like the linear growth in TTFT.

**Decode gets dramatically cheaper per token when batched.** If 32 sequences decode together, you
read the 15 GB *once* and produce 32 tokens. Per-token cost falls by 32×, and arithmetic
intensity rises from 1 to 32. This is the single largest lever in LLM serving, and Module 5 is
about the machinery required to actually exploit it.

Note what does *not* change: the latency of a single step. Batching improves throughput while
leaving per-user speed roughly flat (slightly worse, in practice). Latency and throughput are
different currencies, and batch size is the exchange rate.`,
      ascii: `  PREFILL                          DECODE
  W [4096 x 14336]                 W [4096 x 14336]
  x [2000 x 4096]                  x [1 x 4096]
  ---------------                  ---------------
  read 117 MB of W                 read 117 MB of W
  do 2000 positions                do 1 position
  ~2000 FLOP/byte                  ~1 FLOP/byte
  GPU: saturated                   GPU: idle, waiting on HBM`,
    },
    {
      name: 'The naive recomputation problem',
      keyPoint: 'Without a cache, generating N tokens costs O(N^2) attention work and re-reads the weights N times — and the second cost is the one that hurts first.',
      body: `Suppose you implement generation the obvious way: keep a list of tokens, append the
sampled one, and re-run the full forward pass on the whole list.

This is correct. It is also profoundly wasteful, in two independent ways that are worth
separating because people usually only notice one.

**Waste 1 — the attention quadratic.** At step \`t\` you compute a \`[t, t]\` score matrix, but you
only use the last row. Total attention work over \`N\` steps is proportional to
\`sum_{t=1..N} t^2\`, which is O(N³) in FLOPs — versus O(N²) if you only ever computed the new
row. This is the waste that gets mentioned in textbooks.

**Waste 2 — the weight re-reads.** At step \`t\` you re-read all 15 GB of weights and re-derive the
K and V vectors for all \`t\` positions, when \`t-1\` of them are bit-identical to last step. This is
the waste that actually dominates in practice at realistic sequence lengths, because it is
bandwidth and bandwidth is what you are short of.

Concretely, generating 512 tokens from a 512-token prompt on Llama-3-8B:

| | attention FLOPs | weight bytes read |
|---|---|---|
| no cache | ~O(N³), tens of TFLOP wasted | 512 × 15.0 GB = **7.7 TB** |
| with cache | only the new row each step | 512 × 15.0 GB = **7.7 TB** |

Look carefully: **the weight traffic is the same either way.** The cache does not save weight
reads — those are irreducible for a single sequence, and only batching fixes them. What the cache
eliminates is the redundant recomputation of K and V and the quadratic attention blowup.

That is a genuinely important distinction and it is worth sitting with. The KV cache is not the
answer to the memory-bandwidth problem. It is the answer to a *different* problem — redundant
recomputation — and it happens to be the prerequisite that makes the real answer (batching)
affordable, by keeping per-step work small enough that many sequences fit in flight at once.`,
      ascii: '',
    },
    {
      name: 'Anatomy of one decode step',
      keyPoint: 'A decode step is a long chain of small bandwidth-bound operations between the big matmuls, and the glue is not free.',
      body: `Zoom into a single decode step so you know what is actually being timed:

\`\`\`
1.  embedding lookup           one row, 8 KB
2.  for each of 32 layers:
      a. RMSNorm               read 4096, write 4096      bandwidth-bound
      b. q/k/v projections     read 25 MB of weights      bandwidth-bound
      c. RoPE on q, k          tiny elementwise           bandwidth-bound
      d. append k,v to cache   write ~4 KB                bandwidth-bound
      e. attention over cache  read the whole KV cache    bandwidth-bound
      f. o_projection          read 33 MB                 bandwidth-bound
      g. residual add          trivial                    bandwidth-bound
      h. RMSNorm               trivial                    bandwidth-bound
      i. gate/up/down          read 352 MB of weights     bandwidth-bound
      j. residual add          trivial                    bandwidth-bound
3.  final RMSNorm
4.  lm_head                    read 1.05 GB               bandwidth-bound
5.  sample from 128256 logits
\`\`\`

Every single line is bandwidth-bound at batch 1. There is not one compute-bound operation in a
decode step.

Two things this makes visible that the FLOP count hides:

**Kernel launch overhead is real.** That is on the order of 200+ separate CUDA kernel launches
per token if nothing is fused. At roughly 5 µs of launch overhead each, you are looking at ~1 ms
of pure overhead against a ~4.5 ms theoretical step — over 20% of your budget spent launching
work rather than doing it. This is why CUDA Graphs (capture the launch sequence once, replay it)
and aggressive kernel fusion are standard in serving engines, and why a naive PyTorch loop is
often 2–3× slower than the roofline says it should be.

**Step 4 is expensive and easy to forget.** The LM head is a \`[1, 4096] @ [4096, 128256]\` matmul
reading 1.05 GB. That is 7% of your entire per-token byte budget spent turning a hidden state
into logits, for a model whose vocabulary is large. It is one reason vocabulary size is an
inference cost, not just a modelling choice.`,
      ascii: '',
    },
    {
      name: 'What throughput means when the loop is serial',
      keyPoint: 'Per-user speed and system throughput are different numbers that move in opposite directions as you turn the batch dial.',
      body: `"How fast is your model?" is not a well-posed question. There are at least four
distinct numbers and they trade against each other:

- **TTFT** (time to first token) — how long until the user sees anything. Dominated by prefill,
  so it scales with prompt length and with how long you queued behind other work.
- **TPOT / ITL** (time per output token, inter-token latency) — the gap between successive
  tokens. Dominated by decode. This is what "feels" fast.
- **End-to-end latency** — \`TTFT + TPOT × output_tokens\`, roughly. What actually matters to a
  user waiting for a complete answer.
- **Throughput** — total tokens per second across all concurrent requests. What determines your
  cost per million tokens.

The tension: throughput is maximized by large batches, because one weight load serves many
sequences. Per-user TPOT is (slightly) hurt by large batches, and TTFT is badly hurt by them
because your request waits for a slot and then contends with everyone else's prefill.

A useful frame is **Little's Law**: \`concurrency = arrival_rate × latency\`. If you want to serve
100 requests per second and each takes 2 seconds, you need 200 requests in flight. That number
must fit in your KV cache budget — which is Module 2's subject, and the reason memory management
turns out to be the discipline that governs everything.

The practical upshot is that any benchmark quoting a single "tokens per second" figure without
saying **at what batch size, at what context length, and at which percentile** is not telling you
anything. A system doing 15,000 tok/s aggregate at batch 256 may deliver a miserable 12 tok/s to
each user. Both numbers are true. Module 10 returns to how to benchmark honestly.`,
      ascii: '',
    },
  ],

  mathLab: {
    prompt: `Take **Llama-3-8B** (8.03 B parameters, fp16) on an **H100 SXM**
(3.35 TB/s HBM bandwidth, 989 TFLOP/s dense BF16). A request has a **2,048-token prompt** and
generates **512 tokens**. Batch size 1. Ignore the KV cache for now.

Work out:

1. Prefill matmul FLOPs, and the theoretical prefill time if compute-bound.
2. Bytes that must be read during prefill, and the theoretical prefill time if bandwidth-bound.
3. Which of (1) and (2) is the binding constraint? That answers "is prefill compute- or
   memory-bound?" with a number rather than an assertion.
4. Per-decode-step bytes, and theoretical TPOT.
5. Total generation time, and the split between prefill and decode.
6. The arithmetic intensity of prefill and of decode. Compare both to the H100's ratio of
   \`989e12 / 3.35e12 = 295 FLOP/byte\`.
7. Now suppose you batch **64** such requests. What happens to decode arithmetic intensity, and
   what does that tell you?`,

    solution: `**1. Prefill FLOPs**

\`\`\`
2 x N_params x S = 2 x 8.03e9 x 2048 = 3.29e13 = 32.9 TFLOP
compute time = 32.9e12 / 989e12 = 33.3 ms
\`\`\`

(Plus the attention term: \`4 x S^2 x d_model x n_layers = 4 x 2048^2 x 4096 x 32 = 2.20 TFLOP\`,
about 7% more. Call it ~35.5 ms. Small at this length; dominant past ~30k.)

**2. Prefill bytes**

The weights are read once for the whole pass, regardless of \`S\`:

\`\`\`
15.0 GB / 3350 GB/s = 4.48 ms
\`\`\`

**3. Binding constraint**

\`\`\`
compute-bound time   33.3 ms
bandwidth-bound time  4.48 ms
\`\`\`

Compute is 7.4× larger, so prefill is **compute-bound**. TTFT ≈ 33–36 ms in the ideal case.

**4. Decode**

Every step re-reads every weight:

\`\`\`
bytes  = 15.0 GB
TPOT   = 15.0 / 3350 = 4.48 ms
\`\`\`

FLOPs per step are \`2 x 8.03e9 = 16.1 GFLOP\`, which at 989 TFLOP/s would take **0.016 ms**. The
memory time is **275× larger**. Decode is emphatically **memory-bound**.

**5. Total**

\`\`\`
prefill:  35.5 ms
decode:   512 x 4.48 ms = 2293 ms
                          ---------
total:                    2329 ms  (2.33 s)
\`\`\`

Prefill is **1.5%** of wall-clock. Decode is **98.5%**. For any request generating a meaningful
number of tokens, you are optimising decode.

**6. Arithmetic intensity**

\`\`\`
prefill:  32.9e12 FLOP / 15.0e9 bytes  = 2193 FLOP/byte
decode:   16.1e9  FLOP / 15.0e9 bytes  =  1.07 FLOP/byte
H100 ridge point:                        295 FLOP/byte
\`\`\`

Prefill sits at 2193, far to the right of the ridge — compute-bound with room to spare. Decode
sits at 1.07, far to the left — memory-bound by a factor of about 275. **The two phases are on
opposite sides of the roofline by three orders of magnitude.** That is the central asymmetry of
the field, and you have now derived it rather than been told it.

**7. Batch 64**

Arithmetic intensity for a matmul \`W[n,k] @ x[k,B]\` is:

\`\`\`
FLOPs = 2nkB ,  bytes = 2nk (fp16 weights)  ->  intensity = B FLOP/byte
\`\`\`

The intensity of decode **equals the batch size**. At \`B = 64\` you are at 64 FLOP/byte — still
below the 295 ridge, so still memory-bound, but you are now doing 64 tokens per 15 GB read
instead of 1:

\`\`\`
per-token bytes:  15.0 GB / 64 = 234 MB
per-token time:   4.48 ms / 64 = 0.07 ms  ->  ~14,300 tok/s aggregate
\`\`\`

A 64× throughput improvement from a single change, with per-user TPOT unchanged at 4.48 ms.

And the striking corollary: to reach the H100's ridge point you would need **batch ≈ 295**. Below
that you are leaving arithmetic on the table no matter what else you do. That single number
explains why serving systems fight so hard for large batches — and why Module 2's KV cache
budget, which is what caps batch size, is the real constraint on the whole system.`,
  },

  codeLab: {
    goal: `Write a generation loop with **no KV cache** and measure how the per-token cost grows
with sequence length. Then plot it. You are looking for the quadratic.

Use a small real model on CPU (GPT-2 is fine and downloads in seconds) so the numbers are real
rather than simulated. The absolute times do not matter; the *shape* of the curve does.`,
    code: `"""
Naive generation with no KV cache. Measures per-token latency vs context length.

    pip install torch transformers matplotlib

Runs on CPU in a couple of minutes. A GPU makes it faster but the shape is the same.
"""
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "gpt2"          # 124M params -- small enough to be quick, real enough to be honest
N_TOKENS = 256

tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL)
model.eval()

prompt = "The history of computing begins"
ids = tok(prompt, return_tensors="pt").input_ids


@torch.no_grad()
def generate_no_cache(ids, n):
    """The obvious implementation: re-run the whole forward pass every step."""
    times = []
    for _ in range(n):
        t0 = time.perf_counter()
        # use_cache=False forces a full recompute over the entire sequence
        out = model(ids, use_cache=False)
        next_id = out.logits[:, -1, :].argmax(-1, keepdim=True)
        ids = torch.cat([ids, next_id], dim=1)
        times.append(time.perf_counter() - t0)
    return ids, times


@torch.no_grad()
def generate_with_cache(ids, n):
    """The same thing, but keep K and V around. Module 2 explains why this is legal."""
    times = []
    out = model(ids, use_cache=True)          # prefill
    past = out.past_key_values
    next_id = out.logits[:, -1, :].argmax(-1, keepdim=True)
    ids = torch.cat([ids, next_id], dim=1)

    for _ in range(n - 1):
        t0 = time.perf_counter()
        # feed ONLY the new token; the cache supplies everything else
        out = model(next_id, past_key_values=past, use_cache=True)
        past = out.past_key_values
        next_id = out.logits[:, -1, :].argmax(-1, keepdim=True)
        ids = torch.cat([ids, next_id], dim=1)
        times.append(time.perf_counter() - t0)
    return ids, times


print("running without cache...")
_, slow = generate_no_cache(ids, N_TOKENS)
print("running with cache...")
_, fast = generate_with_cache(ids, N_TOKENS)

start_len = ids.shape[1] - N_TOKENS
lengths_slow = [start_len + i for i in range(len(slow))]
lengths_fast = [start_len + 1 + i for i in range(len(fast))]

print(f"\\n{'ctx':>6} {'no cache (ms)':>15} {'cached (ms)':>13} {'ratio':>8}")
for i in range(0, min(len(slow), len(fast)), 32):
    r = slow[i] / fast[i]
    print(f"{lengths_slow[i]:>6} {slow[i]*1000:>15.2f} {fast[i]*1000:>13.2f} {r:>8.1f}x")

print(f"\\ntotal without cache: {sum(slow):.2f} s")
print(f"total with cache:    {sum(fast):.2f} s")
print(f"speedup:             {sum(slow)/sum(fast):.1f}x")

# --- the plot is the point ---
try:
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(lengths_slow, [t * 1000 for t in slow], label="no KV cache", lw=2)
    ax.plot(lengths_fast, [t * 1000 for t in fast], label="with KV cache", lw=2)
    ax.set_xlabel("context length (tokens)")
    ax.set_ylabel("time for ONE token (ms)")
    ax.set_title("Per-token cost vs context length")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig("kv_cache_effect.png", dpi=130)
    print("\\nwrote kv_cache_effect.png")
except ImportError:
    print("\\n(install matplotlib to get the plot)")

# --- TODO for you ---
#   1. Fit a curve to the no-cache line. Is it linear or quadratic in context length?
#      Think about which parts of the forward pass grow with S and which do not.
#   2. The cached line is nearly flat but not perfectly flat. What is the slope?
#      (Hint: one operation in a decode step DOES read something that grows with S.)
`,
    expect: `The no-cache line grows visibly and superlinearly. The cached line is close to flat.

On CPU with GPT-2 you should see roughly: at 32 tokens of context the two are within ~2–4× of
each other; by 256 tokens the gap is 10–30×. Overall speedup for the 256-token run is typically
**10–25×**. Exact numbers depend heavily on your machine — do not chase a specific figure, chase
the shape.

The no-cache curve is dominated by the *linear* term at these lengths (you re-read the weights
every step and reprocess \`S\` positions of matmul), with the quadratic attention term becoming
visible only at longer contexts. If you extend to 1024+ tokens the curvature becomes obvious.

The cached line has a small positive slope, not zero. That slope is the KV cache being read: it
grows linearly with context. At GPT-2 scale it is barely detectable; at 128k context on a 70B
model it is the dominant term in a decode step. That is the whole of Module 2.`,
    stretch: `Instrument \`generate_with_cache\` to report the size of \`past_key_values\` in bytes
at each step, and plot it alongside the timings. Then compute, for GPT-2 (12 layers, 12 heads,
head_dim 64, fp32), the KV bytes per token by hand and check your instrument agrees. You will use
that formula constantly from Module 2 onward.`,
  },

  papers: [
    {
      title: 'Efficiently Scaling Transformer Inference',
      by: 'Pope et al. (Google), 2022',
      url: 'https://arxiv.org/abs/2211.05102',
      why: 'The paper that made prefill/decode asymmetry and inference-time roofline analysis legible to the field. Still one of the best treatments of how partitioning interacts with the two phases.',
      frame: `Read **Section 2** (inference cost model) and **Section 3** (partitioning
strategies) carefully — Section 2 is essentially this module written by the people who first laid
it out rigorously. The multi-chip partitioning notation in Section 3 is heavy going; skim it now
and come back after Module 9.`,
    },
    {
      title: 'Transformer Inference Arithmetic',
      by: 'Carol Chen (kipply), 2022',
      url: 'https://kipp.ly/transformer-inference-arithmetic/',
      why: 'The standard back-of-envelope reference for inference cost. Short, numeric, and the source of most of the mental arithmetic practitioners actually use.',
      frame: `Work every calculation yourself as you read rather than accepting the results. The
sections on the KV cache and on the memory-bandwidth-bound nature of decoding are the payload.
Some absolute hardware numbers have aged; the method has not.`,
    },
    {
      title: 'Language Models are Few-Shot Learners (GPT-3)',
      by: 'Brown et al., 2020',
      url: 'https://arxiv.org/abs/2005.14165',
      why: 'Not an inference paper, but the one that made long prompts the norm — which is exactly what turned prefill into a workload worth engineering around.',
      frame: 'Read only **Section 2.1** on the in-context learning setup. Everything else is out of scope here. The relevant realization: few-shot prompting means prompts got 10–100× longer, and TTFT became a product problem.',
    },
  ],

  checkpoint: {
    claim: `You can explain, with numbers rather than adjectives, why prefill and decode sit on
opposite sides of the roofline, and you can predict what happens to TTFT and TPOT when you change
prompt length, output length, or batch size.`,
    questions: [
      {
        q: 'Your TTFT doubled but TPOT is unchanged. What changed about the workload?',
        a: `Almost certainly the prompt got longer — roughly twice as long, since prefill is
compute-bound and scales close to linearly with prompt length (with a quadratic attention term
that shows up at long context). TPOT is set by streaming the weights, which does not depend on
prompt length, so it stays flat. Other candidates: you got queued behind more prefill work, or
the scheduler started admitting more requests so your prefill contends for the GPU. What it is
*not* is a change in model size or quantization — those would move TPOT too.`,
      },
      {
        q: 'Why does batching improve throughput enormously for decode but barely at all for prefill?',
        a: `Because prefill is already compute-bound. A single 2,000-token prefill saturates the
tensor cores; adding more sequences just queues more work for a machine that is already busy, so
aggregate throughput is roughly flat and latency rises. Decode is memory-bound with arithmetic
intensity equal to the batch size — at batch 1 you read 15 GB of weights to produce one token,
which is a ~0.4% utilization of an H100's arithmetic. Batching 64 sequences reads the same 15 GB
and produces 64 tokens. The weight load is amortized, so throughput scales nearly linearly until
you approach the ridge point or run out of KV cache memory.`,
      },
      {
        q: 'Without a KV cache, what exactly is being wasted? Be precise about bandwidth versus compute.',
        a: `Two distinct things. First, redundant *compute*: you recompute the K and V vectors for
every past position on every step, and you compute a full \`[t, t]\` attention matrix when only the
last row is needed — pushing total attention work to O(N³) over a generation instead of O(N²).
Second, redundant *activation traffic* for those recomputed positions. What is **not** wasted, and
this is the part people get wrong, is weight bandwidth: you re-read all 15 GB of weights every
step either way. The cache does not fix the bandwidth problem — batching does. The cache fixes
recomputation, and by keeping per-step work small it is what makes large batches affordable.`,
      },
      {
        q: 'A request has a 100-token prompt and generates 1,000 tokens. Another has a 10,000-token prompt and generates 10 tokens. Which costs the serving system more, and which feels slower to the user?',
        a: `They stress different resources. Using Llama-3-8B on an H100: the first costs about
1.6 TFLOP of prefill plus 1,000 decode steps ≈ 4.5 s of bandwidth-bound work — it dominates GPU
*time* and occupies a batch slot for a long while. The second costs about 161 TFLOP of prefill
(~163 ms compute-bound) plus 10 decode steps (~45 ms) ≈ 0.2 s total — far more FLOPs but far less
wall-clock, and it releases its slot quickly. So the first costs the system more; the second has
much worse TTFT. To the user, the first feels responsive but takes 4.5 s to finish; the second
stalls for 160 ms then finishes almost instantly. This is exactly why chunked prefill exists
(Module 5): a 10,000-token prefill run as one blocking unit stalls every other request's decode.`,
      },
      {
        q: 'Why is decode latency roughly independent of context length, given that attention is O(S²)?',
        a: `Because the O(S²) applies to prefill, where you compute an \`[S, S]\` score matrix. At
decode you have one query attending to \`S\` keys, so attention is O(S) per step, not O(S²) — and
that linear term is reading the KV cache, which is typically much smaller than the weights. For
Llama-3-8B at 8k context the KV cache is 1 GiB against 15 GB of weights, so it contributes about
7% of the bytes moved. Doubling context to 16k adds another 7%, not 100%. The weight traffic is
constant and dominant, so TPOT is nearly flat. This stops being true at very long context or very
large batch, where the KV cache overtakes the weights — which is exactly the regime Module 6
attacks.`,
      },
    ],
  },

  pitfalls: [
    {
      wrong: 'Generation is slow because language models have a lot of FLOPs to do.',
      right: `At decode time the FLOPs are trivial — 16 GFLOP against an H100 that does 989,000
GFLOP/s, so about 0.016 ms of arithmetic. The step takes 4.5 ms. You are 99.6% idle, waiting on
memory. Generation is slow because of *bytes*, not FLOPs, and every optimization that helps is
best understood as reducing bytes moved or increasing work done per byte.`,
    },
    {
      wrong: 'The KV cache is what makes decoding memory-bound.',
      right: `Backwards. Decoding is memory-bound because you must stream every model weight to
produce one token — that is true with or without a cache, and the weights usually outweigh the
cache by an order of magnitude. The KV cache is a *response* to a different problem
(recomputation). At very long context or very large batch the cache does become the dominant
memory term, but that is a regime, not the default.`,
    },
    {
      wrong: 'A faster GPU with more FLOPs will speed up token generation.',
      right: `Only if it also has more bandwidth. Compare an H100 (3.35 TB/s, 989 TFLOP/s) to an
H200: same compute, 4.8 TB/s. The H200 decodes about 1.43× faster at batch 1 purely from
bandwidth, and prefills at the same speed. Meanwhile a hypothetical chip with double the FLOPs
and identical bandwidth would decode at *exactly* the same speed. For decode, read the bandwidth
line on the spec sheet first.`,
    },
    {
      wrong: 'Tokens per second is a single number that describes a serving system.',
      right: `There are at least four numbers — TTFT, TPOT, end-to-end latency, and aggregate
throughput — and they trade against each other through batch size. A system at 15,000 tok/s
aggregate might deliver 12 tok/s per user. Any benchmark without batch size, context length, and
percentiles attached is unfalsifiable.`,
    },
  ],

  glossary: [
    { term: 'prefill', def: 'The forward pass over the entire prompt, done in one shot. Compute-bound; sets TTFT.' },
    { term: 'decode', def: 'The per-token forward passes that follow prefill, one position at a time. Memory-bound; sets TPOT.' },
    { term: 'TTFT', def: 'Time to first token. How long the user waits before anything appears.' },
    { term: 'TPOT / ITL', def: 'Time per output token / inter-token latency. The gap between successive tokens once streaming starts.' },
    { term: 'arithmetic intensity', def: 'FLOPs performed per byte moved from memory. Compare it against the hardware ratio to find your bottleneck.' },
    { term: 'ridge point', def: 'The arithmetic intensity at which a machine transitions from memory-bound to compute-bound. About 295 FLOP/byte for an H100.' },
    { term: "Little's Law", def: 'concurrency = arrival rate x latency. Tells you how many requests must be in flight to hit a throughput target.' },
    { term: 'CUDA Graph', def: 'A captured, replayable sequence of kernel launches. Removes per-launch CPU overhead, which is significant when a decode step is hundreds of tiny kernels.' },
  ],
};
