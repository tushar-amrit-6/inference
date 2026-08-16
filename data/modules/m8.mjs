export default {
  n: 8,
  slug: 'speculative-decoding',
  title: 'SPECULATIVE DECODING',
  tagline: 'Buy back the compute that memory-bound decoding leaves on the floor — and provably change nothing.',
  hours: '6–8 hours',
  prereqs: ['Module 1', 'Module 3', 'Module 4'],

  bigIdea: `Every module so far has attacked the *bytes* side of the ratio. This one attacks the
other side: the wasted arithmetic.

Here is the setup. At batch 1, a decode step moves 15 GB and does 16 GFLOP — arithmetic intensity
of about 1 against an H100's ridge point of 295. **You are using roughly 0.4% of the machine's
arithmetic.** The tensor cores sit idle for the entire step.

Now notice something about that idle capacity. Verifying whether a *guessed* token is correct
costs no extra memory traffic — the weights are already being streamed. If you guess five tokens
and check all five in one forward pass, you move the same 15 GB and do five positions' worth of
arithmetic instead of one. Intensity goes from 1 to 5. Still nowhere near the ridge, so the extra
arithmetic is genuinely free.

That is speculative decoding: **draft cheaply, verify in parallel, and spend idle FLOPs to shorten
the sequential chain.** It is the only technique in these chapters that reduces the *number* of
forward passes rather than the cost of each one.

The part that makes it remarkable rather than merely clever is that the output distribution is
preserved **exactly**. Not approximately, not usually — provably, by a modified rejection sampling
scheme. You are not trading quality for speed. You are getting the same samples faster, and the
proof fits on half a page.`,

  concepts: [
    {
      name: 'Draft and verify: why verification is nearly free',
      keyPoint: 'Checking gamma guessed tokens costs one forward pass, the same weight traffic as generating one token — so the extra positions ride in idle arithmetic.',
      body: `The loop:

\`\`\`
1. DRAFT   run a cheap model autoregressively for gamma steps,
           producing candidate tokens x1..x_gamma and its
           distributions q(.|prefix), q(.|prefix,x1), ...

2. VERIFY  run the target model ONCE on [prefix, x1, ..., x_gamma].
           Because attention is causal, one pass gives you the target's
           distribution p at every one of those positions simultaneously.

3. ACCEPT  walk left to right, accepting or rejecting each x_i by the
           rule in the next concept. Stop at the first rejection.

4. On rejection at position i, sample a corrected token from a residual
   distribution. On accepting all gamma, sample one bonus token from
   p(.|prefix, x1..x_gamma) -- which you already have.
\`\`\`

Step 2 is the crux. **A single forward pass over \`γ+1\` positions costs the same weight traffic as
a forward pass over one position.** You read all 15 GB either way. The extra positions cost extra
arithmetic — which you had in surplus.

Per accepted token, the cost falls dramatically. If you accept 3 of 4 drafts plus a bonus token,
you produced 4 tokens for one target-model pass instead of four passes. **Four times fewer weight
reads.**

Note also step 4's bonus token. If every draft is accepted, the verification pass has already
computed \`p\` at the final position, so you get one extra token for free. This is why the expected
yield formula has \`γ+1\` in it rather than \`γ\`, and it is a meaningful contribution at high
acceptance rates.

Two things this does *not* do, worth being clear about. It does not reduce total FLOPs — it
increases them, since drafts that get rejected were computed for nothing. And it does not help
when you are already compute-bound, because then there is no idle arithmetic to spend. Both
qualifications become important later in this module.`,
      ascii: `  STANDARD DECODE -- 4 tokens, 4 target passes, 4 x 15 GB
    [target] -> t1
    [target] -> t2
    [target] -> t3
    [target] -> t4                        60 GB moved

  SPECULATIVE -- 4 tokens, 1 target pass, 4 draft passes
    [draft][draft][draft] -> x1 x2 x3     3 x 0.5 GB
    [target on x1,x2,x3 at once]          1 x 15 GB
      accept x1  ✓
      accept x2  ✓
      reject x3  ✗ -> resample t3
                                          16.5 GB moved, 3 tokens
                                          -> 3.6x less traffic per token`,
    },
    {
      name: 'The acceptance rule, and the proof that nothing changes',
      keyPoint: 'Accept with probability min(1, p/q) and on rejection sample from the normalized positive part of p−q; the composition is exactly p.',
      body: `The naive approach — accept the draft token if the target agrees, otherwise take the
target's token — is wrong. It biases the output toward tokens the draft model likes, because the
draft gets a veto over which tokens are even considered.

The correct rule is modified rejection sampling. Let \`q\` be the draft's distribution and \`p\` the
target's, at some position, with draft token \`x\`:

\`\`\`
1. accept x with probability  min(1, p(x) / q(x))
2. if rejected, sample from the residual:

       p'(z) = max(0, p(z) - q(z)) / sum_w max(0, p(w) - q(w))
\`\`\`

Read the intuition first. If the target likes \`x\` at least as much as the draft did
(\`p(x) ≥ q(x)\`), accept unconditionally. If the target likes it less, accept with probability
\`p/q\` — proportionally to how much less. When you reject, you must not simply sample from \`p\`,
because you would double-count the mass \`p\` and \`q\` already agreed on. The residual
\`max(0, p−q)\` is precisely the mass \`p\` wanted that \`q\` under-supplied.

**The proof.** Compute the total probability of emitting token \`z\`.

Accepting \`z\` requires drafting it and then accepting:

\`\`\`
P(draft z and accept) = q(z) * min(1, p(z)/q(z)) = min(q(z), p(z))
\`\`\`

Rejection happens with probability:

\`\`\`
P(reject) = 1 - sum_w min(q(w), p(w))
\`\`\`

And a useful identity, since both \`p\` and \`q\` sum to 1:

\`\`\`
sum_w max(0, p(w) - q(w)) = sum_w [ p(w) - min(p(w), q(w)) ]
                          = 1 - sum_w min(p(w), q(w))
                          = P(reject)
\`\`\`

The normalizer of the residual is exactly the rejection probability. So:

\`\`\`
P(emit z) = min(q(z), p(z))  +  P(reject) * max(0, p(z) - q(z)) / P(reject)
          = min(q(z), p(z))  +  max(0, p(z) - q(z))
\`\`\`

Two cases. If \`p(z) ≤ q(z)\`: \`= p(z) + 0 = p(z)\`. If \`p(z) > q(z)\`:
\`= q(z) + p(z) − q(z) = p(z)\`.

**\`P(emit z) = p(z)\` in both cases.** The output distribution is the target model's, exactly,
regardless of how bad the draft model is. ∎

This is why speculative decoding is free performance rather than a quality trade. A poor draft
model does not corrupt your output — it just gets rejected more often and saves you less. The
draft affects *speed only*. That property is unusual enough in this field to be worth pausing on:
almost every other optimization here asks you to give something up.

*(One caveat for practice: the proof assumes \`p\` and \`q\` are the distributions you are actually
sampling from. If you apply top-p or temperature, both must see the same processed distribution,
or you lose the guarantee. Implementations that apply sampling parameters to the target but not
the draft are subtly wrong — and it is a common bug because the output still looks fine.)*`,
      ascii: `  q (draft)   ████████░░░░░░░░
  p (target)  ████░░░░████████

  min(p,q)    ████             <- accepted directly
  max(0,p-q)      ░░░░████     <- the residual, sampled on rejection

  accepted mass + residual mass = p, exactly`,
    },
    {
      name: 'Acceptance rate: the number that governs everything',
      keyPoint: 'Expected tokens per verification is (1 − α^(γ+1))/(1 − α), and it saturates fast — so long draft lengths pay only at very high acceptance.',
      body: `Let \`α\` be the acceptance rate: the probability that a given draft token survives.
Leviathan et al. show that under a reasonable independence assumption, the expected number of
tokens produced per verification pass is:

\`\`\`
E[tokens] = (1 - alpha^(gamma+1)) / (1 - alpha)
\`\`\`

The \`γ+1\` accounts for the bonus token you get when all \`γ\` drafts are accepted.

Tabulate it:

| α \\ γ | 1 | 2 | 3 | 4 | 5 | 7 | 10 |
|---|---|---|---|---|---|---|---|
| 0.5 | 1.50 | 1.75 | 1.88 | 1.94 | 1.97 | 1.99 | 2.00 |
| 0.7 | 1.70 | 2.19 | 2.53 | 2.77 | 2.94 | 3.15 | 3.28 |
| 0.8 | 1.80 | 2.44 | 2.95 | 3.36 | 3.69 | 4.16 | 4.57 |
| 0.9 | 1.90 | 2.71 | 3.44 | 4.10 | 4.69 | 5.70 | 6.86 |

Two patterns matter.

**Returns diminish sharply.** At \`α = 0.5\`, going from \`γ = 3\` to \`γ = 10\` improves yield from
1.88 to 2.00 — 7% more tokens for 233% more draft work. The limit as \`γ → ∞\` is \`1/(1−α)\`, so
\`α = 0.5\` can never exceed 2 tokens per pass no matter how far you draft.

**Acceptance rate matters far more than draft length.** Moving \`α\` from 0.7 to 0.9 at \`γ = 4\`
takes you from 2.77 to 4.10 — a 48% gain. Moving \`γ\` from 4 to 10 at \`α = 0.7\` gains 18%. If you
are tuning, tune the draft model, not the draft length.

Now the actual speedup, which must account for the cost of drafting. Let \`c\` be the draft model's
cost as a fraction of the target's:

\`\`\`
                E[tokens]           (1 - alpha^(gamma+1)) / (1 - alpha)
speedup  =  ----------------  =  ------------------------------------
             gamma * c + 1                  gamma * c + 1
\`\`\`

The denominator is the cost of one speculation round in units of target passes: \`γ\` draft passes
at cost \`c\` each, plus one verification.

**Break-even** is where the speedup equals 1:

\`\`\`
(1 - alpha^(gamma+1)) / (1 - alpha)  =  gamma * c + 1
\`\`\`

Worked cases at \`γ = 4\`:

\`\`\`
alpha = 0.8, c = 0.05 (a 1B draft for a 70B target):
    3.36 / (0.20 + 1) = 2.80x    strong win

alpha = 0.8, c = 0.15 (a 7B draft for a 70B target):
    3.36 / (0.60 + 1) = 2.10x    still good

alpha = 0.5, c = 0.15:
    1.94 / (0.60 + 1) = 1.21x    marginal

alpha = 0.4, c = 0.30:
    1.64 / (2.20)     = 0.75x    LOSS -- slower than not speculating
\`\`\`

The last row is the important one. **Speculation can make you slower**, and it does so silently —
the output is still correct, it just costs more. Any deployment should measure the realized
acceptance rate rather than assume it.

Acceptance rate is not a constant, either. It varies by domain (code drafts better than creative
prose, because it is more predictable), by position (the token after "def " is easy, the token
starting a new idea is hard), and by sampling temperature (higher temperature flattens \`p\`, which
reduces agreement). Good implementations adapt \`γ\` dynamically based on recent acceptance.`,
      ascii: '',
    },
    {
      name: 'Self-speculation: Medusa, EAGLE, and n-gram lookup',
      keyPoint: 'A separate draft model costs memory and alignment effort, so the practical methods generate drafts from the target model itself.',
      body: `Classical speculative decoding needs a second model, and that has real costs: extra
weights in memory competing with your KV cache, extra weight bytes streamed per draft step, and
the requirement that the draft be well-aligned with the target — a 1B model drafting for a 70B
model from a different family will have poor \`α\`. Self-speculation avoids all three.

**Medusa** attaches extra decoding heads to the target model's final hidden state. Head \`k\`
predicts the token \`k\` positions ahead. One forward pass produces the current token plus several
speculative future tokens, with no separate model at all. Because the heads are individually
weak, Medusa proposes multiple candidates per position and verifies them as a **tree** rather than
a single chain. The heads are cheap to train — the backbone is frozen.

The trade: independent heads cannot condition on each other, so head 3's guess does not know what
head 2 guessed. Accuracy falls off with distance.

**EAGLE** fixes that by autoregressing at the *feature* level rather than the token level. It runs
a small autoregressive head over the target model's second-to-top-layer hidden states, using both
the feature sequence and the token sequence one step shifted. Because the features are far more
informative than tokens, a very small head achieves high acceptance — EAGLE reports substantially
better acceptance rates than Medusa, and successive versions (EAGLE-2's dynamic draft trees,
EAGLE-3's training changes) have pushed it further. EAGLE is currently the strongest general
self-speculation approach and is supported in the major serving engines.

**N-gram / prompt lookup decoding** is the cheapest idea in the family and requires no model at
all. Search the existing context for the current suffix; if you find it, propose whatever followed
it last time. Zero training, zero extra parameters, near-zero cost.

It sounds too crude to work, and for open-ended chat it largely does not. But for
**high-repetition workloads it is excellent**: code editing where most of a file is unchanged,
summarization that quotes the source, RAG where the answer copies from retrieved documents,
structured output with a fixed schema. In those settings acceptance rates can be very high, and
since \`c ≈ 0\` the break-even condition is trivially satisfied. It is worth trying first precisely
because it costs nothing to try.

**Tree attention** is the shared machinery for Medusa and EAGLE. Rather than verifying one
candidate sequence, verify a tree of them in a single pass, using a carefully constructed
attention mask so each node attends only to its ancestors. Verifying 64 tree nodes costs one
forward pass, and the probability that *some* path through the tree is accepted is much higher
than for a single chain. The cost is a more complex mask and more arithmetic — again, arithmetic
you were not using.

| method | extra params | training | typical strength |
|---|---|---|---|
| draft model | a whole small model | none if one exists | general, if well aligned |
| Medusa | a few heads | light, backbone frozen | moderate |
| EAGLE | one small head | light | strongest general method |
| n-gram lookup | none | none | excellent on repetitive text |`,
      ascii: `  TREE VERIFICATION -- one forward pass covers every path

                    [prefix]
                   /        \\
              "the"          "a"
             /     \\           \\
        "cat"     "dog"       "bird"
         /                       \\
     "sat"                      "flew"

  attention mask: each node sees only its ancestors
  -> 8 candidate continuations verified in ONE pass
  -> P(some path accepted) >> P(one chain accepted)`,
    },
    {
      name: 'Why speculation stops working at high batch',
      keyPoint: 'The free arithmetic exists only because you are memory-bound; batching consumes exactly the same headroom, so the two techniques compete.',
      body: `The single most important practical caveat, and the one most often missed.

Speculation is free because decode has idle arithmetic. But **batching consumes that same idle
arithmetic** — recall from Module 4 that decode arithmetic intensity equals batch size. The two
techniques are drawing on one pool of spare compute.

\`\`\`
batch 1, no speculation:      intensity ~1     ridge 295   0.3% used
batch 1, gamma=4 speculation: intensity ~5                 1.7% used
batch 64, no speculation:     intensity ~64               22%   used
batch 64, gamma=4:            intensity ~320             OVER the ridge
\`\`\`

At batch 64 with \`γ = 4\`, verification processes \`64 × 5 = 320\` positions per pass. You have
crossed the ridge point and become **compute-bound** — at which point the extra positions are no
longer free, they cost time proportional to their count.

Now add the second problem: rejected drafts are wasted work. At batch 1 that waste falls into idle
capacity and costs nothing. At batch 64, where you are compute-bound, a rejected draft consumes
arithmetic that another sequence's real token needed. Speculation stops being free and starts
being a tax.

The empirical picture:

| batch | typical effect of speculation |
|---|---|
| 1–4 | large win — 2–3× is achievable |
| 8–32 | moderate win, shrinking with batch |
| 64+ | marginal, often a net loss |

This creates a genuine architectural tension. Throughput-oriented serving wants large batches.
Latency-oriented serving wants speculation. **You usually cannot have both**, and which you choose
should follow from your SLO:

- **Low-latency, low-concurrency** (local inference, interactive coding, a single user on
  dedicated hardware): speculate aggressively, keep the batch small.
- **High-throughput, high-concurrency** (a public API optimizing cost per token): large batches,
  speculation off or adaptive.
- **In between:** adapt. Modern schedulers monitor batch occupancy and acceptance rate and adjust
  \`γ\` per step, disabling speculation when the batch is full. Doing this well is an open problem
  — see Module 10.

One nuance worth adding: reasoning models change the calculation. A long chain-of-thought trace
is thousands of decode tokens whose *content* is never shown to the user, so per-token latency
matters less than total time, and the tokens are often highly predictable. That combination
favours aggressive speculation even at moderate batch, and it is an active area.`,
      ascii: '',
    },
  ],

  mathLab: {
    prompt: `**Part 1 — verify the acceptance rule preserves the distribution.**
Vocabulary of 4 tokens. Draft \`q = [0.5, 0.3, 0.15, 0.05]\`, target \`p = [0.3, 0.4, 0.2, 0.1]\`.

  a) For each token, the probability it is drafted **and** accepted.
  b) The total rejection probability.
  c) The residual distribution \`p'\`.
  d) Confirm that \`P(emit z)\` equals \`p(z)\` for all four tokens. Show the arithmetic.
  e) What is the acceptance rate \`α\` for this pair? Relate it to \`Σ min(p, q)\`.

**Part 2 — the speedup formula.**
Target: Llama-3-70B. Draft: Llama-3-8B. Assume the draft's cost per step is proportional to
parameters, so \`c = 8.03/70.6 = 0.114\`.

  a) Expected tokens per verification for \`α = 0.7\` at \`γ = 1, 2, 3, 4, 5, 8\`.
  b) Speedup for each. Which \`γ\` is optimal?
  c) Repeat for \`α = 0.85\`. Does the optimal \`γ\` move? In which direction, and why?
  d) With \`α = 0.7\`, how small must \`c\` be for \`γ = 8\` to beat \`γ = 3\`?

**Part 3 — break-even.**
  a) With \`γ = 4\` and \`c = 0.15\`, what is the minimum \`α\` for any speedup at all?
  b) With \`α = 0.6\`, what is the maximum \`c\` that still gives a speedup at \`γ = 4\`?
  c) A 7B draft for a 13B target gives \`c = 0.54\` and \`α = 0.8\`. Speedup at \`γ = 2\`? Comment.

**Part 4 — the batching interaction.**
Llama-3-70B on 4×H100 (aggregate bandwidth 13.4 TB/s, aggregate dense BF16 3,956 TFLOP/s, so a
ridge point of 295). Weights 141 GB. Ignore KV cache.

  a) Arithmetic intensity of plain decode at batch 1, 8, 32, 64.
  b) With \`γ = 4\` speculation, intensity becomes roughly \`5 ×\` batch. Recompute for each.
  c) At which batch size does speculation push you past the ridge?
  d) At batch 64 with \`γ = 4\`, you are compute-bound. Estimate the time for one verification pass
     from the FLOP side and compare it to the memory-bound time. What has happened to the "free
     arithmetic" argument?`,

    solution: `**Part 1**

a) \`P(draft z and accept) = q(z) × min(1, p(z)/q(z)) = min(p(z), q(z))\`:

\`\`\`
z=0: min(0.30, 0.50) = 0.30      (ratio p/q = 0.6, accept w.p. 0.6: 0.5 x 0.6 = 0.30)
z=1: min(0.40, 0.30) = 0.30      (p >= q, accept always: 0.3 x 1.0 = 0.30)
z=2: min(0.20, 0.15) = 0.15      (p >= q, accept always)
z=3: min(0.10, 0.05) = 0.05      (p >= q, accept always)
                       ----
             sum min = 0.80
\`\`\`

b) \`P(reject) = 1 − 0.80 = 0.20\`

c) Residual \`max(0, p − q)\`:

\`\`\`
z=0: max(0, 0.30-0.50) = 0.00
z=1: max(0, 0.40-0.30) = 0.10
z=2: max(0, 0.20-0.15) = 0.05
z=3: max(0, 0.10-0.05) = 0.05
                         ----
                  sum =  0.20      <- equals P(reject), as the identity requires

p' = [0.00, 0.50, 0.25, 0.25]
\`\`\`

d) \`P(emit z) = min(p,q) + P(reject) × p'(z)\`:

\`\`\`
z=0: 0.30 + 0.20 x 0.00 = 0.30   = p(0)  ✓
z=1: 0.30 + 0.20 x 0.50 = 0.40   = p(1)  ✓
z=2: 0.15 + 0.20 x 0.25 = 0.20   = p(2)  ✓
z=3: 0.05 + 0.20 x 0.25 = 0.10   = p(3)  ✓
\`\`\`

Exact for every token, and note it did not depend on \`q\` being any good — a different draft
would change the acceptance rate but not this table.

e) \`α = Σ min(p, q) = 0.80\`. In general the acceptance rate is exactly the overlap
\`Σ_z min(p(z), q(z))\`, which equals \`1 − TV(p, q)\` where TV is total variation distance. **A
better-aligned draft model is literally one whose distribution is closer to the target's in TV
distance** — which is a satisfying way to state what "well aligned" means.

**Part 2**

With \`c = 0.114\`, \`E = (1 − α^(γ+1))/(1 − α)\`, speedup \`= E/(γc + 1)\`.

a) and b), at \`α = 0.7\`:

\`\`\`
gamma   E[tokens]   cost (gamma*c+1)   speedup
  1       1.700          1.114          1.526
  2       2.190          1.228          1.783
  3       2.533          1.342          1.887
  4       2.773          1.456          1.905   <- optimal
  5       2.941          1.570          1.873
  8       3.212          1.912          1.680
\`\`\`

Working one: \`γ=4\`: \`E = (1 − 0.7^5)/0.3 = (1 − 0.16807)/0.3 = 2.773\`; cost \`= 4(0.114) + 1 =
1.456\`; speedup \`= 1.905\`.

**Optimal γ = 4**, giving 1.91×.

c) At \`α = 0.85\`:

\`\`\`
gamma   E[tokens]   cost    speedup
  1       1.850     1.114    1.661
  2       2.573     1.228    2.095
  3       3.187     1.342    2.375
  4       3.709     1.456    2.547
  5       4.153     1.570    2.645
  8       5.130     1.912    2.683   <- optimal
 10       5.591     2.140    2.613
\`\`\`

**Optimal γ moves up to about 8**, giving 2.68×. Higher acceptance means drafts survive longer, so
longer speculation pays. The general rule: **optimal draft length increases with acceptance rate**,
which is exactly why adaptive-\`γ\` schemes track recent acceptance.

d) At \`α = 0.7\`, \`E(3) = 2.533\` and \`E(8) = 3.212\`. For \`γ=8\` to beat \`γ=3\`:

\`\`\`
3.212 / (8c + 1)  >  2.533 / (3c + 1)
3.212(3c + 1)     >  2.533(8c + 1)
9.636c + 3.212    >  20.264c + 2.533
0.679             >  10.628c
c                 <  0.0639
\`\`\`

You would need a draft costing under **6.4%** of the target — roughly a 4.5B draft for a 70B
target, or smaller. With the 8B draft at \`c = 0.114\`, \`γ = 8\` is worse than \`γ = 3\`.

**Part 3**

a) \`γ = 4, c = 0.15\`, so cost \`= 1.6\`. Need \`E ≥ 1.6\`:

\`\`\`
(1 - alpha^5)/(1 - alpha) = 1.6
\`\`\`

Solving numerically: at \`α = 0.35\`, \`E = (1 − 0.00525)/0.65 = 1.530\`. At \`α = 0.40\`,
\`E = (1 − 0.01024)/0.60 = 1.650\`. At \`α = 0.38\`, \`E = (1 − 0.00792)/0.62 = 1.600\`.

**Minimum α ≈ 0.38.**

b) \`α = 0.6, γ = 4\`: \`E = (1 − 0.6^5)/0.4 = (1 − 0.07776)/0.4 = 2.306\`. Need
\`2.306 / (4c + 1) > 1\`:

\`\`\`
4c + 1 < 2.306  ->  c < 0.3264
\`\`\`

**Maximum c ≈ 0.33** — the draft may cost up to a third of the target.

c) \`α = 0.8, γ = 2, c = 0.54\`:

\`\`\`
E = (1 - 0.8^3)/0.2 = (1 - 0.512)/0.2 = 2.44
cost = 2(0.54) + 1 = 2.08
speedup = 2.44 / 2.08 = 1.17x
\`\`\`

Only 1.17× despite an excellent acceptance rate of 0.8. **The draft is too expensive relative to
the target.** A 7B draft for a 13B target is a bad pairing — the models are too close in size. The
general guidance that follows: you want roughly a **10× or greater** size gap, which is why 1B
drafts for 70B targets are the classic configuration, and why self-speculation (where \`c\` is
tiny) often beats a separate draft model even at lower \`α\`.

**Part 4**

a) Plain decode intensity equals batch size: **1, 8, 32, 64.**

b) With \`γ = 4\`, verification processes \`5 ×\` batch positions per pass:

\`\`\`
batch 1:   ~5
batch 8:   ~40
batch 32:  ~160
batch 64:  ~320
\`\`\`

c) Ridge point is 295. \`5 × batch > 295\` gives \`batch > 59\`. **Speculation pushes you past the
ridge at around batch 60**, whereas plain decode would not reach it until batch 295.

d) At batch 64, \`γ = 4\` — 320 positions per verification pass:

\`\`\`
memory time:  141 GB / 13,400 GB/s                  = 10.5 ms
compute time: 2 x 70.6e9 x 320 / 3,956e12           = 11.4 ms
\`\`\`

Compute time now **exceeds** memory time. The pass is compute-bound, so those extra positions are
no longer riding in idle capacity — each one costs real time.

And the arithmetic is worse than that comparison suggests, because with \`α = 0.7\` you only
harvest 2.77 tokens per sequence per pass out of the 5 positions you paid for. You are spending
compute-bound time on 320 positions to obtain \`64 × 2.77 = 177\` tokens, versus a plain
memory-bound pass costing 10.5 ms for 64 tokens:

\`\`\`
speculative:  177 tokens / 11.4 ms = 15.5 tokens/ms
plain:         64 tokens / 10.5 ms =  6.1 tokens/ms
\`\`\`

Still ahead here — but the margin is collapsing, and pushing to batch 128 or \`γ = 8\` inverts it.
**The "free arithmetic" argument holds only while you are on the memory-bound side of the ridge.**
Speculation and batching draw on the same finite pool of spare compute, and past the ridge that
pool is empty.`,
  },

  codeLab: {
    goal: `Implement the modified rejection sampling scheme and verify empirically that the output
distribution matches the target's — that is the claim the whole technique rests on, and it is
worth proving to yourself rather than trusting.

Then build the speedup model and find the break-even surface, so you can tell in advance whether
speculation will pay for a given deployment.`,
    code: `"""
Speculative decoding: correctness, then economics.

    pip install numpy
"""
import numpy as np

rng = np.random.default_rng(17)


# ==========================================================================
# Part A -- the acceptance rule, and proof by simulation
# ==========================================================================
def speculative_step(p, q, rng):
    """One draft token, verified. Returns the emitted token.

    p: target distribution, q: draft distribution (both over the vocabulary)
    """
    x = rng.choice(len(q), p=q)                      # draft samples from q

    if rng.random() < min(1.0, p[x] / q[x]):         # accept with prob min(1, p/q)
        return x, True

    residual = np.maximum(0.0, p - q)                # the mass p wanted that q undersupplied
    s = residual.sum()
    if s <= 0:                                       # p == q everywhere: cannot happen
        return int(rng.choice(len(p), p=p)), False
    return int(rng.choice(len(p), p=residual / s)), False


def naive_step(p, q, rng):
    """The WRONG rule: take the draft if the target's argmax agrees, else target's token.
    Included so you can see the bias it introduces."""
    x = rng.choice(len(q), p=q)
    if x == p.argmax():
        return x, True
    return int(rng.choice(len(p), p=p)), False


print("=== Part A: does the acceptance rule preserve the target distribution? ===")
p = np.array([0.30, 0.40, 0.20, 0.10])
q = np.array([0.50, 0.30, 0.15, 0.05])

N = 400_000
counts = np.zeros(4)
accepts = 0
for _ in range(N):
    tok, acc = speculative_step(p, q, rng)
    counts[tok] += 1
    accepts += acc
emp = counts / N

print(f"  target p     {np.array2string(p, precision=4)}")
print(f"  draft  q     {np.array2string(q, precision=4)}")
print(f"  empirical    {np.array2string(emp, precision=4)}")
print(f"  max abs err  {np.abs(emp - p).max():.4f}   (sampling noise ~ {1/np.sqrt(N):.4f})")
print(f"  acceptance   {accepts/N:.4f}   predicted sum min(p,q) = {np.minimum(p,q).sum():.4f}")

counts = np.zeros(4)
for _ in range(N):
    tok, _ = naive_step(p, q, rng)
    counts[tok] += 1
print(f"\\n  the NAIVE rule gives {np.array2string(counts/N, precision=4)}")
print(f"  max abs err  {np.abs(counts/N - p).max():.4f}   <- biased, and not by a little")

# does it hold for arbitrary distributions?
print("\\n  checking 200 random (p, q) pairs over a 50-token vocabulary...")
worst = 0.0
for _ in range(200):
    pv = rng.dirichlet(np.ones(50) * rng.uniform(0.2, 3))
    qv = rng.dirichlet(np.ones(50) * rng.uniform(0.2, 3))
    c = np.zeros(50)
    for _ in range(4000):
        t, _ = speculative_step(pv, qv, rng)
        c[t] += 1
    worst = max(worst, np.abs(c / 4000 - pv).max())
print(f"  worst deviation across all pairs: {worst:.4f} (noise floor ~{1/np.sqrt(4000):.4f})")
print("  -> holds regardless of how badly q matches p. The draft affects SPEED only.")


# ==========================================================================
# Part B -- multi-token speculation and the yield formula
# ==========================================================================
def speculate_gamma(p_fn, q_fn, gamma, rng):
    """Draft gamma tokens, verify all at once. Returns the accepted tokens.

    p_fn(prefix) / q_fn(prefix) return distributions -- here they are stubs, but
    the accept/reject logic is exactly what a real implementation does.
    """
    prefix, drafts, qs = [], [], []
    for _ in range(gamma):                            # DRAFT sequentially
        qd = q_fn(prefix + drafts)
        x = int(rng.choice(len(qd), p=qd))
        drafts.append(x)
        qs.append(qd)

    # VERIFY: one target pass gives p at every drafted position
    ps = [p_fn(prefix + drafts[:i]) for i in range(gamma + 1)]

    out = []
    for i in range(gamma):
        x, pd, qd = drafts[i], ps[i], qs[i]
        if rng.random() < min(1.0, pd[x] / qd[x]):
            out.append(x)
        else:
            resid = np.maximum(0.0, pd - qd)
            out.append(int(rng.choice(len(pd), p=resid / resid.sum())))
            return out                                # stop at first rejection
    out.append(int(rng.choice(len(ps[gamma]), p=ps[gamma])))   # bonus token
    return out


def make_pair(alpha_target, vocab=32, rng=rng):
    """Build (p, q) whose overlap sum min(p,q) is approximately alpha_target."""
    base = rng.dirichlet(np.ones(vocab))
    noise = rng.dirichlet(np.ones(vocab))
    lo, hi = 0.0, 1.0
    for _ in range(40):                               # bisect on the mixing weight
        w = (lo + hi) / 2
        qv = (1 - w) * base + w * noise
        if np.minimum(base, qv).sum() > alpha_target:
            lo = w
        else:
            hi = w
    return base, (1 - lo) * base + lo * noise


print("\\n=== Part B: expected tokens per verification ===")
print(f"  {'alpha':>7} {'gamma':>6} {'predicted':>11} {'simulated':>11}")
for alpha in (0.5, 0.7, 0.9):
    for gamma in (1, 2, 4, 8):
        pv, qv = make_pair(alpha)
        a_real = np.minimum(pv, qv).sum()
        pred = (1 - a_real ** (gamma + 1)) / (1 - a_real)
        got = np.mean([len(speculate_gamma(lambda _: pv, lambda _: qv, gamma, rng))
                       for _ in range(3000)])
        print(f"  {a_real:>7.3f} {gamma:>6} {pred:>11.3f} {got:>11.3f}")


# ==========================================================================
# Part C -- the economics
# ==========================================================================
def expected_tokens(alpha, gamma):
    if alpha >= 1.0:
        return gamma + 1
    return (1 - alpha ** (gamma + 1)) / (1 - alpha)


def speedup(alpha, gamma, c):
    return expected_tokens(alpha, gamma) / (gamma * c + 1)


print("\\n=== Part C: speedup surface (c = 0.114, an 8B draft for a 70B target) ===")
c = 8.03 / 70.6
print(f"  {'alpha':>7} " + " ".join(f"g={g:<5}" for g in (1, 2, 3, 4, 6, 8, 12)) + "   best")
for alpha in (0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95):
    row, best_g, best_s = [], 0, 0
    for g in (1, 2, 3, 4, 6, 8, 12):
        s = speedup(alpha, g, c)
        row.append(f"{s:<7.2f}")
        if s > best_s:
            best_s, best_g = s, g
    print(f"  {alpha:>7.2f} " + " ".join(row) + f"   g={best_g} ({best_s:.2f}x)")

print("\\n=== break-even: minimum alpha for a speedup at all ===")
print(f"  {'c':>8} " + " ".join(f"g={g:<7}" for g in (1, 2, 4, 8)))
for c_ in (0.02, 0.05, 0.10, 0.20, 0.35, 0.50):
    cells = []
    for g in (1, 2, 4, 8):
        lo, hi = 0.0, 0.999
        for _ in range(60):
            mid = (lo + hi) / 2
            if speedup(mid, g, c_) > 1.0:
                hi = mid
            else:
                lo = mid
        cells.append(f"{hi:<9.3f}" if hi < 0.99 else "never    ")
    print(f"  {c_:>8.2f} " + " ".join(cells))

print("\\n=== the batching interaction (ridge point 295) ===")
print(f"  {'batch':>6} {'plain I':>9} {'g=4 I':>8} {'g=8 I':>8}  regime at g=4")
for b in (1, 4, 16, 32, 64, 128, 256):
    i1, i4, i8 = b, b * 5, b * 9
    reg = "compute-bound (speculation no longer free)" if i4 > 295 else "memory-bound (speculation free)"
    print(f"  {b:>6} {i1:>9} {i4:>8} {i8:>8}  {reg}")

# --- TODO for you ---
#   1. Add temperature. Recompute alpha as sum min(p,q) after applying the same
#      temperature to both. Plot alpha vs temperature -- it should fall, which
#      is why speculation works better at low temperature.
#   2. Model tree verification: with a branching factor b and depth d, you verify
#      b^d paths in one pass. Estimate P(some path accepted) and redo the speedup.
#   3. Write an adaptive-gamma controller: track a rolling acceptance rate and
#      pick gamma to maximize predicted speedup each step. Test it on a workload
#      whose alpha shifts halfway through.
`,
    expect: `**Part A** is the important one. The empirical distribution matches \`p\` to within
sampling noise — a max absolute error around \`0.002\` at 400,000 samples, which is the
\`1/√N ≈ 0.0016\` noise floor. The measured acceptance rate lands on \`Σ min(p,q) = 0.80\`.

The naive rule shows an error of \`0.05–0.15\` — clearly, visibly biased. Seeing the two side by
side makes it obvious why the rejection scheme is necessary rather than pedantic.

The 200-random-pair check confirms the guarantee holds for arbitrary \`q\`. **The draft model
cannot corrupt your output no matter how bad it is.**

**Part B** shows simulated yields matching \`(1 − α^(γ+1))/(1 − α)\` closely — within a few percent
at 3,000 trials.

**Part C** reproduces the math lab: at \`α = 0.7\` the optimum is around \`γ = 4\` at ~1.9×, and at
\`α = 0.9\` the optimum moves to \`γ = 8\` or beyond at ~3×. The break-even table shows that with an
expensive draft (\`c = 0.5\`) you need \`α > 0.6\` even at \`γ = 1\`, and \`γ = 8\` may never pay.

The final table is the one to remember: at \`γ = 4\`, speculation crosses the ridge point at around
**batch 60**. Past that, the free-arithmetic argument no longer holds.`,
    stretch: `Run it against a real model pair. Use \`gpt2\` as the target and \`distilgpt2\` as the
draft (same tokenizer, which is required), and measure the actual acceptance rate on three
different prompt types: natural prose, Python code, and a repeated-structure prompt such as a JSON
list. You should find code and structured text accepting substantially better than prose, because
they are more predictable. Then implement n-gram lookup speculation and compare — on the
repeated-structure prompt it will likely beat the draft model outright, at zero cost.`,
  },

  papers: [
    {
      title: 'Fast Inference from Transformers via Speculative Decoding',
      by: 'Leviathan, Kalman & Matias (Google), 2022',
      url: 'https://arxiv.org/abs/2211.17192',
      why: 'The original. Contains the acceptance rule, the distribution-preservation proof, and the expected-yield analysis this module is built on.',
      frame: `**Section 2.1** defines the algorithm and **Theorem 3.5** with its appendix proof is
the payload — work through it yourself; it is half a page and it is the reason the technique is
adoptable. **Section 3.2** derives the expected number of accepted tokens. Pay attention to their
discussion of choosing \`γ\`, which is where the practical guidance lives.`,
    },
    {
      title: 'Accelerating Large Language Model Decoding with Speculative Sampling',
      by: 'Chen et al. (DeepMind), 2023',
      url: 'https://arxiv.org/abs/2302.01318',
      why: 'Independent concurrent discovery, with a cleaner presentation of the sampling argument and results on Chinchilla at scale.',
      frame: 'Read alongside Leviathan — the two derivations of the same result illuminate each other. This one is more direct about the modified rejection sampling and has a clearer treatment of what happens with different sampling parameters.',
    },
    {
      title: 'Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads',
      by: 'Cai et al., 2024',
      url: 'https://arxiv.org/abs/2401.10774',
      why: 'Self-speculation via extra decoding heads, removing the need for a separate draft model. Also introduces tree attention for multi-candidate verification.',
      frame: 'Read Section 3 for the heads and Section 3.2 for tree attention and the mask construction — the tree idea generalizes beyond Medusa and is the more durable contribution. Note the two training regimes (frozen backbone vs joint) and what each costs.',
    },
    {
      title: 'EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty',
      by: 'Li et al., 2024',
      url: 'https://arxiv.org/abs/2401.15077',
      why: 'The strongest general self-speculation method. Autoregresses at the feature level rather than the token level, which is a genuinely better idea than Medusa\'s independent heads.',
      frame: `The key insight is in Section 3: features are more predictable than tokens, but
feature-level autoregression alone is ambiguous because the sampled token is not determined — so
EAGLE conditions on the token sequence shifted one step. Understand why that shift is necessary.
Follow up with EAGLE-2 for dynamic draft trees.`,
    },
    {
      title: 'Break the Sequential Dependency of LLM Inference Using Lookahead Decoding',
      by: 'Fu et al., 2024',
      url: 'https://arxiv.org/abs/2402.02057',
      why: 'A different route: generate and verify n-grams in parallel using Jacobi iteration, with no draft model and no extra training at all.',
      frame: 'Read Sections 2 and 3. The Jacobi-iteration framing of autoregressive decoding as a fixed-point problem is a genuinely different way to look at the sequential dependency, and worth having in your head even if you never deploy it.',
    },
  ],

  checkpoint: {
    claim: `You can explain why speculative decoding is free performance rather than a
quality/speed trade-off — including the proof — and you can compute in advance whether it will
pay for a given model pair, acceptance rate and batch size.`,
    questions: [
      {
        q: 'Prove that the acceptance rule preserves the target distribution.',
        a: `Let \`q\` be the draft distribution and \`p\` the target's. A token \`z\` is emitted either
by being drafted and accepted, or by being sampled from the residual after a rejection. First
path: \`P = q(z) × min(1, p(z)/q(z)) = min(p(z), q(z))\`. Rejection probability is
\`1 − Σ_w min(p(w), q(w))\`, and since both distributions sum to 1,
\`Σ_w max(0, p(w) − q(w)) = 1 − Σ_w min(p(w), q(w))\` — the residual's normalizer equals the
rejection probability exactly, so they cancel. Therefore
\`P(emit z) = min(p(z), q(z)) + max(0, p(z) − q(z))\`. If \`p(z) ≤ q(z)\` that is \`p(z) + 0\`; if
\`p(z) > q(z)\` it is \`q(z) + p(z) − q(z) = p(z)\`. Either way \`P(emit z) = p(z)\`, for any \`q\`
whatsoever.`,
      },
      {
        q: 'Your acceptance rate is 0.6 and your draft costs 40% of your target. Should you speculate?',
        a: `Probably not. At \`γ = 2\`: \`E = (1 − 0.6³)/0.4 = 2.06\`, cost \`= 2(0.4) + 1 = 1.8\`, so
speedup \`= 1.14×\`. At \`γ = 4\`: \`E = 2.31\`, cost \`= 2.6\`, speedup \`= 0.89×\` — an actual slowdown.
The best you can do is around 1.1× at \`γ = 1–2\`, which is not worth the implementation and
operational complexity. The problem is \`c = 0.4\`: the draft is far too expensive relative to the
target. You want roughly a 10× size gap. Better options here would be a much smaller draft model,
self-speculation (EAGLE, where \`c\` is tiny), or n-gram lookup if the workload is repetitive.`,
      },
      {
        q: 'Why does speculation stop helping at large batch size?',
        a: `Because it and batching draw on the same resource: idle arithmetic. Decode is free to
speculate only because arithmetic intensity is far below the ridge point — at batch 1 you use
about 0.3% of an H100's compute. But intensity equals batch size, so batching consumes that
headroom too. With \`γ = 4\`, verification processes \`5 × batch\` positions, so you cross the
295 ridge at around batch 60. Past that you are compute-bound and the extra positions cost real
time rather than riding in idle capacity — and rejected drafts now consume arithmetic that another
sequence's real token needed. Speculation is a latency technique for low-concurrency serving;
batching is a throughput technique for high-concurrency serving, and you generally pick one.`,
      },
      {
        q: 'Why does the optimal draft length increase with the acceptance rate?',
        a: `Because the marginal value of the \`(γ+1)\`-th draft token is the probability that all
\`γ\` before it were accepted, which is \`α^γ\` — it decays geometrically in \`γ\` at a rate set by
\`α\`. At \`α = 0.5\` the fifth draft token is worth \`0.5⁴ = 6%\` of a token while costing a full
draft step, so long speculation is wasted. At \`α = 0.9\` it is worth \`0.9⁴ = 66%\`, so drafting
further still pays. Concretely with \`c = 0.114\`: at \`α = 0.7\` the optimum is \`γ ≈ 4\` (1.91×),
and at \`α = 0.85\` it moves to \`γ ≈ 8\` (2.68×). This is why production implementations track a
rolling acceptance rate and adapt \`γ\` rather than fixing it.`,
      },
      {
        q: 'When would n-gram lookup beat a trained draft model?',
        a: `Whenever the output substantially repeats the context, because then the lookup's
acceptance rate is high and its cost is essentially zero — so the break-even condition is
trivially satisfied and the speedup approaches the raw yield. Good cases: code editing where most
of a file is unchanged, summarization that quotes the source, RAG where answers copy from
retrieved documents, structured output with a fixed schema, and any diff-style task. A trained
draft model has \`c > 0\` and needs a meaningfully higher \`α\` to justify itself. It is also worth
trying first for the practical reason that it needs no extra weights, no training, no alignment,
and no memory — if it works for your workload you are done.`,
      },
      {
        q: 'A colleague applies top-p 0.9 to the target model but samples the draft at full distribution. What is wrong?',
        a: `The distribution-preservation proof assumes \`p\` and \`q\` are the distributions actually
being sampled from. If the target's \`p\` is truncated by top-p but the draft's \`q\` is not, then \`q\`
can place mass on tokens where \`p\` is exactly zero. Those drafts are always rejected — wasting
speculation — and more importantly the acceptance/residual arithmetic no longer composes to the
intended distribution, so you lose the exactness guarantee. Both models must see the same
processed distribution: apply identical temperature, top-p, top-k and penalties to each before
the accept/reject step. This is a common bug precisely because the output still looks fine — it is
plausible text, just not samples from the distribution you specified.`,
      },
    ],
  },

  pitfalls: [
    {
      wrong: 'Accept the draft token if the target model would have picked it too.',
      right: `That rule is biased. It gives the draft model an effective veto over which tokens
are considered, skewing output toward the draft's preferences. The correct rule accepts with
probability \`min(1, p/q)\` and, on rejection, samples from the normalized \`max(0, p − q)\`. Only
that composition provably yields \`p\`. The naive version produces plausible text, which is exactly
what makes the bug hard to notice.`,
    },
    {
      wrong: 'A better draft model always means a better speedup.',
      right: `Speedup is \`E(α, γ) / (γc + 1)\`, so a draft that raises \`α\` but also raises \`c\` can
easily be worse. A 7B draft for a 13B target with \`α = 0.8\` gives only 1.17×, because \`c = 0.54\`
dominates. You want roughly a 10× size gap — or self-speculation, where \`c\` is near zero and a
lower \`α\` still wins.`,
    },
    {
      wrong: 'Turning on speculation in a high-throughput serving deployment.',
      right: `Speculation converts idle arithmetic into tokens, and at large batch there is no
idle arithmetic left — batching already consumed it. At \`γ = 4\` you cross an H100's ridge point
around batch 60, after which extra draft positions cost real time and rejected drafts steal
compute from real tokens. Speculation is for low-concurrency, latency-sensitive serving. Measure
before enabling it at scale.`,
    },
    {
      wrong: 'Assuming a published acceptance rate transfers to your workload.',
      right: `Acceptance varies substantially by domain (code drafts far better than creative
prose), by sampling temperature (higher temperature flattens \`p\`, reducing overlap with \`q\`), and
by position within a sequence. A configuration measured at 0.8 on one benchmark can be 0.5 on
yours, which moves you from a 2.8× win to a 1.2× marginal one. Instrument the realized rate in
production and adapt \`γ\` to it.`,
    },
  ],

  glossary: [
    { term: 'draft model', def: 'A small, cheap model that proposes candidate tokens for the target model to verify.' },
    { term: 'acceptance rate (alpha)', def: 'Probability a drafted token survives verification. Equals sum min(p, q), i.e. 1 minus the total variation distance between draft and target.' },
    { term: 'gamma', def: 'Draft length: how many tokens are proposed before each verification pass.' },
    { term: 'modified rejection sampling', def: 'Accept with probability min(1, p/q); on rejection sample from normalized max(0, p - q). Provably yields p.' },
    { term: 'residual distribution', def: 'The normalized positive part of p - q. The mass the target wanted that the draft undersupplied.' },
    { term: 'bonus token', def: 'The extra token available when all gamma drafts are accepted, since the verification pass already computed p at the final position.' },
    { term: 'self-speculation', def: 'Generating drafts from the target model itself via extra heads or a small feature-level head, avoiding a separate draft model.' },
    { term: 'Medusa', def: 'Self-speculation with multiple independent decoding heads predicting several positions ahead, verified as a tree.' },
    { term: 'EAGLE', def: 'Self-speculation by autoregressing over the target\'s hidden features rather than tokens. The strongest general method.' },
    { term: 'tree attention', def: 'Verifying many candidate continuations in one pass using a mask where each node attends only to its ancestors.' },
    { term: 'n-gram lookup', def: 'Drafting by searching the context for the current suffix and proposing what followed it. Free, and excellent on repetitive text.' },
  ],
};
