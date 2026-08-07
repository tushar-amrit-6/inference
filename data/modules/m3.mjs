export default {
  n: 3,
  slug: 'decoding-and-sampling',
  title: 'DECODING AND SAMPLING',
  tagline: 'The model gives you a distribution. Turning it into a token is a separate design problem.',
  hours: '5–7 hours',
  prereqs: ['Module 0', 'Module 1', 'Basic probability'],

  bigIdea: `A language model does not output text. It outputs a vector of 128,256 logits. Every
choice about how to turn that vector into a token is *yours*, sits outside the model, and can be
changed without retraining anything.

This is the cheapest lever in the entire stack and the most frequently misconfigured. Identical
weights with different sampling parameters produce output that ranges from robotically repetitive
to incoherent. Most complaints that "the model is bad at X" are, on inspection, complaints about
a temperature setting.

The module also covers two things that surprise people. First, **maximizing likelihood produces
bad text** — this is an empirical finding with a solid explanation, and it is why beam search,
which dominated machine translation, lost decisively for open-ended generation. Second,
**inference is not deterministic even at temperature 0**, for reasons that have nothing to do with
sampling and everything to do with floating-point arithmetic on parallel hardware.

Sampling also has a systems cost. Constrained decoding, logprobs, and penalties all add per-token
work in the hot loop, and at batch 256 a naive sampler can become a measurable fraction of your
step time.`,

  concepts: [
    {
      name: 'Greedy decoding, and why the highest-probability token is not the goal',
      keyPoint: 'Greedy is locally optimal and globally arbitrary: picking the best token at each step does not produce the best sequence.',
      body: `The simplest rule: take \`argmax(logits)\`. Deterministic, free, and reproducible.

It has two distinct problems, and they are worth separating.

**It is locally greedy.** The most likely first token may lead into a region of low-probability
continuations. The sequence "The" → "cat" might have higher joint probability than "A" → "cat",
but if \`P(A) > P(The)\` greedy commits to "A" and never looks back. This is the standard search
critique, and it motivates beam search.

**It produces degenerate text.** This one is more surprising and more important. Greedy output on
open-ended prompts falls into loops — repeating a phrase, then a sentence, then a paragraph,
indefinitely. Not occasionally: reliably. Holtzman et al. documented this carefully in 2019, and
the mechanism is a positive feedback loop. A repeated phrase raises the model's estimate that it
is in a repetitive context, which raises the probability of repeating again.

The deep point is that **high probability is not the objective**. Human text is *not* the most
likely text under a language model. Holtzman's Figure 2 is the memorable evidence: the per-token
probability of real human writing fluctuates wildly, dipping low constantly, while beam-search
output sits at a consistently high probability. Real language is full of surprise — that is what
makes it informative. A decoder that maximizes likelihood systematically strips the surprise out.

Where greedy is still right:

- **Reproducibility matters** — evaluation, regression tests, debugging.
- **The output space is narrow** — extraction, classification, format conversion. If there is one
  correct answer, sampling can only hurt.
- **Speculative decoding verification** — the target model's greedy choice defines acceptance.

Where it is wrong: anything open-ended. Creative writing, dialogue, brainstorming, and — perhaps
counterintuitively — long chains of reasoning, where a little diversity helps escape a bad line
of argument.`,
      ascii: '',
    },
    {
      name: 'Beam search, and why it lost',
      keyPoint: 'Beam search finds higher-probability sequences, and higher-probability sequences are exactly what you do not want in open-ended generation.',
      body: `Beam search keeps the \`k\` highest-probability *partial sequences* at each step rather
than one. Expand all \`k\`, score all \`k × V\` continuations by cumulative log-probability, keep the
best \`k\`. It is a heuristic search for the maximum-probability sequence, and it works — it
reliably finds sequences with higher joint probability than greedy.

It dominated neural machine translation for years, and still does. So why is it absent from
essentially every chat model's default configuration?

**Because it succeeds at the wrong objective.** Higher probability means blander, more generic,
more repetitive text. Beam search finds the safest possible continuation, which for open-ended
generation is exactly the failure mode. The classic symptom: raise the beam width and output
quality gets *worse*, monotonically. That is the signature of a search that is working correctly
on a bad objective.

The task distinction is sharp and worth internalizing:

| task | output space | beam search |
|---|---|---|
| translation | narrow — one meaning, few phrasings | **helps** |
| summarization | fairly narrow | often helps |
| code completion | narrow, constrained | can help |
| dialogue | wide open | **hurts** |
| creative writing | wide open | **hurts badly** |

There is also a systems argument, and it is nearly as decisive. Beam search multiplies your
memory and compute by \`k\`: you maintain \`k\` KV caches per request and run \`k\` sequences through
every forward pass. From Module 2, KV cache is what caps batch size — so beam width 4 cuts your
concurrency by roughly 4×. You pay 4× the cost for output most users rate as worse. In a
throughput-driven serving business that argument alone ends the discussion.

(Beam search does have one elegant systems property: the beams share a prefix, so with
copy-on-write block sharing under PagedAttention the memory cost is far below 4×. vLLM implements
this. It softens the cost argument but not the quality one.)`,
      ascii: '',
    },
    {
      name: 'Temperature: sharpening and flattening',
      keyPoint: 'Temperature divides the logits before softmax, so it rescales the distribution multiplicatively in log-space — it does not add or remove options.',
      body: `Temperature is a single division applied to the logits before the softmax:

\`\`\`
p_i = exp(z_i / T) / sum_j exp(z_j / T)
\`\`\`

- \`T = 1\` — the model's own distribution, unmodified.
- \`T < 1\` — sharpens. Gaps between logits are magnified, high-probability tokens get more
  probability. As \`T → 0\` it converges to greedy.
- \`T > 1\` — flattens. Gaps shrink, the tail gets more mass. As \`T → ∞\` it converges to uniform
  over the whole vocabulary.

Work a concrete example. Logits \`[4.0, 3.0, 1.0, 0.0]\`:

| T | p1 | p2 | p3 | p4 |
|---|---|---|---|---|
| 0.5 | 0.867 | 0.117 | 0.002 | 0.000 |
| 1.0 | 0.643 | 0.236 | 0.032 | 0.012 |
| 1.5 | 0.526 | 0.270 | 0.071 | 0.036 |
| 2.0 | 0.456 | 0.277 | 0.102 | 0.062 |

Two things to notice. At \`T = 0.5\` the top token has 87% of the mass and the fourth is
effectively unreachable. At \`T = 2.0\` the fourth token — which the model gave 1.2% — now has 6.2%,
a **5× increase in the junk**.

That is the failure mode at high temperature, and it is why temperature alone is a poor
diversity knob. A 128,256-token vocabulary has a *very* long tail. Even if each tail token has
tiny probability, there are 128,000 of them, and flattening the distribution hands them
collectively a large share. Sample from that often enough and you get a token that derails the
sequence — and because generation is autoregressive, one bad token poisons everything after it.

This is why temperature is almost always combined with a **truncation** method. Truncation
removes the tail; temperature reshapes what remains.

**Order of operations matters and is a real source of confusion between libraries.** Most
implementations (including Hugging Face and vLLM) apply temperature *first*, then truncate. If
you truncate first and then apply temperature, the same nominal settings give different behaviour
— because top-p computes its cutoff on the sharpened or flattened distribution rather than the
original. When comparing sampling configurations across systems, check the order before
concluding the models differ.`,
      ascii: '',
    },
    {
      name: 'Truncation: top-k, top-p, and min-p',
      keyPoint: 'Top-k truncates by rank, top-p by cumulative mass, min-p by relative probability — and only the last adapts correctly to how confident the model is.',
      body: `All three zero out part of the distribution and renormalize what is left. They differ
in how they draw the line.

**Top-k.** Keep the \`k\` highest-probability tokens. Simple, but \`k\` is fixed while the
distribution is not. When the model is confident (one token at 95%), \`k = 50\` drags in 49 tokens
that should never be considered. When the model is genuinely uncertain across 500 plausible
tokens, \`k = 50\` cuts off legitimate options.

**Top-p (nucleus).** Sort descending, accumulate probability, keep tokens until the cumulative
sum reaches \`p\`. The *number* of candidates adapts to the shape of the distribution:

\`\`\`
confident:  [0.95, 0.03, 0.01, ...]  ->  p=0.9 keeps 1 token
uncertain:  [0.08, 0.07, 0.07, ...]  ->  p=0.9 keeps ~30 tokens
\`\`\`

This is the Holtzman et al. contribution and it was a genuine advance. But it has a known flaw:
at high temperature the distribution flattens, so reaching cumulative \`p\` requires many more
tokens, and top-p lets in the tail exactly when the tail is most dangerous. Temperature and top-p
interact badly at the extremes.

**Min-p.** Keep every token whose probability is at least \`min_p × p_max\`, where \`p_max\` is the
top token's probability. The threshold is *relative to the model's confidence*:

\`\`\`
min_p = 0.1
confident:  p_max = 0.95  ->  threshold 0.095  ->  very few survive
uncertain:  p_max = 0.08  ->  threshold 0.008  ->  many survive
\`\`\`

This is the behaviour you actually want, and it degrades gracefully under temperature: since the
threshold scales with \`p_max\`, flattening the distribution does not automatically widen the
candidate set the way it does for top-p. Min-p is the reason people can run temperature 1.5+
without the output falling apart, and it has become a common default in local-inference
communities.

A rough guide to defaults, offered with the caveat that the right values are task- and
model-specific and worth measuring rather than copying:

| use case | temperature | truncation |
|---|---|---|
| factual QA, extraction | 0 – 0.3 | greedy or top-p 0.9 |
| general chat | 0.7 – 1.0 | top-p 0.9–0.95, or min-p 0.05–0.1 |
| creative writing | 1.0 – 1.4 | min-p 0.05–0.1 |
| code generation | 0 – 0.2 | greedy or top-p 0.95 |

The cost of all of these is a sort over the vocabulary. A full sort of 128,256 logits per sequence
per token is not free at batch 256 — production samplers use partial sorts (top-k selection) and
fused kernels for exactly this reason.`,
      ascii: `  logits -> [temperature] -> softmax -> [truncate] -> renormalize -> sample

  TOP-K (k=3)              TOP-P (p=0.9)            MIN-P (0.1)
  ████ 0.45  keep          ████ 0.45  keep  (0.45)  ████ 0.45  keep
  ███  0.30  keep          ███  0.30  keep  (0.75)  ███  0.30  keep
  ██   0.15  keep          ██   0.15  keep  (0.90)  ██   0.15  keep
  █    0.06  CUT           █    0.06  cut           █    0.06  keep (>0.045)
  ·    0.04  CUT           ·    0.04  cut           ·    0.04  CUT  (<0.045)

  fixed count              fixed mass               fixed RATIO to the top`,
    },
    {
      name: 'Repetition penalties and how they misfire',
      keyPoint: 'Penalties operate on token identity with no notion of syntax, so they degrade the text they are meant to fix.',
      body: `Even with good sampling, models loop. The standard patches all modify logits based on
what has already been generated.

**Repetition penalty** (Keskar et al., CTRL) divides the logit of any previously-seen token by
\`r > 1\`. The implementation detail that catches people:

\`\`\`
z_i = z_i / r   if z_i > 0
z_i = z_i * r   if z_i < 0
\`\`\`

The sign check exists because dividing a negative logit would *increase* it. This asymmetry means
the penalty's strength depends on the sign of the logit, which is a fairly arbitrary property.

**Frequency penalty** subtracts \`alpha × count(token)\` — scales with how often the token appeared.
**Presence penalty** subtracts a flat \`beta\` if the token appeared at all. Both are additive in
logit space, which is better behaved than division.

Now the failure modes, which are more serious than they are usually presented:

**Function words are punished first.** The most-repeated tokens in any English text are "the",
"a", "of", "to", ",". A repetition penalty hits them hardest. Set \`r = 1.2\` and the model starts
dropping articles and prepositions — output becomes subtly telegraphic in a way that reads as
"off" without an obvious cause.

**Code and structured output break.** Python repeats \`self\`, \`return\`, \`def\`, \`import\`,
indentation tokens. JSON repeats \`{\`, \`}\`, \`"\`, \`:\`. Penalizing repetition in structured output
means penalizing correctness. **Repetition penalties should be off for code and off for
constrained JSON.** This is a common and avoidable production bug.

**Long contexts poison the penalty.** If the penalty applies over the whole context, then in a
32k-token conversation nearly every common token has appeared and is penalized. Good
implementations apply it over a sliding window (the last \`n\` tokens); many do not, or do not
expose the window.

**It treats identity, not structure.** The real problem is repeated *phrases*, and a token-level
penalty cannot see phrases. It over-penalizes legitimate reuse and under-penalizes a loop built
from varied tokens.

The honest summary: penalties are a blunt patch for a symptom whose real cause is the model's
distribution. Prefer fixing the distribution — better truncation, min-p, a better prompt — and
treat penalties as a last resort with narrow windows and small values (1.05–1.15 for repetition
penalty, 0.1–0.5 for frequency). DRY and similar n-gram-aware samplers are attempts at the
structural version of this idea.`,
      ascii: '',
    },
    {
      name: 'Constrained decoding: making invalid output impossible',
      keyPoint: 'A grammar compiled to a state machine yields a token mask per step, so malformed output cannot be generated rather than being retried.',
      body: `If you need valid JSON, you can ask nicely, validate, and retry — or you can make
invalid output unrepresentable.

The mechanism is a **logit mask**. At each step, given the tokens generated so far, compute which
next tokens could still lead to a valid string. Set every other logit to \`-inf\`. Sample from
what remains. The output is valid by construction, with no retries.

The machinery:

1. Express the target format as a regular expression or context-free grammar.
2. Compile it to a finite state machine (or a pushdown automaton for CFGs).
3. **Precompute, for each FSM state, the set of allowed vocabulary tokens.** This is the
   expensive step and the key insight of the Outlines paper — done ahead of time, it turns
   per-step constraint checking into an O(1) lookup instead of a scan over 128k tokens.
4. At generation time, track the state, look up the mask, apply, sample, advance.

The hard part is that **the FSM is over characters but sampling is over tokens**, and BPE tokens
span multiple characters and do not align to grammar boundaries. A single token might be \`"},{"\`
— valid in some JSON states and not others. Handling this token-boundary mismatch correctly is
most of the implementation difficulty and the source of most bugs in naive versions.

Costs worth knowing before you turn it on:

- **Compilation is not free.** A complex grammar can take hundreds of milliseconds to compile.
  Cache compiled grammars by schema; do not compile per request.
- **Masking is per-sequence work in the hot loop.** At batch 256 with 128k vocabulary, applying
  distinct masks is real overhead. It is why structured output is a first-class engineering
  concern in SGLang and xgrammar rather than a wrapper.
- **Constraints do not confer competence.** Forcing schema-valid JSON guarantees the *shape*, not
  the *content*. A model that does not know the answer will emit a well-formed wrong one. There
  is also evidence that heavy constraint can degrade reasoning quality, presumably by pushing the
  model off its natural distribution — so for reasoning-heavy tasks, prefer letting the model
  think freely and constraining only a final extraction step.`,
      ascii: `  state: expecting a JSON key

    vocabulary            mask        result
    ' "'      logit 3.2   allow  ->    3.2
    ' {'      logit 2.9   BLOCK  ->   -inf
    ' hello'  logit 2.1   BLOCK  ->   -inf
    ' "name'  logit 1.8   allow  ->    1.8
                                       |
                                  softmax over survivors only
                                  -> malformed JSON is unreachable`,
    },
    {
      name: 'Why temperature 0 is not deterministic',
      keyPoint: 'Floating-point addition is not associative, so changing the batch changes the reduction order, which changes the logits, which can flip an argmax.',
      body: `Set temperature to 0, send the same prompt twice, get different completions. This
surprises people, and the explanation has nothing to do with sampling.

**Floating-point addition is not associative.** In real arithmetic \`(a + b) + c = a + (b + c)\`.
In floating point it is not, because each addition rounds:

\`\`\`
(1e10 + 1.0) - 1e10  =  0.0     in fp32
1e10 + (1.0 - 1e10)  =  1.0
\`\`\`

A GPU matmul sums thousands of products in parallel and combines partial results in whatever
order the kernel's parallel reduction dictates. That order depends on the thread and block
configuration, which depends on the tensor shapes, which depend on **the batch size and sequence
lengths of everyone else in your batch**.

So the chain is:

\`\`\`
different batch composition
  -> different kernel/tile configuration selected
  -> different floating-point reduction order
  -> logits differ in the last few bits
  -> if the top two logits are close, argmax flips
  -> a different token
  -> a completely different continuation from there on
\`\`\`

The divergence is usually rare per token but the consequences compound: one flipped token early
changes everything after it.

Contributing factors, roughly in order of how often they bite:

1. **Batch-dependent kernel selection.** The big one. Your request is batched with different
   neighbours each time, so the matmul kernel and its reduction order change.
2. **Atomics and non-deterministic reductions.** Some kernels use \`atomicAdd\`, which completes in
   nondeterministic order by design.
3. **Autotuning.** Libraries pick algorithms by benchmarking at runtime; the winner can vary.
4. **Mixed precision.** bf16 has 8 mantissa bits. Ties and near-ties are common, so small
   perturbations flip argmax more often than you would guess.
5. **MoE routing.** In a sparse model, batch composition can change expert assignment and
   capacity-factor dropping, which is a much larger perturbation than a rounding difference.

Getting real determinism requires **batch-invariant kernels** — kernels whose reduction order does
not depend on batch composition. This is achievable (Thinking Machines published a detailed
treatment of batch-invariant kernels in 2025) but costs performance, because you give up the
freedom to pick the fastest tiling for each shape.

The practical implications: do not build tests that assert exact token equality across runs;
assert on semantic properties instead. Do not promise users bit-reproducible output unless you
have specifically engineered for it. And when debugging a "flaky" model, check whether you are
chasing a bug or chasing floating-point.`,
      ascii: '',
    },
  ],

  mathLab: {
    prompt: `Work these by hand. A calculator is fine; a script is cheating, because the point is
to feel how the transformations behave.

The model gives you logits over a 6-token vocabulary:

\`\`\`
token:   A      B      C      D      E      F
logit:   3.0    2.0    1.0    0.5    0.0   -1.0
\`\`\`

1. Compute the softmax at \`T = 1.0\`. Verify it sums to 1.
2. Recompute at \`T = 0.5\` and at \`T = 2.0\`. Tabulate all three.
3. At \`T = 2.0\`, by what factor did the probability of F (the worst token) increase relative to
   \`T = 1.0\`? This is the "temperature amplifies junk" effect, quantified.
4. At \`T = 1.0\`, apply **top-p = 0.9**. Which tokens survive? What are the renormalized
   probabilities?
5. At \`T = 1.0\`, apply **min-p = 0.1**. Which tokens survive?
6. Now suppose the model is *uncertain*: logits are \`[1.0, 0.9, 0.8, 0.7, 0.6, 0.5]\`. Apply
   top-p = 0.9 and min-p = 0.1 to this. Compare the number of survivors in each case against
   question 4/5.
7. From (4)–(6), state in one sentence the behavioural difference between top-p and min-p.
8. Bonus: at \`T = 2.0\`, apply top-p = 0.9. How many tokens survive now, versus at \`T = 1.0\`?
   What does that tell you about combining high temperature with top-p?`,

    solution: `**1. Softmax at T = 1.0**

\`\`\`
exp(3.0)  = 20.086
exp(2.0)  =  7.389
exp(1.0)  =  2.718
exp(0.5)  =  1.649
exp(0.0)  =  1.000
exp(-1.0) =  0.368
             ------
sum       = 33.210

A = 20.086/33.210 = 0.6048
B =  7.389/33.210 = 0.2225
C =  2.718/33.210 = 0.0818
D =  1.649/33.210 = 0.0496
E =  1.000/33.210 = 0.0301
F =  0.368/33.210 = 0.0111
                    ------
                    1.0000  ✓
\`\`\`

**2. All three temperatures**

At \`T = 0.5\` the logits double to \`[6, 4, 2, 1, 0, -2]\`:

\`\`\`
exp: 403.43, 54.598, 7.389, 2.718, 1.000, 0.135   sum = 469.27
p:   0.8597, 0.1163, 0.0157, 0.0058, 0.0021, 0.0003
\`\`\`

At \`T = 2.0\` the logits halve to \`[1.5, 1.0, 0.5, 0.25, 0, -0.5]\`:

\`\`\`
exp: 4.4817, 2.7183, 1.6487, 1.2840, 1.0000, 0.6065   sum = 11.739
p:   0.3818, 0.2316, 0.1404, 0.1094, 0.0852, 0.0517
\`\`\`

| token | T=0.5 | T=1.0 | T=2.0 |
|---|---|---|---|
| A | 0.8597 | 0.6048 | 0.3818 |
| B | 0.1163 | 0.2225 | 0.2316 |
| C | 0.0157 | 0.0818 | 0.1404 |
| D | 0.0058 | 0.0496 | 0.1094 |
| E | 0.0021 | 0.0301 | 0.0852 |
| F | 0.0003 | 0.0111 | 0.0517 |

**3. Junk amplification**

\`\`\`
F: 0.0517 / 0.0111 = 4.66x
\`\`\`

The worst token became **4.7× more likely** while the best became 1.6× less likely. Now scale
this intuition to a real 128,256-token vocabulary: there are ~128,000 tokens in the tail, each
getting a similar multiplicative boost. Their *collective* mass grows enormously. That is why
high temperature without truncation produces incoherence rather than creativity.

**4. Top-p = 0.9 at T = 1.0**

Accumulate in descending order:

\`\`\`
A: 0.6048   cumulative 0.6048   < 0.9, keep
B: 0.2225   cumulative 0.8273   < 0.9, keep
C: 0.0818   cumulative 0.9091   >= 0.9, keep (this one crosses) and STOP
\`\`\`

Survivors: **A, B, C**. Renormalized by 0.9091:

\`\`\`
A = 0.6048/0.9091 = 0.6653
B = 0.2225/0.9091 = 0.2447
C = 0.0818/0.9091 = 0.0900
\`\`\`

*(Note: implementations differ on whether the crossing token is included. Hugging Face and vLLM
include it, so the kept mass is ≥ p rather than ≤ p. Worth checking in any library you rely on.)*

**5. Min-p = 0.1 at T = 1.0**

\`\`\`
p_max = 0.6048
threshold = 0.1 x 0.6048 = 0.06048

A 0.6048 >= 0.06048  keep
B 0.2225 >= 0.06048  keep
C 0.0818 >= 0.06048  keep
D 0.0496 <  0.06048  CUT
E 0.0301 <  0.06048  CUT
F 0.0111 <  0.06048  CUT
\`\`\`

Survivors: **A, B, C**. Same answer as top-p here — which is exactly why you need question 6 to
see the difference.

**6. The uncertain distribution**

Logits \`[1.0, 0.9, 0.8, 0.7, 0.6, 0.5]\` at \`T = 1.0\`:

\`\`\`
exp: 2.7183, 2.4596, 2.2255, 2.0138, 1.8221, 1.6487   sum = 12.888
p:   0.2109, 0.1908, 0.1727, 0.1563, 0.1414, 0.1279
\`\`\`

*Top-p = 0.9:*

\`\`\`
0.2109  cum 0.2109  keep
0.1908  cum 0.4017  keep
0.1727  cum 0.5744  keep
0.1563  cum 0.7307  keep
0.1414  cum 0.8721  keep
0.1279  cum 1.0000  keep (crosses 0.9)
\`\`\`

**All 6 survive.**

*Min-p = 0.1:*

\`\`\`
p_max = 0.2109,  threshold = 0.02109
every token is >= 0.02109
\`\`\`

**All 6 survive.**

Both adapt here. The distinguishing case is the *confident* distribution combined with
temperature, which is question 8.

**7. The difference in one sentence**

Top-p fixes the **total probability mass** it keeps and lets the candidate count float; min-p
fixes the **ratio to the best token** and lets both count and mass float — so min-p tracks the
model's confidence directly rather than through the proxy of cumulative mass.

**8. Top-p = 0.9 at T = 2.0** (the original confident logits)

\`\`\`
A: 0.3818   cum 0.3818   keep
B: 0.2316   cum 0.6134   keep
C: 0.1404   cum 0.7538   keep
D: 0.1094   cum 0.8632   keep
E: 0.0852   cum 0.9484   keep (crosses)
\`\`\`

**5 tokens survive, versus 3 at T = 1.0.**

This is the flaw. Raising temperature flattens the distribution, so more tokens are needed to
accumulate 0.9 of the mass — top-p *widens* the candidate set exactly when each candidate has
become less trustworthy. The two knobs compound in the wrong direction.

Min-p at \`T = 2.0\`: threshold is \`0.1 × 0.3818 = 0.03818\`, and all six tokens exceed it — so
min-p widens too, but for the honest reason that the model really has become less certain, and
the threshold remains anchored to the top token rather than to an absolute mass target. That
anchoring is why min-p stays usable at temperatures where top-p starts admitting the tail.`,
  },

  codeLab: {
    goal: `Implement greedy, temperature, top-k, top-p, and min-p from scratch in NumPy — no
\`torch.multinomial\`, no library sampling. Then run a temperature sweep on a real model and
characterize how the output degrades at each end.

The implementation is short. The value is in the sweep: seeing the same prompt go from robotic to
incoherent as one number changes is worth more than any amount of reading about it.`,
    code: `"""
Samplers from scratch, then a temperature sweep on a real model.
    pip install numpy torch transformers
"""
import numpy as np


def softmax(logits, temperature=1.0):
    if temperature <= 0:                       # T -> 0 is greedy
        out = np.zeros_like(logits, dtype=np.float64)
        out[np.argmax(logits)] = 1.0
        return out
    z = np.asarray(logits, dtype=np.float64) / temperature
    z = z - z.max()                            # stability: exp(large) overflows
    e = np.exp(z)
    return e / e.sum()


def top_k_filter(probs, k):
    """Keep the k highest-probability tokens."""
    if k is None or k <= 0 or k >= len(probs):
        return probs
    out = np.zeros_like(probs)
    idx = np.argpartition(probs, -k)[-k:]      # O(V), not a full sort
    out[idx] = probs[idx]
    return out / out.sum()


def top_p_filter(probs, p):
    """Nucleus: keep the smallest set whose cumulative mass reaches p.

    Includes the token that crosses the threshold, matching HF/vLLM behaviour,
    so the kept mass is >= p rather than <= p.
    """
    if p is None or p >= 1.0:
        return probs
    order = np.argsort(probs)[::-1]
    sorted_p = probs[order]
    cum = np.cumsum(sorted_p)
    # keep everything strictly before the crossing, plus the crossing token
    n_keep = int(np.searchsorted(cum, p) + 1)
    n_keep = min(n_keep, len(probs))
    out = np.zeros_like(probs)
    out[order[:n_keep]] = sorted_p[:n_keep]
    return out / out.sum()


def min_p_filter(probs, min_p):
    """Keep tokens with probability >= min_p * p_max. Threshold scales with confidence."""
    if min_p is None or min_p <= 0:
        return probs
    threshold = min_p * probs.max()
    out = np.where(probs >= threshold, probs, 0.0)
    return out / out.sum()


def sample(logits, temperature=1.0, top_k=None, top_p=None, min_p=None, rng=None):
    """Temperature first, then truncation -- the HF/vLLM order."""
    rng = rng or np.random.default_rng()
    probs = softmax(logits, temperature)
    if top_k:  probs = top_k_filter(probs, top_k)
    if top_p:  probs = top_p_filter(probs, top_p)
    if min_p:  probs = min_p_filter(probs, min_p)
    return int(rng.choice(len(probs), p=probs))


# ============================================================
# Reproduce the math lab
# ============================================================
logits = np.array([3.0, 2.0, 1.0, 0.5, 0.0, -1.0])
names = list("ABCDEF")

print("=== temperature sweep ===")
print(f"{'tok':>4} {'T=0.5':>8} {'T=1.0':>8} {'T=2.0':>8}")
cols = [softmax(logits, t) for t in (0.5, 1.0, 2.0)]
for i, n in enumerate(names):
    print(f"{n:>4} {cols[0][i]:>8.4f} {cols[1][i]:>8.4f} {cols[2][i]:>8.4f}")
print(f"\\n  junk amplification, F: {cols[2][5]/cols[1][5]:.2f}x")

print("\\n=== truncation on a CONFIDENT distribution (T=1.0) ===")
p1 = softmax(logits, 1.0)
for label, f in [("top-p 0.9", top_p_filter(p1, 0.9)),
                 ("min-p 0.1", min_p_filter(p1, 0.1)),
                 ("top-k 3", top_k_filter(p1, 3))]:
    kept = [names[i] for i in range(6) if f[i] > 0]
    print(f"  {label:<12} keeps {len(kept)}: {' '.join(kept)}")

print("\\n=== truncation on an UNCERTAIN distribution (T=1.0) ===")
p2 = softmax(np.array([1.0, 0.9, 0.8, 0.7, 0.6, 0.5]), 1.0)
for label, f in [("top-p 0.9", top_p_filter(p2, 0.9)),
                 ("min-p 0.1", min_p_filter(p2, 0.1)),
                 ("top-k 3", top_k_filter(p2, 3))]:
    kept = [names[i] for i in range(6) if f[i] > 0]
    print(f"  {label:<12} keeps {len(kept)}: {' '.join(kept)}")

print("\\n=== the top-p flaw: confident distribution at T=2.0 ===")
p3 = softmax(logits, 2.0)
print(f"  top-p 0.9 keeps {(top_p_filter(p3, 0.9) > 0).sum()} tokens (was 3 at T=1.0)")
print(f"  min-p 0.1 keeps {(min_p_filter(p3, 0.1) > 0).sum()} tokens")


# ============================================================
# The sweep on a real model -- this is the part that teaches
# ============================================================
def sweep():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    name = "gpt2"
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(name).eval()

    prompt = "The three most important ideas in computer science are"
    ids = tok(prompt, return_tensors="pt").input_ids

    @torch.no_grad()
    def gen(n=60, seed=0, **kw):
        rng = np.random.default_rng(seed)
        cur, past = ids, None
        out_ids = []
        for _ in range(n):
            o = model(cur if past is None else cur[:, -1:],
                      past_key_values=past, use_cache=True)
            past = o.past_key_values
            nxt = sample(o.logits[0, -1].float().numpy(), rng=rng, **kw)
            out_ids.append(nxt)
            cur = torch.cat([cur, torch.tensor([[nxt]])], dim=1)
        return tok.decode(out_ids)

    print("\\n" + "=" * 70)
    print("TEMPERATURE SWEEP -- same prompt, same seed, one knob")
    print("=" * 70)
    for T in [0.0, 0.5, 0.8, 1.0, 1.3, 1.8, 2.5]:
        label = "greedy" if T == 0 else f"T={T}"
        print(f"\\n--- {label} ---")
        print(gen(temperature=T).strip()[:300])

    print("\\n" + "=" * 70)
    print("T=1.8 WITH TRUNCATION -- the tail is the problem, not the temperature")
    print("=" * 70)
    for label, kw in [("no truncation", {}),
                      ("top-p 0.9", {"top_p": 0.9}),
                      ("min-p 0.1", {"min_p": 0.1})]:
        print(f"\\n--- T=1.8, {label} ---")
        print(gen(temperature=1.8, **kw).strip()[:300])


if __name__ == "__main__":
    try:
        sweep()
    except ImportError:
        print("\\n(pip install torch transformers for the model sweep)")

# --- TODO for you ---
#   1. Implement a repetition penalty and confirm it damages code generation.
#      Prompt with "def fibonacci(n):" and try penalty 1.0, 1.15, 1.3.
#      Watch 'return' and indentation tokens get suppressed.
#   2. Implement typical sampling (Meister et al., 2022): keep tokens whose
#      surprisal is closest to the distribution's entropy. Compare to min-p.
`,
    expect: `The math-lab reproduction matches the hand calculation exactly: junk amplification
of 4.66×, top-p and min-p both keeping 3 tokens on the confident distribution and all 6 on the
uncertain one, and at T=2.0 top-p widening to 5 while min-p stays anchored.

The temperature sweep is where the learning happens. Expect roughly:

- **Greedy / T=0.0** — coherent, then a loop. GPT-2 usually starts repeating within 40 tokens.
- **T=0.5** — coherent, safe, slightly dull.
- **T=0.8–1.0** — the usable range. Varied and mostly sensible.
- **T=1.3** — noticeably looser; occasional non-sequiturs.
- **T=1.8** — frequent nonsense words and broken syntax.
- **T=2.5** — near-total incoherence, often with rare unicode and code fragments.

Then the second block makes the actual point: T=1.8 with min-p 0.1 is *dramatically* more
coherent than T=1.8 raw, while keeping the variety. The temperature was never the problem — the
tail was. That comparison is the single most useful thing in this lab.

(GPT-2 is a weak model, so even T=0.8 wanders. The *relative* progression is what to look at, not
absolute quality.)`,
    stretch: `Quantify the degradation instead of eyeballing it. For each temperature, generate
20 completions and measure: (a) fraction of generated tokens that are repeats of a token in the
previous 20 positions, (b) distinct-2 — the ratio of unique bigrams to total bigrams, and (c) mean
per-token log-probability under the model itself. Plot all three against temperature. You should
find repetition falling and distinct-2 rising with temperature, while mean log-probability falls
monotonically — which is precisely Holtzman's point that likelihood and quality diverge.`,
  },

  papers: [
    {
      title: 'The Curious Case of Neural Text Degeneration',
      by: 'Holtzman et al., 2019',
      url: 'https://arxiv.org/abs/1904.09751',
      why: 'The nucleus sampling paper, and the paper that explained why maximizing likelihood produces bad text. One of the most practically consequential NLP papers of its era.',
      frame: `**Figure 2** is the whole argument in one image: per-token probability of human text
fluctuates constantly while beam-search text sits at a high plateau. Read **Section 3** for why
maximization fails and **Section 4** for nucleus sampling itself. The evaluation in Section 5
(perplexity of generated text against human text, distinct-n) is worth reading for how to measure
generation quality at all.`,
    },
    {
      title: 'Efficient Guided Generation for Large Language Models',
      by: 'Willard & Louf, 2023',
      url: 'https://arxiv.org/abs/2307.09702',
      why: 'The Outlines paper. Shows how to precompute FSM-state-to-token-mask maps so constrained decoding costs O(1) per step instead of a scan over the vocabulary.',
      frame: `**Section 3** is the contribution: indexing the vocabulary by FSM state ahead of
time. **Section 4** extends it to context-free grammars with a pushdown automaton. Pay attention
to the discussion of token-boundary alignment — that mismatch between character-level grammars
and BPE tokens is where the real difficulty lives.`,
    },
    {
      title: 'CTRL: A Conditional Transformer Language Model for Controllable Generation',
      by: 'Keskar et al., 2019',
      url: 'https://arxiv.org/abs/1909.05858',
      why: 'Source of the repetition penalty that nearly every inference library implements, usually without citation.',
      frame: 'Read **Section 4.1** only — the penalized-sampling paragraph. It is a few lines and defines the formula you will find in every serving codebase, including the sign asymmetry.',
    },
    {
      title: 'Locally Typical Sampling',
      by: 'Meister et al., 2022',
      url: 'https://arxiv.org/abs/2202.00666',
      why: 'An information-theoretic alternative to top-p: keep tokens whose surprisal is near the distribution entropy, rather than the most likely ones.',
      frame: 'Read the introduction and Section 3. The framing — that natural language keeps information rate roughly constant, so you want *typical* tokens rather than *likely* ones — is a genuinely different way to think about the decoding objective.',
    },
  ],

  checkpoint: {
    claim: `You can predict what changing any sampling parameter will do to output quality, and
you can explain both why beam search lost for open-ended generation and why temperature 0 does
not guarantee reproducible output.`,
    questions: [
      {
        q: 'Why does beam search improve translation but hurt dialogue?',
        a: `Beam search searches for the highest-probability sequence, so its value depends on
whether high probability is a good proxy for quality. In translation the output space is narrow —
one source meaning, a few acceptable phrasings — so the most likely sequence is usually the right
one. In dialogue the output space is enormous and the most likely response is the blandest: "I
don't know", "That's interesting". Holtzman et al. showed that human text does not sit at high
probability under a language model; it fluctuates, and that variability is what makes it
informative. The diagnostic symptom is that increasing beam width makes open-ended output
monotonically worse — a search working correctly on the wrong objective. There is also a systems
cost: \`k\` beams means \`k\` KV caches, which cuts concurrency.`,
      },
      {
        q: 'You raise temperature from 0.8 to 1.5 and output becomes incoherent. Is the temperature wrong?',
        a: `Not necessarily — the missing truncation is more likely the problem. Temperature
flattens the distribution multiplicatively, and with a 128,256-token vocabulary that hands a
large collective share of mass to ~128,000 tail tokens. Sample one and the sequence derails,
permanently, because generation is autoregressive. The fix is to pair the temperature with min-p
(say 0.05–0.1), which cuts every token below a fixed ratio of the top token's probability. Note
that top-p is a poor partner here: flattening the distribution makes top-p keep *more* tokens, so
it widens the candidate set exactly when candidates are least trustworthy. Min-p's threshold
stays anchored to the top token, so it does not have that failure.`,
      },
      {
        q: 'Explain min-p to someone who already understands top-p, in two sentences.',
        a: `Top-p keeps the smallest set of tokens whose cumulative probability reaches \`p\`, so it
fixes the total *mass* retained and lets the candidate count vary. Min-p instead keeps every token
whose probability is at least \`min_p × p_max\`, fixing the *ratio to the best token* — which means
the cutoff tightens automatically when the model is confident and relaxes when it is genuinely
uncertain, and it does not blow open as temperature rises.`,
      },
      {
        q: 'Your JSON-generating endpoint occasionally emits malformed output. Someone suggests raising the repetition penalty to stop it looping. What is wrong with that?',
        a: `Almost everything. A repetition penalty punishes tokens that have already appeared,
and JSON is *built* from repeated tokens: \`{\`, \`}\`, \`"\`, \`:\`, \`,\`. Penalizing them makes malformed
output more likely, not less. Repetition penalties should be off for JSON and for code generally.
The correct fix is constrained decoding: compile the schema to a state machine and mask the logits
each step so only tokens that can still lead to valid JSON are sampleable. Then malformed output
is not merely unlikely, it is unrepresentable. Worth remembering that this guarantees shape, not
correctness — a model that does not know the answer will emit a well-formed wrong one.`,
      },
      {
        q: 'A user reports the same prompt at temperature 0 gives different answers on different days. Walk through the likely cause.',
        a: `Floating-point non-associativity interacting with batching. GPU matmuls sum thousands
of products via parallel reduction, and the reduction order depends on the tile and thread
configuration the kernel picks, which depends on the tensor shapes, which depend on the batch
composition. The user's request is batched with different neighbours each time, so the logits
differ in their last few bits. Usually harmless — but when the top two logits are close, argmax
flips, and one different token changes the entire continuation. Contributing factors include
atomics, kernel autotuning, bf16's 8-bit mantissa making near-ties common, and for MoE models,
batch-dependent expert routing. Real determinism needs batch-invariant kernels, which cost
throughput. The practical advice is not to promise bit-reproducibility and not to write tests that
assert exact token equality.`,
      },
    ],
  },

  pitfalls: [
    {
      wrong: 'Temperature 0 means deterministic output.',
      right: `Temperature 0 removes randomness from *sampling*, not from the logits. Those still
vary run to run because GPU reductions are order-dependent and the order depends on batch
composition. When the top two logits are close, argmax flips and the continuation diverges
completely. Bit-level reproducibility requires batch-invariant kernels and a throughput
sacrifice.`,
    },
    {
      wrong: 'Higher temperature means more creative.',
      right: `Higher temperature means a flatter distribution, which means more probability on the
long tail — and in a 128k-token vocabulary the tail is mostly junk, not creativity. Without
truncation, high temperature yields broken syntax and invented words. With min-p, the same
temperature yields genuine variety. Creativity comes from sampling widely *within* the plausible
set, not from sampling outside it.`,
    },
    {
      wrong: 'A repetition penalty is a safe default to leave on.',
      right: `It punishes the most frequent tokens first, which in English are function words —
"the", "a", "of", "to" — so output goes subtly telegraphic. In code and JSON it punishes the
tokens that constitute correctness. And if applied over a full 32k context rather than a sliding
window, nearly every common token ends up penalized. Keep it off by default; if you need it, use
a narrow window and a small value.`,
    },
    {
      wrong: 'Constrained decoding makes the model more accurate.',
      right: `It makes the output *well-formed*, which is a different property. A schema
guarantees the shape; it says nothing about whether the values are right. A model that does not
know the answer will emit a syntactically perfect wrong one. There is also evidence that heavy
constraint can hurt reasoning quality by pushing the model off its natural distribution — so for
reasoning tasks, let it think freely and constrain only the final extraction.`,
    },
  ],

  glossary: [
    { term: 'logits', def: 'Raw unnormalized scores over the vocabulary, before softmax. Everything in this module operates on them.' },
    { term: 'temperature', def: 'Divisor applied to logits before softmax. Below 1 sharpens, above 1 flattens.' },
    { term: 'top-k', def: 'Keep the k highest-probability tokens. Fixed count, does not adapt to the distribution.' },
    { term: 'top-p / nucleus', def: 'Keep the smallest set whose cumulative probability reaches p. Fixed mass, adaptive count.' },
    { term: 'min-p', def: "Keep tokens with probability at least min_p times the top token's probability. Fixed ratio, adaptive to confidence." },
    { term: 'repetition penalty', def: 'Divides logits of already-seen tokens by r > 1. Blunt, and harmful for code and structured output.' },
    { term: 'frequency penalty', def: 'Subtracts a term proportional to how many times a token has appeared. Additive in logit space, better behaved than division.' },
    { term: 'constrained decoding', def: 'Masking logits each step so only tokens consistent with a grammar or schema can be sampled.' },
    { term: 'degeneration', def: 'The repetitive-loop failure mode of likelihood-maximizing decoders. Named and diagnosed by Holtzman et al.' },
    { term: 'batch invariance', def: 'The property that a kernel produces bit-identical results regardless of batch composition. Required for true determinism; costs performance.' },
  ],
};
