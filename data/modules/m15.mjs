export default {
  n: 15,
  slug: 'cross-model-kv-transfer',
  title: 'CROSS-MODEL KV CACHE TRANSFER',
  tagline: 'Every earlier level treated the KV cache as private to the model that built it. This is the weeks-old research arguing that was only ever a convenience — and the arithmetic for what it costs to relax it.',
  hours: '6–8 hours',
  prereqs: ['Module 2', 'Module 6', 'Module 9'],

  bigIdea: `Module 2 gave you the KV cache as a private, growing block of memory owned by one
sequence on one model. Module 6 showed you could share it *within* a model — copy-on-write between
sequences, a hash chain over shared prefixes. Module 9 showed you could move it *between machines*
running the same model, and priced the transfer. In every one of those, "which model built this
cache" was fixed. The cache and the model that would read it next were always the same model.

Starting in mid-2026, a run of research — one arXiv preprint (Heo et al., "Cross-Model KV Cache
Transfer in LLM Families") and a string of industrial replications from a company called TrustAI —
has been testing what happens if you let that assumption go. A small, cheap model in a family reads
a prompt and builds its own KV cache the ordinary way. Instead of discarding that cache and having
a larger, more expensive sibling model reread the same prompt from scratch, a **learned mapping**
translates the small model's cache into the large model's coordinate system and injects it directly
— skipping the large model's prefill almost entirely.

This works, in the narrow, mechanistic sense the evidence actually supports: mapped caches recover
most of a benchmark's task accuracy, and a hybrid of mapping and targeted recomputation can recover
essentially all of it. It does not yet work in the sense of being a benchmarked, shipped serving
technique the way PagedAttention or speculative decoding are — the numbers in this level come from
small held-out sets and a single company's own research blog more often than from peer review, and
the honest latency picture (the math lab gets there directly) is more modest than the headline
figures suggest. Read this level the way Module 10 asks you to read the rest of the frontier: as a
real mechanism whose magnitude is still being established, not a settled result.

The throughline that connects it to everything before it is the same one these chapters have been
making since Module 1: **decoding is memory-bound, and prefill is compute-bound, so the amount of
compute you spend rereading a prompt is a cost you get to choose how to pay** — with your own
model, with someone else's cache, or, as the last two concepts in this level show, with some honest
combination of both.`,

  concepts: [
    {
      name: 'A third axis for KV reuse: across model size, not just across machines',
      keyPoint: "Module 9's disaggregation moves a cache between machines running the *same* model; this is a cache moving between two *different* models in the same family, to skip a second model's prefill entirely.",
      body: `Every KV-cache technique these chapters have covered so far treats "which model built this
cache" as fixed. Module 6's prefix caching and copy-on-write share a cache among sequences running
the same weights. Module 9's disaggregation moves a cache between machines, over the network — but
the model on both ends is identical. Disaggregation is a *where* question, not a *which* question.

The research behind this level relaxes that. Inside one model family — Qwen3 at 0.6B, 1.7B, 4B and
30B; an NVIDIA Minitron-pruned Llama sibling next to the full Llama 3.1 8B — a smaller, cheaper
model reads a prompt and builds its own KV cache exactly as it always does. Instead of throwing
that cache away and having the *target* model reread the same prompt from scratch, a learned
mapping translates the small model's cache into the target's coordinate system and injects it
directly into the target's cache before the target starts generating.

The target model never sees the prompt tokens. It only ever sees a KV cache someone else produced,
in translated form. If the mapping is good enough, the target continues as if it had done its own
prefill — at a fraction of the cost, because prefill on the small model is far cheaper than prefill
on the large one. Module 1's compute-bound-prefill argument does not change; you are simply
choosing *which* model pays that bill.

Why this counts as new content rather than a restatement of disaggregation: disaggregation's entire
cost model — Module 9's KV-transfer bandwidth arithmetic — assumes the bytes on both ends mean the
same thing. You are moving an *identical* tensor, so the only cost is bytes over a wire.
Cross-model transfer's bytes do not mean the same thing on both ends: two different models
generally disagree on hidden dimension, layer count, or KV-head count, and even within one family
at matched width they diverge through training. "Just copy it" corrupts the cache outright —
Research Note 001 found direct copy from Qwen3 0.6B into Qwen3 1.7B reached 2% next-token
agreement against a 100% oracle, worse than useless. Everything else in this level is the machinery
that stands between "different tensors" and "usable transfer."`,
      ascii: `  MODULE 9 · DISAGGREGATION                THIS LEVEL · CROSS-MODEL TRANSFER

  [Llama-8B]--KV bytes-->[Llama-8B]      [Qwen-1.7B]--mapped KV-->[Qwen-4B]
   machine A   (network)   machine B      (small, cheap)  (large, expensive)

  SAME model, different machine.          DIFFERENT models, same prompt.
  Cost = bytes / bandwidth.                Cost = bytes / bandwidth + a
                                            learned coordinate change.`,
    },
    {
      name: 'RoPE-stripped, per-head linear mapping: making two different models speak the same coordinates',
      keyPoint: 'Positional rotation is removed before mapping and reapplied after, because it is the one part of a K vector that is not "content" — mapping it would fit position, not meaning.',
      body: `The mechanism, stripped to its three moves, is the one every report in this level's
source material shares:

1. **Strip RoPE from the source keys.** Module 7 covered rotary position embedding as a rotation
   applied to Q and K before the dot product, so attention scores depend on relative position. That
   rotation is a function of *position*, not of what token produced the vector — leave it in, and a
   mapping fit on position-3 vectors would not transfer to position-300, because the rotation angle
   is baked into the numbers the regression sees. Removing it (the exact inverse rotation, which
   RoPE's structure makes cheap) leaves a **content-space** representation: what the source model
   believes about this token, independent of where it sits.
2. **Fit a per-head linear map in content space.** For each target attention head, a small linear
   transform — the reports call this **ridge regression**, ordinary least squares with an L2
   penalty for stability — is fit from source-model K/V vectors onto the corresponding target-model
   K/V vectors, using a calibration set of prompts (500 FineWeb-Edu sequences, in every report cited
   here) run through both models. This is offline work, done once per model pair, long before any
   real request arrives.
3. **Reapply the target's own RoPE**, at the target's own positions, after mapping. The content
   moved; the position did not, so the rotation has to be redone in the target's coordinate system,
   not carried over from the source.

The paper behind this (Heo et al., arXiv:2608.03893) reports why a linear map is a reasonable thing
to try in the first place, not just a convenient one: on a Qwen3 14B→32B pair, "one source layer
explains 56% of variance in the target's keys and 32% in values, rising to 79% and 65% with
multiple source layers." Two different models trained on similar data develop K/V representations
that are substantially *linearly* related to each other — not identical, but not arbitrary either.
A linear map is cheap to fit (a closed-form solve, no gradient descent, no GPU) and captures most
of that relationship; what it leaves on the table is what the next three concepts are about.

One number worth holding onto before the k-selection concept: doing this *without* the map at all —
copying source K/V straight into the target, RoPE and all — is not a weaker version of the
technique, it is a different regime entirely. Research Note 001 measured it directly: 2% next-token
agreement, because a target head attending over vectors from an unrelated coordinate system does
not merely lose information, it actively misreads it.`,
      ascii: `  source K/V --[strip RoPE]--> content space --[per-head ridge map]--> content space
                                                                              |
                                                            [reapply target RoPE]
                                                                              v
                                                                   injected target K/V`,
    },
    {
      name: 'How many source layers feed one target layer: the k trade-off',
      keyPoint: "k is how many source layers are concatenated as inputs to one target layer's mapper; more views improve fit and cost more to compute, and the two costs do not peak at the same k.",
      body: `The mapping in the previous concept glosses over one design choice: a target layer
does not have to be predicted from its single corresponding source layer. It can be predicted from
a **concatenation of several source layers' K/V vectors** — more views of what the source model
computed, at the cost of a bigger regression to solve at every injection.

Research Note 002 swept this directly on the Qwen3 1.7B→4B pair, testing k in {1, 2, 4, 6, 8, 10,
12, 16, 20, 24} against three things that all move as k grows: prefix negative log-likelihood
(lower is better — does the mapped cache make the actual next tokens likely), attention-output
cosine similarity (does the attention *pattern* match, not just the eventual logits), and wall-clock
mapping time.

\`\`\`
k= 4:  prefix NLL 3.058   attn-cosine 0.834   mapping time 22.8 ms
k= 8:  prefix NLL 3.002   attn-cosine 0.849   mapping time 27.7 ms   <- highest attn-cosine
k=16:  prefix NLL 2.934   attn-cosine 0.816   mapping time 49.0 ms
k=24:  prefix NLL 2.930   attn-cosine 0.797   mapping time 70.7 ms   <- lowest prefix NLL
\`\`\`

The two quality metrics **disagree about which k is best**, and neither disagrees by accident.
Prefix NLL is a property of the eventual output distribution — it keeps improving (slowly) as k
grows, because more source context genuinely does carry more predictive signal. Attention-output
cosine is a property of the intermediate representation the mapper is fit to reproduce directly,
and it *peaks* at k=8 and degrades afterward — past a point, a bigger concatenated input to the
same ridge regression means more parameters to fit from the same calibration budget, and the fit
gets noisier even as the thing it indirectly predicts keeps improving. Mapping time is the least
ambiguous of the three: it grows monotonically, 2.6× from k=8 to k=24, because a bigger concatenated
input means solving a bigger linear system per head, per layer, per request.

Note 002's own conclusion is a methodology lesson as much as a numeric one: **"select on behavior,
then ask about speed."** Choosing k by whichever metric is cheapest to compute — mapping time, or
even attention-cosine, which is intermediate rather than behavioral — risks optimizing something
that does not track what you actually care about: the mapped cache's effect on the model's eventual
output. The team's own selection, k=24 on prefix NLL, is *slower*, not faster, than the metric that
looked best at a glance — a deliberate trade of speed for the metric closest to what the cache is
actually for.`,
      ascii: '',
    },
    {
      name: 'Align first, then learn what alignment missed: staged residual correction',
      keyPoint: 'A frozen linear/ridge map handles the coordinate change; a small residual, trained separately and afterward, corrects what a linear map cannot represent — and the two stages have to run in that order.',
      body: `Ridge mapping is linear by construction, and the previous concept's variance-explained
numbers (56–79%) already said a linear map does not fully explain the target's representations.
Research Note 005 ran the natural next experiment: freeze the ridge map, and train a small
additional correction on top of it.

The ordering matters, not just the presence of a correction. Note 005's headline result trains a
**rank-8 residual** — \`ẑ = z + BAz\`, where B and A are the two low-rank factors of an adapter
added on top of the frozen ridge output — and finds it beats fitting one larger single-stage map:
mean KL32 (a rollout-divergence metric, covered two concepts from now) dropped from 0.04287 (ridge
alone) to 0.03813 (ridge, then residual), with a 95% confidence interval entirely below zero.
Crucially, the residual is trained **through a 32-step rollout objective** — backpropagated through
the target model actually generating 32 tokens from the mapped cache and comparing to its own
native continuation — rather than a single-step loss, because a residual that only looks right on
the very next token can still compound into visible drift a few tokens later.

Two findings sharpen what "align first" is doing mechanistically. First, residual gain correlates
with how hard the ridge alignment already was: at 4B scale, the correlation between per-sample
residual improvement and ridge error was r = 0.72 — the hardest quartile of ridge-error cases
improved 4.1× more from the residual than the easiest quartile. The residual is not adding a flat
bonus everywhere; it is specifically fixing the cases the linear map struggled with. Second, that
correlation is scale-dependent: at 30B the same measurement gave r = 0.13, with gains concentrated
on the hardest cases even more sharply rather than spreading evenly — meaning the right place to
spend adapter capacity is not the same fraction-of-effort at every model scale. Note 005's own
conclusion is to reselect adapter placement per model pair rather than trust one fixed recipe — and
the next concept's real-world pair, where the recipe inverts outright, is why that caveat is not
boilerplate.

None of this is free. This is the point where cross-model transfer stops being "one closed-form
regression, computed once offline" and starts requiring an actual training loop per model pair —
the residual is fit by gradient descent through a rollout, not solved in one step the way ridge is.
Whether that training cost is worth paying depends on how many requests will amortize it, the same
build-vs-buy question Module 9 asks about a dedicated KV-transfer link.`,
      ascii: `        frozen ridge map              trained residual (rank-r)
   z ------------------------> ẑ_ridge ------------------------> ẑ = ẑ_ridge + BAz
        (closed-form,                  (SGD through a 32-step
         solved once, offline)          rollout KL objective)`,
    },
    {
      name: 'When mapping hits a ceiling: partial recomputation as the hybrid escape',
      keyPoint: 'Any linear or low-rank correction has a ceiling set by how nonlinear the true source-to-target relationship is; recomputing the last one or two layers exactly, from the shared boundary, removes the ceiling instead of chasing it.',
      body: `Research Note 004 asked a sharper version of the previous concept's question, on a toy
model small enough to see the failure mode directly: two four-layer transformers sharing an
embedding and one exact layer (layer 0), then diverging — the target's layers 1–3 perturbed from
the source's. Four conditions, each anchoring the recomputation boundary further downstream:

\`\`\`
raw cache (no mapping at all)                        logit error 13.96%   agreement 81.67%
affine map (one linear+bias map, all divergent layers) logit error  5.20%   agreement 92.71%
L2 anchor (recompute layers 1-2 exactly, map layer 3)  logit error  4.09%   agreement 95.21%
L3 anchor (recompute layers 1-3 exactly, map nothing)  logit error  0.0000014%   agreement 100%
\`\`\`

A single affine map, applied across every divergent layer, recovers most of the gap over raw
copying — it works well close to the shared boundary, "translating two projections of one
representation," in the note's own words. But it stops well short of oracle-exact, because
"nonlinear representation drift... accumulated" through the deeper layers faster than one linear
map applied uniformly could track. Pushing the recompute boundary progressively deeper — anchor at
layer 2 and only map the last layer, then anchor at layer 3 and map nothing at all — keeps closing
the remaining gap in exactly the direction that pattern predicts: 92.71% → 95.21% → 100.00000% (mean
target-logit error 1.42×10⁻⁶%) across 480 held-out comparisons, because at full recomputation there
is no approximation left anywhere in the pipeline to accumulate error.

This is not a retreat from transfer, it is a **hybrid**. Only the layers close enough to the shared
boundary to still be well-approximated by a map get mapped — cheap, one regression's worth of
compute. The layers where drift has piled up get recomputed — real forward-pass compute, but only
for those layers, on already-mapped context, not the whole prompt from scratch. The same shape of
trade-off appears in a same-model setting these chapters have not covered until now: **CacheBlend**
(Yao et al., arXiv:2405.16444) reuses precomputed KV caches for RAG context chunks that were not
built as a shared prefix, and "selectively recomputes the KV values of a small subset of tokens to
partially update each reused KV cache" rather than trusting the reused cache everywhere or
discarding it entirely. Cross-model transfer and CacheBlend solve different problems — one bridges
two different models, the other stitches contexts within one model — but they converge on the
identical mechanism: *reuse what is safe to reuse, recompute exactly the part that would otherwise
be silently wrong.*

The scope this result was shown at matters: two four-layer toy models, one exactly-shared layer,
single-token continuation, no measured latency. It demonstrates that the mapping ceiling has an
escape hatch, not that the escape hatch is already validated at production scale — Note 004 says so
in its own evidence-boundary section, and the last concept in this level returns to that gap.`,
      ascii: '',
    },
    {
      name: 'Scoring a transferred cache: agreement, KL32, and chance-normalized quality',
      keyPoint: 'Next-token agreement, multi-step KL divergence, and task-quality retention measure three different things a transferred cache can get right or wrong — a cache can score well on one while scoring poorly on another.',
      body: `The reports lean on a small family of metrics repeatedly enough that reading any one
number in isolation is a trap. Four are worth being able to define on sight:

- **Next-token top-1 agreement** — does the target model, continuing from the mapped cache, pick
  the same single next token it would have picked from its own native prefill? Cheap to compute,
  and the metric Research Note 001's 30× improvement (direct injection to learned mapping) is
  stated in.
- **KL32** — the KL divergence between the mapped-cache continuation and the native-prefill
  continuation, measured over a 32-step forced rollout rather than one step. This is what the
  residual two concepts back is trained against, and it exists because single-step agreement can
  look fine while the trajectory quietly diverges a few tokens later — a model 82% likely to pick
  the same *first* token can still be somewhere else entirely by token 32.
- **Chance-normalized task-quality retention** — for a multiple-choice benchmark like HellaSwag or
  PIQA, \`(mapped_score − chance_floor) / (oracle_score − chance_floor)\`. Raw accuracy is not
  comparable across benchmarks with different numbers of options, and therefore different chance
  floors; this normalization is, which is why every report states retention as a percentage rather
  than a raw accuracy delta.
- **Answer evidence** — whether a downstream probe shows the information the cached context should
  have carried, independent of whether the exact next-token path matches.

The reason to track more than one: they are not proxies for each other. Note 005 ran an experiment
specifically to separate cached-context "memory" from exact-continuation "trajectory," comparing
four conditions on a 40-case context-dependent benchmark:

| Cache path                 | Mean evidence | Complete evidence |
|-----------------------------|--------------|--------------------|
| Native Thinking prefill     | 99.4%        | 39/40              |
| Ridge-mapped Instruct cache | 91.9%        | 36/40              |
| Direct Instruct cache       | 91.9%        | 36/40              |
| Uncached-context control    | 5.6%         | 2/40               |

Mapped and direct scored *identically* on whether the answer's evidence survived transfer, while
the RoPE-stripping concept already established they are nowhere near identical on token-level
fidelity (91% vs 88% next-token agreement in a related measurement, and far more divergent under
KL32). Prompt memory and exact behavioral trajectory are, empirically, close to independent axes: a
cache can carry "the information was in there" almost regardless of mapping quality, while "the
model continues exactly as it natively would" depends heavily on it. Which axis matters depends on
the deployment — a RAG lookup cares about the first; a multi-turn agent replaying a specific
reasoning trace cares more about the second.`,
      ascii: '',
    },
    {
      name: 'Where the win actually lives: routing, the latency crossover, and the honest limits',
      keyPoint: "The technique is fastest when a source cache is already sitting in memory for another reason; measured end to end, its latency advantage is real but concentrated at long contexts, and every report so far is a small-sample mechanistic study, not a production benchmark.",
      body: `Three deployment shapes recur across the reports, and they share one precondition: a
source-model cache has to already exist, produced for some reason other than feeding the target.
**Model routing** — a small model handles a request by default and escalates to a larger one only
on difficulty, carrying its reading forward instead of discarding it. **Agent workflows** — a cheap
model handles routine conversation turns and a specialized model is brought in mid-session,
inheriting the accumulated context. **Cache-aware scheduling** — a system with a choice of which
worker serves a request prefers one that already holds a usable cache, the same locality argument
Module 9 makes about prefix-cache placement, extended across model boundaries.

The headline speed numbers are real but need their preconditions attached. "7.91× faster Llama
target start at 8K" measures target-model *start* latency given the source cache is already
resident — it explicitly excludes the time the small model spent reading in the first place.
Include that, and the same pair measures 1.25× end to end. The math lab in this level works
through the underlying latency table directly and finds something neither headline states outright:
at every context length actually measured (up to 8,192 tokens), the transfer path was still
*slower* than fresh target prefill in absolute terms — the reported "1.083× at 8K" means transfer
cost 8.3% more, not less. The trend line only crosses over in transfer's favor beyond the tested
range, because transfer carries a large *fixed* per-request overhead that a straight prefill does
not, and that fixed cost is exactly what the reports flag as the thing left to optimize, not the
underlying idea.

Held next to that arithmetic, the boundaries every report states explicitly are worth taking at
face value rather than as boilerplate: results hold within a model family and mostly within one
exact architecture — cross-*family* transfer (a Qwen source, a Llama target) is not attempted
anywhere in this material; sample sizes are small by the standards of an ML benchmark (27 to 480
held-out comparisons, not the tens of thousands a serving-quality claim would usually want); the
fidelity numbers are mechanistic — forced rollouts, controlled probes — rather than free-generation
task quality at scale; and, worth stating plainly rather than leaving implicit, the primary source
for most of the specific numbers in this level is a company's own research blog, not a
peer-reviewed benchmark. The one peer-reviewable anchor, Heo et al., is itself still a preprint.
Treat the mechanism as real and the magnitude of the win as still being established — the same
posture Module 10 asks you to take toward every other frontier technique in these chapters.`,
      ascii: '',
    },
  ],

  mathLab: {
    prompt: `This level's numbers come from two places you can check without a GPU: the
architecture arithmetic Module 2 already gave you for Llama-3.1-8B (32 layers, 8 KV heads, 128
head_dim, 131,072 bytes per token), and Research Note 007's own measured production numbers for a
Minitron-4B-to-Llama-3.1-8B transfer on H100 (its Experiment 9 latency table).

\`\`\`
rank-r residual params = 4 x r x head_dim x n_layers x n_kv_heads
    (per layer, per KV head: an A and a B factor, each head_dim x r, for BOTH K and V --
     4 low-rank factors total, each contributing r x head_dim parameters)
\`\`\`

1. Derive that formula yourself from the adapter's own description — a rank-r linear residual,
   \`ẑ = z + BAz\`, applied separately to K and V, per attention head, per layer, where A is
   \`r x head_dim\` and B is \`head_dim x r\`. Evaluate it at r = 8 for Llama-3.1-8B's own dimensions
   and compare to Research Note 007's reported "1,048,576 learned parameters."
2. Note 007 also reports a 12.0 MiB checkpoint for that adapter. At 1,048,576 parameters, what
   per-parameter storage size does that imply, in bytes? Which common dtype does that match (fp16,
   bf16, fp32) — or does it match none of them? What does the mismatch tell you about what else is
   likely stored inside a real checkpoint file besides the raw weight values?
3. Note 007's Experiment 9 table gives two measured points each for "ridge + residual transfer"
   latency and for "fresh target prefill" latency: transfer is 117.311 ms at 1,024 tokens and
   256.692 ms at 8,192 tokens; prefill is 28.404 ms at 1,024 tokens and 237.002 ms at 8,192 tokens.
   Fit each path as an affine function of sequence length S — find the slope (ms/token) and
   intercept (ms) implied by its two data points.
4. What do the two intercepts tell you, independent of the slopes? One should come out close to
   zero — say what that means physically for that path. Say what a roughly 97 ms intercept means
   for the other.
5. Using your two affine fits, solve for the crossover token count S* where the transfer path and
   the fresh-prefill path cost the same. Is S* inside or outside the range Note 007 actually
   measured (1,024–8,192 tokens)? What does that imply about how close the reported "1.083× at 8K"
   figure is to the technique's actual break-even point?
6. Given your answer to question 4, which term — the fixed overhead or the per-token slope — would
   a systems engineer prioritize shrinking to pull the crossover down into the tested range? Check
   your answer against what Note 007 itself names as the "primary optimization target."`,

    solution: `**1. The rank-r parameter formula**

Each attention head, each layer, needs its own adapter, and each adapter has four low-rank
factors — an A and a B for K, and an A and a B for V. A is \`r x head_dim\`, B is \`head_dim x r\`,
so each factor contributes \`r x head_dim\` parameters and there are four of them:

\`\`\`
params_per_head_per_layer = 4 x r x head_dim
total params = 4 x r x head_dim x n_layers x n_kv_heads
\`\`\`

For Llama-3.1-8B at r = 8 (n_layers = 32, n_kv_heads = 8, head_dim = 128):

\`\`\`
4 x 8 x 128 x 32 x 8 = 1,048,576
\`\`\`

That is an **exact match** to Research Note 007's reported figure. The formula is not an
approximation of what they built — reading "a rank-8 residual on direct KV" and the four
architecture numbers Module 2 already gave you is enough to reproduce their parameter count to the
last digit, without ever seeing their code.

**2. What the checkpoint size implies**

\`\`\`
12.0 MiB = 12 x 2^20 B = 12,582,912 B
12,582,912 / 1,048,576 = 12.0 bytes per parameter
\`\`\`

That matches none of the standard dtypes — not bf16/fp16 (2 B), not fp32 (4 B), not even fp32 with
a naive 3× safety margin by accident; it is exactly 3× a plain fp32 storage estimate. The honest
read is that the checkpoint is not simply "1,048,576 numbers, serialized." A rank-8 residual for
this architecture is **32 layers × 8 KV heads × 2 (K, V) × 2 (A, B) = 1,024 distinct small
tensors**, and a real serialization format (safetensors, a state-dict pickle) pays fixed per-tensor
overhead — shape metadata, dtype tags, name strings, alignment padding — on every one of them. One
big tensor absorbs that overhead once; a thousand small ones pay it a thousand times. This is the
mirror image of Module 12's block-table result: there, one indirection layer cost two kilobytes to
save a gigabyte, because it was one structure managing many bytes. Here, splitting one adapter into
many small per-head, per-layer tensors multiplies a *fixed* per-object cost by how many objects you
chose to have — a genuinely different overhead direction from anything else in these chapters, and a
concrete reason a "count the parameters" estimate of a real fine-tuning or adapter artifact usually
comes in low.

**3. Two-point affine fits**

\`\`\`
transfer:  slope = (256.692 - 117.311) / (8192 - 1024) = 139.381 / 7168 = 0.019445 ms/token
           intercept = 117.311 - 0.019445 x 1024 = 97.399 ms

prefill:   slope = (237.002 - 28.404) / (8192 - 1024) = 208.598 / 7168 = 0.029101 ms/token
           intercept = 28.404 - 0.029101 x 1024 = -1.396 ms  (≈ 0, within measurement noise)
\`\`\`

**4. What the intercepts mean**

Prefill's intercept is essentially zero, which is exactly what a compute-bound, roughly-linear-in-S
FLOPs model predicts — the same linear approximation these chapters have used for prefill cost since
Module 1. There is no meaningful fixed cost to running the target model's own prefill; the whole
bill scales with tokens.

Transfer's ~97 ms intercept is a large, **fixed** cost paid once per request, essentially
independent of how many tokens are being transferred: computing the ridge map's output, running the
residual adapter, and injecting the result into the target's cache all cost roughly the same
whether the prefix is 1,024 tokens or 8,192.

**5. The crossover**

\`\`\`
97.399 + 0.019445 S  =  -1.396 + 0.029101 S
98.795                =  0.009656 S
S*  ≈  10,231 tokens
\`\`\`

That is **outside** the tested range — Note 007's longest measured prefix was 8,192 tokens, short
of the ~10,231 where this linear extrapolation says transfer starts winning outright. Which means
the reported "1.083× at 8K" is not a near-miss rounding down to "basically break-even" — under this
fit, transfer was still measurably slower than fresh prefill at every length TrustAI actually
tested, and the true crossover sits beyond what the report shows you. The 7.91× headline figure
from elsewhere in this level's source material is a different, narrower comparison — target-start
latency given an already-resident source cache, which specifically excludes the ~97 ms fixed
overhead this question just derived. Both numbers are honestly reported; neither one, read alone,
tells you where the break-even point actually is.

**6. Where to spend engineering effort**

Shrink the fixed ~97 ms overhead, not the per-token slope. The slope is already smaller for
transfer than for prefill (0.0194 vs 0.0291 ms/token) — at long enough context, transfer wins by
more and more per additional token. The entire reason it currently loses at every tested length is
the intercept. That matches Note 007's own conclusion exactly: "static mapper identified as primary
optimization target... concrete 19.69 ms gap remains for fused mapping, lower-precision execution,
and boundary-token path overlapping" — all three of those are ways to cut fixed per-request
overhead, not per-token cost. The math lab and the report agree, which is itself worth noticing:
you did not need their conclusion to find the same answer, only their two data points and an affine
fit.`,
  },

  codeLab: {
    goal: `Build a small, deterministic, standard-library-only simulation of the four-rung ladder
this level climbs — direct copy, a ridge-regression map, a trained residual on top of it, and exact
partial recomputation — on two toy "models" that share one layer and diverge after it through a
transform no linear map can fully undo. Confirm the qualitative shape every research note in this
level reports: direct copy is close to useless, ridge mapping recovers most of the gap, a correctly
designed residual recovers most of what ridge missed, and exact recomputation from the shared
boundary closes the rest completely. No GPU, no real model weights — the point is the mechanism,
isolated from everything real weights would add on top of it.`,
    code: `"""Toy cross-model KV-cache transfer: RoPE strip/reapply, ridge-regression
mapping, a trained low-rank residual, and partial recomputation -- the same
four-rung ladder TrustAI's research notes climb, on synthetic data small
enough to run with the standard library alone.

    python3 kv_transfer_lab.py

Two toy "models" share one layer of representation (the shared boundary) and
diverge after it: the source applies a fixed linear transform, the target
applies its own linear transform PLUS a rank-2 bilinear term with no source
counterpart -- a nonlinearity that makes a *linear* map provably unable to
reach perfect agreement, the same ceiling Research Note 004 hit with real
weights. Positions carry a simplified RoPE: a per-position rotation applied
after the shared boundary, which is why it has to be stripped before mapping
and reapplied after.
"""

import random
import math

random.seed(7)

D = 8            # hidden dim (kept small enough to invert by hand-rolled Gauss-Jordan)
N_CALIB = 500    # calibration sequences, matching the FineWeb-Edu-sized sets in the reports
N_EVAL = 100      # held-out evaluation sequences
SEQ_LEN = 32      # positions per sequence
RIDGE_LAMBDA = 1.0
VOCAB = 200       # toy unembedding size, for a top-1 "agreement" metric

# ---- tiny linear algebra: plain lists of lists, no numpy ------------------

def mat_mul(A, B):
    n, k, m = len(A), len(B), len(B[0])
    return [[sum(A[i][t] * B[t][j] for t in range(k)) for j in range(m)] for i in range(n)]

def transpose(A):
    return [list(row) for row in zip(*A)]

def mat_vec(A, v):
    return [sum(A[i][j] * v[j] for j in range(len(v))) for i in range(len(A))]

def add(A, B):
    return [[A[i][j] + B[i][j] for j in range(len(A[0]))] for i in range(len(A))]

def scale(A, s):
    return [[a * s for a in row] for row in A]

def identity(n):
    return [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]

def solve(A, B):
    """Gauss-Jordan solve of A X = B for square A. A and B are lists of lists;
    B may have multiple columns. Returns X with the same shape as B."""
    n = len(A)
    M = [row[:] + B[i][:] for i, row in enumerate(A)]
    bw = len(B[0])
    for col in range(n):
        piv = max(range(col, n), key=lambda r: abs(M[r][col]))
        M[col], M[piv] = M[piv], M[col]
        pv = M[col][col]
        M[col] = [x / pv for x in M[col]]
        for r in range(n):
            if r == col:
                continue
            f = M[r][col]
            if f != 0:
                M[r] = [M[r][c] - f * M[col][c] for c in range(n + bw)]
    return [row[n:] for row in M]

def ridge_fit(X, Y, lam):
    """W minimising ||XW - Y||^2 + lam*||W||^2, solved per-output-column via
    the normal equations (X^T X + lam I) W = X^T Y."""
    Xt = transpose(X)
    XtX = mat_mul(Xt, X)
    reg = add(XtX, scale(identity(len(XtX)), lam))
    XtY = mat_mul(Xt, Y)
    return solve(reg, XtY)

def rand_vec(d, scale_=1.0):
    return [random.gauss(0, scale_) for _ in range(d)]

def rand_mat(n, m, scale_=1.0):
    return [[random.gauss(0, scale_) for _ in range(m)] for _ in range(n)]

def tanh_v(v):
    return [math.tanh(x) for x in v]

# ---- the two toy models -----------------------------------------------------
# Shared boundary: both models produce the same layer-0 output for a given
# input (a stand-in for a shared embedding + early layers). Source and target
# then each apply their OWN layer-1 transform -- different random weights,
# different models -- source purely linear, target linear plus a bilinear
# term (defined below) with no source counterpart. RoPE is a per-position
# rotation in the first three (x, y) coordinate pairs, applied after layer-1.

W_shared = rand_mat(D, D, 0.5)
W_src1 = rand_mat(D, D, 0.5)
W_tgt1 = rand_mat(D, D, 0.5)
UNEMBED = rand_mat(VOCAB, D, 1.0)   # toy "read out a token id from a hidden state"

def rope(v, pos, dim_pairs=3):
    """Rotate the first \`dim_pairs\` (x, y) pairs of v by an angle proportional
    to position -- a simplified stand-in for real per-frequency RoPE."""
    out = v[:]
    for p in range(dim_pairs):
        i, j = 2 * p, 2 * p + 1
        theta = pos * (0.05 * (p + 1))
        c, s = math.cos(theta), math.sin(theta)
        x, y = out[i], out[j]
        out[i] = c * x - s * y
        out[j] = s * x + c * y
    return out

def unrope(v, pos, dim_pairs=3):
    return rope(v, -pos, dim_pairs)

# The target's transform has a genuine nonlinear component with no source
# counterpart: a fixed rank-2 bilinear term, ((h0.P) elementwise* (h0.Q)) . Cb.
# No linear map from source to target -- ridge or otherwise -- can represent
# it exactly, which is the toy-model version of Research Note 004's "layers
# 1-3 perturbed" nonlinear drift. It is EXACTLY the functional form the
# factorized-quadratic residual (below) is built to fit, so a correctly
# trained rank-2 residual should recover most of it; a plain linear residual,
# stacked on top of a map that already found the best linear fit, cannot
# recover any of it -- try R_TRUE = 2 with a LINEAR-only residual first.
R_TRUE = 2
P_true = rand_mat(R_TRUE, D, 0.6)
Q_true = rand_mat(R_TRUE, D, 0.6)
C_bilin = rand_mat(D, R_TRUE, 0.5)

def bilinear_term(h0):
    p = mat_vec(P_true, h0)
    q = mat_vec(Q_true, h0)
    hh = [p[k] * q[k] for k in range(R_TRUE)]
    return mat_vec(C_bilin, hh)

def layer0(x):
    return tanh_v(mat_vec(W_shared, x))

def source_layer1(h0, pos):
    h1 = mat_vec(W_src1, h0)              # source: linear
    return rope(h1, pos)

def target_layer1_exact(h0, pos):
    h1 = [a + b for a, b in zip(mat_vec(W_tgt1, h0), bilinear_term(h0))]
    return rope(h1, pos)

def token_id(v):
    scores = mat_vec(UNEMBED, v)
    return max(range(VOCAB), key=lambda i: scores[i])

# ---- build calibration and eval sets ---------------------------------------

def make_sequence(seq_len):
    """One synthetic 'prompt': a random walk of embeddings, each carried
    through both models' first layer at its own position."""
    src_cache, tgt_cache, positions = [], [], []
    for pos in range(seq_len):
        x = rand_vec(D, 1.0)
        h0 = layer0(x)
        src_cache.append(source_layer1(h0, pos))
        tgt_cache.append(target_layer1_exact(h0, pos))
        positions.append(pos)
    return src_cache, tgt_cache, positions

def flatten(seqs):
    src_rows, tgt_rows = [], []
    for src_cache, tgt_cache, positions in seqs:
        for s, t, pos in zip(src_cache, tgt_cache, positions):
            src_rows.append(unrope(s, pos))   # strip RoPE before mapping
            tgt_rows.append(unrope(t, pos))
    return src_rows, tgt_rows

calib = [make_sequence(SEQ_LEN) for _ in range(N_CALIB)]
evalset = [make_sequence(SEQ_LEN) for _ in range(N_EVAL)]

calib_src, calib_tgt = flatten(calib)
W_ridge = ridge_fit(calib_src, calib_tgt, RIDGE_LAMBDA)   # content-space linear map

# A FACTORIZED QUADRATIC residual on top of the frozen ridge map -- Research
# Note 005's own adapter formula, verbatim:
#
#     z -> z_hat = z + ((z.A) elementwise* (z.B)) . C      A,B: D->r   C: r->D
#
# The point of the elementwise product is that it is the cheapest possible
# NONLINEAR function of z. A plain linear residual B(A(z)) is still linear in
# z, and ridge regression already found the best linear map -- so a linear
# residual has, by construction, nothing left to correct on the SAME
# objective ridge was fit to. Try that first (see the pitfall below) before
# running this cell, and it reproduces exactly that null result. Trained by
# plain SGD on the ridge map's residual error; the real adapters are trained
# by backpropagating through a multistep rollout, which the math lab in this
# level connects back to a single-step loss.
def ridge_predict(src_vec):
    return mat_vec(transpose(W_ridge), src_vec)

R = 2
resid_pairs = [(s, [t[d] - ridge_predict(s)[d] for d in range(D)]) for s, t in zip(calib_src, calib_tgt)]

A_q = rand_mat(R, D, 0.2)   # D -> R
B_q = rand_mat(R, D, 0.2)   # D -> R
C_q = rand_mat(D, R, 0.2)   # R -> D
LR, EPOCHS = 0.003, 25
train_set = resid_pairs[:4000]   # subsample for pure-Python training speed
clip = lambda x: max(-1.0, min(1.0, x))
for epoch in range(EPOCHS):
    random.shuffle(train_set)
    for s, r in train_set:
        a = mat_vec(A_q, s)                                       # R-vector, z.A
        b = mat_vec(B_q, s)                                       # R-vector, z.B
        h = [a[k] * b[k] for k in range(R)]                        # elementwise product
        pred = mat_vec(C_q, h)                                     # D-vector
        err = [pred[d] - r[d] for d in range(D)]
        for d in range(D):
            for k in range(R):
                C_q[d][k] -= LR * clip(err[d] * h[k])
        dh = [sum(C_q[d][k] * err[d] for d in range(D)) for k in range(R)]
        da = [dh[k] * b[k] for k in range(R)]
        db = [dh[k] * a[k] for k in range(R)]
        for k in range(R):
            for i in range(D):
                A_q[k][i] -= LR * clip(da[k] * s[i])
                B_q[k][i] -= LR * clip(db[k] * s[i])

def residual_predict(src_vec):
    a = mat_vec(A_q, src_vec)
    b = mat_vec(B_q, src_vec)
    h = [a[k] * b[k] for k in range(R)]
    return mat_vec(C_q, h)

# ---- evaluation --------------------------------------------------------------

def evaluate(name, predict_fn):
    agree, total = 0, 0
    sq_err, tgt_sq = 0.0, 0.0
    for src_cache, tgt_cache, positions in evalset:
        for s_r, t_r, pos in zip(src_cache, tgt_cache, positions):
            s = unrope(s_r, pos)
            pred_content = predict_fn(s)
            pred = rope(pred_content, pos)     # reapply target position
            t = t_r
            if token_id(pred) == token_id(t):
                agree += 1
            total += 1
            sq_err += sum((a - b) ** 2 for a, b in zip(pred, t))
            tgt_sq += sum(b * b for b in t)
    top1 = agree / total
    rel_err = math.sqrt(sq_err / tgt_sq)
    return top1, rel_err

results = {}
results['direct (no mapping)'] = evaluate('direct', lambda s: s)
results['ridge map'] = evaluate('ridge', ridge_predict)
results['ridge + rank-2 residual'] = evaluate(
    'ridge+resid', lambda s: [a + b for a, b in zip(ridge_predict(s), residual_predict(s))])
# partial recompute: given the source's content-space vector, invert the
# SOURCE's own layer-1 map to recover h0 exactly (source is purely linear, so
# this inverse is exact), then apply the target's own layer-1 exactly (linear
# + bilinear) -- no mapping error anywhere, by construction.
def evaluate_recompute():
    agree, total = 0, 0
    sq_err, tgt_sq = 0.0, 0.0
    W_src1_inv = solve(W_src1, identity(D))
    for src_cache, tgt_cache, positions in evalset:
        for s_r, t_r, pos in zip(src_cache, tgt_cache, positions):
            s = unrope(s_r, pos)                       # source content vector = W_src1 @ h0
            h0 = mat_vec(W_src1_inv, s)                 # recover the shared boundary exactly
            pred_content = [a + b for a, b in zip(mat_vec(W_tgt1, h0), bilinear_term(h0))]  # target's OWN exact layer
            pred = rope(pred_content, pos)
            t = t_r
            if token_id(pred) == token_id(t):
                agree += 1
            total += 1
            sq_err += sum((a - b) ** 2 for a, b in zip(pred, t))
            tgt_sq += sum(b * b for b in t)
    return agree / total, math.sqrt(sq_err / tgt_sq)

results['partial recompute'] = evaluate_recompute()

print(f"{'method':<26}{'top-1 agreement':>18}{'rel. error':>14}")
for name, (top1, rel_err) in results.items():
    print(f"{name:<26}{top1*100:>17.1f}%{rel_err:>14.4f}")
`,
    expect: `\`\`\`
method                       top-1 agreement    rel. error
direct (no mapping)                     1.4%        1.2645
ridge map                              56.9%        0.4529
ridge + rank-2 residual                90.8%        0.0786
partial recompute                     100.0%        0.0000
\`\`\`

Runs in about eleven seconds on one core and is fully deterministic with \`random.seed(7)\` set once
at the top and nowhere else — rerun it and every digit above should reproduce exactly.

The four rows are, in order, Research Note 001's headline number, Note 002/003's ridge baseline,
Note 005's staged-residual result, and Note 004's recomputation ceiling — reproduced in miniature.
Direct copy is barely better than random (1/200 = 0.5% chance floor on this toy vocabulary, so 1.4%
is close to noise). Ridge mapping alone recovers a large fraction of the gap by capturing the
*linear* relationship between the two models' representations. The rank-2 residual then recovers
most of what ridge left behind — 33.9 additional percentage points of agreement, and the relative
error drops by 82% — because it is trained on exactly the nonlinear (bilinear) functional form the
target's transform actually contains; this is a favorable case by construction, more favorable than
Note 005's own real-weight improvement, precisely because a toy model lets you build the residual's
true target shape into the experiment. Partial recomputation reaches exact agreement, because
inverting the source's own (exactly linear, exactly invertible) transform and then running the
target's real layer removes approximation from the pipeline entirely, not just most of it.

If your numbers differ: check that \`random.seed(7)\` is the only call to \`random.seed\` anywhere,
that \`train_set = resid_pairs[:4000]\` is taken before the shuffle inside the epoch loop (shuffling
first would make the subsample epoch-dependent), and that \`clip\` is applied to each gradient term
individually, not to the final parameter update — the difference changes the effective learning
rate near the clip boundary.`,
    stretch: `Replace the quadratic residual's \`A_q\`/\`B_q\`/\`C_q\` training block with a **plain linear**
residual — \`ẑ = z + BAz\` for two matrices A (D→R) and B (R→D), dropping the elementwise product
entirely — trained by the same SGD loop on the same \`resid_pairs\` target. Keep everything else
identical, including \`R = 2\`. You should get 56.9% agreement and 0.4531 relative error: statistically
indistinguishable from ridge alone (56.9% / 0.4529), reproducing the pitfall's "why doesn't the
residual help" result exactly. The reason is structural, not a training failure: \`ridge_fit\` already
computed the closed-form minimizer of squared error for a *linear* map on this exact calibration
set, so composing it with another linear map cannot reduce that same objective any further — two
linear maps in sequence are still one linear map, and you cannot beat the optimum of a loss you have
already exactly minimized by adding more of the same kind of function.

Then set \`R_TRUE = 4\` (the target's true bilinear rank) while leaving the residual's own \`R = 2\`
unchanged, and rerun the original quadratic-residual version. The residual should recover *less* of
the gap than it did at matched rank — a controlled demonstration of Research Note 005's own
scale-dependent finding that how much a fixed-rank adapter can recover depends on how well its rank
matches the true complexity of what it is correcting, not on the adapter design alone.`,
  },

  papers: [
    {
      title: 'Cross-Model KV Cache Transfer in LLM Families: A Closed-Form Linear Mapping for Prefill Reuse',
      by: 'Heo et al., 2026',
      url: 'https://arxiv.org/abs/2608.03893',
      why: 'The paper this entire level is downstream of. The RoPE-strip-and-reapply mechanism, the per-head ridge map, and the variance-explained numbers the second concept quotes all originate here.',
      frame: `Read for the linear-structure measurements before the method — the 56%/32%,
79%/65% variance-explained figures are the empirical justification for trying a linear map at all,
not an incidental detail. Then compare its "retains 73–98% of standalone-prefill accuracy on four
pairs" headline against this level's own math lab, which works the same kind of honest-accounting
exercise on a different report's latency numbers; the paper's accuracy claim and the blog's speed
claim deserve the identical treatment — read the range, not just the top of it.`,
    },
    {
      title: 'Compact Language Models via Pruning and Knowledge Distillation',
      by: 'Muralidharan et al., NVIDIA, 2024',
      url: 'https://arxiv.org/abs/2407.14679',
      why: "The Minitron family — the width-pruned Llama sibling several of this level's transfer pairs use as the cheap source model. Worth knowing what pruning actually removes before trusting a pruned model's cache to stand in for its parent's.",
      frame: `Read for what "width-pruned" means concretely — depth, width, attention and MLP
pruning combined with distillation-based retraining, at up to 40× fewer tokens than training from
scratch. A width-pruned sibling and its parent share far more architectural DNA than two
independently-trained models of different sizes do, which is a plausible reason cross-model
transfer works at all between them — worth treating as a hypothesis to weigh against the plain
same-family Qwen results, not an established explanation.`,
    },
    {
      title: 'CacheBlend: Fast Large Language Model Serving with Cached Knowledge Fusion',
      by: 'Yao et al., 2024',
      url: 'https://arxiv.org/abs/2405.16444',
      why: "The same-model analog of this level's partial-recomputation concept: selectively recomputing a small subset of a reused cache rather than trusting it everywhere or discarding it entirely.",
      frame: `Read against the partial-recomputation concept side by side. CacheBlend reuses KV
caches for RAG context chunks that were not built as a shared prefix — a same-model, different-
context problem — while this level's technique bridges two different models on the same context.
Different problems, identical fix: reuse what is safe, recompute exactly what would otherwise be
silently wrong. Worth having an opinion on why that shape of solution recurs whenever "reuse a
cache built under slightly different conditions" comes up, across two otherwise-unrelated papers.`,
    },
    {
      title: 'DistServe: Disaggregating Prefill and Decoding for Goodput-optimized Large Language Model Serving',
      by: 'Zhong et al., OSDI 2024',
      url: 'https://arxiv.org/abs/2401.09670',
      why: "Module 9's paper, reread against this level's opening axis distinction: DistServe moves an identical cache across machines; nothing in it needs the mapping machinery this level spends six concepts on, because the model on both ends is the same.",
      frame: `Read it once more for what it does *not* have to solve. DistServe's KV transfer is
bytes over InfiniBand, full stop — no RoPE stripping, no ridge regression, no residual, because
source and target agree on every dimension by construction. That absence is the cleanest possible
statement of why this level exists: relax "same model" to "same family," and an entire mapping
apparatus becomes necessary that a same-model transfer never had to build.`,
    },
  ],

  checkpoint: {
    claim: `You can explain why a KV cache cannot simply be copied between two different models, name
the three-step mechanism — strip RoPE, fit a per-head ridge map, reapply RoPE — that makes transfer
possible, place a trained residual and partial recomputation as the two ways to push past a linear
map's ceiling, distinguish the metrics this technique is evaluated on and why no single one is
sufficient alone, and give an honest account of where the measured latency win actually lives versus
where the headline number implies it lives.`,
    questions: [
      {
        q: `A colleague proposes skipping the ridge-mapping step entirely and running the small
model's cache straight through the large model's attention layers, on the theory that "it's still a
KV cache, the shapes will broadcast." What actually goes wrong, and which reported number would you
show them?`,
        a: `Two separate things go wrong, and conflating them understates the problem. First, a
literal shape mismatch: two different models in a family routinely disagree on hidden dimension,
head_dim, or KV-head count, so the tensors may not even be the same size — there is no "broadcast"
that fixes a dimension mismatch, only an error or silent corruption depending on how the injection
code is written. Second, and this is the part "the shapes will broadcast" misses even when
dimensions happen to match: a target attention head reading raw source K/V is reading vectors from
an unrelated coordinate system, still carrying the source model's own RoPE rotation baked in at the
wrong angles for the target's position scheme. Research Note 001 measured exactly this — direct
injection reached 2% next-token top-1 agreement, against a 100% same-model oracle and a 61% learned-
mapping result on the identical pair. That is not "somewhat worse than mapping," it is close to
useless: the target model does not just fail to benefit from the source's reading, it actively
misreads what the source produced.`,
      },
      {
        q: `Why does RoPE have to be stripped before mapping and reapplied after, rather than just
being included as part of what the ridge regression learns to reproduce?`,
        a: `Because RoPE's rotation angle is a deterministic function of absolute position that the
target model already knows exactly — \`theta(position)\` is closed-form, not something to be
estimated from data. Folding it into the regression would mean fitting a linear map on vectors whose
values are entangled with whatever positions happened to appear in the calibration set, which both
wastes the regression's limited capacity approximating something already computable in closed form,
and risks a map that overfits to calibration-set positions and generalizes poorly to positions it
did not see during fitting — a long request at inference time routinely visits positions a
500-sequence calibration set never sampled. Stripping RoPE first produces a **content-space**
representation that genuinely is position-independent — what the source model believes about a
token, decoupled from where it sits — which is the only thing a single calibration-time-fit linear
map can be expected to generalize correctly across arbitrary sequence lengths. Reapplying the
target's own RoPE afterward is then exact, not learned, because the target's rotation formula and
the token's real position are both already known with certainty.`,
      },
      {
        q: `Research Note 002 found that attention-output cosine similarity peaks at k = 8 while
prefix NLL keeps improving through k = 24, and the team ultimately selected k = 24. Was that the
right call, and what does "select on behavior, then ask about speed" mean operationally?`,
        a: `It was the right call for what the cache is actually for. Attention-output cosine
measures fidelity of an *intermediate* representation — useful as a diagnostic, but not what a
downstream user experiences. Prefix NLL measures the *actual output distribution* the mapped cache
produces — closer to what "the mapped cache behaves like the real thing" means in practice.
Selecting on the intermediate metric because it happens to peak at a cheaper k (27.7 ms vs 70.7 ms
mapping time) would optimize a proxy at the expense of the real target, which is exactly the mistake
"select on behavior, then ask about speed" is naming: pick the configuration that wins on the metric
closest to actual model behavior first, and treat speed as a secondary filter among behaviorally
tied candidates — not as a tiebreaker that gets to override a behavioral difference. Operationally,
that means the selection criterion has an explicit priority order (prefix NLL, then speed) rather
than one blended score where a large speed advantage could silently outvote a real behavioral
regression.`,
      },
      {
        q: `A rank-8 linear residual trained on top of a frozen ridge map showed a real, statistically
significant improvement in Research Note 005. This level's own code lab found a plain linear
residual, trained the same way on top of the same kind of frozen ridge map, showed none. Is that a
contradiction?`,
        a: `No — the two residuals are trained on different objectives, and that difference is the
whole explanation. The code lab's linear residual is trained on the *identical* single-step
squared-error objective the frozen ridge map was already closed-form fit to minimize; composing two
linear maps is still one linear map, and ridge regression already found the exact minimizer of that
objective on that calibration data, so a further linear map trained on the same loss has
mathematically nothing left to improve. Research Note 005's rank-8 residual is trained on a
*different* objective — a 32-step rollout KL divergence, backpropagated through the target model's
own multistep generation — which ridge was never fit to minimize in the first place. A linear
function can still improve a loss landscape it was not the closed-form optimum of, even after
another linear function was the closed-form optimum of a *different* loss on the same data. The
lesson is not "linear residuals never help" — it is that a residual only has room to help when it is
optimizing something the frozen map underneath it was not already exactly solving, whether that
difference comes from a genuinely nonlinear functional form (the code lab's quadratic residual) or
from a genuinely different training objective (Note 005's rollout KL versus ridge's single-step
MSE). Worth also flagging the pitfall on the next page: Note 005's own "ridge, then residual"
ordering is not universal — Research Note 007's Minitron-to-Llama pair found ridge projection
performed *poorly* on that architecture, and the winning recipe there skipped ridge and applied the
residual directly to the raw cache instead.`,
      },
      {
        q: `This level's math lab found the transfer path was still slower than fresh prefill at
every context length actually tested, even though the surrounding material's headline framing is
7.91× faster. Does that mean the technique doesn't work?`,
        a: `No, but it means two separable claims are being made and only one of them is well
supported yet. Claim one: the mechanism produces a usable cache — this is well supported, across
next-token agreement, KL32, task-quality retention, and answer-evidence measurements that agree with
each other about the qualitative ordering of direct-copy versus ridge versus ridge-plus-residual
versus recomputation, even if the exact magnitudes vary by pair. Claim two: the current, unoptimized
implementation is already a net latency win end to end — this is not supported at the lengths
actually measured, and the math lab's own affine fit puts the real break-even point beyond the
tested range, driven by a large fixed per-request overhead rather than by anything wrong with the
per-token cost. Those are different claims, and the honest read is "promising mechanism, immature
latency implementation" rather than either "it works" or "it doesn't" — which is also exactly how
the source material frames its own next steps: kernel fusion, lower-precision execution, and
overlapping the transfer with other work are named as the remaining path to a real end-to-end win,
not offered as a retreat from the underlying idea.`,
      },
    ],
  },

  pitfalls: [
    {
      wrong: 'If a mapped cache achieves high next-token agreement, the model\'s downstream task behavior is preserved.',
      right: `Agreement, multi-step trajectory fidelity, and task-quality retention are measurably
different axes, not three views of one underlying number. Research Note 005's four-condition
comparison found a ridge-mapped cache and a raw direct-injected cache scored *identically* on
whether a downstream probe's answer evidence survived transfer — 91.9% both — while the same pair
diverged sharply on token-level fidelity elsewhere in the same report. A cache can carry "the
information was in there" almost independent of how well it was mapped, while whether the model's
*exact continuation* matches what it would natively produce depends heavily on mapping quality. Ask
which axis your use case actually needs before reading a single metric as the verdict: a retrieval
lookup cares about evidence surviving; a multi-turn agent replaying a specific reasoning chain cares
about the exact trajectory.`,
    },
    {
      wrong: 'The two-stage recipe — a frozen ridge alignment first, then a trained residual on top of it — is the correct order for any model pair.',
      right: `It is the order that won for Research Note 005's same-width Qwen3-4B Instruct/
Thinking sibling pair, where ridge alignment itself was strong (mean K/V reconstruction R² ≈ 0.91).
It was not the winning recipe for Research Note 007's Minitron-4B-to-Llama-3.1-8B pair — a
width-pruned model paired with its non-pruned relative, a structurally different relationship than
two same-width siblings — where "ridge projection... performed poorly," and the method that actually
won applied the rank-8 residual directly to the raw, unmapped cache instead. Note 005's own
conclusion generalizes past just its headline result: which stage should carry the correction is an
empirical question to reselect per model pair and per architecture relationship, not a fixed
pipeline to apply uniformly.`,
    },
    {
      wrong: 'The 7.91× (or 30×, or any other headline multiplier reported here) describes the end-to-end latency a production deployment would see.',
      right: `Each of these numbers is a narrower comparison than "end to end" — read what it holds
fixed before repeating it. 7.91× is target-model *start* latency given an already-resident source
cache, excluding the time the source model spent reading. 30× is next-token agreement improvement
over a direct-injection baseline that was itself close to random, not a speed figure at all. This
level's own math lab, working from a different report's own published latency table, finds the
transfer path was still slower than fresh prefill at every length actually tested once the fixed
per-request mapping overhead is included — a 1.083×, not a win, at the longest length measured. None
of these numbers are wrong; each is honestly reported for what it specifically measures. The mistake
is repeating the largest one as if it summarized the whole system.`,
    },
    {
      wrong: 'This is a shipped, benchmarked serving technique, comparable in maturity to PagedAttention or speculative decoding.',
      right: `Treat the maturity gap as load-bearing information, not a footnote. Most of the
specific figures in this level trace to one company's own research blog published across a span of
days to weeks, evaluated on small held-out sets — 27 to 480 comparisons, not the scale an ML
benchmark claim would usually rest on — using mechanistic probes (forced rollouts, controlled
diagnostic tasks) rather than free-generation task quality at production scale. The one
peer-reviewable anchor, Heo et al., is a preprint, not a published, cited, independently-replicated
result. None of that means the mechanism is wrong — the qualitative pattern (direct fails, mapping
helps a lot, a trained residual helps further, recomputation closes the rest) shows up consistently
across every independent pair tested, including this level's own from-scratch toy reproduction. It
means the *technique* is at the stage Module 10 calls the frontier, not the stage PagedAttention or
speculative decoding reached only after multiple independent, peer-reviewed, large-scale
evaluations. Cite the magnitude of any specific number the way you would cite any single-source,
small-sample result — with the sample size attached.`,
    },
  ],

  glossary: [
    { term: 'cross-model KV cache transfer', def: 'Reusing a KV cache built by one model to skip or shorten prefill on a different, larger model in the same family, via a learned coordinate mapping rather than a direct copy.' },
    { term: 'content space', def: "A source model's K/V representation with RoPE's positional rotation removed, leaving only position-independent information about the token — the space a cross-model map is fit in." },
    { term: 'ridge mapping', def: 'A per-attention-head linear transform, fit by ridge regression (least squares with an L2 penalty) from source-model K/V vectors onto target-model K/V vectors, using an offline calibration set.' },
    { term: 'k (source-layer width)', def: 'How many top-predictive source layers are concatenated as input to one target layer\'s mapper. Larger k improves output-distribution fidelity (prefix NLL) but costs more to compute and can hurt intermediate-representation fidelity past a point.' },
    { term: 'rank-r residual adapter', def: 'A small trained correction added on top of a frozen ridge map, factored through a low-rank bottleneck (rank r) to keep its parameter count small — linear (BAz) or a factorized-quadratic, genuinely nonlinear variant.' },
    { term: 'KL32', def: 'KL divergence between a mapped-cache continuation and the native-prefill continuation, measured over a 32-step forced rollout rather than a single token — designed to catch drift that single-step agreement misses.' },
    { term: 'partial recomputation (cross-model)', def: "Recomputing a target model's last one or two layers exactly from a shared boundary, instead of mapping them, to eliminate the nonlinear-drift ceiling a pure map cannot cross." },
    { term: 'chance-normalized task-quality retention', def: '(mapped_score − chance_floor) / (oracle_score − chance_floor) — a benchmark-accuracy comparison that accounts for different multiple-choice baselines, making retention comparable across benchmarks.' },
    { term: 'answer evidence', def: "Whether a downstream probe shows the information a cached context should have carried, independent of whether the model's exact next-token continuation matches — the \"memory\" axis, distinct from trajectory fidelity." },
    { term: 'next-token top-1 agreement', def: "Whether a target model's single most-likely next token, continuing from a transferred cache, matches what it would have produced from its own native prefill." },
    { term: 'model routing (cross-model transfer)', def: 'A deployment pattern where a cheap model handles a request by default and escalates to a larger model only when needed, carrying its KV cache forward instead of discarding the reading already done.' },
    { term: 'width-pruned sibling', def: "A smaller model derived from a larger one by pruning (reducing width, depth, attention or MLP dimensions) and distillation-based retraining — e.g. NVIDIA's Minitron family — structurally closer to its parent than an independently-trained model of the same size." },
  ],
};
