export default {
  n: 11,
  slug: 'model-architectures',
  title: 'MODEL ARCHITECTURES',
  tagline: 'Every architecture is a different answer to one question: which bytes have to move to produce the next token?',
  hours: '7–9 hours',
  prereqs: ['Module 0', 'Module 2', 'Module 4'],

  bigIdea: `Module 0 traced one architecture — dense, decoder-only, grouped-query attention — and
ended on a ratio: about 15 GB moved to do 16 GFLOP of arithmetic. This module is the other half of
that lesson. **The model you traced is one point in a design space, and the space is organised
almost entirely around that ratio.**

The claim to hold on to is that you can price any language model with two numbers:

- **active bytes** — the weight bytes that must be read from memory to produce one token;
- **cache bytes** — the KV or state bytes that must be read per token, for each sequence in flight.

Everything else about an architecture is quality. Those two numbers are cost. A mixture of experts
attacks the first. Grouped-query attention, latent attention, sliding windows and recurrent states
attack the second. Hybrids attack both. And each of them pays for it somewhere: in memory
capacity, in exact recall, or in being impossible to retrofit onto a model you already have.

That last point is the practical one. Quantization, paging and batching are decisions you make on
the day you deploy. Architecture is a decision someone else made months earlier, and by the time
the weights are on your disk the two numbers are fixed. **Choosing a model to serve is mostly
choosing its bytes-per-token, and you cannot negotiate afterwards.**

So this is a level about reading a config file and predicting a bill. By the end you should be able
to look at \`n_kv_heads\`, \`n_experts\`, \`top_k\` and a layer count, and say — before you download
anything — how fast it can possibly decode and how many users will fit beside it.`,

  concepts: [
    {
      name: 'The decode-step bill: active bytes and cache bytes',
      keyPoint: 'Every architectural choice lands in one of two terms: weight bytes read per step, or cache bytes read per token per sequence.',
      body: `Module 1 established that a decode step reads the weights once to produce one token.
Module 2 added the cache that grows with context. Module 4 turned both into time. Put the three
together and one line prices any architecture:

\`\`\`
bytes per decode step  ~=  W_active  +  B x S x kv_per_token
step time              ~=  bytes / achieved bandwidth
\`\`\`

\`W_active\` is the weight bytes actually read, \`B\` is the number of sequences in flight, \`S\` is
their mean length. The asymmetry between the two terms is the thing to internalise: **the weight
term is paid once per step no matter how many sequences you are serving, and the cache term is
paid once per sequence.** That is the entire reason batching works, and it is why the two terms
have to be tracked separately rather than added into a single "model size".

For Llama-3-8B at fp16, the two numbers are 15.0 GB and 128 KiB. At batch 1 and 8k context:

\`\`\`
weights  15.0 GB
cache     8192 x 131072 = 1.07 GB
                          ---------
                          16.1 GB  ->  4.8 ms on an H100  ->  208 tok/s
\`\`\`

At batch 64 the weight term does not move and the cache term multiplies by 64, to 68.7 GB. The
model has not changed. Which term you are paying has.

Here is the whole level in one table — four families, four different bets:

| family | what it changes | term attacked | what it costs |
|---|---|---|---|
| Dense transformer | nothing; the baseline | neither | it is the thing everything else is measured against |
| Mixture of experts | one MLP becomes \`E\` MLPs, \`k\` read per token | \`W_active\` | memory capacity — every expert must be resident |
| MQA · GQA · MLA · cross-layer sharing | fewer or smaller cached vectors | \`kv_per_token\` | quality, in amounts that vary from negligible to real |
| Sliding window · local-global | old tokens fall out of the window | \`S\` in most layers | exact recall of distant tokens on those layers |
| SSM · linear attention · hybrid | a fixed-size state replaces the cache | \`S\` entirely, in most layers | recall again, harder — Module 7 has the argument |

Two warnings before the details. First, **the axis an architecture does not attack tends to get
worse**, because designers spend the savings: MoE models are enormous, MLA models are dense, and
hybrids that keep a few attention layers keep a cache for them. Second, quantization scales both
terms and is not on this list, because it is a deployment choice rather than an architectural one.
Module 6 covers it, and the two compose — a fp8 MoE moves half of a small number.`,
      ascii: `  ONE DECODE STEP,  batch B,  mean context S

  ┌──────────────────────────────┬──────────────────────────────────┐
  │ W_active                     │ B x S x kv_per_token             │
  │ read once per STEP           │ read once per SEQUENCE           │
  │ independent of B and S       │ grows with BOTH                  │
  └──────────────────────────────┴──────────────────────────────────┘
             ^                                   ^
             │                                   │
     mixture of experts                 GQA · MLA · sliding window
     cuts this term                     · recurrent state cut this one
             │                                   │
     pays in CAPACITY                    pays in RECALL`,
    },
    {
      name: 'Decoder-only, and the two shapes it beat',
      keyPoint: 'An encoder-decoder computes its cross-attention keys and values once per input and never grows them; what made decoder-only win was generalization, not caching.',
      body: `Three shapes came out of the original transformer, and only one of them is what you
serve today.

**Encoder-only** — BERT and its descendants. Bidirectional attention over a fixed input, no
generation. Still the right tool for classification, retrieval embeddings and reranking, and still
deployed everywhere in the retrieval half of a RAG system. It has no decode loop, so none of this
course applies to it.

**Encoder-decoder** — T5, BART, Whisper. The encoder reads the input bidirectionally once. The
decoder generates, and each decoder layer has two attention blocks: self-attention over what it has
generated so far, and **cross-attention** over the encoder's output.

The inference-relevant fact about that shape is one people usually get backwards. The
cross-attention K and V are projections of the encoder output, so they are computed **once per
request and never grow**. Only the decoder's self-attention cache grows, and it grows with the
*output* length rather than the input length. For a task with a long input and a short output —
summarization, translation, transcription — an encoder-decoder has a smaller and flatter cache
profile than a decoder-only model doing the same job. Whisper is the surviving example most people
have running: its encoder consumes a fixed 30-second window of mel spectrogram, so its
cross-attention cache is a constant.

So encoder-decoder is not the expensive shape. It is the **inflexible** one: two parameter sets to
hold, an input that must be fully encoded before any output appears, and no way to append to the
encoded region without recomputing it.

**Decoder-only** won, and it is worth being accurate about why. The engineering conveniences are
real — one set of weights, one code path for prefill and decode, and a cache that stays valid as
the sequence grows (Module 0's causality argument). But the published evidence is about quality:
Wang et al. compared architectures and objectives head to head and found causal decoder-only models
trained with a plain autoregressive objective had the strongest zero-shot generalization after
unsupervised pretraining. The systems properties are why nobody has been tempted back.

**Prefix-LM** is the middle ground that keeps coming up and never quite arrives: bidirectional
attention over the prompt, causal attention over the continuation. UL2's S-denoising is exactly
this, and PaliGemma uses it so that image and prompt tokens see each other in both directions. The
serving consequence is the one that keeps it niche — a bidirectional region cannot be extended
without recomputing it, so multi-turn chat, where each turn appends to the previous prompt, loses
the prefix caching from Module 6.`,
      ascii: `  DECODER-ONLY                     ENCODER-DECODER

  ┌───────────────────────┐        ┌────────────────────────┐
  │ self-attn KV          │        │ cross-attn K,V         │  encoder output,
  │ grows with            │        │ FIXED SIZE             │  projected ONCE
  │ prompt + output       │        ├────────────────────────┤
  └───────────────────────┘        │ self-attn KV           │  grows with
                                   │ grows with OUTPUT only │  the generation
  one parameter set                └────────────────────────┘
  one code path                    two parameter sets
  prefix caching works             input must be encoded first`,
    },
    {
      name: 'The cache axis: MHA, MQA, GQA, MLA, and sharing across layers',
      keyPoint: 'The cache formula has exactly two architectural terms, n_kv_heads and head_dim, and every attention variant since 2019 is an attack on one of them — or on the layer count in front of them.',
      body: `Module 2 derived the formula and Module 6 spent the savings. Here it is as a family
tree, because the variants are usually presented as a list of tricks when they are a single idea
applied at three different points:

\`\`\`
cache bytes per token  =  2 x n_layers x n_kv_heads x head_dim x dtype_bytes
                          ^        ^          ^          ^         ^
                          |        |          |          |         └─ quantization (Module 6)
                          |        |          |          └─ MLA compresses this jointly
                          |        |          └─ MQA and GQA cut this
                          |        └─ cross-layer sharing cuts this
                          └─ MLA removes the factor of 2
\`\`\`

Hold the Llama-3-70B shape fixed — 80 layers, \`head_dim\` 128, fp16 — and vary only the attention
design:

| variant | cached per token per layer | bytes per token | vs MHA |
|---|---|---|---|
| MHA, 64 KV heads | \`2 × 64 × 128\` = 16,384 values | 2.5 MiB | 1× |
| GQA-8 (what Llama 3 ships) | \`2 × 8 × 128\` = 2,048 values | 320 KiB | 8× |
| MQA, 1 KV head | \`2 × 1 × 128\` = 256 values | 40 KiB | 64× |
| MLA, rank 512 + 64 rotary | 576 values, K and V together | 90 KiB | 28× |

MQA is not the winner despite the best number, because a single shared KV head costs more quality
than most models can afford; GQA at 4 to 8 groups is where the field settled. Module 6 has the
mechanism for MLA and the DeepSeek result — a 93% reduction against MHA with benchmark quality that
went *up* rather than down.

The term this level adds is the one on the left. **Cross-layer KV sharing** lets neighbouring
layers reuse a single set of K and V, cutting \`n_layers\` in the formula rather than the head
count. Character.AI reported stacking three of these attacks in production: MQA, a 5:1 pattern of
local to global layers with a 1024-token horizon, and cross-layer sharing worth a further 2–3×,
for a combined reduction of more than 20× with no quality regression they could measure. That is
the shape of the modern answer — not one clever attention variant, but three multiplicative ones
chosen because they compose.

And the constraint that governs all of it, restated because it is the reason this level exists:
**these are training-time decisions.** GQA has an uptraining recipe that costs about 5% of original
pre-training compute. MLA does not — it must be trained from scratch. Cross-layer sharing changes
what the model *is*. When you pick a model to serve you are picking a row of that table, and the
row is not negotiable.`,
      ascii: '',
    },
    {
      name: 'The weight axis: mixture of experts',
      keyPoint: 'MoE reads k of E expert blocks per token, which is a large win at batch 1 and a shrinking one as the batch grows, because a big enough batch touches every expert anyway.',
      body: `Module 9 covered how to *serve* a mixture of experts — expert parallelism, all-to-all,
load imbalance. This is the architecture, and the arithmetic underneath the headline numbers.

An MoE layer replaces the single MLP with \`E\` expert MLPs and a router — a small linear layer
producing one score per expert — that sends each token to the top \`k\` of them. Routing is
per token and per layer, so a token can use different experts at every depth.

Start with the reconciliation everyone skips. Mixtral-8x7B is not 56B parameters, and it does not
activate 14B. **Only the MLP is replicated.** Attention, embeddings and norms are shared by all
experts:

\`\`\`
per layer, attention (GQA-8, d_model 4096):        41,943,040
per layer, one expert (SwiGLU, ffn 14336):        176,160,768
per layer, eight experts:                       1,409,286,144
per layer, total:                               1,451,229,184
x 32 layers:                                   46,439,333,888
+ embedding and LM head (32000 x 4096, x2):       262,144,000
                                               --------------
                                               46,701,477,888   = 46.7B, the published figure

active per token:  41,943,040 + 2 x 176,160,768 = 394,264,576
x 32 layers + LM head:                           12,878,610,432   = 12.9B
\`\`\`

The **activation ratio** — active over total — is the number to read off a model card. Mixtral is
12.9/46.7 = 28%. Qwen3-235B-A22B is 22/235 = 9%, with 128 experts and top-8 and no shared expert at
all. DeepSeek-V3 is 37/671 = 5.5%, with 256 routed experts plus one shared expert and top-8
routing, the first three layers left dense. The trend is unmistakable: **more experts, each
smaller, a smaller fraction active.** That is DeepSeekMoE's contribution — fine-grained expert
segmentation, so routing decisions are finer, plus an isolated shared expert that every token reads
and that therefore absorbs the knowledge which would otherwise be duplicated into all 256.

Routing details worth knowing because they change behaviour under load: Switch Transformer routes
to exactly one expert, GShard to at most two, and DeepSeek-V3 replaced softmax gating and auxiliary
balancing losses with sigmoid affinities and a bias term adjusted during training to keep experts
balanced without a loss that fights the objective.

Now the part that the model card does not tell you. **The activation ratio is a per-token number,
and a batch does not read one token's experts — it reads the union of the batch's experts.** Under
uniform routing, a batch of \`B\` tokens touches

\`\`\`
E x (1 - (1 - k/E)^B)   distinct experts
\`\`\`

For Mixtral, that is 2 experts at batch 1, 7.2 at batch 8, and all 8 by batch 32. For DeepSeek-V3
it is 8 at batch 1, 163 at batch 32, and 252 of 256 by batch 128. The consequence, stated as
carefully as it deserves:

- **Bytes per step rise with batch** for an MoE and are flat for a dense model. At a large enough
batch the step reads every expert, so it moves the bytes of its *total* size, not its active size.
- **Bytes per token still fall**, because those bytes are amortized over \`B\` tokens. MoE does not
stop working at scale.
- What erodes is the *latency* advantage. An MoE at batch 1 decodes like a small model. The same
model at batch 512 has a step time set by its full weight footprint, and what it keeps is the FLOP
saving — which matters on the other side of the ridge point, where Module 4 says you are
compute-bound anyway.

So the honest summary is that **MoE is a bandwidth optimization at small batch and an arithmetic
optimization at large batch**, and it is worth knowing which one you are buying. Module 9's note
that larger batches smooth routing imbalance is the same phenomenon seen from the scheduler's side:
by the time the batch is big enough to balance, it is big enough to touch everything.

One last practical fact. You can convert a dense checkpoint into an MoE — **sparse upcycling**
copies the dense MLP into every expert, initialises a fresh router, and continues training, at
roughly half the sunk cost of the original pretraining. It is the only entry in this level that is
even partly retrofittable, and "half of a pretraining run" is not a deployment-day decision.`,
      ascii: `  ROUTING, one MoE layer, top-2 of 8

     token ──> router ──> scores [8] ──> pick 2
                                          │
     ┌────┬────┬────┬────┬────┬────┬────┬─┴──┐
     │ e0 │ e1 │ e2 │ e3 │ e4 │ e5 │ e6 │ e7 │   all 8 resident
     └────┴────┴────┴─██─┴────┴────┴─██─┴────┘   2 of 8 read
                      └─────┬─────────┘
                            v  weighted sum back into the residual stream

  batch 1:   2 of 8 experts read   ->  22.6 GB per step
  batch 8:   7.2 of 8               ->  81.2 GB per step
  batch 32:  8 of 8                 ->  90.2 GB per step   <- now it reads
                                                              like a 47B dense`,
    },
    {
      name: 'The context axis: sliding windows and local-global stacks',
      keyPoint: 'A window turns the cache from linear in context to constant, and interleaving a few global layers buys back the long-range recall the window gives up.',
      body: `The cache term has one more factor that no attention variant touches: \`S\`, the number
of tokens you keep. Bound it and the cache stops growing.

**Sliding-window attention** lets each position attend only to the last \`w\` tokens. Mistral 7B
shipped with \`w = 4096\` inside an 8192-token context; later Mistral models dropped it and
Ministral reintroduced it in interleaved form. The immediate objection — that the model then cannot
use anything older than the window — is wrong, and worth working through. Attention composes across
depth: a token at layer 2 attends to positions that themselves attended \`w\` tokens further back,
so the **receptive field grows roughly as \`layers × w\`**. What a window costs is not connectivity
but *exact* recall, because information from far away arrives blurred through many layers rather
than retrieved directly. That is the same trade Module 7 describes for state space models, applied
per layer instead of per model.

**Local-global interleaving** buys the exact recall back cheaply. Make most layers local and a few
global. Gemma 2 alternated 1:1 with a 4096-token window. Gemma 3 moved to 5:1 with a 1024-token
window, and only the global layers carry the long-context RoPE scaling. Character.AI's production
stack used the same 5:1 shape with a 1024 horizon.

The arithmetic is the reason. Take 48 layers, GQA-8, \`head_dim\` 128, fp16 — 4 KiB per token per
layer — and a 32k context:

\`\`\`
all-global:      48 layers x 32768 tokens x 4 KiB        =  6.0 GiB
5:1 local-global: 8 global x 32768 x 4 KiB               =  1.0 GiB
                 40 local  x  1024 x 4 KiB               =  0.16 GiB
                                                            ---------
                                                            1.15 GiB   5.2x smaller
\`\`\`

Note where the saving comes from and where it stops. The local layers are already saturated at 1024
tokens, so their contribution is **constant from 1k of context onward** — doubling the context
doubles only the global part. As \`S\` grows the ratio approaches \`L / (L/6)\` = 6×, and it gets
there quickly. This is the cheapest long-context lever there is, which is why every recent model
family has adopted some version of it.

Two things to watch when serving one. Prefix caching (Module 6) has to know the window, because a
cached block older than \`w\` is useful to the global layers and dead weight for the local ones —
engines that page uniformly across layers will happily store both. And the attention-sink result
from Module 6 is the reason naive windowing was worse than it should have been: keep the first few
tokens pinned alongside the window and quality recovers, because those positions absorb attention
mass regardless of their content.`,
      ascii: `  CACHE PER LAYER, 5:1 local-global, w = 1024

  layer  1  local   [....1024....]                 constant
  layer  2  local   [....1024....]                 constant
  layer  3  local   [....1024....]                 constant
  layer  4  local   [....1024....]                 constant
  layer  5  local   [....1024....]                 constant
  layer  6  GLOBAL  [...............32768.......]  grows with context
  ...
                     ^                          ^
                     └── 5/6 of layers flat ─────┘  only 1/6 grows`,
    },
    {
      name: 'Removing the cache: recurrent states and hybrid stacks',
      keyPoint: 'A recurrent layer stores the same bytes at 1k tokens as at 1M, which moves the capacity question from context length to concurrency.',
      body: `The last move is to delete the cache. A state space model or linear-attention layer
carries a **fixed-size recurrent state** instead: every token updates it, and nothing accumulates.
Module 7 covers the mechanism and the reason pure versions lost — a fixed state cannot hold an
arbitrary amount of detail, so exact recall degrades in a way that is information-theoretic rather
than an implementation shortfall. Here the question is only what it costs to serve.

For Mamba-2 the state is \`d_inner × d_state\` values per layer, plus a small convolution cache a
few timesteps wide. Take the 2.7B checkpoint — \`d_model\` 2560, expand 2, \`d_state\` 128, 64
layers:

\`\`\`
per layer:   5120 x 128            =    655,360 values
x 64 layers:                       = 41,943,040 values
at bf16:                           =      83.9 MB per sequence
\`\`\`

83.9 MB at 1k tokens. 83.9 MB at 1M tokens. **The state does not know how long the sequence is.**
Compare that to a KV cache, which at 128k context and GQA-8 is measured in gigabytes per sequence,
and the shape of the trade is clear: a recurrent model's memory scales with **concurrency only**,
where a transformer's scales with concurrency **times** context.

Nobody ships the pure version. **Hybrids** interleave a few full-attention layers among many
recurrent ones, which is enough to restore exact retrieval while keeping most of the memory
saving. The published ratios vary more than the folklore suggests:

| model | pattern | scale |
|---|---|---|
| Jamba | 1 attention : 7 Mamba, MoE every second layer | 52B total, 12B active |
| MiniMax-01 | 7 lightning-attention layers : 1 softmax, 10 of 80 | 456B total, 45.9B active |
| Nemotron-H | Mamba-2 with roughly 8% attention layers | 8B and 56B |
| Qwen3-Next | 1 full attention per 3 linear-attention layers | 80B total |
| Falcon-H1 | attention and SSM heads run in *parallel* inside a block | 0.5B to 34B |

Jamba's own table is the cleanest statement of the payoff: at 256k context with a 16-bit cache it
needs about **4 GB**, against 32 GB for Mixtral and 128 GB for Llama-2-70B at the same context.
That is the entire argument for hybrids in one row.

What it does to a serving stack is less advertised. You now have two allocators: a paged KV cache
for the attention layers, and a constant-size state block per sequence for the rest. The state is
allocated when the sequence starts and freed when it ends, which makes admission control simpler
and preemption harder — you cannot evict half a state the way you can drop and recompute KV blocks.
Prefix caching works only on the attention layers unless the engine also snapshots the recurrent
state at block boundaries, and a snapshot costs the same fixed bytes whether the prefix is 100
tokens or 100,000.`,
      ascii: `  CACHE BYTES vs CONTEXT, one sequence

   bytes
     ^
     │                                        ,-'  transformer, GQA-8
     │                                    ,-'      (linear in S)
     │                                ,-'
     │                            ,-'
     │                        ,-'
     │                    ,-'
     │                ,-'          hybrid 1:7 (linear, 1/8 the slope)
     │            ,-'         ...................................
     │        ,-'   ..........
     │    ,-'.......
     │ ,.'..
     ├───────────────────────────────────────────────  pure SSM: FLAT
     └────────────────────────────────────────────────>  context S
       1k        8k        32k       128k      512k`,
    },
    {
      name: 'The boring shapes that decide how it serves',
      keyPoint: 'Layer count sets how many fixed costs you pay per token, head counts set how many ways you can shard, and vocabulary sets a floor that small models cannot get under.',
      body: `Three parts of a config file have nothing to do with the interesting architecture
debates and quite a lot to do with your latency.

**Depth.** Under tensor parallelism a decode step costs two all-reduces per layer, one after
attention and one after the MLP, so an 80-layer model pays 160 collectives per token. Those
collectives carry small messages, so their cost is mostly fixed latency that does not shrink when
you add GPUs — Module 9 measures it. The same is true of kernel launches, though CUDA graphs remove
most of that particular tax. **A deep-narrow model and a shallow-wide model with equal parameter
counts do not decode equally**, and the deep one loses. It often wins on quality per parameter,
which is why the trade keeps being made.

**Head counts.** Tensor parallelism splits attention by heads, so \`n_heads\` must divide by the TP
degree. The one that surprises people is the KV side. When the TP degree exceeds \`n_kv_heads\`,
engines do not fail — they replicate. A GQA-8 model at TP=16 gives every rank one KV head, held
twice across the group, so **your KV cache costs 2× what the formula says**. The cache arithmetic
from Module 2 silently stops being right at the moment you scale past 8 ways, and nothing in the
logs tells you.

**Vocabulary.** The LM head is a \`d_model × vocab\` matrix and it is read on every decode step. It
has grown: GPT-2 at 50,257, Llama-2 at 32,000, Llama-3 at 128,256, Qwen3 at 151,936, Gemma 3 at
262,208. On Llama-3-8B that is \`4096 × 128256 × 2\` = 1.05 GB of the 15.0 GB read per token — 7%,
noticeable but not decisive. On a 0.6B model with a 151,936-token vocabulary it is 311 MB of about
1.2 GB, **a quarter of the entire decode cost spent on the output projection**. This is why small
models tie their embeddings: Gemma ties at every size, Qwen3 ties at 4B and below, and Llama does
not tie at all. Tying saves one \`vocab × d_model\` matrix, which is rounding error on a 70B model
and a design constraint on a 600M one.

The general lesson is that the config fields nobody argues about on release day are the ones that
decide whether the model shards cleanly, and a model that does not shard cleanly is a model you
serve at 2× the cache cost forever.`,
      ascii: '',
    },
    {
      name: 'Multimodal: the architecture that changes the shape of the prefill',
      keyPoint: 'In the dominant VLM design an image becomes several hundred to several thousand ordinary positions, each costing exactly what a text token costs.',
      body: `A vision-language model is usually three parts: a **vision encoder** (a ViT such as CLIP
or SigLIP), a **projector** that maps its output into the language model's embedding space, and the
language model itself. LLaVA-1.5 uses CLIP ViT-L/14 at 336px with a two-layer MLP projector;
PaliGemma pairs SigLIP with Gemma; Qwen2-VL uses a ViT with an MLP token merger.

The arithmetic that matters is how many positions an image becomes. A ViT at 336px with patch size
14 produces \`(336/14)² = 24 × 24 = 576\` patches, and LLaVA-1.5 passes all of them through, so one
image is 576 tokens. Higher-resolution schemes tile: LLaVA-NeXT's grid reaches 2,880 tokens per
image, and Qwen2-VL's dynamic resolution maps any input size to a variable count bounded in the
thousands.

Then the point of this level: **those are ordinary positions.** They cost the same prefill FLOPs
per token, and the same cache bytes per token, as text. On a Llama-3-8B-shaped model at 128 KiB per
token:

\`\`\`
one 576-token image:    576 x 131072   =  72 MiB of KV
one 2880-token image:  2880 x 131072   = 360 MiB of KV
a 40-token text prompt:   40 x 131072  = 5.0 MiB
\`\`\`

A single image is worth fourteen to seventy text prompts. So in an image-heavy request with a short
question, prefill is dominated by pixels, the cache is dominated by pixels, and every scheduling
decision from Module 5 — chunked prefill, admission control, the prefill/decode interference
trade — is being made about image tokens whether or not the scheduler knows it.

The exception is the **cross-attention** family: Flamingo and Llama 3.2 Vision feed image features
into the language model through dedicated cross-attention layers instead of inserting them into the
sequence. Image content then never enters the self-attention cache at all; it adds a separate,
fixed-size K and V, exactly the encoder-decoder structure from earlier in this level. It costs
extra parameters and buys a cache that does not grow with image count — the same trade, made in a
different place, for the third time in this module.

Two practical notes. The vision encoder is a second model resident on the same GPU, with its own
weights and its own batch behaviour, and it runs in the prefill phase only. And prefix caching over
images requires hashing the image content rather than the token ids, which most engines now do —
worth checking, because in a multi-turn conversation about one image the image is precisely the
part that repeats.`,
      ascii: '',
    },
  ],

  mathLab: {
    prompt: `Price three architectures on the same hardware, then explain the number that comes
out wrong. Assume H100 SXM: 3.35 TB/s, 80 GB, call it 76 GB usable after activations and
workspace.

\`\`\`
A. Llama-3-70B          fp16 weights, fp16 cache
   n_layers 80, d_model 8192, n_heads 64, n_kv_heads 8, head_dim 128
   ffn 28672 (SwiGLU, three matrices), vocab 128256, untied

B. Mixtral-8x7B         fp16 weights, fp16 cache
   n_layers 32, d_model 4096, n_heads 32, n_kv_heads 8, head_dim 128
   ffn 14336 per expert, 8 experts, top-2, vocab 32000, untied

C. DeepSeek-V3          fp8 weights, bf16 cache
   n_layers 61 — the first 3 dense, the remaining 58 MoE
   d_model 7168, ffn 18432 in a dense layer, 2048 per expert
   256 routed experts + 1 shared, top-8, vocab 129280, untied
   MLA caches kv_lora_rank 512 plus a 64-wide decoupled RoPE key
   MLA projections per layer:  q_a 7168x1536,  q_b 1536x24576,
                               kv_a 7168x576,  kv_b 512x32768,  o 16384x7168
\`\`\`

1. Weight bytes read per decode token, for each. The input embedding is a lookup, not a matmul.
2. Cache bytes per token per sequence, for each.
3. Resident bytes, and the minimum number of H100s that can hold the weights.
4. The batch-1 decode step at 8k context, assuming you get the full aggregate bandwidth of that
   many GPUs. Convert to tokens per second.
5. The context length at which the cache term equals the weight term, at batch 1 and at batch 64.
6. Your answer to question 4 for DeepSeek-V3 is a number nobody has ever measured at batch 1. What
   does the arithmetic assume that the deployment does not deliver?`,

    solution: `**1. Weight bytes per token**

**A. Llama-3-70B.** Per layer:

\`\`\`
attention:  W_q  8192 x 8192  =  67,108,864
            W_k  8192 x 1024  =   8,388,608
            W_v  8192 x 1024  =   8,388,608
            W_o  8192 x 8192  =  67,108,864
                                 -----------
                                 150,994,944

mlp:        3 x 8192 x 28672  = 704,643,072
per layer:                      855,638,016
x 80 layers:                 68,451,041,280
+ embedding  128256 x 8192  =  1,050,673,152
+ lm_head    8192 x 128256  =  1,050,673,152
                              --------------
                              70,552,387,584   = 70.55B
\`\`\`

At fp16 that is 141.10 GB resident. Subtract the embedding table, which is a row lookup:
**139.00 GB read per token.**

**B. Mixtral-8x7B.** The trap is that 8 × 7B is not 56B — only the MLP is replicated:

\`\`\`
attention per layer:                        41,943,040
one expert, 3 x 4096 x 14336:              176,160,768
eight experts:                           1,409,286,144
per layer total:                         1,451,229,184
x 32:                                   46,439,333,888
+ embedding + lm_head, 2 x 32000 x 4096:   262,144,000
                                        --------------
                                        46,701,477,888   = 46.70B  -> 93.40 GB at fp16

active: 41,943,040 + 2 x 176,160,768   =   394,264,576
x 32 + lm_head 131,072,000              = 12,747,538,432 read
\`\`\`

**25.50 GB read per token.** (Counting the embedding table too gives the 12.9B on the model card;
only 12.75B of it is actually read.)

**C. DeepSeek-V3.** Four groups:

\`\`\`
one expert, 3 x 7168 x 2048:                    44,040,192
257 experts (256 routed + 1 shared) per layer: 11,318,329,344
x 58 MoE layers:                              656,463,101,952

3 dense layers, 3 x 3 x 7168 x 18432:           1,189,085,184

MLA per layer:  q_a  7168 x 1536  =  11,010,048
                q_b  1536 x 24576 =  37,748,736
                kv_a 7168 x 576   =   4,128,768
                kv_b 512 x 32768  =  16,777,216
                o    16384 x 7168 = 117,440,512
                                    -----------
                                    187,105,280   x 61 = 11,413,422,080

embedding + lm_head, 2 x 129280 x 7168:         1,853,358,080
                                              ---------------
                                              670,918,967,296   = 670.9B  ✓ "671B"
\`\`\`

Active: 9 experts of 257 per MoE layer, everything else unchanged.

\`\`\`
9 x 44,040,192 x 58    = 22,988,980,224
+ dense layers          =  1,189,085,184
+ MLA, all 61 layers    = 11,413,422,080
+ lm_head               =    926,679,040
                          --------------
                          36,518,166,528   = 36.5B  ✓ "37B"
\`\`\`

At fp8, **36.5 GB read per token** out of 671 GB resident.

**2. Cache bytes per token**

\`\`\`
A:  2 x 80 x 8 x 128 x 2  = 327,680 B = 320 KiB
B:  2 x 32 x 8 x 128 x 2  = 131,072 B = 128 KiB
C:      61 x 576     x 2  =  70,272 B = 68.6 KiB      no factor of 2 — the latent is K and V
\`\`\`

**3. Resident bytes and GPUs**

\`\`\`
A:  141.10 GB / 76  = 1.86  ->  2 H100s
B:   93.40 GB / 76  = 1.23  ->  2 H100s
C:  671.0  GB / 76  = 8.83  ->  9, in practice 16 H100s or 8 H200s (1,128 GB)
\`\`\`

Note what just happened to B. Mixtral reads 25.50 GB per token, less than a dense 13B — but it
needs the same two GPUs as the 70B, because capacity is set by the total and bandwidth by the
active. **That is the MoE trade in one line.**

**4. The batch-1 step at 8k**

\`\`\`
A:  139.00 + 8192 x 327,680 = 139.00 + 2.68 = 141.68 GB / 6.70 TB/s = 21.1 ms ->    47 tok/s
B:   25.50 + 8192 x 131,072 =  25.50 + 1.07 =  26.57 GB / 6.70 TB/s =  4.0 ms ->   252 tok/s
C:   36.50 + 8192 x  70,272 =  36.50 + 0.58 =  37.08 GB / 53.6 TB/s =  0.7 ms -> 1,446 tok/s
\`\`\`

A 671B model looks nine times faster than a 70B one. The first part of that is real and is the
whole point of the level: **36.5 GB moved beats 139.00 GB moved, and parameter count did not enter
the calculation.** The second part is not, which is question 6.

**5. Where the cache catches the weights**

Set \`W = B × S × kv\` and solve for \`S\`:

\`\`\`
              batch 1        batch 64
A:   139.00e9 / 327,680   =  424,000 tokens      6,600 tokens
B:    25.50e9 / 131,072   =  195,000 tokens      3,000 tokens
C:    36.50e9 /  70,272   =  520,000 tokens      8,100 tokens
\`\`\`

At batch 1 the cache is irrelevant for every architecture here — you would need a context longer
than any of them supports. At batch 64 it dominates from a few thousand tokens onward. **The cache
term is a concurrency problem wearing a context-length costume**, which is why Module 6 is about
fitting more sequences and not about fitting longer ones.

**6. What the arithmetic assumes**

It assumes all 16 GPUs are streaming useful bytes at the same time. At batch 1 with expert
parallelism, they are not. A GPU holding 16 of the 256 experts is asked for zero or one of the
eight the token selected, so most of the fleet reads nothing it needs and contributes none of its
bandwidth. On top of that, each MoE layer costs two all-to-alls whose latency does not shrink as
you add GPUs (Module 9), and there are 58 of them.

Aggregate bandwidth only becomes real when the batch is large enough to keep every expert busy —
and from the routing arithmetic, that is roughly the batch at which the model reads all 656 GB of
its experts anyway. **Batch 1 on a 671B sparse model is the worst operating point the architecture
has**, and it was designed by people who intended to run it at the other end.`,
  },

  codeLab: {
    goal: `Write the cost model as code: a config in, the two numbers out. Then use it to see the
thing that is hard to see from a model card — that an MoE's step reads more of itself as the batch
grows, until it reads all of itself.

No GPU, no dependencies, and no model downloads. The point is that everything in this level is
derivable from a config file, and once you have typed the derivation once you will read config
files differently.`,
    code: `"""Price an architecture in two numbers: active bytes and cache bytes.

    python3 arch_cost.py        # standard library only

Every model below is described by its config, not by its marketing size. The
point of the exercise is that the two numbers -- not the parameter count --
predict the decode step.
"""

from dataclasses import dataclass

BW = 3350.0e9          # H100 SXM, bytes/second
GB = 1e9


@dataclass
class Arch:
    name: str
    n_layers: int
    d_model: int
    n_heads: int
    n_kv_heads: int
    head_dim: int
    ffn: int
    vocab: int
    n_experts: int = 1        # 1 means a dense MLP
    top_k: int = 1
    n_shared: int = 0         # DeepSeek-style always-on expert
    w_bytes: float = 2.0      # bytes per weight
    kv_bytes: float = 2.0     # bytes per cached value
    latent: int = 0           # MLA: values cached per token per layer
    window: int = 0           # sliding-window size, 0 = every layer sees everything
    global_every: int = 1     # 1 = every layer is global
    reported: tuple = ()      # (total, active) from the tech report, for models
                              # the generic formula below does not describe


def params(a):
    """(total, active-per-token) parameter counts, ignoring norms and router."""
    if a.reported:
        return a.reported
    attn = (a.d_model * a.n_heads * a.head_dim
            + 2 * a.d_model * a.n_kv_heads * a.head_dim
            + a.n_heads * a.head_dim * a.d_model)
    expert = 3 * a.d_model * a.ffn
    per_layer_total = attn + expert * (a.n_experts + a.n_shared)
    per_layer_active = attn + expert * (a.top_k + a.n_shared)
    emb = a.vocab * a.d_model
    total = 2 * emb + a.n_layers * per_layer_total
    # the input embedding is a row lookup, not a matmul: only the LM head is read
    active = emb + a.n_layers * per_layer_active
    return total, active


def kv_bytes_per_token(a):
    """Cache bytes for one token in one global layer."""
    values = a.latent if a.latent else 2 * a.n_kv_heads * a.head_dim
    return values * a.kv_bytes


def kv_total(a, ctx):
    """Cache bytes for one sequence at this context length."""
    per_layer = kv_bytes_per_token(a)
    n_global = a.n_layers // a.global_every
    n_local = a.n_layers - n_global
    kept_local = min(ctx, a.window) if a.window else ctx
    return per_layer * (n_global * ctx + n_local * kept_local)


def step(a, batch, ctx):
    total, active = params(a)
    w_read = active * a.w_bytes
    cache = batch * kv_total(a, ctx)
    ms = (w_read + cache) / BW * 1e3
    return dict(total=total, active=active, w_read=w_read, cache=cache,
                resident=total * a.w_bytes, ms=ms)


MODELS = [
    Arch("Llama-3-8B", 32, 4096, 32, 8, 128, 14336, 128256),
    Arch("Llama-3-70B", 80, 8192, 64, 8, 128, 28672, 128256),
    Arch("Mixtral-8x7B", 32, 4096, 32, 8, 128, 14336, 32000, n_experts=8, top_k=2),
    # MLA changes the q and kv projections and the first three layers are dense,
    # so the formula above does not describe it. Take the two headline numbers
    # from the technical report and derive only the cache term.
    Arch("DeepSeek-V3 (fp8)", 61, 7168, 128, 128, 128, 2048, 129280,
         n_experts=256, top_k=8, n_shared=1, w_bytes=1.0, latent=576,
         reported=(671e9, 37e9)),
    Arch("local-global 5:1", 48, 4096, 32, 8, 128, 14336, 128256,
         window=1024, global_every=6),
]

print("=== Part 1: the two numbers ===\\n")
print(f"{'model':20s} {'total':>9s} {'active':>9s} {'resident':>10s} "
      f"{'read/token':>11s} {'KV/token':>10s}")
for a in MODELS:
    s = step(a, 1, 8192)
    kvt = kv_total(a, 8192) / 8192
    print(f"{a.name:20s} {s['total']/1e9:8.1f}B {s['active']/1e9:8.1f}B "
          f"{s['resident']/GB:9.1f}G {s['w_read']/GB:10.1f}G {kvt/1024:9.1f}K")

print("\\n=== Part 2: the decode step, batch 1 and batch 64 ===\\n")
print(f"{'model':20s} {'B=1 8k':>10s} {'B=64 8k':>10s} {'B=64 128k':>11s}")
for a in MODELS:
    print(f"{a.name:20s} {step(a, 1, 8192)['ms']:9.1f}ms "
          f"{step(a, 64, 8192)['ms']:9.1f}ms {step(a, 64, 131072)['ms']:10.1f}ms")

print("\\n=== Part 3: where the cache term overtakes the weights ===\\n")
for a in MODELS:
    _, active = params(a)
    w = active * a.w_bytes
    per_tok = kv_total(a, 131072) / 131072
    for batch in (1, 64):
        print(f"  {a.name:20s} batch {batch:3d}: cache == weights at "
              f"{w / (batch * per_tok):9,.0f} tokens of context")

print("\\n=== Part 4: an MoE reads more of itself as the batch grows ===\\n")


def distinct_experts(E, k, B):
    """Expected distinct experts touched by B tokens under uniform top-k routing."""
    return E * (1.0 - (1.0 - k / E) ** B)


for (E, k, expert_gb, label) in [
        (8, 2, 90.2, "Mixtral-8x7B    8 experts, top-2, 90.2 GB of expert weights"),
        (256, 8, 656.5, "DeepSeek-V3   256 experts, top-8, 656 GB of expert weights at fp8")]:
    print(f"  {label}")
    print(f"    {'batch':>7s} {'experts touched':>17s} {'GB per step':>12s} {'GB per token':>13s}")
    for B in (1, 8, 32, 128, 512):
        d = distinct_experts(E, k, B)
        per_step = expert_gb * d / E
        print(f"    {B:7d} {d:9.1f} of {E:3d} {per_step:12.1f} {per_step/B:13.2f}")
    print()

# --- TODO for you ---------------------------------------------------------
#   1. Add a hybrid: replace the KV cache of 7 layers out of 8 with a constant
#      per-sequence state of d_inner x d_state values. Find the context length
#      at which it beats the GQA model on total cache bytes at batch 64.
#   2. Teach Arch to express MLA properly (q_a, q_b, kv_a, kv_b, o) and check
#      that you reproduce 671B total and 37B active without the override.
#   3. Real routers are not uniform. Re-run Part 4 with a skewed distribution
#      -- expert i chosen with probability proportional to 1/(i+1) -- and say
#      which way the curve moves, and why that is bad news for the GPU holding
#      expert 0.
`,
    expect: `Part 1 reproduces every number this level has quoted: 15.0 GB and 128 KiB for
Llama-3-8B, 139.0 GB and 320 KiB for the 70B, 25.5 GB read against 93.4 GB resident for Mixtral,
36.5 GB read against 671 GB resident for DeepSeek-V3, and 68.6 KiB per token for MLA against 320
KiB for GQA-8. If your Mixtral total is 56B you replicated the attention block as well as the MLP.

Part 2 is the level's headline: at batch 1 and 8k context Mixtral steps in 7.9 ms against
Llama-3-70B's 42.3 ms, despite holding two thirds of its parameters. The last column is the other
lesson — at batch 64 and 128k context every model is dominated by the cache term, and the ranking
changes: the local-global model at 91.8 ms beats the dense 8B at 332.7 ms, because five of every
six of its layers stopped growing at 1,024 tokens.

Part 3 prints the crossover contexts: ~424,000 tokens for Llama-3-70B at batch 1, but 6,628 at
batch 64. Every model shows the same 64× collapse, which is the point — the cache term is about
concurrency.

Part 4 is the one worth staring at. Mixtral touches 2 of 8 experts at batch 1 and all 8 by batch
32, so bytes per step go 22.6 GB, 81.2 GB, 90.2 GB while bytes per token fall 22.55, 10.15, 2.82.
Both statements are true simultaneously and people usually only know one of them. DeepSeek-V3 shows
the same curve stretched: 8 of 256 at batch 1, 163 at batch 32, 252 at batch 128.

These are single-device figures. In a real expert-parallel deployment those bytes are spread across
GPUs and the step is also paying for two all-to-alls per MoE layer, which is Module 9's subject and
the reason the math lab's 1,446 tok/s is fiction.`,
    stretch: `Pull three real \`config.json\` files from HuggingFace — pick a dense model, an MoE
and a hybrid — and feed their actual fields into \`Arch\` without adjusting anything. Two things
will happen. Some fields will not fit the dataclass, which tells you where that family's cost model
genuinely differs. And at least one model's total will come out wrong, which tells you which
parameters the config does not mention. Track down the discrepancy; it is usually a shared expert,
a tied embedding, or a first-\`k\`-layers-dense rule.

Then plot step time against batch size for a dense model and an MoE on the same axes, from batch 1
to 512, at 8k context. The dense line is flat until the cache term bites. The MoE line rises
steeply, then flattens once every expert is touched. Where they cross is the batch size at which
you should stop describing your MoE as "decodes like a 13B model".`,
  },

  papers: [
    {
      title: 'Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity',
      by: 'Fedus, Zoph & Shazeer, 2021',
      url: 'https://arxiv.org/abs/2101.03961',
      why: 'The paper that made sparse routing simple enough to be default: one expert per token, and a clear account of what that buys and what it costs to keep balanced.',
      frame: `Read **Section 2** for the routing mechanism and the argument for top-1, which had
been assumed impossible. **Section 2.2** on expert capacity and the capacity factor is the piece
that shows up in every serving system since. Skim the training-stability material in Section 3 and
the distillation results at the end unless you are training one yourself.`,
    },
    {
      title: 'DeepSeek-V3 Technical Report',
      by: 'DeepSeek-AI, 2024',
      url: 'https://arxiv.org/abs/2412.19437',
      why: 'The clearest example of a model whose architecture was chosen for its inference bill: latent attention for the cache term, fine-grained sparsity for the weight term, fp8 for both.',
      frame: `**Section 2.1** is the architecture: read MLA and DeepSeekMoE together and notice that
they attack different terms of the same equation. **Section 2.1.2** on auxiliary-loss-free load
balancing is the newest idea in the paper. Then work out the parameter breakdown yourself from the
config numbers and check that it lands on 671B and 37B — the math lab in this level does exactly
that. Skip the training infrastructure sections on a first pass, then come back to them after
Module 9.`,
    },
    {
      title: 'Jamba: A Hybrid Transformer-Mamba Language Model',
      by: 'Lieber et al. (AI21), 2024',
      url: 'https://arxiv.org/abs/2403.19887',
      why: 'A production hybrid with the cache numbers stated plainly: 4 GB at 256k context against 32 GB for Mixtral and 128 GB for Llama-2-70B.',
      frame: `**Table 1** is the argument — read it before anything else and make sure you can
derive why the numbers differ. **Section 3.1** gives the block structure: 1 attention layer per 7
Mamba layers, MoE every second layer. The ablations in **Section 6** on how few attention layers
you can get away with are the interesting part; they are measuring exactly the recall trade Module
7 describes.`,
    },
    {
      title: 'Gemma 3 Technical Report',
      by: 'Google DeepMind, 2025',
      url: 'https://arxiv.org/abs/2503.19786',
      why: 'The clearest published statement of local-global interleaving as a KV-cache decision rather than a modelling one, including what moving from 1:1 to 5:1 cost in quality.',
      frame: `Read the architecture section on the 5:1 local-global pattern and the 1024-token
window, and note that only the global layers get long-context RoPE scaling. The KV-cache ablations
are the payload — they show the memory curve flattening while perplexity barely moves. Skim the
distillation and evaluation sections.`,
    },
    {
      title: 'What Language Model Architecture and Pretraining Objective Work Best for Zero-Shot Generalization?',
      by: 'Wang et al., 2022',
      url: 'https://arxiv.org/abs/2204.05832',
      why: 'The controlled comparison behind the claim that decoder-only won, rather than the folklore version of it.',
      frame: `Read the setup in **Section 3** carefully — the whole value of the paper is that
architecture, objective and adaptation are varied independently, which almost nothing else in this
literature does. The result to extract is conditional: causal decoder-only plus autoregressive
pretraining wins zero-shot, while non-causal encoder-decoder plus masked pretraining wins once you
allow multitask finetuning. Skip the adaptation experiments unless the conditionality surprises
you, in which case they are the reason to believe it.`,
    },
  ],

  checkpoint: {
    claim: `You can read an unfamiliar model's config file and produce its two numbers — weight
bytes read per token, and cache bytes per token per sequence — without downloading anything. And
you can say which architectural family each field belongs to, what that family is buying, and what
it is paying with.`,
    questions: [
      {
        q: 'Mixtral-8x7B holds 46.7B parameters, not 56B, and activates 12.9B, not 14B. Why both?',
        a: `Because only the MLP block is replicated across experts. Attention, the embeddings, the
LM head and the norms are shared by every expert, so eight experts multiply one part of the layer
and leave the rest alone. Per layer: attention is 41,943,040 parameters and one SwiGLU expert is
\`3 × 4096 × 14336\` = 176,160,768, so eight experts plus shared attention is 1,451,229,184 rather
than eight times a whole 7B model. Times 32 layers plus 262,144,000 for the two embedding matrices
gives 46,701,477,888. The same logic caps the active count: a token reads attention once plus two
experts, \`41,943,040 + 2 × 176,160,768\` = 394,264,576 per layer, which over 32 layers plus the LM
head is 12.88B. The operational point is that the shared part is a floor — you cannot make an MoE
arbitrarily cheap per token by adding experts, because attention is read every time regardless.`,
      },
      {
        q: 'An MoE has an activation ratio of 5%. Does that mean the decode step reads 5% of the weights?',
        a: `Only at batch 1. The ratio is a per-token property, and a batch reads the *union* of its
tokens' experts. With top-\`k\` of \`E\` under roughly uniform routing, a batch of \`B\` tokens
touches \`E × (1 − (1 − k/E)^B)\` distinct experts: for 256 experts and top-8 that is 8 at batch 1,
163 at batch 32, and 252 by batch 128. So bytes per *step* climb toward the full weight footprint
as you batch, while bytes per *token* keep falling because they are amortized over more tokens.
Both are true at once. What actually degrades is the latency advantage — an MoE at batch 1 steps
like a small dense model, and the same MoE at batch 512 steps like a dense model of its total size
while doing a small fraction of the arithmetic. It changes from a bandwidth optimization into a
FLOP optimization somewhere in between, which is fine, because by then Module 4 says you are on the
compute-bound side of the ridge.`,
      },
      {
        q: 'A 48-layer model moves from all-global attention to 5:1 local-global with a 1024-token window. What happens to the cache at 32k context, and what happens at 256k?',
        a: `At GQA-8 and \`head_dim\` 128 in fp16, each layer stores 4 KiB per token. All-global at
32k is \`48 × 32768 × 4 KiB\` = 6.0 GiB. With 5:1, the 8 global layers still store everything —
\`8 × 32768 × 4 KiB\` = 1.0 GiB — while the 40 local layers saturated at 1,024 tokens and hold
\`40 × 1024 × 4 KiB\` = 0.16 GiB, for 1.15 GiB total, a 5.2× reduction. At 256k the local part is
unchanged, because it stopped growing at 1k, so the total is dominated by the global layers and the
ratio approaches \`L / (L/6)\` = 6×. That is the appeal: the saving is largest exactly where the
problem is worst, and it is achieved by making five sixths of the model stop caring about context
length. What it costs is exact recall on the local layers — information from beyond the window
still reaches a token through stacked layers, but blurred rather than retrieved.`,
      },
      {
        q: 'You are choosing between a 70B dense model and a 671B sparse one for a latency-sensitive single-user deployment. The sparse model reads 36.5 GB per token and the dense one 139 GB. Is the sparse model the right answer?',
        a: `On bytes moved per token, yes, and by a factor of 3.8 — which is the whole argument of
this level, since parameter count did not enter it. On everything else, probably not. The 671B
model needs 671 GB resident, so 9 H100s minimum and 16 in practice against 2 for the dense model,
and at batch 1 with expert parallelism most of those GPUs contribute no useful bandwidth: a GPU
holding 16 of 256 experts is asked for zero or one of the eight a token selects. You also pay two
all-to-alls per MoE layer, 58 of them, with latency that does not shrink as you add GPUs. The
sparse model's arithmetic is designed for a batch large enough to keep every expert busy, and a
single-user deployment is the operating point it was least designed for. The general rule this
gives you: **active bytes tell you the ceiling, resident bytes tell you what hardware you need to
approach it, and the gap between those two is where sparse models are won or lost.**`,
      },
      {
        q: 'Why is a KV cache formula that was correct for a model on 8 GPUs sometimes wrong on 16?',
        a: `Because tensor parallelism splits attention by heads, and a GQA model usually has 8 KV
heads. At TP=8 each rank holds one KV head and the formula \`2 × n_layers × n_kv_heads × head_dim
× dtype_bytes\` is exact. At TP=16 there are not enough KV heads to go around, so engines replicate:
every rank still holds one KV head, but each head now exists on two ranks. Aggregate cache memory
is 2× what the formula predicts, your maximum batch size falls accordingly, and nothing in the logs
mentions it — you scaled out for bandwidth and paid for it in the term that caps concurrency. It is
the clearest case of a "boring" config field (\`n_kv_heads\`) setting a hard limit on how a model
can be deployed, and it is decided at training time.

The limit is on head-sharding specifically, and Module 9's decode context parallelism is the way
out: shard the same cache by token position and the divisor keeps working past the head count.
Worth noting where that leaves MLA, which looks like the opposite case. Because MLA compresses K
and V into one latent vector there is no head axis at all, so tensor parallelism replicates the
whole cache on every rank at every degree — DeepSeek-V3's 68.6 KiB per token is 68.6 KiB on all
sixteen. The architecture that caches least per token is also the one that gets least help from
sharding it by head, which is why position-sharding arrived with it.`,
      },
    ],
  },

  pitfalls: [
    {
      wrong: 'Parameter count tells you how fast a model will decode.',
      right: `It tells you how much memory you need to hold it, which is a different question.
Decode speed at small batch is set by **active bytes**: DeepSeek-V3 has 671B parameters and reads
36.5 GB per token at fp8, while Llama-3-70B has one tenth the parameters and reads 139 GB at fp16.
The sparse model moves 3.8× fewer bytes per token than the model that is 9.5× smaller. Parameter
count and decode cost were joined at the hip for exactly as long as every model was dense, and
people have not updated. Read the activation ratio, then the dtype, then the layer count — and only
then the number in the model's name.`,
    },
    {
      wrong: 'A model with no KV cache has no per-sequence memory cost.',
      right: `The recurrent state is per sequence and per layer — it is constant in context length,
not zero. Mamba-2 at 2.7B carries \`5120 × 128\` values per layer across 64 layers, about 83.9 MB
per sequence at bf16, whether the sequence is 1,000 tokens or 1,000,000. At batch 64 that is 5.4 GB
of state you must have. What changes is *which* variable your memory scales with: concurrency
instead of concurrency times context. And every hybrid keeps real attention layers, so it keeps a
real KV cache for them — Jamba's 4 GB at 256k is small, not absent.`,
    },
    {
      wrong: 'Sliding-window attention means the model cannot see beyond the window.',
      right: `Attention composes across depth. A token attends to \`w\` positions, each of which
attended to \`w\` positions of its own, so the receptive field grows roughly as \`layers × w\` and
reaches far past the window. What a window bounds is the **cache**, and what it degrades is *exact*
recall — retrieving one specific distant token, which is what long-context benchmarks measure.
That is why local-global interleaving exists: a few full-attention layers restore the exact path
while the local majority keeps the cache flat. Judge a windowed model on retrieval evaluations, not
on the arithmetic of its receptive field.`,
    },
    {
      wrong: 'Architecture is something you can choose at deployment time.',
      right: `Almost none of it is. GQA can be uptrained from an MHA checkpoint at about 5% of
original pre-training compute; MLA cannot be retrofitted at all; sparse upcycling turns a dense
model into an MoE at roughly half the original pretraining cost. Those are training budgets, not
config flags. What you genuinely control on deployment day is quantization, paging, batching and
scheduling — Modules 5 and 6 — and they operate on whatever bytes-per-token the architecture handed
you. This is the practical reason to be able to read a config file: **by the time you can measure a
model, its two numbers have been fixed for months.**`,
    },
  ],

  glossary: [
    { term: 'active parameters', def: 'The parameters read to produce one token. Equal to the total in a dense model; a small fraction of it in a sparse one.' },
    { term: 'activation ratio', def: 'Active parameters divided by total parameters. DeepSeek-V3 is 5.5%, Qwen3-235B-A22B is 9%, Mixtral-8x7B is 28%.' },
    { term: 'router', def: 'The small linear layer in an MoE that scores every expert for a token, so the top k can be selected. Runs per token, per layer.' },
    { term: 'top-k routing', def: 'Sending each token to the k highest-scoring experts. Switch Transformer uses k=1, GShard and Mixtral k=2, DeepSeek-V3 k=8 of 256.' },
    { term: 'shared expert', def: 'An expert every token reads, alongside its routed ones. Absorbs the common knowledge that would otherwise be duplicated into every expert.' },
    { term: 'fine-grained experts', def: 'Many small experts instead of a few large ones, so routing is finer-grained at the same active parameter count. The DeepSeekMoE design.' },
    { term: 'sparse upcycling', def: 'Turning a dense checkpoint into an MoE by copying its MLP into every expert and continuing training. Costs roughly half the original pretraining.' },
    { term: 'cross-layer KV sharing', def: 'Letting neighbouring layers reuse one set of K and V, cutting n_layers in the cache formula rather than the head count.' },
    { term: 'sliding-window attention', def: 'Attending only to the last w positions, which makes the cache constant in context rather than linear. Bounds the cache, not the receptive field.' },
    { term: 'local-global interleaving', def: 'A stack of mostly windowed layers with occasional full-attention layers. Gemma 2 used 1:1 at 4096; Gemma 3 uses 5:1 at 1024.' },
    { term: 'recurrent state', def: 'The fixed-size per-sequence memory of an SSM or linear-attention layer. Constant in context length, so it scales with concurrency alone.' },
    { term: 'cross-attention', def: 'Decoder attention over an encoder output. Its K and V are computed once per input and never grow, unlike self-attention KV.' },
    { term: 'tied embeddings', def: 'Reusing the input embedding matrix as the LM head. Saves vocab x d_model parameters — rounding error at 70B, a quarter of a 0.6B model.' },
    { term: 'image tokens', def: 'The positions a vision encoder and projector contribute to the language model sequence. 576 for LLaVA-1.5, thousands under tiling schemes.' },
  ],
};
