export default {
  n: 14,
  slug: 'fine-tuning',
  title: 'FINE-TUNING',
  tagline: 'Pretraining teaches a model to complete anything. Fine-tuning teaches it to do what you actually asked — and the newest alignment methods do that by running the exact decode loop the rest of this book spent optimizing.',
  hours: '7–9 hours',
  prereqs: ['Module 0', 'Module 1', 'Module 6'],

  bigIdea: `Every module so far has taken the weights as given and asked how to run them fast. This one
asks where those weights come from after pretraining, and it earns a place in a book about
inference for two reasons.

First: the cheapest and most widely used fine-tuning methods work by leaving the pretrained
weights completely alone and training a small delta beside them. That delta is small enough to be
swapped in and out of a shared base model at serving time — the same shared-resource trick that
makes batching (Module 5) work for KV caches, applied to the model's parameters instead. A fleet
that would otherwise need a dedicated deployment per fine-tuned customer can hold one base model
resident and stream a few megabytes of adapter per request.

Second, and newer: the fine-tuning methods behind today's best reasoning models don't train on a
fixed dataset at all. They generate their own training data by sampling from the model itself —
dozens to hundreds of full decode passes per prompt, scored, and turned into a gradient. A
training loop built this way is bottlenecked by the exact arithmetic Module 1 derived for a
chatbot serving real users: decode is memory-bound, and however many tokens per second your
inference engine produces per GPU is now also your training throughput. Fine-tuning used to be a
training-time concern this book could leave alone. Increasingly, it is not.`,

  concepts: [
    {
      name: 'Pretraining gives you completion; fine-tuning gives you compliance',
      keyPoint: 'A base model completes text; an instruction-tuned model answers you — and the gap between them is a training stage, not an architecture change.',
      body: `Pretraining optimizes exactly one thing: predict the next token, over trillions of tokens
of ordinary text. Nothing in that objective knows about "questions" or "answers" or "being
helpful." Hand a raw base model the prompt \`How do I boil an egg?\` and it is just as likely to
continue with \`How long does it take? What about a soft-boiled one?\` — more plausible internet
text — as it is to answer you. The model is not being difficult. It is doing exactly what it was
trained to do: continue the distribution it saw.

**Supervised fine-tuning (SFT)** — what most people mean when they say "fine-tuning" without
qualification — closes that gap with no architecture change at all. Same transformer, same
next-token cross-entropy loss, same optimizer. The only thing that changes is the data: instead of
"the internet," the model trains on curated \`(instruction, response)\` pairs, wrapped in a chat
template that marks who is speaking. A few thousand to a few hundred thousand well-written
examples are usually enough to shift a model from "completes anything" to "answers you" — SFT is
teaching a *format and a role*, which turns out to need far less data than teaching new knowledge
does (a distinction the pitfalls below return to).

One detail that trips people up the first time they write an SFT loop by hand: you do not compute
loss on the whole sequence. The prompt tokens are masked out — the model is not being trained to
predict *your* question, only the assistant's response to it. Get this wrong and the model
partially learns to imitate users instead of answering them, and burns capacity re-deriving
prompts it will never need to generate at inference time.

Base model and instruction-tuned model are usually the exact same architecture at the exact same
size; Llama-3-8B and Llama-3-8B-Instruct differ only in what was done to the weights after
pretraining finished. Everything else in this module is either a cheaper way to do SFT, or a
second stage that runs after it.`,
      ascii: `  <user> How do I boil an egg?    <assistant> Bring water to a boil...   <eos>
    NO    NO NO NO NO NO NO  NO         NO          YES   YES   YES  ...    YES

  \\_________ prompt tokens: masked out, zero gradient _________/\\___ response: normal loss ___/`,
    },

    {
      name: "Full fine-tuning's memory bill",
      keyPoint: 'Adam under mixed precision needs about 16 bytes of state per parameter you update — four times the weights themselves — and that single number is the reason everything else in this module exists.',
      body: `Fine-tuning the entire model looks simple: unfreeze every parameter, run SFT (or any of the
objectives later in this module), backprop, step the optimizer. The reason it is not the default
is memory, and the amount is easy to underestimate if you only count the weights.

Training almost universally uses **mixed-precision Adam**: forward and backward passes run in
fp16/bf16 for speed, but a full-precision "master copy" of the weights is kept in fp32 so tiny
gradient updates do not get rounded away, and Adam's own running statistics are kept in fp32 too.
Per trainable parameter:

\`\`\`
2 bytes   fp16/bf16 weight    (used in the forward/backward matmuls)
2 bytes   fp16/bf16 gradient
4 bytes   fp32 master weight  (what the optimizer actually updates)
4 bytes   fp32 Adam moment 1  (m, the running mean of the gradient)
4 bytes   fp32 Adam moment 2  (v, the running mean of the squared gradient)
--------
16 bytes  per parameter, before a single activation is stored
\`\`\`

For Llama-3-8B that is \`8.03e9 x 16 = 128.5 GB\` — before you have loaded a single training
example, and on a single 80 GB H100 or A100 you are already out of memory. This is not an
inefficiency to be optimized away; it is the honest cost of Adam remembering enough about every
parameter's gradient history to take a stable step. \`ZeRO\`/FSDP-style sharding spreads this
16 bytes/parameter across many GPUs rather than reducing it — the same "spread the state, don't
shrink it" idea Module 9 used for inference-time tensor parallelism, applied to training instead.

Multiply by 70B and the picture gets uglier before it gets better — the math lab works both
scales by hand. Every method in the rest of this module is, one way or another, an answer to this
number: LoRA answers it by making almost all of a model's parameters permanently ineligible for
gradients or optimizer state; QLoRA answers it again by shrinking what is left; DPO and GRPO
answer a related but distinct memory bill, the cost of keeping *several models* resident at once
for alignment.`,
      ascii: '',
    },

    {
      name: 'LoRA: freeze the model, train a low-rank detour',
      keyPoint: "Instead of updating W directly, LoRA learns a rank-r product BA and adds it: W' = W0 + (alpha/r) B A. Freezing W0 removes gradient and optimizer memory for the vast majority of parameters, and the update folds back into one matrix at merge time, so it costs nothing at inference.",
      body: `Whatever fine-tuning teaches a weight matrix \`W0\` [d_out x d_in], the *update* \`delta W\` it
learns is itself a matrix of that same shape. LoRA's premise — argued empirically in the original
paper on GPT-3 and RoBERTa — is that \`delta W\` during task adaptation does not need the full
\`d_out x d_in\` degrees of freedom; it has low **intrinsic rank**. So instead of learning
\`delta W\` directly, parameterize it as a product of two much smaller matrices:

\`\`\`
delta W  =  (alpha / r) * B @ A

  A : [r, d_in]      B : [d_out, r]      r << min(d_out, d_in)
  W0 stays frozen — no gradient, no optimizer state
  only A and B are trained
\`\`\`

The forward pass becomes \`h = W0 x + (alpha/r) B(Ax)\` — one extra small matmul through the
r-dimensional bottleneck, added to the frozen path. \`A\` is initialized to small random values and
\`B\` to exactly zero, so at training step zero the LoRA path contributes nothing at all and the
model starts training from the pretrained model's *exact* behavior — a nice property full
fine-tuning does not give you for free. \`alpha\` is a fixed scaling constant that keeps the
update's effective magnitude roughly stable as you change \`r\`, so you are not re-tuning the
learning rate every time you try a different rank; \`alpha = 2r\` is the common starting point, and
pushing it much higher tends to hurt rather than help.

Which weight matrices get a LoRA adapter is a real design knob. The original paper found adapting
only \`W_q\` and \`W_v\` in attention sufficient at GPT-3 scale; most current recipes adapt all four
attention projections (\`q, k, v, o\`) and often the MLP's \`gate/up/down\` projections too, trading
more trainable parameters for a smaller quality gap to full fine-tuning. On a GQA model like
Llama-3-8B (Module 11), \`k_proj\` and \`v_proj\` are already narrower than \`q_proj\`/\`o_proj\` — the
math lab uses exactly this to compute a real trainable-parameter count.`,
      ascii: `  frozen W0 [d_out x d_in]          trainable, rank r << d_in, d_out
        |                              A [r x d_in]        B [d_out x r]
        |                                   |                    |
   x ---+--------------------------> x ---> A ---> (r-dim) -----> B ---> * (alpha/r)
        |                                                                    |
        +-------------------------------- + <----------------------------- -+
                                            |
                                            v
                              h = W0 x + (alpha/r) B(Ax)

  merge:  W' = W0 + (alpha/r) B A     <- same shape as W0, one matrix, zero extra latency`,
    },

    {
      name: 'Quantize the frozen part too: QLoRA',
      keyPoint: 'QLoRA keeps the LoRA adapter in bf16 but stores the large frozen base model in 4-bit NF4, cutting the base-weight footprint another 4x on top of what freezing already saved — enough to fine-tune a 65B model on one 48 GB GPU.',
      body: `LoRA stops the optimizer from paying for the base model's weights, but the base weights
still have to sit in GPU memory in some precision so the forward pass can read them. QLoRA's move
is to apply Module 6's quantization idea at fine-tuning time rather than only at serving time:
store the frozen base model in 4-bit **NF4** (4-bit NormalFloat) instead of fp16 or bf16.

NF4 is not a generic 4-bit integer format. Its quantization bins are spaced to be
information-theoretically optimal for weights that are roughly zero-centered and Gaussian —
which pretrained transformer weights empirically are, once you normalize per small block — rather
than the uniform bins a plain INT4 scheme would use. Crucially, compute does not happen in 4-bit:
every matmul dequantizes the relevant weight block to bf16 on the fly, multiplies, and discards
the bf16 copy. Storage shrinks four-fold; the arithmetic itself still runs at bf16 precision
throughout training. Two more tricks round out the recipe: **double quantization**, which
quantizes the per-block scale factors themselves, saving a further fraction of a bit per
parameter; and **paged optimizers**, which use unified GPU/CPU memory so an activation-memory
spike (gradient checkpointing, in particular, causes these) pages optimizer state out to host RAM
instead of crashing the run.

The headline result — fine-tuning a 65B-parameter model on a single 48 GB GPU while matching
16-bit full fine-tuning quality — is the demonstration that these three tricks compose cleanly
with LoRA's already-small optimizer footprint. The honest cost is speed, not just an efficiency
footnote: independent benchmarks report QLoRA runs on the order of 30-40% slower in wall-clock
time than plain bf16 LoRA at comparable memory-unconstrained settings, because every forward and
backward pass now pays for a dequantize step the bf16 version does not. QLoRA buys memory, not
speed — you reach for it specifically when the model would not fit any other way.`,
      ascii: '',
    },

    {
      name: 'The rest of the PEFT family, and why LoRA merges for free but the others do not',
      keyPoint: "Adapters, prefix-tuning, and prompt-tuning are older or parallel ideas for training few parameters, but only an update that is *additive to an existing weight matrix* can be folded back into it — which is exactly the property LoRA and DoRA have and the others don't.",
      body: `LoRA is the dominant parameter-efficient fine-tuning (PEFT) method today, but it was not the
first, and it is worth knowing what it beat and why.

**Adapters (Houlsby et al., 2019)** insert a small bottleneck feed-forward block — down-project to
a few dozen dimensions, a nonlinearity, up-project back — *inline*, after the attention and FFN
sublayers, with everything else frozen. On GLUE, adapters land within 0.4% of full fine-tuning
while training only about 3.6% of the parameters per task, robust across a wide range of bottleneck
sizes. The catch is architectural: an adapter sits in the middle of the residual stream with a
nonlinearity inside it, so there is no algebraic move that folds it back into the surrounding
frozen weights. Every inference request pays its extra forward-pass compute, forever.

**Prefix-tuning and prompt-tuning (2021)** don't touch any weight at all. They prepend a short
sequence of trainable "virtual token" embeddings — a soft prompt — to the input, at every layer
(prefix-tuning) or just the embedding layer (prompt-tuning), and train only those vectors. This
ties directly back to Module 2: those virtual tokens occupy real sequence positions, so they
consume context-window budget and KV-cache memory on every single request, exactly as if the
prompt were physically longer. Cheap to train, but not free at serving time either — just charged
in a different currency than adapters.

**BitFit** trains only the bias terms — tiny parameter count, correspondingly weaker quality,
rarely the first choice today. **(IA)^3** learns per-channel rescaling vectors that multiply key,
value, and hidden activations rather than adding a low-rank detour; fewer parameters than LoRA in
many configurations, and in several cases those vectors can also be absorbed into an adjacent
weight matrix.

**DoRA — Weight-Decomposed LoRA (Liu et al., 2024)** is the modern refinement most current
recipes have converged toward. It decomposes each frozen weight into a magnitude vector and a
direction matrix (\`W0 = m * V / ||V||\`), trains a standard LoRA update on the direction and the
magnitude vector directly, and — crucially — still merges back into a single weight matrix for
zero added latency, the same property LoRA has. Separating "how far" from "which way" lets DoRA's
updates resemble full fine-tuning's more closely than plain LoRA's additive delta can, at the same
rank; it beats vanilla LoRA by a small, consistent margin on standard benchmarks and is close to a
drop-in swap in current PEFT libraries.

The dividing line, restated: anything that is *additive to an existing weight matrix* — LoRA,
DoRA — produces a delta of the same shape as the weight it modifies, so \`W0 + delta W\` is again
just a matrix, substitutable with zero change to the computation graph. Anything that adds new
layers (adapters) or new sequence positions (soft prompts) changes the graph itself, and no amount
of algebra removes that cost at inference time.`,
      ascii: '',
    },

    {
      name: 'RLHF vs DPO: two ways to learn from preferences',
      keyPoint: 'RLHF trains a reward model on human preference pairs, then runs PPO against it with a KL penalty back to the SFT policy — four models in memory. DPO folds the same objective into a single classification-style loss over the policy\'s own log-probabilities — two models in memory, no reward model, no rollouts.',
      body: `SFT teaches a model to imitate the responses it is shown. It does not teach the model to
prefer a subtly better response over a subtly worse one written in a similar style — that is a
*comparative* signal, and getting a human to say "A is better than B" is far cheaper and more
reliable than getting them to write a gold response from scratch. That is what preference data is,
and aligning a model to it is the second post-training stage, after SFT.

**RLHF**, the recipe InstructGPT made standard, runs in three stages. Collect pairs of model
outputs and human preferences between them. Train a **reward model** \`r_phi(x, y)\` on those pairs
with a Bradley-Terry pairwise loss — maximize the log-sigmoid gap between the chosen and rejected
response's scores, the same functional form behind Elo ratings. Then run **PPO**: treat generation
as an RL episode, reward each rollout with \`r_phi\` minus a KL penalty \`beta * KL(pi_theta || pi_ref)\`
that stops the policy drifting so far from the SFT model that it degenerates into
reward-hacking the reward model rather than actually improving. The engineering cost is real: PPO
needs **four models resident simultaneously** — the policy being trained, a frozen reference
policy for the KL term, the frozen reward model scoring rollouts, and a value/critic network
estimating a baseline for the advantage — and the RL loop itself is notoriously fiddly to keep
stable.

**DPO (Rafailov et al., 2023)** starts from an observation about that same KL-regularized
objective: for a fixed reward function, the optimal policy has a closed form in terms of the
reward and the reference policy. Substitute that relationship into the Bradley-Terry preference
loss, and the reward model's normalizing term cancels out algebraically. What is left is a loss
computable directly from the *policy's own* log-probabilities of the chosen and rejected response,
compared against the frozen reference's log-probabilities of the same two responses:

\`\`\`
loss = -log sigmoid( beta * [ (logpi_theta(y_w|x) - logpi_ref(y_w|x))
                             - (logpi_theta(y_l|x) - logpi_ref(y_l|x)) ] )

  y_w = preferred ("winner") response,  y_l = rejected ("loser") response
\`\`\`

Two forward passes per preference pair, no sampling, no reward model, no critic, no RL loop — just
a supervised classification loss over data you already collected. Only two models are resident:
the policy and the frozen reference. That simplicity is DPO's entire appeal, and it is why most
open-source alignment recipes today skip PPO.

The tradeoff is that DPO optimizes entirely against a fixed, already-collected preference dataset
— there is no rollout step where it can score fresh, on-policy samples the way an RL loop
conceptually can. Several studies since have found DPO trailing a well-tuned reward-model-plus-PPO
pipeline in ceiling quality once enough compute and data go into the RL side, which is one reason
interest in RL-based alignment did not disappear — it fed directly into the reasoning-focused
methods in the next concept. A family of DPO variants has grown up around trimming its remaining
rough edges: **KTO** learns from unpaired desirable/undesirable labels rather than requiring
matched pairs; **ORPO** folds preference optimization into the SFT stage itself, needing no
separate reference model at all; **SimPO** drops the reference model and uses a length-normalized
log-probability directly as the implicit reward.`,
      ascii: `  SFT model
     |
     +--> RLHF ---------------------------------------+
     |     reward model (pairwise)  -->  PPO rollout   |  4 models resident:
     |                                                  |  policy + ref + reward model + critic
     |
     +--> DPO ----------------------------------------+
           preference pairs  -->  classification loss  |  2 models resident:
           (no rollouts, no reward model, no critic)    |  policy + reference`,
    },

    {
      name: "GRPO and RLVR: when the reward is just 'is the answer right'",
      keyPoint: 'For tasks with a checkable answer — math, code, logic — skip human preference labels entirely: sample a group of completions per prompt, reward each by verifiable correctness, and set its advantage relative to its own group\'s mean and std. Generating that group is pure decode, which is why this training loop\'s speed is now an inference-engine problem.',
      body: `RLHF and DPO both depend on a learned proxy for quality — a reward model, or the implicit
reward baked into DPO's loss — because "is this response good" is usually subjective. For a
narrower but very important class of tasks it is not subjective at all: does the final boxed
answer match the known solution, does the generated code pass its test suite, does a symbolic
verifier accept a proof step. **RLVR** — reinforcement learning with verifiable rewards — uses a
deterministic checker instead of a learned reward model wherever one is available: no
reward-model training, no reward-hacking-the-reward-model failure mode, and a signal that scales
with however many problems you have known answers to.

**GRPO** (Group Relative Policy Optimization, from DeepSeekMath, and the algorithm behind
DeepSeek-R1) is the policy-gradient method built for this setting. It keeps PPO's clipped
policy-gradient update but removes the critic network entirely. Instead of training a value
function to estimate a baseline, GRPO samples a **group** of \`G\` completions for the same prompt
from the current policy, scores each one (with the verifiable reward above, or a reward model —
the group-relative trick is separable from RLVR specifically), and sets each completion's
advantage as its own reward normalized against the group's mean and standard deviation:

\`\`\`
advantage_i  =  (reward_i - mean(reward_1 .. reward_G)) / std(reward_1 .. reward_G)
\`\`\`

That is a Monte-Carlo baseline computed for free from rollouts you were generating anyway, so an
entire model's worth of memory and compute — the critic — disappears relative to PPO. A frozen
reference policy is still kept for a KL penalty, so GRPO needs two models resident, like DPO, not
four like full PPO. What it needs *instead* is samples: \`G\` full completions per prompt, every
single training step, before a gradient exists to take.

That is where this module closes the loop with the rest of the book. Generating \`G\` completions
for every prompt in a batch is \`G x batch_size\` full decode sequences — at \`G = 64\` and a batch
of 256 prompts, that is 16,384 sequences to decode before one gradient step happens. That workload
is exactly the large-batch, memory-bound decode this entire book has been about optimizing, and it
now sits on the *training* throughput critical path rather than a user's. It is why modern
RL post-training frameworks embed a real serving engine — frequently vLLM itself — as their
rollout generator instead of a naive generation loop: the training loop's steps-per-hour ceiling
is, quite literally, the inference engine's decode-throughput ceiling from Module 1, Module 5, and
Module 7 combined. One more data point worth knowing: DeepSeek's R1-Zero variant skipped the SFT
warm-start stage entirely, running GRPO directly against a base model, and still reached strong
reasoning behavior — evidence that for verifiable domains, the RL signal alone can carry more of
the weight than the earlier RLHF-era pipeline assumed.`,
      ascii: `  one prompt --> sample G completions from the CURRENT policy   (G full decode passes)

                   r1    r2    r3    r4   ...   rG      <- verifiable reward per completion
                    |
                    v
       advantage_i = (r_i - mean(r_1..rG)) / std(r_1..rG)
                    |
                    v
       clipped policy-gradient update — no critic network needed

  G=64, batch=256  -->  16,384 decode sequences generated before ONE gradient step`,
    },
  ],

  mathLab: {
    prompt: `Take **Llama-3-8B** (8.03 B parameters: hidden 4096, 32 layers, 32 query heads, 8 KV
heads, head_dim 128 — GQA, as in Module 11) and its larger sibling **Llama-3-70B** (≈70.6 B
parameters: hidden 8192, 80 layers, 64 query heads, 8 KV heads, same head_dim). You want to adapt
each one using Adam with mixed precision, on **A100-80GB** GPUs, with no ZeRO/FSDP sharding —
assume the model states have to fit on one card.

Work out:

1. Full fine-tuning: using the 16-bytes-per-parameter figure this module derived, compute total
   memory for both models. Do either fit on one 80 GB card?
2. LoRA on \`q_proj, k_proj, v_proj, o_proj\` in every layer, rank \`r = 16\`. Using each model's
   actual projection shapes (note the GQA-narrowed \`k_proj\`/\`v_proj\`), how many trainable
   parameters does this add to Llama-3-8B? What fraction of the 8.03 B total is that, and how
   does the reduction ratio compare to the ~10,000x the original LoRA paper reports for GPT-3?
   Why the difference?
3. Total memory to LoRA-fine-tune Llama-3-8B: frozen fp16 base weights, plus Adam state (still
   16 bytes/parameter — Adam does not care whether a parameter belongs to a LoRA matrix or the
   original model) for only the trainable parameters from (2). Does it fit on a single 24 GB
   consumer GPU?
4. QLoRA: same adapter, but the frozen base is stored in 4-bit NF4 (0.5 bytes/parameter) instead
   of fp16. Recompute the total for Llama-3-8B, then repeat steps (2)-(4) for Llama-3-70B. At
   what point does each of full fine-tuning, plain LoRA, and QLoRA stop fitting on a single 48 GB
   GPU?
5. None of the above counts activation memory. Does LoRA reduce it? Why or why not?`,

    solution: `**1. Full fine-tuning**

\`\`\`
bytes/param = 2 (fp16 weight) + 2 (fp16 grad) + 4 (fp32 master) + 4 (fp32 m) + 4 (fp32 v) = 16

8B:   8.03e9  x 16 = 128.5 GB
70B:  70.6e9  x 16 = 1129.6 GB
\`\`\`

Neither fits on one 80 GB GPU — the 8B model needs about 1.6 cards' worth of state alone, before a
single activation is stored, and the 70B model needs on the order of 15 GPUs' worth (via
ZeRO-3/FSDP sharding across a pod, not a single-card option at all).

**2. LoRA trainable parameters, Llama-3-8B**

Projection shapes: \`q_proj, o_proj\` are \`[4096, 4096]\` (32 heads x 128); \`k_proj, v_proj\` are
\`[4096, 1024]\` (8 KV heads x 128, GQA-narrowed). LoRA parameters per matrix = \`r * (d_in + d_out)\`:

\`\`\`
q_proj, o_proj:  16 x (4096 + 4096) = 131,072 each
k_proj, v_proj:  16 x (4096 + 1024) =  81,920 each

per layer:  131,072 x 2 + 81,920 x 2 = 425,984
x 32 layers:                          13,631,488  ~=  13.6 M trainable parameters
\`\`\`

Fraction of the 8.03 B total: \`13.6e6 / 8.03e9 = 0.17%\` — a **589x** reduction in trainable
parameters (\`8.03e9 / 13.6e6\`), not the ~10,000x the LoRA paper quotes for GPT-3. Two reasons for
the gap, and both are worth re-deriving rather than trusting: GPT-3's LoRA config in the paper
targets only \`W_q\` and \`W_v\` (not all four projections, as here), and GQA has already shrunk
\`k_proj\`/\`v_proj\` relative to an MHA model, so this model's total parameter count buys
proportionally less attention-matrix real estate to begin with. The headline ratio is a function
of exactly which matrices you target and exactly which architecture you target them on — never a
universal constant.

**3. LoRA total memory, Llama-3-8B**

\`\`\`
frozen fp16 base weights:      8.03e9 x 2 bytes           = 16.06 GB   (no grad, no optimizer state)
trainable LoRA state:          13.6e6 x 16 bytes/param     =  0.22 GB
                                                             --------
                                                              16.28 GB
\`\`\`

Comfortably fits a single 24 GB consumer GPU, with room left for activations at a reasonable batch
size and sequence length. The frozen weights, not the optimizer, are now the dominant cost.

**4. QLoRA, both models**

\`\`\`
Llama-3-8B:
  4-bit NF4 base:   8.03e9 x 0.5 bytes  = 4.02 GB
  LoRA state:       13.6e6 x 16 bytes   = 0.22 GB
                                          --------
                                           4.24 GB

Llama-3-70B LoRA trainable params (same method as step 2, scaled up):
  q_proj, o_proj [8192,8192]:  16 x 16384 = 262,144 each
  k_proj, v_proj [8192,1024]:  16 x  9216 = 147,456 each
  per layer: 262,144 x 2 + 147,456 x 2 = 819,200
  x 80 layers: 65,536,000  ~=  65.5 M trainable parameters

Llama-3-70B:
  full fine-tuning:  70.6e9 x 16 bytes           = 1129.6 GB
  plain LoRA:        70.6e9 x 2 (frozen fp16)
                      + 65.5e6 x 16 (LoRA state)  = 141.2 + 1.05  = 142.3 GB
  QLoRA:              70.6e9 x 0.5 (frozen NF4)
                      + 65.5e6 x 16 (LoRA state)  =  35.3 + 1.05  =  36.3 GB
\`\`\`

Laid against a single 48 GB card: full fine-tuning never fits either model without multi-GPU
sharding. Plain LoRA fits the 8B model (16.3 GB) but not the 70B model (142.3 GB — needs at least
two 80 GB cards just for the frozen weights). QLoRA fits **both**: 4.24 GB for the 8B model, 36.3
GB for the 70B model, comfortably under 48 GB with room for activations — which is exactly the
"65B on one 48 GB GPU" result the QLoRA paper demonstrated, re-derived here at a slightly larger
scale.

**5. Activation memory**

No — LoRA does not touch it, and this is the detail people most often miss. Activation memory
scales with batch size x sequence length x hidden dim x layers, because the forward pass has to
run through *every* layer to reach the loss, frozen or not — LoRA does not skip a single layer's
forward computation, it only decides which parameters accumulate a gradient afterward. Full
fine-tuning and LoRA store the *same* activations for the *same* batch and sequence length. The
memory LoRA saves is entirely in gradients and optimizer state for the frozen majority of
parameters; shrinking activation memory is a separate lever (gradient checkpointing — recompute
activations during the backward pass instead of storing them, trading compute for memory) that
composes with LoRA rather than being subsumed by it.`,
  },

  codeLab: {
    goal: `Apply LoRA to GPT-2 with the \`peft\` library, confirm the trainable-parameter drop this
module's math lab predicts, fine-tune on a tiny synthetic task, and then merge the adapter into
the base weights and check that the merged model's logits are numerically identical to running
base-plus-adapter unmerged — the property that makes LoRA free at inference time once merged.

Runs on CPU in a couple of minutes; GPT-2 is small enough that the download and the loop are both
fast. The point is the parameter arithmetic and the merge property, not the resulting model's
quality.`,
    code: `"""
LoRA fine-tuning on GPT-2, plus the merge-invariance check.

    pip install torch transformers peft

Runs on CPU. GPT-2 is 124M parameters, so this downloads and trains in well under a minute.
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model

MODEL = "gpt2"

tok = AutoTokenizer.from_pretrained(MODEL)
tok.pad_token = tok.eos_token
base = AutoModelForCausalLM.from_pretrained(MODEL)

total_params = sum(p.numel() for p in base.parameters())
print(f"base model: {total_params:,} parameters")

# GPT-2's attention block is one fused q/k/v projection stored as a Conv1D
# (not a Linear) -- peft handles both, this is the standard target for gpt2.
config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["c_attn"],
    lora_dropout=0.0,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(base, config)
model.print_trainable_parameters()

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"\\ntrainable: {trainable:,} / {total_params:,}  ({trainable / total_params:.4%})")
print(f"reduction: {total_params / trainable:,.0f}x fewer trainable parameters")

# --- a tiny synthetic task: teach it one fixed Q/A pattern by rote ---
example = "Q: What is the capital of memory-bound inference?\\nA: The KV cache.\\n"
optim = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-3)

model.train()
for step in range(24):
    batch = tok(example, return_tensors="pt")
    out = model(**batch, labels=batch["input_ids"])
    out.loss.backward()
    optim.step()
    optim.zero_grad()
    if step % 6 == 0:
        print(f"step {step:2d}  loss {out.loss.item():.3f}")

# --- the merge-invariance check ---
model.eval()
prompt = tok("Q: What is the capital of memory-bound inference?\\nA:", return_tensors="pt")
with torch.no_grad():
    unmerged_logits = model(**prompt).logits

merged = model.merge_and_unload()   # folds B @ A into the base weight, in place
with torch.no_grad():
    merged_logits = merged(**prompt).logits

max_diff = (unmerged_logits - merged_logits).abs().max().item()
print(f"\\nmax logit difference, merged vs unmerged: {max_diff:.2e}")
assert torch.allclose(unmerged_logits, merged_logits, atol=1e-4), "merge changed the forward pass!"
print("merged model is numerically identical -- this is why LoRA costs zero extra latency once merged.")

# --- TODO for you ---
#   1. Add "c_proj" (the attention output projection) to target_modules. Work out the new
#      trainable-parameter count by hand first, then check the script's printed number against it.
#   2. Try r=1 and r=64. Does the loss on this toy task move much? On a real task it would --
#      this example is too easy to show the capacity gap the pitfalls section describes.
`,
    expect: `\`print_trainable_parameters\` should report something in the neighbourhood of
\`294,912\` trainable parameters out of GPT-2's \`124,439,808\` total — about **0.24%**, which is
the same \`r * (d_in + d_out)\`-per-matrix arithmetic the math lab used for Llama-3-8B, just at
GPT-2's dimensions (\`d_model = 768\`, 12 layers). The exact figure can drift slightly with your
\`peft\` version's accounting; what matters is landing in that ballpark, not matching a specific
digit.

The loss should drop sharply within the first handful of steps — the "task" is a single memorized
sentence repeated 24 times, so this is a plumbing check, not a real training run. The max logit
difference after merging should land around \`1e-6\` to \`1e-5\`: floating-point associativity noise
from re-ordering the addition, not a bug. If you see a difference anywhere near the scale of the
logits themselves, the merge is broken.`,
    stretch: `Call \`merged.save_pretrained("merged-gpt2")\` and reload it with a plain
\`AutoModelForCausalLM.from_pretrained("merged-gpt2")\` — confirm it is an ordinary GPT-2 checkpoint
with no \`peft\` import needed to load it. That is what makes a merged LoRA adapter deployable to
any inference stack that already knows how to serve the base model, no adapter-aware code path
required. Then read about S-LoRA or vLLM's multi-LoRA support and write one paragraph explaining
why a production system serving hundreds of customer-specific adapters on one base model would
deliberately choose *not* to merge, even though merging is free.`,
  },

  papers: [
    {
      title: 'LoRA: Low-Rank Adaptation of Large Language Models',
      by: 'Hu et al., 2021',
      url: 'https://arxiv.org/abs/2106.09685',
      why: 'The paper this module\'s central algebraic trick comes from, and the reason parameter-efficient fine-tuning became a real alternative to full fine-tuning rather than a compromise.',
      frame: `Read **Section 4** (the low-rank hypothesis and the \`delta W = BA\` parameterization) and
the GPT-3 175B results in **Section 5** closely — that is the 10,000x/3x headline the math lab
re-derives at a different scale and gets a different number from. The rank-search ablation in
**Section 7** is the empirical argument for why a small \`r\` turns out to be enough; it's worth
sitting with rather than taking on faith.`,
    },
    {
      title: 'QLoRA: Efficient Finetuning of Quantized LLMs',
      by: 'Dettmers, Pagnoni, Holtzman, Zettlemoyer, 2023',
      url: 'https://arxiv.org/abs/2305.14314',
      why: 'Combines this module\'s PEFT material with Module 6\'s quantization material into the single most-used recipe for fine-tuning a large model on one GPU.',
      frame: `**Section 3** covers all three innovations — NF4, double quantization, paged
optimizers — read it next to Module 6's quantization concepts, since NF4's whole pitch is
"quantization bins matched to the actual weight distribution" rather than uniform INT4 bins. The
Guanaco results in **Section 5** are the "65B on one 48 GB GPU" headline this module's math lab
checks by hand.`,
    },
    {
      title: 'Training language models to follow instructions with human feedback',
      by: 'Ouyang et al. (OpenAI), 2022',
      url: 'https://arxiv.org/abs/2203.02155',
      why: 'The InstructGPT paper that made the SFT -> reward model -> PPO pipeline the default recipe, and the one every later alignment paper — DPO included — defines itself against.',
      frame: `**Figure 2**'s three-stage pipeline diagram is the one to internalize before reading
anything else about RLHF. **Section 3.5**'s discussion of the KL penalty and the "alignment tax"
measured on public NLP benchmarks is the honest cost side of the story that summaries of this
paper tend to skip.`,
    },
    {
      title: 'Direct Preference Optimization: Your Language Model is Secretly a Reward Model',
      by: 'Rafailov, Sharma, Mitchell, Ermon, Manning, Finn, 2023',
      url: 'https://arxiv.org/abs/2305.18290',
      why: 'Derives the reparameterization this module\'s DPO loss comes from, and is the reason most open-source alignment recipes today skip PPO entirely.',
      frame: `**Section 4**'s derivation — from the KL-constrained reward-maximization objective to
the final classification loss — is dense on a first read but is a five-line algebra trick once you
see that the reward model's partition function cancels out of the preference probability. Worth
doing by hand once rather than taking the final loss as given.`,
    },
  ],

  checkpoint: {
    claim: `You can explain why the cheap fine-tuning methods are cheap — what memory they do and
do not save, and why — why LoRA specifically costs nothing at inference while adapters and
prefixes don't, and why the newest alignment methods make your training loop's speed a function of
your inference engine's decode throughput.`,
    questions: [
      {
        q: 'Your team tries to full-fine-tune Llama-3-8B with Adam on a single 80 GB GPU and gets an out-of-memory error before training even starts. Roughly how much memory were you asking for, and which piece of it does switching to LoRA remove?',
        a: `About 128.5 GB (\`8.03e9 x 16 bytes/parameter\` under mixed-precision Adam), against an
80 GB card — a 1.6x overshoot before a single activation is stored. LoRA removes the gradient and
optimizer-state portion (12 of those 16 bytes) for every parameter except the roughly 13.6 M in
the LoRA matrices, dropping the requirement to about 16.3 GB. What LoRA does *not* remove is the
frozen base weights themselves — 16.06 GB of that total — which is the next lever QLoRA pulls if
even that doesn't fit.`,
      },
      {
        q: 'Why does a merged LoRA adapter add zero latency at inference, but a Houlsby adapter always adds some?',
        a: `A LoRA update is additive to an existing weight: \`W' = W0 + (alpha/r) BA\` is a matrix of
the exact same shape as \`W0\`, so it substitutes directly into the same matmul the base model
already runs — one weight, no new step. A Houlsby adapter is an extra bottleneck block with a
nonlinearity sitting *inside* the residual stream, after attention or the FFN. There is no
algebraic move that folds a nonlinearity into a preceding frozen linear layer, so its down-project,
activation, and up-project run as genuinely extra compute on every single forward pass, forever —
merging simply isn't an operation that applies to it.`,
      },
      {
        q: 'A team wants a chat model that reliably outputs valid JSON in a fixed schema. Another wants a model that has absorbed their company\'s internal engineering wiki as usable knowledge. Which fine-tuning approach fits each, and why?',
        a: `The JSON-formatting task is a narrow behavioral shift — low intrinsic rank, exactly what
LoRA (even a small rank) was built for, and cheap enough to iterate on repeatedly. Absorbing a
large internal knowledge base is injecting a large volume of new facts rather than steering
existing behavior — the regime where "LoRA Learns Less and Forgets Less" found a real quality gap
against full fine-tuning, so continued pretraining or full fine-tuning (or LoRA at much higher
rank across more target modules, accepting some of that gap) fits better. Worth flagging past both
options: for pure factual lookup rather than a change in the model's behavior, retrieval (RAG) is
frequently a better answer than fine-tuning at all — it doesn't fight catastrophic forgetting and
the knowledge stays trivially updatable.`,
      },
      {
        q: 'Why does DPO need only two models in memory during training, while "vanilla" RLHF with PPO needs four? Name all four.',
        a: `DPO's loss only ever needs the policy being trained and a frozen reference policy — the
reward model's contribution cancels out algebraically in the derivation, so it never needs to
exist as a separate network at all. PPO-based RLHF keeps four models resident simultaneously: the
**policy** being updated, the frozen **reference/SFT policy** (for the KL penalty), the frozen
**reward model** (scores each rollout), and a **value/critic network** (estimates the baseline
PPO's advantage calculation needs).`,
      },
      {
        q: 'GRPO drops PPO\'s critic network by using a "group-relative" baseline instead. What does that phrase actually mean, and what does it cost you in return?',
        a: `For each prompt, sample a group of \`G\` completions from the current policy, score each
one, and use the group's own mean and standard deviation as the baseline instead of a learned
value function: \`advantage_i = (reward_i - mean(group)) / std(group)\`. It's a Monte-Carlo baseline
computed for free from rollouts you needed anyway, which is why the critic disappears. The cost is
that you must generate all \`G\` completions per prompt — a full decode pass each — before a single
gradient step exists to take. At \`G = 64\` and a batch of 256 prompts, that's 16,384 sequences to
decode per step, which is exactly the large-batch, memory-bound decode workload the rest of this
book optimizes, now sitting on the training loop's critical path instead of a user's.`,
      },
    ],
  },

  pitfalls: [
    {
      wrong: 'LoRA is just a cheaper way to do the same thing as full fine-tuning.',
      right: `Not quite — restricting the update to rank \`r\` is a real capacity constraint, not
only a compute optimization. "LoRA Learns Less and Forgets Less" (Biderman et al., 2024) found
that full fine-tuning learns weight updates with an effective rank 10-100x higher than typical
LoRA configurations, and that the resulting gap shows up specifically on knowledge-heavy domains
like code and math — LoRA underperformed full fine-tuning there, especially at low rank and in the
continued-pretraining regime. The flip side is real too: LoRA forgets less of the base model's
other capabilities and produces more diverse generations, precisely because it physically cannot
move the frozen weights far from where they started. Reach for LoRA when you're steering style,
format, or a narrow behavior; reach for full fine-tuning (or QLoRA at higher rank, across more
target modules) when you're injecting a large amount of new knowledge.`,
    },
    {
      wrong: 'DPO is strictly better than RLHF — it gets the same result with less machinery.',
      right: `It gets a much *simpler* result with less machinery, which is not the same claim.
DPO trades away the reward model and the RL loop, but also trades away the ability to score fresh,
on-policy samples during training — it optimizes entirely against a fixed, already-collected
preference dataset. Several studies since DPO's release have found it trailing a well-tuned
reward-model-plus-PPO pipeline in ceiling quality, particularly as more compute and data go into
the RL side. DPO's real advantage is stability and engineering simplicity — two models in memory
instead of four, a supervised-style loss instead of an RL loop that can silently reward-hack — not
a strictly better optimum.`,
    },
    {
      wrong: 'You should always merge a LoRA adapter into the base model before deploying it.',
      right: `Merging is free and correct — if you're serving *one* fine-tuned model. It stops
being the right call the moment you're serving many. A production system with hundreds of
customer-specific adapters on the same base model keeps them unmerged and swaps the small delta in
per request — Module 5's batching logic, extended to adapters instead of just KV caches — the way
S-LoRA and vLLM's multi-LoRA support do. Merge once per customer and you've cloned the entire base
model's memory footprint per customer; keep every adapter unmerged and each one costs only its own
few megabytes, batched together against one resident copy of the base weights.`,
    },
    {
      wrong: 'QLoRA fine-tunes in 4-bit, so the model you get out the other end is a 4-bit model.',
      right: `The 4-bit NF4 representation is a training-time memory trick for the *frozen* base
weights only — every matmul dequantizes to bf16 on the fly before multiplying, so training compute
runs at bf16 precision throughout, and the LoRA adapter itself is trained in bf16. What comes out
is a bf16 adapter you can merge into a bf16 (or fp32) copy of the base weights, completely
independent of whatever quantization scheme you then pick for *serving* — INT8, AWQ, GPTQ, or NF4
again, per Module 6. Training-time quantization and serving-time quantization are two separate
decisions that happen to reuse the same bit-width vocabulary.`,
    },
  ],

  glossary: [
    { term: 'SFT (supervised fine-tuning)', def: 'Fine-tuning a pretrained model on curated (instruction, response) pairs with the ordinary next-token loss, masked so gradient only flows through the response tokens.' },
    { term: 'PEFT', def: 'Parameter-efficient fine-tuning — the family of methods that update a small fraction of parameters, or none of the original ones, instead of every weight.' },
    { term: 'LoRA', def: "Low-Rank Adaptation. Freezes W0 and trains a rank-r product BA added beside it; merges back into W0 for zero added inference cost." },
    { term: 'rank (r) / alpha', def: 'r sets the dimensionality of a LoRA update; alpha scales it (the update is multiplied by alpha/r) so quality is stable as r changes.' },
    { term: 'QLoRA', def: 'LoRA with the frozen base model stored in 4-bit NF4, dequantized on the fly per matmul — cuts base-weight storage 4x on top of what freezing already saves.' },
    { term: 'NF4', def: '4-bit NormalFloat — a quantization data type whose bins are spaced to match a roughly-Gaussian weight distribution, rather than uniform INT4 bins.' },
    { term: 'DoRA', def: 'Weight-Decomposed LoRA. Splits a frozen weight into magnitude and direction, trains both, and still merges for free — usually a small quality step up from plain LoRA.' },
    { term: 'adapter (Houlsby adapter)', def: 'A small bottleneck feed-forward block inserted inline into each transformer sublayer. Cannot be merged away, so it adds latency at every inference call.' },
    { term: 'reward model', def: 'A model trained on pairwise human preference comparisons to output a scalar score, used as the reward signal in RLHF\'s PPO stage.' },
    { term: 'RLHF', def: 'Reinforcement learning from human feedback: train a reward model on preference pairs, then run PPO against it with a KL penalty back to the SFT policy.' },
    { term: 'DPO', def: "Direct Preference Optimization — reparameterizes RLHF's objective so the reward model cancels out, turning preference learning into a classification loss over the policy's own log-probabilities." },
    { term: 'GRPO', def: "Group Relative Policy Optimization — drops PPO's critic network by scoring each sampled completion's advantage against the mean/std of a group sampled for the same prompt." },
    { term: 'RLVR', def: 'RL with verifiable rewards — reward comes from a deterministic checker (does the math answer match, do the tests pass) instead of a learned reward model.' },
    { term: 'catastrophic forgetting', def: 'A model loses previously-held capabilities while being fine-tuned on new data. PEFT methods generally cause less of it than full fine-tuning does.' },
  ],
};
