export default {
  n: 0,
  slug: 'transformer-fundamentals',
  title: 'TRANSFORMER FUNDAMENTALS',
  tagline: 'You cannot reason about inference without a mechanical picture of the forward pass.',
  hours: '8–12 hours',
  prereqs: ['Python and NumPy', 'Matrix multiplication', 'What a softmax is'],

  bigIdea: `Everything in this course is downstream of one question: **what has to happen in
memory for a model to produce one token?** You cannot answer that until you can trace a tensor
through a transformer layer and say its shape at every arrow.

So Module 0 is not a transformers tutorial with an inference flavour. It is a shape-counting
exercise. By the end you should be able to look at a config file — \`n_layers\`, \`d_model\`,
\`n_heads\`, \`n_kv_heads\`, \`vocab_size\` — and derive, on paper, how many parameters the model
has, how many bytes that is on disk, and which of those bytes must be re-read for every single
token generated.

The last part is the punchline. A transformer's weights are used *once* per forward pass. There
is no reuse. When you generate one token you stream all 16 GB of a Llama-3-8B off the memory bus
to do about 16 GFLOPs of arithmetic with them. Hold that ratio in your head — 16 GB moved,
16 GFLOP done — because the rest of the course is a sustained assault on it.`,

  concepts: [
    {
      name: 'Tokens, embeddings, and the residual stream',
      keyPoint: 'The residual stream is a fixed-width bus of shape [batch, seq, d_model] that every layer reads from and writes back into.',
      body: `Text becomes integers before it becomes anything else. A tokenizer (BPE for most
modern models) maps a string to a list of integers in \`[0, vocab_size)\`. Llama 3 uses a
128,256-entry vocabulary; GPT-2 used 50,257. Nothing about this step is learned during
pretraining — the merges were fit once on a corpus and frozen.

Those integers index into an embedding table of shape \`[vocab_size, d_model]\`. For Llama-3-8B
that is \`[128256, 4096]\` = 525 million parameters, just to turn integers into vectors. The
lookup produces a tensor of shape \`[batch, seq, d_model]\`.

That tensor is the **residual stream**, and it is the single most useful mental object in the
architecture. Its width never changes. Every block in the network reads the stream, computes
something, and *adds* the result back:

\`\`\`
x = x + attention(norm(x))
x = x + mlp(norm(x))
\`\`\`

Note the \`+\`. Layers do not transform the stream, they *contribute* to it. This is why you can
delete a layer from a trained transformer and often get degraded-but-coherent output: you removed
one contribution from a sum of 32, not a link in a chain.

For inference, the shape \`[batch, seq, d_model]\` is where your intuition should live. During
**prefill** \`seq\` is the whole prompt — maybe 2,000. During **decode** \`seq\` is 1. Same weights,
same code path, wildly different arithmetic. That gap is the entire subject of Module 4.

A "hidden state" is just one slice of this stream: a single \`d_model\`-length vector at one
position, at one layer. When people say a model "represents" something, they mean a direction in
this 4096-dimensional space.`,
      ascii: `  token ids  [B, S]
      |
      v  embedding lookup  [128256, 4096]
  ┌─────────────────────────────────────────┐
  │  RESIDUAL STREAM      [B, S, 4096]      │  <-- width never changes
  └─────────────────────────────────────────┘
      |         ^                |         ^
      v         |                v         |
   norm->attn --+             norm->mlp ---+     (x32 layers)
      |
      v  final norm + lm_head  [4096, 128256]
   logits  [B, S, 128256]`,
    },
    {
      name: 'Attention: Q, K, V and what a score actually means',
      keyPoint: 'Attention is a weighted average of value vectors, where the weights come from the dot product of one query against every key.',
      body: `Every position produces three vectors from its residual-stream slice, via three learned
projections:

\`\`\`
Q = x @ W_q     "what am I looking for?"
K = x @ W_k     "what do I offer?"
V = x @ W_v     "what do I pass on if selected?"
\`\`\`

The computation is four steps, and the shapes matter more than the formula:

\`\`\`
scores  = Q @ K.T                    [S, S]     every query vs every key
scores  = scores / sqrt(head_dim)    [S, S]     keep logits out of softmax saturation
scores  = scores + causal_mask       [S, S]     -inf above the diagonal
weights = softmax(scores, dim=-1)    [S, S]     each ROW sums to 1
out     = weights @ V                [S, d_h]   weighted average of value vectors
\`\`\`

Read row \`i\` of \`weights\`: it is a probability distribution over positions \`0..i\` saying how
much position \`i\` cares about each earlier position. The output at \`i\` is that distribution
applied to the value vectors. Attention is *retrieval by soft lookup* — dot-product similarity
picks what to read, and V is what gets read.

The \`1/sqrt(head_dim)\` is not cosmetic. If Q and K entries are roughly unit-variance and
independent, their dot product over \`head_dim=128\` terms has variance ~128, standard deviation
~11.3. Feed logits of that magnitude into a softmax and you get a near-one-hot distribution with
vanishing gradients. Dividing by \`sqrt(128) = 11.31\` puts the variance back at 1.

The inference-critical observation is asymmetry. **Q comes only from the current token. K and V
come from every token so far.** When you generate token 501, you need a fresh Q for position 501,
but you need the same K and V for positions 0–500 that you computed on the previous step. They
have not changed and they never will — this is precisely the fact that Module 2 exploits.`,
      ascii: `  Q [S, 128] · K^T [128, S]  ->  scores [S, S]

        k0   k1   k2   k3
   q0 [ ·  |-inf|-inf|-inf]     row 0 sees only position 0
   q1 [ ·  | ·  |-inf|-inf]
   q2 [ ·  | ·  | ·  |-inf]     softmax over each row
   q3 [ ·  | ·  | ·  | ·  ]     row 3 sees everything

   then  weights [S,S] @ V [S,128] -> out [S,128]`,
    },
    {
      name: 'Causal masking, and why it makes cached generation legal',
      keyPoint: 'Because position i can never attend to position j > i, the K and V of past tokens are frozen forever — which is what makes a cache correct rather than merely convenient.',
      body: `The causal mask sets \`scores[i][j] = -inf\` for all \`j > i\`, so \`softmax\` gives those
entries exactly zero weight. Implementations add a matrix of \`0\` and \`-inf\` (or \`-1e9\`) rather
than slicing, because a dense add is faster on a GPU than irregular indexing.

The training reason for this is the obvious one: it lets you compute the loss for all \`S\`
positions in a single forward pass without any position cheating by looking at its own answer.
That is **teacher forcing**, and it is why training is so much more hardware-efficient than
generation.

The inference reason is the one people skip, and it is the load-bearing fact of this entire
course. Causality means the representation of token \`i\` **cannot be influenced by anything that
comes after it**. So when you append token 501 to the sequence:

- the K and V vectors for positions 0–500 are bit-for-bit identical to what they were before;
- the *only* new work is computing K, V, and Q for position 501;
- everything else can be read from storage.

Without causality this would be false. In a bidirectional encoder like BERT, adding a token at
the end changes the representation of every earlier token, so there is nothing to cache. That is
the real reason decoder-only models won for generation: not that they are more expressive, but
that the causal constraint turns an O(n²) recomputation into an O(n) append.

Worth being precise about what is *not* cached. Q is not cached, because the query for a position
is used once, at the step where that position is the newest token, and then never again. Attention
outputs are not cached either. Only K and V — hence "KV cache" and not "QKV cache."`,
      ascii: `  step 500                          step 501
  ┌────────────────────────┐        ┌────────────────────────┬───┐
  │ K,V for tokens 0..500  │   ->   │ K,V for tokens 0..500  │new│
  └────────────────────────┘        └────────────────────────┴───┘
       already computed                  UNCHANGED           only
                                                             this
                                                             is new`,
    },
    {
      name: 'Multi-head attention, and every shape in the block',
      keyPoint: 'Heads are a reshape, not a loop: one [4096, 4096] matmul produces all 32 heads at once.',
      body: `One attention pattern per layer is too few — a token usually needs to track several
relationships at once. So the \`d_model\` vector is split into \`n_heads\` independent
\`head_dim\`-sized subspaces, attention runs in each, and the results are concatenated.

The efficiency point that trips people up: heads are *not* implemented as a loop. The projection
\`W_q\` is a single \`[4096, 4096]\` matrix. You do one matmul, then \`reshape\` and \`transpose\` the
result into \`[B, n_heads, S, head_dim]\`. The heads only exist as a view.

Here is the full trace for **Llama-3-8B** (\`d_model=4096\`, \`n_heads=32\`, \`n_kv_heads=8\`,
\`head_dim=128\`, \`ffn=14336\`), with batch \`B\` and sequence \`S\`:

| step | operation | output shape |
|---|---|---|
| input | residual stream | \`[B, S, 4096]\` |
| q_proj | \`@ [4096, 4096]\` | \`[B, S, 4096]\` |
| k_proj | \`@ [4096, 1024]\` | \`[B, S, 1024]\` |
| v_proj | \`@ [4096, 1024]\` | \`[B, S, 1024]\` |
| reshape Q | | \`[B, 32, S, 128]\` |
| reshape K,V | | \`[B, 8, S, 128]\` |
| repeat K,V | each kv head serves 4 q heads | \`[B, 32, S, 128]\` |
| scores | \`Q @ K.T\` | \`[B, 32, S, S]\` |
| weights | softmax | \`[B, 32, S, S]\` |
| context | \`@ V\` | \`[B, 32, S, 128]\` |
| merge | transpose + reshape | \`[B, S, 4096]\` |
| o_proj | \`@ [4096, 4096]\` | \`[B, S, 4096]\` |

Notice \`k_proj\` and \`v_proj\` output **1024**, not 4096. That is grouped-query attention: 8 KV
heads instead of 32, each shared by 4 query heads. It shrinks the KV cache by 4× on this model
and is the single most consequential architectural decision for inference cost. Module 6 covers
what it trades away.

Notice also \`[B, 32, S, S]\`. At \`S = 8192\` and \`B = 1\` in fp16 that one intermediate is
\`32 × 8192 × 8192 × 2 = 4.3 GB\` — for a single layer, for a single sequence. Never
materialising that matrix is what FlashAttention is for (Module 7).`,
      ascii: '',
    },
    {
      name: 'The MLP block: two thirds of the weights',
      keyPoint: 'The MLP holds roughly twice the parameters of the attention block, so at decode time it dominates the bytes you move.',
      body: `After attention, each position independently passes through a feed-forward network.
"Independently" is exact — there is no mixing across positions here, which makes it trivially
parallel and, at decode time, a plain matrix-vector product.

The classic transformer used \`W_2 @ relu(W_1 @ x)\` with an expansion factor of 4. Llama-family
models use **SwiGLU**, which has three matrices instead of two:

\`\`\`
gate = x @ W_gate        [B, S, 14336]
up   = x @ W_up          [B, S, 14336]
h    = silu(gate) * up   [B, S, 14336]     elementwise
out  = h @ W_down        [B, S, 4096]
\`\`\`

where \`silu(z) = z * sigmoid(z)\`. The \`gate\` branch acts as a learned, input-dependent filter on
the \`up\` branch. Because there are three matrices, the hidden width is chosen smaller than 4×
to keep the parameter count comparable — 14336 here, which is 3.5× \`d_model\`.

Now count parameters per layer for Llama-3-8B:

\`\`\`
attention:  W_q  4096 x 4096  = 16,777,216
            W_k  4096 x 1024  =  4,194,304
            W_v  4096 x 1024  =  4,194,304
            W_o  4096 x 4096  = 16,777,216
                               ------------
                                41,943,040

mlp:        W_gate 4096 x 14336 = 58,720,256
            W_up   4096 x 14336 = 58,720,256
            W_down 14336 x 4096 = 58,720,256
                                 ------------
                                 176,160,768
\`\`\`

The MLP is **4.2× the attention block**. Across 32 layers: 1.34 B attention parameters,
5.64 B MLP parameters. Add the 525 M embedding and 525 M unembedding and you get 8.03 B — the
"8B" on the label.

For inference this ranking matters. People assume attention dominates because it is the
interesting part and because of the O(S²) term. At decode time with a modest context, it does
not: you move 176 MB of MLP weights per layer versus 42 MB of attention weights, and the
attention arithmetic touches a KV cache that is usually far smaller than either.`,
      ascii: '',
    },
    {
      name: 'Normalization and the pre-norm residual stream',
      keyPoint: 'Modern models normalize the input to each block, never the residual stream itself, which keeps a clean gradient path from output to input.',
      body: `The original transformer put LayerNorm *after* each block:
\`x = norm(x + attn(x))\`. This is **post-norm**, and it is hard to train deep — the residual path
is interrupted by a normalization at every layer, so gradients get rescaled 2L times on the way
back.

Every modern LLM uses **pre-norm**: \`x = x + attn(norm(x))\`. The normalization is applied to the
*input* of the block; the residual stream itself is never touched. There is now an unbroken
identity path from the final layer to the embedding, which is why 80-layer models train at all.
A single final norm is applied before the LM head.

Most current models also replace LayerNorm with **RMSNorm**:

\`\`\`
LayerNorm:  y = g * (x - mean(x)) / sqrt(var(x) + eps) + b
RMSNorm:    y = g * x / sqrt(mean(x^2) + eps)
\`\`\`

RMSNorm drops the mean subtraction and the bias. It works about as well and is measurably
cheaper — one pass over the vector instead of two, and one fewer parameter tensor.

Why an inference course cares: normalization layers are **memory-bound with essentially zero
arithmetic intensity**. An RMSNorm over a \`[1, 4096]\` decode vector reads 4096 values, does a
handful of FLOPs each, and writes 4096 values back. If you launch it as its own kernel you pay a
full round trip to HBM for nothing. This is the canonical case for **kernel fusion** — folding
the norm into the matmul that follows it so the data never leaves on-chip memory. Modules 4 and 7
return to this. A transformer at decode time is a long chain of these tiny bandwidth-bound
operations glued between big matmuls, and the glue is not free.`,
      ascii: '',
    },
    {
      name: 'RoPE: position without a position vector',
      keyPoint: 'RoPE rotates Q and K by an angle proportional to absolute position, so their dot product depends only on relative distance.',
      body: `Attention as defined is permutation-invariant — shuffle the tokens and the scores follow
them unchanged. Position has to be injected explicitly.

The original approach added a learned or sinusoidal position vector to the embedding. Almost
every current model instead uses **RoPE** (Rotary Position Embedding), which does something
cleverer: it *rotates* Q and K, in-place, by an angle that depends on position.

Split each \`head_dim=128\` vector into 64 pairs of coordinates. Treat each pair as a point in 2D
and rotate it by \`m * theta_i\`, where \`m\` is the absolute position and \`theta_i =
base^(-2i/head_dim)\` with \`base\` typically 10000 (Llama 3 uses 500000 to extend context). Low-index
pairs rotate fast, high-index pairs rotate slowly — a positional code across many frequencies.

The property that makes it work falls out of rotation algebra: for a rotation \`R\`,

\`\`\`
(R_m q) · (R_n k) = q · R_(n-m) k
\`\`\`

The score between positions \`m\` and \`n\` depends only on \`n - m\`. You get **relative** position
from an operation applied using **absolute** position. No extra parameters, nothing added to the
residual stream.

Three inference consequences:

1. **RoPE is applied to Q and K before they are cached.** The cache stores rotated keys. Get the
   position index wrong on a cache hit and you silently corrupt the attention pattern — a
   classic prefix-caching bug (Module 6).
2. **V is not rotated.** Only Q and K participate in the dot product that needs position.
3. **Context extension is a RoPE trick.** Methods like position interpolation and YaRN rescale
   the frequencies so a model trained at 8k can attend over 128k. This is why long-context
   support often arrives without full retraining — and why long-context *quality* frequently
   lags long-context *capability*.`,
      ascii: '',
    },
    {
      name: 'The LM head, and the training/inference split',
      keyPoint: 'Training computes loss at all S positions in one pass; generation throws away all but the last, then does it all again.',
      body: `The final residual stream is normalized and projected to vocabulary size by the LM head,
a \`[4096, 128256]\` matrix. Output shape: \`[B, S, 128256]\`. Those are **logits** — unnormalized
scores, one per vocabulary entry.

That tensor is enormous relative to everything else. At \`B=1, S=4096\` in fp32 it is
\`4096 × 128256 × 4 = 2.1 GB\`. Serving implementations therefore compute the LM head only for
positions they actually need — during decode, that is the single last position, turning a
\`[4096, 128256]\` matmul into a matrix-vector product.

**Training** runs one forward pass over the whole sequence, computes cross-entropy at every
position against the next token, and updates. Every position is supervised in parallel. The GPU
is doing \`S\` positions' worth of arithmetic per weight load, and arithmetic intensity is high.

**Inference** cannot do that, because position \`i+1\`'s input is position \`i\`'s output. You run
the forward pass, take the logits at the last position, sample a token, append it, and run again.
The loop is serial by construction.

Here is the asymmetry stated as a ratio. Generating 500 tokens from a 500-token prompt requires:

- **Training-style work:** one pass, 1000 positions, weights read **once**.
- **Generation:** one prefill pass over 500 positions, then 500 decode passes each producing one
  token. Weights read **501 times**.

Same model, same total tokens, 500× the memory traffic. The prefill pass is doing hundreds of
positions per weight load and the GPU is busy. Each decode pass does *one* position per weight
load and the GPU is idle waiting on HBM.

That is the whole problem. Everything from here is an attempt to get more useful work out of each
weight load.`,
      ascii: '',
    },
  ],

  mathLab: {
    prompt: `Derive the parameter count of **Llama-3-8B** from its config, without looking up
the answer, and check that it lands on 8.03 B.

Config:

\`\`\`
n_layers        = 32
d_model         = 4096
n_heads         = 32
n_kv_heads      = 8          (grouped-query attention)
head_dim        = 128
ffn_hidden      = 14336      (SwiGLU: gate, up, down)
vocab_size      = 128256
tied_embeddings = false      (separate input embedding and LM head)
\`\`\`

Compute each of these, showing the multiplication:

1. Embedding table parameters.
2. Per-layer attention parameters (\`W_q, W_k, W_v, W_o\`). Watch the GQA shapes.
3. Per-layer MLP parameters (all three SwiGLU matrices).
4. Total for all 32 layers.
5. LM head parameters.
6. Grand total. Then: size in GB at fp16, and at fp8.
7. Finally — the question that matters — **how many bytes must be read from memory to generate
   one token**, ignoring the KV cache?`,

    solution: `**1. Embedding**

\`\`\`
128256 x 4096 = 525,336,576
\`\`\`

**2. Attention, per layer.** With GQA the K and V projections output \`n_kv_heads x head_dim =
8 x 128 = 1024\`, not 4096:

\`\`\`
W_q:  4096 x (32 x 128) = 4096 x 4096 = 16,777,216
W_k:  4096 x (8  x 128) = 4096 x 1024 =  4,194,304
W_v:  4096 x (8  x 128) = 4096 x 1024 =  4,194,304
W_o:  (32 x 128) x 4096 = 4096 x 4096 = 16,777,216
                                        -----------
                                         41,943,040
\`\`\`

**3. MLP, per layer.** Three matrices, all \`4096 x 14336\` in size:

\`\`\`
W_gate: 4096 x 14336 = 58,720,256
W_up:   4096 x 14336 = 58,720,256
W_down: 14336 x 4096 = 58,720,256
                       -----------
                       176,160,768
\`\`\`

**4. All 32 layers**

\`\`\`
per layer = 41,943,040 + 176,160,768 = 218,103,808
x 32      = 6,979,321,856
\`\`\`

(RMSNorm gains are \`2 x 4096\` per layer = 262,144 total. Negligible, but they exist.)

**5. LM head**

\`\`\`
4096 x 128256 = 525,336,576
\`\`\`

**6. Grand total**

\`\`\`
  525,336,576   embedding
6,979,321,856   layers
  525,336,576   lm head
      262,144   norms
-------------
8,030,257,152   = 8.03 B  ✓
\`\`\`

At fp16 (2 bytes): \`8.03e9 x 2 = 16.06 GB\`.
At fp8  (1 byte):  \`8.03 GB\`.

**7. Bytes per generated token**

Every weight participates in every forward pass, and each is read once. So generating one token
requires reading **~16.06 GB** at fp16.

Two refinements worth noting. The embedding table is a *lookup*, not a matmul — you touch one row
of 4096 values, not all 525 M parameters. So the honest figure is:

\`\`\`
16.06 GB - 1.05 GB (embedding) + 8 KB (one row) ~= 15.01 GB
\`\`\`

Sanity-check it against hardware. An H100 SXM moves 3.35 TB/s:

\`\`\`
15.01 GB / 3350 GB/s = 4.48 ms  ->  ~223 tokens/second, absolute ceiling
\`\`\`

That is the fastest this model can possibly decode at batch size 1 on that GPU, even with a
perfect implementation and infinite compute. It is set entirely by bandwidth. Module 4 makes this
reasoning systematic.`,
  },

  codeLab: {
    goal: `Implement single-head causal attention in pure NumPy and print the shape at every
step. No PyTorch, no \`nn.MultiheadAttention\`. The point is that after doing this you will never
again be unsure what \`[B, H, S, D]\` means.

Then verify the two facts that the rest of the course rests on: that the mask makes the output at
position \`i\` independent of everything after \`i\`, and that appending a token leaves the earlier
K and V untouched.`,
    code: `import numpy as np

np.random.seed(0)

B, S, D_MODEL, D_HEAD = 1, 6, 16, 8


def softmax(x, axis=-1):
    # subtract the max for numerical stability -- exp(1000) overflows
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


def attention(x, Wq, Wk, Wv, show=False):
    """Single-head causal attention. x: [B, S, D_MODEL] -> [B, S, D_HEAD]"""
    Q = x @ Wq                                   # [B, S, D_HEAD]
    K = x @ Wk                                   # [B, S, D_HEAD]
    V = x @ Wv                                   # [B, S, D_HEAD]

    scores = Q @ K.transpose(0, 2, 1)            # [B, S, S]
    scores = scores / np.sqrt(D_HEAD)            # keep softmax out of saturation

    s = x.shape[1]
    mask = np.triu(np.ones((s, s), dtype=bool), k=1)   # True strictly above diagonal
    scores = np.where(mask, -np.inf, scores)

    weights = softmax(scores)                    # [B, S, S], each row sums to 1
    out = weights @ V                            # [B, S, D_HEAD]

    if show:
        for name, t in [("x", x), ("Q", Q), ("K", K), ("V", V),
                        ("scores", scores), ("weights", weights), ("out", out)]:
            print(f"  {name:8s} {str(t.shape):16s}")
        print("\\n  row sums of weights (must all be 1.0):",
              np.round(weights[0].sum(-1), 6))
        print("\\n  attention weights, rounded -- note the zeros above the diagonal:")
        print(np.round(weights[0], 3))
    return out, K, V


x = np.random.randn(B, S, D_MODEL)
Wq = np.random.randn(D_MODEL, D_HEAD) / np.sqrt(D_MODEL)
Wk = np.random.randn(D_MODEL, D_HEAD) / np.sqrt(D_MODEL)
Wv = np.random.randn(D_MODEL, D_HEAD) / np.sqrt(D_MODEL)

print("=== forward pass ===")
out, K, V = attention(x, Wq, Wk, Wv, show=True)

# ---- claim 1: causality. Changing the LAST token must not move earlier outputs.
print("\\n=== claim 1: causal mask actually works ===")
x2 = x.copy()
x2[0, -1] = np.random.randn(D_MODEL)          # scribble over the final token
out2, _, _ = attention(x2, Wq, Wk, Wv)
print("  max change at positions 0..S-2:", np.abs(out[0, :-1] - out2[0, :-1]).max())
print("  max change at the last position:", np.abs(out[0, -1] - out2[0, -1]).max())

# ---- claim 2: appending a token leaves earlier K/V bit-identical.
#      This is the entire justification for the KV cache.
print("\\n=== claim 2: past K/V are frozen ===")
x3 = np.concatenate([x, np.random.randn(1, 1, D_MODEL)], axis=1)   # S -> S+1
_, K3, V3 = attention(x3, Wq, Wk, Wv)
print("  K identical for old positions:", np.array_equal(K, K3[:, :S]))
print("  V identical for old positions:", np.array_equal(V, V3[:, :S]))
print("  -> so recomputing them every step is pure waste. See Module 2.")

# ---- TODO for you:
#   1. Extend to multi-head: project to [B, S, N_HEADS * D_HEAD], reshape to
#      [B, N_HEADS, S, D_HEAD], attend, then merge back. Print shapes throughout.
#   2. Make it grouped-query: use 2 KV heads for 4 query heads. You will need
#      np.repeat on axis 1. Confirm the K/V tensors shrink by 2x.
`,
    expect: `\`weights\` is strictly lower-triangular after softmax — every entry above the
diagonal is exactly \`0.0\`, and every row sums to \`1.0\`.

Claim 1 prints a max change at positions 0..S−2 of \`0.0\` (exactly zero, not merely small) and a
nonzero change at the last position. If the earlier positions moved at all, your mask is wrong.

Claim 2 prints \`True\` twice. That is the licence to cache.

Common failure: using \`-1e9\` instead of \`-np.inf\` gives \`~1e-40\` above the diagonal rather than
exact zero. Fine numerically, but claim 1 will print something like \`3e-38\` instead of \`0.0\`.
Worth seeing both.`,
    stretch: `Time \`attention()\` at \`S = 128, 256, 512, 1024, 2048\` with everything else fixed
and plot time against \`S\`. Fit a curve. You should get a clean quadratic, because \`scores\` is
\`[S, S]\`. Then compute how much memory that \`[B, 32, S, S]\` intermediate would need for a real
model at \`S = 8192\` in fp16 — the answer is 4.3 GB per layer, and it is why Module 7 exists.`,
  },

  papers: [
    {
      title: 'Attention Is All You Need',
      by: 'Vaswani et al., 2017',
      url: 'https://arxiv.org/abs/1706.03762',
      why: 'The original. Read it for the architecture, not for the state of the art — almost every detail (post-norm, learned positional encodings, ReLU FFN, the encoder-decoder structure) has since been replaced in decoder-only LLMs.',
      frame: `Read **Section 3.2** (scaled dot-product and multi-head attention) closely and
make sure you can reproduce Figure 2 from memory. Read **Section 3.5** on positional encoding to
understand what RoPE later replaced. Skim Sections 4–6 entirely — the machine-translation results
are of historical interest only. The single sentence to internalize is the note in 3.2.1 on why
they divide by \`sqrt(d_k)\`.`,
    },
    {
      title: 'The Illustrated Transformer',
      by: 'Jay Alammar, 2018',
      url: 'https://jalammar.github.io/illustrated-transformer/',
      why: 'The best visual explanation of the forward pass. If the shapes in this module have not clicked yet, this is the fastest fix.',
      frame: `Work through it with a pen and write the tensor shape next to every diagram. The
post uses the original encoder-decoder architecture, so mentally delete the encoder and the
cross-attention — a modern LLM is only the decoder stack, minus its cross-attention block.`,
    },
    {
      title: "Let's build GPT: from scratch, in code, spelled out",
      by: 'Andrej Karpathy, 2023',
      url: 'https://www.youtube.com/watch?v=kCc8FmEb1nY',
      why: 'Two hours, and you leave with a working transformer you typed yourself. There is no substitute for this.',
      frame: `Type it, do not watch it. Pause at every tensor operation and predict the output
shape before he says it. The companion repo is **nanoGPT**, which is the cleanest readable GPT
implementation in existence and worth keeping open for the rest of this course.`,
    },
    {
      title: 'RoFormer: Enhanced Transformer with Rotary Position Embedding',
      by: 'Su et al., 2021',
      url: 'https://arxiv.org/abs/2104.09864',
      why: 'RoPE, which essentially every current open-weight LLM uses. It matters for inference because the cache stores rotated keys.',
      frame: `**Section 3.4.2** has the derivation that the inner product depends only on relative
position — that is the whole idea, and it is four lines of algebra. Skim the theoretical
long-term-decay analysis in 3.4.3. Skip the experiments.`,
    },
    {
      title: 'Root Mean Square Layer Normalization',
      by: 'Zhang & Sennrich, 2019',
      url: 'https://arxiv.org/abs/1910.07467',
      why: 'RMSNorm, used by Llama, Mistral, Qwen, Gemma and most others. Short, and a good example of a simplification that stuck.',
      frame: 'Read Section 3 and stop. The claim is that re-centering does not matter, only re-scaling does. Ten minutes.',
    },
  ],

  checkpoint: {
    claim: `You can hand-draw the data flow for one transformer layer and state the tensor shape
at every arrow — from token ids to logits — without notes. And you can explain why the causal
mask is what makes a KV cache correct, rather than just a speed hack.`,
    questions: [
      {
        q: 'Why are keys and values cached during generation, but not queries?',
        a: `Because of what each one is used for across steps. The key and value for position \`i\`
are consumed by *every* future position that attends back to \`i\` — they are read again on every
subsequent decode step, forever. The query for position \`i\` is used exactly once, on the step
where \`i\` is the newest token, to compute that token's output; after that it is never referenced
again. Caching it would store data nothing will read. The causal mask is what guarantees K and V
are stable: since no position can attend forward, adding tokens at the end cannot change the K or
V of anything earlier.`,
      },
      {
        q: 'A model has d_model=4096 and n_heads=32. What is head_dim, and what is the shape of the attention score matrix for a batch of 4 sequences of length 1024?',
        a: `\`head_dim = d_model / n_heads = 4096 / 32 = 128\`. The score matrix is
\`[batch, n_heads, seq, seq] = [4, 32, 1024, 1024]\`. That is 134,217,728 elements, or 268 MB in
fp16 — for one layer. Materialising it for all 32 layers at once would take 8.6 GB, which is why
attention is computed layer by layer, and why FlashAttention avoids writing it to HBM at all.
Note that \`head_dim\` is only forced to be \`d_model / n_heads\` by convention; some models set it
independently.`,
      },
      {
        q: 'Which holds more parameters in a Llama-style layer: attention or the MLP? By how much, and why should an inference engineer care?',
        a: `The MLP, by roughly 4×. For Llama-3-8B it is 176.2 M parameters in the MLP versus
41.9 M in attention per layer — the SwiGLU block has three \`4096 × 14336\` matrices while
attention has two \`4096 × 4096\` and, thanks to GQA, two small \`4096 × 1024\` ones. It matters
because at decode time your cost is bytes moved, and about 80% of the weight bytes you move are
MLP weights. Attention gets disproportionate attention (so to speak) because of its O(S²) term,
but that term concerns the KV cache and activations, not weights.`,
      },
      {
        q: 'Explain the difference between teacher forcing and the autoregressive loop in terms of how many times the weights are read.',
        a: `Under teacher forcing the whole target sequence is already known, so a single forward
pass computes the loss at all \`S\` positions simultaneously. The weights are read **once** and
amortized across \`S\` positions of arithmetic — high arithmetic intensity, compute-bound, GPU
busy. In the autoregressive loop each token depends on the previous one, so you cannot batch
positions. Generating \`N\` tokens takes \`N\` sequential forward passes and the weights are read
**N times**, each amortized over a single position. Same model and same token count, but roughly
\`N\`× the memory traffic — and that traffic, not the arithmetic, is what you wait on.`,
      },
      {
        q: 'Why does RoPE get applied before the keys are written to the cache, and what breaks if you get the position index wrong?',
        a: `RoPE rotates Q and K by an angle proportional to absolute position, and it is applied
inside the attention block before the dot product, so the cache naturally stores already-rotated
keys — recomputing rotations on every read would waste work. The correctness of attention then
depends on each cached key having been rotated by *its own* position index. If you reuse a cached
prefix at a different offset — as prefix caching does — and do not account for the position
change, the keys carry the wrong rotation, the relative-distance property breaks, and attention
scores are silently wrong. No crash, no error, just degraded output. It is one of the more common
and harder-to-spot bugs in serving systems.`,
      },
    ],
  },

  pitfalls: [
    {
      wrong: 'Multi-head attention runs a loop over heads.',
      right: `Heads are a reshape of a single large matmul, not separate operations. \`W_q\` is one
\`[d_model, d_model]\` matrix; you multiply once, then view the result as
\`[B, n_heads, S, head_dim]\`. Thinking of heads as a loop leads to badly wrong performance
intuitions — you start believing 32 heads costs 32 kernel launches when it costs one.`,
    },
    {
      wrong: 'Attention is the expensive part of a transformer.',
      right: `It depends entirely on the regime. At long context during prefill, attention's
O(S²) term dominates. At decode time with moderate context, the MLP dominates: about 80% of the
weight bytes you move per token are MLP weights, and attention is reading a KV cache that is
often much smaller. "Attention is expensive" is a statement about prefill and long context, not a
universal truth.`,
    },
    {
      wrong: 'The KV cache is an optimization you could skip if you had enough compute.',
      right: `No amount of compute makes recomputation free, because the bottleneck is not
compute. Without a cache, generating token \`n\` means re-running the full forward pass over all
\`n\` positions — re-reading every weight *and* redoing all the attention work. It is O(n²) in
total attention FLOPs and it wastes bandwidth on top. The cache is not a nice-to-have; it is what
makes generation tractable at all.`,
    },
    {
      wrong: 'GQA is a quantization or compression technique applied after training.',
      right: `GQA is an *architectural* choice baked in at training time: the model has fewer K
and V projection matrices than Q projections. You cannot switch a trained MHA model to GQA for
free — you have to fuse the KV heads and fine-tune to recover quality (the "uptraining" procedure
in the GQA paper). What GQA shrinks is the KV cache, not the weights.`,
    },
  ],

  glossary: [
    { term: 'residual stream', def: 'The [batch, seq, d_model] tensor that every block reads from and adds back into. Its width is constant through the network.' },
    { term: 'hidden state', def: 'One d_model-length slice of the residual stream, at one position and one layer.' },
    { term: 'head_dim', def: 'The width of one attention head. Usually d_model / n_heads, though it can be set independently.' },
    { term: 'GQA', def: 'Grouped-query attention. Fewer K/V heads than Q heads, with each K/V head shared by a group of query heads. Shrinks the KV cache proportionally.' },
    { term: 'SwiGLU', def: 'The gated feed-forward block used by Llama-family models: down(silu(gate(x)) * up(x)). Three weight matrices instead of two.' },
    { term: 'RMSNorm', def: 'Normalization by root-mean-square only, with no mean subtraction and no bias. Cheaper than LayerNorm and works as well.' },
    { term: 'pre-norm', def: 'Normalizing the input to each block rather than its output, leaving the residual path unbroken. Required in practice for deep transformers.' },
    { term: 'RoPE', def: 'Rotary position embedding. Rotates Q and K by an angle proportional to absolute position so their dot product depends only on relative distance.' },
    { term: 'logits', def: 'The unnormalized [.., vocab_size] scores produced by the LM head, before any softmax or sampling.' },
    { term: 'teacher forcing', def: 'Training on the known target sequence so all positions are computed in one parallel pass. The reason training is far more hardware-efficient than generation.' },
  ],
};
