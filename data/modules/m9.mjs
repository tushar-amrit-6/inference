export default {
  n: 9,
  slug: 'distributed-inference',
  title: 'DISTRIBUTED INFERENCE',
  tagline: 'When the model does not fit, the interconnect becomes the memory bus — and it is much slower.',
  hours: '8–10 hours',
  prereqs: ['Module 2', 'Module 4', 'Module 5'],

  bigIdea: `A 70B model at fp16 is 141 GB of weights. No single GPU holds that. So you split it,
and the moment you do, a new tier appears in the memory hierarchy: **the interconnect**.

NVLink between H100s moves about 900 GB/s. HBM moves 3,350 GB/s. PCIe Gen5 moves about 64 GB/s.
Splitting a model means that some of the data movement in every forward pass happens across a link
that is 3.7× slower than memory at best, and 52× slower at worst. Every technique in this module
is about controlling how much traffic crosses that link.

There is a compensation, and it is the reason multi-GPU inference works at all: **splitting a model
across \`N\` GPUs also multiplies your aggregate memory bandwidth by \`N\`.** Four H100s give you
13.4 TB/s of HBM. Since decode is bandwidth-bound, four GPUs can decode a model roughly four times
faster than one could — if the communication does not eat the gain.

So distributed inference is a trade between two effects that both scale with GPU count: more
bandwidth (good, linear) versus more communication (bad, and dependent on topology). Getting it
right means knowing which parallelism strategy puts the traffic where the fast links are.

The module ends with the idea that follows most directly from everything in these chapters:
**prefill and decode are different workloads, so stop running them on the same machines.**`,

  concepts: [
    {
      name: 'Tensor parallelism: split inside the layer',
      keyPoint: 'Column-parallel then row-parallel matmuls let a layer be split with exactly one all-reduce at the end, and there are two such all-reduces per transformer layer.',
      body: `Tensor parallelism (Megatron-LM) splits the weight matrices themselves across GPUs.
The arrangement is chosen so that communication happens as rarely as possible.

**The MLP block.** Two matmuls in sequence: \`Y = GeLU(X @ A)\`, then \`Z = Y @ B\`.

Split \`A\` **by columns** across GPUs. Each GPU computes a slice of \`Y\`. Crucially, GeLU is
elementwise, so each GPU can apply it to its own slice without needing anyone else's — **no
communication**.

Then split \`B\` **by rows**, matching the column split of \`A\`. Each GPU computes a partial \`Z\`
using its slice of \`Y\` and its rows of \`B\`. The partials sum to the correct answer, so one
**all-reduce** finishes the block.

\`\`\`
GPU 0:  X @ A[:, :h/2]  -> Y0 -> GeLU -> Y0 @ B[:h/2, :] -> Z_partial_0
GPU 1:  X @ A[:, h/2:]  -> Y1 -> GeLU -> Y1 @ B[h/2:, :] -> Z_partial_1
                                                 all-reduce -> Z
\`\`\`

One all-reduce for two matmuls. Splitting the other way round would need communication between
them.

**The attention block** splits naturally by head: give each GPU a subset of heads, let it compute
those heads' attention independently, then all-reduce after the output projection. Also one
all-reduce.

**So: two all-reduces per transformer layer.** For an 80-layer model that is 160 all-reduces per
forward pass, on the critical path.

**Communication volume.** A ring all-reduce moves \`2(N−1)/N\` times the data per GPU. For an
activation of shape \`[batch, seq, d_model]\` in fp16:

\`\`\`
bytes per all-reduce = 2 * batch * seq * d_model * 2 * (N-1)/N
\`\`\`

At decode (\`seq = 1\`), batch 32, \`d_model = 8192\`, TP=4:

\`\`\`
2 * 32 * 1 * 8192 * 2 * 0.75 = 786,432 bytes = 0.79 MB per all-reduce
x 160 all-reduces = 126 MB per token
\`\`\`

Over NVLink at 900 GB/s that is 0.14 ms — against a decode step of a few milliseconds, so roughly
5% overhead. **Over PCIe at 64 GB/s it is 1.97 ms**, which can double your step time. This is the
single most important practical fact in this module: **TP belongs inside a node, on NVLink.**

**The GQA ceiling.** TP splits attention by head, and a KV head cannot be split across GPUs
without replicating its cache. With 8 KV heads you can run TP-8 cleanly, one KV head per GPU. At
TP-16 you must duplicate KV heads, so per-GPU cache stops shrinking as you add GPUs — you pay for
hardware that adds bandwidth and compute but no cache capacity. **The KV head count silently caps
useful TP degree** — or rather, it caps the degree to which *this* way of splitting the cache keeps
paying. Splitting it along a different axis lifts the cap, which is what the decode
context-parallelism concept below is about.`,
      ascii: `  MLP under TP-2: column-split then row-split, one all-reduce

         X (replicated on both GPUs)
         |                    |
    A[:, :h/2]           A[:, h/2:]      <- column-parallel
         |                    |
       GeLU                 GeLU         <- elementwise, NO comms needed
         |                    |
    B[:h/2, :]           B[h/2:, :]      <- row-parallel
         |                    |
      Z_part_0            Z_part_1
         \\__________ + __________/       <- ONE all-reduce
                     |
                     Z`,
    },
    {
      name: 'Pipeline parallelism: split across layers',
      keyPoint: 'Pipelining sends only activations between stages, so it survives slow links — but a single request gets no latency benefit and creates a bubble.',
      body: `Pipeline parallelism assigns different *layers* to different GPUs. With 80 layers on
4 GPUs, each holds 20 consecutive layers. Activations flow forward through the stages.

**Communication is tiny.** Between stages you send one activation tensor:
\`batch × seq × d_model × 2\` bytes, once per stage boundary. At decode with batch 32 and
\`d_model = 8192\` that is 0.52 MB per boundary, three boundaries — about 1.6 MB per token, versus
126 MB for TP-4. **Roughly 80× less traffic.**

That is why PP tolerates slow interconnects. Across nodes, over Ethernet or InfiniBand, PP works
and TP does not.

**The bubble is the cost.** A single request must traverse all four stages sequentially. While it
is in stage 2, stages 1, 3 and 4 have nothing to do. For one request in flight, utilization is
\`1/N\`.

The fix is to keep multiple microbatches in flight so every stage always has work:

\`\`\`
time ->
stage 0:  [m1][m2][m3][m4][m5]...
stage 1:      [m1][m2][m3][m4]...
stage 2:          [m1][m2][m3]...
stage 3:              [m1][m2]...
          ^^^^^^^^^^^^ the fill bubble
\`\`\`

Bubble fraction is \`(N−1)/(M + N−1)\` for \`M\` microbatches and \`N\` stages. With 4 stages and 32
microbatches: \`3/35 = 8.6%\`. Acceptable. With 4 microbatches: \`3/7 = 43%\`. Not.

**In LLM serving the microbatches are your concurrent requests**, so pipeline efficiency depends
on having enough concurrency — which brings you back to KV cache capacity. Under light load a
pipelined deployment is badly underutilized.

**PP does not reduce single-request latency.** A request still passes through every layer once; you
have only distributed where they live. TP, by contrast, genuinely speeds up each layer because you
have \`N×\` the bandwidth working on it. **This is the key difference: TP improves latency, PP
improves capacity.**

**The standard combination** follows directly:

\`\`\`
TP within a node   (NVLink, 900 GB/s, high-volume all-reduces)
PP across nodes    (InfiniBand/Ethernet, low-volume activation passing)
\`\`\`

An 8-GPU node running TP-8, with PP across nodes. Sequence or context parallelism (splitting the
sequence dimension, as in Ring Attention) is a third axis used for very long context, where even
one sequence's activations and KV cache exceed a single GPU — and, in the decode-specific form two
concepts down, as the answer to the KV-head ceiling.`,
      ascii: '',
    },
    {
      name: 'What TP degree actually does to your numbers',
      keyPoint: 'TP divides weight bytes and KV cache per GPU while multiplying aggregate bandwidth, so it improves both latency and capacity — until communication or KV head count stops it.',
      body: `Work through TP for Llama-3-70B on H100s, 8k context, so the effects are concrete.

**Weights per GPU:** \`141.2 / N\` GB.
**KV cache per GPU:** \`320 KiB per token / N\` — because each GPU holds only its own heads' K and V,
provided \`N ≤ n_kv_heads\`.
**Aggregate bandwidth:** \`3.35 × N\` TB/s.

\`\`\`
TP   weights/GPU   KV/tok/GPU   agg BW      fits in 80GB?   TPOT floor
 1     141.2 GB      320 KiB    3.35 TB/s        NO             --
 2      70.6 GB      160 KiB    6.70 TB/s        yes         21.1 ms
 4      35.3 GB       80 KiB   13.40 TB/s        yes         10.5 ms
 8      17.7 GB       40 KiB   26.80 TB/s        yes          5.3 ms
16       8.8 GB       40 KiB   53.60 TB/s        yes          2.6 ms
\`\`\`

TPOT floor is \`141.2 GB / aggregate bandwidth\` — the weights are read once collectively per step.

Three things to read out of this table.

**TP improves latency roughly linearly.** From TP-2 to TP-8, TPOT falls 4×. You are not just
fitting the model, you are decoding it faster, because you added memory bandwidth.

**KV cache per GPU stops shrinking at TP-8.** Llama-3-70B has 8 KV heads. At TP-16 two GPUs share
each KV head and must both hold its cache, so per-GPU KV stays at 40 KiB/token. You doubled the
hardware and got no additional cache capacity — only bandwidth. That is the GQA ceiling biting, and
that flat 40 KiB row is what the next concept exists to fix: layering DCP-2 on top of TP-16 shards
the same cache by position and takes it to 20 KiB.

**Communication grows with TP degree.** The all-reduce volume scales as \`(N−1)/N\`, approaching a
constant, but the *number of participants* grows and ring all-reduce latency grows with hop count.
Beyond a node, where you leave NVLink, the cost jumps discontinuously.

Now the effect that surprises people: **TP degree affects TTFT and throughput differently.**

**TTFT improves with TP** because prefill is compute-bound and you added compute. Communication
during prefill is larger in absolute terms (activations are \`seq\` times bigger) but it is
amortized over far more arithmetic, so the fraction of time spent communicating is *lower* during
prefill than decode.

**Throughput per GPU usually falls with TP.** You are paying communication overhead on every layer
and getting sub-linear scaling. Two independent TP-4 replicas will generally serve more total
tokens per second than one TP-8 deployment — but each request will be slower.

That is the real decision:

\`\`\`
minimize latency        ->  maximize TP (up to the node boundary)
maximize cache capacity ->  TP to the KV head count, then DCP for the rest
maximize throughput/$   ->  minimum TP that fits the model, then replicate
\`\`\`

Most production deployments choose the smallest TP that fits the model with acceptable latency,
then scale out with replicas.`,
      ascii: '',
    },
    {
      name: 'Decode context parallelism: shard the cache by position, not by head',
      keyPoint: 'DCP splits the KV cache along the sequence dimension, so per-GPU cache keeps shrinking past the KV-head count — paid for with one extra collective per layer, which is affordable at decode precisely because the query is one token wide.',
      body: `The ceiling in the last two concepts is not a law of nature. It is a consequence of
*which axis* the cache gets sharded on, and the cache has more than one axis.

Tensor parallelism shards it by **head**, because that is the axis TP is already splitting for the
attention matmuls — it comes free with the strategy. But heads are a small, fixed number set by the
model architect, so the divisor runs out. **Decode context parallelism (DCP) shards the same cache
by token position instead**, and positions do not run out: for a 200k-token request across four
ranks, rank 0 holds the cache for tokens 0–50k, rank 1 holds 50k–100k, and so on. Per-GPU cache
keeps falling as you add GPUs, which is exactly the property TP loses at \`n_kv_heads\`.

**The mechanism, one decode step.** Each rank holds a slice of the positions but needs to attend
over all of them, so:

\`\`\`
AllGather Q  ->  attend locally  ->  AllGather + ReduceScatter
\`\`\`

Every rank gathers the full query, computes attention between it and *its own* KV slice, and
produces a partial output plus the log-sum-exp of its slice's scores. Those partials are then
combined — weighted by their LSE values — and the result is redistributed by head so the rest of
the layer proceeds in its usual TP layout. **The LSE combination is exactly Module 7's online
softmax**, applied across devices instead of across tiles in shared memory: the same
mathematics that lets FlashAttention never materialize the score matrix lets DCP never gather the
K and V.

**Why "decode" specifically.** The operand you move is the query, and at decode the query is one
token per sequence — about 0.5 MB at batch 32 for a \`d_model\` of 8192, the same order as the
all-reduce you are already paying twice a layer. The data you *avoid* moving is the KV cache,
which is gigabytes. You are moving the small operand to the big data, and the ratio between them is
what makes it worth an extra collective. At prefill the query is \`seq\` times larger and that ratio
inverts, which is why Ring Attention — the context-parallel scheme in this module's papers —
makes the opposite choice and passes K/V blocks around a ring instead. Same axis, different
regime, opposite answer about what should move.

**The constraint, and why it differs by architecture.** In vLLM's implementation a GQA model caps
DCP at \`tp_size // num_kv_heads\`, with even divisibility. Read that as: DCP picks up exactly the
TP degree that ran out of heads to split. Llama-3-70B at TP-8 has eight heads for eight ranks and
no use for DCP; at TP-16 it gets DCP-2, and per-GPU cache halves again from 40 to 20 KiB per token,
roughly doubling how many sequences fit.

An MLA model is a different story, and a more urgent one. MLA compresses K and V into a *single*
latent vector per token, so there is no head axis to split at all — every TP rank holds the entire
cache, at every TP degree. DeepSeek-V3's 68.6 KiB per token from Module 11 is 68.6 KiB on rank 0
and 68.6 KiB on rank 15 alike. That is why the MLA constraint is only \`tp_size >= dcp_size\`:
sharding by position is not an optimization there, it is the sole available lever, and it is the
reason the technique showed up when MLA models did.

**What it does not do.** DCP moves no weights, so it does nothing for the model-fitting problem
that PP and TP solve. It does not directly reduce single-stream latency either — what it buys is
cache capacity, which becomes batch size, which becomes throughput, which is the chain Module 6
established. And it adds a collective per layer to the attention path, so it belongs on the same
fast link as TP for the same reason TP does.

**The reported payoff, with the usual caveat.** vLLM's benchmark on an 8×B200 node serving Kimi
K2.6 (an MLA model, NVFP4) against agentic traces with roughly 67k-token median inputs: baseline
TP plateaus at 1,863 tok/s/GPU at concurrency 64, where it hits 100% KV-cache utilization and
cannot admit another sequence; with DCP it reaches 6,091 tok/s/GPU at concurrency 512 while still
sitting at 82% KV usage. Treat the mechanism and the constraint formulas as durable and that 3.3×
as what it is — one model, one node, one trace shape, measured by the people who wrote the
feature. The number your workload gets is a function of how much of your memory the cache was
actually taking, which is the arithmetic the math lab makes you do.`,
      ascii: `  ONE CACHE, TWO AXES TO SHARD IT ON

  BY HEAD (tensor parallelism)        BY POSITION (decode context parallelism)

  rank 0   kv head 0  ████            rank 0   tokens      0- 50k  ████
  rank 1   kv head 1  ████            rank 1   tokens  50k-100k    ████
    ...                                 ...
  rank 7   kv head 7  ████            rank 7   tokens 350k-400k    ████
  rank 8   kv head 0  ████ REPLICA    rank 8   tokens 400k-450k    ████
           ^ only 8 heads exist,               ^ positions do not run out
             so the divisor stops

  one decode step, per layer:

     AllGather Q  ──►  attend to the local slice  ──►  AllGather + ReduceScatter
     one token wide     gigabytes, never moves        combine partials by LSE
     (~0.5 MB @ b=32)                                  (Module 7's online softmax,
                                                        across devices this time)`,
    },
    {
      name: 'Mixture of experts: a bandwidth win and a capacity problem',
      keyPoint: 'MoE activates a fraction of its parameters per token, so it moves far fewer bytes than a dense model of the same size — but every expert must still be resident somewhere.',
      body: `An MoE layer replaces the single MLP with \`E\` expert MLPs and a router that sends each
token to the top \`k\` of them, typically \`k = 2\` out of 8 to 256 experts.

The inference consequence follows straight from these chapters' thesis. **You only read the weights
of the experts you activate.** A model with 8 experts and top-2 routing has roughly 4× the
parameters of its dense equivalent but moves about the same bytes per token.

For decode, which is memory-bound, this is close to free quality. Mixtral-8x7B has ~47B total
parameters but activates ~13B per token: it decodes at roughly the speed of a 13B model while
being substantially better than one. **MoE is the most direct possible exploitation of the fact
that bytes moved, not parameters held, is what costs you.**

The problems are all about capacity and routing.

**Every expert must be resident.** You do not know in advance which experts a token will need, so
all of them must be in memory somewhere. Mixtral-8x7B needs 94 GB at fp16 even though only 26 GB
is read per token. **MoE trades memory capacity for memory bandwidth**, which is a good trade when
capacity is what you have.

**Expert parallelism** distributes experts across GPUs, so each holds \`E/N\` of them. Now routing
requires an **all-to-all**: send each token's hidden state to whichever GPUs hold its chosen
experts, compute, send results back. Two all-to-alls per MoE layer, and all-to-all is the most
demanding collective — every GPU talks to every other.

**Load imbalance is the hard part.** Routing is learned and data-dependent, so nothing guarantees
even distribution. If 40% of tokens in a batch pick expert 3, the GPU holding expert 3 does 40% of
the work while others idle. Since the layer cannot complete until every GPU finishes, **the
slowest GPU sets the pace for all of them.**

Mitigations, none complete:

- **Capacity factor.** Cap tokens per expert; drop or reroute the overflow. Bounds the imbalance
  and costs quality on dropped tokens.
- **Auxiliary load-balancing loss** during training. Helps on average, not per batch.
- **Expert replication.** Duplicate hot experts across GPUs. Costs memory.
- **Larger batches.** Averaging over more tokens smooths the distribution — one case where
  batching helps for a reason unrelated to arithmetic intensity.

**Combining parallelism.** Production MoE deployments typically use expert parallelism for the MoE
layers and tensor parallelism for attention and the dense layers, which means switching
communication patterns within each layer. This is genuinely intricate, and it is why MoE serving
is harder than dense serving even though the arithmetic is friendlier.`,
      ascii: `  DENSE 47B                    MoE 8x7B (top-2)
  read all 94 GB per token     read ~26 GB per token
  -> 28 ms at 3.35 TB/s        -> 7.8 ms at 3.35 TB/s

  but both need 94 GB resident.
  MoE trades CAPACITY for BANDWIDTH.

  expert parallelism, 4 GPUs, 8 experts:
    GPU0 [e0 e1]   GPU1 [e2 e3]   GPU2 [e4 e5]   GPU3 [e6 e7]
              \\        |        /        /
               all-to-all: route tokens to their experts
              /        |        \\        \\
    if 40% of tokens want e3, GPU1 is the bottleneck for everyone`,
    },
    {
      name: 'Prefill/decode disaggregation',
      keyPoint: 'Two workloads with opposite bottlenecks and opposite ideal configurations should not share hardware; the cost is transferring the KV cache between pools.',
      body: `This is the conclusion these chapters have been building toward.

Prefill is compute-bound, latency-critical at the request level, and benefits from high TP degree
and large FLOP capacity. Decode is memory-bound, latency-critical per token, benefits from
bandwidth and cache capacity, and wants large batches. Module 5 showed that co-locating them means
one interferes with the other, and chunked prefill only mitigates it.

**Disaggregation** puts them on separate machine pools:

\`\`\`
  request
     |
     v
  [ PREFILL POOL ]        compute-optimized, high TP, small batches
     |                    produces the prompt's KV cache
     |  transfer KV
     v
  [ DECODE POOL ]         bandwidth-optimized, large batches, big KV budget
     |
     v
  streamed tokens
\`\`\`

What this buys:

**No interference.** A 100k-token prefill cannot stall anyone's token stream, because it is not on
the same GPU.

**Independent tuning.** Each pool gets its own parallelism strategy, batch size and even hardware.
Prefill wants FLOPs; decode wants bandwidth and capacity. You can buy different GPUs for each — an
H200 or a large-memory part for decode, a compute-dense part for prefill.

**Independent scaling.** Traffic mix shifts. A day of long-document summarization is prefill-heavy;
a day of chat is decode-heavy. Separate pools scale separately.

**The cost is the KV transfer.** After prefill you must move the prompt's entire KV cache to the
decode pool. For Llama-3-70B with a 4k prompt that is \`320 KiB × 4096 = 1.25 GiB\`. Over
InfiniBand at ~50 GB/s that is 25 ms; over NVLink where available, ~1.4 ms.

Whether disaggregation wins depends on comparing that transfer against the interference it
removes. DistServe's argument, and it is convincing, is that for most realistic workloads the
transfer is cheaper — especially because it can be **overlapped layer by layer**: start sending
layer 0's KV as soon as it is computed, rather than waiting for the whole prefill to finish. Done
well, most of the transfer hides behind the remaining prefill compute.

Splitwise reached similar conclusions independently, and Mooncake built a production system around
a disaggregated KV store shared across the cluster — which also makes prefix caching global rather
than per-node, a significant additional benefit.

**When it is not worth it:** short prompts (little to transfer, but also little interference),
low load (no contention to eliminate), or a slow interconnect between pools. Like everything in
these chapters, it is a trade you should compute rather than assume.`,
      ascii: '',
    },
    {
      name: 'The interconnect as the real constraint',
      keyPoint: 'Every parallelism decision is a decision about which link carries the traffic, and the links differ by more than 50x.',
      body: `Collect the numbers, because every choice in this module reduces to them:

| link | bandwidth | relative to HBM |
|---|---|---|
| HBM3 (on-package) | 3,350 GB/s | 1× |
| NVLink 4 (H100↔H100) | ~900 GB/s | 0.27× |
| NVSwitch (all-to-all in node) | ~900 GB/s per GPU | 0.27× |
| InfiniBand NDR 400G | ~50 GB/s | 0.015× |
| PCIe Gen5 x16 | ~64 GB/s | 0.019× |
| 100 GbE | ~12 GB/s | 0.004× |

**NVLink is 3.7× slower than HBM. PCIe is 52× slower. Ethernet is 280× slower.**

Now map the strategies onto them:

| strategy | traffic per layer | needs |
|---|---|---|
| tensor parallel | 2 all-reduces of the full activation | NVLink |
| pipeline parallel | 1 activation pass per boundary | anything |
| expert parallel | 2 all-to-alls | NVLink strongly preferred |
| sequence parallel | ring exchange of K/V blocks | NVLink |
| disaggregation | one KV transfer per request | InfiniBand acceptable |

The design rule that falls out:

\`\`\`
within a node (NVLink):    TP, EP, SP -- the chatty strategies
across nodes (IB/Eth):     PP, disaggregation, replication -- the quiet ones
\`\`\`

Violating it is the most common and most expensive mistake in multi-GPU deployment. Running TP-8
across two 4-GPU nodes connected by PCIe means every one of your 160 all-reduces per token crosses
a link 52× slower than HBM. Deployments have been observed running slower on 8 GPUs than on 4 for
exactly this reason.

A few practical notes:

**Check your topology before choosing TP degree.** \`nvidia-smi topo -m\` tells you which GPUs have
NVLink and which are behind PCIe switches. A "4-GPU machine" may be two NVLink pairs connected by
PCIe, in which case TP-4 is a mistake and TP-2 with PP-2 is correct.

**All-reduce cost is not purely bandwidth.** Ring all-reduce has \`2(N−1)\` communication steps, so
latency grows with participant count even at fixed volume. At decode, where messages are small
(under a megabyte), you are often *latency*-bound on the collective rather than bandwidth-bound —
which is why decode suffers proportionally more from high TP degree than prefill does.

**Overlap what you can.** Communication and computation can overlap if the schedule allows it.
Fine-grained overlapping of all-reduce with the next matmul is standard in mature implementations
and is worth a meaningful fraction of the communication cost.`,
      ascii: '',
    },
  ],

  mathLab: {
    prompt: `**Llama-3-70B**: 80 layers, \`d_model = 8192\`, 64 query heads, 8 KV heads,
head_dim 128, 70.6 B parameters. H100 80GB nodes: NVLink 900 GB/s within a node, InfiniBand
50 GB/s between nodes, HBM 3.35 TB/s, dense BF16 989 TFLOP/s per GPU.

**Part 1 — minimum GPU count.**
  a) Weight bytes at fp16, fp8, and int4 (assume 4.5 bits effective).
  b) Minimum H100s to hold the weights alone at each precision.
  c) Now require room for 32 concurrent sequences at 8k context, plus 20 GB workspace total. How
     many GPUs at each precision?

**Part 2 — what TP does.**
For TP ∈ {2, 4, 8, 16} at fp16:
  a) Weights per GPU. Does it fit in 80 GB?
  b) KV cache per token per GPU at 8k context. Careful past TP-8.
  c) Aggregate bandwidth and the TPOT floor.
  d) At which TP does the KV cache per GPU stop improving, and why?
  e) Now add decode context parallelism. vLLM caps DCP for a GQA model at
     \`tp_size // num_kv_heads\`. What is the maximum DCP degree at TP-8, TP-16 and TP-32, and what
     does KV per token per GPU become in each case? At TP-16 with 20 GB of weights and workspace
     per GPU, how many 8k-context sequences fit in an 80 GB card with and without DCP?

**Part 3 — communication cost.**
Decode, batch 32, \`seq = 1\`, \`d_model = 8192\`, fp16. Ring all-reduce moves
\`2 × (N−1)/N × bytes\` per GPU.
  a) Bytes per all-reduce at TP-4 and TP-8.
  b) There are 2 all-reduces per layer, 80 layers. Total communication bytes per token.
  c) Time over NVLink at TP-4 and TP-8.
  d) Time over PCIe Gen5 (64 GB/s) at TP-4.
  e) Compare each against the TPOT floor from Part 2. Express communication as a percentage of
     step time. What is the conclusion?

**Part 4 — prefill contrast.**
Same model, prefill of 2,048 tokens, batch 1, TP-8.
  a) Communication bytes per all-reduce, and total for the pass.
  b) Time over NVLink.
  c) Prefill compute: \`2 × 70.6e9 × 2048\` FLOP across 8 GPUs at 989 TFLOP/s each.
  d) Communication as a percentage of prefill time. Compare to your Part 3 answer and explain the
     difference.

**Part 5 — disaggregation.**
A request has a 4,096-token prompt. Prefill on one pool, decode on another.
  a) KV cache bytes to transfer.
  b) Transfer time over InfiniBand at 50 GB/s, and over NVLink.
  c) If co-locating instead would cost every decoding sequence a 190 ms stall, and there are 40
     sequences decoding, what is the total decode time lost to interference?
  d) Which is cheaper? What if the transfer overlaps with prefill compute?`,

    solution: `**Part 1**

a)
\`\`\`
fp16:  70.6e9 x 2      = 141.2 GB
fp8:   70.6e9 x 1      =  70.6 GB
int4:  70.6e9 x 0.5625 =  39.7 GB    (4.5 bits = 0.5625 bytes)
\`\`\`

b) At 80 GB per H100, usable ~76 GB after CUDA context and fragmentation:
\`\`\`
fp16: 141.2 / 76 = 1.86  ->  2 GPUs
fp8:   70.6 / 76 = 0.93  ->  1 GPU
int4:  39.7 / 76 = 0.52  ->  1 GPU
\`\`\`

c) KV for 32 sequences at 8k:
\`\`\`
327,680 B/token x 8192 x 32 = 85.9 GB
\`\`\`
Total needed:
\`\`\`
fp16: 141.2 + 85.9 + 20 = 247.1 GB  ->  247.1/76 = 3.25  ->  4 GPUs
fp8:   70.6 + 85.9 + 20 = 176.5 GB  ->  2.32          ->  3 GPUs
int4:  39.7 + 85.9 + 20 = 145.6 GB  ->  1.92          ->  2 GPUs
\`\`\`

Note that at int4 the **KV cache is more than twice the weights** (85.9 vs 39.7 GB). Quantizing
weights further would barely help; quantizing the KV cache would. Module 6's two-regime rule, in
a capacity-planning context.

**Part 2**

\`\`\`
TP   weights/GPU   KV/token/GPU   agg BW      fits?   TPOT floor
 2     70.6 GB       160 KiB      6.70 TB/s   yes    141.2/6700  = 21.1 ms
 4     35.3 GB        80 KiB     13.40 TB/s   yes    141.2/13400 = 10.5 ms
 8     17.7 GB        40 KiB     26.80 TB/s   yes    141.2/26800 =  5.3 ms
16      8.8 GB        40 KiB     53.60 TB/s   yes    141.2/53600 =  2.6 ms
\`\`\`

d) **KV per GPU stops improving at TP-8.** The model has 8 KV heads and a KV head cannot be split
across devices without replicating its cache. At TP-8 each GPU owns exactly one KV head
(\`2 × 80 × 1 × 128 × 2 = 40,960 B = 40 KiB\`). At TP-16, two GPUs share each KV head and both must
hold its cache, so per-GPU KV stays at 40 KiB. You doubled the hardware and gained bandwidth and
compute but **no additional cache capacity** — which means no additional batch size, which means
no additional throughput from that axis.

e) DCP picks up precisely where the head divisor gave out:

\`\`\`
TP-8:   8 // 8 = 1   ->  no DCP available    kv/token/GPU = 40 KiB
TP-16: 16 // 8 = 2   ->  DCP-2               kv/token/GPU = 40 / 2 = 20 KiB
TP-32: 32 // 8 = 4   ->  DCP-4               kv/token/GPU = 40 / 4 = 10 KiB
\`\`\`

At TP-16, weights are 8.8 GB per GPU; with 20 GB of weights-plus-workspace budgeted and ~76 GB
usable, about 56 GB is left for cache:

\`\`\`
without DCP:  56 GB / (40 KiB x 8192)  =  56e9 / 327.7e6  =  170 sequences
with DCP-2:   56 GB / (20 KiB x 8192)  =  56e9 / 163.8e6  =  341 sequences
\`\`\`

Twice the concurrent sequences on the same sixteen GPUs, from re-sharding a cache that was already
there. Note what this does *not* change: the TPOT floor, the weight bytes, and the number of
all-reduces the MLP still needs. DCP buys capacity, and capacity becomes throughput only if you
actually raise the batch size to use it.

**Part 3**

a) Activation bytes: \`batch × seq × d_model × 2 = 32 × 1 × 8192 × 2 = 524,288 B = 0.52 MB\`.
\`\`\`
TP-4: 2 x (3/4) x 524,288 = 786,432 B   = 0.79 MB
TP-8: 2 x (7/8) x 524,288 = 917,504 B   = 0.92 MB
\`\`\`

b) 2 all-reduces × 80 layers = 160 per token:
\`\`\`
TP-4: 160 x 786,432   = 125.8 MB per token
TP-8: 160 x 917,504   = 146.8 MB per token
\`\`\`

c) Over NVLink at 900 GB/s:
\`\`\`
TP-4: 0.1258 / 900 = 0.140 ms
TP-8: 0.1468 / 900 = 0.163 ms
\`\`\`

d) Over PCIe at 64 GB/s, TP-4: \`0.1258 / 64 = 1.97 ms\`

e) Against the TPOT floors:
\`\`\`
TP-4, NVLink:  0.140 / 10.5 =  1.3%    negligible
TP-8, NVLink:  0.163 /  5.3 =  3.1%    fine
TP-4, PCIe:    1.97  / 10.5 = 18.8%    serious
TP-8, PCIe:    2.29  /  5.3 = 43.2%    catastrophic
\`\`\`

**Conclusion: tensor parallelism over NVLink costs a few percent; over PCIe it costs a fifth to
nearly half your step time.** And note the trend — the NVLink percentage *rises* with TP degree
(1.3% → 3.1%) because communication grows slightly while step time falls. TP has diminishing
returns even on fast links.

*(These are bandwidth-only estimates. At these small message sizes, collective **latency** —
roughly \`2(N−1)\` hops of a few microseconds each — is often comparable to or larger than the
transfer time, so real overheads are higher than the table suggests.)*

**Part 4**

a) Activation is now \`1 × 2048 × 8192 × 2 = 33.55 MB\`.
\`\`\`
per all-reduce, TP-8: 2 x (7/8) x 33.55 MB = 58.7 MB
total: 160 x 58.7 MB = 9.39 GB
\`\`\`

b) Over NVLink: \`9.39 / 900 = 10.4 ms\`

c) Prefill compute:
\`\`\`
2 x 70.6e9 x 2048 = 2.892e14 FLOP = 289.2 TFLOP
across 8 GPUs at 989 TFLOP/s: 8 x 989 = 7,912 TFLOP/s
time = 289.2 / 7912 = 36.6 ms
\`\`\`

d) \`10.4 / (36.6 + 10.4) = 22.1%\` of prefill time.

Compare to TP-8 decode at 3.1%. **Prefill spends a much larger *fraction* on communication in
absolute terms** — 10.4 ms versus 0.163 ms, a 64× larger absolute cost, because activations scale
with sequence length.

But the ratio tells a subtler story. Prefill communication is 22% of a 47 ms step; decode
communication is 3% of a 5.3 ms step. Prefill moves vastly more bytes but has vastly more
arithmetic to hide them behind — and crucially, prefill's large messages achieve near-peak link
bandwidth, while decode's sub-megabyte messages are latency-dominated and get nowhere near it. So
the honest summary is: **prefill pays more in absolute communication and can overlap most of it;
decode pays little in absolute terms but overlaps poorly.** Good implementations overlap
prefill's all-reduces with subsequent matmuls, cutting that 22% substantially.

**Part 5**

a) \`327,680 B/token × 4096 = 1.342 GB\`

b)
\`\`\`
InfiniBand 50 GB/s: 1.342 / 50 = 26.8 ms
NVLink 900 GB/s:    1.342 / 900 = 1.5 ms
\`\`\`

c) Co-located interference: 40 sequences each stalled 190 ms:
\`\`\`
40 x 190 ms = 7,600 ms = 7.6 seconds of aggregate decode time lost
\`\`\`

Per-user, each of those 40 sees a 190 ms gap in their token stream — roughly 19 missed tokens at
a 10 ms TPOT.

d) **Disaggregation is dramatically cheaper.** 26.8 ms of transfer, paid once by one request,
against 7.6 seconds of aggregate stall inflicted on 40 others. Two orders of magnitude.

And with layer-by-layer overlapping, most of the 26.8 ms disappears: start streaming layer 0's KV
as soon as it is computed, roughly 1/80th of the way into prefill. Since prefill for a 4k prompt
takes tens of milliseconds and the transfer is 26.8 ms spread across it, the exposed cost is
mostly the final layer's transfer — a few hundred microseconds.

This is why DistServe, Splitwise and Mooncake all converged on disaggregation despite the obvious
objection that moving gigabytes between machines sounds expensive. It is expensive. Interference
is more expensive.`,
  },

  codeLab: {
    goal: `You almost certainly do not have a multi-GPU cluster to hand, so this lab is a
**planner**: given a model, hardware, and a topology, it tells you which parallelism configuration
to use and why, with all the arithmetic shown.

Build it properly. This is the tool you would actually use before provisioning anything.`,
    code: `"""
A distributed inference planner. Answers: how many GPUs, what parallelism,
and where does the time go?

    pip install numpy
"""
from dataclasses import dataclass
from itertools import product

GB = 10 ** 9
GiB = 1024 ** 3


@dataclass
class GPU:
    name: str
    memory_gb: float
    hbm_gbs: float
    dense_tflops: float
    per_node: int = 8
    nvlink_gbs: float = 900.0
    internode_gbs: float = 50.0      # InfiniBand NDR


@dataclass
class Model:
    name: str
    n_params: float
    n_layers: int
    d_model: int
    n_heads: int
    n_kv_heads: int
    head_dim: int

    def weight_bytes(self, dtype=2):
        return self.n_params * dtype

    def kv_per_token(self, dtype=2, tp=1, dcp=1):
        """Per GPU. A KV head cannot be split, so the head divisor saturates at
        n_kv_heads -- that is the GQA ceiling. Decode context parallelism adds a
        second divisor that does not saturate, because it shards the same cache
        by token position instead of by head."""
        heads_per_gpu = max(1, self.n_kv_heads // min(tp, self.n_kv_heads))
        return 2 * self.n_layers * heads_per_gpu * self.head_dim * dtype / dcp

    def max_dcp(self, tp):
        """vLLM's constraint for GQA models: DCP picks up exactly the TP degree
        that ran out of KV heads to split."""
        return max(1, tp // self.n_kv_heads)


H100 = GPU("H100 SXM", 80, 3350, 989)
H200 = GPU("H200 SXM", 141, 4800, 989)

L70B = Model("Llama-3-70B", 70.6e9, 80, 8192, 64, 8, 128)
L8B = Model("Llama-3-8B", 8.03e9, 32, 4096, 32, 8, 128)


def allreduce_bytes(batch, seq, d_model, tp, dtype=2):
    """Ring all-reduce: each GPU moves 2*(N-1)/N of the tensor."""
    if tp <= 1:
        return 0
    return 2 * (tp - 1) / tp * batch * seq * d_model * dtype


def plan(model, gpu, tp, pp=1, dcp=1, batch=32, seq=8192,
         w_dtype=2, kv_dtype=2, workspace_gb=10, efficiency=0.75):
    n_gpus = tp * pp
    same_node = n_gpus <= gpu.per_node
    link = gpu.nvlink_gbs if same_node else gpu.internode_gbs

    # --- memory ---
    w_per_gpu = model.weight_bytes(w_dtype) / n_gpus
    kv_per_gpu = model.kv_per_token(kv_dtype, tp, dcp) * seq * batch / pp
    used = w_per_gpu + kv_per_gpu + workspace_gb * GB
    fits = used < gpu.memory_gb * GB * 0.95

    # --- decode time ---
    bytes_read = w_per_gpu + kv_per_gpu
    mem_ms = bytes_read / (gpu.hbm_gbs * GB * efficiency) * 1000

    # TP all-reduces: 2 per layer, on the layers this GPU holds
    layers_here = model.n_layers / pp
    comm_b = allreduce_bytes(batch, 1, model.d_model, tp) * 2 * layers_here
    comm_ms = comm_b / (link * GB) * 1000 if tp > 1 else 0.0

    # DCP: an extra collective per layer -- AllGather Q, then combine the
    # per-shard partials by their log-sum-exp. Modelled here as one all-reduce
    # over the DCP group, which is the right order of magnitude at decode.
    dcp_b = allreduce_bytes(batch, 1, model.d_model, dcp) * layers_here if dcp > 1 else 0
    comm_ms += dcp_b / (link * GB) * 1000

    # PP: one activation hand-off per stage boundary
    pp_b = batch * 1 * model.d_model * 2 * (pp - 1) if pp > 1 else 0
    pp_ms = pp_b / (link * GB) * 1000

    tpot = mem_ms + comm_ms + pp_ms
    free = gpu.memory_gb * GB * 0.95 - w_per_gpu - workspace_gb * GB
    kv_tok = model.kv_per_token(kv_dtype, tp, dcp)
    return {
        "n_gpus": n_gpus, "tp": tp, "pp": pp, "dcp": dcp, "fits": fits,
        "kv_tok": kv_tok, "seats": max(0, free) / (kv_tok * seq), "same_node": same_node,
        "w_gb": w_per_gpu / GB, "kv_gb": kv_per_gpu / GB, "used_gb": used / GB,
        "mem_ms": mem_ms, "comm_ms": comm_ms + pp_ms, "tpot_ms": tpot,
        "comm_pct": 100 * (comm_ms + pp_ms) / tpot if tpot else 0,
        "throughput": batch / (tpot / 1000) if tpot else 0,
        "tput_per_gpu": batch / (tpot / 1000) / n_gpus if tpot else 0,
    }


def min_gpus(model, gpu, batch, seq, w_dtype=2, kv_dtype=2, workspace_gb=10):
    need = (model.weight_bytes(w_dtype)
            + model.kv_per_token(kv_dtype, 1) * seq * batch
            + workspace_gb * GB)
    return int(-(-need // (gpu.memory_gb * GB * 0.95)))


# ==========================================================================
print("=== Part 1: minimum GPUs for Llama-3-70B, batch 32 @ 8k ===")
print(f"  {'precision':<12} {'weights':>10} {'kv':>10} {'total':>10} {'H100s':>7}")
for label, wd, kd in [("fp16/fp16", 2, 2), ("fp8/fp16", 1, 2),
                      ("fp8/fp8", 1, 1), ("int4/fp8", 0.5625, 1)]:
    w = L70B.weight_bytes(wd) / GB
    kv = L70B.kv_per_token(kd, 1) * 8192 * 32 / GB
    n = min_gpus(L70B, H100, 32, 8192, wd, kd)
    print(f"  {label:<12} {w:>9.1f}G {kv:>9.1f}G {w+kv+10:>9.1f}G {n:>7}")

print("\\n=== Part 2/3: what TP degree does (Llama-3-70B, batch 32 @ 8k, fp16) ===")
print(f"  {'TP':>3} {'w/GPU':>8} {'kv/GPU':>8} {'kv/tok':>8} {'mem ms':>8} "
      f"{'comm ms':>8} {'TPOT':>7} {'comm%':>7} {'tok/s':>8} {'/GPU':>7}")
for tp in (1, 2, 4, 8, 16):
    r = plan(L70B, H100, tp=tp)
    kvt = L70B.kv_per_token(2, tp) / 1024
    flag = "" if r["fits"] else "  OOM"
    print(f"  {tp:>3} {r['w_gb']:>7.1f}G {r['kv_gb']:>7.1f}G {kvt:>7.0f}K "
          f"{r['mem_ms']:>8.1f} {r['comm_ms']:>8.2f} {r['tpot_ms']:>7.1f} "
          f"{r['comm_pct']:>6.1f}% {r['throughput']:>8.0f} {r['tput_per_gpu']:>7.0f}{flag}")
print("  note: kv/tok stops improving past TP-8 -- only 8 KV heads to split")

print("\\n=== Part 2b: decode context parallelism lifts the KV-head ceiling ===\\n")
print(f"  {'TP':>3} {'DCP':>4} {'GPUs':>5} {'kv/tok/GPU':>11} {'seqs @ 8k':>10} "
      f"{'TPOT':>8} {'comm%':>7}")
for tp, dcp in ((8, 1), (16, 1), (16, 2), (32, 1), (32, 4)):
    if dcp > L70B.max_dcp(tp):
        continue
    r = plan(L70B, H100, tp=tp, dcp=dcp)
    print(f"  {tp:>3} {dcp:>4} {r['n_gpus']:>5} {r['kv_tok']/1024:>10.0f}K "
          f"{r['seats']:>10.0f} {r['tpot_ms']:>7.1f}ms {r['comm_pct']:>6.1f}%")
print(f"  max DCP for {L70B.name} ({L70B.n_kv_heads} KV heads): "
      f"TP-8 -> {L70B.max_dcp(8)}, TP-16 -> {L70B.max_dcp(16)}, TP-32 -> {L70B.max_dcp(32)}")
print("  note: DCP shards the SAME cache by position, so the divisor keeps going.")
print("  note: TP-16 and TP-32 leave the 8-GPU node, so their comm% is an")
print("        internode figure -- DCP's extra collective wants a fast link too.")

# MLA caches one latent vector per token per layer and cannot split it by head
# at all, so TP replicates it on every rank. DeepSeek-V3: 61 layers, a 576-wide
# latent (512 compressed + 64 rope), fp16 -- the 68.6 KiB/token from Module 11.
MLA_KV_PER_TOKEN = 61 * 576 * 2

print("\\n=== why the DCP rule differs for MLA: there is nothing to split ===\\n")
print(f"  {'TP':>3} {'GQA kv/tok/GPU':>15} {'MLA kv/tok/GPU':>15}   max DCP: GQA / MLA")
for tp in (1, 2, 4, 8, 16):
    print(f"  {tp:>3} {L70B.kv_per_token(2, tp)/1024:>14.0f}K "
          f"{MLA_KV_PER_TOKEN/1024:>14.1f}K        "
          f"{L70B.max_dcp(tp):>3} / {tp:>3}")
print("  GQA: the head divisor works until it runs out of heads, then DCP takes over.")
print("  MLA: the head divisor never worked at all, so DCP is the only lever there is.")

print("\\n=== the PCIe trap: same config, slow link ===")
PCIE = GPU("H100 (PCIe-only)", 80, 3350, 989, per_node=8, nvlink_gbs=64.0)
print(f"  {'TP':>3} {'NVLink TPOT':>13} {'PCIe TPOT':>11} {'penalty':>9}")
for tp in (2, 4, 8):
    a = plan(L70B, H100, tp=tp)
    b = plan(L70B, PCIE, tp=tp)
    print(f"  {tp:>3} {a['tpot_ms']:>12.1f}ms {b['tpot_ms']:>10.1f}ms "
          f"{b['tpot_ms']/a['tpot_ms']:>8.2f}x")

print("\\n=== TP x PP search: minimize latency vs maximize throughput/GPU ===")
print(f"  {'TP':>3} {'PP':>3} {'GPUs':>5} {'node?':>6} {'TPOT':>8} {'tok/s':>8} {'/GPU':>7}")
rows = []
for tp, pp in product((1, 2, 4, 8), (1, 2, 4)):
    r = plan(L70B, H100, tp=tp, pp=pp)
    if r["fits"]:
        rows.append(r)
        print(f"  {tp:>3} {pp:>3} {r['n_gpus']:>5} "
              f"{'yes' if r['same_node'] else 'NO':>6} "
              f"{r['tpot_ms']:>7.1f}ms {r['throughput']:>8.0f} {r['tput_per_gpu']:>7.0f}")

if rows:
    best_lat = min(rows, key=lambda r: r["tpot_ms"])
    best_eff = max(rows, key=lambda r: r["tput_per_gpu"])
    print(f"\\n  lowest latency:      TP={best_lat['tp']} PP={best_lat['pp']} "
          f"-> {best_lat['tpot_ms']:.1f} ms/token")
    print(f"  best throughput/GPU: TP={best_eff['tp']} PP={best_eff['pp']} "
          f"-> {best_eff['tput_per_gpu']:.0f} tok/s/GPU")
    print("  -> these are different configurations. Pick based on your SLO.")

print("\\n=== Part 5: disaggregation vs interference ===")
for prompt in (1024, 4096, 16384, 65536):
    kv = L70B.kv_per_token(2, 1) * prompt
    ib = kv / (50 * GB) * 1000
    nv = kv / (900 * GB) * 1000
    # interference: a blocking prefill stalls every decoding sequence
    prefill_ms = 2 * L70B.n_params * prompt / (989e12 * 8 * 0.75) * 1000
    stalled = 40
    print(f"  prompt {prompt:>6}: KV {kv/GB:>6.2f} GB | transfer IB {ib:>6.1f}ms "
          f"NVLink {nv:>5.1f}ms | co-located stall {prefill_ms:>6.1f}ms "
          f"x {stalled} seqs = {prefill_ms*stalled/1000:>6.2f}s lost")

# --- TODO for you ---
#   1. Add expert parallelism: E experts, top-k routing, all-to-all cost, and a
#      load-imbalance factor. Model Mixtral-8x7B and find the best EP degree.
#   2. Add collective LATENCY (2*(N-1) hops at ~3 us) on top of bandwidth. At
#      decode the messages are small -- does latency dominate? At what TP?
#   3. Model overlapping: assume a fraction f of communication hides behind
#      compute. How large must f be for TP-8 over PCIe to be viable?
`,
    expect: `The precision table shows int4 weights needing only 2 H100s while fp16 needs 4 — and
makes visible that at int4 the KV cache (85.9 GB) is more than twice the weights (39.7 GB), so
further weight quantization is the wrong lever.

The TP table reproduces the math lab: TPOT falling roughly linearly from TP-2 to TP-8, communication
staying at a few percent over NVLink, and — the key line — \`kv/tok\` stopping at 40 KiB past TP-8
because there are only 8 KV heads to distribute. Note also that \`tok/s per GPU\` **falls** as TP
rises: you buy latency with efficiency.

Part 2b is the new lever. At TP-8 the DCP column is empty, because eight KV heads across eight
ranks leaves nothing for DCP to pick up — the constraint \`tp // n_kv_heads\` evaluates to 1. At
TP-16 it engages: \`kv/tok/GPU\` halves from 40 KiB to 20 KiB and the seat count roughly doubles
from 170 to 341 sequences, on the same sixteen GPUs. At TP-32 with DCP-4 it is 10 KiB and 734
seats. Read the \`comm%\` column with the node boundary in mind — TP-16 and TP-32 have already
left the 8-GPU node, so those percentages are internode figures inflated by the same 50 GB/s link
this module keeps warning about. DCP's extra collective is subject to exactly that warning: it
wants a fast link for the same reason TP does.

The MLA table underneath is the sharper version of the argument. The GQA column falls 320 → 160 →
80 → 40 KiB and then stops; the MLA column sits at 68.6 KiB at *every* TP degree, because a single
latent vector per token has no head axis to split and TP simply replicates it on every rank. For a
GQA model DCP is an extra lever available past TP-8; for an MLA model it is the only lever there
has ever been, which is why the technique arrived alongside MLA-based models rather than before
them.

The PCIe comparison is the one to internalize. Expect penalties in the range of **1.2× at TP-2 to
1.5–1.8× at TP-8** on these bandwidth-only estimates — and reality is worse, because small-message
collectives are latency-bound and this model ignores latency entirely.

The TP×PP search should show that lowest latency and best throughput-per-GPU are **different
configurations** — typically high TP for latency, low TP with replication for efficiency. That
divergence is the central planning decision in multi-GPU serving.

The disaggregation table shows transfer cost growing linearly with prompt length while
interference cost grows linearly too — but multiplied by the number of stalled sequences. At 40
concurrent decoders, interference exceeds transfer by roughly two orders of magnitude at every
prompt length.`,
    stretch: `Add collective latency (TODO 2). Model each all-reduce as
\`2(N−1) × 3 µs + bytes/bandwidth\` and re-run the TP table. You should find that at decode, where
messages are under a megabyte, **latency dominates bandwidth** at TP-8 and above — 14 hops × 3 µs
= 42 µs against 0.92 MB / 900 GB/s ≈ 1 µs of transfer. That completely changes the picture and
explains why real TP scaling is worse than bandwidth arithmetic predicts, and why techniques that
reduce the *number* of collectives matter more than those that reduce their size.`,
  },

  papers: [
    {
      title: 'Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism',
      by: 'Shoeybi et al., 2019',
      url: 'https://arxiv.org/abs/1909.08053',
      why: 'The origin of the column-parallel/row-parallel tensor-parallel scheme every framework uses. Written for training, but the forward pass is identical.',
      frame: `**Section 3** is the whole thing. Understand precisely why splitting the first MLP
matrix by columns and the second by rows means the nonlinearity needs no communication — that
choice is the entire contribution, and it is one paragraph. Note the two all-reduces per layer and
carry that number forward.`,
    },
    {
      title: 'Efficiently Scaling Transformer Inference',
      by: 'Pope et al., 2022',
      url: 'https://arxiv.org/abs/2211.05102',
      why: 'The definitive treatment of how partitioning strategy interacts with the prefill/decode split, with an explicit communication cost model.',
      frame: `**Section 3** is now readable with the background you have. Their partitioning
notation takes effort but is worth learning — it expresses exactly which tensor dimension lives on
which device. **Section 3.3** on the different optimal strategies for prefill versus decode is the
payload of this module, derived rigorously.`,
    },
    {
      title: 'DistServe: Disaggregating Prefill and Decoding for Goodput-optimized LLM Serving',
      by: 'Zhong et al., OSDI 2024',
      url: 'https://arxiv.org/abs/2401.09670',
      why: 'The clearest argument that the two phases should run on separate hardware, with the placement algorithm to do it.',
      frame: `Read **Section 3** for the interference quantification and **Sections 4–5** for the
placement algorithm and KV transfer. Pay attention to the argument that transfer cost is
overlappable layer-by-layer — that is what makes the whole approach viable and it is easy to miss.
Read alongside Sarathi-Serve from Module 5: same problem, opposite conclusion.`,
    },
    {
      title: 'Splitwise: Efficient Generative LLM Inference Using Phase Splitting',
      by: 'Patel et al., ISCA 2024',
      url: 'https://arxiv.org/abs/2311.18677',
      why: 'Independent arrival at disaggregation, with a stronger focus on using *different hardware* for each pool — which is the argument that most changes procurement decisions.',
      frame: 'Read the characterization in Section 3: their measurements of how differently the two phases use the GPU are the empirical backing for everything in this module. The cluster-design section is worth reading if you ever influence hardware purchasing.',
    },
    {
      title: 'Mixtral of Experts',
      by: 'Jiang et al., 2024',
      url: 'https://arxiv.org/abs/2401.04088',
      why: 'A production sparse MoE with a clear account of the inference trade: 47B parameters resident, ~13B active per token.',
      frame: 'Read Section 2 for the architecture and routing. The number to extract is the ratio of total to active parameters, and what that means for the memory-capacity versus memory-bandwidth trade. The load-balancing discussion is brief; supplement with the DeepSeek-V3 report for a more detailed treatment of expert parallelism at scale.',
    },
    {
      title: 'Efficient Decode Context Parallelism with vLLM for Long Context Workloads',
      by: 'vLLM team, August 2026',
      url: 'https://vllm.ai/blog/2026-08-07-decode-context-parallelism',
      why: 'The engineering write-up of DCP: sharding the KV cache by position so per-GPU cache keeps shrinking past the KV-head count.',
      frame: `An engineering blog post rather than a paper, so read it for two things and treat the
rest as vendor benchmarking. First, the collective pattern — AllGather Q, attend locally, combine
partials by log-sum-exp — and how little it is, given what it avoids moving. Second, the divisibility
constraints, which encode the whole architectural argument in two lines: \`tp // num_kv_heads\` for
GQA, \`tp >= dcp\` for MLA. The 3.3× throughput headline is one model on one 8×B200 node against
one trace shape; re-derive what your own deployment would get from the capacity arithmetic in this
module's math lab instead of importing it. Read against Ring Attention below, which shards the same
axis and reaches the opposite conclusion about what should move, because it is solving the prefill
case.`,
    },
    {
      title: 'Ring Attention with Blockwise Transformers for Near-Infinite Context',
      by: 'Liu, Zaharia & Abbeel, 2023',
      url: 'https://arxiv.org/abs/2310.01889',
      why: 'Sequence/context parallelism: distribute the sequence dimension across devices, passing K/V blocks around a ring while computing. The third parallelism axis.',
      frame: 'Read Section 3. The key trick is overlapping the ring communication with blockwise attention computation so the transfer is hidden. Relevant when a single sequence is too long for one GPU — the regime long-context serving is heading toward.',
    },
  ],

  checkpoint: {
    claim: `Given a model, a GPU count and a network topology, you can choose a parallelism
strategy and justify it with numbers — and you can explain what TP degree does to TTFT versus what
it does to throughput.`,
    questions: [
      {
        q: 'Why is tensor parallelism within a node and pipeline parallelism across nodes?',
        a: `Because they generate wildly different amounts of traffic. TP does two all-reduces of
the full activation per layer — 160 collectives per token for an 80-layer model, around 126 MB per
token at batch 32 with TP-4. That needs NVLink at 900 GB/s, where it costs about 1% of step time;
over PCIe at 64 GB/s it costs 19% and over Ethernet it is hopeless. PP sends one activation tensor
per stage boundary — about 1.6 MB per token, roughly 80× less — so it tolerates InfiniBand or
Ethernet comfortably. Put the chatty strategy on the fast link and the quiet one on the slow link.
Violating this is the most expensive common mistake in multi-GPU deployment: TP-8 across two nodes
on PCIe can be slower than TP-4 on one.`,
      },
      {
        q: 'What does increasing TP degree do to TTFT, and what does it do to throughput per GPU?',
        a: `TTFT improves, roughly linearly at first, because prefill is compute-bound and you
added compute — and because prefill's large messages achieve near-peak link bandwidth and can be
overlapped with subsequent matmuls. Throughput per GPU *falls*, because you pay communication
overhead on every layer and scaling is sub-linear. Two independent TP-4 replicas will generally
serve more total tokens per second than one TP-8 deployment, while each individual request is
slower. So the rule is: maximize TP if you are optimizing latency, use the minimum TP that fits
the model and then replicate if you are optimizing cost per token.`,
      },
      {
        q: 'Why does a model with 8 KV heads cap useful tensor parallelism at TP-8?',
        a: `TP splits attention by head, and a KV head cannot be split across two GPUs without
replicating its cache on both. At TP-8 each GPU owns exactly one KV head, so per-GPU KV cache is
1/8th of the total — the maximum possible reduction. At TP-16, two GPUs share each KV head and
both must store its cache, so per-GPU KV stays flat at 40 KiB/token for Llama-3-70B. You added
hardware that brings bandwidth and compute but no additional cache capacity, and since cache
capacity is what caps batch size, that axis of scaling has stopped paying. It is a real constraint
on large-model deployment and one reason KV head counts are chosen with parallelism in mind.

What the cap actually says, precisely, is that *this* way of splitting the cache stops paying —
splitting by head. Decode context parallelism splits the same cache by token position instead, and
positions do not run out: at TP-16 a DCP degree of 2 (\`16 // 8\`) takes per-GPU cache from 40 KiB
back down to 20 KiB per token. So the honest form of the claim is that eight KV heads cap the
useful degree of *head-sharded* parallelism at 8, and anything beyond that has to shard a
different axis.`,
      },
      {
        q: 'Decode context parallelism gathers the query on every rank each step. Why is that affordable at decode but not at prefill?',
        a: `Because at decode the query is one token per sequence. At batch 32 with
\`d_model = 8192\` in fp16 that is about 0.5 MB — the same order as the all-reduce the layer
already does twice — while the KV cache it is being matched against is gigabytes and stays exactly
where it is. You are moving the small operand to the big data, and the whole trade is that ratio.
At prefill the query is \`seq\` times larger: for a 2,048-token prompt it is not 0.5 MB but roughly
a gigabyte's worth of traffic per layer, and the ratio inverts — now the queries are the bulk and
the sensible thing is to move K and V instead, which is precisely what Ring Attention does by
passing K/V blocks around a ring and overlapping the transfer with blockwise computation. Same
sharding axis, opposite decision about what travels, decided entirely by which operand is bigger
in that phase. This is also why the technique is named for decode rather than for context.`,
      },
      {
        q: 'Why is MoE good for inference bandwidth but awkward for memory capacity?',
        a: `Because you only read the weights of the experts a token actually activates.
Mixtral-8x7B holds ~47B parameters but activates ~13B per token, so it moves roughly the bytes of
a 13B model while having the quality of something much larger. Since decode is memory-bandwidth
bound, that is close to free quality. The catch is that routing is data-dependent and unknown in
advance, so **every** expert must be resident somewhere — 94 GB at fp16 even though only 26 GB is
read per token. MoE trades memory capacity for memory bandwidth. Expert parallelism spreads the
experts across GPUs but introduces all-to-all communication and load imbalance: if 40% of tokens
route to one expert, that GPU sets the pace for every other.`,
      },
      {
        q: 'A 4k-token prefill would stall 40 decoding sequences for 190 ms. Disaggregating requires transferring 1.34 GB over InfiniBand. Which is cheaper?',
        a: `Disaggregation, by roughly two orders of magnitude. The transfer is
\`1.34 GB / 50 GB/s = 26.8 ms\`, paid once by the one request being moved. Co-location inflicts a
190 ms stall on each of 40 decoding sequences — 7.6 seconds of aggregate lost decode time, and a
visible 190 ms gap in 40 users' token streams. And the transfer can be overlapped layer by layer:
start streaming layer 0's KV as soon as it is computed rather than waiting for prefill to finish,
which hides most of the 26.8 ms behind the remaining prefill compute. This asymmetry is why
DistServe, Splitwise and Mooncake all converged on disaggregation despite the objection that
moving gigabytes between machines sounds expensive.`,
      },
      {
        q: 'Your bandwidth arithmetic says TP-8 should cost 3% overhead, but you measure 15%. What did you forget?',
        a: `Collective latency. A ring all-reduce takes \`2(N−1)\` communication steps — 14 hops at
TP-8 — each with a few microseconds of fixed cost regardless of message size. At decode the
messages are small: 0.92 MB at batch 32, which is about 1 µs of actual transfer at 900 GB/s
against roughly 42 µs of hop latency. **You are latency-bound, not bandwidth-bound**, and a
bandwidth-only model understates the cost by an order of magnitude. This is why decode suffers
disproportionately from high TP degree while prefill — whose messages are hundreds of times larger
— does not, and why reducing the *number* of collectives matters more than reducing their size.
Other candidates: no computation/communication overlap, and kernel launch overhead on the
collectives themselves.`,
      },
    ],
  },

  pitfalls: [
    {
      wrong: 'Running tensor parallelism across nodes to use more GPUs.',
      right: `TP does 160 all-reduces per token for an 80-layer model. Over NVLink that costs a
few percent; over InfiniBand or PCIe it can double or triple your step time. Deployments have
been measured running *slower* on 8 GPUs than on 4 for exactly this reason. Check the topology
with \`nvidia-smi topo -m\` before choosing TP degree, and use pipeline parallelism or replication
to cross node boundaries.`,
    },
    {
      wrong: 'More GPUs always means more throughput.',
      right: `Throughput per GPU generally falls as TP degree rises, because communication
overhead is paid on every layer and scaling is sub-linear. Two TP-4 replicas usually beat one
TP-8 deployment on total tokens per second, while being slower per request. And past the KV head
count, extra GPUs add bandwidth but no cache capacity *under head-sharded TP alone*, so they buy no
additional batch size unless you also shard the cache by position with DCP — which is a
configuration flag you have to actually set, not something that happens because you bought more
GPUs.`,
    },
    {
      wrong: 'Estimating communication cost from bandwidth alone.',
      right: `At decode the messages are under a megabyte, so ring all-reduce latency —
\`2(N−1)\` hops at a few microseconds each — often exceeds the transfer time by an order of
magnitude. A bandwidth-only model will tell you TP-8 costs 3% when it actually costs 15%. Model
latency explicitly, and prefer strategies that reduce the number of collectives rather than their
size.`,
    },
    {
      wrong: 'MoE models are cheap to serve because they only activate a few experts.',
      right: `They are cheap in *bandwidth* and expensive in *capacity*. Every expert must be
resident because routing is data-dependent, so Mixtral-8x7B needs 94 GB of memory to read 26 GB
per token. You also inherit all-to-all communication and load imbalance, where one overloaded
expert's GPU sets the pace for every other. Cheaper per token, harder to deploy.`,
    },
  ],

  glossary: [
    { term: 'decode context parallelism (DCP)', def: 'Sharding the KV cache by token position across ranks so per-GPU cache keeps shrinking past the KV-head count. One extra collective per layer: AllGather the query, attend locally, combine the partials by their log-sum-exp.' },
    { term: 'log-sum-exp (LSE) combination', def: 'Merging attention outputs computed over disjoint slices of the sequence by reweighting them with each slice’s softmax normalizer. FlashAttention’s online softmax, applied across devices instead of across tiles.' },
    { term: 'tensor parallelism (TP)', def: 'Splitting weight matrices within a layer across GPUs. Two all-reduces per transformer layer. Needs NVLink.' },
    { term: 'pipeline parallelism (PP)', def: 'Assigning different layers to different GPUs. One activation hand-off per stage boundary. Tolerates slow links.' },
    { term: 'expert parallelism (EP)', def: 'Distributing MoE experts across GPUs. Requires all-to-all routing and suffers from load imbalance.' },
    { term: 'sequence / context parallelism', def: 'Splitting the sequence dimension across devices, as in Ring Attention. Used when one sequence exceeds a single GPU.' },
    { term: 'column-parallel / row-parallel', def: "Megatron's arrangement: split the first matmul by columns and the second by rows, so the nonlinearity between them needs no communication." },
    { term: 'all-reduce', def: 'Collective that sums a tensor across all GPUs and returns the result to all. Ring implementation moves 2(N-1)/N of the data per GPU.' },
    { term: 'pipeline bubble', def: 'Idle stage time while the pipeline fills and drains. Fraction is (N-1)/(M+N-1) for M microbatches and N stages.' },
    { term: 'capacity factor', def: 'A cap on tokens routed to any single MoE expert, bounding load imbalance at the cost of dropping or rerouting overflow.' },
    { term: 'disaggregation', def: 'Running prefill and decode on separate machine pools, transferring the KV cache between them.' },
    { term: 'NVLink', def: 'High-bandwidth GPU-to-GPU interconnect, ~900 GB/s on H100. Roughly 14x faster than PCIe Gen5.' },
  ],
};
