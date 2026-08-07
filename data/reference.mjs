export default {
  /* ======================================================================
     TIMELINE
     ====================================================================== */
  timelineLede:
    'The papers and systems that actually changed how inference is done — each one an answer to the same question about bytes moved from memory.',

  timelineIntro: `Read this as a single argument rather than a list. In 2017 nobody was thinking
about inference cost, because models were small enough that it did not matter. By 2020 prompts
had got long. By 2022 the models had got big enough that the memory wall became the only thing
that mattered, and every year since has produced another attack on it — architectural, algorithmic,
systems-level, or hardware.

Dates are the arXiv preprint month where one exists, since that is usually when the idea entered
circulation rather than when a venue accepted it.`,

  timeline: [
    {
      date: '2017-06',
      name: 'Attention Is All You Need',
      url: 'https://arxiv.org/abs/1706.03762',
      what: 'The transformer. Encoder-decoder, post-norm, learned positional encodings, and no thought given to generation cost.',
      why: 'Establishes the architecture whose weights you will spend the next decade trying to move less often. Also establishes causal masking, which is what makes a KV cache legal at all.',
    },
    {
      date: '2019-04',
      name: 'The Curious Case of Neural Text Degeneration',
      url: 'https://arxiv.org/abs/1904.09751',
      what: 'Nucleus (top-p) sampling, and the demonstration that maximizing likelihood produces degenerate text.',
      why: 'Kills beam search for open-ended generation — which incidentally removes a k-fold multiplier on KV cache and compute that nobody could have afforded at scale.',
    },
    {
      date: '2019-09',
      name: 'Megatron-LM',
      url: 'https://arxiv.org/abs/1909.08053',
      what: 'Tensor parallelism: split each layer across GPUs with column-parallel then row-parallel matmuls, costing two all-reduces per layer.',
      why: 'The mechanism by which a model too large for one GPU becomes servable. It also multiplies aggregate bandwidth, which is why TP helps decode latency and not just capacity.',
    },
    {
      date: '2019-11',
      name: 'Fast Transformer Decoding: One Write-Head is All You Need',
      url: 'https://arxiv.org/abs/1911.02150',
      what: 'Multi-query attention. One KV head shared by all query heads, shrinking the cache by the head count.',
      why: 'The first paper to name the KV cache as *the* inference bottleneck and attack it architecturally. Four years ahead of the field, and largely ignored until the field caught up.',
    },
    {
      date: '2020-05',
      name: 'GPT-3 / in-context learning',
      url: 'https://arxiv.org/abs/2005.14165',
      what: 'Few-shot prompting as the standard interface, which made prompts 10–100x longer overnight.',
      why: 'Turns prefill from a rounding error into a distinct, compute-bound workload with its own latency budget. Everything about TTFT starts here.',
    },
    {
      date: '2022-05',
      name: 'FlashAttention',
      url: 'https://arxiv.org/abs/2205.14135',
      what: 'Tiling plus online softmax, computing exact attention without ever materializing the N×N score matrix in HBM.',
      why: 'The clearest single demonstration of the thesis: same arithmetic, far fewer bytes moved, 2–4x faster. Memory traffic drops from O(N²) to O(N²d²/M) where M is SRAM size.',
    },
    {
      date: '2022-07',
      name: 'Orca (OSDI ’22)',
      url: 'https://www.usenix.org/conference/osdi22/presentation/yu',
      what: 'Continuous / iteration-level batching: evict finished sequences and admit new ones at every decode step rather than every request.',
      why: 'Turns the batch dial from a static setting into a live one. Since decode arithmetic intensity equals batch size, keeping the batch full is the single largest throughput lever in serving.',
    },
    {
      date: '2022-10',
      name: 'GPTQ',
      url: 'https://arxiv.org/abs/2210.17323',
      what: 'Accurate post-training weight quantization to 3–4 bits using approximate second-order information.',
      why: 'Weights are most of the bytes you move at small batch, so 4-bit weights are close to a 4x decode speedup in the weight-dominated regime. No retraining required.',
    },
    {
      date: '2022-11',
      name: 'Efficiently Scaling Transformer Inference',
      url: 'https://arxiv.org/abs/2211.05102',
      what: 'A rigorous cost model for inference, partitioning strategies derived from it, and the latency/throughput Pareto frontier made explicit.',
      why: 'The paper that made roofline reasoning standard practice for inference. If you internalize one analytical framework, take it from here.',
    },
    {
      date: '2022-11',
      name: 'Speculative decoding',
      url: 'https://arxiv.org/abs/2211.17192',
      what: 'Draft several tokens with a cheap model, verify them in one parallel pass of the target model, and accept via modified rejection sampling.',
      why: 'The only technique that shortens the sequential dependency chain rather than making each link cheaper — and it provably preserves the output distribution exactly.',
    },
    {
      date: '2023-05',
      name: 'GQA',
      url: 'https://arxiv.org/abs/2305.13245',
      what: 'Grouped-query attention: an interpolation between MHA and MQA, plus a recipe for converting existing MHA checkpoints cheaply.',
      why: 'The compromise the whole field adopted. Cuts the KV cache 8x on a 70B model with minimal quality loss, and is what makes long-context serving economically possible.',
    },
    {
      date: '2023-07',
      name: 'FlashAttention-2',
      url: 'https://arxiv.org/abs/2307.08691',
      what: 'Better work partitioning across warps and thread blocks, and fewer non-matmul FLOPs. Roughly 2x over FlashAttention-1.',
      why: 'A reminder that on modern GPUs, non-tensor-core arithmetic is comparatively expensive and scheduling is as important as the algorithm.',
    },
    {
      date: '2023-09',
      name: 'vLLM / PagedAttention (SOSP ’23)',
      url: 'https://arxiv.org/abs/2309.06180',
      what: 'Virtual-memory paging applied to the KV cache: fixed-size blocks, per-sequence block tables, non-contiguous physical storage, copy-on-write sharing.',
      why: 'Measured KV memory utilization in prior systems at 20.4–38.2%. Paging pushes waste below 4%, and the recovered memory converts directly into batch size and therefore throughput.',
    },
    {
      date: '2023-11',
      name: 'Splitwise',
      url: 'https://arxiv.org/abs/2311.18677',
      what: 'Run prefill and decode on separate machine pools, transferring the KV cache between them.',
      why: 'Takes the prefill/decode asymmetry to its logical conclusion: two workloads with opposite bottlenecks should not share a scheduler, let alone a GPU.',
    },
    {
      date: '2023-12',
      name: 'SGLang / RadixAttention',
      url: 'https://arxiv.org/abs/2312.07104',
      what: 'A radix tree over cached KV prefixes, enabling automatic longest-prefix reuse across arbitrary request patterns, plus a frontend language for structured generation.',
      why: 'Turns prefix sharing from a special case (one fixed system prompt) into a general mechanism. Decisive for agentic and multi-turn workloads, where history is re-sent every step.',
    },
    {
      date: '2023-12',
      name: 'Mamba',
      url: 'https://arxiv.org/abs/2312.00752',
      what: 'A selective state-space model with constant-size recurrent state, so there is no KV cache that grows with sequence length.',
      why: 'The alternative bet: rather than compress the cache, remove it. Pure SSMs lost ground on exact recall, but hybrid attention/SSM stacks took the idea mainstream.',
    },
    {
      date: '2024-01',
      name: 'Medusa and EAGLE',
      url: 'https://arxiv.org/abs/2401.15077',
      what: 'Self-speculation: extra decoding heads (Medusa) or feature-level autoregression (EAGLE) generate drafts without a separate draft model.',
      why: 'Removes the main practical objection to speculative decoding — that you need a second well-aligned model with its own memory footprint and its own weights to stream.',
    },
    {
      date: '2024-01',
      name: 'DistServe (OSDI ’24)',
      url: 'https://arxiv.org/abs/2401.09670',
      what: 'Prefill/decode disaggregation with per-phase parallelism, optimizing goodput under explicit TTFT and TPOT SLOs.',
      why: 'Makes goodput — throughput that actually meets its latency target — the metric, and shows that co-locating the two phases costs you more than the KV transfer does.',
    },
    {
      date: '2024-05',
      name: 'DeepSeek-V2 / Multi-head Latent Attention',
      url: 'https://arxiv.org/abs/2405.04434',
      what: 'Compress K and V into a shared low-rank latent vector per token, cached in place of the full per-head keys and values.',
      why: 'The most aggressive architectural attack on the cache to date — a reported ~93% KV reduction versus MHA — while claiming *better* quality than MHA rather than a trade.',
    },
    {
      date: '2024-07',
      name: 'FlashAttention-3',
      url: 'https://arxiv.org/abs/2407.08608',
      what: 'Hopper-specific: warp specialization, asynchronous TMA copies, and interleaved matmul/softmax to hide the non-tensor-core work. Adds FP8 support.',
      why: 'Shows that extracting the last factor of two now requires exploiting specific hardware asynchrony, not just better asymptotics.',
    },
    {
      date: '2025-01',
      name: 'Reasoning models change the workload shape',
      url: 'https://arxiv.org/abs/2501.12948',
      what: 'Long chain-of-thought models that spend thousands of tokens thinking before answering.',
      why: 'Shifts the decode-to-prefill ratio dramatically toward decode — the memory-bound half. A workload that was 60% prefill by FLOPs can become almost entirely decode by wall-clock.',
    },
  ],

  /* ======================================================================
     HARDWARE
     ====================================================================== */
  hardwareLede:
    'Bandwidth, dense FLOPs, and the ridge point for the accelerators you will actually meet. Keep this open from Level 04 onward.',

  hardware: `## How to read this table

Two columns matter more than the rest.

**Bandwidth** sets your decode speed. Time per token at batch 1 is approximately
\`weight_bytes / bandwidth\`, and nothing else in the spec sheet enters that calculation.

**Ridge point** is \`dense FLOP/s ÷ bandwidth\` — the arithmetic intensity at which the machine
stops being memory-bound. Since decode intensity equals your batch size, the ridge point *is* the
batch size you would need to saturate the machine's arithmetic. A lower ridge is easier to
saturate, and therefore better for decode.

All compute figures below are **dense** BF16. Vendors usually headline the 2:4 structured
sparsity number, which is double and does not apply to production LLMs.

## Datacenter GPUs

| GPU | memory | bandwidth | dense BF16 | dense FP8 | ridge point |
|---|---|---|---|---|---|
| A100 40GB SXM | 40 GB HBM2e | 1.56 TB/s | 312 TFLOP/s | — | 200 |
| A100 80GB SXM | 80 GB HBM2e | 2.04 TB/s | 312 TFLOP/s | — | **153** |
| H100 SXM | 80 GB HBM3 | 3.35 TB/s | 989 TFLOP/s | 1979 TFLOP/s | **295** |
| H100 PCIe | 80 GB HBM3 | 2.0 TB/s | 756 TFLOP/s | 1513 TFLOP/s | 378 |
| H200 SXM | 141 GB HBM3e | 4.8 TB/s | 989 TFLOP/s | 1979 TFLOP/s | **206** |
| B200 | 192 GB HBM3e | ~8 TB/s | ~2.25 PFLOP/s | ~4.5 PFLOP/s | ~281 |
| L40S | 48 GB GDDR6 | 0.86 TB/s | 362 TFLOP/s | 733 TFLOP/s | 419 |
| AMD MI300X | 192 GB HBM3 | 5.3 TB/s | ~1.31 PFLOP/s | ~2.61 PFLOP/s | ~247 |

## Consumer

| GPU | memory | bandwidth | dense BF16 | ridge point |
|---|---|---|---|---|
| RTX 4090 | 24 GB GDDR6X | 1.01 TB/s | 165 TFLOP/s | 164 |
| RTX 3090 | 24 GB GDDR6X | 0.94 TB/s | 71 TFLOP/s | 76 |

> **Verify before you rely on these.** Vendors quote compute under varying assumptions and
> revise specifications between silicon revisions. Blackwell and MI300X figures in particular
> should be checked against a current datasheet — they are given here as working estimates, not
> authoritative values. Bandwidth figures are the most stable and the most important.

## The trend that explains the whole field

| | A100 (2020) | H100 (2022) | H200 (2023) | B200 (2024) |
|---|---|---|---|---|
| bandwidth | 2.04 TB/s | 3.35 TB/s | 4.8 TB/s | ~8 TB/s |
| dense BF16 | 312 TF/s | 989 TF/s | 989 TF/s | ~2250 TF/s |
| ridge point | 153 | 295 | 206 | ~281 |

Between the A100 and the H100, compute grew **3.2×** while bandwidth grew **1.6×**. The ridge
point nearly doubled, meaning it takes twice the batch size to keep the machine busy. This is the
memory wall, and it has been widening for a decade — which is why an entire subfield exists to
work around it.

The H200 is the interesting counterexample: identical compute to the H100 with 43% more
bandwidth, which pushes the ridge back *down* to 206. It is a part designed for decode.

## Memory hierarchy (H100, approximate)

The roofline above concerns HBM traffic. Inside the chip there are faster tiers, and exploiting
them is what Level 07 is about.

| level | capacity | bandwidth | latency |
|---|---|---|---|
| registers | 256 KB per SM | ~100 TB/s | ~1 cycle |
| shared memory / L1 | 228 KB per SM | ~20 TB/s | ~30 cycles |
| L2 cache | 50 MB | ~10 TB/s | ~200 cycles |
| HBM3 | 80 GB | 3.35 TB/s | ~500 cycles |
| NVLink to peer GPU | — | 900 GB/s | ~2 µs |
| PCIe Gen5 x16 | — | ~64 GB/s | ~10 µs |
| host DRAM | TBs | ~200 GB/s | ~1 µs |

Shared memory is roughly **6× the bandwidth of HBM** and about 16× lower latency. FlashAttention
is, in one sentence, the observation that if you can keep a tile of the computation in that tier
you never pay for the N×N matrix at all.

Note also the NVLink-to-PCIe gap: **14×**. Tensor parallelism does two all-reduces per layer, so
running TP across a PCIe link rather than NVLink is usually a mistake.

## Quick calculations

\`\`\`
TPOT_floor  = weight_bytes / bandwidth
max_batch   = (memory - weights - workspace) / (kv_per_token * seq_len)
ridge       = dense_flops / bandwidth
saturated?  = max_batch > ridge          # almost always "no"
\`\`\`

\`\`\`
kv_per_token = 2 * n_layers * n_kv_heads * head_dim * dtype_bytes

  Llama-3-8B  fp16:  2 x 32 x 8 x 128 x 2 = 131,072 B = 128 KiB
  Llama-3-70B fp16:  2 x 80 x 8 x 128 x 2 = 327,680 B = 320 KiB
\`\`\``,

  /* ======================================================================
     ENGINES
     ====================================================================== */
  enginesLede:
    'Not a feature table. An argument about what each engine optimizes for, and what it gives up to get there.',

  engines: `Feature tables go stale in weeks and tell you nothing about why a system is shaped
the way it is. What follows is the design philosophy of each engine — the thing that stays true
between releases.

## vLLM — memory management as the organizing principle

**The bet:** if you fix KV cache fragmentation, everything else follows. Recovered memory becomes
batch size, batch size becomes arithmetic intensity, arithmetic intensity becomes throughput.

PagedAttention is the mechanism, and it drove the whole design. Blocks of 16 tokens, per-sequence
block tables, non-contiguous physical storage, copy-on-write for shared prefixes. Continuous
batching sits on top and depends on it — you cannot cheaply admit and evict sequences mid-flight
if each one owns a contiguous reservation.

**What it gives up:** peak single-request latency. vLLM is built around keeping many sequences in
flight, and the scheduler, block manager and Python-side machinery cost you something at batch 1.
It has also historically trailed TensorRT-LLM on raw kernel performance for specific
model/hardware pairs, because it optimizes for generality across a very large model zoo.

**Pick it when:** you are serving many concurrent users, you want broad model support, and you
want to deploy today without a compilation step.

## SGLang — the workload is not one request at a time

**The bet:** real traffic is structured and repetitive. Requests share system prompts, few-shot
examples, documents, and conversation history. An engine that treats each request as independent
is leaving most of the work on the table.

RadixAttention is the mechanism: a radix tree over cached KV prefixes with automatic
longest-prefix matching and LRU eviction. Where vLLM's prefix caching handles the common case,
SGLang treats sharing as the *default* structure of the workload. Its frontend language makes
that explicit, letting you express branching, parallel calls and constrained generation as a
program rather than a sequence of opaque API calls. Structured output is a first-class concern
rather than a wrapper.

**What it gives up:** simplicity, and some generality. The programming model is more opinionated,
and the benefits are largest for workloads that genuinely share prefixes.

**Pick it when:** agentic loops, multi-turn chat with long histories, RAG over a shared corpus, or
heavy structured/JSON output — anywhere the prefix reuse rate is high.

## TensorRT-LLM — compile ahead of time, run with no surprises

**The bet:** the last 20–30% of performance lives in kernel selection, fusion and layout decisions
that are only correct for a specific model on specific hardware at a specific batch size. So make
those decisions ahead of time, at build time, with full knowledge of the deployment.

You compile a model into an engine artifact for a target GPU, precision, batch size range and
sequence length range. The result is typically the fastest option on NVIDIA hardware, with the
tightest and most predictable latency.

**What it gives up:** flexibility, and a great deal of operational convenience. The engine is tied
to the GPU it was built for. Changing max batch size or context length means recompiling. Build
times are long, and new model architectures take longer to land than in the Python-first engines.

**Pick it when:** a fixed model on fixed NVIDIA hardware at scale, where a 25% efficiency gain is
worth a compilation pipeline in your deploy process.

## llama.cpp — the weight-dominated regime, taken seriously

**The bet:** most people running a model locally have one user, a short context, and not enough
memory. That is the weight-dominated regime from Level 02, where weight bytes are essentially the
entire cost of a decode step — so weight quantization is not one optimization among many, it is
*the* optimization.

Hence the k-quant family and an obsessive focus on getting good quality out of 2–5 bits per
weight. Hence also CPU inference, GPU offloading of a subset of layers, Metal and Vulkan
backends, and a zero-dependency C++ build. Different regime, different correct answers.

**What it gives up:** throughput at scale. Continuous batching and paged KV arrived late and are
not the design centre. Serving hundreds of concurrent users is not what it is for.

**Pick it when:** local or edge inference, consumer hardware, one or few users, or you need the
model to fit in memory it does not obviously fit in.

## How to choose, in one table

| your situation | engine |
|---|---|
| many users, many models, deploy today | vLLM |
| high prefix reuse: agents, chat, RAG | SGLang |
| heavy structured / JSON output | SGLang |
| one model, NVIDIA, squeeze the last 25% | TensorRT-LLM |
| strict and predictable p99 latency | TensorRT-LLM |
| laptop, desktop, edge, single user | llama.cpp |
| CPU-only or partial GPU offload | llama.cpp |

## What they all agree on

The convergence is as informative as the differences. Every serious engine now implements
continuous batching, paged KV cache, some form of prefix caching, chunked prefill, CUDA graphs,
fused kernels, FlashAttention-family kernels, quantization (weights, and increasingly KV), and
speculative decoding.

Those are no longer differentiators. They are table stakes, and the fact that four independently
designed systems converged on the same list is decent evidence that the list is correct.

> Version numbers and specific feature availability move fast. Treat the philosophies above as
> durable and check current documentation for anything you are about to depend on.`,

  /* ======================================================================
     FRONTIER
     ====================================================================== */
  frontierLede:
    'What is genuinely unsolved. Read it last, or read it first to know where you are heading.',

  frontier: `Every problem below is open in the sense that people are actively publishing on it
and no answer has clearly won. Where I name a system or paper, treat it as a pointer to a line of
work rather than a settled result.

## 1. Long context is still fundamentally unaffordable

The arithmetic from Level 02 does not improve with cleverness. Llama-3-70B at 128k context needs
40 GiB of KV cache **per sequence**. On four H100s that is three concurrent users.

The attacks all trade something real:

- **Eviction** — H2O, SnapKV, StreamingLLM's attention sinks. Drop tokens judged unimportant.
  Fast and simple, but you cannot know in advance which token a future query will need, so it is
  lossy in a way that is hard to characterize and worse under adversarial or retrieval-heavy use.
- **Compression** — quantize the cache, or project it to a lower rank. Reliable to 8 bits, harder
  below. Quality loss concentrates in exactly the long-context tasks you bought the context for.
- **Architecture** — MLA compresses to a latent; sliding-window and sparse attention bound what
  must be kept. Requires training the model that way; no help for existing checkpoints.
- **Offload** — push cold cache to CPU DRAM or NVMe. PCIe at ~64 GB/s against HBM at 3.35 TB/s is
  a 50× cliff, so this only works when access is predictable.

Nobody has an approach that is simultaneously lossless, architecture-agnostic and cheap. The
honest state of the art is a menu of trades.

## 2. Reasoning models inverted the workload

Long chain-of-thought models spend thousands of tokens thinking before answering. A request that
was 2,000 prompt tokens and 200 output tokens becomes 2,000 and 20,000.

That change is not incremental — it moves nearly all the wall-clock into decode, the memory-bound
half. It also means the KV cache grows for far longer per request, so sequences occupy batch slots
much longer, and it makes TTFT nearly irrelevant next to total time.

Open questions: can you batch reasoning traces more aggressively given they are less
latency-sensitive? Should thinking tokens use a cheaper decode path, or a smaller model, or
aggressive speculation given they are never shown to the user? Can you prune a reasoning trace
mid-flight when it is clearly going nowhere? There is no consensus on any of these.

## 3. Speculative decoding at high batch

Speculation converts spare compute into throughput, which works beautifully at batch 1 and stops
working as batch grows — because at large batch you are approaching the ridge point and no longer
have spare compute to convert. Worse, a rejected draft wastes work that a non-speculating batch
would have spent usefully.

The open problem is adaptive speculation: choosing draft length per request, per step, based on
current batch occupancy and observed acceptance rate. Some systems do simple versions. Doing it
well, with a scheduler that understands the interaction between speculation and batching, is
unsolved.

## 4. Agentic and multi-turn workloads

An agent re-sends its whole growing trajectory every step. Without prefix caching, total prefill
work is quadratic in the number of steps. With it, the cache becomes the working set of the
system and the interesting questions are all about cache policy:

- What do you evict when the cache holds hundreds of partially-shared trajectories?
- Should you keep a trajectory's cache warm while the agent waits on a slow external tool call —
  paying memory for latency you do not control?
- How do you schedule fairly when one agent's cache is worth 50× another's?

This is a caching and scheduling problem more than a kernels problem, and it is comparatively
under-explored.

## 5. Evaluating optimized models honestly

We are still not good at this. Perplexity is insufficient — a 4-bit model can match fp16
perplexity and fail badly on multi-step reasoning or long-context retrieval. Quality loss from
quantization is not uniform: it concentrates in rare tokens, long contexts, and exactly the
capabilities that are hardest to measure.

There is no standard, trusted evaluation suite for "is this quantized model still the model I
tested?" — which means a great deal of production quantization is deployed on faith and
spot-checks. This is arguably the most important unglamorous problem in the list.

## 6. Disaggregation at scale

Prefill/decode disaggregation is clearly right in principle and awkward in practice. You must
move the KV cache between pools, and for a long prompt that is gigabytes over an interconnect.
Open: how to place and schedule across heterogeneous pools, how to size the two pools as traffic
shifts, and whether the transfer can be overlapped well enough to disappear.

## 7. Hardware and the widening wall

Compute has grown far faster than bandwidth for a decade, and the ridge point has drifted from
153 on an A100 to 295 on an H100. Each generation makes decode relatively worse.

FP4 and FP8 help by shrinking bytes. Larger HBM stacks help capacity. But the structural fix
would be a memory-centric architecture — much more bandwidth per FLOP, or compute placed nearer
the memory. Several companies are betting on variations of this. Whether any of it displaces
GPUs at scale is genuinely uncertain, and it is the question that would most change everything
else in this course.

## 8. The economics

Inference is now the dominant cost of operating a model, and the field has no shared vocabulary
for cost-quality trade-offs. What is the right price for a token that took 20,000 thinking tokens
to produce? How should a provider expose the latency/throughput dial to customers who do not know
they want it? These are not technical questions, but they increasingly determine which technical
work gets done.

---

> **A note on currency.** This page reflects the state of things as of the author's knowledge and
> was not verified against live sources at build time. Inference moves quickly, particularly on
> points 2, 3 and 7. Before relying on any specific claim here, search for recent work — and if
> something below has obviously been solved since, that is a good sign the field is healthy.`,

  /* ======================================================================
     HOW TO USE
     ====================================================================== */
  howToLede:
    'Eleven levels, roughly three months at a few hours a week. Levels 00 to 03 are the foundation — do not rush them.',

  howTo: `## The one idea that organizes everything

Generating a token requires reading **every** model weight from memory. On modern hardware,
moving those bytes takes far longer than the arithmetic performed on them. So decoding a single
sequence leaves the GPU almost entirely idle.

Nearly every technique in this course is an answer to one question: **how do we do more useful
work per byte moved from memory?** Batching, quantization, GQA, speculative decoding,
PagedAttention — different attacks on the same problem. Keep this in view and the field stops
looking like a pile of tricks.

## How each level is built

Every level has the same seven parts, and they are colour-coded so you can navigate without
reading:

| marker | part | what to do with it |
|---|---|---|
| **Pinky** (pink) | The big idea | Read first. Intuition before mechanism. |
| **Pellets** (peach) | Concepts | The substance. Read in order. |
| **Inky** (cyan) | Math by hand | **Do not skip.** Work it on paper before opening the solution. |
| **Clyde** (orange) | Code lab | Small, runnable, CPU-friendly. Type it, do not read it. |
| **The key** (yellow) | Papers | Each with a reading frame: what to extract, what to skim. |
| **Blinky** (red) | Pitfalls | The specific things people get wrong. |
| **Power pellet** | Checkpoint | Answer out loud before revealing. Honest self-assessment. |

The ghost roles are not arbitrary — each one matches how that ghost actually behaves in the
arcade game. Pinky targets four tiles ahead of you, so she is the idea out in front. Inky's
target is *computed* from Blinky's position, so he is the derived arithmetic. Blinky chases you
directly, so he is the misconception coming for you.

## The method

**1. Concept pass.** Read the big idea and the concepts straight through. Do not stop to look
things up; get the shape first.

**2. Math by hand.** Work every math lab on paper. This is the non-negotiable part. The
arithmetic is where the understanding actually lives, and reading a worked solution feels like
learning without being learning. Get a number, *then* check it.

**3. Code lab.** Type the code. Run it. Change something and predict what happens before you
re-run. The TODO items at the bottom of each lab are the real exercise.

**4. Papers.** Use the reading frame — most of these papers have a payload of two or three
sections and a lot of context you can skim. Read those sections properly.

**5. Checkpoint.** Answer each question out loud, in full sentences, before revealing the answer.
If you cannot, go back. Marking a level cleared is for you, not for anyone else, so be honest
about it.

## Pacing

| levels | why |
|---|---|
| **00–03** | The foundation. Take your time. Everything later assumes you can trace a forward pass and compute a KV cache without notes. |
| **04** | The analytical core. If you only truly master one level, make it this one — the rest of the course is applied roofline reasoning. |
| **05–07** | Systems and kernels. Faster going, because you now have the framework. |
| **08–10** | The frontier. More reading, less arithmetic. |

At a few hours a week this is roughly three months. There is no benefit to going faster; the
checkpoints are honest gates and the material compounds.

## What you need

- Python and NumPy. PyTorch for a few labs.
- No GPU required. Every code lab runs on CPU; a couple are more interesting on a free Colab GPU
  and say so.
- A pen and paper. Genuinely — the math labs do not work typed.

## Progress

Marking a level cleared stores it in your browser's local storage. Nothing is sent anywhere,
there is no account, and clearing your browser data resets it. The score is cosmetic: 2,200
points a level, 24,200 for all eleven, which is the Pac-Man perfect-game score.

Arrow keys move between levels.

## A note on currency

The foundational papers here are stable and will not go out of date. The systems material,
hardware numbers and frontier discussion will. Where a claim matters to a decision you are
making, verify it — particularly hardware specifications, which change between silicon
revisions, and anything about which serving engine currently does what.

The habit worth building is not trusting this site. It is being able to re-derive its numbers
yourself, at which point you no longer need it.`,
};
