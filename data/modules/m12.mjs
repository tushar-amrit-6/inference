export default {
  n: 12,
  slug: 'how-vllm-works',
  title: 'HOW VLLM WORKS',
  tagline: 'The serving-engines reference gave you the philosophy. This level is the mechanism: the block table, the scheduler loop, and the arithmetic behind preemption.',
  hours: '6–8 hours',
  prereqs: ['Module 5', 'Module 6', 'Module 9'],

  bigIdea: `Every serving engine needs exactly two things: an allocator that does not waste memory,
and a scheduler that does not waste time. The reference page on serving engines told you vLLM's
answer to both in one sentence — PagedAttention for the first, continuous batching for the second —
and that they are not really two techniques. **The scheduler cannot do its job without the
allocator, and the allocator has no reason to exist without the scheduler.**

A scheduler that adds and evicts sequences every iteration needs allocation and freeing to be
cheap and fine-grained, or the bookkeeping eats the throughput it was trying to buy. An allocator
that hands out fixed-size blocks on demand only pays for that flexibility if something is actually
admitting and evicting sequences fast enough to use it. Module 6 gave you the first half — the
block table, and the fragmentation arithmetic it fixes. Module 5 gave you the second — the
running/waiting model, and why iteration-level scheduling beats static batching. This level is
where they turn out to be one system, and goes to the places neither module had room for: what the
scheduler actually does when the block pool runs dry, what sharing a block costs and who pays for
it, and what changes when the scheduler is handing work to eight GPUs instead of one.

By the end you should be able to read vLLM's own scheduler and block-manager source and recognise
every piece of it — because you will have built a working, if miniature, copy of both yourself.`,

  concepts: [
    {
      name: 'The block table: PagedAttention as OS-style paging',
      keyPoint: 'A block table is a small array of physical block ids; the attention kernel reads through it instead of assuming one contiguous buffer per sequence.',
      body: `Module 6 derived how much memory paging recovers. This concept is about the data
structure that recovers it.

The KV cache is never reserved for a sequence's maximum possible length. Instead it is carved into
fixed-size **blocks** — 16 tokens each, in vLLM's original design — drawn from a single pool
shared by every sequence resident on the GPU. Each sequence owns a **block table**: a small array
mapping its logical block indices (0, 1, 2, …, in the order the sequence filled them) to physical
block indices anywhere in that shared pool. That is exactly the indirection an operating system
uses for virtual-memory pages, mapped onto a much smaller address space — a sequence at 8k context
needs 512 logical blocks, not the millions of pages a real OS manages, so the whole table for one
sequence is a couple of kilobytes (the math lab makes this precise).

A sequence claims a new physical block only when its current one fills — after its 17th token, not
before. Allocation tracks *actual* usage instead of a worst-case reservation, which is the entire
mechanism behind Module 6's fragmentation numbers: nothing is wasted because nothing is reserved
ahead of need.

The attention kernel has to cooperate with this or the indirection is pointless. Given a block
table and a **slot mapping** — which physical slot each token in the current step's batch belongs
to — a paged-attention kernel gathers the right K and V vectors from scattered physical memory
instead of assuming one contiguous run per sequence. This is the detail Module 7's kernel material
calls out: paging changes the *memory access pattern* attention has to support, not just the
allocator sitting above it. A kernel written for a contiguous cache cannot serve a paged one; the
two were co-designed.

One consequence worth stating plainly, because it explains why the mechanism is boring rather than
clever: **nothing here is lossy.** Every byte in the cache is still exactly the K and V vectors the
model would have produced with a contiguous buffer. Paging is pure bookkeeping — it changes *where*
bytes live, never *what* they are — which is why Module 6 could call it free.`,
      ascii: `  LOGICAL (per sequence)              PHYSICAL POOL (shared)

  seq A block table                    ┌───┬───┬───┬───┬───┬───┬───┬───┐
  L0 ──────────────────┐               │ P0│ P1│ P2│ P3│ P4│ P5│ P6│ P7│
  L1 ──────────┐        └────────────► │   │   │███│   │   │███│   │███│
  L2 ──┐        │                      └───┴───┴───┴───┴───┴───┴───┴───┘
       │        └───────────────────►                 ▲       ▲       ▲
       └────────────────────────────────────────────► L2      L1      L0
                                                     (16 tokens each, non-contiguous)`,
    },
    {
      name: 'The scheduler: continuous batching as a loop, not a batch',
      keyPoint: 'The scheduler runs once per token, not once per request; a sequence can join or leave the batch on any step instead of waiting for a fixed-size batch to finish.',
      body: `Module 5 established why continuous batching beats static batching: intensity equals
batch size, so an idle slot is wasted arithmetic, and a static batch runs at the speed of its
slowest member. This concept is the loop that implements that idea inside vLLM specifically.

The scheduler runs once per **iteration**, and one iteration is one forward pass — not one
request processed to completion. Every sequence vLLM is tracking sits in one of three states:
**running** (it will advance by one token this iteration, if it can), **waiting** (admitted to the
system but not yet given a first block), or **swapped** (preempted, its cache moved to host memory
rather than dropped — the alternative to the recomputation policy the next concept covers). Each
step, the scheduler decides who moves between those states before anyone's forward pass runs.

Admission is deliberately cheap: a waiting sequence only needs *one* free block to start, not its
eventual total, because paging means "start" and "reserve everything you might need" are different
operations for the first time. That is what makes iteration-level scheduling affordable — Orca
established the scheduling idea in 2022, a year before PagedAttention, using a coarser memory
model; vLLM's contribution was making the memory side cheap enough that the scheduling side could
run this fine-grained.

Chunked prefill (Module 5's Sarathi-Serve material) is not a separate scheduling mode bolted on
afterward — it is one more admission rule inside the same loop. Instead of running a long prompt's
prefill as one uninterrupted unit that blocks every decode step behind it, the scheduler caps how
many prefill tokens it will admit in a single iteration and carries the rest over to the next one,
interleaving those chunks with whatever decode work is already running. The loop does not change;
what counts as "one iteration's work" gets more flexible.

The property worth holding onto, because the next two concepts are both about what threatens it:
**a sequence can enter or leave the batch on any single step.** Nothing waits for a fixed-size
batch to drain. That is only true as long as the scheduler can always find a free block for
whoever needs one next — which is exactly the condition the next concept is about losing.`,
      ascii: `  ONE ITERATION = ONE FORWARD PASS

  WAITING  ──admit, if a block is free──►  RUNNING  ──finishes──►  done, blocks freed
     ▲                                        │  │
     │                                        │  └──preempt, if the pool is empty──┐
     └───────────────re-admit later◄──────────┘                                    ▼
                                                                                 SWAPPED
                                                                          (or dropped and
                                                                        recomputed — next)`,
    },
    {
      name: 'Preemption: what happens when the pool runs dry',
      keyPoint: 'When no block is free, the scheduler preempts a running sequence by swapping its blocks to host memory or by dropping them and recomputing later; which is cheaper is not simply a function of sequence length.',
      body: `Every promise the last concept made — cheap admission, a sequence joining or leaving on
any step — depends on a free block being available when a running sequence needs one. Sometimes
none is. The scheduler then **preempts** the lowest-priority running sequence rather than failing
the request in front of it.

There are two ways to do it. **Swap** copies the victim's blocks to host DRAM over PCIe and copies
them back when the sequence is re-admitted — the cache survives, at the cost of a round trip over a
link roughly 50 to 60 times slower than HBM. **Recompute** simply drops the blocks and reruns
prefill over the sequence's tokens-so-far once it is rescheduled — no host memory required, at the
cost of redoing work the GPU had already done once.

The intuition most people reach for is "recompute is cheap for short sequences, swap wins once a
sequence is long enough that its cache is expensive to redo." The math lab in this level does the
arithmetic and that intuition turns out to be *incomplete*: under the same linear-in-length
approximation this course uses for prefill FLOPs everywhere else, both costs scale with sequence
length, which means their **ratio is constant** — for Llama-3-8B on an H100 with a real PCIe link,
swap comes out cheaper than recompute at every length, not just long ones. What actually produces a
length-dependent crossover is something the simple bytes-and-FLOPs model leaves out entirely: a
**fixed per-transfer overhead** on the swap path — DMA setup, driver bookkeeping, synchronization —
that does not shrink with sequence length. Work through the math lab and you will find that
overhead only has to be small for the crossover to sit at a handful of tokens, which reframes the
folklore rather than simply confirming it.

Victim selection is its own small policy question, independent of swap-versus-recompute. FCFS —
preempt whoever was admitted most recently — is simple, requires no bookkeeping about how close
anyone is to finishing, and is close to what vLLM defaults to. It is not obviously optimal: a
sequence one token from completion is a strange thing to throw away over a sequence that just
started, and the code lab's stretch goal asks you to try a policy that accounts for it.

One thing every version of preemption costs that neither policy avoids: **it is visible to the
preempted request as a latency spike**, not a throughput loss. The GPU is still working the whole
time; the victim's user is the one waiting.`,
      ascii: '',
    },
    {
      name: 'Copy-on-write and reference counting: sharing without copying',
      keyPoint: 'Two sequences that share tokens can point their block tables at the same physical block, tracked by a reference count; a write forks only the one block being written to.',
      body: `Parallel sampling (several completions from one prompt) and beam search both start
several sequences from an identical prefix. Duplicating that prefix's cache once per sequence would
waste exactly the kind of memory Module 6 spent an entire level recovering.

Because the block table is already an indirection, sharing costs nothing extra to *represent* —
two sequences' block tables can simply point at the same physical block. What they need beyond that
is a **reference count** per physical block, so the allocator knows a block is still in use by
someone even after one of its sharers releases it.

Sharing only works cleanly as long as nobody writes to a shared block. The moment two sequences'
continuations diverge — sequence A generates one token, sequence B generates a different one, and
the position that used to hold "the next slot in the shared prefix" now needs to hold two different
things — the block that both were about to write into gets **copied** first, for whichever sequence
is doing the writing, and the two block tables split there. This is copy-on-write, the identical
idea operating systems use for \`fork()\`: shared as long as possible, copied at the last possible
moment, and only the one block that actually diverged, not the whole cache either sequence holds.

The cost model this produces is worth stating precisely, because it is easy to overestimate. Two
sequences sharing a 500-token prompt (32 blocks at the standard block size) pay for 32 shared
blocks plus exactly one new block each, the moment they first diverge — not 64 blocks, and not a
full duplicate cache re-copied on every subsequent token. Copy-on-write is a one-time fork per
divergence point, not an ongoing tax.`,
      ascii: '',
    },
    {
      name: 'Automatic prefix caching: sharing without being asked',
      keyPoint: 'Blocks are hashed on their token contents chained with the previous block’s hash, so identical prefixes across unrelated requests resolve to the same physical blocks without anyone requesting it.',
      body: `Copy-on-write shares a prefix between sequences that started life sharing it — one
prompt, several samples. It does nothing for two *unrelated* requests that happen to carry the
same 2,000-token system prompt, because as far as the block manager is concerned they are unrelated
requests with unrelated block tables. **Automatic prefix caching** is the mechanism that catches
that case too.

Each physical block is hashed on its own token contents, chained with the hash of the block before
it — so the hash of block *k* commits to every token from the start of the sequence through the end
of block *k*, not just the 16 tokens inside it. A new request's first block is hashed and looked up
in a table of existing blocks; on a hit, its block table points at the existing physical block
(reference-counted, exactly as in copy-on-write) instead of allocating and writing a new one. The
lookup then continues into the request's second block using the chained hash, and keeps matching
for as long as the new request's prefix keeps agreeing with something already cached — which is why
the mechanism only recognises a shared prefix down to a **block boundary**: a match that starts one
token off from where an existing block began is not a match at all under this scheme, hash chains
included.

This is the same problem SGLang's RadixAttention solves, with a different structure and a different
trade-off. A radix tree matches at the token, not the block, and evicts by tree structure rather
than a flat LRU list — more precise, and more machinery. A hash chain over fixed blocks is coarser
but is the same block manager every other feature in this level already needs; there is no second
data structure to keep consistent with the first. Neither is strictly better; they are the two
answers you get from optimizing the same idea against two different constraints, which is exactly
the contrast the serving-engines reference page draws between the two projects.

The payoff shows up as time, not memory. A cache hit is prefill work the GPU never does — the math
lab prices exactly this for a system prompt shared across fifty concurrent requests, and it shows
up as a drop in time-to-first-token, not a change in how many sequences fit.`,
      ascii: '',
    },
    {
      name: 'Multi-GPU: one scheduler, many workers',
      keyPoint: 'A single driver process holds the scheduler and the block tables and broadcasts each step’s batch to one worker per GPU; the block manager itself stays single-writer.',
      body: `Module 9 covered tensor and pipeline parallelism as a way of splitting a model too
large for one GPU across several. What it did not cover is who runs the scheduler once the model
is split — a question this level is now equipped to answer, because the scheduler this level has
been describing is a single piece of sequential logic making yes/no decisions about admission and
preemption, and that logic cannot simply be duplicated once per GPU without every replica
disagreeing about which blocks are free.

vLLM's answer is a **driver process plus worker processes**. The driver owns the scheduler and the
authoritative block tables — the same structures this level has been describing all along, just
now living in exactly one place instead of one per GPU. Each iteration, it decides the batch the
way a single-GPU deployment would, then broadcasts that decision — which sequences, which token
positions, which block tables — to one worker process per GPU. Each worker holds its shard of the
model's weights and runs its slice of the forward pass; under tensor parallelism the workers
exchange partial activations through NCCL collectives at the points Module 9 identified, once per
attention block and once per MLP block.

The block manager staying single-writer is the detail that makes this whole level's machinery port
to multiple GPUs without a redesign. Workers do not negotiate allocation, do not run their own
copy of the scheduler, and do not need to agree with each other about which physical block holds
which sequence — they simply execute against the block table and slot mapping the driver already
computed for them. Every mechanism above — admission, preemption, copy-on-write, prefix-cache
lookup — is a decision made exactly once per iteration, by exactly one process, regardless of how
many GPUs are executing the result.

What this buys you is that nothing in this level's first five concepts needed to know a second GPU
existed. What it costs is a broadcast every iteration and a driver process that is now a
sequential bottleneck other systems can choose to remove — which is precisely DistServe's argument
in this level's papers, and the reason it is here as the closing citation rather than a footnote.`,
      ascii: '',
    },
  ],

  mathLab: {
    prompt: `All of this reuses figures already established elsewhere in the course rather than
inventing new ones — Llama-3-8B's 8.03e9 active parameters and its 131,072 B (128 KiB) per-token
cache, the ~50 GB/s achieved PCIe bandwidth from Module 5's swap example, and the 16-token block
size from Module 6. One new number: back out the achieved H100 compute throughput implied by
Module 5's own worked example — "32k tokens on Llama-3-8B is roughly 2 × 8.03e9 × 32000 = 514
TFLOP, about 740 ms at realistic H100 throughput" — and use *that* derived figure everywhere below,
so this level's arithmetic is a strict continuation of Module 5's rather than a second, disagreeing
estimate.

\`\`\`
recompute_time(S)  ~=  2 x N_active x S / C            # C = achieved compute, FLOP/s
swap_time(S)       ~=  2 x kv_per_token x S / BW_pcie   # round trip: out, then back
\`\`\`

1. Derive \`C\`, the achieved H100 compute throughput, from Module 5's own numbers.
2. Module 5's checkpoint gives a worked example: "a 2.5 GiB cache at ~50 GB/s is roughly 50 ms out
   and 50 ms back." What sequence length \`S\` does 2.5 GiB correspond to for Llama-3-8B, and what
   are the precise (unrounded) single-leg and round-trip swap times?
3. Compute the recompute time at that same \`S\`. Which policy actually wins, and by how much?
4. \`recompute_time(S)\` and \`swap_time(S)\` are both linear in \`S\`. What does that imply about
   their ratio as \`S\` varies? Derive the ratio symbolically, evaluate it for Llama-3-8B on this
   hardware, and check it at \`S\` = 10, 1,000 and 100,000 tokens to confirm what the algebra
   predicts.
5. Question 4 should have surprised you if you believed "short sequences favour recompute." Assume
   swap carries a fixed overhead of about 200 microseconds per transfer — DMA setup and
   synchronization that the pure bytes/bandwidth model above omits — that recompute does not pay.
   At what \`S\` does recompute start winning? Recompute the crossover for 500 μs, 1 ms and 2 ms of
   overhead as well, and say what this implies about how short "short" really has to be for the
   folklore to hold.
6. A sequence table entry is one 4-byte physical block id per logical block. At 8,192 tokens of
   context, how many blocks does a Llama-3-8B sequence need, how many bytes does its block table
   cost, and what fraction is that of the KV bytes those blocks actually hold?
7. Fifty concurrent requests share an identical 2,000-token system prompt. Using automatic prefix
   caching, only the first pays full prefill on those 2,000 tokens. How much prefill compute (FLOP)
   and wall-clock GPU time does caching save the other forty-nine, at the achieved throughput from
   question 1?
8. What does the arithmetic in questions 4 and 5 assume that a real multi-tenant deployment does
   not deliver?`,

    solution: `**1. Achieved compute throughput**

\`\`\`
2 x 8.03e9 x 32000 = 5.1392e14 FLOP = 513.9 TFLOP   (matches Module 5's "~514 TFLOP")
C = 513.9 TFLOP / 0.740 s = 694.5 TFLOP/s            (70.2% of the H100's 989 TFLOP/s peak)
\`\`\`

A 70% achieved fraction on a compute-bound matmul is a believable real-world figure, and it is the
number question 3 onward is built on.

**2. What 2.5 GiB means**

\`\`\`
S = 2.5 x 2^30 B / 131,072 B  =  20,480 tokens
single leg  = 2.5 x 2^30 / 50e9  =  0.0537 s  = 53.7 ms   (Module 5 rounds this to "~50 ms")
round trip  = 2 x 53.7 ms        =  107.4 ms              (Module 5's "50 out and 50 back")
\`\`\`

The rounding in Module 5's checkpoint answer was for readability, not error — 53.7 rounds
comfortably to "roughly 50."

**3. Recompute at the same S = 20,480**

\`\`\`
recompute_time(20480) = 2 x 8.03e9 x 20480 / 694.5e12  =  0.4736 s  =  473.6 ms
swap round trip                                         =  107.4 ms
\`\`\`

Swap wins here by **4.41×** — consistent with "long sequences favour swap." So far the folklore
holds.

**4. The ratio is S-independent**

Both times are \`(constant) x S\`, so:

\`\`\`
recompute_time(S) / swap_time(S)  =  (2 N_active / C) / (2 kv_per_token / BW_pcie)
                                   =  (N_active x BW_pcie) / (C x kv_per_token)
\`\`\`

\`S\` cancels completely. Plugging in numbers:

\`\`\`
(8.03e9 x 50e9) / (694.5e12 x 131,072)  =  4.41
\`\`\`

Checking directly at three lengths:

\`\`\`
S =     10:  recompute = 0.231 ms   swap = 0.052 ms   ratio = 4.41x
S =  1,000:  recompute = 23.13 ms   swap = 5.24 ms     ratio = 4.41x
S = 100,000: recompute = 2312.5 ms  swap = 524.3 ms    ratio = 4.41x
\`\`\`

The ratio is exactly 4.41 at every length. Under this linear model, **swap is cheaper than
recompute at every sequence length, including very short ones** — not just long ones. The
"short favours recompute" intuition does not survive contact with the arithmetic, at least not for
this model and this hardware pair.

**5. Where a real crossover would have to come from**

Per-token cost:

\`\`\`
recompute: 2 x 8.03e9 / 694.5e12  =  23.12 us/token
swap:      2 x 131,072 / 50e9     =  5.24 us/token
gap:       17.88 us/token
\`\`\`

A fixed swap overhead \`OVERHEAD\` produces a crossover where \`OVERHEAD = S x gap\`, so
\`S* = OVERHEAD / gap\`:

\`\`\`
  200 us overhead  ->  S* = 11.2 tokens
  500 us overhead  ->  S* = 28.0 tokens
1,000 us overhead  ->  S* = 55.9 tokens
2,000 us overhead  ->  S* = 111.8 tokens
\`\`\`

Even a generous 2 ms of fixed transfer overhead only buys recompute a win below about 112 tokens —
a handful of turns into a conversation. If real deployments do see recompute favoured over a wider
range than that, the reason is very unlikely to be pure DMA latency (the hardware table's PCIe
figure is ~10 μs, two orders of magnitude below what would be needed); it is more likely
**contention** — a swap consumes PCIe bandwidth that other transfers also want, which Module 5's
checkpoint already flags as a separate reason to prefer recompute under load, independent of the
bytes-and-FLOPs comparison this question is making.

**6. Block-table overhead**

\`\`\`
blocks needed at 8,192 tokens: ceil(8192 / 16) = 512
block-table bytes: 512 x 4 B = 2,048 B = 2.00 KiB
KV bytes those blocks hold:  8192 x 131,072 B = 1,073,741,824 B = 1024.0 MiB
overhead fraction: 2,048 / 1,073,741,824 = 0.0002%
\`\`\`

The indirection this entire level is about costs two kilobytes to save a gigabyte. That ratio is
why "the block table adds overhead" is true and irrelevant in the same breath.

**7. Prefix-cache savings, 50 requests, 2,000-token shared prompt**

\`\`\`
FLOPs saved = 2 x 8.03e9 x 2000 x 49 = 1.5739e15 FLOP = 1,573.9 TFLOP
time saved  = 1,573.9e12 / 694.5e12  = 2.266 s of GPU time
\`\`\`

Over two seconds of prefill compute the GPU never has to spend, recovered by one hash lookup per
block on 49 of the 50 requests. That is the number that shows up to users as a shorter
time-to-first-token, not as a change in how many sequences fit in memory — prefix caching is a
latency win dressed up as a memory feature.

**8. What the arithmetic assumes**

That the PCIe link is otherwise idle, that host memory is actually provisioned and pinned for
however many sequences you might want to swap out simultaneously, and that the *linear* prefill
FLOPs approximation used everywhere in this course holds — it undercounts recompute's true cost at
very long context, where attention's quadratic term stops being negligible, which if anything makes
recompute look *relatively better* here than it actually is and would push the true crossover even
shorter. None of those hold under real multi-tenant load: PCIe bandwidth is shared with weight
loading, tensor-parallel activation transfers and other sequences' swaps at the same moment;
pinned host memory has its own capacity ceiling; and a deployment under enough memory pressure to
be preempting sequences at all is precisely the deployment where "otherwise idle" is least true.
The honest conclusion is not "swap is always better" — it is that a length-based threshold is the
wrong lever to reach for first, and contention for shared resources is a bigger factor than the
bytes-and-FLOPs comparison this question modeled.`,
  },

  codeLab: {
    goal: `Build a working, miniature copy of vLLM's block manager and scheduler — paged
allocation, admission control, and preemption by recomputation — then use it to reproduce two
claims from this level: that paging supports far more concurrent sequences than reserving each
one's maximum length up front, and that pushing the admission cap past a certain point buys almost
no throughput for a steep rise in preemption. No GPU, no attention, no dependencies — the point is
the bookkeeping, isolated from everything it is bookkeeping for.`,
    code: `"""A toy version of vLLM's block manager and scheduler: paged allocation,
continuous batching, admission control, and preemption by recomputation.

    python3 vllm_scheduler.py        # standard library only

Every design choice below has a real counterpart: BLOCK_SIZE is vLLM's
original default, the block table is a list of physical block ids,
max_num_seqs is the same admission-control knob vLLM exposes, and
admission/preemption run once per simulated decode step -- one step is one
forward pass. Preemption drops a sequence's blocks and sends it back to the
queue: the recompute policy from the math lab. What is missing on purpose:
attention, swap, prefix caching, anything that touches a GPU. The point is
the bookkeeping, isolated from everything it is bookkeeping for.
"""

import random
from collections import deque

BLOCK_SIZE = 16            # tokens per physical block, vLLM's original default
TOTAL_BLOCKS = 400          # the whole simulated KV-cache budget, in blocks
MAX_LEN = 512               # longest any one sequence can run (prompt + output)


def ceil_div(a, b):
    return -(-a // b)


class Seq:
    __slots__ = ("id", "target_len", "len", "blocks", "preemptions")

    def __init__(self, sid, target_len):
        self.id = sid
        self.target_len = target_len
        self.len = 0
        self.blocks = []
        self.preemptions = 0


class BlockPool:
    """A free list of physical block ids. Allocation is O(1) and does not
    care which sequence a block belonged to a moment ago."""

    def __init__(self, total):
        self.free = list(range(total))
        self.total = total

    def alloc(self):
        return self.free.pop() if self.free else None

    def release(self, block_ids):
        self.free.extend(block_ids)

    @property
    def used(self):
        return self.total - len(self.free)


class Scheduler:
    """Iteration-level scheduling: one call to step() is one forward pass.
    Sequences move WAITING -> RUNNING -> DONE, or RUNNING -> WAITING again if
    preempted. max_num_seqs is the same admission-control cap vLLM exposes,
    independent of block availability -- both can be the reason a waiting
    sequence does not get in. Preemption priority is simple FCFS: the
    sequence admitted least recently goes first, the newest guest in the
    room. vLLM's default policy is close to this."""

    def __init__(self, pool, max_num_seqs):
        self.pool = pool
        self.max_num_seqs = max_num_seqs
        self.waiting = deque()
        self.running = []
        self.done = []
        self.preemptions = 0
        self.util_samples = []

    def submit(self, seq):
        self.waiting.append(seq)

    def _admit(self):
        while self.waiting and self.pool.free and len(self.running) < self.max_num_seqs:
            seq = self.waiting.popleft()
            seq.blocks.append(self.pool.alloc())
            self.running.append(seq)

    def step(self):
        self._admit()
        i = 0
        while i < len(self.running):
            seq = self.running[i]
            needed = ceil_div(seq.len + 1, BLOCK_SIZE)

            while needed > len(seq.blocks) and not self.pool.free:
                if len(self.running) - 1 <= i:
                    break  # nobody left to preempt for seq; it stalls this step
                victim = self.running.pop()
                self.pool.release(victim.blocks)
                victim.blocks = []
                victim.len = 0
                victim.preemptions += 1
                self.preemptions += 1
                self.waiting.appendleft(victim)

            if needed > len(seq.blocks) and self.pool.free:
                seq.blocks.append(self.pool.alloc())

            if needed <= len(seq.blocks):
                seq.len += 1

            if seq.len >= seq.target_len:
                self.pool.release(seq.blocks)
                seq.blocks = []
                self.done.append(seq)
                self.running.pop(i)
                continue
            i += 1

        self.util_samples.append(self.pool.used / self.pool.total)


def run_simulation(max_num_seqs, seed=7, n_requests=300, arrivals_per_step=2, max_steps=8000):
    random.seed(seed)
    pool = BlockPool(TOTAL_BLOCKS)
    sched = Scheduler(pool, max_num_seqs)
    arrivals = deque(Seq(i, random.randint(20, MAX_LEN)) for i in range(n_requests))

    tokens_generated = 0
    peak_concurrency = 0
    step = 0
    while (arrivals or sched.waiting or sched.running) and step < max_steps:
        for _ in range(min(arrivals_per_step, len(arrivals))):
            sched.submit(arrivals.popleft())
        sched.step()
        tokens_generated += len(sched.running)  # one token per running seq this step
        peak_concurrency = max(peak_concurrency, len(sched.running))
        step += 1

    return sched, step, tokens_generated, peak_concurrency


if __name__ == "__main__":
    print("=== One run, max_num_seqs=32 ===\\n")
    sched, steps, tokens, peak = run_simulation(max_num_seqs=32)
    avg_util = sum(sched.util_samples) / len(sched.util_samples)
    completed = len(sched.done)
    preempted_seqs = sum(1 for s in sched.done if s.preemptions)

    print(f"  requests completed:        {completed}")
    print(f"  simulated decode steps:    {steps}")
    print(f"  tokens generated:          {tokens}")
    print(f"  peak concurrent sequences: {peak}")
    print(f"  average pool utilization:  {avg_util * 100:.1f}%")
    print(f"  preemption events:         {sched.preemptions}")
    print(f"  sequences preempted >=1x:  {preempted_seqs} of {completed}")
    print(f"  throughput:                {tokens / steps:.2f} tokens/step")

    naive_blocks_per_seq = ceil_div(MAX_LEN, BLOCK_SIZE)
    naive_max_concurrency = TOTAL_BLOCKS // naive_blocks_per_seq
    print()
    print("=== Naive comparison: reserve MAX_LEN contiguously, up front ===\\n")
    print(f"  blocks reserved per sequence (worst case): {naive_blocks_per_seq}")
    print(f"  max concurrent sequences the same {TOTAL_BLOCKS}-block pool allows: {naive_max_concurrency}")
    print(f"  the paged run's peak concurrency was {peak / naive_max_concurrency:.1f}x that")

    print()
    print("=== Part 2: what max_num_seqs actually trades off ===\\n")
    print(f"  {'cap':>5} {'peak':>6} {'util%':>7} {'preempt_ev':>11} {'preempt_seqs':>13} {'tok/step':>9}")
    for cap in (16, 24, 32, 40, 48, 56, 64, 96, 128):
        s, st, tk, pk = run_simulation(max_num_seqs=cap)
        u = sum(s.util_samples) / len(s.util_samples)
        pseq = sum(1 for sq in s.done if sq.preemptions)
        print(f"  {cap:5d} {pk:6d} {u*100:6.1f}% {s.preemptions:11d} {pseq:13d} {tk/st:9.2f}")

# --- TODO for you ----------------------------------------------------------
#   1. Swap policy: instead of dropping a preempted sequence's blocks, copy
#      them into a Python list standing in for host memory, and restore them
#      on readmission instead of resetting .len to 0. At what max_num_seqs
#      does swap start beating recompute on total simulated steps to
#      completion? Compare against the math lab's per-token ratio.
#   2. Give ten sequences an identical 200-token prefix and add a prefix
#      cache: a dict from (block index, token tuple) to a physical block id,
#      consulted before allocating a fresh block for those positions. How
#      many blocks does the run save?
#   3. Replace FCFS preemption with shortest-remaining-first (preempt
#      whoever is furthest from target_len). Does throughput improve, and
#      what happens to the longest sequence in the batch?
`,
    expect: `The single run at \`max_num_seqs=32\` completes all 300 requests in 2,739 simulated
steps, generating 75,830 tokens (27.69 tokens/step), holding average pool utilization at 73.1%,
with 77 preemption events touching 35 of the 300 sequences. The naive comparison shows why the cap
was set where it was: reserving each sequence's full 512-token maximum up front (32 blocks each)
caps the same 400-block pool at 12 concurrent sequences, so the paged run's 32-deep batch is
**2.7× that** — and that is with the admission cap itself, not the pool, doing the limiting.

Part 2 is the one worth reading as a curve rather than a single number:

\`\`\`
  cap   peak   util%  preempt_ev  preempt_seqs  tok/step
   16     16   39.8%           0             0     14.85
   24     24   57.6%           0             0     21.50
   32     32   73.1%          77            35     27.69
   40     40   80.9%         564           166     33.20
   48     48   85.2%       1,214           230     37.79
   56     56   86.3%       1,864           236     40.37
   64     64   87.1%       2,378           239     42.15
   96     96   86.2%       3,267           238     43.71
  128    128   86.4%       3,441           239     44.23
\`\`\`

Below cap 32, preemption is zero — the pool never comes under enough pressure to evict anyone.
From 32 to 64 the cap buys real throughput (27.69 to 42.15 tokens/step, +52%) at a steeply rising
preemption cost (77 to 2,378 events). From 64 to 128 — doubling the cap again — throughput moves
almost nothing (42.15 to 44.23, +5%) while preemption keeps climbing and utilization has already
flattened around 86–87%. That flattening is the same knee Module 5's checkpoint describes for raw
batch size: past a point you are not buying throughput, you are buying preemption. If you let the
cap go effectively unbounded, this workload settles at a natural peak of 154 concurrent sequences —
**12.8× the naive ceiling** — which is the number to reach for if someone asks what paging is
"really" worth here, with the caveat that reaching it costs the most preemption of any row in the
table.

If your numbers differ, check three things first: that \`random.seed(7)\` is set before the
arrivals are generated and nowhere else; that preemption pops from the *end* of \`self.running\`
(the most recently admitted, not the current sequence); and that \`_admit\` checks
\`len(self.running) < self.max_num_seqs\` before allocating a block, not after — admitting first
and checking the cap second silently turns the cap into a no-op.`,
    stretch: `Change the arrival distribution from \`random.randint(20, MAX_LEN)\` to something
bimodal — say, 80% short requests (20–60 tokens) and 20% long ones (400–512) — matching the shape
Module 5 describes for real chat traffic. Re-run the Part 2 sweep. The naive baseline does not
move, because it is a function of \`MAX_LEN\` alone; the paged numbers should move quite a lot,
because they are a function of what sequences actually cost, not their ceiling. That gap — one
number that is blind to the traffic shape and one that tracks it — is the entire argument for
paging in one experiment.

Then implement TODO 3 (shortest-remaining-first preemption) for real and re-run the cap=64 row.
vLLM does not do this by default; after you see the effect on tail latency for the longest
sequence in the batch, you will understand why "obviously better" throughput policies are not
always shipped.`,
  },

  papers: [
    {
      title: 'Efficient Memory Management for Large Language Model Serving with PagedAttention',
      by: 'Kwon et al., SOSP 2023',
      url: 'https://arxiv.org/abs/2309.06180',
      why: 'Module 6 read this for the fragmentation numbers and Module 5 for the scheduling policy in Section 4.3. This pass is for what both skip.',
      frame: `Read for the memory-sharing mechanism — copy-on-write over reference-counted blocks
— and the distributed execution design, the driver/worker split this level's last concept
describes. Notice how much of the systems design was already settled before a single kernel got
optimized; the paper is a systems paper first and a kernel paper second, which the name makes easy
to miss.`,
    },
    {
      title: 'Orca: A Distributed Serving System for Transformer-Based Generative Models',
      by: 'Yu et al., OSDI 2022',
      url: 'https://www.usenix.org/conference/osdi22/presentation/yu',
      why: 'The paper vLLM’s scheduler is downstream of. Continuous batching as a scheduling idea is Orca’s; PagedAttention is what let vLLM run it without the memory getting in the way, a year later.',
      frame: `Read for iteration-level scheduling on its own terms, independent of any particular
memory manager — Orca predates PagedAttention and pays for it in memory reuse, not in the
scheduling idea, which is exactly right. Compare its selective batching, which batches only the
operations that can be batched across different-length sequences, against what a paged block table
buys for free once it exists.`,
    },
    {
      title: 'Taming Throughput-Latency Tradeoff in LLM Inference with Sarathi-Serve',
      by: 'Agrawal et al., OSDI 2024',
      url: 'https://arxiv.org/abs/2403.02310',
      why: 'The concrete algorithm behind "chunked prefill slices a long prompt into pieces" above. Module 5 covers the general prefill/decode interference argument; this pass is the scheduler-implementation angle.',
      frame: `**Section 4**'s token-budget rule is the thing to extract: a per-iteration cap on
prefill tokens, spent alongside whatever decode work is already scheduled. It slots directly into
the running-queue loop this level describes — chunked prefill is not a separate scheduler, it is
one more admission rule inside the same one.`,
    },
    {
      title: 'SGLang: Efficient Execution of Structured Language Model Programs',
      by: 'Zheng et al., 2023',
      url: 'https://arxiv.org/abs/2312.07104',
      why: 'The same prefix-sharing problem vLLM’s automatic prefix caching solves, solved with a different data structure: a radix tree instead of a hash chain over fixed blocks.',
      frame: `Read the RadixAttention section against this level's prefix-caching concept side by
side. A hash chain can only match a prefix at a block boundary and evicts by plain LRU; a radix
tree matches at the token and evicts by tree structure. Worth having an opinion on which cost is
worth which flexibility, rather than treating one as simply better.`,
    },
    {
      title: 'DistServe: Disaggregating Prefill and Decoding for Goodput-optimized Large Language Model Serving',
      by: 'Zhong et al., OSDI 2024',
      url: 'https://arxiv.org/abs/2401.09670',
      why: 'Everything in this level describes one scheduler making prefill and decode decisions together, on the same GPUs. This is the paper arguing you should not.',
      frame: `Read it as the rebuttal to this level's architecture, not an extension of it —
disaggregation gives up the single block pool and the cheap admission this level spent five
concepts describing, in exchange for removing prefill/decode interference entirely. Module 9 has
the KV-transfer mechanics; this is the argument for why you would pay for them.`,
    },
  ],

  checkpoint: {
    claim: `You can read vLLM's own scheduler and block-manager source and recognise every piece
of it: the block table as page-table indirection, the running/waiting/swapped loop, swap-versus-
recompute as an arithmetic question rather than a rule of thumb, copy-on-write as a reference
count plus a fork on divergence, prefix caching as a hash chain over the same blocks, and the
driver/worker split that lets all of it run unmodified across many GPUs.`,
    questions: [
      {
        q: 'A colleague says PagedAttention and continuous batching are two separate vLLM features you could in principle ship independently. What is wrong with that claim?',
        a: `They are separable in name only. Continuous batching needs to cheaply admit and evict
sequences every iteration, which means allocating and freeing memory in small, fast operations —
exactly what a block table provides and a contiguous per-sequence reservation does not. Run
continuous batching over a contiguous allocator and admitting a new sequence mid-flight means
finding a free reservation-sized hole, which is the fragmentation problem Module 6 measured at
60–80% waste; you would be paying that cost on every single admission decision. PagedAttention, in
turn, has no reason to exist without a scheduler that actually adds and evicts sequences fine-
grainedly enough to use the flexibility — a system that runs static batches to completion gets
almost nothing from paging, since it allocates once per batch and frees once per batch regardless
of how cheap allocation is. Orca proved the scheduling idea works; PagedAttention is what let vLLM
run it without the memory getting in the way. Neither paper's system is the other's without both
pieces.`,
      },
      {
        q: 'Work through the swap-versus-recompute arithmetic for Llama-3-8B on an H100 with a real PCIe link. Which one actually wins, and does it depend on sequence length?',
        a: `Under the linear approximation this course uses for prefill FLOPs everywhere else,
recompute costs about 23.1 microseconds per token and swap costs about 5.2 microseconds per token
— a ratio of 4.41, and because both scale linearly in sequence length, that ratio is the same at
10 tokens and at 100,000. Swap wins at every length under this model, which contradicts the
intuition that short sequences favour recomputation. A real crossover has to come from something
the bytes-and-FLOPs model leaves out entirely — a fixed per-transfer overhead on the swap path,
like DMA setup and driver synchronization — and even a generous 2 millisecond estimate for that
overhead only pushes the crossover to around 112 tokens. The honest takeaway is not "always swap."
It is that a pure length threshold is the wrong first-order model, and that PCIe contention with
other traffic — which the bytes/bandwidth comparison assumes away — is a more likely reason a real
engine would still prefer recompute under load.`,
      },
      {
        q: 'Two parallel samples share a 1,000-token prompt and then diverge at the very next token. How many physical blocks get copied at the moment of divergence?',
        a: `One. At a 16-token block size, 1,000 shared tokens occupy 63 full blocks (1008 tokens'
worth) with the 1,000th token sitting partway into the 63rd block — call it the boundary case and
say the shared prefix cleanly fills 62 or 63 blocks; either way, only the single block that both
sequences are about to write their next, different token into needs to fork. The other 61 or 62
already-full blocks stay shared, still reference-counted at two, because nothing about to be
written touches them. Copy-on-write is a one-block operation triggered by a write, not a whole-
cache duplication triggered by branching — the point of tracking divergence at block granularity
rather than sequence granularity is that "two sequences forked" and "one block forked" are
different events, and only the second one costs memory.`,
      },
      {
        q: 'Automatic prefix caching and SGLang’s RadixAttention solve the same problem. Why might an engine choose the coarser, block-hash version anyway?',
        a: `Because it is not a second system — it reuses the block table and the physical pool
every other feature in this level already depends on, with no additional data structure to keep
consistent as blocks are allocated, freed and shared. A hash chain only recognises a shared prefix
down to a block boundary, so a prefix that diverges one token before the end of a block gets zero
credit for that block, where a radix tree would still match everything up to the exact token. That
imprecision is a real cost on workloads with lots of near-miss prefixes. But it buys operational
simplicity: the same allocator, the same reference counting, the same admission logic, one fewer
subsystem that can disagree with the rest of the engine about what is free. Whether that trade is
worth it depends on how much of your traffic actually shares prefixes only up to odd token
boundaries — which for a fixed system prompt or a repeated few-shot header, the common cases, is
rarely the failure mode that matters.`,
      },
      {
        q: 'You move a vLLM deployment from one GPU to eight, with tensor parallelism across all eight. What changes about the scheduler, and what does not?',
        a: `The scheduler itself does not change — it is still one sequential process deciding
admission and preemption once per iteration, and it still owns one set of block tables. What
changes is that its decision is now broadcast to eight worker processes rather than executed
in-process, and each worker runs its shard of the model against block tables and slot mappings it
received rather than computed. The workers additionally exchange partial activations through NCCL
collectives at the points tensor parallelism requires them, once per attention block and once per
MLP block, which Module 9 prices in fixed per-collective latency that does not shrink as you add
GPUs. What this design buys is that none of this level's first five concepts — the block table,
the admit/preempt loop, copy-on-write, prefix caching — needed to be rewritten to run on eight
GPUs instead of one. What it costs is a single driver process that is now a sequential bottleneck
serving eight workers instead of doing the work itself, which is the architectural choice
DistServe's disaggregation argument is aimed at.`,
      },
    ],
  },

  pitfalls: [
    {
      wrong: 'PagedAttention is a kernel optimization.',
      right: `It is a memory-management redesign that a kernel then has to be written to support.
The saving Module 6 measures — internal fragmentation dropping from 60–80% to under 4% — comes
entirely from changing what gets *allocated*, before a single FLOP is computed differently. What
does live at the kernel level is the consequence: because the cache is no longer contiguous, the
attention kernel has to gather K and V through a block table and slot mapping instead of reading a
flat buffer, which is real engineering work but is downstream of the allocator decision, not the
source of the saving. Confusing the two makes "just write a faster paged-attention kernel" sound
like it could recover memory it cannot — the kernel makes the layout usable, the allocator is what
made the layout smaller in the first place.`,
    },
    {
      wrong: 'Swap is always cheaper than recomputing, because copying is faster than redoing work.',
      right: `Under the arithmetic this level works through for Llama-3-8B on an H100, that
happens to be true — but it is true because of this specific model-and-hardware ratio, not as a
general law, and even here it stops being clean once you account for what the simple model leaves
out. The ratio \`(N_active x BW_pcie) / (C x kv_per_token)\` is what actually decides it, and every
term is hardware- and model-specific: a slower PCIe link, a model with a larger cache per token
relative to its active parameters, or a GPU with unusually high achieved compute throughput can all
push the ratio the other way. Treat "swap or recompute" as an arithmetic question you re-derive
for your own model and hardware, the way this level's math lab does, rather than folklore imported
from somewhere else's numbers.`,
    },
    {
      wrong: 'Copy-on-write means the two sequences’ caches stay in sync until you explicitly branch them.',
      right: `They diverge automatically, the instant generation produces different tokens — there
is no explicit branch operation to forget to call. The block manager forks a block the moment a
write to a shared block would make it disagree with what another sequence expects to find there;
by the time you would think to ask whether the sequences have "branched yet," the answer was
decided by their first differing token, several tokens ago in wall-clock terms if you were not
watching for it. The mental model to use is closer to a persistent data structure than to a manual
git branch: sharing is the passive default, and copying is a side effect of writing, not a step
anyone requests.`,
    },
    {
      wrong: 'Automatic prefix caching means you no longer need to think about prompt structure.',
      right: `It means the *scheduler* no longer needs to be told about shared prefixes — it will
find them on its own. It does not mean prompt structure stops mattering to the result. A cache hit
requires the new request's prefix to match an existing one exactly, down to the token, up to a
block boundary; a system prompt that gets reformatted, reordered, or has a timestamp spliced into
it defeats the hash chain from the very first block, silently, with no error and no warning — the
request just runs as a full-cost prefill and nobody is told why. If your team is not seeing the
time-to-first-token improvement prefix caching should be delivering, check whether something
upstream is mutating the "identical" prompt before it reaches the engine, because the caching layer
will not tell you it happened.`,
    },
  ],

  glossary: [
    { term: 'block table', def: 'A per-sequence array mapping logical block indices to physical block indices in the shared KV-cache pool. The core PagedAttention data structure.' },
    { term: 'slot mapping', def: 'The per-token mapping from a batch position to a physical KV-cache slot, which the attention kernel reads through to gather scattered K and V.' },
    { term: 'iteration-level scheduling', def: 'Scheduling once per forward pass rather than once per request, so a sequence can join or leave the batch on any step. Introduced by Orca; the mechanism continuous batching runs on.' },
    { term: 'admission control', def: 'The scheduler decision of whether a waiting sequence can start this iteration, gated by both free blocks and a configured concurrency cap (max_num_seqs).' },
    { term: 'preemption', def: 'Evicting a running sequence when the block pool has none free for it, by swap or by recomputation, rather than failing the request.' },
    { term: 'swap', def: 'Preemption that copies a sequence’s KV blocks to host memory and back, preserving the cache at the cost of a PCIe round trip.' },
    { term: 'recomputation', def: 'Preemption that drops a sequence’s KV blocks and reruns prefill over its tokens-so-far when it is re-admitted, at the cost of redone compute.' },
    { term: 'copy-on-write (KV cache)', def: 'Sharing a physical block between sequences via a reference count, and copying it only at the moment one sequence’s write would make it diverge from another’s.' },
    { term: 'automatic prefix caching', def: 'Resolving a new request’s prefix to existing physical blocks via a hash chained over each block’s tokens and its predecessor’s hash, without the request explicitly declaring a shared prefix.' },
    { term: 'reference count', def: 'The count of sequences whose block table points at a given physical block, used to decide whether releasing a sequence actually frees the block.' },
    { term: 'driver process', def: 'In a multi-GPU vLLM deployment, the single process holding the scheduler and authoritative block tables, which broadcasts each iteration’s decisions to the worker processes.' },
    { term: 'worker process', def: 'One per GPU in a multi-GPU deployment; holds a shard of the model and executes the batch the driver assigned, exchanging activations with other workers via NCCL under tensor parallelism.' },
    { term: 'chunked prefill', def: 'Capping how many prefill tokens the scheduler admits in one iteration and carrying the remainder to later iterations, so a long prompt interleaves with ongoing decode instead of blocking it.' },
  ],
};
