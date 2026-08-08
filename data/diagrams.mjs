/* ==========================================================================
   One grid-notes summary per level.

   Each entry gets the drawing kit `g` and a unique `id` (arrowhead markers are
   namespaced per figure) and returns SVG markup. Coordinates are GRID CELLS.

   These are teaching figures, not decoration: every number in them appears in
   the level's prose, and the arithmetic is the same arithmetic.

   `npm run check:figures` (see build/) measures the real rendered text boxes
   and fails on any collision — do not trust the eye alone on these.
   ========================================================================== */

const A = 'var(--accent)';
const DIM = 'var(--ink-dim)';
const FAINT = 'var(--ink-faint)';
const PAC = 'var(--pac)';
const INKY = 'var(--inky)';
const BLINKY = 'var(--blinky)';
const PINKY = 'var(--pinky)';
const CLYDE = 'var(--clyde)';
const MAZE = 'var(--maze)';
const TINT = 'color-mix(in srgb, var(--maze) 14%, transparent)';

export default {
  /* ---------------------------------------------------------------- 00 --- */
  0: {
    w: 58, h: 28,
    title: 'The forward pass and its tensor shapes',
    desc: 'Token ids become an embedding, which feeds a residual stream of shape batch by ' +
      'sequence by 4096. That width never changes through all 32 layers. Attention and the MLP ' +
      'each read the stream and add their result back into it at a plus junction. The stream ' +
      'ends at the LM head, producing logits over 128256 tokens.',
    caption: 'The residual stream is a fixed-width bus. Every block reads it and adds back — ' +
      'nothing transforms it in place, which is why you can delete a layer and still get text.',
    draw: (g, id) => [
      g.heading(1.5, 1.6, 'LEVEL 00 · WHERE THE SHAPES GO'),
      g.line(1.5, 2.4, 56.5, 2.4, { stroke: FAINT, width: 1 }),
      g.note(1.5, 3.7, ['every block reads the stream and adds back into it — the width never changes'],
        { fill: FAINT }),

      g.chip(1.5, 5.4, '[B, S]', { accent: FAINT }),
      g.chip(50.6, 5.4, '[B,S,128256]', { accent: FAINT }),

      g.box(1.5, 7.6, 5.5, 3, 'token ids'),
      g.arrow(7.2, 9.1, 9, 9.1, { id }),
      g.box(9.2, 7.6, 6, 3, 'embed', { sub: '128256×4096' }),
      g.arrow(15.4, 9.1, 17.2, 9.1, { id }),

      // the bus — the plus junctions sit on its lower edge, clear of the label
      g.box(17.4, 7.8, 22.4, 2.6, 'RESIDUAL STREAM   [B, S, 4096]', { accent: A, fill: TINT }),
      g.arrow(40, 9.1, 41.8, 9.1, { id }),
      g.box(42, 7.6, 6.4, 3, 'lm_head', { sub: '4096×128256' }),
      g.arrow(48.6, 9.1, 50.4, 9.1, { id }),
      g.box(50.6, 7.6, 5.9, 3, 'logits'),

      // the two taps
      g.box(17.4, 12.4, 22.4, 4.6, null, { accent: FAINT, fill: 'none', dashed: true }),
      g.arrow(20.5, 10.5, 20.5, 12.8, { id }),
      g.box(18.4, 12.9, 8.4, 3.2, 'ATTENTION', { sub: 'mixes positions', accent: PINKY }),
      g.arrow(26, 12.8, 26, 10.6, { id }),
      g.plus(26, 10.4),
      g.arrow(32, 10.5, 32, 12.8, { id }),
      g.box(30.4, 12.9, 8.4, 3.2, 'MLP · SwiGLU', { sub: 'per position', accent: CLYDE }),
      g.arrow(37.5, 12.8, 37.5, 10.6, { id }),
      g.plus(37.5, 10.4),
      g.text(39.6, 18.2, '× 32 layers', { font: 'tag', fill: FAINT, anchor: 'end' }),

      // details, in two columns with a hard divider between them
      g.line(28.6, 19.4, 28.6, 25.4, { stroke: FAINT, width: 1 }),

      g.heading(1.5, 20.2, 'INSIDE ATTENTION', { fill: PINKY }),
      g.chip(1.5, 21.4, 'Q    [B, 32, S, 128]', { accent: PINKY }),
      g.chip(1.5, 23.2, 'K,V  [B,  8, S, 128]', { accent: PINKY }),
      g.chip(1.5, 25, 'scores [B, 32, S, S]', { accent: BLINKY }),
      g.note(14.5, 22.6, ['4.3 GB per layer at S = 8192.', 'Never materialised — level 07.'],
        { fill: FAINT }),

      g.heading(30.4, 20.2, 'INSIDE THE MLP', { fill: CLYDE }),
      g.chip(30.4, 21.4, 'gate/up [B, S, 14336]', { accent: CLYDE }),
      g.chip(30.4, 23.2, 'down    [B, S, 4096]', { accent: CLYDE }),
      g.note(43.5, 22.6, ['176 M params vs 42 M —', 'the MLP is 4.2× the bytes.'], { fill: FAINT }),

      g.line(1.5, 26.4, 56.5, 26.4, { stroke: FAINT, width: 1 }),
      g.note(1.5, 27.3, ['8.03 B parameters · 16.06 GB at fp16 · all of them read once per forward pass'],
        { fill: DIM }),
    ].join(''),
  },

  /* ---------------------------------------------------------------- 01 --- */
  1: {
    w: 58, h: 25,
    title: 'Prefill and decode are two different workloads',
    desc: 'Prefill processes the whole prompt in one pass, amortising the weight read over ' +
      'thousands of positions, and is compute-bound at 2193 FLOPs per byte. Decode runs one ' +
      'position per pass and re-reads all 15 GB of weights each time, so it is memory-bound at ' +
      '1.07 FLOPs per byte. For a 2048-token prompt and 512 output tokens, prefill takes 35 ' +
      'milliseconds and decode takes 2293, so decode is 98.5 per cent of the wall clock.',
    caption: 'Same weights, same code path, three orders of magnitude apart in arithmetic ' +
      'intensity. Everything after this level follows from that gap.',
    draw: (g, id) => [
      g.heading(1.5, 1.6, 'LEVEL 01 · ONE PASS vs FIVE HUNDRED'),
      g.line(1.5, 2.4, 56.5, 2.4, { stroke: FAINT, width: 1 }),

      g.heading(1.5, 4.4, 'PREFILL · one pass, 2048 positions', { fill: INKY }),
      g.box(1.5, 5.4, 26, 4, null, { accent: INKY, fill: TINT }),
      ...Array.from({ length: 16 }, (_, i) =>
        g.box(2.3 + i * 1.55, 6.2, 1.15, 2.4, null, { accent: INKY, fill: 'var(--phosphor)' })),
      g.text(14.5, 10.6, 'weights read ONCE, used by 2048 positions',
        { font: 'sub', fill: DIM, anchor: 'middle' }),
      g.chip(1.5, 11.8, '2193 FLOP/byte', { accent: INKY }),
      g.chip(11.5, 11.8, 'compute-bound', { accent: INKY }),
      g.chip(21, 11.8, '35 ms', { accent: INKY }),

      g.heading(31, 4.4, 'DECODE · 512 passes, 1 position each', { fill: BLINKY }),
      ...Array.from({ length: 8 }, (_, i) => [
        g.box(31 + i * 2.7, 5.4, 2.1, 4, null, { accent: BLINKY, fill: TINT }),
        g.box(31.45 + i * 2.7, 6.2, 1.2, 2.4, null, { accent: BLINKY, fill: 'var(--phosphor)' }),
      ].join('')),
      g.text(53.4, 7.4, '…×512', { font: 'sub', fill: BLINKY }),
      g.text(42, 10.6, 'weights read 512 TIMES, one position each',
        { font: 'sub', fill: DIM, anchor: 'middle' }),
      g.chip(31, 11.8, '1.07 FLOP/byte', { accent: BLINKY }),
      g.chip(41, 11.8, 'memory-bound', { accent: BLINKY }),
      g.chip(50, 11.8, '2293 ms', { accent: BLINKY }),

      g.line(1.5, 14.6, 56.5, 14.6, { stroke: FAINT, width: 1 }),
      g.heading(1.5, 16, 'WHERE THE WALL-CLOCK GOES · 2048 in, 512 out'),
      g.bar(1.5, 16.8, 2.4, [
        { w: 0.83, fill: INKY, label: '' },
        { w: 54.17, fill: PAC, label: 'DECODE   98.5%', ink: 'var(--on-pac)' },
      ]),
      g.text(1.5, 20.6, 'prefill is that 1.5% sliver on the left',
        { font: 'sub', fill: INKY }),

      g.note(1.5, 23, [
        'Batching amortises the weight read: at batch 64 you move the same 15 GB and get 64 tokens out.',
      ], { fill: DIM }),
    ].join(''),
  },

  /* ---------------------------------------------------------------- 02 --- */
  2: {
    w: 58, h: 27,
    title: 'The KV cache size formula and the memory budget',
    desc: 'The formula multiplies two for keys and values, by 80 layers, by 8 KV heads, by head ' +
      'dimension 128, by two bytes, giving 320 kibibytes per token for Llama-3-70B. At 8192 ' +
      'context that is 2.5 gibibytes per sequence and 80 gibibytes for a batch of 32. On four ' +
      'H100s, 141 GB of weights and 20 GB of workspace leave about 159 GB, which fits 59 ' +
      'sequences at 8k context but only three at 128k.',
    caption: 'Batch size is not a setting you choose. It is what is left after the weights, ' +
      'divided by the cache each sequence needs.',
    draw: (g, id) => {
      const scale = 7 / 59;
      const rows = [['1k ctx', 477, 'no'], ['8k ctx', 59, 'no'], ['32k ctx', 14, 'no'], ['128k ctx', 3, 'yes']];
      return [
        g.heading(1.5, 1.6, 'LEVEL 02 · BUILDING THE NUMBER THAT CAPS EVERYTHING'),
        g.line(1.5, 2.4, 56.5, 2.4, { stroke: FAINT, width: 1 }),

        g.heading(1.5, 4.2, 'PER TOKEN · LLAMA-3-70B · fp16'),
        ...[['2', 'K and V'], ['× 80', 'layers'], ['× 8', 'kv_heads ← not 64'],
            ['× 128', 'head_dim'], ['× 2 B', 'fp16']].map((r, i) => [
          g.text(2, 6 + i * 1.7, r[0], { font: 'num', fill: A }),
          g.text(6.8, 6 + i * 1.7, r[1], { font: 'sub', fill: DIM }),
        ].join('')),
        g.line(1.5, 14.6, 17, 14.6, { stroke: A, width: 1.4 }),
        g.text(2, 15.7, '= 320 KiB', { font: 'num', fill: A }),
        g.text(9.5, 15.7, 'per token', { font: 'sub', fill: DIM }),

        g.arrow(17.6, 10.6, 19.8, 10.6, { id }),
        g.box(20.2, 6.2, 11.4, 3.2, '× 8192 tokens', { sub: '= 2.5 GiB / seq', accent: A }),
        g.box(20.2, 10.6, 11.4, 3.2, '× 32 sequences', { sub: '= 80 GiB', accent: A }),
        g.note(20.2, 15.3, ['an entire H100,', 'holding nothing but cache'], { fill: FAINT }),

        g.heading(34.5, 4.2, 'FOUR H100s = 320 GB'),
        g.bar(34.5, 5.4, 2.6, [
          { w: 8.6, fill: BLINKY, label: 'weights 141', ink: 'var(--on-maze)' },
          { w: 1.3, fill: FAINT, label: '' },
          { w: 12.1, fill: MAZE, label: 'KV 159 GB', ink: 'var(--on-maze)' },
        ]),
        g.text(39.6, 9, 'workspace 20 →', { font: 'sub', fill: FAINT, anchor: 'end' }),

        g.heading(34.5, 11.4, 'WHAT FITS IN THAT 159 GB'),
        ...rows.map((r, i) => [
          g.text(34.5, 13.4 + i * 2.1, r[0], { font: 'sub', fill: r[2] === 'yes' ? BLINKY : DIM }),
          g.bar(41, 12.7 + i * 2.1, 1.4,
            [{ w: Math.max(0.35, r[1] * scale), fill: r[2] === 'yes' ? BLINKY : A, label: '' }]),
          g.text(50.5, 13.4 + i * 2.1, `${r[1]} seqs`,
            { font: 'num', fill: r[2] === 'yes' ? BLINKY : A }),
        ].join('')),
        g.note(34.5, 21.4, ['At 128k a $120,000 machine holds', 'three concurrent users.'],
          { fill: BLINKY }),

        g.line(1.5, 23.6, 56.5, 23.6, { stroke: FAINT, width: 1 }),
        g.text(1.5, 25, 'max_batch = (memory − weights − workspace) ÷ kv_bytes_per_sequence',
          { font: 'label', fill: A }),
        g.note(1.5, 26.3, ['Levels 05 to 09 are all attempts to grow the numerator or shrink the denominator.'],
          { fill: DIM }),
      ].join('');
    },
  },

  /* ---------------------------------------------------------------- 03 --- */
  3: {
    w: 58, h: 25,
    title: 'How sampling reshapes the distribution',
    desc: 'Logits pass through temperature, then truncation, then renormalisation, then ' +
      'sampling. Low temperature sharpens the distribution and high temperature flattens it, ' +
      'which makes the worst token 4.7 times more likely. Top-k keeps a fixed count, top-p a ' +
      'fixed cumulative mass, and min-p everything above a fixed ratio of the top token.',
    caption: 'Temperature was never the problem. The tail is: 128,000 unlikely tokens are ' +
      'collectively very likely once you flatten the distribution.',
    draw: (g, id) => {
      const sharp = [1, .28, .1, .05, .03, .02, .015, .01];
      const flat = [1, .72, .55, .45, .38, .33, .3, .27];
      return [
        g.heading(1.5, 1.6, 'LEVEL 03 · WHAT EACH KNOB DOES TO THE SHAPE'),
        g.line(1.5, 2.4, 56.5, 2.4, { stroke: FAINT, width: 1 }),

        g.box(1.5, 4.4, 7, 2.6, 'logits'),
        g.arrow(8.7, 5.7, 10.3, 5.7, { id }),
        g.box(10.5, 4.4, 8.4, 2.6, 'temperature', { accent: INKY }),
        g.arrow(19.1, 5.7, 20.7, 5.7, { id }),
        g.box(20.9, 4.4, 7.6, 2.6, 'truncate', { accent: PINKY }),
        g.arrow(28.7, 5.7, 30.3, 5.7, { id }),
        g.box(30.5, 4.4, 9.2, 2.6, 'renormalise'),
        g.arrow(39.9, 5.7, 41.5, 5.7, { id }),
        g.box(41.7, 4.4, 7, 2.6, 'sample', { accent: A }),
        g.note(1.5, 8.4, ['order matters — most libraries apply temperature first, then truncate'],
          { fill: FAINT }),

        g.heading(1.5, 11, 'TEMPERATURE reshapes', { fill: INKY }),
        g.dist(1.5, 12.2, 8.5, 4.4, sharp, { fill: INKY }),
        g.text(5.75, 17.6, 'T = 0.5', { font: 'sub', fill: DIM, anchor: 'middle' }),
        g.dist(12, 12.2, 8.5, 4.4, flat, { fill: INKY }),
        g.text(16.25, 17.6, 'T = 2.0', { font: 'sub', fill: DIM, anchor: 'middle' }),
        g.note(1.5, 19.4, ['the worst token became', '4.7× more likely'], { fill: BLINKY }),

        g.line(23.5, 10.2, 23.5, 21.4, { stroke: FAINT, width: 1 }),

        g.heading(25.5, 11, 'TRUNCATION cuts — same confident input', { fill: PINKY }),
        g.dist(25.5, 12.2, 8.5, 4.4, sharp, { fill: PINKY, cut: 3 }),
        g.text(29.75, 17.6, 'top-k 3', { font: 'sub', fill: DIM, anchor: 'middle' }),
        g.text(29.75, 18.8, 'fixed count', { font: 'sub', fill: FAINT, anchor: 'middle' }),
        g.dist(36, 12.2, 8.5, 4.4, sharp, { fill: PINKY, cut: 3 }),
        g.text(40.25, 17.6, 'top-p 0.9', { font: 'sub', fill: DIM, anchor: 'middle' }),
        g.text(40.25, 18.8, 'fixed mass', { font: 'sub', fill: FAINT, anchor: 'middle' }),
        g.dist(46.5, 12.2, 8.5, 4.4, sharp, { fill: A, cut: 3 }),
        g.text(50.75, 17.6, 'min-p 0.1', { font: 'sub', fill: DIM, anchor: 'middle' }),
        g.text(50.75, 18.8, 'fixed RATIO', { font: 'sub', fill: A, anchor: 'middle' }),

        g.line(1.5, 22, 56.5, 22, { stroke: FAINT, width: 1 }),
        g.note(1.5, 23.2, [
          'At T = 2.0 top-p keeps 5 tokens instead of 3 — it widens exactly when the candidates got less trustworthy.',
          'Min-p is pinned to p_max, so it tracks the model’s confidence rather than a mass target.',
        ], { fill: DIM }),
      ].join('');
    },
  },

  /* ---------------------------------------------------------------- 04 --- */
  4: {
    w: 58, h: 27,
    title: 'The roofline, with prefill and decode plotted on it',
    desc: 'Attainable performance is the minimum of peak FLOPs and arithmetic intensity times ' +
      'bandwidth. The two lines meet at the ridge point, 295 FLOPs per byte on an H100. Decode ' +
      'at batch 1 sits at intensity 1, far to the left and memory-bound. Decode at batch 64 sits ' +
      'at 64, still memory-bound. Prefill sits at 2193, compute-bound. Decode intensity equals ' +
      'the batch size exactly.',
    caption: 'Decode intensity equals your batch size. Put that number next to 295 and you know ' +
      'your regime before running anything.',
    draw: (g, id) => {
      const X0 = 7, X1 = 39, Y0 = 5, Y1 = 16.5;
      const sx = (log) => X0 + (log / 4) * (X1 - X0);
      const ridgeX = sx(Math.log10(295));
      const roofY = Y0 + 1.2;
      return [
        g.heading(1.5, 1.6, 'LEVEL 04 · THE ONLY CHART YOU NEED'),
        g.line(1.5, 2.4, 56.5, 2.4, { stroke: FAINT, width: 1 }),

        g.text(X0 - 0.4, Y0 - 1.4, 'attainable FLOP/s', { font: 'sub', fill: DIM }),
        g.line(X0, Y1, X1, Y1, { stroke: DIM, width: 1.4 }),
        g.line(X0, Y0, X0, Y1, { stroke: DIM, width: 1.4 }),
        ...[0, 1, 2, 3, 4].map((d) =>
          g.text(sx(d), Y1 + 1.2, ['1', '10', '100', '1k', '10k'][d],
            { font: 'sub', fill: FAINT, anchor: 'middle' })),
        g.text((X0 + X1) / 2, Y1 + 2.8, 'arithmetic intensity   FLOP / byte  →',
          { font: 'sub', fill: DIM, anchor: 'middle' }),

        g.poly([[X0, Y1], [ridgeX, roofY], [X1, roofY]], { stroke: A, width: 2.4 }),
        g.text(X1, roofY - 1.1, 'peak 989 TFLOP/s', { font: 'sub', fill: A, anchor: 'end' }),
        g.text(11.5, 12.4, 'slope = bandwidth', { font: 'sub', fill: A }),

        g.line(ridgeX, roofY, ridgeX, Y1, { stroke: FAINT, dashed: true, width: 1.2 }),
        g.dot(ridgeX, roofY, 4.5, A),
        g.text(ridgeX, roofY - 1.1, 'RIDGE 295', { font: 'tag', fill: A, anchor: 'middle' }),

        g.dot(sx(0), Y1 - 0.05, 5, BLINKY),
        g.dot(sx(Math.log10(64)), Y1 - (Math.log10(64) / Math.log10(295)) * (Y1 - roofY), 5, CLYDE),
        g.dot(sx(Math.log10(2193)), roofY, 5, INKY),

        // legend, kept out of the plot so nothing collides with the roof
        g.line(1.5, 20.4, 39, 20.4, { stroke: FAINT, width: 1 }),
        ...[
          [BLINKY, 'decode, batch 1', 'intensity 1 — 0.3% of the machine'],
          [CLYDE, 'decode, batch 64', 'intensity 64 — 22% of the ridge'],
          [INKY, 'prefill, 2048 tok', 'intensity 2193 — compute-bound'],
        ].map((r, i) => [
          g.dot(2.2, 21.6 + i * 1.5, 4.5, r[0]),
          g.text(3.4, 21.6 + i * 1.5, r[1], { font: 'num', fill: r[0] }),
          g.text(15.5, 21.6 + i * 1.5, r[2], { font: 'sub', fill: DIM }),
        ].join('')),

        g.line(41.5, 4, 41.5, 25, { stroke: FAINT, width: 1 }),
        g.heading(43.5, 5, 'THE RECIPE'),
        g.note(43.5, 6.4, ['intensity = FLOPs ÷ bytes', 'ridge = peak ÷ bandwidth'], { fill: DIM }),
        g.text(43.5, 9.2, 'below ridge → memory-bound', { font: 'sub', fill: BLINKY }),
        g.text(43.5, 10.6, 'above ridge → compute-bound', { font: 'sub', fill: INKY }),
        g.heading(43.5, 13, 'RIDGE POINTS'),
        ...[['A100', 153], ['H100', 295], ['H200', 206], ['L40S', 419]].map((r, i) =>
          g.text(43.5, 14.4 + i * 1.5, `${r[0]}   ${r[1]}`, { font: 'num', fill: DIM })),
        g.note(43.5, 21.4, ['H200: same compute,', 'more bandwidth — so it is', 'EASIER to saturate.'],
          { fill: FAINT }),
      ].join('');
    },
  },

  /* ---------------------------------------------------------------- 05 --- */
  5: {
    w: 58, h: 29,
    title: 'Static versus continuous batching',
    desc: 'A static batch runs until its longest sequence finishes, so shorter sequences leave ' +
      'their slots idle and utilisation falls to about twenty per cent. Continuous batching ' +
      'refills a slot the moment a sequence finishes, keeping utilisation above ninety per cent. ' +
      'Static batching gets worse as batch size grows, because the longest of N samples grows ' +
      'while the mean does not.',
    caption: 'Static batching is one of the few optimisations that degrades as you scale it. ' +
      'That is why scheduling moved from the request to the iteration.',
    draw: (g, id) => {
      const lens = [4, 6, 7, 10, 13, 20, 33, 52];
      const W = 51;
      return [
        g.heading(1.5, 1.6, 'LEVEL 05 · KEEPING THE BATCH FULL'),
        g.line(1.5, 2.4, 56.5, 2.4, { stroke: FAINT, width: 1 }),

        g.heading(1.5, 4.2, 'STATIC · the whole batch waits for the LONGEST sequence', { fill: BLINKY }),
        ...lens.map((L, i) => [
          g.text(2.9, 5.5 + i * 1.35, `${i}`, { font: 'sub', fill: FAINT, anchor: 'end' }),
          g.bar(3.5, 5 + i * 1.35, 1.05, [{ w: L, fill: A, label: '' }]),
          g.bar(3.5 + L, 5 + i * 1.35, 1.05, [{ w: W - L, fill: 'none', label: '' }]),
        ].join('')),
        g.text(3.5 + W / 2, 16.3, 'idle slots — allocated, unusable, still costing you',
          { font: 'sub', fill: FAINT, anchor: 'middle' }),
        g.chip(3.5, 17.4, 'utilisation 20.2%', { accent: BLINKY }),
        g.chip(18, 17.4, 'and it gets WORSE at larger batch', { accent: BLINKY }),

        g.line(1.5, 19.8, 56.5, 19.8, { stroke: FAINT, width: 1 }),
        g.heading(1.5, 21.2, 'CONTINUOUS · a finished slot is refilled on the next step', { fill: INKY }),
        ...Array.from({ length: 5 }, (_, row) => {
          let x = 3.5, out = '', k = 0;
          while (x < 3.5 + W - 1) {
            const seg = Math.min(3 + ((row * 7 + k * 5) % 11), 3.5 + W - x);
            out += g.bar(x, 22 + row * 1.35, 1.05, [{ w: Math.max(0.6, seg - 0.5), fill: INKY, label: '' }]);
            x += seg;
            k++;
          }
          return out;
        }),
        g.chip(3.5, 29 - 1.9, 'utilisation > 90%', { accent: INKY }),
        g.text(56.5, 27.85, 'gaps = a slot changing hands', { font: 'sub', fill: FAINT, anchor: 'end' }),
      ].join('');
    },
  },

  /* ---------------------------------------------------------------- 06 --- */
  6: {
    w: 58, h: 26,
    title: 'Four attacks on the batch equation',
    desc: 'Paging stops wasting allocated memory and is free. Prefix caching stops storing the ' +
      'same thing twice and is also free and exact. Architectural changes like GQA and MLA store ' +
      'less by design but must be trained in. Quantization stores the same thing in fewer bits ' +
      'and is the only one that costs quality. Stacking them takes concurrent sequences on four ' +
      'H100s from 14 to over 300, crossing the H100 ridge point of 295.',
    caption: 'Do the exact optimisations before the lossy ones. Paging alone is 4.2× and costs ' +
      'nothing; it would be perverse to spend accuracy while still discarding 75% of the cache.',
    draw: (g, id) => {
      const x0 = 34.5, maxW = 15, scale = maxW / 303;
      const rows = [
        ['naive alloc', 14, BLINKY], ['+ paging', 59, A], ['+ fp8 KV', 118, A],
        ['+ fp8 weights', 170, A], ['+ int4 weights', 193, A], ['MLA + fp8', 303, INKY],
      ];
      return [
        g.heading(1.5, 1.6, 'LEVEL 06 · FOUR ATTACKS ON ONE DENOMINATOR'),
        g.line(1.5, 2.4, 56.5, 2.4, { stroke: FAINT, width: 1 }),

        g.heading(1.5, 4.2, 'IN THIS ORDER — EXACT FIRST, LOSSY LAST'),
        ...[
          ['1  PagedAttention', 'stop wasting what you allocate', 'FREE', INKY],
          ['2  Prefix caching', 'stop storing the same prefix twice', 'FREE', INKY],
          ['3  GQA · MLA', 'store less per token, by design', 'TRAIN-TIME', CLYDE],
          ['4  Quantization', 'store the same thing in fewer bits', 'COSTS QUALITY', BLINKY],
        ].map((r, i) => [
          g.box(1.5, 5.4 + i * 3.6, 24.5, 2.9, null, { accent: r[3], fill: 'none' }),
          g.text(2.6, 6.4 + i * 3.6, r[0], { font: 'label', fill: 'var(--ink)' }),
          g.text(2.6, 7.7 + i * 3.6, r[1], { font: 'sub', fill: DIM }),
          g.text(23.5, 6.4 + i * 3.6, r[2], { font: 'tag', fill: r[3], anchor: 'end' }),
        ].join('')),

        g.heading(28.5, 4.2, 'SEQUENCES THAT FIT · 70B @ 8k · 4×H100'),
        ...rows.map((r, i) => [
          g.text(34, 6.6 + i * 2.3, r[0], { font: 'sub', fill: DIM, anchor: 'end' }),
          g.bar(x0, 5.9 + i * 2.3, 1.5, [{ w: Math.max(0.3, r[1] * scale), fill: r[2], label: '' }]),
          g.text(51, 6.6 + i * 2.3, `${r[1]}`, { font: 'num', fill: r[2] }),
        ].join('')),
        g.line(x0 + 295 * scale, 5.4, x0 + 295 * scale, 19.4,
          { stroke: PAC, dashed: true, width: 1.4 }),
        g.text(x0 + 295 * scale, 20.4, 'ridge 295', { font: 'tag', fill: PAC, anchor: 'middle' }),
        g.note(28.5, 22, ['only the last row crosses the ridge —', 'and it needs a different model.'],
          { fill: FAINT }),

        g.line(1.5, 23.6, 30, 23.6, { stroke: FAINT, width: 1 }),
        g.note(1.5, 24.8, ['Paging alone: 14 → 59 sequences, at zero quality cost.'], { fill: DIM }),
      ].join('');
    },
  },

  /* ---------------------------------------------------------------- 07 --- */
  7: {
    w: 58, h: 26,
    title: 'Why attention is decided by the memory hierarchy',
    desc: 'Shared memory has about six times the bandwidth of HBM but only 228 kilobytes per ' +
      'streaming multiprocessor. Naive attention writes and reads the N by N score matrix four ' +
      'times, which for an 8k prefill is 550 gigabytes of HBM traffic against only 35 teraFLOPs ' +
      'of arithmetic. FlashAttention keeps tiles in shared memory and never materialises the ' +
      'matrix, cutting traffic to single-digit gigabytes for the same arithmetic.',
    caption: 'The arithmetic never changed. FlashAttention removes an intermediate that never ' +
      'needed to exist in HBM — and it is exact, which is why it was adoptable at all.',
    draw: (g, id) => [
      g.heading(1.5, 1.6, 'LEVEL 07 · THE TIER YOU COMPUTE IN DECIDES THE SPEED'),
      g.line(1.5, 2.4, 56.5, 2.4, { stroke: FAINT, width: 1 }),

      g.heading(1.5, 4.2, 'THE HIERARCHY'),
      ...[
        ['registers', '256 KB/SM', '~100 TB/s', INKY],
        ['shared / L1', '228 KB/SM', '~20 TB/s', INKY],
        ['L2', '50 MB', '~10 TB/s', CLYDE],
        ['HBM3', '80 GB', '3.35 TB/s', BLINKY],
      ].map((r, i) => [
        g.box(1.5, 5.4 + i * 2.7, 21, 2.1, null, { accent: r[3], fill: 'none' }),
        g.text(2.4, 6.45 + i * 2.7, r[0], { font: 'num', fill: 'var(--ink)' }),
        g.text(11, 6.45 + i * 2.7, r[1], { font: 'sub', fill: FAINT }),
        g.text(21.6, 6.45 + i * 2.7, r[2], { font: 'sub', fill: r[3], anchor: 'end' }),
      ].join('')),
      g.note(1.5, 17, ['shared memory is ~6× HBM bandwidth.', 'That gap IS the algorithm.'],
        { fill: A }),

      g.line(24.5, 4, 24.5, 22.5, { stroke: FAINT, width: 1 }),

      g.heading(26.5, 4.2, 'HBM TRAFFIC · ONE 8k PREFILL, 32 LAYERS'),
      g.text(26.5, 6.6, 'naive', { font: 'num', fill: BLINKY }),
      g.bar(32, 5.9 + 0, 1.5, [{ w: 19, fill: BLINKY, label: '', ink: 'var(--on-maze)' }]),
      g.text(52, 6.6, '550 GB', { font: 'num', fill: BLINKY }),
      g.text(26.5, 9.4, 'flash', { font: 'num', fill: INKY }),
      g.bar(32, 8.7, 1.5, [{ w: 0.4, fill: INKY, label: '' }]),
      g.text(52, 9.4, '~9 GB', { font: 'num', fill: INKY }),
      g.note(26.5, 11.4, ['identical arithmetic either way: 35 TFLOP.', 'Only the byte traffic changed.'],
        { fill: DIM }),

      g.heading(26.5, 14.6, 'WHY IT IS EXACT'),
      g.text(26.5, 16.2, 'exp(a−m₁) · exp(m₁−m)  =  exp(a−m)', { font: 'num', fill: A }),
      g.note(26.5, 17.6, ['an identity, not an approximation — so it is', 'a drop-in for any trained checkpoint'],
        { fill: FAINT }),

      g.heading(26.5, 20.4, 'DECODE NEEDS A DIFFERENT KERNEL', { fill: CLYDE }),
      g.text(26.5, 21.9, 'q_len 1 → 32 work units for 132 SMs', { font: 'sub', fill: DIM }),
      g.text(26.5, 23.2, 'split KV into 16 → 512 units', { font: 'sub', fill: CLYDE }),

      g.line(1.5, 24.4, 56.5, 24.4, { stroke: FAINT, width: 1 }),
      g.note(1.5, 25.4, ['The backward pass RECOMPUTES attention rather than storing it — more FLOPs, less traffic, still faster.'],
        { fill: DIM }),
    ].join(''),
  },

  /* ---------------------------------------------------------------- 08 --- */
  8: {
    w: 58, h: 26,
    title: 'Draft and verify, and when it stops paying',
    desc: 'Standard decoding runs one target pass per token, each moving 15 gigabytes. ' +
      'Speculative decoding drafts several cheap tokens then verifies them all in one target ' +
      'pass, so three accepted tokens cost one weight read instead of three. The acceptance rule ' +
      'provably returns the target distribution. Expected tokens per verification is one minus ' +
      'alpha to the gamma plus one, over one minus alpha. Because verification uses idle ' +
      'arithmetic, it stops paying once batching has consumed that headroom, around batch 60.',
    caption: 'Speculation spends the arithmetic that memory-bound decoding leaves idle — which ' +
      'is exactly the headroom batching also wants. You rarely get both.',
    draw: (g, id) => [
      g.heading(1.5, 1.6, 'LEVEL 08 · BUYING BACK THE IDLE ARITHMETIC'),
      g.line(1.5, 2.4, 56.5, 2.4, { stroke: FAINT, width: 1 }),

      g.heading(1.5, 4.2, 'STANDARD · one target pass per token', { fill: FAINT }),
      ...Array.from({ length: 4 }, (_, i) =>
        g.box(1.5 + i * 6.4, 5.2, 5.6, 2.6, 'target', { sub: '15 GB', accent: FAINT })),
      g.text(27.5, 6.5, '= 4 tokens for 60 GB', { font: 'num', fill: FAINT }),

      g.heading(1.5, 10, 'SPECULATIVE · draft cheap, verify once', { fill: A }),
      ...Array.from({ length: 3 }, (_, i) =>
        g.box(1.5 + i * 3.3, 11, 2.9, 2.6, 'd', { sub: '0.5', accent: CLYDE })),
      g.text(11.6, 12.3, '→', { font: 'label', fill: DIM }),
      g.box(13.2, 11, 8.4, 2.6, 'VERIFY', { sub: 'one pass, 15 GB', accent: A }),
      g.text(22.6, 12.3, '✓ ✓ ✗', { font: 'label', fill: 'var(--ink)' }),
      g.text(27.5, 12.3, '= 3 tokens for 16.5 GB', { font: 'num', fill: A }),
      g.note(1.5, 15, ['verifying γ guesses costs ONE weight read — the extra positions ride in arithmetic that was idle'],
        { fill: DIM }),

      g.line(1.5, 17, 30, 17, { stroke: FAINT, width: 1 }),
      g.heading(1.5, 18.2, 'EXPECTED TOKENS PER VERIFICATION'),
      g.text(1.5, 19.8, 'E = (1 − αᵞ⁺¹) ÷ (1 − α)', { font: 'num', fill: A }),
      ...[['α=0.5', '2.0 max, ever'], ['α=0.7', '2.8 at γ=4'], ['α=0.9', '4.1 at γ=4']]
        .map((r, i) => [
          g.text(1.5, 21.4 + i * 1.4, r[0], { font: 'num', fill: DIM }),
          g.text(8, 21.4 + i * 1.4, r[1], { font: 'sub', fill: FAINT }),
        ].join('')),

      g.line(31.5, 17, 31.5, 25.4, { stroke: FAINT, width: 1 }),
      g.heading(33.5, 18.2, 'AND WHY IT STOPS PAYING', { fill: BLINKY }),
      ...[[1, 5], [16, 80], [32, 160], [64, 320]].map((r, i) => [
        g.text(33.5, 19.8 + i * 1.4, `batch ${r[0]}`, { font: 'sub', fill: DIM }),
        g.text(41, 19.8 + i * 1.4, `→ intensity ${r[1]}`,
          { font: 'num', fill: r[1] > 295 ? BLINKY : A }),
        g.text(51.5, 19.8 + i * 1.4, r[1] > 295 ? 'past ridge' : '',
          { font: 'sub', fill: BLINKY }),
      ].join('')),
      g.note(33.5, 25.4, ['batching wants the same idle FLOPs'], { fill: BLINKY }),
    ].join(''),
  },

  /* ---------------------------------------------------------------- 09 --- */
  9: {
    w: 58, h: 26,
    title: 'Tensor parallelism, pipeline parallelism, and the interconnect',
    desc: 'Tensor parallelism splits each layer across GPUs and costs two all-reduces per layer, ' +
      '160 per token for an 80-layer model, about 126 megabytes. Pipeline parallelism assigns ' +
      'whole layers to each GPU and passes one activation per boundary, about 1.6 megabytes, ' +
      'roughly 80 times less. NVLink runs at 900 gigabytes per second and PCIe at 64, a 14-fold ' +
      'gap, so the chatty strategy belongs inside a node and the quiet one across nodes.',
    caption: 'Every parallelism decision is a decision about which link carries the traffic. ' +
      'The links differ by more than 50×.',
    draw: (g, id) => [
      g.heading(1.5, 1.6, 'LEVEL 09 · WHICH LINK CARRIES THE TRAFFIC'),
      g.line(1.5, 2.4, 56.5, 2.4, { stroke: FAINT, width: 1 }),

      g.heading(1.5, 4.2, 'TENSOR PARALLEL · split inside the layer', { fill: BLINKY }),
      ...Array.from({ length: 4 }, (_, i) =>
        g.box(1.5 + i * 6.3, 5.4, 5.5, 3, `GPU ${i}`, { sub: '¼ layer', accent: BLINKY })),
      g.line(1.5, 9.2, 25.3, 9.2, { stroke: BLINKY, width: 1.6 }),
      ...Array.from({ length: 4 }, (_, i) => g.arrow(4.25 + i * 6.3, 8.5, 4.25 + i * 6.3, 9.1, { id })),
      g.text(1.5, 10.4, 'all-reduce ×2 per layer', { font: 'sub', fill: BLINKY }),
      g.chip(1.5, 11.4, '160 per token · 126 MB', { accent: BLINKY }),
      g.note(1.5, 13.6, ['needs NVLink. Over PCIe this', 'is 19% of your step time.'], { fill: FAINT }),

      g.line(27, 4, 27, 16.4, { stroke: FAINT, width: 1 }),

      g.heading(29, 4.2, 'PIPELINE PARALLEL · split across layers', { fill: INKY }),
      ...Array.from({ length: 4 }, (_, i) => [
        g.box(29 + i * 6.9, 5.4, 5.5, 3, `GPU ${i}`, { sub: `L${i * 20}–${i * 20 + 19}`, accent: INKY }),
        i < 3 ? g.arrow(34.6 + i * 6.9, 6.9, 35.8 + i * 6.9, 6.9, { id }) : '',
      ].join('')),
      g.text(29, 10.4, 'one activation per boundary', { font: 'sub', fill: INKY }),
      g.chip(29, 11.4, '3 per token · 1.6 MB', { accent: INKY }),
      g.note(29, 13.6, ['~80× less traffic, so it survives', 'InfiniBand or Ethernet.'], { fill: FAINT }),

      g.line(1.5, 17, 56.5, 17, { stroke: FAINT, width: 1 }),
      g.heading(1.5, 18.4, 'THE LADDER · LOG SCALE — the real ratios are far wider'),
      ...[['HBM3', 3350, A], ['NVLink 4', 900, INKY], ['PCIe Gen5', 64, CLYDE], ['100 GbE', 12, BLINKY]]
        .map((r, i) => [
          g.text(11, 20 + i * 1.5, r[0], { font: 'num', fill: DIM, anchor: 'end' }),
          g.bar(12, 19.4 + i * 1.5, 1.2,
            [{ w: Math.max(0.3, (Math.log10(r[1]) / Math.log10(3350)) * 26), fill: r[2], label: '' }]),
          g.text(39.5, 20 + i * 1.5, `${r[1]} GB/s`, { font: 'num', fill: r[2] }),
        ].join('')),
      g.note(46, 21.5, ['PCIe is 52× slower than', 'HBM, not 2× — chatty', 'strategies stay in-node.'], { fill: A }),
    ].join(''),
  },

  /* ---------------------------------------------------------------- 10 --- */
  10: {
    w: 58, h: 26,
    title: 'Engine philosophies and the benchmark that lies to you',
    desc: 'The four serving engines have converged on the same feature list, so what separates ' +
      'them is the operating point each was designed for. Separately: a closed-loop benchmark ' +
      'keeps a fixed number of requests in flight, so when the server slows it sends fewer and ' +
      'the curve never breaks. An open-loop benchmark sends requests on a fixed schedule, so ' +
      'goodput rises and then collapses at the capacity cliff — the number you actually needed.',
    caption: 'A closed-loop harness cannot show you overload: it throttles itself exactly when ' +
      'the system slows down. Only the open-loop curve has a cliff to find.',
    draw: (g, id) => {
      const X0 = 32, X1 = 55, Y0 = 7, Y1 = 17.5;
      return [
        g.heading(1.5, 1.6, 'LEVEL 10 · WHAT SEPARATES THEM, AND HOW TO MEASURE'),
        g.line(1.5, 2.4, 56.5, 2.4, { stroke: FAINT, width: 1 }),

        g.heading(1.5, 4.2, 'THE BET EACH ENGINE MAKES'),
        ...[
          ['vLLM', 'fix fragmentation and the rest follows', 'gives up: peak single-request latency', A],
          ['SGLang', 'real traffic is structured and shared', 'gives up: simplicity, generality', INKY],
          ['TensorRT-LLM', 'decide everything at build time', 'gives up: flexibility, new models', CLYDE],
          ['llama.cpp', 'one user, short context, tight memory', 'gives up: throughput at scale', PINKY],
        ].map((r, i) => [
          g.box(1.5, 5.4 + i * 3.2, 27, 2.6, null, { accent: r[3], fill: 'none' }),
          g.text(2.5, 6.3 + i * 3.2, r[0], { font: 'label', fill: r[3] }),
          g.text(11, 6.3 + i * 3.2, r[1], { font: 'sub', fill: 'var(--ink)' }),
          g.text(2.5, 7.5 + i * 3.2, r[2], { font: 'sub', fill: FAINT }),
        ].join('')),

        g.line(30, 4, 30, 22, { stroke: FAINT, width: 1 }),
        g.heading(32, 4.2, 'GOODPUT vs OFFERED LOAD'),
        g.line(X0, Y1, X1, Y1, { stroke: DIM, width: 1.4 }),
        g.line(X0, Y0, X0, Y1, { stroke: DIM, width: 1.4 }),
        g.text(X0 - 0.4, Y0 - 1.2, 'goodput', { font: 'sub', fill: DIM }),
        g.text((X0 + X1) / 2, Y1 + 1.4, 'requests / second  →', { font: 'sub', fill: DIM, anchor: 'middle' }),

        // closed loop: smooth, never breaks
        g.poly([[X0, Y1], [X0 + 6, Y1 - 4], [X0 + 12, Y1 - 6], [X0 + 18, Y1 - 6.8], [X1, Y1 - 7.2]],
          { stroke: FAINT, width: 2, dashed: true }),
        g.text(X1, Y1 - 8.2, 'closed loop', { font: 'sub', fill: FAINT, anchor: 'end' }),

        // open loop: rises then collapses
        g.poly([[X0, Y1], [X0 + 5, Y1 - 4.2], [X0 + 9, Y1 - 7], [X0 + 11, Y1 - 7.6],
                [X0 + 13, Y1 - 5], [X0 + 16, Y1 - 1.6], [X0 + 19, Y1 - 0.5]],
          { stroke: BLINKY, width: 2.4 }),
        g.dot(X0 + 11, Y1 - 7.6, 4.5, BLINKY),
        g.text(X0 + 11, Y1 - 9, 'THE CLIFF', { font: 'tag', fill: BLINKY, anchor: 'middle' }),
        g.text(X0 + 19.5, Y1 - 1.6, 'open loop', { font: 'sub', fill: BLINKY }),

        g.note(32, 20.8, ['closed loop hides the cliff: it stops sending',
          'exactly when the system starts to struggle.'], { fill: DIM }),

        g.line(1.5, 22.8, 56.5, 22.8, { stroke: FAINT, width: 1 }),
        g.note(1.5, 24, [
          'Report p50/p95/p99, not means. State batch size and length distribution, or the number means nothing.',
          'Goodput counts only the requests that met their SLO — the rest were produced for free and sold for nothing.',
        ], { fill: DIM }),
      ].join('');
    },
  },
};
