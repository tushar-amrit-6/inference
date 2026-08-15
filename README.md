# LLM Inference — from scratch, one token at a time

A self-paced course on how LLM inference works and why it is hard, built as a static site with a
Pac-Man arcade theme.

Thirteen levels, from the transformer forward pass to distributed serving, the architecture zoo,
and inside vLLM itself, organised around one idea:

> Generating a token requires reading **every** model weight from memory. Moving those bytes takes
> far longer than the arithmetic performed on them, so decoding a single sequence leaves the GPU
> almost entirely idle. Nearly every technique in the field is an answer to one question: **how do
> we do more useful work per byte moved from memory?**

Pac-Man is memory-bound too — one pellet per move, however fast he runs.

## The levels

| # | Level | Hours |
|---|---|---|
| 00 | Transformer fundamentals | 8–12 |
| 01 | The autoregressive loop | 4–6 |
| 02 | The KV cache | 6–8 |
| 03 | Decoding and sampling | 5–7 |
| 04 | Metrics and the roofline | 8–10 |
| 05 | Batching | 5–7 |
| 06 | Memory optimization | 8–10 |
| 07 | Attention kernels | 7–9 |
| 08 | Speculative decoding | 6–8 |
| 09 | Distributed inference | 7–9 |
| 10 | Systems and the frontier | 6–8 |
| 11 | Model architectures | 7–9 |
| 12 | How vLLM works | 6–8 |

Plus reference pages: a GPU spec table with ridge points, a timeline of the field, a serving-engine
comparison, open problems, and a combined glossary.

Every level has the same seven parts, colour-coded by which Pac-Man ghost owns them — and the
roles match how each ghost actually behaves in the arcade game:

| Marker | Part | Why that ghost |
|---|---|---|
| **Pinky** (pink) | The big idea | They target four tiles *ahead* of you |
| **Pellets** (peach) | Concepts | Eaten one at a time |
| **Inky** (cyan) | Math by hand | Their target is *computed*, not chased |
| **Clyde** (orange) | Code lab | Chases, then retreats to their corner |
| **Blinky** (red) | Pitfalls | They come straight at you |
| **Power pellet** | Checkpoint | Eat it and the ghosts turn blue |

Every level opens with a **grid-notes summary figure** — the whole level as one diagram on graph
paper: the residual stream's shape at every step, the KV-cache formula built up term by term, the
roofline with prefill and decode plotted on it, the draft-and-verify timeline. They are SVG, so
they re-theme with the site rather than being baked images.

## Running it

Everything is static. Open `index.html`, or serve the directory:

```bash
python3 -m http.server 8000
```

## Building

The HTML is generated from the data files and committed, so the site works on GitHub Pages with no
build step. To regenerate after editing content:

```bash
node build/build.mjs
```

No dependencies and no install step for the build — Node 18+ is the only requirement.

The build refuses to ship a broken figure: it asserts every level has a tile on the map and that
every node is reachable from Pac-Man's start tile, and that the two copies of the light palette (one under
`prefers-color-scheme`, one under `[data-theme]`) have not drifted apart. Diagram labels are
checked separately, because estimating text extents from character counts is not trustworthy:

```bash
npm install          # playwright, the only dev dependency
npm run check:figures
```

That loads each page in a real browser, measures the actual `getBBox` of every label, and fails
on any overlap or anything drawn outside the canvas.

```
data/modules/m*.mjs    course content, one file per level
data/reference.mjs     timeline, hardware table, engines, frontier, how-to
data/diagrams.mjs      the per-level summary figures
build/build.mjs        the generator
build/md.mjs           a small dependency-free Markdown renderer
build/maze.mjs         the maze, rendered to static SVG from a tile grid
build/diagram.mjs      grid-paper drawing primitives, authored in grid cells
build/check-figures.mjs  measures rendered label boxes and fails on collisions
assets/css/arcade.css  the design system
assets/js/app.js       progress tracking, scroll spy, keyboard nav
start.md               the original course outline this was built from
```

The maze on the landing page is authored as a 21×19 tile grid in `build/maze.mjs`, the same way the
real game does it, and rendered to SVG at build time — so it works with JavaScript disabled.
Progress is stored in `localStorage`; nothing is sent anywhere and there is no account. Arrow keys
move between levels.

## A note on accuracy

This is a teaching resource built around arithmetic, so the arithmetic needs to be right. Every
number is meant to be re-derived rather than trusted — the math labs exist precisely so you check
them yourself.

Two caveats worth stating plainly:

- **Content was written from the author's knowledge and not verified against live sources at build
  time.** The foundational papers are stable; hardware specifications, engine features, and the
  frontier discussion will age. Verify anything load-bearing against a current datasheet or the
  paper itself.
- **Hardware figures are working estimates.** Blackwell and MI300X numbers in particular should be
  checked against a current vendor datasheet. Bandwidth figures are the most stable and the most
  important — for decode, bandwidth is essentially the only spec that matters.

If you find an error, the fastest way to confirm it is to do the multiplication.
