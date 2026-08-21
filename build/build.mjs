#!/usr/bin/env node
/* ==========================================================================
   Static site generator for LLM INFERENCE.

     node build/build.mjs

   No dependencies, no install step. Reads data/, writes HTML to the repo
   root so GitHub Pages can serve it directly.
   ========================================================================== */

import { readdir, readFile, writeFile, mkdir } from 'node:fs/promises';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';

import { md, esc, plain } from './md.mjs';
import { renderMaze, mazeData, assertReachable, icon } from './maze.mjs';
import { figure, kit } from './diagram.mjs';

const { default: diagrams } = await import('../data/diagrams.mjs');

const HERE = dirname(fileURLToPath(import.meta.url));
const ROOT = join(HERE, '..');

const SITE = {
  title: 'LLM INFERENCE',
  subtitle: 'From scratch, one token at a time',
  repo: 'https://github.com/tushar-amrit-6/inference',
};

/* --- ghost-coded section identities ---------------------------------------
   Each ghost's role matches its actual behaviour in the arcade game. The
   colour is the wayfinding: learn it once, scan any page. */
const GHOSTS = {
  bigIdea:    { key: 'pinky',  tag: 'PINKY · THE BIG IDEA',  note: 'targets four tiles ahead of you' },
  concepts:   { key: 'pellet', tag: 'PELLETS · CONCEPTS',    note: 'eat them one at a time' },
  math:       { key: 'inky',   tag: 'INKY · MATH BY HAND',   note: 'their target is computed' },
  code:       { key: 'clyde',  tag: 'CLYDE · CODE LAB',      note: 'go hands-on, then retreat' },
  papers:     { key: 'pac',    tag: 'THE KEY · PAPERS',      note: 'unlocks the literature' },
  pitfalls:   { key: 'blinky', tag: 'BLINKY · PITFALLS',     note: 'they chase you directly' },
  checkpoint: { key: 'pac',    tag: 'POWER PELLET · CHECKPOINT', note: 'eat it and the ghosts turn blue' },
  glossary:   { key: 'pellet', tag: 'GLOSSARY',              note: '' },
};

const GHOST_ICON = {
  pinky: icon.ghost('var(--pinky)'),
  inky: icon.ghost('var(--inky)'),
  clyde: icon.ghost('var(--clyde)'),
  blinky: icon.ghost('var(--blinky)'),
  pac: icon.power(),
  pellet: icon.pellet(),
};

/* --- page shell ------------------------------------------------------------ */

function layout({ title, description, body, depth = 0, bodyClass = '' }) {
  const up = depth ? '../'.repeat(depth) : '';
  return `<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>${esc(title)}</title>
<meta name="description" content="${esc(description)}">
<meta name="color-scheme" content="dark light">
<script>/* set the stored theme before first paint, so there is no flash of the wrong one */
(function(){try{var t=localStorage.getItem('llm-inference:theme');
if(t==='light'||t==='dark')document.documentElement.setAttribute('data-theme',t);}catch(e){}})();</script>
<meta property="og:title" content="${esc(title)}">
<meta property="og:description" content="${esc(description)}">
<meta property="og:type" content="website">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Press+Start+2P&family=IBM+Plex+Sans:ital,wght@0,300;0,400;0,600;1,400&family=IBM+Plex+Mono:wght@400;600&display=swap" rel="stylesheet">
<link rel="stylesheet" href="${up}assets/css/arcade.css">
<link rel="icon" href="data:image/svg+xml,${encodeURIComponent(
    '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 32 32"><rect width="32" height="32" fill="#000"/><path d="M16,16 L29,9 A15,15 0 1 0 29,23 Z" fill="#FFE800"/></svg>'
  )}">
</head>
<body${bodyClass ? ` class="${bodyClass}"` : ''}>
<a class="skip" href="#main">Skip to content</a>
${body}
<script src="${up}assets/js/app.js" defer></script>
</body>
</html>`;
}

function scorebar(up, { level } = {}) {
  return `<header class="scorebar">
  <a class="scorebar__home" href="${up}index.html">${icon.pac(14)} LLM INFERENCE</a>
  ${level != null ? `<span class="scorebar__item"><span class="scorebar__label">LEVEL</span><span class="scorebar__value scorebar__value--pac">${String(level).padStart(2, '0')}</span></span>` : ''}
  <span class="scorebar__item"><span class="scorebar__label">SCORE</span><span class="scorebar__value" data-score>000000</span></span>
  <span class="scorebar__item"><span class="scorebar__label">CLEARED</span><span class="scorebar__value" data-cleared>00/15</span></span>
  <span class="scorebar__spacer"></span>
  <span class="scorebar__item scorebar__lives" data-lives aria-label="Modules remaining"></span>
  <button class="theme-toggle" type="button" data-theme-toggle aria-label="Switch theme">
    <svg class="icon-moon" width="15" height="15" viewBox="0 0 16 16" fill="currentColor" aria-hidden="true"><path d="M6.2 1.4A6.6 6.6 0 1 0 14.6 9.8 5.2 5.2 0 0 1 6.2 1.4z"/></svg>
    <svg class="icon-sun" width="15" height="15" viewBox="0 0 16 16" fill="currentColor" aria-hidden="true"><circle cx="8" cy="8" r="3.2"/><path d="M8 .8v2.1M8 13.1v2.1M.8 8h2.1M13.1 8h2.1M2.9 2.9l1.5 1.5M11.6 11.6l1.5 1.5M13.1 2.9l-1.5 1.5M4.4 11.6l-1.5 1.5" stroke="currentColor" stroke-width="1.4" stroke-linecap="round"/></svg>
  </button>
</header>`;
}

function footer(up, modules) {
  return `<footer class="foot">
  <div class="wrap">
    <div class="foot__grid">
      <div>
        <p class="foot__h">LEVELS</p>
        <ul>${modules.slice(0, 8).map((m) => `<li><a href="${up}modules/${m.slug}.html">${String(m.n).padStart(2, '0')} · ${esc(m.title)}</a></li>`).join('')}</ul>
      </div>
      <div>
        <p class="foot__h">&nbsp;</p>
        <ul>${modules.slice(8).map((m) => `<li><a href="${up}modules/${m.slug}.html">${String(m.n).padStart(2, '0')} · ${esc(m.title)}</a></li>`).join('')}</ul>
      </div>
      <div>
        <p class="foot__h">REFERENCE</p>
        <ul>
          <li><a href="${up}reference/timeline.html">Timeline</a></li>
          <li><a href="${up}reference/hardware.html">GPU spec table</a></li>
          <li><a href="${up}reference/engines.html">Serving engines</a></li>
          <li><a href="${up}reference/frontier.html">Open problems</a></li>
          <li><a href="${up}reference/glossary.html">Glossary</a></li>
        </ul>
      </div>
      <div>
        <p class="foot__h">SOURCE</p>
        <ul>
          <li><a href="${SITE.repo}">GitHub repository</a></li>
          <li><a href="${up}reference/how-to-use.html">How to work through this</a></li>
        </ul>
      </div>
    </div>
    <p class="foot__note">Built as a self-paced set of chapters. Every number here is meant to be
    re-derived rather than trusted — if one of them is wrong, the fastest way to find out is to
    do the arithmetic yourself.</p>
  </div>
</footer>`;
}

function ghostTag(kind) {
  const g = GHOSTS[kind];
  return `<p class="ghost__tag">${GHOST_ICON[g.key]}${g.tag}</p>`;
}

/* --- module page ----------------------------------------------------------- */

function railFor(m, modules) {
  const items = [
    { id: 'big-idea', label: 'The big idea', kind: 'dot' },
    ...m.concepts.map((c, i) => ({ id: `concept-${i}`, label: c.name, kind: 'dot' })),
    { id: 'math-lab', label: 'Math by hand', kind: 'dot' },
    { id: 'code-lab', label: 'Code lab', kind: 'dot' },
    { id: 'papers', label: 'Papers', kind: 'dot' },
    { id: 'pitfalls', label: 'Pitfalls', kind: 'dot' },
    { id: 'checkpoint', label: 'Checkpoint', kind: 'power' },
    { id: 'glossary', label: 'Glossary', kind: 'dot' },
  ];
  const prev = modules.find((x) => x.n === m.n - 1);
  const next = modules.find((x) => x.n === m.n + 1);

  return `<nav class="rail" aria-label="Module contents">
  <p class="rail__head">ON THIS LEVEL</p>
  <ul class="rail__list">
    ${items.map((it) => `<li><a class="rail__link" href="#${it.id}" data-kind="${it.kind}"><span class="rail__dot"></span><span>${esc(it.label)}</span></a></li>`).join('')}
  </ul>
  <div class="rail__nav">
    ${prev ? `<a class="rail__step" href="${prev.slug}.html">◀ ${String(prev.n).padStart(2, '0')} ${esc(prev.title)}</a>` : ''}
    <a class="rail__step" href="../index.html">▲ ALL LEVELS</a>
    ${next ? `<a class="rail__step" href="${next.slug}.html">${String(next.n).padStart(2, '0')} ${esc(next.title)} ▶</a>` : ''}
  </div>
</nav>`;
}

function conceptBlock(c, i) {
  return `<article class="concept" id="concept-${i}">
  <div class="concept__head">
    <span class="concept__n">${String(i + 1).padStart(2, '0')}</span>
    <h3 class="concept__title">${esc(c.name)}</h3>
  </div>
  <div class="prose">${md(c.body)}</div>
  ${c.ascii && c.ascii.trim() ? `<pre class="ascii" aria-label="diagram">${esc(c.ascii)}</pre>` : ''}
  ${c.keyPoint ? `<p class="keypoint"><b>REMEMBER</b>${esc(c.keyPoint)}</p>` : ''}
</article>`;
}

function modulePage(m, modules) {
  const prev = modules.find((x) => x.n === m.n - 1);
  const next = modules.find((x) => x.n === m.n + 1);

  const body = `${scorebar('../', { level: m.n })}
<div class="wrap">
  <div class="shell">
    ${railFor(m, modules)}
    <main id="main">

      <div class="mod-head">
        <p class="mod-head__level">LEVEL ${String(m.n).padStart(2, '0')}</p>
        <h1 class="mod-head__title">${esc(m.title)}</h1>
        <p class="mod-head__tag">${esc(m.tagline)}</p>
        <div class="mod-head__meta">
          <span>⏱ ${esc(m.hours)}</span>
          <span>● ${m.concepts.length} concepts</span>
          <span>📄 ${m.papers.length} papers</span>
          ${m.prereqs?.length ? `<span>▸ needs: ${m.prereqs.map(esc).join(', ')}</span>` : ''}
        </div>
      </div>

      ${diagrams[m.n] ? `<section class="section" id="at-a-glance">
        <p class="eyebrow" style="margin-bottom:var(--s5)">AT A GLANCE · THE WHOLE LEVEL ON ONE PAGE</p>
        ${figure({
          w: diagrams[m.n].w,
          h: diagrams[m.n].h,
          id: `fig${m.n}`,
          title: diagrams[m.n].title,
          desc: diagrams[m.n].desc,
          caption: diagrams[m.n].caption,
          body: diagrams[m.n].draw(kit, `fig${m.n}`),
        })}
      </section>` : ''}

      <section class="section ghost ghost--pinky" id="big-idea">
        ${ghostTag('bigIdea')}
        <div class="prose">${md(m.bigIdea)}</div>
      </section>

      <section class="section ghost ghost--pellet" id="concepts">
        ${ghostTag('concepts')}
        ${m.concepts.map(conceptBlock).join('\n')}
      </section>

      <section class="section ghost ghost--inky" id="math-lab">
        ${ghostTag('math')}
        <h2 class="section__title">Math by hand</h2>
        <div class="prose">${md(m.mathLab.prompt)}</div>
        <details class="spoiler">
          <summary>SHOW THE WORKED SOLUTION</summary>
          <div class="spoiler__body prose">${md(m.mathLab.solution)}</div>
        </details>
      </section>

      <section class="section ghost ghost--clyde" id="code-lab">
        ${ghostTag('code')}
        <h2 class="section__title">Code lab</h2>
        <div class="prose">${md(m.codeLab.goal)}</div>
        <div class="prose" style="max-width:none"><pre><code class="lang-python">${esc(m.codeLab.code)}</code></pre></div>
        <div class="prose"><h3>What you should see</h3>${md(m.codeLab.expect)}</div>
        <div class="prose"><h3>Stretch</h3>${md(m.codeLab.stretch)}</div>
      </section>

      <section class="section ghost ghost--pac" id="papers">
        ${ghostTag('papers')}
        <h2 class="section__title">Papers, with a reading frame</h2>
        <div class="papers">
          ${m.papers.map((p) => `<article class="paper">
            <h3 class="paper__title"><a href="${esc(p.url)}" target="_blank" rel="noopener">${esc(p.title)}</a></h3>
            <p class="paper__by">${esc(p.by)}</p>
            <p class="paper__why">${esc(p.why)}</p>
            <div class="paper__frame">${md(p.frame)}</div>
          </article>`).join('\n')}
        </div>
      </section>

      <section class="section ghost ghost--blinky" id="pitfalls">
        ${ghostTag('pitfalls')}
        <h2 class="section__title">What people get wrong</h2>
        ${m.pitfalls.map((p) => `<div class="pitfall">
          <p class="pitfall__wrong"><span>${esc(p.wrong)}</span></p>
          <div class="pitfall__right"><div class="prose" style="max-width:none">${md(p.right)}</div></div>
        </div>`).join('\n')}
      </section>

      <section class="section" id="checkpoint">
        <div class="wall checkpoint">
          ${ghostTag('checkpoint')}
          <p class="checkpoint__claim">${esc(m.checkpoint.claim)}</p>
          ${m.checkpoint.questions.map((q) => `<details class="qa">
            <summary>${esc(q.q)}</summary>
            <div class="qa__a prose">${md(q.a)}</div>
          </details>`).join('\n')}
          <p style="margin-top:var(--s7)">
            <button class="done-toggle" type="button" data-complete="${m.n}" aria-pressed="false">
              MARK LEVEL ${String(m.n).padStart(2, '0')} CLEARED
            </button>
          </p>
        </div>
      </section>

      <section class="section ghost ghost--pellet" id="glossary">
        ${ghostTag('glossary')}
        <h2 class="section__title">Glossary</h2>
        <div class="gloss">
          ${m.glossary.map((g) => `<div class="gloss__row"><div class="gloss__term">${esc(g.term)}</div><div class="gloss__def">${esc(g.def)}</div></div>`).join('')}
        </div>
      </section>

      <nav class="pagenav">
        ${prev ? `<a href="${prev.slug}.html" data-nav="prev"><span>◀ PREVIOUS LEVEL</span>${String(prev.n).padStart(2, '0')} ${esc(prev.title)}</a>` : '<span></span>'}
        ${next ? `<a href="${next.slug}.html" data-nav="next" style="text-align:right"><span>NEXT LEVEL ▶</span>${String(next.n).padStart(2, '0')} ${esc(next.title)}</a>` : '<span></span>'}
      </nav>

    </main>
  </div>
</div>
${footer('../', modules)}`;

  return layout({
    title: `${String(m.n).padStart(2, '0')} · ${m.title} — ${SITE.title}`,
    description: plain(m.tagline, 175),
    body,
    depth: 1,
  });
}

/* --- landing page ---------------------------------------------------------- */

function indexPage(modules) {
  const nodes = modules.map((m) => ({
    n: m.n,
    href: `modules/${m.slug}.html`,
    title: m.title,
    slug: m.slug,
  }));

  const legend = [
    ['pinky', 'PINKY', 'The big idea', 'They target four tiles ahead of you.'],
    ['pellet', 'PELLETS', 'Concepts', 'Eaten one at a time.'],
    ['inky', 'INKY', 'Math by hand', 'Their target is computed, not chased.'],
    ['clyde', 'CLYDE', 'Code lab', 'Chases, then retreats to their corner.'],
    ['blinky', 'BLINKY', 'Pitfalls', 'They come straight at you.'],
    ['pac', 'POWER PELLET', 'Checkpoint', 'Eat it and the ghosts turn blue.'],
  ];

  const body = `${scorebar('')}
<div class="wrap">
  <main id="main">

    <section class="hero">
      <div class="hero__grid">
        <div>
          <p class="eyebrow hero__kicker">FIFTEEN CHAPTERS</p>
          <h1 class="hero__title">LLM<br>INFERENCE</h1>
          <p class="hero__thesis">Pac-Man is <b>memory-bound</b>. One pellet per move, however fast he runs.</p>
          <p class="hero__body">So is your GPU. Generating a single token means reading
          <strong>every weight in the model</strong> out of memory — 15 GB for an 8B model — to do
          about 16 GFLOP of arithmetic with them. On an H100 that is roughly
          <strong>0.4% of the machine's arithmetic capability</strong>; the other 99.6% of the time
          it is sitting still, waiting on the memory bus.</p>
          <p class="hero__body">Batching, quantization, GQA, PagedAttention, FlashAttention,
          speculative decoding — every technique in these chapters is an answer to one question:
          <strong>how do we do more useful work per byte moved from memory?</strong> Hold that in
          view and the field stops looking like a pile of tricks.</p>
          <div class="hero__cta">
            <!-- Plays the maze when JS is available; falls back to level 00
                 without it. A link rather than a JS-revealed button, so the
                 primary CTA never pops in after load. -->
            <a class="btn" href="modules/${modules[0].slug}.html" data-game-play>${icon.pac(14)} START</a>
            <a class="btn btn--ghost" href="reference/how-to-use.html">HOW TO WORK THROUGH IT</a>
            <span class="coin" data-resume hidden></span>
          </div>
        </div>

        <aside class="gauge wall" aria-label="Where the time goes in one decode step">
          <p class="gauge__h">ONE DECODE STEP</p>
          <p class="gauge__sub">Llama-3-8B · fp16 · batch 1 · H100</p>
          <div class="gauge__row">
            <span class="gauge__label">MEMORY BUS</span>
            <span class="gauge__bar"><i style="width:100%"></i></span>
            <span class="gauge__val">4.48 ms</span>
          </div>
          <div class="gauge__row">
            <span class="gauge__label">TENSOR CORES</span>
            <span class="gauge__bar"><i class="gauge__thin" style="width:0.4%"></i></span>
            <span class="gauge__val">0.02 ms</span>
          </div>
          <p class="gauge__foot">15.0 GB moved, to do 16.1 GFLOP. The arithmetic is finished in
          <b>0.4%</b> of the time the bytes take. Everything here is about closing that gap.</p>
        </aside>
      </div>
    </section>

    <section class="section">
      <p class="eyebrow" style="margin-bottom:var(--s5)">THE MAZE · PICK A LEVEL</p>
      <div class="wall maze-frame">
        ${renderMaze(nodes)}
      </div>

      <div class="game-bar">
        <button class="btn game-start" type="button" data-game-start hidden>▶ START</button>
        <div class="game-bar__text">
          <span class="px px-10" data-maze-readout style="color:var(--pac)">HOVER A NODE TO SEE THE LEVEL</span>
          <span class="px px-10 game-hint" data-game-status></span>
        </div>
      </div>
      <p class="dim" style="margin-top:var(--s4);font-size:14.5px;max-width:70ch">
        Or play it. Press start — or just tap left or right while the maze is on screen — and
        Pac-Man finds a level on his own; take over with the arrow keys, WASD or a swipe to pick
        one yourself. Whichever numbered node he eats is the level that opens. The ghosts doze
        until you take control — after that they will send you back to the start, and the power
        pellets in the bottom corners make them edible. Clicking a node still just opens it.
      </p>

      <ul class="maze-legend">
        ${legend.map(([k, name, role, note]) => `<li>${GHOST_ICON[k]}<span><b style="--key:var(--${k === 'pellet' ? 'pellet' : k === 'pac' ? 'pac' : k})">${name}</b>${esc(role)} — ${esc(note)}</span></li>`).join('')}
      </ul>
      <p class="dim" style="margin-top:var(--s5);font-size:14.5px;max-width:70ch">
        The colours are the navigation. Every level uses the same six, so once you know that pink
        means "the big idea" and red means "here is what people get wrong", you can scan any page
        without reading it.
      </p>
    </section>

    <section class="section">
      <p class="eyebrow" style="margin-bottom:var(--s5)">LEVEL SELECT</p>
      <ol class="levels">
        ${modules.map((m) => `<li><a class="level" href="modules/${m.slug}.html" data-level="${m.n}">
          <span class="level__num">${String(m.n).padStart(2, '0')}</span>
          <span>
            <span class="level__title">${esc(m.title)}</span>
            <span class="level__tag">${esc(m.tagline)}</span>
          </span>
          <span class="level__meta">${esc(m.hours)}<br>${m.concepts.length} concepts</span>
        </a></li>`).join('')}
      </ol>
    </section>

    <section class="section">
      <p class="eyebrow" style="margin-bottom:var(--s5)">REFERENCE · ALWAYS OPEN IN A SECOND TAB</p>
      <ol class="levels">
        <li><a class="level" href="reference/hardware.html">
          <span class="level__num">GPU</span>
          <span><span class="level__title">HARDWARE TABLE</span>
          <span class="level__tag">Bandwidth, dense FLOPs and ridge points for the accelerators you will actually meet. You need this from Level 04 onward.</span></span>
          <span class="level__meta">reference</span></a></li>
        <li><a class="level" href="reference/timeline.html">
          <span class="level__num">📅</span>
          <span><span class="level__title">TIMELINE</span>
          <span class="level__tag">The papers and systems that changed how inference is done, 2017 to now, each tied back to the bandwidth thesis.</span></span>
          <span class="level__meta">reference</span></a></li>
        <li><a class="level" href="reference/engines.html">
          <span class="level__num">⚙</span>
          <span><span class="level__title">SERVING ENGINES</span>
          <span class="level__tag">vLLM, SGLang, TensorRT-LLM, llama.cpp — what each one optimizes for and what it gives up.</span></span>
          <span class="level__meta">reference</span></a></li>
        <li><a class="level" href="reference/frontier.html">
          <span class="level__num">?</span>
          <span><span class="level__title">OPEN PROBLEMS</span>
          <span class="level__tag">What is genuinely unsolved. Read this last, or read it first to know where you are heading.</span></span>
          <span class="level__meta">reference</span></a></li>
        <li><a class="level" href="reference/glossary.html">
          <span class="level__num">A–Z</span>
          <span><span class="level__title">GLOSSARY</span>
          <span class="level__tag">Every term defined across all sixteen levels, in one place.</span></span>
          <span class="level__meta">reference</span></a></li>
      </ol>
    </section>

  </main>
</div>
${footer('', modules)}
<script type="application/json" id="maze-data">${JSON.stringify(mazeData(nodes)).replace(/</g, '\\u003c')}</script>
<script src="assets/js/maze-game.js" defer></script>`;

  return layout({
    title: `${SITE.title} — ${SITE.subtitle}`,
    description:
      'A self-paced set of chapters on how LLM inference works and why it is hard. Sixteen levels, from the transformer forward pass to distributed serving, the architecture zoo, inside vLLM itself, down to the silicon, and cross-model KV cache transfer, organized around one idea: decoding is memory-bound.',
    body,
  });
}

/* --- reference pages -------------------------------------------------------- */

function refPage({ slug, title, eyebrow, lede, bodyHtml }, modules) {
  const body = `${scorebar('../')}
<div class="wrap wrap--narrow">
  <main id="main">
    <div class="mod-head">
      <p class="mod-head__level">${esc(eyebrow)}</p>
      <h1 class="mod-head__title">${esc(title)}</h1>
      ${lede ? `<p class="mod-head__tag">${esc(lede)}</p>` : ''}
    </div>
    ${bodyHtml}
  </main>
</div>
${footer('../', modules)}`;
  return layout({ title: `${title} — ${SITE.title}`, description: plain(lede, 175), body, depth: 1 });
}

function timelinePage(ref, modules) {
  const html = `<section class="section">
    <div class="prose" style="margin-bottom:var(--s8)">${md(ref.timelineIntro)}</div>
    <ol class="tl">
      ${ref.timeline.map((t) => `<li class="tl__item">
        <p class="tl__date">${esc(t.date)}</p>
        <h2 class="tl__name">${t.url ? `<a href="${esc(t.url)}" target="_blank" rel="noopener">${esc(t.name)}</a>` : esc(t.name)}</h2>
        <p class="tl__what">${esc(t.what)}</p>
        <p class="tl__why">${esc(t.why)}</p>
      </li>`).join('')}
    </ol>
  </section>`;
  return refPage({ slug: 'timeline', title: 'TIMELINE', eyebrow: 'REFERENCE', lede: ref.timelineLede, bodyHtml: html }, modules);
}

function glossaryPage(modules) {
  const all = new Map();
  for (const m of modules) {
    for (const g of m.glossary) {
      if (!all.has(g.term.toLowerCase())) all.set(g.term.toLowerCase(), { ...g, n: m.n, slug: m.slug });
    }
  }
  const sorted = [...all.values()].sort((a, b) => a.term.localeCompare(b.term, 'en', { sensitivity: 'base' }));
  const html = `<section class="section">
    <div class="gloss">
      ${sorted.map((g) => `<div class="gloss__row">
        <div class="gloss__term">${esc(g.term)}<br><a href="../modules/${g.slug}.html" style="font-family:var(--display);font-size:8px;color:var(--ink-faint);text-decoration:none">LEVEL ${String(g.n).padStart(2, '0')}</a></div>
        <div class="gloss__def">${esc(g.def)}</div>
      </div>`).join('')}
    </div>
  </section>`;
  return refPage({
    slug: 'glossary', title: 'GLOSSARY', eyebrow: 'REFERENCE',
    lede: `${sorted.length} terms from across all sixteen levels, each linked back to where it is explained.`,
    bodyHtml: html,
  }, modules);
}

function proseRefPage(cfg, modules) {
  return refPage({ ...cfg, bodyHtml: `<section class="section"><div class="prose" style="max-width:none">${md(cfg.markdown)}</div></section>` }, modules);
}

/* --- guards ------------------------------------------------------------------ */

/**
 * The light palette is written twice — once under `prefers-color-scheme` and
 * once under an explicit `[data-theme="light"]` — because CSS cannot share one
 * declaration block between a media query and a plain selector. Nothing stops
 * the two drifting apart, and a drift would only show up as one theme path
 * looking subtly wrong. So the build compares them and fails on any mismatch.
 */
async function assertThemeParity() {
  const css = await readFile(join(ROOT, 'assets', 'css', 'arcade.css'), 'utf8');
  const tokens = (src) =>
    [...src.matchAll(/(--[\w-]+):\s*([^;]+);/g)]
      .map((m) => `${m[1]}=${m[2].trim().replace(/\s*\/\*.*$/, '').trim()}`);

  const media = css.match(/:root:not\(\[data-theme="dark"\]\)\s*\{([\s\S]*?)\n {2}\}/);
  const explicit = css.match(/:root\[data-theme="light"\]\s*\{([\s\S]*?)\n\}/);
  if (!media || !explicit) throw new Error('theme: could not find both light palette blocks');

  const a = tokens(media[1]);
  const b = tokens(explicit[1]);
  const setA = new Set(a);
  const setB = new Set(b);
  const onlyA = a.filter((t) => !setB.has(t));
  const onlyB = b.filter((t) => !setA.has(t));

  if (onlyA.length || onlyB.length) {
    throw new Error(
      'theme: the two light palettes have drifted apart.\n' +
      (onlyA.length ? `  only under prefers-color-scheme: ${onlyA.join(', ')}\n` : '') +
      (onlyB.length ? `  only under [data-theme="light"]: ${onlyB.join(', ')}\n` : '')
    );
  }
  return a.length;
}

/* --- main ------------------------------------------------------------------- */

async function main() {
  const dir = join(ROOT, 'data', 'modules');
  const files = (await readdir(dir)).filter((f) => /^m\d+\.mjs$/.test(f));

  const modules = [];
  for (const f of files) {
    const mod = await import(join(dir, f));
    modules.push(mod.default);
  }
  modules.sort((a, b) => a.n - b.n);

  if (!modules.length) throw new Error('no modules found in data/modules');

  const { default: ref } = await import(join(ROOT, 'data', 'reference.mjs'));

  // fail the build rather than ship a level the game cannot drive to
  const { walkable, nodes } = assertReachable(modules);
  console.log(`\n  maze: ${walkable} walkable tiles, all ${nodes} nodes reachable`);

  const themeTokens = await assertThemeParity();
  console.log(`  theme: light palette has ${themeTokens} tokens, both copies in step`);

  await mkdir(join(ROOT, 'modules'), { recursive: true });
  await mkdir(join(ROOT, 'reference'), { recursive: true });

  const written = [];
  const put = async (rel, html) => {
    await writeFile(join(ROOT, rel), html, 'utf8');
    written.push([rel, html.length]);
  };

  await put('index.html', indexPage(modules));
  for (const m of modules) await put(`modules/${m.slug}.html`, modulePage(m, modules));

  await put('reference/timeline.html', timelinePage(ref, modules));
  await put('reference/glossary.html', glossaryPage(modules));
  await put('reference/hardware.html', proseRefPage({
    title: 'HARDWARE TABLE', eyebrow: 'REFERENCE', lede: ref.hardwareLede, markdown: ref.hardware,
  }, modules));
  await put('reference/engines.html', proseRefPage({
    title: 'SERVING ENGINES', eyebrow: 'REFERENCE', lede: ref.enginesLede, markdown: ref.engines,
  }, modules));
  await put('reference/frontier.html', proseRefPage({
    title: 'OPEN PROBLEMS', eyebrow: 'REFERENCE', lede: ref.frontierLede, markdown: ref.frontier,
  }, modules));
  await put('reference/how-to-use.html', proseRefPage({
    title: 'HOW TO WORK THROUGH THIS', eyebrow: 'START HERE', lede: ref.howToLede, markdown: ref.howTo,
  }, modules));

  await writeFile(join(ROOT, '.nojekyll'), '', 'utf8');

  const total = written.reduce((a, [, n]) => a + n, 0);
  console.log(`\n  ${modules.length} modules -> ${written.length} pages\n`);
  for (const [rel, n] of written) console.log(`    ${rel.padEnd(44)} ${(n / 1024).toFixed(0).padStart(5)} KB`);
  console.log(`\n  total ${(total / 1024).toFixed(0)} KB\n`);
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
