#!/usr/bin/env node
/* ==========================================================================
   Figure collision check.

     node build/build.mjs && node build/check-figures.mjs

   Estimating text extents from character counts is not reliable enough to
   trust, so this loads each built page in a real browser and measures the
   actual rendered getBBox of every label. Exits non-zero on any overlap or
   anything drawn outside the canvas.

   Requires playwright. It is a dev check, not part of the default build,
   so a clone without playwright can still build the site.
   ========================================================================== */
import fs from 'node:fs';

let chromium;
try {
  ({ chromium } = await import('playwright'));
} catch (e) {
  console.log('check-figures: playwright is not installed — skipping.');
  console.log('  npm install   (playwright is an optional devDependency)');
  process.exit(0);
}
const b = await chromium.launch();
const mods = await Promise.all(
  fs.readdirSync(new URL('../data/modules', import.meta.url))
    .filter((f) => /^m\d+\.mjs$/.test(f))
    .map((f) => import(new URL('../data/modules/' + f, import.meta.url))));
const slugs = mods.map((m) => [m.default.n, m.default.slug]).sort((a, b) => a[0] - b[0]);
const ROOT = new URL('..', import.meta.url).pathname;
let total = 0;
for (const [n, slug] of slugs) {
  const p = await b.newPage({ viewport:{width:1500,height:1000} });
  await p.goto('file://' + ROOT + 'modules/' + slug + '.html', { waitUntil: 'load' });
  const r = await p.evaluate(() => {
    const svg = document.querySelector('.fig__svg');
    if (!svg) return null;
    const vb = svg.viewBox.baseVal;
    const items = [...svg.querySelectorAll('text')].map((t,i) => {
      const bb = t.getBBox();
      return { i, s: (t.textContent||'').slice(0,42),
               x1: bb.x, y1: bb.y, x2: bb.x+bb.width, y2: bb.y+bb.height };
    });
    const pad = 1.5;
    const hits = [];
    for (let a=0;a<items.length;a++) for (let c=a+1;c<items.length;c++){
      const A=items[a], B=items[c];
      const ox = Math.min(A.x2,B.x2)-Math.max(A.x1,B.x1) - pad;
      const oy = Math.min(A.y2,B.y2)-Math.max(A.y1,B.y1) - pad;
      if (ox>0 && oy>0) hits.push({a:A.s,b:B.s,ox:Math.round(ox),oy:Math.round(oy)});
    }
    // anything drawn outside the canvas
    const out = items.filter(t => t.x1 < -2 || t.x2 > vb.width+2 || t.y1 < -2 || t.y2 > vb.height+2)
                     .map(t=>({s:t.s, x2:Math.round(t.x2), y2:Math.round(t.y2)}));
    return { hits, out, w: vb.width, h: vb.height, n: items.length };
  });
  await p.close();
  if (!r) { console.log(`level ${n}: NO FIGURE`); continue; }
  const bad = r.hits.length + r.out.length;
  total += bad;
  console.log(`level ${String(n).padStart(2)}  ${r.n} labels  ${bad ? bad+' PROBLEMS' : 'clean'}`);
  r.hits.slice(0,6).forEach(h=>console.log(`     overlap ${h.ox}x${h.oy}px: "${h.a}" / "${h.b}"`));
  r.out.slice(0,6).forEach(o=>console.log(`     outside canvas (${r.w}x${r.h}): "${o.s}" ends ${o.x2},${o.y2}`));
}
console.log(total ? `\n${total} problems` : '\nall figures clean');
await b.close();
process.exit(total ? 1 : 0);
