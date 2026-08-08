/* ==========================================================================
   Grid-notes diagrams.

   A tiny SVG vocabulary for the one-page summary at the top of each level:
   boxes, arrows, chips, brackets, bars and annotations drawn on graph paper.

   Everything is authored in GRID CELLS, not pixels. `box(2, 3, 10, 4)` means
   "two cells across, three down, ten wide, four tall" — so figures land on the
   grid by construction and stay aligned without fiddling with coordinates.

   All colour comes from CSS custom properties, so the figures re-theme with
   the rest of the site rather than being baked images.
   ========================================================================== */

export const U = 18;                    // pixels per grid cell

const esc = (s) =>
  String(s ?? '')
    .replace(/&/g, '&amp;').replace(/</g, '&lt;')
    .replace(/>/g, '&gt;').replace(/"/g, '&quot;');

const px = (n) => +(n * U).toFixed(2);

/* --- text ----------------------------------------------------------------- */

/**
 * Sizes are in SVG user units, and the figure is scaled to the column width
 * (roughly 78% on a desktop layout). They are set larger than their rendered
 * target so they survive that scale-down: 14 here lands at about 11px.
 */
const FONT = {
  chip:  'font-family="var(--mono)" font-size="14"',
  label: 'font-family="var(--mono)" font-size="16" font-weight="600"',
  sub:   'font-family="var(--mono)" font-size="13.5"',
  note:  'font-family="var(--body)" font-size="15.5" font-style="italic"',
  tag:   'font-family="var(--display)" font-size="10.5"',
  num:   'font-family="var(--mono)" font-size="14" font-weight="600"',
};

/** Rough advance width of the mono face, for auto-sizing chips. */
const monoW = (text, size = 14) => text.length * size * 0.601;

export function text(x, y, s, { font = 'sub', fill = 'var(--ink)', anchor = 'start', dy = 0 } = {}) {
  return `<text x="${px(x)}" y="${px(y) + dy}" ${FONT[font]} fill="${fill}" ` +
         `text-anchor="${anchor}" dominant-baseline="middle">${esc(s)}</text>`;
}

/** Multi-line annotation. Pass an array of lines; they stack downward. */
export function note(x, y, lines, { fill = 'var(--ink-dim)', anchor = 'start', gap = 1 } = {}) {
  return (Array.isArray(lines) ? lines : [lines])
    .map((l, i) => text(x, y + i * gap, l, { font: 'note', fill, anchor }))
    .join('');
}

/* --- shapes --------------------------------------------------------------- */

export function box(x, y, w, h, label, {
  sub = null, accent = 'var(--maze)', fill = 'var(--phosphor)',
  dashed = false, labelFont = 'label',
} = {}) {
  const cx = x + w / 2;
  const hasSub = sub != null && sub !== '';
  return (
    `<g>` +
    `<rect x="${px(x)}" y="${px(y)}" width="${px(w)}" height="${px(h)}" rx="4" ` +
      `fill="${fill}" stroke="${accent}" stroke-width="1.6"` +
      `${dashed ? ' stroke-dasharray="5 4"' : ''}/>` +
    (label ? text(cx, hasSub ? y + h / 2 - 0.45 : y + h / 2, label,
                  { font: labelFont, anchor: 'middle' }) : '') +
    (hasSub ? text(cx, y + h / 2 + 0.55, sub,
                   { font: 'sub', fill: 'var(--ink-dim)', anchor: 'middle' }) : '') +
    `</g>`
  );
}

/** A small rounded tag, e.g. a tensor shape. Width is derived from the text. */
export function chip(x, y, s, { accent = 'var(--pellet)', anchor = 'start' } = {}) {
  const w = (monoW(s) + 14) / U;
  const left = anchor === 'middle' ? x - w / 2 : anchor === 'end' ? x - w : x;
  return (
    `<g>` +
    `<rect x="${px(left)}" y="${px(y)}" width="${px(w)}" height="${px(1.5)}" rx="3" ` +
      `fill="none" stroke="${accent}" stroke-width="1.2" stroke-dasharray="3 2.5"/>` +
    text(left + w / 2, y + 0.75, s, { font: 'chip', fill: accent, anchor: 'middle' }) +
    `</g>`
  );
}

export function line(x1, y1, x2, y2, { stroke = 'var(--maze-dim)', dashed = false, width = 1.2 } = {}) {
  return `<line x1="${px(x1)}" y1="${px(y1)}" x2="${px(x2)}" y2="${px(y2)}" ` +
         `stroke="${stroke}" stroke-width="${width}"${dashed ? ' stroke-dasharray="4 4"' : ''}/>`;
}

export function arrow(x1, y1, x2, y2, {
  stroke = 'var(--ink-dim)', dashed = false, label = null, labelSide = 'above', id = 'ah',
} = {}) {
  let lbl = '';
  if (label) {
    const mx = (x1 + x2) / 2, my = (y1 + y2) / 2;
    const vertical = Math.abs(x2 - x1) < Math.abs(y2 - y1);
    lbl = vertical
      ? text(mx + 0.6, my, label, { font: 'sub', fill: 'var(--ink-dim)' })
      : text(mx, my + (labelSide === 'above' ? -1 : 1.1), label,
             { font: 'sub', fill: 'var(--ink-dim)', anchor: 'middle' });
  }
  // the marker is defined as `${id}-ah` in figure(); referencing bare `id`
  // silently drops every arrowhead
  return `<line x1="${px(x1)}" y1="${px(y1)}" x2="${px(x2)}" y2="${px(y2)}" ` +
         `stroke="${stroke}" stroke-width="1.5" marker-end="url(#${id}-ah)"` +
         `${dashed ? ' stroke-dasharray="5 4"' : ''}/>${lbl}`;
}

/** A dimension bracket, for annotating an extent. side: 'top' | 'bottom'. */
export function bracket(x, y, w, label, { side = 'bottom', stroke = 'var(--ink-faint)' } = {}) {
  const tick = side === 'bottom' ? 0.45 : -0.45;
  return (
    `<g>` +
    line(x, y, x + w, y, { stroke, width: 1.1 }) +
    line(x, y, x, y + tick, { stroke, width: 1.1 }) +
    line(x + w, y, x + w, y + tick, { stroke, width: 1.1 }) +
    text(x + w / 2, y + (side === 'bottom' ? 1.15 : -1.05), label,
         { font: 'sub', fill: 'var(--ink-faint)', anchor: 'middle' }) +
    `</g>`
  );
}

/** Horizontal stacked bar. segments: [{w, fill, label}] widths in cells. */
export function bar(x, y, h, segments, { outline = 'var(--maze-dim)' } = {}) {
  let cx = x;
  const parts = segments.map((s) => {
    const r =
      `<rect x="${px(cx)}" y="${px(y)}" width="${px(s.w)}" height="${px(h)}" ` +
        `fill="${s.fill}" stroke="${outline}" stroke-width="1"/>` +
      (s.label && s.w > 2.2
        ? text(cx + s.w / 2, y + h / 2, s.label,
               { font: 'chip', fill: s.ink || 'var(--on-pac)', anchor: 'middle' })
        : '');
    cx += s.w;
    return r;
  });
  return parts.join('');
}

export function dot(x, y, r, fill = 'var(--accent)') {
  return `<circle cx="${px(x)}" cy="${px(y)}" r="${r}" fill="${fill}"/>`;
}

/** A short caption pinned to a point by a leader line. */
export function callout(x, y, tx, ty, lines, { stroke = 'var(--ink-faint)', anchor = 'start' } = {}) {
  return line(x, y, tx, ty, { stroke, dashed: true, width: 1 }) +
         note(tx + (anchor === 'end' ? -0.4 : 0.4), ty, lines, { anchor });
}

/** A residual-add junction: the ⊕ on the stream. */
export function plus(x, y, { r = 7, stroke = 'var(--accent)' } = {}) {
  const cx = px(x), cy = px(y);
  return (
    `<g>` +
    `<circle cx="${cx}" cy="${cy}" r="${r}" fill="var(--grid-bg)" stroke="${stroke}" stroke-width="1.5"/>` +
    `<path d="M ${cx - r / 2} ${cy} H ${cx + r / 2} M ${cx} ${cy - r / 2} V ${cy + r / 2}" ` +
      `stroke="${stroke}" stroke-width="1.5"/>` +
    `</g>`
  );
}

/**
 * A miniature distribution, for showing a vector of probabilities being
 * reshaped. `values` are 0..1 and drawn as columns filling w x h cells.
 */
export function dist(x, y, w, h, values, { fill = 'var(--accent)', cut = -1 } = {}) {
  const n = values.length;
  const bw = w / n;
  return values.map((v, i) => {
    const bh = Math.max(0.08, v) * h;
    const faded = cut >= 0 && i >= cut;
    return `<rect x="${px(x + i * bw + bw * 0.15)}" y="${px(y + h - bh)}" ` +
           `width="${px(bw * 0.7)}" height="${px(bh)}" ` +
           `fill="${faded ? 'var(--ink-faint)' : fill}" opacity="${faded ? 0.3 : 1}"/>`;
  }).join('') + line(x, y + h, x + w, y + h, { stroke: 'var(--ink-faint)', width: 1 });
}

/** An open polyline through grid points, for plotted curves. */
export function poly(points, { stroke = 'var(--accent)', width = 2, dashed = false } = {}) {
  const d = points.map(([x, y], i) => `${i ? 'L' : 'M'} ${px(x)} ${px(y)}`).join(' ');
  return `<path d="${d}" fill="none" stroke="${stroke}" stroke-width="${width}" ` +
         `stroke-linejoin="round" stroke-linecap="round"` +
         `${dashed ? ' stroke-dasharray="5 4"' : ''}/>`;
}

/** Section heading inside a figure. */
export function heading(x, y, s, { fill = 'var(--accent)', anchor = 'start' } = {}) {
  return text(x, y, s, { font: 'tag', fill, anchor });
}

/* --- frame ---------------------------------------------------------------- */

/**
 * Wrap a body in graph paper. `w`/`h` are in cells. The figure scrolls
 * horizontally on narrow screens rather than shrinking its text to nothing.
 */
export function figure({ w, h, id, title, desc, body, caption }) {
  const W = px(w), H = px(h);
  return (
    `<figure class="fig">` +
      `<div class="fig__scroll">` +
        `<svg class="fig__svg" viewBox="0 0 ${W} ${H}" width="${W}" height="${H}" ` +
          `role="img" aria-labelledby="${id}-t ${id}-d" ` +
          `xmlns="http://www.w3.org/2000/svg">` +
          `<title id="${id}-t">${esc(title)}</title>` +
          `<desc id="${id}-d">${esc(desc)}</desc>` +
          `<defs>` +
            `<pattern id="${id}-grid" width="${U}" height="${U}" patternUnits="userSpaceOnUse">` +
              `<path d="M ${U} 0 L 0 0 0 ${U}" fill="none" ` +
                `stroke="var(--grid-line)" stroke-width="1"/>` +
            `</pattern>` +
            `<marker id="${id}-ah" viewBox="0 0 10 10" refX="9" refY="5" ` +
              `markerWidth="6" markerHeight="6" orient="auto-start-reverse">` +
              `<path d="M 0 0 L 10 5 L 0 10 z" fill="var(--ink-dim)"/>` +
            `</marker>` +
            `<marker id="${id}-ah-accent" viewBox="0 0 10 10" refX="9" refY="5" ` +
              `markerWidth="6" markerHeight="6" orient="auto-start-reverse">` +
              `<path d="M 0 0 L 10 5 L 0 10 z" fill="var(--accent)"/>` +
            `</marker>` +
          `</defs>` +
          `<rect width="${W}" height="${H}" fill="var(--grid-bg)"/>` +
          `<rect width="${W}" height="${H}" fill="url(#${id}-grid)"/>` +
          body +
        `</svg>` +
      `</div>` +
      (caption ? `<figcaption class="fig__cap">${esc(caption)}</figcaption>` : '') +
    `</figure>`
  );
}

/** Everything a diagram spec receives, so specs never import anything. */
export const kit = {
  U, text, note, box, chip, line, arrow, bracket, bar, dot, callout, heading, plus, dist, poly,
};
