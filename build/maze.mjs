/* ==========================================================================
   Maze renderer — builds the course map as static SVG at build time.

   The map is a tile grid, the way the real game is (Pac-Man is a 28x36 grid
   of 8px tiles). Ours is 21x15 with bigger tiles so the module discs have
   room to sit in the corridors.

     #  wall        .  pellet       o  power pellet
     (space) void   0-9,A  module node (A = module 10)

   Walls render as rounded outlines, not fills — that is how the arcade maze
   is actually drawn.
   ========================================================================== */

export const MAP = [
  '#####################',
  '#...................#',
  '#.0.......1.......2.#',
  '#.#.#####.#.#####.#.#',
  '#.#.......#.......#.#',
  '#.#.#####.#.#####.#.#',
  '#.3.......4.......5.#',
  '#.#.#####.#.#####.#.#',
  '#.#.......#.......#.#',
  '#.#.#####.#.#####.#.#',
  '#.6.......7.......8.#',
  '#.###.#########.###.#',
  '#.....#       #.....#',
  '#.o.9.#       #.A.o.#',
  '#####################',
];

const T = 32;      // tile size
const PAD = 18;    // breathing room inside the frame
const COLS = MAP[0].length;
const ROWS = MAP.length;

/** Where Pac-Man sits before the game starts, and respawns to. [row, col] */
export const PAC_START = [1, 15];

/**
 * The four ghosts. `seat` is their idle x-offset inside the house; `spawn` is
 * the corridor tile they take up station on when a game begins. The house has
 * no gate in the tilemap, so rather than fake one we simply station them on
 * row 10 — the long corridor that runs directly above it.
 */
export const GHOSTS = [
  { name: 'blinky', seat: -42, spawn: [10, 8], style: 'chase' },
  { name: 'pinky', seat: -14, spawn: [10, 9], style: 'ambush' },
  { name: 'inky', seat: 14, spawn: [10, 11], style: 'flank' },
  { name: 'clyde', seat: 42, spawn: [10, 12], style: 'shy' },
];

/** A tile you can walk on: corridor, pellet, power pellet, or a module node. */
const isOpen = (r, c) =>
  r >= 0 && r < ROWS && c >= 0 && c < COLS && /[.o0-9A]/.test(MAP[r][c]);

const cx = (c) => PAD + c * T + T / 2;
const cy = (r) => PAD + r * T + T / 2;
const isWall = (r, c) => r >= 0 && r < ROWS && c >= 0 && c < COLS && MAP[r][c] === '#';

/* --- sprites -------------------------------------------------------------- */

/** Pac-Man: a disc with a wedge removed. Mouth opens toward +x. */
export function pac(x, y, r = 12, cls = 'pac-sprite') {
  const dx = (0.866 * r).toFixed(2), dy = (0.5 * r).toFixed(2);
  // sweep-flag 0, not 1: with 1 you get the 60-degree wedge instead of the disc.
  return `<path class="${cls}" fill="var(--pac)" d="M${x},${y} L${x + +dx},${y - +dy} A${r},${r} 0 1 0 ${x + +dx},${y + +dy} Z"/>`;
}

/**
 * Pac-Man drawn at the origin inside a translatable group, as two frames:
 * an open wedge and a closed disc. The chomp is the classic two-frame toggle —
 * you cannot animate an arc's sweep in CSS, so you alternate the frames.
 * The group is rotated to face its direction of travel.
 */
function pacSprite(id, x, y, r = 12) {
  const dx = (0.866 * r).toFixed(2), dy = (0.5 * r).toFixed(2);
  return (
    `<g id="${id}" class="pac" transform="translate(${x},${y})">` +
      `<g class="pac__spin">` +
        `<path class="pac__open" fill="var(--pac)" d="M0,0 L${dx},-${dy} A${r},${r} 0 1 0 ${dx},${dy} Z"/>` +
        `<circle class="pac__closed" r="${r}" fill="var(--pac)"/>` +
      `</g>` +
    `</g>`
  );
}

/**
 * Ghost: semicircular hood, straight flanks, four feet. Drawn at the origin
 * so it can be translated. Body uses currentColor so frightened mode is a
 * single `color` change on the group rather than a path rewrite.
 */
export function ghost(x, y, color, w = 24, id = '') {
  const s = w / 24, X = -w / 2, Y = -w / 2;
  const p = (px, py) => `${(X + px * s).toFixed(2)},${(Y + py * s).toFixed(2)}`;
  const body =
    `M${p(1, 22)} V${(Y + 12 * s).toFixed(2)} ` +
    `A${(11 * s).toFixed(2)},${(11 * s).toFixed(2)} 0 0 1 ${p(23, 12)} ` +
    `L${p(23, 22)} L${p(17.5, 18)} L${p(12, 22)} L${p(6.5, 18)} Z`;
  const eye = (ex) =>
    `<ellipse class="ghost__eye" cx="${(X + ex * s).toFixed(2)}" cy="${(Y + 11 * s).toFixed(2)}" rx="${(3.1 * s).toFixed(2)}" ry="${(3.9 * s).toFixed(2)}" fill="#fff"/>` +
    `<circle class="ghost__pupil" cx="${(X + (ex + 1) * s).toFixed(2)}" cy="${(Y + 12 * s).toFixed(2)}" r="${(1.7 * s).toFixed(2)}" fill="#1414A0"/>`;
  return (
    `<g${id ? ` id="${id}"` : ''} class="ghost-sprite" style="color:${color}" ` +
    `transform="translate(${x},${y})">` +
      `<path class="ghost__body" d="${body}" fill="currentColor"/>${eye(8.5)}${eye(15.5)}` +
    `</g>`
  );
}

/* --- wall geometry --------------------------------------------------------- */

/**
 * Flood-fill the interior wall tiles into connected components, then draw each
 * component's bounding box as a rounded outline. Every interior wall shape in
 * this map is a rectangle except the ghost house, whose bounding box IS the
 * chamber we want to draw. The outer ring is handled separately.
 */
function interiorWallRects() {
  const seen = Array.from({ length: ROWS }, () => new Array(COLS).fill(false));
  const rects = [];
  const border = (r, c) => r === 0 || c === 0 || r === ROWS - 1 || c === COLS - 1;

  for (let r = 1; r < ROWS - 1; r++) {
    for (let c = 1; c < COLS - 1; c++) {
      if (!isWall(r, c) || seen[r][c]) continue;
      let r0 = r, r1 = r, c0 = c, c1 = c;
      const stack = [[r, c]];
      seen[r][c] = true;
      while (stack.length) {
        const [y, x] = stack.pop();
        r0 = Math.min(r0, y); r1 = Math.max(r1, y);
        c0 = Math.min(c0, x); c1 = Math.max(c1, x);
        for (const [dy, dx] of [[1, 0], [-1, 0], [0, 1], [0, -1]]) {
          const ny = y + dy, nx = x + dx;
          if (isWall(ny, nx) && !border(ny, nx) && !seen[ny][nx]) {
            seen[ny][nx] = true;
            stack.push([ny, nx]);
          }
        }
      }
      rects.push([r0, c0, r1, c1]);
    }
  }
  return rects;
}

/* --- render ---------------------------------------------------------------- */

/**
 * @param {Array<{n:number,href:string,title:string,slug:string}>} modules
 * @returns {string} SVG markup
 */
export function renderMaze(modules) {
  const W = COLS * T + PAD * 2;
  const H = ROWS * T + PAD * 2;
  const out = [];

  // outer wall — a rounded ring, inset to sit on the tile centreline
  const i = PAD + T / 2;
  out.push(
    `<rect class="maze__wall" x="${i}" y="${i}" width="${W - i * 2}" height="${H - i * 2}" rx="14"/>`
  );

  // interior walls
  for (const [r0, c0, r1, c1] of interiorWallRects()) {
    const x = cx(c0) - T * 0.34, y = cy(r0) - T * 0.34;
    const w = (c1 - c0) * T + T * 0.68, h = (r1 - r0) * T + T * 0.68;
    out.push(`<rect class="maze__wall" x="${x.toFixed(1)}" y="${y.toFixed(1)}" width="${w.toFixed(1)}" height="${h.toFixed(1)}" rx="${Math.min(11, Math.min(w, h) / 2).toFixed(1)}"/>`);
  }

  // pellets
  const nodeAt = new Map();
  for (let r = 0; r < ROWS; r++) {
    for (let c = 0; c < COLS; c++) {
      const ch = MAP[r][c];
      if (ch === '.') {
        out.push(`<circle class="maze__pellet" data-pellet="${r}-${c}" cx="${cx(c)}" cy="${cy(r)}" r="2.7"/>`);
      } else if (ch === 'o') {
        out.push(`<circle class="maze__power" data-pellet="${r}-${c}" data-power="1" cx="${cx(c)}" cy="${cy(r)}" r="6.5"/>`);
      } else if (/[0-9A]/.test(ch)) {
        nodeAt.set(ch === 'A' ? 10 : Number(ch), [r, c]);
      }
    }
  }

  // ghost house residents — the four pedagogical modes live here.
  // These are their idle seats; the game moves them out onto the corridors.
  const houseX = cx(10), houseY = cy(12) + 10;
  for (let i = 0; i < GHOSTS.length; i++) {
    const g = GHOSTS[i];
    out.push(ghost(houseX + g.seat, houseY, `var(--${g.name})`, 22, `ghost-${i}`));
  }

  // Pac-Man, mid-corridor on the top row, heading for module 0
  out.push(pacSprite('pac', cx(PAC_START[1]), cy(PAC_START[0]), 11));

  // module nodes
  for (const m of modules) {
    const pos = nodeAt.get(m.n);
    if (!pos) continue;
    const [r, c] = pos;
    const x = cx(c), y = cy(r);
    out.push(
      `<a class="node" href="${m.href}" data-node="${m.n}" data-title="${esc(m.title)}" aria-label="Module ${m.n}: ${esc(m.title)}">` +
        `<title>Module ${m.n} — ${esc(m.title)}</title>` +
        `<circle class="node__disc" cx="${x}" cy="${y}" r="14.5"/>` +
        `<text class="node__num" x="${x}" y="${y + 0.5}">${m.n}</text>` +
      `</a>`
    );
  }

  return (
    `<svg class="maze" viewBox="0 0 ${W} ${H}" role="group" ` +
    `aria-label="Course map: 11 modules laid out as a Pac-Man maze" ` +
    `xmlns="http://www.w3.org/2000/svg">${out.join('')}</svg>`
  );
}

function esc(s) {
  return String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
}

/* --- data handed to the client game ---------------------------------------- */

/** Row/col of every module node, keyed by module number. */
function nodePositions() {
  const at = new Map();
  for (let r = 0; r < ROWS; r++) {
    for (let c = 0; c < COLS; c++) {
      const ch = MAP[r][c];
      if (/[0-9A]/.test(ch)) at.set(ch === 'A' ? 10 : Number(ch), [r, c]);
    }
  }
  return at;
}

/**
 * Build-time sanity check: every module node must be walkable-reachable from
 * Pac-Man's start tile. If a map edit ever strands a node, the build fails
 * here rather than shipping a level nobody can drive to.
 */
export function assertReachable() {
  const seen = new Set([PAC_START.join(',')]);
  const q = [PAC_START];
  while (q.length) {
    const [r, c] = q.shift();
    for (const [dr, dc] of [[1, 0], [-1, 0], [0, 1], [0, -1]]) {
      const nr = r + dr, nc = c + dc, k = nr + ',' + nc;
      if (isOpen(nr, nc) && !seen.has(k)) { seen.add(k); q.push([nr, nc]); }
    }
  }
  const unreachable = [];
  for (const [n, [r, c]] of nodePositions()) {
    if (!seen.has(r + ',' + c)) unreachable.push(n);
  }
  if (unreachable.length) {
    throw new Error(`maze: module nodes unreachable from Pac-Man's start: ${unreachable.join(', ')}`);
  }
  return seen.size;
}

/**
 * Everything the browser needs to run the game, serialized into the page.
 * Keeping geometry in one place means the JS never re-derives tile maths that
 * the renderer already did.
 */
export function mazeData(modules) {
  const at = nodePositions();
  return {
    map: MAP,
    tile: T,
    pad: PAD,
    cols: COLS,
    rows: ROWS,
    pacStart: PAC_START,
    ghosts: GHOSTS.map((g) => ({ name: g.name, spawn: g.spawn, style: g.style })),
    nodes: modules
      .filter((m) => at.has(m.n))
      .map((m) => ({ n: m.n, rc: at.get(m.n), href: m.href, title: m.title })),
  };
}

/* --- small inline sprites for the legend and chrome ----------------------- */
export const icon = {
  pellet: (color = 'var(--pellet)') => `<svg width="14" height="14" viewBox="0 0 14 14" aria-hidden="true"><circle cx="7" cy="7" r="3" fill="${color}"/></svg>`,
  power: (color = 'var(--pac)') => `<svg width="14" height="14" viewBox="0 0 14 14" aria-hidden="true"><circle cx="7" cy="7" r="6" fill="${color}"/></svg>`,
  ghost: (color) => `<svg width="16" height="16" viewBox="0 0 24 24" aria-hidden="true">${ghost(12, 12, color, 24)}</svg>`,
  pac: (size = 16) => `<svg width="${size}" height="${size}" viewBox="0 0 24 24" aria-hidden="true">${pac(12, 12, 10, '')}</svg>`,
  key: (color = 'var(--inky)') => `<svg width="14" height="16" viewBox="0 0 14 16" aria-hidden="true" fill="${color}"><path d="M7 0a4 4 0 0 0-1 7.87V13H4v2h2v1h2v-1h2v-2H8V7.87A4 4 0 0 0 7 0zm0 2.2A1.8 1.8 0 1 1 7 5.8a1.8 1.8 0 0 1 0-3.6z"/></svg>`,
  cherry: () => `<svg width="16" height="16" viewBox="0 0 16 16" aria-hidden="true"><path d="M9 2c-2 1-3 3-4 6" stroke="#22B14C" stroke-width="1.6" fill="none"/><circle cx="4.5" cy="11" r="3.2" fill="#FF0000"/><circle cx="11" cy="10" r="3.2" fill="#FF0000"/></svg>`,
};
