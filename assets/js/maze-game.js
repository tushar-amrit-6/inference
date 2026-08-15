/* ==========================================================================
   The maze, playable.

   Press START, steer with the arrow keys (or WASD, or a swipe), and whichever
   module node Pac-Man eats is the level that opens. Ghosts patrol; getting
   caught sends you back to the start rather than ending anything, because this
   is navigation first and a game second.

   The maze is server-rendered SVG with real <a> nodes, so clicking a level and
   tabbing to it keep working whether or not this file ever runs.
   ========================================================================== */

(function () {
  'use strict';

  var el = document.getElementById('maze-data');
  var svg = document.querySelector('.maze');
  var startBtn = document.querySelector('[data-game-start]');
  if (!el || !svg || !startBtn) return;

  var D;
  try { D = JSON.parse(el.textContent); } catch (e) { return; }

  var TILE = D.tile, PAD = D.pad, MAP = D.map, ROWS = D.rows, COLS = D.cols;

  var readout = document.querySelector('[data-maze-readout]');
  var status = document.querySelector('[data-game-status]');
  var pacEl = document.getElementById('pac');
  var spinEl = pacEl && pacEl.querySelector('.pac__spin');
  if (!pacEl || !spinEl) return;

  /* --- geometry ---------------------------------------------------------- */

  var cx = function (c) { return PAD + c * TILE + TILE / 2; };
  var cy = function (r) { return PAD + r * TILE + TILE / 2; };

  function open(r, c) {
    return r >= 0 && r < ROWS && c >= 0 && c < COLS && /[.o0-9ABC]/.test(MAP[r].charAt(c));
  }

  var DIRS = {
    left:  { dr: 0, dc: -1, deg: 180 },
    right: { dr: 0, dc: 1, deg: 0 },
    up:    { dr: -1, dc: 0, deg: -90 },
    down:  { dr: 1, dc: 0, deg: 90 },
  };
  var DIR_NAMES = ['left', 'right', 'up', 'down'];
  var OPPOSITE = { left: 'right', right: 'left', up: 'down', down: 'up' };

  /* --- node lookup ------------------------------------------------------- */

  var nodeByTile = {};
  D.nodes.forEach(function (n) { nodeByTile[n.rc[0] + ',' + n.rc[1]] = n; });

  /* --- state ------------------------------------------------------------- */

  var running = false;
  var raf = 0;
  var lastT = 0;
  var frightenedUntil = 0;
  var navigating = false;
  var eaten = {};                    // "r,c" -> true
  var score = 0;

  var PAC_SPEED = 6.4;               // tiles per second
  var GHOST_SPEED = 5.0;
  var GHOST_FRIGHT_SPEED = 3.2;
  var FRIGHT_MS = 6500;

  var pac = null;
  var ghosts = [];

  var hasInput = false;              // has the player steered yet this run?

  // r,c is the tile we are leaving; t is progress (0..1) toward r+dr, c+dc.
  // `parked` means we reached a tile and the way ahead is blocked.
  function mover(r, c, dir) {
    return { r: r, c: c, dir: dir, next: dir, t: 0, parked: true };
  }

  function resetActors() {
    pac = mover(D.pacStart[0], D.pacStart[1], 'left');
    ghosts = D.ghosts.map(function (g, i) {
      var m = mover(g.spawn[0], g.spawn[1], i % 2 ? 'left' : 'right');
      m.style = g.style;
      m.el = document.getElementById('ghost-' + i);
      m.dead = false;
      return m;
    });
  }

  /* --- rendering --------------------------------------------------------- */

  function lerp(m) {
    // interpolate between the tile we left and the tile we are entering
    var d = DIRS[m.dir];
    return { x: cx(m.c + d.dc * m.t), y: cy(m.r + d.dr * m.t) };
  }

  function draw() {
    var p = lerp(pac);
    pacEl.setAttribute('transform', 'translate(' + p.x.toFixed(2) + ',' + p.y.toFixed(2) + ')');
    spinEl.setAttribute('transform', 'rotate(' + DIRS[pac.dir].deg + ')');

    var fright = Date.now() < frightenedUntil;
    for (var i = 0; i < ghosts.length; i++) {
      var g = ghosts[i];
      if (!g.el) continue;
      var q = lerp(g);
      g.el.setAttribute('transform', 'translate(' + q.x.toFixed(2) + ',' + q.y.toFixed(2) + ')');
      g.el.classList.toggle('is-asleep', !hasInput);
      g.el.classList.toggle('is-frightened', fright && !g.dead);
      // flash for the last 1.5s so the player knows it is ending
      g.el.classList.toggle('is-flashing', fright && frightenedUntil - Date.now() < 1500);
    }
  }

  /* --- movement ---------------------------------------------------------- */

  function canGo(m, dir) {
    var d = DIRS[dir];
    return open(m.r + d.dr, m.c + d.dc);
  }

  /**
   * Advance one mover by dt seconds at `speed` tiles/sec.
   * Direction is only ever reconsidered at a tile centre, which is what makes
   * turns feel like the arcade rather than like free-roaming.
   */
  function step(m, dt, speed, chooseDir) {
    if (m.parked) {
      chooseDir(m);
      if (!canGo(m, m.dir)) return;      // still nowhere to go
      m.parked = false;
    }
    m.t += dt * speed;
    if (m.t >= 1) {
      var d = DIRS[m.dir];
      m.r += d.dr;
      m.c += d.dc;
      m.t = 0;
      chooseDir(m);
      if (!canGo(m, m.dir)) m.parked = true;
    }
  }

  function legalDirs(m, allowReverse) {
    return DIR_NAMES.filter(function (d) {
      return canGo(m, d) && (allowReverse || d !== OPPOSITE[m.dir]);
    });
  }

  function pacChoose(m) {
    // Until the player takes over, Pac-Man drives himself, so pressing START
    // and just watching still lands on a level. He always turns when blocked
    // and sometimes turns at a junction — without the second rule the route is
    // fully determined by the maze and he picks the same level every time.
    if (!hasInput) {
      var opts = legalDirs(m, false);
      if (!opts.length) opts = legalDirs(m, true);
      var blocked = !canGo(m, m.dir);
      if (opts.length && (blocked || (opts.length > 1 && Math.random() < 0.35))) {
        m.dir = opts[Math.floor(Math.random() * opts.length)];
        // clear the queued turn too, or the stale one reverses him at the next
        // tile where it happens to be legal
        m.next = m.dir;
      }
      return;
    }

    // take the queued turn the moment it becomes legal — classic Pac-Man
    if (m.next !== m.dir && canGo(m, m.next)) m.dir = m.next;
  }

  function dist2(ar, ac, br, bc) {
    var dr = ar - br, dc = ac - bc;
    return dr * dr + dc * dc;
  }

  function ghostChoose(g) {
    var fright = Date.now() < frightenedUntil;
    // ghosts never reverse unless it is the only option (as in the arcade)
    var pool = legalDirs(g, false);
    if (!pool.length) pool = legalDirs(g, true);
    if (!pool.length) return;

    if (fright) {
      g.dir = pool[Math.floor(Math.random() * pool.length)];
      return;
    }

    // each ghost gets a different target, matching its arcade personality
    var tr = pac.r, tc = pac.c;
    var pd = DIRS[pac.dir];
    if (g.style === 'ambush') { tr = pac.r + pd.dr * 4; tc = pac.c + pd.dc * 4; }
    else if (g.style === 'flank') { tr = pac.r + pd.dr * -3; tc = pac.c + pd.dc * -3; }
    else if (g.style === 'shy' && dist2(g.r, g.c, pac.r, pac.c) < 36) { tr = ROWS - 1; tc = 0; }

    var best = pool[0], bestD = Infinity;
    for (var i = 0; i < pool.length; i++) {
      var d = DIRS[pool[i]];
      var v = dist2(g.r + d.dr, g.c + d.dc, tr, tc);
      if (v < bestD) { bestD = v; best = pool[i]; }
    }
    g.dir = best;
  }

  /* --- eating ------------------------------------------------------------ */

  function eatAt(r, c) {
    var key = r + ',' + c;
    if (eaten[key]) return;
    var dot = svg.querySelector('[data-pellet="' + r + '-' + c + '"]');
    if (!dot) return;
    eaten[key] = true;
    if (dot.getAttribute('data-power')) {
      score += 50;
      frightenedUntil = Date.now() + FRIGHT_MS;
      dot.classList.add('is-eaten');
      say('POWER PELLET — GHOSTS ARE EDIBLE');
    } else {
      score += 10;
      dot.classList.add('is-eaten');
    }
  }

  function say(msg) {
    if (readout) readout.textContent = msg;
  }

  function setStatus(msg) {
    if (status) status.textContent = msg;
  }

  /* --- the loop ---------------------------------------------------------- */

  function tick(now) {
    if (!running) return;
    if (!lastT) lastT = now;
    var dt = Math.min((now - lastT) / 1000, 0.05);   // clamp after a tab switch
    lastT = now;

    step(pac, dt, PAC_SPEED, pacChoose);
    eatAt(pac.r, pac.c);

    var fright = Date.now() < frightenedUntil;
    var gs = fright ? GHOST_FRIGHT_SPEED : GHOST_SPEED;
    for (var i = 0; i < ghosts.length; i++) step(ghosts[i], dt, gs, ghostChoose);

    // Collisions. Ghosts are asleep until the player takes the wheel: in
    // autopilot the maze is a level picker, and being knocked back to the
    // start forever would stop it ever picking one.
    var p = lerp(pac);
    for (var j = 0; hasInput && j < ghosts.length; j++) {
      var g = ghosts[j];
      if (g.dead) continue;
      var q = lerp(g);
      if (Math.abs(p.x - q.x) < TILE * 0.6 && Math.abs(p.y - q.y) < TILE * 0.6) {
        if (fright) {
          g.dead = true;
          score += 200;
          if (g.el) g.el.classList.add('is-dead');
          say('ATE ' + D.ghosts[j].name.toUpperCase() + ' — +200');
          (function (gg, idx) {
            setTimeout(function () {
              gg.r = D.ghosts[idx].spawn[0];
              gg.c = D.ghosts[idx].spawn[1];
              gg.t = 0;
              gg.parked = true;
              gg.dead = false;
              if (gg.el) gg.el.classList.remove('is-dead');
            }, 2200);
          })(g, j);
        } else {
          caught();
          return;
        }
      }
    }

    // did we land on a module node?
    var node = nodeByTile[pac.r + ',' + pac.c];
    if (node && !navigating) { arrive(node); return; }

    draw();
    raf = requestAnimationFrame(tick);
  }

  function caught() {
    running = false;
    cancelAnimationFrame(raf);
    svg.classList.add('is-caught');
    say('CAUGHT — BACK TO THE START');
    setTimeout(function () {
      svg.classList.remove('is-caught');
      resetActors();
      draw();
      lastT = 0;
      running = true;
      raf = requestAnimationFrame(tick);
    }, 900);
  }

  function arrive(node) {
    navigating = true;
    running = false;
    cancelAnimationFrame(raf);
    draw();

    var a = svg.querySelector('[data-node="' + node.n + '"]');
    if (a) a.classList.add('is-eaten');
    say('LEVEL ' + String(node.n).padStart(2, '0') + ' — ' + node.title);
    setStatus('OPENING…');
    setTimeout(function () { window.location.href = node.href; }, 620);
  }

  /* --- controls ---------------------------------------------------------- */

  function turn(dir) {
    if (!running || !pac) return;
    if (!hasInput) {                   // player takes over; the ghosts wake up
      hasInput = true;
      setStatus('GHOSTS ARE AWAKE — EAT A NODE TO OPEN THAT LEVEL');
    }
    pac.next = dir;

    // Mid-corridor reversal is instant, as in the arcade. The tile we were
    // heading into becomes the tile we are leaving, and the remaining progress
    // flips — which keeps the on-screen position exactly where it was.
    if (!pac.parked && pac.t > 0 && dir === OPPOSITE[pac.dir]) {
      var d = DIRS[pac.dir];
      pac.r += d.dr;
      pac.c += d.dc;
      pac.t = 1 - pac.t;
      pac.dir = dir;
    }
  }

  var KEYS = {
    ArrowLeft: 'left', ArrowRight: 'right', ArrowUp: 'up', ArrowDown: 'down',
    a: 'left', d: 'right', w: 'up', s: 'down',
    A: 'left', D: 'right', W: 'up', S: 'down',
  };

  function mazeOnScreen() {
    var r = svg.getBoundingClientRect();
    var vh = window.innerHeight || document.documentElement.clientHeight;
    return r.bottom > vh * 0.15 && r.top < vh * 0.85;
  }

  document.addEventListener('keydown', function (e) {
    if (e.metaKey || e.ctrlKey || e.altKey) return;
    var t = e.target;
    if (t && (t.tagName === 'INPUT' || t.tagName === 'TEXTAREA' || t.isContentEditable)) return;
    var dir = KEYS[e.key];

    if (!running) {
      // Left or right starts a game, but only while the maze is actually on
      // screen. Deliberately not up/down: those are how people scroll, and
      // hijacking them anywhere on the page would be hostile.
      if (!navigating && (dir === 'left' || dir === 'right') && mazeOnScreen()) {
        e.preventDefault();
        start();
        turn(dir);
      }
      return;
    }

    if (dir) { e.preventDefault(); turn(dir); }
    if (e.key === 'Escape') stop('STOPPED');
  });

  // swipe, for touch
  var tsx = 0, tsy = 0;
  svg.addEventListener('touchstart', function (e) {
    if (!running) return;
    tsx = e.changedTouches[0].clientX;
    tsy = e.changedTouches[0].clientY;
  }, { passive: true });
  svg.addEventListener('touchend', function (e) {
    if (!running) return;
    var dx = e.changedTouches[0].clientX - tsx;
    var dy = e.changedTouches[0].clientY - tsy;
    if (Math.abs(dx) < 24 && Math.abs(dy) < 24) return;
    turn(Math.abs(dx) > Math.abs(dy) ? (dx > 0 ? 'right' : 'left') : (dy > 0 ? 'down' : 'up'));
  }, { passive: true });

  /* --- start / stop ------------------------------------------------------ */

  function clearEaten() {
    Array.prototype.forEach.call(svg.querySelectorAll('.is-eaten'), function (n) {
      n.classList.remove('is-eaten');
    });
    eaten = {};
  }

  function start() {
    clearEaten();
    score = 0;
    frightenedUntil = 0;
    navigating = false;
    hasInput = false;
    resetActors();
    draw();

    svg.classList.add('is-playing');
    startBtn.textContent = '■ STOP';
    startBtn.setAttribute('data-playing', 'true');
    setStatus('STEER WITH ARROWS, WASD OR A SWIPE — OR JUST WATCH');
    say('HE WILL FIND A LEVEL ON HIS OWN. TAKE OVER TO CHOOSE ONE.');

    running = true;
    lastT = 0;
    raf = requestAnimationFrame(tick);
  }

  function stop(msg) {
    running = false;
    cancelAnimationFrame(raf);
    svg.classList.remove('is-playing', 'is-caught');
    startBtn.textContent = '▶ START';
    startBtn.removeAttribute('data-playing');
    setStatus('');
    say(msg || 'HOVER A NODE TO SEE THE LEVEL');
    // put everyone back in their idle positions and restore the pellets
    clearEaten();
    resetActors();
    Array.prototype.forEach.call(svg.querySelectorAll('.ghost-sprite'), function (g) {
      g.removeAttribute('transform');
      g.classList.remove('is-frightened', 'is-flashing', 'is-dead', 'is-asleep');
    });
    pacEl.setAttribute('transform',
      'translate(' + cx(D.pacStart[1]) + ',' + cy(D.pacStart[0]) + ')');
    spinEl.setAttribute('transform', 'rotate(0)');
  }

  startBtn.addEventListener('click', function () {
    if (running || navigating) stop();
    else start();
  });

  // The hero CTA plays the maze. It is a real link to level 00, so with JS
  // disabled it still takes you somewhere useful; here we take it over.
  var reduce = window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  Array.prototype.forEach.call(document.querySelectorAll('[data-game-play]'), function (btn) {
    btn.addEventListener('click', function (e) {
      e.preventDefault();
      try {
        svg.scrollIntoView({ behavior: reduce ? 'auto' : 'smooth', block: 'center' });
      } catch (err) {
        svg.scrollIntoView();
      }
      if (!running && !navigating) start();
    });
  });

  // stop cleanly if the tab goes away mid-run
  document.addEventListener('visibilitychange', function () {
    if (document.hidden && running) stop('PAUSED — PRESS START AGAIN');
  });

  startBtn.hidden = false;
})();
