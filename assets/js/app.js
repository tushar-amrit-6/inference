/* ==========================================================================
   Progress, wayfinding, and the arcade chrome.

   Everything here is enhancement. The site is fully readable and navigable
   with JavaScript disabled — the maze is server-rendered SVG with real <a>
   elements, and the rail is anchor links.
   ========================================================================== */

(function () {
  'use strict';

  var KEY = 'llm-inference:cleared';
  var TOTAL = 12;
  var POINTS = 2200;          // per level. 12 x 2200 = 26400 for a clean sweep.

  /* --- state ------------------------------------------------------------- */

  function read() {
    try {
      var raw = localStorage.getItem(KEY);
      if (!raw) return [];
      var v = JSON.parse(raw);
      return Array.isArray(v) ? v.filter(function (n) { return typeof n === 'number'; }) : [];
    } catch (e) {
      return [];
    }
  }

  function write(list) {
    try {
      localStorage.setItem(KEY, JSON.stringify(list));
    } catch (e) {
      /* private browsing, quota, whatever — progress just won't persist */
    }
  }

  var cleared = read();
  var has = function (n) { return cleared.indexOf(n) !== -1; };

  function toggle(n) {
    var i = cleared.indexOf(n);
    if (i === -1) cleared.push(n);
    else cleared.splice(i, 1);
    cleared.sort(function (a, b) { return a - b; });
    write(cleared);
    paint();
  }

  /* --- chrome ------------------------------------------------------------ */

  // class rather than an inline fill, so the theme can give it a keyline on paper
  var PAC =
    '<svg class="life-pac" width="13" height="13" viewBox="0 0 24 24" aria-hidden="true">' +
    '<path d="M12,12 L20.66,7 A10,10 0 1 0 20.66,17 Z"/></svg>';

  function paint() {
    var n = cleared.length;

    var score = document.querySelector('[data-score]');
    if (score) score.textContent = String(n * POINTS).padStart(6, '0');

    var cl = document.querySelector('[data-cleared]');
    if (cl) cl.textContent = String(n).padStart(2, '0') + '/' + TOTAL;

    // "lives" are levels still to clear, capped so the bar never wraps
    var lives = document.querySelector('[data-lives]');
    if (lives) {
      var left = Math.max(0, TOTAL - n);
      lives.innerHTML = left ? new Array(Math.min(left, 11) + 1).join(PAC) : '<span style="color:var(--pac)">PERFECT</span>';
    }

    Array.prototype.forEach.call(document.querySelectorAll('[data-node]'), function (el) {
      el.setAttribute('data-done', has(Number(el.getAttribute('data-node'))) ? '1' : '0');
    });

    Array.prototype.forEach.call(document.querySelectorAll('[data-level]'), function (el) {
      el.setAttribute('data-done', has(Number(el.getAttribute('data-level'))) ? '1' : '0');
    });

    Array.prototype.forEach.call(document.querySelectorAll('[data-complete]'), function (btn) {
      var done = has(Number(btn.getAttribute('data-complete')));
      btn.setAttribute('aria-pressed', done ? 'true' : 'false');
      btn.textContent = done
        ? '✓ LEVEL ' + String(btn.getAttribute('data-complete')).padStart(2, '0') + ' CLEARED'
        : 'MARK LEVEL ' + String(btn.getAttribute('data-complete')).padStart(2, '0') + ' CLEARED';
    });

    // resume hint on the landing page
    var resume = document.querySelector('[data-resume]');
    if (resume) {
      if (!n) {
        resume.hidden = true;
      } else {
        var next = 0;
        while (has(next) && next < TOTAL) next++;
        resume.hidden = false;
        resume.textContent = next >= TOTAL
          ? '★ ALL LEVELS CLEARED'
          : '● ' + n + ' CLEARED — RESUME AT LEVEL ' + String(next).padStart(2, '0');
      }
    }
  }

  document.addEventListener('click', function (e) {
    var btn = e.target.closest ? e.target.closest('[data-complete]') : null;
    if (btn) toggle(Number(btn.getAttribute('data-complete')));
  });

  /* --- maze readout ------------------------------------------------------ */

  var readout = document.querySelector('[data-maze-readout]');
  if (readout) {
    var idle = readout.textContent;
    var show = function (e) {
      var node = e.target.closest ? e.target.closest('[data-node]') : null;
      if (!node) return;
      readout.textContent =
        'LEVEL ' + String(node.getAttribute('data-node')).padStart(2, '0') +
        ' — ' + node.getAttribute('data-title');
    };
    var hide = function () { readout.textContent = idle; };
    var maze = document.querySelector('.maze');
    if (maze) {
      maze.addEventListener('mouseover', show);
      maze.addEventListener('focusin', show);
      maze.addEventListener('mouseleave', hide);
      maze.addEventListener('focusout', hide);
    }
  }

  /* --- theme -------------------------------------------------------------- */
  // Default is whatever the OS asks for; an explicit choice overrides it and
  // persists. The <head> applies a stored choice before first paint, so this
  // only has to handle the toggle itself.
  var THEME_KEY = 'llm-inference:theme';
  var root = document.documentElement;

  function systemTheme() {
    return window.matchMedia && window.matchMedia('(prefers-color-scheme: light)').matches
      ? 'light' : 'dark';
  }
  function activeTheme() {
    return root.getAttribute('data-theme') || systemTheme();
  }
  function labelTheme() {
    var next = activeTheme() === 'dark' ? 'light' : 'dark';
    Array.prototype.forEach.call(document.querySelectorAll('[data-theme-toggle]'), function (b) {
      b.setAttribute('aria-label', 'Switch to ' + next + ' mode');
      b.setAttribute('title', 'Switch to ' + next + ' mode');
    });
  }

  document.addEventListener('click', function (e) {
    var btn = e.target.closest ? e.target.closest('[data-theme-toggle]') : null;
    if (!btn) return;
    var next = activeTheme() === 'dark' ? 'light' : 'dark';
    root.setAttribute('data-theme', next);
    try { localStorage.setItem(THEME_KEY, next); } catch (err) { /* private mode */ }
    labelTheme();
  });

  // follow the OS if the user has not made an explicit choice
  if (window.matchMedia) {
    var mq = window.matchMedia('(prefers-color-scheme: light)');
    var onChange = function () { if (!root.getAttribute('data-theme')) labelTheme(); };
    if (mq.addEventListener) mq.addEventListener('change', onChange);
    else if (mq.addListener) mq.addListener(onChange);
  }

  labelTheme();

  /* --- SVG link activation ------------------------------------------------ */
  // SVG <a> elements are focusable and Tab reaches them, but Chromium does not
  // fire activation on Enter the way it does for HTML anchors. Wire it up so
  // the maze is keyboard-navigable, not just mouse-navigable.
  document.addEventListener('keydown', function (e) {
    if (e.key !== 'Enter' && e.key !== ' ' && e.key !== 'Spacebar') return;
    if (e.metaKey || e.ctrlKey || e.altKey) return;
    var node = e.target && e.target.closest ? e.target.closest('a.node[href]') : null;
    if (!node) return;
    e.preventDefault();
    window.location.href = node.getAttribute('href');
  });

  /* --- rail scroll spy --------------------------------------------------- */

  var links = Array.prototype.slice.call(document.querySelectorAll('.rail__link'));
  if (links.length && 'IntersectionObserver' in window) {
    var byId = {};
    var targets = [];
    links.forEach(function (a) {
      var id = a.getAttribute('href').slice(1);
      var el = document.getElementById(id);
      if (el) { byId[id] = a; targets.push(el); }
    });

    var visible = {};
    var obs = new IntersectionObserver(function (entries) {
      entries.forEach(function (en) {
        visible[en.target.id] = en.isIntersecting ? en.intersectionRatio : 0;
      });
      var best = null, bestRatio = 0;
      targets.forEach(function (t) {
        var r = visible[t.id] || 0;
        if (r > bestRatio) { bestRatio = r; best = t.id; }
      });
      links.forEach(function (a) { a.removeAttribute('aria-current'); });
      if (best && byId[best]) byId[best].setAttribute('aria-current', 'true');
    }, { rootMargin: '-70px 0px -55% 0px', threshold: [0, 0.15, 0.5, 1] });

    targets.forEach(function (t) { obs.observe(t); });
  }

  /* --- keyboard: arrows move between levels ------------------------------ */

  document.addEventListener('keydown', function (e) {
    if (e.metaKey || e.ctrlKey || e.altKey || e.shiftKey) return;
    var t = e.target;
    if (t && (t.tagName === 'INPUT' || t.tagName === 'TEXTAREA' || t.isContentEditable)) return;

    var dir = null;
    if (e.key === 'ArrowLeft') dir = 'prev';
    if (e.key === 'ArrowRight') dir = 'next';
    if (!dir) return;

    var link = document.querySelector('.pagenav a[data-nav="' + dir + '"]');
    if (link) {
      e.preventDefault();
      window.location.href = link.href;
    }
  });

  paint();
})();
