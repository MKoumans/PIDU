/* Project page: fit-animation pool, PIDU explorer and BibTeX copy. No dependencies. */
(function () {
  'use strict';

  const $ = (root, sel) => root.querySelector(sel);
  const el = (tag, attrs = {}, text) => {
    const n = document.createElement(tag);
    for (const [k, v] of Object.entries(attrs)) n.setAttribute(k, v);
    if (text != null) n.textContent = text;
    return n;
  };
  const shuffle = (a) => {
    a = a.slice();
    for (let i = a.length - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1));
      [a[i], a[j]] = [a[j], a[i]];
    }
    return a;
  };
  const reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;

  /* ── Fit-animation pool ────────────────────────────── */
  function initFits() {
    const fig = document.getElementById('fit-pool');
    if (!fig) return;
    const pool = window.PIDU_FITS || {};
    const CASES = [
      { key: 'case2', label: 'SrTiO₃ measured' },
      { key: 'case1', label: 'simulated' },
    ].filter((c) => pool[c.key] && pool[c.key].items.length);
    if (!CASES.length) {
      // No animations built yet (docs/tools/build_site_assets.py): say so instead of an empty frame.
      $(fig, '[data-fits="img"]').hidden = true;
      $(fig, '[data-fits="next"]').hidden = true;
      $(fig, '[data-fits="loading"]').textContent = 'fit animations not built yet';
      $(fig, '.fits-frame').classList.add('is-loading');
      return;
    }

    const img = $(fig, '[data-fits="img"]');
    const chips = $(fig, '[data-fits="cases"]');
    const frame = img.parentElement;
    const DEFAULT_MS = 8000;
    let caseKey = CASES[0].key, order = [], pos = 0, timer = null, visible = true;

    const pressed = () => chips.querySelectorAll('.chip').forEach((b) =>
      b.setAttribute('aria-pressed', String(b.dataset.key === caseKey)));

    function show() {
      const item = order[pos % order.length];
      const methods = pool[caseKey].methods.join(' vs ');
      frame.classList.add('is-loading');
      const next = new Image();
      next.onload = () => {
        img.src = next.src;
        img.alt = `${item.title}: ${methods}, step by step`;
        frame.classList.remove('is-loading');
        // Preload the following animation so switching is instant.
        new Image().src = order[(pos + 1) % order.length].src;
        schedule(item.ms || DEFAULT_MS);
      };
      next.src = item.src;
    }
    function schedule(ms) {
      clearTimeout(timer);
      if (reducedMotion || !visible) return;
      timer = setTimeout(() => { pos++; show(); }, ms * 2); // let each fit play twice
    }
    function selectCase(key) {
      caseKey = key;
      order = shuffle(pool[key].items);
      pos = 0;
      pressed();
      show();
    }

    CASES.forEach((c) => {
      const b = el('button', { class: 'chip', type: 'button', 'data-key': c.key, 'aria-pressed': 'false' }, c.label);
      b.addEventListener('click', () => selectCase(c.key));
      chips.appendChild(b);
    });
    $(fig, '[data-fits="next"]').addEventListener('click', () => { pos++; show(); });

    new IntersectionObserver((entries) => {
      visible = entries[0].isIntersecting;
      if (visible) schedule(DEFAULT_MS / 2); else clearTimeout(timer);
    }).observe(fig);

    selectCase(caseKey);
  }

  /* ── Explorer ──────────────────────────────────────── */
  const GREEK = { w0: 'ω₀', wp: 'ωₚ', g: 'γ' };
  function paramLabel(name) {
    if (name === 'eps_inf') return 'ε∞';
    let m = name.match(/^(w0|wp|g)_before$/);
    if (m) return `${GREEK[m[1]]} soft mode`;
    m = name.match(/^(w0|wp|g)_(\d+)$/);
    if (m) return `${GREEK[m[1]]} osc. ${m[2]}`;
    return name;
  }
  const fmt = (v) => {
    const a = Math.abs(v);
    if (a === 0) return '0';
    if (a >= 1000) return v.toFixed(0);
    if (a >= 100) return v.toFixed(1);
    if (a >= 1) return v.toFixed(2);
    return v.toPrecision(3);
  };

  function initExplorer() {
    const fig = document.getElementById('explorer-figure');
    const manifest = window.PIDU_EXPLORER;
    if (!fig || !manifest) return;
    const cases = Object.keys(manifest).sort();
    const q = (k) => $(fig, `[data-ex="${k}"]`);
    const stage = $(fig, '.explorer-stage');
    const ui = {
      cases: q('cases'), sample: q('sample'), sampleOut: q('sample-out'), step: q('step'), stepOut: q('step-out'),
      play: q('play'), plots: q('plots'), params: q('params'), mape: q('mape'),
    };
    const cache = new Map();
    let caseKey = cases[0], record = null, plots = [], playTimer = null, request = 0;

    function load(key, index) {
      const url = `static/explorer/${key}/${String(index).padStart(4, '0')}.json`;
      if (!cache.has(url)) cache.set(url, fetch(url).then((r) => {
        if (!r.ok) throw new Error(`${url}: ${r.status}`);
        return r.json();
      }));
      return cache.get(url);
    }

    /* One SVG line plot per spectral channel; the estimate path is updated per step. */
    function buildPlot(channel, spec, steps) {
      const tick = (t) => String(+t.toFixed(4));
    const W = 640, H = 220, M = { l: 44, r: 10, t: 22, b: 38 };
      const wn = spec.wn, n = wn.length;
      const all = spec.y.concat(...steps.map((s) => spec.y_hat[s] || []));
      let lo = Math.min(...all), hi = Math.max(...all);
      const pad = (hi - lo || 1) * 0.06; lo -= pad; hi += pad;
      const x = (v) => M.l + ((v - wn[0]) / (wn[n - 1] - wn[0])) * (W - M.l - M.r);
      const y = (v) => M.t + (1 - (v - lo) / (hi - lo)) * (H - M.t - M.b);
      const path = (ys) => ys.map((v, i) => `${i ? 'L' : 'M'}${x(wn[i]).toFixed(1)},${y(v).toFixed(1)}`).join('');

      const ns = 'http://www.w3.org/2000/svg';
      const s = (tag, attrs, text) => {
        const node = document.createElementNS(ns, tag);
        for (const [k, v] of Object.entries(attrs)) node.setAttribute(k, v);
        if (text != null) node.textContent = text;
        return node;
      };
      const svg = s('svg', { viewBox: `0 0 ${W} ${H}`, role: 'img', 'aria-label': `${channel.label}: measured and PIDU estimate` });
      const grid = s('g', { class: 'grid' }), axis = s('g', { class: 'axis' });
      const ticks = (a, b, k) => {
        const step = Math.pow(10, Math.floor(Math.log10((b - a) / k)));
        const mult = [1, 2, 5, 10].find((m) => (b - a) / (step * m) <= k) * step;
        const out = [];
        for (let t = Math.ceil(a / mult) * mult; t <= b + 1e-9; t += mult) out.push(+t.toFixed(10));
        return out;
      };
      ticks(lo, hi, 4).forEach((t) => {
        grid.appendChild(s('line', { x1: M.l, x2: W - M.r, y1: y(t), y2: y(t) }));
        axis.appendChild(s('text', { x: M.l - 6, y: y(t) + 4, 'text-anchor': 'end' }, tick(t)));
      });
      ticks(wn[0], wn[n - 1], 6).forEach((t) => {
        grid.appendChild(s('line', { x1: x(t), x2: x(t), y1: M.t, y2: H - M.b }));
        axis.appendChild(s('text', { x: x(t), y: H - M.b + 16, 'text-anchor': 'middle' }, tick(t)));
      });
      axis.appendChild(s('line', { x1: M.l, x2: W - M.r, y1: H - M.b, y2: H - M.b }));
      axis.appendChild(s('text', { x: (M.l + W - M.r) / 2, y: H - 1, 'text-anchor': 'middle' }, 'wavenumber (cm⁻¹)'));
      svg.append(grid, axis, s('text', { x: M.l, y: 13, class: 'title' }, channel.label));
      svg.appendChild(s('path', { class: 'measured', d: path(spec.y) }));
      const est = s('path', { class: 'estimate', d: path(spec.y_hat[0]) });
      svg.appendChild(est);

      const box = el('div', { class: 'plot' });
      box.appendChild(svg);
      return { box, set: (k) => est.setAttribute('d', path(spec.y_hat[k])) };
    }

    function render() {
      const k = +ui.step.value;
      plots.forEach((p) => p.set(k));
      ui.stepOut.textContent = `${k} / ${record.n_unfolds}`;
      ui.params.replaceChildren(...record.names.map((name, i) => {
        const t = record.p_true[i], v = record.p_hat[k][i];
        const err = Math.abs(v - t) / Math.max(Math.abs(t), 1e-12) * 100;
        const tr = el('tr');
        tr.append(el('td', {}, paramLabel(name)), el('td', {}, fmt(t)), el('td', {}, fmt(v)),
          el('td', { class: `err${err > 25 ? ' high' : err > 8 ? ' mid' : ''}` }, `${err.toFixed(1)}%`));
        return tr;
      }));
    }

    async function showSample() {
      const indices = manifest[caseKey].indices;
      const i = +ui.sample.value;
      const index = indices[i];
      ui.sampleOut.textContent = `#${index}`;
      const ticket = ++request;
      stage.classList.add('is-loading');
      let rec;
      try { rec = await load(caseKey, index); } catch (e) { console.error(e); return; }
      if (ticket !== request) return;
      stage.classList.remove('is-loading');
      record = rec;
      ui.step.max = rec.n_unfolds;
      if (+ui.step.value > rec.n_unfolds) ui.step.value = rec.n_unfolds;
      const steps = Array.from({ length: rec.n_unfolds + 1 }, (_, s) => s);
      const legend = el('div', { class: 'legend' });
      legend.innerHTML = '<span><i style="background:var(--measured)"></i>measured</span>' +
        '<span><i style="background:var(--estimate)"></i>PIDU estimate</span>';
      plots = manifest[caseKey].channels.filter((c) => rec.spectra[c.id]).map((c) => buildPlot(c, rec.spectra[c.id], steps));
      ui.plots.replaceChildren(legend, ...plots.map((p) => p.box));
      ui.mape.textContent = `${rec.mape.toFixed(1)}%`;
      render();
      // Warm the cache for the neighbours.
      [i - 1, i + 1].filter((j) => j >= 0 && j < indices.length).forEach((j) => load(caseKey, indices[j]).catch(() => {}));
    }

    function selectCase(key) {
      caseKey = key;
      ui.cases.querySelectorAll('.chip').forEach((b) => b.setAttribute('aria-pressed', String(b.dataset.key === key)));
      ui.sample.max = manifest[key].indices.length - 1;
      ui.sample.value = 0;
      ui.step.value = 3;
      stopPlay();
      showSample();
    }

    function stopPlay() {
      clearInterval(playTimer); playTimer = null;
      ui.play.classList.remove('is-playing');
      ui.play.setAttribute('aria-label', 'Play the unfolding steps');
    }
    ui.play.addEventListener('click', () => {
      if (playTimer) return stopPlay();
      ui.play.classList.add('is-playing');
      ui.play.setAttribute('aria-label', 'Pause');
      ui.step.value = 0; render();
      playTimer = setInterval(() => {
        if (+ui.step.value >= +ui.step.max) return stopPlay();
        ui.step.value = +ui.step.value + 1; render();
      }, 800);
    });

    cases.forEach((key) => {
      const b = el('button', { class: 'chip', type: 'button', 'data-key': key, 'aria-pressed': 'false' }, manifest[key].label);
      b.addEventListener('click', () => selectCase(key));
      ui.cases.appendChild(b);
    });
    ui.sample.addEventListener('input', showSample);
    ui.step.addEventListener('input', () => { stopPlay(); if (record) render(); });

    selectCase(caseKey);
  }

  /* ── BibTeX copy ───────────────────────────────────── */
  function initCopy() {
    document.querySelectorAll('[data-copy]').forEach((btn) => {
      btn.addEventListener('click', async () => {
        const text = document.querySelector(btn.dataset.copy).textContent;
        try { await navigator.clipboard.writeText(text); btn.textContent = 'copied'; }
        catch { btn.textContent = 'select & copy'; }
        setTimeout(() => { btn.textContent = 'copy'; }, 1600);
      });
    });
  }

  initFits();
  initExplorer();
  initCopy();
})();
