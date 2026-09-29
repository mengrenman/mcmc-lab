import { mountChrome, $, setText, bindRange, Runner, Pacer, throttle } from '../lib/ui.js';
import { Plot, legend, palette, hexToRgb, nearestIndex, fmtNum, fmtSig, fmtInt } from '../lib/plot.js';
import { RNG, randomSeed } from '../lib/random.js';
import { Trace, chainDiagnostics } from '../lib/stats.js';
import { Ising, T_CRITICAL, onsagerMagnetization, onsagerEnergy } from '../samplers/ising.js';

mountChrome();

const ALG_NAMES = { metropolis: 'Metropolis', heatbath: 'Heat bath', wolff: 'Wolff' };
const ALG_COLOR_VAR = { metropolis: '--s1', heatbath: '--s2', wolff: '--s3' };
const algColor = (alg, c) => ({ metropolis: c.s1, heatbath: c.s2, wolff: c.s3 })[alg];

// =====================================================================
// Live lattice
// =====================================================================

const HISTORY = 100000;
const rng = new RNG();
let model = null;
const histM = new Trace(HISTORY), histE = new Trace(HISTORY);
let changes = []; // { at: history index, T, alg } for each change of settings
let seg = null; // work counters since the last change
let stats = null; // averages since the last change
const pacer = new Pacer();

const Tctl = bindRange('ig-T', { format: (v) => fmtNum(v, 2), onInput: onSettingsChange });
const speed = bindRange('ig-speed', { log: true, format: (v) => (v < 10 ? fmtNum(v, 1) : fmtInt(v)) + ' sweeps/s' });
const algorithm = () => $('ig-alg').value;

const canvas = $('ig-lattice');
const lctx = canvas.getContext('2d');
let img = null;

legend('ig-lattice-legend', [
  { label: 'spin up', color: 'var(--spin-up)', kind: 'rect' },
  { label: 'spin down', color: 'var(--spin-down)', kind: 'rect' },
  { label: 'cluster just flipped (Wolff)', color: 'var(--cluster)', kind: 'rect' },
]);

function drawLattice() {
  const c = palette();
  const up = hexToRgb(c.spinUp), down = hexToRgb(c.spinDown), cl = hexToRgb(c.cluster);
  const showCluster = model.showCluster && (!runner.running || speed.get() <= 3);
  const d = img.data, s = model.s, mark = model.inCluster;
  for (let i = 0; i < model.N; i++) {
    const col = showCluster && mark[i] ? cl : s[i] > 0 ? up : down;
    d[4 * i] = col[0];
    d[4 * i + 1] = col[1];
    d[4 * i + 2] = col[2];
    d[4 * i + 3] = 255;
  }
  lctx.putImageData(img, 0, 0);
}

/** Sweep index of change k and where its segment ends. */
function segmentBounds(k) {
  return [changes[k].at, k + 1 < changes.length ? changes[k + 1].at : Math.max(changes[k].at, histM.n - 1)];
}

function historyPlot(id, { history, exact, y, label }) {
  return new Plot(id, {
    height: 200,
    x: { domain: [0, 50], label: 'sweep' },
    y: { domain: y },
    draw(p) {
      const c = p.c;
      for (let k = 0; k < changes.length; k++) {
        const [a, b] = segmentBounds(k);
        if (k > 0) p.vline(a, { color: c.ink2, alpha: 0.35 });
        const v = exact(changes[k].T);
        p.line([a, Math.max(b, a + 0.5)], [v, v], { color: c.exact, width: 2 });
      }
      p.line(null, history.a, { color: c.s1, width: history.n > 3000 ? 1 : 1.5, n: history.n });
    },
    hover(x) {
      if (!history.n) return null;
      const i = Math.max(0, Math.min(history.n - 1, Math.round(x)));
      let k = 0;
      while (k + 1 < changes.length && changes[k + 1].at <= i) k++;
      const ch = changes[k];
      return {
        x: i,
        title: `sweep ${fmtInt(i)} · ${ALG_NAMES[ch.alg]}, T = ${fmtNum(ch.T, 2)}`,
        rows: [
          { value: fmtNum(history.a[i], 3), label, color: 'var(--s1)' },
          { value: fmtNum(exact(ch.T), 3), label: 'exact, infinite lattice', color: 'var(--exact)' },
        ],
      };
    },
  });
}

const mPlot = historyPlot('ig-m', { history: histM, exact: onsagerMagnetization, y: [0, 1.02], label: '|m|' });
const ePlot = historyPlot('ig-e', { history: histE, exact: onsagerEnergy, y: [-2.05, 0.05], label: 'e' });
for (const [id, name] of [['ig-m-legend', '|m| after each sweep'], ['ig-e-legend', 'e after each sweep']]) {
  legend(id, [
    { label: name, color: 'var(--s1)', kind: 'line' },
    { label: 'exact (Onsager, infinite lattice)', color: 'var(--exact)', kind: 'line' },
  ]);
}

function record() {
  histM.push(Math.abs(model.M) / model.N);
  histE.push(model.E / model.N);
}

function markChange() {
  const entry = { at: Math.max(0, histM.n - 1), T: Tctl.get(), alg: algorithm() };
  // Dragging the slider fires many changes; merge those that happen between sweeps.
  if (changes.length && changes[changes.length - 1].at === entry.at) changes[changes.length - 1] = entry;
  else changes.push(entry);
  seg = { start: entry.at, work: 0, clusters: 0, flipped: 0, sweeps: 0 };
  stats = null;
}

function onSettingsChange() {
  model.setT(Tctl.get());
  $('ig-cluster').disabled = algorithm() !== 'wolff';
  markChange();
  updateStats();
  drawAll();
}

function sweepOnce() {
  const alg = algorithm();
  const r = model.sweep(alg);
  if (alg === 'wolff') {
    seg.clusters += r;
    seg.flipped += model.lastSweepFlipped;
  } else {
    seg.work += r;
  }
  seg.sweeps++;
  record();
  return !histM.full;
}

function updateStats() {
  const n = histM.n;
  const from = seg.start + Math.floor(0.2 * (n - seg.start));
  stats = n - from >= 20 ? { m: chainDiagnostics(histM.a, from, n), e: chainDiagnostics(histE.a, from, n) } : null;
  const T = Tctl.get();
  const pm = (d) => `${fmtNum(d.mean, 3)} ± ${fmtSig(d.error, 1)}`;
  setText('ig-t-m', stats?.m ? pm(stats.m) : '—');
  setText('ig-t-e', stats?.e ? pm(stats.e) : '—');
  setText('ig-t-m-exact', `exact ${fmtNum(onsagerMagnetization(T), 3)} (infinite lattice)`);
  setText('ig-t-e-exact', `exact ${fmtNum(onsagerEnergy(T), 3)} (infinite lattice)`);
  const alg = algorithm();
  if (alg === 'wolff') {
    setText('ig-t-work-label', 'Mean cluster size');
    setText('ig-t-work', seg.clusters ? fmtSig(seg.flipped / seg.clusters, 3) : '—');
    setText('ig-t-work-sub', seg.sweeps ? `${fmtSig(seg.clusters / seg.sweeps, 3)} clusters per sweep` : 'spins per cluster');
  } else {
    setText('ig-t-work-label', alg === 'metropolis' ? 'Acceptance rate' : 'Spins changed per sweep');
    setText('ig-t-work', seg.sweeps ? `${fmtNum((100 * seg.work) / (seg.sweeps * model.N), 1)}%` : '—');
    setText('ig-t-work-sub', alg === 'metropolis' ? 'accepted ÷ proposed flips' : 'fraction of spins that changed');
  }
}
const statsSoon = throttle(updateStats, 250);

function drawAll() {
  drawLattice();
  const n = Math.max(50, histM.n - 1);
  mPlot.setX([0, n]).redraw();
  ePlot.setX([0, n]).redraw();
  const T = Tctl.get();
  setText('ig-t-sweeps', fmtInt(histM.n - 1));
  setText('ig-t-since', `${fmtInt(histM.n - 1 - seg.start)} since the last change`);
  setText('ig-t-ratio', fmtNum(T / T_CRITICAL, 3));
  setText('ig-t-phase', Math.abs(T / T_CRITICAL - 1) < 0.02 ? 'critical region' : T < T_CRITICAL ? 'ordered phase' : 'disordered phase');
  setText('ig-t-mnow', fmtNum(Math.abs(model.M) / model.N, 3));
}

function reset() {
  runner?.stop();
  const L = Number($('ig-L').value);
  if (!model || model.L !== L) {
    model = new Ising(L, rng);
    canvas.width = L;
    canvas.height = L;
    img = lctx.createImageData(L, L);
  }
  rng.seed(randomSeed());
  model.setT(Tctl.get());
  if ($('ig-init').value === 'cold') model.order();
  else model.randomize();
  setText('ig-lattice-sub', `${L} × ${L} = ${fmtInt(L * L)} spins, periodic boundaries.`);
  histM.clear();
  histE.clear();
  record();
  changes = [];
  markChange();
  $('ig-cluster').disabled = algorithm() !== 'wolff';
  $('ig-full').hidden = true;
  pacer.reset();
  updateStats();
  drawAll();
}

const runner = new Runner({
  button: 'ig-run',
  root: 'ig-lab',
  tick(dt) {
    const k = pacer.take(speed.get(), dt, 1000);
    if (k === 0) return true;
    const t0 = performance.now();
    let ok = true;
    for (let i = 0; i < k && ok; i++) {
      ok = sweepOnce();
      if (performance.now() - t0 > 30) break; // keep the page responsive on big lattices
    }
    if (!ok) $('ig-full').hidden = false;
    drawAll();
    statsSoon();
    return ok;
  },
  onChange(running) {
    if (!running) {
      updateStats();
      drawAll();
    }
  },
});

$('ig-sweep').addEventListener('click', () => {
  runner.stop();
  if (sweepOnce()) {
    updateStats();
    drawAll();
  }
});
$('ig-cluster').addEventListener('click', () => {
  runner.stop();
  model.wolffStep();
  model.recompute();
  drawAll();
});
$('ig-reset').addEventListener('click', reset);
$('ig-L').addEventListener('change', reset);
$('ig-init').addEventListener('change', reset);
$('ig-alg').addEventListener('change', onSettingsChange);
window.addEventListener('themechange', () => drawLattice());
reset();

// =====================================================================
// Temperature scan
// =====================================================================

const TEMPS = [3.5, 3.0, 2.7, 2.5, 2.4, 2.35, 2.3, 2.27, 2.25, 2.2, 2.1, 2.0, 1.8, 1.5]; // hot to cold
const results = {}; // algorithm -> { L, points: [{ T, m, mErr, e, eErr, tau, converged }] }
let job = null;

function sortedPoints(alg) {
  return results[alg] ? [...results[alg].points].sort((a, b) => a.T - b.T) : [];
}

function scanPlot(id, { y, yType, value, err, exact, fmt, label }) {
  return new Plot(id, {
    height: yType === 'log' ? 230 : 220,
    x: { domain: [1.4, 3.6], label: 'temperature T' },
    y: { domain: y, type: yType || 'linear' },
    draw(p) {
      const c = p.c;
      p.vline(T_CRITICAL, { color: c.ink2, alpha: 0.4 });
      p.text(T_CRITICAL, p.y.domain[1], 'Tc', { dx: 4, dy: 8, color: c.muted });
      if (exact) p.fn(exact, { color: c.exact, width: 2 });
      for (const alg of Object.keys(ALG_NAMES)) {
        const pts = sortedPoints(alg);
        if (!pts.length) continue;
        const color = algColor(alg, c);
        const xs = pts.map((q) => q.T), ys = pts.map(value);
        p.line(xs, ys, { color, width: 1.5, alpha: 0.8 });
        if (err) p.errorBars(xs, ys, pts.map(err), { color });
        for (const q of pts) {
          const hollow = yType === 'log' && !q.converged;
          p.marker(q.T, value(q), { color, r: 4, hollow });
        }
      }
    },
    hover(x) {
      const rows = [];
      const Ts = [...TEMPS].sort((a, b) => a - b);
      const T = Ts[nearestIndex(Ts, x)];
      for (const alg of Object.keys(ALG_NAMES)) {
        const q = results[alg]?.points.find((pt) => pt.T === T);
        if (!q) continue;
        const bound = yType === 'log' && !q.converged ? '≳ ' : '';
        rows.push({ value: bound + fmt(q), label: `${ALG_NAMES[alg]} (L = ${results[alg].L})`, color: `var(${ALG_COLOR_VAR[alg]})` });
      }
      if (exact) rows.push({ value: fmtNum(exact(T), 3), label: 'exact, infinite lattice', color: 'var(--exact)' });
      if (!rows.length) return null;
      return { x: T, title: `T = ${fmtNum(T, 2)} · ${label}`, rows };
    },
  });
}

const scM = scanPlot('sc-m', {
  y: [0, 1.02], value: (q) => q.m, err: (q) => q.mErr, exact: onsagerMagnetization,
  fmt: (q) => `${fmtNum(q.m, 3)} ± ${fmtSig(q.mErr, 1)}`, label: '⟨|m|⟩',
});
const scE = scanPlot('sc-e', {
  y: [-2.05, 0], value: (q) => q.e, err: (q) => q.eErr, exact: onsagerEnergy,
  fmt: (q) => `${fmtNum(q.e, 3)} ± ${fmtSig(q.eErr, 1)}`, label: '⟨e⟩',
});
const scTau = scanPlot('sc-tau', {
  y: [0.5, 100], yType: 'log', value: (q) => q.tau,
  fmt: (q) => `${fmtSig(q.tau, 3)} sweeps`, label: 'τint of |m|',
});

function refreshScanLegend() {
  const items = Object.keys(ALG_NAMES)
    .filter((alg) => results[alg])
    .map((alg) => ({ label: `${ALG_NAMES[alg]} (L = ${results[alg].L})`, color: `var(${ALG_COLOR_VAR[alg]})`, kind: 'line' }));
  items.push({ label: 'exact (Onsager, infinite lattice)', color: 'var(--exact)', kind: 'line' });
  legend('sc-legend', items);
}

function drawScan() {
  let top = 10, bottom = 0.5;
  for (const r of Object.values(results)) {
    for (const q of r.points) {
      if (!Number.isFinite(q.tau)) continue;
      top = Math.max(top, q.tau);
      bottom = Math.min(bottom, q.tau);
    }
  }
  scTau.setY([Math.min(0.7, bottom * 0.7), 10 ** Math.ceil(Math.log10(top * 1.5))]);
  scM.redraw();
  scE.redraw();
  scTau.redraw();
  refreshScanLegend();
}

function* scanJob(alg, L, meas, res) {
  const therm = Math.max(1000, meas / 5);
  const model = new Ising(L, new RNG());
  model.randomize();
  const absM = new Float64Array(meas), en = new Float64Array(meas);
  const total = TEMPS.length * (therm + meas);
  let done = 0;
  for (let ti = 0; ti < TEMPS.length; ti++) {
    const T = TEMPS[ti];
    model.setT(T);
    for (let s = 0; s < therm; s++) {
      model.sweep(alg);
      done++;
      if ((s & 15) === 0) yield { done, total, T, ti };
    }
    for (let s = 0; s < meas; s++) {
      model.sweep(alg);
      absM[s] = Math.abs(model.M) / model.N;
      en[s] = model.E / model.N;
      done++;
      if ((s & 15) === 0) yield { done, total, T, ti };
    }
    const dm = chainDiagnostics(absM, 0, meas);
    const de = chainDiagnostics(en, 0, meas);
    res.points.push({ T, m: dm.mean, mErr: dm.error, e: de.mean, eErr: de.error, tau: dm.tau, converged: dm.converged });
    yield { done, total, T, ti, point: true };
  }
}

const channel = new MessageChannel();
channel.port1.onmessage = pump;

function pump() {
  if (!job) return;
  if (job.cancel) return finishScan('Scan canceled. The finished temperatures are kept.');
  const t0 = performance.now();
  let r;
  do {
    r = job.gen.next();
    if (r.done) break;
    if (r.value.point) drawScan();
  } while (performance.now() - t0 < 14);
  if (r.done) return finishScan(`Done: ${ALG_NAMES[job.alg]}, L = ${job.L}, in ${fmtNum((performance.now() - job.started) / 1000, 1)} s.`);
  const v = r.value;
  $('sc-bar').style.width = `${(100 * v.done) / v.total}%`;
  setText('sc-status', `${ALG_NAMES[job.alg]}, L = ${job.L}: T = ${fmtNum(v.T, 2)} (${v.ti + 1} of ${TEMPS.length})`);
  channel.port2.postMessage(0);
}

function finishScan(message) {
  job = null;
  $('sc-run').textContent = 'Run scan';
  $('sc-bar').style.width = '0%';
  setText('sc-status', message);
  for (const id of ['sc-alg', 'sc-L', 'sc-meas', 'sc-clear']) $(id).disabled = false;
  drawScan();
}

$('sc-run').addEventListener('click', () => {
  if (job) {
    job.cancel = true;
    return;
  }
  const alg = $('sc-alg').value, L = Number($('sc-L').value), meas = Number($('sc-meas').value);
  const res = { L, points: [] };
  results[alg] = res;
  job = { gen: scanJob(alg, L, meas, res), alg, L, cancel: false, started: performance.now() };
  $('sc-run').textContent = 'Cancel';
  for (const id of ['sc-alg', 'sc-L', 'sc-meas', 'sc-clear']) $(id).disabled = true;
  drawScan();
  channel.port2.postMessage(0);
});
$('sc-clear').addEventListener('click', () => {
  for (const k of Object.keys(results)) delete results[k];
  drawScan();
  setText('sc-status', 'Results cleared.');
});
drawScan();
