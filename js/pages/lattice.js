import { mountChrome, $, setText, bindRange, Runner, Pacer, throttle } from '../lib/ui.js';
import { Plot, legend, palette, hexToRgb, nearestIndex, fmtNum, fmtSig, fmtInt } from '../lib/plot.js';
import { RNG, randomSeed } from '../lib/random.js';
import { Trace, chainDiagnostics } from '../lib/stats.js';
import { U1Lattice, hmcTrajectory, u1Exact as u1ExactUncached } from '../samplers/gauge.js';
import { WilsonDirac, Pseudofermion, schwingerTrajectory } from '../samplers/schwinger.js';

mountChrome();

const pct = (v) => `${fmtNum(100 * v, 1)}%`;
const fmtRate = (unit) => (v) => (v < 10 ? fmtNum(v, 1) : fmtInt(v)) + ` ${unit}/s`;
const pm = (d, digits = 4) => `${fmtNum(d.mean, digits)} ± ${d.converged ? '' : '≳ '}${fmtSig(d.error, 1)}`;
const hmcStep = (beta) => 0.2 / Math.sqrt(Math.max(1, beta));
const exactCache = new Map();
function u1Exact(beta, V) {
  const key = `${beta}|${V}`;
  if (!exactCache.has(key)) {
    if (exactCache.size > 200) exactCache.clear();
    exactCache.set(key, u1ExactUncached(beta, V));
  }
  return exactCache.get(key);
}

// =====================================================================
// Pure gauge theory
// =====================================================================

const HISTORY = 100000;
const LOOPS = [[1, 1], [1, 2], [2, 2], [2, 3], [3, 3], [3, 4], [4, 4]];
const glRng = new RNG();
let lat = null;
const histP = new Trace(HISTORY), histQ = new Trace(HISTORY);
let loopTraces = LOOPS.map(() => new Trace(HISTORY));
let changes = [];
let seg = null;
let exact = null;
let loopStats = [];
const glPacer = new Pacer();

const glBeta = bindRange('gl-beta', { log: true, format: (v) => fmtSig(v, 3), onInput: onGaugeSettings });
const glSpeed = bindRange('gl-speed', { log: true, format: fmtRate('sweeps') });
const glAlg = () => $('gl-alg').value;

const fieldCanvas = $('gl-field');
const fctx = fieldCanvas.getContext('2d');
let fieldImg = null;

legend('gl-field-legend', [
  { label: '\\(\\theta_P < 0\\)', color: 'var(--div-neg)', kind: 'rect' },
  { label: '\\(\\theta_P \\approx 0\\)', color: 'var(--div-mid)', kind: 'rect' },
  { label: '\\(\\theta_P > 0\\)', color: 'var(--div-pos)', kind: 'rect' },
]);

function drawField() {
  const c = palette();
  const neg = hexToRgb(c.divNeg), mid = hexToRgb(c.divMid), pos = hexToRgb(c.divPos);
  const d = fieldImg.data, L = lat.L;
  for (let s = 0; s < lat.V; s++) {
    let a = lat.plaquette(s);
    a -= 2 * Math.PI * Math.round(a / (2 * Math.PI));
    const t = Math.min(1, Math.abs(a) / Math.PI) ** 0.6;
    const end = a < 0 ? neg : pos;
    // Put time upward: row 0 of the image is the top of the lattice.
    const x = s % L, y = L - 1 - Math.floor(s / L), k = 4 * (y * L + x);
    d[k] = mid[0] + t * (end[0] - mid[0]);
    d[k + 1] = mid[1] + t * (end[1] - mid[1]);
    d[k + 2] = mid[2] + t * (end[2] - mid[2]);
    d[k + 3] = 255;
  }
  fctx.putImageData(fieldImg, 0, 0);
}

function segmentBounds(k) {
  return [changes[k].at, k + 1 < changes.length ? changes[k + 1].at : Math.max(changes[k].at, histP.n - 1)];
}

const plaqPlot = new Plot('gl-plaq', {
  height: 150,
  x: { domain: [0, 50], label: 'sweep' },
  y: { domain: [-0.05, 1.02] },
  draw(p) {
    if (!lat) return;
    const c = p.c;
    for (let k = 0; k < changes.length; k++) {
      const [a, b] = segmentBounds(k);
      if (k > 0) p.vline(a, { color: c.ink2, alpha: 0.35 });
      const v = u1Exact(changes[k].beta, lat.V).plaquette;
      p.line([a, Math.max(b, a + 0.5)], [v, v], { color: c.exact, width: 2 });
    }
    p.line(null, histP.a, { color: c.s1, width: histP.n > 3000 ? 1 : 1.5, n: histP.n });
  },
  hover(x) {
    if (!histP.n) return null;
    const i = Math.max(0, Math.min(histP.n - 1, Math.round(x)));
    let k = 0;
    while (k + 1 < changes.length && changes[k + 1].at <= i) k++;
    return {
      x: i,
      title: `sweep ${fmtInt(i)} · β = ${fmtSig(changes[k].beta, 3)}`,
      rows: [
        { value: fmtNum(histP.a[i], 4), label: 'cos θ_P, lattice average', color: 'var(--s1)' },
        { value: fmtNum(u1Exact(changes[k].beta, lat.V).plaquette, 4), label: 'exact', color: 'var(--exact)' },
      ],
    };
  },
});

const qPlot = new Plot('gl-q', {
  height: 220,
  x: { domain: [0, 50], label: 'sweep' },
  y: { domain: [-4, 4] },
  draw(p) {
    if (!lat) return;
    const c = p.c;
    for (let k = 1; k < changes.length; k++) p.vline(changes[k].at, { color: c.ink2, alpha: 0.35 });
    p.hline(0, { color: c.axis });
    p.line(null, histQ.a, { color: c.s1, width: 1.5, n: histQ.n });
  },
  hover(x) {
    if (!histQ.n) return null;
    const i = Math.max(0, Math.min(histQ.n - 1, Math.round(x)));
    return { x: i, title: `sweep ${fmtInt(i)}`, rows: [{ value: String(histQ.a[i]), label: 'Q', color: 'var(--s1)' }] };
  },
});

const wilsonPlot = new Plot('gl-wilson', {
  height: 220,
  x: { domain: [0, 17], label: 'area A (plaquettes)' },
  y: { domain: [0, 6] },
  draw(p) {
    if (!lat || !exact) return;
    const c = p.c;
    const As = [], ys = [];
    for (let A = 0; A <= 16; A++) { As.push(A); ys.push(A === 0 ? 0 : -Math.log(exact.wilson(A))); }
    p.line(As, ys, { color: c.exact, width: 2 });
    const pts = loopStats.filter((q) => q && q.ok);
    if (pts.length) {
      p.errorBars(pts.map((q) => q.A), pts.map((q) => q.y), pts.map((q) => q.err), { color: c.s1 });
      p.points(pts.map((q) => q.A), pts.map((q) => q.y), { color: c.s1, r: 4, ring: true });
    }
  },
  hover(x) {
    if (!exact) return null;
    const areas = LOOPS.map(([R, T]) => R * T);
    const i = nearestIndex(areas, x);
    const [R, T] = LOOPS[i], A = R * T, q = loopStats[i];
    const rows = [{ value: fmtNum(-Math.log(exact.wilson(A)), 3), label: 'exact', color: 'var(--exact)' }];
    if (q && q.ok) rows.unshift({ value: `${fmtNum(q.y, 3)} ± ${fmtSig(q.err, 1)}`, label: 'measured', color: 'var(--s1)' });
    else if (q) rows.unshift({ value: 'too noisy yet', label: 'measured', color: 'var(--s1)' });
    return { x: A, title: `${R} × ${T} loop, A = ${A}`, rows };
  },
});
legend('gl-wilson-legend', [
  { label: 'measured \\(-\\ln\\langle W \\rangle\\)', color: 'var(--s1)', kind: 'dot' },
  { label: 'exact on this lattice', color: 'var(--exact)', kind: 'line' },
]);

function activeLoops() {
  return LOOPS.map(([R, T]) => R <= lat.L / 2 && T <= lat.L / 2);
}

function markChange() {
  const entry = { at: Math.max(0, histP.n - 1), beta: glBeta.get(), alg: glAlg() };
  if (changes.length && changes[changes.length - 1].at === entry.at) changes[changes.length - 1] = entry;
  else changes.push(entry);
  seg = { start: entry.at, acc: 0, prop: 0, qChanges: 0, sweeps: 0 };
  loopTraces.forEach((t) => t.clear());
  exact = u1Exact(entry.beta, lat.V);
}

function onGaugeSettings() {
  if (!lat) return;
  lat.setBeta(glBeta.get());
  markChange();
  updateGaugeStats();
  drawGauge();
}

function record() {
  histP.push(lat.meanCosPlaquette());
  const q = Math.round(lat.topologicalCharge());
  if (histQ.n && q !== histQ.last()) seg.qChanges++;
  histQ.push(q);
}

function sweepOnce() {
  if (glAlg() === 'hmc') {
    const eps = hmcStep(lat.beta);
    const r = hmcTrajectory(lat, { eps, steps: Math.round(1 / eps), forceFn: (f) => lat.gaugeForce(f), actionFn: () => lat.action() });
    seg.acc += r.accept ? 1 : 0;
    seg.prop += 1;
  } else {
    seg.acc += lat.sweepMetropolis();
    seg.prop += 2 * lat.V;
  }
  seg.sweeps++;
  record();
  const on = activeLoops();
  LOOPS.forEach(([R, T], i) => { if (on[i]) loopTraces[i].push(lat.wilsonLoop(R, T)); });
  return !histP.full;
}

function updateGaugeStats() {
  const n = histP.n, from = seg.start + Math.floor(0.2 * (n - seg.start));
  const d = n - from >= 20 ? chainDiagnostics(histP.a, from, n) : null;
  setText('gl-t-plaq', d ? pm(d) : '—');
  // Frozen topology: the error bars ignore the slow sector mode, so say so.
  const frozen = seg.sweeps >= 1000 && seg.qChanges / seg.sweeps < 0.01;
  const note = $('gl-frozen');
  note.hidden = !frozen;
  if (frozen) {
    note.textContent = `Topology is frozen: Q has changed only ${fmtInt(seg.qChanges)} times in ${fmtInt(seg.sweeps)} sweeps. ` +
      'Plaquettes and Wilson loops are being measured in just a few charge sectors, so they can miss the exact values by more than their error bars. See "Topological freezing" below.';
  }
  setText('gl-t-plaq-exact', `exact ${fmtNum(exact.plaquette, 4)} on ${lat.L} × ${lat.L}`);
  setText('gl-t-sigma', fmtNum(exact.stringTension, 4));
  const on = activeLoops();
  loopStats = LOOPS.map(([R, T], i) => {
    if (!on[i]) return null;
    const tr = loopTraces[i], m = tr.n, a = Math.floor(0.2 * m);
    if (m - a < 20) return { A: R * T, ok: false };
    const dd = chainDiagnostics(tr.a, a, m);
    // −ln⟨W⟩ is only meaningful while ⟨W⟩ is clearly above zero.
    const ok = dd && dd.mean > 3 * dd.error && Number.isFinite(dd.error);
    return ok ? { A: R * T, y: -Math.log(dd.mean), err: dd.error / dd.mean, ok } : { A: R * T, ok: false };
  });
  wilsonPlot.redraw();
}
const gaugeStatsSoon = throttle(updateGaugeStats, 300);

function drawGauge() {
  drawField();
  const n = Math.max(50, histP.n - 1);
  plaqPlot.setX([0, n]).redraw();
  let qmax = 3;
  for (let i = 0; i < histQ.n; i++) qmax = Math.max(qmax, Math.abs(histQ.a[i]));
  qPlot.setX([0, n]).setY([-qmax - 1, qmax + 1]).redraw();
  setText('gl-t-sweeps', fmtInt(histP.n - 1));
  setText('gl-t-since', `${fmtInt(seg.sweeps)} since the last change`);
  setText('gl-t-acc', seg.prop ? pct(seg.acc / seg.prop) : '—');
  setText('gl-t-acc-sub', glAlg() === 'hmc' ? 'HMC trajectories accepted' : 'single-link proposals accepted');
  setText('gl-t-q', String(histQ.last()));
  setText('gl-t-qrate', seg.sweeps ? fmtNum((1000 * seg.qChanges) / seg.sweeps, 1) : '—');
}

function resetGauge() {
  glRunner?.stop();
  const L = Number($('gl-L').value);
  if (!lat || lat.L !== L) {
    lat = new U1Lattice(L, glRng);
    fieldCanvas.width = L;
    fieldCanvas.height = L;
    fieldImg = fctx.createImageData(L, L);
  }
  glRng.seed(randomSeed());
  lat.setBeta(glBeta.get());
  if ($('gl-init').value === 'cold') lat.cold();
  else lat.hot();
  setText('gl-field-sub', `${L} × ${L} plaquettes, periodic in both directions. Color shows each plaquette's angle.`);
  histP.clear();
  histQ.clear();
  changes = [];
  seg = { start: 0, acc: 0, prop: 0, qChanges: 0, sweeps: 0 };
  record();
  markChange();
  glPacer.reset();
  updateGaugeStats();
  drawGauge();
}

const glRunner = new Runner({
  button: 'gl-run',
  root: 'gl-lab',
  tick(dt) {
    const k = glPacer.take(glSpeed.get(), dt, 2000);
    if (k === 0) return true;
    const t0 = performance.now();
    let ok = true;
    for (let i = 0; i < k && ok; i++) {
      ok = sweepOnce();
      if (performance.now() - t0 > 20) break;
    }
    drawGauge();
    gaugeStatsSoon();
    return ok;
  },
  onChange(running) { if (!running) { updateGaugeStats(); drawGauge(); } },
});
$('gl-sweep').addEventListener('click', () => {
  glRunner.stop();
  sweepOnce();
  updateGaugeStats();
  drawGauge();
});
$('gl-reset').addEventListener('click', resetGauge);
$('gl-L').addEventListener('change', resetGauge);
$('gl-init').addEventListener('change', resetGauge);
$('gl-alg').addEventListener('change', onGaugeSettings);
window.addEventListener('themechange', () => lat && drawField());
resetGauge();

// =====================================================================
// The Schwinger model: quarks via pseudofermion HMC
// =====================================================================

const SM_MAX = 50000;
const smRng = new RNG();
let sLat = null, dirac = null, pf = null;
const smPlaq = new Trace(SM_MAX), smCg = new Trace(SM_MAX);
let smStats = null;
const smPacer = new Pacer();

const smBeta = bindRange('sm-beta', { log: true, format: (v) => fmtSig(v, 3), onInput: onQuarkSettings });
const smMass = bindRange('sm-mass', { log: true, format: (v) => fmtSig(v, 2), onInput: onQuarkSettings });
const smSteps = bindRange('sm-steps', { format: (v) => `${v} (ε = ${fmtSig(1 / v, 2)})`, onInput: onQuarkSettings });
const smSpeed = bindRange('sm-speed', { log: true, format: fmtRate('trajectories') });
const quarksOn = () => $('sm-quarks').checked;

const smPlaqPlot = new Plot('sm-plaq', {
  height: 220,
  x: { domain: [0, 50], label: 'trajectory' },
  y: { domain: [0, 1] },
  draw(p) {
    if (!sLat) return;
    const c = p.c;
    p.hline(u1Exact(sLat.beta, sLat.V).plaquette, { color: c.exact, width: 2 });
    p.line(null, smPlaq.a, { color: c.s1, width: smPlaq.n > 2000 ? 1 : 1.5, n: smPlaq.n });
  },
  hover(x) {
    if (!smPlaq.n) return null;
    const i = Math.max(0, Math.min(smPlaq.n - 1, Math.round(x)));
    return {
      x: i,
      title: `trajectory ${fmtInt(i + 1)}`,
      rows: [
        { value: fmtNum(smPlaq.a[i], 4), label: 'plaquette', color: 'var(--s1)' },
        { value: fmtNum(u1Exact(sLat.beta, sLat.V).plaquette, 4), label: 'quenched, exact', color: 'var(--exact)' },
      ],
    };
  },
});
legend('sm-plaq-legend', [
  { label: 'this simulation', color: 'var(--s1)', kind: 'line' },
  { label: 'quenched (no quarks), exact', color: 'var(--exact)', kind: 'line' },
]);

const smCgPlot = new Plot('sm-cg', {
  height: 220,
  x: { domain: [0, 50], label: 'trajectory' },
  y: { domain: [0, 100] },
  draw(p) {
    if (!smCg.n) return;
    p.line(null, smCg.a, { color: p.c.s1, width: 1.5, n: smCg.n });
  },
  hover(x) {
    if (!smCg.n) return null;
    const i = Math.max(0, Math.min(smCg.n - 1, Math.round(x)));
    return { x: i, title: `trajectory ${fmtInt(i + 1)}`, rows: [{ value: fmtNum(smCg.a[i], 1), label: 'CG iterations per solve', color: 'var(--s1)' }] };
  },
});

function resetQuarkStats() {
  smPlaq.clear();
  smCg.clear();
  smStats = { traj: 0, acc: 0, time: 0, cgTime: 0, cgIters: 0, cgSolves: 0, lastDH: NaN };
  smPacer.reset();
}

function onQuarkSettings() {
  if (!sLat) return;
  sLat.setBeta(smBeta.get());
  dirac.mass = smMass.get();
  $('sm-mass-field').classList.toggle('is-disabled', !quarksOn());
  smMass.input.disabled = !quarksOn();
  resetQuarkStats();
  drawQuarks();
}

function resetQuarks() {
  smRunner?.stop();
  const L = Number($('sm-L').value);
  smRng.seed(randomSeed());
  sLat = new U1Lattice(L, smRng);
  sLat.setBeta(smBeta.get());
  sLat.hot();
  dirac = new WilsonDirac(sLat, smMass.get());
  pf = new Pseudofermion(dirac);
  onQuarkSettings();
}

function trajectoryOnce() {
  const steps = smSteps.get(), eps = 1 / steps;
  const t0 = performance.now();
  let r;
  if (quarksOn()) {
    const it0 = pf.cgIterations, so0 = pf.cgSolves, ct0 = pf.cgTime;
    r = schwingerTrajectory(sLat, pf, { eps, steps });
    const its = pf.cgIterations - it0, solves = pf.cgSolves - so0;
    smStats.cgIters += its;
    smStats.cgSolves += solves;
    smStats.cgTime += pf.cgTime - ct0;
    smCg.push(its / solves);
  } else {
    r = hmcTrajectory(sLat, { eps, steps, forceFn: (f) => sLat.gaugeForce(f), actionFn: () => sLat.action() });
  }
  smStats.time += performance.now() - t0;
  smStats.traj++;
  if (r.accept) smStats.acc++;
  smStats.lastDH = r.dH;
  smPlaq.push(sLat.meanCosPlaquette());
  return !smPlaq.full;
}

function drawQuarks() {
  const n = Math.max(50, smPlaq.n);
  let lo = 1, hi = 0;
  for (let i = 0; i < smPlaq.n; i++) { lo = Math.min(lo, smPlaq.a[i]); hi = Math.max(hi, smPlaq.a[i]); }
  const q = u1Exact(sLat.beta, sLat.V).plaquette;
  lo = Math.min(lo, q); hi = Math.max(hi, q);
  const pad = 0.1 * (hi - lo || 0.2);
  smPlaqPlot.setX([0, n]).setY([Math.max(-1, lo - pad), Math.min(1, hi + pad)]).redraw();
  let cmax = 10;
  for (let i = 0; i < smCg.n; i++) cmax = Math.max(cmax, smCg.a[i]);
  smCgPlot.setX([0, n]).setY([0, cmax * 1.15]).redraw();

  const s = smStats;
  setText('sm-t-traj', fmtInt(s.traj));
  setText('sm-t-ms', s.traj ? `${fmtSig(s.time / s.traj, 2)} ms each` : '');
  setText('sm-t-acc', s.traj ? pct(s.acc / s.traj) : '—');
  setText('sm-t-dh', Number.isFinite(s.lastDH) ? `last ΔH = ${fmtSig(s.lastDH, 2)}` : '');
  const from = Math.floor(0.2 * smPlaq.n);
  const d = smPlaq.n - from >= 20 ? chainDiagnostics(smPlaq.a, from, smPlaq.n) : null;
  setText('sm-t-plaq', d ? pm(d) : '—');
  setText('sm-t-quenched', `quenched, exact: ${fmtNum(q, 4)}`);
  const on = quarksOn();
  setText('sm-t-cg', on && s.cgSolves ? fmtNum(s.cgIters / s.cgSolves, 0) : on ? '—' : 'no quarks');
  setText('sm-t-cgshare', on && s.time ? pct(s.cgTime / s.time) : on ? '—' : 'no quarks');
  setText('sm-t-solves', on && s.traj ? fmtNum(s.cgSolves / s.traj, 0) : on ? '—' : 'no quarks');
}
const drawQuarksSoon = throttle(drawQuarks, 150);

const smRunner = new Runner({
  button: 'sm-run',
  root: 'sm-lab',
  tick(dt) {
    const k = smPacer.take(smSpeed.get(), dt, 100);
    if (k === 0) return true;
    const t0 = performance.now();
    let ok = true;
    for (let i = 0; i < k && ok; i++) {
      ok = trajectoryOnce();
      if (performance.now() - t0 > 30) break;
    }
    drawQuarksSoon();
    return ok;
  },
  onChange(running) { if (!running) drawQuarks(); },
});
$('sm-reset').addEventListener('click', resetQuarks);
$('sm-L').addEventListener('change', resetQuarks);
$('sm-quarks').addEventListener('change', onQuarkSettings);
resetQuarks();
