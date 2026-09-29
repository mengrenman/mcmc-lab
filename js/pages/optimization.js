import { mountChrome, $, setText, bindRange, Runner, Pacer, throttle } from '../lib/ui.js';
import { Plot, legend, nearestIndex, fmtNum, fmtSig, fmtInt } from '../lib/plot.js';
import { RNG, randomSeed } from '../lib/random.js';
import { Trace, chainDiagnostics, histogram } from '../lib/stats.js';
import { logGamma } from '../lib/bayes.js';
import { doubleWell, equilibrium, WalkerPopulation, geometricLadder, ReplicaExchange1D } from '../samplers/annealing.js';
import { TspProblem, greedyDescent, bruteForce, TspReplicaExchange } from '../samplers/tsp.js';

mountChrome();

const fmtRate = (unit) => (v) => (v < 10 ? fmtNum(v, 1) : fmtInt(v)) + ` ${unit}/s`;
const pct = (v) => `${fmtNum(100 * v, 1)}%`;
const FRAME_BUDGET_MS = 12;

/** Exact P(x > 0) as a function of T, tabulated once on a log grid and interpolated. */
const probRight = (() => {
  const lo = Math.log(0.0005), hi = Math.log(4), K = 400;
  const table = new Float64Array(K + 1);
  for (let i = 0; i <= K; i++) table[i] = equilibrium(Math.exp(lo + ((hi - lo) * i) / K), -2.2, 2.2, 2001).probRight;
  return (T) => {
    const u = ((Math.log(T) - lo) / (hi - lo)) * K;
    const i = Math.max(0, Math.min(K - 1, Math.floor(u)));
    const t = Math.max(0, Math.min(1, u - i));
    return table[i] * (1 - t) + table[i + 1] * t;
  };
})();

// =====================================================================
// Simulated annealing with a population of walkers
// =====================================================================

const WALKERS = 400, STEP = 0.1, T_END = 0.001, SA_MAX = 400000;
const saRng = new RNG();
const pop = new WalkerPopulation(WALKERS, saRng);
const saFrac = new Trace(SA_MAX), saExact = new Trace(SA_MAX), saTemp = new Trace(SA_MAX);
const saPacer = new Pacer();

const saT = bindRange('sa-T', { log: true, format: (v) => fmtSig(v, 2), onInput: resetAnneal });
const saL = bindRange('sa-L', { log: true, format: (v) => fmtInt(v), onInput: resetAnneal });
const saSpeed = bindRange('sa-speed', { log: true, format: fmtRate('steps') });
const annealing = () => $('sa-mode').value === 'anneal';
const annealLength = () => Math.round(saL.get());

function temperatureAt(step) {
  if (!annealing()) return saT.get();
  const L = annealLength(), T0 = saT.get();
  return T0 * Math.pow(T_END / T0, Math.min(1, step / (L - 1)));
}

const land = new Plot('sa-land', {
  height: 240,
  x: { domain: [-1.7, 1.7], label: 'x' },
  y: { domain: [-0.05, 1.7] },
  draw(p) {
    const c = p.c;
    p.fn(doubleWell, { color: c.exact, width: 2 });
    const ctx = p.ctx;
    ctx.save();
    ctx.fillStyle = c.s1;
    ctx.globalAlpha = 0.18;
    for (const x of pop.x) {
      ctx.beginPath();
      ctx.arc(p.sx(x), p.sy(doubleWell(x)), 4, 0, 2 * Math.PI);
      ctx.fill();
    }
    ctx.restore();
    const right = Math.round(pop.fractionRight() * WALKERS);
    p.text(-1, 0.55, `${WALKERS - right} walkers`, { align: 'center', color: c.ink2, weight: 600, size: 12 });
    p.text(1, 0.55, `${right} walkers`, { align: 'center', color: c.ink2, weight: 600, size: 12 });
    p.text(-1, 0.42, 'local minimum', { align: 'center', color: c.muted });
    p.text(1, 0.42, 'global minimum', { align: 'center', color: c.muted });
  },
  hover(x) {
    return { x, title: `x = ${fmtNum(x, 2)}`, rows: [{ value: fmtNum(doubleWell(x), 3), label: 'f(x)', color: 'var(--exact)' }] };
  },
});
legend('sa-land-legend', [
  { label: 'f(x)', color: 'var(--exact)', kind: 'line' },
  { label: 'walkers', color: 'var(--s1)', kind: 'dot' },
]);

const fracPlot = new Plot('sa-frac', {
  height: 210,
  x: { domain: [0, 100], label: 'step' },
  y: { domain: [0, 1.02] },
  draw(p) {
    const c = p.c;
    p.hline(0.5, { color: c.axis });
    // Index i holds the state after step i + 1 (at step 0 every walker sits exactly on x = 0).
    p.line(null, saExact.a, { color: c.exact, width: 2, n: saExact.n, x0: 1 });
    p.line(null, saFrac.a, { color: c.s1, width: 1.5, n: saFrac.n, x0: 1 });
  },
  hover(x) {
    if (!saFrac.n) return null;
    const i = Math.max(0, Math.min(saFrac.n - 1, Math.round(x) - 1));
    return {
      x: i + 1,
      title: `step ${fmtInt(i + 1)} · T = ${fmtSig(saTemp.a[i], 3)}`,
      rows: [
        { value: pct(saFrac.a[i]), label: 'walkers', color: 'var(--s1)' },
        { value: pct(saExact.a[i]), label: 'equilibrium', color: 'var(--exact)' },
      ],
    };
  },
});
legend('sa-frac-legend', [
  { label: 'walkers with x > 0', color: 'var(--s1)', kind: 'line' },
  { label: 'exact equilibrium at this T', color: 'var(--exact)', kind: 'line' },
]);

const tempPlot = new Plot('sa-temp', {
  height: 210,
  x: { domain: [0, 100], label: 'step' },
  y: { type: 'log', domain: [0.0008, 2.5] },
  draw(p) {
    const c = p.c;
    const n = annealing() ? annealLength() : Math.max(saTemp.n, 100);
    const xs = [], ys = [];
    for (let i = 0; i <= 200; i++) {
      const s = (i / 200) * (n - 1);
      xs.push(s);
      ys.push(temperatureAt(s));
    }
    p.line(xs, ys, { color: c.ink2, width: 1.5, alpha: 0.5 });
    if (pop.steps) p.marker(pop.steps, temperatureAt(pop.steps), { color: c.s1, r: 5 });
  },
  hover(x) {
    return { x, title: `step ${fmtInt(x)}`, rows: [{ value: fmtSig(temperatureAt(Math.max(0, x)), 3), label: 'T' }] };
  },
});

function drawAnneal() {
  const n = annealing() ? annealLength() : Math.max(100, saFrac.n + 1);
  fracPlot.setX([0, n]);
  tempPlot.setX([0, n]);
  land.redraw();
  fracPlot.redraw();
  tempPlot.redraw();
  const T = temperatureAt(pop.steps);
  setText('sa-t-step', fmtInt(pop.steps));
  setText('sa-t-step-sub', annealing() ? `of ${fmtInt(annealLength())}` : 'fixed temperature');
  setText('sa-t-T', fmtSig(T, 3));
  setText('sa-t-right', pct(pop.fractionRight()));
  setText('sa-t-exact', pct(probRight(T)));
}

function record() {
  const T = temperatureAt(pop.steps - 1); // the temperature the last step used
  saFrac.push(pop.fractionRight());
  saExact.push(probRight(T));
  saTemp.push(T);
}

function resetAnneal() {
  saRunner?.stop();
  const anneal = annealing();
  setText('sa-T-label', anneal ? 'Starting temperature T₀' : 'Temperature T');
  $('sa-L-field').classList.toggle('is-disabled', !anneal);
  saL.input.disabled = !anneal;
  saRng.seed(randomSeed());
  pop.reset(0);
  saFrac.clear();
  saExact.clear();
  saTemp.clear();
  saPacer.reset();
  drawAnneal();
}

const saRunner = new Runner({
  button: 'sa-run',
  root: 'sa-lab',
  tick(dt) {
    const k = saPacer.take(saSpeed.get(), dt, 100000);
    if (k === 0) return true;
    const t0 = performance.now();
    const end = annealing() ? annealLength() : SA_MAX - 1;
    for (let i = 0; i < k && pop.steps < end; i++) {
      pop.step(temperatureAt(pop.steps), STEP);
      record();
      if ((i & 15) === 15 && performance.now() - t0 > FRAME_BUDGET_MS) break;
    }
    drawAnneal();
    return pop.steps < end;
  },
});
$('sa-reset').addEventListener('click', resetAnneal);
$('sa-mode').addEventListener('change', resetAnneal);
resetAnneal();

// =====================================================================
// Replica exchange on the double well
// =====================================================================

const RX_MAX = 400000, TRAIL = 400, BINS = 128, H_LO = -1.6, H_HI = 1.6;
const rxRng = new RNG();
let rx = null, temps = [];
let rungX = new Trace(RX_MAX); // x on the measured rung, one per step
let cumRight = new Trace(RX_MAX + 1); // cumulative count of x > 0 on the measured rung
let trailX = new Float64Array(TRAIL), trailT = new Float64Array(TRAIL), trailN = 0;
let exactBins = null, rxDiag = null;
const rxPacer = new Pacer();

const rxM = bindRange('rx-M', { format: String, onInput: resetReplicas });
const rxRung = bindRange('rx-rung', { format: (m) => `${m} (T = ${fmtSig(temps[m] ?? NaN, 2)})`, onInput: resetRungStats });
const rxSpeed = bindRange('rx-speed', { log: true, format: fmtRate('steps') });
const measured = () => Math.min(Number(rxRung.input.value), temps.length - 1);

const ladder = new Plot('rx-ladder', {
  height: 300,
  x: { domain: [-1.7, 1.7], label: 'x' },
  y: { type: 'log', domain: [0.0006, 3.5], label: 'temperature T' },
  draw(p) {
    if (!rx) return;
    const c = p.c;
    for (const [x, label] of [[-1, 'local min'], [1, 'global min']]) {
      p.vline(x, { color: c.exact, alpha: 0.35 });
      p.text(x, p.y.domain[1], label, { align: 'center', dy: 8, color: c.muted });
    }
    p.vline(0, { color: c.axis });
    const m = measured();
    p.hline(temps[m], { color: c.ink2, alpha: 0.25 });
    // The tracked walker's recent path, oldest first.
    const k = Math.min(trailN, TRAIL);
    if (k > 1) {
      const xs = new Float64Array(k), ts = new Float64Array(k);
      const start = trailN > TRAIL ? trailN % TRAIL : 0;
      for (let i = 0; i < k; i++) {
        xs[i] = trailX[(start + i) % TRAIL];
        ts[i] = trailT[(start + i) % TRAIL];
      }
      p.line(xs, ts, { color: c.s2, width: 1.2, alpha: 0.55 });
    }
    p.points(rx.x, temps, { color: c.s1, r: 4, ring: true });
    p.marker(rx.x[m], temps[m], { color: c.ink2, r: 8, hollow: true });
    const w = rx.rungOf(0);
    p.marker(rx.x[w], temps[w], { color: c.s2, r: 5 });
  },
  hover(x, T) {
    if (!rx) return null;
    const logs = Array.from(temps, (t) => -Math.log(t));
    const m = nearestIndex(logs, -Math.log(T));
    return { title: `rung ${m} · T = ${fmtSig(temps[m], 3)}`, rows: [{ value: fmtNum(rx.x[m], 3), label: 'x on this rung', color: 'var(--s1)' }, { value: pct(probRight(temps[m])), label: 'exact P(x > 0)', color: 'var(--exact)' }] };
  },
});
legend('rx-ladder-legend', [
  { label: 'replica on each rung', color: 'var(--s1)', kind: 'dot' },
  { label: 'one walker followed through swaps', color: 'var(--s2)', kind: 'line' },
  { label: 'measured rung', color: 'var(--ink-2)', kind: 'dot' },
]);

let rxHist = null;
const rxHistPlot = new Plot('rx-hist', {
  height: 220,
  x: { domain: [H_LO, H_HI], label: 'x' },
  draw(p) {
    const c = p.c;
    if (rxHist) p.bars(rxHist.edges, rxHist.heights, { color: c.s1, alpha: 0.85, gap: 0 });
    if (exactBins) p.line(exactBins.centers, exactBins.density, { color: c.exact, width: 2 });
  },
  hover(x) {
    const w = (H_HI - H_LO) / BINS;
    const k = Math.floor((x - H_LO) / w);
    if (k < 0 || k >= BINS || !exactBins) return null;
    return {
      title: `x in [${fmtNum(H_LO + k * w, 3)}, ${fmtNum(H_LO + (k + 1) * w, 3)})`,
      rows: [
        { value: rxHist ? fmtSig(rxHist.heights[k], 3) : '—', label: 'samples', color: 'var(--s1)' },
        { value: fmtSig(exactBins.density[k], 3), label: 'exact, bin average', color: 'var(--exact)' },
      ],
    };
  },
});
legend('rx-hist-legend', [
  { label: 'samples on the measured rung', color: 'var(--s1)', kind: 'rect' },
  { label: 'exact e^(−f/T), bin-averaged', color: 'var(--exact)', kind: 'line' },
]);

const rxFrac = new Plot('rx-frac', {
  height: 220,
  x: { domain: [0, 100], label: 'step' },
  y: { domain: [0, 1.02] },
  draw(p) {
    if (!rx) return;
    const c = p.c;
    p.hline(probRight(temps[measured()]), { color: c.exact, width: 2 });
    const n = rungX.n;
    if (n < 2) return;
    const xs = [], ys = [];
    const K = Math.min(400, n);
    for (let i = 1; i <= K; i++) {
      const s = Math.max(1, Math.round((i / K) * n));
      xs.push(s);
      ys.push(runningP(s));
    }
    p.line(xs, ys, { color: c.s1, width: 1.5 });
  },
  hover(x) {
    if (!rx || rungX.n < 2) return null;
    const s = Math.max(1, Math.min(rungX.n, Math.round(x)));
    return { x: s, title: `after ${fmtInt(s)} steps`, rows: [{ value: pct(runningP(s)), label: 'running P(x > 0)', color: 'var(--s1)' }, { value: pct(probRight(temps[measured()])), label: 'exact', color: 'var(--exact)' }] };
  },
});

/** P(x > 0) over steps (0.1 s, s], i.e. after discarding the first 10%. */
function runningP(s) {
  const a = Math.floor(0.1 * s);
  return (cumRight.a[s] - cumRight.a[a]) / (s - a);
}

function computeExactBins(T) {
  const eq = equilibrium(T, -2.2, 2.2, 8001);
  const w = (H_HI - H_LO) / BINS, dx = eq.x[1] - eq.x[0];
  const mass = new Float64Array(BINS);
  for (let i = 0; i < eq.x.length; i++) {
    const k = Math.floor((eq.x[i] - H_LO) / w);
    if (k >= 0 && k < BINS) mass[k] += eq.p[i] * dx;
  }
  return {
    centers: Float64Array.from({ length: BINS }, (_, k) => H_LO + (k + 0.5) * w),
    density: mass.map((m) => m / w),
  };
}

function resetRungStats() {
  rungX.clear();
  cumRight.clear();
  cumRight.push(0);
  exactBins = computeExactBins(temps[measured()]);
  rxDiag = null;
  updateReplicaStats();
  drawReplicas();
}

function resetReplicas() {
  rxRunner?.stop();
  const M = Math.round(rxM.get());
  temps = geometricLadder(2, 0.001, M);
  rxRung.input.max = M - 1;
  if (Number(rxRung.input.value) > M - 1) rxRung.input.value = M - 1;
  rxRung.set(Number(rxRung.input.value)); // refresh the label
  rxRng.seed(randomSeed());
  rx = new ReplicaExchange1D(temps, rxRng);
  rx.reset(-1);
  trailN = 0;
  rxPacer.reset();
  resetRungStats();
}

function updateReplicaStats() {
  const n = rungX.n, a = Math.floor(0.1 * n);
  if (n - a >= 50) {
    const ind = new Float64Array(n - a);
    for (let i = a; i < n; i++) ind[i - a] = rungX.a[i] > 0 ? 1 : 0;
    rxDiag = chainDiagnostics(ind);
  } else {
    rxDiag = null;
  }
  rxHist = n - a > 0 ? histogram(rungX.a, H_LO, H_HI, BINS, a, n) : null;
  let top = 0;
  for (const v of exactBins.density) top = Math.max(top, v);
  if (rxHist) for (const v of rxHist.heights) top = Math.max(top, v);
  rxHistPlot.setY([0, top * 1.1 || 1]);
  rxHistPlot.redraw();
  const T = temps[measured()];
  setText('rx-t-T', fmtSig(T, 3));
  setText('rx-t-exact', pct(probRight(T)));
  if (rxDiag && !Number.isFinite(rxDiag.error)) {
    // A constant indicator: the chain has stayed on one side of the hill the whole time.
    setText('rx-t-p', `${pct(rxDiag.mean)}, never crossed`);
  } else if (rxDiag) {
    const approx = rxDiag.converged ? '' : '≳ ';
    setText('rx-t-p', `${pct(rxDiag.mean)} ± ${approx}${fmtNum(100 * rxDiag.error, 1)}`);
  } else {
    setText('rx-t-p', '—');
  }
}
const replicaStatsSoon = throttle(updateReplicaStats, 300);

function drawReplicas() {
  rxFrac.setX([0, Math.max(100, rungX.n)]);
  ladder.redraw();
  rxFrac.redraw();
  setText('rx-t-steps', fmtInt(rx.steps));
  setText('rx-t-cost', `${temps.length} Metropolis updates per step`);
  setText('rx-t-swap', $('rx-swap').checked ? (rx.steps ? pct(rx.swapRate) : '—') : 'swaps off');
  setText('rx-t-walker', `${rx.rungOf(0)} of ${temps.length - 1}`);
}

const rxRunner = new Runner({
  button: 'rx-run',
  root: 'rx-lab',
  tick(dt) {
    const k = rxPacer.take(rxSpeed.get(), dt, 50000);
    if (k === 0) return true;
    const t0 = performance.now();
    const swaps = $('rx-swap').checked, m = measured();
    for (let i = 0; i < k && !rungX.full; i++) {
      rx.step(swaps);
      const x = rx.x[m];
      rungX.push(x);
      cumRight.push(cumRight.last() + (x > 0 ? 1 : 0));
      const w = rx.rungOf(0);
      trailX[trailN % TRAIL] = rx.x[w];
      trailT[trailN % TRAIL] = temps[w];
      trailN++;
      if ((i & 63) === 63 && performance.now() - t0 > FRAME_BUDGET_MS) break;
    }
    drawReplicas();
    replicaStatsSoon();
    return !rungX.full;
  },
  onChange(running) { if (!running) updateReplicaStats(); },
});
$('rx-reset').addEventListener('click', resetReplicas);
$('rx-swap').addEventListener('change', resetReplicas);
resetReplicas();

// =====================================================================
// Traveling salesman via replica exchange
// =====================================================================

const TS_MAX = 1000000;
let citySeed = randomSeed();
const tsRng = new RNG();
let problem = null, tsRx = null, exact = null, greedyLength = NaN, lastImproved = 0;
let lenLo = Infinity, lenHi = -Infinity; // running range of the recorded lengths
const coldLen = new Trace(TS_MAX), bestLen = new Trace(TS_MAX);
const tsPacer = new Pacer();

const tsN = bindRange('ts-N', { format: String, onInput: setupCities });
const tsM = bindRange('ts-M', { format: String, onInput: resetTsp });
const tsBeta = bindRange('ts-beta', { log: true, format: (v) => fmtSig(v, 3), onInput: resetTsp });
const tsSpeed = bindRange('ts-speed', { log: true, format: fmtRate('steps') });

function routeCount(N) {
  if (N <= 18) {
    let f = 1;
    for (let k = 2; k < N; k++) f *= k;
    return fmtInt(f / 2);
  }
  const log10 = (logGamma(N) - Math.log(2)) / Math.LN10;
  const e = Math.floor(log10);
  const sup = String(e).replace(/\d/g, (d) => '⁰¹²³⁴⁵⁶⁷⁸⁹'[d]);
  return `${fmtNum(10 ** (log10 - e), 1)} × 10${sup}`;
}

const tsMap = new Plot('ts-map', {
  square: true,
  maxHeight: 400,
  grid: 'none',
  x: { domain: [0, 1], ticks: 5 },
  y: { domain: [0, 1], ticks: 5 },
  margin: { l: 36 },
  draw(p) {
    if (!problem) return;
    const c = p.c;
    const route = (tour, opts) => {
      const xs = [], ys = [];
      for (let i = 0; i <= tour.length; i++) {
        const city = tour[i % tour.length];
        xs.push(problem.xs[city]);
        ys.push(problem.ys[city]);
      }
      p.line(xs, ys, opts);
    };
    if (exact) route(exact.tour, { color: c.exact, width: 7, alpha: 0.25 });
    if (tsRx) route(tsRx.best, { color: c.s1, width: 2 });
    p.points(problem.xs, problem.ys, { color: c.ink, r: 3 });
    p.marker(problem.xs[0], problem.ys[0], { color: c.s2, r: 6 });
  },
  hover(x, y) {
    if (!problem) return null;
    let best = 0, bd = Infinity;
    for (let i = 0; i < problem.N; i++) {
      const d = Math.hypot(problem.xs[i] - x, problem.ys[i] - y);
      if (d < bd) { bd = d; best = i; }
    }
    if (bd > 0.05) return null;
    return { title: best === 0 ? 'city 0 (home)' : `city ${best}`, rows: [{ value: `(${fmtNum(problem.xs[best], 3)}, ${fmtNum(problem.ys[best], 3)})`, label: 'position' }] };
  },
});

const tsLen = new Plot('ts-len', {
  height: (w) => Math.min(400, Math.max(240, w * 0.85)),
  x: { domain: [0, 100], label: 'step' },
  y: { domain: [0, 1] },
  draw(p) {
    if (!tsRx) return;
    const c = p.c;
    if (exact) p.hline(exact.length, { color: c.exact, width: 2 });
    p.hline(greedyLength, { color: c.s2, width: 1.5 });
    p.line(null, coldLen.a, { color: c.s1, width: 1, alpha: 0.6, n: coldLen.n });
    p.line(null, bestLen.a, { color: c.s3, width: 2, n: bestLen.n });
  },
  hover(x) {
    if (!coldLen.n) return null;
    const i = Math.max(0, Math.min(coldLen.n - 1, Math.round(x)));
    const rows = [
      { value: fmtNum(coldLen.a[i], 4), label: 'coldest replica', color: 'var(--s1)' },
      { value: fmtNum(bestLen.a[i], 4), label: 'best so far', color: 'var(--s3)' },
      { value: fmtNum(greedyLength, 4), label: 'greedy descent', color: 'var(--s2)' },
    ];
    if (exact) rows.push({ value: fmtNum(exact.length, 4), label: 'true shortest', color: 'var(--exact)' });
    return { x: i, title: `step ${fmtInt(i)}`, rows };
  },
});

function refreshLengthLegend() {
  const items = [
    { label: 'coldest replica', color: 'var(--s1)', kind: 'line' },
    { label: 'best so far', color: 'var(--s3)', kind: 'line' },
    { label: 'greedy descent', color: 'var(--s2)', kind: 'line' },
  ];
  if (exact) items.push({ label: 'true shortest (brute force)', color: 'var(--exact)', kind: 'line' });
  legend('ts-len-legend', items);
}

function setupCities() {
  const N = Math.round(tsN.get());
  const r = new RNG(citySeed);
  const xs = [], ys = [];
  for (let i = 0; i < N; i++) {
    xs.push(r.uniform());
    ys.push(r.uniform());
  }
  problem = new TspProblem(xs, ys);
  if (N <= 10) {
    const t0 = performance.now();
    exact = bruteForce(problem);
    let perms = 1;
    for (let k = 2; k < N; k++) perms *= k;
    setText('ts-t-exact-sub', `checked all ${fmtInt(perms)} orderings in ${fmtNum(performance.now() - t0, 0)} ms`);
  } else {
    exact = null;
    setText('ts-t-exact-sub', 'too many routes to check (N > 10)');
  }
  setText('ts-t-exact', exact ? fmtNum(exact.length, 4) : '—');
  setText('ts-t-routes', routeCount(N));
  refreshLengthLegend();
  resetTsp();
}

function resetTsp() {
  tsRunner?.stop();
  const M = Math.round(tsM.get()), bmax = tsBeta.get();
  const move = $('ts-move').value;
  tsRng.seed(randomSeed());
  const identity = Int32Array.from({ length: problem.N }, (_, i) => i);
  greedyLength = greedyDescent(problem, identity, move, tsRng);
  tsRx = new TspReplicaExchange(problem, Array.from({ length: M }, (_, m) => (bmax * (m + 1)) / M), tsRng, move);
  coldLen.clear();
  bestLen.clear();
  coldLen.push(tsRx.L[M - 1]);
  bestLen.push(tsRx.bestLength);
  lenLo = tsRx.bestLength;
  lenHi = tsRx.L[M - 1];
  lastImproved = 0;
  tsPacer.reset();
  setText('ts-t-greedy', fmtNum(greedyLength, 4));
  drawTsp();
}

function drawTsp() {
  const n = coldLen.n;
  const lo = Math.min(lenLo, greedyLength, exact ? exact.length : Infinity);
  // The starting route is far longer than anything interesting; clip the axis above greedy descent.
  const hi = Math.min(lenHi, 1.3 * Math.max(greedyLength, lenLo));
  const pad = 0.05 * (hi - lo || 1);
  tsLen.setX([0, Math.max(100, n - 1)]).setY([Math.max(0, lo - pad), hi + pad]);
  tsMap.redraw();
  tsLen.redraw();
  const M = tsRx.beta.length;
  setText('ts-t-best', fmtNum(tsRx.bestLength, 4));
  const gap = exact ? tsRx.bestLength - exact.length : NaN;
  setText('ts-t-best-sub', exact && gap < 1e-9 ? `optimal, found at step ${fmtInt(lastImproved)}` : `last improved at step ${fmtInt(lastImproved)}`);
  setText('ts-t-steps', fmtInt(tsRx.steps));
  setText('ts-t-moves', `${fmtInt(tsRx.moveTried)} moves over ${M} replicas`);
  setText('ts-t-acc', tsRx.moveTried ? pct(tsRx.moveDone / tsRx.moveTried) : '—');
  setText('ts-t-swap', tsRx.swapTried ? `swaps: ${pct(tsRx.swapDone / tsRx.swapTried)}` : 'moves; swaps not yet tried');
}

const tsRunner = new Runner({
  button: 'ts-run',
  root: 'ts-lab',
  tick(dt) {
    const k = tsPacer.take(tsSpeed.get(), dt, 200000);
    if (k === 0) return true;
    const t0 = performance.now();
    const M = tsRx.beta.length;
    for (let i = 0; i < k && !coldLen.full; i++) {
      const before = tsRx.bestLength;
      tsRx.step();
      if (tsRx.bestLength < before) lastImproved = tsRx.steps;
      if (tsRx.steps % 5000 === 0) tsRx.refresh();
      coldLen.push(tsRx.L[M - 1]);
      bestLen.push(tsRx.bestLength);
      if (tsRx.L[M - 1] > lenHi) lenHi = tsRx.L[M - 1];
      if (tsRx.bestLength < lenLo) lenLo = tsRx.bestLength;
      if ((i & 31) === 31 && performance.now() - t0 > FRAME_BUDGET_MS) break;
    }
    drawTsp();
    return !coldLen.full;
  },
});
$('ts-reset').addEventListener('click', resetTsp);
$('ts-move').addEventListener('change', resetTsp);
$('ts-new').addEventListener('click', () => {
  citySeed = randomSeed();
  setupCities();
});
setupCities();
