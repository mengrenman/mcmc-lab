import { mountChrome, $, setText, bindRange, Runner, Pacer, throttle } from '../lib/ui.js';
import { Plot, legend, nearestIndex, densityImage, palette, withAlpha, fmtNum, fmtSig, fmtInt } from '../lib/plot.js';
import { RNG, randomSeed } from '../lib/random.js';
import { Trace, chainDiagnostics, histogram } from '../lib/stats.js';
import { normPdf } from '../lib/targets.js';
import { Metropolis1D, Metropolis2D } from '../samplers/metropolis.js';
import {
  coinLogPrior, coinLogLikelihood, coinPosterior,
  gaussianStats, gaussianAction, gaussianPosteriorExact,
} from '../lib/bayes.js';

mountChrome();

const fmtRate = (unit) => (v) => (v < 10 ? fmtNum(v, 1) : fmtInt(v)) + ` ${unit}/s`;
const MAX_TOSSES = 100000;

// =====================================================================
// Coin: data, prior and the exact posterior
// =====================================================================

const coinRng = new RNG();
let heads = []; // heads[i] = number of heads after i + 1 tosses
let replayed = false; // true when loaded data are replayed in a random order
let prior = { kind: 'bump', p0: 0.5, M: 100 };
let post = null; // exact posterior for the current data
let view = null; // every 4th grid point of post, for drawing smooth curves
let priorOnly = null; // posterior with no data, i.e. the normalized prior
let history = []; // [{ n, mean, lo, hi, ml }]
const summaryCache = new Map(); // n -> summary, valid for the current prior

const trueBias = bindRange('co-true', { format: (v) => fmtNum(v, 2) });
const p0 = bindRange('co-p0', { format: (v) => fmtNum(v, 2), onInput: onPriorChange });
const strength = bindRange('co-M', { log: true, format: (v) => fmtInt(v), onInput: onPriorChange });

const coinN = () => heads.length;
const coinK = () => (heads.length ? heads[heads.length - 1] : 0);

const curves = new Plot('co-curves', {
  height: 240,
  x: { domain: [0, 1], label: 'p' },
  y: { domain: [0, 3] },
  draw(p) {
    const c = p.c;
    if (!view) return;
    p.line(view.p, view.prior, { color: c.s2 });
    if (coinN() > 0) p.line(view.p, view.likelihood, { color: c.s3 });
    const ctx = p.ctx;
    ctx.save();
    ctx.fillStyle = withAlpha(c.exact, 0.1);
    ctx.beginPath();
    ctx.moveTo(p.sx(0), p.sy(0));
    for (let i = 0; i < view.p.length; i++) ctx.lineTo(p.sx(view.p[i]), p.sy(view.posterior[i]));
    ctx.lineTo(p.sx(1), p.sy(0));
    ctx.closePath();
    ctx.fill();
    ctx.restore();
    p.line(view.p, view.posterior, { color: c.exact, width: 2.5 });
  },
  hover(x) {
    if (!post) return null;
    const i = Math.max(0, Math.min(post.p.length - 1, Math.round(x * (post.p.length - 1))));
    const rows = [{ value: fmtSig(post.posterior[i], 3), label: 'posterior', color: 'var(--exact)' }];
    if (coinN() > 0) rows.push({ value: fmtSig(post.likelihood[i], 3), label: 'likelihood', color: 'var(--s3)' });
    rows.push({ value: fmtSig(post.prior[i], 3), label: 'prior', color: 'var(--s2)' });
    return { x: post.p[i], title: `p = ${fmtNum(post.p[i], 3)}`, rows };
  },
});
legend('co-curves-legend', [
  { label: 'prior P(p)', color: 'var(--s2)', kind: 'line' },
  { label: 'likelihood P(data | p)', color: 'var(--s3)', kind: 'line' },
  { label: 'posterior P(p | data)', color: 'var(--exact)', kind: 'line' },
]);

const historyPlot = new Plot('co-history', {
  height: 220,
  x: { domain: [0, 10], label: 'tosses n' },
  y: { domain: [0, 1] },
  draw(p) {
    const c = p.c;
    if (!history.length) return;
    const ctx = p.ctx;
    ctx.save();
    ctx.fillStyle = withAlpha(c.exact, 0.15);
    ctx.beginPath();
    ctx.moveTo(p.sx(0), p.sy(priorOnly.quantile(0.975)));
    for (const h of history) ctx.lineTo(p.sx(h.n), p.sy(h.hi));
    for (let i = history.length - 1; i >= 0; i--) ctx.lineTo(p.sx(history[i].n), p.sy(history[i].lo));
    ctx.lineTo(p.sx(0), p.sy(priorOnly.quantile(0.025)));
    ctx.closePath();
    ctx.fill();
    ctx.restore();
    p.hline(trueBias.get(), { color: c.ink2, alpha: 0.6 });
    p.text(p.x.domain[1], trueBias.get(), `true bias ${fmtNum(trueBias.get(), 2)}`, { align: 'right', dy: -8, color: c.muted });
    const ns = [0, ...history.map((h) => h.n)];
    p.line(history.map((h) => h.n), history.map((h) => h.ml), { color: c.s3, width: 1.5 });
    p.line(ns, [priorOnly.mean, ...history.map((h) => h.mean)], { color: c.exact, width: 2 });
  },
  hover(x) {
    if (!history.length) return null;
    const ns = history.map((h) => h.n);
    const h = history[nearestIndex(ns, x)];
    return {
      x: h.n,
      title: `after ${fmtInt(h.n)} toss${h.n === 1 ? '' : 'es'}`,
      rows: [
        { value: fmtNum(h.mean, 3), label: 'posterior mean', color: 'var(--exact)' },
        { value: `${fmtNum(h.lo, 3)} to ${fmtNum(h.hi, 3)}`, label: '95% interval' },
        { value: fmtNum(h.ml, 3), label: 'maximum likelihood k/n', color: 'var(--s3)' },
      ],
    };
  },
});
legend('co-history-legend', [
  { label: 'posterior mean and 95% interval', color: 'var(--exact)', kind: 'line' },
  { label: 'maximum likelihood k/n', color: 'var(--s3)', kind: 'line' },
]);

function summaryAt(n) {
  let s = summaryCache.get(n);
  if (!s) {
    const k = n ? heads[n - 1] : 0;
    const q = coinPosterior(prior, n, k, 801);
    s = { n, mean: q.mean, lo: q.quantile(0.025), hi: q.quantile(0.975), ml: k / n };
    summaryCache.set(n, s);
  }
  return s;
}

function rebuildHistory() {
  const n = coinN();
  history = [];
  if (!n) return;
  const step = Math.max(1, Math.ceil(n / 300));
  for (let i = step; i <= n; i += step) history.push(summaryAt(i));
  if (history[history.length - 1].n !== n) history.push(summaryAt(n));
}

function updateCoin() {
  const n = coinN(), k = coinK();
  post = coinPosterior(prior, n, k);
  const every4 = (a) => a.filter((_, i) => i % 4 === 0);
  view = { p: every4(post.p), prior: every4(post.prior), likelihood: every4(post.likelihood), posterior: every4(post.posterior) };
  priorOnly = coinPosterior(prior, 0, 0, 801);
  rebuildHistory();

  let top = 0;
  for (let i = 0; i < post.p.length; i++) {
    top = Math.max(top, post.posterior[i], post.prior[i], n ? post.likelihood[i] : 0);
  }
  curves.setY([0, Math.min(top * 1.08, 1e6)]);
  curves.redraw();
  historyPlot.setX([0, Math.max(10, n)]);
  historyPlot.redraw();
  setText('co-history-sub', replayed
    ? 'Loaded data are replayed in a random order. The shaded band is the 95% credible interval after each toss.'
    : 'The shaded band is the 95% credible interval after each toss.');

  setText('co-t-data', `${fmtInt(k)} / ${fmtInt(n)}`);
  setText('co-t-data-sub', n ? `${fmtNum((100 * k) / n, 1)}% heads` : 'no data yet');
  setText('co-t-ml', n ? fmtNum(k / n, 3) : '—');
  setText('co-t-mean', fmtNum(post.mean, 3));
  setText('co-t-sd', `posterior sd ${fmtSig(post.sd, 2)}`);
  setText('co-t-ci', `${fmtNum(post.quantile(0.025), 3)} – ${fmtNum(post.quantile(0.975), 3)}`);
  setText('co-t-p2', fmtNum(post.second, 3));
  setText('co-t-p2-prior', `before any data: ${fmtNum(priorOnly.second, 3)}`);
  setText('co-t-above', `${fmtNum(100 * post.probAboveHalf, 1)}%`);

  restartCoinChain();
}

function onPriorChange() {
  const kind = $('co-prior').value;
  prior = { kind, p0: p0.get(), M: strength.get() };
  for (const id of ['co-p0-field', 'co-M-field']) $(id).classList.toggle('is-disabled', kind !== 'bump');
  p0.input.disabled = strength.input.disabled = kind !== 'bump';
  summaryCache.clear();
  updateCoin();
}

function toss(count) {
  const b = trueBias.get();
  let k = coinK();
  for (let i = 0; i < count && heads.length < MAX_TOSSES; i++) {
    if (coinRng.uniform() < b) k++;
    heads.push(k);
  }
  updateCoin();
}

/** Load n tosses with k heads, replayed in a random order so the history chart has a path. */
function loadData(n, k) {
  const seq = new Uint8Array(n);
  seq.fill(1, 0, k);
  for (let i = n - 1; i > 0; i--) {
    const j = coinRng.int(i + 1);
    const t = seq[i]; seq[i] = seq[j]; seq[j] = t;
  }
  heads = [];
  let c = 0;
  for (let i = 0; i < n; i++) heads.push((c += seq[i]));
  replayed = true;
  summaryCache.clear();
}

$('co-toss1').addEventListener('click', () => toss(1));
$('co-toss10').addEventListener('click', () => toss(10));
$('co-toss100').addEventListener('click', () => toss(100));
$('co-reset').addEventListener('click', () => {
  heads = [];
  replayed = false;
  summaryCache.clear();
  updateCoin();
});
$('co-prior').addEventListener('change', onPriorChange);
$('co-preset').addEventListener('change', (e) => {
  const v = e.target.value;
  e.target.value = '';
  if (v === 'book9') loadData(10, 9);
  else if (v === 'book5') loadData(10, 5);
  else if (v === 'book515') {
    loadData(1000, 515);
    $('co-prior').value = 'bump';
    p0.set(0.9);
    strength.set(100);
  } else return;
  onPriorChange();
});

// =====================================================================
// Sampling the coin posterior with Metropolis
// =====================================================================

const CM_MAX = 200000;
const CM_BINS = 50;
const cmRng = new RNG();
const coinTarget = { logp: (p) => coinLogPrior(prior, p) + coinLogLikelihood(coinN(), coinK(), p) };
const cmChain = new Metropolis1D(coinTarget, cmRng);
const cmTrace = new Trace(CM_MAX + 1);
const cmPacer = new Pacer();
let cmDiag = null, cmDiag2 = null, cmHist = null, cmWindow = [0, 1];

const cmStep = bindRange('cm-step', { log: true, format: (v) => fmtSig(v, 2), onInput: restartCoinChain });
const cmStart = bindRange('cm-start', { format: (v) => fmtNum(v, 2), onInput: restartCoinChain });
const cmBurn = bindRange('cm-burn', {
  format: fmtInt,
  onInput() {
    updateCoinChainStats();
    cmTracePlot.redraw();
  },
});
const cmSpeed = bindRange('cm-speed', { log: true, format: fmtRate('steps') });

const cmTracePlot = new Plot('cm-trace', {
  height: 180,
  x: { domain: [0, 50], label: 'step' },
  y: { domain: [0, 1] },
  draw(p) {
    const c = p.c;
    const b = cmBurn.get();
    if (b > 0) p.vband(0, b, { color: c.ink, alpha: 0.06 });
    if (post) p.hline(post.mean, { color: c.exact, width: 1.5 });
    p.line(null, cmTrace.a, { color: c.s1, width: cmTrace.n > 2000 ? 1 : 1.5, n: cmTrace.n });
    if (cmTrace.n) p.marker(cmTrace.n - 1, cmTrace.last(), { color: c.s1, r: 4 });
  },
  hover(x) {
    if (!cmTrace.n) return null;
    const i = Math.max(0, Math.min(cmTrace.n - 1, Math.round(x)));
    return { x: i, title: i === 0 ? 'start' : `step ${fmtInt(i)}`, rows: [{ value: fmtNum(cmTrace.a[i], 4), label: 'p', color: 'var(--s1)' }] };
  },
});

const cmHistPlot = new Plot('cm-hist', {
  height: 230,
  x: { domain: [0, 1], label: 'p' },
  draw(p) {
    const c = p.c;
    if (cmHist) p.bars(cmHist.edges, cmHist.heights, { color: c.s1, alpha: 0.85 });
    if (post) p.line(post.p, post.posterior, { color: c.exact, width: 2 });
  },
  hover(x) {
    if (!post) return null;
    const [lo, hi] = cmWindow;
    const w = (hi - lo) / CM_BINS;
    const k = Math.floor((x - lo) / w);
    if (k < 0 || k >= CM_BINS) return null;
    const mid = lo + (k + 0.5) * w;
    const i = Math.round(mid * (post.p.length - 1));
    return {
      title: `p in [${fmtNum(lo + k * w, 3)}, ${fmtNum(lo + (k + 1) * w, 3)})`,
      rows: [
        { value: cmHist ? fmtSig(cmHist.heights[k], 3) : '—', label: 'samples', color: 'var(--s1)' },
        { value: fmtSig(post.posterior[i], 3), label: 'exact posterior', color: 'var(--exact)' },
      ],
    };
  },
});
legend('cm-hist-legend', [
  { label: 'kept samples', color: 'var(--s1)', kind: 'rect' },
  { label: 'exact posterior', color: 'var(--exact)', kind: 'line' },
]);

function restartCoinChain() {
  if (!post) return;
  cmRunner?.stop();
  cmRng.seed(randomSeed());
  let x0 = cmStart.get();
  if (!Number.isFinite(coinTarget.logp(x0))) x0 = post.mode; // e.g. the suspicious prior at exactly 1/2
  cmChain.reset(x0);
  cmTrace.clear();
  cmTrace.push(cmChain.x);
  cmPacer.reset();
  const lo = post.quantile(0.0005), hi = post.quantile(0.9995);
  const pad = 0.15 * (hi - lo) + 0.002;
  cmWindow = [Math.max(0, lo - pad), Math.min(1, hi + pad)];
  cmHistPlot.setX(cmWindow);
  $('cm-full').hidden = true;
  setText('cm-t-target', `${fmtInt(coinK())} heads in ${fmtInt(coinN())} tosses`);
  updateCoinChainStats();
  drawCoinChain();
}

function drawCoinChain() {
  cmTracePlot.setX([0, Math.max(50, cmTrace.n - 1)]);
  cmTracePlot.redraw();
  setText('cm-t-steps', fmtInt(cmChain.steps));
  setText('cm-t-acc', cmChain.steps ? `${fmtNum(100 * cmChain.acceptance, 1)}%` : '—');
}

function updateCoinChainStats() {
  const n = cmTrace.n;
  const s = Math.min(cmBurn.get(), n);
  cmDiag = chainDiagnostics(cmTrace.a, s, n);
  if (n - s >= 20) {
    const sq = new Float64Array(n - s);
    for (let i = s; i < n; i++) sq[i - s] = cmTrace.a[i] * cmTrace.a[i];
    cmDiag2 = chainDiagnostics(sq);
  } else {
    cmDiag2 = null;
  }
  cmHist = n - s > 0 ? histogram(cmTrace.a, cmWindow[0], cmWindow[1], CM_BINS, s, n) : null;

  let top = 0;
  for (let i = 0; i < post.p.length; i++) if (post.p[i] >= cmWindow[0] && post.p[i] <= cmWindow[1]) top = Math.max(top, post.posterior[i]);
  if (cmHist) for (const h of cmHist.heights) top = Math.max(top, h);
  cmHistPlot.setY([0, top * 1.1 || 1]);
  cmHistPlot.redraw();

  setText('cm-t-mean-exact', `exact ${fmtNum(post.mean, 4)}`);
  setText('cm-t-p2-exact', `exact ${fmtNum(post.second, 4)}`);
  if (cmDiag) {
    const approx = cmDiag.converged ? '' : '≳ ';
    setText('cm-t-mean', `${fmtNum(cmDiag.mean, 4)} ± ${approx}${fmtSig(cmDiag.error, 2)}`);
    setText('cm-t-tau', approx + fmtSig(cmDiag.tau, 3));
    setText('cm-t-ess', `${cmDiag.converged ? '' : '≲ '}${fmtInt(cmDiag.ess)} effective samples`);
    const z = Math.abs(cmDiag.mean - post.mean) / cmDiag.error;
    setText('cm-t-z', cmDiag.converged ? fmtNum(z, 1) : '—');
  } else {
    for (const id of ['cm-t-mean', 'cm-t-tau', 'cm-t-z']) setText(id, '—');
    setText('cm-t-ess', n <= s ? 'still inside the burn-in' : '');
  }
  setText('cm-t-p2', cmDiag2 ? `${fmtNum(cmDiag2.mean, 4)} ± ${cmDiag2.converged ? '' : '≳ '}${fmtSig(cmDiag2.error, 2)}` : '—');
}
const coinChainStatsSoon = throttle(updateCoinChainStats, 300);

function advanceCoinChain(k) {
  const c = cmStep.get();
  for (let i = 0; i < k; i++) {
    cmChain.step(c);
    cmTrace.push(cmChain.x);
    if (cmTrace.full) {
      $('cm-full').hidden = false;
      return false;
    }
  }
  return true;
}

const cmRunner = new Runner({
  button: 'cm-run',
  root: 'cm-lab',
  tick(dt) {
    const k = cmPacer.take(cmSpeed.get(), dt, 20000);
    if (k === 0) return true;
    const ok = advanceCoinChain(k);
    drawCoinChain();
    coinChainStatsSoon();
    return ok;
  },
  onChange(running) {
    if (!running) {
      drawCoinChain();
      updateCoinChainStats();
    }
  },
});
$('cm-once').addEventListener('click', () => {
  cmRunner.stop();
  advanceCoinChain(1);
  drawCoinChain();
  updateCoinChainStats();
});
$('cm-reset').addEventListener('click', restartCoinChain);

onPriorChange(); // computes the initial posterior and starts the chain

// =====================================================================
// Mean and width of a Gaussian
// =====================================================================

const GS_MAX = 200000;
const GS_DRAWN = 20000; // points drawn in the plane
const dataRng = new RNG();
let dataZ = new Float64Array(1000); // standard normals; data are mu + sigma * z
const gsRng = new RNG();
let gs = null; // { xs, stats, exact, action, muDom, sigDom, start }
const gsTarget = { U: () => 0 };
const gsChain = new Metropolis2D(gsTarget, gsRng);
const gsMu = new Trace(GS_MAX + 1), gsSig = new Trace(GS_MAX + 1);
const gsPacer = new Pacer();
let gsDiag = null;

const trueMu = bindRange('gs-mu', { format: (v) => fmtNum(v, 1), onInput: updateGaussianData });
const trueSigma = bindRange('gs-sigma', { format: (v) => fmtNum(v, 2), onInput: updateGaussianData });
const gsN = bindRange('gs-n', { log: true, format: (v) => fmtInt(Math.round(v)), onInput: updateGaussianData });
const gsStep = bindRange('gs-step', { log: true, format: (v) => fmtSig(v, 2), onInput: restartGaussianChain });
const gsSpeed = bindRange('gs-speed', { log: true, format: fmtRate('steps') });

/** Histogram bins for the data chart: about half a σ wide for small n, finer as n grows. */
const dataBins = (n) => Math.max(12, Math.min(60, Math.round(6 * Math.cbrt(n))));

function drawData() {
  dataRng.seed(randomSeed());
  for (let i = 0; i < dataZ.length; i++) dataZ[i] = dataRng.normal();
}

const dataPlot = new Plot('gs-data', {
  height: (w) => Math.min(360, Math.max(220, w * 0.75)),
  x: { domain: [-5, 5], label: 'x' },
  draw(p) {
    if (!gs) return;
    const c = p.c;
    const [lo, hi] = p.x.domain;
    const h = histogram(gs.xs, lo, hi, dataBins(gs.stats.n));
    p.bars(h.edges, h.heights, { color: c.s1, alpha: 0.8 });
    p.fn((x) => normPdf(x, trueMu.get(), trueSigma.get()), { color: c.exact, width: 2 });
    p.fn((x) => normPdf(x, gs.stats.mean, gs.stats.s), { color: c.s2, width: 1.5 });
  },
  hover(x) {
    if (!gs) return null;
    return {
      x,
      title: `x = ${fmtNum(x, 2)}`,
      rows: [
        { value: fmtNum(normPdf(x, trueMu.get(), trueSigma.get()), 3), label: 'true density', color: 'var(--exact)' },
        { value: fmtNum(normPdf(x, gs.stats.mean, gs.stats.s), 3), label: 'max-likelihood fit', color: 'var(--s2)' },
      ],
    };
  },
});
legend('gs-data-legend', [
  { label: 'data', color: 'var(--s1)', kind: 'rect' },
  { label: 'true density', color: 'var(--exact)', kind: 'line' },
  { label: 'max-likelihood fit N(x̄, s²)', color: 'var(--s2)', kind: 'line' },
]);

const heatmaps = new Map();
function gaussianHeatmap() {
  const c = palette();
  const key = `${gs.stats.n}|${gs.stats.mean}|${gs.stats.s}|${c.surface}|${c.seqHi}`;
  let img = heatmaps.get(key);
  if (!img) {
    const Smin = gs.action(gs.stats.mean, gs.stats.s);
    img = densityImage((mu, sigma) => Math.exp(Smin - gs.action(mu, sigma)), gs.muDom, gs.sigDom, 160, c.surface, c.seqHi, 0.55);
    heatmaps.clear();
    heatmaps.set(key, img);
  }
  return img;
}

const plane = new Plot('gs-plane', {
  square: true,
  maxHeight: 380,
  grid: 'none',
  x: { domain: [0, 1], ticks: 5, label: 'μ' },
  y: { domain: [0, 1], ticks: 5, label: 'σ' },
  margin: { l: 52 },
  draw(p) {
    if (!gs) return;
    const c = p.c;
    p.image(gaussianHeatmap(), gs.muDom[0], gs.muDom[1], gs.sigDom[0], gs.sigDom[1]);
    const start = Math.max(0, gsMu.n - GS_DRAWN);
    p.points(gsMu.a, gsSig.a, { color: c.s1, r: gsMu.n > 2000 ? 1 : 1.8, alpha: 0.55, n: gsMu.n, start });
    p.marker(gs.stats.mean, gs.stats.s, { color: c.s2, r: 5 });
    p.marker(trueMu.get(), trueSigma.get(), { color: c.exact, r: 7, hollow: true });
  },
  hover(mu, sigma) {
    if (!gs) return null;
    return { title: `μ = ${fmtNum(mu, 3)}, σ = ${fmtNum(sigma, 3)}`, rows: [{ value: fmtSig(gs.action(mu, sigma) - gs.action(gs.stats.mean, gs.stats.s), 3), label: 'S − S_min' }] };
  },
});
legend('gs-plane-legend', [
  { label: 'samples', color: 'var(--s1)', kind: 'dot' },
  { label: 'max likelihood (x̄, s)', color: 'var(--s2)', kind: 'dot' },
  { label: 'true (μ, σ)', color: 'var(--exact)', kind: 'dot' },
]);

function marginalPlot(id, { trace, pdf, dom, label }) {
  const BINS = 40;
  let hist = null;
  const plot = new Plot(id, {
    height: 200,
    x: { domain: [0, 1], label },
    draw(p) {
      if (!gs) return;
      const c = p.c;
      if (hist) p.bars(hist.edges, hist.heights, { color: c.s1, alpha: 0.85 });
      p.fn(pdf(), { color: c.exact, width: 2 });
    },
    hover(x) {
      if (!gs) return null;
      return { x, title: `${label} = ${fmtNum(x, 3)}`, rows: [{ value: fmtSig(pdf()(x), 3), label: 'exact density', color: 'var(--exact)' }] };
    },
  });
  return {
    plot,
    update(burn) {
      const [lo, hi] = dom();
      plot.setX([lo, hi]);
      hist = trace.n - burn > 0 ? histogram(trace.a, lo, hi, BINS, burn, trace.n) : null;
      let top = 0;
      const f = pdf();
      for (let i = 0; i <= 200; i++) top = Math.max(top, f(lo + ((hi - lo) * i) / 200));
      if (hist) for (const h of hist.heights) top = Math.max(top, h);
      plot.setY([0, top * 1.1 || 1]);
      plot.redraw();
    },
  };
}

const muMarginal = marginalPlot('gs-mu-hist', { trace: gsMu, pdf: () => (m) => gs.exact.muPdf(m), dom: () => gs.muDom, label: 'μ' });
const sigmaMarginal = marginalPlot('gs-sigma-hist', { trace: gsSig, pdf: () => (s) => gs.exact.sigmaPdf(s), dom: () => gs.sigDom, label: 'σ' });

function gsBurn() {
  return Math.min(200, Math.floor(gsMu.n / 5));
}

function updateGaussianData() {
  const n = Math.round(gsN.get());
  const mu = trueMu.get(), sigma = trueSigma.get();
  const xs = new Float64Array(n);
  for (let i = 0; i < n; i++) xs[i] = mu + sigma * dataZ[i];
  const stats = gaussianStats(xs);
  const exact = gaussianPosteriorExact(stats);
  const muDom = [stats.mean - 5 * exact.muSd, stats.mean + 5 * exact.muSd];
  const sigDom = [Math.max(0.01 * stats.s, exact.sigmaMean - 4 * exact.sigmaSd), exact.sigmaMean + 5 * exact.sigmaSd];
  gs = {
    xs, stats, exact, muDom, sigDom,
    action: gaussianAction(stats),
    start: [stats.mean + 3 * exact.muSd, exact.sigmaMean + 3 * exact.sigmaSd],
  };
  gsTarget.U = gs.action;

  dataPlot.setX([mu - 4.5 * sigma, mu + 4.5 * sigma]);
  let top = normPdf(0, 0, sigma);
  top = Math.max(top, normPdf(0, 0, stats.s));
  const h = histogram(xs, mu - 4.5 * sigma, mu + 4.5 * sigma, dataBins(n));
  for (const v of h.heights) top = Math.max(top, v);
  dataPlot.setY([0, top * 1.1]);
  dataPlot.redraw();
  setText('gs-data-sub', `${fmtInt(n)} points drawn from N(${fmtNum(mu, 1)}, ${fmtNum(sigma, 2)}²)`);

  plane.setX(muDom).setY(sigDom);
  setText('gs-t-ml', `${fmtNum(stats.mean, 3)}, ${fmtNum(stats.s, 3)}`);
  setText('gs-t-true', `true ${fmtNum(mu, 2)}, ${fmtNum(sigma, 2)}`);
  setText('gs-t-mu-exact', `exact ${fmtNum(exact.muMean, 3)}`);
  setText('gs-t-musd-exact', `exact ${fmtNum(exact.muSd, 3)} = s/√(n − 4)`);
  setText('gs-t-sig-exact', `exact ${fmtNum(exact.sigmaMean, 3)}`);
  setText('gs-t-sigsd-exact', `exact ${fmtNum(exact.sigmaSd, 3)}`);
  restartGaussianChain();
}

function restartGaussianChain() {
  if (!gs) return;
  gsRunner?.stop();
  gsRng.seed(randomSeed());
  gsChain.reset(...gs.start);
  gsMu.clear();
  gsSig.clear();
  gsMu.push(gsChain.x);
  gsSig.push(gsChain.y);
  gsPacer.reset();
  updateGaussianStats();
  drawGaussian();
}

function drawGaussian() {
  plane.redraw();
  setText('gs-t-acc', gsChain.steps ? `${fmtNum(100 * gsChain.acceptance, 1)}%` : '—');
  setText('gs-t-steps', `${fmtInt(gsChain.steps)} steps`);
}

function updateGaussianStats() {
  const burn = gsBurn();
  const n = gsMu.n;
  gsDiag = n - burn >= 50 ? { mu: chainDiagnostics(gsMu.a, burn, n), sigma: chainDiagnostics(gsSig.a, burn, n) } : null;
  const show = (d) => `${fmtNum(d.mean, 3)} ± ${d.converged ? '' : '≳ '}${fmtSig(d.error, 1)}`;
  setText('gs-t-mu', gsDiag ? show(gsDiag.mu) : '—');
  setText('gs-t-sig', gsDiag ? show(gsDiag.sigma) : '—');
  setText('gs-t-musd', gsDiag ? fmtNum(Math.sqrt(gsDiag.mu.variance), 3) : '—');
  setText('gs-t-sigsd', gsDiag ? fmtNum(Math.sqrt(gsDiag.sigma.variance), 3) : '—');
  muMarginal.update(burn);
  sigmaMarginal.update(burn);
}
const gaussianStatsSoon = throttle(updateGaussianStats, 300);

const gsRunner = new Runner({
  button: 'gs-run',
  root: 'gs-lab',
  tick(dt) {
    const k = gsPacer.take(gsSpeed.get(), dt, 20000);
    if (k === 0) return true;
    const c = gsStep.get() * (gs.stats.s / Math.sqrt(gs.stats.n));
    for (let i = 0; i < k; i++) {
      gsChain.step(c);
      gsMu.push(gsChain.x);
      gsSig.push(gsChain.y);
      if (gsMu.full) break;
    }
    drawGaussian();
    gaussianStatsSoon();
    return !gsMu.full;
  },
  onChange(running) {
    if (!running) {
      drawGaussian();
      updateGaussianStats();
    }
  },
});
$('gs-reset').addEventListener('click', restartGaussianChain);
$('gs-new').addEventListener('click', () => {
  drawData();
  updateGaussianData();
});

drawData();
updateGaussianData();
