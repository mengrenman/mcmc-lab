import { mountChrome, $, setText, bindRange, Runner, Pacer } from '../lib/ui.js';
import { Plot, legend, nearestIndex, fmtNum, fmtSig, fmtInt } from '../lib/plot.js';
import { RNG, randomSeed } from '../lib/random.js';
import { integrand, importanceWeightVariance, uniformWeightVariance, SQRT2PI } from '../lib/targets.js';

mountChrome();

const fmtRate = (unit) => (v) => (v < 10 ? fmtNum(v, 1) : fmtInt(v)) + ` ${unit}/s`;
const logDomainFloor = (v) => 10 ** Math.floor(Math.log10(v));

// =====================================================================
// Estimating π
// =====================================================================

const PI_DRAW = 20000;
const PI_SIGMA = Math.sqrt(Math.PI * (4 - Math.PI)); // sd of one dart's contribution 4·1[inside]

const piRng = new RNG();
const dx = new Float32Array(PI_DRAW), dy = new Float32Array(PI_DRAW);
const dIn = new Uint8Array(PI_DRAW);
let piN = 0, piHits = 0, piNext = 1;
let ckN = [], ckErr = [];
const piPacer = new Pacer();
const piSpeed = bindRange('pi-speed', { log: true, format: fmtRate('darts') });

const board = new Plot('pi-board', {
  square: true,
  maxHeight: 380,
  grid: 'none',
  x: { domain: [0, 1], ticks: 4 },
  y: { domain: [0, 1], ticks: 4 },
  margin: { l: 36 },
  draw(p) {
    const c = p.c;
    const shown = Math.min(piN, PI_DRAW);
    const small = shown < 400;
    const ctx = p.ctx;
    for (const [flag, color] of [[1, c.s1], [0, c.s2]]) {
      ctx.fillStyle = color;
      ctx.globalAlpha = small ? 1 : 0.75;
      for (let i = 0; i < shown; i++) {
        if (dIn[i] !== flag) continue;
        const x = p.sx(dx[i]), y = p.sy(dy[i]);
        if (small) {
          ctx.beginPath();
          ctx.arc(x, y, 3, 0, 2 * Math.PI);
          ctx.fill();
        } else {
          ctx.fillRect(x - 1, y - 1, 2, 2);
        }
      }
    }
    ctx.globalAlpha = 1;
    p.fn((x) => Math.sqrt(Math.max(0, 1 - x * x)), { color: c.exact, width: 2 });
  },
});
legend('pi-board-legend', [
  { label: 'inside', color: 'var(--s1)', kind: 'dot' },
  { label: 'outside', color: 'var(--s2)', kind: 'dot' },
]);

const errPlot = new Plot('pi-error', {
  height: (w) => Math.min(380, Math.max(240, w * 0.8)),
  x: { type: 'log', domain: [1, 1000], label: 'darts N' },
  y: { type: 'log', domain: [1e-4, 4] },
  draw(p) {
    const c = p.c;
    p.fn((n) => PI_SIGMA / Math.sqrt(n), { color: c.exact, width: 2 });
    p.line(ckN, ckErr, { color: c.s1, width: 1.5 });
    if (ckN.length) p.marker(ckN[ckN.length - 1], ckErr[ckErr.length - 1], { color: c.s1, r: 4 });
  },
  hover(x) {
    if (!ckN.length) return null;
    const i = nearestIndex(ckN, x);
    return {
      x: ckN[i],
      title: `N = ${fmtInt(ckN[i])}`,
      rows: [
        { value: fmtSig(ckErr[i], 3), label: '|estimate − π|', color: 'var(--s1)' },
        { value: fmtSig(PI_SIGMA / Math.sqrt(ckN[i]), 3), label: 'expected 1.64/√N', color: 'var(--exact)' },
      ],
    };
  },
});
legend('pi-error-legend', [
  { label: 'this run: \\(|\\hat{\\pi} - \\pi|\\)', color: 'var(--s1)', kind: 'line' },
  { label: 'expected error \\(1.64/\\sqrt{N}\\)', color: 'var(--exact)', kind: 'line' },
]);

function throwDarts(k) {
  for (let i = 0; i < k; i++) {
    const x = piRng.uniform(), y = piRng.uniform();
    const inside = x * x + y * y < 1 ? 1 : 0;
    if (piN < PI_DRAW) {
      dx[piN] = x;
      dy[piN] = y;
      dIn[piN] = inside;
    }
    piN++;
    piHits += inside;
    if (piN >= piNext) {
      ckN.push(piN);
      ckErr.push(Math.abs((4 * piHits) / piN - Math.PI));
      piNext = Math.max(piN + 1, Math.ceil(piN * 1.03));
    }
  }
}

let boardShown = -1;
function drawPi() {
  // The board only changes while darts are still being recorded.
  if (Math.min(piN, PI_DRAW) !== boardShown) {
    boardShown = Math.min(piN, PI_DRAW);
    board.redraw();
  }
  let minErr = 1;
  for (const e of ckErr) if (e > 0 && e < minErr) minErr = e;
  errPlot.setX([1, Math.max(1000, piN * 1.2)]).setY([Math.min(1e-3, logDomainFloor(minErr * 0.8)), 4]);
  errPlot.redraw();
  setText('pi-n', fmtInt(piN));
  setText('pi-hits', fmtInt(piHits));
  setText('pi-est', piN ? fmtNum((4 * piHits) / piN, 5) : '—');
  setText('pi-err', piN ? fmtSig(Math.abs((4 * piHits) / piN - Math.PI), 2) : '—');
  setText('pi-sigma', piN ? `expected ≈ ${fmtSig(PI_SIGMA / Math.sqrt(piN), 2)}` : 'expected ≈ 1.64/√N');
}

function resetPi() {
  piRunner?.stop();
  piRng.seed(randomSeed());
  piN = piHits = 0;
  piNext = 1;
  ckN = [];
  ckErr = [];
  piPacer.reset();
  drawPi();
}

const piRunner = new Runner({
  button: 'pi-run',
  root: 'pi-lab',
  tick(dt) {
    const k = piPacer.take(piSpeed.get(), dt, 200000);
    if (k > 0) {
      throwDarts(k);
      drawPi();
    }
  },
});
$('pi-reset').addEventListener('click', resetPi);
resetPi();

// =====================================================================
// Importance sampling
// =====================================================================

const BOX = integrand.box;
const RUG = 150;
const uRng = new RNG(), iRng = new RNG();
let isN = 0, sumU = 0, sumU2 = 0, sumI = 0, sumI2 = 0, isNext = 1;
let isCk = { n: [], u: [], i: [] };
const rugU = new Float64Array(RUG), rugI = new Float64Array(RUG);
const isPacer = new Pacer();

const width = bindRange('is-width', { log: true, format: (v) => fmtNum(v, 2), onInput: resetIS });
const isSpeed = bindRange('is-speed', { log: true, format: fmtRate('samples') });

const gaussQ = (x, s) => Math.exp((-0.5 * x * x) / (s * s)) / (s * SQRT2PI);

const funcs = new Plot('is-funcs', {
  height: 230,
  x: { domain: [-BOX, BOX], label: 'x' },
  y: { domain: [0, 1.1] },
  draw(p) {
    const c = p.c;
    const s = width.get();
    const I = integrand.exact;
    p.area(integrand.f, { color: c.exact, alpha: 0.08 });
    p.fn(() => I / (2 * BOX), { color: c.s1 });
    p.fn((x) => I * gaussQ(x, s), { color: c.s2 });
    p.fn(integrand.f, { color: c.exact });
    // Rug of recent samples: uniform on the lower row, Gaussian proposal above it.
    const ctx = p.ctx;
    const shown = Math.min(isN, RUG);
    const base = p.m.t + p.ih;
    ctx.lineWidth = 1.5;
    for (const [arr, color, y0] of [[rugU, c.s1, base - 2], [rugI, c.s2, base - 14]]) {
      ctx.strokeStyle = color;
      ctx.globalAlpha = 0.7;
      ctx.beginPath();
      for (let k = 0; k < shown; k++) {
        const x = Math.round(p.sx(arr[k])) + 0.5;
        ctx.moveTo(x, y0);
        ctx.lineTo(x, y0 - 9);
      }
      ctx.stroke();
    }
    ctx.globalAlpha = 1;
  },
  hover(x) {
    const s = width.get();
    return {
      x,
      title: `x = ${fmtNum(x, 2)}`,
      rows: [
        { value: fmtNum(integrand.f(x), 3), label: 'f(x)', color: 'var(--exact)' },
        { value: fmtNum(integrand.exact * gaussQ(x, s), 3), label: 'I·q(x), Gaussian', color: 'var(--s2)' },
        { value: fmtNum(integrand.exact / (2 * BOX), 3), label: 'I·q(x), uniform', color: 'var(--s1)' },
      ],
    };
  },
});
legend('is-funcs-legend', [
  { label: 'integrand \\(f(x)\\)', color: 'var(--exact)', kind: 'line' },
  { label: 'uniform proposal', color: 'var(--s1)', kind: 'line' },
  { label: 'Gaussian proposal \\(\\mathcal{N}(0, s^2)\\)', color: 'var(--s2)', kind: 'line' },
]);

const running = new Plot('is-running', {
  height: 230,
  x: { type: 'log', domain: [1, 1000], label: 'samples N' },
  y: { domain: [0, 5] },
  draw(p) {
    const c = p.c;
    p.hline(integrand.exact, { color: c.exact, width: 2 });
    p.line(isCk.n, isCk.u, { color: c.s1, width: 1.5 });
    p.line(isCk.n, isCk.i, { color: c.s2, width: 1.5 });
  },
  hover(x) {
    if (!isCk.n.length) return null;
    const i = nearestIndex(isCk.n, x);
    return {
      x: isCk.n[i],
      title: `N = ${fmtInt(isCk.n[i])}`,
      rows: [
        { value: fmtNum(isCk.u[i], 4), label: 'uniform', color: 'var(--s1)' },
        { value: fmtNum(isCk.i[i], 4), label: 'importance', color: 'var(--s2)' },
        { value: fmtNum(integrand.exact, 4), label: 'exact √(2π)', color: 'var(--exact)' },
      ],
    };
  },
});
legend('is-running-legend', [
  { label: 'uniform sampling', color: 'var(--s1)', kind: 'line' },
  { label: 'importance sampling', color: 'var(--s2)', kind: 'line' },
  { label: 'exact \\(\\sqrt{2\\pi}\\)', color: 'var(--exact)', kind: 'line' },
]);

function sampleIS(k) {
  const s = width.get();
  const c = SQRT2PI * s;
  const e = 1 / (2 * s * s) - 0.5;
  for (let j = 0; j < k; j++) {
    const xu = BOX * (2 * uRng.uniform() - 1);
    const wu = 2 * BOX * integrand.f(xu);
    const xi = s * iRng.normal();
    // f(x)/q(x) = x^2 · s·sqrt(2π) · exp(x^2 (1/(2s^2) − 1/2)), written to avoid 0/0 in the tails
    const wi = xi * xi * c * Math.exp(xi * xi * e);
    rugU[isN % RUG] = xu;
    rugI[isN % RUG] = xi;
    isN++;
    sumU += wu; sumU2 += wu * wu;
    sumI += wi; sumI2 += wi * wi;
    if (isN >= isNext) {
      isCk.n.push(isN);
      isCk.u.push(sumU / isN);
      isCk.i.push(sumI / isN);
      isNext = Math.max(isN + 1, Math.ceil(isN * 1.02));
    }
  }
}

function stdErr(sum, sum2, n) {
  if (n < 2) return NaN;
  const v = (sum2 - (sum * sum) / n) / (n - 1);
  return Math.sqrt(Math.max(0, v) / n);
}

function drawIS() {
  const zoom = $('is-zoom').checked;
  funcs.setX(zoom ? [-5, 5] : [-BOX, BOX]);
  funcs.redraw();
  running.setX([1, Math.max(1000, isN * 1.2)]);
  running.redraw();
  setText('is-n', fmtInt(isN));
  setText('is-u', isN ? `${fmtNum(sumU / isN, 3)} ± ${fmtSig(stdErr(sumU, sumU2, isN), 2)}` : '—');
  setText('is-i', isN ? `${fmtNum(sumI / isN, 3)} ± ${fmtSig(stdErr(sumI, sumI2, isN), 2)}` : '—');
}

function describeTheory() {
  const vu = uniformWeightVariance(), vi = importanceWeightVariance(width.get());
  setText('is-u-sd', `σ of one weight: ${fmtNum(Math.sqrt(vu), 2)}`);
  setText('is-i-sd', Number.isFinite(vi) ? `σ of one weight: ${fmtNum(Math.sqrt(vi), 2)}` : 'σ of one weight: ∞ (s ≤ 0.71)');
  if (Number.isFinite(vi)) {
    setText('is-gain', `${fmtSig(vu / vi, 2)}×`);
    setText('is-gain-sub', 'fewer samples for the same error, in theory');
  } else {
    setText('is-gain', 'none');
    setText('is-gain-sub', 'infinite variance: no reliable error bar');
  }
}

function resetIS() {
  isRunner?.stop();
  uRng.seed(randomSeed());
  iRng.seed(randomSeed());
  isN = sumU = sumU2 = sumI = sumI2 = 0;
  isNext = 1;
  isCk = { n: [], u: [], i: [] };
  isPacer.reset();
  describeTheory();
  drawIS();
}

const isRunner = new Runner({
  button: 'is-run',
  root: 'is-lab',
  tick(dt) {
    const k = isPacer.take(isSpeed.get(), dt, 100000);
    if (k > 0) {
      sampleIS(k);
      drawIS();
    }
  },
});
$('is-reset').addEventListener('click', resetIS);
$('is-zoom').addEventListener('change', drawIS);
resetIS();
