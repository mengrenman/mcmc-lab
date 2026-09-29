import { mountChrome, $, setText, bindRange, Runner, Pacer, throttle } from '../lib/ui.js';
import { Plot, legend, densityImage, palette, fmtNum, fmtSig, fmtInt } from '../lib/plot.js';
import { RNG, randomSeed } from '../lib/random.js';
import { Trace, chainDiagnostics } from '../lib/stats.js';
import { correlatedGaussian, banana } from '../lib/targets.js';
import { HMC2D } from '../samplers/hmc.js';
import { Metropolis2D } from '../samplers/metropolis.js';
import { Gibbs2D } from '../samplers/gibbs.js';

mountChrome();

const makeTarget = (kind, rho) => (kind === 'banana' ? banana() : correlatedGaussian(rho));

/** Density heatmaps, cached per target and theme (both labs share the cache). */
const heatmaps = new Map();
function heatmapFor(target) {
  const c = palette();
  const key = `${target.name}|${c.surface}|${c.seqHi}`;
  let img = heatmaps.get(key);
  if (!img) {
    const [dx, dy] = target.domain;
    img = densityImage((x, y) => Math.exp(-target.U(x, y)), dx, dy, 160, c.surface, c.seqHi, 0.55);
    if (heatmaps.size > 8) heatmaps.delete(heatmaps.keys().next().value);
    heatmaps.set(key, img);
  }
  return img;
}

/** Small ring buffer of the most recent path points. */
class RecentPath {
  constructor(size = 60) {
    this.x = new Float64Array(size);
    this.y = new Float64Array(size);
    this.size = size;
    this.clear();
  }
  clear() { this.n = 0; this.head = 0; }
  push(x, y) {
    this.x[this.head] = x;
    this.y[this.head] = y;
    this.head = (this.head + 1) % this.size;
    this.n = Math.min(this.n + 1, this.size);
  }
  /** Points in chronological order. */
  ordered() {
    const xs = new Float64Array(this.n), ys = new Float64Array(this.n);
    const start = (this.head - this.n + this.size) % this.size;
    for (let i = 0; i < this.n; i++) {
      xs[i] = this.x[(start + i) % this.size];
      ys[i] = this.y[(start + i) % this.size];
    }
    return [xs, ys];
  }
}

// =====================================================================
// HMC explorer
// =====================================================================

const rngA = new RNG();
let targetA = makeTarget('gaussian', 0.9);
const hmc = new HMC2D(targetA, rngA);
const sampX = new Trace(50000), sampY = new Trace(50000);
const pacerA = new Pacer();
let cur = { x: 0, y: 0 }; // displayed state (lags the sampler while a trajectory animates)
let anim = null; // { t, dur } while a trajectory is being revealed
let animLoop = 0;

const fmtRate = (unit) => (v) => (v < 10 ? fmtNum(v, 1) : fmtInt(v)) + ` ${unit}/s`;
const rhoA = bindRange('hx-rho', { format: (v) => fmtNum(v, 2), onInput: resetA });
const epsA = bindRange('hx-eps', { log: true, format: (v) => fmtSig(v, 2) });
const LA = bindRange('hx-L', { format: String });
const speedA = bindRange('hx-speed', { log: true, format: fmtRate('steps') });

function revealed() {
  const n = hmc.path.n;
  if (!anim) return n;
  return Math.max(1, Math.min(n, 1 + Math.floor((anim.t / anim.dur) * (n - 1) * 1.0001)));
}

const plane = new Plot('hx-plane', {
  square: true,
  maxHeight: 420,
  grid: 'none',
  x: { domain: targetA.domain[0], ticks: 5 },
  y: { domain: targetA.domain[1], ticks: 5 },
  margin: { l: 36 },
  draw(p) {
    const c = p.c;
    const [dx, dy] = targetA.domain;
    p.image(heatmapFor(targetA), dx[0], dx[1], dy[0], dy[1]);
    p.points(sampX.a, sampY.a, { color: c.s1, r: sampX.n > 3000 ? 1.2 : 2.5, alpha: 0.6, n: sampX.n });
    const path = hmc.path;
    if (path.n > 0 && hmc.last) {
      const k = revealed();
      p.line(path.x, path.y, { color: c.s2, width: 2, n: k });
      p.points(path.x, path.y, { color: c.s2, r: 2.5, n: k });
      if (!anim) {
        const end = path.n - 1;
        p.marker(path.x[end], path.y[end], { color: hmc.last.accepted ? c.good : c.bad, r: 7, hollow: true });
      }
    }
    p.marker(cur.x, cur.y, { color: c.s1, r: 5 });
  },
  hover(x, y) {
    return { title: `(${fmtNum(x, 2)}, ${fmtNum(y, 2)})`, rows: [{ value: fmtNum(targetA.U(x, y), 2), label: 'U = −log p' }] };
  },
});
legend('hx-plane-legend', [
  { label: 'samples', color: 'var(--s1)', kind: 'dot' },
  { label: 'trajectory', color: 'var(--s2)', kind: 'line' },
  { label: 'accepted end', color: 'var(--good)', kind: 'dot' },
  { label: 'rejected end', color: 'var(--bad)', kind: 'dot' },
]);

const energy = new Plot('hx-energy', {
  height: (w) => Math.min(420, Math.max(240, w * 0.85)),
  x: { domain: [0, 20], label: 'leapfrog step' },
  y: { domain: [0, 5] },
  draw(p) {
    const c = p.c;
    const path = hmc.path;
    if (!hmc.last) return;
    const k = revealed();
    p.line(null, path.K, { color: c.s3, n: k });
    p.line(null, path.U, { color: c.s2, n: k });
    p.line(null, path.H, { color: c.s1, n: k, width: 2.5 });
    p.hline(path.H[0], { color: c.ink2, alpha: 0.5 });
  },
  hover(x) {
    if (!hmc.last) return null;
    const k = revealed();
    const i = Math.max(0, Math.min(k - 1, Math.round(x)));
    const path = hmc.path;
    return {
      x: i,
      title: `after ${i} leapfrog step${i === 1 ? '' : 's'}`,
      rows: [
        { value: fmtSig(path.H[i], 4), label: 'total H', color: 'var(--s1)' },
        { value: fmtSig(path.U[i], 3), label: 'potential U', color: 'var(--s2)' },
        { value: fmtSig(path.K[i], 3), label: 'kinetic K', color: 'var(--s3)' },
      ],
    };
  },
});
legend('hx-energy-legend', [
  { label: 'total \\(H = U + K\\)', color: 'var(--s1)', kind: 'line' },
  { label: 'potential \\(U\\)', color: 'var(--s2)', kind: 'line' },
  { label: 'kinetic \\(K\\)', color: 'var(--s3)', kind: 'line' },
]);

function fitEnergyAxes() {
  const path = hmc.path;
  let lo = Infinity, hi = -Infinity;
  for (const arr of [path.U, path.K, path.H]) {
    for (let i = 0; i < path.n; i++) {
      const v = arr[i];
      if (Number.isFinite(v) && Math.abs(v) < 1e8) {
        lo = Math.min(lo, v);
        hi = Math.max(hi, v);
      }
    }
  }
  if (!Number.isFinite(lo)) { lo = 0; hi = 1; }
  const pad = (hi - lo) * 0.08 || 0.5;
  energy.setX([0, Math.max(1, path.n - 1)]).setY([Math.min(0, lo - pad), hi + pad]);
}

function describeA() {
  const el = $('hx-decision');
  const L = hmc.last;
  if (!L) {
    el.textContent = 'Press One step to launch a trajectory from the current point, or Run to keep going.';
    return;
  }
  if (anim) {
    el.textContent = `Integrating the trajectory: leapfrog step ${revealed() - 1} of ${hmc.path.n - 1}…`;
    return;
  }
  const verdict = L.accepted
    ? '<span class="badge good">✓ Accepted</span> The chain jumps to the end of the trajectory.'
    : '<span class="badge bad">✗ Rejected</span> The chain stays where it was and records that point again.';
  const dh = Number.isFinite(L.dH) ? fmtSig(L.dH, 3) : '∞ (the trajectory diverged)';
  const detail = !Number.isFinite(L.dH) ? '' : L.dH <= 0 ? ' ΔH ≤ 0, so the move is always accepted.' : ` Accept with probability e<sup>−ΔH</sup> = ${fmtNum(L.prob, 3)}. Drew u = ${fmtNum(L.u, 3)}.`;
  el.innerHTML = `ΔH = ${dh}.${detail} ${verdict}`;
}

function drawA() {
  plane.redraw();
  energy.redraw();
  describeA();
  setText('hx-steps', fmtInt(hmc.steps));
  setText('hx-acc', hmc.steps ? `${fmtNum(100 * hmc.acceptance, 1)}%` : '—');
  setText('hx-dh', hmc.last ? (Number.isFinite(hmc.last.dH) ? fmtSig(hmc.last.dH, 3) : '∞') : '—');
  setText('hx-grads', fmtInt(hmc.grads));
}

function commitA() {
  anim = null;
  cur = { x: hmc.x, y: hmc.y };
  sampX.push(hmc.x);
  sampY.push(hmc.y);
}

function beginStep(animate, dur) {
  hmc.step(epsA.get(), LA.get());
  fitEnergyAxes();
  if (animate) anim = { t: 0, dur };
  else commitA();
}

function advanceAnim(dt) {
  if (!anim) return;
  anim.t += dt;
  if (anim.t >= anim.dur) commitA();
}

function playOnce() {
  if (animLoop) return;
  let last = performance.now();
  const frame = (now) => {
    advanceAnim(Math.min(0.1, (now - last) / 1000));
    last = now;
    drawA();
    animLoop = anim ? requestAnimationFrame(frame) : 0;
  };
  animLoop = requestAnimationFrame(frame);
}

function resetA() {
  runnerA?.stop();
  if (anim) commitA();
  const kind = $('hx-target').value;
  $('hx-rho-field').classList.toggle('is-disabled', kind === 'banana');
  rhoA.input.disabled = kind === 'banana';
  targetA = makeTarget(kind, rhoA.get());
  hmc.target = targetA;
  rngA.seed(randomSeed());
  hmc.reset(...targetA.typical);
  hmc.path.n = 0;
  cur = { x: hmc.x, y: hmc.y };
  sampX.clear();
  sampY.clear();
  sampX.push(cur.x);
  sampY.push(cur.y);
  pacerA.reset();
  plane.setX(targetA.domain[0]).setY(targetA.domain[1]);
  energy.setX([0, LA.get()]).setY([0, 5]);
  drawA();
}

const runnerA = new Runner({
  button: 'hx-run',
  root: 'hx-lab',
  tick(dt) {
    if (animLoop) return true; // a single-step animation is still playing
    const rate = speedA.get();
    if (anim) {
      advanceAnim(dt);
      drawA();
      return true;
    }
    const k = pacerA.take(rate, dt, 500);
    if (k === 0) return true;
    if (rate <= 5) {
      beginStep(true, Math.min(1.6, Math.max(0.35, 0.8 / rate)));
    } else {
      for (let i = 0; i < k && !sampX.full; i++) beginStep(false);
    }
    drawA();
    return !sampX.full;
  },
});

$('hx-once').addEventListener('click', () => {
  runnerA.stop();
  if (anim) commitA();
  beginStep(true, 1.1);
  drawA();
  playOnce();
});
$('hx-reset').addEventListener('click', resetA);
$('hx-target').addEventListener('change', resetA);
resetA();

// =====================================================================
// The race
// =====================================================================

const RACE_MAX = 20000;
let targetR = makeTarget('gaussian', 0.95);

const rhoR = bindRange('rc-rho', { format: (v) => fmtNum(v, 2), onInput: resetRace });
const cR = bindRange('rc-c', { log: true, format: (v) => fmtSig(v, 2), onInput: resetRace });
const epsR = bindRange('rc-eps', { log: true, format: (v) => fmtSig(v, 2), onInput: resetRace });
const LR = bindRange('rc-L', { format: String, onInput: resetRace });
const speedR = bindRange('rc-speed', { log: true, format: fmtRate('iterations') });
const pacerR = new Pacer();

function makeLane(key, Sampler) {
  const rng = new RNG();
  const lane = {
    key,
    rng,
    sampler: new Sampler(targetR, rng),
    xs: new Trace(RACE_MAX),
    ys: new Trace(RACE_MAX),
    recent: new RecentPath(key === 'gibbs' ? 41 : 21),
    diag: null,
  };
  lane.plane = new Plot(`rc-${key}-plane`, {
    square: true,
    maxHeight: 300,
    grid: 'none',
    x: { domain: targetR.domain[0], ticks: 4 },
    y: { domain: targetR.domain[1], ticks: 4 },
    margin: { l: 30, r: 8, b: 22 },
    draw(p) {
      const c = p.c;
      const [dx, dy] = targetR.domain;
      p.image(heatmapFor(targetR), dx[0], dx[1], dy[0], dy[1]);
      if (!lane.active) return;
      p.points(lane.xs.a, lane.ys.a, { color: c.s1, r: lane.xs.n > 2000 ? 1 : 1.8, alpha: 0.55, n: lane.xs.n });
      const [rx, ry] = lane.recent.ordered();
      p.line(rx, ry, { color: c.s2, width: 1.5 });
      if (lane.xs.n) p.marker(lane.xs.last(), lane.ys.last(), { color: c.s2, r: 4 });
    },
  });
  lane.trace = new Plot(`rc-${key}-trace`, {
    height: 96,
    x: { domain: [0, 100], ticks: 3 },
    y: { domain: targetR.domain[0], ticks: 3 },
    margin: { l: 30, r: 8, b: 20, t: 8 },
    draw(p) {
      if (!lane.active) return;
      p.hline(0, { color: p.c.exact, alpha: 0.6 });
      p.line(null, lane.xs.a, { color: p.c.s1, width: 1, n: lane.xs.n });
    },
    hover(x) {
      if (!lane.active || !lane.xs.n) return null;
      const i = Math.max(0, Math.min(lane.xs.n - 1, Math.round(x)));
      return { x: i, title: `iteration ${fmtInt(i)}`, rows: [{ value: fmtNum(lane.xs.a[i], 3), label: 'x', color: 'var(--s1)' }] };
    },
  });
  return lane;
}

const lanes = [makeLane('mh', Metropolis2D), makeLane('gibbs', Gibbs2D), makeLane('hmc', HMC2D)];
const [mhLane, gibbsLane, hmcLane] = lanes;
legend('rc-legend', [
  { label: 'samples', color: 'var(--s1)', kind: 'dot' },
  { label: 'most recent moves', color: 'var(--s2)', kind: 'line' },
  { label: '\\(x = 0\\) (true mean)', color: 'var(--exact)', kind: 'line' },
]);

function costPerIteration(lane) {
  if (lane.key === 'mh') return { per: 1, text: '1 density evaluation' };
  if (lane.key === 'gibbs') return { per: 2, text: '2 conditional draws' };
  return { per: LR.get() + 1, text: `${LR.get() + 1} gradient evaluations` };
}

function renderStats(lane) {
  const ul = $(`rc-${lane.key}-stats`);
  const rows = [];
  if (!lane.active) {
    rows.push(['Status', 'not available for this target']);
  } else {
    const n = lane.xs.n - 1;
    const s = lane.sampler;
    const cost = costPerIteration(lane);
    const d = lane.diag;
    const perIter = d ? (1000 * d.ess) / d.n : NaN;
    // Short chains: tau is underestimated, so ESS is an upper bound and the error bar a lower bound.
    // Anticorrelated chains (tau < 1): ESS is capped at N, which is conservative.
    const essText = (v) => (!d ? '—' : !d.converged ? `≲ ${fmtSig(v, 3)}` : d.tau < 1 ? `${fmtSig(v, 3)} (capped)` : fmtSig(v, 3));
    rows.push(['Iterations', fmtInt(n)]);
    rows.push(['Acceptance', lane.key === 'gibbs' ? '100% (always)' : n ? `${fmtNum(100 * s.acceptance, 1)}%` : '—']);
    rows.push(['Cost per iteration', cost.text]);
    rows.push(['ESS of x per 1,000 iterations', essText(perIter)]);
    rows.push(['ESS of x per 1,000 evaluations', essText(perIter / cost.per)]);
    rows.push(['Mean of x (exact 0)', d ? `${fmtNum(d.mean, 2)} ± ${d.converged ? '' : '≳ '}${fmtSig(d.error, 2)}` : '—']);
  }
  ul.replaceChildren(...rows.map(([k, v]) => {
    const li = document.createElement('li');
    const a = document.createElement('span');
    a.textContent = k;
    const b = document.createElement('b');
    b.textContent = v;
    li.append(a, b);
    return li;
  }));
}

function updateRaceDiagnostics() {
  for (const lane of lanes) {
    if (!lane.active) { lane.diag = null; continue; }
    const n = lane.xs.n;
    const burn = Math.min(200, Math.floor(n / 10));
    lane.diag = n - burn >= 100 ? chainDiagnostics(lane.xs.a, burn, n) : null;
  }
  lanes.forEach(renderStats);
  const short = lanes.some((l) => l.diag && !l.diag.converged);
  const capped = lanes.some((l) => l.diag && l.diag.converged && l.diag.tau < 1);
  const notes = [];
  if (short) notes.push('≲ and ≳: the chain is still too short to measure its autocorrelation time, so ESS is at most the value shown and the error bar at least the value shown. Keep running.');
  if (capped) notes.push('Capped: successive samples are anticorrelated (τint < 1), so the true ESS exceeds the number of iterations. It is shown capped at that number.');
  $('rc-note').textContent = notes.join(' ');
  $('rc-note').hidden = !notes.length;
}
const raceDiagnosticsSoon = throttle(updateRaceDiagnostics, 500);

function drawRace() {
  for (const lane of lanes) {
    lane.plane.redraw();
    lane.trace.setX([0, Math.max(100, lane.xs.n - 1)]);
    lane.trace.redraw();
  }
}

function stepRace(k) {
  const c = cR.get(), eps = epsR.get(), L = LR.get();
  for (let i = 0; i < k; i++) {
    if (mhLane.xs.full) return false;
    mhLane.sampler.step(c);
    mhLane.xs.push(mhLane.sampler.x);
    mhLane.ys.push(mhLane.sampler.y);
    mhLane.recent.push(mhLane.sampler.x, mhLane.sampler.y);

    if (gibbsLane.active) {
      const g = gibbsLane.sampler;
      g.step();
      gibbsLane.recent.push(g.midX, g.midY); // the horizontal move...
      gibbsLane.recent.push(g.x, g.y); // ...then the vertical one
      gibbsLane.xs.push(g.x);
      gibbsLane.ys.push(g.y);
    }

    hmcLane.sampler.step(eps, L);
    hmcLane.xs.push(hmcLane.sampler.x);
    hmcLane.ys.push(hmcLane.sampler.y);
    hmcLane.recent.push(hmcLane.sampler.x, hmcLane.sampler.y);
  }
  return true;
}

function resetRace() {
  runnerR?.stop();
  const kind = $('rc-target').value;
  $('rc-rho-field').classList.toggle('is-disabled', kind === 'banana');
  rhoR.input.disabled = kind === 'banana';
  targetR = makeTarget(kind, rhoR.get());
  for (const lane of lanes) {
    lane.sampler.target = targetR;
    lane.rng.seed(randomSeed());
    lane.active = lane.key !== 'gibbs' || gibbsLane.sampler.available;
    lane.sampler.reset(...targetR.start);
    lane.xs.clear();
    lane.ys.clear();
    lane.recent.clear();
    lane.xs.push(targetR.start[0]);
    lane.ys.push(targetR.start[1]);
    lane.recent.push(...targetR.start);
    lane.diag = null;
    lane.plane.setX(targetR.domain[0]).setY(targetR.domain[1]);
    lane.trace.setY(targetR.domain[0]);
  }
  $('rc-gibbs-na').hidden = gibbsLane.active;
  pacerR.reset();
  drawRace();
  updateRaceDiagnostics();
}

const runnerR = new Runner({
  button: 'rc-run',
  root: 'rc-lab',
  tick(dt) {
    const k = pacerR.take(speedR.get(), dt, 2000);
    if (k === 0) return true;
    const ok = stepRace(k);
    drawRace();
    raceDiagnosticsSoon();
    return ok;
  },
  onChange(running) { if (!running) updateRaceDiagnostics(); },
});
$('rc-reset').addEventListener('click', resetRace);
$('rc-target').addEventListener('change', resetRace);
resetRace();
