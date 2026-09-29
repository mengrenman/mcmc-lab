// Statistical and exact checks of every sampler and helper, run in the browser.
// Statistical checks compare an estimate with the exact value using the chain's own
// error bar, sigma * sqrt(tau_int / N), and pass within 4 error bars (plus any stated allowance).

import { mountChrome, $, setText } from './lib/ui.js';
import { fmtNum, fmtSig } from './lib/plot.js';
import { RNG, randomSeed } from './lib/random.js';
import { mean, variance, autocorrelation, integratedTime, jackknifeMeanError, chainDiagnostics } from './lib/stats.js';
import { targets1D, correlatedGaussian, banana, integrand, importanceWeightVariance, uniformWeightVariance, SQRT2PI } from './lib/targets.js';
import { Metropolis1D, Metropolis2D } from './samplers/metropolis.js';
import { HMC2D, leapfrog } from './samplers/hmc.js';
import { Gibbs2D } from './samplers/gibbs.js';
import { Ising, T_CRITICAL, onsagerMagnetization, onsagerEnergy } from './samplers/ising.js';

mountChrome();

let seedBase = 20260929;
const tests = [];
const test = (group, name, fn) => tests.push({ group, name, fn });

/** Pass if |observed - expected| <= k * err + allowance. */
function statCheck(observed, err, expected, { k = 4, allowance = 0 } = {}) {
  const dev = Math.abs(observed - expected);
  return {
    pass: Number.isFinite(dev) && dev <= k * err + allowance,
    expected: fmtSig(expected, 5),
    observed: `${fmtSig(observed, 5)} ± ${fmtSig(err, 2)}`,
    detail: `${fmtNum(dev / err, 1)} error bars off${allowance ? ` (allowance ${allowance})` : ''}`,
  };
}

function exactCheck(observed, expected, tol, label = '') {
  const dev = Math.abs(observed - expected);
  return {
    pass: dev <= tol,
    expected: fmtSig(expected, 6),
    observed: fmtSig(observed, 6),
    detail: `|difference| = ${dev.toExponential(1)}, tolerance ${tol}${label ? '; ' + label : ''}`,
  };
}

/** Mean of f(sample) along a chain, with its autocorrelation-aware error. */
function chainMean(values) {
  const d = chainDiagnostics(values, 0, values.length);
  return { mean: d.mean, err: d.error, tau: d.tau };
}

// ---------------- Random numbers ----------------

test('Random numbers', 'Uniform mean is 1/2', () => {
  const r = new RNG(seedBase), n = 1e6;
  let s = 0;
  for (let i = 0; i < n; i++) s += r.uniform();
  return statCheck(s / n, Math.sqrt(1 / 12 / n), 0.5);
});

test('Random numbers', 'Box–Muller normals: mean 0, variance 1', () => {
  const r = new RNG(seedBase + 1), n = 1e6;
  const a = new Float64Array(n);
  for (let i = 0; i < n; i++) a[i] = r.normal();
  const v = variance(a);
  const res = statCheck(v, Math.sqrt(2 / n), 1);
  const m = mean(a);
  res.pass = res.pass && Math.abs(m) < 4 / Math.sqrt(n);
  res.observed = `mean ${fmtSig(m, 2)}, variance ${fmtSig(v, 5)}`;
  res.expected = 'mean 0, variance 1';
  return res;
});

test('Random numbers', 'Same seed gives the same stream', () => {
  const a = new RNG(12345), b = new RNG(12345);
  let same = true;
  for (let i = 0; i < 1000; i++) same = same && a.nextU32() === b.nextU32();
  return { pass: same, expected: 'identical', observed: same ? 'identical' : 'different', detail: '1,000 draws compared' };
});

// ---------------- Statistics ----------------

function ar1(n, phi, seed) {
  const r = new RNG(seed), a = new Float64Array(n);
  let x = 0;
  const sc = Math.sqrt(1 - phi * phi);
  for (let i = 0; i < n; i++) {
    x = phi * x + sc * r.normal();
    a[i] = x;
  }
  return a;
}

test('Statistics', 'FFT autocorrelation equals the direct sum', () => {
  const a = ar1(3000, 0.7, seedBase + 2);
  const rho = autocorrelation(a);
  const m = mean(a);
  let c0 = 0;
  for (const v of a) c0 += (v - m) ** 2;
  let worst = 0;
  for (let t = 0; t < 200; t++) {
    let c = 0;
    for (let i = 0; i + t < a.length; i++) c += (a[i] - m) * (a[i + t] - m);
    worst = Math.max(worst, Math.abs(c / c0 - rho[t]));
  }
  return exactCheck(worst, 0, 1e-10, 'largest difference over lags 0–199');
});

test('Statistics', 'τ_int of an AR(1) process with φ = 0.9 is (1 + φ)/(1 − φ) = 19', () => {
  const n = 1 << 18;
  const { tau, window } = integratedTime(autocorrelation(ar1(n, 0.9, seedBase + 3)));
  // Statistical error of the windowed estimator (Sokal): tau * sqrt(2 (2W + 1) / n).
  const err = tau * Math.sqrt((2 * (2 * window + 1)) / n);
  return statCheck(tau, err, 19);
});

test('Statistics', 'Jackknife with bin size 1 equals the naive standard error', () => {
  const a = ar1(10000, 0.5, seedBase + 4);
  return exactCheck(jackknifeMeanError(a, 1), Math.sqrt(variance(a) / a.length), 1e-12);
});

test('Statistics', 'Jackknife plateau agrees with σ √(τ/N) for AR(1), φ = 0.8', () => {
  const a = ar1(1 << 18, 0.8, seedBase + 5);
  const d = chainDiagnostics(a);
  const jk = jackknifeMeanError(a, 1024);
  // The jackknife error with 256 bins is itself uncertain by about 1/sqrt(2 * 256) ≈ 4.4%.
  return statCheck(jk, d.error * 0.044, d.error);
});

// ---------------- Metropolis ----------------

function runMetropolis1D(target, c, n, seed, x0 = target.start) {
  const m = new Metropolis1D(target, new RNG(seed));
  m.reset(x0);
  for (let i = 0; i < 2000; i++) m.step(c); // burn-in
  const x = new Float64Array(n), x2 = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    m.step(c);
    x[i] = m.x;
    x2[i] = m.x * m.x;
  }
  return { x: chainMean(x), x2: chainMean(x2), acc: m.acceptance };
}

test('Metropolis', 'N(0, 1), c = 3.5: E[x] = 0', () => {
  const r = runMetropolis1D(targets1D.gaussian, 3.5, 300000, seedBase + 10);
  return statCheck(r.x.mean, r.x.err, 0);
});
test('Metropolis', 'N(0, 1), c = 3.5: E[x²] = 1', () => {
  const r = runMetropolis1D(targets1D.gaussian, 3.5, 300000, seedBase + 11);
  return statCheck(r.x2.mean, r.x2.err, 1);
});
test('Metropolis', 'N(0, 1): τ_int is smallest near c = 3.5 (≈ 3.7) and ≈ 16 at c = 1', () => {
  const t1 = runMetropolis1D(targets1D.gaussian, 1, 300000, seedBase + 12).x.tau;
  const t35 = runMetropolis1D(targets1D.gaussian, 3.5, 300000, seedBase + 13).x.tau;
  const t8 = runMetropolis1D(targets1D.gaussian, 8, 300000, seedBase + 14).x.tau;
  const pass = t35 < t1 && t35 < t8 && t35 > 3 && t35 < 4.5 && t1 > 13 && t1 < 20;
  return { pass, expected: 'τ(1) ≈ 16, τ(3.5) ≈ 3.7, τ(8) larger', observed: `τ(1) = ${fmtSig(t1, 3)}, τ(3.5) = ${fmtSig(t35, 3)}, τ(8) = ${fmtSig(t8, 3)}`, detail: 'backs the step-size claim on the Metropolis page' };
});
test('Metropolis', 'Gamma(3, 1): E[x] = 3 (boundary at x = 0)', () => {
  const r = runMetropolis1D(targets1D.gamma, 3, 300000, seedBase + 15);
  return statCheck(r.x.mean, r.x.err, 3);
});
test('Metropolis', 'Gamma(3, 1): E[x²] = 12', () => {
  const r = runMetropolis1D(targets1D.gamma, 3, 300000, seedBase + 16);
  return statCheck(r.x2.mean, r.x2.err, 12);
});
test('Metropolis', 'Two modes, c = 6: E[x²] = 9.49', () => {
  const r = runMetropolis1D(targets1D.bimodal, 6, 300000, seedBase + 17);
  return statCheck(r.x2.mean, r.x2.err, targets1D.bimodal.variance);
});
test('Metropolis', '2D, ρ = 0.5: E[xy] = 0.5', () => {
  const t = correlatedGaussian(0.5);
  const m = new Metropolis2D(t, new RNG(seedBase + 18));
  m.reset(0, 0);
  const n = 300000, xy = new Float64Array(n);
  for (let i = 0; i < 2000; i++) m.step(1.5);
  for (let i = 0; i < n; i++) { m.step(1.5); xy[i] = m.x * m.y; }
  const r = chainMean(xy);
  return statCheck(r.mean, r.err, 0.5);
});

// ---------------- HMC and Gibbs ----------------

test('HMC', 'Leapfrog is time-reversible (correlated Gaussian and banana)', () => {
  let worst = 0;
  for (const t of [correlatedGaussian(0.9), banana()]) {
    const [qx, qy, px, py] = leapfrog(t, 0.3, -0.5, 0.7, 1.1, 0.05, 40);
    const [bx, by] = leapfrog(t, qx, qy, -px, -py, 0.05, 40);
    worst = Math.max(worst, Math.abs(bx - 0.3), Math.abs(by + 0.5));
  }
  return exactCheck(worst, 0, 1e-10, 'forward, flip momentum, back');
});

test('HMC', 'Leapfrog energy error shrinks like ε²', () => {
  // The energy error oscillates along a trajectory, so compare its largest value, not the endpoint.
  const t = correlatedGaussian(0.9);
  const H = (s) => t.U(s[0], s[1]) + 0.5 * (s[2] ** 2 + s[3] ** 2);
  const maxErr = (eps) => {
    let s = [0.5, 0.2, 0.3, -0.8];
    const H0 = H(s);
    let worst = 0;
    for (let i = 0; i < Math.round(3 / eps); i++) {
      s = leapfrog(t, ...s, eps, 1);
      worst = Math.max(worst, Math.abs(H(s) - H0));
    }
    return worst;
  };
  const ratio = maxErr(0.04) / maxErr(0.02);
  return { pass: ratio > 3.5 && ratio < 4.5, expected: '≈ 4', observed: fmtSig(ratio, 3), detail: 'max |ΔH| along a trajectory of length 3, at ε = 0.04 divided by ε = 0.02' };
});

function runSampler2D(sampler, stepFn, n, stats) {
  sampler.reset(0.5, 0.5);
  for (let i = 0; i < 1000; i++) stepFn();
  const arrs = stats.map(() => new Float64Array(n));
  for (let i = 0; i < n; i++) {
    stepFn();
    stats.forEach((f, k) => { arrs[k][i] = f(sampler.x, sampler.y); });
  }
  return arrs.map(chainMean);
}

test('HMC', 'Correlated Gaussian ρ = 0.9: E[x²] = 1 and E[xy] = 0.9', () => {
  const t = correlatedGaussian(0.9);
  const h = new HMC2D(t, new RNG(seedBase + 20));
  const [x2, xy] = runSampler2D(h, () => h.step(0.15, 20), 60000, [(x) => x * x, (x, y) => x * y]);
  const a = statCheck(x2.mean, x2.err, 1), b = statCheck(xy.mean, xy.err, 0.9);
  return { pass: a.pass && b.pass, expected: 'E[x²] = 1, E[xy] = 0.9', observed: `${a.observed}; ${b.observed}`, detail: `${a.detail}; ${b.detail}` };
});

test('HMC', 'Banana: E[y] = 0, E[x²] = 1.44, E[y²] = 1.1968', () => {
  const t = banana();
  const h = new HMC2D(t, new RNG(seedBase + 21));
  const [y, x2, y2] = runSampler2D(h, () => h.step(0.1, 25), 60000, [(x, yy) => yy, (x) => x * x, (x, yy) => yy * yy]);
  const checks = [statCheck(y.mean, y.err, 0), statCheck(x2.mean, x2.err, t.variance[0]), statCheck(y2.mean, y2.err, t.variance[1])];
  return {
    pass: checks.every((c) => c.pass),
    expected: 'E[y] = 0, E[x²] = 1.44, E[y²] = 1.1968',
    observed: checks.map((c) => c.observed).join('; '),
    detail: checks.map((c) => c.detail).join('; '),
  };
});

test('Gibbs', 'Correlated Gaussian ρ = 0.9: E[x²] = 1 and E[xy] = 0.9', () => {
  const t = correlatedGaussian(0.9);
  const g = new Gibbs2D(t, new RNG(seedBase + 22));
  const [x2, xy] = runSampler2D(g, () => g.step(), 300000, [(x) => x * x, (x, y) => x * y]);
  const a = statCheck(x2.mean, x2.err, 1), b = statCheck(xy.mean, xy.err, 0.9);
  return { pass: a.pass && b.pass, expected: 'E[x²] = 1, E[xy] = 0.9', observed: `${a.observed}; ${b.observed}`, detail: `${a.detail}; ${b.detail}` };
});

// ---------------- Ising ----------------

test('Ising', 'All spins up: E = −2N and M = N', () => {
  const m = new Ising(16, new RNG(1));
  m.order();
  const pass = m.E === -2 * m.N && m.M === m.N;
  return { pass, expected: `E = ${-2 * m.N}, M = ${m.N}`, observed: `E = ${m.E}, M = ${m.M}`, detail: '16 × 16 periodic lattice' };
});

test('Ising', 'Incremental E and M match a full recount after Metropolis and Wolff updates', () => {
  const m = new Ising(24, new RNG(seedBase + 30));
  m.setT(2.3);
  let worst = 0;
  for (let i = 0; i < 200; i++) {
    m.sweepMetropolis();
    const E = m.E, M = m.M;
    m.recompute();
    worst = Math.max(worst, Math.abs(E - m.E), Math.abs(M - m.M));
    m.wolffStep();
    const M2 = m.M;
    m.recompute();
    worst = Math.max(worst, Math.abs(M2 - m.M));
  }
  return exactCheck(worst, 0, 0, '200 rounds');
});

test('Ising', 'Onsager energy: e(1), e(2), e(3) and continuity at T_c = −√2', () => {
  // Reference values from an independent evaluation of the same formula with K(k) by direct
  // quadrature instead of the arithmetic–geometric mean used here.
  const ref = [[1, -1.99716], [2, -1.745565], [3, -0.81731]];
  const worst = Math.max(...ref.map(([T, e]) => Math.abs(onsagerEnergy(T) - e)));
  const below = onsagerEnergy(T_CRITICAL - 1e-7), above = onsagerEnergy(T_CRITICAL + 1e-7);
  const pass = worst < 1e-5 && Math.abs(below + Math.SQRT2) < 1e-4 && Math.abs(above + Math.SQRT2) < 1e-4;
  return {
    pass,
    expected: 'e(1) = −1.99716, e(2) = −1.74557, e(3) = −0.81731, e(T_c) = −1.41421',
    observed: `worst |difference| ${worst.toExponential(1)}; e(T_c ± 10⁻⁷) = ${fmtNum(below, 5)}, ${fmtNum(above, 5)}`,
    detail: 'AGM elliptic integral against quadrature',
  };
});

function isingRun(alg, L, T, therm, meas, seed) {
  const m = new Ising(L, new RNG(seed));
  m.setT(T);
  for (let i = 0; i < therm; i++) m.sweep(alg);
  const am = new Float64Array(meas), e = new Float64Array(meas);
  for (let i = 0; i < meas; i++) {
    m.sweep(alg);
    am[i] = Math.abs(m.M) / m.N;
    e[i] = m.E / m.N;
  }
  return { m: chainMean(am), e: chainMean(e) };
}

test('Ising', 'Wolff, L = 32, T = 1.5: ⟨|m|⟩ matches Onsager', () => {
  const r = isingRun('wolff', 32, 1.5, 500, 4000, seedBase + 31);
  // Finite-size corrections are tiny deep in the ordered phase; allow 0.002.
  return statCheck(r.m.mean, r.m.err, onsagerMagnetization(1.5), { allowance: 0.002 });
});

test('Ising', 'Metropolis, L = 32, T = 3.0: ⟨e⟩ matches Onsager', () => {
  const r = isingRun('metropolis', 32, 3.0, 1000, 8000, seedBase + 32);
  return statCheck(r.e.mean, r.e.err, onsagerEnergy(3.0), { allowance: 0.003 });
});

test('Ising', 'All three algorithms agree on ⟨e⟩ at L = 16, T = 2.0', () => {
  // A regression check for Wolff: stopping each sweep once N spins had flipped biased ⟨e⟩ low.
  const runs = ['metropolis', 'heatbath', 'wolff'].map((alg, i) => isingRun(alg, 16, 2.0, 1000, 40000, seedBase + 40 + i));
  let worst = 0;
  const parts = [];
  for (let i = 0; i < 3; i++) {
    for (let j = i + 1; j < 3; j++) {
      const z = Math.abs(runs[i].e.mean - runs[j].e.mean) / Math.hypot(runs[i].e.err, runs[j].e.err);
      worst = Math.max(worst, z);
    }
    parts.push(`${fmtNum(runs[i].e.mean, 4)} ± ${fmtSig(runs[i].e.err, 1)}`);
  }
  return { pass: worst < 4, expected: 'pairwise within 4 combined error bars', observed: parts.join(', '), detail: `worst pair: ${fmtNum(worst, 1)} combined error bars` };
});

// ---------------- Monte Carlo integration ----------------

test('Monte Carlo', 'Importance-weight variance formula (s = 1.5)', () => {
  const r = new RNG(seedBase + 50), n = 1e6, s = 1.5;
  let sw = 0, sw2 = 0;
  for (let i = 0; i < n; i++) {
    const x = s * r.normal();
    const w = integrand.f(x) / (Math.exp((-0.5 * x * x) / (s * s)) / (s * SQRT2PI));
    sw += w;
    sw2 += w * w;
  }
  const v = (sw2 - (sw * sw) / n) / (n - 1);
  const exact = importanceWeightVariance(s);
  return { pass: Math.abs(v / exact - 1) < 0.05, expected: fmtSig(exact, 4), observed: fmtSig(v, 4), detail: `mean weight ${fmtSig(sw / n, 5)} (exact ${fmtSig(integrand.exact, 5)}); 5% tolerance on the variance` };
});

test('Monte Carlo', 'Uniform-weight variance formula on [−20, 20]', () => {
  const r = new RNG(seedBase + 51), n = 1e6, L = integrand.box;
  let sw = 0, sw2 = 0;
  for (let i = 0; i < n; i++) {
    const w = 2 * L * integrand.f(L * (2 * r.uniform() - 1));
    sw += w;
    sw2 += w * w;
  }
  const v = (sw2 - (sw * sw) / n) / (n - 1);
  const exact = uniformWeightVariance();
  return { pass: Math.abs(v / exact - 1) < 0.05, expected: fmtSig(exact, 4), observed: fmtSig(v, 4), detail: '5% tolerance' };
});

// ---------------- Runner ----------------

function render(results) {
  const body = $('test-body');
  body.replaceChildren();
  for (const r of results) {
    const tr = document.createElement('tr');
    const status = document.createElement('td');
    const badge = document.createElement('span');
    badge.className = `badge ${r.pass ? 'good' : 'bad'}`;
    badge.textContent = r.pass ? '✓ Pass' : '✗ Fail';
    status.append(badge);
    const cells = [r.group, r.name, r.expected, r.observed, r.detail, `${fmtNum(r.ms, 0)} ms`];
    tr.append(status);
    cells.forEach((text, i) => {
      const td = document.createElement('td');
      td.textContent = text ?? '';
      if (i >= 2) td.className = 'num';
      tr.append(td);
    });
    body.append(tr);
  }
}

async function runAll() {
  $('test-run').disabled = true;
  const results = [];
  for (let i = 0; i < tests.length; i++) {
    const t = tests[i];
    setText('test-status', `Running ${i + 1} of ${tests.length}: ${t.name}`);
    await new Promise((r) => setTimeout(r, 0));
    const t0 = performance.now();
    let res;
    try {
      res = t.fn();
    } catch (err) {
      res = { pass: false, expected: '', observed: 'threw an error', detail: String(err) };
    }
    results.push({ ...res, group: t.group, name: t.name, ms: performance.now() - t0 });
    render(results);
  }
  const failed = results.filter((r) => !r.pass).length;
  setText('test-status', failed ? `${failed} of ${results.length} checks failed (seed base ${seedBase}).` : `All ${results.length} checks passed (seed base ${seedBase}).`);
  document.body.dataset.testsDone = failed ? 'fail' : 'pass';
  $('test-run').disabled = false;
}

$('test-run').addEventListener('click', () => {
  seedBase = randomSeed();
  runAll();
});
runAll();
