// Simulated annealing and replica exchange on a one-dimensional double well (book §6.3).

/** The book's test function: global minimum at x = 1, a slightly higher local minimum near x = -1. */
export const doubleWell = (x) => (x - 1) ** 2 * ((x + 1) ** 2 + 0.01);

/**
 * Exact equilibrium density exp(-f(x)/T) / Z on a grid over [lo, hi], and P(x > 0).
 * Used for the gray reference curves; the samplers never see it.
 */
export function equilibrium(T, lo = -2.2, hi = 2.2, points = 4001) {
  const dx = (hi - lo) / (points - 1);
  const x = new Float64Array(points), p = new Float64Array(points);
  let fmin = Infinity;
  for (let i = 0; i < points; i++) {
    x[i] = lo + i * dx;
    fmin = Math.min(fmin, doubleWell(x[i]));
  }
  let z = 0, right = 0;
  for (let i = 0; i < points; i++) {
    p[i] = Math.exp(-(doubleWell(x[i]) - fmin) / T);
    z += p[i];
    if (x[i] > 0) right += p[i];
  }
  for (let i = 0; i < points; i++) p[i] /= z * dx;
  return { x, p, probRight: right / z };
}

/** Many independent Metropolis walkers sharing one temperature (for annealing schedules). */
export class WalkerPopulation {
  constructor(count, rng) {
    this.rng = rng;
    this.x = new Float64Array(count);
    this.f = new Float64Array(count);
  }

  reset(x0) {
    this.x.fill(x0);
    this.f.fill(doubleWell(x0));
    this.steps = 0;
  }

  /** One Metropolis update of every walker at temperature T with step size c. */
  step(T, c) {
    const { x, f, rng } = this;
    for (let i = 0; i < x.length; i++) {
      const xp = x[i] + c * (2 * rng.uniform() - 1);
      const fp = doubleWell(xp);
      if (fp <= f[i] || rng.uniform() < Math.exp((f[i] - fp) / T)) {
        x[i] = xp;
        f[i] = fp;
      }
    }
    this.steps++;
  }

  fractionRight() {
    let k = 0;
    for (const v of this.x) if (v > 0) k++;
    return k / this.x.length;
  }
}

/** Geometric temperature ladder from hot to cold. */
export function geometricLadder(tHot, tCold, count) {
  const r = Math.pow(tCold / tHot, 1 / (count - 1));
  return Array.from({ length: count }, (_, m) => tHot * r ** m);
}

/**
 * Replica exchange (parallel tempering). Rung m holds one configuration at temperature T[m],
 * ordered hot to cold. Each step updates every rung with Metropolis, then offers swaps between
 * neighboring rungs m and m + 1 with probability min(1, e^{-ΔS}),
 * ΔS = (1/T[m+1] - 1/T[m]) (f(X_m) - f(X_{m+1})).
 * A swap exchanges configurations between temperatures; walkerAt[m] follows which original
 * walker sits on each rung, so one walker's journey up and down the ladder can be drawn.
 */
export class ReplicaExchange1D {
  constructor(temps, rng, stepFor = (T) => Math.min(0.5, Math.sqrt(T))) {
    this.T = Float64Array.from(temps);
    this.beta = this.T.map((t) => 1 / t);
    this.c = this.T.map(stepFor);
    this.rng = rng;
    const M = temps.length;
    this.x = new Float64Array(M);
    this.f = new Float64Array(M);
    this.walkerAt = new Int32Array(M);
    this.swapTried = new Float64Array(M - 1);
    this.swapDone = new Float64Array(M - 1);
  }

  reset(x0) {
    this.x.fill(x0);
    this.f.fill(doubleWell(x0));
    this.walkerAt = Int32Array.from(this.walkerAt, (_, m) => m);
    this.swapTried.fill(0);
    this.swapDone.fill(0);
    this.steps = 0;
    this.accepted = 0;
  }

  /** One Metropolis update per rung, then (unless disabled) a sweep of neighbor swaps. */
  step(allowSwaps = true) {
    const { x, f, T, c, rng, beta } = this;
    const M = x.length;
    for (let m = 0; m < M; m++) {
      const xp = x[m] + c[m] * (2 * rng.uniform() - 1);
      const fp = doubleWell(xp);
      if (fp <= f[m] || rng.uniform() < Math.exp((f[m] - fp) / T[m])) {
        x[m] = xp;
        f[m] = fp;
        this.accepted++;
      }
    }
    for (let m = 0; allowSwaps && m < M - 1; m++) {
      const dS = (beta[m + 1] - beta[m]) * (f[m] - f[m + 1]);
      this.swapTried[m]++;
      if (dS <= 0 || rng.uniform() < Math.exp(-dS)) {
        let t = x[m]; x[m] = x[m + 1]; x[m + 1] = t;
        t = f[m]; f[m] = f[m + 1]; f[m + 1] = t;
        const w = this.walkerAt[m]; this.walkerAt[m] = this.walkerAt[m + 1]; this.walkerAt[m + 1] = w;
        this.swapDone[m]++;
      }
    }
    this.steps++;
  }

  rungOf(walker) {
    return this.walkerAt.indexOf(walker);
  }

  get swapRate() {
    let a = 0, b = 0;
    for (let m = 0; m < this.swapTried.length; m++) { a += this.swapDone[m]; b += this.swapTried[m]; }
    return b ? a / b : NaN;
  }
}
