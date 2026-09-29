// Two-dimensional compact U(1) lattice gauge theory: the simplest relative of lattice QCD (book §6.4).
// The gauge field lives on links as phases U = e^{iθ}. The Wilson action is
// S = β Σ_P (1 − cos θ_P), where θ_P is the sum of link angles around a plaquette.
// Site s = x + L t; link (s, μ) is stored at index 2s + μ, with μ = 0 (x) and μ = 1 (t).

const TWO_PI = 2 * Math.PI;
const wrap = (a) => a - TWO_PI * Math.round(a / TWO_PI); // to (−π, π]

export class U1Lattice {
  constructor(L, rng) {
    this.L = L;
    this.V = L * L;
    this.rng = rng;
    this.theta = new Float64Array(2 * this.V);
    this.up = [new Int32Array(this.V), new Int32Array(this.V)];
    this.dn = [new Int32Array(this.V), new Int32Array(this.V)];
    for (let t = 0; t < L; t++) {
      for (let x = 0; x < L; x++) {
        const s = x + L * t;
        this.up[0][s] = ((x + 1) % L) + L * t;
        this.dn[0][s] = ((x - 1 + L) % L) + L * t;
        this.up[1][s] = x + L * ((t + 1) % L);
        this.dn[1][s] = x + L * ((t - 1 + L) % L);
      }
    }
    this.sinP = new Float64Array(this.V);
    this.setBeta(1);
  }

  setBeta(beta) {
    this.beta = beta;
    this.step = Math.min(Math.PI, 2 / Math.sqrt(beta)); // Metropolis proposal width
  }

  cold() { this.theta.fill(0); }

  hot() { for (let i = 0; i < this.theta.length; i++) this.theta[i] = TWO_PI * this.rng.uniform() - Math.PI; }

  /** Plaquette angle at s: θ_0(s) + θ_1(s + x̂) − θ_0(s + t̂) − θ_1(s). */
  plaquette(s) {
    const th = this.theta;
    return th[2 * s] + th[2 * this.up[0][s] + 1] - th[2 * this.up[1][s]] - th[2 * s + 1];
  }

  meanCosPlaquette() {
    let c = 0;
    for (let s = 0; s < this.V; s++) c += Math.cos(this.plaquette(s));
    return c / this.V;
  }

  action() {
    let S = 0;
    for (let s = 0; s < this.V; s++) S += 1 - Math.cos(this.plaquette(s));
    return this.beta * S;
  }

  /** Topological charge Q = (1/2π) Σ_P θ_P wrapped into (−π, π]. An integer on a torus. */
  topologicalCharge() {
    let q = 0;
    for (let s = 0; s < this.V; s++) q += wrap(this.plaquette(s));
    return q / TWO_PI;
  }

  /** The two plaquettes that contain link (s, μ). */
  plaquettesOf(s, mu) {
    return mu === 0 ? [s, this.dn[1][s]] : [s, this.dn[0][s]];
  }

  /** One Metropolis update of every link. Returns the number of accepted proposals. */
  sweepMetropolis() {
    const { theta, beta, rng, step } = this;
    let acc = 0;
    for (let s = 0; s < this.V; s++) {
      for (let mu = 0; mu < 2; mu++) {
        const [a, b] = this.plaquettesOf(s, mu);
        const before = Math.cos(this.plaquette(a)) + Math.cos(this.plaquette(b));
        const i = 2 * s + mu, old = theta[i];
        theta[i] = wrap(old + step * (2 * rng.uniform() - 1));
        const after = Math.cos(this.plaquette(a)) + Math.cos(this.plaquette(b));
        const dS = -beta * (after - before);
        if (dS <= 0 || rng.uniform() < Math.exp(-dS)) acc++;
        else theta[i] = old;
      }
    }
    return acc;
  }

  /** dS/dθ for every link, written into out (length 2V). */
  gaugeForce(out) {
    const { beta, sinP } = this;
    for (let s = 0; s < this.V; s++) sinP[s] = Math.sin(this.plaquette(s));
    for (let s = 0; s < this.V; s++) {
      out[2 * s] = beta * (sinP[s] - sinP[this.dn[1][s]]); // +θ_0(s) in P(s), −θ_0(s) in P(s − t̂)
      out[2 * s + 1] = beta * (sinP[this.dn[0][s]] - sinP[s]); // −θ_1(s) in P(s), +θ_1(s) in P(s − x̂)
    }
    return out;
  }

  /** Average of cos(angle around an R × T rectangle) over all positions. */
  wilsonLoop(R, T) {
    const { L, theta } = this;
    let sum = 0;
    for (let t0 = 0; t0 < L; t0++) {
      for (let x0 = 0; x0 < L; x0++) {
        let a = 0;
        for (let i = 0; i < R; i++) {
          a += theta[2 * (((x0 + i) % L) + L * t0)];
          a -= theta[2 * (((x0 + i) % L) + L * ((t0 + T) % L))];
        }
        for (let j = 0; j < T; j++) {
          a += theta[2 * (((x0 + R) % L) + L * ((t0 + j) % L)) + 1];
          a -= theta[2 * (x0 + L * ((t0 + j) % L)) + 1];
        }
        sum += Math.cos(a);
      }
    }
    return sum / this.V;
  }

  /** Apply a random gauge transformation θ_μ(s) → θ_μ(s) + α(s) − α(s + μ̂). Physics must not change. */
  randomGaugeTransform() {
    const alpha = Float64Array.from({ length: this.V }, () => TWO_PI * this.rng.uniform());
    for (let s = 0; s < this.V; s++) {
      for (let mu = 0; mu < 2; mu++) this.theta[2 * s + mu] = wrap(this.theta[2 * s + mu] + alpha[s] - alpha[this.up[mu][s]]);
    }
  }
}

/**
 * Leapfrog HMC for the link angles. `forceFn(out)` must write dS/dθ; `actionFn()` returns S.
 * Shared by the pure-gauge lab and the Schwinger model, which adds the quark force.
 */
export function hmcTrajectory(lat, { eps, steps, forceFn, actionFn, kinetic = true }) {
  const n = lat.theta.length, rng = lat.rng;
  const p = new Float64Array(n), f = new Float64Array(n), old = Float64Array.from(lat.theta);
  let K0 = 0;
  for (let i = 0; i < n; i++) { p[i] = rng.normal(); K0 += 0.5 * p[i] * p[i]; }
  const S0 = actionFn();
  forceFn(f);
  for (let k = 0; k < steps; k++) {
    for (let i = 0; i < n; i++) p[i] -= 0.5 * eps * f[i];
    for (let i = 0; i < n; i++) lat.theta[i] += eps * p[i];
    forceFn(f);
    for (let i = 0; i < n; i++) p[i] -= 0.5 * eps * f[i];
  }
  let K1 = 0;
  for (let i = 0; i < n; i++) K1 += 0.5 * p[i] * p[i];
  const dH = K1 + actionFn() - K0 - S0;
  const accept = Number.isFinite(dH) && (dH <= 0 || rng.uniform() < Math.exp(-dH));
  if (accept) for (let i = 0; i < n; i++) lat.theta[i] = wrap(lat.theta[i]);
  else lat.theta.set(old);
  return { accept, dH };
}

// ---------- Exact results on an L × L torus (character expansion) ----------

/** log I_n(x), the modified Bessel function, from its power series in log space (x > 0). */
export function logBesselI(n, x) {
  n = Math.abs(n);
  const lx = Math.log(x / 2);
  let lf = 0; // log(k!) and log((k + n)!) built incrementally
  let lfn = 0;
  for (let j = 2; j <= n; j++) lfn += Math.log(j);
  const terms = [];
  let m = -Infinity;
  for (let k = 0; k < 1000; k++) {
    if (k > 0) { lf += Math.log(k); lfn += Math.log(k + n); }
    const t = (2 * k + n) * lx - lf - lfn;
    terms.push(t);
    if (t > m) m = t;
    else if (t < m - 40) break; // past the peak and e^-40 below it
  }
  let s = 0;
  for (const t of terms) s += Math.exp(t - m);
  return m + Math.log(s);
}

/**
 * Exact expectation of a contractible Wilson loop enclosing A plaquettes on a torus of V plaquettes:
 * ⟨W_A⟩ = Σ_n I_n(β)^(V−A) I_(n+1)(β)^A / Σ_n I_n(β)^V. A = 1 gives ⟨cos θ_P⟩.
 * For V → ∞ this becomes (I_1/I_0)^A: an area law, with string tension σ = −ln(I_1/I_0).
 */
export function u1Exact(beta, V) {
  const N = 40;
  const lI = new Map();
  for (let n = -N; n <= N + 1; n++) lI.set(n, logBesselI(n, beta));
  const lse = (arr) => {
    const m = Math.max(...arr);
    return m + Math.log(arr.reduce((acc, v) => acc + Math.exp(v - m), 0));
  };
  const ns = Array.from({ length: 2 * N + 1 }, (_, i) => i - N);
  const logZ = lse(ns.map((n) => V * lI.get(n)));
  return {
    wilson: (A) => Math.exp(lse(ns.map((n) => (V - A) * lI.get(n) + A * lI.get(n + 1))) - logZ),
    plaquette: Math.exp(lse(ns.map((n) => (V - 1) * lI.get(n) + lI.get(n + 1))) - logZ),
    stringTension: -(lI.get(1) - lI.get(0)),
  };
}
