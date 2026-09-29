// The Schwinger model: 2D U(1) gauge theory with two flavors of Wilson quarks (book §6.4.1).
// Integrating out the quarks leaves the weight det(D D†) e^{−S_G}. As in lattice QCD, the
// determinant is replaced by a pseudofermion F: det(D D†) ∝ ∫ dF exp(−F† (D D†)^{-1} F).
// Each trajectory draws Φ from e^{−Φ†Φ}, sets F = D Φ, then moves the links with HMC,
// solving (D D†) χ = F by conjugate gradient at every force evaluation.
//
// Spinors have two complex components per site, stored interleaved:
// v[4s] = Re up, v[4s+1] = Im up, v[4s+2] = Re down, v[4s+3] = Im down.
// Gamma matrices: γ_0 = σ1, γ_1 = σ2, γ_5 = σ3. Fermions are antiperiodic in time (μ = 1).

export class WilsonDirac {
  constructor(lattice, mass) {
    this.lat = lattice;
    this.mass = mass;
    this.n = 4 * lattice.V;
    this.tmp = new Float64Array(this.n);
  }

  /** Boundary sign for the hop from site s forward in direction mu. */
  sign(s, mu) {
    const { L } = this.lat;
    return mu === 1 && Math.floor(s / L) === L - 1 ? -1 : 1;
  }

  /** out = D v, with D = (m + 2) − ½ Σ_μ [(1 − γ_μ) U_μ(x) δ_{x+μ} + (1 + γ_μ) U_μ(x−μ)* δ_{x−μ}]. */
  apply(out, v) {
    const { lat, mass } = this;
    const th = lat.theta, V = lat.V, L = lat.L;
    const diag = mass + 2;
    for (let s = 0; s < V; s++) {
      const o = 4 * s;
      let ar = diag * v[o], ai = diag * v[o + 1], br = diag * v[o + 2], bi = diag * v[o + 3];
      const t = Math.floor(s / L);
      for (let mu = 0; mu < 2; mu++) {
        // Forward hop to y = s + μ with U_μ(s).
        {
          const y = lat.up[mu][s], q = 4 * y;
          const sg = mu === 1 && t === L - 1 ? -0.5 : 0.5;
          const c = Math.cos(th[2 * s + mu]), sn = Math.sin(th[2 * s + mu]);
          // w = U v_y
          const w0r = c * v[q] - sn * v[q + 1], w0i = c * v[q + 1] + sn * v[q];
          const w1r = c * v[q + 2] - sn * v[q + 3], w1i = c * v[q + 3] + sn * v[q + 2];
          let pr, pi, qr, qi; // (1 − γ_μ) w
          if (mu === 0) { pr = w0r - w1r; pi = w0i - w1i; qr = w1r - w0r; qi = w1i - w0i; }
          else { pr = w0r - w1i; pi = w0i + w1r; qr = w1r + w0i; qi = w1i - w0r; } // (a + i b, b − i a)
          ar -= sg * pr; ai -= sg * pi; br -= sg * qr; bi -= sg * qi;
        }
        // Backward hop to y = s − μ with U_μ(y)*.
        {
          const y = lat.dn[mu][s], q = 4 * y;
          const sg = mu === 1 && t === 0 ? -0.5 : 0.5;
          const c = Math.cos(th[2 * y + mu]), sn = -Math.sin(th[2 * y + mu]);
          const w0r = c * v[q] - sn * v[q + 1], w0i = c * v[q + 1] + sn * v[q];
          const w1r = c * v[q + 2] - sn * v[q + 3], w1i = c * v[q + 3] + sn * v[q + 2];
          let pr, pi, qr, qi; // (1 + γ_μ) w
          if (mu === 0) { pr = w0r + w1r; pi = w0i + w1i; qr = pr; qi = pi; }
          else { pr = w0r + w1i; pi = w0i - w1r; qr = w1r - w0i; qi = w1i + w0r; } // (a − i b, b + i a)
          ar -= sg * pr; ai -= sg * pi; br -= sg * qr; bi -= sg * qi;
        }
      }
      out[o] = ar; out[o + 1] = ai; out[o + 2] = br; out[o + 3] = bi;
    }
    return out;
  }

  /** out = D† v, using γ5-hermiticity: D† = γ5 D γ5. */
  applyDagger(out, v) {
    const g = this.tmp;
    for (let i = 0; i < this.n; i += 4) {
      g[i] = v[i]; g[i + 1] = v[i + 1]; g[i + 2] = -v[i + 2]; g[i + 3] = -v[i + 3];
    }
    this.apply(out, g);
    for (let i = 0; i < this.n; i += 4) { out[i + 2] = -out[i + 2]; out[i + 3] = -out[i + 3]; }
    return out;
  }
}

const dot = (a, b) => { let s = 0; for (let i = 0; i < a.length; i++) s += a[i] * b[i]; return s; }; // Re⟨a, b⟩

/**
 * Solve (D D†) x = b by conjugate gradient, starting from the guess in x.
 * Returns the number of iterations. Stops when |residual| ≤ tol |b|.
 */
export function solveCG(dirac, x, b, { tol = 1e-10, maxIter = 5000 } = {}) {
  const n = b.length;
  const r = new Float64Array(n), p = new Float64Array(n), Ap = new Float64Array(n), t = new Float64Array(n);
  const M = (out, v) => { dirac.applyDagger(t, v); dirac.apply(out, t); };
  M(Ap, x);
  for (let i = 0; i < n; i++) { r[i] = b[i] - Ap[i]; p[i] = r[i]; }
  let rr = dot(r, r);
  const stop = tol * tol * dot(b, b);
  let k = 0;
  for (; k < maxIter && rr > stop; k++) {
    M(Ap, p);
    const alpha = rr / dot(p, Ap);
    for (let i = 0; i < n; i++) { x[i] += alpha * p[i]; r[i] -= alpha * Ap[i]; }
    const rrNew = dot(r, r);
    const beta = rrNew / rr;
    rr = rrNew;
    for (let i = 0; i < n; i++) p[i] = r[i] + beta * p[i];
  }
  return k;
}

/** Pseudofermion bookkeeping for one trajectory, with timing of the CG solves. */
export class Pseudofermion {
  constructor(dirac) {
    this.dirac = dirac;
    const n = dirac.n;
    this.F = new Float64Array(n);
    this.chi = new Float64Array(n); // (D D†)^{-1} F, reused as the next CG starting guess
    this.Y = new Float64Array(n);
    this.cgIterations = 0;
    this.cgSolves = 0;
    this.cgTime = 0;
  }

  /** Heat bath: Φ with density e^{−Φ†Φ}, then F = D Φ. Returns Φ†Φ (= S_F at the start). */
  refresh(rng) {
    const n = this.dirac.n, phi = new Float64Array(n);
    const sd = Math.SQRT1_2; // each real component of Φ has variance 1/2
    let s = 0;
    for (let i = 0; i < n; i++) { phi[i] = sd * rng.normal(); s += phi[i] * phi[i]; }
    this.dirac.apply(this.F, phi);
    this.chi.fill(0);
    return s;
  }

  solve() {
    const t0 = performance.now();
    this.cgIterations += solveCG(this.dirac, this.chi, this.F);
    this.cgSolves++;
    this.cgTime += performance.now() - t0;
  }

  /** S_F = F† (D D†)^{-1} F, using the current solution χ. */
  action() {
    return dot(this.F, this.chi);
  }

  /**
   * Adds dS_F/dθ to out. With X = χ and Y = D† X,
   * dS_F/dθ_μ(x) = b Re[ i U X_x† (1 − γ_μ) Y_{x+μ} − i U* X_{x+μ}† (1 + γ_μ) Y_x ],
   * where b is the boundary sign of the link. (Derived from δS_F = −2 Re[X† δD Y].)
   */
  addForce(out) {
    const { dirac, chi: X, Y } = this;
    const lat = dirac.lat, th = lat.theta, V = lat.V;
    dirac.applyDagger(Y, X);
    for (let s = 0; s < V; s++) {
      for (let mu = 0; mu < 2; mu++) {
        const y = lat.up[mu][s];
        const b = dirac.sign(s, mu);
        const c = Math.cos(th[2 * s + mu]), sn = Math.sin(th[2 * s + mu]);
        const xs = 4 * s, xy = 4 * y;
        // A = X_s† (1 − γ_μ) Y_y
        let m0r, m0i, m1r, m1i;
        const ar = Y[xy], ai = Y[xy + 1], br = Y[xy + 2], bi = Y[xy + 3];
        if (mu === 0) { m0r = ar - br; m0i = ai - bi; m1r = br - ar; m1i = bi - ai; }
        else { m0r = ar - bi; m0i = ai + br; m1r = br + ai; m1i = bi - ar; }
        const Ar = X[xs] * m0r + X[xs + 1] * m0i + X[xs + 2] * m1r + X[xs + 3] * m1i;
        const Ai = X[xs] * m0i - X[xs + 1] * m0r + X[xs + 2] * m1i - X[xs + 3] * m1r;
        // B = X_y† (1 + γ_μ) Y_s
        const cr = Y[xs], ci = Y[xs + 1], dr = Y[xs + 2], di = Y[xs + 3];
        let n0r, n0i, n1r, n1i;
        if (mu === 0) { n0r = cr + dr; n0i = ci + di; n1r = n0r; n1i = n0i; }
        else { n0r = cr + di; n0i = ci - dr; n1r = dr - ci; n1i = di + cr; }
        const Br = X[xy] * n0r + X[xy + 1] * n0i + X[xy + 2] * n1r + X[xy + 3] * n1i;
        const Bi = X[xy] * n0i - X[xy + 1] * n0r + X[xy + 2] * n1i - X[xy + 3] * n1r;
        // Re[i U A] = −Im[U A] ; Re[−i U* B] = Im[U* B]
        const UAim = c * Ai + sn * Ar;
        const UsBim = c * Bi - sn * Br;
        out[2 * s + mu] += b * (-UAim + UsBim);
      }
    }
    return out;
  }
}

/**
 * One HMC trajectory for the Schwinger model (the book's "HMC with pseudofermion" recipe):
 * refresh Φ and F = D Φ, then leapfrog the links under S_G + F† (D D†)^{-1} F.
 */
export function schwingerTrajectory(lat, pf, { eps, steps }) {
  const n = lat.theta.length, rng = lat.rng;
  const p = new Float64Array(n), f = new Float64Array(n), old = Float64Array.from(lat.theta);
  const force = () => { lat.gaugeForce(f); pf.solve(); pf.addForce(f); };
  let K0 = 0;
  for (let i = 0; i < n; i++) { p[i] = rng.normal(); K0 += 0.5 * p[i] * p[i]; }
  const SF0 = pf.refresh(rng);
  const H0 = K0 + lat.action() + SF0;
  force();
  for (let k = 0; k < steps; k++) {
    for (let i = 0; i < n; i++) p[i] -= 0.5 * eps * f[i];
    for (let i = 0; i < n; i++) lat.theta[i] += eps * p[i];
    force();
    for (let i = 0; i < n; i++) p[i] -= 0.5 * eps * f[i];
  }
  let K1 = 0;
  for (let i = 0; i < n; i++) K1 += 0.5 * p[i] * p[i];
  const dH = K1 + lat.action() + pf.action() - H0;
  const accept = Number.isFinite(dH) && (dH <= 0 || rng.uniform() < Math.exp(-dH));
  if (accept) for (let i = 0; i < n; i++) lat.theta[i] -= 2 * Math.PI * Math.round(lat.theta[i] / (2 * Math.PI));
  else lat.theta.set(old);
  return { accept, dH };
}

// ---------- Dense linear algebra, for checks on tiny lattices ----------

/** log |det D| by LU decomposition with partial pivoting of the dense 2V × 2V matrix. */
export function logAbsDet(dirac) {
  const N = 2 * dirac.lat.V;
  const re = new Float64Array(N * N), im = new Float64Array(N * N);
  const e = new Float64Array(4 * dirac.lat.V), col = new Float64Array(4 * dirac.lat.V);
  for (let j = 0; j < N; j++) {
    e.fill(0);
    e[2 * j] = 1;
    dirac.apply(col, e);
    for (let i = 0; i < N; i++) { re[i * N + j] = col[2 * i]; im[i * N + j] = col[2 * i + 1]; }
  }
  let logdet = 0;
  for (let k = 0; k < N; k++) {
    let piv = k, best = -1;
    for (let i = k; i < N; i++) {
      const m = re[i * N + k] ** 2 + im[i * N + k] ** 2;
      if (m > best) { best = m; piv = i; }
    }
    if (best === 0) return -Infinity;
    if (piv !== k) {
      for (let j = 0; j < N; j++) {
        let t = re[k * N + j]; re[k * N + j] = re[piv * N + j]; re[piv * N + j] = t;
        t = im[k * N + j]; im[k * N + j] = im[piv * N + j]; im[piv * N + j] = t;
      }
    }
    const pr = re[k * N + k], pi = im[k * N + k];
    logdet += 0.5 * Math.log(pr * pr + pi * pi);
    const den = pr * pr + pi * pi;
    for (let i = k + 1; i < N; i++) {
      const ar = re[i * N + k], ai = im[i * N + k];
      const fr = (ar * pr + ai * pi) / den, fi = (ai * pr - ar * pi) / den; // a / pivot
      for (let j = k; j < N; j++) {
        const br = re[k * N + j], bi = im[k * N + j];
        re[i * N + j] -= fr * br - fi * bi;
        im[i * N + j] -= fr * bi + fi * br;
      }
    }
  }
  return logdet;
}
