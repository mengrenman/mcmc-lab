// Hybrid (Hamiltonian) Monte Carlo in two dimensions (book §5.1).
// Treat U(q) = -log p(q) as a potential energy, draw a Gaussian momentum p, follow
// H = U(q) + |p|^2 / 2 with the leapfrog integrator, then accept with min(1, e^{-ΔH}).
// Leapfrog is reversible and area-preserving, which is what makes this simple test exact.

export class HMC2D {
  constructor(target, rng, maxSteps = 200) {
    this.target = target;
    this.rng = rng;
    this.g = new Float64Array(2);
    // Trajectory of the latest proposal: positions and energies after each leapfrog step.
    this.path = {
      x: new Float64Array(maxSteps + 1),
      y: new Float64Array(maxSteps + 1),
      U: new Float64Array(maxSteps + 1),
      K: new Float64Array(maxSteps + 1),
      H: new Float64Array(maxSteps + 1),
      n: 0,
    };
    this.maxSteps = maxSteps;
  }

  reset(x, y) {
    this.x = x;
    this.y = y;
    this.steps = 0;
    this.accepted = 0;
    this.grads = 0; // gradient evaluations
    this.last = null;
  }

  /** One HMC update with step size eps and L leapfrog steps. */
  step(eps, L) {
    const { target, rng, g, path } = this;
    L = Math.min(L, this.maxSteps);
    let qx = this.x, qy = this.y;
    let px = rng.normal(), py = rng.normal();
    const U0 = target.U(qx, qy);
    const K0 = 0.5 * (px * px + py * py);
    path.x[0] = qx; path.y[0] = qy; path.U[0] = U0; path.K[0] = K0; path.H[0] = U0 + K0;

    target.grad(qx, qy, g);
    this.grads++;
    let U1 = U0, K1 = K0;
    for (let i = 1; i <= L; i++) {
      // half kick, drift, half kick: the leapfrog step written with synchronized momenta
      px -= 0.5 * eps * g[0];
      py -= 0.5 * eps * g[1];
      qx += eps * px;
      qy += eps * py;
      target.grad(qx, qy, g);
      this.grads++;
      px -= 0.5 * eps * g[0];
      py -= 0.5 * eps * g[1];
      U1 = target.U(qx, qy);
      K1 = 0.5 * (px * px + py * py);
      path.x[i] = qx; path.y[i] = qy; path.U[i] = U1; path.K[i] = K1; path.H[i] = U1 + K1;
    }
    path.n = L + 1;

    const dH = U1 + K1 - (U0 + K0);
    const u = rng.uniform();
    // A diverging trajectory gives a non-finite dH; treat it as a rejection.
    const accept = Number.isFinite(dH) && (dH <= 0 || u < Math.exp(-dH));
    this.last = { dH, u, prob: Number.isFinite(dH) ? Math.min(1, Math.exp(-dH)) : 0, accepted: accept, fromX: this.x, fromY: this.y };
    if (accept) {
      this.x = qx;
      this.y = qy;
      this.accepted++;
    }
    this.steps++;
    return accept;
  }

  get acceptance() {
    return this.steps ? this.accepted / this.steps : NaN;
  }
}

/**
 * Run L leapfrog steps from (qx, qy, px, py) and return the end state.
 * Used by the checks page to test reversibility.
 */
export function leapfrog(target, qx, qy, px, py, eps, L) {
  const g = new Float64Array(2);
  target.grad(qx, qy, g);
  for (let i = 0; i < L; i++) {
    px -= 0.5 * eps * g[0];
    py -= 0.5 * eps * g[1];
    qx += eps * px;
    qy += eps * py;
    target.grad(qx, qy, g);
    px -= 0.5 * eps * g[0];
    py -= 0.5 * eps * g[1];
  }
  return [qx, qy, px, py];
}
