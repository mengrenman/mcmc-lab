// Target distributions used across the labs, with the exact answers the samplers are checked against.

const SQRT2PI = Math.sqrt(2 * Math.PI);
const normPdf = (x, mu, s) => Math.exp(-0.5 * ((x - mu) / s) ** 2) / (s * SQRT2PI);

function logAddExp(a, b) {
  const m = Math.max(a, b);
  return m === -Infinity ? -Infinity : m + Math.log(Math.exp(a - m) + Math.exp(b - m));
}

// ---------- 1D targets (Metropolis lab) ----------
// logp is the log of an unnormalized density; pdf is normalized (for the overlay).

const BI_MU = 3, BI_SD = 0.7;

export const targets1D = {
  gaussian: {
    name: 'Gaussian N(0, 1)',
    domain: [-5, 5],
    mean: 0,
    variance: 1,
    start: 0,
    startRange: [-15, 15],
    pdf: (x) => normPdf(x, 0, 1),
    logp: (x) => -0.5 * x * x,
  },
  bimodal: {
    name: 'Two separated modes',
    domain: [-6.5, 6.5],
    mean: 0,
    variance: BI_SD * BI_SD + BI_MU * BI_MU,
    start: 3,
    startRange: [-15, 15],
    pdf: (x) => 0.5 * normPdf(x, -BI_MU, BI_SD) + 0.5 * normPdf(x, BI_MU, BI_SD),
    logp: (x) => logAddExp(-0.5 * ((x - BI_MU) / BI_SD) ** 2, -0.5 * ((x + BI_MU) / BI_SD) ** 2),
  },
  gamma: {
    name: 'Gamma(3, 1), x > 0',
    domain: [-1, 12],
    mean: 3,
    variance: 3,
    start: 2,
    startRange: [0.1, 30],
    pdf: (x) => (x > 0 ? (x * x * Math.exp(-x)) / 2 : 0),
    logp: (x) => (x > 0 ? 2 * Math.log(x) - x : -Infinity),
  },
};

// ---------- 2D targets (HMC & Gibbs lab) ----------
// Each defines the potential U = -log p (up to a constant), its gradient, and, when they
// are standard distributions, the full conditionals Gibbs sampling needs.

export function correlatedGaussian(rho) {
  const k = 1 / (1 - rho * rho);
  const sc = Math.sqrt(1 - rho * rho);
  return {
    id: 'gaussian',
    name: `Correlated Gaussian (ρ = ${rho.toFixed(2)})`,
    domain: [[-3.6, 3.6], [-3.6, 3.6]],
    start: [-2.5, -2.5], // one end of the ridge: the chains must travel its length
    typical: [-1, -0.6], // a point of ordinary probability, for the single-trajectory view
    mean: [0, 0],
    variance: [1, 1],
    U: (x, y) => 0.5 * k * (x * x - 2 * rho * x * y + y * y),
    grad(x, y, out) {
      out[0] = k * (x - rho * y);
      out[1] = k * (y - rho * x);
    },
    condX: (y, rng) => rho * y + sc * rng.normal(),
    condY: (x, rng) => rho * x + sc * rng.normal(),
  };
}

// Banana: x ~ N(0, SX^2) and y | x ~ N(B (x^2 - SX^2), SY^2), so E[x] = E[y] = 0.
const SX = 1.2, SY = 0.4, B = 0.5;

export function banana() {
  return {
    id: 'banana',
    name: 'Banana (curved ridge)',
    domain: [[-3.6, 3.6], [-2, 5.2]],
    start: [-2.8, 3.2], // tip of the left arm
    typical: [-1, -0.2],
    mean: [0, 0],
    variance: [SX * SX, SY * SY + 2 * B * B * SX ** 4],
    U(x, y) {
      const r = y - B * (x * x - SX * SX);
      return (0.5 * x * x) / (SX * SX) + (0.5 * r * r) / (SY * SY);
    },
    grad(x, y, out) {
      const r = y - B * (x * x - SX * SX);
      out[0] = x / (SX * SX) - (2 * B * x * r) / (SY * SY);
      out[1] = r / (SY * SY);
    },
    condX: null, // p(x | y) is not a standard distribution
    condY: (x, rng) => B * (x * x - SX * SX) + SY * rng.normal(),
  };
}

// ---------- Monte Carlo integration lab ----------

/** Integrand f(x) = x^2 exp(-x^2 / 2); its integral over the real line is sqrt(2 pi). */
export const integrand = {
  f: (x) => x * x * Math.exp(-0.5 * x * x),
  exact: SQRT2PI,
  box: 20, // the uniform proposal covers [-box, box]; f is negligible outside
};

/** Variance of one importance weight f/q for a N(0, s^2) proposal (infinite when s <= 1/sqrt 2). */
export function importanceWeightVariance(s) {
  const a = 1 - 1 / (2 * s * s);
  if (a <= 0) return Infinity;
  return s * SQRT2PI * 0.75 * Math.sqrt(Math.PI) * a ** -2.5 - 2 * Math.PI;
}

/** Variance of one weight f/q for the uniform proposal on [-L, L]. */
export function uniformWeightVariance(L = integrand.box) {
  return 2 * L * 0.75 * Math.sqrt(Math.PI) - 2 * Math.PI;
}

export { normPdf, SQRT2PI };
