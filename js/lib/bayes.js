// Bayesian inference helpers (book §6.1): coin-toss priors and posteriors on a grid, and the
// posterior for the mean and width of a Gaussian with flat priors.

// ---------- Special functions ----------

const LANCZOS = [
  676.5203681218851, -1259.1392167224028, 771.32342877765313, -176.61502916214059,
  12.507343278686905, -0.13857109526572012, 9.9843695780195716e-6, 1.5056327351493116e-7,
];

/** log Γ(x) for x > 0 (Lanczos approximation, about 15 significant digits). */
export function logGamma(x) {
  if (x < 0.5) return Math.log(Math.PI / Math.sin(Math.PI * x)) - logGamma(1 - x);
  x -= 1;
  let a = 0.99999999999980993;
  const t = x + 7.5;
  for (let i = 0; i < 8; i++) a += LANCZOS[i] / (x + i + 1);
  return 0.5 * Math.log(2 * Math.PI) + (x + 0.5) * Math.log(t) - t + Math.log(a);
}

// ---------- Coin toss: p = probability of heads ----------

/**
 * Log prior density (unnormalized) for p in [0, 1].
 * 'bump':       exp(-M (p - p0)^2), trust that p is near p0 (the book's first example)
 * 'flat':       every p equally likely
 * 'suspicious': proportional to (p - 1/2)^2, a fair coin is ruled out
 */
export function coinLogPrior(prior, p) {
  if (p < 0 || p > 1) return -Infinity;
  if (prior.kind === 'flat') return 0;
  if (prior.kind === 'suspicious') return 2 * Math.log(Math.abs(p - 0.5));
  return -prior.M * (p - prior.p0) ** 2;
}

/** Log likelihood of k heads in n tosses, p^k (1 - p)^(n - k), with 0 · log 0 = 0. */
export function coinLogLikelihood(n, k, p) {
  if (p < 0 || p > 1) return -Infinity;
  const a = k === 0 ? 0 : k * Math.log(p);
  const b = n - k === 0 ? 0 : (n - k) * Math.log(1 - p);
  return a + b;
}

function normalizeOnGrid(logf, dp) {
  let max = -Infinity;
  for (const v of logf) if (v > max) max = v;
  const f = new Float64Array(logf.length);
  let total = 0;
  for (let i = 0; i < f.length; i++) {
    f[i] = Number.isFinite(logf[i]) ? Math.exp(logf[i] - max) : 0;
    total += (i === 0 || i === f.length - 1 ? 0.5 : 1) * f[i];
  }
  total *= dp;
  for (let i = 0; i < f.length; i++) f[i] /= total;
  return f;
}

/**
 * Prior, likelihood and posterior on an evenly spaced grid over [0, 1], each normalized to
 * integrate to 1, plus posterior summaries. The grid answer is exact to plotting accuracy,
 * which is what the MCMC lab is checked against.
 */
export function coinPosterior(prior, n, k, points = 4001) {
  const dp = 1 / (points - 1);
  const p = new Float64Array(points);
  const lp = new Float64Array(points), ll = new Float64Array(points), lpost = new Float64Array(points);
  for (let i = 0; i < points; i++) {
    p[i] = i * dp;
    lp[i] = coinLogPrior(prior, p[i]);
    ll[i] = coinLogLikelihood(n, k, p[i]);
    lpost[i] = lp[i] + ll[i];
  }
  const post = normalizeOnGrid(lpost, dp);
  // Cumulative distribution by the trapezoid rule.
  const cdf = new Float64Array(points);
  for (let i = 1; i < points; i++) cdf[i] = cdf[i - 1] + 0.5 * dp * (post[i - 1] + post[i]);
  const moment = (fn) => {
    let t = 0;
    for (let i = 0; i < points; i++) t += (i === 0 || i === points - 1 ? 0.5 : 1) * post[i] * fn(p[i]);
    return t * dp;
  };
  const quantile = (q) => {
    let i = 1;
    while (i < points - 1 && cdf[i] < q) i++;
    const f = cdf[i] > cdf[i - 1] ? (q - cdf[i - 1]) / (cdf[i] - cdf[i - 1]) : 0;
    return p[i - 1] + f * dp;
  };
  const mean = moment((x) => x);
  const second = moment((x) => x * x);
  let mode = 0;
  for (let i = 1; i < points; i++) if (post[i] > post[mode]) mode = i;
  return {
    p,
    prior: normalizeOnGrid(lp, dp),
    likelihood: n > 0 ? normalizeOnGrid(ll, dp) : new Float64Array(points).fill(1),
    posterior: post,
    mean,
    second, // E[p^2]: the probability of two heads in a row
    sd: Math.sqrt(Math.max(0, second - mean * mean)),
    mode: p[mode],
    quantile,
    probAboveHalf: 1 - cdf[Math.round(0.5 / dp)],
  };
}

// ---------- Gaussian with unknown mean and width ----------

/** Sufficient statistics: n, the sample mean and the ML width s = sqrt(mean(x^2) - mean(x)^2). */
export function gaussianStats(xs, n = xs.length) {
  let s1 = 0, s2 = 0;
  for (let i = 0; i < n; i++) {
    s1 += xs[i];
    s2 += xs[i] * xs[i];
  }
  const mean = s1 / n;
  return { n, mean, s: Math.sqrt(Math.max(0, s2 / n - mean * mean)) };
}

/**
 * Minus the log posterior for (mu, sigma) with flat priors on mu and on sigma > 0 (book eq. 6.10):
 * S = n [ ((mu - xbar)^2 + s^2) / (2 sigma^2) + log sigma ].
 */
export function gaussianAction({ n, mean, s }) {
  return (mu, sigma) => (sigma <= 0 ? Infinity : n * (((mu - mean) ** 2 + s * s) / (2 * sigma * sigma) + Math.log(sigma)));
}

/**
 * Exact posterior summaries for the flat-prior Gaussian model (needs n >= 5).
 * mu | data is Student-t with n - 2 degrees of freedom, center xbar and scale s / sqrt(n - 2),
 * so sd(mu) = s / sqrt(n - 4). sigma | data has density proportional to
 * sigma^-(n-1) exp(-A / sigma^2) with A = n s^2 / 2, giving
 * E[sigma^m] = A^(m/2) Γ((n - 2 - m)/2) / Γ((n - 2)/2).
 */
export function gaussianPosteriorExact({ n, mean, s }) {
  const A = (n * s * s) / 2;
  const g0 = logGamma((n - 2) / 2);
  const eSigma = Math.sqrt(A) * Math.exp(logGamma((n - 3) / 2) - g0);
  const eSigma2 = (n * s * s) / (n - 4);
  const nu = n - 2, scale = s / Math.sqrt(n - 2);
  const tNorm = logGamma((nu + 1) / 2) - logGamma(nu / 2) - 0.5 * Math.log(nu * Math.PI) - Math.log(scale);
  const sigmaNorm = Math.log(2) + ((n - 2) / 2) * Math.log(A) - g0;
  return {
    muMean: mean,
    muSd: s / Math.sqrt(n - 4),
    sigmaMean: eSigma,
    sigmaSd: Math.sqrt(Math.max(0, eSigma2 - eSigma * eSigma)),
    /** Marginal posterior density of mu. */
    muPdf(mu) {
      const t = (mu - mean) / scale;
      return Math.exp(tNorm - ((nu + 1) / 2) * Math.log(1 + (t * t) / nu));
    },
    /** Marginal posterior density of sigma. */
    sigmaPdf(sigma) {
      if (sigma <= 0) return 0;
      return Math.exp(sigmaNorm - (n - 1) * Math.log(sigma) - A / (sigma * sigma));
    },
  };
}
