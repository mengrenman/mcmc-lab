// Statistics for MCMC output: means, autocorrelation, integrated autocorrelation time,
// effective sample size, jackknife errors and histograms (book §4.3).

/** Growable Float64Array for chain histories, with a hard cap on length. */
export class Trace {
  constructor(max = 200000, initial = 4096) {
    this.a = new Float64Array(Math.min(initial, max));
    this.n = 0;
    this.max = max;
  }
  push(v) {
    if (this.n === this.a.length) {
      if (this.n >= this.max) return false;
      const b = new Float64Array(Math.min(this.max, this.a.length * 2));
      b.set(this.a);
      this.a = b;
    }
    this.a[this.n++] = v;
    return true;
  }
  get full() { return this.n >= this.max; }
  get data() { return this.a.subarray(0, this.n); }
  last() { return this.n ? this.a[this.n - 1] : NaN; }
  clear() { this.n = 0; }
}

export function mean(a, s = 0, e = a.length) {
  let t = 0;
  for (let i = s; i < e; i++) t += a[i];
  return t / (e - s);
}

/** Unbiased sample variance. */
export function variance(a, s = 0, e = a.length) {
  const m = mean(a, s, e);
  let t = 0;
  for (let i = s; i < e; i++) {
    const d = a[i] - m;
    t += d * d;
  }
  return t / (e - s - 1);
}

// In-place iterative radix-2 FFT; re and im have a power-of-two length.
function fft(re, im, inverse) {
  const n = re.length;
  for (let i = 1, j = 0; i < n; i++) {
    let bit = n >> 1;
    for (; j & bit; bit >>= 1) j ^= bit;
    j ^= bit;
    if (i < j) {
      let t = re[i]; re[i] = re[j]; re[j] = t;
      t = im[i]; im[i] = im[j]; im[j] = t;
    }
  }
  for (let len = 2; len <= n; len <<= 1) {
    const half = len >> 1;
    const ang = ((inverse ? 2 : -2) * Math.PI) / len;
    const wr0 = Math.cos(ang), wi0 = Math.sin(ang);
    for (let i = 0; i < n; i += len) {
      let wr = 1, wi = 0;
      for (let k = 0; k < half; k++) {
        const a = i + k, b = a + half;
        const br = re[b] * wr - im[b] * wi;
        const bi = re[b] * wi + im[b] * wr;
        re[b] = re[a] - br; im[b] = im[a] - bi;
        re[a] += br; im[a] += bi;
        const t = wr * wr0 - wi * wi0;
        wi = wr * wi0 + wi * wr0;
        wr = t;
      }
    }
  }
  if (inverse) for (let i = 0; i < n; i++) { re[i] /= n; im[i] /= n; }
}

/**
 * Normalized autocorrelation rho(t) for t = 0..n-1, using the usual 1/n estimator
 * of the autocovariance. Returns null when the series is constant (a stuck chain).
 */
export function autocorrelation(a, s = 0, e = a.length) {
  const n = e - s;
  if (n < 2) return null;
  const m = mean(a, s, e);
  let size = 1;
  while (size < 2 * n) size <<= 1;
  const re = new Float64Array(size), im = new Float64Array(size);
  for (let i = 0; i < n; i++) re[i] = a[s + i] - m;
  fft(re, im, false);
  for (let i = 0; i < size; i++) {
    re[i] = re[i] * re[i] + im[i] * im[i];
    im[i] = 0;
  }
  fft(re, im, true);
  const c0 = re[0];
  if (!(c0 > 1e-300)) return null;
  const rho = new Float64Array(n);
  for (let t = 0; t < n; t++) rho[t] = re[t] / c0;
  return rho;
}

/**
 * Integrated autocorrelation time tau = 1 + 2 * sum_{t=1}^{W} rho(t), with Sokal's
 * automatic window: W is the first lag with W >= c * tau(W).
 * With this convention, n correlated samples carry about n / tau independent ones.
 */
export function integratedTime(rho, c = 5) {
  if (!rho) return { tau: Infinity, window: 0, converged: false };
  const maxLag = Math.floor(rho.length / 4);
  let tau = 1;
  for (let t = 1; t <= maxLag; t++) {
    tau += 2 * rho[t];
    if (t >= c * tau) return { tau: Math.max(tau, 1e-9), window: t, converged: true };
  }
  return { tau: Math.max(tau, 1e-9), window: maxLag, converged: false };
}

/**
 * Autocorrelation diagnostics for the samples a[s..e). The ACF is computed on at most
 * the last `cap` samples (tau is a property of the chain, so a long recent stretch
 * estimates it well); ESS then uses the full count.
 */
export function chainDiagnostics(a, s = 0, e = a.length, cap = 65536) {
  const n = e - s;
  if (n < 20) return null;
  const s2 = Math.max(s, e - cap);
  const rho = autocorrelation(a, s2, e);
  const { tau, window, converged } = integratedTime(rho);
  const m = mean(a, s, e);
  const v = variance(a, s, e);
  const ess = Number.isFinite(tau) ? n / Math.max(tau, 1) : 0;
  return {
    n, mean: m, variance: v, rho, tau, window, converged, ess,
    naiveError: Math.sqrt(v / n),
    error: Number.isFinite(tau) ? Math.sqrt((v * Math.max(tau, 1)) / n) : Infinity,
  };
}

/**
 * Jackknife error of the mean with samples grouped into bins of size w.
 * Bin-to-bin correlations vanish once w is much larger than tau, so the estimate rises
 * with w and levels off at the honest error bar.
 */
export function jackknifeMeanError(a, w, s = 0, e = a.length) {
  const k = Math.floor((e - s) / w);
  if (k < 2) return NaN;
  const bins = new Float64Array(k);
  let total = 0;
  for (let b = 0; b < k; b++) {
    let t = 0;
    const off = s + b * w;
    for (let i = 0; i < w; i++) t += a[off + i];
    bins[b] = t;
    total += t;
  }
  const n = k * w;
  let jm = 0;
  const est = new Float64Array(k);
  for (let b = 0; b < k; b++) {
    est[b] = (total - bins[b]) / (n - w); // mean with bin b left out
    jm += est[b];
  }
  jm /= k;
  let v = 0;
  for (let b = 0; b < k; b++) v += (est[b] - jm) ** 2;
  return Math.sqrt(((k - 1) / k) * v);
}

/** Jackknife error for bin sizes 1, 2, 4, ... while at least `minBins` bins remain. */
export function jackknifeCurve(a, s = 0, e = a.length, minBins = 16) {
  const sizes = [], errors = [];
  for (let w = 1; (e - s) / w >= minBins; w *= 2) {
    sizes.push(w);
    errors.push(jackknifeMeanError(a, w, s, e));
  }
  return { sizes, errors };
}

/** Histogram over [lo, hi); returns bin edges and heights (a density if `density`). */
export function histogram(a, lo, hi, bins, s = 0, e = a.length, density = true) {
  const counts = new Float64Array(bins);
  const width = (hi - lo) / bins;
  for (let i = s; i < e; i++) {
    const k = Math.floor((a[i] - lo) / width);
    if (k >= 0 && k < bins) counts[k]++;
  }
  const n = e - s;
  if (density && n > 0) for (let k = 0; k < bins; k++) counts[k] /= n * width;
  const edges = new Float64Array(bins + 1);
  for (let k = 0; k <= bins; k++) edges[k] = lo + k * width;
  return { edges, heights: counts };
}
