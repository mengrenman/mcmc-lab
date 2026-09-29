// Two-dimensional Ising model on an L x L periodic lattice (book §6.2).
// E = -J * sum over nearest-neighbor pairs of s_i s_j, with J = 1 and no external field.

export const T_CRITICAL = 2 / Math.log(1 + Math.SQRT2); // ≈ 2.269
export const CALIBRATION_SWEEPS = 20;

export class Ising {
  constructor(L, rng) {
    this.L = L;
    this.N = L * L;
    this.rng = rng;
    this.s = new Int8Array(this.N);
    // Neighbor table: right, left, down, up for each site.
    this.nb = new Int32Array(4 * this.N);
    for (let r = 0; r < L; r++) {
      for (let c = 0; c < L; c++) {
        const i = r * L + c;
        this.nb[4 * i] = r * L + ((c + 1) % L);
        this.nb[4 * i + 1] = r * L + ((c - 1 + L) % L);
        this.nb[4 * i + 2] = ((r + 1) % L) * L + c;
        this.nb[4 * i + 3] = ((r - 1 + L) % L) * L + c;
      }
    }
    this.stack = new Int32Array(this.N);
    this.cluster = new Int32Array(this.N); // sites of the latest Wolff cluster
    this.clusterSize = 0;
    this.showCluster = false; // true right after a Wolff flip, for highlighting
    this.marked = 0; // entries of inCluster currently set
    this.inCluster = new Uint8Array(this.N);
    this.setT(T_CRITICAL);
    this.randomize();
  }

  setT(T) {
    this.T = T;
    const beta = 1 / T;
    // Metropolis acceptance for the two positive energy changes, ΔE = 4 and 8.
    this.w4 = Math.exp(-4 * beta);
    this.w8 = Math.exp(-8 * beta);
    // Heat bath: probability of spin up for local field h in {-4, -2, 0, 2, 4}.
    this.pUp = new Float64Array(9);
    for (let h = -4; h <= 4; h += 2) this.pUp[h + 4] = 1 / (1 + Math.exp(-2 * beta * h));
    // Wolff bond probability.
    this.pAdd = 1 - Math.exp(-2 * beta);
    // The mean cluster size depends on T, so recalibrate the clusters per Wolff sweep.
    this.wolffK = 0;
    this.calib = { sweeps: 0, clusters: 0, flipped: 0 };
  }

  randomize() {
    for (let i = 0; i < this.N; i++) this.s[i] = this.rng.uniform() < 0.5 ? 1 : -1;
    this.showCluster = false;
    this.recompute();
  }

  order() {
    this.s.fill(1);
    this.showCluster = false;
    this.recompute();
  }

  field(i) {
    const s = this.s, nb = this.nb, k = 4 * i;
    return s[nb[k]] + s[nb[k + 1]] + s[nb[k + 2]] + s[nb[k + 3]];
  }

  /** Recompute total energy E and magnetization M from scratch. */
  recompute() {
    const { s, nb, N } = this;
    let E = 0, M = 0;
    for (let i = 0; i < N; i++) {
      M += s[i];
      E -= s[i] * (s[nb[4 * i]] + s[nb[4 * i + 2]]); // each bond once
    }
    this.E = E;
    this.M = M;
  }

  /** One sweep of single-spin Metropolis updates in typewriter order. Returns accepted flips. */
  sweepMetropolis() {
    const { s, N, rng, w4, w8 } = this;
    let acc = 0, dEsum = 0, dM = 0;
    for (let i = 0; i < N; i++) {
      const dE = 2 * s[i] * this.field(i);
      if (dE <= 0 || rng.uniform() < (dE === 4 ? w4 : w8)) {
        s[i] = -s[i];
        dEsum += dE;
        dM += 2 * s[i];
        acc++;
      }
    }
    this.E += dEsum;
    this.M += dM;
    this.showCluster = false;
    return acc;
  }

  /** One sweep of heat-bath updates: each spin is redrawn from its conditional distribution. */
  sweepHeatBath() {
    const { s, N, rng, pUp } = this;
    let changed = 0;
    for (let i = 0; i < N; i++) {
      const h = this.field(i);
      const next = rng.uniform() < pUp[h + 4] ? 1 : -1;
      if (next !== s[i]) {
        s[i] = next;
        changed++;
      }
    }
    this.recompute();
    this.showCluster = false;
    return changed;
  }

  /** Grow one Wolff cluster from a random seed and flip it. Returns the cluster size. */
  wolffStep() {
    const { s, nb, rng, pAdd, stack, cluster, inCluster } = this;
    for (let k = 0; k < this.marked; k++) inCluster[cluster[k]] = 0;
    const seed = rng.int(this.N);
    const spin = s[seed];
    let top = 0, size = 0;
    stack[top++] = seed;
    inCluster[seed] = 1;
    cluster[size++] = seed;
    while (top > 0) {
      const i = stack[--top];
      for (let k = 4 * i; k < 4 * i + 4; k++) {
        const j = nb[k];
        if (!inCluster[j] && s[j] === spin && rng.uniform() < pAdd) {
          inCluster[j] = 1;
          cluster[size++] = j;
          stack[top++] = j;
        }
      }
    }
    for (let k = 0; k < size; k++) s[cluster[k]] = -spin;
    this.marked = size;
    this.clusterSize = size;
    this.showCluster = true;
    this.M -= 2 * spin * size;
    return size;
  }

  /**
   * One Wolff "sweep": a fixed number K of cluster flips, with K ≈ N / (mean cluster size),
   * so on average N spins flip and autocorrelation times compare with the single-spin
   * algorithms at equal work. Returns the number of clusters flipped.
   *
   * K must be fixed. Stopping as soon as N spins have flipped would be a state-dependent
   * stopping rule: the cluster that crosses the line tends to be a large one (the
   * inspection paradox), and measurements taken there are biased. So for the first
   * CALIBRATION_SWEEPS after a temperature change (part of the burn-in), the sweep flips
   * until N spins have flipped and records cluster sizes. After that, K is frozen.
   */
  sweepWolff() {
    let flipped = 0, clusters = 0;
    if (this.wolffK) {
      for (; clusters < this.wolffK; clusters++) flipped += this.wolffStep();
    } else {
      while (flipped < this.N) {
        flipped += this.wolffStep();
        clusters++;
      }
      const c = this.calib;
      c.sweeps++;
      c.clusters += clusters;
      c.flipped += flipped;
      if (c.sweeps >= CALIBRATION_SWEEPS) this.wolffK = Math.max(1, Math.round((this.N * c.clusters) / c.flipped));
    }
    this.recompute();
    this.lastSweepClusters = clusters;
    this.lastSweepFlipped = flipped;
    return clusters;
  }

  sweep(algorithm) {
    if (algorithm === 'wolff') return this.sweepWolff();
    if (algorithm === 'heatbath') return this.sweepHeatBath();
    return this.sweepMetropolis();
  }
}

// ---------- Exact results for the infinite lattice (Onsager, Yang) ----------

/** Spontaneous magnetization per spin: (1 - sinh(2/T)^-4)^(1/8) below T_c, 0 above. */
export function onsagerMagnetization(T) {
  if (T >= T_CRITICAL) return 0;
  const sh = Math.sinh(2 / T);
  return Math.pow(1 - Math.pow(sh, -4), 0.125);
}

// Complete elliptic integral of the first kind via the arithmetic–geometric mean.
function ellipticK(k) {
  let a = 1, b = Math.sqrt(1 - k * k);
  for (let i = 0; i < 40 && Math.abs(a - b) > 1e-15 * a; i++) {
    const an = 0.5 * (a + b);
    b = Math.sqrt(a * b);
    a = an;
  }
  return Math.PI / (2 * a);
}

/** Energy per spin of the infinite lattice. */
export function onsagerEnergy(T) {
  const b2 = 2 / T;
  const t = Math.tanh(b2);
  const c = 2 * t * t - 1;
  if (Math.abs(c) < 1e-12) return -Math.SQRT2;
  const k = (2 * Math.sinh(b2)) / (Math.cosh(b2) ** 2);
  return -(1 / Math.tanh(b2)) * (1 + (2 / Math.PI) * c * ellipticK(Math.min(k, 1 - 1e-16)));
}
