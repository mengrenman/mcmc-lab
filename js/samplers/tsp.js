// Traveling salesman problem via replica exchange (book §6.3.3).
// A route visits every city once and returns to city 0, which stays in position 0.

export class TspProblem {
  constructor(xs, ys) {
    this.N = xs.length;
    this.xs = Float64Array.from(xs);
    this.ys = Float64Array.from(ys);
    const N = this.N;
    this.d = new Float64Array(N * N);
    for (let i = 0; i < N; i++) {
      for (let j = 0; j < N; j++) this.d[i * N + j] = Math.hypot(xs[i] - xs[j], ys[i] - ys[j]);
    }
  }

  dist(a, b) {
    return this.d[a * this.N + b];
  }

  length(tour) {
    let L = 0;
    for (let i = 0; i < tour.length; i++) L += this.dist(tour[i], tour[(i + 1) % tour.length]);
    return L;
  }

  /** Sum of the edges (tour[e], tour[e + 1]) for the given distinct edge indices. */
  edgeSum(tour, edges) {
    const N = this.N;
    let s = 0;
    for (const e of edges) s += this.dist(tour[e], tour[(e + 1) % N]);
    return s;
  }
}

export function randomCities(n, rng) {
  const xs = [], ys = [];
  for (let i = 0; i < n; i++) {
    xs.push(rng.uniform());
    ys.push(rng.uniform());
  }
  return new TspProblem(xs, ys);
}

const EDGES3 = new Int32Array(3), EDGES4 = new Int32Array(4);

/** Two distinct positions 1 <= i < j <= N - 1 (position 0 is the fixed home city). */
function pickPair(N, rng) {
  let i = 1 + rng.int(N - 1);
  let j = 1 + rng.int(N - 2);
  if (j >= i) j++;
  return i < j ? [i, j] : [j, i];
}

/**
 * Propose a move on `tour` in place and return the change in length.
 * 'swap' exchanges the cities at positions i and j (the book's move).
 * '2opt' reverses the segment i..j, which replaces two edges by two others.
 * Call undoMove with the same arguments to revert a rejected proposal.
 */
export function applyMove(problem, tour, move, i, j) {
  const N = problem.N;
  if (move === '2opt') {
    const a = tour[i - 1], b = tour[i], c = tour[j], e = tour[(j + 1) % N];
    const delta = problem.dist(a, c) + problem.dist(b, e) - problem.dist(a, b) - problem.dist(c, e);
    reverse(tour, i, j);
    return delta;
  }
  // Edges touching positions i and j; neighbors (j = i + 1) share one edge, counted once.
  const edges = j === i + 1 ? EDGES3 : EDGES4;
  edges[0] = i - 1; edges[1] = i; edges[2] = j;
  if (edges === EDGES4) edges[3] = j - 1;
  const before = problem.edgeSum(tour, edges);
  const t = tour[i]; tour[i] = tour[j]; tour[j] = t;
  return problem.edgeSum(tour, edges) - before;
}

export function undoMove(tour, move, i, j) {
  if (move === '2opt') reverse(tour, i, j);
  else { const t = tour[i]; tour[i] = tour[j]; tour[j] = t; }
}

function reverse(tour, i, j) {
  for (; i < j; i++, j--) { const t = tour[i]; tour[i] = tour[j]; tour[j] = t; }
}

/** The book's naive algorithm: accept a move only if it shortens the route. Stops at a local optimum. */
export function greedyDescent(problem, tour, move, rng, patience = 20 * problem.N * problem.N) {
  const N = problem.N;
  let L = problem.length(tour), fails = 0;
  while (fails < patience) {
    const [i, j] = pickPair(N, rng);
    const delta = applyMove(problem, tour, move, i, j);
    if (delta < -1e-12) { L += delta; fails = 0; }
    else { undoMove(tour, move, i, j); fails++; }
  }
  return problem.length(tour);
}

/** Shortest route by checking all (N - 1)! orderings with city 0 fixed. Feasible up to N ≈ 10. */
export function bruteForce(problem) {
  const N = problem.N;
  const rest = Array.from({ length: N - 1 }, (_, i) => i + 1);
  let best = Infinity, bestTour = null;
  const tour = new Int32Array(N);
  const c = new Int32Array(rest.length);
  const check = () => {
    tour[0] = 0;
    for (let k = 0; k < rest.length; k++) tour[k + 1] = rest[k];
    const L = problem.length(tour);
    if (L < best - 1e-12) { best = L; bestTour = Int32Array.from(tour); }
  };
  check();
  // Heap's algorithm over positions 1..N-1.
  for (let i = 1; i < rest.length;) {
    if (c[i] < i) {
      const k = i % 2 === 0 ? 0 : c[i];
      const t = rest[k]; rest[k] = rest[i]; rest[i] = t;
      check();
      c[i]++;
      i = 1;
    } else {
      c[i] = 0;
      i++;
    }
  }
  return { length: best, tour: bestTour };
}

/**
 * Replica exchange for the TSP. Replica m samples routes with weight exp(-beta[m] L), where
 * beta increases with m (m = 0 is the hottest). Each step makes one Metropolis move per
 * replica, then offers swaps between neighbors with min(1, e^{-ΔS}),
 * ΔS = (beta[m+1] - beta[m]) (L_m - L_{m+1}).
 */
export class TspReplicaExchange {
  constructor(problem, betas, rng, move = 'swap') {
    this.problem = problem;
    this.beta = Float64Array.from(betas);
    this.rng = rng;
    this.move = move;
    this.reset();
  }

  reset() {
    const { problem, beta } = this;
    const M = beta.length, N = problem.N;
    this.tours = Array.from({ length: M }, () => Int32Array.from({ length: N }, (_, i) => i));
    const L0 = problem.length(this.tours[0]);
    this.L = new Float64Array(M).fill(L0);
    this.best = Int32Array.from(this.tours[0]);
    this.bestLength = L0;
    this.steps = 0;
    this.moveTried = 0;
    this.moveDone = 0;
    this.swapTried = 0;
    this.swapDone = 0;
  }

  step() {
    const { problem, beta, rng, move, tours, L } = this;
    const M = beta.length, N = problem.N;
    for (let m = 0; m < M; m++) {
      const [i, j] = pickPair(N, rng);
      const delta = applyMove(problem, tours[m], move, i, j);
      this.moveTried++;
      if (delta <= 0 || rng.uniform() < Math.exp(-beta[m] * delta)) {
        L[m] += delta;
        this.moveDone++;
        if (L[m] < this.bestLength - 1e-9) {
          L[m] = problem.length(tours[m]); // refresh to avoid drift from summed deltas
          if (L[m] < this.bestLength - 1e-12) {
            this.bestLength = L[m];
            this.best.set(tours[m]);
          }
        }
      } else {
        undoMove(tours[m], move, i, j);
      }
    }
    for (let m = 0; m < M - 1; m++) {
      const dS = (beta[m + 1] - beta[m]) * (L[m] - L[m + 1]);
      this.swapTried++;
      if (dS <= 0 || rng.uniform() < Math.exp(-dS)) {
        const t = tours[m]; tours[m] = tours[m + 1]; tours[m + 1] = t;
        const l = L[m]; L[m] = L[m + 1]; L[m + 1] = l;
        this.swapDone++;
      }
    }
    this.steps++;
  }

  /** Recompute every replica's length from scratch (removes floating-point drift). */
  refresh() {
    for (let m = 0; m < this.L.length; m++) this.L[m] = this.problem.length(this.tours[m]);
  }
}

export { pickPair };
