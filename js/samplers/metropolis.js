// Metropolis algorithm with a uniform proposal (book ch. 4).
// Propose x' = x + c * (2u - 1) and accept with probability min(1, p(x') / p(x)).
// The proposal is symmetric, so this ratio alone gives detailed balance.

export class Metropolis1D {
  constructor(target, rng) {
    this.rng = rng;
    this.setTarget(target);
    // The most recent decision, kept for the step-by-step display.
    this.last = { from: 0, to: 0, ratio: 0, u: 0, accepted: false };
  }

  setTarget(target) {
    this.target = target;
  }

  reset(x0) {
    this.x = x0;
    this.logp = this.target.logp(x0);
    this.steps = 0;
    this.accepted = 0;
    this.hasLast = false;
  }

  step(c) {
    const { rng, target, last } = this;
    const xp = this.x + c * (2 * rng.uniform() - 1);
    const lpp = target.logp(xp);
    const logRatio = lpp - this.logp;
    const u = rng.uniform();
    // Compare in log space; a proposal outside the support has logp = -Infinity and is rejected.
    const accept = logRatio >= 0 || Math.log(u) < logRatio;
    last.from = this.x;
    last.to = xp;
    last.ratio = Math.exp(logRatio);
    last.u = u;
    last.accepted = accept;
    this.hasLast = true;
    if (accept) {
      this.x = xp;
      this.logp = lpp;
      this.accepted++;
    }
    this.steps++;
    return accept;
  }

  get acceptance() {
    return this.steps ? this.accepted / this.steps : NaN;
  }
}

/** Multivariate Metropolis: each coordinate is shifted by an independent uniform in [-c, c]. */
export class Metropolis2D {
  constructor(target, rng) {
    this.target = target;
    this.rng = rng;
  }

  reset(x, y) {
    this.x = x;
    this.y = y;
    this.U = this.target.U(x, y);
    this.steps = 0;
    this.accepted = 0;
    this.evals = 0; // density evaluations
  }

  step(c) {
    const { rng, target } = this;
    const xp = this.x + c * (2 * rng.uniform() - 1);
    const yp = this.y + c * (2 * rng.uniform() - 1);
    const Up = target.U(xp, yp);
    this.evals++;
    const accept = Up <= this.U || rng.uniform() < Math.exp(this.U - Up);
    if (accept) {
      this.x = xp;
      this.y = yp;
      this.U = Up;
      this.accepted++;
    }
    this.steps++;
    return accept;
  }

  get acceptance() {
    return this.steps ? this.accepted / this.steps : NaN;
  }
}
