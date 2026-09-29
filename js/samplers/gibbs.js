// Gibbs sampling (heat bath) in two dimensions (book §5.2).
// Draw x from p(x | y), then y from p(y | x). Every draw is accepted, but the chain can
// only move parallel to the axes, so strong correlations make it crawl.

export class Gibbs2D {
  constructor(target, rng) {
    this.target = target;
    this.rng = rng;
  }

  get available() {
    return Boolean(this.target.condX && this.target.condY);
  }

  reset(x, y) {
    this.x = x;
    this.y = y;
    this.steps = 0;
    this.draws = 0; // conditional draws
    this.midX = x; // after the x-update, before the y-update (for drawing the zigzag)
  }

  step() {
    const { target, rng } = this;
    this.x = target.condX(this.y, rng);
    this.midX = this.x;
    this.midY = this.y;
    this.y = target.condY(this.x, rng);
    this.draws += 2;
    this.steps++;
  }
}
