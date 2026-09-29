// Seeded pseudorandom numbers (book ch. 2).
// xoshiro128** generates the bits; splitmix32 expands a single seed into its state.
// Seeding makes every run reproducible, which is what you want when debugging a sampler.

function splitmix32(seed) {
  let a = seed | 0;
  return () => {
    a = (a + 0x9e3779b9) | 0;
    let t = a ^ (a >>> 16);
    t = Math.imul(t, 0x21f0aaad);
    t ^= t >>> 15;
    t = Math.imul(t, 0x735a2d97);
    return (t ^ (t >>> 15)) >>> 0;
  };
}

const rotl = (x, k) => (x << k) | (x >>> (32 - k));

export class RNG {
  constructor(seed = randomSeed()) {
    this.s = new Uint32Array(4);
    this.seed(seed);
  }

  seed(seed) {
    this.initialSeed = seed >>> 0;
    const next = splitmix32(seed);
    for (let i = 0; i < 4; i++) this.s[i] = next();
    this.spare = null;
    return this;
  }

  /** Uniform 32-bit unsigned integer. */
  nextU32() {
    const s = this.s;
    const result = Math.imul(rotl(Math.imul(s[1], 5), 7), 9) >>> 0;
    const t = s[1] << 9;
    s[2] ^= s[0];
    s[3] ^= s[1];
    s[1] ^= s[2];
    s[0] ^= s[3];
    s[2] ^= t;
    s[3] = rotl(s[3], 11);
    return result;
  }

  /** Uniform on [0, 1). */
  uniform() {
    return this.nextU32() / 4294967296;
  }

  /** Uniform integer in {0, ..., n - 1}. */
  int(n) {
    return Math.floor(this.uniform() * n);
  }

  /** Standard normal via the Box–Muller method; each pair of uniforms yields two normals. */
  normal() {
    if (this.spare !== null) {
      const z = this.spare;
      this.spare = null;
      return z;
    }
    const u1 = 1 - this.uniform(); // in (0, 1], so the log is finite
    const u2 = this.uniform();
    const r = Math.sqrt(-2 * Math.log(u1));
    const theta = 2 * Math.PI * u2;
    this.spare = r * Math.sin(theta);
    return r * Math.cos(theta);
  }
}

export function randomSeed() {
  if (globalThis.crypto?.getRandomValues) {
    return globalThis.crypto.getRandomValues(new Uint32Array(1))[0];
  }
  return (Math.random() * 4294967296) >>> 0;
}
