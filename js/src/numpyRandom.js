/**
 * NumPy legacy `RandomState` (MT19937 + randomkit), matching
 * `np.random.seed` / `np.random.randn` / `np.random.randint` used by the
 * Python ONNX path (`np.random.randn` in equivariant_diffusion.py).
 *
 * With the same integer seed, `randn` draws match NumPy bit-for-bit
 * (float64). Unseeded mode is initialised from crypto entropy (not reproducible).
 */

function getCrypto() {
  return globalThis.crypto?.getRandomValues ? globalThis.crypto : null;
}

const N = 624;
const M = 397;
const MATRIX_A = 0x9908b0df;
const UPPER_MASK = 0x80000000;
const LOWER_MASK = 0x7fffffff;

class NumpyRandomState {
  constructor(seed = null) {
    this.key = new Uint32Array(N);
    this.pos = N;
    this.hasGauss = 0;
    this.gaussSpare = 0;
    if (seed == null) this.seedFromEntropy();
    else this.seed(seed);
  }

  /** `np.random.seed(seed)` for a scalar integer in [0, 2**32). */
  seed(seed) {
    let s = seed >>> 0;
    for (let pos = 0; pos < N; pos += 1) {
      this.key[pos] = s;
      // rk_seed: seed = (1812433253 * (seed ^ (seed >> 30)) + pos + 1) & 0xffffffff
      s = (Math.imul(1812433253, s ^ (s >>> 30)) + pos + 1) >>> 0;
    }
    this.pos = N;
    this.hasGauss = 0;
    this.gaussSpare = 0;
  }

  /** Entropy init approximating `rk_randomseed` (non-reproducible). */
  seedFromEntropy() {
    const buf = new Uint32Array(N);
    const crypto = getCrypto();
    if (crypto) {
      crypto.getRandomValues(buf);
    } else {
      // Fallback when Web Crypto is unavailable (some Node 18 test runners).
      let s = (Date.now() ^ (Math.random() * 0x100000000)) >>> 0;
      for (let i = 0; i < N; i += 1) {
        s = (Math.imul(1664525, s) + 1013904223) >>> 0;
        buf[i] = s;
      }
    }
    buf[0] |= 0x80000000;
    this.key.set(buf);
    this.pos = N;
    this.hasGauss = 0;
    this.gaussSpare = 0;
  }

  /** `rk_random` — one tempered uint32. */
  randomUint32() {
    const key = this.key;
    if (this.pos === N) {
      let i = 0;
      for (; i < N - M; i += 1) {
        const y = (key[i] & UPPER_MASK) | (key[i + 1] & LOWER_MASK);
        key[i] = (key[i + M] ^ (y >>> 1) ^ ((y & 1) * MATRIX_A)) >>> 0;
      }
      for (; i < N - 1; i += 1) {
        const y = (key[i] & UPPER_MASK) | (key[i + 1] & LOWER_MASK);
        key[i] = (key[i + (M - N)] ^ (y >>> 1) ^ ((y & 1) * MATRIX_A)) >>> 0;
      }
      const y = (key[N - 1] & UPPER_MASK) | (key[0] & LOWER_MASK);
      key[N - 1] = (key[M - 1] ^ (y >>> 1) ^ ((y & 1) * MATRIX_A)) >>> 0;
      this.pos = 0;
    }
    let y = key[this.pos++];
    y ^= y >>> 11;
    y ^= (y << 7) & 0x9d2c5680;
    y ^= (y << 15) & 0xefc60000;
    y ^= y >>> 18;
    return y >>> 0;
  }

  /** `rk_double` — uniform in [0, 1) with 53 bits. */
  random() {
    const a = this.randomUint32() >>> 5;
    const b = this.randomUint32() >>> 6;
    return (a * 67108864.0 + b) / 9007199254740992.0;
  }

  /** `rk_gauss` — standard normal via Marsaglia polar (NumPy `randn`). */
  standardNormal() {
    if (this.hasGauss) {
      const tmp = this.gaussSpare;
      this.gaussSpare = 0;
      this.hasGauss = 0;
      return tmp;
    }
    let x1;
    let x2;
    let r2;
    do {
      x1 = 2.0 * this.random() - 1.0;
      x2 = 2.0 * this.random() - 1.0;
      r2 = x1 * x1 + x2 * x2;
    } while (r2 >= 1.0 || r2 === 0.0);
    const f = Math.sqrt((-2.0 * Math.log(r2)) / r2);
    this.gaussSpare = f * x1;
    this.hasGauss = 1;
    return f * x2;
  }

  /**
   * `rk_interval(max)` — uniform integer in [0, max] inclusive.
   * Used by `np.random.randint(low, high)` as `low + interval(high-low-1)`.
   */
  interval(max) {
    if (max <= 0) return 0;
    let mask = max >>> 0;
    mask |= mask >>> 1;
    mask |= mask >>> 2;
    mask |= mask >>> 4;
    mask |= mask >>> 8;
    mask |= mask >>> 16;
    let value;
    do {
      value = this.randomUint32() & mask;
    } while (value > max);
    return value;
  }

  /** `np.random.randint(low, high)` — half-open [low, high). */
  randint(low, high) {
    return low + this.interval(high - low - 1);
  }
}

/** Global state — same role as `np.random`'s module-level RandomState. */
let globalState = null;

function getGlobalState() {
  if (!globalState) globalState = new NumpyRandomState(null);
  return globalState;
}

/** `np.random.seed(seed)`. Pass `null` to re-seed from entropy. */
export function seed(seedValue = null) {
  if (!globalState) globalState = new NumpyRandomState(0);
  if (seedValue == null) globalState.seedFromEntropy();
  else globalState.seed(seedValue);
}

/** Expose state for tests / advanced use. */
export function getRandomState() {
  return getGlobalState();
}

export function setRandomState(state) {
  globalState = state;
}

export { NumpyRandomState };

/**
 * `np.random.randn(*shape)` — fills a Float64Array in C order with N(0,1).
 * @param {number[]} shape
 * @returns {{ data: Float64Array, shape: number[] }}
 */
export function numpyRandn(shape) {
  const state = getGlobalState();
  const size = shape.reduce((a, b) => a * b, 1);
  const data = new Float64Array(size);
  for (let i = 0; i < size; i += 1) data[i] = state.standardNormal();
  return { data, shape };
}

/** `np.random.randint(low, high)` for a single int, or fill an Int32Array. */
export function numpyRandint(low, high, size = null) {
  const state = getGlobalState();
  if (size == null) return state.randint(low, high);
  const out = new Int32Array(size);
  for (let i = 0; i < size; i += 1) out[i] = state.randint(low, high);
  return out;
}
