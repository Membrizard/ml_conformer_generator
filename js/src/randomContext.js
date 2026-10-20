import { NumpyRandomState } from "./numpyRandom.js";
import SEEDS from "./randomMoleculeSeeds.js";

/**
 * Pick a random context seed from the bundled list of molecule contexts
 * sampled from ChEMBL. Mirrors Python `utils.common.random_context`.
 *
 * The pick uses its own generator and leaves the global `seed()` state alone.
 * The same `seed` always gives the same context in JS, but seeds are **not**
 * cross-compatible with Python: Python draws with `numpy.random.default_rng`
 * (PCG64), this package with a NumPy-compatible MT19937, so `seed: 42` picks
 * different contexts in the two.
 *
 * @param {number|null} [seed=null] integer in [0, 2**32); null draws from entropy
 * @returns {{ context: Float32Array, nAtoms: number }}
 */
export function randomContext(seed = null) {
  if (
    seed != null &&
    !(Number.isInteger(seed) && seed >= 0 && seed < 2 ** 32)
  ) {
    throw new RangeError(
      `seed must be an integer in [0, 2**32) or null, got ${seed}.`,
    );
  }
  const rng = new NumpyRandomState(seed);
  const { context, nAtoms } = SEEDS[rng.randint(0, SEEDS.length)];
  return { context: Float32Array.from(context), nAtoms };
}
