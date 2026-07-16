import * as ort from "onnxruntime-node";
import { MLConformerGenerator } from "./conformerGenerator.js";
import { registerDefaultRdkit } from "./rdkit.js";

// Wire package dependencies on import.
registerDefaultRdkit();

/** @param {object} [options] same as `MLConformerGenerator.create` (`ort` optional) */
export async function create(options = {}) {
  return MLConformerGenerator.create({ ort, ...options });
}

export { ort, MLConformerGenerator };
export { EquivariantDiffusion } from "./equivariantDiffusion.js";
export {
  Molecule,
  contextFromCoordinates,
  isValidMol,
  standardizeMol,
} from "./mol.js";
export { seed, NumpyRandomState } from "./numpyRandom.js";
export {
  setRdkitLoader,
  clearRdkitLoader,
  hasRdkitLoader,
  getRdkit,
  registerDefaultRdkit,
} from "./rdkit.js";
export {
  ATOM_DECODER,
  CONTEXT_NORMS,
  DIMENSION,
  MAX_N_NODES,
  MIN_N_NODES,
} from "./constants.js";
