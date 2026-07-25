import { MLConformerGenerator } from "./conformerGenerator.js";

/**
 * Create a conformer generator.
 *
 * @param {object} options  Generation options.
 * @param {object} options.ort  ONNX Runtime namespace (required) — e.g.
 *   `import * as ort from "onnxruntime-node"`, or an `onnxruntime-web` build.
 * @param {string|Uint8Array} [options.egnnOnnx]  EGNN model path or bytes.
 * @param {string|Uint8Array} [options.adjMatSeerOnnx]  AdjMatSeer model path or bytes.
 * @param {string|Uint8Array|null} [options.finetuneCheckpointOnnx]  Optional EDM adapter.
 * @param {number} [options.diffusionSteps]  Reverse-process steps.
 * @returns {Promise<object>}  A ready generator exposing `generateConformers(...)`.
 */
export async function createGenerator(options = {}) {
  return MLConformerGenerator.create(options);
}

export { EquivariantDiffusion } from "./equivariantDiffusion.js";
export {
  Molecule,
  contextFromCoordinates,
  isValidMol,
  standardizeMol,
} from "./mol.js";
export { seed, NumpyRandomState } from "./numpyRandom.js";

export {
  ATOM_DECODER,
  CONTEXT_NORMS,
  DIMENSION,
  MAX_N_NODES,
  MIN_N_NODES,
} from "./constants.js";
