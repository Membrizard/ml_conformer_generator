import { MLConformerGenerator } from "./conformerGenerator.js";

/**
 * Create a conformer generator.
 *
 * @param {object} options  Generation options.
 * @param {object} options.ort  ONNX Runtime namespace (required) — e.g.
 *   `import * as ort from "onnxruntime-node"`, or an `onnxruntime-web` build.
 * @param {string|Uint8Array} [options.egnnOnnx]  EGNN model: path, bare file
 *   name (resolved via the weights manager, downloaded on first use) or bytes.
 *   Defaults to `"egnn_chembl_15_39.onnx"`.
 * @param {string|Uint8Array} [options.adjMatSeerOnnx]  AdjMatSeer model, same
 *   forms. Defaults to `"adj_mat_seer_chembl_15_39.onnx"`.
 * @param {string|Uint8Array|null} [options.finetuneCheckpointOnnx]  Optional EDM adapter (path or bytes).
 * @param {number} [options.diffusionSteps]  Reverse-process steps.
 * @param {WeightsManager} [options.weightsManager]  Custom weights manager.
 * @param {string} [options.cacheDir]  Weights cache directory (Node).
 * @param {string} [options.baseUrl]  Download location for bare file names.
 * @param {string} [options.cacheName]  Cache API cache name (browser).
 * @param {Function} [options.onProgress]  Weights download progress callback,
 *   called with `{ file, loaded, total }`.
 * @returns {Promise<object>}  A ready generator exposing `generateConformers(...)`,
 *   `random(...)`, `listWeights()` and `clearCache()`.
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
export { WeightsManager } from "./weightsManager.js";

export {
  ATOM_DECODER,
  CONTEXT_NORMS,
  DIMENSION,
  HF_REPO,
  MAX_N_NODES,
  MIN_N_NODES,
} from "./constants.js";
