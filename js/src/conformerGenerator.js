import {
  ATOM_DECODER,
  CONTEXT_NORMS,
  DIMENSION,
  MAX_N_NODES,
  MIN_N_NODES,
} from "./constants.js";
import { EquivariantDiffusion } from "./equivariantDiffusion.js";
import {
  contextFromCoordinates,
  prepareAdjMatSeerInput,
  prepareEdmInput,
  redefineBonds,
  samplesToMolecules,
  standardizeMol,
} from "./mol.js";
import { clearRdkitLoader, registerDefaultRdkit, setRdkitLoader } from "./rdkit.js";

/**
 * ONNX Runtime conformer generator.
 *
 * Mirrors `MLConformerGeneratorONNX` for the main generation path:
 * context → EDM samples → AdjMatSeer bonds → standardize / validity filter.
 *
 * The ONNX Runtime is supplied by the caller as `ort` — pass `onnxruntime-node`
 * for Node, or an `onnxruntime-web` build for the browser.
 *
 * MMFF geometry optimisation from the Python API is not available in RDKit.js;
 * validity uses RDKit sanitize when an RDKit loader is configured.
 */
export class MLConformerGenerator {
  constructor({
    ort,
    generativeModel,
    adjMatSeer,
    edmAdapter = null,
    dimension = DIMENSION,
    minNNodes = MIN_N_NODES,
    maxNNodes = MAX_N_NODES,
    contextNorms = CONTEXT_NORMS,
    atomDecoder = ATOM_DECODER,
  }) {
    if (!ort?.Tensor) {
      throw new TypeError(
        "MLConformerGenerator requires an ONNX Runtime module (`ort`).",
      );
    }
    this.ort = ort;
    this.generativeModel = generativeModel;
    this.adjMatSeer = adjMatSeer;
    this.edmAdapter = edmAdapter;
    this.dimension = dimension;
    this.minNNodes = minNNodes;
    this.maxNNodes = maxNNodes;
    this.contextNorms = contextNorms;
    this.atomDecoder = atomDecoder;
  }

  /**
   * @param {object} [options]
   * @param {object} options.ort ONNX Runtime namespace (required, e.g. onnxruntime-node)
   * @param {() => Promise<any>} [options.rdkitLoader] custom RDKit loader; defaults to the bundled @rdkit/rdkit
   * @param {string|Uint8Array} [options.egnnOnnx]
   * @param {string|Uint8Array} [options.adjMatSeerOnnx]
   * @param {string|Uint8Array|null} [options.finetuneCheckpointOnnx]
   * @param {number} [options.diffusionSteps]
   */
  static async create({
    ort,
    rdkitLoader,
    egnnOnnx = "./egnn_chembl_15_39.onnx",
    adjMatSeerOnnx = "./adj_mat_seer_chembl_15_39.onnx",
    finetuneCheckpointOnnx = null,
    diffusionSteps = 100,
    dimension = DIMENSION,
    minNNodes = MIN_N_NODES,
    maxNNodes = MAX_N_NODES,
    contextNorms = CONTEXT_NORMS,
    atomDecoder = ATOM_DECODER,
  } = {}) {
    if (!ort?.InferenceSession) {
      throw new TypeError(
        "Invalid `ort` module — expected onnxruntime-node (or compatible) with InferenceSession.",
      );
    }

    if (rdkitLoader) {
      setRdkitLoader(rdkitLoader);
    } else {
      // Force the bundled default even if a prior generator set a custom
      // loader — the RDKit loader is module-global and registerDefaultRdkit
      // alone no-ops when any loader is already set.
      clearRdkitLoader();
      registerDefaultRdkit();
    }

    const loads = [
      EquivariantDiffusion.create(egnnOnnx, ort, { timesteps: diffusionSteps }),
      ort.InferenceSession.create(adjMatSeerOnnx),
    ];
    if (finetuneCheckpointOnnx) {
      loads.push(ort.InferenceSession.create(finetuneCheckpointOnnx));
    }

    const [generativeModel, adjMatSeer, edmAdapter = null] =
      await Promise.all(loads);

    return new MLConformerGenerator({
      ort,
      generativeModel,
      adjMatSeer,
      edmAdapter,
      dimension,
      minNNodes,
      maxNNodes,
      contextNorms,
      atomDecoder,
    });
  }

  /**
   * Resolve generation inputs.
   * Provide either `referenceContext` + `nAtoms`, or a `referenceConformer`
   * with `{ positions, atomicNumbers? }` / flat positions.
   */
  prepareInputs({
    referenceConformer = null,
    referenceContext = null,
    nAtoms = null,
  } = {}) {
    if (referenceConformer) {
      const positions = flattenPositions(referenceConformer);
      const n = positions.length / 3;
      const { context } = contextFromCoordinates(positions, n);
      return { referenceContext: context, nAtoms: nAtoms ?? n };
    }
    if (referenceContext == null || nAtoms == null) {
      throw new Error(
        "Provide referenceConformer, or both referenceContext and nAtoms.",
      );
    }
    return {
      referenceContext: Float32Array.from(referenceContext),
      nAtoms,
    };
  }

  async edmSamples({
    referenceContext,
    nSamples = 100,
    maxNNodes = 32,
    minNNodes = 25,
    resampleSteps = 0,
  }) {
    minNNodes = Math.max(minNNodes, this.minNNodes);
    maxNNodes = Math.min(maxNNodes, this.maxNNodes);
    if (minNNodes > maxNNodes) {
      throw new RangeError(
        `Invalid node range [${minNNodes}, ${maxNNodes}] after clamping.`,
      );
    }

    const { nodeMask, edgeMask, batchContext } = prepareEdmInput({
      nSamples,
      referenceContext,
      contextNorms: this.contextNorms,
      minNNodes,
      maxNNodes,
    });

    let { x, h } = await this.generativeModel.sample(
      nodeMask,
      edgeMask,
      batchContext,
      resampleSteps,
    );

    if (this.edmAdapter) {
      ({ x, h } = await this.applyEdmAdapter(x, h, nodeMask, edgeMask));
    }

    return samplesToMolecules(x, h, nodeMask, this.atomDecoder);
  }

  /** Optional RL fine-tune adapter: (x, h, masks) → (x_new, h_new). */
  async applyEdmAdapter(x, h, nodeMask, edgeMask) {
    const { Tensor } = this.ort;
    const feeds = {
      x: new Tensor("float32", Float32Array.from(x.data), x.shape),
      h: new Tensor("float32", Float32Array.from(h.data), h.shape),
      node_mask: new Tensor(
        "float32",
        Float32Array.from(nodeMask.data),
        nodeMask.shape,
      ),
      edge_mask: new Tensor(
        "float32",
        Float32Array.from(edgeMask.data),
        edgeMask.shape,
      ),
    };
    const out = await this.edmAdapter.run(feeds);
    return {
      x: {
        data: Float32Array.from(out.x_new.data, Number),
        shape: out.x_new.dims.slice(),
      },
      h: {
        data: Float32Array.from(out.h_new.data, Number),
        shape: out.h_new.dims.slice(),
      },
    };
  }

  async predictBonds(edmSamples) {
    const { elements, distMat, adjMat, mols } = await prepareAdjMatSeerInput(
      edmSamples,
      this.dimension,
    );

    const { Tensor } = this.ort;
    const feeds = {
      elements: new Tensor("int64", elements, [mols.length, this.dimension]),
      dist_mat: new Tensor("float32", distMat, [
        mols.length,
        this.dimension,
        this.dimension,
      ]),
      adj_mat: new Tensor("float32", adjMat, [
        mols.length,
        this.dimension,
        this.dimension,
      ]),
    };

    const results = await this.adjMatSeer.run(feeds);
    const out = results[this.adjMatSeer.outputNames[0]];
    // shape (B, D, D, 5)
    const bonded = [];
    const stride = this.dimension * this.dimension * 5;
    for (let i = 0; i < mols.length; i += 1) {
      const slice = out.data.subarray(i * stride, (i + 1) * stride);
      bonded.push(redefineBonds(mols[i], slice, this.dimension));
    }
    return bonded;
  }

  async generateConformers({
    referenceConformer = null,
    referenceContext = null,
    nAtoms = null,
    nSamples = 10,
    variance = 2,
    resampleSteps = 0,
    keepLargestFragment = true,
    filterInvalid = true,
  } = {}) {
    const prepared = this.prepareInputs({
      referenceConformer,
      referenceContext,
      nAtoms,
    });

    const samples = await this.edmSamples({
      referenceContext: prepared.referenceContext,
      nSamples,
      minNNodes: prepared.nAtoms - variance,
      maxNNodes: prepared.nAtoms + variance,
      resampleSteps,
    });

    const bonded = await this.predictBonds(samples);

    // Python ONNX: standardize_mol each sample; drop failures (None).
    if (!filterInvalid) {
      return keepLargestFragment
        ? bonded.map((m) => m.largestFragment())
        : bonded;
    }

    const valid = [];
    for (const mol of bonded) {
      const std = await standardizeMol(mol, { keepLargestFragment });
      if (std) valid.push(std);
    }
    return valid;
  }

  /**
   * Streaming variant of `generateConformers`: yields one frame per denoising
   * step, each frame being the batch decoded to molecules at that step. The
   * optional EDM (RL fine-tune) adapter is not applied per frame.
   *
   * Bond prediction is controlled by `predictBonds`:
   *   - `"last"` (default) — AdjMatSeer on the final frame only. Intermediate
   *     frames are atom clouds, which is what a trajectory view wants, and the
   *     whole animation costs one extra inference.
   *   - `"always"` — AdjMatSeer on every frame, so a viewer can show real
   *     connectivity forming instead of guessing bonds by distance. Costs one
   *     AdjMatSeer call per denoising step.
   *   - `"never"` — atom clouds only, no bonds at any point.
   *
   * By default the final frame carries bonds but is NOT put through the
   * `standardizeMol` / largest-fragment / validity pass that
   * `generateConformers` applies, so it may be multi-fragment or fail RDKit
   * sanitize. Pass `filterInvalid: true` to run that pass on the final frame
   * and get a frame equivalent to a `generateConformers` result — note it can
   * drop molecules, so the final frame may be shorter than the frames before
   * it (possibly empty). Filtering is only ever applied to the final frame.
   *
   * @param {"last"|"never"|"always"} [predictBonds="last"]
   * @param {boolean} [filterInvalid=false] standardize + validity-filter the final frame
   * @param {boolean} [keepLargestFragment=true] only meaningful with `filterInvalid`
   * @returns {AsyncGenerator<{ step: number, total: number, molecules: object[] }>}
   */
  async *animateGeneration({
    referenceConformer = null,
    referenceContext = null,
    nAtoms = null,
    nSamples = 1,
    variance = 2,
    resampleSteps = 0,
    predictBonds = "last",
    filterInvalid = false,
    keepLargestFragment = true,
  } = {}) {
    if (!["last", "never", "always"].includes(predictBonds)) {
      throw new TypeError(
        `Invalid predictBonds "${predictBonds}" — expected "last", "never" or "always".`,
      );
    }
    if (filterInvalid && predictBonds === "never") {
      throw new TypeError(
        'filterInvalid needs bonds on the final frame — use predictBonds "last" or "always".',
      );
    }

    const prepared = this.prepareInputs({
      referenceConformer,
      referenceContext,
      nAtoms,
    });

    const minNNodes = Math.max(prepared.nAtoms - variance, this.minNNodes);
    const maxNNodes = Math.min(prepared.nAtoms + variance, this.maxNNodes);
    if (minNNodes > maxNNodes) {
      throw new RangeError(
        `Invalid node range [${minNNodes}, ${maxNNodes}] after clamping.`,
      );
    }

    const { nodeMask, edgeMask, batchContext } = prepareEdmInput({
      nSamples,
      referenceContext: prepared.referenceContext,
      contextNorms: this.contextNorms,
      minNNodes,
      maxNNodes,
    });

    for await (const frame of this.generativeModel.animate(
      nodeMask,
      edgeMask,
      batchContext,
      resampleSteps,
    )) {
      const isFinal = frame.step === frame.total;
      let mols = samplesToMolecules(frame.x, frame.h, nodeMask, this.atomDecoder);

      if (predictBonds === "always" || (predictBonds === "last" && isFinal)) {
        mols = await this.predictBonds(mols);
      }

      // Final frame only: mirror the generateConformers post-processing so the
      // last frame can be used as the result rather than just as a picture.
      if (isFinal && filterInvalid) {
        const valid = [];
        for (const mol of mols) {
          const std = await standardizeMol(mol, { keepLargestFragment });
          if (std) valid.push(std);
        }
        mols = valid;
      }

      yield { step: frame.step, total: frame.total, molecules: mols };
    }
  }
}

function flattenPositions(ref) {
  if (ref instanceof Float32Array) return ref;
  if (Array.isArray(ref) && typeof ref[0] === "number") {
    return Float32Array.from(ref);
  }
  if (ref.positions) {
    const p = ref.positions;
    if (p instanceof Float32Array) return p;
    if (Array.isArray(p[0])) {
      const out = new Float32Array(p.length * 3);
      for (let i = 0; i < p.length; i += 1) {
        out[i * 3] = p[i][0];
        out[i * 3 + 1] = p[i][1];
        out[i * 3 + 2] = p[i][2];
      }
      return out;
    }
    return Float32Array.from(p);
  }
  throw new TypeError("referenceConformer must provide positions.");
}
