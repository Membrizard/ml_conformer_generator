/**
 * Slow ONNX generation tests mirroring tests/test_generation_onnx.py.
 *
 * Requires ChemBL ONNX weights next to the package (or via EGNN_ONNX / ADJ_ONNX).
 * Fixed-fragment / IFM paths from the Python suite are not ported yet.
 */
import assert from "node:assert/strict";
import { before, describe, it } from "node:test";
import { createGenerator, seed } from "../src/index.js";
import { contextFromCoordinates } from "../src/mol.js";
import {
  ceyyagMolPath,
  loadMolFile,
  resolveOnnxPaths,
} from "./helpers.js";
import * as ort from "onnxruntime-node";

const DIFFUSION_STEPS = 50;
const N_SAMPLES = 20;

const onnx = resolveOnnxPaths();
const skipReason = onnx.available
  ? false
  : "ONNX weights not found (set EGNN_ONNX / ADJ_ONNX or place models in js/)";

describe("ONNX generation (slow)", { skip: skipReason }, () => {
  /** @type {MLConformerGenerator} */
  let generator;
  /** @type {import("../src/mol.js").Molecule} */
  let ceyyag;
  /** @type {Float32Array} */
  let refContext;

  before(async () => {
    generator = await createGenerator({
      ort,
      egnnOnnx: onnx.egnn,
      adjMatSeerOnnx: onnx.adj,
      diffusionSteps: DIFFUSION_STEPS,
    });
    ceyyag = await loadMolFile(ceyyagMolPath(), { removeHs: true });
    refContext = contextFromCoordinates(ceyyag.positions, ceyyag.nAtoms).context;
  });

  it("basic generation from reference conformer coords", async () => {
    seed(0);
    const samples = await generator.generateConformers({
      referenceConformer: { positions: ceyyag.positions },
      nSamples: N_SAMPLES,
      variance: 1,
      resampleSteps: 0,
      filterInvalid: true,
    });
    const validRate = samples.length / N_SAMPLES;
    assert.ok(
      validRate > 0.3,
      `expected valid rate > 0.3, got ${validRate} (${samples.length}/${N_SAMPLES})`,
    );
  });

  it("basic generation from reference context", async () => {
    seed(0);
    const samples = await generator.generateConformers({
      referenceContext: refContext,
      nAtoms: ceyyag.nAtoms,
      nSamples: N_SAMPLES,
      variance: 1,
      filterInvalid: true,
    });
    const validRate = samples.length / N_SAMPLES;
    assert.ok(
      validRate > 0.3,
      `expected valid rate > 0.3, got ${validRate} (${samples.length}/${N_SAMPLES})`,
    );
  });
});
