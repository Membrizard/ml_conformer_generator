/**
 * Pipeline plumbing test for MLConformerGenerator using a MOCK ONNX Runtime.
 *
 * Exercises the full path prepareInputs → edmSamples → predictBonds →
 * (standardize) WITHOUT model weights, by injecting a fake `ort` whose sessions
 * return deterministic tensors. This locks the wiring — tensor packing, mask
 * prep, decoding, bond slicing — so a refactor can't silently rewire it.
 *
 * Requires `onnxruntime-node` to be INSTALLED (conformerGenerator.js statically
 * imports it), but does NOT require the NC-ND weights. Once the static import is
 * removed (review §2/§10) this becomes runnable with zero native deps.
 *
 * Run: node --test test/conformer_generator_mock.test.js
 */
import assert from "node:assert/strict";
import { describe, it } from "node:test";
import { createGenerator } from "../src/index.js";
import { DIMENSION } from "../src/constants.js";
import { seed } from "../src/numpyRandom.js";

class MockTensor {
  constructor(type, data, dims) {
    this.type = type;
    this.data = data;
    this.dims = dims;
  }
}

function egnnSession() {
  return {
    outputNames: ["out"],
    async run(feeds) {
      const xh = feeds.xh;
      // Small deterministic net output → finite decoded coordinates.
      return {
        out: {
          data: Float32Array.from({ length: xh.data.length }, () => 0.01),
          dims: xh.dims.slice(),
        },
      };
    },
  };
}

function adjSession() {
  return {
    outputNames: ["out"],
    async run(feeds) {
      const B = feeds.elements.dims[0];
      // (B, D, D, 5): prefer single-bond (channel 1) on the lower triangle so
      // redefineBonds produces edges (needed to assert final-frame bonding).
      const data = new Float32Array(B * DIMENSION * DIMENSION * 5);
      for (let b = 0; b < B; b += 1) {
        for (let i = 0; i < DIMENSION; i += 1) {
          for (let j = 0; j < i; j += 1) {
            const offset = ((b * DIMENSION + i) * DIMENSION + j) * 5;
            data[offset] = 0; // no-bond
            data[offset + 1] = 1; // single bond
          }
        }
      }
      return {
        out: { data, dims: [B, DIMENSION, DIMENSION, 5] },
      };
    },
  };
}

const mockOrt = {
  Tensor: MockTensor,
  InferenceSession: {
    async create(path) {
      return String(path).includes("egnn") ? egnnSession() : adjSession();
    },
  },
};

describe("MLConformerGenerator pipeline (mock ORT, no weights)", () => {
  it("generateConformers returns nSamples molecules with valid MOL blocks", async () => {
    const gen = await createGenerator({
      ort: mockOrt,
      egnnOnnx: "egnn.onnx",
      adjMatSeerOnnx: "adj.onnx",
      diffusionSteps: 5,
    });

    seed(42);
    const nSamples = 4;
    const mols = await gen.generateConformers({
      referenceContext: [89.8693, 210.783, 217.7825],
      nAtoms: 20,
      nSamples,
      variance: 0,
      filterInvalid: false, // skip RDKit validity so the test needs no weights/RDKit
    });

    assert.equal(mols.length, nSamples);
    for (const mol of mols) {
      assert.ok(mol.nAtoms >= 1, "molecule has atoms");
      const mb = mol.toMolBlock();
      assert.match(mb, /V2000/);
      assert.match(mb, /M {2}END$/);
    }
  });

  it("is seed-deterministic across two runs (same mock)", async () => {
    const gen = await createGenerator({
      ort: mockOrt,
      egnnOnnx: "egnn.onnx",
      adjMatSeerOnnx: "adj.onnx",
      diffusionSteps: 5,
    });
    const opts = {
      referenceContext: [89.8693, 210.783, 217.7825],
      nAtoms: 18,
      nSamples: 2,
      variance: 1,
      filterInvalid: false,
    };
    seed(1);
    const a = (await gen.generateConformers(opts)).map((m) => m.toMolBlock());
    seed(1);
    const b = (await gen.generateConformers(opts)).map((m) => m.toMolBlock());
    assert.deepEqual(a, b);
  });

  it("animateGeneration streams one frame per diffusion step", async () => {
    const gen = await createGenerator({
      ort: mockOrt,
      egnnOnnx: "egnn.onnx",
      adjMatSeerOnnx: "adj.onnx",
      diffusionSteps: 5,
    });

    seed(3);
    const frames = [];
    for await (const f of gen.animateGeneration({
      referenceContext: [89.8693, 210.783, 217.7825],
      nAtoms: 20,
      nSamples: 2,
      variance: 0,
    })) {
      frames.push(f);
    }

    assert.equal(frames.length, 5);
    assert.deepEqual(frames.map((f) => f.step), [1, 2, 3, 4, 5]);
    for (const f of frames) {
      assert.equal(f.total, 5);
      assert.equal(f.molecules.length, 2);
      for (const m of f.molecules) assert.ok(m.nAtoms >= 1);
    }
    // Bonds only on the final frame (AdjMatSeer).
    for (const f of frames.slice(0, -1)) {
      for (const m of f.molecules) assert.equal(m.bonds.length, 0);
    }
    for (const m of frames[frames.length - 1].molecules) {
      assert.ok(m.bonds.length > 0);
    }
  });
});
