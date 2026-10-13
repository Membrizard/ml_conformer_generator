/**
 * Pipeline plumbing test for MLConformerGenerator using a MOCK ONNX Runtime.
 *
 * Exercises the full path prepareInputs → edmSamples → predictBonds →
 * (standardize) WITHOUT model weights, by injecting a fake `ort` whose sessions
 * return deterministic tensors. This locks the wiring — tensor packing, mask
 * prep, decoding, bond slicing — so a refactor can't silently rewire it.
 *
 * Needs neither `onnxruntime-node` nor the model weights. Weights resolution
 * goes through a pass-through stub manager, so nothing touches the network.
 *
 * Run: node --test test/conformer_generator_mock.test.js
 */
import assert from "node:assert/strict";
import { mkdtemp, readdir, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, beforeEach, describe, it } from "node:test";
import { createGenerator, randomContext, WeightsManager } from "../src/index.js";
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

/** Pass-through weights manager: hands the name straight to the mock ORT. */
const stubWeightsManager = {
  async resolve(nameOrPath) {
    return nameOrPath;
  },
};

const mockOrt = {
  Tensor: MockTensor,
  InferenceSession: {
    async create(path) {
      return String(path).includes("egnn") ? egnnSession() : adjSession();
    },
  },
};

describe("MLConformerGenerator pipeline (mock ORT, no weights)", () => {
  /** Generator wired to the mock ORT with a short 5-step trajectory. */
  async function animGenerator() {
    return createGenerator({
      ort: mockOrt,
      egnnOnnx: "egnn.onnx",
      adjMatSeerOnnx: "adj.onnx",
      diffusionSteps: 5,
      weightsManager: stubWeightsManager,
    });
  }

  /** Drain animateGeneration into an array, with a fixed reference context. */
  async function collectFrames(gen, opts = {}) {
    const frames = [];
    for await (const f of gen.animateGeneration({
      referenceContext: [89.8693, 210.783, 217.7825],
      nAtoms: 20,
      nSamples: 2,
      variance: 0,
      ...opts,
    })) {
      frames.push(f);
    }
    return frames;
  }

  it("generateConformers returns nSamples molecules with valid MOL blocks", async () => {
    const gen = await createGenerator({
      ort: mockOrt,
      egnnOnnx: "egnn.onnx",
      adjMatSeerOnnx: "adj.onnx",
      diffusionSteps: 5,
      weightsManager: stubWeightsManager,
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
      weightsManager: stubWeightsManager,
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
      weightsManager: stubWeightsManager,
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
    // Bonds only on the final frame (AdjMatSeer), i.e. predictBonds: "last".
    for (const f of frames.slice(0, -1)) {
      for (const m of f.molecules) assert.equal(m.bonds.length, 0);
    }
    for (const m of frames[frames.length - 1].molecules) {
      assert.ok(m.bonds.length > 0);
    }
  });

  it('animateGeneration predictBonds: "always" bonds every frame', async () => {
    const gen = await animGenerator();
    const frames = await collectFrames(gen, { predictBonds: "always" });

    assert.equal(frames.length, 5);
    for (const f of frames) {
      for (const m of f.molecules) assert.ok(m.bonds.length > 0);
    }
  });

  it('animateGeneration predictBonds: "never" bonds no frame', async () => {
    const gen = await animGenerator();
    const frames = await collectFrames(gen, { predictBonds: "never" });

    assert.equal(frames.length, 5);
    for (const f of frames) {
      for (const m of f.molecules) assert.equal(m.bonds.length, 0);
    }
  });

  it("animateGeneration rejects an unknown predictBonds mode", async () => {
    const gen = await animGenerator();
    await assert.rejects(
      () => collectFrames(gen, { predictBonds: "sometimes" }),
      /Invalid predictBonds/,
    );
  });

  it('animateGeneration rejects filterInvalid with predictBonds: "never"', async () => {
    const gen = await animGenerator();
    await assert.rejects(
      () => collectFrames(gen, { predictBonds: "never", filterInvalid: true }),
      /filterInvalid needs bonds/,
    );
  });

  it("animateGeneration filterInvalid standardizes only the final frame", async () => {
    const gen = await animGenerator();

    seed(3);
    const unfiltered = await collectFrames(gen, {});
    seed(3);
    const filtered = await collectFrames(gen, { filterInvalid: true });

    // Same trajectory length and identical intermediate frames — filtering is
    // confined to the last frame and must not perturb the ones before it.
    assert.equal(filtered.length, unfiltered.length);
    for (let i = 0; i < filtered.length - 1; i += 1) {
      assert.equal(filtered[i].molecules.length, 2);
      assert.deepEqual(
        filtered[i].molecules.map((m) => m.nAtoms),
        unfiltered[i].molecules.map((m) => m.nAtoms),
      );
    }

    // Final frame: a subset of the unfiltered batch, every survivor a single
    // connected fragment (keepLargestFragment defaults to true).
    const finalFiltered = filtered[filtered.length - 1].molecules;
    assert.equal(unfiltered[unfiltered.length - 1].molecules.length, 2);
    assert.ok(finalFiltered.length <= 2);
    for (const m of finalFiltered) {
      assert.equal(m.fragmentCount(), 1);
    }
  });
});

describe("MLConformerGenerator weights resolution (mock ORT, stubbed fetch)", () => {
  const HF_RESOLVE =
    "https://huggingface.co/Membrizard/ml_conformer_generator/resolve/main";
  const realFetch = globalThis.fetch;
  const realEnvCache = process.env.MLCONFGEN_CACHE;
  const originalCwd = process.cwd();

  let dir;
  /** Empty working directory, so weights next to the package can't win over the cache. */
  let cwd;
  let fetched;
  /** Models handed to `InferenceSession.create`, in call order. */
  let created;

  const recordingOrt = {
    Tensor: MockTensor,
    InferenceSession: {
      async create(model) {
        created.push(model);
        return String(model).includes("egnn") ? egnnSession() : adjSession();
      },
    },
  };

  beforeEach(async () => {
    dir = await mkdtemp(join(tmpdir(), "mlconfgen-gen-"));
    cwd = await mkdtemp(join(tmpdir(), "mlconfgen-cwd-"));
    process.chdir(cwd);
    process.env.MLCONFGEN_CACHE = dir;
    fetched = [];
    created = [];
    globalThis.fetch = async (url) => {
      fetched.push(url);
      if (url.includes("/api/models/")) {
        return new Response(
          JSON.stringify({
            siblings: [
              { rfilename: "egnn_chembl_15_39.onnx" },
              { rfilename: "edm_moi_chembl_15_39.pt" },
            ],
          }),
        );
      }
      return new Response(new Uint8Array([1, 2]), {
        headers: { "content-length": "2" },
      });
    };
  });

  afterEach(async () => {
    process.chdir(originalCwd);
    globalThis.fetch = realFetch;
    if (realEnvCache === undefined) delete process.env.MLCONFGEN_CACHE;
    else process.env.MLCONFGEN_CACHE = realEnvCache;
    await rm(dir, { recursive: true, force: true });
    await rm(cwd, { recursive: true, force: true });
  });

  it("create() with no options builds a manager with defaults", async () => {
    const gen = await createGenerator({ ort: recordingOrt });
    const wm = gen.weightsManager;

    assert.ok(wm instanceof WeightsManager);
    assert.equal(wm.baseUrl, HF_RESOLVE);
    assert.equal(wm.cacheName, "mlconfgen-weights");
    assert.equal(wm.cacheDir, null);
    assert.equal(wm.onProgress, null);
    assert.equal(await wm.getCacheDir(), dir);
  });

  it("resolves the default bare names into the cache", async () => {
    await createGenerator({ ort: recordingOrt });

    assert.deepEqual(created, [
      join(dir, "egnn_chembl_15_39.onnx"),
      join(dir, "adj_mat_seer_chembl_15_39.onnx"),
    ]);
    assert.deepEqual(fetched, [
      `${HF_RESOLVE}/egnn_chembl_15_39.onnx`,
      `${HF_RESOLVE}/adj_mat_seer_chembl_15_39.onnx`,
    ]);
  });

  it("reuses the cache on a second create()", async () => {
    await createGenerator({ ort: recordingOrt });
    await createGenerator({ ort: recordingOrt });
    assert.equal(fetched.length, 2);
  });

  it("forwards cacheDir, baseUrl, cacheName and onProgress to the default manager", async () => {
    const own = join(dir, "own");
    const progress = [];
    const gen = await createGenerator({
      ort: recordingOrt,
      cacheDir: own,
      baseUrl: "https://mirror.example/w",
      cacheName: "c2",
      onProgress: (p) => progress.push(p.file),
    });

    assert.equal(gen.weightsManager.cacheName, "c2");
    assert.equal(created[0], join(own, "egnn_chembl_15_39.onnx"));
    assert.deepEqual(fetched.sort(), [
      "https://mirror.example/w/adj_mat_seer_chembl_15_39.onnx",
      "https://mirror.example/w/egnn_chembl_15_39.onnx",
    ]);
    assert.deepEqual([...new Set(progress)].sort(), [
      "adj_mat_seer_chembl_15_39.onnx",
      "egnn_chembl_15_39.onnx",
    ]);
  });

  it("passes bytes through and does not resolve the fine-tune checkpoint", async () => {
    const bytes = new Uint8Array([5]);
    await createGenerator({
      ort: recordingOrt,
      egnnOnnx: bytes,
      adjMatSeerOnnx: bytes,
      finetuneCheckpointOnnx: "my_finetune.onnx",
    });
    assert.deepEqual(created, [bytes, bytes, "my_finetune.onnx"]);
    assert.equal(fetched.length, 0);
  });

  it("keeps an injected weights manager", async () => {
    const own = new WeightsManager({ cacheDir: dir });
    const bytes = new Uint8Array([5]);
    const gen = await createGenerator({
      ort: recordingOrt,
      weightsManager: own,
      egnnOnnx: bytes,
      adjMatSeerOnnx: bytes,
    });
    assert.equal(gen.weightsManager, own);
  });

  it("rejects a missing weights path without starting other downloads", async () => {
    await assert.rejects(
      createGenerator({ ort: recordingOrt, egnnOnnx: "./nope/egnn.onnx" }),
      /Weights file not found/,
    );
    assert.equal(fetched.length, 0);
    assert.deepEqual(await readdir(dir), []);
  });

  it("listWeights lists .onnx only, and clearCache empties the cache", async () => {
    const gen = await createGenerator({ ort: recordingOrt });

    const listed = await gen.listWeights();
    assert.deepEqual(listed.remote, ["egnn_chembl_15_39.onnx"]);
    assert.deepEqual(listed.local.sort(), [
      "adj_mat_seer_chembl_15_39.onnx",
      "egnn_chembl_15_39.onnx",
    ]);

    await gen.clearCache();
    assert.deepEqual(await readdir(dir), []);
  });
});

describe("MLConformerGenerator with an injected weights manager", () => {
  it("resolves the default names through it and delegates listWeights/clearCache", async () => {
    const calls = [];
    const manager = {
      async resolve(name) {
        calls.push(["resolve", name]);
        return name;
      },
      async listAvailableWeights(options) {
        calls.push(["list", [...options.suffixes]]);
        return { remote: ["r.onnx"], local: [] };
      },
      async clearCache() {
        calls.push(["clear"]);
      },
    };
    const gen = await createGenerator({ ort: mockOrt, weightsManager: manager });

    assert.deepEqual(await gen.listWeights(), { remote: ["r.onnx"], local: [] });
    await gen.clearCache();
    assert.deepEqual(calls, [
      ["resolve", "egnn_chembl_15_39.onnx"],
      ["resolve", "adj_mat_seer_chembl_15_39.onnx"],
      ["list", [".onnx"]],
      ["clear"],
    ]);
  });
});

describe("MLConformerGenerator.random (mock ORT, no weights)", () => {
  async function generator() {
    return createGenerator({
      ort: mockOrt,
      egnnOnnx: "egnn.onnx",
      adjMatSeerOnnx: "adj.onnx",
      diffusionSteps: 3,
      weightsManager: stubWeightsManager,
    });
  }

  it("returns size molecules, reproducibly with seed() and seed", async () => {
    const gen = await generator();
    const opts = { size: 3, seed: 7, filterInvalid: false };

    seed(1);
    const a = await gen.random(opts);
    seed(1);
    const b = await gen.random(opts);

    assert.equal(a.length, 3);
    assert.deepEqual(
      a.map((m) => m.toMolBlock()),
      b.map((m) => m.toMolBlock()),
    );
  });

  it("generates from the picked context with variance 0 and forwards options", async () => {
    const gen = await generator();
    let seen;
    gen.generateConformers = async (options) => {
      seen = options;
      return [];
    };

    await gen.random({
      seed: 3,
      referenceConformer: { positions: [0, 0, 0] },
      variance: 5,
      nSamples: 99,
      resampleSteps: 2,
      keepLargestFragment: false,
    });

    const expected = randomContext(3);
    assert.equal(seen.referenceConformer, null);
    assert.deepEqual(seen.referenceContext, expected.context);
    assert.equal(seen.nAtoms, expected.nAtoms);
    assert.equal(seen.variance, 0);
    assert.equal(seen.nSamples, 1);
    assert.equal(seen.resampleSteps, 2);
    assert.equal(seen.keepLargestFragment, false);
  });

  it("rejects an invalid seed", async () => {
    const gen = await generator();
    await assert.rejects(gen.random({ seed: -3 }), RangeError);
  });
});
