import assert from "node:assert/strict";
import { describe, it } from "node:test";
import { EquivariantDiffusion } from "../../src/equivariantDiffusion.js";
import { seed } from "../../src/numpyRandom.js";

// Characterization tests for the EDM sampler math and control flow.
// A mock ONNX Runtime + fake EGNN session lets us exercise the full reverse
// process WITHOUT native onnxruntime-node and WITHOUT model weights. Goldens
// are captured from the current implementation.
const TOL = 1e-9;
function close(actual, expected, tol = TOL) {
  assert.equal(actual.length, expected.length, "length");
  for (let i = 0; i < expected.length; i += 1) {
    assert.ok(
      Math.abs(actual[i] - expected[i]) <= Math.max(tol, tol * Math.abs(expected[i])),
      `index ${i}: got ${actual[i]}, want ${expected[i]}`,
    );
  }
}

class MockTensor {
  constructor(type, data, dims) {
    this.type = type;
    this.data = data;
    this.dims = dims;
  }
}
const ort = { Tensor: MockTensor };
// Deterministic fake EGNN: net output depends only on tensor position.
const session = {
  outputNames: ["out"],
  async run(feeds) {
    const xh = feeds.xh;
    const n = xh.data.length;
    return {
      out: { data: Float32Array.from({ length: n }, (_, i) => ((i % 7) - 3) * 0.01), dims: xh.dims.slice() },
    };
  },
};

function makeEd() {
  return new EquivariantDiffusion(session, ort, { timesteps: 3, noisePrecision: 1e-5 });
}

describe("EquivariantDiffusion: constructor guards", () => {
  it("throws without an ORT module", () => {
    assert.throws(() => new EquivariantDiffusion(session, {}, {}), TypeError);
  });
});

describe("EquivariantDiffusion: schedule-derived coefficients", () => {
  const ed = makeEd();
  const target = { data: new Float64Array(11), shape: [1, 1, 11] };
  const gammaT = ed.gamma.call({ data: Float64Array.from([2 / 3]), shape: [1, 1] });
  const gammaS = ed.gamma.call({ data: Float64Array.from([1 / 3]), shape: [1, 1] });

  it("sigma = sqrt(sigmoid(gamma))", () => {
    close([...ed.sigma(gammaT, target).data].slice(0, 2), [0.8314771187040743, 0.8314771187040743]);
  });
  it("alpha = sqrt(sigmoid(-gamma))", () => {
    close([...ed.alpha(gammaT, target).data].slice(0, 2), [0.5555589987315216, 0.5555589987315216]);
  });
  it("snr = exp(-gamma)", () => {
    close([...ed.snr(gammaT).data], [0.4464365755642466]);
  });
  it("sigmaAndAlphaTGivenS", () => {
    const sat = ed.sigmaAndAlphaTGivenS(gammaT, gammaS, target);
    close([...sat.sigma2TGivenS.data].slice(0, 1), [0.609367286775727]);
    close([...sat.alphaTGivenS.data].slice(0, 1), [0.6250061705489578]);
    close([...sat.sigmaTGivenS.data].slice(0, 1), [0.7806198093667154]);
  });
});

describe("EquivariantDiffusion: unnormalize", () => {
  it("scales coords by 1, features by 9, zeros padded nodes", () => {
    const ed = makeEd();
    const x = { data: Float64Array.from([1, 2, 3, 4, 5, 6]), shape: [1, 2, 3] };
    const h = { data: Float64Array.from([1, 1, 2, 2]), shape: [1, 2, 2] };
    const nodeMask = { data: Float64Array.from([1, 0]), shape: [1, 2] };
    const out = ed.unnormalize(x, h, nodeMask);
    close([...out.x.data], [1, 2, 3, 4, 5, 6]);
    close([...out.hCat.data], [9, 9, 0, 0]);
  });
});

describe("EquivariantDiffusion: full reverse process (sample)", () => {
  it("is seed-deterministic end to end", async () => {
    const ed = makeEd();
    seed(2024);
    const nodeMask = { data: Float64Array.from([1, 1]), shape: [1, 2] };
    const edgeMask = { data: Float64Array.from([0, 1, 1, 0]), shape: [4, 1] };
    const ctx = { data: Float64Array.from([0.5, 0.5, 0.5]), shape: [1, 3] };
    const { x, h } = await ed.sample(nodeMask, edgeMask, ctx, 0);
    assert.deepEqual(x.shape, [1, 2, 3]);
    close(
      [...x.data],
      [51.63963832, -4.435472061, -34.64954822, -51.63957508, 4.435472061, 34.64948498],
      1e-6,
    );
    assert.deepEqual([...h.data], [0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]);
  });
});

describe("EquivariantDiffusion: animate (trajectory)", () => {
  const nodeMask = { data: Float64Array.from([1, 1]), shape: [1, 2] };
  const edgeMask = { data: Float64Array.from([0, 1, 1, 0]), shape: [4, 1] };
  const ctx = { data: Float64Array.from([0.5, 0.5, 0.5]), shape: [1, 3] };

  it("yields one decoded frame per denoising step", async () => {
    const ed = makeEd();
    seed(7);
    const frames = [];
    for await (const f of ed.animate(nodeMask, edgeMask, ctx, 0)) frames.push(f);

    assert.equal(frames.length, ed.T);
    assert.deepEqual(frames.map((f) => f.step), [1, 2, 3]);
    assert.deepEqual(frames.map((f) => f.total), [3, 3, 3]);
    for (const f of frames) {
      assert.deepEqual(f.x.shape, [1, 2, 3]);
      assert.ok([...f.x.data].every(Number.isFinite));
    }
  });

  it("is seed-deterministic end to end", async () => {
    const runOnce = async () => {
      const ed = makeEd();
      seed(11);
      const frames = [];
      for await (const f of ed.animate(nodeMask, edgeMask, ctx, 0)) frames.push(f);
      return frames.at(-1);
    };
    const a = await runOnce();
    const b = await runOnce();
    close([...a.x.data], [...b.x.data]);
    assert.deepEqual([...a.h.data], [...b.h.data]);
  });
});
