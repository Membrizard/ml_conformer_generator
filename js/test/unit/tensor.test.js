import assert from "node:assert/strict";
import { describe, it } from "node:test";
import {
  add,
  argmaxLast,
  concatLast,
  div,
  inflateBatch,
  logSigmoid,
  mul,
  oneHot,
  removeMeanWithMask,
  sampleCenterGravityZeroGaussianWithMask,
  sampleGaussianWithMask,
  sigmoid,
  sliceLast,
  softplus,
  sub,
} from "../../src/tensor.js";
import { seed } from "../../src/numpyRandom.js";

// Characterization tests: goldens captured from the current implementation.
// They lock present behavior so a refactor cannot silently change the math.
const TOL = 1e-9;
function close(actual, expected, tol = TOL) {
  assert.equal(actual.length, expected.length, "length");
  for (let i = 0; i < expected.length; i += 1) {
    assert.ok(
      Math.abs(actual[i] - expected[i]) <= tol,
      `index ${i}: got ${actual[i]}, want ${expected[i]}`,
    );
  }
}

const A = { data: Float64Array.from([1, 2, 3, 4, 5, 6]), shape: [1, 2, 3] };
const B = { data: Float64Array.from([6, 5, 4, 3, 2, 1]), shape: [1, 2, 3] };

describe("tensor: elementwise ops", () => {
  it("add / sub", () => {
    close([...add(A, B).data], [7, 7, 7, 7, 7, 7]);
    close([...sub(A, B).data], [-5, -3, -1, 1, 3, 5]);
  });

  it("mul: elementwise and scalar", () => {
    close([...mul(A, B).data], [6, 10, 12, 12, 10, 6]);
    close([...mul(A, 2).data], [2, 4, 6, 8, 10, 12]);
  });

  it("div: elementwise", () => {
    close(
      [...div(A, B).data],
      [
        0.16666666666666666, 0.4, 0.75, 1.3333333333333333, 2.5, 6,
      ],
    );
  });
});

describe("tensor: shape ops", () => {
  it("concatLast joins along the last axis", () => {
    const c = concatLast(A, B);
    assert.deepEqual(c.shape, [1, 2, 6]);
    close([...c.data], [1, 2, 3, 6, 5, 4, 4, 5, 6, 3, 2, 1]);
  });

  it("sliceLast [0,2)", () => {
    const s = sliceLast(A, 0, 2);
    assert.deepEqual(s.shape, [1, 2, 2]);
    close([...s.data], [1, 2, 4, 5]);
  });

  it("sliceLast [1,-1) resolves negative end", () => {
    const s = sliceLast(A, 1, -1);
    assert.deepEqual(s.shape, [1, 2, 1]);
    close([...s.data], [2, 5]);
  });

  it("inflateBatch broadcasts a per-batch scalar", () => {
    const out = inflateBatch(
      { data: Float64Array.from([10, 20]), shape: [2] },
      { data: new Float64Array(12), shape: [2, 2, 3] },
    );
    assert.deepEqual(out.shape, [2, 2, 3]);
    close([...out.data], [10, 10, 10, 10, 10, 10, 20, 20, 20, 20, 20, 20]);
  });
});

describe("tensor: math helpers (match Python ONNX formulas)", () => {
  const t = { data: Float64Array.from([-2, 0, 2]), shape: [3] };
  it("softplus = log1p(exp(z))", () => {
    close(
      [...softplus(t).data],
      [0.1269280110429725, 0.6931471805599453, 2.1269280110429727],
    );
  });
  it("logSigmoid = -log1p(exp(-z))", () => {
    close(
      [...logSigmoid(t).data],
      [-2.1269280110429727, -0.6931471805599453, -0.1269280110429725],
    );
  });
  it("sigmoid = 1/(1+exp(-z))", () => {
    close(
      [...sigmoid(t).data],
      [0.11920292202211755, 0.5, 0.8807970779778823],
    );
  });
});

describe("tensor: masked reductions and decoding", () => {
  const nodeMask = { data: Float64Array.from([1, 1, 0]), shape: [1, 3] };

  it("removeMeanWithMask subtracts masked mean", () => {
    const x = { data: Float64Array.from([1, 1, 1, 3, 3, 3, 9, 9, 9]), shape: [1, 3, 3] };
    close(
      [...removeMeanWithMask(x, nodeMask).data],
      [-5.5, -5.5, -5.5, -3.5, -3.5, -3.5, 9, 9, 9],
    );
  });

  it("argmaxLast picks first-max per (B,N) row", () => {
    const t = { data: Float64Array.from([0.1, 0.9, 0.2, 0.5, 0.4, 0.6]), shape: [1, 2, 3] };
    assert.deepEqual([...argmaxLast(t)], [1, 2]);
  });

  it("oneHot encodes labels into numClasses", () => {
    close([...oneHot(Int32Array.from([1, 0]), 3, [1, 2]).data], [0, 1, 0, 1, 0, 0]);
  });

  it("sampleGaussianWithMask is seed-deterministic and zeros padded nodes", () => {
    seed(123);
    close(
      [...sampleGaussianWithMask([1, 3, 2], nodeMask).data],
      [-1.0856306033005612, 0.9973454465835858, 0.28297849805199204, -1.506294713918092, 0, 0],
    );
  });

  it("sampleCenterGravityZeroGaussianWithMask projects out the COM", () => {
    seed(123);
    close(
      [...sampleCenterGravityZeroGaussianWithMask([1, 3, 2], nodeMask).data],
      [-0.6843045506762766, 1.2518200802508388, 0.6843045506762766, -1.2518200802508388, 0, 0],
    );
  });
});
