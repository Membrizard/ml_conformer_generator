import assert from "node:assert/strict";
import { describe, it } from "node:test";
import {
  NumpyRandomState,
  numpyRandint,
  numpyRandn,
  seed,
} from "../../src/numpyRandom.js";

describe("NumpyRandomState (NumPy RandomState parity)", () => {
  it("seed(42) randn matches NumPy bit-for-bit", () => {
    seed(42);
    const { data } = numpyRandn([5]);
    assert.deepEqual([...data], [
      0.4967141530112327,
      -0.13826430117118466,
      0.6476885381006925,
      1.5230298564080254,
      -0.23415337472333597,
    ]);
  });

  it("seed(42) randint matches NumPy", () => {
    seed(42);
    const draws = Array.from({ length: 10 }, () => numpyRandint(15, 40));
    assert.deepEqual(draws, [21, 34, 29, 25, 22, 35, 21, 33, 37, 25]);
  });

  it("independent instances do not share state", () => {
    const a = new NumpyRandomState(1);
    const b = new NumpyRandomState(1);
    assert.equal(a.standardNormal(), b.standardNormal());
    a.standardNormal();
    assert.notEqual(a.standardNormal(), b.standardNormal());
  });

  // Longer characterization goldens (lock the current RNG stream so a refactor
  // can't shift it). First 5 of randn20 also match the NumPy golden above.
  it("seed(42) randn(20) — full stream golden", () => {
    seed(42);
    assert.deepEqual([...numpyRandn([20]).data], [
      0.4967141530112327, -0.13826430117118466, 0.6476885381006925,
      1.5230298564080254, -0.23415337472333597, -0.23413695694918055,
      1.5792128155073915, 0.7674347291529088, -0.4694743859349521,
      0.5425600435859647, -0.46341769281246226, -0.46572975357025687,
      0.24196227156603412, -1.913280244657798, -1.7249178325130328,
      -0.5622875292409727, -1.0128311203344238, 0.3142473325952739,
      -0.9080240755212109, -1.4123037013352915,
    ]);
  });

  it("seed(7) randint(15,40) x30 — stream golden", () => {
    seed(7);
    const draws = Array.from({ length: 30 }, () => numpyRandint(15, 40));
    assert.deepEqual(draws, [
      30, 19, 37, 18, 34, 38, 22, 29, 38, 23, 29, 25, 23, 22, 21, 19, 31, 22,
      27, 15, 26, 38, 21, 34, 27, 20, 39, 39, 38, 36,
    ]);
  });

  it("randint(low, high) stays within [low, high)", () => {
    seed(99);
    for (let i = 0; i < 200; i += 1) {
      const v = numpyRandint(15, 40);
      assert.ok(v >= 15 && v < 40, `out of range: ${v}`);
    }
  });
});
