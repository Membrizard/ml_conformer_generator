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
});
