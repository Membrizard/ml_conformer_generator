import assert from "node:assert/strict";
import { existsSync, readFileSync } from "node:fs";
import { join } from "node:path";
import { describe, it } from "node:test";

import { MAX_N_NODES, MIN_N_NODES } from "../../src/constants.js";
import { getRandomState, seed } from "../../src/numpyRandom.js";
import { randomContext } from "../../src/randomContext.js";
import SEEDS from "../../src/randomMoleculeSeeds.js";
import { REPO_ROOT } from "../helpers.js";

const PYTHON_SEEDS = join(
  REPO_ROOT,
  "src",
  "mlconfgen",
  "utils",
  "random_molecule_seeds.json",
);

describe("randomContext", () => {
  it("returns the same context for the same seed", () => {
    const a = randomContext(42);
    assert.deepEqual(a, randomContext(42));
    assert.ok(a.context instanceof Float32Array);
    assert.equal(a.context.length, 3);
  });

  it("varies with the seed", () => {
    const picks = new Set();
    for (let i = 0; i < 50; i += 1) {
      const { context, nAtoms } = randomContext(i);
      picks.add(`${nAtoms}:${context[0]}`);
    }
    assert.ok(picks.size > 30, `only ${picks.size} distinct picks for 50 seeds`);
  });

  it("picks one of the bundled seeds, within the model's atom range", () => {
    for (let i = 0; i < 200; i += 1) {
      const { context, nAtoms } = randomContext();
      assert.ok(nAtoms >= MIN_N_NODES && nAtoms <= MAX_N_NODES);
      assert.ok(
        SEEDS.some(
          (s) =>
            s.nAtoms === nAtoms &&
            s.context.every((v, j) => Math.fround(v) === context[j]),
        ),
      );
    }
  });

  it("rejects seeds outside [0, 2**32)", () => {
    for (const bad of [-1, 1.5, 2 ** 32, "7", Number.NaN]) {
      assert.throws(() => randomContext(bad), RangeError, String(bad));
    }
  });

  it("leaves the global random state untouched", () => {
    seed(5);
    const expected = getRandomState().randomUint32();
    seed(5);
    randomContext(1);
    assert.equal(getRandomState().randomUint32(), expected);
  });
});

describe("bundled random molecule seeds", () => {
  it("all lie within the model's atom range", () => {
    for (const { context, nAtoms } of SEEDS) {
      assert.equal(context.length, 3);
      assert.ok(nAtoms >= MIN_N_NODES && nAtoms <= MAX_N_NODES);
    }
  });

  it(
    "match the Python package's random_molecule_seeds.json",
    { skip: existsSync(PYTHON_SEEDS) ? false : "Python sources not present" },
    () => {
      const python = JSON.parse(readFileSync(PYTHON_SEEDS, "utf8"));
      assert.equal(SEEDS.length, python.length);
      python.forEach((p, i) => {
        assert.equal(SEEDS[i].nAtoms, p.n_atoms, `entry ${i}`);
        assert.deepEqual(SEEDS[i].context, p.context, `entry ${i}`);
      });
    },
  );
});
