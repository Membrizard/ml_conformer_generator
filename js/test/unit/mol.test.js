import assert from "node:assert/strict";
import { describe, it } from "node:test";
import { registerDefaultRdkit } from "../../src/rdkit.js";
import {
  Molecule,
  contextFromCoordinates,
  isValidMol,
  redefineBonds,
  standardizeMol,
} from "../../src/mol.js";

// Register RDKit directly (not via index.js) so these run without onnxruntime-node.
registerDefaultRdkit();

const TOL = 1e-4; // context uses float32 internally
function close(actual, expected, tol = TOL) {
  assert.equal(actual.length, expected.length, "length");
  for (let i = 0; i < expected.length; i += 1) {
    assert.ok(
      Math.abs(actual[i] - expected[i]) <= tol,
      `index ${i}: got ${actual[i]}, want ${expected[i]}`,
    );
  }
}

describe("Molecule: MOL block (V2000) export", () => {
  it("emits counts line, atom block and bond block", () => {
    const mol = new Molecule({
      atomicNumbers: [6, 7, 8],
      positions: Float32Array.from([0, 0, 0, 1.2, 0, 0, 2.4, 0.5, 0]),
      bonds: [
        { i: 0, j: 1, type: 1 },
        { i: 1, j: 2, type: 2 },
      ],
    });
    const expected = [
      "MLConfGen",
      "  mlconfgen",
      "",
      "  3  2  0  0  0  0  0  0  0  0999 V2000",
      "    0.0000    0.0000    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0",
      "    1.2000    0.0000    0.0000 N   0  0  0  0  0  0  0  0  0  0  0  0",
      "    2.4000    0.5000    0.0000 O   0  0  0  0  0  0  0  0  0  0  0  0",
      "  1  2  1  0  0  0  0",
      "  2  3  2  0  0  0  0",
      "M  END",
    ].join("\n");
    assert.equal(mol.toMolBlock(), expected);
  });
});

describe("Molecule: MOI / context", () => {
  it("returns 3 ascending finite principal values (unit mass)", () => {
    const pos = Float32Array.from([0, 0, 0, 1.5, 0, 0, 0, 1.5, 0, 0, 0, 1.5, 1, 1, 1]);
    const { context, aligned } = contextFromCoordinates(pos, 5);
    close([...context], [3.75, 3.749999761581421, 4.5]);
    close([...aligned.slice(0, 3)], [0, 0, -0.8660253882408142]);
  });
});

describe("Molecule: bond redefinition from AdjMatSeer output", () => {
  it("reads lower-triangle argmax over 5 bond-type channels", () => {
    const mol = new Molecule({
      atomicNumbers: [6, 6, 8],
      positions: Float32Array.from([0, 0, 0, 1.3, 0, 0, 2.6, 0, 0]),
      bonds: [],
    });
    const D = 3;
    const slice = new Float32Array(D * D * 5);
    const setBond = (i, j, ch) => {
      slice[(i * D + j) * 5 + ch] = 1;
    };
    setBond(1, 0, 2); // double
    setBond(2, 1, 1); // single
    const bonded = redefineBonds(mol, slice, D);
    assert.deepEqual(bonded.bonds, [
      { i: 1, j: 0, type: 2 },
      { i: 2, j: 1, type: 1 },
    ]);
  });
});

describe("Molecule: validity and standardization", () => {
  it("isValidMol accepts a sane 3-atom chain", async () => {
    const mol = new Molecule({
      atomicNumbers: [6, 6, 8],
      positions: Float32Array.from([0, 0, 0, 1.5, 0, 0, 2.9, 0, 0]),
      bonds: [
        { i: 0, j: 1, type: 1 },
        { i: 1, j: 2, type: 1 },
      ],
    });
    assert.equal(await isValidMol(mol), 1);
  });

  it("standardizeMol keeps all atoms of a connected molecule", async () => {
    const mol = new Molecule({
      atomicNumbers: [6, 6, 8],
      positions: Float32Array.from([0, 0, 0, 1.5, 0, 0, 2.9, 0, 0]),
      bonds: [
        { i: 0, j: 1, type: 1 },
        { i: 1, j: 2, type: 1 },
      ],
    });
    const std = await standardizeMol(mol, { keepLargestFragment: true });
    assert.ok(std);
    assert.equal(std.nAtoms, 3);
    assert.equal(await isValidMol(std), 1);
  });

  it("standardizeMol returns null for an empty molecule", async () => {
    const mol = new Molecule({ atomicNumbers: [], positions: [], bonds: [] });
    assert.equal(await standardizeMol(mol), null);
  });
});
