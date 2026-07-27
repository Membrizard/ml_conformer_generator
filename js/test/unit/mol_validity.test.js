import assert from "node:assert/strict";
import { describe, it } from "node:test";
// Register RDKit directly (not via index.js) so the unit suite runs without
// onnxruntime-node — index.js statically imports the native ORT addon.
import { registerDefaultRdkit } from "../../src/rdkit.js";
import {
  Molecule,
  contextFromCoordinates,
  isValidMol,
  standardizeMol,
} from "../../src/mol.js";
import { ceyyagMolPath, loadMolFile } from "../helpers.js";

registerDefaultRdkit();

describe("Molecule validity / MOI context", () => {
  it("isValidMol accepts a simple ethane-like molecule", async () => {
    const mol = new Molecule({
      atomicNumbers: [6, 6],
      positions: [0, 0, 0, 1.5, 0, 0],
      bonds: [{ i: 0, j: 1, type: 1 }],
    });
    assert.equal(await isValidMol(mol), 1);
  });

  it("standardizeMol returns null for empty mol", async () => {
    const mol = new Molecule({
      atomicNumbers: [],
      positions: [],
      bonds: [],
    });
    assert.equal(await standardizeMol(mol), null);
  });

  it("ceyyag heavy-atom count is 17 and MOI is finite", async () => {
    const mol = await loadMolFile(ceyyagMolPath(), { removeHs: true });
    assert.equal(mol.nAtoms, 17);
    const { context } = contextFromCoordinates(mol.positions, mol.nAtoms);
    assert.equal(context.length, 3);
    for (const v of context) assert.ok(Number.isFinite(v) && v > 0);
  });

  it("prepareInputs-compatible context is stable under re-alignment", async () => {
    const mol = await loadMolFile(ceyyagMolPath(), { removeHs: true });
    const once = contextFromCoordinates(mol.positions, mol.nAtoms);
    const twice = contextFromCoordinates(once.aligned, mol.nAtoms);
    for (let i = 0; i < 3; i += 1) {
      assert.ok(
        Math.abs(once.context[i] - twice.context[i]) < 1e-3,
        `context[${i}]`,
      );
    }
  });
});
