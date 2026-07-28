import assert from "node:assert/strict";
import test from "node:test";

import { isValidMol, Molecule, standardizeMol } from "../../src/mol.js";
import {
  clearRdkitLoader,
  hasRdkitLoader,
  RdkitLoadError,
  registerDefaultRdkit,
  setRdkitLoader,
} from "../../src/rdkit.js";

/** Ethanol-like C-C-O; sanitizes cleanly. */
function validMolecule() {
  return new Molecule({
    atomicNumbers: [6, 6, 8],
    positions: Float32Array.from([0, 0, 0, 1.5, 0, 0, 2.4, 1.0, 0]),
    bonds: [
      { i: 0, j: 1, type: 1 },
      { i: 1, j: 2, type: 1 },
    ],
  });
}

/** Hypervalent carbon: six single bonds, so sanitize must reject it. */
function invalidMolecule() {
  return new Molecule({
    atomicNumbers: [6, 6, 6, 6, 6, 6, 6],
    positions: Float32Array.from([
      0, 0, 0, 1.5, 0, 0, -1.5, 0, 0, 0, 1.5, 0, 0, -1.5, 0, 0, 0, 1.5, 0, 0,
      -1.5,
    ]),
    bonds: [1, 2, 3, 4, 5, 6].map((j) => ({ i: 0, j, type: 1 })),
  });
}

test("rdkit loader", async (t) => {
  t.afterEach(() => {
    clearRdkitLoader();
    registerDefaultRdkit();
  });

  await t.test("a failing loader throws instead of reporting invalid", async () => {
    // Regression: the browser default used to import `node:module`, which threw
    // and was swallowed by the call sites, so every molecule silently came back
    // invalid and generateConformers returned []. A broken RDKit must be loud.
    setRdkitLoader(async () => {
      throw new Error("Failed to resolve module specifier 'node:module'");
    });

    await assert.rejects(() => isValidMol(validMolecule()), RdkitLoadError);
    await assert.rejects(() => standardizeMol(validMolecule()), RdkitLoadError);
  });

  await t.test("load failure is distinguishable from invalid chemistry", async () => {
    setRdkitLoader(async () => {
      throw new Error("boom");
    });

    const err = await isValidMol(validMolecule()).catch((e) => e);
    assert.ok(err instanceof RdkitLoadError);
    assert.match(err.message, /rdkitLoader/);
    assert.equal(err.cause?.message, "boom");
  });

  await t.test("default loader works with no configuration", async () => {
    clearRdkitLoader();
    registerDefaultRdkit();
    assert.ok(hasRdkitLoader());

    assert.equal(await isValidMol(validMolecule()), 1);
    assert.notEqual(await standardizeMol(validMolecule()), null);
  });

  await t.test("genuinely invalid molecules are still rejected, not thrown", async () => {
    clearRdkitLoader();
    registerDefaultRdkit();

    assert.equal(await isValidMol(invalidMolecule()), 0);
    assert.equal(await standardizeMol(invalidMolecule()), null);
  });

  await t.test("an explicit rdkitLoader overrides the default", async () => {
    clearRdkitLoader();
    let called = 0;
    setRdkitLoader(async () => {
      called += 1;
      const { default: init } = await import("@rdkit/rdkit");
      const { createRequire } = await import("node:module");
      const path = await import("node:path");
      const require = createRequire(import.meta.url);
      const dir = path.dirname(require.resolve("@rdkit/rdkit/package.json"));
      return init({
        locateFile: () => path.join(dir, "dist", "RDKit_minimal.wasm"),
      });
    });

    assert.equal(await isValidMol(validMolecule()), 1);
    assert.equal(called, 1, "custom loader should be used");
  });
});
