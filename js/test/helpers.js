import { existsSync } from "node:fs";
import { readFile } from "node:fs/promises";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { Molecule } from "../src/mol.js";

const __dirname = dirname(fileURLToPath(import.meta.url));
export const PACKAGE_ROOT = resolve(__dirname, "..");
export const REPO_ROOT = resolve(PACKAGE_ROOT, "..");

const Z_MAP = {
  H: 1,
  C: 6,
  N: 7,
  O: 8,
  F: 9,
  P: 15,
  S: 16,
  Cl: 17,
  Br: 35,
  I: 53,
};

/** Resolve ONNX weights from env, package dir, or repo root. */
export function resolveOnnxPaths() {
  const egnn =
    process.env.EGNN_ONNX ||
    firstExisting([
      join(PACKAGE_ROOT, "egnn_chembl_15_39.onnx"),
      join(REPO_ROOT, "egnn_chembl_15_39.onnx"),
    ]);
  const adj =
    process.env.ADJ_ONNX ||
    firstExisting([
      join(PACKAGE_ROOT, "adj_mat_seer_chembl_15_39.onnx"),
      join(REPO_ROOT, "adj_mat_seer_chembl_15_39.onnx"),
    ]);
  return { egnn, adj, available: Boolean(egnn && adj) };
}

function firstExisting(paths) {
  for (const p of paths) {
    if (existsSync(p)) return p;
  }
  return null;
}

export function ceyyagMolPath() {
  return join(REPO_ROOT, "assets", "demo_files", "ceyyag.mol");
}

/**
 * Minimal V2000 parser. Optionally drop hydrogens (Python `RemoveHs`).
 * @returns {Promise<Molecule>}
 */
export async function loadMolFile(path, { removeHs = true } = {}) {
  const text = await readFile(path, "utf8");
  const lines = text.replace(/\r\n/g, "\n").split("\n");
  const counts = lines[3];
  const nAtoms = parseInt(counts.slice(0, 3), 10);
  const nBonds = parseInt(counts.slice(3, 6), 10);

  const atomicNumbers = [];
  const positions = [];
  for (let i = 0; i < nAtoms; i += 1) {
    const L = lines[4 + i];
    const sym = L.slice(31, 34).trim();
    const z = Z_MAP[sym];
    if (z == null) throw new Error(`Unknown element: ${sym}`);
    if (removeHs && z === 1) continue;
    atomicNumbers.push(z);
    positions.push(
      parseFloat(L.slice(0, 10)),
      parseFloat(L.slice(10, 20)),
      parseFloat(L.slice(20, 30)),
    );
  }

  // Rebuild bonds among kept atoms when stripping H.
  const keep = [];
  for (let i = 0; i < nAtoms; i += 1) {
    const L = lines[4 + i];
    const sym = L.slice(31, 34).trim();
    const z = Z_MAP[sym];
    if (!(removeHs && z === 1)) keep.push(i);
  }
  const map = new Int32Array(nAtoms).fill(-1);
  keep.forEach((old, neu) => {
    map[old] = neu;
  });

  const bonds = [];
  for (let i = 0; i < nBonds; i += 1) {
    const L = lines[4 + nAtoms + i];
    const a = parseInt(L.slice(0, 3), 10) - 1;
    const b = parseInt(L.slice(3, 6), 10) - 1;
    const t = parseInt(L.slice(6, 9), 10);
    const i2 = map[a];
    const j2 = map[b];
    if (i2 < 0 || j2 < 0) continue;
    bonds.push(
      i2 < j2 ? { i: i2, j: j2, type: t } : { i: j2, j: i2, type: t },
    );
  }

  return new Molecule({
    atomicNumbers,
    positions: Float32Array.from(positions),
    bonds,
  });
}
