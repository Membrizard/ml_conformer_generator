import { createGenerator, seed } from "../src/index.js";
import * as ort from "onnxruntime-node";
import { writeFileSync } from "node:fs";

/** Minimal Z → element symbol for the 8 supported atom types. */
const Z_TO_SYMBOL = { 6: "C", 7: "N", 8: "O", 9: "F", 15: "P", 16: "S", 17: "Cl", 35: "Br" };

function radiusOfGyration(mol) {
  const n = mol.nAtoms;
  let cx = 0, cy = 0, cz = 0;
  for (let i = 0; i < n; i += 1) {
    cx += mol.positions[i * 3];
    cy += mol.positions[i * 3 + 1];
    cz += mol.positions[i * 3 + 2];
  }
  cx /= n; cy /= n; cz /= n;
  let s = 0;
  for (let i = 0; i < n; i += 1) {
    const dx = mol.positions[i * 3] - cx;
    const dy = mol.positions[i * 3 + 1] - cy;
    const dz = mol.positions[i * 3 + 2] - cz;
    s += dx * dx + dy * dy + dz * dz;
  }
  return Math.sqrt(s / n);
}

function toXyzFrame(mol, comment) {
  const n = mol.nAtoms;
  const lines = [String(n), comment];
  for (let i = 0; i < n; i += 1) {
    const sym = Z_TO_SYMBOL[mol.atomicNumbers[i]] || "C";
    lines.push(
      `${sym} ${mol.positions[i * 3].toFixed(4)} ${mol.positions[i * 3 + 1].toFixed(4)} ${mol.positions[i * 3 + 2].toFixed(4)}`,
    );
  }
  return lines.join("\n");
}

async function main() {
  const steps = Number(process.env.STEPS) || 50;
  const gen = await createGenerator({
    ort,
    egnnOnnx: process.env.EGNN_ONNX || "./egnn_chembl_15_39.onnx",
    adjMatSeerOnnx: process.env.ADJ_ONNX || "./adj_mat_seer_chembl_15_39.onnx",
    diffusionSteps: steps,
  });

  seed(42);

  const frames = [];
  let last = null;
  for await (const { step, total, molecules } of gen.animateGeneration({
    referenceContext: [89.8693, 210.783, 217.7825], // MOI eigenvalues
    nAtoms: 20,
    nSamples: 1,
    variance: 0,
  })) {
    const mol = molecules[0];
    last = mol;
    const rg = radiusOfGyration(mol);
    frames.push(toXyzFrame(mol, `step ${step}/${total} Rg=${rg.toFixed(3)}`));
    if (step === 1 || step === total || step % 10 === 0) {
      console.log(`step ${String(step).padStart(3)}/${total}  atoms=${mol.nAtoms}  Rg=${rg.toFixed(3)}`);
    }
  }

  writeFileSync("trajectory.xyz", `${frames.join("\n")}\n`);
  console.log(`\nWrote trajectory.xyz — ${frames.length} frames, ${last.nAtoms} atoms.`);
  console.log("View it: drag into https://molview.org (or open in VMD / PyMOL / Avogadro) and play frames.");
}

main().catch((err) => {
  console.error(err);
  process.exitCode = 1;
});
