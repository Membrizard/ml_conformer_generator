import { MLConformerGenerator } from "../src/index.js";

async function main() {
  const generator = await MLConformerGenerator.create({
    egnnOnnx: "./egnn_chembl_15_39.onnx",
    adjMatSeerOnnx: "./adj_mat_seer_chembl_15_39.onnx",
    diffusionSteps: 100,
  });

  const molecules = await generator.generateConformers({
    // Example MOI eigenvalues (principal frame)
    referenceContext: [89.8693, 210.783, 217.7825],
    nAtoms: 20,
    nSamples: 2,
    variance: 0,
  });

  console.log(`Generated ${molecules.length} molecules.`);
  for (const mol of molecules) {
    console.log(`atoms=${mol.nAtoms} bonds=${mol.bonds.length}`);
    console.log(mol.toMolBlock());
  }
}

main().catch((err) => {
  console.error(err);
  process.exitCode = 1;
});
