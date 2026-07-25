/**
 * End-to-end GOLDEN snapshot test — the strongest refactor safety net.
 *
 * With seed(42) the whole pipeline is deterministic, so we snapshot the exact
 * MOL blocks and compare on every run. ANY behavioral change (RNG, math, bond
 * decoding, standardization) shifts the output and fails the test.
 *
 * Requires the ChemBL ONNX weights (NC-ND, not in the repo). Auto-skips if
 * absent. Capture / refresh the golden after an INTENTIONAL change with:
 *   MLCONFGEN_UPDATE_GOLDEN=1 node --test test/generation_golden.test.js
 *
 * Note on portability: coordinates may differ from a Python run at ~ULP scale
 * (JS Math.log vs libm — see review §2). This golden locks the JS output only;
 * for JS↔Python parity use a separate tolerance-based comparison.
 */
import assert from "node:assert/strict";
import { existsSync, readFileSync, writeFileSync } from "node:fs";
import { before, describe, it } from "node:test";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { createGenerator, seed } from "../src/index.js";
import { resolveOnnxPaths } from "./helpers.js";
import * as ort from "onnxruntime-node";

const __dirname = dirname(fileURLToPath(import.meta.url));
const GOLDEN_PATH = join(__dirname, "__goldens__", "generation_seed42.json");
const UPDATE = process.env.MLCONFGEN_UPDATE_GOLDEN === "1";

const onnx = resolveOnnxPaths();
const skipReason = onnx.available
  ? false
  : "ONNX weights not found (set EGNN_ONNX / ADJ_ONNX or place models in js/)";

const SEED = 42;
const DIFFUSION_STEPS = 50;
const REFERENCE_CONTEXT = [89.8693, 210.783, 217.7825];

describe("E2E generation golden (seed=42)", { skip: skipReason }, () => {
  /** @type {MLConformerGenerator} */
  let generator;

  before(async () => {
    generator = await createGenerator({
      ort,
      egnnOnnx: onnx.egnn,
      adjMatSeerOnnx: onnx.adj,
      diffusionSteps: DIFFUSION_STEPS,
    });
  });

  it("produces the recorded MOL blocks", async () => {
    seed(SEED);
    const mols = await generator.generateConformers({
      referenceContext: REFERENCE_CONTEXT,
      nAtoms: 20,
      nSamples: 5,
      variance: 2,
      filterInvalid: true,
    });
    const actual = mols.map((m) => m.toMolBlock());

    if (UPDATE || !existsSync(GOLDEN_PATH)) {
      writeFileSync(GOLDEN_PATH, `${JSON.stringify(actual, null, 2)}\n`);
      console.log(`[golden] wrote ${actual.length} molecules → ${GOLDEN_PATH}`);
      return;
    }

    const golden = JSON.parse(readFileSync(GOLDEN_PATH, "utf8"));
    assert.deepEqual(
      actual,
      golden,
      "generation output drifted from golden — if intentional, re-run with MLCONFGEN_UPDATE_GOLDEN=1",
    );
  });
});
