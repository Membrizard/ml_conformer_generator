/**
 * Preloaded by every test script (`node --require ./test/setup.cjs --test …`).
 *
 * Points the weights cache at a throwaway directory for the whole run, so no
 * test can write to the user's ~/.mlconfgen_weights. The test runner's child
 * processes inherit the variable; the directory is removed when the process
 * that created it exits. An MLCONFGEN_CACHE set by the caller is left alone.
 */
const { mkdtempSync, rmSync } = require("node:fs");
const { tmpdir } = require("node:os");
const { join } = require("node:path");

if (process.env.MLCONFGEN_CACHE == null) {
  const cacheDir = mkdtempSync(join(tmpdir(), "mlconfgen-test-cache-"));
  process.env.MLCONFGEN_CACHE = cacheDir;
  process.on("exit", () => rmSync(cacheDir, { recursive: true, force: true }));
}
