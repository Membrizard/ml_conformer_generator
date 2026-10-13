/**
 * The package must stay loadable in the browser: Node built-ins may only be
 * imported dynamically inside Node-only code paths. This guards against
 * *static* imports and re-exports of built-ins only. It does not check
 * dynamic `import("node:…")` calls, which esbuild browser builds still
 * resolve unless `node:*` is marked external.
 */
import assert from "node:assert/strict";
import { readdirSync, readFileSync } from "node:fs";
import { builtinModules } from "node:module";
import { join, relative } from "node:path";
import { it } from "node:test";

import { PACKAGE_ROOT } from "../helpers.js";

const SRC = join(PACKAGE_ROOT, "src");

/** Every built-in, without its optional `node:` prefix. */
const BUILTINS = new Set(builtinModules.map((m) => m.replace(/^node:/, "")));

/**
 * Specifiers of static `import … from`, `import "…"` and `export … from`
 * statements. `import(…)` and `import.meta` are not statements.
 */
const STATIC_SPECIFIER =
  /^(?:import\b(?!\s*[(.])|export\b)[^;]*?\bfrom\s*["']([^"']+)["']|^import\s*["']([^"']+)["']/gm;

function staticBuiltinImports(text) {
  return [...text.matchAll(STATIC_SPECIFIER)]
    .map((m) => m[1] ?? m[2])
    .filter((spec) => spec.startsWith("node:") || BUILTINS.has(spec));
}

function jsFiles(dir) {
  return readdirSync(dir, { withFileTypes: true }).flatMap((entry) => {
    const full = join(dir, entry.name);
    if (entry.isDirectory()) return jsFiles(full);
    return entry.name.endsWith(".js") ? [full] : [];
  });
}

it("the scan finds every static form and skips dynamic imports", () => {
  const flagged = `
import fs from "fs";
import { join } from "node:path";
import "os";
import {
  Worker,
} from "worker_threads";
export { createHash } from "node:crypto";
export * from "stream";
import x from "fs/promises";
`;
  assert.deepEqual(staticBuiltinImports(flagged), [
    "fs",
    "node:path",
    "os",
    "worker_threads",
    "node:crypto",
    "stream",
    "fs/promises",
  ]);

  const allowed = `
import { a } from "./fs.js";
import { b } from "@rdkit/rdkit";
const url = import.meta.url;
  const fs = await import("node:fs/promises");
import(/* webpackIgnore: true */ "node:os").then((os) => os);
export const from = "fs";
`;
  assert.deepEqual(staticBuiltinImports(allowed), []);
});

it("src/ loads the random seeds only through a dynamic import", () => {
  // A static import or re-export of randomContext.js would put the ~200 KB
  // seed list into every browser bundle.
  const offenders = jsFiles(SRC).flatMap((file) =>
    [...readFileSync(file, "utf8").matchAll(STATIC_SPECIFIER)]
      .map((m) => m[1] ?? m[2])
      .filter((spec) => /\/randomContext\.js$/.test(spec))
      .map((spec) => `${relative(SRC, file)}: ${spec}`),
  );
  assert.deepEqual(offenders, []);
});

it("src/ has no static imports or re-exports of Node built-ins", () => {
  const files = jsFiles(SRC);
  assert.ok(files.length > 0);
  const offenders = files.flatMap((file) =>
    staticBuiltinImports(readFileSync(file, "utf8")).map(
      (spec) => `${relative(SRC, file)}: ${spec}`,
    ),
  );
  assert.deepEqual(offenders, []);
});
