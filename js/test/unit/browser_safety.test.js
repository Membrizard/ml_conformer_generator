/**
 * The package must stay loadable in the browser and by bundlers: Node
 * built-ins may only be imported dynamically inside Node-only code paths.
 */
import assert from "node:assert/strict";
import { readdirSync, readFileSync } from "node:fs";
import { join } from "node:path";
import { it } from "node:test";

import { PACKAGE_ROOT } from "../helpers.js";

const SRC = join(PACKAGE_ROOT, "src");

it("src/ has no static imports of Node built-ins", () => {
  const offenders = [];
  for (const file of readdirSync(SRC)) {
    if (!file.endsWith(".js")) continue;
    const text = readFileSync(join(SRC, file), "utf8");
    // Static `import … from "node:…"` / `import "node:…"`; `import("node:…")`
    // calls are fine.
    const pattern =
      /^import\s(?:[^;"'()]*?\sfrom\s*)?["'](node:[^"']+|fs|path|os|url|module)["']/gm;
    for (const match of text.matchAll(pattern)) {
      offenders.push(`${file}: ${match[0].trim()}`);
    }
  }
  assert.deepEqual(offenders, []);
});
