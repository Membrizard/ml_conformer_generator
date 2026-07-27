/**
 * Optional RDKit.js hook for SMILES canonicalisation and sanitize-based validity.
 * The package registers a Node WASM loader by default via `registerDefaultRdkit()`.
 */

/** @type {null | (() => Promise<any>)} */
let rdkitLoader = null;
/** @type {Promise<any> | null} */
let rdkitPromise = null;

/**
 * @param {() => Promise<any>} loader async factory returning an initialised RDKit module
 */
export function setRdkitLoader(loader) {
  rdkitLoader = loader;
  rdkitPromise = null;
}

/** Disable RDKit (generation still runs; no sanitize / SMILES reorder). */
export function clearRdkitLoader() {
  rdkitLoader = null;
  rdkitPromise = null;
}

/** @returns {boolean} */
export function hasRdkitLoader() {
  return typeof rdkitLoader === "function";
}

/**
 * @returns {Promise<any>}
 * @throws {Error} if no loader has been registered
 */
export function getRdkit() {
  if (!rdkitLoader) {
    throw new Error(
      "RDKit is not configured. It is registered automatically on import of mlconfgen; or call setRdkitLoader(...).",
    );
  }
  if (!rdkitPromise) {
    rdkitPromise = Promise.resolve().then(() => rdkitLoader());
  }
  return rdkitPromise;
}

/**
 * Default Node loader for `@rdkit/rdkit` (WASM next to the package).
 * Called once from the package entry so `npm install mlconfgen` is enough.
 */
export function registerDefaultRdkit() {
  if (hasRdkitLoader()) return;
  setRdkitLoader(async () => {
    const { createRequire } = await import("node:module");
    const path = await import("node:path");
    const require = createRequire(import.meta.url);
    const pkgDir = path.dirname(require.resolve("@rdkit/rdkit/package.json"));
    const initRDKitModule = (await import("@rdkit/rdkit")).default;
    return initRDKitModule({
      locateFile: () => path.join(pkgDir, "dist", "RDKit_minimal.wasm"),
    });
  });
}
