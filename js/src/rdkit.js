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
 * Raised when a registered RDKit loader exists but fails to initialise.
 *
 * Distinct from "no loader configured" (see `hasRdkitLoader`) so callers can
 * tell a broken RDKit apart from a genuinely invalid molecule instead of
 * silently reporting every molecule as invalid.
 */
export class RdkitLoadError extends Error {
  constructor(cause) {
    super(
      `RDKit failed to initialise: ${cause?.message ?? cause}. ` +
        "In the browser, ensure the bundled @rdkit/rdkit can resolve its " +
        "RDKit_minimal.wasm, or pass a custom `rdkitLoader` to createGenerator().",
      { cause },
    );
    this.name = "RdkitLoadError";
  }
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
    rdkitPromise = Promise.resolve()
      .then(() => rdkitLoader())
      .catch((err) => {
        throw new RdkitLoadError(err);
      });
  }
  return rdkitPromise;
}

const IS_NODE =
  typeof process !== "undefined" && process.versions?.node != null;

/**
 * Node: resolve the bundled WASM through the module graph, since Emscripten
 * cannot infer its own location outside a browser.
 */
async function loadRdkitNode() {
  const { createRequire } = await import("node:module");
  const path = await import("node:path");
  const require = createRequire(import.meta.url);
  const pkgDir = path.dirname(require.resolve("@rdkit/rdkit/package.json"));
  const initRDKitModule = (await import("@rdkit/rdkit")).default;
  return initRDKitModule({
    locateFile: () => path.join(pkgDir, "dist", "RDKit_minimal.wasm"),
  });
}

/**
 * Browser / worker: `RDKit_minimal.js` locates its own `.wasm` relative to the
 * script URL, so no `locateFile` is needed.
 *
 * Three setups, tried in order:
 *   1. a global `initRDKitModule` from a plain <script> tag;
 *   2. a bundler (Vite/webpack/…) that interops the CommonJS build into a
 *      usable `default` export;
 *   3. no bundler — `RDKit_minimal.js` is UMD and yields *no* ES exports when
 *      imported directly, so it has to be injected as a classic script.
 *
 * Anything more exotic should pass an explicit `rdkitLoader`.
 */
async function loadRdkitBrowser() {
  if (typeof globalThis.initRDKitModule === "function") {
    return globalThis.initRDKitModule();
  }

  try {
    const mod = await import("@rdkit/rdkit");
    const init = mod?.default ?? mod?.initRDKitModule;
    if (typeof init === "function") return init();
  } catch {
    // Not resolvable as a module here; fall through to the script-tag path.
  }

  const init = await loadRdkitViaScriptTag();
  return init();
}

const RDKIT_CDN =
  "https://cdn.jsdelivr.net/npm/@rdkit/rdkit@2025.3.4-1.0.0/dist/RDKit_minimal.js";

/** @type {Promise<Function> | null} */
let scriptTagPromise = null;

/**
 * Inject the UMD build so it installs `globalThis.initRDKitModule`.
 * Only used when there is no bundler and no pre-existing global.
 */
function loadRdkitViaScriptTag(src = RDKIT_CDN) {
  if (scriptTagPromise) return scriptTagPromise;
  if (typeof document === "undefined") {
    return Promise.reject(
      new Error(
        "No DOM available to load RDKit — pass an explicit `rdkitLoader` " +
          "(e.g. importScripts in a worker).",
      ),
    );
  }

  scriptTagPromise = new Promise((resolve, reject) => {
    const script = document.createElement("script");
    script.src = src;
    script.async = true;
    script.onload = () => {
      const init = globalThis.initRDKitModule;
      if (typeof init === "function") resolve(init);
      else reject(new Error(`Loaded ${src} but initRDKitModule is missing.`));
    };
    script.onerror = () => reject(new Error(`Failed to load ${src}`));
    document.head.appendChild(script);
  }).catch((err) => {
    scriptTagPromise = null;
    throw err;
  });

  return scriptTagPromise;
}

/**
 * Default loader for the bundled `@rdkit/rdkit`, picked to match the runtime so
 * that `npm install mlconfgen` is enough in both Node and the browser.
 * Override with `setRdkitLoader` / the `rdkitLoader` option when embedding a
 * custom RDKit build.
 */
export function registerDefaultRdkit() {
  if (hasRdkitLoader()) return;
  setRdkitLoader(() => (IS_NODE ? loadRdkitNode() : loadRdkitBrowser()));
}
