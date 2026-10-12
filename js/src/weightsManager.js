import { CACHE_DIR_NAME, HF_BASE_URL, HF_REPO } from "./constants.js";

/**
 * Resolves model weights by name, mirroring the Python `WeightsManager`:
 * existing path → cache → download from `baseUrl` (the Hugging Face repo by
 * default).
 *
 * Node: downloads are cached on disk under `~/.mlconfgen_weights`. The cache
 * directory is shared with the Python package, and so is the `.lock` file
 * guarding a download, so the two can run side by side.
 *
 * Browser: a bare file name is fetched from `baseUrl` and cached with the
 * Cache API (HTTPS or localhost only; the browser may evict it). Without the
 * Cache API it is fetched on every call. A relative path or URL is passed to
 * onnxruntime-web unchanged, i.e. self-hosted weights work exactly as before.
 *
 * Node built-ins are imported lazily so the module stays bundler-safe.
 */

const IS_NODE =
  typeof process !== "undefined" && process.versions?.node != null;

/** File names that may be looked up in the cache or downloaded. */
const BARE_NAME = /^[\w.-]+\.onnx$/;

const LOCK_POLL_MS = 200;

export const DEFAULT_BASE_URL = `${HF_BASE_URL}/${HF_REPO}/resolve/main`;
export const DEFAULT_CACHE_NAME = "mlconfgen-weights";

let nodeModulesPromise = null;

function nodeModules() {
  if (!nodeModulesPromise) {
    nodeModulesPromise = Promise.all([
      import("node:fs/promises"),
      import("node:path"),
      import("node:os"),
    ]).then(([fs, path, os]) => ({ fs, path, os }));
  }
  return nodeModulesPromise;
}

/** `true` when `name` is a plain weights file name with no path component. */
export function isBareWeightsName(name) {
  return typeof name === "string" && BARE_NAME.test(name);
}

export class WeightsManager {
  /**
   * @param {object} [options]
   * @param {string} [options.cacheDir] cache directory (Node). Defaults to the
   *   `MLCONFGEN_CACHE` environment variable, then `~/.mlconfgen_weights`.
   * @param {string} [options.baseUrl] where bare names are downloaded from;
   *   defaults to the Hugging Face repo
   * @param {string} [options.cacheName="mlconfgen-weights"] Cache API cache
   *   name (browser)
   * @param {(p: { file: string, loaded: number, total: number|null }) => void} [options.onProgress]
   *   called as download bytes arrive; `total` is null when the size is unknown
   * @param {number} [options.lockTimeout=60000] ms to wait for another
   *   process's download of the same file before giving up (Node)
   * @param {boolean} [options.isNode] override runtime detection, e.g. to
   *   exercise the browser behaviour under Node
   */
  constructor({
    cacheDir = null,
    baseUrl = DEFAULT_BASE_URL,
    cacheName = DEFAULT_CACHE_NAME,
    onProgress = null,
    lockTimeout = 60_000,
    isNode = IS_NODE,
  } = {}) {
    this.isNode = isNode;
    this.cacheDir = cacheDir;
    this.baseUrl = baseUrl.replace(/\/+$/, "");
    this.cacheName = cacheName;
    this.onProgress = onProgress;
    this.lockTimeout = lockTimeout;
  }

  /** Download URL of a weights file. */
  weightsUrl(name) {
    return `${this.baseUrl}/${encodeURIComponent(name)}`;
  }

  /** Absolute cache directory (Node only). */
  async getCacheDir() {
    if (!this.isNode) {
      throw new Error("The weights cache directory is only available in Node.");
    }
    const { path, os } = await nodeModules();
    const dir =
      this.cacheDir ||
      process.env.MLCONFGEN_CACHE ||
      path.join(os.homedir(), CACHE_DIR_NAME);
    return path.resolve(dir);
  }

  /**
   * Resolve weights to something `ort.InferenceSession.create` accepts.
   *
   * Byte buffers are returned unchanged. In Node a string resolves to a file
   * path: an existing file is used as is, a bare file name is looked up in the
   * cache and downloaded on a miss, anything else is an error. In the browser
   * a bare file name resolves to its bytes; other strings pass through.
   *
   * @param {string|Uint8Array|ArrayBuffer} nameOrPath
   * @param {object} [options]
   * @param {boolean} [options.forceDownload=false] re-download even if cached
   * @returns {Promise<string|Uint8Array|ArrayBuffer>}
   */
  async resolve(nameOrPath, { forceDownload = false } = {}) {
    if (typeof nameOrPath !== "string") return nameOrPath;
    if (!this.isNode) return this.#resolveBrowser(nameOrPath, forceDownload);

    const { fs, path } = await nodeModules();

    if (await isFile(fs, nameOrPath)) return nameOrPath;

    if (!isBareWeightsName(nameOrPath)) {
      throw new Error(`Weights file not found: ${nameOrPath}`);
    }

    const cacheDir = await this.getCacheDir();
    await fs.mkdir(cacheDir, { recursive: true });

    const cachePath = path.join(cacheDir, nameOrPath);
    if (!forceDownload && (await isFile(fs, cachePath))) return cachePath;

    // Same lock name as Python (`Path.with_suffix(".lock")`).
    const lockPath = path.join(cacheDir, `${stripSuffix(nameOrPath)}.lock`);
    await this.#withLock(fs, lockPath, async () => {
      // Another process may have finished the download while we waited.
      if (!forceDownload && (await isFile(fs, cachePath))) return;
      await this.#download(fs, nameOrPath, cachePath);
    });

    return cachePath;
  }

  /**
   * List weights files on Hugging Face and in the local cache.
   *
   * `remote` always lists the Hugging Face repo, regardless of `baseUrl`.
   * A failed Hugging Face request does not hide the local list: `remote` is
   * then empty and `remoteError` holds the error, so callers can decide
   * whether it matters.
   *
   * `local` holds paths relative to the cache directory in Node, and the file
   * names stored in the `cacheName` Cache API cache in the browser.
   *
   * @param {object} [options]
   * @param {Iterable<string>} [options.suffixes=[".pt", ".onnx"]]
   * @returns {Promise<{ remote: string[], local: string[], remoteError?: Error }>}
   */
  async listAvailableWeights({ suffixes = [".pt", ".onnx"] } = {}) {
    const wanted = new Set([...suffixes].map((s) => s.toLowerCase()));
    const keep = (name) => wanted.has(suffixOf(name));

    const out = { remote: [], local: [] };

    const url = `${HF_BASE_URL}/api/models/${HF_REPO}`;
    try {
      const res = await getFetch()(url);
      if (!res.ok) {
        throw new Error(
          `Could not list weights at ${url}: HTTP ${res.status} ${res.statusText}`,
        );
      }
      const info = await res.json();
      out.remote = (info.siblings ?? []).map((s) => s.rfilename).filter(keep);
    } catch (err) {
      out.remoteError = err;
    }

    if (this.isNode) {
      const { fs, path } = await nodeModules();
      const cacheDir = await this.getCacheDir();
      out.local = (await walkFiles(fs, path, cacheDir))
        .map((p) => path.relative(cacheDir, p))
        .filter(keep);
    } else {
      const cache = await this.#openBrowserCache();
      if (cache) {
        out.local = (await cache.keys())
          .map((req) => lastPathSegment(req.url))
          .filter(keep);
      }
    }

    return out;
  }

  /**
   * Delete every cached weights file: the cache directory in Node, the
   * `cacheName` Cache API cache in the browser.
   */
  async clearCache() {
    if (!this.isNode) {
      if (typeof globalThis.caches?.delete === "function") {
        await globalThis.caches.delete(this.cacheName);
      }
      return;
    }
    const { fs } = await nodeModules();
    const cacheDir = await this.getCacheDir();
    await fs.rm(cacheDir, { recursive: true, force: true });
    await fs.mkdir(cacheDir, { recursive: true });
  }

  async #resolveBrowser(nameOrPath, forceDownload) {
    if (!isBareWeightsName(nameOrPath)) return nameOrPath;

    const url = this.weightsUrl(nameOrPath);
    const cache = await this.#openBrowserCache();

    if (cache && !forceDownload) {
      const hit = await cache.match(url);
      if (hit) return new Uint8Array(await hit.arrayBuffer());
    }

    const res = await this.#fetchWeights(nameOrPath);
    // Caching is best effort: a failed put (quota, private mode) must not
    // fail the load.
    const stored = cache
      ? cache.put(url, res.clone()).catch(() => {})
      : Promise.resolve();

    const chunks = [];
    let size = 0;
    for await (const chunk of this.#readChunks(nameOrPath, res)) {
      chunks.push(chunk);
      size += chunk.length;
    }
    await stored;

    const out = new Uint8Array(size);
    let offset = 0;
    for (const chunk of chunks) {
      out.set(chunk, offset);
      offset += chunk.length;
    }
    return out;
  }

  /** The Cache API cache, or null where it is unavailable. */
  async #openBrowserCache() {
    if (typeof globalThis.caches?.open !== "function") return null;
    try {
      return await globalThis.caches.open(this.cacheName);
    } catch {
      return null;
    }
  }

  /** Stream to `<name>.part`, then rename so readers never see a partial file. */
  async #download(fs, name, cachePath) {
    const partPath = `${cachePath}.part`;
    let handle = null;
    try {
      handle = await fs.open(partPath, "w");
      const res = await this.#fetchWeights(name);
      for await (const chunk of this.#readChunks(name, res)) {
        await handle.write(chunk);
      }
      await handle.close();
      handle = null;
      await fs.rename(partPath, cachePath);
    } catch (err) {
      if (handle) await handle.close().catch(() => {});
      await fs.rm(partPath, { force: true });
      throw new Error(
        `Could not resolve path to the weights file: ${name}: ${err.message}`,
        { cause: err },
      );
    }
  }

  async #fetchWeights(name) {
    const url = this.weightsUrl(name);
    const res = await getFetch()(url);
    if (!res.ok || !res.body) {
      throw new Error(`HTTP ${res.status} ${res.statusText} for ${url}`);
    }
    return res;
  }

  async *#readChunks(name, res) {
    const total = Number(res.headers.get("content-length")) || null;
    let loaded = 0;
    let finished = false;
    const reader = res.body.getReader();
    try {
      for (;;) {
        const { done, value } = await reader.read();
        if (done) break;
        loaded += value.length;
        this.onProgress?.({ file: name, loaded, total });
        yield value;
      }
      finished = true;
    } finally {
      // Stop the transfer if the consumer bailed out mid-download.
      if (!finished) await reader.cancel().catch(() => {});
      reader.releaseLock();
    }
  }

  async #withLock(fs, lockPath, fn) {
    const start = Date.now();
    let handle;
    for (;;) {
      try {
        handle = await fs.open(lockPath, "wx");
        break;
      } catch (err) {
        if (err.code !== "EEXIST") throw err;
      }
      if (Date.now() - start > this.lockTimeout) {
        const err = new Error(
          `Timeout waiting for lock ${lockPath} — delete it if no other download is running.`,
        );
        err.name = "TimeoutError";
        throw err;
      }
      await new Promise((r) => setTimeout(r, LOCK_POLL_MS));
    }

    try {
      await handle.close();
      return await fn();
    } finally {
      await fs.rm(lockPath, { force: true });
    }
  }
}

function getFetch() {
  if (typeof globalThis.fetch !== "function") {
    throw new Error("Downloading weights requires a global `fetch`.");
  }
  return globalThis.fetch;
}

async function isFile(fs, p) {
  try {
    return (await fs.stat(p)).isFile();
  } catch {
    return false;
  }
}

async function walkFiles(fs, path, dir) {
  let entries;
  try {
    entries = await fs.readdir(dir, { withFileTypes: true });
  } catch (err) {
    if (err.code === "ENOENT") return [];
    throw err;
  }
  const out = [];
  for (const entry of entries) {
    const full = path.join(dir, entry.name);
    if (entry.isDirectory()) out.push(...(await walkFiles(fs, path, full)));
    else if (entry.isFile()) out.push(full);
  }
  return out;
}

/** Lower-cased extension including the dot (`Path.suffix`), or "". */
function suffixOf(name) {
  const base = name.slice(name.search(/[^\\/]*$/));
  const dot = base.lastIndexOf(".");
  return dot > 0 ? base.slice(dot).toLowerCase() : "";
}

/** Decoded file name at the end of a URL. */
function lastPathSegment(url) {
  const { pathname } = new URL(url);
  return decodeURIComponent(pathname.slice(pathname.lastIndexOf("/") + 1));
}

function stripSuffix(name) {
  const dot = name.lastIndexOf(".");
  return dot > 0 ? name.slice(0, dot) : name;
}
