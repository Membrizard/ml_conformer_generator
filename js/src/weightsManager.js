import { CACHE_DIR_NAME, HF_BASE_URL, HF_REPO } from "./constants.js";

/**
 * Resolves model weights by name, mirroring the Python `WeightsManager`:
 * existing path → cache (`~/.mlconfgen_weights/<name>`) → download from the
 * Hugging Face repo.
 *
 * Node: downloads are cached on disk. The cache directory is shared with the
 * Python package, and so is the `.lock` file guarding a download, so the two
 * can run side by side.
 *
 * Browser: a bare file name is fetched from Hugging Face on every call and
 * returned as bytes — there is no persistent cache. Any other string (a URL
 * or a relative path) is passed through for the ONNX Runtime to load.
 *
 * Node built-ins are imported lazily so the module stays bundler-safe.
 */

const IS_NODE =
  typeof process !== "undefined" && process.versions?.node != null;

/** File names that may be looked up in the cache or downloaded. */
const BARE_NAME = /^[\w.-]+\.onnx$/;

const LOCK_POLL_MS = 200;

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

/** Download URL of a weights file in the Hugging Face repo. */
export function weightsUrl(name) {
  return `${HF_BASE_URL}/${HF_REPO}/resolve/main/${encodeURIComponent(name)}`;
}

export class WeightsManager {
  /**
   * @param {object} [options]
   * @param {string} [options.cacheDir] cache directory (Node). Defaults to the
   *   `MLCONFGEN_CACHE` environment variable, then `~/.mlconfgen_weights`.
   * @param {(p: { file: string, loaded: number, total: number|null }) => void} [options.onProgress]
   *   called as download bytes arrive; `total` is null when the size is unknown
   * @param {number} [options.lockTimeout=60000] ms to wait for another
   *   process's download of the same file before giving up
   */
  constructor({ cacheDir = null, onProgress = null, lockTimeout = 60_000 } = {}) {
    this.cacheDir = cacheDir;
    this.onProgress = onProgress;
    this.lockTimeout = lockTimeout;
  }

  /** Absolute cache directory (Node only). */
  async getCacheDir() {
    if (!IS_NODE) {
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
   * cache and downloaded on a miss, anything else is an error.
   *
   * @param {string|Uint8Array|ArrayBuffer} nameOrPath
   * @param {object} [options]
   * @param {boolean} [options.forceDownload=false] re-download even if cached
   * @returns {Promise<string|Uint8Array|ArrayBuffer>}
   */
  async resolve(nameOrPath, { forceDownload = false } = {}) {
    if (typeof nameOrPath !== "string") return nameOrPath;
    if (!IS_NODE) return this.#resolveBrowser(nameOrPath);

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
   * @param {object} [options]
   * @param {Iterable<string>} [options.suffixes=[".pt", ".onnx"]]
   * @returns {Promise<{ remote: string[], local: string[] }>} `local` is
   *   relative to the cache directory, and always empty in the browser
   */
  async listAvailableWeights({ suffixes = [".pt", ".onnx"] } = {}) {
    const wanted = new Set([...suffixes].map((s) => s.toLowerCase()));
    const keep = (name) => wanted.has(suffixOf(name));

    const url = `${HF_BASE_URL}/api/models/${HF_REPO}`;
    const res = await getFetch()(url);
    if (!res.ok) {
      throw new Error(
        `Could not list weights at ${url}: HTTP ${res.status} ${res.statusText}`,
      );
    }
    const info = await res.json();
    const remote = (info.siblings ?? [])
      .map((s) => s.rfilename)
      .filter(keep);

    let local = [];
    if (IS_NODE) {
      const { fs, path } = await nodeModules();
      const cacheDir = await this.getCacheDir();
      local = (await walkFiles(fs, path, cacheDir))
        .map((p) => path.relative(cacheDir, p))
        .filter(keep);
    }

    return { remote, local };
  }

  /** Delete every cached weights file (Node). No-op in the browser. */
  async clearCache() {
    if (!IS_NODE) return;
    const { fs } = await nodeModules();
    const cacheDir = await this.getCacheDir();
    await fs.rm(cacheDir, { recursive: true, force: true });
    await fs.mkdir(cacheDir, { recursive: true });
  }

  async #resolveBrowser(nameOrPath) {
    if (!isBareWeightsName(nameOrPath)) return nameOrPath;
    const chunks = [];
    let size = 0;
    for await (const chunk of this.#fetchChunks(nameOrPath)) {
      chunks.push(chunk);
      size += chunk.length;
    }
    const out = new Uint8Array(size);
    let offset = 0;
    for (const chunk of chunks) {
      out.set(chunk, offset);
      offset += chunk.length;
    }
    return out;
  }

  /** Stream to `<name>.part`, then rename so readers never see a partial file. */
  async #download(fs, name, cachePath) {
    const partPath = `${cachePath}.part`;
    let handle = null;
    try {
      handle = await fs.open(partPath, "w");
      for await (const chunk of this.#fetchChunks(name)) {
        await handle.write(chunk);
      }
      await handle.close();
      handle = null;
      await fs.rename(partPath, cachePath);
    } catch (err) {
      if (handle) await handle.close().catch(() => {});
      await fs.rm(partPath, { force: true });
      throw new Error(`Could not resolve path to the weights file: ${name}`, {
        cause: err,
      });
    }
  }

  async *#fetchChunks(name) {
    const url = weightsUrl(name);
    const res = await getFetch()(url);
    if (!res.ok || !res.body) {
      throw new Error(`HTTP ${res.status} ${res.statusText} for ${url}`);
    }
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

function stripSuffix(name) {
  const dot = name.lastIndexOf(".");
  return dot > 0 ? name.slice(0, dot) : name;
}
