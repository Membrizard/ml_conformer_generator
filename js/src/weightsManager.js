import { CACHE_DIR_NAME, HF_BASE_URL, HF_REPO } from "./constants.js";

/**
 * Resolves model weights by name, in the same order as the Python
 * `WeightsManager`: existing path → cache → download from `baseUrl` (the
 * Hugging Face repo by default).
 *
 * Node: downloads are cached on disk under `~/.mlconfgen_weights`. The cache
 * directory is shared with the Python package, and so is the `.lock` file
 * guarding a download, so the two can run side by side.
 *
 * Browser: a bare file name is fetched from `baseUrl` and cached with the
 * Cache API (HTTPS or localhost only; the browser may evict it). Without the
 * Cache API it is fetched on every call. A bare name is not page-relative:
 * paths with a directory part (`./…`, `/…`) and URLs are passed to
 * onnxruntime-web unchanged.
 *
 * Node built-ins are imported dynamically, inside Node-only code, as in
 * `rdkit.js`. esbuild browser builds need `--external:node:*`.
 */

const IS_NODE =
  typeof process !== "undefined" && process.versions?.node != null;

/** File names that may be looked up in the cache or downloaded. */
const BARE_NAME = /^[\w.-]+\.onnx$/;

const LOCK_POLL_MS = 200;

const DEFAULT_BASE_URL = `${HF_BASE_URL}/${HF_REPO}/resolve/main`;
const DEFAULT_CACHE_NAME = "mlconfgen-weights";

let nodeModulesPromise = null;

function nodeModules() {
  if (!nodeModulesPromise) {
    // The comments keep webpack and Vite from resolving these for browser
    // builds; this code only runs in Node.
    nodeModulesPromise = Promise.all([
      import(/* webpackIgnore: true */ /* @vite-ignore */ "node:fs/promises"),
      import(/* webpackIgnore: true */ /* @vite-ignore */ "node:path"),
      import(/* webpackIgnore: true */ /* @vite-ignore */ "node:os"),
    ]).then(([fs, path, os]) => ({ fs, path, os }));
  }
  return nodeModulesPromise;
}

/** `true` when `name` is a plain weights file name with no path component. */
function isBareWeightsName(name) {
  return typeof name === "string" && BARE_NAME.test(name);
}

export class WeightsManager {
  #warnedCachePut = false;

  /**
   * @param {object} [options]
   * @param {string} [options.cacheDir] cache directory (Node). Defaults to the
   *   `MLCONFGEN_CACHE` environment variable, then `~/.mlconfgen_weights`.
   * @param {string} [options.baseUrl] where bare names are downloaded from;
   *   defaults to the Hugging Face repo. A mirror must serve the files
   *   uncompressed with `Content-Length` (`Content-Encoding` is not visible
   *   cross-origin, so a compressed body fails the size check).
   * @param {string} [options.cacheName="mlconfgen-weights"] Cache API cache
   *   name (browser)
   * @param {(p: { file: string, loaded: number, total: number|null }) => void} [options.onProgress]
   *   called as download bytes arrive; `total` is null when the size is unknown
   * @param {number} [options.lockTimeout=60000] ms to wait for another
   *   process's download of the same file before giving up (Node). The wait
   *   restarts whenever that download makes progress.
   */
  constructor({
    cacheDir = null,
    baseUrl = DEFAULT_BASE_URL,
    cacheName = DEFAULT_CACHE_NAME,
    onProgress = null,
    lockTimeout = 60_000,
    _isNode = IS_NODE, // test hook: exercise the browser path under Node
  } = {}) {
    this._isNode = _isNode;
    this.cacheDir = cacheDir;
    this.baseUrl = baseUrl.replace(/\/+$/, "");
    this.cacheName = cacheName;
    this.onProgress = onProgress;
    this.lockTimeout = lockTimeout;
  }

  /** Download URL of a weights file. */
  #weightsUrl(name) {
    return `${this.baseUrl}/${encodeURIComponent(name)}`;
  }

  /** Absolute cache directory (Node only). */
  async getCacheDir() {
    if (!this._isNode) {
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
    if (!this._isNode) return this.#resolveBrowser(nameOrPath, forceDownload);

    const { fs, path, os } = await nodeModules();

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
    const partPath = `${cachePath}.part`;
    await this.#withLock(fs, os.hostname(), lockPath, partPath, async () => {
      // Another process may have finished the download while we waited.
      if (!forceDownload && (await isFile(fs, cachePath))) return;
      await this.#download(fs, nameOrPath, cachePath, partPath);
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

    if (this._isNode) {
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
   *
   * In Node this refuses to delete a cache directory that is the filesystem
   * root, the home directory or the working directory.
   */
  async clearCache() {
    if (!this._isNode) {
      if (typeof globalThis.caches?.delete === "function") {
        await globalThis.caches.delete(this.cacheName);
      }
      return;
    }
    const { fs, path, os } = await nodeModules();
    const cacheDir = await this.getCacheDir();
    const target = await realpathOr(fs, cacheDir);
    const guarded = await Promise.all(
      [path.parse(target).root, os.homedir(), process.cwd()].map((p) =>
        realpathOr(fs, path.resolve(p)),
      ),
    );
    if (guarded.includes(target)) {
      throw new Error(
        `Refusing to clear ${cacheDir}: it is the filesystem root, the home directory or the working directory.`,
      );
    }
    await fs.rm(cacheDir, { recursive: true, force: true });
    await fs.mkdir(cacheDir, { recursive: true });
  }

  async #resolveBrowser(nameOrPath, forceDownload) {
    if (!isBareWeightsName(nameOrPath)) return nameOrPath;

    const url = this.#weightsUrl(nameOrPath);
    const cache = await this.#openBrowserCache();

    if (cache && !forceDownload) {
      const hit = await cache.match(url);
      if (hit) return new Uint8Array(await hit.arrayBuffer());
    }

    try {
      return await this.#fetchBytes(nameOrPath, url, cache);
    } catch (err) {
      throw downloadError(nameOrPath, err);
    }
  }

  async #fetchBytes(name, url, cache) {
    const res = await this.#fetchWeights(name);
    // Caching is best effort: a failed put (quota, private mode) must not
    // fail the load.
    const stored = cache
      ? cache.put(url, res.clone()).then(
          () => null,
          (err) => err,
        )
      : Promise.resolve(null);

    // With a known size, chunks are copied into one buffer as they arrive,
    // so the weights are not held twice.
    const total = contentLength(res);
    const out = total === null ? null : new Uint8Array(total);
    const chunks = [];
    let size = 0;
    for await (const chunk of this.#readChunks(name, res)) {
      if (out) out.set(chunk, size);
      else chunks.push(chunk);
      size += chunk.length;
    }

    const putError = await stored;
    if (putError && !this.#warnedCachePut) {
      this.#warnedCachePut = true;
      console.warn(
        `mlconfgen: could not store ${name} in the "${this.cacheName}" cache ` +
          `(${putError.name}: ${putError.message}); it will be downloaded again next time.`,
      );
    }

    return out ?? concat(chunks, size);
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
  async #download(fs, name, cachePath, partPath) {
    let handle = null;
    try {
      // We hold the lock, so whatever is at `.part` is a leftover. Remove it
      // and create the file exclusively, so a planted symlink is never
      // followed.
      await fs.rm(partPath, { force: true });
      handle = await fs.open(partPath, "wx");
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
      throw downloadError(name, err);
    }
  }

  async #fetchWeights(name) {
    const url = this.#weightsUrl(name);
    const res = await getFetch()(url);
    if (!res.ok || !res.body) {
      throw new Error(`HTTP ${res.status} ${res.statusText} for ${url}`);
    }
    return res;
  }

  async *#readChunks(name, res) {
    const total = contentLength(res);
    let loaded = 0;
    let finished = false;
    const reader = res.body.getReader();
    try {
      for (;;) {
        const { done, value } = await reader.read();
        if (done) break;
        loaded += value.length;
        if (total !== null && loaded > total) break;
        this.onProgress?.({ file: name, loaded, total });
        yield value;
      }
      if (total !== null && loaded !== total) {
        throw new Error(
          `Size mismatch for ${name}: expected ${total} bytes, received ${loaded}.`,
        );
      }
      finished = true;
    } finally {
      // Stop the transfer if the consumer bailed out mid-download.
      if (!finished) await reader.cancel().catch(() => {});
      reader.releaseLock();
    }
  }

  /**
   * Run `fn` holding `lockPath`. The lock holds `<pid>@<hostname>`, so a lock
   * left by a killed process on this host can be recognised and removed.
   * Locks from other hosts and Python's empty locks are waited on. The wait
   * times out after `lockTimeout` ms without progress on `partPath`, the file
   * the holder downloads into.
   */
  async #withLock(fs, hostname, lockPath, partPath, fn) {
    let start = Date.now();
    let lastPartMtime = null;
    let handle;
    for (;;) {
      try {
        handle = await fs.open(lockPath, "wx");
        break;
      } catch (err) {
        if (err.code !== "EEXIST") throw err;
      }
      if (await isStaleLock(fs, lockPath, hostname)) {
        // Move the lock aside before deleting it. ENOENT means another
        // waiter reclaimed it first.
        const stalePath = `${lockPath}.stale-${process.pid}`;
        try {
          await fs.rename(lockPath, stalePath);
          await fs.rm(stalePath, { force: true });
        } catch (err) {
          if (err.code !== "ENOENT") throw err;
        }
        continue;
      }
      const partMtime = await mtimeOf(fs, partPath);
      if (partMtime !== null && partMtime !== lastPartMtime) start = Date.now();
      lastPartMtime = partMtime;
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
      try {
        await handle.writeFile(`${process.pid}@${hostname}`);
      } finally {
        await handle.close();
      }
      return await fn();
    } finally {
      await fs.rm(lockPath, { force: true });
    }
  }
}

function downloadError(name, cause) {
  return new Error(
    `Could not resolve path to the weights file: ${name}: ${cause.message}`,
    { cause },
  );
}

/**
 * Body size from `content-length`, or null when unknown. A compressed body is
 * decoded by `fetch`, so its `content-length` does not match what we read.
 */
function contentLength(res) {
  const encoding = res.headers.get("content-encoding");
  if (encoding && encoding !== "identity") return null;
  return Number(res.headers.get("content-length")) || null;
}

function concat(chunks, size) {
  const out = new Uint8Array(size);
  let offset = 0;
  for (const chunk of chunks) {
    out.set(chunk, offset);
    offset += chunk.length;
  }
  return out;
}

/**
 * `true` when the lock holds the PID of a process on this host that no longer
 * exists. A bare `<pid>` is taken to be from this host.
 */
async function isStaleLock(fs, lockPath, hostname) {
  let text;
  try {
    text = (await fs.readFile(lockPath, "utf8")).trim();
  } catch {
    return false;
  }
  const match = /^(\d+)(?:@(.+))?$/.exec(text);
  if (!match || (match[2] !== undefined && match[2] !== hostname)) return false;
  try {
    process.kill(Number(match[1]), 0);
    return false;
  } catch (err) {
    return err.code === "ESRCH";
  }
}

async function mtimeOf(fs, p) {
  try {
    return (await fs.stat(p)).mtimeMs;
  } catch {
    return null;
  }
}

async function realpathOr(fs, p) {
  try {
    return await fs.realpath(p);
  } catch {
    return p;
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
