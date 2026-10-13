/**
 * WeightsManager unit tests. Fully offline: `globalThis.fetch` (and, for the
 * browser path, `globalThis.caches`) are stubbed, and every test gets its own
 * temporary cache directory. Node tests run from an empty working directory,
 * since a weights file in the cwd would otherwise win over the cache.
 */
import assert from "node:assert/strict";
import { existsSync } from "node:fs";
import { mkdir, mkdtemp, readdir, readFile, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, beforeEach, describe, it } from "node:test";

import { WeightsManager } from "../../src/weightsManager.js";

const HF_RESOLVE =
  "https://huggingface.co/Membrizard/ml_conformer_generator/resolve/main";
const HF_API = "https://huggingface.co/api/models/Membrizard/ml_conformer_generator";
const EGNN = "egnn_chembl_15_39.onnx";
const BYTES = [1, 2, 3, 4];

const realFetch = globalThis.fetch;
const realCaches = globalThis.caches;
const realEnvCache = process.env.MLCONFGEN_CACHE;

/** JSON response without `Response.json` (missing on older Node 18). */
function jsonResponse(body) {
  return new Response(JSON.stringify(body), {
    headers: { "content-type": "application/json" },
  });
}

function bytesResponse(bytes = BYTES) {
  return new Response(new Uint8Array(bytes), {
    headers: { "content-length": String(bytes.length) },
  });
}

const API_LISTING = {
  siblings: [
    { rfilename: EGNN },
    { rfilename: "small_egnn_chembl_15_39.pt" },
    { rfilename: "README.md" },
  ],
};

/**
 * Install a fetch stub. `handler(url)` returns a Response; by default the HF
 * API returns `API_LISTING` and anything else returns `BYTES`.
 * @returns {string[]} the URLs fetched, in order
 */
function stubFetch(handler = null) {
  const calls = [];
  globalThis.fetch = async (url) => {
    calls.push(url);
    if (handler) return handler(url);
    if (url === HF_API) return jsonResponse(API_LISTING);
    return bytesResponse();
  };
  return calls;
}

/** Map-backed stand-in for the Cache API (`open/match/put/keys/delete`). */
function stubCaches() {
  const stores = new Map();
  globalThis.caches = {
    async open(name) {
      if (!stores.has(name)) stores.set(name, new Map());
      const store = stores.get(name);
      return {
        async match(url) {
          const body = store.get(url);
          return body ? new Response(body) : undefined;
        },
        async put(url, res) {
          store.set(url, new Uint8Array(await res.arrayBuffer()));
        },
        async keys() {
          return [...store.keys()].map((u) => new Request(u));
        },
      };
    },
    async delete(name) {
      return stores.delete(name);
    },
  };
  return stores;
}

function restoreGlobals() {
  globalThis.fetch = realFetch;
  if (realCaches === undefined) delete globalThis.caches;
  else globalThis.caches = realCaches;
  if (realEnvCache === undefined) delete process.env.MLCONFGEN_CACHE;
  else process.env.MLCONFGEN_CACHE = realEnvCache;
}

describe("WeightsManager (Node)", () => {
  const originalCwd = process.cwd();
  let dir;
  let cwd;

  beforeEach(async () => {
    dir = await mkdtemp(join(tmpdir(), "mlconfgen-wm-"));
    cwd = await mkdtemp(join(tmpdir(), "mlconfgen-cwd-"));
    process.chdir(cwd);
  });

  afterEach(async () => {
    process.chdir(originalCwd);
    restoreGlobals();
    await rm(dir, { recursive: true, force: true });
    await rm(cwd, { recursive: true, force: true });
  });

  it("downloads a bare name on a cache miss from the Hugging Face URL", async () => {
    const calls = stubFetch();
    const wm = new WeightsManager({ cacheDir: dir });

    const resolved = await wm.resolve(EGNN);

    assert.equal(resolved, join(dir, EGNN));
    assert.deepEqual(calls, [`${HF_RESOLVE}/${EGNN}`]);
    assert.deepEqual([...(await readFile(resolved))], BYTES);
  });

  it("leaves no .part or .lock file behind after a download", async () => {
    stubFetch();
    await new WeightsManager({ cacheDir: dir }).resolve(EGNN);
    assert.deepEqual(await readdir(dir), [EGNN]);
  });

  it("writes to <name>.part and renames on completion", async () => {
    stubFetch();
    const seen = [];
    const wm = new WeightsManager({
      cacheDir: dir,
      onProgress: () => {
        seen.push({
          part: existsSync(join(dir, `${EGNN}.part`)),
          final: existsSync(join(dir, EGNN)),
        });
      },
    });

    await wm.resolve(EGNN);

    assert.ok(seen.length > 0);
    for (const s of seen) assert.deepEqual(s, { part: true, final: false });
    assert.ok(existsSync(join(dir, EGNN)));
  });

  it("holds a Python-compatible lock file (<stem>.lock) while downloading", async () => {
    let lockSeen = null;
    stubFetch(() => {
      lockSeen = existsSync(join(dir, "egnn_chembl_15_39.lock"));
      return bytesResponse();
    });
    await new WeightsManager({ cacheDir: dir }).resolve(EGNN);
    assert.equal(lockSeen, true);
  });

  it("reports progress with file, loaded and total", async () => {
    stubFetch();
    const progress = [];
    const wm = new WeightsManager({
      cacheDir: dir,
      onProgress: (p) => progress.push(p),
    });
    await wm.resolve(EGNN);
    assert.deepEqual(progress.at(-1), { file: EGNN, loaded: 4, total: 4 });
  });

  it("fetches once for concurrent resolves of the same name", async () => {
    const calls = stubFetch(async () => {
      await new Promise((r) => setTimeout(r, 50));
      return bytesResponse();
    });
    const wm = new WeightsManager({ cacheDir: dir });

    const [a, b] = await Promise.all([wm.resolve(EGNN), wm.resolve(EGNN)]);

    assert.equal(a, join(dir, EGNN));
    assert.equal(a, b);
    assert.equal(calls.length, 1);
  });

  it("uses the cache on a hit without fetching", async () => {
    const calls = stubFetch();
    const wm = new WeightsManager({ cacheDir: dir });
    await wm.resolve(EGNN);
    await wm.resolve(EGNN);
    assert.equal(calls.length, 1);
  });

  it("re-downloads with forceDownload", async () => {
    const calls = stubFetch();
    const wm = new WeightsManager({ cacheDir: dir });
    await wm.resolve(EGNN);
    await wm.resolve(EGNN, { forceDownload: true });
    assert.equal(calls.length, 2);
  });

  it("includes the cause in the error of a failed download", async () => {
    stubFetch(() => new Response("nope", { status: 404, statusText: "Not Found" }));
    const wm = new WeightsManager({ cacheDir: dir });
    await assert.rejects(wm.resolve("missing.onnx"), (err) => {
      assert.match(
        err.message,
        /^Could not resolve path to the weights file: missing\.onnx: HTTP 404 Not Found/,
      );
      assert.ok(err.cause instanceof Error);
      return true;
    });
  });

  it("leaves nothing behind after a failed download", async () => {
    stubFetch(() => new Response("nope", { status: 404, statusText: "Not Found" }));
    await assert.rejects(new WeightsManager({ cacheDir: dir }).resolve("missing.onnx"));
    assert.deepEqual(await readdir(dir), []);
  });

  it("removes the partial file when the stream fails mid-download", async () => {
    stubFetch(() => {
      let sent = false;
      const body = new ReadableStream({
        pull(controller) {
          if (!sent) {
            sent = true;
            controller.enqueue(new Uint8Array([1, 2]));
          } else {
            controller.error(new Error("connection reset"));
          }
        },
      });
      return new Response(body);
    });
    await assert.rejects(
      new WeightsManager({ cacheDir: dir }).resolve(EGNN),
      /connection reset/,
    );
    assert.deepEqual(await readdir(dir), []);
  });

  it("throws for a missing path that is not a bare name, without fetching", async () => {
    const calls = stubFetch();
    const wm = new WeightsManager({ cacheDir: dir });
    await assert.rejects(wm.resolve("./nowhere/egnn.onnx"), /Weights file not found/);
    assert.equal(calls.length, 0);
  });

  it("rejects names that would escape the cache directory", async () => {
    const calls = stubFetch();
    const wm = new WeightsManager({ cacheDir: dir });
    await assert.rejects(wm.resolve("../evil.onnx"), /Weights file not found/);
    assert.equal(calls.length, 0);
  });

  it("returns an existing path as is", async () => {
    const calls = stubFetch();
    const existing = join(dir, "local.onnx");
    await writeFile(existing, "x");
    assert.equal(await new WeightsManager({ cacheDir: dir }).resolve(existing), existing);
    assert.equal(calls.length, 0);
  });

  it("returns byte buffers unchanged", async () => {
    const bytes = new Uint8Array([9]);
    assert.equal(await new WeightsManager({ cacheDir: dir }).resolve(bytes), bytes);
  });

  it("times out on a lock held by another process", async () => {
    const calls = stubFetch();
    await writeFile(join(dir, "egnn_chembl_15_39.lock"), "");
    const wm = new WeightsManager({ cacheDir: dir, lockTimeout: 300 });
    await assert.rejects(wm.resolve(EGNN), (err) => {
      assert.equal(err.name, "TimeoutError");
      assert.match(err.message, /egnn_chembl_15_39\.lock/);
      return true;
    });
    assert.equal(calls.length, 0);
  });

  it("proceeds once a held lock is released", async () => {
    const calls = stubFetch();
    const lock = join(dir, "egnn_chembl_15_39.lock");
    await writeFile(lock, "");
    setTimeout(() => rm(lock), 300);

    const resolved = await new WeightsManager({ cacheDir: dir }).resolve(EGNN);

    assert.equal(resolved, join(dir, EGNN));
    assert.equal(calls.length, 1);
  });

  it("uses the file another process downloaded while it waited for the lock", async () => {
    const calls = stubFetch();
    const lock = join(dir, "egnn_chembl_15_39.lock");
    await writeFile(lock, "");
    setTimeout(async () => {
      await writeFile(join(dir, EGNN), "done");
      await rm(lock);
    }, 300);

    const resolved = await new WeightsManager({ cacheDir: dir }).resolve(EGNN);

    assert.equal(await readFile(resolved, "utf8"), "done");
    assert.equal(calls.length, 0);
  });

  it("downloads from a custom baseUrl", async () => {
    const calls = stubFetch();
    const wm = new WeightsManager({ cacheDir: dir, baseUrl: "https://mirror.example/w/" });
    await wm.resolve("a.onnx");
    assert.deepEqual(calls, ["https://mirror.example/w/a.onnx"]);
  });

  it("lists remote and local weights filtered by suffix", async () => {
    stubFetch();
    const wm = new WeightsManager({ cacheDir: dir });
    await wm.resolve(EGNN);
    await mkdir(join(dir, "nested"));
    await writeFile(join(dir, "nested", "b.onnx"), "x");
    await writeFile(join(dir, "notes.txt"), "x");

    const listed = await wm.listAvailableWeights({ suffixes: [".onnx"] });

    assert.deepEqual(listed.remote, [EGNN]);
    assert.deepEqual(listed.local.sort(), [EGNN, join("nested", "b.onnx")].sort());
    assert.equal("remoteError" in listed, false);
  });

  it("lists .pt and .onnx by default", async () => {
    stubFetch();
    const listed = await new WeightsManager({ cacheDir: dir }).listAvailableWeights();
    assert.deepEqual(listed.remote, [EGNN, "small_egnn_chembl_15_39.pt"]);
  });

  it("lists remote from Hugging Face even with a custom baseUrl", async () => {
    const calls = stubFetch();
    const wm = new WeightsManager({ cacheDir: dir, baseUrl: "https://mirror.example/w" });
    await wm.listAvailableWeights();
    assert.deepEqual(calls, [HF_API]);
  });

  it("keeps the local list and reports remoteError when the network fails", async () => {
    stubFetch();
    const wm = new WeightsManager({ cacheDir: dir });
    await wm.resolve(EGNN);
    globalThis.fetch = async () => {
      throw new TypeError("fetch failed");
    };

    const listed = await wm.listAvailableWeights({ suffixes: [".onnx"] });

    assert.deepEqual(listed.remote, []);
    assert.deepEqual(listed.local, [EGNN]);
    assert.match(listed.remoteError.message, /fetch failed/);
  });

  it("reports an HTTP error from the Hugging Face API as remoteError", async () => {
    stubFetch(() => new Response("down", { status: 503, statusText: "Service Unavailable" }));
    const listed = await new WeightsManager({ cacheDir: dir }).listAvailableWeights();
    assert.deepEqual(listed.remote, []);
    assert.match(listed.remoteError.message, /HTTP 503/);
  });

  it("returns an empty local list when the cache directory does not exist", async () => {
    stubFetch();
    const wm = new WeightsManager({ cacheDir: join(dir, "absent") });
    assert.deepEqual((await wm.listAvailableWeights()).local, []);
  });

  it("clearCache empties and recreates the cache directory", async () => {
    stubFetch();
    const wm = new WeightsManager({ cacheDir: dir });
    await wm.resolve(EGNN);
    await wm.clearCache();
    assert.deepEqual(await readdir(dir), []);
  });

  it("takes the cache directory from MLCONFGEN_CACHE, and cacheDir over it", async () => {
    process.env.MLCONFGEN_CACHE = join(dir, "env");
    assert.equal(await new WeightsManager().getCacheDir(), join(dir, "env"));
    assert.equal(await new WeightsManager({ cacheDir: dir }).getCacheDir(), dir);
  });

  it("defaults the cache directory to ~/.mlconfgen_weights", async () => {
    delete process.env.MLCONFGEN_CACHE;
    const { homedir } = await import("node:os");
    assert.equal(
      await new WeightsManager().getCacheDir(),
      join(homedir(), ".mlconfgen_weights"),
    );
  });
});

describe("WeightsManager (browser)", () => {
  const URL_EGNN = `${HF_RESOLVE}/${EGNN}`;

  afterEach(restoreGlobals);

  function browserManager(options = {}) {
    return new WeightsManager({ isNode: false, ...options });
  }

  it("fetches a bare name on a miss and stores it in the named cache", async () => {
    const stores = stubCaches();
    const calls = stubFetch();

    const bytes = await browserManager().resolve(EGNN);

    assert.ok(bytes instanceof Uint8Array);
    assert.deepEqual([...bytes], BYTES);
    assert.deepEqual(calls, [URL_EGNN]);
    assert.deepEqual([...stores.get("mlconfgen-weights").get(URL_EGNN)], BYTES);
  });

  it("reports progress in the browser", async () => {
    stubCaches();
    stubFetch();
    const progress = [];
    await browserManager({ onProgress: (p) => progress.push(p) }).resolve(EGNN);
    assert.deepEqual(progress.at(-1), { file: EGNN, loaded: 4, total: 4 });
  });

  it("serves a cache hit without fetching", async () => {
    stubCaches();
    const calls = stubFetch();
    const wm = browserManager();
    await wm.resolve(EGNN);

    const bytes = await wm.resolve(EGNN);

    assert.deepEqual([...bytes], BYTES);
    assert.equal(calls.length, 1);
  });

  it("fetches with forceDownload even on a hit", async () => {
    stubCaches();
    const calls = stubFetch();
    const wm = browserManager();
    await wm.resolve(EGNN);
    await wm.resolve(EGNN, { forceDownload: true });
    assert.equal(calls.length, 2);
  });

  it("passes relative paths and URLs through unchanged", async () => {
    stubCaches();
    const calls = stubFetch();
    const wm = browserManager();
    assert.equal(await wm.resolve("./models/egnn.onnx"), "./models/egnn.onnx");
    assert.equal(await wm.resolve("https://x.example/y.onnx"), "https://x.example/y.onnx");
    assert.equal(calls.length, 0);
  });

  it("uses a custom baseUrl and cacheName", async () => {
    const stores = stubCaches();
    const calls = stubFetch();
    await browserManager({ baseUrl: "https://self.host/w", cacheName: "c2" }).resolve("a.onnx");
    assert.deepEqual(calls, ["https://self.host/w/a.onnx"]);
    assert.ok(stores.get("c2").has("https://self.host/w/a.onnx"));
  });

  it("lists cached file names as local, filtered by suffix", async () => {
    stubCaches();
    stubFetch();
    const wm = browserManager();
    await wm.resolve(EGNN);
    const cache = await globalThis.caches.open("mlconfgen-weights");
    await cache.put(`${HF_RESOLVE}/sub/x.pt`, new Response("p"));

    assert.deepEqual(await wm.listAvailableWeights({ suffixes: [".onnx"] }), {
      remote: [EGNN],
      local: [EGNN],
    });
    assert.deepEqual((await wm.listAvailableWeights()).local, [EGNN, "x.pt"]);
  });

  it("reports remoteError in the browser and keeps the cached list", async () => {
    stubCaches();
    stubFetch();
    const wm = browserManager();
    await wm.resolve(EGNN);
    globalThis.fetch = async () => {
      throw new TypeError("Failed to fetch");
    };

    const listed = await wm.listAvailableWeights({ suffixes: [".onnx"] });

    assert.deepEqual(listed.local, [EGNN]);
    assert.match(listed.remoteError.message, /Failed to fetch/);
  });

  it("clearCache deletes the named cache", async () => {
    const stores = stubCaches();
    stubFetch();
    const wm = browserManager();
    await wm.resolve(EGNN);
    await wm.clearCache();
    assert.equal(stores.has("mlconfgen-weights"), false);
  });

  it("still returns the bytes when storing in the cache fails", async () => {
    globalThis.caches = {
      async open() {
        return {
          async match() {
            return undefined;
          },
          async put() {
            throw new DOMException("quota", "QuotaExceededError");
          },
        };
      },
    };
    stubFetch();
    assert.deepEqual([...(await browserManager().resolve(EGNN))], BYTES);
  });

  it("falls back to a plain fetch without the Cache API", async () => {
    delete globalThis.caches;
    const calls = stubFetch();
    const wm = browserManager();
    assert.deepEqual([...(await wm.resolve(EGNN))], BYTES);
    assert.deepEqual([...(await wm.resolve(EGNN))], BYTES);
    assert.equal(calls.length, 2);
    assert.deepEqual((await wm.listAvailableWeights()).local, []);
    await wm.clearCache();
  });

  it("falls back to a plain fetch when opening the cache throws", async () => {
    globalThis.caches = {
      async open() {
        throw new DOMException("insecure", "SecurityError");
      },
    };
    stubFetch();
    assert.deepEqual([...(await browserManager().resolve(EGNN))], BYTES);
  });

  it("rejects on an HTTP error", async () => {
    stubCaches();
    stubFetch(() => new Response("nope", { status: 404, statusText: "Not Found" }));
    await assert.rejects(browserManager().resolve("c.onnx"), /HTTP 404/);
  });

  it("has no cache directory in the browser", async () => {
    await assert.rejects(browserManager().getCacheDir(), /only available in Node/);
  });
});
