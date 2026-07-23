import http from "node:http";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import {
  createGenerator,
  clearRdkitLoader,
  hasRdkitLoader,
} from "../src/index.js";
import * as ort from "onnxruntime-node";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const ROOT = path.resolve(__dirname, "..");
const PUBLIC = path.join(__dirname, "public");
const PORT = Number(process.env.PORT) || 3847;
/** Set NO_RDKIT=1 to skip RDKit (no sanitize / SMILES reorder). */
const USE_RDKIT = process.env.NO_RDKIT !== "1";

const EGNN = process.env.EGNN_ONNX || path.join(ROOT, "egnn_chembl_15_39.onnx");
const ADJ = process.env.ADJ_ONNX || path.join(ROOT, "adj_mat_seer_chembl_15_39.onnx");

if (!USE_RDKIT) clearRdkitLoader();

let generator = null;
let readyError = null;

async function ensureGenerator({ diffusionSteps }) {
  const key = `${diffusionSteps}:${USE_RDKIT ? "rdkit" : "nordkit"}`;
  if (generator?.__smokeKey === key) return generator;

  generator = await createGenerator({
    ort,
    egnnOnnx: EGNN,
    adjMatSeerOnnx: ADJ,
    diffusionSteps,
  });
  generator.__smokeKey = key;
  return generator;
}

function sendJson(res, status, body) {
  const payload = JSON.stringify(body);
  res.writeHead(status, {
    "Content-Type": "application/json",
    "Content-Length": Buffer.byteLength(payload),
  });
  res.end(payload);
}

function readBody(req) {
  return new Promise((resolve, reject) => {
    const chunks = [];
    req.on("data", (c) => chunks.push(c));
    req.on("end", () => {
      try {
        const raw = Buffer.concat(chunks).toString("utf8") || "{}";
        resolve(JSON.parse(raw));
      } catch (err) {
        reject(err);
      }
    });
    req.on("error", reject);
  });
}

function contentType(filePath) {
  if (filePath.endsWith(".html")) return "text/html; charset=utf-8";
  if (filePath.endsWith(".css")) return "text/css; charset=utf-8";
  if (filePath.endsWith(".js")) return "text/javascript; charset=utf-8";
  return "application/octet-stream";
}

const server = http.createServer(async (req, res) => {
  try {
    if (req.method === "GET" && req.url === "/api/status") {
      return sendJson(res, 200, {
        ready: !readyError,
        error: readyError,
        rdkit: hasRdkitLoader(),
        models: { egnn: EGNN, adj: ADJ },
      });
    }

    if (req.method === "POST" && req.url === "/api/generate") {
      const body = await readBody(req);
      const referenceContext = (body.referenceContext || []).map(Number);
      const nAtoms = Number(body.nAtoms) || 20;
      const nSamples = Math.min(Math.max(Number(body.nSamples) || 1, 1), 8);
      const variance = Number(body.variance) || 0;
      const diffusionSteps = Number(body.diffusionSteps) || 20;
      const keepLargestFragment = body.keepLargestFragment !== false;
      const filterInvalid = body.filterInvalid !== false;

      if (referenceContext.length !== 3 || referenceContext.some((v) => !Number.isFinite(v))) {
        return sendJson(res, 400, {
          error: "referenceContext must be three finite numbers (MOI eigenvalues).",
        });
      }

      const started = Date.now();
      const gen = await ensureGenerator({ diffusionSteps });
      const mols = await gen.generateConformers({
        referenceContext,
        nAtoms,
        nSamples,
        variance,
        keepLargestFragment,
        filterInvalid,
      });

      return sendJson(res, 200, {
        elapsedMs: Date.now() - started,
        nRequested: nSamples,
        nValid: mols.length,
        molecules: mols.map((m, i) => ({
          index: i,
          nAtoms: m.nAtoms,
          nBonds: m.bonds.length,
          molfile: m.toMolBlock(),
        })),
      });
    }

    // static files
    let urlPath = req.url === "/" ? "/index.html" : req.url.split("?")[0];
    const filePath = path.normalize(path.join(PUBLIC, urlPath));
    if (!filePath.startsWith(PUBLIC)) {
      res.writeHead(403).end("Forbidden");
      return;
    }
    if (!fs.existsSync(filePath) || fs.statSync(filePath).isDirectory()) {
      res.writeHead(404).end("Not found");
      return;
    }
    const data = fs.readFileSync(filePath);
    res.writeHead(200, { "Content-Type": contentType(filePath) });
    res.end(data);
  } catch (err) {
    console.error(err);
    sendJson(res, 500, { error: err.message || String(err) });
  }
});

console.log(
  USE_RDKIT
    ? "Loading ONNX models (RDKit enabled)…"
    : "Loading ONNX models (NO_RDKIT=1 — no sanitize / SMILES reorder)…",
);
ensureGenerator({ diffusionSteps: 20 })
  .then(() => {
    console.log("Models ready.");
    server.listen(PORT, () => {
      console.log(`Smoke UI → http://localhost:${PORT}`);
    });
  })
  .catch((err) => {
    readyError = err.message || String(err);
    console.error("Failed to preload models:", err);
    server.listen(PORT, () => {
      console.log(`Smoke UI (models failed) → http://localhost:${PORT}`);
    });
  });
