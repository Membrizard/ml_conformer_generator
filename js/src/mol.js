import {
  ATOM_DECODER,
  COVALENT_RADII,
  DIMENSION,
  ELEMENT_TO_Z,
  NUM_BOND_TYPES,
} from "./constants.js";
import { numpyRandint } from "./numpyRandom.js";
import { getRdkit, hasRdkitLoader } from "./rdkit.js";
import { zeros } from "./tensor.js";

/** Lightweight molecule: atoms + coordinates + bonds. */
export class Molecule {
  /**
   * @param {object} opts
   * @param {Int32Array|number[]} opts.atomicNumbers
   * @param {Float32Array|number[]} opts.positions flat [n*3]
   * @param {{i:number,j:number,type:number}[]} [opts.bonds]
   */
  constructor({ atomicNumbers, positions, bonds = [] }) {
    this.atomicNumbers = Int32Array.from(atomicNumbers);
    this.positions =
      positions instanceof Float32Array
        ? positions
        : Float32Array.from(positions);
    this.bonds = bonds.map((b) => ({ i: b.i, j: b.j, type: b.type }));
    if (this.positions.length !== this.atomicNumbers.length * 3) {
      throw new RangeError("positions length must be nAtoms * 3");
    }
  }

  get nAtoms() {
    return this.atomicNumbers.length;
  }

  /** Number of connected components (ignores isolated atoms with no bonds as separate if n>0). */
  fragmentCount() {
    const n = this.nAtoms;
    if (n === 0) return 0;
    if (this.bonds.length === 0) return n;

    const adj = Array.from({ length: n }, () => []);
    for (const { i, j } of this.bonds) {
      adj[i].push(j);
      adj[j].push(i);
    }
    const seen = new Uint8Array(n);
    let count = 0;
    for (let start = 0; start < n; start += 1) {
      if (seen[start]) continue;
      count += 1;
      const stack = [start];
      seen[start] = 1;
      while (stack.length) {
        const u = stack.pop();
        for (const v of adj[u]) {
          if (!seen[v]) {
            seen[v] = 1;
            stack.push(v);
          }
        }
      }
    }
    return count;
  }

  /** Keep the largest connected component (by atom count). */
  largestFragment() {
    const n = this.nAtoms;
    if (n === 0 || this.bonds.length === 0) return this;

    const adj = Array.from({ length: n }, () => []);
    for (const { i, j } of this.bonds) {
      adj[i].push(j);
      adj[j].push(i);
    }

    const seen = new Uint8Array(n);
    let best = [];
    for (let start = 0; start < n; start += 1) {
      if (seen[start]) continue;
      const stack = [start];
      const comp = [];
      seen[start] = 1;
      while (stack.length) {
        const u = stack.pop();
        comp.push(u);
        for (const v of adj[u]) {
          if (!seen[v]) {
            seen[v] = 1;
            stack.push(v);
          }
        }
      }
      if (comp.length > best.length) best = comp;
    }

    if (best.length === n) return this;

    const map = new Int32Array(n).fill(-1);
    best.forEach((old, neu) => {
      map[old] = neu;
    });
    const atomicNumbers = Int32Array.from(best, (i) => this.atomicNumbers[i]);
    const positions = new Float32Array(best.length * 3);
    best.forEach((old, neu) => {
      positions[neu * 3] = this.positions[old * 3];
      positions[neu * 3 + 1] = this.positions[old * 3 + 1];
      positions[neu * 3 + 2] = this.positions[old * 3 + 2];
    });
    const bonds = [];
    for (const b of this.bonds) {
      const i = map[b.i];
      const j = map[b.j];
      if (i >= 0 && j >= 0) bonds.push({ i, j, type: b.type });
    }
    return new Molecule({ atomicNumbers, positions, bonds });
  }

  /** CTAB V2000 — preferred for RDKit.js `get_mol` / depiction. */
  toMolBlock(name = "MLConfGen") {
    const n = this.nAtoms;
    const m = this.bonds.length;
    if (n > 999 || m > 999) {
      throw new RangeError("V2000 molfiles support at most 999 atoms/bonds");
    }

    const lines = [
      name,
      "  mlconfgen",
      "",
      `${padInt(n, 3)}${padInt(m, 3)}  0  0  0  0  0  0  0  0999 V2000`,
    ];

    for (let i = 0; i < n; i += 1) {
      const sym = atomicSymbol(this.atomicNumbers[i]).padEnd(3, " ");
      lines.push(
        `${fmtFixed(this.positions[i * 3])}${fmtFixed(this.positions[i * 3 + 1])}${fmtFixed(this.positions[i * 3 + 2])} ${sym} 0  0  0  0  0  0  0  0  0  0  0  0`,
      );
    }

    for (const b of this.bonds) {
      lines.push(
        `${padInt(b.i + 1, 3)}${padInt(b.j + 1, 3)}${padInt(b.type, 3)}  0  0  0  0`,
      );
    }

    lines.push("M  END");
    return lines.join("\n");
  }

  toMolBlockV2000(name = "MLConfGen") {
    return this.toMolBlock(name);
  }

  toMolBlockV3000() {
    const n = this.nAtoms;
    const m = this.bonds.length;
    const lines = [
      "",
      "     MLConfGen",
      "",
      "  0  0  0     0  0            999 V3000",
      "M  V30 BEGIN CTAB",
      `M  V30 COUNTS ${n} ${m} 0 0 0`,
      "M  V30 BEGIN ATOM",
    ];
    for (let i = 0; i < n; i += 1) {
      const z = this.atomicNumbers[i];
      const sym = atomicSymbol(z);
      const x = this.positions[i * 3];
      const y = this.positions[i * 3 + 1];
      const zc = this.positions[i * 3 + 2];
      lines.push(
        `M  V30 ${i + 1} ${sym} ${fmt(x)} ${fmt(y)} ${fmt(zc)} 0`,
      );
    }
    lines.push("M  V30 END ATOM");
    lines.push("M  V30 BEGIN BOND");
    this.bonds.forEach((b, idx) => {
      lines.push(`M  V30 ${idx + 1} ${b.type} ${b.i + 1} ${b.j + 1}`);
    });
    lines.push("M  V30 END BOND");
    lines.push("M  V30 END CTAB");
    lines.push("M  END");
    return lines.join("\n");
  }
}

function fmt(v) {
  return Number(v).toFixed(4);
}

function fmtFixed(v) {
  return Number(v).toFixed(4).padStart(10, " ");
}

function padInt(v, width) {
  return String(v).padStart(width, " ");
}

function atomicSymbol(z) {
  for (const [sym, num] of Object.entries(ELEMENT_TO_Z)) {
    if (num === z) return sym;
  }
  return String(z);
}

/** Convert EDM `(x, h, nodeMask)` batch → Molecule[] (no bonds yet). */
export function samplesToMolecules(x, h, nodeMask, atomDecoder = ATOM_DECODER) {
  const [B, N] = nodeMask.shape;
  const mols = [];
  for (let b = 0; b < B; b += 1) {
    let nAtoms = 0;
    for (let i = 0; i < N; i += 1) {
      if (nodeMask.data[b * N + i] > 0) nAtoms += 1;
    }
    const atomicNumbers = new Int32Array(nAtoms);
    const positions = new Float32Array(nAtoms * 3);
    for (let i = 0; i < nAtoms; i += 1) {
      let best = 0;
      let bestV = h.data[(b * N + i) * h.shape[2]];
      for (let c = 1; c < h.shape[2]; c += 1) {
        const v = h.data[(b * N + i) * h.shape[2] + c];
        if (v > bestV) {
          bestV = v;
          best = c;
        }
      }
      const sym = atomDecoder[best];
      atomicNumbers[i] = ELEMENT_TO_Z[sym] ?? best;
      positions[i * 3] = x.data[(b * N + i) * 3];
      positions[i * 3 + 1] = x.data[(b * N + i) * 3 + 1];
      positions[i * 3 + 2] = x.data[(b * N + i) * 3 + 2];
    }
    mols.push(new Molecule({ atomicNumbers, positions, bonds: [] }));
  }
  return mols;
}

/** Max heavy-atom valence for connectivity guessing (RDKit ConnectTheDots-style). */
const MAX_VALENCE = Object.freeze({
  6: 4,
  7: 3,
  8: 2,
  9: 1,
  15: 5,
  16: 6,
  17: 1,
  35: 1,
});

/**
 * Guess single-bond connectivity from covalent radii + valence caps
 * (stand-in for RDKit `DetermineConnectivity` / ConnectTheDots).
 */
export function guessConnectivity(mol, scale = 1.2) {
  const n = mol.nAtoms;
  const candidates = [];
  for (let i = 0; i < n; i += 1) {
    const ri = COVALENT_RADII[mol.atomicNumbers[i]] ?? 0.8;
    for (let j = i + 1; j < n; j += 1) {
      const rj = COVALENT_RADII[mol.atomicNumbers[j]] ?? 0.8;
      const dx = mol.positions[i * 3] - mol.positions[j * 3];
      const dy = mol.positions[i * 3 + 1] - mol.positions[j * 3 + 1];
      const dz = mol.positions[i * 3 + 2] - mol.positions[j * 3 + 2];
      const dist = Math.sqrt(dx * dx + dy * dy + dz * dz);
      if (dist > 0.5 && dist < (ri + rj) * scale) {
        candidates.push({ i, j, dist });
      }
    }
  }
  candidates.sort((a, b) => a.dist - b.dist);

  const valence = new Int32Array(n);
  const bonds = [];
  for (const { i, j } of candidates) {
    const maxI = MAX_VALENCE[mol.atomicNumbers[i]] ?? 4;
    const maxJ = MAX_VALENCE[mol.atomicNumbers[j]] ?? 4;
    if (valence[i] >= maxI || valence[j] >= maxJ) continue;
    bonds.push({ i, j, type: 1 });
    valence[i] += 1;
    valence[j] += 1;
  }

  return new Molecule({
    atomicNumbers: mol.atomicNumbers,
    positions: mol.positions,
    bonds,
  });
}

/**
 * Python `canonicalise`: DetermineConnectivity + MolToSmiles +
 * `_smilesAtomOutputOrder` + RenumberAtoms.
 *
 * RDKit.js exposes the same `_smilesAtomOutputOrder` mol prop after `get_smiles()`.
 */
export async function canonicalise(mol) {
  const connected = guessConnectivity(mol);
  const n = connected.nAtoms;
  if (n <= 1) return connected;

  try {
    if (!hasRdkitLoader()) return connected;
    const RDKit = await getRdkit();
    const rdMol = RDKit.get_mol(
      connected.toMolBlock(),
      JSON.stringify({ sanitize: false }),
    );
    if (!rdMol) return connected;

    let order = null;
    try {
      rdMol.get_smiles();
      if (rdMol.has_prop("_smilesAtomOutputOrder")) {
        order = parseSmilesAtomOutputOrder(
          rdMol.get_prop("_smilesAtomOutputOrder"),
          n,
        );
      }
    } finally {
      rdMol.delete();
    }

    if (!order) return connected;
    return permuteMolecule(connected, order);
  } catch {
    return connected;
  }
}

/** Parse RDKit `[3,1,2,0]` / `[3,1,2,0,]` into an Int32Array of length n. */
function parseSmilesAtomOutputOrder(orderStr, n) {
  const order = orderStr
    .replace(/[\[\]]/g, "")
    .split(",")
    .map((x) => x.trim())
    .filter((x) => x !== "")
    .map(Number);
  if (order.length !== n || order.some((i) => !Number.isInteger(i) || i < 0 || i >= n)) {
    return null;
  }
  const seen = new Uint8Array(n);
  for (const i of order) {
    if (seen[i]) return null;
    seen[i] = 1;
  }
  return Int32Array.from(order);
}

function permuteMolecule(mol, order) {
  const n = order.length;
  const inv = new Int32Array(n);
  for (let neu = 0; neu < n; neu += 1) inv[order[neu]] = neu;

  const atomicNumbers = Int32Array.from(order, (old) => mol.atomicNumbers[old]);
  const positions = new Float32Array(n * 3);
  for (let neu = 0; neu < n; neu += 1) {
    const old = order[neu];
    positions[neu * 3] = mol.positions[old * 3];
    positions[neu * 3 + 1] = mol.positions[old * 3 + 1];
    positions[neu * 3 + 2] = mol.positions[old * 3 + 2];
  }

  const bonds = [];
  for (const b of mol.bonds) {
    const i = inv[b.i];
    const j = inv[b.j];
    bonds.push(i < j ? { i, j, type: b.type } : { i: j, j: i, type: b.type });
  }

  return new Molecule({ atomicNumbers, positions, bonds });
}

/** Pack molecules for AdjMatSeer ONNX: elements, dist_mat, binary adj. */
export async function prepareAdjMatSeerInput(mols, dimension = DIMENSION) {
  const nSamples = mols.length;
  const elements = new BigInt64Array(nSamples * dimension);
  const distMat = new Float32Array(nSamples * dimension * dimension);
  const adjMat = new Float32Array(nSamples * dimension * dimension);
  const prepared = [];

  for (let s = 0; s < nSamples; s += 1) {
    const mol = await canonicalise(mols[s]);
    prepared.push(mol);
    const n = mol.nAtoms;
    const base = s * dimension;

    for (let i = 0; i < n; i += 1) {
      elements[base + i] = BigInt(mol.atomicNumbers[i]);
    }

    const matBase = s * dimension * dimension;
    for (let i = 0; i < dimension; i += 1) {
      for (let j = 0; j < dimension; j += 1) {
        const idx = matBase + i * dimension + j;
        if (i === j) {
          distMat[idx] = 1;
          adjMat[idx] = 1;
        } else if (i < n && j < n) {
          const dx = mol.positions[i * 3] - mol.positions[j * 3];
          const dy = mol.positions[i * 3 + 1] - mol.positions[j * 3 + 1];
          const dz = mol.positions[i * 3 + 2] - mol.positions[j * 3 + 2];
          distMat[idx] = Math.sqrt(dx * dx + dy * dy + dz * dz);
        }
      }
    }
    for (const { i, j } of mol.bonds) {
      adjMat[matBase + i * dimension + j] = 1;
      adjMat[matBase + j * dimension + i] = 1;
    }
  }

  return { elements, distMat, adjMat, mols: prepared };
}

/**
 * Apply AdjMatSeer logits (Python redefine_bonds) with connectivity fallback:
 * prefer AdjMatSeer bond type; if it predicts no bond but distance-connectivity
 * had an edge, keep a single bond (helps when sanitization-free samples are noisy).
 */
export function redefineBonds(mol, adjLogits, dimension = DIMENSION) {
  const data = adjLogits.data ?? adjLogits;
  const n = mol.nAtoms;
  const guessed = new Set(
    mol.bonds.map(
      (b) => `${Math.max(b.i, b.j)},${Math.min(b.i, b.j)}`,
    ),
  );

  const bonds = [];
  for (let i = 0; i < n; i += 1) {
    for (let j = 0; j < i; j += 1) {
      const offset = (i * dimension + j) * NUM_BOND_TYPES;
      let best = 0;
      let bestV = data[offset];
      for (let t = 1; t < NUM_BOND_TYPES; t += 1) {
        const v = data[offset + t];
        if (v > bestV) {
          bestV = v;
          best = t;
        }
      }
      if (best !== 0) {
        bonds.push({ i, j, type: best });
      } else if (guessed.has(`${i},${j}`)) {
        bonds.push({ i, j, type: 1 });
      }
    }
  }

  return new Molecule({
    atomicNumbers: mol.atomicNumbers,
    positions: mol.positions,
    bonds,
  });
}

/**
 * Python `is_valid_mol`: copy + `Chem.SanitizeMol` → 1.0 / 0.0.
 * RDKit.js: `get_mol(..., { sanitize: true })` returns null on failure.
 */
export async function isValidMol(mol) {
  if (!mol || mol.nAtoms === 0) return 0;
  if (!hasRdkitLoader()) return 1;
  try {
    const RDKit = await getRdkit();
    const rdMol = RDKit.get_mol(
      mol.toMolBlock(),
      JSON.stringify({ sanitize: true }),
    );
    if (!rdMol) return 0;
    const ok = rdMol.is_valid();
    rdMol.delete();
    return ok ? 1 : 0;
  } catch {
    return 0;
  }
}

/**
 * Python `standardize_mol` (without MMFF): largest fragment or reject
 * disconnected, then sanitize. Returns the cleaned `Molecule` or `null`.
 */
export async function standardizeMol(
  mol,
  { keepLargestFragment = true } = {},
) {
  if (!mol || mol.nAtoms === 0) return null;

  let m = mol;
  if (keepLargestFragment) {
    m = mol.largestFragment();
  } else if (mol.fragmentCount() > 1) {
    // Python ifm_mode: discard multi-fragment molecules
    return null;
  }

  if (!hasRdkitLoader()) return m;

  try {
    const RDKit = await getRdkit();
    const rdMol = RDKit.get_mol(
      m.toMolBlock(),
      JSON.stringify({ sanitize: true }),
    );
    if (!rdMol || !rdMol.is_valid()) {
      if (rdMol) rdMol.delete();
      return null;
    }
    // Sanitize can update bond orders / aromaticity; re-read CTAB if possible.
    let out = m;
    try {
      const block = rdMol.get_molblock();
      const parsed = moleculeFromMolBlock(block);
      if (parsed) out = parsed;
    } catch {
      // keep original geometry/topology if molblock round-trip fails
    }
    rdMol.delete();
    return out;
  } catch {
    return null;
  }
}

/** Minimal V2000 molblock → Molecule (for post-sanitize round-trip). */
function moleculeFromMolBlock(block) {
  const lines = block.replace(/\r\n/g, "\n").split("\n");
  if (lines.length < 4) return null;
  const counts = lines[3];
  const nAtoms = parseInt(counts.slice(0, 3), 10);
  const nBonds = parseInt(counts.slice(3, 6), 10);
  if (!Number.isFinite(nAtoms) || !Number.isFinite(nBonds)) return null;

  const zMap = {
    C: 6,
    N: 7,
    O: 8,
    F: 9,
    P: 15,
    S: 16,
    Cl: 17,
    Br: 35,
  };
  const atomicNumbers = [];
  const positions = new Float32Array(nAtoms * 3);
  for (let i = 0; i < nAtoms; i += 1) {
    const L = lines[4 + i];
    if (!L || L.length < 34) return null;
    positions[i * 3] = parseFloat(L.slice(0, 10));
    positions[i * 3 + 1] = parseFloat(L.slice(10, 20));
    positions[i * 3 + 2] = parseFloat(L.slice(20, 30));
    const sym = L.slice(31, 34).trim();
    const z = zMap[sym];
    if (!z) return null;
    atomicNumbers.push(z);
  }
  const bonds = [];
  for (let i = 0; i < nBonds; i += 1) {
    const L = lines[4 + nAtoms + i];
    if (!L || L.length < 9) return null;
    const a = parseInt(L.slice(0, 3), 10) - 1;
    const b = parseInt(L.slice(3, 6), 10) - 1;
    const t = parseInt(L.slice(6, 9), 10);
    if (a < 0 || b < 0 || a >= nAtoms || b >= nAtoms) return null;
    bonds.push(a < b ? { i: a, j: b, type: t } : { i: b, j: a, type: t });
  }
  return new Molecule({ atomicNumbers, positions, bonds });
}

export function prepareMasks(nNodes, maxNNodes) {
  // nNodes: Int32Array length B
  const B = nNodes.length;
  const nodeMask = zeros([B, maxNNodes, 1]);
  for (let b = 0; b < B; b += 1) {
    for (let i = 0; i < nNodes[b]; i += 1) nodeMask.data[b * maxNNodes + i] = 1;
  }

  // edge_mask: (B*N*N, 1), pairwise node_mask outer product, diagonal 0
  const edgeMask = zeros([B * maxNNodes * maxNNodes, 1]);
  for (let b = 0; b < B; b += 1) {
    for (let i = 0; i < maxNNodes; i += 1) {
      const mi = nodeMask.data[b * maxNNodes + i];
      for (let j = 0; j < maxNNodes; j += 1) {
        if (i === j) continue;
        const mj = nodeMask.data[b * maxNNodes + j];
        edgeMask.data[(b * maxNNodes + i) * maxNNodes + j] = mi * mj;
      }
    }
  }
  return { nodeMask, edgeMask };
}

export function prepareEdmInput({
  nSamples,
  referenceContext,
  contextNorms,
  minNNodes,
  maxNNodes,
}) {
  const nNodes = new Int32Array(nSamples);
  for (let i = 0; i < nSamples; i += 1) {
    // Match np.random.randint(min, max+1) on the shared RandomState.
    nNodes[i] = numpyRandint(minNNodes, maxNNodes + 1);
  }
  const { nodeMask, edgeMask } = prepareMasks(nNodes, maxNNodes);

  const mean = contextNorms.mean;
  const mad = contextNorms.mad;
  const normed = new Float32Array(3);
  for (let d = 0; d < 3; d += 1) {
    normed[d] = (referenceContext[d] - mean[d]) / mad[d];
  }

  const batchContext = zeros([nSamples, maxNNodes, 3]);
  for (let b = 0; b < nSamples; b += 1) {
    for (let i = 0; i < maxNNodes; i += 1) {
      const m = nodeMask.data[b * maxNNodes + i];
      for (let d = 0; d < 3; d += 1) {
        batchContext.data[(b * maxNNodes + i) * 3 + d] = normed[d] * m;
      }
    }
  }
  return { nodeMask, edgeMask, batchContext };
}

/**
 * Principal-frame MOI eigenvalues for a set of equal-mass points.
 * @returns {{ context: Float32Array, aligned: Float32Array }}
 */
export function contextFromCoordinates(positions, nAtoms = positions.length / 3) {
  const coord = new Float32Array(nAtoms * 3);
  let cx = 0;
  let cy = 0;
  let cz = 0;
  for (let i = 0; i < nAtoms; i += 1) {
    cx += positions[i * 3];
    cy += positions[i * 3 + 1];
    cz += positions[i * 3 + 2];
  }
  cx /= nAtoms;
  cy /= nAtoms;
  cz /= nAtoms;
  for (let i = 0; i < nAtoms; i += 1) {
    coord[i * 3] = positions[i * 3] - cx;
    coord[i * 3 + 1] = positions[i * 3 + 1] - cy;
    coord[i * 3 + 2] = positions[i * 3 + 2] - cz;
  }

  const moi = momentOfInertia(coord, nAtoms);
  const { values, vectors } = eigh3(moi);

  const aligned = new Float32Array(nAtoms * 3);
  for (let i = 0; i < nAtoms; i += 1) {
    const x = coord[i * 3];
    const y = coord[i * 3 + 1];
    const z = coord[i * 3 + 2];
    // coord @ eigenvectors (columns)
    aligned[i * 3] = x * vectors[0] + y * vectors[3] + z * vectors[6];
    aligned[i * 3 + 1] = x * vectors[1] + y * vectors[4] + z * vectors[7];
    aligned[i * 3 + 2] = x * vectors[2] + y * vectors[5] + z * vectors[8];
  }

  const moi2 = momentOfInertia(aligned, nAtoms);
  const context = Float32Array.from([moi2[0], moi2[4], moi2[8]]);
  return { context, aligned };
}

function momentOfInertia(coord, n) {
  let xx = 0;
  let yy = 0;
  let zz = 0;
  let xy = 0;
  let xz = 0;
  let yz = 0;
  for (let i = 0; i < n; i += 1) {
    const x = coord[i * 3];
    const y = coord[i * 3 + 1];
    const z = coord[i * 3 + 2];
    xx += y * y + z * z;
    yy += x * x + z * z;
    zz += x * x + y * y;
    xy -= x * y;
    xz -= x * z;
    yz -= y * z;
  }
  return Float32Array.from([xx, xy, xz, xy, yy, yz, xz, yz, zz]);
}

/** Symmetric 3×3 eigen-decomposition (Jacobi). Returns ascending eigenvalues. */
function eigh3(m) {
  // m row-major 3x3
  let a = [[m[0], m[1], m[2]], [m[3], m[4], m[5]], [m[6], m[7], m[8]]];
  let v = [
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1],
  ];
  for (let iter = 0; iter < 32; iter += 1) {
    let p = 0;
    let q = 1;
    let max = Math.abs(a[0][1]);
    for (const [i, j] of [
      [0, 2],
      [1, 2],
    ]) {
      const val = Math.abs(a[i][j]);
      if (val > max) {
        max = val;
        p = i;
        q = j;
      }
    }
    if (max < 1e-10) break;
    const app = a[p][p];
    const aqq = a[q][q];
    const apq = a[p][q];
    const tau = (aqq - app) / (2 * apq);
    const t =
      tau === 0
        ? 1
        : (tau >= 0 ? 1 : -1) / (Math.abs(tau) + Math.sqrt(1 + tau * tau));
    const c = 1 / Math.sqrt(1 + t * t);
    const s = t * c;
    const newA = a.map((row) => row.slice());
    newA[p][p] = app - t * apq;
    newA[q][q] = aqq + t * apq;
    newA[p][q] = newA[q][p] = 0;
    for (let r = 0; r < 3; r += 1) {
      if (r === p || r === q) continue;
      const arp = a[r][p];
      const arq = a[r][q];
      newA[r][p] = newA[p][r] = c * arp - s * arq;
      newA[r][q] = newA[q][r] = c * arq + s * arp;
    }
    a = newA;
    const newV = v.map((row) => row.slice());
    for (let r = 0; r < 3; r += 1) {
      const vip = v[r][p];
      const viq = v[r][q];
      newV[r][p] = c * vip - s * viq;
      newV[r][q] = c * viq + s * vip;
    }
    v = newV;
  }
  const vals = [a[0][0], a[1][1], a[2][2]];
  const order = [0, 1, 2].sort((i, j) => vals[i] - vals[j]);
  const values = Float32Array.from(order, (i) => vals[i]);
  // columns of vectors are eigenvectors
  const vectors = new Float32Array(9);
  for (let col = 0; col < 3; col += 1) {
    const src = order[col];
    vectors[col] = v[0][src];
    vectors[3 + col] = v[1][src];
    vectors[6 + col] = v[2][src];
  }
  return { values, vectors };
}
