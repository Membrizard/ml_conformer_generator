/** Defaults for the ChemBL 15–39 ONNX models. */

export const DIMENSION = 42;
export const MIN_N_NODES = 15;
export const MAX_N_NODES = 39;
export const NUM_BOND_TYPES = 5;
export const IN_NODE_NF = 8;
export const N_DIMS = 3;

export const CONTEXT_NORMS = Object.freeze({
  // float64 arrays — match Python ONNX `np.array(value)` (default float64)
  mean: Float64Array.from([105.0766, 473.1938, 537.4675]),
  mad: Float64Array.from([52.0409, 219.7475, 232.9718]),
});

export const ATOM_DECODER = Object.freeze({
  0: "C",
  1: "N",
  2: "O",
  3: "F",
  4: "P",
  5: "S",
  6: "Cl",
  7: "Br",
});

export const ELEMENT_TO_Z = Object.freeze({
  C: 6,
  N: 7,
  O: 8,
  F: 9,
  P: 15,
  S: 16,
  Cl: 17,
  Br: 35,
});

/** Approximate covalent radii (Å) for distance-based connectivity. */
export const COVALENT_RADII = Object.freeze({
  6: 0.76,
  7: 0.71,
  8: 0.66,
  9: 0.57,
  15: 1.07,
  16: 1.05,
  17: 1.02,
  35: 1.2,
});
