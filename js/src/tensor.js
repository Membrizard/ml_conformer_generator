import { numpyRandn } from "./numpyRandom.js";

/**
 * Flat Float64Array tensors with an explicit shape.
 *
 * Matches the Python ONNX path: `np.random.randn` and reverse-process math run in
 * float64; values are cast to float32 only at the ONNX Runtime boundary.
 */

export function tensor(shape, fill = 0) {
  const size = shape.reduce((a, b) => a * b, 1);
  const data = new Float64Array(size);
  if (fill !== 0) data.fill(fill);
  return { data, shape };
}

export function zeros(shape) {
  return tensor(shape, 0);
}

export function full(shape, value) {
  return tensor(shape, value);
}

/**
 * Standard normal — `np.random.randn(*shape)` via NumPy RandomState
 * (MT19937 + Marsaglia polar). Same seed → bit-identical draws.
 */
export function randn(shape) {
  return numpyRandn(shape);
}

export function clone(t) {
  return { data: new Float64Array(t.data), shape: t.shape.slice() };
}

export function reshape(t, shape) {
  const size = shape.reduce((a, b) => a * b, 1);
  if (size !== t.data.length) {
    throw new RangeError(`Cannot reshape ${t.shape} → ${shape}`);
  }
  return { data: t.data, shape };
}

/** Broadcast a (B,) or (B,1,...) array to match `target.shape`. */
export function inflateBatch(array, target) {
  const out = zeros(target.shape);
  const batch = target.shape[0];
  const stride = out.data.length / batch;
  for (let b = 0; b < batch; b += 1) {
    const v = array.data[b];
    const offset = b * stride;
    for (let i = 0; i < stride; i += 1) out.data[offset + i] = v;
  }
  return out;
}

export function add(a, b) {
  const out = zeros(a.shape);
  for (let i = 0; i < out.data.length; i += 1) out.data[i] = a.data[i] + b.data[i];
  return out;
}

export function sub(a, b) {
  const out = zeros(a.shape);
  for (let i = 0; i < out.data.length; i += 1) out.data[i] = a.data[i] - b.data[i];
  return out;
}

export function mul(a, b) {
  const out = zeros(a.shape);
  const bd = typeof b === "number" ? null : b.data;
  for (let i = 0; i < out.data.length; i += 1) {
    out.data[i] = a.data[i] * (bd ? bd[i] : b);
  }
  return out;
}

export function div(a, b) {
  const out = zeros(a.shape);
  const bd = typeof b === "number" ? null : b.data;
  for (let i = 0; i < out.data.length; i += 1) {
    out.data[i] = a.data[i] / (bd ? bd[i] : b);
  }
  return out;
}

export function concatLast(a, b) {
  const [B, N, Da] = a.shape;
  const Db = b.shape[2];
  const out = zeros([B, N, Da + Db]);
  for (let i = 0; i < B * N; i += 1) {
    const ao = i * Da;
    const bo = i * Db;
    const oo = i * (Da + Db);
    for (let d = 0; d < Da; d += 1) out.data[oo + d] = a.data[ao + d];
    for (let d = 0; d < Db; d += 1) out.data[oo + Da + d] = b.data[bo + d];
  }
  return out;
}

export function sliceLast(t, start, end = null) {
  const [B, N, D] = t.shape;
  const e = end == null ? D : end < 0 ? D + end : end;
  const outD = e - start;
  const out = zeros([B, N, outD]);
  for (let i = 0; i < B * N; i += 1) {
    const src = i * D + start;
    const dst = i * outD;
    for (let d = 0; d < outD; d += 1) out.data[dst + d] = t.data[src + d];
  }
  return out;
}

export function mapData(t, fn) {
  const out = zeros(t.shape);
  for (let i = 0; i < t.data.length; i += 1) out.data[i] = fn(t.data[i]);
  return out;
}

export function sigmoid(t) {
  return mapData(t, (z) => 1 / (1 + Math.exp(-z)));
}

/** Match Python ONNX: `np.log1p(np.exp(z))`. */
export function softplus(t) {
  return mapData(t, (z) => Math.log1p(Math.exp(z)));
}

/** Match Python ONNX: `-np.log1p(np.exp(-z))`. */
export function logSigmoid(t) {
  return mapData(t, (z) => -Math.log1p(Math.exp(-z)));
}

/**
 * Python `remove_mean_with_mask`:
 *   n = sum(mask, 1); mean = sum(x, 1) / n; x = x - mean * mask
 * (x is already zero on padded nodes when called after masking.)
 */
export function removeMeanWithMask(x, nodeMask) {
  const [B, N, D] = x.shape;
  const out = zeros(x.shape);
  for (let b = 0; b < B; b += 1) {
    let n = 0;
    const mean = new Float64Array(D);
    for (let i = 0; i < N; i += 1) {
      const m = nodeMask.data[b * N + i];
      n += m;
      for (let d = 0; d < D; d += 1) {
        // Sum x directly (padded entries are 0 after mask multiply).
        mean[d] += x.data[(b * N + i) * D + d];
      }
    }
    const invN = 1 / (n || 1);
    for (let d = 0; d < D; d += 1) mean[d] *= invN;
    for (let i = 0; i < N; i += 1) {
      const m = nodeMask.data[b * N + i];
      for (let d = 0; d < D; d += 1) {
        const idx = (b * N + i) * D + d;
        out.data[idx] = x.data[idx] - mean[d] * m;
      }
    }
  }
  return out;
}

/** Python `sample_gaussian_with_mask`: randn then `x * node_mask`. */
export function sampleGaussianWithMask(shape, nodeMask) {
  const x = randn(shape);
  const [B, N, D] = shape;
  for (let b = 0; b < B; b += 1) {
    for (let i = 0; i < N; i += 1) {
      const m = nodeMask.data[b * N + i];
      for (let d = 0; d < D; d += 1) x.data[(b * N + i) * D + d] *= m;
    }
  }
  return x;
}

/**
 * Python `sample_center_gravity_zero_gaussian_with_mask`:
 * mask randn, then project out the centre of mass.
 */
export function sampleCenterGravityZeroGaussianWithMask(shape, nodeMask) {
  const xMasked = sampleGaussianWithMask(shape, nodeMask);
  return removeMeanWithMask(xMasked, nodeMask);
}

export function argmaxLast(t) {
  const [B, N, C] = t.shape;
  const out = new Int32Array(B * N);
  for (let i = 0; i < B * N; i += 1) {
    let best = 0;
    let bestV = t.data[i * C];
    for (let c = 1; c < C; c += 1) {
      const v = t.data[i * C + c];
      if (v > bestV) {
        bestV = v;
        best = c;
      }
    }
    out[i] = best;
  }
  return out;
}

export function oneHot(labels, numClasses, shapeBN) {
  const [B, N] = shapeBN;
  const out = zeros([B, N, numClasses]);
  for (let i = 0; i < labels.length; i += 1) {
    out.data[i * numClasses + labels[i]] = 1;
  }
  return out;
}

export function toOrt(ort, t, type = "float32") {
  // Cast float64 working tensors → float32 for ORT (Python `.astype(np.float32)`).
  const data =
    type === "float32"
      ? Float32Array.from(t.data)
      : type === "int64"
        ? BigInt64Array.from(t.data, (v) => BigInt(v))
        : t.data;
  return new ort.Tensor(type, data, t.shape);
}

export function fromOrt(ortTensor) {
  // Promote ORT float32 outputs back to float64 for the reverse process.
  return {
    data: Float64Array.from(ortTensor.data, Number),
    shape: ortTensor.dims.slice(),
  };
}
