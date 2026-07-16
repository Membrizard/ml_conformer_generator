import { IN_NODE_NF, N_DIMS } from "./constants.js";
import { PredefinedNoiseSchedule } from "./noiseSchedule.js";
import {
  add,
  argmaxLast,
  concatLast,
  div,
  fromOrt,
  full,
  inflateBatch,
  logSigmoid,
  mul,
  oneHot,
  removeMeanWithMask,
  sampleCenterGravityZeroGaussianWithMask,
  sampleGaussianWithMask,
  sliceLast,
  softplus,
  sub,
  toOrt,
  zeros,
} from "./tensor.js";

/**
 * E(n) diffusion sampler backed by an EGNN ONNX session.
 * Mirrors `EquivariantDiffusionONNX` in the Python package.
 *
 * Pass any ONNX Runtime namespace (`onnxruntime-node`, `onnxruntime-web`, …)
 * via `create` / the constructor — the core package does not depend on one.
 */
export class EquivariantDiffusion {
  constructor(egnnSession, ort, { timesteps = 100, noisePrecision = 1e-5 } = {}) {
    if (!ort?.Tensor) {
      throw new TypeError(
        "EquivariantDiffusion requires an ONNX Runtime module (`ort`).",
      );
    }
    this.ort = ort;
    this.dynamics = egnnSession;
    this.gamma = new PredefinedNoiseSchedule(timesteps, noisePrecision);
    this.inNodeNf = IN_NODE_NF;
    this.nDims = N_DIMS;
    this.numClasses = IN_NODE_NF;
    this.T = timesteps;
    this.normValues = [1.0, 9.0];
  }

  static async create(egnnOnnxPath, ort, options) {
    if (!ort?.InferenceSession) {
      throw new TypeError(
        "Pass an ONNX Runtime module as the second argument (e.g. import * as ort from 'onnxruntime-node').",
      );
    }
    const session = await ort.InferenceSession.create(egnnOnnxPath);
    return new EquivariantDiffusion(session, ort, options);
  }

  async phi(x, t, nodeMask, edgeMask, context) {
    const feeds = {
      t: toOrt(this.ort, t),
      xh: toOrt(this.ort, x),
      node_mask: toOrt(this.ort, nodeMask),
      edge_mask: toOrt(this.ort, edgeMask),
      context: toOrt(this.ort, context),
    };
    const out = await this.dynamics.run(feeds);
    return fromOrt(out[this.dynamics.outputNames[0]]);
  }

  sigma(gamma, target) {
    return inflateBatch(
      {
        data: Float64Array.from(gamma.data, (g) =>
          Math.sqrt(1 / (1 + Math.exp(-g))),
        ),
        shape: gamma.shape,
      },
      target,
    );
  }

  alpha(gamma, target) {
    return inflateBatch(
      {
        data: Float64Array.from(gamma.data, (g) =>
          Math.sqrt(1 / (1 + Math.exp(g))),
        ),
        shape: gamma.shape,
      },
      target,
    );
  }

  snr(gamma) {
    return {
      data: Float64Array.from(gamma.data, (g) => Math.exp(-g)),
      shape: gamma.shape,
    };
  }

  sigmaAndAlphaTGivenS(gammaT, gammaS, target) {
    const softS = softplus(gammaS);
    const softT = softplus(gammaT);
    const sigma2 = inflateBatch(
      {
        data: Float64Array.from(
          softS.data,
          (_, i) => 1 - Math.exp(softS.data[i] - softT.data[i]),
        ),
        shape: softS.shape,
      },
      target,
    );

    const logAlpha2T = logSigmoid(mul(gammaT, -1));
    const logAlpha2S = logSigmoid(mul(gammaS, -1));
    const alphaTGivenS = inflateBatch(
      {
        data: Float64Array.from(logAlpha2T.data, (_, i) =>
          Math.exp(0.5 * (logAlpha2T.data[i] - logAlpha2S.data[i])),
        ),
        shape: logAlpha2T.shape,
      },
      target,
    );

    const sigmaTGivenS = {
      data: Float64Array.from(sigma2.data, Math.sqrt),
      shape: sigma2.shape,
    };
    return { sigma2TGivenS: sigma2, sigmaTGivenS, alphaTGivenS };
  }

  unnormalize(x, hCat, nodeMask) {
    x = mul(x, this.normValues[0]);
    hCat = mul(hCat, this.normValues[1]);
    const [B, N, C] = hCat.shape;
    for (let b = 0; b < B; b += 1) {
      for (let i = 0; i < N; i += 1) {
        const m = nodeMask.data[b * N + i];
        for (let c = 0; c < C; c += 1) hCat.data[(b * N + i) * C + c] *= m;
      }
    }
    return { x, hCat };
  }

  computeXPred(netOut, zt, gammaT) {
    const sigmaT = this.sigma(gammaT, netOut);
    const alphaT = this.alpha(gammaT, netOut);
    return div(sub(zt, mul(sigmaT, netOut)), alphaT);
  }

  /**
   * Python `sample_combined_position_feature_noise`:
   * COM-zero Gaussian for coordinates, plain Gaussian for atom features.
   */
  sampleCombinedNoise(nSamples, nNodes, nodeMask) {
    const zX = sampleCenterGravityZeroGaussianWithMask(
      [nSamples, nNodes, this.nDims],
      nodeMask,
    );
    const zH = sampleGaussianWithMask(
      [nSamples, nNodes, this.inNodeNf],
      nodeMask,
    );
    return concatLast(zX, zH);
  }

  sampleNormal(mu, sigma, nodeMask) {
    const eps = this.sampleCombinedNoise(mu.shape[0], mu.shape[1], nodeMask);
    return add(mu, mul(sigma, eps));
  }

  async samplePZsGivenZt(s, t, zt, nodeMask, edgeMask, context) {
    const gammaS = this.gamma.call(s);
    const gammaT = this.gamma.call(t);
    const { sigma2TGivenS, sigmaTGivenS, alphaTGivenS } =
      this.sigmaAndAlphaTGivenS(gammaT, gammaS, zt);

    const sigmaS = this.sigma(gammaS, zt);
    const sigmaT = this.sigma(gammaT, zt);
    const epsT = await this.phi(zt, t, nodeMask, edgeMask, context);

    // mu = zt / alpha - (sigma2 / alpha / sigma_t) * eps
    const mu = sub(
      div(zt, alphaTGivenS),
      mul(div(div(sigma2TGivenS, alphaTGivenS), sigmaT), epsT),
    );
    const sigma = div(mul(sigmaTGivenS, sigmaS), sigmaT);
    let zs = this.sampleNormal(mu, sigma, nodeMask);

    const coords = removeMeanWithMask(sliceLast(zs, 0, this.nDims), nodeMask);
    const feats = sliceLast(zs, this.nDims);
    return concatLast(coords, feats);
  }

  async samplePXhGivenZ0(z0, nodeMask, edgeMask, context) {
    const zerosT = zeros([z0.shape[0], 1]);
    const gamma0 = this.gamma.call(zerosT);
    const snrVal = this.snr(mul(gamma0, -0.5));
    // expand_dims(snr, 1) → (B, 1, 1) broadcast via inflate
    const sigmaX = inflateBatch(snrVal, z0);

    const netOut = await this.phi(z0, zerosT, nodeMask, edgeMask, context);
    const muX = this.computeXPred(netOut, z0, gamma0);
    const xh = this.sampleNormal(muX, sigmaX, nodeMask);
    const x = sliceLast(xh, 0, this.nDims);

    // Match Python exactly: z0[:, :, n_dims:-1] (drops last latent channel).
    const { x: xOut, hCat } = this.unnormalize(
      x,
      sliceLast(z0, this.nDims, -1),
      nodeMask,
    );

    let h = oneHot(argmaxLast(hCat), this.numClasses, [z0.shape[0], z0.shape[1]]);
    const [B, N, C] = h.shape;
    for (let b = 0; b < B; b += 1) {
      for (let i = 0; i < N; i += 1) {
        const m = nodeMask.data[b * N + i];
        for (let c = 0; c < C; c += 1) h.data[(b * N + i) * C + c] *= m;
      }
    }
    return { x: xOut, h };
  }

  async sample(nodeMask, edgeMask, context, resampleSteps = 0) {
    const [nSamples, nNodes] = nodeMask.shape;
    let z = this.sampleCombinedNoise(nSamples, nNodes, nodeMask);

    for (let step = this.T - 1; step >= 0; step -= 1) {
      const sArray = full([nSamples, 1], step / this.T);
      const tArray = full([nSamples, 1], (step + 1) / this.T);

      for (let r = 0; r < resampleSteps; r += 1) {
        z = await this.samplePZsGivenZt(sArray, tArray, z, nodeMask, edgeMask, context);
      }
      z = await this.samplePZsGivenZt(sArray, tArray, z, nodeMask, edgeMask, context);
    }

    return this.samplePXhGivenZ0(z, nodeMask, edgeMask, context);
  }
}
