function clipNoiseSchedule(alphas2, clipValue = 0.001) {
  // Match Python: prepend 1, take consecutive ratios, clip, cumprod.
  const n = alphas2.length;
  const step = new Float64Array(n);
  let prev = 1;
  for (let i = 0; i < n; i += 1) {
    let s = alphas2[i] / prev;
    if (s < clipValue) s = clipValue;
    if (s > 1) s = 1;
    step[i] = s;
    prev = alphas2[i];
  }
  const out = new Float64Array(n);
  let prod = 1;
  for (let i = 0; i < n; i += 1) {
    prod *= step[i];
    out[i] = prod;
  }
  return out;
}

function polynomialSchedule(timesteps, s = 1e-4, power = 2) {
  // Python: x = np.linspace(0, steps, steps); alphas2 = (1 - (x/steps)^power)^2
  // All math in float64, then gamma is stored as float32 (Python `.astype(np.float32)`).
  const steps = timesteps + 1;
  let alphas2 = new Float64Array(steps);
  const denom = steps - 1 || 1;
  for (let i = 0; i < steps; i += 1) {
    const xOverSteps = i / denom; // == linspace(0, steps, steps)[i] / steps
    alphas2[i] = (1 - xOverSteps ** power) ** 2;
  }
  alphas2 = clipNoiseSchedule(alphas2);
  const precision = 1 - 2 * s;
  for (let i = 0; i < alphas2.length; i += 1) {
    alphas2[i] = precision * alphas2[i] + s;
  }
  return alphas2;
}

/** Lookup table for γ_t used by the EDM sampler. */
export class PredefinedNoiseSchedule {
  constructor(timesteps, precision, power = 2) {
    this.timesteps = timesteps;
    const alphas2 = polynomialSchedule(timesteps, precision, power);
    // Python: self.gamma = (-log_alphas2_to_sigmas2).astype(np.float32)
    this.gamma = new Float32Array(alphas2.length);
    for (let i = 0; i < alphas2.length; i += 1) {
      const sigmas2 = 1 - alphas2[i];
      this.gamma[i] = -(Math.log(alphas2[i]) - Math.log(sigmas2));
    }
  }

  /** @param {{data: ArrayLike<number>, shape?: number[]} | ArrayLike<number>} t values in [0, 1] */
  call(t) {
    const data = t.data ?? t;
    const out = {
      data: new Float64Array(data.length),
      shape: t.shape ?? [data.length],
    };
    for (let i = 0; i < data.length; i += 1) {
      const idx = Math.round(data[i] * this.timesteps);
      out.data[i] = this.gamma[idx];
    }
    return out;
  }
}
