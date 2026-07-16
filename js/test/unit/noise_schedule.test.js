import assert from "node:assert/strict";
import { describe, it } from "node:test";
import { PredefinedNoiseSchedule } from "../../src/noiseSchedule.js";

describe("PredefinedNoiseSchedule", () => {
  it("builds gamma table of length timesteps+1", () => {
    const sched = new PredefinedNoiseSchedule(50, 1e-4, 2);
    assert.equal(sched.gamma.length, 51);
    assert.equal(sched.timesteps, 50);
  });

  it("gamma entries are finite Float32 values", () => {
    const sched = new PredefinedNoiseSchedule(100, 1e-4, 2);
    for (let i = 0; i < sched.gamma.length; i += 1) {
      assert.ok(Number.isFinite(sched.gamma[i]), `gamma[${i}]`);
    }
  });

  it("call() indexes by round(t * timesteps)", () => {
    const sched = new PredefinedNoiseSchedule(10, 1e-4, 2);
    const out = sched.call({ data: [0, 0.5, 1], shape: [3] });
    assert.equal(out.data[0], sched.gamma[0]);
    assert.equal(out.data[1], sched.gamma[5]);
    assert.equal(out.data[2], sched.gamma[10]);
  });

  it("matches fixed golden gamma endpoints for timesteps=50", () => {
    // Python PredefinedNoiseSchedule(gamma_interpolator) polynomial power=2
    const sched = new PredefinedNoiseSchedule(50, 1e-4, 2);
    assert.ok(Math.abs(sched.gamma[0] - -9.210240364074707) < 1e-5);
    assert.ok(Math.abs(sched.gamma[50] - 9.194682121276855) < 1e-5);
  });
});
