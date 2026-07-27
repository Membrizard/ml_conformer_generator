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

  // Full-array characterization golden (timesteps=10, precision=1e-5) — catches
  // any drift in the schedule, incl. a float32→float64 regression on gamma.
  it("full gamma table golden (timesteps=10, precision=1e-5)", () => {
    const sched = new PredefinedNoiseSchedule(10, 1e-5, 2);
    const golden = [
      -11.51291561126709, -3.8964426517486572, -2.4641706943511963,
      -1.5721749067306519, -0.8740893006324768, -0.251309335231781,
      0.365611732006073, 1.045423984527588, 1.9044344425201416,
      3.284428119659424, 9.98466682434082,
    ];
    assert.equal(sched.gamma.length, golden.length);
    for (let i = 0; i < golden.length; i += 1) {
      assert.ok(
        Math.abs(sched.gamma[i] - golden[i]) < 1e-5,
        `gamma[${i}]: got ${sched.gamma[i]}, want ${golden[i]}`,
      );
    }
  });

  it("100-step schedule has 101 entries", () => {
    assert.equal(new PredefinedNoiseSchedule(100, 1e-5, 2).gamma.length, 101);
  });
});
