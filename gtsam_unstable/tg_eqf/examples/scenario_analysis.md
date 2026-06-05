# TG-EqF example scenarios: propagation accuracy analysis

Context notes for the two IMU-only example trajectories
(`TGEqFStraightLineExample`, `TGEqFCircleExample`) driven through
`runScenario` in [`TGEqFScenarioExample.h`](TGEqFScenarioExample.h).

These runs use a **perfect IMU** (no bias, no noise) and **no measurement
updates** — pure dead-reckoning of the EqF mean. They exist to validate that
the lift and group propagation integrate the kinematics correctly, not to
demonstrate filtering performance. The configurable error model
(see [README](README.md)) lets you relax these idealizations.

## IMU rate

`RunOptions::dt` defaults to `0.01 s`, and neither example overrides it, so the
IMU rate is **100 Hz**. Both the `ScenarioRunner` sample time and the filter
`propagate(...)` step use the same `dt`.

## How the mean is propagated

Each step advances the group estimate by the exponential of the per-step lift:

```
g  <-  g * Expmap(Lambda(xi, u) * dt)
```

For the SE_2(3) navigation part this is the **exact** matrix exponential of the
lifted twist held constant over the step — strictly better than naive Euler
integration of `R, p, v`. The discretization error therefore comes only from
`Lambda` changing *within* a step, not from the exponential itself.

## Why velocity error is exactly zero (for these two scenarios)

This is expected, and it is a property of the specific trajectories, not a
general guarantee.

**Circle = constant twist.** `ConstantTwistScenario` has constant body-frame
angular rate and velocity, i.e. it is a one-parameter subgroup. The lifted
`Lambda` is constant in the body frame, so

```
Expmap(Lambda*dt) composed N times  ==  Expmap(Lambda * N*dt)   (exact)
```

Both position and velocity are reproduced exactly (0 error, any `dt`).

**Straight line = constant acceleration, no rotation.** With `R = I`
throughout, the velocity-rate slot of the lift is

```
Lambda_1.a_tilde = (a - g) - b_a + R^T g = a    (constant)
```

Velocity is linear in `t`, and integrating a constant rate is exact, so the
velocity error is zero.

## Why the straight-line position error is *not* zero

The position-rate slot is `Lambda_1.v_tilde = R^T v = v`, which **grows** as the
body accelerates. With `omega = 0` the SE_2(3) exponential has no
velocity-into-position coupling (left Jacobian is `I`), so each step adds
exactly `v_k * dt` — first-order (Euler) integration of a quadratic position.
The per-step lag is `½ a dt²`, accumulating to

```
error ≈ ½ |a| dt T = ½ (0.1)(0.01)(30) = 0.015 m
```

which matches the reported value. Being first-order, it halves with `dt`:

| `dt` (s) | final position error (m) |
|----------|--------------------------|
| 0.01     | 0.015000                 |
| 0.005    | 0.007500                 |
| 0.0025   | 0.003750                 |

(The circle stays at 0 because the constant-twist subgroup is integrated
exactly regardless of `dt`.)

## When velocity/position error stops being zero

The zeros above rely on three idealizations. Relax any of them and the
estimate degrades in the expected ways:

- **Non-zero true bias** (gyro/accel): IMU-only EqF cannot observe bias without
  measurement updates, so the filter's bias estimate stays at its initial value
  and the integrated trajectory drifts. Gyro bias tilts attitude, which
  mis-rotates gravity compensation and feeds a growing velocity/position error.
- **IMU noise:** the mean becomes a random walk; velocity and position errors
  grow roughly as `√t` (velocity) and faster for position.
- **Trajectory with intra-step variation** (jerk, time-varying `omega`):
  `Lambda` is no longer constant over a step, so even the SE_2(3) part incurs
  first-order error like the straight-line position above.

All of these are exercised by the configurable error model on `runScenario`
(`gyro_bias`, `accel_bias`, `gyro_noise_sigma`, `accel_noise_sigma`); defaults
are zero so the clean baseline above is preserved.
