# TG-EqF trajectory examples

Runnable C++ examples that simulate ground-truth motion with GTSAM
`Scenario` / `ScenarioRunner`, propagate `TGEqF` with IMU only, and write CSV
trajectories for comparison with Python plots in `../scripts/`.

## Build and run

```bash
cd gtsam/build
cmake --build . --target TGEqFStraightLineExample TGEqFCircleExample

./gtsam_unstable/tg_eqf/examples/TGEqFStraightLineExample --output tg_eqf_straight_line.csv
./gtsam_unstable/tg_eqf/examples/TGEqFCircleExample --output tg_eqf_circle.csv

python3 ../gtsam_unstable/tg_eqf/scripts/plot_trajectory.py tg_eqf_straight_line.csv
python3 ../gtsam_unstable/tg_eqf/scripts/plot_trajectory.py tg_eqf_circle.csv --output-dir plots/
```

### CLI options (both examples)

| Flag | Default | Description |
|------|---------|-------------|
| `--output <path>` | scenario-specific (below) | Output CSV path |
| `--duration <s>` | `30.0` | Simulation length (seconds) |
| `--dt <s>` | `0.01` | IMU sample period (100 Hz) |
| `--gyro-bias <x,y,z>` | `0,0,0` | Constant true gyro bias (rad/s) |
| `--accel-bias <x,y,z>` | `0,0,0` | Constant true accel bias (m/s²) |
| `--gyro-noise <σ>` | `0` | Gyro noise density (rad/s/√Hz) |
| `--accel-noise <σ>` | `0` | Accel noise density (m/s²/√Hz) |

Default output files:

- Straight line: `tg_eqf_straight_line.csv`
- Circle: `tg_eqf_circle.csv`

### IMU error model

Defaults give an **ideal IMU**, reproducing the clean baseline (see
[`scenario_analysis.md`](scenario_analysis.md)). Any non-zero bias/noise switches
the runner to corrupted measurements (`measuredAngularVelocity` /
`measuredSpecificForce`): the *true* bias is injected into the IMU but the filter
does not know it. Noise `σ` is a continuous density; the discrete per-sample
stddev is `σ/√dt`.

These examples run **IMU-only with no aiding**, so unaided inertial
dead-reckoning drifts unbounded under any imperfection — expected physics, not a
filter fault. Bounding the drift requires `update_dvl` / `update_position`.
Illustrative values:

```bash
# attitude/velocity drift from a small gyro bias
./TGEqFCircleExample --gyro-bias 0,0,0.002

# realistic MEMS noise densities (drifts ~tens of metres over 30 s, unaided)
./TGEqFCircleExample --gyro-noise 1e-3 --accel-noise 1e-2
```

Noise samplers in `ScenarioRunner` use fixed internal seeds, so corrupted runs
are reproducible.

## Simulation settings (shared)

Defined in [`TGEqFScenarioExample.h`](TGEqFScenarioExample.h).

| Setting | Value | Notes |
|---------|-------|-------|
| Filter | `TGEqF` | IMU propagation only; no `update_position` / `update_dvl` |
| Reference state `xi_ref` | `scenario.navState(0)` | EqF chart origin = true initial state (IMU-only cannot observe it) |
| Initial covariance `Sigma0` | `0.01 * I_18` | Same scale as `testTGEqF.cpp` |
| Gravity | `9.81 m/s²`, Z-up ENU | `PreintegrationParams::MakeSharedU(9.81)` → `n_gravity = (0, 0, -9.81)` |
| IMU bias (simulation) | Zero (configurable) | `--gyro-bias` / `--accel-bias` → `imuBias::ConstantBias` |
| IMU noise | None (configurable) | `--gyro-noise` / `--accel-noise`; clean run uses `actual*`, corrupted uses `measured*` |
| Process noise `Qc` | Block-diagonal | See table below |
| GT source | `scenario.navState(t)` | GTSAM `Scenario` interface |
| IMU synthesis | `ScenarioRunner` | `actualAngularVelocity(t)`, `actualSpecificForce(t)` |

### Process noise `Qc` (continuous-time, 18×18)

| Tangent block | Index | Variance scale |
|---------------|-------|----------------|
| Attitude | 0–2 | `1e-4 * I_3` |
| Position | 3–5 | (zero) |
| Velocity | 6–8 | `1e-3 * I_3` |
| `b_omega` | 9–11 | `1e-6 * I_3` |
| `b_v` | 12–14 | (zero) |
| `b_a` | 15–17 | `1e-5 * I_3` |

Matches `defaultQc()` in `testTGEqF.cpp`.

### Logged estimate trajectory

The lift integrates both velocity and position into the group element (the
`T^{-1} f1(T)` term encodes `dp = v`). The CSV logs the filter state directly:

- **Velocity:** `filter.velocity()` after each `propagate` step
- **Position:** `filter.position()` after each `propagate` step

Ground truth position and velocity come directly from `scenario.navState(t)`.

## Scenario: straight line

**Executable:** `TGEqFStraightLineExample.cpp`  
**GTSAM class:** `AcceleratingScenario`

| Parameter | Value |
|-----------|-------|
| Initial pose | `R = I`, `p = 0` |
| Initial velocity | `v0 = 0.5 * dir` (oblique) |
| Navigation acceleration | `a_n = 0.1 * dir`, `dir = normalize(2, 1, 0.5)` |
| Body angular rate | `omega_b = 0` |

A non-axis-aligned 3D line: velocity and acceleration are parallel (`dir`) so the
path stays straight while exercising all three axes and a non-rest initial state.
The constant acceleration also gives the accelerometer specific force to observe.

## Scenario: circle

**Executable:** `TGEqFCircleExample.cpp`  
**GTSAM class:** `ConstantTwistScenario` (same pattern as `NavStateImuExample.cpp`)

| Parameter | Value |
|-----------|-------|
| Radius | `R = 10 m` |
| Speed | `v = 1 m/s` |
| Angular rate | `omega = v/R = 0.1 rad/s` |
| Body twist | `w_b = (0, 0, -omega)` |
| Body linear twist | `v_b = (R*omega, 0, 0) = (1, 0, 0) m/s` |

Motion is a horizontal-plane constant-twist orbit starting from the origin at
`t = 0` with initial tangential velocity `(1, 0, 0) m/s`.

## CSV format

```
t,gt_px,gt_py,gt_pz,gt_vx,gt_vy,gt_vz,est_px,est_py,est_pz,est_vx,est_vy,est_vz
```

| Column | Unit | Description |
|--------|------|-------------|
| `t` | s | Simulation time |
| `gt_p*` | m | Ground-truth position (navigation frame) |
| `gt_v*` | m/s | Ground-truth velocity (navigation frame) |
| `est_p*` | m | Estimate position from `TGEqF::position()` |
| `est_v*` | m/s | Estimate velocity from `TGEqF::velocity()` |

## Expected behaviour (30 s, default `dt`)

| Scenario | Velocity tracking | Position tracking |
|----------|-------------------|-------------------|
| Straight line (accel) | ~0 m/s error | ~1.5 cm final error (first-order discretization) |
| Circle | ~0 m/s error | ~0 error (constant twist integrates exactly) |

With the filter initialized at the true start state and the lift correctly
encoding `dp = v`, IMU-only propagation reproduces both scenarios with only
discretization error (no position/DVL updates needed for these noise-free runs).

## Files

| File | Role |
|------|------|
| `TGEqFScenarioExample.h` | Shared run loop, CSV writer, defaults |
| `TGEqFStraightLineExample.cpp` | Accelerating straight-line scenario |
| `TGEqFCircleExample.cpp` | Constant-twist circle scenario |
| `CMakeLists.txt` | Registers examples via `gtsamAddExamplesGlob` |
| `../scripts/plot_trajectory.py` | Plot GT vs estimate and error time series |
| `../scripts/README.md` | Plotting workflow |
