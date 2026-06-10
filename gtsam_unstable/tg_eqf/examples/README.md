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
| `--gyro-noise <σ>` | `0` | Gyro white-noise density (rad/s/√Hz) |
| `--accel-noise <σ>` | `0` | Accel white-noise density (m/s²/√Hz) |
| `--gyro-bias-rw <σ>` | `0` | Gyro bias random-walk rate (rad/s/√s) |
| `--accel-bias-rw <σ>` | `0` | Accel bias random-walk rate (m/s²/√s) |
| `--init-sigma <σ>` | `0.1` | Initial state stddev: `Sigma0 = σ²·I₁₈` |
| `--log-decim <n>` | `1` | Log every Nth step (bounds CSV size for sweeps) |
| `--seed <n>` | `42` | RNG seed for IMU noise (vary for Monte Carlo) |
| `--pos-rate <Hz>` | `0` | GNSS-like position update rate (0 = disabled) |
| `--pos-noise <σ>` | `0.1` | Position measurement stddev (m); must be > 0 when `--pos-rate` > 0 |
| `--dvl-rate <Hz>` | `0` | DVL body-velocity update rate (0 = disabled) |
| `--dvl-noise <σ>` | `0.02` | DVL measurement stddev (m/s); must be > 0 when `--dvl-rate` > 0 |

When IMU noise/RW is set, the filter `Qc` is matched to it (gyro noise → attitude
block, accel noise → velocity block, bias-RW rates → bias blocks) so the
covariance is physically meaningful for consistency analysis.

Default output files:

- Straight line: `tg_eqf_straight_line.csv`
- Circle: `tg_eqf_circle.csv`

### IMU error model

Defaults give an **ideal IMU**, reproducing the clean baseline (see
[`scenario_analysis.md`](scenario_analysis.md)). `runScenario` takes the runner's
ideal measurements and injects the error model itself: `actual + bias + noise`.
The *true* bias is added to the IMU but the filter does not know it. White-noise
`σ` is a continuous density; the discrete per-sample stddev is `σ/√dt`, drawn
from a `std::mt19937` seeded by `--seed`.

**Bias random walk.** `--gyro-bias-rw` / `--accel-bias-rw` drive the *true* bias
as a random walk (`b += rate·√dt·N(0,1)`), starting from the constant
`--*-bias` offset. The filter supports this directly: the bias states have a
process-noise model, so the matching continuous PSD (`rate²`) is wired into the
`Qc` gyro/accel-bias blocks and the filter's bias covariance grows accordingly.
Without measurement updates the bias is unobservable, so the estimate still
drifts — but the covariance now honestly reflects it.

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

Same `--seed` reproduces a run exactly; each distinct seed is an independent
noise realization, so you can Monte-Carlo by sweeping seeds:

```bash
for s in $(seq 1 50); do
  ./TGEqFCircleExample --gyro-noise 1e-3 --accel-noise 1e-2 \
    --seed $s --output /tmp/mc_$s.csv | grep "final position error"
done
```

(Bias is deterministic, so a bias-only run is identical regardless of `--seed`.)

## Simulation settings (shared)

Defined in [`TGEqFScenarioExample.h`](TGEqFScenarioExample.h).

| Setting | Value | Notes |
|---------|-------|-------|
| Filter | `TGEqF` | IMU propagation (virtual input ν = 0, b_v anchor on by default); optional GNSS position aiding via `--pos-rate` |
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
| Velocity | 3–5 | `1e-3 * I_3` |
| Position | 6–8 | (zero) |
| `b_omega` | 9–11 | `1e-6 * I_3` |
| `b_a` | 12–14 | `1e-5 * I_3` |
| `b_v` | 15–17 | (zero) |

Matches `defaultQc()` in `testTGEqF.cpp` (tangent order `[att, vel, pos, b_w,
b_a, b_v]`). The attitude (0–2) / velocity (3–5) blocks are overridden to
`sigma²·I_3` when `--gyro-noise` / `--accel-noise` are set, and `b_omega` (9–11)
/ `b_a` (12–14) to `rate²·I_3` when `--gyro-bias-rw` / `--accel-bias-rw` are set,
matching the simulated IMU noise and bias random walk.

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

Columns are keyed by name (read with `csv.DictReader`), so consumers select what
they need. The full set (see `csvHeader()` in `TGEqFScenarioExample.h`):

| Columns | Unit | Description |
|---------|------|-------------|
| `t` | s | Simulation time |
| `gt_p*`, `gt_v*` | m, m/s | Ground-truth position / velocity (nav frame) |
| `est_p*`, `est_v*` | m, m/s | Estimate from `TGEqF::position()` / `velocity()` |
| `att_err_*` | rad | Attitude error `Logmap(gt_Rᵀ·est_R)` |
| `est_bg*`, `est_bv*`, `est_ba*` | — | Estimated gyro / virtual-vel / accel bias |
| `true_bg*`, `true_ba*` | — | True (simulated) gyro / accel bias |
| `eps_<grp>_*` | — | EqF origin-frame error per group (att,pos,vel,bg,bv,ba) |
| `P_<grp>_ij` | — | Upper triangle of each 3×3 origin-frame covariance block |

`plot_trajectory.py` uses only the `gt_*`/`est_*` columns; `monte_carlo.py` uses
`eps_*` and `P_*` for the per-group NEES consistency analysis. See
[`scenario_analysis.md`](scenario_analysis.md) for the accuracy discussion and
[`../scripts/README.md`](../scripts/README.md) for the Monte-Carlo workflow.

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
| `../scripts/monte_carlo.py` | Seed-sweep Monte-Carlo: error envelopes + NEES |
| `../scripts/README.md` | Plotting workflow |
