# TFG-IEKF — Implementation

Code map, API, build, and tests for the `tfg_inekf` module. For the theory see
[design.md](design.md).

## File map

```
tfg_inekf/
  Group.h / .cpp          TwoFrameGroup: G_TF as a gtsam LieGroup (traits)
  Symmetry.h / .cpp       action phi, lift Lambda, increment, gravity
  PositionOutput.h / .cpp body-residual measurement h, output action, C0 / C*
  InEKF.h / .cpp          TfgInEKF: gtsam::InvariantEKF<TwoFrameGroup> wrapper
  CMakeLists.txt          header install + tests + examples
  tests/                  testTFG{Group,Symmetry,PositionOutput,InEKF}.cpp
  examples/               IMU-only trajectory runners (see below)
  docs/                   design.md, implementation.md, reference PDF
```

Layering: `Group` <- `Symmetry`, `PositionOutput` <- `InEKF`. No separate
`State` type — the filter state *is* the group element (cf. design §2).

## Group.h — `tfg::TwoFrameGroup`

15-dim Lie group, members `R (Rot3), v, p, gamma_omega, gamma_accel`.

| Member | Meaning |
|--------|---------|
| `TwoFrameGroup(R, v, p, g_omega, g_accel)` | from group coordinates |
| `FromState(R, v, p, b_omega, b_accel)` | from physical state, `gamma = -R*b` |
| `bias_omega()`, `bias_accel()` | physical biases `A^{-1}*(-gamma)` (Eq. B.31) |
| `operator*`, `inverse()`, `Identity()` | group ops (Sec. 5.3) |
| `Expmap(xi)`, `Logmap()` | exact, shared left Jacobian on all blocks |
| `AdjointMap()` | 15x15 (design §2) |
| `C_matrix()` | 5x5 SE_2(3) realisation `[R|v|p; 0 1 0; 0 0 1]` |

Full `gtsam::traits<TwoFrameGroup>` specialisation models the `LieGroup`
concept, so the type is a valid `InvariantEKF<G>` parameter. `Retract = X*Expmap`,
`Local(X,Y) = Logmap(X^-1 Y)` (right convention). Compose/Inverse Jacobians use
the adjoint (`Hx = Ad_{Y^-1}`, `Inverse = -Ad_X`).

## Symmetry.h — action and lift

```cpp
struct ImuInput { Vector3 omega, accel, tau_omega{0}, tau_accel{0}; };

Vector3       gravity();                              // (0,0,-9.81)
TwoFrameGroup phi(const TwoFrameGroup& X, const TwoFrameGroup& xi); // = xi*X
Tangent       lift(const TwoFrameGroup& X, const ImuInput& u);      // Eq. 17-18
TwoFrameGroup increment(const TwoFrameGroup& X, const ImuInput& u, double dt);
```

`lift` evaluates the closed form (design §3) directly — no 5x5 wedge/inverse.

## PositionOutput.h — `tfg::PositionOutput`

```cpp
enum class Variant { C0, Cstar };

Vector3            predict(xi, pi);                    // h = R^T(pi - p)   (Eq.34)
Vector3            output_action(X, y);                // rho_X             (Eq.35)
Matrix<3,15>       jacobian(xi_hat, pi, variant = Cstar);  // C0 / C* (Eq.B.35)
Vector3            innovation(xi_hat, pi);             // z - h = -y_hat
```

## InEKF.h — `tfg::TfgInEKF`

```cpp
TfgInEKF(const TwoFrameGroup& X0, const Covariance& P0);   // Covariance = 15x15

void propagate(const Vector3& omega, const Vector3& accel,
               const Covariance& Qc, double dt);            // Qc continuous

void update_position(const Vector3& pi, const Covariance3& R_pos,
                     bool use_cstar = true);                // C* by default

// accessors (physical state, Eq. B.31)
Rot3    attitude();   Vector3 velocity();   Vector3 position();
Vector3 bias_gyro();  Vector3 bias_accel();
```

`propagate` builds `U = increment(state(), u, dt)` and calls
`InvariantEKF::predict(U, Qc*dt)`. `update_position` forms the body residual and
calls `ManifoldEKF::updateWithVector(prediction, H, z=0, R_meas)` with
`R_meas = R^T R_pos R` and `performReset = false`.

### Minimal usage

```cpp
using namespace tfg;
auto X0 = TwoFrameGroup::FromState(R0, v0, p0, bg0, ba0);
TfgInEKF ekf(X0, P0);                       // 15x15 P0
for (each IMU sample) {
  ekf.propagate(omega, accel, Qc, dt);      // Qc 15x15 continuous PSD
  if (gnss_available) ekf.update_position(pi, R_pos);   // C* default
}
Rot3 R = ekf.attitude(); Vector3 p = ekf.position();
```

## Build & test

Configure + build from the gtsam root (see repo memory):

```
cd <gtsam>
cmake -S . -B build -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
cmake --build build --target testTFGGroup testTFGSymmetry \
      testTFGPositionOutput testTFGInEKF -j$(nproc)
```

Binaries: `build/gtsam_unstable/tfg_inekf/tests/`. The module is wired into
`gtsam_unstable/CMakeLists.txt` (subdir + `${tfg_inekf_srcs}`).

| Suite | Covers |
|-------|--------|
| `testTFGGroup` | group axioms, matrix-realisation ground truth, Exp/Log, Adjoint (conjugation + homomorphism), Retract/Local |
| `testTFGSymmetry` | right-action axiom, lift closed form vs Eq. 3, predict-vs-world-dynamics, bias random walk |
| `testTFGPositionOutput` | h value, equivariance, C0 finite-difference, C* structure, C*→C0 when centred |
| `testTFGInEKF` | construct, predict SPD/mean, position update, static convergence (C0), C* improves convergence over C0 |

## Examples (IMU-only)

Dead-reckoning with `propagate` only (no position updates) — validates the
predict step. Ideal IMU must reproduce ground truth; bias/noise must drift
physically. Shared runner `examples/TFGInEKFScenarioExample.h` (gtsam
`Scenario`/`ScenarioRunner`, CSV + RMS summary).

```
Static  ConstantVelocity  CoordinatedTurn  PureRotation
Vertical  StraightLine  Sinusoid
```

Constant-twist scenarios track exactly; the accelerating ones show sub-cm RMS
(O(dt^2) discretisation between the continuous scenario and the per-step
increment). Common flags:

```
--duration --dt --gyro-bias x,y,z --accel-bias x,y,z
--gyro-noise --accel-noise --gyro-bias-rw --accel-bias-rw
--init-sigma --seed --log-decim --output
```

Example:

```
build/gtsam_unstable/tfg_inekf/examples/TFGInEKFCoordinatedTurnExample \
    --duration 20 --accel-bias 0.05,0,0 --output turn.csv
```

CSV columns: time, ground-truth pos/vel, estimate pos/vel, attitude error,
estimated/true biases, and the upper triangle of each 3x3 covariance block in
tangent order `[att, vel, pos, bg, ba]`.

## Status & follow-ups

45 unit tests pass; 7 IMU-only examples build and run. Open items (see design §6):

- GNSS update mode in the example runner (currently IMU-only).
- Exact bias-error coupling `A_t^0` (Eq. B.34) instead of `Ad_{U^-1}`.
