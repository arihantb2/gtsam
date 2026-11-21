## Summary

Adds three biased inertial navigation system (INS) filters to `gtsam_unstable/`. All three share one scenario harness, so they can be compared on the same trajectories and noise draws.

| Directory | Filter | State | Built on |
|---|---|---|---|
| `tg_eqf/` | **TG-EqF**: Equivariant Filter on the tangent group SE₂(3) ⋉ se₂(3) | 18-dim `[R, v, p, b_w, b_a, b_v]` | `EquivariantFilter` |
| `tfg_inekf/` | **TFG-InEKF**: Invariant EKF on the Two-Frames Group SO(3) × ℝ¹² | 15-dim | `InvariantEKF` |
| `mekf/` | **MEKF**: multiplicative EKF baseline | 15-dim | `ManifoldEKF` |

**Measurements:** position, DVL body-frame velocity, depth, and a virtual-bias output (TG-EqF).

**Shared harness (`examples_common/`):** ground-truth scenarios (typical navigation, vertical, waypoint spline, and a survey path taken from a real AUV mission), IMU and aiding simulation, a common run loop, and CSV logging. There is one example executable per filter for each scenario.

**Python:** TG-EqF is wrapped as `gtsam_unstable.tgeqf`. An example is in `python/gtsam_unstable/examples/TGEqFExample.py`.

## Tests

Each filter has unit tests in `tests/` covering:
- group, symmetry and lift operations, checked against numerical derivatives
- the Jacobian of each output model
- a TG-EqF regression on biased IMU input with position aiding
🤖 Generated with [Claude Code](https://claude.com/claude-code)

https://claude.ai/code/session_01XCGpUjEFrj2U13ACUVDBkh
