# Code review: `gtsam_unstable/tfg_inekf` (TFG-IEKF)

Reference: A. Fornasier, Y. Ng, C. van Goor, T. Hamel, R. Mahony, S. Weiss,
*"Equivariant Symmetries for Inertial Navigation Systems"*, arXiv:2309.03765v3.
All equation numbers below (Eq. 3, 16–18, 33–35, Table 2, B.31–B.35) refer to
that paper. Scope: `Group.{h,cpp}`, `Symmetry.{h,cpp}`, `PositionOutput.{h,cpp}`,
`InEKF.{h,cpp}`, `tests/`, `examples/`.

Every claim below was re-derived independently and cross-checked against the
paper text. Findings are ordered by severity.

---

## Summary

| ID | Severity | Component | Issue |
|----|----------|-----------|-------|
| F1 | **Critical** | `InEKF.cpp::propagate` | Covariance propagation omits the bias/navigation error coupling of the TFG-IEKF state matrix (Table 2, Eq. B.34). Consequence: IMU biases are **structurally unobservable** — position updates can never correct them. |
| F2 | **Critical** | `PositionOutput.cpp::jacobian` (`Cstar`) | The C\* output matrix (Eq. B.35) is transplanted from the paper's global/origin error chart into this implementation's body-frame chart without the required chart transform, and misreads the meaning of `y`. The implemented coupling `½(ŷ + p̂)^` breaks translation invariance; the correct body-chart value is `½ ŷ^`. |
| F3 | High (latent) | `Group.cpp` traits `Expmap`/`Logmap` | Chart Jacobians returned as Identity (marked TODO). Silently wrong for any consumer that requests them — including the natural fix for F1 via `LieGroupEKF::predict`. |
| F4 | Medium | `InEKF.cpp::propagate`, examples `defaultQc` | Process noise is added as a user-supplied diagonal `Qc·dt` without mapping IMU input noise through the input matrix; gyro-noise coupling into the bias (γ) rows is dropped. |
| F5 | Medium | `tests/` | The test suite is constructed (inadvertently) so that F1 and F2 are invisible: no bias-convergence test, C\* tested only with truth at the origin and `R = I`, static test pinned to C0. |
| F6 | Low | `PositionOutput` | `innovation()` is dead code with an unresolved sign-convention comment; misuse-prone. |
| F7 | Low | `examples/` | Consequences of F1 visible in CSV output (bias estimates frozen); hard-coded gravity; minor CLI parsing robustness. |

What was checked and found **correct** is listed at the end — the group
algebra, the lift, the equivariance properties, and the C0 update path are all
right; the two critical problems are confined to covariance propagation and the
C\* variant.

---

## F1 (critical, logical): predict step drops the bias→navigation error coupling — biases are unobservable

**Where:** `InEKF.cpp:37-38` (`propagate` → `InvariantEKF::predict(U, Qc*dt)`),
acknowledged in the header comment `InEKF.cpp:10-15`.

**What the paper requires.** Table 2 (TFG-IEKF row) and Eq. (B.34) give the
linearized error dynamics `ε̇ ≃ A⁰ₜ ε`:

```
ε̇_R  = −ε_bω
ε̇_v  =  g^ ε_R − v̂^ ε_bω − ε_ba
ε̇_p  =  ε_v − p̂^ ε_bω
ε̇_bω = (R̂(ω − b̂ω))^ ε_bω
ε̇_ba = (R̂(ω − b̂ω))^ ε_ba
```

The defining feature of this matrix is that **bias errors drive every
navigation error row** (the `−ε_bω`, `−v̂^ε_bω − ε_ba`, `−p̂^ε_bω` terms). This
coupling exists because the lift Λ (Eq. 17–18) is state-dependent: it is
evaluated at the *estimated* biases, so an error in the bias estimate produces
a first-order error in the propagated navigation state.

**What the code does.** `propagate` calls `InvariantEKF::predict(U, Q)`, which
propagates `P ← Ad_{U⁻¹} P Ad_{U⁻¹}ᵀ + Q`. For this group, `Ad` (see
`Group.cpp::AdjointMap`) has *block-diagonal-only* columns for the two γ
blocks: the only off-diagonal entries are `skew(uᵢ)·A` under the `d_theta`
column. Hence the state-transition matrix used by the filter contains **no
bias→attitude, bias→velocity, or bias→position terms at all** — not an
approximation of (B.34), but its complete omission. The comment in
`InEKF.cpp:12-15` calls this "a possible refinement"; it is in fact load-bearing.

**Provable consequence.** Write `P` in blocks over (nav = θ,v,p | bias = γω,γa).
With `A = Ad_{U⁻¹}` block-lower-triangular w.r.t. this split (bias columns
diagonal-only) and the measurement Jacobian `H` having zero bias columns
(`PositionOutput::jacobian` zeroes columns 9–14, correctly):

1. If `P₀` has zero nav–bias cross-covariance (true for every test and every
   example: all use diagonal `P₀`), then `P_{nav,bias} = 0` is invariant under
   both the predict (`A P Aᵀ` preserves the zero block) and the Joseph update.
2. The bias rows of the Kalman gain are `K_bias = P_{bias,:} Hᵀ S⁻¹ =
   P_{bias,nav} H_navᵀ S⁻¹ = 0` — identically, forever.

So the bias estimates produced by this filter **never move** in response to
measurements, and the navigation covariance never inflates due to bias
uncertainty (overconfident, inconsistent filter). The filter implemented here
is an unbiased-IMU left-IEKF carrying two dead 3-vectors, not the TFG-IEKF of
the paper. Run any example with `--gyro-bias 0.01,0,0 --pos-rate 10` and the
`est_bg*` CSV columns stay at zero while the position error exhibits a
persistent bias-induced offset.

**Correction.** Propagate `P` with the full linearization. Two equivalent
routes:

*Route A — explicit state matrix in the implementation's chart.* The module's
error chart is right-multiplicative (`X = X̂·Exp(ε)`, tangent order
`[θ, v, p, γω, γa]`), which is a different chart from the paper's
(`e = φ(X̂⁻¹, ξ)`, global frame), so (B.34) cannot be pasted verbatim. In this
chart the continuous-time error dynamics are `ε̇ = (−ad_λ + DΛ) ε` where
`−ad_λ` is what `Ad_{U⁻¹}` already provides over a step, and `DΛ` is the
missing differential of the lift. Using `b̂ = bias estimate`,
`ŵ = ω − b̂ω` and `δb = −ε_γ + b̂^ ε_θ` (differential of `b = −Rᵀγ`):

```
DΛ[θ ,θ] = −b̂ω^                DΛ[θ ,γω] = I
DΛ[v ,θ] = (R̂ᵀg)^ − b̂a^       DΛ[v ,γa] = I
DΛ[p ,θ] = (R̂ᵀv̂)^             DΛ[p ,v ] = I
DΛ[γω,θ] = −ω^ b̂ω^             DΛ[γω,γω] = ω^
DΛ[γa,θ] = −ŵ^ b̂a^ − b̂a^ b̂ω^  DΛ[γa,γω] = b̂a^ ,  DΛ[γa,γa] = ŵ^
```

(Each row is the directional derivative of the corresponding block of
`lift()` in `Symmetry.cpp:33-45` w.r.t. the right-chart perturbation; the
`[γω,θ]` entry simplifies via `δb×ŵ − b̂ω×δb = δb×ω`.) Then propagate with
`F = Ad_{U⁻¹}·(I + DΛ·dt)` (first order) instead of `Ad_{U⁻¹}`:

```cpp
// in TfgInEKF::propagate, replacing this->predict(U, Qc*dt):
const auto F = (U.inverse().AdjointMap() *
                (Covariance::Identity() + Dlambda(state(), u) * dt)).eval();
// X ← X*U;  P ← F P Fᵀ + Qc dt   (ManifoldEKF::predict(X_next, F, Q))
```

Verify `F` against a finite difference of
`ε ↦ Log( (X̂ Exp(ε) · U(X̂ Exp(ε)))⁻¹ · noise-free-propagation )` — a unit
test that currently does not exist (see F5).

*Route B — use `gtsam::LieGroupEKF`.* `LieGroupEKF<G>::predict(f, dt, Q)`
already accepts state-dependent dynamics `λ = f(X, Df)` and composes the
transition matrix for you; pass `f = lift(X, u)` with `Df = −ad_λ + DΛ`-style
Jacobian per its convention. **Caveat:** this path consumes
`traits<G>::Expmap(xi, H)` — whose Jacobian is currently a stub (F3); fix F3
first.

Either way, also assert in a test that a GNSS-aided run converges to a nonzero
true gyro/accel bias (see F5).

---

## F2 (critical, logical): C\* is the paper's matrix in the wrong chart, with the wrong `y` — breaks translation invariance

**Where:** `PositionOutput.cpp:33-34` (`coupling = 0.5*(y_hat + xi_hat.p)`),
documented at `PositionOutput.h:62` and `PositionOutput.cpp:14`; used by
default in `InEKF.cpp::update_position` (`use_cstar = true`).

**What the paper says.** Eq. (B.35): `C⋆ = [ ½(y + p̂)^  0₃ₓ₃  −I₃  0₃ₓ₆ ]`.
This matrix is expressed in the **paper's error coordinates**
`ε = ϑ(φ(X̂⁻¹, ξ))` — a global-frame chart in which the innovation is
`δ = ρ_{X̂⁻¹}(0) − π = p̂ − π` (paper, Sec. 7.1), and `y` pairs with `p̂` as a
*global* position. Expanding the exact innovation in that chart gives

```
δ = (π + ½δ)^ ε_R − ε_p + O(ε³)      with δ = p̂ − π
  ⇒ C⋆_paper = [ ½(π + p̂)^  0  −I  0 ]
```

i.e. the `y` in (B.35) is the **raw global position measurement π** — the
attitude coupling is the *midpoint of measured and estimated global position*.

**What the code does.** It substitutes `y = ŷ = R̂ᵀ(π − p̂)` — the *body-frame
residual* — and adds the *global* `p̂`, mixing two frames, and then uses the
result as `H` in **this module's chart** (the right-multiplicative chart of
`ManifoldEKF`: `X̂ ← X̂·Exp(δ)`, covariance in body coordinates), which is not
the paper's chart in the first place.

**The correct C\* in this implementation's chart** can be derived exactly. With
`X_true = X̂ Exp(ε)` the predicted residual is *exactly*
`ŷ = J_l(ε_θ) ε_p + R̂ᵀn` (left Jacobian of SO(3); no `p̂` anywhere). Expanding
`J_l` to second order and using `ε_p = ŷ + O(ε²)`:

```
ŷ = ε_p − ½ ŷ^ ε_θ + O(ε³)   ⇒   C⋆_body = [ ½ ŷ^   0₃ₓ₃   −I₃   0₃ₓ₆ ]
```

Cross-check: transforming the paper matrix into this chart,
`R̂ᵀ · C⋆_paper · Ad_{X̂}` with `C⋆_paper = [½(π+p̂)^ 0 −I 0]`, gives exactly
`[½(π−p̂)^R̂ … ] → [½ ŷ^ 0 −I 0]` (using `R a^ Rᵀ = (Ra)^`). The two agree;
the implemented `½(ŷ + p̂)^` agrees with neither.

**Consequences.**
- *Translation variance:* `ŷ` depends only on `π − p̂`, but the implemented
  coupling contains the absolute coordinate `p̂`. Shift the world origin by
  1 km and the update behaves completely differently: the `H` attitude column
  acquires a ~500 m lever arm, producing large spurious attitude corrections
  from ordinary position innovations and badly mis-scaled innovation
  covariance. An equivariant filter must be (and the correct C\* is)
  invariant to this.
- The existing tests pass only because they sit exactly on the cancellation
  point: in `testTFGInEKF.cpp::CstarImprovesConvergenceOverC0` the truth is at
  the origin with `R = I`, so `ŷ = −p̂` and the implemented coupling is
  `½(ŷ+p̂) = 0` — by accident close to the correct `½ŷ` for that geometry.
  See F5.
- All examples run with the default `use_cstar = true`; trajectories that
  drift from the origin (constant-velocity: 30 m, coordinated turn, etc.) with
  `--pos-rate > 0` exercise the bug directly.

**Correction** (one line, plus doc fix):

```cpp
// PositionOutput.cpp::jacobian
const Eigen::Vector3d coupling =
    (variant == Variant::Cstar) ? (0.5 * y_hat).eval() : y_hat;
```

and update the comments in `PositionOutput.h:57-72` / `InEKF.cpp:19`: the
implemented `H` lives in the right-multiplicative body chart, where (B.35)
transforms to `½ ŷ^`. Add a translation-invariance regression test (F5).

---

## F3 (high, latent): traits `Expmap`/`Logmap` chart Jacobians are Identity stubs

**Where:** `Group.cpp:202-210`.

```cpp
G T::Expmap(const TangentVector& xi, ChartJacobian H) {
  if (H) *H = Jacobian::Identity();  // TODO: exact right Jacobian ...
```

For a 15-dim group whose vector blocks all carry the SO(3) left Jacobian, the
right ExpmapDerivative is materially different from `I` for non-small `ξ`.
Today no in-module call path requests these Jacobians (`InvariantEKF::predict`
uses only `AdjointMap`; `ManifoldEKF::reset` detects that `Retract` has no
Jacobian overload and skips transport), so the bug is dormant — but:

- the recommended fix for F1 via `LieGroupEKF::predict` *does* consume
  `traits<G>::Expmap(xi, H)` and would silently inherit the wrong Jacobian;
- any future use of the group in factors / `numericalDerivative` checks
  silently gets `I`.

**Correction.** Implement the closed form. For this group structure the
right Jacobian is block-structured exactly like `AdjointMap` (rotation
`Jr_SO3(θ)` on the diagonal, `Q`-type coupling blocks under the `θ` column for
each of the four vector blocks — same `Q(θ, u)` block as SE(3)/SE₂(3),
repeated per block), or at minimum throw on `H` being requested instead of
returning a wrong value.

---

## F4 (medium): process noise is not mapped through the input matrix

**Where:** `InEKF.cpp:38` (`predict(U, Qc*dt)`), `examples/TFGInEKFScenarioExample.h:98-106,190-199`.

The paper's filters use `B_t Q B_tᵀ`-type input-noise mapping. Here a
user-supplied diagonal 15×15 `Qc` is added directly in tangent coordinates:

- Gyro white noise `n_ω` enters the error dynamics not only in the `θ` row
  (identity) but also in both γ rows through `Λ₂` (Eq. 18:
  `b̂ω^ n_ω`, `b̂a^ n_ω`) — those cross terms (and the resulting
  `θ`–`γ` noise correlation) are dropped. Small for small biases, but it is
  exactly the bias subsystem that the TFG structure is supposed to get right.
- `examples/defaultQc()` maps `gyro σ² → θ`, `accel σ² → v` directly, which is
  the leading-order `B = I` approximation; fine, but worth a comment, and the
  bias-random-walk τ noise enters γ through `R̂` rotation (isotropic here, so
  harmless — fragile if anisotropic noise is ever used).

**Correction.** Build `Q_d = B(X̂) Q_imu B(X̂)ᵀ dt` inside `propagate` from a
6+6 IMU noise spec (`n_ω, n_a, τ_ω, τ_a`), with
`B[θ,ω]=I, B[v,a]=I, B[γω,ω]=b̂ω^, B[γa,ω]=b̂a^, B[γ,τ]=−R̂…` per the lift —
or at minimum document that the user-facing `Qc` is already in lifted tangent
coordinates and must account for this.

---

## F5 (medium): the test suite is shaped so F1 and F2 cannot be seen

`tests/` is well built for the group algebra (axioms, adjoint conjugation,
matrix-realisation cross-checks, FD-validated C0 — good), but the filter-level
tests have three blind spots:

1. **No bias estimability test.** Every filter test uses zero true biases.
   A test like *"static vehicle, true `b_ω = (0.01,0,0)`, GNSS at 10 Hz, expect
   `bias_gyro()` → truth and attitude error bounded"* fails immediately under
   F1 (gain rows identically zero) and should be the acceptance test for its fix.
2. **C\* tested only at the cancellation point.**
   `CstarImprovesConvergenceOverC0` places the truth at the origin with
   `R = I`, where the implemented coupling `½(ŷ+p̂)^ = 0`. Add a
   *translation-invariance* test: run the identical scenario shifted by a
   constant offset (e.g. `(500, −300, 100)` m) and assert the error trajectory
   is identical to the unshifted run. Current C\* fails it; C0 and the
   corrected C\* pass.
3. **`StaticConvergence` pins `use_cstar=false` and asserts only 60 % error
   reduction**, with a comment (`testTFGInEKF.cpp:130-137,163-166`)
   rationalizing the residual as expected C0 behaviour. With F2 fixed, run the
   same test with C\* and tighten the bound.

Minor: in `testTFGSymmetry.cpp:75`, `Y = makeXi(), xi = makeXi()` — the
right-action axiom is tested with `Y == xi`; use a third distinct element.

---

## F6 (low): `PositionOutput::innovation` is dead, with an unresolved sign comment

`PositionOutput.h:79-84` / `PositionOutput.cpp:42-45`. `update_position` never
calls it (it passes `prediction` and `z = 0` and lets `ManifoldEKF` form
`prediction − z` internally, which is correct). The helper returns
`−ŷ = z − ŷ` with a comment "Sign convention to be matched against
InEKF::update_position" — anyone who passes this value into
`updateWithVector` as the prediction flips the update sign. Remove it, or
rename/document it as the *signed innovation* `z − ŷ` for logging only.

---

## F7 (low): examples

- **Frozen bias columns (symptom of F1):** the CSV schema
  (`est_bg*, est_ba*, P_bg_*, P_ba_*`) implies bias estimation works; with
  GNSS enabled the estimates stay at the initial zero and `P_bg/P_ba` grow
  monotonically. After fixing F1 these become meaningful; until then the
  header comment should say biases are propagate-only.
- **Gravity:** `tfg::gravity()` hard-codes `(0,0,−9.81)` while the examples
  independently hard-code `MakeSharedU(9.81)` (`TFGInEKFScenarioExample.h:162`).
  Consistent today; make the magnitude a single shared constant (or a
  `propagate` parameter) so they cannot drift apart, and to support 9.80665 /
  non-Earth use.
- **`RunOptions.log_decim`:** the CLI parser clamps to ≥ 1, but direct struct
  use with `log_decim = 0` divides by zero at
  `TFGInEKFScenarioExample.h:226`. Clamp in `runScenario`.
- **`parseVector3`** silently truncates extra components (`"1,2,3,4"` → ok) and
  `parseRunOptions` silently ignores unknown flags; both deserve a warning or
  error.
- `runScenario` performs the position update *before* the propagate within the
  step and times the first update at `t = 1/rate` — correct, just noting it
  was checked.

---

## Verified correct (no action)

For completeness, the following were independently re-derived and match the
paper and/or numerically validated conventions:

- **Group structure** (`Group.cpp`): product `(C_X C_Y, γ_X + A_X ∗ γ_Y)`,
  inverse `(C⁻¹, −Aᵀ ∗ γ)` — Sec. 5.3 exactly; nav part composes as the 5×5
  SE₂(3) matrix product (cross-checked by `ComposeMatchesMatrixRealisation`).
- **Exp/Log** are exact for `SO(3) ⋉ ℝ¹²`: `Exp(θ,u) = (Exp θ, J_l(θ)u)` per
  block — correct, including the small-angle guards.
- **AdjointMap** block structure and sign — validated by the conjugation
  identity test; `Compose`/`Between`/`Inverse` trait Jacobians match the
  standard gtsam conventions.
- **State ↔ group correspondence** `b = A⁻¹ ∗ (−γ)` (Eq. B.31) and the action
  `φ(X, ξ) = (TC, Aᵀ ∗ (b − γ))` (Eq. 16, Lemma 5) — `phi(X, ξ) = ξ·X` is the
  correct realisation when states are stored as group elements.
- **Lift** (`Symmetry.cpp::lift`): the closed form
  `(ω−b_ω, (a−b_a)+Rᵀg, Rᵀv, b_ω×(ω−b_ω)−τ_ω, b_a×(ω−b_ω)−τ_a)` is exactly
  the vee form of Eq. (17)–(18); one predict step reproduces the world-frame
  dynamics Eq. (3) to O(dt²) (and the tests check this).
- **Position output** (Eq. 34) and output action (Eq. 35), and the
  equivariance identity `h(φ(X,ξ)) = ρ_X(h(ξ))` — correct.
- **C0** is the true Jacobian of `h` in the chart `ManifoldEKF` actually uses
  (finite-difference test agrees), and the measurement noise rotation
  `R_meas = R̂ᵀ R_pos R̂` matches the exact residual noise `R̂ᵀ n`.
- **No reset step** (`performReset=false`) matches the paper: "the
  Imperfect-IEKFs and the TFG-IEKF have been implemented without reset step"
  (Sec. 7.2).
- Build wiring (`gtsam_unstable/CMakeLists.txt` subdir list, test/example
  globs, header install) is in place.

## Suggested fix order

1. F2 (one-line, plus docs) with the translation-invariance test from F5.
2. F3 (exact right Jacobian) — prerequisite for the clean version of 3.
3. F1 (full state-transition matrix; Route A or B) with the bias-convergence
   test from F5.
4. F4–F7 as cleanups.
