# TG-EqF Code Review

Review of `gtsam_unstable/tg_eqf` (code, tests, examples, scripts) against the
design/implementation docs and the reference paper:

> Fornasier, Ge, van Goor, Mahony, Weiss, *Equivariant Symmetries for Inertial
> Navigation Systems*, arXiv:2309.03765 ("the paper"). All Lemma/Theorem/
> Equation numbers below refer to it. The TG-EqF is the filter of Sec. 5.4
> (symmetry), Sec. 7 (position output), and Appendix B.4 (filter matrices).

Reviewed at commit `72676b2` ("Changed R,p,v convention to R,v,p and added a
virtual bias velocity output") on branch `claude/focused-faraday-l2hxl2`. The
rest of the GTSAM tree is upstream except for the root `CMakeLists.txt`
(`-Wno-error`, commit `82e6359`).

**Verification performed:** all 18×18/9×9 algebra was checked symbolically
against the paper; the full unit-test suite (`check.tg_eqf_unstable`) was
built and run; the chart-consistency finding (F1) was additionally confirmed
with an independent numerical program (Appendix A of this report).

---

## Summary verdict

The *symmetry-algebra layer is faithful to the paper and well tested*: the
group product/inverse (Sec. 5.4), the state action φ (Lemma 7, Eq. 19), the
input action ψ including the Ω-correction (Lemma 8, Eq. 20 and the Ω
definition on p. 4), the lift Λ₁/Λ₂ (Theorem 9, Eq. 21–22), and the output
actions for body velocity and reformulated position (Lemma 15, Eq. 34–35) all
match the paper exactly, and each is guarded by axiom and finite-difference
tests.

There are, however, **two significant inconsistencies** and several moderate
ones:

| # | Severity | Finding |
|---|----------|---------|
| F1 | **Major (correctness)** | All measurement Jacobians are linearized in the chart at the *current estimate*, while the filter's covariance, error dynamics, and innovation lift live in the chart at the *fixed origin* `xi_ref`. The missing factor is exactly the `Diffeomorphism` Jacobian that is implemented but never used in updates. Updates are only correct when the estimate is near the origin. |
| F2 | **Major (bug)** | The `R,p,v → R,v,p` convention change (commit `72676b2`) was not propagated to the example harness: process-noise `Qc` blocks are written to the wrong tangent slots (accel noise → *position* block, accel-bias RW → *virtual-bias* block), and the CSV header labels `eps_*`/`P_*` in the legacy order, corrupting the Monte-Carlo NEES/ANEES analysis. |
| F3 | Moderate | The position update's `C*`/innovation pairing deviates from the paper's (App. B, Eq. B.19 + Sec. 2.2): the paper pairs an *origin-frame* innovation with `C*` built from the *origin output* `ẙ`; the code pairs a *body-frame* innovation with `C*` built from the body residual — giving a factor-2-wrong attitude coupling at convergence even at the origin. |
| F4 | Moderate | No covariance **reset step**. The paper's experimental TG-EqF includes the reset (Sec. 2.2, "Reset"; Sec. 7.2), and its consistency results (ANEES ≈ 1) are reported *with* it. |
| F5 | Moderate | Process noise is injected directly as a diagonal `Qc` in state coordinates with no input matrix `B_t` mapping (no `Dphi0·DΛ/du` transport); acceptable as an approximation, but undocumented and compounded by F2. |
| F6 | Minor | The virtual input is fed `ν = b̂_v` where the paper prescribes `ν = 0` for implementation (App. B.4.3, final note). Defensible, but makes the input state-dependent and should be documented as a deviation. |
| F7 | Minor | Several derivation comments still describe the **old** `(R,p,v)` slot order while the code below them is correct (`Group.cpp`, `Group.h`, `Symmetry.cpp`). |
| F8 | Minor | Upstream `EquivariantFilter::covariance()` transports `P` as `Jᵀ P J` where the push-forward of a covariance is `J P Jᵀ`; tg_eqf tests and any consumer of `covariance()` inherit this. |
| F9 | Info | Smaller items: header-file anonymous namespace, duplicated adjoint helpers, global `-Wno-error`, dangling "proposal" references, missing guard for `pos_noise_sigma == 0`. |

F1+F3 explain why everything *appears* to work: every unit test exercises the
filter at (or one small update away from) the reference state, where the two
charts coincide to first order. The scenario examples are IMU-only by default,
so the predict path (which **is** chart-consistent) dominates. The errors
surface exactly in the regime the examples are being extended toward: aided
updates on trajectories whose attitude/position depart from the origin.

---

## F1 — Measurement Jacobians are in the wrong chart (major)

### The filter's chart

`gtsam::EquivariantFilter` (upstream, `gtsam/navigation/EquivariantFilter.h`)
keeps:

* the mean as a group element `g_`, with the state reconstructed as
  `X_ = act_on_ref_(g_) = φ(g_, ξ_ref)`;
* the covariance `P_` of the error `ε = Local(ξ_ref, φ(g_⁻¹, ξ_true))` —
  coordinates in the **fixed chart at `ξ_ref`** (this is what the predict path
  assumes: `A = Dphi0_ · D_lift` is evaluated at `ξ_ref`, matching the paper's
  `A_t⁰` in Sec. 2.2 / Eq. B.18);
* the update (lines 291–296):

  ```cpp
  TangentM delta_xi = -K * innovation;
  TangentG delta_x  = InnovationLift_ * delta_xi;   // pinv(Dphi0)
  g_ = traits<G>::Compose(traits<G>::Expmap(delta_x), g_);   // LEFT multiply
  ```

Because φ is a **right** action, left-composition acts at the origin:

```
φ(Exp(δx)·g, ξ_ref) = φ(g, φ(Exp(δx), ξ_ref)) ≈ φ(g, Retract(ξ_ref, Dphi0·δx))
```

so the correction `delta_xi` displaces the state **in the origin chart** and
is then pushed through the diffeomorphism `φ_g`. Consequently the `H` consumed
by `updateWithVector` must be

```
H_origin = d/dε  h( φ(g, Retract(ξ_ref, ε)) ) |_{ε=0}
         = Dh|_{ξ̂}  ·  Dφ_g|_{ξ_ref}            (chain rule)
```

This is also exactly the paper's prescription (Sec. 2.2):
`C⁰ = D δ(y) · D h(e) · D ϑ⁻¹` is differentiated **on the error `e` at the
origin `ξ̊`**, not at the estimate.

### What the code supplies

All four Jacobians differentiate `predict` through `Retract` **at the current
estimate** and omit the `Dφ_g` transport:

* `BodyVelocityOutput.cpp:11-22` — `H = [skew(R̂ᵀv̂) | R̂ᵀ | 0 …]`;
* `PositionOutput.cpp:22-33` — `C* = [½(y+p̂)^∧ | 0 | −I | 0…]` with
  `y = R̂ᵀ(π−p̂)`;
* `PositionOutput.cpp:12-20` — `C0 = [0 | 0 | −I | 0…]`;
* `VirtualBiasOutput.cpp:11-25` — `C0 = [0 … I₃]`.

The finite-difference guards in
`testTGBodyVelocityOutput.cpp:48-59` / `testTGPositionOutput.cpp:48-59` /
`testTGVirtualBiasOutput.cpp:60-71` all perturb via
`traits<TGState>::Retract(xi, e)` around the *same* point the analytic
Jacobian is evaluated at — so they confirm the matrices are correct
derivatives **in the wrong chart**. They never compose with `φ_g`.

### Concrete error (DVL example)

Take `ξ_ref = identity`, estimate `ĝ = ((R_X, v_X, p_X), 0)`, so
`ξ̂ = (R_X, v_X, p_X, 0)`. The true measurement as a function of the
origin-chart error `ε` is

```
h(φ(ĝ, Retract(ξ_ref, ε))) = (e_R R_X)ᵀ(e_R v_X + ε_v) = R_Xᵀ v_X + R_Xᵀ ε_v + O(ε²)
```

so `H_origin = [0 | R̂ᵀ | 0 …]` — **zero rotation block**. The implemented
`H = [skew(R̂ᵀv̂) | R̂ᵀ | 0 …]` has a spurious rotation column: every DVL
innovation leaks into an attitude correction proportional to the vehicle
speed. One can verify directly that
`H_impl · Dφ_ĝ(ξ_ref) = [skew(R̂ᵀv̂)·R_Xᵀ − R̂ᵀ·skew(v_X) | R̂ᵀ | 0] = [0 | R̂ᵀ | 0]`
using `skew(Rᵀu)Rᵀ = Rᵀ skew(u)` — i.e. the transport exactly cancels the
spurious block. (Numerically confirmed; see Appendix A.)

For the position output the same analysis gives
`H_origin = [R̂ᵀπ^∧ | 0 | −R̂ᵀ | 0…]`: the implemented `−I` position block is
mis-rotated by the full current attitude `R̂` relative to what the filter
consumes, so position innovations are applied in the wrong direction whenever
the vehicle has yawed/rolled away from the origin attitude (e.g. everywhere on
the Circle scenario).

For the virtual-bias output the correct origin-chart Jacobian is the `b_ν`-row
of the bias transport `Ad⁻¹` (cf. `Ad9inv` in `Symmetry.cpp:13-25`):

```
H_origin = [0₃ₓ₉ | −R̂ᵀ p̂^∧ | 0₃ₓ₃ | R̂ᵀ]      (columns b_ω, b_a, b_ν)
```

— this is precisely why the paper's Eq. (B.20) matrix is `R̂`- and
`p̂`-dependent. The comment block in `VirtualBiasOutput.cpp:13-21` (and
`docs/implementation.md` §6.3/§9) asserting that B.20 differs only by a
benign "`Ad_T`-transported chart" relabeling is a misdiagnosis: the paper's
normal-coordinates chart and the code's `Retract` chart agree at the origin up
to the constant `Dphi0 = diag(I₉, −I₉)`; the `R̂, p̂` dependence in B.20 is
genuine transport that the code drops.

### Why the tests don't catch it

Every aided-update test in `testTGEqF.cpp` constructs the filter **at** the
reference (`TGEqF filter(TGState::identity(), …)`) and immediately updates.
At `ĝ = id` the transport `Dφ_ĝ(ξ_ref) = I` and both charts coincide. No test
propagates to a rotated/translated state *and then* checks an update against
ground truth.

### How to fix

Minimal, structure-preserving fix — compose with the already-implemented
`Diffeomorphism` Jacobian (chain rule) in each `TGEqF::update_*`:

```cpp
// EqF.cpp — e.g. update_dvl
Eigen::Matrix<double, 18, 18> Dphi_g;
TGSymmetry::Diffeomorphism(groupEstimate())(referenceState(), &Dphi_g);
const Eigen::Matrix<double, 3, 18> H =
    DVLMeasurement::jacobian(xi_hat) * Dphi_g;          // H_origin
Base::updateWithVector(prediction, H, z_dvl, R_dvl);
```

The same one-liner applies to `update_position` (both `C0` and `C*`) and
`update_virtual_bias` (whose product `[0…I₃]·Dφ_g` reproduces the
`[−R̂ᵀp̂^∧ | 0 | R̂ᵀ]` structure of Eq. B.20 automatically). Note the
measurement noise should then also be expressed consistently (for the
body-frame residual with isotropic `R_pos` this is a no-op; for non-isotropic
GNSS noise use `R̂ᵀ R_pos R̂`).

Add a regression test that (i) propagates or constructs the filter at a state
with `R̂` ≈ 90° away from `ξ_ref` and non-zero `v̂, p̂`, (ii) applies a
position/DVL update with a known offset, and (iii) asserts the state moves in
the correct *global* direction (cf. Appendix A, where the current code moves
the position estimate **+0.60 m in x** for a **+0.5 m in y** innovation).

A more literal alternative is to implement the paper's own pairing (Sec. 2.2
update with innovation `δ(ρ(X̂⁻¹, y))` and Eq. B.19's `C*`) — see F3.

---

## F2 — Stale tangent-slot offsets after the convention change (major)

Commit `72676b2` changed the state/tangent order from the legacy

```
[R(0:3), p(3:6), v(6:9), b_ω(9:12), b_v(12:15), b_a(15:18)]   (old)
```

(verifiable via `git show 72676b2^:gtsam_unstable/tg_eqf/State.cpp`, lines
40-51) to the paper-conformant

```
[R(0:3), v(3:6), p(6:9), b_w(9:12), b_a(12:15), b_v(15:18)]   (new, State.cpp:42-52)
```

but the example harness and test fixtures still use the **old offsets**:

### F2a — `Qc` written to the wrong blocks (functional bug)

`examples/TGEqFScenarioExample.h:256-269`:

```cpp
if (opts.accel_noise_sigma > 0.0) {
  Qc.block<3, 3>(6, 6) = opts.accel_noise_sigma * opts.accel_noise_sigma * I3;   // ← position block!
}
...
if (opts.accel_bias_rw > 0.0) {
  Qc.block<3, 3>(15, 15) = opts.accel_bias_rw * opts.accel_bias_rw * I3;          // ← b_v block!
}
```

In the new order, block `(6,6)` is **position** and `(15,15)` is the
**virtual bias**. Accelerometer white noise must drive the **velocity** error
(block `(3,3)`; the TG error dynamics are `ε̇_v ≃ g^∧ε_R + ε_ba` with input
noise entering through `Λ₁`'s accel slot — Table 2 / Eq. B.18), and the accel
bias random walk must drive `b_a` (block `(12,12)`). As written, simulated
accel noise/bias-RW is fed to states it does not excite, while the states it
does excite keep the (zero / default) PSD — the filter's covariance is
inconsistent with the simulation by construction, which defeats the stated
purpose ("so the covariance is physically meaningful (a prerequisite for any
consistency / NEES analysis)", lines 251-255, and the same claim in
`examples/README.md`).

`defaultQc()` in both `TGEqFScenarioExample.h:108-115` and
`tests/testTGEqF.cpp:27-34` carries the same stale pattern
(`(6,6) = 1e-3` "accel", `(15,15) = 1e-5` "accel-bias", nothing in `(3,3)` /
`(12,12)`).

### F2b — CSV header / doc-comment labels in the legacy order

`writeCsvRow` (`TGEqFScenarioExample.h:190-196`) dumps `eps` and the diagonal
blocks of `P` in **raw tangent order** `[att, vel, pos, b_w, b_a, b_v]`, but
`csvHeader()` (lines 118-136) labels those columns

```
eps_att, eps_pos, eps_vel, eps_bg, eps_bv, eps_ba
P_att,  P_pos,  P_vel,  P_bg,  P_bv,  P_ba
```

i.e. the legacy order. So the column called `eps_pos` actually contains the
**velocity** error, `P_pos` the velocity covariance, `eps_ba` ↔ `eps_bv`
swapped, etc. The doc comment above `writeCsvRow` ("in TGState tangent order
[att, pos, vel, bg, bv, ba]", lines 152-160) is likewise stale.

`scripts/monte_carlo.py` selects these columns **by name**
(`eps_and_cov`, lines 99-113; `GROUPS = ["att","pos","vel","bg","bv","ba"]`):
its per-group output therefore pairs the *position* RMSE (computed from
correctly-named `est_px/gt_px`) with what is actually the *velocity*
ANEES, and `ba` ↔ `bv` likewise. Any consistency conclusions drawn from
those plots/summaries for pos/vel/ba/bv are mislabeled.

(The estimated-bias columns `est_bg, est_bv, est_ba` are *not* affected —
`writeCsvRow` writes them explicitly in that order, matching the header.)

### Fix

* `Qc`: accel noise → `block<3,3>(3,3)`, accel-bias RW → `block<3,3>(12,12)`;
  update `defaultQc()` in both files and the README sentence.
* Header: relabel to `[att, vel, pos, bg, ba, bv]` (or reorder the writes);
  update the `writeCsvRow` doc comment and `scripts/monte_carlo.py`'s
  `GROUPS`/labels to match.
* Cheap guard: a unit test that round-trips one CSV row and asserts
  `eps_pos` equals the position component of a known `eps`.

---

## F3 — `C*`/innovation pairing differs from the paper (moderate)

The paper's update (Sec. 2.2) uses the **origin-frame** innovation
`δ(ρ(X̂⁻¹, y))`; for the position output (Sec. 7.1) the received equivariant
measurement is `y = 0`, `ρ_X(y) = Aᵀ(y − b)` (Eq. 35, with `b` = position
column of `C_X`), so

```
innovation_paper = ρ_{X̂⁻¹}(0) − ẙ = p̂ − π          (global frame)
```

paired with (Eq. B.19)

```
C* = [ ½(ẙ + p̂)^∧  0₃ₓ₃  −I₃  0₃ₓ₉ ],   ẙ = h(ξ̊) = R̊ᵀ(π − p̊)  ( = π at the identity origin)
```

The `C*` construction (Sec. 2.2) averages `D_E ρ_E(·)` at the **origin
output** `ẙ` and at the **back-transported measurement** `ρ_{X̂⁻¹}(0) = p̂`;
both arguments are global-frame points. First-order check: the innovation
satisfies `p̂ − π ≈ −ε_p + p̂^∧ε_R`, and as `π → p̂` the paper's rotation block
`½(π + p̂)^∧ → p̂^∧` — consistent.

The implementation (`PositionOutput.cpp:22-33`, `EqF.cpp:44-56`) instead pairs
the **body-frame** innovation `R̂ᵀ(π − p̂)` with a `C*` whose first argument is
the **body residual** `y = R̂ᵀ(π − p̂)`:

```cpp
const Eigen::Vector3d y = predict(xi_hat, pi);      // R̂ᵀ(π − p̂) → 0 at convergence
Cstar.block<3,3>(0,0) = 0.5 * gtsam::skewSymmetric(y + xi_hat.p);
```

Two consequences, independent of F1:

1. At convergence (`y → 0`) the implemented rotation block tends to `½p̂^∧`
   where the consistent first-order value is `p̂^∧` — the attitude coupling of
   the position update is **half** its correct value even at `R̂ = I`.
2. Away from the origin the body-frame innovation needs the `R̂ᵀ` transport on
   every block (this is F1's manifestation for this output).

The `docs/design.md` §4 note ("`jacobian_Cstar` … is only checked against a
restated formula — no numerical ground-truth guard") correctly flags the lack
of a guard; the restated formula itself, however, is a misreading of B.19's
`y` (origin output `ẙ`, not the body residual).

**Fix options** (pick one, consistently):

* (a) Keep the body-frame innovation and use the exact first-order
  origin-chart Jacobian `H = [(y + R̂ᵀp̂)^∧ R̂ᵀ | 0 | −R̂ᵀ | 0…]`
  (equivalently `Dh|ξ̂ · Dφ_ĝ` per F1) — loses the third-order property but is
  a correct EKF.
* (b) Implement the paper's pairing literally: innovation `p̂ − π` (global),
  `C* = [½(ẙ + p̂)^∧ | 0 | −I | 0…]`, noise `R_pos` already in the right
  frame. This preserves the `O(‖ε‖³)` output linearization the docs claim.
  (Sign bookkeeping: gtsam's `update` computes `δξ = −K·(prediction − z)`, so
  feed `prediction = p̂ − π`, `z = 0`.)

---

## F4 — Missing reset step (moderate)

Paper Sec. 2.2 lists the EqF as Predict / Update / **Reset**
(`Σ ← exp(Γ dφ_ξ̊ Δ) Σ exp(Γ dφ_ξ̊ Δ)ᵀ`), and Sec. 7.2 states explicitly that
"each of the aforementioned EqFs have been implemented **including the reset
step**" for the published comparison (only the IEKFs/TFG-IEKF run without).
The upstream `gtsam::EquivariantFilter` has no reset hook and `TGEqF` adds
none. The header claim in `EqF.h:91-93` / `PositionOutput.h:11-12`
("Directly reproduces the TG-EqF filter from Fornasier …") should therefore
be qualified, and the consistency (ANEES) targets of the paper should not be
expected to reproduce exactly. Either implement the reset in the wrapper
(post-update covariance conjugation) or document the deviation in the docs'
"known limitations".

---

## F5 — Process noise: no input matrix `B_t` (moderate)

`TGEqF::propagate` forwards a user-supplied diagonal `Qc` straight into
`P ← Φ P Φᵀ + Qc·dt` (origin chart). In the paper's EqF the input noise enters
through the lift differential (`B_t = Dφ_ξ̊ · D_u Λ`), which for the TG lift
(Eq. 21–22):

* maps gyro/accel/virtual-input white noise into the navigation rows through
  `Λ₁` (identity-ish at the origin — benign),
* **also** maps the same IMU noise into the **bias-error rows** through
  `Λ₂ = ad_b[Λ₁] − τ` with gain `ad_b` (small when `b̂` is small),
* and transports `τ`-noise by `Ad`.

Treating `Qc` as block-diagonal in state coordinates is a common and usually
acceptable approximation for small biases, but it is currently neither stated
in `docs/design.md` ("known approximations") nor in the header docs, and F2a
makes the realized mapping wrong outright. Recommend documenting it and, if
NEES-grade consistency is the goal, adding the `B_t Q_u B_tᵀ` mapping (the
numerical `D_lift` machinery in `Lift.cpp:84-107` can be reused for
`D_u Λ` by perturbing `u` instead of `ξ`).

---

## F6 — Virtual input `ν = b̂_v` vs. the paper's `ν = 0` (minor)

`TGEqF::propagate` (`EqF.cpp:18`) sets `u.v = xi_hat.b_v`, which makes the
lift's position-rate slot exactly `Rᵀv` (guarded by
`testTGLift.cpp:130-138`). The paper instead says, "Note that for a practical
implementation of the presented EqF the virtual inputs ν is set to zero"
(App. B.4.3), relying on the `b_ν = 0` anchor (Eq. B.20) to keep the
discrepancy `R(ν − b_ν)` in `ṗ` negligible. The implementation's choice is
arguably cleaner (exact `ṗ = v` at the operating point even when `b̂_v ≠ 0`)
but (i) it makes the "input" a function of the state estimate, stepping
outside the exogenous-input setting in which Theorem 9's equivariance and the
error-dynamics derivation are stated, and (ii) it changes `u°` and hence `A`.
Since the virtual-bias anchor is **off by default**
(`anchor_virtual_bias_ = false`, `EqF.h:131-132`), `b̂_v` can drift via
update cross-covariance and then feeds back into the dynamics through `u.v`.
Document the deviation; consider enabling the anchor by default (matching the
paper, which always imposes B.20) or zeroing `ν` when the anchor is off.

---

## F7 — Stale `(R,p,v)`-order comments over correct code (minor)

The convention change fixed the code but left several derivation comments
describing the old slot order — a trap for the next editor (the docs
themselves warn that "mixing in the legacy `[R,p,v]` order is the classic bug
source here"):

* `Group.cpp:75` — comment says
  `Ad_{A_X}[xi] = (R xi_w, R xi_v + p×(R xi_w), R xi_a + v×(R xi_w))`;
  the code (and `Group.h:165-170`) correctly computes
  `(R xi_w, R xi_a + v×(R xi_w), R xi_v + p×(R xi_w))`.
* `Group.cpp:85` — same swap for `Ad_{A^{-1}}`
  (code is right: `(Rᵀxi_w, Rᵀ(xi_a − v×xi_w), Rᵀ(xi_v − p×xi_w))`).
* `Group.h:188-191` — `ad_a(xi)` comment swaps the `a`/`v` rows; the matrix
  beneath it (and `ad_se23`) is correct.
* `Symmetry.cpp:7-8` — `Ad9inv` comment pairs `eta` with `p` and `alpha` with
  `v`; the matrix is correct (`a`-row couples `v`, `v`-row couples `p`).

Fix: update the four comments to the `(w, a, v) ↔ (rot, vel-col, pos-col)`
convention.

---

## F8 — Upstream `covariance()` transports `P` the wrong way (minor, upstream)

`EquivariantFilter::covariance()` returns `Jᵀ P J` with
`J = Dφ_g(ξ_ref)` (the push-forward from origin chart to current-state
chart). The covariance of `ε_state = J ε_origin` is `J P Jᵀ`, so unless `J` is
orthogonal (it is not: see the `−R·skew(v_X)` and `Ad⁻¹` blocks in
`Symmetry.cpp:135-160`) the returned matrix is the transport of `P` under
`Jᵀ = (J⁻¹ only-if-orthogonal)`. tg_eqf consumes `covariance()` in the
covariance-shrink tests (`testTGEqF.cpp:174-186, 221-233, 282-291`) — those
remain valid as monotonicity checks — but any quantitative use of
`covariance()` blocks (e.g. gating) would be wrong. The harness correctly logs
`errorCovariance()` (origin chart) instead. Worth reporting upstream; within
tg_eqf, prefer `errorCovariance()` everywhere and note the caveat in the docs.

---

## F9 — Smaller items

1. **Anonymous namespace in a header** (`Group.h:153-209`): `toSe23`,
   `Ad_SE23`, `ad_se23` get internal linkage *per translation unit*; every
   includer carries (possibly unused) copies — this is what forced the global
   `-Wno-error` workaround (root `CMakeLists.txt`, commit `82e6359`,
   "silence warning as error"), which now masks **all** warnings-as-errors for
   the whole GTSAM build. Move the helpers into a named detail namespace in a
   `.cpp`/`-inl.h`, drop the global `-Wno-error`.
2. **Duplicated adjoint code**: `Symmetry.cpp:13-44` re-implements `Ad9inv` /
   `ad9` already available as `Ad_SE23` (inverse) / `ad_se23` in `Group.h` and
   as `TGGroupElement::Ad_A_inv`. Two hand-maintained copies of the same
   9×9 algebra is exactly the drift channel the convention change already bit
   once (F7).
3. **Dangling references**: comments cite "proposal Eq. (7b)/(9)/(13)/(14),
   (15)" and `docs/EqF_design_for_DVL_Depth_aided_INS.pdf`, none of which are
   in the repo (the `docs/` directory was removed in commit `1c2b9d4`). Either
   restore the docs (without the copyrighted PDF) or renumber citations
   against arXiv:2309.03765 (e.g. action = Eq. 19, ψ = Eq. 20, lift =
   Eq. 21–22, ρ = Eq. 35).
4. **`EqF.h:34-36` ctor doc**: "must have gravity direction fixed if using
   depth" — depth aiding is not implemented (also listed as a non-goal in
   `docs/design.md` §5); drop or qualify.
5. **Harness edge case**: `pos_rate > 0` with `pos_noise_sigma = 0` builds
   `R_pos = 0` (`TGEqFScenarioExample.h:292-298`) — singular-`S` risk in the
   gain; clamp or assert.
6. **`testTGEqF.cpp:304`**: `filter.propagate(omega, g_vec, g_vec, …)` feeds
   specific force `= g` (free-fall) where a stationary body should read `−g`;
   harmless for what the test asserts, but confusing as an "end-to-end"
   exemplar.
7. **`D_lift` recomputation**: `computeErrorDynamicsMatrix` re-runs the
   19-evaluation finite-difference at the *same* `ξ_ref` every step; only
   `u°` changes. Cheap (≈ 19 5×5 inverses) but trivially cacheable; or supply
   the closed-form `A` of Eq. B.18, which is constant in `ẘ` and would also
   remove the FD step-size sensitivity noted in `docs/design.md` §3.4.
8. **NEES doc-comment** (`TGEqFScenarioExample.h:158-160`): "per group g the
   proper NEES is eps_gᵀ P_g⁻¹ eps_g ~ chi^2_3" — block-marginal NEES is fine,
   but the phrase "per group g" (g = CSV group key) collides with the group
   element `g_`; reword. The 18-DoF full-state NEES would use the full `P`.

---

## What checks out (verified against the paper)

For balance — these were each verified symbolically and are correct,
matching the paper one-to-one:

| Item | Paper | Code |
|------|-------|------|
| State `ξ = (T, b) ∈ SE₂(3) × R⁹`, `b = (b_ω, b_a, b_ν)` | Sec. 5.4 | `State.h` (dim 18) |
| Group product `(C_X C_Y, γ_X + Ad_{C_X}γ_Y)`, inverse `(C⁻¹, −Ad_{C⁻¹}γ)` | Sec. 5.4 | `Group.cpp:51-73` |
| `Ad_{SE₂(3)}` and inverse, `ad_{se₂(3)}` matrices | standard | `Group.h:165-207` (matrices correct; comments F7) |
| Tangent-group `exp` fiber = `J_l(λ₁)λ₂`, code truncates `J_l ≈ I` | B.4.2 | `Group.cpp:92-111`; truncation documented in `docs/design.md` §3.2 |
| Full 18×18 `Ad_{(A,a)} = [[Ad_A, 0],[ad_a Ad_A, Ad_A]]` | semidirect structure | `Group.cpp:178-193` |
| Action `φ(X, ξ) = (TC, Ad_{C⁻¹}(b − γ))` (right action, verified axioms) | Lemma 7, Eq. 19 | `Symmetry.cpp:62-80` + tests |
| Input action `ψ = (Ad_{C⁻¹}(w − γ) + Ω(C⁻¹), Ad_{C⁻¹}τ)`, `Ω(X) = (0,0,a)` | Lemma 8, Eq. 20; p.4 | `Lift.cpp:118-139` (`Ω(C⁻¹)` slot = `−Rᵀv` in the position-rate slot ✓) |
| Lift `Λ₁ = (W−B+N) + T⁻¹(G−N)T`, `Λ₂ = ad_b[Λ₁] − τ`; `N(3,4)=1`; gravity in the velocity slot | Thm 9, Eq. 21-22; Eq. 12 | `Lift.cpp:39-82`; equivariance `Λ(φ(X,ξ), ψ(X,u)) = Ad_{X⁻¹}Λ(ξ,u)` numerically tested |
| Mean propagation `g ← g·Exp(Λ(ξ̂,u)dt)` (left-trivialized lifted dynamics) | Sec. 2.2 Predict | `EquivariantFilter::predictWithJacobian` |
| `A = Dphi0·D_lift` at `ξ_ref` with `u° = ψ(ĝ⁻¹, u)` (origin-chart error dynamics — the EqF property) | Sec. 2.2, Eq. B.18 | `computeErrorDynamicsMatrix` + `Orbit`/`Lift` functors |
| DVL output `h = Rᵀv`, action `ψ_X(y) = R_Xᵀy + R_Xᵀv_X`, equivariance | (natively equivariant) | `BodyVelocityOutput` + tests |
| Position reformulation `h = Rᵀ(π − p)`, `ρ_X(y) = R_Xᵀ(y − p_X)`, target 0 | Lemma 15, Eq. 34-35 | `PositionOutput` + tests |
| Orbit / Diffeomorphism Jacobians (18×18) | — | `Symmetry.cpp:98-163`, FD-verified |

The test suite design is also commendable: group axioms, action axioms,
adjoint homomorphism, lift equivariance for generic and `Exp`-generated
elements, and FD guards on every analytic Jacobian *within its chart*.

---

## Appendix A — numerical evidence for F1/F3

Standalone program [`CODE_REVIEW_evidence.cpp`](CODE_REVIEW_evidence.cpp)
(not wired into the build; compile with
`g++ -std=c++17 -I<gtsam-root> -I<build> -I<eigen> CODE_REVIEW_evidence.cpp
-lgtsam_unstable -lgtsam`), evaluated at
`ξ_ref = id`, `ĝ = ((Rz(90°), v=(2,0,0), p=(5,−3,1)), 0)`:

```
DVL rotation block, origin chart (consumed by filter):
  ≈ 0 (max entry 2e-10, finite-difference noise)
DVL rotation block, implemented (chart at xi_hat):
   0  0 -2
   0  0  0
   2  0  0        = skew(R̂ᵀv̂), v̂ = (2,0,0)

max |origin-chart H − H_impl · Dphi_g(xi_ref)|  (DVL):  2.2e-10
  → composing with the Diffeomorphism Jacobian exactly closes the gap (the F1 fix).

Position block (cols 6-8), origin chart:        Position block, implemented C*:
   0 -1  0                                        -1  0  0
   1  0  0     ( = −R̂ᵀ, R̂ = Rz(90°) )             0 -1  0
   0  0 -1                                         0  0 -1

End-to-end (EquivariantFilter::updateWithVector, R = 1e-4·I):
  global position innovation:  +0.5 m in y
  actual position change:      (+0.603, +0.175, −0.058) m   ← mostly +x: rotated
                                                              by the unaccounted R̂
  (a consistent update would move ≈ +0.5 m in y)
```

## Appendix B — build & test status

* Configured with `-DGTSAM_BUILD_PYTHON=OFF -DGTSAM_WITH_TBB=OFF
  -DGTSAM_BUILD_WITH_MARCH_NATIVE=OFF`; module builds clean (Release).
* `check.tg_eqf_unstable`: **100% tests passed, 0 tests failed out of 9** —
  consistent with the analysis above that the suite only exercises the
  near-origin regime where F1/F3 vanish to first order.
