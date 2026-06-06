# TFG-IEKF — Design

Two-Frames Group Invariant EKF for biased inertial navigation with direct
global position measurements.

All equation / section numbers refer to Fornasier, Ge, van Goor, Mahony, Weiss,
*"Equivariant Symmetries for Inertial Navigation Systems"* (arXiv:2309.03765v3),
in `docs/`. The filter is the **TFG-IEKF** of Appendix B.7 (symmetry group
`G_TF` of Sec. 5.3, originally Barrau & Bonnabel [4]).

## 1. Problem

Biased INS kinematics (Eq. 3), navigation + bias states:

```
R_dot = R (omega - b_omega)^         (3a)   R    in SO(3)   body orientation
v_dot = R (accel - b_accel) + g      (3b)   v,p  in R^3     nav-frame vel/pos
p_dot = v                            (3c)   b_*  in R^3     IMU biases
b_omega_dot = tau_omega              (3d)
b_accel_dot = tau_accel              (3e)
```

IMU input `u = (omega, accel, tau_omega, tau_accel)` (Eq. 5); `tau` are the
bias random-walk drivers (zero for constant bias). Output: a global position
fix `pi in R^3` (GNSS), modelled in Sec. 7.

State manifold `M = SE_2(3) x R^6`, `xi = (T, b)`, `T = (R, v, p) in SE_2(3)`,
`b = (b_omega, b_accel) in R^6`. dim(M) = 15.

## 2. Symmetry group `G_TF` (Sec. 5.3)

`G_TF = SO(3) |x (R^6 (+) R^6)`, dim 15. An element `X = (C, gamma)` with the
navigation part `C = (A, (a, b)) in SE_2(3)` and the bias part
`gamma = (g_omega, g_accel) in R^6`, living in the SO(3)-rotated second frame.

```
Group product (Sec. 5.3, p.7):  X*Y     = ( C_X C_Y , gamma_X + A_X * gamma_Y )
Inverse:                         X^{-1}  = ( C^{-1}  , -A^T * gamma )
Right action (Eq. 16):           phi(X, xi) = ( T C , A^T * (b - gamma) )
```

where `A * (x1, x2) = (A x1, A x2)` rotates each R^3 block.

### Key structural fact

At the **group-product** level every R^3 block of the 12-vector
`(a, b, g_omega, g_accel)` transforms identically by pure rotation
(`u_XY = u_X + A_X u_Y`), so

```
G_TF  ==  SO(3) |x R^12   (one rotation acting on four R^3 blocks)
```

The SE_2(3)-style coupling (`v^A`, `p^A` in the adjoint; the left Jacobian
`J_l` in Exp/Log) is **shared by all four blocks**. This makes every group
operation a single loop over the four blocks, and Exp/Log are *exact* (no fiber
truncation), unlike the tangent-group construction used by the TG-EqF.

### Representation choice

The state is identified with a group element (the IEKF runs directly on
`G_TF`). We store the navigation part as `(R, v, p)` and keep the bias
*generator* `gamma`. Physical biases are recovered with `b = A^{-1}*(-gamma)`
(Eq. B.31). Storing `gamma` (not `b`) keeps `Compose`/`Inverse` matching the
paper's `(C, gamma)` formulas one-to-one. `FromState` / `bias_omega()` /
`bias_accel()` convert at the boundary.

### Tangent block order (fixed everywhere)

```
[ d_theta(3) | d_v(3) | d_p(3) | d_gamma_omega(3) | d_gamma_accel(3) ]
```

The leading 9 match gtsam's SE_2(3) Logmap ordering (rot, vel, pos).

### Adjoint (15x15)

Diagonal `A` on every block; the `d_theta` column couples each vector block
`u_i` via `skew(u_i) A`:

```
Ad = [ A        0  0  0  0 ]
     [ v^A      A  0  0  0 ]
     [ p^A      0  A  0  0 ]
     [ g_w^A    0  0  A  0 ]
     [ g_a^A    0  0  0  A ]
```

Required by `InvariantEKF` for covariance propagation.

## 3. Lift (Theorem 6, Eq. 17-18)

The lift `Lambda(X, u) in g_TF` turns the group-affine dynamics (Eq. 3) into a
body-frame increment so the left-invariant EKF can predict by composition. In
vee coordinates `Lambda_1` (Eq. 17) reduces to the closed form of Eq. 3, and
`Lambda_2` (Eq. 18) is the bias generator:

```
d_theta       = omega - b_omega                          (3a)
d_v           = (accel - b_accel) + R^T g                (3b)
d_p           = R^T v                                    (3c)
d_gamma_omega = b_omega x (omega - b_omega) - tau_omega  (18)
d_gamma_accel = b_accel x (omega - b_omega) - tau_accel  (18)
```

Predict increment: `U = Expmap(Lambda(X, u) * dt)`, then `X <- X * U`.

## 4. Position output (Sec. 7, Lemma 15)

The raw model `h(xi) = p` (Eq. 33) is **not** output-equivariant for `G_TF`.
The body-frame residual reformulation (Lemma 15) is:

```
h(xi)     = R^T (pi - p)            (Eq. 34)   noise-free value 0 when pi = p
rho_X(y)  = R_X^T (y - p_X)         (Eq. 35)   output action
```

Equivariance `h(xi * X) = rho_X(h(xi))` is verified in tests.

### Output Jacobians

For the chart `xi(eps) = xi_hat * Expmap(eps)`, output equivariance gives
`h(xi(eps)) = rho_{Exp(eps)}(y_hat)` **exactly**, with `y_hat = R_hat^T(pi - p_hat)`.
Differentiating at `eps = 0`:

```
C0    = [ y_hat^             0  -I  0 ]   true first-order Jacobian
Cstar = [ 1/2 (y_hat+p_hat)^ 0  -I  0 ]   midpoint-symmetrised (Eq. B.35)
```

`Cstar` averages the skew of the predicted output `y_hat` with that of the
**back-transported measurement** `rho_{X_hat^-1}(0) = p_hat` (this is where the
`p_hat` in Eq. B.35 comes from). It centres the attitude coupling between
estimate and measurement, giving cubic `O(||e||^3)` linearisation error and
avoiding spurious tilt when the position innovation is large.

`Cstar` is the **default**; `C0` is available via a flag. Both pair with the
same innovation `y_hat`, so only the matrix `H` changes.

Note: for a level vehicle with an accurate fix (`R ~ I`, `pi ~ truth`),
`y_hat ~ -p_hat`, so `Cstar` zeroes the position->attitude coupling — correct,
but it then does not drive horizontal velocity through the gravity feedback
loop that `C0` exploits in pure-static scenarios.

## 5. Filter (gtsam::InvariantEKF)

`TfgInEKF : public InvariantEKF<TwoFrameGroup>`.

- **Predict**: build the state-dependent increment `U` and call
  `predict(U, Qc*dt)`: `X <- X*U`, `P <- Ad_{U^-1} P Ad_{U^-1}^T + Q`.
- **Update**: body residual `y_hat`, target `z = 0`, `H = Cstar` (or `C0`),
  measurement noise rotated into the residual frame `R_meas = R^T R_pos R`.
  No reset step (the TFG-IEKF runs without curvature correction, Table 2).

## 6. Design decisions & known approximations

1. **Covariance uses `A = Ad_{U^-1}`** (standard InvariantEKF propagation). For
   the two-frames group the navigation error is autonomous (exact); the exact
   bias-error coupling of `A_t^0` (Eq. B.34) is approximated. Navigation error
   linearisation is exact — the property that distinguishes the TFG geometry.
2. **`Cstar` default** for the position update (cubic error). `C0` is the exact
   first-order Jacobian, kept as a flag and for finite-difference validation.
3. **No reset step**, matching the published TFG-IEKF (Table 2). The state's
   `Retract` has no Jacobian overload, so covariance transport is skipped.
4. **Gravity** defaults to `(0, 0, -9.81)` (z-up), consistent with
   `PreintegrationParams::MakeSharedU`.

## 7. References

- Fornasier et al., arXiv:2309.03765v3 — Sec. 5.3, Thm 6, Sec. 7, App. B.7.
- Barrau & Bonnabel, *The Geometry of Navigation Problems* (TFG-IEKF) [4].
- gtsam `InvariantEKF` / `LeftLinearEKF` / `ManifoldEKF`.
