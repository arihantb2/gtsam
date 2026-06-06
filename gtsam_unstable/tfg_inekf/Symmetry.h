#pragma once
#include "Group.h"
#include <Eigen/Dense>

// ============================================================================
//  Two-Frames symmetry of the biased INS:  action phi + lift Lambda
// ----------------------------------------------------------------------------
//  Reference: Fornasier et al. (arXiv:2309.03765v3)
//    - Section 4, Eq. (3)   : continuous-time biased INS dynamics
//    - Section 5.3          : group action phi (Lemma 5, Eq. 16)
//    - Theorem 6, Eq.(17-18): equivariant lift Lambda for G_TF
//    - Appendix B.7         : resulting TFG-IEKF error/state matrices
//
//  IMU input (Eq. 5):  u = (omega, accel, tau_omega, tau_accel)
//    omega, accel        : gyro / accelerometer readings (body frame {I})
//    tau_omega, tau_accel: bias random-walk drivers (zero for constant bias)
//
//  The lift turns the system dynamics (3) into a curve on the group so that the
//  left-invariant EKF can propagate the state by group composition,
//    X_{k+1} = X_k * U,   U = Expmap(Lambda(X_k, u) * dt).
//  Because Lambda depends on the state (the biases), the increment U is built
//  here and passed to gtsam::InvariantEKF::predict(U, Q) (not the
//  state-independent predict(u, dt, Q) overload).
// ============================================================================

namespace tfg {

/// Lifted IMU input, fed to InvariantEKF::predict.
/// Mirrors u in Eq. (5); biases are part of the state, not this struct.
struct ImuInput {
    Eigen::Vector3d omega;       ///< gyroscope reading      (body frame)
    Eigen::Vector3d accel;       ///< accelerometer reading  (body frame)
    Eigen::Vector3d tau_omega = Eigen::Vector3d::Zero(); ///< gyro  bias driver
    Eigen::Vector3d tau_accel = Eigen::Vector3d::Zero(); ///< accel bias driver
};

/// Gravity in the global frame {G} (Eq. 3b), default -z.
Eigen::Vector3d gravity();

// ----------------------------------------------------------------------------
//  Right group action  phi : G_TF x M -> M           (Sec. 5.3, Eq. 16)
//    phi(X, xi) = ( T C , A^T * (b - gamma) )
//  Transitive, so the origin xi0 = (I_5, 0) generates the whole state space.
//  Used to (a) recover the state estimate from X, (b) document equivariance.
// ----------------------------------------------------------------------------
TwoFrameGroup phi(const TwoFrameGroup& X, const TwoFrameGroup& xi);

// ----------------------------------------------------------------------------
//  Equivariant lift  Lambda : M x L -> g_TF          (Theorem 6, Eq. 17-18)
//
//    Lambda_1(xi,u) = (W - B + N) + T^{-1}(G - N) T          in se_2(3)
//    Lambda_2(xi,u) = ( b_w^ (omega - b_w) - tau_w ,
//                       b_a^ (omega - b_w) - tau_a )         in R^6
//
//  with W, B, N, G the se_2(3) matrices built from (omega, accel) and the
//  biases (Sec. 5.2, Eq. 12). In vee coordinates Lambda_1 reduces to the closed
//  form of Eq. 3 (theta = omega - b_w, v_dot = (a - b_a) + R^T g, p_dot = R^T v),
//  which is what the implementation evaluates. Returns the 15-vector in g_TF
//  (block order of Group.h::Tangent), the per-step generator for InvariantEKF.
// ----------------------------------------------------------------------------
Tangent lift(const TwoFrameGroup& X, const ImuInput& u);

/// Convenience: group increment  U = Expmap(lift(X,u) * dt)  for
/// InvariantEKF::predict(U, Q). Keeps the EKF call site symmetry-agnostic.
TwoFrameGroup increment(const TwoFrameGroup& X, const ImuInput& u, double dt);

} // namespace tfg
