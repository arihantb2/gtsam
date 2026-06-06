#include <gtsam_unstable/tfg_inekf/Symmetry.h>

// ============================================================================
//  Two-Frames action phi and lift Lambda  (Sec. 5.3, Theorem 6, Eq. 16-18).
//
//  The lift Lambda_1 (Eq. 17, the se_2(3) navigation generator) is the matrix
//  expression (W - B) + T^{-1}(G - N)T which, in vee coordinates, reduces to
//  the closed form of the biased INS dynamics (Eq. 3). We implement that closed
//  form directly. Block order: [d_theta | d_v | d_p | d_gamma_w | d_gamma_a].
// ============================================================================

namespace tfg {

using Vec3 = Eigen::Vector3d;

Eigen::Vector3d gravity() { return {0.0, 0.0, -9.81}; }

// Right group action (Eq. 16). With the state space M = G_TF (the filter state
// IS the group element), the action is right multiplication:
//   phi(X, xi) = xi * X.
// Check in gamma coords: gamma_out = gamma_xi + R_xi gamma_X reproduces
//   b_out = A_X^T (b_xi - gamma_X)   since   gamma = -R b   (Eq. B.31).
TwoFrameGroup phi(const TwoFrameGroup& X, const TwoFrameGroup& xi) {
  return xi * X;
}

// Lambda(X, u): closed form of Eq. 17-18.
//   d_theta       = omega - b_omega                          (Eq. 3a)
//   d_v           = (accel - b_accel) + R^T g                (Eq. 3b)
//   d_p           = R^T v                                    (Eq. 3c)
//   d_gamma_omega = b_omega x (omega - b_omega) - tau_omega  (Eq. 18)
//   d_gamma_accel = b_accel x (omega - b_omega) - tau_accel  (Eq. 18)
Tangent lift(const TwoFrameGroup& X, const ImuInput& u) {
  const Vec3 b_omega = X.bias_omega();
  const Vec3 b_accel = X.bias_accel();
  const Vec3 gyro_err = u.omega - b_omega;

  Tangent xi;
  xi.segment<3>(0)  = gyro_err;
  xi.segment<3>(3)  = (u.accel - b_accel) + X.R.unrotate(gravity());
  xi.segment<3>(6)  = X.R.unrotate(X.v);
  xi.segment<3>(9)  = b_omega.cross(gyro_err) - u.tau_omega;
  xi.segment<3>(12) = b_accel.cross(gyro_err) - u.tau_accel;
  return xi;
}

TwoFrameGroup increment(const TwoFrameGroup& X, const ImuInput& u, double dt) {
  return TwoFrameGroup::Expmap(lift(X, u) * dt);
}

}  // namespace tfg
