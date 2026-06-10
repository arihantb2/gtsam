#include <gtsam_unstable/tfg_inekf/Symmetry.h>

#include <gtsam/base/Matrix.h>

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
using Matrix3 = Eigen::Matrix3d;

// skew(x) = x^ (3x3 skew-symmetric matrix), via gtsam.
static inline Matrix3 skew(const Vec3& x) { return gtsam::skewSymmetric(x); }

Eigen::Vector3d gravity() { return {0.0, 0.0, -kGravity}; }

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

// Df = d lift(X * Exp(eps), u) / d eps at eps = 0 (right-retract chart). Only
// the blocks below are non-zero; derivation in CODE_REVIEW.md (F1). With
// b_w = bias_omega, b_a = bias_accel, w_hat = omega - b_w, g = gravity():
//   Df[theta , theta ] = -b_w^                  Df[theta , g_w] =  I
//   Df[v     , theta ] = (R^T g)^ - b_a^         Df[v     , g_a] =  I
//   Df[p     , theta ] = (R^T v)^               Df[p     , v  ] =  I
//   Df[g_w   , theta ] = -omega^ b_w^            Df[g_w   , g_w] =  omega^
//   Df[g_a   , theta ] = -w_hat^ b_a^ - b_a^ b_w^
//                        Df[g_a, g_w] = b_a^     Df[g_a   , g_a] =  w_hat^
Eigen::Matrix<double, 15, 15> liftJacobian(const TwoFrameGroup& X,
                                           const ImuInput& u) {
  const Vec3 b_omega = X.bias_omega();
  const Vec3 b_accel = X.bias_accel();
  const Vec3 w_hat = u.omega - b_omega;
  const Vec3 Rt_g = X.R.unrotate(gravity());
  const Vec3 Rt_v = X.R.unrotate(X.v);
  const Matrix3 I3 = Matrix3::Identity();

  Eigen::Matrix<double, 15, 15> Df = Eigen::Matrix<double, 15, 15>::Zero();
  // d_theta row
  Df.block<3, 3>(0, 0) = -skew(b_omega);
  Df.block<3, 3>(0, 9) = I3;
  // d_v row
  Df.block<3, 3>(3, 0) = skew(Rt_g) - skew(b_accel);
  Df.block<3, 3>(3, 12) = I3;
  // d_p row
  Df.block<3, 3>(6, 0) = skew(Rt_v);
  Df.block<3, 3>(6, 3) = I3;
  // d_gamma_omega row
  Df.block<3, 3>(9, 0) = -skew(u.omega) * skew(b_omega);
  Df.block<3, 3>(9, 9) = skew(u.omega);
  // d_gamma_accel row
  Df.block<3, 3>(12, 0) = -skew(w_hat) * skew(b_accel) - skew(b_accel) * skew(b_omega);
  Df.block<3, 3>(12, 9) = skew(b_accel);
  Df.block<3, 3>(12, 12) = skew(w_hat);
  return Df;
}

// Qd = B(X) diag(sigma^2) B(X)^T dt. Input noise order (12-vector):
// (n_omega, n_accel, tau_omega, tau_accel). B is the differential of the lift
// w.r.t. these inputs (see Symmetry.h for the block layout).
Eigen::Matrix<double, 15, 15> inputNoiseCov(const TwoFrameGroup& X,
                                            const ImuNoise& noise, double dt) {
  const Vec3 b_omega = X.bias_omega();
  const Vec3 b_accel = X.bias_accel();
  const Matrix3 I3 = Matrix3::Identity();

  Eigen::Matrix<double, 15, 12> B = Eigen::Matrix<double, 15, 12>::Zero();
  B.block<3, 3>(0, 0) = I3;             // theta <- n_omega
  B.block<3, 3>(3, 3) = I3;             // v     <- n_accel
  B.block<3, 3>(9, 0) = skew(b_omega);  // g_w   <- n_omega
  B.block<3, 3>(9, 6) = -I3;            // g_w   <- tau_omega
  B.block<3, 3>(12, 0) = skew(b_accel); // g_a   <- n_omega
  B.block<3, 3>(12, 9) = -I3;           // g_a   <- tau_accel

  Eigen::Matrix<double, 12, 1> sig;
  sig.segment<3>(0).setConstant(noise.gyro);
  sig.segment<3>(3).setConstant(noise.accel);
  sig.segment<3>(6).setConstant(noise.gyro_rw);
  sig.segment<3>(9).setConstant(noise.accel_rw);

  return (B * sig.asDiagonal() * B.transpose() * dt).eval();
}

TwoFrameGroup increment(const TwoFrameGroup& X, const ImuInput& u, double dt) {
  return TwoFrameGroup::Expmap(lift(X, u) * dt);
}

}  // namespace tfg
