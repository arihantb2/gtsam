#include <gtsam/base/Matrix.h>
#include <gtsam_unstable/tfg_inekf/Symmetry.h>

namespace tfg {

using Vec3 = Eigen::Vector3d;
using Matrix3 = Eigen::Matrix3d;

static inline Matrix3 skew(const Vec3& x) { return gtsam::skewSymmetric(x); }

Eigen::Vector3d gravity() { return {0.0, 0.0, -kGravity}; }

TwoFrameGroup phi(const TwoFrameGroup& X, const TwoFrameGroup& xi) {
  return xi * X;
}

Tangent lift(const TwoFrameGroup& X, const ImuInput& u) {
  const Vec3 b_omega = X.bias_omega();
  const Vec3 b_accel = X.bias_accel();
  const Vec3 gyro_err = u.omega - b_omega;

  Tangent xi;
  xi.segment<3>(0) = gyro_err;
  xi.segment<3>(3) = (u.accel - b_accel) + X.R.unrotate(gravity());
  xi.segment<3>(6) = X.R.unrotate(X.v);
  xi.segment<3>(9) = b_omega.cross(gyro_err) - u.tau_omega;
  xi.segment<3>(12) = b_accel.cross(gyro_err) - u.tau_accel;
  return xi;
}

// Df = d lift(X * Exp(eps), u) / d eps at eps = 0. Non-zero blocks only:
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
  Df.block<3, 3>(0, 0) = -skew(b_omega);
  Df.block<3, 3>(0, 9) = I3;
  Df.block<3, 3>(3, 0) = skew(Rt_g) - skew(b_accel);
  Df.block<3, 3>(3, 12) = I3;
  Df.block<3, 3>(6, 0) = skew(Rt_v);
  Df.block<3, 3>(6, 3) = I3;
  Df.block<3, 3>(9, 0) = -skew(u.omega) * skew(b_omega);
  Df.block<3, 3>(9, 9) = skew(u.omega);
  Df.block<3, 3>(12, 0) =
      -skew(w_hat) * skew(b_accel) - skew(b_accel) * skew(b_omega);
  Df.block<3, 3>(12, 9) = skew(b_accel);
  Df.block<3, 3>(12, 12) = skew(w_hat);
  return Df;
}

// Qd = B diag(sigma^2) B^T dt. Input order: (n_omega, n_accel, tau_omega,
// tau_accel).
Eigen::Matrix<double, 15, 15> inputNoiseCov(const TwoFrameGroup& X,
                                            const ImuNoise& noise, double dt) {
  const Vec3 b_omega = X.bias_omega();
  const Vec3 b_accel = X.bias_accel();
  const Matrix3 I3 = Matrix3::Identity();

  Eigen::Matrix<double, 15, 12> B = Eigen::Matrix<double, 15, 12>::Zero();
  B.block<3, 3>(0, 0) = I3;
  B.block<3, 3>(3, 3) = I3;
  B.block<3, 3>(9, 0) = skew(b_omega);
  B.block<3, 3>(9, 6) = -I3;
  B.block<3, 3>(12, 0) = skew(b_accel);
  B.block<3, 3>(12, 9) = -I3;

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
