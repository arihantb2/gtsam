#include <gtsam_unstable/mekf/Dynamics.h>

// Strapdown mean (forward Euler):
//   R+ = R * Exp((omega - b_gyro) dt)
//   v+ = v + (R (accel - b_accel) + g) dt
//   p+ = p + v dt
// F: finite-diff of discrete map in local coords, re-evaluated each step.

namespace mekf {

using Vec3 = Eigen::Vector3d;

Eigen::Vector3d gravity() { return {0.0, 0.0, -9.81}; }

MekfState propagateMean(const MekfState& X, const ImuInput& u, double dt) {
  const Vec3 gyro_corr = u.omega - X.b_gyro;
  const Vec3 acc_corr = u.accel - X.b_accel;

  MekfState Xn;
  Xn.R = X.R * gtsam::Rot3::Expmap(gyro_corr * dt);
  Xn.v = X.v + (X.R.rotate(acc_corr) + gravity()) * dt;
  Xn.p = X.p + X.v * dt;
  Xn.b_gyro = X.b_gyro;
  Xn.b_accel = X.b_accel;
  return Xn;
}

Eigen::Matrix<double, 15, 15> transitionJacobian(const MekfState& X,
                                                 const ImuInput& u, double dt) {
  using Tangent = Eigen::Matrix<double, 15, 1>;
  constexpr double h = 1e-7;

  const MekfState Xn_ref = propagateMean(X, u, dt);

  Eigen::Matrix<double, 15, 15> F;
  for (int j = 0; j < 15; ++j) {
    Tangent e = Tangent::Zero();
    e(j) = h;
    const MekfState Xp = gtsam::traits<MekfState>::Retract(X, e);
    const MekfState Xpn = propagateMean(Xp, u, dt);
    F.col(j) = gtsam::traits<MekfState>::Local(Xn_ref, Xpn) / h;
  }
  return F;
}

}  // namespace mekf
