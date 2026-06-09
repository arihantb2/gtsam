#include <gtsam_unstable/tg_eqf/PositionOutput.h>

#include <gtsam/geometry/Rot3.h>

namespace tgeqf {

Eigen::Vector3d PositionMeasurement::predict(const TGState& xi,
                                              const Eigen::Vector3d& pi) {
  return xi.R.unrotate(pi - xi.p);
}

Eigen::Matrix<double, 3, 18> PositionMeasurement::jacobian_C0(
    const TGState& /*xi_ref*/) {
  // Tangent order is [delta_R | delta_v | delta_p | delta_b] (State::Retract).
  // d(R^T(pi - p))/d(delta_p) = -I sits in the position columns (6..8).
  Eigen::Matrix<double, 3, 18> C0 =
      Eigen::Matrix<double, 3, 18>::Zero();
  C0.block<3, 3>(0, 6) = -Eigen::Matrix3d::Identity();
  return C0;
}

Eigen::Matrix<double, 3, 18> PositionMeasurement::jacobian_Cstar(
    const TGState& xi_hat, const Eigen::Vector3d& pi) {
  const Eigen::Vector3d y = predict(xi_hat, pi);
  const Eigen::Vector3d vec = y + xi_hat.p;

  Eigen::Matrix<double, 3, 18> Cstar =
      Eigen::Matrix<double, 3, 18>::Zero();
  Cstar.block<3, 3>(0, 0) =
      0.5 * gtsam::skewSymmetric(vec);
  Cstar.block<3, 3>(0, 6) = -Eigen::Matrix3d::Identity();
  return Cstar;
}

Eigen::Vector3d PositionMeasurement::innovation(
    const Eigen::Vector3d& pi, const TGState& xi_hat) {
  // Equivariant target is zero when pi == p; residual is -h'(xi_hat).
  return -predict(xi_hat, pi);
}

Eigen::Vector3d PositionMeasurement::output_action(
    const TGGroupElement& X, const Eigen::Vector3d& y) {
  return X.R.unrotate(y - X.p);
}

}  // namespace tgeqf
