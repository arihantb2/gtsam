#include <gtsam/geometry/Rot3.h>
#include <gtsam_unstable/tg_eqf/BodyVelocityOutput.h>

namespace gtsam {
namespace tgeqf {

Eigen::Vector3d DVLMeasurement::predict(const State& xi) {
  return xi.R.unrotate(xi.v);
}

Eigen::Matrix<double, 3, 18> DVLMeasurement::jacobian_C0(const State& xi_ref) {
  const Eigen::Vector3d y0 = predict(xi_ref);
  Eigen::Matrix<double, 3, 18> C0 = Eigen::Matrix<double, 3, 18>::Zero();
  C0.block<3, 3>(0, 0) = gtsam::skewSymmetric(y0);
  C0.block<3, 3>(0, 3) = Eigen::Matrix3d::Identity();
  return C0;
}

Eigen::Matrix<double, 3, 18> DVLMeasurement::jacobian_Cstar(
    const State& xi_ref, const TGElement& g, const Eigen::Vector3d& z_dvl) {
  const Eigen::Vector3d y0 = predict(xi_ref);
  const Eigen::Vector3d vec = y0 + inverse_output_action(g, z_dvl);

  Eigen::Matrix<double, 3, 18> Cstar = Eigen::Matrix<double, 3, 18>::Zero();
  Cstar.block<3, 3>(0, 0) = 0.5 * gtsam::skewSymmetric(vec);
  Cstar.block<3, 3>(0, 3) = Eigen::Matrix3d::Identity();
  return Cstar;
}

Eigen::Vector3d DVLMeasurement::output_action(const TGElement& X,
                                              const Eigen::Vector3d& y) {
  return X.R.unrotate(y + X.v);
}

Eigen::Vector3d DVLMeasurement::inverse_output_action(
    const TGElement& X, const Eigen::Vector3d& y) {
  return X.R.rotate(y) - X.v;
}

}  // namespace tgeqf
}  // namespace gtsam
