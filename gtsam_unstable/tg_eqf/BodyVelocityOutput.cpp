#include <gtsam/geometry/Rot3.h>
#include <gtsam_unstable/tg_eqf/BodyVelocityOutput.h>

namespace gtsam {
namespace tgeqf {

Eigen::Vector3d DVLMeasurement::predict(const State& xi) {
  return xi.R.unrotate(xi.v);
}

Eigen::Matrix<double, 3, 18> DVLMeasurement::stateJacobian(
    const State& xi_hat) {
  const Eigen::Vector3d body_v = predict(xi_hat);

  Eigen::Matrix<double, 3, 18> H = Eigen::Matrix<double, 3, 18>::Zero();
  H.block<3, 3>(0, 0) = gtsam::skewSymmetric(body_v);
  H.block<3, 3>(0, 3) = xi_hat.R.matrix().transpose();
  return H;
}

Eigen::Vector3d DVLMeasurement::innovation(const Eigen::Vector3d& z,
                                           const State& xi_hat) {
  return z - predict(xi_hat);
}

Eigen::Vector3d DVLMeasurement::output_action(const TGElement& X,
                                              const Eigen::Vector3d& y) {
  return X.R.unrotate(y + X.v);
}

}  // namespace tgeqf
}  // namespace gtsam
