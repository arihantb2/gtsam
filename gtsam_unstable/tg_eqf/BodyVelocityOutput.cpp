#include <gtsam_unstable/tg_eqf/BodyVelocityOutput.h>

#include <gtsam/geometry/Rot3.h>

namespace tgeqf {

Eigen::Vector3d DVLMeasurement::predict(const TGState& xi) {
  return xi.R.unrotate(xi.v);
}

Eigen::Matrix<double, 3, 18> DVLMeasurement::jacobian(
    const TGState& xi_ref) {
  const Eigen::Vector3d body_v = predict(xi_ref);

  Eigen::Matrix<double, 3, 18> H =
      Eigen::Matrix<double, 3, 18>::Zero();
  H.block<3, 3>(0, 0) = gtsam::skewSymmetric(body_v);
  H.block<3, 3>(0, 6) = xi_ref.R.matrix().transpose();
  return H;
}

Eigen::Vector3d DVLMeasurement::innovation(
    const Eigen::Vector3d& z, const TGState& xi_hat) {
  return z - predict(xi_hat);
}

Eigen::Vector3d DVLMeasurement::output_action(
    const TGGroupElement& X, const Eigen::Vector3d& y) {
  return X.R_X.unrotate(y) + X.R_X.unrotate(X.v_X);
}

}  // namespace tgeqf
