#include <gtsam/base/Matrix.h>
#include <gtsam_unstable/mekf/BodyVelocityOutput.h>

namespace mekf {

Eigen::Vector3d BodyVelocityOutput::predict(const MekfState& X) {
  return X.R.unrotate(X.v);
}

Eigen::Matrix<double, 3, 15> BodyVelocityOutput::jacobian(const MekfState& X) {
  const Eigen::Vector3d body_v = predict(X);

  Eigen::Matrix<double, 3, 15> H = Eigen::Matrix<double, 3, 15>::Zero();
  H.block<3, 3>(0, 0) = gtsam::skewSymmetric(body_v);
  H.block<3, 3>(0, 3) = X.R.matrix().transpose();
  return H;
}

}  // namespace mekf
