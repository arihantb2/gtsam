#include <gtsam/base/Matrix.h>
#include <gtsam_unstable/tfg_inekf/BodyVelocityOutput.h>

namespace tfg {

Eigen::Vector3d BodyVelocityOutput::predict(const TwoFrameGroup& xi) {
  return xi.R.unrotate(xi.v);
}

Eigen::Vector3d BodyVelocityOutput::output_action(const TwoFrameGroup& X,
                                                  const Eigen::Vector3d& y) {
  return X.R.unrotate(y) + X.R.unrotate(X.v);
}

Eigen::Matrix<double, 3, 15> BodyVelocityOutput::jacobian(
    const TwoFrameGroup& xi_hat) {
  const Eigen::Vector3d y_hat = predict(xi_hat);

  Eigen::Matrix<double, 3, 15> H = Eigen::Matrix<double, 3, 15>::Zero();
  H.block<3, 3>(0, 0) = gtsam::skewSymmetric(y_hat);
  H.block<3, 3>(0, 3) = Eigen::Matrix3d::Identity();
  return H;
}

}  // namespace tfg
