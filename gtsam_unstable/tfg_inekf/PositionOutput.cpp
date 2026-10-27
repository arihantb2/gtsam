#include <gtsam/base/Matrix.h>
#include <gtsam_unstable/tfg_inekf/PositionOutput.h>

namespace tfg {

Eigen::Vector3d PositionOutput::predict(const TwoFrameGroup& xi,
                                        const Eigen::Vector3d& pi) {
  return xi.R.unrotate(pi - xi.p);
}

Eigen::Vector3d PositionOutput::output_action(const TwoFrameGroup& X,
                                              const Eigen::Vector3d& y) {
  return X.R.unrotate(y - X.p);
}

Eigen::Matrix<double, 3, 18> PositionOutput::jacobian(
    const TwoFrameGroup& xi_hat, const Eigen::Vector3d& pi) {
  Eigen::Matrix<double, 3, 18> C = Eigen::Matrix<double, 3, 18>::Zero();
  C.block<3, 3>(0, 0) = gtsam::skewSymmetric(predict(xi_hat, pi));
  C.block<3, 3>(0, 6) = -Eigen::Matrix3d::Identity();
  return C;
}

}  // namespace tfg
