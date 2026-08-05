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

Eigen::Matrix<double, 3, 15> PositionOutput::jacobian(
    const TwoFrameGroup& xi_hat, const Eigen::Vector3d& pi, Variant variant) {
  const Eigen::Vector3d y_hat = predict(xi_hat, pi);
  const Eigen::Vector3d coupling =
      (variant == Variant::Cstar) ? (0.5 * y_hat).eval() : y_hat;

  Eigen::Matrix<double, 3, 15> C = Eigen::Matrix<double, 3, 15>::Zero();
  C.block<3, 3>(0, 0) = gtsam::skewSymmetric(coupling);
  C.block<3, 3>(0, 6) = -Eigen::Matrix3d::Identity();
  return C;
}

}  // namespace tfg
