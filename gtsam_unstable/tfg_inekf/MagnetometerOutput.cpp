#include <gtsam/base/Matrix.h>
#include <gtsam_unstable/tfg_inekf/MagnetometerOutput.h>

namespace tfg {

Eigen::Vector3d MagnetometerOutput::predict(const TwoFrameGroup& xi,
                                            const Eigen::Vector3d& m) {
  return xi.R.unrotate(m - xi.gamma_mag);
}

Eigen::Vector3d MagnetometerOutput::output_action(const TwoFrameGroup& X,
                                                  const Eigen::Vector3d& y) {
  return X.R.unrotate(y - X.gamma_mag);
}

Eigen::Matrix<double, 3, 18> MagnetometerOutput::jacobian(
    const TwoFrameGroup& xi_hat, const Eigen::Vector3d& m) {
  Eigen::Matrix<double, 3, 18> C = Eigen::Matrix<double, 3, 18>::Zero();
  C.block<3, 3>(0, 0) = gtsam::skewSymmetric(predict(xi_hat, m));
  C.block<3, 3>(0, 15) = -Eigen::Matrix3d::Identity();
  return C;
}

}  // namespace tfg
