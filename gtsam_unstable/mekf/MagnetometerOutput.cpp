#include <gtsam/base/Matrix.h>
#include <gtsam_unstable/mekf/MagnetometerOutput.h>

namespace mekf {

Eigen::Vector3d MagnetometerOutput::predict(const MekfState& X,
                                            const Eigen::Vector3d& m) {
  return X.R.unrotate(m) + X.b_mag;
}

Eigen::Matrix<double, 3, 18> MagnetometerOutput::jacobian(
    const MekfState& X, const Eigen::Vector3d& m) {
  Eigen::Matrix<double, 3, 18> H = Eigen::Matrix<double, 3, 18>::Zero();
  H.block<3, 3>(0, 0) = gtsam::skewSymmetric(X.R.unrotate(m));
  H.block<3, 3>(0, 15) = Eigen::Matrix3d::Identity();
  return H;
}

}  // namespace mekf
