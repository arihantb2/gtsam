#include <gtsam_unstable/mekf/PositionOutput.h>

namespace mekf {

Eigen::Vector3d PositionOutput::predict(const MekfState& X) { return X.p; }

Eigen::Matrix<double, 3, 15> PositionOutput::jacobian() {
  Eigen::Matrix<double, 3, 15> H = Eigen::Matrix<double, 3, 15>::Zero();
  H.block<3, 3>(0, 6) = Eigen::Matrix3d::Identity();
  return H;
}

}  // namespace mekf
