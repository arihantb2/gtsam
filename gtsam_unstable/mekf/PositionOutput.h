#pragma once
#include <gtsam_unstable/mekf/State.h>

#include <Eigen/Dense>

namespace mekf {

struct PositionOutput {
  static Eigen::Vector3d predict(const MekfState& X);
  static Eigen::Matrix<double, 3, 15> jacobian();
};

}  // namespace mekf
