#pragma once
#include <gtsam_unstable/tg_eqf/Group.h>
#include <gtsam_unstable/tg_eqf/State.h>

#include <Eigen/Dense>

namespace tgeqf {

/**
 * Virtual-velocity-bias pseudo-measurement h(xi) = b_v = 0.
 */
struct VirtualBiasMeasurement {
  static Eigen::Vector3d predict(const State& xi);

  static Eigen::Matrix<double, 3, 18> jacobian_C0(const State& xi_hat);

  static Eigen::Vector3d innovation(const State& xi_hat);
};

}  // namespace tgeqf
