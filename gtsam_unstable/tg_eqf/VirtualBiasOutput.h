#pragma once
#include <gtsam_unstable/tg_eqf/Group.h>
#include <gtsam_unstable/tg_eqf/State.h>

#include <Eigen/Dense>

namespace gtsam {
namespace tgeqf {

/**
 * Virtual-velocity-bias pseudo-measurement h(xi) = b_v = 0.
 */
struct VirtualBiasMeasurement {
  static Eigen::Vector3d predict(const State& xi);

  /**
   * Measurement Jacobian in the Retract chart at the state it is evaluated at,
   * a plain selector on the virtual-bias block. This is **not** error
   * coordinates: run it through EquivariantFilter::outputMatrix() before
   * passing it to update().
   */
  static Eigen::Matrix<double, 3, 18> stateJacobian(const State& xi_hat);

  static Eigen::Vector3d innovation(const State& xi_hat);
};

}  // namespace tgeqf
}  // namespace gtsam
