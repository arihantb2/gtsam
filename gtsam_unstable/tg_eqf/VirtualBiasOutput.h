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
   * Output matrix C* in R^{3x18}, in error coordinates
   *
   *   C* = [ 0  0  0  R_X^T p_X^  0  -R_X^T ]
   *
   * This chart folds eps_b through the fiber exponential Xi(eps_tau) rather
   * than a plain vector difference (State.h), so this is a first-order
   * linearization only, and it depends on xi_ref's bias too, via the
   * chart's nav-to-bias coupling.
   */
  static Eigen::Matrix<double, 3, 18> jacobian_Cstar(const State& xi_ref,
                                                     const TGElement& g);
};

}  // namespace tgeqf
}  // namespace gtsam
