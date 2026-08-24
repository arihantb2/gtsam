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
  /// Output matrix in error coordinates, d(output)/d(eps) at eps = 0.
  using Jacobian = Eigen::Matrix<double, 3, 18>;

  static Eigen::Vector3d predict(const State& xi);

  /**
   * Output matrix C* in R^{3x18}, in error coordinates
   *
   *   C* = [ M ad(b_ref)  -M ],   M = [ -R_X^T p_X^  0  R_X^T ]
   *
   * (M's three 3x3 slots line up with b_w, b_a, b_v; only the b_w and b_v
   * slots are nonzero). This chart folds eps_b through the fiber exponential
   * Xi(eps_tau) rather than a plain vector difference (State.h), so this is a
   * first-order linearization only. The ad(b_ref) term is the chart's
   * nav-to-bias coupling and vanishes only when xi_ref's bias is zero.
   */
  static Jacobian jacobian_Cstar(const State& xi_ref, const TGElement& g);
};

}  // namespace tgeqf
}  // namespace gtsam
