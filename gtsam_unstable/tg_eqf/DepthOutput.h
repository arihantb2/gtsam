#pragma once
#include <gtsam_unstable/tg_eqf/Group.h>
#include <gtsam_unstable/tg_eqf/State.h>

#include <Eigen/Dense>

namespace gtsam {
namespace tgeqf {

/**
 * Non-equivariant pressure-sensor depth output h(xi) = e_3^T p.
 *
 * The state action right-translates the position slot, p -> p + R p_A, so
 * h(phi(X, xi)) = h(xi) + e_3^T R p_A. That extra term depends on the
 * unmeasured rotation, so no output action can reproduce it. Hence no C*, and
 * no output_action or predictAtOrigin here: their absence is the point.
 */
struct DepthMeasurement {
  /// Vertical position e_3^T p.
  static double predict(const State& xi);

  /**
   * Output matrix C in R^{1x18}, in error coordinates at xi_ref:
   *
   *   C = [ (g.p x n)^T  0  n^T  0 ],   n = R0^T e_3
   *
   * The exact chart derivative of eps -> h(phi(g, Retract(xi_ref, eps))) at
   * eps = 0. Only second-order accurate, but unlike the equivariant C*
   * matrices it can be checked against a numerical derivative.
   */
  static Eigen::Matrix<double, 1, 18> jacobian_C(const State& xi_ref,
                                                 const TGElement& g);
};

}  // namespace tgeqf
}  // namespace gtsam
