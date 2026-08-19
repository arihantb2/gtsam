#pragma once
#include <gtsam_unstable/tg_eqf/Group.h>
#include <gtsam_unstable/tg_eqf/State.h>

#include <Eigen/Dense>

namespace gtsam {
namespace tgeqf {

/**
 * Equivariant GNSS position measurement h'(xi) = R^T(pi - p).
 *
 * Both Jacobians below are built at the reference state and are already in
 * **error coordinates**: pass them to TGEqF::updateWithReset() unchanged.
 */
struct PositionMeasurement {
  static Eigen::Vector3d predict(const State& xi, const Eigen::Vector3d& pi);

  /**
   * Origin-chart Jacobian C0 in R^{3x18}
   *
   *   C0 = [ y0^  0  -I_3  0 ],   y0 = R0^T (pi - p0)
   *
   * The plain origin endpoint, with a second-order linearization error. Kept
   * as a reference point for the order-of-accuracy tests; the filter uses C*.
   */
  static Eigen::Matrix<double, 3, 18> jacobian_C0(const State& xi_ref,
                                                  const Eigen::Vector3d& pi);

  /**
   * Equivariant output matrix C*, Fornasier et al. Equ. (B.19)
   *
   *   C* = [ 0.5 (y0 + p_X)^  0  -I_3  0 ]
   *
   * with y0 = R0^T (pi - p0), p_X = g.p. The half is the midpoint of the two
   * output-action differentials, at the origin output y0 and at the transported
   * measurement p_X, which buys a third-order linearization error where C0 is
   * only second order. This is what the filter uses.
   */
  static Eigen::Matrix<double, 3, 18> jacobian_Cstar(const State& xi_ref,
                                                     const TGElement& g,
                                                     const Eigen::Vector3d& pi);

  /// Output group action psi_X(y) = R_X^T (y - p_X).
  static Eigen::Vector3d output_action(const TGElement& X,
                                       const Eigen::Vector3d& y);
};

}  // namespace tgeqf
}  // namespace gtsam
