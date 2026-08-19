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
   * The output differential taken at the origin output alone, so its
   * linearization error is second order in the state error. The filter uses C*
   * below; C0 is the baseline the accuracy tests measure against.
   */
  static Eigen::Matrix<double, 3, 18> jacobian_C0(const State& xi_ref,
                                                  const Eigen::Vector3d& pi);

  /**
   * Equivariant output matrix C* in R^{3x18}, in error coordinates
   *
   *   C* = [ 0.5 (y0 + p_X)^  0  -I_3  0 ]
   *
   * with y0 = R0^T (pi - p0) the origin output and p_X = g.p the transported
   * measurement. Taking the output differential at the midpoint of y0 and p_X
   * instead of at y0 alone makes the linearization error third order in the
   * state error, one order better than C0. This is the matrix the filter uses.
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
