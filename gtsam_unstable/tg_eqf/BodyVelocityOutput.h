#pragma once
#include <gtsam_unstable/tg_eqf/Group.h>
#include <gtsam_unstable/tg_eqf/State.h>

#include <Eigen/Dense>

namespace gtsam {
namespace tgeqf {

/**
 * Equivariant DVL body-frame velocity measurement h_d(xi) = R^T v.
 *
 * Both Jacobians below are built at the reference state and are already in
 * **error coordinates**: pass them to TGEqF::updateWithReset() unchanged. The
 * measurement arrives in the body frame of the true state, so pull it back to
 * the origin output space with inverse_output_action() first.
 */
struct DVLMeasurement {
  static Eigen::Vector3d predict(const State& xi);

  /**
   * Origin-chart Jacobian C0 in R^{3x18}
   *
   *   C0 = [ y0^  I_3  0  0 ],   y0 = R0^T v0
   *
   * The output differential taken at the origin output alone, so its
   * linearization error is second order in the state error. The filter uses C*
   * below; C0 is the baseline the accuracy tests measure against.
   */
  static Eigen::Matrix<double, 3, 18> jacobian_C0(const State& xi_ref);

  /**
   * Equivariant output matrix C* in R^{3x18}, in error coordinates
   *
   *   C* = [ 0.5 (y0 + y_tilde)^  I_3  0  0 ]
   *
   * with y0 = R0^T v0 the origin output and y_tilde = psi_X^{-1}(z) the
   * measurement pulled back to the origin output space. Taking the output
   * differential at the midpoint of y0 and y_tilde instead of at y0 alone
   * makes the linearization error third order in the state error, one order
   * better than C0. This is the matrix the filter uses.
   */
  static Eigen::Matrix<double, 3, 18> jacobian_Cstar(
      const State& xi_ref, const TGElement& g, const Eigen::Vector3d& z_dvl);

  /// Output group action psi_X(y) = R_X^T y + R_X^T v_X.
  static Eigen::Vector3d output_action(const TGElement& X,
                                       const Eigen::Vector3d& y);

  /// Inverse output action psi_X^{-1}(y) = R_X y - v_X, which carries a
  /// measurement from the body frame of the true state to the origin output
  /// space the filter compares in.
  static Eigen::Vector3d inverse_output_action(const TGElement& X,
                                               const Eigen::Vector3d& y);
};

}  // namespace tgeqf
}  // namespace gtsam
