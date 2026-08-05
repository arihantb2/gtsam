#pragma once
#include <gtsam_unstable/tg_eqf/Group.h>
#include <gtsam_unstable/tg_eqf/State.h>

#include <Eigen/Dense>

namespace tgeqf {

/// Equivariant GNSS position measurement h'(xi) = R^T(pi - p).
struct PositionMeasurement {
  static Eigen::Vector3d predict(const State& xi, const Eigen::Vector3d& pi);

  /**
   * Origin-chart Jacobian C0 in R^{3x18}
   *
   *   C0 = [ y0^  0  -R0^T  0 ],   y0 = R0^T (pi - p0)
   */
  static Eigen::Matrix<double, 3, 18> jacobian_C0(const State& xi_ref,
                                                  const Eigen::Vector3d& pi);

  /**
   * Equivariant Jacobian C*
   *
   *   C* = [ 0.5 (y0 + p_X)^  0  -R0^T  0 ]
   *
   * with y0 = R0^T (pi - p0), p_X = g.p. Preferred over C0.
   */
  static Eigen::Matrix<double, 3, 18> jacobian_Cstar(const State& xi_ref,
                                                     const TGElement& g,
                                                     const Eigen::Vector3d& pi);

  static Eigen::Vector3d innovation(const Eigen::Vector3d& pi,
                                    const State& xi_hat);

  /// Output group action psi_X(y) = R_X^T (y - p_X).
  static Eigen::Vector3d output_action(const TGElement& X,
                                       const Eigen::Vector3d& y);
};

}  // namespace tgeqf
