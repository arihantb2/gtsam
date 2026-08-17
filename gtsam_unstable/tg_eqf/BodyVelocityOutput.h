#pragma once
#include <gtsam_unstable/tg_eqf/Group.h>
#include <gtsam_unstable/tg_eqf/State.h>

#include <Eigen/Dense>

namespace gtsam {
namespace tgeqf {

/// Unbiased DVL body-frame velocity h_d(xi) = R^T v.
struct DVLMeasurement {
  static Eigen::Vector3d predict(const State& xi);

  /**
   * Measurement Jacobian in R^{3 x 18}, in the Retract chart at the state it is
   * evaluated at. This is **not** error coordinates: run it through
   * EquivariantFilter::outputMatrix() before passing it to update().
   *
   *   d(h_d)/d(delta_R) =  [R^T v]^x
   *   d(h_d)/d(delta_v) =  R^T
   *
   * At an identity state (R=I, v=0):
   *   [0_{3x3}  I_3  0_{3x3}  0_{3x9}]
   */
  static Eigen::Matrix<double, 3, 18> stateJacobian(const State& xi_hat);

  static Eigen::Vector3d innovation(const Eigen::Vector3d& z,
                                    const State& xi_hat);

  /// Output group action psi_d(X, y_d) = R_X^T y_d + R_X^T v_X.
  static Eigen::Vector3d output_action(const TGElement& X,
                                       const Eigen::Vector3d& y);
};

}  // namespace tgeqf
}  // namespace gtsam
