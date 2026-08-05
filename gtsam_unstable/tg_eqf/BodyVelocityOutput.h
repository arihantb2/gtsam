#pragma once
#include <gtsam_unstable/tg_eqf/Group.h>
#include <gtsam_unstable/tg_eqf/State.h>

#include <Eigen/Dense>

namespace tgeqf {

/// Unbiased DVL body-frame velocity h_d(xi) = R^T v.
struct DVLMeasurement {
  static Eigen::Vector3d predict(const State& xi);

  /**
   * Measurement Jacobian H_d in R^{3 x 18}.
   *
   * At general xi_ref:
   *   d(h_d)/d(delta_R) =  [R^T v]^x
   *   d(h_d)/d(delta_v) =  R^T
   *
   * At identity origin (R=I, v=0):
   *   H_d = [0_{3x3}  I_3  0_{3x3}  0_{3x9}]
   */
  static Eigen::Matrix<double, 3, 18> jacobian(const State& xi_ref);

  static Eigen::Vector3d innovation(const Eigen::Vector3d& z,
                                    const State& xi_hat);

  /// Output group action psi_d(X, y_d) = R_X^T y_d + R_X^T v_X.
  static Eigen::Vector3d output_action(const TGElement& X,
                                       const Eigen::Vector3d& y);
};

}  // namespace tgeqf
