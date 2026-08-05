#pragma once
#include <Eigen/Dense>

#include "Group.h"

// DVL body-frame velocity output (natively equivariant).

namespace tfg {

struct BodyVelocityOutput {
  static constexpr int dim = 3;

  /// Predicted body-frame velocity h_d = R^T v.
  static Eigen::Vector3d predict(const TwoFrameGroup& xi);

  /// Output action psi_d(X, y) = R_X^T y + R_X^T v_X; for equivariance checks.
  static Eigen::Vector3d output_action(const TwoFrameGroup& X,
                                       const Eigen::Vector3d& y);

  /// Output Jacobian H = [ y_hat^  I  0  0 ] in the right-retract chart.
  /// d_v block is identity (velocity perturbed in rotated frame, not R^T).
  static Eigen::Matrix<double, 3, 15> jacobian(const TwoFrameGroup& xi_hat);
};

}  // namespace tfg
