#pragma once
#include <Eigen/Dense>

#include "Group.h"

// Biased magnetometer output (equivariant). With the hard-iron bias stored as
// gamma_m = -R b_m, h = R^T m + b_m = R^T (m - gamma_m): a known-landmark
// output with the reference field m in place of a landmark and gamma_m in
// place of the position, the same form as PositionOutput.

namespace tfg {

struct MagnetometerOutput {
  static constexpr int dim = 3;

  /// Predicted body-frame field h = R^T (m - gamma_m) = R^T m + b_m.
  static Eigen::Vector3d predict(const TwoFrameGroup& xi,
                                 const Eigen::Vector3d& m);

  /// Output action rho_X(y) = R_X^T (y - gamma_m,X); for equivariance checks.
  static Eigen::Vector3d output_action(const TwoFrameGroup& X,
                                       const Eigen::Vector3d& y);

  /// First-order output Jacobian C0 = [ y_hat^  0  0  0  0  -I ] for the
  /// chart xi(eps) = xi_hat * Expmap(eps), with y_hat = predict(xi_hat, m).
  static Eigen::Matrix<double, 3, 18> jacobian(const TwoFrameGroup& xi_hat,
                                               const Eigen::Vector3d& m);
};

}  // namespace tfg
