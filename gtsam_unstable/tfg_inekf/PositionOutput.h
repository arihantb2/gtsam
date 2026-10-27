#pragma once
#include <Eigen/Dense>

#include "Group.h"

// Position output (equivariant body-frame residual).

namespace tfg {

struct PositionOutput {
  static constexpr int dim = 3;

  /// Body-frame residual h = R^T (pi - p).
  static Eigen::Vector3d predict(const TwoFrameGroup& xi,
                                 const Eigen::Vector3d& pi);

  /// Output action rho_X(y) = R_X^T (y - p_X); for equivariance checks.
  static Eigen::Vector3d output_action(const TwoFrameGroup& X,
                                       const Eigen::Vector3d& y);

  /// First-order output Jacobian C0 = [ y_hat^  0  -I  0  0  0 ] for the
  /// chart xi(eps) = xi_hat * Expmap(eps).
  static Eigen::Matrix<double, 3, 18> jacobian(const TwoFrameGroup& xi_hat,
                                               const Eigen::Vector3d& pi);
};

}  // namespace tfg
