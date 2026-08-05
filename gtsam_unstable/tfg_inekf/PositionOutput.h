#pragma once
#include <Eigen/Dense>

#include "Group.h"

// Position output (equivariant body-frame residual).

namespace tfg {

struct PositionOutput {
  static constexpr int dim = 3;

  enum class Variant {
    C0,     ///< first-order Jacobian
    Cstar,  ///< midpoint-symmetrised (cubic linearisation error)
  };

  /// Body-frame residual h = R^T (pi - p).
  static Eigen::Vector3d predict(const TwoFrameGroup& xi,
                                 const Eigen::Vector3d& pi);

  /// Output action rho_X(y) = R_X^T (y - p_X); for equivariance checks.
  static Eigen::Vector3d output_action(const TwoFrameGroup& X,
                                       const Eigen::Vector3d& y);

  /// Output Jacobian for chart xi(eps) = xi_hat * Expmap(eps).
  /// Right-retract chart: noise-free residual y_hat = J_l(eps_theta) eps_p.
  ///   C0    = [ y_hat^      0  -I  0 ]
  ///   Cstar = [ 1/2 y_hat^  0  -I  0 ]  (halves attitude coupling ->
  ///   O(||e||^3) error)
  static Eigen::Matrix<double, 3, 15> jacobian(const TwoFrameGroup& xi_hat,
                                               const Eigen::Vector3d& pi,
                                               Variant variant = Variant::C0);
};

}  // namespace tfg
