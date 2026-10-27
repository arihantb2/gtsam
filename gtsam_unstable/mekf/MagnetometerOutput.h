#pragma once
#include <gtsam_unstable/mekf/State.h>

#include <Eigen/Dense>

// Biased magnetometer output: h(xi) = R^T m + b_mag, with m the reference
// field in the navigation frame and b_mag the body-frame hard-iron bias.
// Jacobian w.r.t. tangent under R_true = R_hat * Exp(d_theta):
//   d(R^T m)/d(d_theta) = [R^T m]^x,  dh/d(d_b_mag) = I
//   H = [ [R^T m]^x | 0 | 0 | 0 | 0 | I ]

namespace mekf {

struct MagnetometerOutput {
  static Eigen::Vector3d predict(const MekfState& X, const Eigen::Vector3d& m);
  static Eigen::Matrix<double, 3, 18> jacobian(const MekfState& X,
                                               const Eigen::Vector3d& m);
};

}  // namespace mekf
