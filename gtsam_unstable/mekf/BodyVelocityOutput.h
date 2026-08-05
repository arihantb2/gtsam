#pragma once
#include <gtsam_unstable/mekf/State.h>

#include <Eigen/Dense>

// Body-frame velocity output: h(xi) = R^T v.
// Jacobian w.r.t. tangent under R_true = R_hat * Exp(d_theta):
//   d(R^T v)/d(d_theta) = [R^T v]^x,  d(R^T v)/d(d_v) = R^T
//   H = [ [R^T v]^x | R^T | 0 | 0 | 0 ]

namespace mekf {

struct BodyVelocityOutput {
  static Eigen::Vector3d predict(const MekfState& X);
  static Eigen::Matrix<double, 3, 15> jacobian(const MekfState& X);
};

}  // namespace mekf
