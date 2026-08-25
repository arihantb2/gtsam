#pragma once
#include <gtsam_unstable/mekf/State.h>

#include <Eigen/Dense>

namespace mekf {

struct ImuInput {
  Eigen::Vector3d omega = Eigen::Vector3d::Zero();
  Eigen::Vector3d accel = Eigen::Vector3d::Zero();

  /// Gravity vector in the navigation frame. The filter is frame-agnostic: the
  /// caller's choice of sign is what fixes which way +z points (Z-down/NED
  /// gives +g on z, Z-up/ENU gives -g). Defaults to zero, i.e. no gravity.
  Eigen::Vector3d g_vec = Eigen::Vector3d::Zero();
};

MekfState propagateMean(const MekfState& X, const ImuInput& u, double dt);

/// Finite-difference transition Jacobian F (state-dependent, recomputed each
/// step).
Eigen::Matrix<double, 15, 15> transitionJacobian(const MekfState& X,
                                                 const ImuInput& u, double dt);

}  // namespace mekf
