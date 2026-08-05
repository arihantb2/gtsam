#pragma once
#include <gtsam_unstable/mekf/State.h>

#include <Eigen/Dense>

namespace mekf {

struct ImuInput {
  Eigen::Vector3d omega = Eigen::Vector3d::Zero();
  Eigen::Vector3d accel = Eigen::Vector3d::Zero();
};

Eigen::Vector3d gravity();

MekfState propagateMean(const MekfState& X, const ImuInput& u, double dt);

/// Finite-difference transition Jacobian F (state-dependent, recomputed each
/// step).
Eigen::Matrix<double, 15, 15> transitionJacobian(const MekfState& X,
                                                 const ImuInput& u, double dt);

}  // namespace mekf
