#pragma once
#include <Eigen/Dense>

#include "Group.h"

// Two-Frames symmetry (action phi, lift Lambda).

namespace tfg {

/// Lifted IMU input for InvariantEKF::predict (biases are in the state).
struct ImuInput {
  Eigen::Vector3d omega;
  Eigen::Vector3d accel;
  Eigen::Vector3d tau_omega = Eigen::Vector3d::Zero();
  Eigen::Vector3d tau_accel = Eigen::Vector3d::Zero();

  /// Gravity vector in the global frame {G}. The filter is frame-agnostic: the
  /// caller's choice of sign is what fixes which way +z points (Z-down/NED
  /// gives +g on z, Z-up/ENU gives -g). Defaults to zero, i.e. no gravity.
  Eigen::Vector3d g_vec = Eigen::Vector3d::Zero();
};

/// Continuous-time IMU noise PSDs, per-axis (diagonal, not isotropic).
struct ImuNoise {
  Eigen::Vector3d gyro = Eigen::Vector3d::Zero();
  Eigen::Vector3d accel = Eigen::Vector3d::Zero();
  Eigen::Vector3d gyro_rw = Eigen::Vector3d::Zero();
  Eigen::Vector3d accel_rw = Eigen::Vector3d::Zero();
};

/// Right group action phi(X, xi) = xi * X (state space M = G_TF).
TwoFrameGroup phi(const TwoFrameGroup& X, const TwoFrameGroup& xi);

/// Equivariant lift Lambda(X, u).
Tangent lift(const TwoFrameGroup& X, const ImuInput& u);

/// Df = d lift(X * Expmap(eps), u) / d eps at eps = 0 (bias->navigation
/// coupling).
Eigen::Matrix<double, 15, 15> liftJacobian(const TwoFrameGroup& X,
                                           const ImuInput& u);

/// Qd = B diag(sigma^2) B^T dt; B maps (n_omega, n_accel, tau_w, tau_a) into
/// lift. Non-zero blocks: B[theta,n_w]=I, B[v,n_a]=I, B[g_w,n_w]=b_w^,
/// B[g_w,tau_w]=-I, B[g_a,n_w]=b_a^, B[g_a,tau_a]=-I.
Eigen::Matrix<double, 15, 15> inputNoiseCov(const TwoFrameGroup& X,
                                            const ImuNoise& noise, double dt);

/// Group increment U = Expmap(lift(X,u) * dt) for InvariantEKF::predict.
TwoFrameGroup increment(const TwoFrameGroup& X, const ImuInput& u, double dt);

}  // namespace tfg
