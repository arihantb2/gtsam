#pragma once
#include <gtsam_unstable/mekf/State.h>
#include <Eigen/Dense>

// ============================================================================
//  MEKF dynamics : strapdown mean propagation + state-dependent Jacobian F
// ----------------------------------------------------------------------------
//  Forward-Euler integration of the biased-INS kinematics (see Dynamics.cpp),
//  plus the transition Jacobian F. F is STATE-DEPENDENT, re-linearised about the
//  current estimate each step (the MEKF trait under study), and computed by
//  finite-difference, mirroring tg_eqf's lift-Jacobian approach.
// ============================================================================

namespace mekf {

/// IMU reading (body frame).
struct ImuInput {
    Eigen::Vector3d omega = Eigen::Vector3d::Zero();  ///< gyroscope (rad/s)
    Eigen::Vector3d accel = Eigen::Vector3d::Zero();  ///< accelerometer (m/s^2)
};

/// Global gravity vector (0, 0, -9.81), z-up (matches MakeSharedU).
Eigen::Vector3d gravity();

/// One Euler strapdown step of the nominal state (Eq. 3).
MekfState propagateMean(const MekfState& X, const ImuInput& u, double dt);

/// Transition Jacobian F (15x15): finite-difference of propagateMean in the
/// local chart at X (step 1e-7). State-dependent, recomputed every step.
Eigen::Matrix<double, 15, 15> transitionJacobian(const MekfState& X,
                                                 const ImuInput& u, double dt);

} // namespace mekf
