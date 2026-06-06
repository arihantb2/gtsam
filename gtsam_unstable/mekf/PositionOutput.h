#pragma once
#include <gtsam_unstable/mekf/State.h>
#include <Eigen/Dense>

// ============================================================================
//  Position output : naive world-frame global position measurement
// ----------------------------------------------------------------------------
//  The deliberate contrast with the geometric filters: the MEKF uses the raw
//  world-frame residual with NO equivariant reformulation and NO C*/C0 choice.
//      h(xi) = p           (predicted position, global frame)
//      z     = pi          (measured position)
//      H     = [ 0_3 | 0_3 | I_3 | 0_3 | 0_3 ]   (the d_p block)
//  measurement noise R_pos is used directly in the world frame (NOT rotated into
//  a body-frame residual as tfg_inekf does).
// ============================================================================

namespace mekf {

struct PositionOutput {
    /// Predicted global position h(xi) = p.
    static Eigen::Vector3d predict(const MekfState& X);

    /// Measurement Jacobian H (3x15), selecting the d_p block.
    static Eigen::Matrix<double, 3, 15> jacobian();
};

} // namespace mekf
