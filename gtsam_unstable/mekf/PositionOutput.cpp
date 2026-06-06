#include <gtsam_unstable/mekf/PositionOutput.h>

// ============================================================================
//  Naive world-frame position output.
//    h(xi) = p                       predicted global position
//    H     = [ 0 0 I_3 0 0 ]         d_p block (cols 6..8)
//  No equivariant reformulation, no C*/C0 (the contrast with tfg_inekf): the
//  Jacobian is constant and the measurement noise is used directly in the
//  world frame by the filter.
// ============================================================================

namespace mekf {

Eigen::Vector3d PositionOutput::predict(const MekfState& X) { return X.p; }

Eigen::Matrix<double, 3, 15> PositionOutput::jacobian() {
    Eigen::Matrix<double, 3, 15> H = Eigen::Matrix<double, 3, 15>::Zero();
    H.block<3, 3>(0, 6) = Eigen::Matrix3d::Identity();  // d_p block
    return H;
}

} // namespace mekf
