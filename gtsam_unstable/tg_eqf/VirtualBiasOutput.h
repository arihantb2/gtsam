#pragma once
#include "State.h"
#include "Group.h"
#include <Eigen/Dense>

namespace tgeqf {

/**
 * Virtual-velocity-bias pseudo-measurement (configuration output) for TG-EqF.
 *
 * The Tangent Group adds a purely virtual velocity bias b_v to enable the full
 * se_2(3) bias symmetry. b_v has no physical meaning, so it is anchored with
 * the soft constraint h(xi) = b_v = 0 (Fornasier 2023 Eq. B.20).
 *
 * B.20 gives the output matrix in the paper's Ad_T-transported chart; the gtsam
 * EquivariantFilter consumes State::Retract-chart Jacobians (like DVL/position),
 * where b_v is a plain coordinate, so C0 = [0 0 0 0 0 I_3] in R^{3x18}.
 *
 * Reference: Fornasier et al. [1] Appendix B, Eq. (B.20).
 */
struct VirtualBiasMeasurement {

    /// Predicted virtual bias h(xi) = b_v; zero is the noise-free target.
    static Eigen::Vector3d predict(const TGState& xi);

    /// Output Jacobian C0 = [0 0 0 0 0 I_3] (Retract chart). xi_hat unused.
    static Eigen::Matrix<double, 3, 18> jacobian_C0(const TGState& xi_hat);

    /// Innovation z - h(xi_hat) with constraint target z = 0; = -b_v.
    static Eigen::Vector3d innovation(const TGState& xi_hat);
};

} // namespace tgeqf
