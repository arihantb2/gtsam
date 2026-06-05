#pragma once
#include "State.h"
#include "Group.h"
#include <Eigen/Dense>

namespace tgeqf {

/**
 * DVL measurement model (unbiased body-frame velocity)
 *
 * h_d(xi) = R^T v  in R^3
 *
 * Equivariant output action (for reference):
 *   psi_d(X, y_d) = R_X^T y_d + R_X^T v_X
 *
 * Reference: proposal Eq. (14), (15)
 */
struct DVLMeasurement {

    /**
     * Predicted measurement at state xi.
     * h_d(xi) = R^T v
     */
    static Eigen::Vector3d predict(const TGState& xi);

    /**
     * Measurement Jacobian H_d in R^{3 x 18}
     *
     * H_d = d(h_d)/d(epsilon) |_{epsilon=0, xi=xi_ref}
     *
     * Evaluated at the fixed origin xi_ref (not the current estimate).
     * If origin is chosen as identity, this matrix is constant.
     *
     * Layout of epsilon (18-vector):
     *   [delta_R(3), delta_p(3), delta_v(3), delta_b_omega(3),
     *    delta_b_v(3), delta_b_a(3)]
     */
    static Eigen::Matrix<double, 3, 18> jacobian(const TGState& xi_ref);

    /**
     * Innovation: difference between observed and predicted measurement.
     * innovation = z - h_d(xi_hat)  in R^3
     *
     * NOTE: GTSAM update uses sign convention Local(z, prediction),
     * which corresponds to (prediction - z) in Euclidean space.
     * Verify sign convention against EquivariantFilter::update.
     */
    static Eigen::Vector3d innovation(
        const Eigen::Vector3d& z,
        const TGState& xi_hat);
};

} // namespace tgeqf