#pragma once
#include "State.h"
#include "Group.h"
#include <Eigen/Dense>

namespace tgeqf {

/**
 * Unbiased DVL body-frame velocity measurement for TG-EqF.
 *
 * Raw measurement y_d in R^3 (body-frame velocity from DVL).
 * Predicted output:
 *   h_d(xi) = R^T v  in R^3
 *
 * This output is equivariant under phi without reformulation:
 *   h_d(phi(X, xi)) = psi_d(X, h_d(xi))
 *
 * Output group action (proposal Eq. 15, unbiased case):
 *   psi_d(X, y_d) = R_X^T y_d + R_X^T v_X
 *
 * Reference: proposal Eq. (14), (15) (unbiased case)
 */
struct DVLMeasurement {

    /**
     * Predicted body-frame velocity at state xi.
     * h_d(xi) = R^T v
     */
    static Eigen::Vector3d predict(const TGState& xi);

    /**
     * Measurement Jacobian H_d in R^{3 x 18}
     *
     * H_d = d(h_d)/d(epsilon) |_{epsilon=0, xi=xi_ref}
     *
     * At general xi_ref:
     *   d(h_d)/d(delta_R) =  [R^T v]^x
     *   d(h_d)/d(delta_v) =  R^T
     *
     * At identity origin (R=I, v=0):
     *   H_d = [0_{3x3}  I_3  0_{3x3}  0_{3x9}]
     *
     * Layout of epsilon (18-vector):
     *   [delta_R(3), delta_v(3), delta_p(3), delta_b_omega(3),
     *    delta_b_a(3), delta_b_v(3)]
     */
    static Eigen::Matrix<double, 3, 18> jacobian(const TGState& xi_ref);

    /**
     * Innovation: z - h_d(xi_hat)
     *
     * NOTE: GTSAM update uses sign convention Local(z, prediction),
     * which corresponds to (prediction - z) in Euclidean space.
     */
    static Eigen::Vector3d innovation(
        const Eigen::Vector3d& z,
        const TGState& xi_hat);

    /**
     * Equivariant output group action psi_d : G x R^3 -> R^3
     *
     * psi_d(X, y_d) = R_X^T y_d + R_X^T v_X
     *
     * Used for equivariance verification: h_d(phi(X,xi)) == psi_d(X, h_d(xi))
     */
    static Eigen::Vector3d output_action(
        const TGGroupElement& X,
        const Eigen::Vector3d& y);
};

} // namespace tgeqf
