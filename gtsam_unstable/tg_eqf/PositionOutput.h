#pragma once
#include "State.h"
#include "Group.h"
#include <Eigen/Dense>

namespace tgeqf {

/**
 * Equivariant global position measurement for TG-EqF.
 *
 * Directly reproduces the TG-EqF filter from Fornasier et al. [5] Sec. 7
 * using a reformulated body-frame residual measurement (Lemma 15).
 *
 * Raw position measurement (GNSS): pi in R^3
 * NOT directly equivariant under phi: h(phi(X,xi)) = R*p_X + p (state-dependent)
 *
 * Equivariant reformulation (Fornasier [5] Lemma 15, Eq. 34-35):
 *   h'(xi) = R^T (pi - p)  in R^3
 *
 * Output group action:
 *   psi_X(y) = R_X^T (y - p_X)
 *
 * Verification: h'(phi(X,xi)) = (R*R_X)^T(pi - (R*p_X + p))
 *                              = R_X^T (R^T(pi - p) - p_X)
 *                              = R_X^T (h'(xi) - p_X)
 *                              = psi_X(h'(xi))  ✓
 */
struct PositionMeasurement {

    /**
     * Equivariant measurement function h'(xi).
     *
     * h'(xi) = R^T (pi - p)  in R^3
     *
     * Noise-free value is zero when pi == p (filter correct).
     *
     * @param xi    Current filter state
     * @param pi    Raw global position measurement (e.g. GNSS fix)
     * @return      Equivariant residual y' in R^3
     */
    static Eigen::Vector3d predict(
        const TGState& xi,
        const Eigen::Vector3d& pi);

    /**
     * Standard output Jacobian C0 in R^{3x18} (first-order, at origin xi_ref).
     *
     * C0 = d(h')/d(epsilon) |_{epsilon=0, xi=xi_ref}
     *
     * At identity origin (R=I, p=0):
     *   d(h')/d(delta_R) = 0   (pi^∧ term vanishes at origin)
     *   d(h')/d(delta_p) = -I
     *   d(h')/d(delta_v) = 0
     *
     * C0 = [0_{3x3}  0_{3x3}  -I_3  0_{3x9}]  in R^{3x18}
     *      [delta_R | delta_v | delta_p | delta_b]
     *
     * Reference: Fornasier [5] Sec. 7.1, Fornasier 2023 (2309.03765)
     *
     * @param xi_ref    Fixed origin state
     * @return          Jacobian C0 in R^{3x18}
     */
    static Eigen::Matrix<double, 3, 18> jacobian_C0(
        const TGState& xi_ref);

    /**
     * Equivariant output approximation C* (third-order linearisation error).
     *
     * C* = [0.5*(y + p_hat)^  0_{3x3}  -I_3  0_{3x9}]  in R^{3x18}
     *      [delta_R           | delta_v | delta_p | delta_b]
     *
     * where y     = h'(xi_hat) = R_hat^T(pi - p_hat)  (predicted measurement)
     *       p_hat = current position estimate
     *       (y + p_hat)^ = skew-symmetric matrix of (y + p_hat)
     *
     * Reduces linearisation error from O(||e||^2) to O(||e||^3).
     * Use in preference to jacobian_C0.
     *
     * Reference: Fornasier [5] Lemma 15, Fornasier 2023 (2309.03765) Sec. 7.1
     *
     * @param xi_hat    Current filter state estimate
     * @param pi        Raw global position measurement
     * @return          C* in R^{3x18}
     */
    static Eigen::Matrix<double, 3, 18> jacobian_Cstar(
        const TGState& xi_hat,
        const Eigen::Vector3d& pi);

    /**
     * Innovation: z - h'(xi_hat)
     *
     * NOTE: verify sign convention against EquivariantFilter::update.
     *
     * @param pi        Raw global position measurement
     * @param xi_hat    Current filter state estimate
     */
    static Eigen::Vector3d innovation(
        const Eigen::Vector3d& pi,
        const TGState& xi_hat);

    /**
     * Equivariant output group action psi_X : G x R^3 -> R^3
     *
     * psi_X(y) = R_X^T (y - p_X)
     *
     * Used for equivariance verification: h'(phi(X,xi)) == psi_X(h'(xi))
     *
     * @param X     Group element
     * @param y     Equivariant measurement y' = h'(xi)
     * @return      psi_X(y)
     */
    static Eigen::Vector3d output_action(
        const TGGroupElement& X,
        const Eigen::Vector3d& y);
};

} // namespace tgeqf
