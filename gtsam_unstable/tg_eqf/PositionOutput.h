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
     * Origin-chart output Jacobian C0 in R^{3x18} (first-order).
     *
     * C0 = [ y0^  0_{3x3}  -R0^T  0_{3x9} ],   y0 = R0^T (pi - p0)
     *      [ delta_R | delta_v | delta_p | delta_b ]
     *
     * This is the limit of C* (below) as the estimate reaches the origin; it
     * is built entirely from the fixed reference state xi_ref, so it is already
     * in the origin chart the filter consumes (no Dphi_g composition). At the
     * identity origin it reduces to [pi^ | 0 | -I | 0].
     *
     * Reference: Fornasier et al., arXiv:2309.03765, Sec. 2.2 / Eq. B.19.
     *
     * @param xi_ref    Fixed origin (reference) state
     * @param pi        Raw global position measurement
     * @return          Jacobian C0 in R^{3x18}
     */
    static Eigen::Matrix<double, 3, 18> jacobian_C0(
        const TGState& xi_ref,
        const Eigen::Vector3d& pi);

    /**
     * Paper-literal equivariant Jacobian C* (Eq. B.19), origin chart.
     *
     * C* = [ 0.5 (y0 + p_X)^  0_{3x3}  -R0^T  0_{3x9} ]  in R^{3x18}
     *      [ delta_R          | delta_v | delta_p | delta_b ]
     *
     * where y0  = R0^T (pi - p0)   (origin output ẙ),
     *       p_X = g.p              (back-transported measurement = R0^T(p_hat - p0)).
     *
     * Averages D_E rho_E at the origin output and at the back-transported
     * measurement (both global-frame), giving the O(||e||^3) output
     * linearisation. At the identity origin this is [0.5(pi + p_hat)^ | 0 | -I
     * | 0]; preferred over jacobian_C0.
     *
     * Reference: Fornasier et al., arXiv:2309.03765, Lemma 15 / Sec. 2.2 / Eq. B.19.
     *
     * @param xi_ref    Fixed origin (reference) state
     * @param g         Current group estimate (groupEstimate())
     * @param pi        Raw global position measurement
     * @return          C* in R^{3x18}
     */
    static Eigen::Matrix<double, 3, 18> jacobian_Cstar(
        const TGState& xi_ref,
        const TGGroupElement& g,
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
