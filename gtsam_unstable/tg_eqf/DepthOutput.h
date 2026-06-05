#pragma once
#include "State.h"
#include "Group.h"
#include <Eigen/Dense>

namespace tgeqf {

/**
 * Pressure sensor measurement model for the TG-EqF.
 *
 * The pressure sensor directly measures depth (z-component of position):
 *   z_p = e3^T p + noise,   e3 = [0, 0, 1]^T
 *
 * This raw measurement is NOT equivariant under the TG group action phi,
 * because:
 *   hp(phi(X, xi)) = e3^T (R_X p_X + p) = e3^T R_X p_X + hp(xi)
 * which depends on the state through the extra term.
 * Reference: proposal Eq. (17), Section 6.6.
 *
 * Two-step equivariant reformulation (proposal Eq. 18, Fornasier [5]):
 *
 * Step 1 — Pseudo full-position measurement:
 *   Augment the scalar depth measurement with the x,y position components
 *   taken from the current filter estimate (not from a sensor).
 *   pi_tilde = [p_x_hat, p_y_hat, z_p]^T  in R^3
 *   NOTE: p_x_hat, p_y_hat are from the filter state, NOT from a sensor.
 *   The covariance assigned to those components must reflect this —
 *   they carry no new information and must be assigned large variance.
 *   Reference: Potokar et al. [3], proposal Eq. (18a).
 *
 * Step 2 — Body-frame residual (right-error reformulation):
 *   Transform the position residual from global frame to body frame:
 *   y'_p = h'_p(xi) = R^T (pi_tilde - p)  in R^3
 *   This IS equivariant under phi with output action:
 *   psi'_p(X, y'_p) = R_X^T (y'_p - p_X)
 *   Reference: Fornasier et al. [5] Lemma 15, proposal Eq. (18c),
 *              Eq. (34)-(35) in project knowledge (2309.03765).
 *
 * The equivariant output approximation C* (third-order linearisation error)
 * is available for this reformulation.
 * Reference: Fornasier [5], TG-EqF row of the C* table (2309.03765 / 2407.14297):
 *   C* = [0.5*(y + p_hat)^  0_{3x3}  -I_3  0_{3x9}]  in R^{3x18}
 * where ^ denotes the skew-symmetric (cross-product) matrix.
 */
struct PressureMeasurement {

    // -----------------------------------------------------------------------
    // Step 1: Raw sensor and pseudo full-position construction
    // -----------------------------------------------------------------------

    /**
     * Extract scalar depth from the raw pressure sensor measurement.
     * Converts pressure reading to depth (metres) using water density model.
     *
     * @param pressure_pa   Raw pressure measurement (Pascals)
     * @param surface_pa    Surface (atmospheric) pressure (Pascals)
     * @param rho           Water density (kg/m^3), default 1025 for seawater
     * @param g_mag         Gravity magnitude (m/s^2), default 9.81
     * @return              Depth z_p (metres, positive downward)
     *
     * NOTE: Adjust sign convention to match your frame definition.
     *       proposal uses e3^T p for depth, confirm e3 direction.
     */
    static double depth_from_pressure(
        double pressure_pa,
        double surface_pa,
        double rho = 1025.0,
        double g_mag = 9.81);

    /**
     * Construct pseudo full-position measurement pi_tilde.
     *
     * Fills x,y components from the current filter position estimate,
     * and z component from the raw depth measurement.
     *
     * IMPORTANT: The returned vector is NOT a sensor measurement for x,y.
     * The caller must assign appropriately large covariance to those components.
     *
     * @param xi_hat    Current filter state estimate
     * @param z_depth   Scalar depth from pressure sensor (metres)
     * @return          pi_tilde = [p_x_hat, p_y_hat, z_depth]^T in R^3
     *
     * Reference: proposal Eq. (18a), Potokar et al. [3]
     */
    static Eigen::Vector3d pseudo_position(
        const TGState& xi_hat,
        double z_depth);

    // -----------------------------------------------------------------------
    // Step 2: Equivariant body-frame residual measurement
    // -----------------------------------------------------------------------

    /**
     * Equivariant measurement function h'_p(xi).
     *
     * h'_p(xi) = R^T (pi_tilde - p)  in R^3
     *
     * Noise-free value is zero when pi_tilde == p (filter is correct).
     * Reference: proposal Eq. (18c), Fornasier [5] Lemma 15, Eq. (34).
     *
     * @param xi        Current filter state
     * @param pi_tilde  Pseudo full-position measurement from Step 1
     * @return          Equivariant residual measurement y'_p in R^3
     */
    static Eigen::Vector3d predict(
        const TGState& xi,
        const Eigen::Vector3d& pi_tilde);

    /**
     * Standard output Jacobian C0 (first-order, at origin xi_ref).
     *
     * C0 = d(h'_p)/d(epsilon) |_{epsilon=0, xi=xi_ref}
     *
     * For the TG-EqF with identity origin, this simplifies to:
     *   C0 = [0_{3x3}  0_{3x3}  -I_3  0_{3x9}]  in R^{3x18}
     * (block layout: [delta_R | delta_p | delta_v | delta_b])
     *       position block is -I_3.
     * Reference: TG-EqF C0 in Fornasier thesis table, proposal Section 6.6.
     *
     * @param xi_ref    Fixed origin state (not the current estimate)
     * @return          Jacobian matrix in R^{3x18}
     */
    static Eigen::Matrix<double, 3, 18> jacobian_C0(
        const TGState& xi_ref);

    /**
     * Equivariant output approximation C* (third-order linearisation error).
     *
     * C* = [0.5*(y + p_hat)^  0_{3x3}  -I_3  0_{3x9}]  in R^{3x18}
     *
     * where y   = current measurement y'_p = h'_p(xi_hat)
     *       p_hat = current position estimate from xi_hat
     *       (y + p_hat)^ = skew-symmetric matrix of (y + p_hat)
     *
     * Available because h'_p is equivariant and the local coordinates
     * at xi_ref are normal w.r.t. the group action phi.
     * Reference: Fornasier [5] Eq. (3.21)/(3.25), TG-EqF row,
     *            proposal Section 6.6 (TODO note on equivariant approx).
     *
     * USE THIS in preference to jacobian_C0 for better linearisation.
     *
     * @param xi_hat    Current filter state estimate (for y and p_hat)
     * @param pi_tilde  Pseudo full-position measurement from Step 1
     * @return          Equivariant output matrix C* in R^{3x18}
     */
    static Eigen::Matrix<double, 3, 18> jacobian_Cstar(
        const TGState& xi_hat,
        const Eigen::Vector3d& pi_tilde);

    // -----------------------------------------------------------------------
    // Measurement noise covariance
    // -----------------------------------------------------------------------

    /**
     * Build the 3x3 measurement noise covariance for the pseudo-position update.
     *
     * The z-component gets the raw sensor noise sigma_depth^2.
     * The x,y components get large assigned variance sigma_xy^2, reflecting
     * that those components came from the filter estimate, not the sensor.
     *
     * This is an engineering choice — the variance sigma_xy must be large
     * enough that the x,y pseudo-components do not pull the filter.
     * There is no principled lower bound; tuning is required.
     * Reference: Potokar et al. [3], proposal Section 3.4 (assumptions).
     *
     * @param sigma_depth  1-sigma depth measurement noise (metres)
     * @param sigma_xy     1-sigma assigned uncertainty for x,y pseudo-components
     * @return             Diagonal 3x3 covariance matrix R_p
     */
    static Eigen::Matrix3d noise_covariance(
        double sigma_depth,
        double sigma_xy);

    // -----------------------------------------------------------------------
    // Output group action (for reference / verification)
    // -----------------------------------------------------------------------

    /**
     * Equivariant output group action psi'_p : G x R^3 -> R^3
     *
     * psi'_p(X, y'_p) = R_X^T (y'_p - p_X)
     *
     * Used to verify equivariance: h'_p(phi(X, xi)) == psi'_p(X, h'_p(xi))
     * Reference: proposal Eq. (18c) derivation, Fornasier [5] Eq. (35).
     *
     * @param X     Group element
     * @param y_p   Equivariant measurement y'_p
     * @return      Output group action applied to y'_p
     */
    static Eigen::Vector3d output_action(
        const TGGroupElement& X,
        const Eigen::Vector3d& y_p);
};

} // namespace tgeqf