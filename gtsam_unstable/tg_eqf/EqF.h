#pragma once
#include "State.h"
#include "Symmetry.h"
#include "Lift.h"
#include "BodyVelocityOutput.h"
#include "PositionOutput.h"
#include "VirtualBiasOutput.h"
#include <gtsam/navigation/EquivariantFilter.h>
#include <Eigen/Dense>

namespace tgeqf {

/**
 * TG-EqF for biased INS with unbiased DVL velocity measurement.
 *
 * Thin wrapper over gtsam::EquivariantFilter<TGState, TGSymmetry>
 * providing INS-specific predict() and update() entry points.
 *
 * State: xi = [R, v, p, b_omega, b_a, b_v] in SE_2(3) x R^9
 * Group: G  = SE_2(3) ⋉ se_2(3)  (Tangent Group of SE_2(3))
 *
 * Reference: Fornasier et al. [1], van Goor et al. [6]
 */
class TGEqF : public gtsam::EquivariantFilter<TGState, TGSymmetry> {

public:
    using Base = gtsam::EquivariantFilter<TGState, TGSymmetry>;
    using Covariance18 = Eigen::Matrix<double, 18, 18>;
    using Covariance3  = Eigen::Matrix<double, 3, 3>;

    /**
     * Construct the filter.
     *
     * @param xi_ref   Fixed origin state (choose identity for simplicity).
     * @param Sigma0   Initial covariance on the 18-dim tangent space at xi_ref.
     * @param X0       Initial group estimate (default identity). The recovered
     *                 manifold state is phi(X0, xi_ref); pass a non-identity X0
     *                 to start the filter at a state away from the origin.
     */
    explicit TGEqF(
        const TGState& xi_ref,
        const Covariance18& Sigma0,
        const TGGroupElement& X0 = TGGroupElement::Identity());

    /**
     * Enable/disable automatic virtual-bias anchoring inside propagate().
     *
     * When enabled, every propagate() call applies the b_v = 0 pseudo-measurement
     * (see update_virtual_bias) right after the predict, so the unobservable
     * virtual bias stays pinned without the caller doing it manually. Disabled by
     * default to keep propagate() a pure predict step.
     *
     * @param enable  Turn auto-anchoring on/off
     * @param R_vb    Pseudo-measurement noise to use (small => tighter anchor)
     */
    void set_virtual_bias_anchor(
        bool enable,
        const Covariance3& R_vb = 1e-4 * Covariance3::Identity());

    /**
     * Enable/disable the EqF covariance reset after each update.
     *
     * Paper Sec. 2.2 lists the EqF as Predict / Update / Reset. The reset
     * re-centres the covariance after the state correction so the (fixed) origin
     * chart stays a good linearisation point as the estimate moves away from
     * xi_ref — essential for consistency far from the origin (Sec. 7.2 reports
     * ANEES ~ 1 with the reset). Enabled by default; disable for a pure
     * Predict/Update filter (e.g. to compare against the no-reset behaviour).
     */
    void set_reset_step(bool enable) { reset_step_ = enable; }

    /**
     * Reset transport matrix J(delta_xi, delta_x): the linearisation of the
     * post-update error re-centring eps+ = vartheta(phi(Exp(-delta_x),
     * vartheta^{-1}(eps))) at eps = delta_xi. The reset conjugates the
     * covariance as P <- J P J^T. Exposed for testing; J -> I as the
     * correction -> 0.
     */
    Eigen::Matrix<double, 18, 18> resetMatrix(
        const Eigen::Matrix<double, 18, 1>& delta_xi,
        const Eigen::Matrix<double, 18, 1>& delta_x) const;

    /**
     * IMU propagation step.
     *
     * Constructs TGInput from raw IMU measurements, forms Lift and InputOrbit
     * functors, and calls Base::predict. If auto-anchoring is enabled
     * (set_virtual_bias_anchor), also applies the b_v = 0 constraint afterward.
     *
     * @param w_meas       Raw gyroscope measurement in body frame (rad/s)
     * @param a_meas       Raw accelerometer measurement in body frame (m/s^2)
     * @param g_vec        Gravity vector in global frame (m/s^2)
     * @param Qc           Continuous-time process noise on the 18-dim manifold.
     *                     Known approximation: Qc is injected directly in
     *                     origin-chart state coordinates (P += Qc*dt); there is
     *                     no input matrix B_t = Dphi0 * D_u(Lambda) mapping IMU
     *                     noise through the lift (which would also couple IMU
     *                     noise into the bias rows via Lambda_2). Acceptable for
     *                     small biases; see docs/design.md "Known approximations".
     * @param dt           Time step (s)
     */
    void propagate(
        const Eigen::Vector3d& w_meas,
        const Eigen::Vector3d& a_meas,
        const Eigen::Vector3d& g_vec,
        const Covariance18& Qc,
        double dt);

    /**
     * DVL measurement update (unbiased body-frame velocity).
     *
     * @param z_dvl   DVL body-frame velocity measurement (m/s)
     * @param R_dvl   DVL measurement noise covariance (3x3)
     */
    void update_dvl(
        const Eigen::Vector3d& z_dvl,
        const Covariance3& R_dvl);

    /**
     * Global position measurement update (e.g. GNSS).
     *
     * Uses equivariant reformulation h'(xi) = R^T(pi - p) from
     * Fornasier [5] Lemma 15. Reproduces TG-EqF from Fornasier 2023
     * (2309.03765) when used with IMU propagation.
     *
     * Calls PositionMeasurement::jacobian_Cstar (preferred) or
     * jacobian_C0, then Base::updateWithVector.
     *
     * @param pi      Global position measurement in R^3 (e.g. GNSS fix)
     * @param R_pos   Position measurement noise covariance (3x3)
     * @param use_Cstar  Use equivariant C* Jacobian (default true)
     */
    void update_position(
        const Eigen::Vector3d& pi,
        const Covariance3& R_pos,
        bool use_Cstar = true);

    /**
     * Virtual-velocity-bias pseudo-measurement update (configuration output).
     *
     * Imposes the soft constraint b_v = 0 (Fornasier 2023 Eq. B.20), anchoring
     * the otherwise unobservable virtual bias at its operating point. Use a
     * small R_vb to enforce the constraint more tightly.
     *
     * @param R_vb   Pseudo-measurement noise covariance (3x3)
     */
    void update_virtual_bias(
        const Covariance3& R_vb);

    // -----------------------------------------------------------------------
    // State accessors
    // -----------------------------------------------------------------------

    gtsam::Rot3     attitude()  const;
    Eigen::Vector3d velocity()  const;
    Eigen::Vector3d position()  const;
    Eigen::Vector3d bias_gyro() const;
    Eigen::Vector3d bias_accel() const;
    /// Virtual velocity bias (internal TG state; not directly observable)
    Eigen::Vector3d bias_vel()  const;

private:
    /**
     * 18x18 differential of phi_g at the origin: Dphi_g = d phi(g, xi)/dxi
     * evaluated at (groupEstimate(), referenceState()).
     *
     * Output Jacobians are derived in the chart at the current estimate, but
     * the filter's covariance / innovation lift live in the fixed origin chart.
     * Composing a measurement Jacobian H_est with this transport (chain rule)
     * gives the origin-chart H_origin = H_est * Dphi_g the filter consumes
     * (CODE_REVIEW F1). At g = identity this is the 18x18 identity.
     */
    Eigen::Matrix<double, 18, 18> originChartTransport() const;

    /**
     * Run a vector measurement update, then (if enabled) the covariance reset.
     * Captures the manifold correction delta_xi via the base filter's custom
     * innovation-lift hook so the reset transport can be formed from it.
     */
    void updateWithReset(const Eigen::VectorXd& prediction,
                         const Eigen::MatrixXd& H, const Eigen::VectorXd& z,
                         const Eigen::MatrixXd& R);

    /// Auto-anchor b_v = 0 inside propagate() (off by default).
    bool anchor_virtual_bias_ = false;
    /// Pseudo-measurement noise used by the auto-anchor.
    Covariance3 virtual_bias_R_ = 1e-4 * Covariance3::Identity();
    /// Apply the EqF covariance reset after each update (on by default).
    bool reset_step_ = true;
    /// Innovation lift pinv(Dphi0) cached from xi_ref (base keeps it private).
    Eigen::Matrix<double, 18, 18> innovation_lift_;
};

} // namespace tgeqf