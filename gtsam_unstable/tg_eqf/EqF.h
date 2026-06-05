#pragma once
#include "State.h"
#include "Symmetry.h"
#include "Lift.h"
#include "BodyVelocityOutput.h"
#include <gtsam/navigation/EquivariantFilter.h>
#include <Eigen/Dense>

namespace tgeqf {

/**
 * TG-EqF for biased INS with unbiased DVL velocity measurement.
 *
 * Thin wrapper over gtsam::EquivariantFilter<TGState, TGSymmetry>
 * providing INS-specific predict() and update() entry points.
 *
 * State: xi = [R, p, v, b_omega, b_v, b_a] in SE_2(3) x R^9
 * Group: G  = SE_2(3) ⋉ se_2(3)  (Tangent Group of SE_2(3))
 *
 * Reference: Fornasier et al. [1], van Goor et al. [6]
 */
class TGEqF : public gtsam::EquivariantFilter<TGState, TGSymmetry> {

    using Base = gtsam::EquivariantFilter<TGState, TGSymmetry>;
    using Covariance18 = Eigen::Matrix<double, 18, 18>;
    using Covariance3  = Eigen::Matrix<double, 3, 3>;

public:

    /**
     * Construct the filter.
     *
     * @param xi_ref   Fixed origin state (choose identity for simplicity;
     *                 must have gravity direction fixed if using depth).
     * @param Sigma0   Initial covariance on the 18-dim tangent space at xi_ref.
     */
    explicit TGEqF(
        const TGState& xi_ref,
        const Covariance18& Sigma0);

    /**
     * IMU propagation step.
     *
     * Constructs TGInput from raw IMU measurements, forms Lift and InputOrbit
     * functors, and calls Base::predict.
     *
     * @param omega_meas   Raw gyroscope measurement in body frame (rad/s)
     * @param a_meas       Raw accelerometer measurement in body frame (m/s^2)
     * @param g_vec        Gravity vector in global frame (m/s^2)
     * @param Qc           Continuous-time process noise on the 18-dim manifold
     * @param dt           Time step (s)
     */
    void propagate(
        const Eigen::Vector3d& omega_meas,
        const Eigen::Vector3d& a_meas,
        const Eigen::Vector3d& g_vec,
        const Covariance18& Qc,
        double dt);

    /**
     * DVL measurement update (unbiased body-frame velocity).
     *
     * Calls DVLMeasurement::predict and DVLMeasurement::jacobian,
     * then Base::updateWithVector.
     *
     * @param z_dvl   DVL body-frame velocity measurement (m/s)
     * @param R_dvl   DVL measurement noise covariance (3x3)
     */
    void update_dvl(
        const Eigen::Vector3d& z_dvl,
        const Covariance3& R_dvl);

    // -----------------------------------------------------------------------
    // State accessors
    // -----------------------------------------------------------------------

    gtsam::Rot3     attitude()  const;
    Eigen::Vector3d position()  const;
    Eigen::Vector3d velocity()  const;
    Eigen::Vector3d bias_gyro() const;
    Eigen::Vector3d bias_accel() const;
    /// Virtual velocity bias (internal TG state; not directly observable)
    Eigen::Vector3d bias_vel()  const;
};

} // namespace tgeqf