#pragma once
#include <gtsam_unstable/mekf/State.h>
#include <gtsam_unstable/mekf/Dynamics.h>
#include <gtsam_unstable/mekf/PositionOutput.h>
#include <gtsam/navigation/ManifoldEKF.h>
#include <Eigen/Dense>

// ============================================================================
//  MultiplicativeEKF : baseline MEKF for biased INS with position output
// ----------------------------------------------------------------------------
//  Thin wrapper over gtsam::ManifoldEKF<MekfState>. Baseline for comparison
//  against the geometric filters (tg_eqf TG-EqF, tfg_inekf TFG-IEKF).
//
//  Distinguishing traits (the contrast the comparison study exposes):
//    - propagate(): state-DEPENDENT transition Jacobian F (finite-difference,
//      re-linearised at the current estimate each step).
//    - update_position(): naive world-frame residual h=p, raw R_pos (no
//      equivariant reformulation). Attitude correction enters via Retract.
//
//  Covariance / tangent block order = State.h
//    [ d_theta(3) ; d_v(3) ; d_p(3) ; d_b_gyro(3) ; d_b_accel(3) ].
// ============================================================================

namespace mekf {

class MultiplicativeEKF : public gtsam::ManifoldEKF<MekfState> {
public:
    using Base        = gtsam::ManifoldEKF<MekfState>;
    using Covariance  = Eigen::Matrix<double, 15, 15>;
    using Covariance3 = Eigen::Matrix<double, 3, 3>;

    /**
     * @param X0  initial physical state (R, v, p, b_gyro, b_accel).
     * @param P0  initial covariance on the 15-dim tangent at X0.
     */
    MultiplicativeEKF(const MekfState& X0, const Covariance& P0);

    /**
     * IMU propagation. Strapdown mean step + finite-difference F:
     *   X <- propagateMean(X, u, dt),  P <- F P F^T + Qc dt.
     *
     * @param omega  gyroscope reading (body frame, rad/s)
     * @param accel  accelerometer reading (body frame, m/s^2)
     * @param Qc     continuous-time process noise (15x15), scaled by dt
     * @param dt     time step (s)
     */
    void propagate(const Eigen::Vector3d& omega, const Eigen::Vector3d& accel,
                   const Covariance& Qc, double dt);

    /**
     * Global position update (GNSS). Naive world-frame residual h=p, z=pi,
     * H=[0 0 I 0 0], with R_pos used directly in the world frame.
     *
     * @param pi     global position fix (R^3)
     * @param R_pos  measurement noise covariance (3x3, world frame)
     */
    void update_position(const Eigen::Vector3d& pi, const Covariance3& R_pos);

    // --- Physical state accessors -------------------------------------------
    gtsam::Rot3     attitude()   const;
    Eigen::Vector3d velocity()   const;
    Eigen::Vector3d position()   const;
    Eigen::Vector3d bias_gyro()  const;
    Eigen::Vector3d bias_accel() const;
};

} // namespace mekf
