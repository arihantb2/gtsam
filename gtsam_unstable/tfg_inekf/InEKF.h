#pragma once
#include "Group.h"
#include "Symmetry.h"
#include "PositionOutput.h"
#include <gtsam/navigation/InvariantEKF.h>
#include <Eigen/Dense>

// ============================================================================
//  TFG-IEKF : Two-Frames Group Invariant EKF for biased INS
// ----------------------------------------------------------------------------
//  Reference: Fornasier et al. (arXiv:2309.03765v3), Sec. 5.3 + App. B.7;
//             Barrau & Bonnabel [4].
//
//  Thin wrapper over gtsam::InvariantEKF<TwoFrameGroup>. The state lives on the
//  group G_TF (no separate manifold type, cf. Group.h):
//    - propagate(): builds the lifted IMU increment U = Symmetry::increment and
//      calls InvariantEKF::predict(U, Q).
//    - update_position(): feeds the equivariant body-frame residual model
//      (PositionOutput) to the EKF update.
//
//  Covariance / tangent block order = Group.h::Tangent
//    [ d_theta(3) ; d_v(3) ; d_p(3) ; d_g_omega(3) ; d_g_accel(3) ].
// ============================================================================

namespace tfg {

class TfgInEKF : public gtsam::InvariantEKF<TwoFrameGroup> {
public:
    using Base        = gtsam::InvariantEKF<TwoFrameGroup>;
    using Covariance  = Eigen::Matrix<double, 15, 15>;
    using Covariance3 = Eigen::Matrix<double, 3, 3>;

    /**
     * @param X0  initial state as a group element (use TwoFrameGroup::FromState
     *            to build it from physical R,v,p,b_omega,b_accel).
     * @param P0  initial covariance on the 15-dim tangent at X0.
     */
    TfgInEKF(const TwoFrameGroup& X0, const Covariance& P0);

    /**
     * IMU propagation. Forms ImuInput, lifts to a group increment, and runs the
     * left-invariant predict  X <- X * U,  P <- Ad_{U^-1} P Ad_{U^-1}^T + Q dt.
     *
     * @param omega  gyroscope reading (body frame, rad/s)
     * @param accel  accelerometer reading (body frame, m/s^2)
     * @param Qc     continuous-time process noise (15x15), scaled by dt
     * @param dt     time step (s)
     */
    void propagate(const Eigen::Vector3d& omega, const Eigen::Vector3d& accel,
                   const Covariance& Qc, double dt);

    /**
     * Global position update (GNSS). Uses the equivariant residual
     * h = R^T(pi - p) (PositionOutput, Lemma 15).
     *
     * @param pi          global position fix (R^3)
     * @param R_pos       measurement noise covariance (3x3)
     * @param use_cstar   if true (default), use the midpoint-symmetrised output
     *                    matrix C* (Eq. B.35, cubic linearisation error); set
     *                    false for the plain first-order C0.
     */
    void update_position(const Eigen::Vector3d& pi, const Covariance3& R_pos,
                         bool use_cstar = true);

    // --- Physical state accessors (recovered via Eq. B.31) ------------------
    gtsam::Rot3     attitude()   const;
    Eigen::Vector3d velocity()   const;
    Eigen::Vector3d position()   const;
    Eigen::Vector3d bias_gyro()  const;
    Eigen::Vector3d bias_accel() const;
};

} // namespace tfg
