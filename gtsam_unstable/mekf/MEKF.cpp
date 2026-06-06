#include <gtsam_unstable/mekf/MEKF.h>

// ============================================================================
//  MultiplicativeEKF: wires the biased-INS dynamics into gtsam::ManifoldEKF.
//
//  Predict (state-dependent linearisation, the MEKF trait):
//    X_next = propagateMean(X, u, dt)             forward-Euler strapdown
//    F      = transitionJacobian(X, u, dt)        finite-diff at current X
//    P      <- F P F^T + Qc dt
//
//  Update (naive world-frame position):
//    prediction = p_hat,  z = pi,  H = [0 0 I 0 0],  R = R_pos (world frame)
//  The attitude correction enters multiplicatively through Retract
//  (R <- R Exp(d_theta)); MekfState has no Retract Jacobian, so covariance
//  transport on the correction is skipped.
// ============================================================================

namespace mekf {

MultiplicativeEKF::MultiplicativeEKF(const MekfState& X0, const Covariance& P0)
    : Base(X0, P0) {}

void MultiplicativeEKF::propagate(const Eigen::Vector3d& omega,
                                  const Eigen::Vector3d& accel,
                                  const Covariance& Qc, double dt) {
    ImuInput u;
    u.omega = omega;
    u.accel = accel;

    const MekfState X_next = propagateMean(this->state(), u, dt);
    const Jacobian F = transitionJacobian(this->state(), u, dt);
    this->predict(X_next, F, Qc * dt);  // Qc continuous-time; discretise by dt
}

void MultiplicativeEKF::update_position(const Eigen::Vector3d& pi,
                                        const Covariance3& R_pos) {
    // Naive world-frame residual: predicted h = p, measured z = pi, raw R_pos.
    const Eigen::Vector3d prediction = PositionOutput::predict(this->state());
    const Eigen::Matrix<double, 3, 15> H = PositionOutput::jacobian();
    this->updateWithVector(prediction, H, pi, R_pos);
}

gtsam::Rot3 MultiplicativeEKF::attitude() const { return this->state().R; }
Eigen::Vector3d MultiplicativeEKF::velocity() const { return this->state().v; }
Eigen::Vector3d MultiplicativeEKF::position() const { return this->state().p; }
Eigen::Vector3d MultiplicativeEKF::bias_gyro() const {
    return this->state().b_gyro;
}
Eigen::Vector3d MultiplicativeEKF::bias_accel() const {
    return this->state().b_accel;
}

} // namespace mekf
