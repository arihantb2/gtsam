#include <gtsam_unstable/tfg_inekf/InEKF.h>

// ============================================================================
//  TfgInEKF: wires the two-frames symmetry into gtsam::InvariantEKF.
//
//  Predict (Sec. 4 dynamics via the lift):
//    U = Expmap(lift(X, u) * dt)                 state-dependent increment
//    X <- X * U                                  (exact, reproduces Eq. 3)
//    P <- Ad_{U^-1} P Ad_{U^-1}^T + Q dt         left-invariant propagation
//
//  Note: the covariance uses A = Ad_{U^-1}. For the two-frames group the
//  navigation error is autonomous (exact), while the bias-error coupling of the
//  exact TFG-IEKF state matrix (Fornasier B.34) is approximated here. This is
//  the standard InvariantEKF propagation; the explicit B.34 coupling is a
//  possible refinement.
//
//  Update (global position, Sec. 7, Lemma 15):
//    prediction y_hat = R^T (pi - p),  target z = 0  (noise-free residual)
//    H = C0 or (use_cstar) the midpoint-symmetrised C* (Eq. B.35)
//    R_meas = R^T R_pos R   (raw GNSS noise rotated into the residual frame)
//    No reset step (TFG-IEKF runs without curvature correction, Table 2).
// ============================================================================

namespace tfg {

TfgInEKF::TfgInEKF(const TwoFrameGroup& X0, const Covariance& P0)
    : Base(X0, P0) {}

void TfgInEKF::propagate(const Eigen::Vector3d& omega,
                         const Eigen::Vector3d& accel, const Covariance& Qc,
                         double dt) {
  ImuInput u;
  u.omega = omega;
  u.accel = accel;
  // tau (bias random-walk drivers) default to zero in ImuInput.

  const TwoFrameGroup U = increment(this->state(), u, dt);
  this->predict(U, Qc * dt);  // Q interpreted as continuous-time; discretise
}

void TfgInEKF::update_position(const Eigen::Vector3d& pi,
                               const Covariance3& R_pos, bool use_cstar) {
  const TwoFrameGroup& X = this->state();

  const auto variant = use_cstar ? PositionOutput::Variant::Cstar
                                 : PositionOutput::Variant::C0;
  const Eigen::Vector3d prediction = PositionOutput::predict(X, pi);  // y_hat
  const Eigen::Matrix<double, 3, 15> H = PositionOutput::jacobian(X, pi, variant);
  const Eigen::Vector3d z = Eigen::Vector3d::Zero();  // residual target

  // Rotate raw measurement noise into the body-residual frame: R^T R_pos R.
  const Eigen::Matrix3d Rm = X.R.matrix();
  const Covariance3 R_meas = Rm.transpose() * R_pos * Rm;

  this->updateWithVector(prediction, H, z, R_meas, /*performReset=*/false);
}

// --- Physical state accessors (Eq. B.31) ------------------------------------

gtsam::Rot3 TfgInEKF::attitude() const { return this->state().R; }
Eigen::Vector3d TfgInEKF::velocity() const { return this->state().v; }
Eigen::Vector3d TfgInEKF::position() const { return this->state().p; }
Eigen::Vector3d TfgInEKF::bias_gyro() const { return this->state().bias_omega(); }
Eigen::Vector3d TfgInEKF::bias_accel() const {
  return this->state().bias_accel();
}

}  // namespace tfg
