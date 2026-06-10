#include <gtsam_unstable/tfg_inekf/InEKF.h>

// ============================================================================
//  TfgInEKF: wires the two-frames symmetry into gtsam::InvariantEKF.
//
//  Predict (Sec. 4 dynamics via the lift):
//    U = Expmap(lift(X, u) * dt)                 state-dependent increment
//    X <- X * U                                  (exact, reproduces Eq. 3)
//    P <- F P F^T + Q,   F = Ad_{U^-1} (I + Df dt)
//
//  The lift depends on the estimated biases, so the error transition carries
//  the bias->navigation coupling of the TFG-IEKF state matrix (Fornasier
//  Table 2 / Eq. B.34) through Df = d lift / d eps (Symmetry::liftJacobian).
//  Without Df the nav/bias cross-covariance would stay zero and biases would
//  be unobservable.
//
//  Update (global position, Sec. 7, Lemma 15):
//    prediction y_hat = R^T (pi - p),  target z = 0  (noise-free residual)
//    H = C0 = [y_hat^ 0 -I 0] or (use_cstar) C* = [1/2 y_hat^ 0 -I 0]
//      (midpoint-symmetrised in this right-retract chart; see PositionOutput)
//    R_meas = R^T R_pos R   (raw GNSS noise rotated into the residual frame)
//    No reset step (TFG-IEKF runs without curvature correction, Table 2).
// ============================================================================

namespace tfg {

TfgInEKF::TfgInEKF(const TwoFrameGroup& X0, const Covariance& P0)
    : Base(X0, P0) {}

namespace {
// State-dependent left-invariant predict transition. The lift Lambda depends on
// the (estimated) biases, so the error transition is NOT the bare adjoint: it
// carries the bias->navigation coupling of the TFG-IEKF error dynamics
// (Fornasier Table 2 / Eq. B.34).
//   F = Ad_{U^-1} (I + Df dt),   Df = d lift / d eps   (right chart)
// This is the standard IEKF discretisation, O(dt^2)-equivalent to the exact
// Ad_{U^-1} + J_r(lambda dt) Df dt; the Ad block is kept exact.
ImuInput makeInput(const Eigen::Vector3d& omega, const Eigen::Vector3d& accel) {
  ImuInput u;
  u.omega = omega;
  u.accel = accel;  // tau (bias random-walk drivers) default to zero.
  return u;
}

void transition(const TwoFrameGroup& X, const ImuInput& u, double dt,
                TwoFrameGroup& X_next, TfgInEKF::Covariance& F) {
  const TwoFrameGroup U = increment(X, u, dt);
  const TfgInEKF::Covariance Df = liftJacobian(X, u);
  F = (U.inverse().AdjointMap() *
       (TfgInEKF::Covariance::Identity() + Df * dt))
          .eval();
  X_next = X * U;
}
}  // namespace

void TfgInEKF::propagate(const Eigen::Vector3d& omega,
                         const Eigen::Vector3d& accel, const Covariance& Qc,
                         double dt) {
  const ImuInput u = makeInput(omega, accel);
  TwoFrameGroup X_next;
  Covariance F;
  transition(this->state(), u, dt, X_next, F);
  // Qc is continuous-time process noise in lifted tangent coordinates.
  this->gtsam::ManifoldEKF<TwoFrameGroup>::predict(X_next, F, Qc * dt);
}

void TfgInEKF::propagate(const Eigen::Vector3d& omega,
                         const Eigen::Vector3d& accel, const ImuNoise& noise,
                         double dt) {
  const ImuInput u = makeInput(omega, accel);
  // Map raw IMU/bias noise into the lifted tangent (already discrete: includes dt).
  const Covariance Qd = inputNoiseCov(this->state(), noise, dt);
  TwoFrameGroup X_next;
  Covariance F;
  transition(this->state(), u, dt, X_next, F);
  this->gtsam::ManifoldEKF<TwoFrameGroup>::predict(X_next, F, Qd);
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
