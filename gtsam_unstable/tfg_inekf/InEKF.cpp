#include <gtsam_unstable/tfg_inekf/InEKF.h>

#include "BodyVelocityOutput.h"
#include "PositionOutput.h"

namespace tfg {

TfgInEKF::TfgInEKF(const TwoFrameGroup& X0, const Covariance& P0)
    : Base(X0, P0) {}

namespace {
ImuInput makeInput(const Eigen::Vector3d& omega, const Eigen::Vector3d& accel) {
  ImuInput u;
  u.omega = omega;
  u.accel = accel;
  return u;
}

// F = Ad_{U^-1} (I + Df dt), Df = d lift / d eps (bias->navigation coupling).
void transition(const TwoFrameGroup& X, const ImuInput& u, double dt,
                TwoFrameGroup& X_next, TfgInEKF::Covariance& F) {
  const TwoFrameGroup U = increment(X, u, dt);
  const TfgInEKF::Covariance Df = liftJacobian(X, u);
  F = (U.inverse().AdjointMap() * (TfgInEKF::Covariance::Identity() + Df * dt))
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
  this->gtsam::ManifoldEKF<TwoFrameGroup>::predict(X_next, F, Qc * dt);
}

void TfgInEKF::propagate(const Eigen::Vector3d& omega,
                         const Eigen::Vector3d& accel, const ImuNoise& noise,
                         double dt) {
  const ImuInput u = makeInput(omega, accel);
  const Covariance Qd = inputNoiseCov(this->state(), noise, dt);
  TwoFrameGroup X_next;
  Covariance F;
  transition(this->state(), u, dt, X_next, F);
  this->gtsam::ManifoldEKF<TwoFrameGroup>::predict(X_next, F, Qd);
}

void TfgInEKF::update_position(const Eigen::Vector3d& pi,
                               const Covariance3& R_pos, bool use_cstar) {
  const TwoFrameGroup& X = this->state();

  const auto variant =
      use_cstar ? PositionOutput::Variant::Cstar : PositionOutput::Variant::C0;
  const Eigen::Vector3d prediction = PositionOutput::predict(X, pi);
  const Eigen::Matrix<double, 3, 15> H =
      PositionOutput::jacobian(X, pi, variant);
  const Eigen::Vector3d z = Eigen::Vector3d::Zero();

  const Eigen::Matrix3d Rm = X.R.matrix();
  const Covariance3 R_meas = Rm.transpose() * R_pos * Rm;

  this->updateWithVector(prediction, H, z, R_meas);
}

void TfgInEKF::update_dvl(const Eigen::Vector3d& z_dvl,
                          const Covariance3& R_dvl) {
  const TwoFrameGroup& X = this->state();

  // Body-frame residual; R_dvl used directly (no R^T(.)R rotation).
  const Eigen::Vector3d prediction = BodyVelocityOutput::predict(X);
  const Eigen::Matrix<double, 3, 15> H = BodyVelocityOutput::jacobian(X);

  this->updateWithVector(prediction, H, z_dvl, R_dvl);
}

TfgInEKF::TangentVector TfgInEKF::errorStateVector(
    const TwoFrameGroup& X_true) const {
  return gtsam::traits<TwoFrameGroup>::Local(this->state(), X_true);
}

gtsam::Rot3 TfgInEKF::attitude() const { return this->state().R; }
Eigen::Vector3d TfgInEKF::velocity() const { return this->state().v; }
Eigen::Vector3d TfgInEKF::position() const { return this->state().p; }
Eigen::Vector3d TfgInEKF::bias_gyro() const {
  return this->state().bias_omega();
}
Eigen::Vector3d TfgInEKF::bias_accel() const {
  return this->state().bias_accel();
}

}  // namespace tfg
