#include <gtsam_unstable/mekf/MEKF.h>

// Attitude correction via Retract; MekfState has no Retract Jacobian, so
// covariance transport on the correction is skipped.

namespace mekf {

MultiplicativeEKF::MultiplicativeEKF(const MekfState& X0, const Covariance& P0)
    : Base(X0, P0) {}

void MultiplicativeEKF::propagate(const Eigen::Vector3d& omega,
                                  const Eigen::Vector3d& accel,
                                  const Eigen::Vector3d& g_vec,
                                  const Covariance& Qc, double dt) {
  ImuInput u;
  u.omega = omega;
  u.accel = accel;
  u.g_vec = g_vec;

  const MekfState X_next = propagateMean(this->state(), u, dt);
  const Jacobian F = transitionJacobian(this->state(), u, dt);
  this->predict(X_next, F, Qc * dt);
}

void MultiplicativeEKF::propagate(const Eigen::Vector3d& omega,
                                  const Eigen::Vector3d& accel,
                                  const Eigen::Vector3d& g_vec,
                                  const ImuNoise& noise, double dt) {
  const Eigen::Matrix3d I3 = Eigen::Matrix3d::Identity();
  Covariance Qc = Covariance::Zero();
  if (noise.gyro > 0.0) {
    Qc.block<3, 3>(0, 0) = noise.gyro * I3;
  }
  if (noise.accel > 0.0) {
    Qc.block<3, 3>(3, 3) = noise.accel * I3;
  }
  if (noise.gyro_rw > 0.0) {
    Qc.block<3, 3>(9, 9) = noise.gyro_rw * I3;
  }
  if (noise.accel_rw > 0.0) {
    Qc.block<3, 3>(12, 12) = noise.accel_rw * I3;
  }
  propagate(omega, accel, g_vec, Qc, dt);
}

void MultiplicativeEKF::update_position(const Eigen::Vector3d& pi,
                                        const Covariance3& R_pos) {
  const Eigen::Vector3d prediction = PositionOutput::predict(this->state());
  const Eigen::Matrix<double, 3, 15> H = PositionOutput::jacobian();
  this->updateWithVector(prediction, H, pi, R_pos);
}

void MultiplicativeEKF::update_dvl(const Eigen::Vector3d& z_dvl,
                                   const Covariance3& R_dvl) {
  const Eigen::Vector3d prediction = BodyVelocityOutput::predict(this->state());
  const Eigen::Matrix<double, 3, 15> H =
      BodyVelocityOutput::jacobian(this->state());
  this->updateWithVector(prediction, H, z_dvl, R_dvl);
}

void MultiplicativeEKF::update_depth(double z_depth, const Covariance1& R_depth,
                                     double horizontal_variance) {
  Eigen::Vector3d pseudo_position = position();
  pseudo_position.z() = z_depth;

  Covariance3 R_pseudo = Covariance3::Zero();
  R_pseudo(0, 0) = horizontal_variance;
  R_pseudo(1, 1) = horizontal_variance;
  R_pseudo(2, 2) = R_depth(0, 0);

  update_position(pseudo_position, R_pseudo);
}

MultiplicativeEKF::TangentVector MultiplicativeEKF::errorStateVector(
    const MekfState& X_true) const {
  return gtsam::traits<MekfState>::Local(this->state(), X_true);
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

}  // namespace mekf
