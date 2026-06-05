#include <gtsam_unstable/tg_eqf/EqF.h>

namespace tgeqf {

TGEqF::TGEqF(const TGState& xi_ref, const Covariance18& Sigma0)
    : Base(xi_ref, Sigma0) {}

void TGEqF::propagate(const Eigen::Vector3d& omega_meas,
                      const Eigen::Vector3d& a_meas,
                      const Eigen::Vector3d& g_vec,
                      const Covariance18& Qc, double dt) {
  const TGState xi_hat = state();

  TGInput u;
  u.omega = omega_meas;
  u.a_tilde = a_meas;
  u.g_vec = g_vec;
  u.v_tilde = xi_hat.b_v;
  u.tau_omega = Eigen::Vector3d::Zero();
  u.tau_v = Eigen::Vector3d::Zero();
  u.tau_a = Eigen::Vector3d::Zero();

  const Lift lift(u);
  const InputOrbit psi_u(u);
  Base::predict(lift, psi_u, Qc, dt);
}

void TGEqF::update_dvl(const Eigen::Vector3d& z_dvl,
                       const Covariance3& R_dvl) {
  const TGState xi_hat = state();
  const Eigen::Vector3d prediction = DVLMeasurement::predict(xi_hat);
  const Eigen::Matrix<double, 3, 18> H = DVLMeasurement::jacobian(xi_hat);
  Base::updateWithVector(prediction, H, z_dvl, R_dvl);
}

void TGEqF::update_position(const Eigen::Vector3d& pi,
                            const Covariance3& R_pos, bool use_Cstar) {
  const TGState xi_hat = state();
  const Eigen::Vector3d prediction =
      PositionMeasurement::predict(xi_hat, pi);
  const Eigen::Matrix<double, 3, 18> H =
      use_Cstar ? PositionMeasurement::jacobian_Cstar(xi_hat, pi)
                : PositionMeasurement::jacobian_C0(referenceState());

  // Equivariant position target is zero: h'(xi) = 0 when pi == p.
  const Eigen::Vector3d z = Eigen::Vector3d::Zero();
  Base::updateWithVector(prediction, H, z, R_pos);
}

gtsam::Rot3 TGEqF::attitude() const { return state().R; }

Eigen::Vector3d TGEqF::position() const { return state().p; }

Eigen::Vector3d TGEqF::velocity() const { return state().v; }

Eigen::Vector3d TGEqF::bias_gyro() const { return state().b_omega; }

Eigen::Vector3d TGEqF::bias_accel() const { return state().b_a; }

Eigen::Vector3d TGEqF::bias_vel() const { return state().b_v; }

}  // namespace tgeqf
