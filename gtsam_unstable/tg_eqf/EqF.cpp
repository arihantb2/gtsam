#include <gtsam_unstable/tg_eqf/BodyVelocityOutput.h>
#include <gtsam_unstable/tg_eqf/EqF.h>
#include <gtsam_unstable/tg_eqf/Lift.h>
#include <gtsam_unstable/tg_eqf/PositionOutput.h>
#include <gtsam_unstable/tg_eqf/VirtualBiasOutput.h>

#include <iomanip>
#include <iostream>

namespace tgeqf {

namespace {

/**
 * Transport an initial covariance given in the tangent chart at the initial
 * estimate phi(X0, xi_ref) back to the EqF origin chart at xi_ref.
 *
 * The filter's Riccati state is the covariance of the origin-chart error
 * eps = Local(xi_ref, phi(X^-1, xi)). A perturbation of the estimate is
 * delta = J * eps with J = d phi(X0, .)/dxi |_{xi_ref}, so the origin-chart
 * covariance is J^-1 Sigma0 J^-T. This is the exact inverse of the transport
 * EquivariantFilter::covariance() applies in the forward direction, and it
 * reduces to the identity when X0 is the identity.
 */
Eigen::Matrix<double, 18, 18> toOriginChart(
    const State& xi_ref, const Eigen::Matrix<double, 18, 18>& Sigma0,
    const TGElement& X0) {
  Eigen::Matrix<double, 18, 18> J;
  const TGSymmetry::Diffeomorphism phi_X0(X0);
  phi_X0(xi_ref, &J);

  // J is block triangular with invertible diagonal blocks (R_X^T, I, I,
  // Ad_SE23_inv(X)), so the solve is always well posed. Sigma0 is symmetric,
  // hence (J^-1 Sigma0)^T = Sigma0 J^-T and a second solve completes the
  // congruence.
  const Eigen::PartialPivLU<Eigen::Matrix<double, 18, 18>> lu = J.partialPivLu();
  const Eigen::Matrix<double, 18, 18> half = lu.solve(Sigma0).transpose();
  const Eigen::Matrix<double, 18, 18> Sigma_eps = lu.solve(half);
  return 0.5 * (Sigma_eps + Sigma_eps.transpose());
}

}  // namespace

TGEqF::Covariance18 TGEqF::initialCovariance(
    const Covariance15& Sigma_physical) {
  Covariance18 Sigma0 = Covariance18::Zero();
  Sigma0.topLeftCorner<15, 15>() = Sigma_physical;
  Sigma0.bottomRightCorner<3, 3>() =
      kVirtualBiasInitialSigma * kVirtualBiasInitialSigma *
      Covariance3::Identity();
  return Sigma0;
}

TGEqF::TGEqF(const State& xi_ref, const Covariance18& Sigma0,
             const TGElement& X0)
    : Base(xi_ref, toOriginChart(xi_ref, Sigma0, X0), X0) {
  computeOriginCaches();
}

void TGEqF::computeOriginCaches() {
  const State& xi_ref = referenceState();

  // Cache the orbit Jacobian Dphi0 and the innovation lift pinv(Dphi0) the base
  // keeps private.
  const TGSymmetry::Orbit orbit(xi_ref);
  orbit(TGElement::Identity(), &orbit_jacobian0_);
  innovation_lift_ =
      orbit_jacobian0_.completeOrthogonalDecomposition().pseudoInverse();

  const se2_3 b_ref = {xi_ref.b_w, xi_ref.b_a, xi_ref.b_v};
  Eigen::Matrix<double, 18, 18> G = Eigen::Matrix<double, 18, 18>::Zero();
  G.block<9, 9>(0, 0).setIdentity();
  G.block<9, 9>(9, 0) = detail::ad_se23(b_ref);
  G.block<9, 9>(9, 9) = -Eigen::Matrix<double, 9, 9>::Identity();
  input_lift_ = orbit_jacobian0_ * G;
}

Eigen::Matrix<double, 18, 18> TGEqF::originChartTransport() const {
  Eigen::Matrix<double, 18, 18> Dphi_g;
  const TGSymmetry::Diffeomorphism phi_g(groupEstimate());
  phi_g(referenceState(), &Dphi_g);
  return Dphi_g;
}

Eigen::Matrix<double, 18, 18> TGEqF::resetMatrix(
    const Eigen::Matrix<double, 18, 1>& delta_xi,
    const Eigen::Matrix<double, 18, 1>& delta_x) const {
  const TGElement Xd = TGElement::Expmap(-delta_x);
  const State& xi_ref = referenceState();
  auto recentre = [&](const Eigen::Matrix<double, 18, 1>& eps) {
    return gtsam::traits<State>::Local(
        xi_ref, phi(Xd, gtsam::traits<State>::Retract(xi_ref, eps)));
  };
  const Eigen::Matrix<double, 18, 1> f0 = recentre(delta_xi);
  constexpr double h = 1e-7;
  Eigen::Matrix<double, 18, 18> J;
  for (int j = 0; j < 18; ++j) {
    Eigen::Matrix<double, 18, 1> e = Eigen::Matrix<double, 18, 1>::Zero();
    e(j) = h;
    J.col(j) = (recentre(delta_xi + e) - f0) / h;
  }
  return J;
}

void TGEqF::updateWithReset(const Eigen::VectorXd& prediction,
                            const Eigen::MatrixXd& H, const Eigen::VectorXd& z,
                            const Eigen::MatrixXd& R) {
  Eigen::Matrix<double, 18, 1> delta_xi = Eigen::Matrix<double, 18, 1>::Zero();
  Base::updateWithVector(
      prediction, H, z, R, [&](const Eigen::Matrix<double, 18, 1>& dxi) {
        delta_xi = dxi;
        return Eigen::Matrix<double, 18, 1>(innovation_lift_ * dxi);
      });

  if (reset_step_) {
    const Eigen::Matrix<double, 18, 1> delta_x = innovation_lift_ * delta_xi;
    const Eigen::Matrix<double, 18, 18> J = resetMatrix(delta_xi, delta_x);
    this->P_ = J * this->P_ * J.transpose();
  }
}

void TGEqF::propagate(const Eigen::Vector3d& w_meas,
                      const Eigen::Vector3d& a_meas,
                      const Eigen::Vector3d& g_vec, const Covariance18& Qc,
                      double dt) {
  Input u;
  u.w = w_meas;
  u.a = a_meas;
  u.g_vec = g_vec;
  // Virtual input nu = 0
  u.v = Eigen::Vector3d::Zero();
  u.tau_w = Eigen::Vector3d::Zero();
  u.tau_a = Eigen::Vector3d::Zero();
  u.tau_v = Eigen::Vector3d::Zero();

  const Lift lift(u);
  const InputOrbit psi_u(u);
  // The virtual_bias_Q_ PSD keeps a steady-state floor ~sqrt(Q*R) instead.
  Covariance18 Qc_eff = Qc;
  Qc_eff.block<3, 3>(15, 15) += virtual_bias_Q_;

  Base::predict(lift, psi_u, Qc_eff, dt);

  // Keep the unobservable virtual bias pinned at zero (Eq. B.20) if requested.
  if (anchor_virtual_bias_) {
    update_virtual_bias(virtual_bias_R_);
  }
}

void TGEqF::propagate(const Eigen::Vector3d& w_meas,
                      const Eigen::Vector3d& a_meas,
                      const Eigen::Vector3d& g_vec, const ImuNoise& noise,
                      double dt) {
  // Map the IMU noise PSDs through the lift.
  propagate(w_meas, a_meas, g_vec, inputNoiseCov(w_meas, a_meas, g_vec, noise),
            dt);
}

TGEqF::Covariance18 TGEqF::inputNoiseCov(const Eigen::Vector3d& /*w_meas*/,
                                         const Eigen::Vector3d& /*a_meas*/,
                                         const Eigen::Vector3d& /*g_vec*/,
                                         const ImuNoise& noise) const {
  const Eigen::Matrix<double, 9, 9> Ad_hat = detail::Ad_SE23(groupEstimate());
  Eigen::Matrix<double, 18, 12> AdSel = Eigen::Matrix<double, 18, 12>::Zero();
  AdSel.block<9, 6>(0, 0) = Ad_hat.leftCols<6>();
  AdSel.block<9, 6>(9, 6) = Ad_hat.leftCols<6>();
  const Eigen::Matrix<double, 18, 12> B = input_lift_ * AdSel;

  Eigen::Matrix<double, 12, 12> Sigma = Eigen::Matrix<double, 12, 12>::Zero();
  const Eigen::Matrix3d I3 = Eigen::Matrix3d::Identity();
  Sigma.block<3, 3>(0, 0) = noise.gyro * I3;
  Sigma.block<3, 3>(3, 3) = noise.accel * I3;
  Sigma.block<3, 3>(6, 6) = noise.gyro_rw * I3;
  Sigma.block<3, 3>(9, 9) = noise.accel_rw * I3;

  auto Qc = B * Sigma * B.transpose();

  // Continuous-time density; Base::predict applies the discretization dt
  // factor.
  return Qc;
}

void TGEqF::set_virtual_bias_anchor(bool enable,
                                    const std::optional<Covariance3>& R_vb) {
  anchor_virtual_bias_ = enable;
  if (R_vb) {
    virtual_bias_R_ = *R_vb;
  }
}

void TGEqF::update_dvl(const Eigen::Vector3d& z_dvl, const Covariance3& R_dvl) {
  const State xi_hat = state();
  const Eigen::Vector3d prediction = DVLMeasurement::predict(xi_hat);
  const Eigen::Matrix<double, 3, 18> H =
      DVLMeasurement::jacobian(xi_hat) * originChartTransport();
  updateWithReset(prediction, H, z_dvl, R_dvl);
}

void TGEqF::update_position(const Eigen::Vector3d& pi, const Covariance3& R_pos,
                            bool use_Cstar) {
  const State xi_ref = referenceState();
  const Eigen::Matrix3d R0 = xi_ref.R.matrix();
  const Eigen::Vector3d prediction = xi_ref.R.unrotate(pi - state().p);
  const Eigen::Matrix<double, 3, 18> H =
      use_Cstar
          ? PositionMeasurement::jacobian_Cstar(xi_ref, groupEstimate(), pi)
          : PositionMeasurement::jacobian_C0(xi_ref, pi);

  const Eigen::Vector3d z = Eigen::Vector3d::Zero();
  const Covariance3 R_eff = R0.transpose() * R_pos * R0;
  updateWithReset(prediction, H, z, R_eff);
}

void TGEqF::update_virtual_bias(const Covariance3& R_vb) {
  const State xi_hat = state();
  const Eigen::Vector3d prediction = VirtualBiasMeasurement::predict(xi_hat);
  const Eigen::Matrix<double, 3, 18> H =
      VirtualBiasMeasurement::jacobian_C0(xi_hat) * originChartTransport();

  // Constraint target b_v = 0.
  const Eigen::Vector3d z = Eigen::Vector3d::Zero();
  updateWithReset(prediction, H, z, R_vb);
}

State TGEqF::errorState(const State& xi_true) const {
  const TGElement& Xhat = groupEstimate();
  return phi(Xhat.inverse(), xi_true);
}

TGEqF::TangentVector TGEqF::errorStateVector(const State& xi_true) const {
  const State e = errorState(xi_true);
  return gtsam::traits<State>::Local(referenceState(), e);
}

gtsam::Rot3 TGEqF::attitude() const { return state().R; }

Eigen::Vector3d TGEqF::velocity() const { return state().v; }

Eigen::Vector3d TGEqF::position() const { return state().p; }

Eigen::Vector3d TGEqF::bias_gyro() const { return state().b_w; }

Eigen::Vector3d TGEqF::bias_accel() const { return state().b_a; }

Eigen::Vector3d TGEqF::bias_vel() const { return state().b_v; }

}  // namespace tgeqf
