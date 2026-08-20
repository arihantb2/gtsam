#include <gtsam_unstable/tg_eqf/BodyVelocityOutput.h>
#include <gtsam_unstable/tg_eqf/EqF.h>
#include <gtsam_unstable/tg_eqf/Lift.h>
#include <gtsam_unstable/tg_eqf/PositionOutput.h>
#include <gtsam_unstable/tg_eqf/VirtualBiasOutput.h>

namespace gtsam {
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

  // J = Ad_{X0^-1} is always invertible; Sigma0 symmetric lets the second
  // solve complete the congruence.
  const Eigen::PartialPivLU<Eigen::Matrix<double, 18, 18>> lu =
      J.partialPivLu();
  const Eigen::Matrix<double, 18, 18> half = lu.solve(Sigma0).transpose();
  const Eigen::Matrix<double, 18, 18> Sigma_eps = lu.solve(half);
  return 0.5 * (Sigma_eps + Sigma_eps.transpose());
}

}  // namespace

TGEqF::Covariance18 TGEqF::initialCovariance(
    const Covariance15& Sigma_physical) {
  Covariance18 Sigma0 = Covariance18::Zero();
  Sigma0.topLeftCorner<15, 15>() = Sigma_physical;
  Sigma0.bottomRightCorner<3, 3>() = kVirtualBiasInitialSigma *
                                     kVirtualBiasInitialSigma *
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

Eigen::Matrix<double, 18, 18> TGEqF::resetMatrix(
    const Eigen::Matrix<double, 18, 1>& delta_xi,
    const Eigen::Matrix<double, 18, 1>& delta_x) const {
  // r(eps) = Local(xi_ref, phi(Xd, Retract(xi_ref, eps))) reduces to
  // Logmap(Expmap(eps) . Xd), whose derivative has a closed form in terms of
  // the group's own left/right Jacobians:
  //
  //   J(eps) = J_r(r(eps))^{-1} Ad_{Xd^{-1}} J_r(eps),   J_r(w) = J_l(-w).
  const TGElement Xd = TGElement::Expmap(-delta_x);
  const Eigen::Matrix<double, 18, 18> Ad_Xd_inv =
      gtsam::traits<TGElement>::AdjointMap(Xd.inverse());
  const Eigen::Matrix<double, 18, 18> Jr_eps = detail::LeftJacobianG(-delta_xi);

  const Eigen::Matrix<double, 18, 1> r =
      (TGElement::Expmap(delta_xi) * Xd).Logmap();
  const Eigen::Matrix<double, 18, 18> Jr_r = detail::LeftJacobianG(-r);

  return Jr_r.partialPivLu().solve(Ad_Xd_inv * Jr_eps);
}

void TGEqF::updateWithReset(const Eigen::VectorXd& prediction,
                            const Eigen::MatrixXd& Cstar,
                            const Eigen::VectorXd& z,
                            const Eigen::MatrixXd& R) {
  Eigen::Matrix<double, 18, 1> delta_xi = Eigen::Matrix<double, 18, 1>::Zero();
  Eigen::Matrix<double, 18, 1> delta_x = Eigen::Matrix<double, 18, 1>::Zero();
  Base::updateWithVector(
      prediction, Cstar, z, R, [&](const Eigen::Matrix<double, 18, 1>& dxi) {
        delta_xi = dxi;
        delta_x = innovation_lift_ * dxi;
        return delta_x;
      });

  if (reset_step_) {
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
  // Give the virtual bias its own process noise so the b_v = 0 anchor leaves it
  // a steady-state floor of about sqrt(Q R) instead of driving its covariance
  // to zero.
  Covariance18 Qc_eff = Qc;
  Qc_eff.block<3, 3>(15, 15) += virtual_bias_Q_;

  // Discretize with a 4-term truncation of exp(A dt) rather than the Euler
  // default.
  Base::predict<4>(lift, psi_u, Qc_eff, dt);

  // Pin the unobservable virtual bias at zero unless disabled.
  if (anchor_virtual_bias_) {
    update_virtual_bias(virtual_bias_R_);
  }
}

void TGEqF::propagate(const Eigen::Vector3d& w_meas,
                      const Eigen::Vector3d& a_meas,
                      const Eigen::Vector3d& g_vec, const ImuNoise& noise,
                      double dt) {
  // Map the IMU noise PSDs through the lift.
  propagate(w_meas, a_meas, g_vec, inputNoiseCov(noise), dt);
}

TGEqF::Covariance18 TGEqF::inputNoiseCov(const ImuNoise& noise) const {
  // B is d/du [Dphi0 * Lambda(xi_ref, psi_{Xhat^-1}(u))] restricted to the
  // gyro/accel/bias-RW channels of u. psi_{Xhat^-1} carries its own
  // Ad_{Xhat^-1} factor, which cancels one power of Ad_hat by lift
  // equivariance, leaving the single Ad_hat below rather than the two an
  // Ad_hat * G * Ad_hat chain would suggest.
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
  const State xi_ref = referenceState();
  const TGElement X = groupEstimate();
  const Eigen::Matrix3d R_X = X.R.matrix();

  // The comparison happens in the origin output space: the reference output
  // against the measurement pulled back through the output action.
  const Eigen::Vector3d prediction = DVLMeasurement::predict(xi_ref);
  const Eigen::Vector3d z = DVLMeasurement::inverse_output_action(X, z_dvl);
  const Covariance3 R_eff = R_X * R_dvl * R_X.transpose();

  const Eigen::Matrix<double, 3, 18> Cstar =
      DVLMeasurement::jacobian_Cstar(xi_ref, X, z_dvl);
  updateWithReset(prediction, Cstar, z, R_eff);
}

void TGEqF::update_position(const Eigen::Vector3d& pi,
                            const Covariance3& R_pos) {
  const State xi_ref = referenceState();
  const Eigen::Matrix3d R0 = xi_ref.R.matrix();

  // Both the residual and R_pos are expressed in the reference rotation frame,
  // the origin output space that C* is built in. R_pos arrives in the global
  // frame, one R0 away, so it is conjugated to match.
  const Eigen::Vector3d prediction =
      PositionMeasurement::predictAtOrigin(xi_ref, groupEstimate(), pi);
  const Eigen::Matrix<double, 3, 18> Cstar =
      PositionMeasurement::jacobian_Cstar(xi_ref, groupEstimate(), pi);

  const Eigen::Vector3d z = Eigen::Vector3d::Zero();
  const Covariance3 R_eff = R0.transpose() * R_pos * R0;
  updateWithReset(prediction, Cstar, z, R_eff);
}

void TGEqF::update_depth(double z_depth, const Covariance1& R_depth,
                         double horizontal_variance) {
  Eigen::Vector3d pseudo_position = position();
  pseudo_position.z() = z_depth;

  Covariance3 R_pseudo = Covariance3::Zero();
  R_pseudo(0, 0) = horizontal_variance;
  R_pseudo(1, 1) = horizontal_variance;
  R_pseudo(2, 2) = R_depth(0, 0);

  update_position(pseudo_position, R_pseudo);
}

void TGEqF::update_virtual_bias(const Covariance3& R_vb) {
  // Constraint target b_v = 0, compared against the estimated virtual bias.
  // The output space is the one the estimate reports in, so R_vb needs no
  // transport.
  const Eigen::Vector3d prediction = VirtualBiasMeasurement::predict(state());
  const Eigen::Vector3d z = Eigen::Vector3d::Zero();

  const Eigen::Matrix<double, 3, 18> Cstar =
      VirtualBiasMeasurement::jacobian_Cstar(referenceState(), groupEstimate());
  updateWithReset(prediction, Cstar, z, R_vb);
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
}  // namespace gtsam
