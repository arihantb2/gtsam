#include <gtsam_unstable/tg_eqf/EqF.h>

namespace tgeqf {

TGEqF::TGEqF(const TGState& xi_ref, const Covariance18& Sigma0,
             const TGGroupElement& X0)
    : Base(xi_ref, Sigma0, X0) {
  // Recompute the innovation lift pinv(Dphi0) the base keeps private (Dphi0 is
  // d phi(X, xi_ref)/dX at identity). xi_ref is fixed for this filter's
  // lifetime, so caching once is sound (do not use resetReferenceAndGroup).
  Eigen::Matrix<double, 18, 18> Dphi0;
  const TGSymmetry::Orbit orbit(xi_ref);
  orbit(TGGroupElement::Identity(), &Dphi0);
  innovation_lift_ = Dphi0.completeOrthogonalDecomposition().pseudoInverse();
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
  // Exact Jacobian of the error re-centring map
  //   f(eps) = Local(xi_ref, phi(Exp(-delta_x), Retract(xi_ref, eps)))
  // at eps = delta_xi (the post-update mean in the old chart). Differentiating
  // the analytic diffeomorphism alone drops the SO(3) Retract/Local chart
  // factors, which are first order in the rotation correction, so finite-
  // difference the full composition instead. J -> I as the correction -> 0.
  const TGGroupElement Xd = TGGroupElement::Expmap(-delta_x);
  const TGState& xi_ref = referenceState();
  auto recentre = [&](const Eigen::Matrix<double, 18, 1>& eps) {
    return gtsam::traits<TGState>::Local(
        xi_ref, phi(Xd, gtsam::traits<TGState>::Retract(xi_ref, eps)));
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
  // Capture the manifold correction through the base's custom innovation-lift
  // hook (it receives delta_xi and must return delta_x = pinv(Dphi0)*delta_xi).
  Base::updateWithVector(
      prediction, H, z, R,
      [&](const Eigen::Matrix<double, 18, 1>& dxi) {
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
                      const Eigen::Vector3d& g_vec,
                      const Covariance18& Qc, double dt) {
  const TGState xi_hat = state();

  TGInput u;
  u.w = w_meas;
  u.a = a_meas;
  u.g_vec = g_vec;
  u.v = xi_hat.b_v;
  u.tau_w = Eigen::Vector3d::Zero();
  u.tau_a = Eigen::Vector3d::Zero();
  u.tau_v = Eigen::Vector3d::Zero();

  const Lift lift(u);
  const InputOrbit psi_u(u);
  Base::predict(lift, psi_u, Qc, dt);

  // Keep the unobservable virtual bias pinned at zero (Eq. B.20) if requested.
  if (anchor_virtual_bias_) update_virtual_bias(virtual_bias_R_);
}

void TGEqF::set_virtual_bias_anchor(bool enable, const Covariance3& R_vb) {
  anchor_virtual_bias_ = enable;
  virtual_bias_R_ = R_vb;
}

void TGEqF::update_dvl(const Eigen::Vector3d& z_dvl,
                       const Covariance3& R_dvl) {
  const TGState xi_hat = state();
  const Eigen::Vector3d prediction = DVLMeasurement::predict(xi_hat);
  // Compose the chart-at-estimate Jacobian with the origin-chart transport so
  // the filter consumes H in its own (origin) chart (CODE_REVIEW F1). The DVL
  // residual R^T v is body-frame, so R_dvl needs no extra transport.
  const Eigen::Matrix<double, 3, 18> H =
      DVLMeasurement::jacobian(xi_hat) * originChartTransport();
  updateWithReset(prediction, H, z_dvl, R_dvl);
}

void TGEqF::update_position(const Eigen::Vector3d& pi,
                            const Covariance3& R_pos, bool use_Cstar) {
  // Paper-literal position pairing (CODE_REVIEW F3, Eq. B.19): C0/C* are built
  // from the fixed reference state, so they are already in the filter's origin
  // chart (no Dphi_g composition). The residual is the origin-frame
  //   prediction = R0^T (pi - p_hat) = y0 - p_X,   target z = 0,
  // which gtsam pairs as delta_xi = -K (prediction - z); with the -R0^T
  // position block this moves p_hat toward pi (verified at the origin and at a
  // rotated state). Express R_pos in the same R0-rotated frame.
  const TGState xi_ref = referenceState();
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
  const TGState xi_hat = state();
  const Eigen::Vector3d prediction = VirtualBiasMeasurement::predict(xi_hat);
  // [0...I3] * Dphi_g reproduces Eq. B.20's R_hat,p_hat-dependent bias
  // transport automatically (it is the b_v row of Ad^-1) — CODE_REVIEW F1.
  const Eigen::Matrix<double, 3, 18> H =
      VirtualBiasMeasurement::jacobian_C0(xi_hat) * originChartTransport();

  // Constraint target b_v = 0.
  const Eigen::Vector3d z = Eigen::Vector3d::Zero();
  updateWithReset(prediction, H, z, R_vb);
}

gtsam::Rot3 TGEqF::attitude() const { return state().R; }

Eigen::Vector3d TGEqF::velocity() const { return state().v; }

Eigen::Vector3d TGEqF::position() const { return state().p; }

Eigen::Vector3d TGEqF::bias_gyro() const { return state().b_w; }

Eigen::Vector3d TGEqF::bias_accel() const { return state().b_a; }

Eigen::Vector3d TGEqF::bias_vel() const { return state().b_v; }

}  // namespace tgeqf
