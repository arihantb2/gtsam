#include <gtsam/geometry/Rot3.h>
#include <gtsam_unstable/tg_eqf/TGEqF.h>

#include "gtsam/base/Matrix.h"

namespace gtsam {
namespace tgeqf {

Eigen::Matrix<double, 21, 1> Input::vector() const {
  Eigen::Matrix<double, 21, 1> result;
  result.segment<3>(0) = w;
  result.segment<3>(3) = a;
  result.segment<3>(6) = v;
  result.segment<3>(9) = tau_w;
  result.segment<3>(12) = tau_a;
  result.segment<3>(15) = tau_v;
  result.segment<3>(18) = g_vec;
  return result;
}

Input Input::from_vector(const Eigen::Matrix<double, 21, 1>& v) {
  Input u;
  u.w = v.segment<3>(0);
  u.a = v.segment<3>(3);
  u.v = v.segment<3>(6);
  u.tau_w = v.segment<3>(9);
  u.tau_a = v.segment<3>(12);
  u.tau_v = v.segment<3>(15);
  u.g_vec = v.segment<3>(18);
  return u;
}

Lift::Lift(const Input& u) : u(u) {}

// Closed form; see the Lift functor's doc comment in TGEqF.h.
Eigen::Matrix<double, 18, 1> Lift::operator()(
    const State& xi, OptionalJacobian<18, 18> D_lift) const {
  const Eigen::Matrix3d Rt = xi.T.rotation().transpose();
  const Vector9& b = xi.b;
  Vector9 Lambda1;
  Lambda1.segment<3>(0) = u.w - b.segment<3>(0);
  Lambda1.segment<3>(3) = u.a - b.segment<3>(3) + Rt * u.g_vec;
  Lambda1.segment<3>(6) = u.v - b.segment<3>(6) + Rt * xi.T.x(0);

  Vector9 tau;
  tau.segment<3>(0) = u.tau_w;
  tau.segment<3>(3) = u.tau_a;
  tau.segment<3>(6) = u.tau_v;

  const Matrix9 ad_b = ExtendedPose3<2>::adjointMap(b);
  const Vector9 Lambda2 = ad_b * Lambda1 - tau;

  Eigen::Matrix<double, 18, 1> result;
  result.head<9>() = Lambda1;
  result.tail<9>() = Lambda2;

  if (D_lift) {
    // db/d(eps_tau) = adjointMap(b), db/d(eps_b) = -I_9; Lambda_2 =
    // ad_b[Lambda_1] - tau, so d(Lambda_2) = ad_b d(Lambda_1) - ad_{Lambda_1}
    // db.
    const Matrix9& db_deps_tau = ad_b;
    const Matrix9 db_deps_b = -Matrix9::Identity();
    const Matrix9 ad_Lambda1 = ExtendedPose3<2>::adjointMap(Lambda1);

    D_lift->setZero();
    D_lift->block<3, 3>(3, 0) = gtsam::skewSymmetric(Rt * u.g_vec);
    D_lift->block<3, 3>(6, 0) = gtsam::skewSymmetric(Rt * xi.T.x(0));
    D_lift->block<3, 3>(6, 3) = Eigen::Matrix3d::Identity();
    D_lift->block<9, 9>(0, 0) -= db_deps_tau;
    D_lift->block<9, 9>(0, 9) = -db_deps_b;

    D_lift->block<9, 18>(9, 0) = ad_b * D_lift->block<9, 18>(0, 0);
    D_lift->block<9, 9>(9, 0) -= ad_Lambda1 * db_deps_tau;
    D_lift->block<9, 9>(9, 9) -= ad_Lambda1 * db_deps_b;
  }

  return result;
}

InputOrbit::InputOrbit(const Input& u) : u(u) {}

Input InputOrbit::operator()(const TGElement& X) const {
  Vector9 w;
  w.segment<3>(0) = u.w;
  w.segment<3>(3) = u.a;
  w.segment<3>(6) = u.v;
  Vector9 tau;
  tau.segment<3>(0) = u.tau_w;
  tau.segment<3>(3) = u.tau_a;
  tau.segment<3>(6) = u.tau_v;

  const Matrix9 Ad_A_inv_matrix = X.first.inverse().AdjointMap();
  const Vector9 w_minus_a = w - X.second;
  Vector9 Omega_A_inv = Vector9::Zero();
  Omega_A_inv.segment<3>(6) = X.first.rotation().unrotate(-X.first.x(0));
  const Vector9 w_out = Ad_A_inv_matrix * w_minus_a + Omega_A_inv;
  const Vector9 tau_out = Ad_A_inv_matrix * tau;

  Input result;
  result.w = w_out.segment<3>(0);
  result.a = w_out.segment<3>(3);
  result.v = w_out.segment<3>(6);
  result.tau_w = tau_out.segment<3>(0);
  result.tau_a = tau_out.segment<3>(3);
  result.tau_v = tau_out.segment<3>(6);
  result.g_vec = u.g_vec;
  return result;
}

Vector3 DVLMeasurement::predict(const State& xi) {
  return xi.T.rotation().unrotate(xi.T.x(0));
}

Eigen::Matrix<double, 3, 18> DVLMeasurement::jacobian_C0(const State& xi_ref) {
  const Vector3 y0 = predict(xi_ref);
  Eigen::Matrix<double, 3, 18> C0 = Eigen::Matrix<double, 3, 18>::Zero();
  C0.block<3, 3>(0, 0) = gtsam::skewSymmetric(y0);
  C0.block<3, 3>(0, 3) = Eigen::Matrix3d::Identity();
  return C0;
}

Eigen::Matrix<double, 3, 18> DVLMeasurement::jacobian_Cstar(
    const State& xi_ref, const TGElement& g, const Vector3& z_dvl) {
  const Vector3 y0 = predict(xi_ref);
  const Vector3 vec = y0 + inverse_output_action(g, z_dvl);

  Eigen::Matrix<double, 3, 18> Cstar = Eigen::Matrix<double, 3, 18>::Zero();
  Cstar.block<3, 3>(0, 0) = 0.5 * gtsam::skewSymmetric(vec);
  Cstar.block<3, 3>(0, 3) = Eigen::Matrix3d::Identity();
  return Cstar;
}

Vector3 DVLMeasurement::output_action(const TGElement& X, const Vector3& y) {
  return X.first.rotation().unrotate(y + X.first.x(0));
}

Vector3 DVLMeasurement::inverse_output_action(const TGElement& X,
                                              const Vector3& y) {
  return X.first.rotation().rotate(y) - X.first.x(0);
}

Vector3 PositionMeasurement::predict(const State& xi, const Vector3& pi) {
  return xi.T.rotation().unrotate(pi - xi.T.x(1));
}

Vector3 PositionMeasurement::predictAtOrigin(const State& xi_ref,
                                             const TGElement& g,
                                             const Vector3& pi) {
  const Vector3 y0 = xi_ref.T.rotation().unrotate(pi - xi_ref.T.x(1));
  return y0 - g.first.x(1);
}

Eigen::Matrix<double, 3, 18> PositionMeasurement::jacobian_C0(
    const State& xi_ref, const Vector3& pi) {
  const Vector3 y0 = xi_ref.T.rotation().unrotate(pi - xi_ref.T.x(1));
  Eigen::Matrix<double, 3, 18> C0 = Eigen::Matrix<double, 3, 18>::Zero();
  C0.block<3, 3>(0, 0) = gtsam::skewSymmetric(y0);
  C0.block<3, 3>(0, 6) = -Eigen::Matrix3d::Identity();
  return C0;
}

Eigen::Matrix<double, 3, 18> PositionMeasurement::jacobian_Cstar(
    const State& xi_ref, const TGElement& g, const Vector3& pi) {
  const Vector3 y0 = xi_ref.T.rotation().unrotate(pi - xi_ref.T.x(1));
  const Vector3 vec = y0 + g.first.x(1);

  Eigen::Matrix<double, 3, 18> Cstar = Eigen::Matrix<double, 3, 18>::Zero();
  Cstar.block<3, 3>(0, 0) = 0.5 * gtsam::skewSymmetric(vec);
  Cstar.block<3, 3>(0, 6) = -Eigen::Matrix3d::Identity();
  return Cstar;
}

Vector3 PositionMeasurement::output_action(const TGElement& X,
                                           const Vector3& y) {
  return X.first.rotation().unrotate(y - X.first.x(1));
}

double DepthMeasurement::predict(const State& xi) { return xi.T.x(1).z(); }

Eigen::Matrix<double, 1, 18> DepthMeasurement::jacobian_C(const State& xi_ref,
                                                          const TGElement& g) {
  const Vector3 n = xi_ref.T.rotation().unrotate(Vector3::UnitZ());

  Eigen::Matrix<double, 1, 18> C = Eigen::Matrix<double, 1, 18>::Zero();
  C.block<1, 3>(0, 0) = g.first.x(1).cross(n).transpose();
  C.block<1, 3>(0, 6) = n.transpose();
  return C;
}

namespace {

/**
 * Van Loan discretization of the process noise,
 *
 *   Qd = int_0^dt exp(A s) Qc exp(A^T s) ds,
 *
 * from the truncated exponential of the block matrix C = [-A, Qc; 0, A^T] dt:
 * expm(C) = [F11, G; 0, F22] with F22 = exp(A^T dt) and
 * G = exp(-A dt) int_0^dt exp(A s) Qc exp(A^T s) ds, so Qd = F22^T G.
 */
Eigen::Matrix<double, 18, 18> discretizeNoiseVanLoan(
    const Eigen::Matrix<double, 18, 18>& A,
    const Eigen::Matrix<double, 18, 18>& Qc, double dt) {
  Eigen::Matrix<double, 36, 36> C = Eigen::Matrix<double, 36, 36>::Zero();
  C.topLeftCorner<18, 18>() = -A;
  C.topRightCorner<18, 18>() = Qc;
  C.bottomRightCorner<18, 18>() = A.transpose();

  const Matrix E = expm(C * dt, 4);
  const Eigen::Matrix<double, 18, 18> Qd =
      E.bottomRightCorner<18, 18>().transpose() * E.topRightCorner<18, 18>();
  return 0.5 * (Qd + Qd.transpose());
}

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

// Left Jacobian of G, J_l(gamma) = sum_{k>=0} ad_gamma^k / (k+1)!. Kept as a
// free function (TangentLieGroup<G>'s own right Jacobian is a different,
// private kernel) built from TGElement::adjointMap. The series converges on
// rotation angle alone, so 28 terms reach machine precision over Logmap's
// full [-pi, pi] range.
Eigen::Matrix<double, 18, 18> leftJacobianG(
    const Eigen::Matrix<double, 18, 1>& gamma) {
  const Eigen::Matrix<double, 18, 18> ad = TGElement::adjointMap(gamma);
  Eigen::Matrix<double, 18, 18> term =
      Eigen::Matrix<double, 18, 18>::Identity();
  Eigen::Matrix<double, 18, 18> sum = Eigen::Matrix<double, 18, 18>::Zero();
  double fact = 1.0;
  for (int k = 0; k < 28; ++k) {
    fact *= (k + 1);
    sum += term / fact;
    term = term * ad;
  }
  return sum;
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

TGEqF::TGEqF(const TGElement& X0, const Covariance18& P0)
    : TGEqF(State::identity(), P0, X0) {}

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

  Eigen::Matrix<double, 18, 18> G = Eigen::Matrix<double, 18, 18>::Zero();
  G.block<9, 9>(0, 0).setIdentity();
  G.block<9, 9>(9, 0) = ExtendedPose3<2>::adjointMap(xi_ref.b);
  G.block<9, 9>(9, 9) = -Matrix9::Identity();
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
  const Eigen::Matrix<double, 18, 18> Ad_Xd_inv = Xd.inverse().AdjointMap();
  const Eigen::Matrix<double, 18, 18> Jr_eps = leftJacobianG(-delta_xi);

  const Eigen::Matrix<double, 18, 1> r =
      TGElement::Logmap(TGElement::Expmap(delta_xi) * Xd);
  const Eigen::Matrix<double, 18, 18> Jr_r = leftJacobianG(-r);

  return Jr_r.partialPivLu().solve(Ad_Xd_inv * Jr_eps);
}

void TGEqF::updateWithReset(const Eigen::VectorXd& prediction,
                            const Eigen::MatrixXd& Cstar,
                            const Eigen::VectorXd& z,
                            const Eigen::MatrixXd& R) {
  Eigen::Matrix<double, 18, 1> delta_xi = Eigen::Matrix<double, 18, 1>::Zero();
  Eigen::Matrix<double, 18, 1> delta_x = Eigen::Matrix<double, 18, 1>::Zero();
  Base::updateWithVector(prediction, Cstar, z, R,
                         [&](const Eigen::Matrix<double, 18, 1>& dxi) {
                           delta_xi = dxi;
                           delta_x = innovation_lift_ * dxi;
                           return delta_x;
                         });

  if (reset_step_) {
    const Eigen::Matrix<double, 18, 18> J = resetMatrix(delta_xi, delta_x);
    this->P_ = J * this->P_ * J.transpose();
  }
}

void TGEqF::propagate(const Vector3& w_meas, const Vector3& a_meas,
                      const Vector3& g_vec, const Covariance18& Qc, double dt) {
  Input u;
  u.w = w_meas;
  u.a = a_meas;
  u.g_vec = g_vec;
  // Virtual input nu = 0
  u.v = Vector3::Zero();
  u.tau_w = Vector3::Zero();
  u.tau_a = Vector3::Zero();
  u.tau_v = Vector3::Zero();

  const Lift lift(u);
  const InputOrbit psi_u(u);

  // 4-term truncation of exp(A dt) for the transition matrix, Van Loan for
  // the process noise. predictWithJacobian scales Qc by dt, so Qd is
  // pre-divided.
  const MatrixM A = computeErrorDynamicsMatrix<Lift>(psi_u);
  const Covariance18 Qc_eff =
      dt > 0.0 ? Covariance18(discretizeNoiseVanLoan(A, Qc, dt) / dt) : Qc;
  Base::predictWithJacobian<4>(lift, A, Qc_eff, dt);
}

void TGEqF::propagate(const Vector3& w_meas, const Vector3& a_meas,
                      const Vector3& g_vec, const ImuNoise& noise, double dt) {
  // Map the IMU noise PSDs through the lift.
  propagate(w_meas, a_meas, g_vec, inputNoiseCov(noise), dt);
}

TGEqF::Covariance18 TGEqF::inputNoiseCov(const ImuNoise& noise) const {
  // B is d/du [Dphi0 * Lambda(xi_ref, psi_{Xhat^-1}(u))], gyro/accel/bias-RW
  // channels only. psi_{Xhat^-1}'s own Ad_{Xhat^-1} cancels one power of
  // Ad_hat by lift equivariance, leaving one Ad_hat rather than the
  // Ad_hat * G * Ad_hat a naive reading would suggest.
  const Matrix9 Ad_hat = groupEstimate().first.AdjointMap();
  Eigen::Matrix<double, 18, 12> AdSel = Eigen::Matrix<double, 18, 12>::Zero();
  AdSel.block<9, 6>(0, 0) = Ad_hat.leftCols<6>();
  AdSel.block<9, 6>(9, 6) = Ad_hat.leftCols<6>();
  const Eigen::Matrix<double, 18, 12> B = input_lift_ * AdSel;

  Eigen::Matrix<double, 12, 12> Sigma = Eigen::Matrix<double, 12, 12>::Zero();
  Sigma.block<3, 3>(0, 0) = noise.gyro.asDiagonal();
  Sigma.block<3, 3>(3, 3) = noise.accel.asDiagonal();
  Sigma.block<3, 3>(6, 6) = noise.gyro_rw.asDiagonal();
  Sigma.block<3, 3>(9, 9) = noise.accel_rw.asDiagonal();

  auto Qc = B * Sigma * B.transpose();

  // Continuous-time density; propagate() applies the Van Loan discretization.
  return Qc;
}

void TGEqF::update_dvl(const Vector3& z_dvl, const Covariance3& R_dvl) {
  const State xi_ref = referenceState();
  const TGElement X = groupEstimate();
  const Eigen::Matrix3d R_X = X.first.rotation().matrix();

  // Compared in the origin output space: reference output against the
  // measurement pulled back through the output action.
  const Vector3 prediction = DVLMeasurement::predict(xi_ref);
  const Vector3 z = DVLMeasurement::inverse_output_action(X, z_dvl);
  const Covariance3 R_eff = R_X * R_dvl * R_X.transpose();

  const Eigen::Matrix<double, 3, 18> C =
      dvl_jacobian_ == DvlJacobian::C0
          ? DVLMeasurement::jacobian_C0(xi_ref)
          : DVLMeasurement::jacobian_Cstar(xi_ref, X, z_dvl);
  updateWithReset(prediction, C, z, R_eff);
}

void TGEqF::update_position(const Vector3& pi, const Covariance3& R_pos) {
  const State xi_ref = referenceState();
  const Eigen::Matrix3d R0 = xi_ref.T.rotation().matrix();

  // C*'s origin output space is the reference rotation frame; R_pos arrives
  // in the global frame, so conjugate it by R0 to match.
  const Vector3 prediction =
      PositionMeasurement::predictAtOrigin(xi_ref, groupEstimate(), pi);
  const Eigen::Matrix<double, 3, 18> Cstar =
      PositionMeasurement::jacobian_Cstar(xi_ref, groupEstimate(), pi);

  const Vector3 z = Vector3::Zero();
  const Covariance3 R_eff = R0.transpose() * R_pos * R0;
  updateWithReset(prediction, Cstar, z, R_eff);
}

void TGEqF::update_depth(double z_depth, const Covariance1& R_depth,
                         double horizontal_variance) {
  Vector3 pseudo_position = position();
  pseudo_position.z() = z_depth;

  Covariance3 R_pseudo = Covariance3::Zero();
  R_pseudo(0, 0) = horizontal_variance;
  R_pseudo(1, 1) = horizontal_variance;
  R_pseudo(2, 2) = R_depth(0, 0);

  update_position(pseudo_position, R_pseudo);
}

void TGEqF::update_depth_direct(double z_depth, const Covariance1& R_depth) {
  // No output action exists for h(xi) = e_3^T p, so the innovation is formed
  // in the raw sensor space: prediction at the estimate, R_depth untransported.
  Covariance1 prediction, z;
  prediction(0, 0) = DepthMeasurement::predict(state());
  z(0, 0) = z_depth;

  const Eigen::Matrix<double, 1, 18> C =
      DepthMeasurement::jacobian_C(referenceState(), groupEstimate());
  updateWithReset(prediction, C, z, R_depth);
}

State TGEqF::errorState(const State& xi_true) const {
  const TGElement& Xhat = groupEstimate();
  return phi(Xhat.inverse(), xi_true);
}

TGEqF::TangentVector TGEqF::errorStateVector(const State& xi_true) const {
  const State e = errorState(xi_true);
  return gtsam::traits<State>::Local(referenceState(), e);
}

gtsam::Rot3 TGEqF::attitude() const { return state().T.rotation(); }

Vector3 TGEqF::velocity() const { return state().T.x(0); }

Vector3 TGEqF::position() const { return state().T.x(1); }

Vector3 TGEqF::bias_gyro() const { return state().b.segment<3>(0); }

Vector3 TGEqF::bias_accel() const { return state().b.segment<3>(3); }

Vector3 TGEqF::bias_vel() const { return state().b.segment<3>(6); }

}  // namespace tgeqf
}  // namespace gtsam
