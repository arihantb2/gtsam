#include <CppUnitLite/TestHarness.h>
#include <gtsam/base/TestableAssertions.h>
#include <gtsam_unstable/tg_eqf/EqF.h>
#include <gtsam_unstable/tg_eqf/PositionOutput.h>
#include <gtsam_unstable/tg_eqf/BodyVelocityOutput.h>
#include <gtsam_unstable/tg_eqf/VirtualBiasOutput.h>
#include <gtsam_unstable/tg_eqf/Symmetry.h>

using namespace tgeqf;
using namespace gtsam;

static constexpr double kTol = 1e-6;

using Vector18 = Eigen::Matrix<double, 18, 1>;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

static bool veq(const Eigen::Vector3d& a, const Eigen::Vector3d& b,
                double tol = kTol) {
  return assert_equal((Vector)a, (Vector)b, tol);
}

static TGEqF::Covariance18 defaultSigma() {
  return 0.01 * TGEqF::Covariance18::Identity();
}

static TGEqF::Covariance18 defaultQc() {
  // Tangent order [att(0:3), vel(3:6), pos(6:9), b_w(9:12), b_a(12:15),
  // b_v(15:18)]: accel noise drives velocity, accel bias RW drives b_a.
  TGEqF::Covariance18 Qc = TGEqF::Covariance18::Zero();
  Qc.block<3, 3>(0, 0) = 1e-4 * Eigen::Matrix3d::Identity();    // attitude
  Qc.block<3, 3>(3, 3) = 1e-3 * Eigen::Matrix3d::Identity();    // velocity
  Qc.block<3, 3>(9, 9) = 1e-6 * Eigen::Matrix3d::Identity();    // gyro bias
  Qc.block<3, 3>(12, 12) = 1e-5 * Eigen::Matrix3d::Identity();  // accel bias
  return Qc;
}

static TGInput makeImuInput(const Eigen::Vector3d& omega,
                            const Eigen::Vector3d& accel,
                            const Eigen::Vector3d& g_vec,
                            const Eigen::Vector3d& b_v =
                                Eigen::Vector3d::Zero()) {
  TGInput u;
  u.w = omega;
  u.a = accel;
  u.v = b_v;
  u.tau_w = Eigen::Vector3d::Zero();
  u.tau_a = Eigen::Vector3d::Zero();
  u.tau_v = Eigen::Vector3d::Zero();
  u.g_vec = g_vec;
  return u;
}

// ---------------------------------------------------------------------------
// Construction and accessors
// ---------------------------------------------------------------------------

TEST(TGEqF, InitializesAtReferenceState) {
  const TGState xi_ref = TGState::identity();
  const TGEqF::Covariance18 Sigma0 = defaultSigma();
  TGEqF filter(xi_ref, Sigma0);

  EXPECT(traits<TGState>::Equals(xi_ref, filter.state(), kTol));
  EXPECT(traits<TGGroupElement>::Equals(TGGroupElement::Identity(),
                                        filter.groupEstimate(), kTol));
  EXPECT(assert_equal(Sigma0, filter.errorCovariance(), kTol));
}

TEST(TGEqF, AccessorsMatchInitialState) {
  TGEqF filter(TGState::identity(), defaultSigma());

  EXPECT(filter.attitude().equals(Rot3::Identity(), kTol));
  EXPECT(veq(Eigen::Vector3d::Zero(), filter.position()));
  EXPECT(veq(Eigen::Vector3d::Zero(), filter.velocity()));
  EXPECT(veq(Eigen::Vector3d::Zero(), filter.bias_gyro()));
  EXPECT(veq(Eigen::Vector3d::Zero(), filter.bias_accel()));
  EXPECT(veq(Eigen::Vector3d::Zero(), filter.bias_vel()));
}

// ---------------------------------------------------------------------------
// Predict
// ---------------------------------------------------------------------------

TEST(TGEqF, PropagateAdvancesGroupAndCovariance) {
  TGEqF filter(TGState::identity(), defaultSigma());
  filter.set_virtual_bias_anchor(false);  // pure predict: no anchor shrink

  const Eigen::Vector3d omega(0.05, -0.02, 0.01);
  const Eigen::Vector3d g_vec(0.0, 0.0, -9.81);
  const Eigen::Vector3d accel(0.1, -0.2, 0.05);
  const double dt = 0.1;

  const TGGroupElement g_before = filter.groupEstimate();
  const TGEqF::Covariance18 P_before = filter.errorCovariance();

  filter.propagate(omega, accel, g_vec, defaultQc(), dt);

  EXPECT(!traits<TGGroupElement>::Equals(g_before, filter.groupEstimate(),
                                         1e-12));
  EXPECT(filter.errorCovariance().trace() > P_before.trace());
}

// The filter feeds the virtual input nu = 0 (paper App. B.4.3), independent of
// the b_v estimate. With a non-zero reference b_v, propagate (anchor disabled
// for a clean predict) must equal a manual lift built with u.v = 0.
TEST(TGEqF, PropagateUsesZeroVirtualInput) {
  TGState xi_ref = TGState::identity();
  xi_ref.b_v = Eigen::Vector3d(0.1, -0.05, 0.08);  // non-zero virtual bias
  TGEqF filter(xi_ref, defaultSigma());
  filter.set_virtual_bias_anchor(false);

  const Eigen::Vector3d omega = Eigen::Vector3d::Zero();
  const Eigen::Vector3d g_vec(0.0, 0.0, -9.81);
  const Eigen::Vector3d accel(0.1, -0.2, 0.05);
  const double dt = 0.05;

  filter.propagate(omega, accel, g_vec, defaultQc(), dt);
  const TGGroupElement g_imu = filter.groupEstimate();

  TGEqF filter2(xi_ref, defaultSigma());
  filter2.set_virtual_bias_anchor(false);
  // u.v = 0 (zero virtual input), regardless of b_v.
  const TGInput u = makeImuInput(omega, accel, g_vec, Eigen::Vector3d::Zero());
  const Lift lift(u);
  const InputOrbit psi_u(u);
  filter2.template predict<1>(lift, psi_u, defaultQc(), dt);

  EXPECT(traits<TGGroupElement>::Equals(g_imu, filter2.groupEstimate(), 1e-9));
}

// Regression: with a non-zero initial velocity, IMU propagation must advance
// the position estimate by ~v*dt (dp = v). The constant-N lift froze position
// at zero while velocity tracked, so guard the integrated position explicitly.
TEST(TGEqF, PropagateIntegratesPosition) {
  TGState xi0 = TGState::identity();
  xi0.v = Eigen::Vector3d(1.0, -0.5, 0.2);  // non-rest start
  TGEqF filter(xi0, defaultSigma());
  filter.set_virtual_bias_anchor(false);  // pure predict

  const Eigen::Vector3d g_vec(0.0, 0.0, -9.81);
  const Eigen::Vector3d accel = -g_vec;  // specific force of a level body at rest
  const double dt = 0.01;

  const Eigen::Vector3d p_before = filter.position();
  filter.propagate(Eigen::Vector3d::Zero(), accel, g_vec, defaultQc(), dt);
  const Eigen::Vector3d p_after = filter.position();

  // Position advanced by velocity*dt (velocity ~constant over one small step).
  EXPECT(veq(p_after - p_before, xi0.v * dt, 1e-4));
}

// ---------------------------------------------------------------------------
// Position update
// ---------------------------------------------------------------------------

TEST(TGEqF, PositionUpdateReducesEquivariantInnovation) {
  TGEqF filter(TGState::identity(), defaultSigma());

  const Eigen::Vector3d pi(2.0, -1.0, 0.5);
  const TGEqF::Covariance3 R_pos = 0.001 * TGEqF::Covariance3::Identity();

  const double err_before =
      PositionMeasurement::predict(filter.state(), pi).norm();
  EXPECT(err_before > 1.0);

  filter.update_position(pi, R_pos, true);

  const double err_after =
      PositionMeasurement::predict(filter.state(), pi).norm();
  EXPECT(err_after < err_before);
  EXPECT(err_after < 0.5 * err_before);
}

TEST(TGEqF, PositionUpdateMovesEstimateTowardMeasurement) {
  TGEqF filter(TGState::identity(), defaultSigma());

  const Eigen::Vector3d pi(1.5, 0.0, -0.25);
  const TGEqF::Covariance3 R_pos = 0.001 * TGEqF::Covariance3::Identity();

  filter.update_position(pi, R_pos, false);

  EXPECT((filter.position() - pi).norm() < 0.2);
}

TEST(TGEqF, PositionUpdateShrinksPositionCovariance) {
  TGEqF filter(TGState::identity(), defaultSigma());

  const Eigen::Vector3d pi(1.0, 0.5, -0.2);
  const TGEqF::Covariance3 R_pos = 0.01 * TGEqF::Covariance3::Identity();

  // Position lives in origin-chart tangent columns 6..8 ([R,v,p] order). Use
  // errorCovariance() (origin chart); covariance() transports as J^T P J (F8).
  const double trace_before = filter.errorCovariance().block<3, 3>(6, 6).trace();
  filter.update_position(pi, R_pos, true);
  const double trace_after = filter.errorCovariance().block<3, 3>(6, 6).trace();

  EXPECT(trace_after < trace_before);
}

// ---------------------------------------------------------------------------
// DVL update
// ---------------------------------------------------------------------------

TEST(TGEqF, DvlUpdateReducesInnovation) {
  TGEqF filter(TGState::identity(), defaultSigma());

  const Eigen::Vector3d z_dvl(0.4, -0.2, 0.1);
  const TGEqF::Covariance3 R_dvl = 0.001 * TGEqF::Covariance3::Identity();

  const double err_before =
      DVLMeasurement::innovation(z_dvl, filter.state()).norm();
  EXPECT(err_before > 0.3);

  filter.update_dvl(z_dvl, R_dvl);

  const double err_after =
      DVLMeasurement::innovation(z_dvl, filter.state()).norm();
  EXPECT(err_after < err_before);
  EXPECT(err_after < 0.5 * err_before);
}

TEST(TGEqF, DvlUpdateMovesVelocityTowardMeasurement) {
  TGEqF filter(TGState::identity(), defaultSigma());

  const Eigen::Vector3d z_dvl(0.8, 0.1, -0.3);
  const TGEqF::Covariance3 R_dvl = 0.001 * TGEqF::Covariance3::Identity();

  filter.update_dvl(z_dvl, R_dvl);

  EXPECT((DVLMeasurement::predict(filter.state()) - z_dvl).norm() < 0.2);
}

TEST(TGEqF, DvlUpdateShrinksVelocityCovariance) {
  TGEqF filter(TGState::identity(), defaultSigma());

  const Eigen::Vector3d z_dvl(0.5, -0.2, 0.3);
  const TGEqF::Covariance3 R_dvl = 0.01 * TGEqF::Covariance3::Identity();

  // Velocity lives in origin-chart tangent columns 3..5 ([R,v,p,...] order).
  const double trace_before = filter.errorCovariance().block<3, 3>(3, 3).trace();
  filter.update_dvl(z_dvl, R_dvl);
  const double trace_after = filter.errorCovariance().block<3, 3>(3, 3).trace();

  EXPECT(trace_after < trace_before);
}

// ---------------------------------------------------------------------------
// Virtual-bias pseudo-measurement (Eq. B.20)
// ---------------------------------------------------------------------------

TEST(TGEqF, VirtualBiasUpdateDrivesBiasToZero) {
  // Start from a reference whose virtual bias is non-zero; the b_v = 0
  // constraint update must pull the estimate toward zero.
  TGState xi_ref = TGState::identity();
  xi_ref.b_v = Eigen::Vector3d(0.2, -0.15, 0.1);
  TGEqF filter(xi_ref, defaultSigma());

  const double err_before = filter.bias_vel().norm();
  EXPECT(err_before > 0.2);

  const TGEqF::Covariance3 R_vb = 1e-4 * TGEqF::Covariance3::Identity();
  filter.update_virtual_bias(R_vb);

  const double err_after = filter.bias_vel().norm();
  EXPECT(err_after < err_before);
  EXPECT(err_after < 0.5 * err_before);
}

TEST(TGEqF, AnchorCanBeDisabledLeavingVirtualBias) {
  TGState xi_ref = TGState::identity();
  xi_ref.b_v = Eigen::Vector3d(0.2, -0.15, 0.1);
  TGEqF filter(xi_ref, defaultSigma());
  filter.set_virtual_bias_anchor(false);  // explicitly disable the default

  const Eigen::Vector3d g_vec(0.0, 0.0, -9.81);
  filter.propagate(Eigen::Vector3d::Zero(), -g_vec, g_vec, defaultQc(), 0.01);

  // No anchoring: bdot = 0 and nu = 0, so b_v is unchanged by propagation.
  EXPECT(veq(xi_ref.b_v, filter.bias_vel(), 1e-9));
}

TEST(TGEqF, AnchorOnByDefaultDrivesVirtualBiasToZeroInPropagate) {
  TGState xi_ref = TGState::identity();
  xi_ref.b_v = Eigen::Vector3d(0.2, -0.15, 0.1);
  TGEqF filter(xi_ref, defaultSigma());  // anchor on by default

  const double before = filter.bias_vel().norm();
  const Eigen::Vector3d g_vec(0.0, 0.0, -9.81);
  filter.propagate(Eigen::Vector3d::Zero(), -g_vec, g_vec, defaultQc(), 0.01);

  EXPECT(filter.bias_vel().norm() < 0.5 * before);
}

TEST(TGEqF, VirtualBiasUpdateShrinksVirtualBiasCovariance) {
  TGEqF filter(TGState::identity(), defaultSigma());

  // b_v lives in origin-chart tangent columns 15..17 ([R,v,p,b_w,b_a,b_v]).
  const double trace_before = filter.errorCovariance().block<3, 3>(15, 15).trace();
  filter.update_virtual_bias(0.01 * TGEqF::Covariance3::Identity());
  const double trace_after = filter.errorCovariance().block<3, 3>(15, 15).trace();

  EXPECT(trace_after < trace_before);
}

// ---------------------------------------------------------------------------
// End-to-end
// ---------------------------------------------------------------------------

TEST(TGEqF, EndToEndPropagateAndUpdates) {
  TGEqF filter(TGState::identity(), defaultSigma());

  const Eigen::Vector3d g_vec(0.0, 0.0, -9.81);
  const Eigen::Vector3d omega(0.02, 0.0, 0.01);
  const double dt = 0.05;

  // A roughly level body at rest reads specific force = -g (not +g): the
  // accelerometer measures the reaction to gravity, not free-fall.
  filter.propagate(omega, -g_vec, g_vec, defaultQc(), dt);

  const Eigen::Vector3d pi = filter.position() + Eigen::Vector3d(0.5, 0.0, 0.0);
  const Eigen::Vector3d z_dvl =
      filter.attitude().unrotate(filter.velocity() + Eigen::Vector3d(0.1, 0.0, 0.0));

  const TGEqF::Covariance3 R = 0.01 * TGEqF::Covariance3::Identity();
  filter.update_position(pi, R, true);
  filter.update_dvl(z_dvl, R);

  EXPECT(PositionMeasurement::predict(filter.state(), pi).norm() < 0.3);
  EXPECT(DVLMeasurement::innovation(z_dvl, filter.state()).norm() < 0.15);
}

TEST(TGEqF, PositionOutputEquivarianceHoldsAfterUpdate) {
  const TGGroupElement X = TGGroupElement::Identity();
  TGState xi = TGState::identity();
  xi.p = Eigen::Vector3d(0.3, -0.1, 0.2);

  TGEqF filter(TGState::identity(), defaultSigma());
  const Eigen::Vector3d pi(1.0, 0.2, -0.4);
  filter.update_position(pi, 0.005 * TGEqF::Covariance3::Identity(), true);

  const Eigen::Vector3d y = PositionMeasurement::predict(filter.state(), pi);
  const Eigen::Vector3d lhs =
      PositionMeasurement::predict(phi(X, filter.state()), pi);
  const Eigen::Vector3d rhs = PositionMeasurement::output_action(X, y);
  EXPECT(veq(lhs, rhs, 1e-7));
}

// ---------------------------------------------------------------------------
// F1: origin-chart measurement Jacobians away from the reference
//
// Build the filter at a state rotated/translated far from xi_ref = identity
// (Rz(90 deg), v=(2,0,0), p=(5,-3,1)) and check that the Jacobian the filter
// actually consumes is the origin-chart one (H_est * Dphi_g), not the
// chart-at-estimate H_est.
// ---------------------------------------------------------------------------

namespace {

// Group element X0 with phi(X0, identity) = (Rz(90), (2,0,0), (5,-3,1), 0).
TGGroupElement rotatedX0() {
  TGGroupElement X;
  X.R = Rot3::Rz(M_PI / 2);
  X.v = Eigen::Vector3d(2.0, 0.0, 0.0);
  X.p = Eigen::Vector3d(5.0, -3.0, 1.0);
  X.a = {Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero(),
         Eigen::Vector3d::Zero()};
  return X;
}

Eigen::Matrix<double, 18, 18> diffeoJacobian(const TGGroupElement& X,
                                             const TGState& xi) {
  Eigen::Matrix<double, 18, 18> J;
  const TGSymmetry::Diffeomorphism phi_X(X);
  phi_X(xi, &J);
  return J;
}

}  // namespace

// The DVL Jacobian the filter consumes is the FD of the origin-chart map
// eps -> h(phi(X0, Retract(xi_ref, eps))). The composed H_est * Dphi_g must
// equal it, and (for this configuration) its rotation block must vanish.
TEST(TGEqF, DvlOriginChartJacobianMatchesFiniteDifference) {
  const TGState xi_ref = TGState::identity();
  const TGGroupElement X0 = rotatedX0();
  const TGState xi_hat = phi(X0, xi_ref);

  // Finite-difference the origin-chart measurement map.
  const Eigen::Vector3d f0 = DVLMeasurement::predict(xi_hat);
  const double h = 1e-6;
  Eigen::Matrix<double, 3, 18> H_fd;
  for (int j = 0; j < 18; ++j) {
    Eigen::Matrix<double, 18, 1> e = Eigen::Matrix<double, 18, 1>::Zero();
    e(j) = h;
    const TGState xi_eps = phi(X0, traits<TGState>::Retract(xi_ref, e));
    H_fd.col(j) = (DVLMeasurement::predict(xi_eps) - f0) / h;
  }

  const Eigen::Matrix<double, 3, 18> H_origin =
      DVLMeasurement::jacobian(xi_hat) * diffeoJacobian(X0, xi_ref);

  EXPECT(assert_equal((Matrix)H_fd, (Matrix)H_origin, 1e-5));

  // The chart-at-estimate Jacobian has a spurious skew(R^T v) rotation block;
  // the origin-chart one does not (cf. CODE_REVIEW Appendix A).
  EXPECT(assert_equal((Matrix)Eigen::Matrix3d::Zero(),
                      (Matrix)H_origin.block<3, 3>(0, 0), 1e-5));
}

// A DVL update at a rotated state must move the velocity estimate in the
// correct *global* direction. A +0.1 m/s global-y offset should raise the
// global-y velocity, not leak into x/z (which the un-transported Jacobian did).
TEST(TGEqF, DvlUpdateMovesVelocityInGlobalFrameAtRotatedState) {
  const TGState xi_ref = TGState::identity();
  const TGGroupElement X0 = rotatedX0();
  TGEqF filter(xi_ref, defaultSigma(), X0);

  const Eigen::Vector3d v_before = filter.velocity();
  const Eigen::Vector3d delta_global(0.0, 0.1, 0.0);
  // Measured body-frame velocity = R^T (v + delta_global).
  const Eigen::Vector3d z_dvl =
      filter.attitude().unrotate(v_before + delta_global);

  filter.update_dvl(z_dvl, 1e-4 * TGEqF::Covariance3::Identity());

  const Eigen::Vector3d dv = filter.velocity() - v_before;
  // Tight R + loose Sigma0 -> near-unit gain, so dv ~ delta_global.
  EXPECT(dv.y() > 0.08);
  EXPECT(std::abs(dv.x()) < 0.02);
  EXPECT(std::abs(dv.z()) < 0.02);
}

// A position update at a rotated state must move the estimate in the correct
// *global* direction. A +0.5 m global-y measurement offset should move the
// position estimate +0.5 m in y, not leak into x/z (the un-transported,
// half-coupled code moved it mostly +x; cf. CODE_REVIEW Appendix A, F3).
TEST(TGEqF, PositionUpdateMovesEstimateInGlobalFrameAtRotatedState) {
  const TGState xi_ref = TGState::identity();
  const TGGroupElement X0 = rotatedX0();
  TGEqF filter(xi_ref, defaultSigma(), X0);

  const Eigen::Vector3d p_before = filter.position();
  const Eigen::Vector3d pi = p_before + Eigen::Vector3d(0.0, 0.5, 0.0);

  filter.update_position(pi, 1e-4 * TGEqF::Covariance3::Identity(), true);

  const Eigen::Vector3d dp = filter.position() - p_before;
  EXPECT(dp.y() > 0.4);
  EXPECT(std::abs(dp.x()) < 0.1);
  EXPECT(std::abs(dp.z()) < 0.1);
}

// ---------------------------------------------------------------------------
// F4: covariance reset step
// ---------------------------------------------------------------------------

// resetMatrix is the first-order approximation (chart factors ~ I) of the exact
// error re-centring map  eps -> Local(xi_ref, phi(Exp(-delta_x), Retract(xi_ref,
// eps)))  at eps = delta_xi. For a small correction the chart factors vanish
// (error O(|delta|^2)), so it matches the exact finite difference tightly.
static Eigen::Matrix<double, 18, 18> resetMatrixFD(const TGState& xi_ref,
                                                   const Vector18& delta_xi,
                                                   const Vector18& delta_x) {
  const TGGroupElement Xd = TGGroupElement::Expmap(-delta_x);
  auto recentre = [&](const Vector18& e) {
    return traits<TGState>::Local(
        xi_ref, phi(Xd, traits<TGState>::Retract(xi_ref, e)));
  };
  const Vector18 f0 = recentre(delta_xi);
  const double h = 1e-7;
  Eigen::Matrix<double, 18, 18> J;
  for (int j = 0; j < 18; ++j) {
    Vector18 e = Vector18::Zero();
    e(j) = h;
    J.col(j) = (recentre(delta_xi + e) - f0) / h;
  }
  return J;
}

TEST(TGEqF, ResetMatrixMatchesFiniteDifferenceForSmallCorrection) {
  const TGState xi_ref = TGState::identity();
  TGEqF filter(xi_ref, defaultSigma());

  // Small correction so the omitted chart factors (O(|delta|^2)) are negligible.
  Vector18 delta_xi;
  delta_xi << 5e-4, -4e-4, 3e-4, 1e-3, -8e-4, 6e-4, 7e-4, -5e-4, 4e-4,
      1e-4, -2e-4, 15e-5, -3e-4, 2e-4, -1e-4, 0.0, 1e-4, -1e-4;
  // delta_x = pinv(Dphi0) delta_xi; at identity Dphi0 = diag(I9,-I9).
  Vector18 delta_x = delta_xi;
  delta_x.tail<9>() = -delta_xi.tail<9>();

  const Eigen::Matrix<double, 18, 18> J = filter.resetMatrix(delta_xi, delta_x);
  const Eigen::Matrix<double, 18, 18> J_fd =
      resetMatrixFD(xi_ref, delta_xi, delta_x);
  EXPECT(assert_equal((Matrix)J_fd, (Matrix)J, 1e-5));
}

TEST(TGEqF, ResetMatrixTendsToIdentityAsCorrectionVanishes) {
  TGEqF filter(TGState::identity(), defaultSigma());
  const Eigen::Matrix<double, 18, 18> J =
      filter.resetMatrix(Vector18::Zero(), Vector18::Zero());
  EXPECT(assert_equal((Matrix)Eigen::Matrix<double, 18, 18>::Identity(),
                      (Matrix)J, 1e-12));
}

// A tiny-innovation update leaves the covariance essentially Joseph-only:
// reset is a near-no-op (J ~ I) when the correction is small.
TEST(TGEqF, ResetIsNoopForTinyInnovation) {
  TGEqF with_reset(TGState::identity(), defaultSigma());
  TGEqF no_reset(TGState::identity(), defaultSigma());
  no_reset.set_reset_step(false);

  const Eigen::Vector3d z_dvl(1e-4, -1e-4, 5e-5);
  const TGEqF::Covariance3 R = 1e-3 * TGEqF::Covariance3::Identity();
  with_reset.update_dvl(z_dvl, R);
  no_reset.update_dvl(z_dvl, R);

  EXPECT(assert_equal(no_reset.errorCovariance(),
                      with_reset.errorCovariance(), 1e-6));
}

// A sizeable update at a rotated state: the reset conjugates the post-update
// covariance by J and keeps it symmetric positive-definite.
TEST(TGEqF, ResetConjugatesCovarianceAndStaysSpd) {
  const TGState xi_ref = TGState::identity();
  const TGGroupElement X0 = rotatedX0();
  TGEqF with_reset(xi_ref, defaultSigma(), X0);
  TGEqF no_reset(xi_ref, defaultSigma(), X0);
  no_reset.set_reset_step(false);

  const Eigen::Vector3d pi = with_reset.position() + Eigen::Vector3d(0.5, 0.3, -0.2);
  const TGEqF::Covariance3 R = 1e-2 * TGEqF::Covariance3::Identity();
  with_reset.update_position(pi, R, true);
  no_reset.update_position(pi, R, true);

  const Eigen::Matrix<double, 18, 18> P = with_reset.errorCovariance();
  // Symmetric.
  EXPECT(assert_equal((Matrix)P, (Matrix)P.transpose(), 1e-12));
  // Positive-definite (Cholesky succeeds).
  Eigen::LLT<Eigen::Matrix<double, 18, 18>> llt(P);
  EXPECT(llt.info() == Eigen::Success);
  // The reset actually changed something vs the no-reset twin.
  EXPECT((P - no_reset.errorCovariance()).norm() > 1e-9);
}

/* ************************************************************************* */
int main() {
  TestResult tr;
  return TestRegistry::runAllTests(tr);
}
