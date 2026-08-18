/**
 * @file  testTGEqF.cpp
 * @brief Unit tests for the TGEqF filter (predict, position/DVL/virtual-bias
 * updates, origin-chart Jacobians, reset step, input-mapped process noise).
 */
#include <CppUnitLite/TestHarness.h>
#include <gtsam/base/TestableAssertions.h>
#include <gtsam/base/numericalDerivative.h>
#include <gtsam_unstable/tg_eqf/BodyVelocityOutput.h>
#include <gtsam_unstable/tg_eqf/EqF.h>
#include <gtsam_unstable/tg_eqf/Lift.h>
#include <gtsam_unstable/tg_eqf/PositionOutput.h>
#include <gtsam_unstable/tg_eqf/Symmetry.h>
#include <gtsam_unstable/tg_eqf/VirtualBiasOutput.h>

#include <iostream>

using namespace gtsam::tgeqf;
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

static Input makeImuInput(const Eigen::Vector3d& omega,
                          const Eigen::Vector3d& accel,
                          const Eigen::Vector3d& g_vec) {
  Input u;
  u.w = omega;
  u.a = accel;
  u.v = Eigen::Vector3d::Zero();  // virtual input nu = 0 (filter feeds 0)
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
  const State xi_ref = State::identity();
  const TGEqF::Covariance18 Sigma0 = defaultSigma();
  TGEqF filter(xi_ref, Sigma0);

  EXPECT(traits<State>::Equals(xi_ref, filter.state(), kTol));
  EXPECT(traits<TGElement>::Equals(TGElement::Identity(),
                                   filter.groupEstimate(), kTol));
  EXPECT(assert_equal(Sigma0, filter.errorCovariance(), kTol));
}

TEST(TGEqF, AccessorsMatchInitialState) {
  TGEqF filter(State::identity(), defaultSigma());

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
  TGEqF filter(State::identity(), defaultSigma());
  filter.set_virtual_bias_anchor(false);  // pure predict: no anchor shrink

  const Eigen::Vector3d omega(0.05, -0.02, 0.01);
  const Eigen::Vector3d g_vec(0.0, 0.0, -9.81);
  const Eigen::Vector3d accel(0.1, -0.2, 0.05);
  const double dt = 0.1;

  const TGElement g_before = filter.groupEstimate();
  const TGEqF::Covariance18 P_before = filter.errorCovariance();

  filter.propagate(omega, accel, g_vec, defaultQc(), dt);

  EXPECT(!traits<TGElement>::Equals(g_before, filter.groupEstimate(), 1e-12));
  EXPECT(filter.errorCovariance().trace() > P_before.trace());
}

// The filter feeds the virtual input nu = 0 independent of
// the b_v estimate. With a non-zero reference b_v, propagate (anchor disabled
// for a clean predict) must equal a manual lift built with u.v = 0.
TEST(TGEqF, PropagateUsesZeroVirtualInput) {
  State xi_ref = State::identity();
  xi_ref.b_v = Eigen::Vector3d(0.1, -0.05, 0.08);  // non-zero virtual bias
  TGEqF filter(xi_ref, defaultSigma());
  filter.set_virtual_bias_anchor(false);

  const Eigen::Vector3d omega = Eigen::Vector3d::Zero();
  const Eigen::Vector3d g_vec(0.0, 0.0, -9.81);
  const Eigen::Vector3d accel(0.1, -0.2, 0.05);
  const double dt = 0.05;

  filter.propagate(omega, accel, g_vec, defaultQc(), dt);
  const TGElement g_imu = filter.groupEstimate();

  TGEqF filter2(xi_ref, defaultSigma());
  filter2.set_virtual_bias_anchor(false);
  // u.v = 0 (zero virtual input), regardless of b_v.
  const Input u = makeImuInput(omega, accel, g_vec);
  const Lift lift(u);
  const InputOrbit psi_u(u);
  filter2.template predict<1>(lift, psi_u, defaultQc(), dt);

  EXPECT(traits<TGElement>::Equals(g_imu, filter2.groupEstimate(), 1e-9));
}

// Regression: with a non-zero initial velocity, IMU propagation must advance
// the position estimate by ~v*dt (dp = v). The constant-N lift froze position
// at zero while velocity tracked, so guard the integrated position explicitly.
TEST(TGEqF, PropagateIntegratesPosition) {
  State xi0 = State::identity();
  xi0.v = Eigen::Vector3d(1.0, -0.5, 0.2);  // non-rest start
  TGEqF filter(xi0, defaultSigma());
  filter.set_virtual_bias_anchor(false);  // pure predict

  const Eigen::Vector3d g_vec(0.0, 0.0, -9.81);
  const Eigen::Vector3d accel =
      -g_vec;  // specific force of a level body at rest
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
  TGEqF filter(State::identity(), defaultSigma());

  const Eigen::Vector3d pi(2.0, -1.0, 0.5);
  const TGEqF::Covariance3 R_pos = 0.001 * TGEqF::Covariance3::Identity();

  const double err_before =
      PositionMeasurement::predict(filter.state(), pi).norm();
  EXPECT(err_before > 1.0);

  filter.update_position(pi, R_pos);

  const double err_after =
      PositionMeasurement::predict(filter.state(), pi).norm();
  EXPECT(err_after < err_before);
  EXPECT(err_after < 0.5 * err_before);
}

TEST(TGEqF, PositionUpdateMovesEstimateTowardMeasurement) {
  TGEqF filter(State::identity(), defaultSigma());

  const Eigen::Vector3d pi(1.5, 0.0, -0.25);
  const TGEqF::Covariance3 R_pos = 0.001 * TGEqF::Covariance3::Identity();

  filter.update_position(pi, R_pos);

  EXPECT((filter.position() - pi).norm() < 0.2);
}

TEST(TGEqF, PositionUpdateShrinksPositionCovariance) {
  TGEqF filter(State::identity(), defaultSigma());

  const Eigen::Vector3d pi(1.0, 0.5, -0.2);
  const TGEqF::Covariance3 R_pos = 0.01 * TGEqF::Covariance3::Identity();

  // Position lives in origin-chart tangent columns 6..8 ([R,v,p] order); use
  // errorCovariance() directly rather than covariance() (which transports to
  // the state chart and would need an extra J P J^T unpacking here).
  const double trace_before =
      filter.errorCovariance().block<3, 3>(6, 6).trace();
  filter.update_position(pi, R_pos);
  const double trace_after = filter.errorCovariance().block<3, 3>(6, 6).trace();

  EXPECT(trace_after < trace_before);
}

// ---------------------------------------------------------------------------
// DVL update
// ---------------------------------------------------------------------------

TEST(TGEqF, DvlUpdateReducesInnovation) {
  TGEqF filter(State::identity(), defaultSigma());

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
  TGEqF filter(State::identity(), defaultSigma());

  const Eigen::Vector3d z_dvl(0.8, 0.1, -0.3);
  const TGEqF::Covariance3 R_dvl = 0.001 * TGEqF::Covariance3::Identity();

  filter.update_dvl(z_dvl, R_dvl);

  EXPECT((DVLMeasurement::predict(filter.state()) - z_dvl).norm() < 0.2);
}

TEST(TGEqF, DvlUpdateShrinksVelocityCovariance) {
  TGEqF filter(State::identity(), defaultSigma());

  const Eigen::Vector3d z_dvl(0.5, -0.2, 0.3);
  const TGEqF::Covariance3 R_dvl = 0.01 * TGEqF::Covariance3::Identity();

  // Velocity lives in origin-chart tangent columns 3..5 ([R,v,p,...] order).
  const double trace_before =
      filter.errorCovariance().block<3, 3>(3, 3).trace();
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
  State xi_ref = State::identity();
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
  State xi_ref = State::identity();
  xi_ref.b_v = Eigen::Vector3d(0.2, -0.15, 0.1);
  TGEqF filter(xi_ref, defaultSigma());
  filter.set_virtual_bias_anchor(false);  // explicitly disable the default

  const Eigen::Vector3d g_vec(0.0, 0.0, -9.81);
  filter.propagate(Eigen::Vector3d::Zero(), -g_vec, g_vec, defaultQc(), 0.01);

  // No anchoring: bdot = 0 and nu = 0, so b_v is unchanged by propagation.
  EXPECT(veq(xi_ref.b_v, filter.bias_vel(), 1e-9));
}

TEST(TGEqF, AnchorOnByDefaultDrivesVirtualBiasToZeroInPropagate) {
  State xi_ref = State::identity();
  xi_ref.b_v = Eigen::Vector3d(0.2, -0.15, 0.1);
  TGEqF filter(xi_ref, defaultSigma());  // anchor on by default

  const double before = filter.bias_vel().norm();
  const Eigen::Vector3d g_vec(0.0, 0.0, -9.81);
  filter.propagate(Eigen::Vector3d::Zero(), -g_vec, g_vec, defaultQc(), 0.01);

  EXPECT(filter.bias_vel().norm() < 0.5 * before);
}

// A configured anchor noise must survive an off/on toggle: calling
// set_virtual_bias_anchor(false) to temporarily disable the anchor, then
// set_virtual_bias_anchor(true) with no R_vb argument, must not silently
// revert to the 1e-4 default.
TEST(TGEqF, AnchorNoiseSurvivesToggle) {
  State xi_ref = State::identity();
  xi_ref.b_v = Eigen::Vector3d(0.2, -0.15, 0.1);
  const TGEqF::Covariance3 tight = 1e-6 * TGEqF::Covariance3::Identity();

  TGEqF toggled(xi_ref, defaultSigma());
  toggled.set_virtual_bias_anchor(true, tight);
  toggled.set_virtual_bias_anchor(false);  // must NOT reset the noise
  toggled.set_virtual_bias_anchor(true);

  TGEqF direct(xi_ref, defaultSigma());
  direct.set_virtual_bias_anchor(true, tight);

  const Eigen::Vector3d g_vec(0.0, 0.0, -9.81);
  toggled.propagate(Eigen::Vector3d::Zero(), -g_vec, g_vec, defaultQc(), 0.01);
  direct.propagate(Eigen::Vector3d::Zero(), -g_vec, g_vec, defaultQc(), 0.01);

  EXPECT(traits<TGElement>::Equals(direct.groupEstimate(),
                                   toggled.groupEstimate(), 1e-12));
  EXPECT(
      assert_equal(direct.errorCovariance(), toggled.errorCovariance(), 1e-12));
}

TEST(TGEqF, VirtualBiasUpdateShrinksVirtualBiasCovariance) {
  TGEqF filter(State::identity(), defaultSigma());

  // b_v lives in origin-chart tangent columns 15..17 ([R,v,p,b_w,b_a,b_v]).
  const double trace_before =
      filter.errorCovariance().block<3, 3>(15, 15).trace();
  filter.update_virtual_bias(0.01 * TGEqF::Covariance3::Identity());
  const double trace_after =
      filter.errorCovariance().block<3, 3>(15, 15).trace();

  EXPECT(trace_after < trace_before);
}

// ---------------------------------------------------------------------------
// End-to-end
// ---------------------------------------------------------------------------

TEST(TGEqF, EndToEndPropagateAndUpdates) {
  TGEqF filter(State::identity(), defaultSigma());

  const Eigen::Vector3d g_vec(0.0, 0.0, -9.81);
  const Eigen::Vector3d omega(0.02, 0.0, 0.01);
  const double dt = 0.05;

  // A roughly level body at rest reads specific force = -g (not +g): the
  // accelerometer measures the reaction to gravity, not free-fall.
  filter.propagate(omega, -g_vec, g_vec, defaultQc(), dt);

  const Eigen::Vector3d pi = filter.position() + Eigen::Vector3d(0.5, 0.0, 0.0);
  const Eigen::Vector3d z_dvl = filter.attitude().unrotate(
      filter.velocity() + Eigen::Vector3d(0.1, 0.0, 0.0));

  const TGEqF::Covariance3 R = 0.01 * TGEqF::Covariance3::Identity();
  filter.update_position(pi, R);
  filter.update_dvl(z_dvl, R);

  EXPECT(PositionMeasurement::predict(filter.state(), pi).norm() < 0.3);
  EXPECT(DVLMeasurement::innovation(z_dvl, filter.state()).norm() < 0.15);
}

TEST(TGEqF, PositionOutputEquivarianceHoldsAfterUpdate) {
  const TGElement X = TGElement::Identity();
  State xi = State::identity();
  xi.p = Eigen::Vector3d(0.3, -0.1, 0.2);

  TGEqF filter(State::identity(), defaultSigma());
  const Eigen::Vector3d pi(1.0, 0.2, -0.4);
  filter.update_position(pi, 0.005 * TGEqF::Covariance3::Identity());

  const Eigen::Vector3d y = PositionMeasurement::predict(filter.state(), pi);
  const Eigen::Vector3d lhs =
      PositionMeasurement::predict(phi(X, filter.state()), pi);
  const Eigen::Vector3d rhs = PositionMeasurement::output_action(X, y);
  EXPECT(veq(lhs, rhs, 1e-7));
}

// ---------------------------------------------------------------------------
// Origin-chart measurement Jacobians away from the reference
//
// Build the filter at a state rotated/translated far from xi_ref = identity
// (Rz(90 deg), v=(2,0,0), p=(5,-3,1)) and check that the Jacobian the filter
// actually consumes is the origin-chart one (H_est * Dphi_g), not the
// chart-at-estimate H_est.
// ---------------------------------------------------------------------------

namespace {

// Group element X0 with phi(X0, identity) = (Rz(90), (2,0,0), (5,-3,1), 0).
TGElement rotatedX0() {
  TGElement X;
  X.R = Rot3::Rz(M_PI / 2);
  X.v = Eigen::Vector3d(2.0, 0.0, 0.0);
  X.p = Eigen::Vector3d(5.0, -3.0, 1.0);
  X.a = {Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero(),
         Eigen::Vector3d::Zero()};
  return X;
}

Eigen::Matrix<double, 18, 18> diffeoJacobian(const TGElement& X,
                                             const State& xi) {
  Eigen::Matrix<double, 18, 18> J;
  const TGSymmetry::Diffeomorphism phi_X(X);
  phi_X(xi, &J);
  return J;
}

}  // namespace

// C0 is stated in the origin output space, so R_X C0 is the FD of the map
// eps -> h(phi(X0, Retract(xi_ref, eps))), which reads the measurement in the
// body frame of the estimate. Composing the chart-at-estimate Jacobian with
// Dphi_g reaches the same matrix, and (for this configuration) its rotation
// block must vanish.
TEST(TGEqF, DvlOriginChartJacobianMatchesFiniteDifference) {
  const State xi_ref = State::identity();
  const TGElement X0 = rotatedX0();
  const State xi_hat = phi(X0, xi_ref);

  // Finite-difference the origin-chart measurement map.
  const Eigen::Vector3d f0 = DVLMeasurement::predict(xi_hat);
  const double h = 1e-6;
  Eigen::Matrix<double, 3, 18> H_fd;
  for (int j = 0; j < 18; ++j) {
    Eigen::Matrix<double, 18, 1> e = Eigen::Matrix<double, 18, 1>::Zero();
    e(j) = h;
    const State xi_eps = phi(X0, traits<State>::Retract(xi_ref, e));
    H_fd.col(j) = (DVLMeasurement::predict(xi_eps) - f0) / h;
  }

  const Eigen::Matrix<double, 3, 18> H_origin =
      X0.R.matrix().transpose() * DVLMeasurement::jacobian_C0(xi_ref);

  EXPECT(assert_equal((Matrix)H_fd, (Matrix)H_origin, 1e-5));

  // The reference velocity is zero here, so the rotation block is skew(0). The
  // chart-at-estimate Jacobian instead carries a skew(R^T v) block, which is
  // what the transport removes.
  EXPECT(assert_equal((Matrix)Eigen::Matrix3d::Zero(),
                      (Matrix)H_origin.block<3, 3>(0, 0), 1e-5));
}

// A DVL update at a rotated state must move the velocity estimate in the
// correct *global* direction. A +0.1 m/s global-y offset should raise the
// global-y velocity, not leak into x/z (which the un-transported Jacobian did).
TEST(TGEqF, DvlUpdateMovesVelocityInGlobalFrameAtRotatedState) {
  const State xi_ref = State::identity();
  const TGElement X0 = rotatedX0();
  // Sigma0 is the prior at the estimate, where the 2 m/s speed makes attitude
  // twice as sensitive as velocity to a body-velocity measurement. An isotropic
  // prior would therefore attribute most of the innovation to yaw, so pin the
  // attitude tightly to isolate the velocity correction under test.
  TGEqF::Covariance18 Sigma0 = defaultSigma();
  Sigma0.block<3, 3>(0, 0) = 1e-8 * Eigen::Matrix3d::Identity();
  TGEqF filter(xi_ref, Sigma0, X0);

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
// half-coupled code moved it mostly +x).
TEST(TGEqF, PositionUpdateMovesEstimateInGlobalFrameAtRotatedState) {
  const State xi_ref = State::identity();
  const TGElement X0 = rotatedX0();
  TGEqF filter(xi_ref, defaultSigma(), X0);

  const Eigen::Vector3d p_before = filter.position();
  const Eigen::Vector3d pi = p_before + Eigen::Vector3d(0.0, 0.5, 0.0);

  filter.update_position(pi, 1e-4 * TGEqF::Covariance3::Identity());

  const Eigen::Vector3d dp = filter.position() - p_before;
  EXPECT(dp.y() > 0.4);
  EXPECT(std::abs(dp.x()) < 0.1);
  EXPECT(std::abs(dp.z()) < 0.1);
}

// ---------------------------------------------------------------------------
// The constructor takes Sigma0 in the tangent chart at the initial estimate and
// transports it to the origin chart as J^-1 Sigma0 J^-T (J = d phi(g,.)/dxi),
// the exact inverse of the J P J^T transport covariance() applies forward. At
// rotatedX0() with an anisotropic Sigma0, J is far from a matrix for which the
// transpose-swapped congruence agrees, so a wrong direction shows up as an O(1)
// discrepancy rather than numerical noise.
// ---------------------------------------------------------------------------

TEST(TGEqF, ConstructorTransportsCovarianceToOriginChart) {
  const State xi_ref = State::identity();
  const TGElement X0 = rotatedX0();
  TGEqF::Covariance18 Sigma0 = TGEqF::Covariance18::Zero();
  for (int i = 0; i < 18; ++i) Sigma0(i, i) = 1e-4 * (1.0 + i);  // anisotropic
  TGEqF filter(xi_ref, Sigma0, X0);

  // Round trip: covariance() undoes exactly what the constructor applied.
  EXPECT(assert_equal((Matrix)Sigma0, (Matrix)filter.covariance(), 1e-9));

  const Eigen::Matrix<double, 18, 18> J_inv =
      diffeoJacobian(X0, xi_ref).inverse();
  const Eigen::Matrix<double, 18, 18> P_expected =
      J_inv * Sigma0 * J_inv.transpose();
  EXPECT(
      assert_equal((Matrix)P_expected, (Matrix)filter.errorCovariance(), 1e-9));

  // Guard the transport direction against a future transpose slip.
  const Eigen::Matrix<double, 18, 18> P_wrong_direction =
      J_inv.transpose() * Sigma0 * J_inv;
  EXPECT((P_wrong_direction - P_expected).norm() > 1e-3);
}

// With X0 = identity the transport is the identity, so Sigma0 must land in the
// origin chart untouched.
TEST(TGEqF, ConstructorTransportIsIdentityAtIdentityGroupElement) {
  const State xi_ref = State::identity();
  const TGEqF::Covariance18 Sigma0 = defaultSigma();
  TGEqF filter(xi_ref, Sigma0, TGElement::Identity());

  EXPECT(assert_equal(Sigma0, filter.errorCovariance(), 1e-12));
  EXPECT(assert_equal((Matrix)Sigma0, (Matrix)filter.covariance(), 1e-12));
}

/* ************************************************************************* */
namespace reset_step {

// Orbit Jacobian Dphi0 at the group identity. It is invertible (block
// triangular with a rotation and -I on the diagonal), so delta_x = pinv(Dphi0)
// delta_xi inverts to delta_xi = Dphi0 delta_x.
Eigen::Matrix<double, 18, 18> orbitJacobian0(const State& xi_ref) {
  Eigen::Matrix<double, 18, 18> H;
  const TGSymmetry::Orbit orbit(xi_ref);
  orbit(TGElement::Identity(), &H);
  return H;
}

// The error re-centring map the group correction applies to the error state.
Vector18 recentre(const State& xi_ref, const TGElement& Xd,
                  const Vector18& eps) {
  return traits<State>::Local(xi_ref,
                              phi(Xd, traits<State>::Retract(xi_ref, eps)));
}

// Independent oracle: central-difference derivative of the re-centring map at
// eps = delta_xi. Shares no code with TGEqF::resetMatrix.
Eigen::Matrix<double, 18, 18> numericalResetJacobian(const State& xi_ref,
                                                     const Vector18& delta_xi,
                                                     const Vector18& delta_x) {
  const TGElement Xd = TGElement::Expmap(-delta_x);
  return numericalDerivative11<Vector18, Vector18>(
      [&](const Vector18& eps) { return recentre(xi_ref, Xd, eps); }, delta_xi);
}

// A correction large enough that the chart factors (the right Jacobian and its
// inverse on the rotation slot) are far from the identity. A 1e-4-sized
// correction makes them identity to 1e-8, which hides a left/right mix-up.
Vector18 bigDeltaXi() {
  Vector18 delta_xi;
  delta_xi << 0.35, -0.25, 0.4, 0.2, -0.15, 0.1, 0.3, -0.2, 0.15, 0.05, -0.04,
      0.03, -0.06, 0.02, -0.01, 0.02, -0.03, 0.01;
  return delta_xi;
}

// A reference state away from the manifold identity, so the chart factors and
// the fiber adjoint are exercised at a non-trivial origin.
State offOriginRef() {
  State xi_ref;
  xi_ref.R = Rot3::Rz(0.7) * Rot3::Ry(-0.3);
  xi_ref.v = Eigen::Vector3d(1.0, -0.5, 0.2);
  xi_ref.p = Eigen::Vector3d(-2.0, 3.0, 0.5);
  xi_ref.b_w = Eigen::Vector3d(0.01, -0.02, 0.03);
  xi_ref.b_a = Eigen::Vector3d(-0.05, 0.04, 0.02);
  xi_ref.b_v = Eigen::Vector3d(0.1, -0.08, 0.06);
  return xi_ref;
}

TGEqF::Covariance18 anisotropicSigma0() {
  TGEqF::Covariance18 Sigma0 = TGEqF::Covariance18::Zero();
  for (int i = 0; i < 18; ++i) Sigma0(i, i) = 1e-4 * (1.0 + i);
  return Sigma0;
}

// One position update at rotatedX0(), run on twins with and without the reset,
// plus the transport rebuilt from the correction the update actually applied.
// Xd is recovered from the public group estimates alone: X_after = Exp(delta_x)
// X_before, so Xd = Exp(-delta_x) = X_before * X_after^-1.
struct UpdatePair {
  Eigen::Matrix<double, 18, 18> P_with, P_without, J;
};

UpdatePair positionUpdatePair(const TGEqF::Covariance18& Sigma0,
                              double offset_scale = 1.0) {
  const State xi_ref = State::identity();
  const TGElement X0 = rotatedX0();
  TGEqF with_reset(xi_ref, Sigma0, X0);
  TGEqF without_reset(xi_ref, Sigma0, X0);
  without_reset.set_reset_step(false);

  const TGElement X_before = with_reset.groupEstimate();
  const Eigen::Vector3d pi =
      with_reset.position() + offset_scale * Eigen::Vector3d(0.5, 0.3, -0.2);
  const TGEqF::Covariance3 R = 1e-2 * TGEqF::Covariance3::Identity();
  with_reset.update_position(pi, R);
  without_reset.update_position(pi, R);

  const TGElement Xd = X_before * with_reset.groupEstimate().inverse();
  const Vector18 delta_x = -Xd.Logmap();
  const Vector18 delta_xi = orbitJacobian0(xi_ref) * delta_x;
  return {with_reset.errorCovariance(), without_reset.errorCovariance(),
          with_reset.resetMatrix(delta_xi, delta_x)};
}

// The analytic Jacobian matches a central-difference derivative of the exact
// re-centring map, at a correction large enough for the chart factors to bite,
// both at the manifold identity and at an off-origin reference state.
//
// Both a matched and an unmatched (delta_xi, delta_x) pair are exercised. The
// matched pair is what the filter passes, but it re-centres the error to nearly
// zero, which leaves the output-side chart factor at ~I and hides a left/right
// Jacobian mix-up there. The unmatched pair lands far from zero and exposes it.
TEST(TGEqF, ResetMatrixMatchesNumericalDerivative) {
  for (const State& xi_ref : {State::identity(), offOriginRef()}) {
    TGEqF filter(xi_ref, defaultSigma());
    const Vector18 delta_xi = bigDeltaXi();
    const Vector18 matched = orbitJacobian0(xi_ref).inverse() * delta_xi;
    const Vector18 unmatched = -0.5 * matched;

    for (const Vector18& delta_x : {matched, unmatched}) {
      EXPECT(assert_equal(
          (Matrix)numericalResetJacobian(xi_ref, delta_xi, delta_x),
          (Matrix)filter.resetMatrix(delta_xi, delta_x), 1e-6));
    }
  }
}

// The Jacobian belongs at eps = delta_xi, the post-update mean of the error,
// not at eps = 0. Linearizing at zero is the classic reset bug: it survives
// every small-correction check but is wrong by O(|delta_xi|).
TEST(TGEqF, ResetMatrixLinearizesAtCorrectionNotZero) {
  const State xi_ref = State::identity();
  TGEqF filter(xi_ref, defaultSigma());
  const Vector18 delta_xi = bigDeltaXi();
  const Vector18 delta_x = orbitJacobian0(xi_ref).inverse() * delta_xi;

  const Eigen::Matrix<double, 18, 18> J_num =
      numericalResetJacobian(xi_ref, delta_xi, delta_x);
  EXPECT(assert_equal((Matrix)J_num,
                      (Matrix)filter.resetMatrix(delta_xi, delta_x), 1e-6));
  EXPECT((filter.resetMatrix(Vector18::Zero(), delta_x) - J_num).norm() > 1e-3);
}

// The fiber block of the transport is the SE_2(3) inverse adjoint of the
// correction. This is the path that shears the bias covariance by the velocity
// and position parts of a navigation correction -- the block whose omission
// (reset off) inflated the accel-bias error in the aiding study.
TEST(TGEqF, ResetMatrixBiasBlockIsFiberAdjoint) {
  const State xi_ref = State::identity();
  TGEqF filter(xi_ref, defaultSigma());
  const Vector18 delta_xi = bigDeltaXi();
  const Vector18 delta_x = orbitJacobian0(xi_ref).inverse() * delta_xi;
  const TGElement Xd = TGElement::Expmap(-delta_x);

  const Eigen::Matrix<double, 9, 9> Ad_inv = detail::Ad_SE23_inv(Xd);
  EXPECT(assert_equal(
      (Matrix)Ad_inv,
      (Matrix)filter.resetMatrix(delta_xi, delta_x).block<9, 9>(9, 9), 1e-9));
  // The coupling is real, not a dressed-up identity.
  EXPECT((Ad_inv - Eigen::Matrix<double, 9, 9>::Identity()).norm() > 0.1);
}

TEST(TGEqF, ResetMatrixTendsToIdentityAsCorrectionVanishes) {
  TGEqF filter(State::identity(), defaultSigma());
  const Eigen::Matrix<double, 18, 18> J =
      filter.resetMatrix(Vector18::Zero(), Vector18::Zero());
  EXPECT(assert_equal((Matrix)Eigen::Matrix<double, 18, 18>::Identity(),
                      (Matrix)J, 1e-12));
}

// A correction confined to the fiber leaves the transport at exactly identity:
// Xd's SE_2(3) part is the identity, so both the chart factors and the fiber
// adjoint collapse. This is why the b_v anchor is free whenever the bias block
// is still uncorrelated with navigation.
TEST(TGEqF, ResetIsIdentityForPureFiberCorrection) {
  const State xi_ref = State::identity();
  TGEqF filter(xi_ref, defaultSigma());

  Vector18 delta_x = Vector18::Zero();
  delta_x.tail<9>() << 0.2, -0.15, 0.1, 0.05, 0.04, -0.03, 0.02, -0.01, 0.03;
  const Vector18 delta_xi = orbitJacobian0(xi_ref) * delta_x;

  EXPECT(assert_equal((Matrix)Eigen::Matrix<double, 18, 18>::Identity(),
                      (Matrix)filter.resetMatrix(delta_xi, delta_x), 1e-12));
}

// The map resetMatrix linearizes is the one the update actually applies to the
// error. With Xd recovered from the group estimates alone, the post-update
// error state must equal phi(Xd, error state before) exactly -- this pins the
// sign of Xd and the side of the base class's Compose(Exp(delta_x), g_), which
// no derivative oracle can see because they all inherit the same Xd.
TEST(TGEqF, ResetTracksGroupCorrectionAppliedByUpdate) {
  const State xi_ref = State::identity();
  TGEqF filter(xi_ref, defaultSigma(), rotatedX0());

  // A truth the estimate is genuinely wrong about, so the error is non-zero.
  State xi_true = filter.state();
  xi_true.v += Eigen::Vector3d(0.1, 0.05, -0.1);
  xi_true.p += Eigen::Vector3d(0.4, -0.2, 0.3);

  const State e_before = filter.errorState(xi_true);
  const TGElement X_before = filter.groupEstimate();

  const TGEqF::Covariance3 R = 1e-2 * TGEqF::Covariance3::Identity();
  filter.update_position(xi_true.p + Eigen::Vector3d(0.05, -0.03, 0.02), R);

  const TGElement Xd = X_before * filter.groupEstimate().inverse();
  EXPECT(traits<State>::Equals(phi(Xd, e_before), filter.errorState(xi_true),
                               1e-9));
}

// End to end: the covariance the filter ships after an update is exactly the
// transport of its Joseph-updated covariance by the reset Jacobian. Closes the
// loop between resetMatrix and updateWithReset.
TEST(TGEqF, ResetTransportsJosephCovarianceByObservedCorrection) {
  const UpdatePair u = positionUpdatePair(anisotropicSigma0());
  EXPECT(assert_equal((Matrix)(u.J * u.P_without * u.J.transpose()),
                      (Matrix)u.P_with, 1e-9));
}

// Guard the transport direction against a transpose slip: with an anisotropic
// covariance at a rotated state, J^T P J is O(1) away from J P J^T.
TEST(TGEqF, ResetTransportDirectionGuard) {
  const UpdatePair u = positionUpdatePair(anisotropicSigma0());
  EXPECT((u.J.transpose() * u.P_without * u.J - u.P_with).norm() > 1e-3);
}

// The reset is a genuine first-order term in the correction: halving the
// innovation halves its effect on the covariance. A test that only asserts
// "small" would pass just as well if the reset were a no-op or O(1).
TEST(TGEqF, ResetDifferenceScalesWithCorrection) {
  const UpdatePair full = positionUpdatePair(defaultSigma(), 1.0);
  const UpdatePair half = positionUpdatePair(defaultSigma(), 0.5);

  const double d_full = (full.P_with - full.P_without).norm();
  const double d_half = (half.P_with - half.P_without).norm();
  EXPECT(d_full > 1e-9);
  EXPECT(d_half / d_full > 0.4);
  EXPECT(d_half / d_full < 0.6);
}

// The b_v anchor fires once per propagate in production, so it has to go
// through the same transport. Once propagation has correlated the bias block
// with navigation the anchor's correction is no longer pure fiber, and the
// reset changes the covariance.
TEST(TGEqF, ResetAppliesToVirtualBiasAnchorUpdate) {
  State xi_ref = State::identity();
  xi_ref.b_v = Eigen::Vector3d(0.2, -0.15, 0.1);
  TGEqF with_reset(xi_ref, defaultSigma());
  TGEqF without_reset(xi_ref, defaultSigma());
  without_reset.set_reset_step(false);

  const Eigen::Vector3d omega(0.05, -0.02, 0.01);
  const Eigen::Vector3d accel(0.1, -0.2, 0.05);
  const Eigen::Vector3d g_vec(0.0, 0.0, -9.81);
  for (int k = 0; k < 10; ++k) {
    with_reset.propagate(omega, accel, g_vec, defaultQc(), 0.01);
    without_reset.propagate(omega, accel, g_vec, defaultQc(), 0.01);
  }

  EXPECT(
      (with_reset.errorCovariance() - without_reset.errorCovariance()).norm() >
      1e-12);
}

// The reset is on unless it is explicitly switched off. The aiding-study
// configs disable it, so guard the default against a silent flip.
TEST(TGEqF, ResetStepOnByDefault) {
  const State xi_ref = State::identity();
  const TGElement X0 = rotatedX0();
  TGEqF by_default(xi_ref, anisotropicSigma0(), X0);
  TGEqF explicitly_on(xi_ref, anisotropicSigma0(), X0);
  TGEqF turned_off(xi_ref, anisotropicSigma0(), X0);
  explicitly_on.set_reset_step(true);
  turned_off.set_reset_step(false);

  const Eigen::Vector3d pi =
      by_default.position() + Eigen::Vector3d(0.5, 0.3, -0.2);
  const TGEqF::Covariance3 R = 1e-2 * TGEqF::Covariance3::Identity();
  by_default.update_position(pi, R);
  explicitly_on.update_position(pi, R);
  turned_off.update_position(pi, R);

  EXPECT(assert_equal(explicitly_on.errorCovariance(),
                      by_default.errorCovariance(), 1e-12));
  EXPECT((turned_off.errorCovariance() - by_default.errorCovariance()).norm() >
         1e-9);
}

// A sizeable update at a rotated state: the reset conjugates the post-update
// covariance by J and keeps it symmetric positive-definite.
TEST(TGEqF, ResetConjugatesCovarianceAndStaysSpd) {
  const UpdatePair u = positionUpdatePair(defaultSigma());
  EXPECT(assert_equal((Matrix)u.P_with, (Matrix)u.P_with.transpose(), 1e-12));
  Eigen::LLT<Eigen::Matrix<double, 18, 18>> llt(u.P_with);
  EXPECT(llt.info() == Eigen::Success);
  EXPECT((u.P_with - u.P_without).norm() > 1e-9);
}

}  // namespace reset_step
/* ************************************************************************* */

// Min eigenvalue of a symmetric matrix (SPD check). The 30 s scenario
// regressions (biased-IMU turn) live in testTGEqFRegression.cpp; this copy is
// for InputNoiseCovSymmetricSpdAtRotatedState below.
static double minEig(const Eigen::MatrixXd& M) {
  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(M);
  return es.eigenvalues().minCoeff();
}

// ---------------------------------------------------------------------------
// Input-mapped process noise: Qc_eff = B Sigma B^T, B = Dphi0 * d(Lambda)/d(u)
// ---------------------------------------------------------------------------

// At the origin (g_ = identity, b_ref = 0) the lift differential has no
// state/transport coupling, so Qc_eff is block-diagonal: the PSDs land on the
// attitude (gyro), velocity (accel), gyro-bias (gyro RW) and accel-bias
// (accel RW) blocks. Position and virtual-bias rows get no process noise, and
// there is no gyro->bias coupling (that needs b_ref != 0; an inherent EqF vs
// IEKF difference).
TEST(TGEqF, InputNoiseCovBlockDiagonalAtOrigin) {
  TGEqF filter(State::identity(), defaultSigma());  // g_ = I, b_ref = 0
  ImuNoise nz;
  nz.gyro = 2e-4;
  nz.accel = 3e-3;
  nz.gyro_rw = 1e-6;
  nz.accel_rw = 5e-5;
  const Eigen::Vector3d g_vec(0.0, 0.0, -9.81);
  const TGEqF::Covariance18 Q = filter.inputNoiseCov(
      Eigen::Vector3d(0.1, -0.05, 0.02), -g_vec, g_vec, nz);

  const Eigen::Matrix3d I3 = Eigen::Matrix3d::Identity();
  EXPECT(
      assert_equal((Matrix)(nz.gyro * I3), (Matrix)Q.block<3, 3>(0, 0), 1e-6));
  EXPECT(
      assert_equal((Matrix)(nz.accel * I3), (Matrix)Q.block<3, 3>(3, 3), 1e-6));
  EXPECT(assert_equal((Matrix)(nz.gyro_rw * I3), (Matrix)Q.block<3, 3>(9, 9),
                      1e-6));
  EXPECT(assert_equal((Matrix)(nz.accel_rw * I3), (Matrix)Q.block<3, 3>(12, 12),
                      1e-6));
  // No position, virtual-bias, or gyro->bias process noise at b_ref = 0.
  EXPECT(assert_equal((Matrix)Eigen::Matrix3d::Zero(),
                      (Matrix)Q.block<3, 3>(6, 6), 1e-6));
  EXPECT(assert_equal((Matrix)Eigen::Matrix3d::Zero(),
                      (Matrix)Q.block<3, 3>(15, 15), 1e-6));
  EXPECT(assert_equal((Matrix)Eigen::Matrix3d::Zero(),
                      (Matrix)Q.block<3, 3>(9, 0), 1e-6));
  EXPECT(assert_equal((Matrix)Q, (Matrix)Q.transpose(), 1e-12));
}

// The ImuNoise propagate overload must equal a raw-Qc predict fed the same
// inputNoiseCov() result (i.e. it is exactly inputNoiseCov + Base::predict).
TEST(TGEqF, PropagateImuNoiseMatchesManualInputCov) {
  ImuNoise nz;
  nz.gyro = 1e-4;
  nz.accel = 1e-3;
  nz.gyro_rw = 1e-7;
  nz.accel_rw = 1e-6;
  const Eigen::Vector3d g_vec(0.0, 0.0, -9.81);
  const Eigen::Vector3d omega(0.05, -0.02, 0.01);
  const Eigen::Vector3d accel(0.1, -0.2, 9.85);
  const double dt = 0.05;

  TGEqF f1(State::identity(), defaultSigma());
  f1.set_virtual_bias_anchor(false);  // isolate the predict
  const double trace_before = f1.errorCovariance().trace();
  f1.propagate(omega, accel, g_vec, nz, dt);
  EXPECT(f1.errorCovariance().trace() > trace_before);  // noise grew it

  TGEqF f2(State::identity(), defaultSigma());
  f2.set_virtual_bias_anchor(false);
  const TGEqF::Covariance18 Qc_eff = f2.inputNoiseCov(omega, accel, g_vec, nz);
  f2.propagate(omega, accel, g_vec, Qc_eff, dt);

  EXPECT(assert_equal(f2.errorCovariance(), f1.errorCovariance(), 1e-10));
  EXPECT(
      traits<TGElement>::Equals(f2.groupEstimate(), f1.groupEstimate(), 1e-10));
}

// Away from the origin the input-orbit transport makes Qc_eff non-trivial; it
// must stay symmetric and positive-semidefinite.
TEST(TGEqF, InputNoiseCovSymmetricSpdAtRotatedState) {
  TGEqF filter(State::identity(), defaultSigma(), rotatedX0());
  ImuNoise nz;
  nz.gyro = 1e-3;
  nz.accel = 1e-3;
  nz.gyro_rw = 1e-6;
  nz.accel_rw = 1e-6;
  const Eigen::Vector3d g_vec(0.0, 0.0, -9.81);
  const TGEqF::Covariance18 Q =
      filter.inputNoiseCov(Eigen::Vector3d(0.1, 0.0, 0.0), -g_vec, g_vec, nz);

  EXPECT(assert_equal((Matrix)Q, (Matrix)Q.transpose(), 1e-10));
  // PSD: eigenvalues >= 0 (the noise has rank 12, so allow a tiny tolerance).
  EXPECT(minEig(Q) > -1e-9);
}

// The cached analytic factorization B = input_lift_ * blkdiag(Ad,Ad) * Ssel
// must reproduce a brute-force numerical derivative of the lift map
// Dphi0 * Lambda(xi_ref, psi_{Xhat^-1}(u)), at a non-identity group estimate.
// Because B is affine in the input, the measurement is irrelevant to B, so any
// u is a valid linearization point.
TEST(TGEqF, InputNoiseCovMatchesNumericalLiftDifferential) {
  TGEqF filter(State::identity(), defaultSigma(), rotatedX0());
  ImuNoise nz;
  nz.gyro = 1e-3;
  nz.accel = 2e-3;
  nz.gyro_rw = 5e-6;
  nz.accel_rw = 7e-6;
  const Eigen::Vector3d g_vec(0.0, 0.0, -9.81);
  const Eigen::Vector3d omega(0.1, -0.05, 0.02);
  const Eigen::Vector3d accel(0.3, 0.2, 9.6);

  // Brute-force reference: forward-difference B over (w, a, tau_w, tau_a).
  const State xi_ref = State::identity();  // filter's fixed reference origin
  const TGElement Xhat_inv = filter.groupEstimate().inverse();
  Eigen::Matrix<double, 18, 18> Dphi0;
  const TGSymmetry::Orbit orbit(xi_ref);
  orbit(TGElement::Identity(), &Dphi0);

  Input u;
  u.w = omega;
  u.a = accel;
  u.g_vec = g_vec;
  auto lambda_origin = [&](const Input& uu) {
    return Eigen::Matrix<double, 18, 1>(Dphi0 *
                                        Lift(InputOrbit(uu)(Xhat_inv))(xi_ref));
  };
  const Eigen::Matrix<double, 18, 1> L0 = lambda_origin(u);
  const double h = 1e-7;
  Eigen::Matrix<double, 18, 12> B_num;
  for (int j = 0; j < 12; ++j) {
    Input uu = u;
    Eigen::Vector3d* slot[4] = {&uu.w, &uu.a, &uu.tau_w, &uu.tau_a};
    (*slot[j / 3])(j % 3) += h;
    B_num.col(j) = (lambda_origin(uu) - L0) / h;
  }
  Eigen::Matrix<double, 12, 12> Sigma = Eigen::Matrix<double, 12, 12>::Zero();
  Sigma.block<3, 3>(0, 0) = nz.gyro * Eigen::Matrix3d::Identity();
  Sigma.block<3, 3>(3, 3) = nz.accel * Eigen::Matrix3d::Identity();
  Sigma.block<3, 3>(6, 6) = nz.gyro_rw * Eigen::Matrix3d::Identity();
  Sigma.block<3, 3>(9, 9) = nz.accel_rw * Eigen::Matrix3d::Identity();
  const TGEqF::Covariance18 Q_num = B_num * Sigma * B_num.transpose();

  const TGEqF::Covariance18 Q = filter.inputNoiseCov(omega, accel, g_vec, nz);
  EXPECT(assert_equal((Matrix)Q_num, (Matrix)Q, 1e-6));

  // B is measurement-independent: a different (w, a, g) yields the same Qc.
  const TGEqF::Covariance18 Q2 = filter.inputNoiseCov(
      Eigen::Vector3d(-0.4, 0.9, 0.05), Eigen::Vector3d(1.0, -2.0, 8.0),
      Eigen::Vector3d(0.1, 0.2, -9.7), nz);
  EXPECT(assert_equal((Matrix)Q, (Matrix)Q2, 1e-12));
}

/* ************************************************************************* */
namespace depth_update {

// 1x1 depth noise matrix from a stddev in metres.
TGEqF::Covariance1 depthNoise(double sigma) {
  TGEqF::Covariance1 R;
  R(0, 0) = sigma * sigma;
  return R;
}

// The pseudo-position update_depth builds internally: the estimate's horizontal
// position with the measured depth substituted for z.
Eigen::Vector3d pseudoPosition(const TGEqF& filter, double z_depth) {
  Eigen::Vector3d p = filter.position();
  p.z() = z_depth;
  return p;
}

// A single depth update pulls the vertical estimate most of the way to the
// measurement when the depth noise is much tighter than the prior.
TEST(TGEqF, DepthUpdateCorrectsVerticalPosition) {
  TGEqF filter(State::identity(), defaultSigma());

  const double z_depth = -0.8;
  filter.update_depth(z_depth, depthNoise(0.01));

  EXPECT(std::abs(filter.position().z() - z_depth) < 0.1);
}

// The inflated horizontal variance keeps a depth update off the x and y axes:
// with an uncorrelated prior the horizontal motion is orders of magnitude
// smaller than the vertical correction the update is there to make.
TEST(TGEqF, DepthUpdateLeavesHorizontalNearlyUnchanged) {
  TGEqF filter(State::identity(), defaultSigma());

  const Eigen::Vector3d before = filter.position();
  filter.update_depth(-0.8, depthNoise(0.01));
  const Eigen::Vector3d delta = filter.position() - before;

  EXPECT(std::abs(delta.z()) > 0.5);
  EXPECT(std::abs(delta.x()) < 0.02);
  EXPECT(std::abs(delta.y()) < 0.02);
}

// Only the vertical position covariance is informed: the horizontal blocks are
// left essentially where they were.
TEST(TGEqF, DepthUpdateShrinksVerticalCovariance) {
  TGEqF filter(State::identity(), defaultSigma());
  // Disable the reset/conjugation step for this test so we observe the
  // pure measurement-induced covariance change (Joseph form) without the
  // reset transport J P J^T, which can spread vertical shrinkage into the
  // horizontal blocks.
  filter.set_reset_step(false);

  // Origin-chart tangent order [att, vel, pos, ...], so position is 6..8.
  const Eigen::Matrix<double, 18, 18> before = filter.errorCovariance();
  filter.update_depth(-0.8, depthNoise(0.01));
  const Eigen::Matrix<double, 18, 18> after = filter.errorCovariance();

  const double drop_z = before(8, 8) - after(8, 8);
  EXPECT(drop_z > 0.0);

  EXPECT(std::abs(before(6, 6) - after(6, 6)) < 0.5 * drop_z);
  EXPECT(std::abs(before(7, 7) - after(7, 7)) < 0.5 * drop_z);
}

// update_depth is exactly the pseudo-position update it documents: the same
// state and covariance as calling update_position with [p_x, p_y, z_depth] and
// the matching anisotropic noise. Guards the delegate against drifting apart.
TEST(TGEqF, DepthUpdateMatchesEquivalentPositionUpdate) {
  const TGElement X0 = rotatedX0();
  TGEqF via_depth(State::identity(), defaultSigma(), X0);
  TGEqF via_position(State::identity(), defaultSigma(), X0);

  const double z_depth = 0.4;
  const double sigma_z = 0.02;
  via_depth.update_depth(z_depth, depthNoise(sigma_z));

  TGEqF::Covariance3 R_pseudo = TGEqF::Covariance3::Zero();
  R_pseudo(0, 0) = TGEqF::kDefaultHorizontalVariance;
  R_pseudo(1, 1) = TGEqF::kDefaultHorizontalVariance;
  R_pseudo(2, 2) = sigma_z * sigma_z;
  via_position.update_position(pseudoPosition(via_position, z_depth), R_pseudo);

  EXPECT(traits<State>::Equals(via_position.state(), via_depth.state(), 1e-12));
  EXPECT(assert_equal((Matrix)via_position.errorCovariance(),
                      (Matrix)via_depth.errorCovariance(), 1e-12));
}

// At a rotated state with a large vertical offset the update still moves the
// estimate toward the measured depth. One update cannot close a metre-sized gap
// against a decimetre prior, so only a strict decrease is asserted.
TEST(TGEqF, DepthUpdateAtRotatedStateReducesVerticalError) {
  TGEqF filter(State::identity(), defaultSigma(), rotatedX0());

  const double z_depth = filter.position().z() + 1.0;
  const double err_before = std::abs(filter.position().z() - z_depth);
  filter.update_depth(z_depth, depthNoise(0.01));
  const double err_after = std::abs(filter.position().z() - z_depth);

  EXPECT(err_after < 0.95 * err_before);
}

// The horizontal-variance argument is live: shrinking it changes the correction
// a correlated prior produces, so the default is not silently ignored.
TEST(TGEqF, DepthUpdateHorizontalVarianceIsRespected) {
  // Correlate p_x with p_z so the depth channel has a horizontal path to act
  // through; the 2x2 block [[0.01, 0.008], [0.008, 0.01]] stays SPD.
  TGEqF::Covariance18 Sigma0 = defaultSigma();
  Sigma0(6, 8) = Sigma0(8, 6) = 0.008;

  TGEqF inflated(State::identity(), Sigma0);
  TGEqF pinned(State::identity(), Sigma0);

  const double z_depth = -0.8;
  inflated.update_depth(z_depth, depthNoise(0.01));
  pinned.update_depth(z_depth, depthNoise(0.01), /*horizontal_variance=*/1e-6);

  EXPECT((inflated.position() - pinned.position()).norm() > 1e-3);
}

}  // namespace depth_update
/* ************************************************************************* */

/* ************************************************************************* */
int main() {
  TestResult tr;
  return TestRegistry::runAllTests(tr);
}
