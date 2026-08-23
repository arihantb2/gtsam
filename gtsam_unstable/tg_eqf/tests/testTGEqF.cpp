/**
 * @file  testTGEqF.cpp
 * @brief The TGEqF filter: construction, propagation, the three updates, the
 *        covariance reset, and input-mapped process noise.
 *
 * Top of the unit suite. The layers below have already pinned the chart, the
 * group, the action, the lift and the output matrices, so the tests here are
 * about the filter's own contracts: what it transports, what it delegates to,
 * and what it does at a state far from the reference. Scenario-level accuracy
 * lives in testTGEqFRegression.cpp.
 */
#include <CppUnitLite/TestHarness.h>
#include <gtsam/base/TestableAssertions.h>
#include <gtsam/base/numericalDerivative.h>
#include <gtsam_unstable/tg_eqf/BodyVelocityOutput.h>
#include <gtsam_unstable/tg_eqf/EqF.h>
#include <gtsam_unstable/tg_eqf/Lift.h>
#include <gtsam_unstable/tg_eqf/PositionOutput.h>
#include <gtsam_unstable/tg_eqf/Symmetry.h>

#include <chrono>
#include <cstdio>
#include <utility>

using namespace gtsam::tgeqf;
using namespace gtsam;

/* ************************************************************************* */
namespace fixture {

constexpr double kTol = 1e-6;

using Vector18 = Eigen::Matrix<double, 18, 1>;

const Eigen::Vector3d kGravity(0.0, 0.0, -9.81);

bool veq(const Eigen::Vector3d& a, const Eigen::Vector3d& b,
         double tol = kTol) {
  return assert_equal((Vector)a, (Vector)b, tol);
}

TGEqF::Covariance18 defaultSigma() {
  return 0.01 * TGEqF::Covariance18::Identity();
}

TGEqF::Covariance18 anisotropicSigma() {
  TGEqF::Covariance18 Sigma0 = TGEqF::Covariance18::Zero();
  for (int i = 0; i < 18; ++i) {
    Sigma0(i, i) = 1e-4 * (1.0 + i);
  }
  return Sigma0;
}

// Tangent order [att(0:3), vel(3:6), pos(6:9), b_w(9:12), b_a(12:15),
// b_v(15:18)]: accel noise drives velocity, accel bias RW drives b_a.
TGEqF::Covariance18 defaultQc() {
  TGEqF::Covariance18 Qc = TGEqF::Covariance18::Zero();
  Qc.block<3, 3>(0, 0) = 1e-4 * Eigen::Matrix3d::Identity();
  Qc.block<3, 3>(3, 3) = 1e-3 * Eigen::Matrix3d::Identity();
  Qc.block<3, 3>(9, 9) = 1e-6 * Eigen::Matrix3d::Identity();
  Qc.block<3, 3>(12, 12) = 1e-5 * Eigen::Matrix3d::Identity();
  return Qc;
}

/// Group element with phi(X0, identity) = (Rz(90 deg), (2,0,0), (5,-3,1), 0).
TGElement rotatedX0() {
  TGElement X;
  X.R = Rot3::Rz(M_PI / 2);
  X.v = Eigen::Vector3d(2.0, 0.0, 0.0);
  X.p = Eigen::Vector3d(5.0, -3.0, 1.0);
  return X;
}

Eigen::Matrix<double, 18, 18> orbitJacobian0(const State& xi_ref) {
  Eigen::Matrix<double, 18, 18> H;
  const TGSymmetry::Orbit orbit(xi_ref);
  orbit(TGElement::Identity(), &H);
  return H;
}

Eigen::Matrix<double, 18, 18> diffeoJacobian(const TGElement& X,
                                             const State& xi) {
  Eigen::Matrix<double, 18, 18> H;
  const TGSymmetry::Diffeomorphism diffeo(X);
  diffeo(xi, &H);
  return H;
}

}  // namespace fixture
/* ************************************************************************* */

/* ************************************************************************* */
namespace construction {

// With the default group estimate the filter sits exactly at the reference
// state and reports it through every accessor.
TEST(TGEqF, InitializesAtTheReferenceState) {
  const State xi_ref = State::identity();
  TGEqF filter(xi_ref, fixture::defaultSigma());

  EXPECT(traits<State>::Equals(xi_ref, filter.state(), fixture::kTol));
  EXPECT(traits<TGElement>::Equals(TGElement::Identity(),
                                   filter.groupEstimate(), fixture::kTol));
  EXPECT(assert_equal(fixture::defaultSigma(), filter.errorCovariance(),
                      fixture::kTol));

  EXPECT(filter.attitude().equals(Rot3::Identity(), fixture::kTol));
  EXPECT(fixture::veq(Eigen::Vector3d::Zero(), filter.velocity()));
  EXPECT(fixture::veq(Eigen::Vector3d::Zero(), filter.position()));
  EXPECT(fixture::veq(Eigen::Vector3d::Zero(), filter.bias_gyro()));
  EXPECT(fixture::veq(Eigen::Vector3d::Zero(), filter.bias_accel()));
  EXPECT(fixture::veq(Eigen::Vector3d::Zero(), filter.bias_vel()));
}

// Sigma0 is given in the chart at the initial estimate, so the constructor
// transports it to the origin chart as J^-1 Sigma0 J^-T, the exact inverse of
// what covariance() applies forward. The last check uses an anisotropic Sigma0
// so the transposed congruence is O(1) away, which fixes the direction of the
// transport.
TEST(TGEqF, ConstructorTransportsCovarianceToTheOriginChart) {
  const State xi_ref = State::identity();
  const TGElement X0 = fixture::rotatedX0();
  const TGEqF::Covariance18 Sigma0 = fixture::anisotropicSigma();
  TGEqF filter(xi_ref, Sigma0, X0);

  EXPECT(assert_equal((Matrix)Sigma0, (Matrix)filter.covariance(), 1e-9));

  const Eigen::Matrix<double, 18, 18> J_inv =
      fixture::diffeoJacobian(X0, xi_ref).inverse();
  const Eigen::Matrix<double, 18, 18> expected =
      J_inv * Sigma0 * J_inv.transpose();
  EXPECT(
      assert_equal((Matrix)expected, (Matrix)filter.errorCovariance(), 1e-9));
  EXPECT((J_inv.transpose() * Sigma0 * J_inv - expected).norm() > 1e-3);

  // At the group identity the transport is the identity.
  TGEqF at_identity(xi_ref, Sigma0, TGElement::Identity());
  EXPECT(assert_equal((Matrix)Sigma0, (Matrix)at_identity.errorCovariance(),
                      1e-12));
}

}  // namespace construction
/* ************************************************************************* */

/* ************************************************************************* */
namespace propagation {

struct IntegrationResult {
  State state;
  TGEqF::Covariance18 covariance;
};

template <size_t K>
IntegrationResult runIntegrationComparison() {
  State xi0 = State::identity();
  xi0.R = Rot3::Rz(0.35) * Rot3::Ry(-0.2);
  xi0.v = Eigen::Vector3d(1.2, -0.7, 0.4);
  xi0.p = Eigen::Vector3d(2.0, -1.0, 0.5);
  xi0.b_w = Eigen::Vector3d(0.03, -0.02, 0.01);
  xi0.b_a = Eigen::Vector3d(-0.08, 0.04, 0.06);

  TGEqF filter(xi0, fixture::defaultSigma());
  filter.set_virtual_bias_anchor(false);

  Input u;
  u.w = Eigen::Vector3d(0.7, -0.4, 1.1);
  u.a = Eigen::Vector3d(1.5, -0.8, 0.6);
  u.g_vec = fixture::kGravity;
  u.v = Eigen::Vector3d::Zero();
  u.tau_w = Eigen::Vector3d::Zero();
  u.tau_a = Eigen::Vector3d::Zero();
  u.tau_v = Eigen::Vector3d::Zero();

  const double dt = 0.05;
  for (int step = 0; step < 20; ++step) {
    filter.predict<K>(Lift(u), InputOrbit(u), fixture::defaultQc(), dt);
  }
  return {filter.state(), filter.errorCovariance()};
}

template <size_t K>
std::pair<IntegrationResult, double> timedIntegrationComparison() {
  constexpr int repetitions = 100;
  const auto start = std::chrono::steady_clock::now();
  IntegrationResult result = runIntegrationComparison<K>();
  for (int repetition = 1; repetition < repetitions; ++repetition) {
    result = runIntegrationComparison<K>();
  }
  const auto elapsed = std::chrono::steady_clock::now() - start;
  const double total_microseconds =
      std::chrono::duration<double, std::micro>(elapsed).count();
  return {result, total_microseconds};
}

void printIntegrationResult(const char* label,
                            const std::pair<IntegrationResult, double>& timed,
                            int repetitions) {
  printf(
      "\n[TGEqF integration diagnostic] %s\n"
      "  timing: total=%.3f us repetitions=%d average=%.3f us/run "
      "(%.3f us/predict)\n",
      label, timed.second, repetitions, timed.second / repetitions,
      timed.second / repetitions / 20.0);
}

// Prints each covariance and timing for Euler and the two expm orders.
TEST_DISABLED(TGEqF, PrintsCovarianceAndTimingByIntegrationOrder) {
  constexpr int repetitions = 100;
  const auto euler = timedIntegrationComparison<1>();
  const auto expm2 = timedIntegrationComparison<2>();
  const auto expm4 = timedIntegrationComparison<4>();

  printf("\n========== TGEqF K timing/covariance (20 x dt=0.05) ==========");
  printIntegrationResult("K=1 (Euler)", euler, repetitions);
  printIntegrationResult("K=2 (expm)", expm2, repetitions);
  printIntegrationResult("K=4 (expm)", expm4, repetitions);
  printf("==============================================================\n");

  EXPECT(true);
}

// The lift carries dp = v, so a moving estimate must advance its position by
// v*dt.
TEST(TGEqF, PropagateIntegratesPosition) {
  State xi0 = State::identity();
  xi0.v = Eigen::Vector3d(1.0, -0.5, 0.2);
  TGEqF filter(xi0, fixture::defaultSigma());
  filter.set_virtual_bias_anchor(false);

  const double dt = 0.01;
  const Eigen::Vector3d p_before = filter.position();
  filter.propagate(Eigen::Vector3d::Zero(), -fixture::kGravity,
                   fixture::kGravity, fixture::defaultQc(), dt);

  EXPECT(fixture::veq(filter.position() - p_before, xi0.v * dt, 1e-4));
}

// propagate() feeds the virtual input nu = 0 regardless of the b_v estimate, so
// it must agree with a lift built by hand with u.v = 0 even when b_v is not.
TEST(TGEqF, PropagateFeedsZeroVirtualInput) {
  State xi_ref = State::identity();
  xi_ref.b_v = Eigen::Vector3d(0.1, -0.05, 0.08);

  const Eigen::Vector3d omega = Eigen::Vector3d::Zero();
  const Eigen::Vector3d accel(0.1, -0.2, 0.05);
  const double dt = 0.05;

  TGEqF via_propagate(xi_ref, fixture::defaultSigma());
  via_propagate.set_virtual_bias_anchor(false);
  via_propagate.propagate(omega, accel, fixture::kGravity, fixture::defaultQc(),
                          dt);

  Input u;
  u.w = omega;
  u.a = accel;
  u.g_vec = fixture::kGravity;
  TGEqF via_lift(xi_ref, fixture::defaultSigma());
  via_lift.set_virtual_bias_anchor(false);
  via_lift.predict<1>(Lift(u), InputOrbit(u), fixture::defaultQc(), dt);

  EXPECT(traits<TGElement>::Equals(via_propagate.groupEstimate(),
                                   via_lift.groupEstimate(), 1e-9));
}

}  // namespace propagation
/* ************************************************************************* */

/* ************************************************************************* */
namespace updates {

// A tight position fix pulls the estimate onto the measurement and shrinks the
// position block of the covariance.
TEST(TGEqF, PositionUpdateCorrectsPosition) {
  TGEqF filter(State::identity(), fixture::defaultSigma());
  const Eigen::Vector3d pi(1.5, 0.0, -0.25);
  const TGEqF::Covariance3 R = 0.001 * TGEqF::Covariance3::Identity();

  const double trace_before =
      filter.errorCovariance().block<3, 3>(6, 6).trace();
  filter.update_position(pi, R);

  const double trace_after = filter.errorCovariance().block<3, 3>(6, 6).trace();
  EXPECT((filter.position() - pi).norm() < 0.2);
  EXPECT(trace_after < trace_before);
}

// A tight DVL fix pulls the body-frame velocity onto the measurement and
// shrinks the velocity block of the covariance.
TEST(TGEqF, DvlUpdateCorrectsVelocity) {
  TGEqF filter(State::identity(), fixture::defaultSigma());
  const Eigen::Vector3d z_dvl(0.8, 0.1, -0.3);
  const TGEqF::Covariance3 R = 0.001 * TGEqF::Covariance3::Identity();

  const double trace_before =
      filter.errorCovariance().block<3, 3>(3, 3).trace();
  filter.update_dvl(z_dvl, R);

  const double trace_after = filter.errorCovariance().block<3, 3>(3, 3).trace();
  EXPECT((DVLMeasurement::predict(filter.state()) - z_dvl).norm() < 0.2);
  EXPECT(trace_after < trace_before);
}

// The b_v = 0 constraint drives a non-zero virtual bias toward zero and
// shrinks its covariance block.
TEST(TGEqF, VirtualBiasUpdateDrivesTheBiasToZero) {
  State xi_ref = State::identity();
  xi_ref.b_v = Eigen::Vector3d(0.2, -0.15, 0.1);
  TGEqF filter(xi_ref, fixture::defaultSigma());

  const double before = filter.bias_vel().norm();
  const double trace_before =
      filter.errorCovariance().block<3, 3>(15, 15).trace();
  filter.update_virtual_bias(1e-4 * TGEqF::Covariance3::Identity());

  const double trace_after =
      filter.errorCovariance().block<3, 3>(15, 15).trace();
  EXPECT(filter.bias_vel().norm() < 0.5 * before);
  EXPECT(trace_after < trace_before);
}

}  // namespace updates
/* ************************************************************************* */

/* ************************************************************************* */
namespace away_from_the_reference {

// C0 is stated in the origin output space, so R_X C0 is the chart derivative of
// eps -> h(phi(X0, Retract(xi_ref, eps))), which reads the measurement in the
// body frame of the estimate. The R_X factor is the transport between the two
// output frames.
TEST(TGEqF, OriginChartJacobianTransportsByTheOutputAction) {
  const State xi_ref = State::identity();
  const TGElement X0 = fixture::rotatedX0();

  const Eigen::Matrix<double, 3, 18> H_num =
      numericalDerivative11<Eigen::Vector3d, State>(
          [&X0](const State& x) { return DVLMeasurement::predict(phi(X0, x)); },
          xi_ref);

  EXPECT(assert_equal(
      (Matrix)H_num,
      (Matrix)(X0.R.matrix().transpose() * DVLMeasurement::jacobian_C0(xi_ref)),
      1e-5));
}

// A DVL update at a state rotated 90 degrees from the reference must move the
// velocity estimate along the correct global axis.
TEST(TGEqF, DvlUpdateMovesVelocityInTheGlobalFrame) {
  // The 2 m/s speed makes attitude twice as sensitive as velocity to a body
  // velocity reading, so an isotropic prior would spend the innovation on yaw.
  // Pin attitude to isolate the velocity correction under test.
  TGEqF::Covariance18 Sigma0 = fixture::defaultSigma();
  Sigma0.block<3, 3>(0, 0) = 1e-8 * Eigen::Matrix3d::Identity();
  TGEqF filter(State::identity(), Sigma0, fixture::rotatedX0());

  const Eigen::Vector3d v_before = filter.velocity();
  const Eigen::Vector3d z_dvl =
      filter.attitude().unrotate(v_before + Eigen::Vector3d(0.0, 0.1, 0.0));
  filter.update_dvl(z_dvl, 1e-4 * TGEqF::Covariance3::Identity());

  const Eigen::Vector3d dv = filter.velocity() - v_before;
  EXPECT(dv.y() > 0.05);
  EXPECT(std::abs(dv.x()) < 0.02);
  EXPECT(std::abs(dv.z()) < 0.02);
}

// The same check on the position channel.
TEST(TGEqF, PositionUpdateMovesEstimateInTheGlobalFrame) {
  TGEqF filter(State::identity(), fixture::defaultSigma(),
               fixture::rotatedX0());

  const Eigen::Vector3d p_before = filter.position();
  const Eigen::Vector3d pi = p_before + Eigen::Vector3d(0.0, 0.5, 0.0);
  filter.update_position(pi, 1e-4 * TGEqF::Covariance3::Identity());

  const Eigen::Vector3d dp = filter.position() - p_before;
  EXPECT(dp.y() > 0.4);
  EXPECT(std::abs(dp.x()) < 0.1);
  EXPECT(std::abs(dp.z()) < 0.1);
}

}  // namespace away_from_the_reference
/* ************************************************************************* */

/* ************************************************************************* */
namespace virtual_bias_anchor {

// The anchor fires once per propagate unless it is switched off. Disabled, b_v
// is untouched by propagation: bdot = 0 and nu = 0.
TEST(TGEqF, AnchorRunsInPropagateUnlessDisabled) {
  State xi_ref = State::identity();
  xi_ref.b_v = Eigen::Vector3d(0.2, -0.15, 0.1);

  TGEqF anchored(xi_ref, fixture::defaultSigma());
  TGEqF unanchored(xi_ref, fixture::defaultSigma());
  unanchored.set_virtual_bias_anchor(false);

  anchored.propagate(Eigen::Vector3d::Zero(), -fixture::kGravity,
                     fixture::kGravity, fixture::defaultQc(), 0.01);
  unanchored.propagate(Eigen::Vector3d::Zero(), -fixture::kGravity,
                       fixture::kGravity, fixture::defaultQc(), 0.01);

  EXPECT(anchored.bias_vel().norm() < 0.5 * xi_ref.b_v.norm());
  EXPECT(fixture::veq(xi_ref.b_v, unanchored.bias_vel(), 1e-9));
}

// Toggling the anchor off and back on keeps the configured noise. Only passing
// a new R_vb changes it.
TEST(TGEqF, AnchorNoiseSurvivesAToggle) {
  State xi_ref = State::identity();
  xi_ref.b_v = Eigen::Vector3d(0.2, -0.15, 0.1);
  const TGEqF::Covariance3 tight = 1e-6 * TGEqF::Covariance3::Identity();

  TGEqF toggled(xi_ref, fixture::defaultSigma());
  toggled.set_virtual_bias_anchor(true, tight);
  toggled.set_virtual_bias_anchor(false);
  toggled.set_virtual_bias_anchor(true);

  TGEqF direct(xi_ref, fixture::defaultSigma());
  direct.set_virtual_bias_anchor(true, tight);

  toggled.propagate(Eigen::Vector3d::Zero(), -fixture::kGravity,
                    fixture::kGravity, fixture::defaultQc(), 0.01);
  direct.propagate(Eigen::Vector3d::Zero(), -fixture::kGravity,
                   fixture::kGravity, fixture::defaultQc(), 0.01);

  EXPECT(traits<TGElement>::Equals(direct.groupEstimate(),
                                   toggled.groupEstimate(), 1e-12));
  EXPECT(
      assert_equal(direct.errorCovariance(), toggled.errorCovariance(), 1e-12));
}

}  // namespace virtual_bias_anchor
/* ************************************************************************* */

/* ************************************************************************* */
namespace reset_step {

using fixture::Vector18;

/// The re-centring map the group correction applies to the error state.
Vector18 recentre(const State& xi_ref, const TGElement& Xd,
                  const Vector18& eps) {
  return traits<State>::Local(xi_ref,
                              phi(Xd, traits<State>::Retract(xi_ref, eps)));
}

/// Independent oracle for resetMatrix: shares no code with it.
Eigen::Matrix<double, 18, 18> numericalResetJacobian(const State& xi_ref,
                                                     const Vector18& delta_xi,
                                                     const Vector18& delta_x) {
  const TGElement Xd = TGElement::Expmap(-delta_x);
  return numericalDerivative11<Vector18, Vector18>(
      [&](const Vector18& eps) { return recentre(xi_ref, Xd, eps); }, delta_xi);
}

/// A correction large enough that the chart factors are far from the identity.
/// At 1e-4 they are the identity to within 1e-8 and there is nothing to check.
Vector18 bigDeltaXi() {
  Vector18 delta_xi;
  delta_xi << 0.35, -0.25, 0.4, 0.2, -0.15, 0.1, 0.3, -0.2, 0.15, 0.05, -0.04,
      0.03, -0.06, 0.02, -0.01, 0.02, -0.03, 0.01;
  return delta_xi;
}

/// A reference state away from the manifold identity.
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

/// One position update at rotatedX0(), on twins with and without the reset,
/// plus the transport rebuilt from the correction the update actually applied.
/// Xd comes from the public group estimates alone: X_after = Exp(delta_x)
/// X_before, so Xd = Exp(-delta_x) = X_before * X_after^-1.
struct UpdatePair {
  Eigen::Matrix<double, 18, 18> P_with, P_without, J;
};

UpdatePair positionUpdatePair(const TGEqF::Covariance18& Sigma0,
                              double offset_scale = 1.0) {
  const State xi_ref = State::identity();
  const TGElement X0 = fixture::rotatedX0();
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
  const Vector18 delta_xi = fixture::orbitJacobian0(xi_ref) * delta_x;
  return {with_reset.errorCovariance(), without_reset.errorCovariance(),
          with_reset.resetMatrix(delta_xi, delta_x)};
}

// resetMatrix is the derivative of the exact re-centring map, at a correction
// large enough for the chart factors to matter, both at the manifold identity
// and at an off-origin reference. Both matched and unmatched (delta_xi,
// delta_x) pairs are used: the matched pair is what the filter passes, but it
// re-centres the error to nearly zero, which leaves the output-side chart
// factor at ~I and nothing to check there.
TEST(TGEqF, ResetMatrixMatchesNumericalDerivative) {
  for (const State& xi_ref : {State::identity(), offOriginRef()}) {
    TGEqF filter(xi_ref, fixture::defaultSigma());
    const Vector18 delta_xi = bigDeltaXi();
    const Vector18 matched =
        fixture::orbitJacobian0(xi_ref).inverse() * delta_xi;

    for (const Vector18& delta_x : {matched, Vector18(-0.5 * matched)}) {
      EXPECT(assert_equal(
          (Matrix)numericalResetJacobian(xi_ref, delta_xi, delta_x),
          (Matrix)filter.resetMatrix(delta_xi, delta_x), 1e-6));
    }

    // The Jacobian belongs at eps = delta_xi, the post-update mean of the
    // error, not at eps = 0; linearizing at zero is wrong by O(|delta_xi|).
    EXPECT((filter.resetMatrix(Vector18::Zero(), matched) -
            filter.resetMatrix(delta_xi, matched))
               .norm() > 1e-3);
  }
}

// A zero correction leaves the transport at exactly the identity.
TEST(TGEqF, ResetAtZeroCorrectionIsIdentity) {
  const State xi_ref = State::identity();
  TGEqF filter(xi_ref, fixture::defaultSigma());

  EXPECT(assert_equal(
      (Matrix)Eigen::Matrix<double, 18, 18>::Identity(),
      (Matrix)filter.resetMatrix(Vector18::Zero(), Vector18::Zero()), 1e-12));
}

// A pure-fiber correction's reset transport is I_18 plus a
// 0.5 ad_se23(z0) coupling block, not the identity.
TEST(TGEqF, ResetAtPureFiberCorrectionHasAdCoupling) {
  const State xi_ref = State::identity();
  TGEqF filter(xi_ref, fixture::defaultSigma());

  Vector18 fiber_only = Vector18::Zero();
  fiber_only.tail<9>() << 0.2, -0.15, 0.1, 0.05, 0.04, -0.03, 0.02, -0.01, 0.03;
  // orbitJacobian0(xi_ref) is I_18 in this chart, so delta_xi = delta_x here.
  const Eigen::Matrix<double, 18, 18> J =
      filter.resetMatrix(fixture::orbitJacobian0(xi_ref) * fiber_only,
                         fiber_only);

  Eigen::Matrix<double, 18, 18> expected =
      Eigen::Matrix<double, 18, 18>::Identity();
  const se2_3 z0 = se2_3::from_vector(fiber_only.tail<9>());
  expected.block<9, 9>(9, 0) = 0.5 * detail::ad_se23(z0);

  EXPECT(assert_equal((Matrix)expected, (Matrix)J, 1e-6));
}

// The map resetMatrix linearizes is the one the update really applies to the
// error. This fixes the sign of Xd and the side the group correction acts on,
// which a derivative check cannot: every oracle there inherits the same Xd.
TEST(TGEqF, ResetTracksTheGroupCorrectionTheUpdateApplies) {
  TGEqF filter(State::identity(), fixture::defaultSigma(),
               fixture::rotatedX0());

  State xi_true = filter.state();
  xi_true.v += Eigen::Vector3d(0.1, 0.05, -0.1);
  xi_true.p += Eigen::Vector3d(0.4, -0.2, 0.3);

  const State e_before = filter.errorState(xi_true);
  const TGElement X_before = filter.groupEstimate();

  filter.update_position(xi_true.p + Eigen::Vector3d(0.05, -0.03, 0.02),
                         1e-2 * TGEqF::Covariance3::Identity());

  const TGElement Xd = X_before * filter.groupEstimate().inverse();
  EXPECT(traits<State>::Equals(phi(Xd, e_before), filter.errorState(xi_true),
                               1e-9));
}

// End to end: the covariance the filter ships is the reset transport of its
// Joseph-updated covariance, it stays symmetric positive-definite, and the
// transposed congruence is O(1) away.
TEST(TGEqF, ResetTransportsTheUpdatedCovariance) {
  const UpdatePair u = positionUpdatePair(fixture::anisotropicSigma());

  EXPECT(assert_equal((Matrix)(u.J * u.P_without * u.J.transpose()),
                      (Matrix)u.P_with, 1e-9));
  EXPECT((u.J.transpose() * u.P_without * u.J - u.P_with).norm() > 1e-3);
  EXPECT(assert_equal((Matrix)u.P_with, (Matrix)u.P_with.transpose(), 1e-12));

  Eigen::LLT<Eigen::Matrix<double, 18, 18>> llt(u.P_with);
  EXPECT(llt.info() == Eigen::Success);
}

// The reset is a first-order term in the correction: halving the innovation
// halves its effect on the covariance.
TEST(TGEqF, ResetEffectScalesWithTheCorrection) {
  const UpdatePair full = positionUpdatePair(fixture::defaultSigma(), 1.0);
  const UpdatePair half = positionUpdatePair(fixture::defaultSigma(), 0.5);

  const double d_full = (full.P_with - full.P_without).norm();
  const double d_half = (half.P_with - half.P_without).norm();

  EXPECT(d_full > 1e-9);
  EXPECT(d_half / d_full > 0.4);
  EXPECT(d_half / d_full < 0.6);
}

// The anchor fires once per propagate, so its correction goes through the same
// transport. Once propagation has correlated the bias block with navigation the
// correction is no longer pure fiber and the reset does move the covariance.
TEST(TGEqF, ResetAppliesToTheVirtualBiasAnchor) {
  State xi_ref = State::identity();
  xi_ref.b_v = Eigen::Vector3d(0.2, -0.15, 0.1);
  TGEqF with_reset(xi_ref, fixture::defaultSigma());
  TGEqF without_reset(xi_ref, fixture::defaultSigma());
  without_reset.set_reset_step(false);

  const Eigen::Vector3d omega(0.05, -0.02, 0.01);
  const Eigen::Vector3d accel(0.1, -0.2, 0.05);
  for (int k = 0; k < 10; ++k) {
    with_reset.propagate(omega, accel, fixture::kGravity, fixture::defaultQc(),
                         0.01);
    without_reset.propagate(omega, accel, fixture::kGravity,
                            fixture::defaultQc(), 0.01);
  }

  EXPECT(
      (with_reset.errorCovariance() - without_reset.errorCovariance()).norm() >
      1e-12);
}

// The reset is on unless explicitly switched off. Callers can disable it, so
// the default is worth pinning.
TEST(TGEqF, ResetStepIsOnByDefault) {
  const State xi_ref = State::identity();
  const TGElement X0 = fixture::rotatedX0();
  TGEqF by_default(xi_ref, fixture::anisotropicSigma(), X0);
  TGEqF turned_off(xi_ref, fixture::anisotropicSigma(), X0);
  turned_off.set_reset_step(false);

  const Eigen::Vector3d pi =
      by_default.position() + Eigen::Vector3d(0.5, 0.3, -0.2);
  const TGEqF::Covariance3 R = 1e-2 * TGEqF::Covariance3::Identity();
  by_default.update_position(pi, R);
  turned_off.update_position(pi, R);

  EXPECT((turned_off.errorCovariance() - by_default.errorCovariance()).norm() >
         1e-9);
}

}  // namespace reset_step
/* ************************************************************************* */

/* ************************************************************************* */
namespace process_noise {

ImuNoise makeNoise() {
  ImuNoise nz;
  nz.gyro = 2e-4;
  nz.accel = 3e-3;
  nz.gyro_rw = 1e-6;
  nz.accel_rw = 5e-5;
  return nz;
}

// At the origin the lift differential has no state or transport coupling, so
// the PSDs land one per block: gyro on attitude, accel on velocity, and the
// random walks on their own biases. Position and virtual bias get none, and
// there is no gyro-to-bias coupling -- that needs b_ref != 0.
TEST(TGEqF, InputNoiseCovIsBlockDiagonalAtTheOrigin) {
  TGEqF filter(State::identity(), fixture::defaultSigma());
  const ImuNoise nz = makeNoise();
  const TGEqF::Covariance18 Q = filter.inputNoiseCov(nz);

  const Eigen::Matrix3d I3 = Eigen::Matrix3d::Identity();
  EXPECT(
      assert_equal((Matrix)(nz.gyro * I3), (Matrix)Q.block<3, 3>(0, 0), 1e-6));
  EXPECT(
      assert_equal((Matrix)(nz.accel * I3), (Matrix)Q.block<3, 3>(3, 3), 1e-6));
  EXPECT(assert_equal((Matrix)(nz.gyro_rw * I3), (Matrix)Q.block<3, 3>(9, 9),
                      1e-6));
  EXPECT(assert_equal((Matrix)(nz.accel_rw * I3), (Matrix)Q.block<3, 3>(12, 12),
                      1e-6));
  EXPECT(assert_equal((Matrix)Eigen::Matrix3d::Zero(),
                      (Matrix)Q.block<3, 3>(6, 6), 1e-6));
  EXPECT(assert_equal((Matrix)Eigen::Matrix3d::Zero(),
                      (Matrix)Q.block<3, 3>(15, 15), 1e-6));
  EXPECT(assert_equal((Matrix)Eigen::Matrix3d::Zero(),
                      (Matrix)Q.block<3, 3>(9, 0), 1e-6));
}

// The cached factorization B = input_lift * blkdiag(Ad, Ad) * selector must
// reproduce a finite-difference derivative of Dphi0 * Lambda(xi_ref,
// psi_{Xhat^-1}(u)) at a non-identity group estimate. B is affine in the
// input, so any u serves as a linearization point; inputNoiseCov() takes none
// itself, since it depends only on the group estimate and noise densities.
TEST(TGEqF, InputNoiseCovMatchesTheNumericalLiftDifferential) {
  TGEqF filter(State::identity(), fixture::defaultSigma(),
               fixture::rotatedX0());
  const ImuNoise nz = makeNoise();
  const Eigen::Vector3d omega(0.1, -0.05, 0.02);
  const Eigen::Vector3d accel(0.3, 0.2, 9.6);

  const State xi_ref = State::identity();
  const TGElement Xhat_inv = filter.groupEstimate().inverse();
  const Eigen::Matrix<double, 18, 18> Dphi0 = fixture::orbitJacobian0(xi_ref);

  Input u;
  u.w = omega;
  u.a = accel;
  u.g_vec = fixture::kGravity;
  const auto lambda_origin = [&](const Input& uu) {
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

  const TGEqF::Covariance18 Q = filter.inputNoiseCov(nz);
  EXPECT(assert_equal((Matrix)(B_num * Sigma * B_num.transpose()), (Matrix)Q,
                      1e-6));
}

// input_lift_ = orbit_jacobian0_ * G collapses to G itself in this chart.
TEST(TGEqF, InputLiftMatchesGAtNonzeroBiasReference) {
  State xi_ref = State::identity();
  xi_ref.b_w = Eigen::Vector3d(0.01, -0.02, 0.03);
  xi_ref.b_a = Eigen::Vector3d(-0.05, 0.04, -0.01);
  xi_ref.b_v = Eigen::Vector3d(0.02, 0.02, -0.03);

  const TGEqF filter(xi_ref, fixture::defaultSigma());

  const se2_3 b_ref = {xi_ref.b_w, xi_ref.b_a, xi_ref.b_v};
  Eigen::Matrix<double, 18, 18> G = Eigen::Matrix<double, 18, 18>::Zero();
  G.block<9, 9>(0, 0).setIdentity();
  G.block<9, 9>(9, 0) = detail::ad_se23(b_ref);
  G.block<9, 9>(9, 9) = -Eigen::Matrix<double, 9, 9>::Identity();

  EXPECT(assert_equal((Matrix)G, (Matrix)filter.input_lift(), 1e-12));
}

// The ImuNoise overload of propagate is exactly inputNoiseCov followed by the
// raw-Qc overload, and the noise it injects does grow the covariance.
TEST(TGEqF, PropagateWithImuNoiseDelegatesToInputNoiseCov) {
  const ImuNoise nz = makeNoise();
  const Eigen::Vector3d omega(0.05, -0.02, 0.01);
  const Eigen::Vector3d accel(0.1, -0.2, 9.85);
  const double dt = 0.05;

  TGEqF via_noise(State::identity(), fixture::defaultSigma());
  via_noise.set_virtual_bias_anchor(false);
  const double trace_before = via_noise.errorCovariance().trace();
  via_noise.propagate(omega, accel, fixture::kGravity, nz, dt);
  EXPECT(via_noise.errorCovariance().trace() > trace_before);

  TGEqF via_qc(State::identity(), fixture::defaultSigma());
  via_qc.set_virtual_bias_anchor(false);
  via_qc.propagate(omega, accel, fixture::kGravity, via_qc.inputNoiseCov(nz),
                   dt);

  EXPECT(assert_equal(via_qc.errorCovariance(), via_noise.errorCovariance(),
                      1e-10));
  EXPECT(traits<TGElement>::Equals(via_qc.groupEstimate(),
                                   via_noise.groupEstimate(), 1e-10));
}

}  // namespace process_noise
/* ************************************************************************* */

/* ************************************************************************* */
namespace depth_update {

TGEqF::Covariance1 depthNoise(double sigma) {
  TGEqF::Covariance1 R;
  R(0, 0) = sigma * sigma;
  return R;
}

// A depth update pulls the vertical estimate onto the measurement and leaves
// the horizontal axes essentially alone, which is what the inflated horizontal
// variance is there to buy.
TEST(TGEqF, DepthUpdateCorrectsVerticalPositionOnly) {
  TGEqF filter(State::identity(), fixture::defaultSigma());
  const double z_depth = -0.8;

  const Eigen::Vector3d before = filter.position();
  filter.update_depth(z_depth, depthNoise(0.01));
  const Eigen::Vector3d delta = filter.position() - before;

  EXPECT(std::abs(filter.position().z() - z_depth) < 0.1);
  EXPECT(std::abs(delta.x()) < 0.02);
  EXPECT(std::abs(delta.y()) < 0.02);
}

// Only the vertical position covariance is informed. The reset transport can
// spread vertical shrinkage sideways, so it is switched off to isolate the
// measurement-induced change.
TEST(TGEqF, DepthUpdateShrinksVerticalCovariance) {
  TGEqF filter(State::identity(), fixture::defaultSigma());
  filter.set_reset_step(false);

  const Eigen::Matrix<double, 18, 18> before = filter.errorCovariance();
  filter.update_depth(-0.8, depthNoise(0.01));
  const Eigen::Matrix<double, 18, 18> after = filter.errorCovariance();

  const double drop_z = before(8, 8) - after(8, 8);
  EXPECT(drop_z > 0.0);
  EXPECT(std::abs(before(6, 6) - after(6, 6)) < 0.5 * drop_z);
  EXPECT(std::abs(before(7, 7) - after(7, 7)) < 0.5 * drop_z);
}

// update_depth is exactly the pseudo-position update it documents.
TEST(TGEqF, DepthUpdateMatchesTheEquivalentPositionUpdate) {
  const TGElement X0 = fixture::rotatedX0();
  TGEqF via_depth(State::identity(), fixture::defaultSigma(), X0);
  TGEqF via_position(State::identity(), fixture::defaultSigma(), X0);

  const double z_depth = 0.4;
  const double sigma_z = 0.02;
  via_depth.update_depth(z_depth, depthNoise(sigma_z));

  Eigen::Vector3d pseudo = via_position.position();
  pseudo.z() = z_depth;
  TGEqF::Covariance3 R_pseudo = TGEqF::Covariance3::Zero();
  R_pseudo(0, 0) = TGEqF::kDefaultHorizontalVariance;
  R_pseudo(1, 1) = TGEqF::kDefaultHorizontalVariance;
  R_pseudo(2, 2) = sigma_z * sigma_z;
  via_position.update_position(pseudo, R_pseudo);

  EXPECT(traits<State>::Equals(via_position.state(), via_depth.state(), 1e-12));
  EXPECT(assert_equal((Matrix)via_position.errorCovariance(),
                      (Matrix)via_depth.errorCovariance(), 1e-12));
}

// The horizontal-variance argument changes the update. It needs a prior that
// correlates p_x with p_z, which gives the depth channel a horizontal path to
// act through.
TEST(TGEqF, DepthUpdateRespectsTheHorizontalVariance) {
  TGEqF::Covariance18 Sigma0 = fixture::defaultSigma();
  Sigma0(6, 8) = Sigma0(8, 6) = 0.008;  // 2x2 block stays SPD

  TGEqF inflated(State::identity(), Sigma0);
  TGEqF pinned(State::identity(), Sigma0);

  inflated.update_depth(-0.8, depthNoise(0.01));
  pinned.update_depth(-0.8, depthNoise(0.01), /*horizontal_variance=*/1e-6);

  EXPECT((inflated.position() - pinned.position()).norm() > 1e-3);
}

// The scalar model informs the vertical position covariance too.
TEST(TGEqF, DirectDepthUpdateShrinksVerticalCovariance) {
  TGEqF filter(State::identity(), fixture::defaultSigma());
  filter.set_reset_step(false);

  const Eigen::Matrix<double, 18, 18> before = filter.errorCovariance();
  filter.update_depth_direct(-0.8, depthNoise(0.01));
  const Eigen::Matrix<double, 18, 18> after = filter.errorCovariance();

  const double drop_z = before(8, 8) - after(8, 8);
  EXPECT(drop_z > 0.0);
  EXPECT(std::abs(before(6, 6) - after(6, 6)) < 0.5 * drop_z);
  EXPECT(std::abs(before(7, 7) - after(7, 7)) < 0.5 * drop_z);
}

// At xi_ref = identity the pseudo-position C* row for depth is exactly the
// negation of the direct C, and so is its innovation, so the two updates
// coincide as sigma_xy^2 -> infinity. The tolerance is loose because the
// neglected terms fall off only as 1/sigma_xy^2.
TEST(TGEqF, DirectDepthUpdateMatchesThePseudoPositionUpdateInTheInflatedLimit) {
  const TGElement X0 = fixture::rotatedX0();
  TGEqF via_pseudo(State::identity(), fixture::defaultSigma(), X0);
  TGEqF via_direct(State::identity(), fixture::defaultSigma(), X0);

  const double z_depth = 0.4;
  const double sigma_z = 0.02;
  via_pseudo.update_depth(z_depth, depthNoise(sigma_z),
                          /*horizontal_variance=*/1e12);
  via_direct.update_depth_direct(z_depth, depthNoise(sigma_z));

  EXPECT(traits<State>::Equals(via_pseudo.state(), via_direct.state(), 1e-7));
  EXPECT(assert_equal((Matrix)via_pseudo.errorCovariance(),
                      (Matrix)via_direct.errorCovariance(), 1e-7));
}

// That equivalence needs the horizontal channel stripped of weight. With a
// prior correlating p_x with p_z and horizontal_variance pinned at 1e-6
// instead, the two are genuinely different models.
TEST(TGEqF, DirectDepthUpdateDiffersWhenTheHorizontalChannelIsActive) {
  TGEqF::Covariance18 Sigma0 = fixture::defaultSigma();
  Sigma0(6, 8) = Sigma0(8, 6) = 0.008;  // 2x2 block stays SPD

  TGEqF via_pseudo(State::identity(), Sigma0);
  TGEqF via_direct(State::identity(), Sigma0);

  via_pseudo.update_depth(-0.8, depthNoise(0.01),
                          /*horizontal_variance=*/1e-6);
  via_direct.update_depth_direct(-0.8, depthNoise(0.01));

  EXPECT((via_pseudo.position() - via_direct.position()).norm() > 1e-3);
}

}  // namespace depth_update
/* ************************************************************************* */

/* ************************************************************************* */
int main() {
  TestResult tr;
  return TestRegistry::runAllTests(tr);
}
