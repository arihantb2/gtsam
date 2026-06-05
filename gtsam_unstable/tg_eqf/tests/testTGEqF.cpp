#include <CppUnitLite/TestHarness.h>
#include <gtsam/base/TestableAssertions.h>
#include <gtsam_unstable/tg_eqf/EqF.h>
#include <gtsam_unstable/tg_eqf/PositionOutput.h>
#include <gtsam_unstable/tg_eqf/BodyVelocityOutput.h>
#include <gtsam_unstable/tg_eqf/Symmetry.h>

using namespace tgeqf;
using namespace gtsam;

static constexpr double kTol = 1e-6;

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
  TGEqF::Covariance18 Qc = TGEqF::Covariance18::Zero();
  Qc.block<3, 3>(0, 0) = 1e-4 * Eigen::Matrix3d::Identity();
  Qc.block<3, 3>(6, 6) = 1e-3 * Eigen::Matrix3d::Identity();
  Qc.block<3, 3>(9, 9) = 1e-6 * Eigen::Matrix3d::Identity();
  Qc.block<3, 3>(15, 15) = 1e-5 * Eigen::Matrix3d::Identity();
  return Qc;
}

static TGInput makeImuInput(const Eigen::Vector3d& omega,
                            const Eigen::Vector3d& accel,
                            const Eigen::Vector3d& g_vec,
                            const Eigen::Vector3d& b_v =
                                Eigen::Vector3d::Zero()) {
  TGInput u;
  u.omega = omega;
  u.a_tilde = accel;
  u.g_vec = g_vec;
  u.v_tilde = b_v;
  u.tau_omega = Eigen::Vector3d::Zero();
  u.tau_v = Eigen::Vector3d::Zero();
  u.tau_a = Eigen::Vector3d::Zero();
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

  const Eigen::Vector3d omega(0.05, -0.02, 0.01);
  const Eigen::Vector3d g_vec(0.0, 0.0, -9.81);
  const Eigen::Vector3d accel = g_vec;
  const double dt = 0.1;

  const TGGroupElement g_before = filter.groupEstimate();
  const TGEqF::Covariance18 P_before = filter.errorCovariance();

  filter.propagate(omega, accel, g_vec, defaultQc(), dt);

  EXPECT(!traits<TGGroupElement>::Equals(g_before, filter.groupEstimate(),
                                         1e-12));
  EXPECT(filter.errorCovariance().trace() > P_before.trace());
}

TEST(TGEqF, PropagateUsesVirtualVelocityBiasInput) {
  const TGState xi_ref = TGState::identity();
  TGEqF filter(xi_ref, defaultSigma());

  const Eigen::Vector3d omega = Eigen::Vector3d::Zero();
  const Eigen::Vector3d g_vec(0.0, 0.0, -9.81);
  const Eigen::Vector3d accel = g_vec;
  const double dt = 0.05;

  filter.propagate(omega, accel, g_vec, defaultQc(), dt);
  const TGGroupElement g_imu = filter.groupEstimate();

  TGEqF filter2(xi_ref, defaultSigma());
  const TGInput u = makeImuInput(omega, accel, g_vec, filter2.bias_vel());
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

  const double trace_before = filter.covariance().block<3, 3>(3, 3).trace();
  filter.update_position(pi, R_pos, true);
  const double trace_after = filter.covariance().block<3, 3>(3, 3).trace();

  EXPECT(trace_after < trace_before);
}

// ---------------------------------------------------------------------------
// DVL update
// ---------------------------------------------------------------------------

TEST(TGEqF, DvlUpdateReducesInnovation) {
  TGEqF filter(TGState::identity(), defaultSigma());

  const Eigen::Vector3d z_dvl(0.4, -0.2, 0.1);
  const TGEqF::Covariance3 R_dvl = 0.01 * TGEqF::Covariance3::Identity();

  const double err_before =
      DVLMeasurement::innovation(z_dvl, filter.state()).norm();
  EXPECT(err_before > 0.3);

  filter.update_dvl(z_dvl, R_dvl);

  const double err_after =
      DVLMeasurement::innovation(z_dvl, filter.state()).norm();
  EXPECT(err_after < err_before);
}

TEST(TGEqF, DvlUpdateMovesVelocityTowardMeasurement) {
  TGEqF filter(TGState::identity(), defaultSigma());

  const Eigen::Vector3d z_dvl(0.8, 0.1, -0.3);
  const TGEqF::Covariance3 R_dvl = 0.001 * TGEqF::Covariance3::Identity();

  filter.update_dvl(z_dvl, R_dvl);

  EXPECT((DVLMeasurement::predict(filter.state()) - z_dvl).norm() < 0.2);
}

// ---------------------------------------------------------------------------
// End-to-end
// ---------------------------------------------------------------------------

TEST(TGEqF, EndToEndPropagateAndUpdates) {
  TGEqF filter(TGState::identity(), defaultSigma());

  const Eigen::Vector3d g_vec(0.0, 0.0, -9.81);
  const Eigen::Vector3d omega(0.02, 0.0, 0.01);
  const double dt = 0.05;

  filter.propagate(omega, g_vec, g_vec, defaultQc(), dt);

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

/* ************************************************************************* */
int main() {
  TestResult tr;
  return TestRegistry::runAllTests(tr);
}
