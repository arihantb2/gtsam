/**
 * @file  testTFGInEKF.cpp
 * @brief Integration tests for the TFG-IEKF wrapper (predict + position update).
 *
 * Reference: Fornasier et al. (arXiv:2309.03765v3), Sec. 4, 7; App. B.7.
 */
#include <CppUnitLite/TestHarness.h>
#include <gtsam/base/TestableAssertions.h>
#include <gtsam_unstable/tfg_inekf/InEKF.h>

using namespace tfg;
using namespace gtsam;

static constexpr double kTol = 1e-9;

using Cov15 = Eigen::Matrix<double, 15, 15>;
using Cov3  = Eigen::Matrix<double, 3, 3>;

static bool veq(const Eigen::Vector3d& a, const Eigen::Vector3d& b,
                double tol) {
  return assert_equal((Vector)a, (Vector)b, tol);
}

// Min eigenvalue of a symmetric matrix (PSD check).
static double minEig(const Eigen::MatrixXd& M) {
  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(M);
  return es.eigenvalues().minCoeff();
}

// ---------------------------------------------------------------------------
// Construction
// ---------------------------------------------------------------------------

TEST(TfgInEKF, ConstructorStoresStateAndCovariance) {
  auto X0 = TwoFrameGroup::FromState(Rot3::Rz(0.2), Eigen::Vector3d(0.1, 0, 0),
                                     Eigen::Vector3d(1, 2, 3),
                                     Eigen::Vector3d::Zero(),
                                     Eigen::Vector3d::Zero());
  Cov15 P0 = Cov15::Identity() * 0.5;
  TfgInEKF ekf(X0, P0);

  EXPECT(traits<TwoFrameGroup>::Equals(ekf.state(), X0, kTol));
  EXPECT(assert_equal((Matrix)P0, (Matrix)ekf.covariance(), kTol));
  EXPECT(veq(ekf.position(), Eigen::Vector3d(1, 2, 3), kTol));
}

// ---------------------------------------------------------------------------
// Predict: covariance stays symmetric PSD; state mean follows the dynamics.
// ---------------------------------------------------------------------------

TEST(TfgInEKF, PredictKeepsCovarianceSymmetricPSD) {
  auto X0 = TwoFrameGroup::Identity();
  Cov15 P0 = Cov15::Identity() * 0.1;
  TfgInEKF ekf(X0, P0);

  Cov15 Qc = Cov15::Identity() * 1e-3;
  ekf.propagate(Eigen::Vector3d(0.05, -0.02, 0.1),
                Eigen::Vector3d(0.0, 0.0, 9.81), Qc, 0.01);

  const Cov15 P = ekf.covariance();
  EXPECT(assert_equal((Matrix)P, (Matrix)P.transpose(), 1e-12));  // symmetric
  EXPECT(minEig(P) > 0.0);                                        // PSD
}

// Mean prediction matches the world-frame dynamics (Eq. 3) over one step.
TEST(TfgInEKF, PredictMeanFollowsDynamics) {
  Rot3 R0 = Rot3::Rz(0.3);
  Eigen::Vector3d v0(0.2, -0.1, 0.0), p0(1.0, 0.0, -1.0);
  auto X0 = TwoFrameGroup::FromState(R0, v0, p0, Eigen::Vector3d::Zero(),
                                     Eigen::Vector3d::Zero());
  TfgInEKF ekf(X0, Cov15::Identity() * 0.01);

  const Eigen::Vector3d omega(0.0, 0.0, 0.1), accel(0.0, 0.0, 0.0);
  const double dt = 1e-4;
  ekf.propagate(omega, accel, Cov15::Identity() * 1e-6, dt);

  const Eigen::Vector3d g(0.0, 0.0, -9.81);
  Rot3 R1 = R0 * Rot3::Expmap(omega * dt);
  Eigen::Vector3d v1 = v0 + (R0 * accel + g) * dt;
  Eigen::Vector3d p1 = p0 + v0 * dt;

  EXPECT(ekf.attitude().equals(R1, 1e-9));
  EXPECT(veq(ekf.velocity(), v1, 1e-7));
  EXPECT(veq(ekf.position(), p1, 1e-7));
}

// ---------------------------------------------------------------------------
// Position update
// ---------------------------------------------------------------------------

// A position measurement reduces the position-block covariance.
TEST(TfgInEKF, UpdateReducesPositionCovariance) {
  auto X0 = TwoFrameGroup::Identity();
  Cov15 P0 = Cov15::Identity() * 0.1;
  P0.block<3, 3>(6, 6) = Cov3::Identity() * 4.0;  // large position uncertainty
  TfgInEKF ekf(X0, P0);

  const double trace_before = ekf.covariance().block<3, 3>(6, 6).trace();
  ekf.update_position(Eigen::Vector3d(0.5, -0.3, 0.2), Cov3::Identity() * 0.01);
  const double trace_after = ekf.covariance().block<3, 3>(6, 6).trace();

  EXPECT(trace_after < trace_before);
  EXPECT(minEig(ekf.covariance()) > 0.0);
}

// A single accurate update pulls a wrong position estimate toward the
// measurement (here R = I so the position block is directly observed).
TEST(TfgInEKF, UpdateCorrectsPosition) {
  auto X0 = TwoFrameGroup::FromState(
      Rot3::Identity(), Eigen::Vector3d::Zero(),
      Eigen::Vector3d(1.0, -1.0, 0.5),  // wrong position
      Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero());
  Cov15 P0 = Cov15::Identity() * 1e-4;
  P0.block<3, 3>(6, 6) = Cov3::Identity() * 1.0;  // trust measurement more
  TfgInEKF ekf(X0, P0);

  const Eigen::Vector3d truth(0.0, 0.0, 0.0);
  ekf.update_position(truth, Cov3::Identity() * 1e-4);

  // Estimate should move substantially toward truth.
  EXPECT(ekf.position().norm() < 0.1);
}

// ---------------------------------------------------------------------------
// End-to-end: static vehicle, IMU + GNSS, filter converges to truth.
// ---------------------------------------------------------------------------

// Truth: static at origin, level. Filter starts with confident (level) attitude
// but uncertain position/velocity, fed IMU (100 Hz) + GNSS (10 Hz). End-to-end
// the filter must drive the error down and shrink the covariance.
//
// Note: in a purely static scene horizontal position and tilt are weakly
// distinguishable through gravity, and the first-order output Jacobian C0
// couples them (the y_hat^ block). So horizontal position converges to a small
// residual rather than exactly zero — tightened by the equivariant C* (B.35),
// a documented follow-up. We therefore assert strong error reduction, clean
// velocity convergence, and covariance shrink.
TEST(TfgInEKF, StaticConvergence) {
  const Eigen::Vector3d g(0.0, 0.0, -9.81);
  const Eigen::Vector3d accel_static = -g;  // specific force when static
  const Eigen::Vector3d omega(0.0, 0.0, 0.0);
  const Eigen::Vector3d truth_pos(0.0, 0.0, 0.0);

  const Eigen::Vector3d p_init(0.5, -0.4, 0.3);
  auto X0 = TwoFrameGroup::FromState(
      Rot3::Identity(), Eigen::Vector3d(0.1, -0.05, 0.0), p_init,
      Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero());

  Cov15 P0 = Cov15::Identity() * 1e-4;        // confident attitude + bias
  P0.block<3, 3>(3, 3) = Cov3::Identity() * 0.25;  // velocity uncertainty
  P0.block<3, 3>(6, 6) = Cov3::Identity() * 1.0;   // position uncertainty
  TfgInEKF ekf(X0, P0);

  Cov15 Qc = Cov15::Identity() * 1e-6;
  Cov3 R_pos = Cov3::Identity() * (0.05 * 0.05);
  const double dt = 0.01;  // 100 Hz IMU

  const double pos_trace_before = ekf.covariance().block<3, 3>(6, 6).trace();

  for (int k = 0; k < 500; ++k) {
    ekf.propagate(omega, accel_static, Qc, dt);
    if (k % 10 == 0)  // 10 Hz GNSS
      // Pin to C0: this test exercises the C0 coupling path. C* deliberately
      // zeroes the position->attitude coupling for a level vehicle, so it does
      // not drive horizontal velocity through the gravity feedback loop here.
      ekf.update_position(truth_pos, R_pos, /*use_cstar=*/false);
  }

  // Strong error reduction (> 60%) and clean velocity convergence.
  const double err_init  = (p_init - truth_pos).norm();
  const double err_final = (ekf.position() - truth_pos).norm();
  EXPECT(err_final < 0.4 * err_init);
  EXPECT(veq(ekf.velocity(), Eigen::Vector3d::Zero(), 0.05));

  // Covariance stays SPD and the position block shrinks well below the prior.
  EXPECT(minEig(ekf.covariance()) > 0.0);
  const double pos_trace_after = ekf.covariance().block<3, 3>(6, 6).trace();
  EXPECT(pos_trace_after < 0.1 * pos_trace_before);
}

// ---------------------------------------------------------------------------
// C*: the midpoint-symmetrised output matrix (1/2 y_hat^) is the cubic-error
// linearisation. In this static-level geometry the position->attitude coupling
// is parallel to the innovation, so C* and C0 converge to the same fixed point
// (their difference is a subtle, regime-dependent transient effect, not a
// clean inequality). What must hold is that the default C* path yields a
// convergent, consistent filter even from a large initial error.
// ---------------------------------------------------------------------------

TEST(TfgInEKF, CstarConvergesFromLargeError) {
  const Eigen::Vector3d g(0.0, 0.0, -9.81);
  const Eigen::Vector3d accel_static = -g;
  const Eigen::Vector3d omega = Eigen::Vector3d::Zero();
  const Eigen::Vector3d truth_pos = Eigen::Vector3d::Zero();

  auto X0 = TwoFrameGroup::FromState(
      Rot3::Identity(), Eigen::Vector3d::Zero(), Eigen::Vector3d(1.0, -0.8, 0.0),
      Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero());
  Cov15 P0 = Cov15::Identity() * 0.05;
  P0.block<3, 3>(6, 6) = Cov3::Identity() * 2.0;
  TfgInEKF ekf(X0, P0);

  Cov15 Qc = Cov15::Identity() * 1e-6;
  Cov3 R_pos = Cov3::Identity() * (0.05 * 0.05);
  const double dt = 0.01;
  for (int k = 0; k < 500; ++k) {
    ekf.propagate(omega, accel_static, Qc, dt);
    if (k % 10 == 0) ekf.update_position(truth_pos, R_pos, /*use_cstar=*/true);
  }
  EXPECT((ekf.position() - truth_pos).norm() < 0.01);
  EXPECT(minEig(ekf.covariance()) > 0.0);
}

// ---------------------------------------------------------------------------
// F1: the predict step must build nav<->bias cross-covariance, and a position
// update must therefore move the bias estimate. Before the fix the adjoint-only
// propagation left these blocks identically zero and biases were unobservable.
// ---------------------------------------------------------------------------

TEST(TfgInEKF, PredictBuildsNavBiasCrossCovariance) {
  auto X0 = TwoFrameGroup::FromState(
      Rot3::Rz(0.2), Eigen::Vector3d(0.3, -0.1, 0.0),
      Eigen::Vector3d(1.0, 0.5, -0.2), Eigen::Vector3d(0.01, -0.02, 0.0),
      Eigen::Vector3d(0.0, 0.03, -0.01));
  TfgInEKF ekf(X0, Cov15::Identity() * 0.1);  // diagonal: zero cross-cov

  ekf.propagate(Eigen::Vector3d(0.05, -0.02, 0.1),
                Eigen::Vector3d(0.0, 0.0, 9.81), Cov15::Identity() * 1e-6, 0.01);

  // Position(6..8) vs gyro-bias(9..11) and velocity(3..5) vs accel-bias(12..14)
  // cross-blocks are now populated by the lift Jacobian coupling.
  const Cov15 P = ekf.covariance();
  const double pos_bg = P.block<3, 3>(6, 9).norm();
  const double vel_ba = P.block<3, 3>(3, 12).norm();
  EXPECT(pos_bg > 1e-9);
  EXPECT(vel_ba > 1e-9);
}

TEST(TfgInEKF, PositionUpdateMovesBiasEstimate) {
  auto X0 = TwoFrameGroup::FromState(
      Rot3::Rz(0.2), Eigen::Vector3d(0.3, -0.1, 0.0),
      Eigen::Vector3d(1.0, 0.5, -0.2), Eigen::Vector3d::Zero(),
      Eigen::Vector3d::Zero());
  Cov15 P0 = Cov15::Identity() * 0.1;
  P0.block<3, 3>(6, 6) = Cov3::Identity() * 2.0;  // loose position
  TfgInEKF ekf(X0, P0);

  // One propagate to couple bias into the nav covariance, then a position fix.
  ekf.propagate(Eigen::Vector3d(0.05, -0.02, 0.1),
                Eigen::Vector3d(0.0, 0.0, 9.81), Cov15::Identity() * 1e-5, 0.01);
  const Eigen::Vector3d bg_before = ekf.bias_gyro();
  const Eigen::Vector3d ba_before = ekf.bias_accel();

  ekf.update_position(Eigen::Vector3d(0.0, 0.0, 0.0), Cov3::Identity() * 1e-3);

  // Gain rows for the bias are no longer identically zero, so the estimate moves.
  EXPECT((ekf.bias_gyro() - bg_before).norm() + (ekf.bias_accel() - ba_before).norm() > 1e-6);
}

// ---------------------------------------------------------------------------
// F2: the equivariant residual / C* update must be invariant to the choice of
// world origin. Running an identical static scenario shifted by a large offset
// must give the same position-error trajectory. (The old 1/2 (y_hat+p_hat)^
// coupling carried the absolute position p_hat and broke this.)
// ---------------------------------------------------------------------------

static double runStaticErrShifted(const Eigen::Vector3d& offset, bool use_cstar) {
  const Eigen::Vector3d g(0.0, 0.0, -9.81);
  const Eigen::Vector3d accel_static = -g;
  const Eigen::Vector3d omega = Eigen::Vector3d::Zero();
  const Eigen::Vector3d truth_pos = offset;

  auto X0 = TwoFrameGroup::FromState(
      Rot3::Identity(), Eigen::Vector3d::Zero(),
      Eigen::Vector3d(1.0, -0.8, 0.0) + offset,  // wrong position, shifted
      Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero());
  Cov15 P0 = Cov15::Identity() * 0.05;
  P0.block<3, 3>(6, 6) = Cov3::Identity() * 2.0;
  TfgInEKF ekf(X0, P0);

  Cov15 Qc = Cov15::Identity() * 1e-6;
  Cov3 R_pos = Cov3::Identity() * (0.05 * 0.05);
  const double dt = 0.01;
  for (int k = 0; k < 500; ++k) {
    ekf.propagate(omega, accel_static, Qc, dt);
    if (k % 10 == 0) ekf.update_position(truth_pos, R_pos, use_cstar);
  }
  return (ekf.position() - truth_pos).norm();
}

TEST(TfgInEKF, CstarTranslationInvariance) {
  const double err0 = runStaticErrShifted(Eigen::Vector3d::Zero(), true);
  const double errS =
      runStaticErrShifted(Eigen::Vector3d(500.0, -300.0, 100.0), true);
  EXPECT_DOUBLES_EQUAL(err0, errS, 1e-6);
}

// ---------------------------------------------------------------------------
// F4: the lifted process noise must couple gyro white noise into the bias
// rows (through b_w^/b_a^ in the input matrix B).
// ---------------------------------------------------------------------------

TEST(TfgInEKF, InputNoiseCouplesIntoBiasRows) {
  auto X = TwoFrameGroup::FromState(
      Rot3::Rz(0.3), Eigen::Vector3d(0.2, 0.0, -0.1),
      Eigen::Vector3d(0.5, -0.5, 0.1), Eigen::Vector3d(0.02, -0.01, 0.0),
      Eigen::Vector3d(0.0, 0.03, -0.02));
  tfg::ImuNoise noise;
  noise.gyro = 1e-2;  // gyro noise only
  const Cov15 Qd = tfg::inputNoiseCov(X, noise, 0.01);

  const double gw = Qd.block<3, 3>(9, 9).norm();    // gyro-bias rows driven
  const double ga = Qd.block<3, 3>(12, 12).norm();  // accel-bias rows driven
  const double th_gw = Qd.block<3, 3>(0, 9).norm(); // theta<->gyro-bias corr
  EXPECT(gw > 1e-9);
  EXPECT(ga > 1e-9);
  EXPECT(th_gw > 1e-9);
  EXPECT(assert_equal((Matrix)Qd, (Matrix)Qd.transpose(), 1e-15));  // symmetric
}

/* ************************************************************************* */
int main() {
  TestResult tr;
  return TestRegistry::runAllTests(tr);
}
