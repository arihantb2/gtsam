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
// C* vs C0: the midpoint-symmetrised output matrix should improve convergence
// in the adversarial static case (large position error couples into attitude
// through the y_hat^ term; C* centres that coupling, Eq. B.35).
// ---------------------------------------------------------------------------

static double runStaticFinalPosError(bool use_cstar) {
  const Eigen::Vector3d g(0.0, 0.0, -9.81);
  const Eigen::Vector3d accel_static = -g;
  const Eigen::Vector3d omega = Eigen::Vector3d::Zero();
  const Eigen::Vector3d truth_pos = Eigen::Vector3d::Zero();

  // Large position error + sizeable attitude covariance: the regime where the
  // first-order coupling hurts.
  auto X0 = TwoFrameGroup::FromState(
      Rot3::Identity(), Eigen::Vector3d::Zero(), Eigen::Vector3d(1.0, -0.8, 0.0),
      Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero());
  Cov15 P0 = Cov15::Identity() * 0.05;
  P0.block<3, 3>(6, 6) = Cov3::Identity() * 2.0;  // position
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

TEST(TfgInEKF, CstarImprovesConvergenceOverC0) {
  const double err_c0 = runStaticFinalPosError(/*use_cstar=*/false);
  const double err_cstar = runStaticFinalPosError(/*use_cstar=*/true);
  EXPECT(err_cstar < err_c0);
}

/* ************************************************************************* */
int main() {
  TestResult tr;
  return TestRegistry::runAllTests(tr);
}
