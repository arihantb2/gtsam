/**
 * @file  testMekf.cpp
 * @brief Integration tests for the MultiplicativeEKF wrapper (predict + update).
 *
 * Block order = [ d_theta(3) | d_v(3) | d_p(3) | d_b_gyro(3) | d_b_accel(3) ].
 * Position block lives at indices 6..8.
 */
#include <CppUnitLite/TestHarness.h>
#include <gtsam/base/TestableAssertions.h>
#include <gtsam_unstable/mekf/MEKF.h>

using namespace mekf;
using namespace gtsam;

static constexpr double kTol = 1e-9;

using Cov15 = Eigen::Matrix<double, 15, 15>;
using Cov3  = Eigen::Matrix<double, 3, 3>;

static bool veq(const Eigen::Vector3d& a, const Eigen::Vector3d& b,
                double tol = kTol) {
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

TEST(Mekf, ConstructorStoresStateAndCovariance) {
  MekfState X0(Rot3::Rz(0.2), Eigen::Vector3d(0.1, 0, 0),
               Eigen::Vector3d(1, 2, 3), Eigen::Vector3d::Zero(),
               Eigen::Vector3d::Zero());
  Cov15 P0 = Cov15::Identity() * 0.5;
  MultiplicativeEKF ekf(X0, P0);

  EXPECT(traits<MekfState>::Equals(ekf.state(), X0, kTol));
  EXPECT(assert_equal((Matrix)P0, (Matrix)ekf.covariance(), kTol));
  EXPECT(veq(ekf.position(), Eigen::Vector3d(1, 2, 3)));
  EXPECT(ekf.attitude().equals(Rot3::Rz(0.2), kTol));
}

// ---------------------------------------------------------------------------
// Predict
// ---------------------------------------------------------------------------

TEST(Mekf, PredictKeepsCovarianceSymmetricPSD) {
  MultiplicativeEKF ekf(MekfState::identity(), Cov15::Identity() * 0.1);

  Cov15 Qc = Cov15::Identity() * 1e-3;
  ekf.propagate(Eigen::Vector3d(0.05, -0.02, 0.1),
                Eigen::Vector3d(0.0, 0.0, 9.81), Qc, 0.01);

  const Cov15 P = ekf.covariance();
  EXPECT(assert_equal((Matrix)P, (Matrix)P.transpose(), 1e-12));  // symmetric
  EXPECT(minEig(P) > 0.0);                                        // PSD
}

// Mean prediction matches the world-frame strapdown step (Eq. 3).
TEST(Mekf, PredictMeanFollowsDynamics) {
  const Rot3 R0 = Rot3::Rz(0.3);
  const Eigen::Vector3d v0(0.2, -0.1, 0.0), p0(1.0, 0.0, -1.0);
  MultiplicativeEKF ekf(
      MekfState(R0, v0, p0, Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero()),
      Cov15::Identity() * 0.01);

  const Eigen::Vector3d omega(0.0, 0.0, 0.1), accel(0.0, 0.0, 0.0);
  const double dt = 1e-4;
  ekf.propagate(omega, accel, Cov15::Identity() * 1e-6, dt);

  const Eigen::Vector3d g(0.0, 0.0, -9.81);
  const Rot3 R1 = R0 * Rot3::Expmap(omega * dt);
  const Eigen::Vector3d v1 = v0 + (R0 * accel + g) * dt;
  const Eigen::Vector3d p1 = p0 + v0 * dt;

  EXPECT(ekf.attitude().equals(R1, 1e-9));
  EXPECT(veq(ekf.velocity(), v1, 1e-7));
  EXPECT(veq(ekf.position(), p1, 1e-7));
}

// Covariance grows when propagating with process noise and no measurement.
TEST(Mekf, PredictInflatesCovariance) {
  MultiplicativeEKF ekf(MekfState::identity(), Cov15::Identity() * 0.1);
  const double trace_before = ekf.covariance().trace();
  ekf.propagate(Eigen::Vector3d(0.01, 0.02, 0.03),
                Eigen::Vector3d(0.0, 0.0, 9.81), Cov15::Identity() * 1e-2, 0.01);
  EXPECT(ekf.covariance().trace() > trace_before);
}

// ---------------------------------------------------------------------------
// Position update
// ---------------------------------------------------------------------------

// A position measurement reduces the position-block covariance.
TEST(Mekf, UpdateReducesPositionCovariance) {
  Cov15 P0 = Cov15::Identity() * 0.1;
  P0.block<3, 3>(6, 6) = Cov3::Identity() * 4.0;  // large position uncertainty
  MultiplicativeEKF ekf(MekfState::identity(), P0);

  const double trace_before = ekf.covariance().block<3, 3>(6, 6).trace();
  ekf.update_position(Eigen::Vector3d(0.5, -0.3, 0.2), Cov3::Identity() * 0.01);
  const double trace_after = ekf.covariance().block<3, 3>(6, 6).trace();

  EXPECT(trace_after < trace_before);
  EXPECT(minEig(ekf.covariance()) > 0.0);
}

// A single accurate update pulls a wrong position estimate toward the fix.
TEST(Mekf, UpdateCorrectsPosition) {
  MekfState X0(Rot3::Identity(), Eigen::Vector3d::Zero(),
               Eigen::Vector3d(1.0, -1.0, 0.5),  // wrong position
               Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero());
  Cov15 P0 = Cov15::Identity() * 1e-4;
  P0.block<3, 3>(6, 6) = Cov3::Identity() * 1.0;  // trust measurement more
  MultiplicativeEKF ekf(X0, P0);

  const Eigen::Vector3d truth(0.0, 0.0, 0.0);
  ekf.update_position(truth, Cov3::Identity() * 1e-4);

  // Estimate should move most of the way to the measurement.
  const double err_before = (Eigen::Vector3d(1.0, -1.0, 0.5) - truth).norm();
  const double err_after = (ekf.position() - truth).norm();
  EXPECT(err_after < 0.05 * err_before);
}

/* ************************************************************************* */
int main() {
  TestResult tr;
  return TestRegistry::runAllTests(tr);
}
