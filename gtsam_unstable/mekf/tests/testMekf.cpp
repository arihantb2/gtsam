/**
 * @file  testMekf.cpp
 * @brief Integration tests for the MultiplicativeEKF wrapper (predict +
 * update).
 *
 * Tangent block order: see ../README.md.
 * Position block lives at indices 6..8, magnetometer bias at 15..17.
 */
#include <CppUnitLite/TestHarness.h>
#include <gtsam/base/TestableAssertions.h>
#include <gtsam/navigation/Scenario.h>
#include <gtsam_unstable/mekf/MEKF.h>

using namespace mekf;
using namespace gtsam;

static constexpr double kTol = 1e-9;

// The filter is frame-agnostic -- gravity is a propagate() argument -- so these
// tests pick a convention of their own: Z-up (ENU), gravity along -z.
static const Eigen::Vector3d kTestGravity(0.0, 0.0, -9.81);

using Cov18 = MultiplicativeEKF::Covariance;
using Cov3 = Eigen::Matrix<double, 3, 3>;

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
  Cov18 P0 = Cov18::Identity() * 0.5;
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
  MultiplicativeEKF ekf(MekfState::identity(), Cov18::Identity() * 0.1);

  Cov18 Qc = Cov18::Identity() * 1e-3;
  ekf.propagate(Eigen::Vector3d(0.05, -0.02, 0.1),
                Eigen::Vector3d(0.0, 0.0, 9.81), kTestGravity, Qc, 0.01);

  const Cov18 P = ekf.covariance();
  EXPECT(assert_equal((Matrix)P, (Matrix)P.transpose(), 1e-12));  // symmetric
  EXPECT(minEig(P) > 0.0);                                        // PSD
}

// Mean prediction matches the world-frame strapdown step (Eq. 3).
TEST(Mekf, PredictMeanFollowsDynamics) {
  const Rot3 R0 = Rot3::Rz(0.3);
  const Eigen::Vector3d v0(0.2, -0.1, 0.0), p0(1.0, 0.0, -1.0);
  MultiplicativeEKF ekf(
      MekfState(R0, v0, p0, Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero()),
      Cov18::Identity() * 0.01);

  const Eigen::Vector3d omega(0.0, 0.0, 0.1), accel(0.0, 0.0, 0.0);
  const double dt = 1e-4;
  ekf.propagate(omega, accel, kTestGravity, Cov18::Identity() * 1e-6, dt);

  const Eigen::Vector3d g = kTestGravity;
  const Rot3 R1 = R0 * Rot3::Expmap(omega * dt);
  const Eigen::Vector3d v1 = v0 + (R0 * accel + g) * dt;
  const Eigen::Vector3d p1 = p0 + v0 * dt;

  EXPECT(ekf.attitude().equals(R1, 1e-9));
  EXPECT(veq(ekf.velocity(), v1, 1e-7));
  EXPECT(veq(ekf.position(), p1, 1e-7));
}

// Covariance grows when propagating with process noise and no measurement.
TEST(Mekf, PredictInflatesCovariance) {
  MultiplicativeEKF ekf(MekfState::identity(), Cov18::Identity() * 0.1);
  const double trace_before = ekf.covariance().trace();
  ekf.propagate(Eigen::Vector3d(0.01, 0.02, 0.03),
                Eigen::Vector3d(0.0, 0.0, 9.81), kTestGravity,
                Cov18::Identity() * 1e-2, 0.01);
  EXPECT(ekf.covariance().trace() > trace_before);
}

// ImuNoise overload must match propagate with the equivalent diagonal Qc.
TEST(Mekf, PropagateImuNoiseMatchesDiagonalQc) {
  ImuNoise nz;
  nz.gyro = Eigen::Vector3d::Constant(1e-4);
  nz.accel = Eigen::Vector3d::Constant(1e-3);
  nz.gyro_rw = Eigen::Vector3d::Constant(1e-6);
  nz.accel_rw = Eigen::Vector3d::Constant(1e-5);
  nz.mag_bias_rw = Eigen::Vector3d::Constant(1e-7);
  const Eigen::Vector3d omega(0.05, -0.02, 0.01);
  const Eigen::Vector3d accel(0.1, -0.2, 9.85);
  const double dt = 0.01;

  MultiplicativeEKF f1(MekfState::identity(), Cov18::Identity() * 0.1);
  MultiplicativeEKF f2(MekfState::identity(), Cov18::Identity() * 0.1);

  Cov18 Qc = Cov18::Zero();
  Qc.block<3, 3>(0, 0) = nz.gyro.asDiagonal();
  Qc.block<3, 3>(3, 3) = nz.accel.asDiagonal();
  Qc.block<3, 3>(9, 9) = nz.gyro_rw.asDiagonal();
  Qc.block<3, 3>(12, 12) = nz.accel_rw.asDiagonal();
  Qc.block<3, 3>(15, 15) = nz.mag_bias_rw.asDiagonal();

  f1.propagate(omega, accel, kTestGravity, nz, dt);
  f2.propagate(omega, accel, kTestGravity, Qc, dt);

  EXPECT(traits<MekfState>::Equals(f1.state(), f2.state(), 1e-12));
  EXPECT(assert_equal((Matrix)f1.covariance(), (Matrix)f2.covariance(), 1e-9));
}

// ---------------------------------------------------------------------------
// Position update
// ---------------------------------------------------------------------------

// A position measurement reduces the position-block covariance.
TEST(Mekf, UpdateReducesPositionCovariance) {
  Cov18 P0 = Cov18::Identity() * 0.1;
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
  Cov18 P0 = Cov18::Identity() * 1e-4;
  P0.block<3, 3>(6, 6) = Cov3::Identity() * 1.0;  // trust measurement more
  MultiplicativeEKF ekf(X0, P0);

  const Eigen::Vector3d truth(0.0, 0.0, 0.0);
  ekf.update_position(truth, Cov3::Identity() * 1e-4);

  // Estimate should move most of the way to the measurement.
  const double err_before = (Eigen::Vector3d(1.0, -1.0, 0.5) - truth).norm();
  const double err_after = (ekf.position() - truth).norm();
  EXPECT(err_after < 0.05 * err_before);
}

// ---------------------------------------------------------------------------
// DVL update
// ---------------------------------------------------------------------------

// A DVL measurement reduces the velocity-block covariance.
TEST(Mekf, UpdateDvlReducesVelocityCovariance) {
  Cov18 P0 = Cov18::Identity() * 0.1;
  P0.block<3, 3>(3, 3) = Cov3::Identity() * 4.0;  // large velocity uncertainty
  MultiplicativeEKF ekf(MekfState::identity(), P0);

  const double trace_before = ekf.covariance().block<3, 3>(3, 3).trace();
  ekf.update_dvl(Eigen::Vector3d(0.3, -0.1, 0.0), Cov3::Identity() * 0.01);
  const double trace_after = ekf.covariance().block<3, 3>(3, 3).trace();

  EXPECT(trace_after < trace_before);
  EXPECT(minEig(ekf.covariance()) > 0.0);
}

// A single accurate DVL update pulls a wrong velocity estimate toward truth.
TEST(Mekf, UpdateDvlCorrectsVelocity) {
  const Rot3 R0 = Rot3::Rz(0.5);
  const Eigen::Vector3d truth_v = R0 * Eigen::Vector3d(0.2, -0.1, 0.05);
  MekfState X0(R0, Eigen::Vector3d(1.0, 1.0, -1.0),  // wrong world velocity
               Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero(),
               Eigen::Vector3d::Zero());
  Cov18 P0 = Cov18::Identity() * 1e-4;
  P0.block<3, 3>(3, 3) = Cov3::Identity() * 1.0;  // trust the DVL more
  MultiplicativeEKF ekf(X0, P0);

  // DVL reports the body-frame velocity consistent with truth_v: R0^T truth_v.
  const Eigen::Vector3d z_dvl = R0.unrotate(truth_v);
  ekf.update_dvl(z_dvl, Cov3::Identity() * 1e-4);

  const double err_before = (Eigen::Vector3d(1.0, 1.0, -1.0) - truth_v).norm();
  const double err_after = (ekf.velocity() - truth_v).norm();
  EXPECT(err_after < 0.05 * err_before);
}

/* ************************************************************************* */
namespace depth_update {

using Cov1 = MultiplicativeEKF::Covariance1;

// 1x1 depth noise matrix from a stddev in metres.
Cov1 depthNoise(double sigma) {
  Cov1 R;
  R(0, 0) = sigma * sigma;
  return R;
}

MekfState offsetState() {
  return MekfState(Rot3::Rx(M_PI / 6), Eigen::Vector3d(0.5, -0.2, 0.1),
                   Eigen::Vector3d(1.0, 2.0, -3.0), Eigen::Vector3d::Zero(),
                   Eigen::Vector3d::Zero());
}

// A single depth update pulls the vertical estimate most of the way to the
// measurement when the depth noise is much tighter than the prior.
TEST(Mekf, DepthUpdateCorrectsVerticalPosition) {
  MultiplicativeEKF ekf(MekfState::identity(), Cov18::Identity() * 0.01);

  const double z_depth = -0.8;
  ekf.update_depth(z_depth, depthNoise(0.01));

  EXPECT(std::abs(ekf.position().z() - z_depth) < 0.1);
}

// The inflated horizontal variance keeps a depth update off the x and y axes
// when the prior is uncorrelated.
TEST(Mekf, DepthUpdateLeavesHorizontalNearlyUnchanged) {
  MultiplicativeEKF ekf(offsetState(), Cov18::Identity() * 0.01);

  const Eigen::Vector3d before = ekf.position();
  ekf.update_depth(before.z() - 0.8, depthNoise(0.01));
  const Eigen::Vector3d delta = ekf.position() - before;

  EXPECT(std::abs(delta.z()) > 0.5);
  EXPECT(std::abs(delta.x()) < 0.02);
  EXPECT(std::abs(delta.y()) < 0.02);
}

// Only the vertical position covariance is informed; the horizontal blocks stay
// essentially where they were.
TEST(Mekf, DepthUpdateShrinksVerticalCovariance) {
  MultiplicativeEKF ekf(MekfState::identity(), Cov18::Identity() * 0.01);

  const Cov18 before = ekf.covariance();
  ekf.update_depth(-0.8, depthNoise(0.01));
  const Cov18 after = ekf.covariance();

  const double drop_z = before(8, 8) - after(8, 8);
  EXPECT(drop_z > 0.0);
  EXPECT(std::abs(before(6, 6) - after(6, 6)) < 0.1 * drop_z);
  EXPECT(std::abs(before(7, 7) - after(7, 7)) < 0.1 * drop_z);
  EXPECT(minEig(after) > 0.0);
}

// update_depth is exactly the pseudo-position update it documents.
TEST(Mekf, DepthUpdateMatchesEquivalentPositionUpdate) {
  MultiplicativeEKF via_depth(offsetState(), Cov18::Identity() * 0.01);
  MultiplicativeEKF via_position(offsetState(), Cov18::Identity() * 0.01);

  const double z_depth = -2.0;
  const double sigma_z = 0.02;
  via_depth.update_depth(z_depth, depthNoise(sigma_z));

  Cov3 R_pseudo = Cov3::Zero();
  R_pseudo(0, 0) = MultiplicativeEKF::kDefaultHorizontalVariance;
  R_pseudo(1, 1) = MultiplicativeEKF::kDefaultHorizontalVariance;
  R_pseudo(2, 2) = sigma_z * sigma_z;
  Eigen::Vector3d pseudo = via_position.position();
  pseudo.z() = z_depth;
  via_position.update_position(pseudo, R_pseudo);

  EXPECT(traits<MekfState>::Equals(via_position.state(), via_depth.state(),
                                   1e-12));
  EXPECT(assert_equal((Matrix)via_position.covariance(),
                      (Matrix)via_depth.covariance(), 1e-12));
}

// The horizontal-variance argument is live: shrinking it changes the correction
// a correlated prior produces, so the default is not silently ignored.
TEST(Mekf, DepthUpdateHorizontalVarianceIsRespected) {
  // Correlate p_x with p_z; the block [[0.01, 0.008], [0.008, 0.01]] is SPD.
  Cov18 P0 = Cov18::Identity() * 0.01;
  P0(6, 8) = P0(8, 6) = 0.008;

  MultiplicativeEKF inflated(MekfState::identity(), P0);
  MultiplicativeEKF pinned(MekfState::identity(), P0);

  const double z_depth = -0.8;
  inflated.update_depth(z_depth, depthNoise(0.01));
  pinned.update_depth(z_depth, depthNoise(0.01), /*horizontal_variance=*/1e-6);

  EXPECT((inflated.position() - pinned.position()).norm() > 1e-3);
}

}  // namespace depth_update
/* ************************************************************************* */

/* ************************************************************************* */
namespace mag_update {

/// Reference field in the navigation frame: horizontal north and east parts
/// and a steeper vertical part, as at mid latitudes (normalized units).
const Eigen::Vector3d kField(0.4, 0.05, 0.9);

/// Prior with the attitude and magnetometer-bias variances set separately.
Cov18 magSigma(double attitude_variance, double mag_bias_variance) {
  Cov18 P0 = Cov18::Identity() * 0.01;
  P0.block<3, 3>(0, 0) = attitude_variance * Cov3::Identity();
  P0.block<3, 3>(15, 15) = mag_bias_variance * Cov3::Identity();
  return P0;
}

// initialCovariance places the navigation block first and the magnetometer
// bias last, uncorrelated.
TEST(Mekf, InitialCovarianceIsBlockDiagonal) {
  const MultiplicativeEKF::Covariance15 Sigma_nav =
      MultiplicativeEKF::Covariance15::Identity() * 0.3;
  const Cov3 Sigma_mag = Cov3::Identity() * 0.7;
  const Cov18 P0 = MultiplicativeEKF::initialCovariance(Sigma_nav, Sigma_mag);

  EXPECT(assert_equal((Matrix)Sigma_nav, (Matrix)P0.topLeftCorner<15, 15>()));
  EXPECT(assert_equal((Matrix)Sigma_mag, (Matrix)P0.block<3, 3>(15, 15)));
  EXPECT(assert_equal((Matrix)Eigen::Matrix<double, 15, 3>::Zero(),
                      (Matrix)P0.topRightCorner<15, 3>()));
}

// With the bias and the roll and pitch known (a level vehicle), a tight
// magnetometer reading of a yawed body pulls the heading onto the truth: the
// field's horizontal part constrains yaw. Roll and pitch are pinned because
// an isotropic attitude prior would spend part of the innovation on rotation
// about the field direction, which the magnetometer cannot see.
TEST(Mekf, MagUpdateCorrectsHeadingWithKnownBias) {
  Cov18 P0 = magSigma(1e-12, 1e-12);
  P0(2, 2) = 0.01;
  MultiplicativeEKF ekf(MekfState::identity(), P0);
  const Rot3 R_true = Rot3::Rz(0.1);

  ekf.update_magnetometer(R_true.unrotate(kField), kField,
                          1e-8 * Cov3::Identity());

  EXPECT(std::abs(ekf.attitude().yaw() - 0.1) < 1e-3);
  EXPECT(veq(Eigen::Vector3d::Zero(), ekf.bias_mag(), 1e-6));
}

// With the attitude known, the same update estimates the hard-iron bias as
// the difference between the reading and the predicted field.
TEST(Mekf, MagUpdateCorrectsBiasAtKnownAttitude) {
  MultiplicativeEKF ekf(MekfState::identity(), magSigma(1e-12, 0.01));
  const Eigen::Vector3d bm_true(0.05, -0.03, 0.02);

  ekf.update_magnetometer(kField + bm_true, kField, 1e-8 * Cov3::Identity());

  EXPECT(veq(bm_true, ekf.bias_mag(), 1e-5));
  EXPECT(ekf.attitude().equals(Rot3::Identity(), 1e-6));
}

// The update is the textbook EKF step: innovation z - h(X_hat), the
// state-dependent H, gain K = P H^T S^-1, correction Retract(X_hat, K nu) and
// the Joseph-form covariance. This rebuilds it by hand at a rotated, biased
// estimate with a correlated prior and an anisotropic R_mag.
TEST(Mekf, MagUpdateMatchesTheHandBuiltEkfUpdate) {
  const MekfState X0(
      Rot3::Rz(M_PI / 2) * Rot3::Rx(0.2), Eigen::Vector3d(1.0, -0.5, 0.3),
      Eigen::Vector3d(2.0, 1.0, -1.0), Eigen::Vector3d(0.01, -0.02, 0.03),
      Eigen::Vector3d(-0.05, 0.1, 0.0), Eigen::Vector3d(0.05, -0.08, 0.03));
  Cov18 P0 = Cov18::Identity() * 0.01;
  P0(0, 15) = P0(15, 0) = 0.004;  // attitude-bias correlation
  P0(2, 4) = P0(4, 2) = 0.003;    // attitude-velocity correlation
  MultiplicativeEKF ekf(X0, P0);

  const Eigen::Vector3d z_mag = Rot3::Rz(M_PI / 2 + 0.05).unrotate(kField) +
                                Eigen::Vector3d(0.01, -0.02, 0);
  const Cov3 R_mag = Eigen::Vector3d(1e-4, 4e-4, 9e-4).asDiagonal();

  const Eigen::Vector3d nu = z_mag - MagnetometerOutput::predict(X0, kField);
  const Eigen::Matrix<double, 3, 18> H =
      MagnetometerOutput::jacobian(X0, kField);
  const Eigen::Matrix<double, 18, 3> K =
      P0 * H.transpose() * (H * P0 * H.transpose() + R_mag).inverse();
  const Cov18 I_KH = Cov18::Identity() - K * H;

  ekf.update_magnetometer(z_mag, kField, R_mag);

  EXPECT(traits<MekfState>::Equals(traits<MekfState>::Retract(X0, K * nu),
                                   ekf.state(), 1e-9));
  EXPECT(assert_equal(
      (Matrix)(I_KH * P0 * I_KH.transpose() + K * R_mag * K.transpose()),
      (Matrix)ekf.covariance(), 1e-12));
}

// With an uncorrelated prior the magnetometer reaches only attitude and its
// own bias; velocity, position and IMU biases stay where they were.
TEST(Mekf, MagUpdateOnlyCorrectsAttitudeAndMagBias) {
  MultiplicativeEKF ekf(MekfState::identity(), Cov18::Identity() * 0.01);

  ekf.update_magnetometer(
      Rot3::Rz(0.2).unrotate(kField) + Eigen::Vector3d(0.02, 0.0, -0.01),
      kField, 1e-4 * Cov3::Identity());

  EXPECT(!ekf.attitude().equals(Rot3::Identity(), 1e-3));
  EXPECT(ekf.bias_mag().norm() > 1e-3);
  EXPECT(veq(Eigen::Vector3d::Zero(), ekf.velocity(), 1e-12));
  EXPECT(veq(Eigen::Vector3d::Zero(), ekf.position(), 1e-12));
  EXPECT(veq(Eigen::Vector3d::Zero(), ekf.bias_gyro(), 1e-12));
  EXPECT(veq(Eigen::Vector3d::Zero(), ekf.bias_accel(), 1e-12));
}

}  // namespace mag_update
/* ************************************************************************* */

/* ************************************************************************* */
namespace decoupling {

/// Propagate with every noise channel, including the bias random walk, and
/// apply DVL, position and depth updates, but no magnetometer update.
MultiplicativeEKF runWithoutMagUpdates(double mag_bias_variance) {
  MultiplicativeEKF::Covariance15 Sigma_nav =
      MultiplicativeEKF::Covariance15::Identity() * 0.01;
  Sigma_nav(0, 4) = Sigma_nav(4, 0) = 0.004;
  MultiplicativeEKF ekf(
      MekfState(Rot3::Rz(M_PI / 2), Eigen::Vector3d::Zero(),
                Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero(),
                Eigen::Vector3d::Zero()),
      MultiplicativeEKF::initialCovariance(
          Sigma_nav, mag_bias_variance * Cov3::Identity()));

  ImuNoise nz;
  nz.gyro = Eigen::Vector3d::Constant(2e-4);
  nz.accel = Eigen::Vector3d::Constant(3e-3);
  nz.gyro_rw = Eigen::Vector3d::Constant(1e-6);
  nz.accel_rw = Eigen::Vector3d::Constant(5e-5);
  nz.mag_bias_rw = Eigen::Vector3d::Constant(1e-5);
  for (int step = 0; step < 20; ++step) {
    ekf.propagate(Eigen::Vector3d(0.05, -0.02, 0.2),
                  Eigen::Vector3d(0.3, -0.1, 9.9), kTestGravity, nz, 0.05);
  }
  ekf.update_dvl(Eigen::Vector3d(2.0, 0.1, -0.05), 1e-3 * Cov3::Identity());
  ekf.update_position(ekf.position() + Eigen::Vector3d(0.2, -0.1, 0.05),
                      1e-2 * Cov3::Identity());
  MultiplicativeEKF::Covariance1 R_depth;
  R_depth(0, 0) = 1e-4;
  ekf.update_depth(ekf.position().z() + 0.05, R_depth);
  return ekf;
}

// Without magnetometer updates the bias block never couples to the rest: the
// navigation estimate and covariance do not depend on the bias prior. The
// cross-covariance and the bias estimate are zero up to the noise floor of
// the finite-difference F (~1e-9 per unit bias variance), whose bias columns
// pick up Logmap(R^T R) / h from round-off.
TEST(Mekf, NavigationIsUnaffectedByTheMagBlockWithoutMagUpdates) {
  const MultiplicativeEKF tight = runWithoutMagUpdates(1e-4);
  const MultiplicativeEKF loose = runWithoutMagUpdates(1.0);

  const MultiplicativeEKF::TangentVector difference =
      traits<MekfState>::Local(tight.state(), loose.state());
  EXPECT(assert_equal((Vector)Eigen::Matrix<double, 15, 1>::Zero(),
                      (Vector)difference.head<15>(), 1e-12));
  EXPECT(assert_equal((Matrix)tight.covariance().topLeftCorner<15, 15>(),
                      (Matrix)loose.covariance().topLeftCorner<15, 15>(),
                      1e-12));
  EXPECT(assert_equal((Matrix)Eigen::Matrix<double, 15, 3>::Zero(),
                      (Matrix)loose.covariance().topRightCorner<15, 3>(),
                      1e-9));
  EXPECT(veq(Eigen::Vector3d::Zero(), loose.bias_mag(), 1e-9));
}

}  // namespace decoupling
/* ************************************************************************* */

/* ************************************************************************* */
namespace mag_heading {

struct HeadingResult {
  double yaw_err = 0.0;    // final absolute heading error (rad)
  double yaw_sigma = 0.0;  // final heading standard deviation (rad)
  Eigen::Vector3d bm_err;  // final magnetometer-bias error
  double min_eig = 0.0;    // min eigenvalue of final covariance
};

/// 60 s at constant yaw rate and 2 m/s surge with 10 Hz DVL and magnetometer,
/// from an initial heading error (rad) and an unknown hard-iron bias. With the
/// default 0.2 rad error this is the TG-EqF heading regression scenario, so
/// the two filters can be compared.
HeadingResult runMagHeading60s(double yaw_rate,
                               double initial_yaw_error = 0.2) {
  const ConstantTwistScenario scenario(Eigen::Vector3d(0, 0, yaw_rate),
                                       Eigen::Vector3d(2.0, 0, 0));
  const Eigen::Vector3d field(0.4, 0.05, 0.9);       // normalized
  const Eigen::Vector3d true_bm(0.08, -0.05, 0.03);  // hard-iron bias

  const double dt = 0.01, duration = 60.0;  // 100 Hz IMU, 60 s
  const int num_steps = static_cast<int>(duration / dt);
  const int aiding_decim = 10;  // 10 Hz DVL and magnetometer
  const Eigen::Vector3d g = kTestGravity;

  // True initial state with the heading rotated by the initial error, bias
  // estimate zero.
  const NavState gt0 = scenario.navState(0.0);
  const MekfState X0(Rot3::Rz(initial_yaw_error) * gt0.attitude(),
                     gt0.velocity(), gt0.position(), Eigen::Vector3d::Zero(),
                     Eigen::Vector3d::Zero());

  MultiplicativeEKF::Covariance15 Sigma_nav =
      MultiplicativeEKF::Covariance15::Zero();
  Sigma_nav.block<3, 3>(0, 0) =
      Eigen::Vector3d(1e-4, 1e-4, 0.09).asDiagonal();       // attitude
  Sigma_nav.block<3, 3>(3, 3) = Cov3::Identity() * 1e-1;    // velocity
  Sigma_nav.block<3, 3>(6, 6) = Cov3::Identity() * 1e-2;    // position
  Sigma_nav.block<3, 3>(9, 9) = Cov3::Identity() * 1e-8;    // gyro bias
  Sigma_nav.block<3, 3>(12, 12) = Cov3::Identity() * 1e-6;  // accel bias
  MultiplicativeEKF ekf(X0, MultiplicativeEKF::initialCovariance(
                                Sigma_nav, Cov3::Identity() * 1e-2));

  ImuNoise noise;
  noise.gyro = Eigen::Vector3d::Constant(1e-8);
  noise.accel = Eigen::Vector3d::Constant(1e-6);
  noise.gyro_rw = Eigen::Vector3d::Constant(1e-12);
  noise.accel_rw = Eigen::Vector3d::Constant(1e-10);
  noise.mag_bias_rw = Eigen::Vector3d::Constant(1e-10);
  const Cov3 R_dvl = Cov3::Identity() * 1e-4;
  const Cov3 R_mag = Cov3::Identity() * 2.5e-5;

  double t = 0.0;
  for (int k = 0; k < num_steps; ++k) {
    const NavState gt = scenario.navState(t);
    const Eigen::Vector3d omega_meas = scenario.omega_b(t);
    const Eigen::Vector3d accel_meas =
        gt.attitude().unrotate(scenario.acceleration_n(t) - g);
    ekf.propagate(omega_meas, accel_meas, g, noise, dt);
    t += dt;
    if ((k + 1) % aiding_decim == 0) {
      const NavState gt_now = scenario.navState(t);
      ekf.update_dvl(gt_now.bodyVelocity(), R_dvl);
      ekf.update_magnetometer(gt_now.attitude().unrotate(field) + true_bm,
                              field, R_mag);
    }
  }

  const NavState gt_end = scenario.navState(duration);
  HeadingResult r;
  r.yaw_err =
      std::abs(Rot3::Logmap(gt_end.attitude().between(ekf.attitude())).z());
  // The attitude block of covariance() is a body-frame rotation; heading is
  // rotation about the navigation z axis.
  const Eigen::Vector3d up_body =
      ekf.attitude().unrotate(Eigen::Vector3d::UnitZ());
  r.yaw_sigma = std::sqrt(up_body.transpose() *
                          ekf.covariance().block<3, 3>(0, 0) * up_body);
  r.bm_err = ekf.bias_mag() - true_bm;
  r.min_eig = minEig(ekf.covariance());
  return r;
}

// With repeated yaw turns the magnetometer recovers both the heading and the
// hard-iron bias from a 0.2 rad heading error and an unknown bias.
TEST(Mekf, YawTurnsSeparateHeadingFromHardIron) {
  const HeadingResult r = runMagHeading60s(/*yaw_rate=*/0.3);
  printf("[mekf mag turning] yaw_err=%.4e  yaw_sigma=%.4e  bm_err=%.4e\n",
         r.yaw_err, r.yaw_sigma, r.bm_err.norm());

  EXPECT(r.yaw_err < 0.01);
  EXPECT(r.bm_err.norm() < 0.01);
  EXPECT(r.min_eig > 0.0);
}

// Without yaw turns the heading error and the horizontal bias stay entangled:
// the heading is not corrected, and its variance stays far above that of the
// turning run.
TEST(Mekf, StraightRunLeavesHeadingEntangledWithHardIron) {
  const HeadingResult turning = runMagHeading60s(/*yaw_rate=*/0.3);
  const HeadingResult straight = runMagHeading60s(/*yaw_rate=*/0.0);
  printf("[mekf mag straight] yaw_err=%.4e  yaw_sigma=%.4e  bm_err=%.4e\n",
         straight.yaw_err, straight.yaw_sigma, straight.bm_err.norm());

  EXPECT(straight.yaw_err > 0.03);
  EXPECT(straight.yaw_sigma > 5.0 * turning.yaw_sigma);
}

// Characterizes the MEKF, unlike the TG-EqF on the same scenario: on a
// straight run its heading covariance is consistent from a small initial
// error, but from 0.2 rad the first DVL and magnetometer updates, linearized
// at the wrong heading, collapse the heading variance within a second while
// the error stays: a transient false observability of the unobservable
// heading/hard-iron direction.
TEST(Mekf, StraightRunHeadingIsOverconfidentOnlyFromALargeError) {
  const HeadingResult small = runMagHeading60s(/*yaw_rate=*/0.0, 0.0);
  const HeadingResult large = runMagHeading60s(/*yaw_rate=*/0.0, 0.2);
  printf("[mekf mag straight] from 0 rad: yaw_err=%.4e  yaw_sigma=%.4e\n",
         small.yaw_err, small.yaw_sigma);

  EXPECT(small.yaw_err < 3.0 * small.yaw_sigma);
  EXPECT(large.yaw_err > 3.0 * large.yaw_sigma);
}

}  // namespace mag_heading
/* ************************************************************************* */

/* ************************************************************************* */
int main() {
  TestResult tr;
  return TestRegistry::runAllTests(tr);
}
