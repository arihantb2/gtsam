/**
 * @file  testTGEqFRegression.cpp
 * @brief Scenario regressions for the TG-EqF filter: 30 s biased IMU with and
 * without position aiding, and magnetometer heading aiding with and without
 * yaw turns.
 *
 * Scenario-level accuracy lives here; the unit tests live in testTGEqF.cpp.
 */
#include <CppUnitLite/TestHarness.h>
#include <gtsam/navigation/Scenario.h>
#include <gtsam_unstable/tg_eqf/TGEqF.h>

using namespace gtsam::tgeqf;
using namespace gtsam;

// Min eigenvalue of a symmetric matrix (SPD check).
static double minEig(const Eigen::MatrixXd& M) {
  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(M);
  return es.eigenvalues().minCoeff();
}

// ---------------------------------------------------------------------------
// 30 s regression: biased IMU, with and without position aiding.
//
// Ground truth is a coordinated turn (gtsam::ConstantTwistScenario: constant
// yaw rate + forward body velocity => circle). The turning motion excites the
// centripetal specific force, which makes yaw and the IMU biases observable
// from position fixes. IMU samples are synthesized from the closed-form
// trajectory and corrupted by CONSTANT biases only (no random noise), so the
// run is fully deterministic and the thresholds are stable regression bounds
// calibrated to the measured behaviour.
// ---------------------------------------------------------------------------

struct RegressionResult {
  Vector3 pos_err;             // final position error (m)
  Vector3 vel_err;             // final velocity error (m/s)
  Vector3 att_err;             // final attitude error, Log(gt^-1 est) (rad)
  Vector3 bg_err, ba_err;      // final bias estimate errors
  Vector3 bg_est, ba_est;      // final bias estimates
  double pos_cov_trace = 0.0;  // trace of final position covariance block
  double bg_cov_trace = 0.0;   // trace of final gyro-bias covariance block
  double ba_cov_trace = 0.0;   // trace of final accel-bias covariance block
  double min_eig = 0.0;        // min eigenvalue of final covariance
};

static RegressionResult runBiasedImu30s(bool with_position_updates) {
  // Coordinated turn: 0.3 rad/s yaw, 2 m/s forward => circle of radius ~6.7 m.
  const gtsam::ConstantTwistScenario scenario(Vector3(0, 0, 0.3),
                                              Vector3(2.0, 0, 0));
  // Constant true biases, unknown to the filter (it starts at zero).
  const Vector3 true_bg(0.01, -0.005, 0.02);  // rad/s
  const Vector3 true_ba(0.05, -0.03, 0.02);   // m/s^2

  const double dt = 0.01, duration = 30.0;  // 100 Hz IMU, 30 s
  const int num_steps = static_cast<int>(duration / dt);
  const int gnss_decim = 10;  // 10 Hz position fixes
  const Vector3 g(0.0, 0.0, -9.81);

  // The origin chart is the true initial state, with zero bias estimate and
  // X0 = identity, so the recovered state starts exactly at ground truth. The
  // reset step keeps the fixed-origin linearization valid as the vehicle
  // circles away from it.
  const gtsam::NavState gt0 = scenario.navState(0.0);
  State xi0 = State::identity();
  Eigen::Matrix<double, 3, 2> vp0;
  vp0.col(0) = gt0.velocity();
  vp0.col(1) = gt0.position();
  xi0.T = ExtendedPose3<2>(gt0.attitude(), vp0);  // biases start at zero

  TGEqF::Covariance21 P0 = TGEqF::Covariance21::Zero();
  P0.block<3, 3>(0, 0) = Eigen::Matrix3d::Identity() * 1e-4;    // attitude
  P0.block<3, 3>(3, 3) = Eigen::Matrix3d::Identity() * 1e-2;    // velocity
  P0.block<3, 3>(6, 6) = Eigen::Matrix3d::Identity() * 1e-2;    // position
  P0.block<3, 3>(9, 9) = Eigen::Matrix3d::Identity() * 1e-3;    // gyro bias
  P0.block<3, 3>(12, 12) = Eigen::Matrix3d::Identity() * 1e-2;  // accel bias
  P0.block<3, 3>(15, 15) = Eigen::Matrix3d::Identity() * 1e-4;  // virtual bias
  P0.block<3, 3>(18, 18) = Eigen::Matrix3d::Identity() * 1e-2;  // mag bias
  TGEqF ekf(xi0, P0);

  // Continuous-time process noise (origin-chart densities, P += Qc*dt): gyro on
  // attitude, accel on velocity, random walks on their own bias blocks.
  TGEqF::Covariance21 Qc = TGEqF::Covariance21::Zero();
  Qc.block<3, 3>(0, 0) = Eigen::Matrix3d::Identity() * 1e-6;    // gyro
  Qc.block<3, 3>(3, 3) = Eigen::Matrix3d::Identity() * 1e-4;    // accel
  Qc.block<3, 3>(9, 9) = Eigen::Matrix3d::Identity() * 1e-9;    // gyro RW
  Qc.block<3, 3>(12, 12) = Eigen::Matrix3d::Identity() * 1e-8;  // accel RW
  const TGEqF::Covariance3 R_pos =
      TGEqF::Covariance3::Identity() * (0.05 * 0.05);

  double t = 0.0;
  for (int k = 0; k < num_steps; ++k) {
    // Ideal IMU at time t (specific force f = R^T (a_n - g)) plus the biases.
    const gtsam::NavState gt = scenario.navState(t);
    const Vector3 omega_meas = scenario.omega_b(t) + true_bg;
    const Vector3 accel_meas =
        gt.attitude().unrotate(scenario.acceleration_n(t) - g) + true_ba;
    ekf.propagate(omega_meas, accel_meas, g, Qc, dt);
    t += dt;
    if (with_position_updates && (k + 1) % gnss_decim == 0) {
      ekf.update_position(scenario.navState(t).position(), R_pos);
    }
  }

  const gtsam::NavState gt_end = scenario.navState(duration);
  RegressionResult r;
  r.pos_err = Vector3(gt_end.position()) - ekf.position();
  r.vel_err = Vector3(gt_end.velocity()) - ekf.velocity();
  r.att_err = gtsam::Rot3::Logmap(gt_end.attitude().between(ekf.attitude()));
  r.bg_est = ekf.bias_gyro();
  r.ba_est = ekf.bias_accel();
  r.bg_err = r.bg_est - true_bg;
  r.ba_err = r.ba_est - true_ba;
  // Origin-chart covariance: canonical [att,vel,pos,b_w,b_a,b_v,b_m] order.
  const TGEqF::Covariance21 P = ekf.errorCovariance();
  r.pos_cov_trace = P.block<3, 3>(6, 6).trace();
  r.bg_cov_trace = P.block<3, 3>(9, 9).trace();
  r.ba_cov_trace = P.block<3, 3>(12, 12).trace();
  r.min_eig = minEig(P);
  return r;
}

// Pretty-print the regression metrics (regression tests only).
static void printRegression(const std::string& tag, const RegressionResult& r) {
  printf(
      "[%s] pos=%.4e m  vel=%.4e m/s  att=%.4e rad  "
      "bg_err=%.4e  ba_err=%.4e  ba_err.z=%.4e  sigma_ba=%.4e  "
      "pos_cov=%.4e  bg_cov=%.4e  min_eig=%.4e\n",
      tag.c_str(), r.pos_err.norm(), r.vel_err.norm(), r.att_err.norm(),
      r.bg_err.norm(), r.ba_err.norm(), std::abs(r.ba_err.z()),
      std::sqrt(r.ba_cov_trace / 3.0), r.pos_cov_trace, r.bg_cov_trace,
      r.min_eig);
}

TEST(TGEqF, Regression30sBiasedImuWithPositionUpdates) {
  const RegressionResult r = runBiasedImu30s(/*with_position_updates=*/true);
  printRegression("aided", r);

  // Navigation errors stay small under aiding even though the filter starts
  // with no bias estimate. Measured: pos 2.5e-4 m, vel 5.5e-4 m/s, att 3.4e-3
  // rad (~0.19 deg). Position and velocity are held by the 5 cm fixes, and
  // attitude follows from the exact navigation-error linearization the SE_2(3)
  // logarithm chart gives.
  EXPECT(r.pos_err.norm() < 1e-3);
  EXPECT(r.vel_err.norm() < 2e-3);
  EXPECT(r.att_err.norm() < 0.01);  // ~0.6 deg

  // Gyro bias is recovered well (truth |bg| = 0.023, measured error 9.0e-4).
  // The accel bias splits by observability: the z component, which pairs with
  // gravity, recovers to ~8.7e-4, while the horizontal components stay partly
  // entangled with the tilt. The filter's own sigma_ba ~ 0.058 covers that
  // residual, so the estimate is consistent rather than diverging. Truth
  // |ba| = 0.062, measured total error 3.0e-2.
  EXPECT(r.bg_err.norm() < 2e-3);
  EXPECT(r.ba_err.norm() < 0.035);        // > 40% of the bias recovered
  EXPECT(std::abs(r.ba_err.z()) < 2e-3);  // observable axis
  const double sigma_ba = std::sqrt(r.ba_cov_trace / 3.0);
  EXPECT(r.ba_err.norm() < 3.0 * sigma_ba);  // filter consistency

  // The covariance stays SPD and the gyro-bias block converges well below its
  // prior: measured trace 1.3e-5 against a 3e-3 prior.
  EXPECT(r.min_eig > 0.0);
  EXPECT(r.bg_cov_trace < 5e-5);
}

TEST(TGEqF, Regression30sAidedVsImuOnly) {
  const RegressionResult aided =
      runBiasedImu30s(/*with_position_updates=*/true);
  const RegressionResult imu_only =
      runBiasedImu30s(/*with_position_updates=*/false);
  printRegression("aided", aided);
  printRegression("imu_only", imu_only);

  // Dead reckoning with a biased IMU drifts far; aiding contains the error to
  // sub-mm (measured: imu_only ~152 m, aided 2.5e-4 m after 30 s).
  EXPECT(imu_only.pos_err.norm() > 10.0);
  EXPECT(aided.pos_err.norm() < 1e-5 * imu_only.pos_err.norm());

  // Without aiding the biases are unobservable: the estimate never moves from
  // its zero initialisation (propagate keeps the physical bias constant).
  EXPECT(imu_only.bg_est.norm() < 1e-6);
  EXPECT(imu_only.ba_est.norm() < 1e-6);

  // The filter knows it: position covariance grows instead of shrinking.
  EXPECT(imu_only.pos_cov_trace > 10.0 * aided.pos_cov_trace);
  EXPECT(imu_only.min_eig > 0.0);
}

// ---------------------------------------------------------------------------
// Magnetometer heading aiding (notes, Sec. Observability of Option B).
//
// DVL alone leaves heading unobservable. A magnetometer with an unknown
// hard-iron bias bm only sees R^T m + bm, so at a fixed heading an error in
// yaw is matched by a bias change. Yaw turns rotate that matching bias in the
// body frame, which separates the two. Both runs below start with a 0.2 rad
// heading error and a zero bias estimate against a true bias of ~0.1; the
// only difference is the yaw rate. IMU and magnetometer data are noise-free
// and the IMU is unbiased, so the runs are deterministic.
// ---------------------------------------------------------------------------

struct HeadingResult {
  double yaw_err = 0.0;    // final |heading error| (rad)
  double yaw_sigma = 0.0;  // final heading standard deviation (rad)
  Vector3 bm_err;          // final magnetometer-bias error
  double min_eig = 0.0;    // min eigenvalue of final covariance
};

static HeadingResult runMagHeading60s(double yaw_rate) {
  const gtsam::ConstantTwistScenario scenario(Vector3(0, 0, yaw_rate),
                                              Vector3(2.0, 0, 0));
  const Vector3 field(0.4, 0.05, 0.9);       // reference field, normalized
  const Vector3 true_bm(0.08, -0.05, 0.03);  // hard-iron bias
  const double initial_yaw_error = 0.2;      // rad

  const double dt = 0.01, duration = 60.0;  // 100 Hz IMU, 60 s
  const int num_steps = static_cast<int>(duration / dt);
  const int aiding_decim = 10;  // 10 Hz DVL and magnetometer
  const Vector3 g(0.0, 0.0, -9.81);

  // Origin at the true initial state with the heading rotated by the initial
  // error, bias estimate zero.
  const gtsam::NavState gt0 = scenario.navState(0.0);
  State xi0 = State::identity();
  Eigen::Matrix<double, 3, 2> vp0;
  vp0.col(0) = gt0.velocity();
  vp0.col(1) = gt0.position();
  xi0.T = ExtendedPose3<2>(Rot3::Rz(initial_yaw_error) * gt0.attitude(), vp0);

  TGEqF::Covariance21 P0 = TGEqF::Covariance21::Zero();
  P0.block<3, 3>(0, 0) = Vector3(1e-4, 1e-4, 0.09).asDiagonal();  // attitude
  P0.block<3, 3>(3, 3) = Eigen::Matrix3d::Identity() * 1e-1;      // velocity
  P0.block<3, 3>(6, 6) = Eigen::Matrix3d::Identity() * 1e-2;      // position
  P0.block<3, 3>(9, 9) = Eigen::Matrix3d::Identity() * 1e-8;      // gyro bias
  P0.block<3, 3>(12, 12) = Eigen::Matrix3d::Identity() * 1e-6;    // accel bias
  P0.block<3, 3>(15, 15) = Eigen::Matrix3d::Identity() * 1e-8;  // virtual bias
  P0.block<3, 3>(18, 18) = Eigen::Matrix3d::Identity() * 1e-2;  // mag bias
  TGEqF ekf(xi0, P0);

  ImuNoise noise;
  noise.gyro = Vector3::Constant(1e-8);
  noise.accel = Vector3::Constant(1e-6);
  noise.gyro_rw = Vector3::Constant(1e-12);
  noise.accel_rw = Vector3::Constant(1e-10);
  noise.mag_bias_rw = Vector3::Constant(1e-10);
  const TGEqF::Covariance3 R_dvl = TGEqF::Covariance3::Identity() * 1e-4;
  const TGEqF::Covariance3 R_mag = TGEqF::Covariance3::Identity() * 2.5e-5;

  double t = 0.0;
  for (int k = 0; k < num_steps; ++k) {
    const gtsam::NavState gt = scenario.navState(t);
    const Vector3 omega_meas = scenario.omega_b(t);
    const Vector3 accel_meas =
        gt.attitude().unrotate(scenario.acceleration_n(t) - g);
    ekf.propagate(omega_meas, accel_meas, g, noise, dt);
    t += dt;
    if ((k + 1) % aiding_decim == 0) {
      const gtsam::NavState gt_now = scenario.navState(t);
      ekf.update_dvl(gt_now.bodyVelocity(), R_dvl);
      ekf.update_magnetometer(gt_now.attitude().unrotate(field) + true_bm,
                              field, R_mag);
    }
  }

  const gtsam::NavState gt_end = scenario.navState(duration);
  HeadingResult r;
  r.yaw_err =
      std::abs(Rot3::Logmap(gt_end.attitude().between(ekf.attitude())).z());
  // covariance() is in the chart at the estimate, whose attitude block is a
  // body-frame rotation; heading is rotation about the global z axis.
  const TGEqF::Covariance21 P_est = ekf.covariance();
  const Vector3 up_body = ekf.attitude().unrotate(Vector3::UnitZ());
  r.yaw_sigma =
      std::sqrt(up_body.transpose() * P_est.block<3, 3>(0, 0) * up_body);
  r.bm_err = ekf.bias_mag() - true_bm;
  r.min_eig = minEig(ekf.errorCovariance());
  return r;
}

// With repeated yaw turns the magnetometer recovers both the heading and the
// hard-iron bias from a 0.2 rad heading error and an unknown bias.
TEST(TGEqF, YawTurnsSeparateHeadingFromHardIron) {
  const HeadingResult r = runMagHeading60s(/*yaw_rate=*/0.3);
  printf("[mag turning] yaw_err=%.4e  yaw_sigma=%.4e  bm_err=%.4e\n", r.yaw_err,
         r.yaw_sigma, r.bm_err.norm());

  EXPECT(r.yaw_err < 0.01);
  EXPECT(r.bm_err.norm() < 0.01);
  EXPECT(r.min_eig > 0.0);
}

// Without yaw turns the heading error and the horizontal bias stay entangled:
// the heading is neither corrected nor claimed as known, while the turning
// run pins it down.
TEST(TGEqF, StraightRunLeavesHeadingEntangledWithHardIron) {
  const HeadingResult turning = runMagHeading60s(/*yaw_rate=*/0.3);
  const HeadingResult straight = runMagHeading60s(/*yaw_rate=*/0.0);
  printf("[mag straight] yaw_err=%.4e  yaw_sigma=%.4e  bm_err=%.4e\n",
         straight.yaw_err, straight.yaw_sigma, straight.bm_err.norm());

  EXPECT(straight.yaw_err > 0.03);
  EXPECT(straight.yaw_err < 3.0 * straight.yaw_sigma);  // still consistent
  EXPECT(straight.yaw_sigma > 5.0 * turning.yaw_sigma);
}

/* ************************************************************************* */
int main() {
  TestResult tr;
  return TestRegistry::runAllTests(tr);
}
