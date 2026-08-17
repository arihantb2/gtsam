/**
 * @file  testTGEqFRegression.cpp
 * @brief 30 s scenario regressions for the TG-EqF filter (biased IMU with/
 * without position aiding).
 *
 * Split out of testTGEqF.cpp to keep unit tests and multi-second scenario
 * regressions in separate files.
 */
#include <CppUnitLite/TestHarness.h>
#include <gtsam/navigation/Scenario.h>
#include <gtsam_unstable/tg_eqf/EqF.h>

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
// run is fully deterministic and thresholds are stable regression bounds.
//
// Ported from the TFG-IEKF regression (testTFGInEKF.cpp). Numbers differ: the
// TG-EqF is a different estimator (equivariant filter on SE_2(3) x R^9 with a
// fixed origin chart + reset step), so the thresholds below are calibrated to
// the measured TG-EqF behaviour, not copied from the IEKF.
// ---------------------------------------------------------------------------

struct RegressionResult {
  Eigen::Vector3d pos_err;         // final position error (m)
  Eigen::Vector3d vel_err;         // final velocity error (m/s)
  Eigen::Vector3d att_err;         // final attitude error, Log(gt^-1 est) (rad)
  Eigen::Vector3d bg_err, ba_err;  // final bias estimate errors
  Eigen::Vector3d bg_est, ba_est;  // final bias estimates
  double pos_cov_trace = 0.0;      // trace of final position covariance block
  double bg_cov_trace = 0.0;       // trace of final gyro-bias covariance block
  double ba_cov_trace = 0.0;       // trace of final accel-bias covariance block
  double min_eig = 0.0;            // min eigenvalue of final covariance
};

static RegressionResult runBiasedImu30s(bool with_position_updates) {
  // Coordinated turn: 0.3 rad/s yaw, 2 m/s forward => circle of radius ~6.7 m.
  const gtsam::ConstantTwistScenario scenario(Eigen::Vector3d(0, 0, 0.3),
                                              Eigen::Vector3d(2.0, 0, 0));
  // Constant true biases, unknown to the filter (it starts at zero).
  const Eigen::Vector3d true_bg(0.01, -0.005, 0.02);  // rad/s
  const Eigen::Vector3d true_ba(0.05, -0.03, 0.02);   // m/s^2

  const double dt = 0.01, duration = 30.0;  // 100 Hz IMU, 30 s
  const int num_steps = static_cast<int>(duration / dt);
  const int gnss_decim = 10;  // 10 Hz position fixes
  const Eigen::Vector3d g(0.0, 0.0, -9.81);

  // Origin chart = the true initial state, zero bias estimate, X0 = identity
  // (so the recovered state starts exactly at ground truth). The reset step
  // keeps the fixed-origin linearisation valid as the vehicle circles away.
  const gtsam::NavState gt0 = scenario.navState(0.0);
  State xi0 = State::identity();
  xi0.R = gt0.attitude();
  xi0.v = gt0.velocity();
  xi0.p = gt0.position();  // biases (b_w, b_a, b_v) start at zero

  TGEqF::Covariance18 P0 = TGEqF::Covariance18::Zero();
  P0.block<3, 3>(0, 0) = Eigen::Matrix3d::Identity() * 1e-4;    // attitude
  P0.block<3, 3>(3, 3) = Eigen::Matrix3d::Identity() * 1e-2;    // velocity
  P0.block<3, 3>(6, 6) = Eigen::Matrix3d::Identity() * 1e-2;    // position
  P0.block<3, 3>(9, 9) = Eigen::Matrix3d::Identity() * 1e-3;    // gyro bias
  P0.block<3, 3>(12, 12) = Eigen::Matrix3d::Identity() * 1e-2;  // accel bias
  P0.block<3, 3>(15, 15) = Eigen::Matrix3d::Identity() * 1e-4;  // virtual bias
  TGEqF ekf(xi0, P0);

  // Continuous-time process noise (origin-chart densities, P += Qc*dt). Mirrors
  // the TFG ImuNoise: gyro -> attitude, accel -> velocity, RW -> bias rows.
  TGEqF::Covariance18 Qc = TGEqF::Covariance18::Zero();
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
    const Eigen::Vector3d omega_meas = scenario.omega_b(t) + true_bg;
    const Eigen::Vector3d accel_meas =
        gt.attitude().unrotate(scenario.acceleration_n(t) - g) + true_ba;
    ekf.propagate(omega_meas, accel_meas, g, Qc, dt);
    t += dt;
    if (with_position_updates && (k + 1) % gnss_decim == 0) {
      ekf.update_position(scenario.navState(t).position(), R_pos);
    }
  }

  const gtsam::NavState gt_end = scenario.navState(duration);
  RegressionResult r;
  r.pos_err = Eigen::Vector3d(gt_end.position()) - ekf.position();
  r.vel_err = Eigen::Vector3d(gt_end.velocity()) - ekf.velocity();
  r.att_err = gtsam::Rot3::Logmap(gt_end.attitude().between(ekf.attitude()));
  r.bg_est = ekf.bias_gyro();
  r.ba_est = ekf.bias_accel();
  r.bg_err = r.bg_est - true_bg;
  r.ba_err = r.ba_est - true_ba;
  // Origin-chart covariance: canonical [att,vel,pos,b_w,b_a,b_v] block order.
  const TGEqF::Covariance18 P = ekf.errorCovariance();
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

  // Navigation errors stay small under aiding despite the uncompensated-at-
  // start biases. Measured (Release, x86-64): pos 2.8e-3 m, vel 6.2e-3 m/s,
  // att 4.7e-2 rad. Position/velocity are pinned by the 5 cm position fixes;
  // attitude carries a larger residual (~2.7 deg) than the TFG-IEKF because the
  // horizontal accel-bias axes stay entangled with the tilt (g * tilt
  // feedback).
  EXPECT(r.pos_err.norm() < 0.05);
  EXPECT(r.vel_err.norm() < 0.05);
  EXPECT(r.att_err.norm() < 0.06);  // ~3.4 deg

  // Gyro bias is recovered well (truth |bg| = 0.023; measured error 1.7e-3).
  // The accel bias splits by observability: the z component (paired with
  // gravity) recovers to ~2e-3, while the horizontal components remain
  // entangled with the few-mrad tilt -- the filter's own sigma_ba ~ 0.057
  // covers this, i.e. it is consistent, not divergent. Truth |ba| = 0.062;
  // measured total error 4.1e-2.
  EXPECT(r.bg_err.norm() < 0.005);
  EXPECT(r.ba_err.norm() < 0.045);         // > 25% recovery overall
  EXPECT(std::abs(r.ba_err.z()) < 0.005);  // observable axis: tight
  const double sigma_ba = std::sqrt(r.ba_cov_trace / 3.0);
  EXPECT(r.ba_err.norm() < 3.0 * sigma_ba);  // filter consistency

  // Covariance stays SPD; the gyro-bias block converges well below its prior.
  EXPECT(r.min_eig > 0.0);
  EXPECT(r.bg_cov_trace < 0.1 * 3e-3);
}

TEST(TGEqF, Regression30sAidedVsImuOnly) {
  const RegressionResult aided =
      runBiasedImu30s(/*with_position_updates=*/true);
  const RegressionResult imu_only =
      runBiasedImu30s(/*with_position_updates=*/false);
  printRegression("aided", aided);
  printRegression("imu_only", imu_only);

  // Dead reckoning with a biased IMU drifts far; aiding contains the error to
  // sub-cm (measured: imu_only ~152 m, aided 2.8e-3 m after 30 s).
  EXPECT(imu_only.pos_err.norm() > 10.0);
  EXPECT(aided.pos_err.norm() < 0.01 * imu_only.pos_err.norm());

  // Without aiding the biases are unobservable: the estimate never moves from
  // its zero initialisation (propagate keeps the physical bias constant).
  EXPECT(imu_only.bg_est.norm() < 1e-6);
  EXPECT(imu_only.ba_est.norm() < 1e-6);

  // The filter knows it: position covariance grows instead of shrinking.
  EXPECT(imu_only.pos_cov_trace > 10.0 * aided.pos_cov_trace);
  EXPECT(imu_only.min_eig > 0.0);
}

/* ************************************************************************* */
int main() {
  TestResult tr;
  return TestRegistry::runAllTests(tr);
}
