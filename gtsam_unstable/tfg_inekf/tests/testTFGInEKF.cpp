/**
 * @file  testTFGInEKF.cpp
 * @brief Integration tests for the TFG-IEKF wrapper (predict + position, DVL,
 * depth and magnetometer updates). Magnetometer bias at indices 15..17.
 */
#include <CppUnitLite/TestHarness.h>
#include <gtsam/base/TestableAssertions.h>
#include <gtsam/navigation/Scenario.h>
#include <gtsam_unstable/tfg_inekf/InEKF.h>
#include <gtsam_unstable/tfg_inekf/MagnetometerOutput.h>

using namespace tfg;
using namespace gtsam;

static constexpr double kTol = 1e-9;

// The filter is frame-agnostic -- gravity is a propagate() argument -- so these
// tests pick a convention of their own: Z-up (ENU), gravity along -z.
static const Eigen::Vector3d kTestGravity(0.0, 0.0, -9.81);

using Cov18 = TfgInEKF::Covariance;
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

TEST(TfgInEKF, ConstructorStoresStateAndCovariance) {
  auto X0 = TwoFrameGroup::FromState(
      Rot3::Rz(0.2), Eigen::Vector3d(0.1, 0, 0), Eigen::Vector3d(1, 2, 3),
      Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero());
  Cov18 P0 = Cov18::Identity() * 0.5;
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
  Cov18 P0 = Cov18::Identity() * 0.1;
  TfgInEKF ekf(X0, P0);

  Cov18 Qc = Cov18::Identity() * 1e-3;
  ekf.propagate(Eigen::Vector3d(0.05, -0.02, 0.1),
                Eigen::Vector3d(0.0, 0.0, 9.81), kTestGravity, Qc, 0.01);

  const Cov18 P = ekf.covariance();
  EXPECT(assert_equal((Matrix)P, (Matrix)P.transpose(), 1e-12));  // symmetric
  EXPECT(minEig(P) > 0.0);                                        // PSD
}

// Mean prediction matches the world-frame dynamics (Eq. 3) over one step.
TEST(TfgInEKF, PredictMeanFollowsDynamics) {
  Rot3 R0 = Rot3::Rz(0.3);
  Eigen::Vector3d v0(0.2, -0.1, 0.0), p0(1.0, 0.0, -1.0);
  auto X0 = TwoFrameGroup::FromState(R0, v0, p0, Eigen::Vector3d::Zero(),
                                     Eigen::Vector3d::Zero());
  TfgInEKF ekf(X0, Cov18::Identity() * 0.01);

  const Eigen::Vector3d omega(0.0, 0.0, 0.1), accel(0.0, 0.0, 0.0);
  const double dt = 1e-4;
  ekf.propagate(omega, accel, kTestGravity, Cov18::Identity() * 1e-6, dt);

  const Eigen::Vector3d g = kTestGravity;
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
  Cov18 P0 = Cov18::Identity() * 0.1;
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
  Cov18 P0 = Cov18::Identity() * 1e-4;
  P0.block<3, 3>(6, 6) = Cov3::Identity() * 1.0;  // trust measurement more
  TfgInEKF ekf(X0, P0);

  const Eigen::Vector3d truth(0.0, 0.0, 0.0);
  ekf.update_position(truth, Cov3::Identity() * 1e-4);

  // Estimate should move substantially toward truth.
  EXPECT(ekf.position().norm() < 0.1);
}

// ---------------------------------------------------------------------------
// DVL update (body-frame velocity)
// ---------------------------------------------------------------------------

// A DVL measurement reduces the velocity-block covariance.
TEST(TfgInEKF, UpdateDvlReducesVelocityCovariance) {
  auto X0 = TwoFrameGroup::Identity();
  Cov18 P0 = Cov18::Identity() * 0.1;
  P0.block<3, 3>(3, 3) = Cov3::Identity() * 4.0;  // large velocity uncertainty
  TfgInEKF ekf(X0, P0);

  const double trace_before = ekf.covariance().block<3, 3>(3, 3).trace();
  ekf.update_dvl(Eigen::Vector3d(0.3, -0.1, 0.0), Cov3::Identity() * 0.01);
  const double trace_after = ekf.covariance().block<3, 3>(3, 3).trace();

  EXPECT(trace_after < trace_before);
  EXPECT(minEig(ekf.covariance()) > 0.0);
}

// A single accurate DVL update pulls a wrong velocity estimate toward truth.
// The DVL reports the body-frame velocity consistent with truth_v: R^T truth_v.
TEST(TfgInEKF, UpdateDvlCorrectsVelocity) {
  const Rot3 R0 = Rot3::Rz(0.5);
  const Eigen::Vector3d truth_v = R0 * Eigen::Vector3d(0.2, -0.1, 0.05);
  auto X0 = TwoFrameGroup::FromState(
      R0, Eigen::Vector3d(1.0, 1.0, -1.0),  // wrong world velocity
      Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero(),
      Eigen::Vector3d::Zero());
  Cov18 P0 = Cov18::Identity() * 1e-4;
  P0.block<3, 3>(3, 3) = Cov3::Identity() * 1.0;  // trust the DVL more
  TfgInEKF ekf(X0, P0);

  const Eigen::Vector3d z_dvl = R0.unrotate(truth_v);
  ekf.update_dvl(z_dvl, Cov3::Identity() * 1e-4);

  const double err_before = (Eigen::Vector3d(1.0, 1.0, -1.0) - truth_v).norm();
  const double err_after = (ekf.velocity() - truth_v).norm();
  EXPECT(err_after < 0.05 * err_before);
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
// residual rather than exactly zero. We therefore assert strong error
// reduction, clean velocity convergence, and covariance shrink.
TEST(TfgInEKF, StaticConvergence) {
  const Eigen::Vector3d g(0.0, 0.0, -9.81);
  const Eigen::Vector3d accel_static = -g;  // specific force when static
  const Eigen::Vector3d omega(0.0, 0.0, 0.0);
  const Eigen::Vector3d truth_pos(0.0, 0.0, 0.0);

  const Eigen::Vector3d p_init(0.5, -0.4, 0.3);
  auto X0 = TwoFrameGroup::FromState(
      Rot3::Identity(), Eigen::Vector3d(0.1, -0.05, 0.0), p_init,
      Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero());

  Cov18 P0 = Cov18::Identity() * 1e-4;             // confident attitude + bias
  P0.block<3, 3>(3, 3) = Cov3::Identity() * 0.25;  // velocity uncertainty
  P0.block<3, 3>(6, 6) = Cov3::Identity() * 1.0;   // position uncertainty
  TfgInEKF ekf(X0, P0);

  Cov18 Qc = Cov18::Identity() * 1e-6;
  Cov3 R_pos = Cov3::Identity() * (0.05 * 0.05);
  const double dt = 0.01;  // 100 Hz IMU

  const double pos_trace_before = ekf.covariance().block<3, 3>(6, 6).trace();

  for (int k = 0; k < 500; ++k) {
    ekf.propagate(omega, accel_static, kTestGravity, Qc, dt);
    if (k % 10 == 0) ekf.update_position(truth_pos, R_pos);  // 10 Hz GNSS
  }

  // Strong error reduction (> 60%) and clean velocity convergence.
  const double err_init = (p_init - truth_pos).norm();
  const double err_final = (ekf.position() - truth_pos).norm();
  EXPECT(err_final < 0.4 * err_init);
  EXPECT(veq(ekf.velocity(), Eigen::Vector3d::Zero(), 0.05));

  // Covariance stays SPD and the position block shrinks well below the prior.
  EXPECT(minEig(ekf.covariance()) > 0.0);
  const double pos_trace_after = ekf.covariance().block<3, 3>(6, 6).trace();
  EXPECT(pos_trace_after < 0.1 * pos_trace_before);
}

// ---------------------------------------------------------------------------
// Predict builds nav<->bias cross-covariance: the predict step must couple
// nav and bias blocks, and a position update must therefore move the bias
// estimate. Before the fix the adjoint-only propagation left these blocks
// identically zero and biases were unobservable.
// ---------------------------------------------------------------------------

TEST(TfgInEKF, PredictBuildsNavBiasCrossCovariance) {
  auto X0 = TwoFrameGroup::FromState(
      Rot3::Rz(0.2), Eigen::Vector3d(0.3, -0.1, 0.0),
      Eigen::Vector3d(1.0, 0.5, -0.2), Eigen::Vector3d(0.01, -0.02, 0.0),
      Eigen::Vector3d(0.0, 0.03, -0.01));
  TfgInEKF ekf(X0, Cov18::Identity() * 0.1);  // diagonal: zero cross-cov

  ekf.propagate(Eigen::Vector3d(0.05, -0.02, 0.1),
                Eigen::Vector3d(0.0, 0.0, 9.81), kTestGravity,
                Cov18::Identity() * 1e-6, 0.01);

  // Position(6..8) vs gyro-bias(9..11) and velocity(3..5) vs accel-bias(12..14)
  // cross-blocks are now populated by the lift Jacobian coupling.
  const Cov18 P = ekf.covariance();
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
  Cov18 P0 = Cov18::Identity() * 0.1;
  P0.block<3, 3>(6, 6) = Cov3::Identity() * 2.0;  // loose position
  TfgInEKF ekf(X0, P0);

  // One propagate to couple bias into the nav covariance, then a position fix.
  ekf.propagate(Eigen::Vector3d(0.05, -0.02, 0.1),
                Eigen::Vector3d(0.0, 0.0, 9.81), kTestGravity,
                Cov18::Identity() * 1e-5, 0.01);
  const Eigen::Vector3d bg_before = ekf.bias_gyro();
  const Eigen::Vector3d ba_before = ekf.bias_accel();

  ekf.update_position(Eigen::Vector3d(0.0, 0.0, 0.0), Cov3::Identity() * 1e-3);

  // Gain rows for the bias are no longer identically zero, so the estimate
  // moves.
  EXPECT((ekf.bias_gyro() - bg_before).norm() +
             (ekf.bias_accel() - ba_before).norm() >
         1e-6);
}

// ---------------------------------------------------------------------------
// Translation invariance: the equivariant residual and its Jacobian depend on
// pi - p only, so the update must be invariant to the choice of world origin.
// Running an identical static scenario shifted by a large offset must give
// the same position-error trajectory.
// ---------------------------------------------------------------------------

static double runStaticErrShifted(const Eigen::Vector3d& offset) {
  const Eigen::Vector3d g(0.0, 0.0, -9.81);
  const Eigen::Vector3d accel_static = -g;
  const Eigen::Vector3d omega = Eigen::Vector3d::Zero();
  const Eigen::Vector3d truth_pos = offset;

  auto X0 = TwoFrameGroup::FromState(
      Rot3::Identity(), Eigen::Vector3d::Zero(),
      Eigen::Vector3d(1.0, -0.8, 0.0) + offset,  // wrong position, shifted
      Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero());
  Cov18 P0 = Cov18::Identity() * 0.05;
  P0.block<3, 3>(6, 6) = Cov3::Identity() * 2.0;
  TfgInEKF ekf(X0, P0);

  Cov18 Qc = Cov18::Identity() * 1e-6;
  Cov3 R_pos = Cov3::Identity() * (0.05 * 0.05);
  const double dt = 0.01;
  for (int k = 0; k < 500; ++k) {
    ekf.propagate(omega, accel_static, kTestGravity, Qc, dt);
    if (k % 10 == 0) ekf.update_position(truth_pos, R_pos);
  }
  return (ekf.position() - truth_pos).norm();
}

TEST(TfgInEKF, TranslationInvariance) {
  const double err0 = runStaticErrShifted(Eigen::Vector3d::Zero());
  const double errS =
      runStaticErrShifted(Eigen::Vector3d(500.0, -300.0, 100.0));
  EXPECT_DOUBLES_EQUAL(err0, errS, 1e-6);
}

// ---------------------------------------------------------------------------
// Input noise couples into bias rows: the lifted process noise must couple
// gyro white noise into the bias rows (through b_w^/b_a^/b_m^ in the input
// matrix B).
// ---------------------------------------------------------------------------

TEST(TfgInEKF, InputNoiseCouplesIntoBiasRows) {
  auto X = TwoFrameGroup::FromState(
      Rot3::Rz(0.3), Eigen::Vector3d(0.2, 0.0, -0.1),
      Eigen::Vector3d(0.5, -0.5, 0.1), Eigen::Vector3d(0.02, -0.01, 0.0),
      Eigen::Vector3d(0.0, 0.03, -0.02), Eigen::Vector3d(0.08, -0.05, 0.03));
  tfg::ImuNoise noise;
  noise.gyro = Eigen::Vector3d::Constant(1e-2);  // gyro noise only
  const Cov18 Qd = tfg::inputNoiseCov(X, noise, 0.01);

  const double gw = Qd.block<3, 3>(9, 9).norm();     // gyro-bias rows driven
  const double ga = Qd.block<3, 3>(12, 12).norm();   // accel-bias rows driven
  const double gm = Qd.block<3, 3>(15, 15).norm();   // mag-bias rows driven
  const double th_gw = Qd.block<3, 3>(0, 9).norm();  // theta<->gyro-bias corr
  EXPECT(gw > 1e-9);
  EXPECT(ga > 1e-9);
  EXPECT(gm > 1e-9);
  EXPECT(th_gw > 1e-9);
  EXPECT(assert_equal((Matrix)Qd, (Matrix)Qd.transpose(), 1e-15));  // symmetric
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
  const Eigen::Vector3d g = kTestGravity;

  // Init at the true initial state with zero bias estimate.
  const gtsam::NavState gt0 = scenario.navState(0.0);
  auto X0 = TwoFrameGroup::FromState(gt0.attitude(), gt0.velocity(),
                                     gt0.position(), Eigen::Vector3d::Zero(),
                                     Eigen::Vector3d::Zero());
  Cov18 P0 = Cov18::Zero();
  P0.block<3, 3>(0, 0) = Cov3::Identity() * 1e-4;    // attitude (confident)
  P0.block<3, 3>(3, 3) = Cov3::Identity() * 1e-2;    // velocity
  P0.block<3, 3>(6, 6) = Cov3::Identity() * 1e-2;    // position
  P0.block<3, 3>(9, 9) = Cov3::Identity() * 1e-3;    // gyro bias
  P0.block<3, 3>(12, 12) = Cov3::Identity() * 1e-2;  // accel bias
  P0.block<3, 3>(15, 15) = Cov3::Identity() * 1e-2;  // mag bias, never updated
  TfgInEKF ekf(X0, P0);

  tfg::ImuNoise noise;
  noise.gyro = Eigen::Vector3d::Constant(1e-6);   // (0.001 rad/s/sqrt(Hz))^2
  noise.accel = Eigen::Vector3d::Constant(1e-4);  // (0.01 m/s^2/sqrt(Hz))^2
  noise.gyro_rw = Eigen::Vector3d::Constant(1e-9);
  noise.accel_rw = Eigen::Vector3d::Constant(1e-8);
  const Cov3 R_pos = Cov3::Identity() * (0.05 * 0.05);

  double t = 0.0;
  for (int k = 0; k < num_steps; ++k) {
    // Ideal IMU at time t (specific force f = R^T (a_n - g)) plus the biases.
    const gtsam::NavState gt = scenario.navState(t);
    const Eigen::Vector3d omega_meas = scenario.omega_b(t) + true_bg;
    const Eigen::Vector3d accel_meas =
        gt.attitude().unrotate(scenario.acceleration_n(t) - g) + true_ba;
    ekf.propagate(omega_meas, accel_meas, kTestGravity, noise, dt);
    t += dt;
    if (with_position_updates && (k + 1) % gnss_decim == 0)
      ekf.update_position(scenario.navState(t).position(), R_pos);
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
  const Cov18 P = ekf.covariance();
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

TEST(TfgInEKF, Regression30sBiasedImuWithPositionUpdates) {
  const RegressionResult r = runBiasedImu30s(/*with_position_updates=*/true);
  printRegression("aided", r);

  // Navigation errors stay small under aiding despite the uncompensated-at-
  // start biases (5 cm GNSS-grade bound on position). Measured (Release,
  // x86-64): pos 3.1e-4 m, vel 6.3e-4 m/s, att 4.3e-3 rad.
  EXPECT(r.pos_err.norm() < 0.05);
  EXPECT(r.vel_err.norm() < 0.05);
  EXPECT(r.att_err.norm() < 0.02);  // ~1.1 deg

  // Gyro bias is recovered almost fully (truth |bg| = 0.023; measured error
  // 8.8e-4). The accel bias splits by observability: the z component (paired
  // with gravity) recovers to ~1e-3, while the horizontal components remain
  // entangled with the few-mrad tilt (g * tilt ~ 0.02 m/s^2) -- the filter's
  // own sigma_ba ~ 0.06 covers this, i.e. it is consistent, not divergent.
  // Truth |ba| = 0.062; measured total error 0.030.
  EXPECT(r.bg_err.norm() < 0.005);
  EXPECT(r.ba_err.norm() < 0.045);         // > 25% recovery overall
  EXPECT(std::abs(r.ba_err.z()) < 0.005);  // observable axis: tight
  const double sigma_ba = std::sqrt(r.ba_cov_trace / 3.0);
  EXPECT(r.ba_err.norm() < 3.0 * sigma_ba);  // filter consistency

  // Covariance stays SPD; the gyro-bias block converges well below its prior.
  EXPECT(r.min_eig > 0.0);
  EXPECT(r.bg_cov_trace < 0.1 * 3e-3);
}

TEST(TfgInEKF, Regression30sAidedVsImuOnly) {
  const RegressionResult aided =
      runBiasedImu30s(/*with_position_updates=*/true);
  const RegressionResult imu_only =
      runBiasedImu30s(/*with_position_updates=*/false);
  printRegression("aided", aided);
  printRegression("imu_only", imu_only);

  // Dead reckoning with a biased IMU drifts far (measured: 152 m after 30 s);
  // aiding contains the error to sub-cm.
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
namespace depth_update {

using Cov1 = TfgInEKF::Covariance1;

// 1x1 depth noise matrix from a stddev in metres.
Cov1 depthNoise(double sigma) {
  Cov1 R;
  R(0, 0) = sigma * sigma;
  return R;
}

// Tilted state: R_meas = R^T R_pseudo R is only diagonal when R is, so a roll
// keeps that transport honest.
TwoFrameGroup tiltedState() {
  return TwoFrameGroup::FromState(
      Rot3::Rx(M_PI / 6), Eigen::Vector3d(0.5, -0.2, 0.1),
      Eigen::Vector3d(1.0, 2.0, -3.0), Eigen::Vector3d::Zero(),
      Eigen::Vector3d::Zero());
}

// A single depth update pulls the vertical estimate most of the way to the
// measurement when the depth noise is much tighter than the prior.
TEST(TfgInEKF, DepthUpdateCorrectsVerticalPosition) {
  TfgInEKF ekf(TwoFrameGroup::Identity(), Cov18::Identity() * 0.01);

  const double z_depth = -0.8;
  ekf.update_depth(z_depth, depthNoise(0.01));

  EXPECT(std::abs(ekf.position().z() - z_depth) < 0.1);
}

// The inflated horizontal variance keeps a depth update off the x and y axes
// when the prior is uncorrelated.
TEST(TfgInEKF, DepthUpdateLeavesHorizontalNearlyUnchanged) {
  TfgInEKF ekf(TwoFrameGroup::Identity(), Cov18::Identity() * 0.01);

  const Eigen::Vector3d before = ekf.position();
  ekf.update_depth(-0.8, depthNoise(0.01));
  const Eigen::Vector3d delta = ekf.position() - before;

  EXPECT(std::abs(delta.z()) > 0.5);
  EXPECT(std::abs(delta.x()) < 0.02);
  EXPECT(std::abs(delta.y()) < 0.02);
}

// Only the vertical position covariance is informed; the horizontal blocks stay
// essentially where they were.
TEST(TfgInEKF, DepthUpdateShrinksVerticalCovariance) {
  TfgInEKF ekf(TwoFrameGroup::Identity(), Cov18::Identity() * 0.01);

  const Cov18 before = ekf.covariance();
  ekf.update_depth(-0.8, depthNoise(0.01));
  const Cov18 after = ekf.covariance();

  const double drop_z = before(8, 8) - after(8, 8);
  EXPECT(drop_z > 0.0);
  EXPECT(std::abs(before(6, 6) - after(6, 6)) < 0.1 * drop_z);
  EXPECT(std::abs(before(7, 7) - after(7, 7)) < 0.1 * drop_z);
  EXPECT(minEig(after) > 0.0);
}

// update_depth is exactly the pseudo-position update it documents. Checked at a
// tilted state, where the R^T(.)R noise transport is not a no-op.
TEST(TfgInEKF, DepthUpdateMatchesEquivalentPositionUpdate) {
  TfgInEKF via_depth(tiltedState(), Cov18::Identity() * 0.01);
  TfgInEKF via_position(tiltedState(), Cov18::Identity() * 0.01);

  const double z_depth = -2.0;
  const double sigma_z = 0.02;
  via_depth.update_depth(z_depth, depthNoise(sigma_z));

  Cov3 R_pseudo = Cov3::Zero();
  R_pseudo(0, 0) = TfgInEKF::kDefaultHorizontalVariance;
  R_pseudo(1, 1) = TfgInEKF::kDefaultHorizontalVariance;
  R_pseudo(2, 2) = sigma_z * sigma_z;
  Eigen::Vector3d pseudo = via_position.position();
  pseudo.z() = z_depth;
  via_position.update_position(pseudo, R_pseudo);

  EXPECT(traits<TwoFrameGroup>::Equals(via_position.state(), via_depth.state(),
                                       1e-12));
  EXPECT(assert_equal((Matrix)via_position.covariance(),
                      (Matrix)via_depth.covariance(), 1e-12));
}

// At a tilted state with a metre-sized vertical offset the update still moves
// the estimate toward the measured depth; one update against a decimetre prior
// cannot close the gap, so only a strict decrease is asserted.
TEST(TfgInEKF, DepthUpdateAtTiltedStateReducesVerticalError) {
  TfgInEKF ekf(tiltedState(), Cov18::Identity() * 0.01);

  const double z_depth = ekf.position().z() + 1.0;
  const double err_before = std::abs(ekf.position().z() - z_depth);
  ekf.update_depth(z_depth, depthNoise(0.01));
  const double err_after = std::abs(ekf.position().z() - z_depth);

  EXPECT(err_after < 0.95 * err_before);
}

// The horizontal-variance argument is live: shrinking it changes the correction
// a correlated prior produces, so the default is not silently ignored.
TEST(TfgInEKF, DepthUpdateHorizontalVarianceIsRespected) {
  // Correlate p_x with p_z; the block [[0.01, 0.008], [0.008, 0.01]] is SPD.
  Cov18 P0 = Cov18::Identity() * 0.01;
  P0(6, 8) = P0(8, 6) = 0.008;

  TfgInEKF inflated(TwoFrameGroup::Identity(), P0);
  TfgInEKF pinned(TwoFrameGroup::Identity(), P0);

  const double z_depth = -0.8;
  inflated.update_depth(z_depth, depthNoise(0.01));
  pinned.update_depth(z_depth, depthNoise(0.01), /*horizontal_variance=*/1e-6);

  EXPECT((inflated.position() - pinned.position()).norm() > 1e-3);
}

}  // namespace depth_update
/* ************************************************************************* */

/* ************************************************************************* */
namespace mag_update {

/// Reference field in the global frame: horizontal north and east parts and a
/// steeper vertical part, as at mid latitudes (normalized units).
const Eigen::Vector3d kField(0.4, 0.05, 0.9);

/// Prior with the attitude and magnetometer-bias variances set separately.
Cov18 magSigma(double attitude_variance, double mag_bias_variance) {
  Cov18 P0 = Cov18::Identity() * 0.01;
  P0.block<3, 3>(0, 0) = attitude_variance * Cov3::Identity();
  P0.block<3, 3>(15, 15) = mag_bias_variance * Cov3::Identity();
  return P0;
}

/// Rotated, biased estimate with a correlated prior, for the hand-built checks.
TwoFrameGroup rotatedBiasedState() {
  return TwoFrameGroup::FromState(
      Rot3::Rz(M_PI / 2) * Rot3::Rx(0.2), Eigen::Vector3d(1.0, -0.5, 0.3),
      Eigen::Vector3d(2.0, 1.0, -1.0), Eigen::Vector3d(0.01, -0.02, 0.03),
      Eigen::Vector3d(-0.05, 0.1, 0.0), Eigen::Vector3d(0.05, -0.08, 0.03));
}
Cov18 correlatedSigma() {
  Cov18 P0 = Cov18::Identity() * 0.01;
  P0(0, 15) = P0(15, 0) = 0.004;  // attitude-bias correlation
  P0(2, 4) = P0(4, 2) = 0.003;    // attitude-velocity correlation
  return P0;
}

// initialCovariance places the navigation block first and the magnetometer
// bias last, uncorrelated.
TEST(TfgInEKF, InitialCovarianceIsBlockDiagonal) {
  const TfgInEKF::Covariance15 Sigma_nav =
      TfgInEKF::Covariance15::Identity() * 0.3;
  const Cov3 Sigma_mag = Cov3::Identity() * 0.7;
  const Cov18 P0 = TfgInEKF::initialCovariance(Sigma_nav, Sigma_mag);

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
TEST(TfgInEKF, MagUpdateCorrectsHeadingWithKnownBias) {
  Cov18 P0 = magSigma(1e-12, 1e-12);
  P0(2, 2) = 0.01;
  TfgInEKF ekf(TwoFrameGroup::Identity(), P0);
  const Rot3 R_true = Rot3::Rz(0.1);

  ekf.update_magnetometer(R_true.unrotate(kField), kField,
                          1e-8 * Cov3::Identity());

  EXPECT(std::abs(ekf.attitude().yaw() - 0.1) < 1e-3);
  EXPECT(veq(Eigen::Vector3d::Zero(), ekf.bias_mag(), 1e-6));
}

// With the attitude known, the same update estimates the hard-iron bias as
// the difference between the reading and the predicted field.
TEST(TfgInEKF, MagUpdateCorrectsBiasAtKnownAttitude) {
  TfgInEKF ekf(TwoFrameGroup::Identity(), magSigma(1e-12, 0.01));
  const Eigen::Vector3d bm_true(0.05, -0.03, 0.02);

  ekf.update_magnetometer(kField + bm_true, kField, 1e-8 * Cov3::Identity());

  EXPECT(veq(bm_true, ekf.bias_mag(), 1e-5));
  EXPECT(ekf.attitude().equals(Rot3::Identity(), 1e-6));
}

// The update is the invariant EKF step in the right-retract chart: body-frame
// innovation z - R^T (m - gamma_m), the output matrix C0, R_mag used
// directly, correction X_hat Exp(K nu) and the Joseph-form covariance. This
// rebuilds it by hand, with an anisotropic R_mag.
TEST(TfgInEKF, MagUpdateMatchesTheHandBuiltUpdate) {
  const TwoFrameGroup X0 = rotatedBiasedState();
  const Cov18 P0 = correlatedSigma();
  TfgInEKF ekf(X0, P0);
  const Eigen::Vector3d z_mag = Rot3::Rz(M_PI / 2 + 0.05).unrotate(kField) +
                                Eigen::Vector3d(0.01, -0.02, 0.0);
  const Cov3 R_mag = Eigen::Vector3d(1e-4, 4e-4, 9e-4).asDiagonal();

  const Eigen::Vector3d nu = z_mag - MagnetometerOutput::predict(X0, kField);
  const Eigen::Matrix<double, 3, 18> H =
      MagnetometerOutput::jacobian(X0, kField);
  const Eigen::Matrix<double, 18, 3> K =
      P0 * H.transpose() * (H * P0 * H.transpose() + R_mag).inverse();
  const Cov18 I_KH = Cov18::Identity() - K * H;

  ekf.update_magnetometer(z_mag, kField, R_mag);

  EXPECT(traits<TwoFrameGroup>::Equals(X0 * TwoFrameGroup::Expmap(K * nu),
                                       ekf.state(), 1e-9));
  EXPECT(assert_equal(
      (Matrix)(I_KH * P0 * I_KH.transpose() + K * R_mag * K.transpose()),
      (Matrix)ekf.covariance(), 1e-12));
}

// With an uncorrelated prior the magnetometer reaches only attitude and its
// own bias; velocity, position and IMU biases stay where they were.
TEST(TfgInEKF, MagUpdateOnlyCorrectsAttitudeAndMagBias) {
  TfgInEKF ekf(TwoFrameGroup::Identity(), Cov18::Identity() * 0.01);

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
TfgInEKF runWithoutMagUpdates(const Eigen::Vector3d& b_mag,
                              double mag_bias_variance) {
  TfgInEKF::Covariance15 Sigma_nav = TfgInEKF::Covariance15::Identity() * 0.01;
  Sigma_nav(0, 4) = Sigma_nav(4, 0) = 0.004;
  TfgInEKF ekf(
      TwoFrameGroup::FromState(Rot3::Rz(M_PI / 2), Eigen::Vector3d::Zero(),
                               Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero(),
                               Eigen::Vector3d::Zero(), b_mag),
      TfgInEKF::initialCovariance(Sigma_nav,
                                  mag_bias_variance * Cov3::Identity()));

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
  TfgInEKF::Covariance1 R_depth;
  R_depth(0, 0) = 1e-4;
  ekf.update_depth(ekf.position().z() + 0.05, R_depth);
  return ekf;
}

/// Navigation part [R v p gamma_w gamma_a] of the error between two filters.
Eigen::Matrix<double, 15, 1> navigationDifference(const TfgInEKF& a,
                                                  const TfgInEKF& b) {
  return traits<TwoFrameGroup>::Local(a.state(), b.state()).head<15>();
}

// Without magnetometer updates the navigation estimate and covariance depend
// on neither the bias estimate nor its prior: F is block lower-triangular,
// with nothing flowing from gamma_m into the navigation blocks.
TEST(TfgInEKF, NavigationIsUnaffectedByTheMagBlockWithoutMagUpdates) {
  const Eigen::Vector3d b_mag(0.08, -0.05, 0.03);
  const TfgInEKF reference = runWithoutMagUpdates(Eigen::Vector3d::Zero(), 1.0);
  const TfgInEKF tight = runWithoutMagUpdates(b_mag, 1e-4);
  const TfgInEKF loose = runWithoutMagUpdates(b_mag, 1.0);

  for (const TfgInEKF* other : {&tight, &loose}) {
    EXPECT(assert_equal((Vector)Eigen::Matrix<double, 15, 1>::Zero(),
                        (Vector)navigationDifference(reference, *other),
                        1e-12));
    EXPECT(assert_equal((Matrix)reference.covariance().topLeftCorner<15, 15>(),
                        (Matrix)other->covariance().topLeftCorner<15, 15>(),
                        1e-12));
  }
}

// The cross-covariance is a different matter: with a zero bias estimate it
// stays exactly zero, but a nonzero one couples gamma_m to the attitude (Ad and
// the b_m^ lift rows) and the cross-covariance grows, one way only.
TEST(TfgInEKF, MagBiasEstimateCouplesTheCrossCovariance) {
  const TfgInEKF zero_bias = runWithoutMagUpdates(Eigen::Vector3d::Zero(), 1.0);
  const TfgInEKF biased =
      runWithoutMagUpdates(Eigen::Vector3d(0.08, -0.05, 0.03), 1.0);

  EXPECT(assert_equal((Matrix)Eigen::Matrix<double, 15, 3>::Zero(),
                      (Matrix)zero_bias.covariance().topRightCorner<15, 3>(),
                      1e-15));
  EXPECT((biased.covariance().topRightCorner<15, 3>().norm() > 1e-6));
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
/// default 0.2 rad error this is the TG-EqF and MEKF heading scenario, so the
/// three filters can be compared.
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
  const TwoFrameGroup X0 = TwoFrameGroup::FromState(
      Rot3::Rz(initial_yaw_error) * gt0.attitude(), gt0.velocity(),
      gt0.position(), Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero());

  TfgInEKF::Covariance15 Sigma_nav = TfgInEKF::Covariance15::Zero();
  Sigma_nav.block<3, 3>(0, 0) =
      Eigen::Vector3d(1e-4, 1e-4, 0.09).asDiagonal();       // attitude
  Sigma_nav.block<3, 3>(3, 3) = Cov3::Identity() * 1e-1;    // velocity
  Sigma_nav.block<3, 3>(6, 6) = Cov3::Identity() * 1e-2;    // position
  Sigma_nav.block<3, 3>(9, 9) = Cov3::Identity() * 1e-8;    // gyro bias
  Sigma_nav.block<3, 3>(12, 12) = Cov3::Identity() * 1e-6;  // accel bias
  TfgInEKF ekf(X0,
               TfgInEKF::initialCovariance(Sigma_nav, Cov3::Identity() * 1e-2));

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
  // rotation about the global z axis.
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
TEST(TfgInEKF, YawTurnsSeparateHeadingFromHardIron) {
  const HeadingResult r = runMagHeading60s(/*yaw_rate=*/0.3);
  printf("[tfg mag turning] yaw_err=%.4e  yaw_sigma=%.4e  bm_err=%.4e\n",
         r.yaw_err, r.yaw_sigma, r.bm_err.norm());

  EXPECT(r.yaw_err < 0.01);
  EXPECT(r.bm_err.norm() < 0.01);
  EXPECT(r.min_eig > 0.0);
}

// Without yaw turns the heading error and the horizontal bias stay entangled:
// the heading is not corrected, and its variance stays far above that of the
// turning run.
TEST(TfgInEKF, StraightRunLeavesHeadingEntangledWithHardIron) {
  const HeadingResult turning = runMagHeading60s(/*yaw_rate=*/0.3);
  const HeadingResult straight = runMagHeading60s(/*yaw_rate=*/0.0);
  printf("[tfg mag straight] yaw_err=%.4e  yaw_sigma=%.4e  bm_err=%.4e\n",
         straight.yaw_err, straight.yaw_sigma, straight.bm_err.norm());

  EXPECT(straight.yaw_err > 0.03);
  EXPECT(straight.yaw_sigma > 5.0 * turning.yaw_sigma);
}

// Characterizes the TFG-IEKF, which on this scenario behaves like the MEKF and
// unlike the TG-EqF: on a straight run its heading covariance is consistent
// from a small initial error, but from 0.2 rad the first DVL and magnetometer
// updates, linearized at the wrong heading, shrink the heading variance while
// the error stays.
TEST(TfgInEKF, StraightRunHeadingIsOverconfidentOnlyFromALargeError) {
  const HeadingResult small = runMagHeading60s(/*yaw_rate=*/0.0, 0.0);
  const HeadingResult large = runMagHeading60s(/*yaw_rate=*/0.0, 0.2);
  printf("[tfg mag straight] from 0 rad: yaw_err=%.4e  yaw_sigma=%.4e\n",
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
