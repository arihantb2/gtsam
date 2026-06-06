/**
 * @file  TFGInEKFScenarioExample.h
 * @brief Shared helpers for TFG-IEKF IMU-only trajectory examples.
 *
 * IMU-only mode: the filter dead-reckons with propagate() alone (no position
 * updates). Validates the predict step under ideal inputs (should track ground
 * truth exactly) and under controlled IMU bias / noise (should drift in a
 * physically meaningful way, consistent with the propagated covariance).
 *
 * Uses gtsam::Scenario for ground truth and ScenarioRunner for IMU synthesis.
 */
#pragma once

#include <gtsam/navigation/NavState.h>
#include <gtsam/navigation/PreintegrationParams.h>
#include <gtsam/navigation/Scenario.h>
#include <gtsam/navigation/ScenarioRunner.h>
#include <gtsam_unstable/tfg_inekf/InEKF.h>

#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>

namespace tfg::examples {

using Cov15 = TfgInEKF::Covariance;

struct RunOptions {
  double duration = 30.0;
  double dt = 0.01;
  std::string output_path = "tfg_inekf_trajectory.csv";

  // IMU error model. Defaults = ideal IMU (no bias, no noise): the clean
  // dead-reckoning baseline, which must reproduce ground truth exactly.
  Eigen::Vector3d gyro_bias = Eigen::Vector3d::Zero();   // rad/s   (constant)
  Eigen::Vector3d accel_bias = Eigen::Vector3d::Zero();  // m/s^2   (constant)
  double gyro_noise_sigma = 0.0;   // rad/s/sqrt(Hz)
  double accel_noise_sigma = 0.0;  // m/s^2/sqrt(Hz)
  double gyro_bias_rw = 0.0;       // gyro  bias random-walk (rad/s/sqrt(s))
  double accel_bias_rw = 0.0;      // accel bias random-walk (m/s^2/sqrt(s))
  double init_sigma = 0.1;         // initial covariance: P0 = init_sigma^2 I
  unsigned seed = 42;              // RNG seed (vary for Monte Carlo)
  int log_decim = 1;               // log every Nth step

  double pos_rate = 0.0;           // Hz
  double pos_noise_sigma = 0.0;    // m
};

/// Parse "x,y,z" into a Vector3 (throws on malformed input).
inline Eigen::Vector3d parseVector3(const std::string& s) {
  Eigen::Vector3d v;
  std::size_t start = 0;
  for (int k = 0; k < 3; ++k) {
    const std::size_t comma = s.find(',', start);
    const std::string tok =
        s.substr(start, comma == std::string::npos ? comma : comma - start);
    v(k) = std::stod(tok);
    if (k < 2) {
      if (comma == std::string::npos)
        throw std::runtime_error("expected 'x,y,z', got: " + s);
      start = comma + 1;
    }
  }
  return v;
}

inline RunOptions parseRunOptions(int argc, char* argv[],
                                  const char* default_output) {
  RunOptions o;
  o.output_path = default_output;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    auto next = [&] { return argv[++i]; };
    if (a == "--output" && i + 1 < argc) o.output_path = next();
    else if (a == "--duration" && i + 1 < argc) o.duration = std::stod(next());
    else if (a == "--dt" && i + 1 < argc) o.dt = std::stod(next());
    else if (a == "--gyro-bias" && i + 1 < argc) o.gyro_bias = parseVector3(next());
    else if (a == "--accel-bias" && i + 1 < argc) o.accel_bias = parseVector3(next());
    else if (a == "--gyro-noise" && i + 1 < argc) o.gyro_noise_sigma = std::stod(next());
    else if (a == "--accel-noise" && i + 1 < argc) o.accel_noise_sigma = std::stod(next());
    else if (a == "--gyro-bias-rw" && i + 1 < argc) o.gyro_bias_rw = std::stod(next());
    else if (a == "--accel-bias-rw" && i + 1 < argc) o.accel_bias_rw = std::stod(next());
    else if (a == "--init-sigma" && i + 1 < argc) o.init_sigma = std::stod(next());
    else if (a == "--log-decim" && i + 1 < argc) o.log_decim = std::max(1, std::stoi(next()));
    else if (a == "--seed" && i + 1 < argc) o.seed = static_cast<unsigned>(std::stoul(next()));
    else if (a == "--pos-rate" && i + 1 < argc) o.pos_rate = std::stod(next());
    else if (a == "--pos-noise" && i + 1 < argc) o.pos_noise_sigma = std::stod(next());
  }
  return o;
}

/// Continuous process-noise PSD, tangent block order [att, vel, pos, bg, ba].
inline Cov15 defaultQc() {
  Cov15 Qc = Cov15::Zero();
  Qc.block<3, 3>(0, 0) = 1e-4 * Eigen::Matrix3d::Identity();   // attitude
  Qc.block<3, 3>(3, 3) = 1e-3 * Eigen::Matrix3d::Identity();   // velocity
  Qc.block<3, 3>(6, 6) = 1e-6 * Eigen::Matrix3d::Identity();   // position
  Qc.block<3, 3>(9, 9) = 1e-6 * Eigen::Matrix3d::Identity();   // gyro bias
  Qc.block<3, 3>(12, 12) = 1e-5 * Eigen::Matrix3d::Identity(); // accel bias
  return Qc;
}

inline const char* csvHeader() {
  return "t,"
         "gt_px,gt_py,gt_pz,gt_vx,gt_vy,gt_vz,"
         "est_px,est_py,est_pz,est_vx,est_vy,est_vz,"
         "att_err_x,att_err_y,att_err_z,"
         "est_bgx,est_bgy,est_bgz,est_bax,est_bay,est_baz,"
         "true_bgx,true_bgy,true_bgz,true_bax,true_bay,true_baz,"
         // upper triangle of each 3x3 covariance block [att,vel,pos,bg,ba]:
         "P_att_00,P_att_01,P_att_02,P_att_11,P_att_12,P_att_22,"
         "P_vel_00,P_vel_01,P_vel_02,P_vel_11,P_vel_12,P_vel_22,"
         "P_pos_00,P_pos_01,P_pos_02,P_pos_11,P_pos_12,P_pos_22,"
         "P_bg_00,P_bg_01,P_bg_02,P_bg_11,P_bg_12,P_bg_22,"
         "P_ba_00,P_ba_01,P_ba_02,P_ba_11,P_ba_12,P_ba_22\n";
}

inline void writeCsvRow(std::ostream& out, double t, const gtsam::NavState& gt,
                        const TfgInEKF& f, const Eigen::Vector3d& true_bg,
                        const Eigen::Vector3d& true_ba) {
  // Attitude error as a right-tangent vector: Logmap(gt_R^-1 * est_R).
  const Eigen::Vector3d att_err =
      gtsam::Rot3::Logmap(gt.attitude().between(f.attitude()));
  const Cov15& P = f.covariance();

  auto v3 = [&out](const Eigen::Vector3d& x) {
    out << ',' << x.x() << ',' << x.y() << ',' << x.z();
  };
  out << std::fixed << std::setprecision(9) << t;
  v3(Eigen::Vector3d(gt.position()));
  v3(Eigen::Vector3d(gt.velocity()));
  v3(f.position());
  v3(f.velocity());
  v3(att_err);
  v3(f.bias_gyro());
  v3(f.bias_accel());
  v3(true_bg);
  v3(true_ba);
  for (int o = 0; o < 15; o += 3) {  // upper triangle of each 3x3 block
    out << ',' << P(o, o) << ',' << P(o, o + 1) << ',' << P(o, o + 2) << ','
        << P(o + 1, o + 1) << ',' << P(o + 1, o + 2) << ',' << P(o + 2, o + 2);
  }
  out << '\n';
}

struct RunSummary {
  size_t num_samples = 0;
  double final_pos_error = 0.0;
  double final_vel_error = 0.0;
  double rms_pos_error = 0.0;
  double rms_vel_error = 0.0;
};

inline RunSummary runScenario(const gtsam::Scenario& scenario,
                              const RunOptions& opts,
                              const std::string& scenario_name) {
  constexpr double kGravity = 9.81;
  auto params = gtsam::PreintegrationParams::MakeSharedU(kGravity);
  gtsam::ScenarioRunner runner(scenario, params, opts.dt);

  // Reproducible noise: discrete stddev = density / sqrt(dt).
  std::mt19937 rng(opts.seed);
  std::normal_distribution<double> normal(0.0, 1.0);
  const double gyro_sd = opts.gyro_noise_sigma / std::sqrt(opts.dt);
  const double accel_sd = opts.accel_noise_sigma / std::sqrt(opts.dt);
  auto sampleNoise = [&](double sd) {
    return Eigen::Vector3d(sd * normal(rng), sd * normal(rng), sd * normal(rng));
  };

  Eigen::Vector3d gyro_b = opts.gyro_bias;
  Eigen::Vector3d accel_b = opts.accel_bias;
  const double gyro_rw_step = opts.gyro_bias_rw * std::sqrt(opts.dt);
  const double accel_rw_step = opts.accel_bias_rw * std::sqrt(opts.dt);

  // Init at the true initial state: IMU-only propagation cannot observe the
  // initial pose/velocity, so the origin must match ground truth.
  const gtsam::NavState gt0 = scenario.navState(0.0);
  auto X0 = TwoFrameGroup::FromState(gt0.attitude(), gt0.velocity(),
                                     gt0.position(), Eigen::Vector3d::Zero(),
                                     Eigen::Vector3d::Zero());
  const Cov15 P0 = opts.init_sigma * opts.init_sigma * Cov15::Identity();
  TfgInEKF filter(X0, P0);

  // Match Qc to the simulated IMU error model so the covariance is meaningful.
  const Eigen::Matrix3d I3 = Eigen::Matrix3d::Identity();
  Cov15 Qc = defaultQc();
  if (opts.gyro_noise_sigma > 0.0)
    Qc.block<3, 3>(0, 0) = opts.gyro_noise_sigma * opts.gyro_noise_sigma * I3;
  if (opts.accel_noise_sigma > 0.0)
    Qc.block<3, 3>(3, 3) = opts.accel_noise_sigma * opts.accel_noise_sigma * I3;
  if (opts.gyro_bias_rw > 0.0)
    Qc.block<3, 3>(9, 9) = opts.gyro_bias_rw * opts.gyro_bias_rw * I3;
  if (opts.accel_bias_rw > 0.0)
    Qc.block<3, 3>(12, 12) = opts.accel_bias_rw * opts.accel_bias_rw * I3;

  std::ofstream csv(opts.output_path);
  if (!csv) throw std::runtime_error("Cannot open output file: " + opts.output_path);
  csv << csvHeader();

  RunSummary summary;
  double sum_pos_sq = 0.0, sum_vel_sq = 0.0;
  double t = 0.0;
  const size_t num_steps =
      static_cast<size_t>(std::llround(opts.duration / opts.dt));

  double next_pos_t = (opts.pos_rate > 0.0) ? (1.0 / opts.pos_rate) : std::numeric_limits<double>::infinity();

  for (size_t k = 0; k <= num_steps; ++k) {
    const gtsam::NavState gt = scenario.navState(t);

    if (opts.pos_rate > 0.0 && t >= next_pos_t - 1e-9) {
      Eigen::Vector3d pos_meas = gt.position() + sampleNoise(opts.pos_noise_sigma);
      Eigen::Matrix3d R_pos = opts.pos_noise_sigma * opts.pos_noise_sigma * Eigen::Matrix3d::Identity();
      filter.update_position(pos_meas, R_pos);
      next_pos_t += 1.0 / opts.pos_rate;
    }

    const Eigen::Vector3d est_p = filter.position();
    const Eigen::Vector3d est_v = filter.velocity();

    if (k % static_cast<size_t>(opts.log_decim) == 0 || k == num_steps)
      writeCsvRow(csv, t, gt, filter, gyro_b, accel_b);

    const double pos_err = (Eigen::Vector3d(gt.position()) - est_p).norm();
    const double vel_err = (Eigen::Vector3d(gt.velocity()) - est_v).norm();
    sum_pos_sq += pos_err * pos_err;
    sum_vel_sq += vel_err * vel_err;
    summary.final_pos_error = pos_err;
    summary.final_vel_error = vel_err;
    ++summary.num_samples;

    if (k == num_steps) break;

    // Corrupt the ideal IMU with the (unknown to the filter) bias + noise.
    const gtsam::Vector3 omega =
        runner.actualAngularVelocity(t) + gyro_b + sampleNoise(gyro_sd);
    const gtsam::Vector3 accel =
        runner.actualSpecificForce(t) + accel_b + sampleNoise(accel_sd);
    filter.propagate(omega, accel, Qc, opts.dt);

    // Random-walk the true bias (no-op when rate is zero).
    if (gyro_rw_step > 0.0)
      gyro_b += gyro_rw_step * Eigen::Vector3d(normal(rng), normal(rng), normal(rng));
    if (accel_rw_step > 0.0)
      accel_b += accel_rw_step * Eigen::Vector3d(normal(rng), normal(rng), normal(rng));
    t += opts.dt;
  }

  if (summary.num_samples > 0) {
    summary.rms_pos_error = std::sqrt(sum_pos_sq / summary.num_samples);
    summary.rms_vel_error = std::sqrt(sum_vel_sq / summary.num_samples);
  }

  std::cout << scenario_name << " scenario complete.\n"
            << "  output: " << opts.output_path << '\n'
            << "  samples: " << summary.num_samples << '\n'
            << std::fixed << std::setprecision(6)
            << "  final position error (m): " << summary.final_pos_error << '\n'
            << "  final velocity error (m/s): " << summary.final_vel_error << '\n'
            << "  RMS position error (m): " << summary.rms_pos_error << '\n'
            << "  RMS velocity error (m/s): " << summary.rms_vel_error << '\n';
  return summary;
}

}  // namespace tfg::examples
