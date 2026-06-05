/**
 * @file TGEqFScenarioExample.h
 * @brief Shared helpers for TG-EqF trajectory example executables.
 *
 * Uses gtsam::Scenario for ground truth and ScenarioRunner for IMU synthesis.
 */

#pragma once

#include <gtsam/navigation/NavState.h>
#include <gtsam/navigation/PreintegrationParams.h>
#include <gtsam/navigation/Scenario.h>
#include <gtsam/navigation/ScenarioRunner.h>
#include <gtsam_unstable/tg_eqf/EqF.h>

#include <cmath>
#include <fstream>
#include <stdexcept>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>

namespace tgeqf::examples {

struct RunOptions {
  double duration = 30.0;
  double dt = 0.01;
  std::string output_path = "tg_eqf_trajectory.csv";

  // IMU error model. Defaults describe an ideal IMU (no bias, no noise) so the
  // clean dead-reckoning baseline is reproduced exactly; see scenario_analysis.md.
  Eigen::Vector3d gyro_bias = Eigen::Vector3d::Zero();   // rad/s (initial)
  Eigen::Vector3d accel_bias = Eigen::Vector3d::Zero();  // m/s^2 (initial)
  double gyro_noise_sigma = 0.0;   // rad/s/sqrt(Hz)
  double accel_noise_sigma = 0.0;  // m/s^2/sqrt(Hz)
  double gyro_bias_rw = 0.0;       // gyro bias random-walk rate (rad/s/sqrt(s))
  double accel_bias_rw = 0.0;      // accel bias random-walk rate (m/s^2/sqrt(s))
  unsigned seed = 42;              // RNG seed for noise (vary it for Monte Carlo)
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
      if (comma == std::string::npos) {
        throw std::runtime_error("expected 'x,y,z', got: " + s);
      }
      start = comma + 1;
    }
  }
  return v;
}

inline RunOptions parseRunOptions(int argc, char* argv[],
                                  const char* default_output) {
  RunOptions opts;
  opts.output_path = default_output;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--output" && i + 1 < argc) {
      opts.output_path = argv[++i];
    } else if (arg == "--duration" && i + 1 < argc) {
      opts.duration = std::stod(argv[++i]);
    } else if (arg == "--dt" && i + 1 < argc) {
      opts.dt = std::stod(argv[++i]);
    } else if (arg == "--gyro-bias" && i + 1 < argc) {
      opts.gyro_bias = parseVector3(argv[++i]);
    } else if (arg == "--accel-bias" && i + 1 < argc) {
      opts.accel_bias = parseVector3(argv[++i]);
    } else if (arg == "--gyro-noise" && i + 1 < argc) {
      opts.gyro_noise_sigma = std::stod(argv[++i]);
    } else if (arg == "--accel-noise" && i + 1 < argc) {
      opts.accel_noise_sigma = std::stod(argv[++i]);
    } else if (arg == "--gyro-bias-rw" && i + 1 < argc) {
      opts.gyro_bias_rw = std::stod(argv[++i]);
    } else if (arg == "--accel-bias-rw" && i + 1 < argc) {
      opts.accel_bias_rw = std::stod(argv[++i]);
    } else if (arg == "--seed" && i + 1 < argc) {
      opts.seed = static_cast<unsigned>(std::stoul(argv[++i]));
    }
  }
  return opts;
}

inline TGEqF::Covariance18 defaultSigma() {
  return 0.01 * TGEqF::Covariance18::Identity();
}

inline TGEqF::Covariance18 defaultQc() {
  TGEqF::Covariance18 Qc = TGEqF::Covariance18::Zero();
  Qc.block<3, 3>(0, 0) = 1e-4 * Eigen::Matrix3d::Identity();
  Qc.block<3, 3>(6, 6) = 1e-3 * Eigen::Matrix3d::Identity();
  Qc.block<3, 3>(9, 9) = 1e-6 * Eigen::Matrix3d::Identity();
  Qc.block<3, 3>(15, 15) = 1e-5 * Eigen::Matrix3d::Identity();
  return Qc;
}

inline void writeCsvRow(std::ostream& out, double t,
                        const gtsam::NavState& gt,
                        const Eigen::Vector3d& est_p,
                        const Eigen::Vector3d& est_v) {
  const gtsam::Point3 gt_p = gt.position();
  const gtsam::Vector3 gt_v = gt.velocity();
  out << std::fixed << std::setprecision(6) << t << ',' << gt_p.x() << ','
      << gt_p.y() << ',' << gt_p.z() << ',' << gt_v.x() << ',' << gt_v.y()
      << ',' << gt_v.z() << ',' << est_p.x() << ',' << est_p.y() << ','
      << est_p.z() << ',' << est_v.x() << ',' << est_v.y() << ',' << est_v.z()
      << '\n';
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
  // Use the ideal (uncorrupted) measurements from the runner and inject the
  // IMU error model ourselves below, so the noise RNG seed is controllable for
  // Monte Carlo runs (ScenarioRunner's own samplers use fixed internal seeds).
  gtsam::ScenarioRunner runner(scenario, params, opts.dt);

  // Discrete per-sample noise stddev = density / sqrt(dt) (same convention as
  // ScenarioRunner). A fresh generator seeded from opts.seed makes each run
  // reproducible and each distinct seed an independent noise realization.
  std::mt19937 rng(opts.seed);
  std::normal_distribution<double> normal(0.0, 1.0);
  const double gyro_sd = opts.gyro_noise_sigma / std::sqrt(opts.dt);
  const double accel_sd = opts.accel_noise_sigma / std::sqrt(opts.dt);
  auto sampleNoise = [&](double sd) {
    return Eigen::Vector3d(sd * normal(rng), sd * normal(rng), sd * normal(rng));
  };

  // True IMU bias drifts as a random walk: b += rate * sqrt(dt) * N(0,1). The
  // matching continuous PSD is rate^2 (wired into Qc below), so the filter's
  // assumed bias process noise is consistent with the simulated drift.
  Eigen::Vector3d gyro_b = opts.gyro_bias;
  Eigen::Vector3d accel_b = opts.accel_bias;
  const double gyro_rw_step = opts.gyro_bias_rw * std::sqrt(opts.dt);
  const double accel_rw_step = opts.accel_bias_rw * std::sqrt(opts.dt);

  // Initialize the filter at the scenario's true initial state. IMU-only
  // propagation cannot observe the initial pose/velocity, so the origin must
  // match ground truth or the estimate drifts by the unmodelled offset.
  const gtsam::NavState gt0 = scenario.navState(0.0);
  TGState xi0 = TGState::identity();
  xi0.R = gt0.attitude();
  xi0.p = gt0.position();
  xi0.v = gt0.velocity();

  TGEqF filter(xi0, defaultSigma());
  // Match the filter's bias process noise to the simulated random-walk rate
  // (continuous PSD = rate^2) so the covariance reflects the true drift.
  TGEqF::Covariance18 Qc = defaultQc();
  if (opts.gyro_bias_rw > 0.0) {
    Qc.block<3, 3>(9, 9) =
        opts.gyro_bias_rw * opts.gyro_bias_rw * Eigen::Matrix3d::Identity();
  }
  if (opts.accel_bias_rw > 0.0) {
    Qc.block<3, 3>(15, 15) =
        opts.accel_bias_rw * opts.accel_bias_rw * Eigen::Matrix3d::Identity();
  }
  const gtsam::Vector3& g_vec = params->n_gravity;

  std::ofstream csv(opts.output_path);
  if (!csv) {
    throw std::runtime_error("Cannot open output file: " + opts.output_path);
  }

  csv << "t,gt_px,gt_py,gt_pz,gt_vx,gt_vy,gt_vz,est_px,est_py,est_pz,est_vx,"
         "est_vy,est_vz\n";

  RunSummary summary;
  double sum_pos_sq = 0.0;
  double sum_vel_sq = 0.0;

  double t = 0.0;
  const size_t num_steps =
      static_cast<size_t>(std::llround(opts.duration / opts.dt));

  for (size_t k = 0; k <= num_steps; ++k) {
    const gtsam::NavState gt = scenario.navState(t);

    // Estimate comes straight from the filter; the lift already integrates
    // position into the group element (no separate dead-reckoning needed).
    const Eigen::Vector3d est_p = filter.position();
    const Eigen::Vector3d est_v = filter.velocity();
    writeCsvRow(csv, t, gt, est_p, est_v);

    const Eigen::Vector3d gt_p(gt.position());
    const Eigen::Vector3d gt_v(gt.velocity());
    const double pos_err = (gt_p - est_p).norm();
    const double vel_err = (gt_v - est_v).norm();
    sum_pos_sq += pos_err * pos_err;
    sum_vel_sq += vel_err * vel_err;
    summary.final_pos_error = pos_err;
    summary.final_vel_error = vel_err;
    ++summary.num_samples;

    if (k == num_steps) {
      break;
    }

    // True bias and noise corrupt the IMU; the filter does not know them.
    // Defaults are zero, so the clean baseline is reproduced exactly.
    const gtsam::Vector3 omega =
        runner.actualAngularVelocity(t) + gyro_b + sampleNoise(gyro_sd);
    const gtsam::Vector3 accel =
        runner.actualSpecificForce(t) + accel_b + sampleNoise(accel_sd);
    filter.propagate(omega, accel, g_vec, Qc, opts.dt);

    // Random-walk the true bias for the next step (no-op when rate is zero).
    if (gyro_rw_step > 0.0) {
      gyro_b += gyro_rw_step *
                Eigen::Vector3d(normal(rng), normal(rng), normal(rng));
    }
    if (accel_rw_step > 0.0) {
      accel_b += accel_rw_step *
                 Eigen::Vector3d(normal(rng), normal(rng), normal(rng));
    }
    t += opts.dt;
  }

  if (summary.num_samples > 0) {
    summary.rms_pos_error =
        std::sqrt(sum_pos_sq / static_cast<double>(summary.num_samples));
    summary.rms_vel_error =
        std::sqrt(sum_vel_sq / static_cast<double>(summary.num_samples));
  }

  std::cout << scenario_name << " scenario complete.\n";
  std::cout << "  output: " << opts.output_path << '\n';
  std::cout << "  samples: " << summary.num_samples << '\n';
  std::cout << std::fixed << std::setprecision(6);
  std::cout << "  final position error (m): " << summary.final_pos_error
            << '\n';
  std::cout << "  final velocity error (m/s): " << summary.final_vel_error
            << '\n';
  std::cout << "  RMS position error (m): " << summary.rms_pos_error << '\n';
  std::cout << "  RMS velocity error (m/s): " << summary.rms_vel_error << '\n';

  return summary;
}

}  // namespace tgeqf::examples
