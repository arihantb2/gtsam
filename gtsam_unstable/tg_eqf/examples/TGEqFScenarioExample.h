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
#include <string>

namespace tgeqf::examples {

struct RunOptions {
  double duration = 30.0;
  double dt = 0.01;
  std::string output_path = "tg_eqf_trajectory.csv";
};

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
  const gtsam::imuBias::ConstantBias zero_bias;
  gtsam::ScenarioRunner runner(scenario, params, opts.dt, zero_bias);

  // Initialize the filter at the scenario's true initial state. IMU-only
  // propagation cannot observe the initial pose/velocity, so the origin must
  // match ground truth or the estimate drifts by the unmodelled offset.
  const gtsam::NavState gt0 = scenario.navState(0.0);
  TGState xi0 = TGState::identity();
  xi0.R = gt0.attitude();
  xi0.p = gt0.position();
  xi0.v = gt0.velocity();

  TGEqF filter(xi0, defaultSigma());
  const TGEqF::Covariance18 Qc = defaultQc();
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

    const gtsam::Vector3 omega = runner.actualAngularVelocity(t);
    const gtsam::Vector3 accel = runner.actualSpecificForce(t);
    filter.propagate(omega, accel, g_vec, Qc, opts.dt);
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
