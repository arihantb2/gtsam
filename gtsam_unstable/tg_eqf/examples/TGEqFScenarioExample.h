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
  double init_sigma = 0.1;         // initial state stddev: Sigma0 = init_sigma^2 I
  unsigned seed = 42;              // RNG seed for noise (vary it for Monte Carlo)
  int log_decim = 1;               // log every Nth step (1 = every step)
  double pos_rate = 0.0;           // GNSS position update rate in Hz (0 = disabled)
  double pos_noise_sigma = 0.1;    // Position measurement noise stddev (m)
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
    } else if (arg == "--init-sigma" && i + 1 < argc) {
      opts.init_sigma = std::stod(argv[++i]);
    } else if (arg == "--log-decim" && i + 1 < argc) {
      opts.log_decim = std::max(1, std::stoi(argv[++i]));
    } else if (arg == "--seed" && i + 1 < argc) {
      opts.seed = static_cast<unsigned>(std::stoul(argv[++i]));
    } else if (arg == "--pos-rate" && i + 1 < argc) {
      opts.pos_rate = std::stod(argv[++i]);
    } else if (arg == "--pos-noise" && i + 1 < argc) {
      opts.pos_noise_sigma = std::stod(argv[++i]);
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

/// Column header matching writeCsvRow() below.
inline const char* csvHeader() {
  return "t,"
         "gt_px,gt_py,gt_pz,gt_vx,gt_vy,gt_vz,"
         "est_px,est_py,est_pz,est_vx,est_vy,est_vz,"
         "att_err_x,att_err_y,att_err_z,"
         "est_bgx,est_bgy,est_bgz,est_bvx,est_bvy,est_bvz,"
         "est_bax,est_bay,est_baz,"
         "true_bgx,true_bgy,true_bgz,true_bax,true_bay,true_baz,"
         "eps_att_x,eps_att_y,eps_att_z,eps_pos_x,eps_pos_y,eps_pos_z,"
         "eps_vel_x,eps_vel_y,eps_vel_z,eps_bg_x,eps_bg_y,eps_bg_z,"
         "eps_bv_x,eps_bv_y,eps_bv_z,eps_ba_x,eps_ba_y,eps_ba_z,"
         // upper-triangle of each 3x3 covariance block (origin frame):
         "P_att_00,P_att_01,P_att_02,P_att_11,P_att_12,P_att_22,"
         "P_pos_00,P_pos_01,P_pos_02,P_pos_11,P_pos_12,P_pos_22,"
         "P_vel_00,P_vel_01,P_vel_02,P_vel_11,P_vel_12,P_vel_22,"
         "P_bg_00,P_bg_01,P_bg_02,P_bg_11,P_bg_12,P_bg_22,"
         "P_bv_00,P_bv_01,P_bv_02,P_bv_11,P_bv_12,P_bv_22,"
         "P_ba_00,P_ba_01,P_ba_02,P_ba_11,P_ba_12,P_ba_22\n";
}

/// Build the true TGState from ground-truth nav and the true IMU biases.
inline TGState trueState(const gtsam::NavState& gt,
                         const Eigen::Vector3d& true_bg,
                         const Eigen::Vector3d& true_ba) {
  TGState xi;
  xi.R = gt.attitude();
  xi.v = gt.velocity();
  xi.p = gt.position();
  xi.b_w = true_bg;
  xi.b_a = true_ba;
  xi.b_v = Eigen::Vector3d::Zero();  // virtual bias: no physical truth
  return xi;
}

/// Write one trajectory row: ground truth, filter estimate, attitude error,
/// estimated/true biases, the EqF origin-frame error eps, and the upper triangle
/// of each 3x3 covariance block (origin frame), in TGState tangent order
/// [att, pos, vel, bg, bv, ba].
///
/// eps is the equivariant error in the fixed reference chart:
///   eps = Local(xi_ref, phi(g^{-1}, xi_true)),
/// which the filter's errorCovariance() P is the covariance of — so per 3-DoF
/// state block b the block-marginal NEES is eps_b^T P_b^{-1} eps_b ~ chi^2_3
/// (the full-state NEES uses the full 18x18 P).
inline void writeCsvRow(std::ostream& out, double t, const gtsam::NavState& gt,
                        const TGEqF& filter, const TGState& xi_ref,
                        const Eigen::Vector3d& true_bg,
                        const Eigen::Vector3d& true_ba) {
  // Attitude error as a right-tangent vector: Logmap(gt_R^-1 * est_R).
  const Eigen::Vector3d att_err =
      gtsam::Rot3::Logmap(gt.attitude().between(filter.attitude()));

  // Origin-frame EqF error and its covariance.
  const TGState xi_true = trueState(gt, true_bg, true_ba);
  const TGState e = phi(filter.groupEstimate().inverse(), xi_true);
  const Eigen::Matrix<double, 18, 1> eps =
      gtsam::traits<TGState>::Local(xi_ref, e);
  const Eigen::Matrix<double, 18, 18>& P = filter.errorCovariance();

  auto v3 = [&out](const Eigen::Vector3d& x) {
    out << ',' << x.x() << ',' << x.y() << ',' << x.z();
  };
  out << std::fixed << std::setprecision(9) << t;
  v3(Eigen::Vector3d(gt.position()));
  v3(Eigen::Vector3d(gt.velocity()));
  v3(filter.position());
  v3(filter.velocity());
  v3(att_err);
  v3(filter.bias_gyro());
  v3(filter.bias_vel());
  v3(filter.bias_accel());
  v3(true_bg);
  v3(true_ba);
  for (int i = 0; i < 18; ++i) {
    out << ',' << eps(i);
  }
  for (int o = 0; o < 18; o += 3) {  // upper triangle of each 3x3 block
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

  const TGEqF::Covariance18 Sigma0 =
      opts.init_sigma * opts.init_sigma * TGEqF::Covariance18::Identity();
  TGEqF filter(xi0, Sigma0);

  // Match the filter's process noise to the simulated IMU error model so the
  // covariance is physically meaningful (a prerequisite for any consistency /
  // NEES analysis): gyro white noise drives attitude, accel white noise drives
  // velocity, and the bias random-walk rates drive the bias blocks (continuous
  // PSD = rate^2). Channels left at zero keep the defaultQc() value.
  const Eigen::Matrix3d I3 = Eigen::Matrix3d::Identity();
  TGEqF::Covariance18 Qc = defaultQc();
  if (opts.gyro_noise_sigma > 0.0) {
    Qc.block<3, 3>(0, 0) = opts.gyro_noise_sigma * opts.gyro_noise_sigma * I3;
  }
  if (opts.accel_noise_sigma > 0.0) {
    Qc.block<3, 3>(6, 6) = opts.accel_noise_sigma * opts.accel_noise_sigma * I3;
  }
  if (opts.gyro_bias_rw > 0.0) {
    Qc.block<3, 3>(9, 9) = opts.gyro_bias_rw * opts.gyro_bias_rw * I3;
  }
  if (opts.accel_bias_rw > 0.0) {
    Qc.block<3, 3>(15, 15) = opts.accel_bias_rw * opts.accel_bias_rw * I3;
  }
  const gtsam::Vector3& g_vec = params->n_gravity;

  std::ofstream csv(opts.output_path);
  if (!csv) {
    throw std::runtime_error("Cannot open output file: " + opts.output_path);
  }

  csv << csvHeader();

  RunSummary summary;
  double sum_pos_sq = 0.0;
  double sum_vel_sq = 0.0;

  double t = 0.0;
  const size_t num_steps =
      static_cast<size_t>(std::llround(opts.duration / opts.dt));

  double next_pos_time = (opts.pos_rate > 0.0) ? (1.0 / opts.pos_rate) : -1.0;

  for (size_t k = 0; k <= num_steps; ++k) {
    const gtsam::NavState gt = scenario.navState(t);

    if (opts.pos_rate > 0.0 && t >= next_pos_time - 1e-5) {
      const Eigen::Vector3d true_pos = gt.position();
      const Eigen::Vector3d pos_meas = true_pos + sampleNoise(opts.pos_noise_sigma);
      const TGEqF::Covariance3 R_pos =
          opts.pos_noise_sigma * opts.pos_noise_sigma * TGEqF::Covariance3::Identity();
      filter.update_position(pos_meas, R_pos, true);
      next_pos_time += 1.0 / opts.pos_rate;
    }

    // Estimate comes straight from the filter; the lift already integrates
    // position into the group element (no separate dead-reckoning needed).
    const Eigen::Vector3d est_p = filter.position();
    const Eigen::Vector3d est_v = filter.velocity();
    // Log every Nth step (and always the final one) to bound MC file sizes.
    if (k % static_cast<size_t>(opts.log_decim) == 0 || k == num_steps) {
      writeCsvRow(csv, t, gt, filter, xi0, gyro_b, accel_b);
    }

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
