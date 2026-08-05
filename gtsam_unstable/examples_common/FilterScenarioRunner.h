/**
 * @file  FilterScenarioRunner.h
 * @brief The one trajectory-example loop, shared by tg_eqf / tfg_inekf / mekf.
 *
 * The loop, the RNG draw order and the logged quantities are identical for
 * every filter; only construction, propagation and the update calls differ.
 * Those live in a small per-filter adapter (see <filter>/examples/
 * <Filter>ScenarioExample.h), which stays next to its filter so this header
 * remains filter-agnostic.
 *
 * Adapter contract:
 * @code
 *   static constexpr int kDim;   // tangent dimension (15, 18, ...)
 *   using Filter;                // the filter type
 *   using TrueState;             // state type holding ground truth + biases
 *   using Covariance = Eigen::Matrix<double, kDim, kDim>;
 *
 *   std::string csvHeader() const;
 *   Filter makeFilter(const gtsam::NavState& gt0, const Covariance& P0) const;
 *   TrueState trueState(const gtsam::NavState& gt, const Eigen::Vector3d& bg,
 *                       const Eigen::Vector3d& ba) const;
 *   void propagate(Filter&, const ImuMeasurement&, double dt) const;
 *   void updatePosition(Filter&, const PositionMeasurement&) const;
 *   void updateDvl(Filter&, const DvlMeasurement&) const;
 *   void writeCsvRow(std::ostream&, double t, const TrueState&,
 *                    const Filter&) const;
 * @endcode
 * The adapter is constructed from RunOptions, so anything it needs per run
 * (process noise, gravity) is set up once before the loop starts.
 */
#pragma once

#include <gtsam/navigation/NavState.h>
#include <gtsam/navigation/PreintegrationParams.h>
#include <gtsam/navigation/Scenario.h>
#include <gtsam/navigation/ScenarioRunner.h>
#include <gtsam_unstable/examples_common/IMUScenarios.h>
#include <gtsam_unstable/examples_common/ScenarioHarness.h>

#include <cmath>
#include <cstddef>
#include <fstream>
#include <random>
#include <stdexcept>
#include <string>

namespace imu_scenarios {

/// Drive `scenario` through the adapter's filter, logging the trajectory CSV to
/// opts.output_path and returning the run's error summary.
template <class Adapter>
inline RunSummary runFilterScenario(const Adapter& adapter,
                                    const gtsam::Scenario& scenario,
                                    const RunOptions& opts,
                                    const std::string& scenario_name) {
  validateRunOptions(opts);

  std::ofstream csv(opts.output_path);
  if (!csv) {
    throw std::runtime_error("Cannot open output file: " + opts.output_path);
  }
  csv << adapter.csvHeader();

  auto params = gtsam::PreintegrationParams::MakeSharedU(kGravity);
  std::mt19937 rng(opts.seed);

  // Ground truth starts away from the state the filter is initialized at, drawn
  // from the init-*-sigma options, so P0 describes an error that actually
  // occurs. true_scenario is what the IMU is driven through and what `gt`
  // refers to below; the filter starts at the nominal scenario's initial state
  // with zero bias.
  const InitialPerturbation init_pert = sampleInitialPerturbation(rng, opts);
  const PerturbedScenario true_scenario(scenario, init_pert.dR, init_pert.dv,
                                        init_pert.dp);
  const gtsam::ScenarioRunner runner(true_scenario, params, opts.dt);

  // The IMU the filter sees: true biases (perturbed off the filter's zero
  // belief) plus white noise. The filter knows neither.
  ImuSimulator sim(opts, rng, opts.gyro_bias + init_pert.dbg,
                   opts.accel_bias + init_pert.dba);

  auto filter = adapter.makeFilter(scenario.navState(0.0),
                                   initialCovariance<Adapter::kDim>(opts));

  PeriodicTrigger pos_updates(opts.pos_rate, opts.aiding_start_time);
  PeriodicTrigger dvl_updates(opts.dvl_rate, opts.aiding_start_time);
  RunSummaryAccumulator errors;

  const size_t num_steps =
      static_cast<size_t>(std::llround(opts.duration / opts.dt));
  double t = 0.0;

  // The filter state is at time t throughout the loop body, matching gt: update
  // and log first, propagate to t + dt last.
  for (size_t k = 0; k <= num_steps; ++k) {
    const gtsam::NavState gt = true_scenario.navState(t);
    const ImuMeasurement imu = sim.sample(runner, t);

    if (pos_updates.due(t)) {
      adapter.updatePosition(filter,
                             sim.samplePosition(gt, opts.pos_noise_sigma));
    }
    if (dvl_updates.due(t)) {
      adapter.updateDvl(filter, sim.sampleDvl(gt, opts.dvl_noise_sigma));
    }

    // Log every Nth step (and always the final one) to bound MC file sizes.
    if (k % static_cast<size_t>(opts.log_decim) == 0 || k == num_steps) {
      adapter.writeCsvRow(
          csv, t, adapter.trueState(gt, sim.gyroBias(), sim.accelBias()),
          filter);
    }
    errors.add(gt, filter.attitude(), filter.velocity(), filter.position());

    if (k == num_steps) {
      break;
    }

    adapter.propagate(filter, imu, opts.dt);
    sim.randomWalkBias();
    t += opts.dt;
  }

  const RunSummary summary = errors.summary();
  printRunSummary(summary, scenario_name, opts.output_path);
  return summary;
}

}  // namespace imu_scenarios
