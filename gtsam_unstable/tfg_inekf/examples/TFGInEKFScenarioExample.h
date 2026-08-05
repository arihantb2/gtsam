/**
 * @file  TFGInEKFScenarioExample.h
 * @brief TFG-IEKF adapter for the shared trajectory-example runner.
 *
 * The run loop, CLI, IMU simulation, CSV format and error reporting are shared
 * with the other filter examples (examples_common/FilterScenarioRunner.h); only
 * what is TFG-IEKF-specific lives here.
 */
#pragma once

#include <gtsam/navigation/NavState.h>
#include <gtsam/navigation/Scenario.h>
#include <gtsam_unstable/examples_common/FilterScenarioRunner.h>
#include <gtsam_unstable/examples_common/ImuProcessNoise.h>
#include <gtsam_unstable/examples_common/ScenarioHarness.h>
#include <gtsam_unstable/examples_common/TrajectoryCsv.h>
#include <gtsam_unstable/tfg_inekf/InEKF.h>

#include <ostream>
#include <string>

namespace tfg::examples {

// Run configuration and reporting are shared with the other filter examples.
using imu_scenarios::parseRunOptions;
using imu_scenarios::RunOptions;
using imu_scenarios::RunSummary;

/// Satisfies the adapter contract in examples_common/FilterScenarioRunner.h.
class ScenarioAdapter {
 public:
  static constexpr int kDim = TwoFrameGroup::dim;
  using Filter = TfgInEKF;
  using TrueState = TwoFrameGroup;
  using Covariance = Eigen::Matrix<double, kDim, kDim>;

  explicit ScenarioAdapter(const RunOptions& opts)
      : noise_(imu_scenarios::toFilterImuNoise<ImuNoise>(
            imu_scenarios::resolvedProcessNoise(opts))) {}

  std::string csvHeader() const {
    return imu_scenarios::trajectoryCsvHeader({"v", "p", "bg", "ba"},
                                              {"v", "p", "bg", "ba"},
                                              {"R", "v", "p", "bg", "ba"});
  }

  /// Start at the scenario's nominal initial state with zero bias.
  Filter makeFilter(const gtsam::NavState& gt0, const Covariance& P0) const {
    const TwoFrameGroup X0 = TwoFrameGroup::FromState(
        gt0.attitude(), gt0.velocity(), gt0.position(), Eigen::Vector3d::Zero(),
        Eigen::Vector3d::Zero());
    return Filter(X0, P0);
  }

  TrueState trueState(const gtsam::NavState& gt, const Eigen::Vector3d& true_bg,
                      const Eigen::Vector3d& true_ba) const {
    return TwoFrameGroup::FromState(gt.attitude(), gt.velocity(), gt.position(),
                                    true_bg, true_ba);
  }

  void propagate(Filter& filter, const imu_scenarios::ImuMeasurement& imu,
                 double dt) const {
    filter.propagate(imu.omega, imu.accel, noise_, dt);
  }

  void updatePosition(Filter& filter,
                      const imu_scenarios::PositionMeasurement& z) const {
    filter.update_position(z.position, z.covariance);
  }

  void updateDvl(Filter& filter, const imu_scenarios::DvlMeasurement& z) const {
    filter.update_dvl(z.body_velocity, z.covariance);
  }

  /// Truth, estimate, tangent error and covariance, all in the group's local
  /// chart. The logged biases are the physical ones, b = R^{-1}(-gamma).
  void writeCsvRow(std::ostream& out, double t, const TrueState& X_true,
                   const Filter& filter) const {
    const TwoFrameGroup& X_hat = filter.state();
    imu_scenarios::CsvRow row(out, t);
    row.quat(X_true.R)
        .v3(X_true.v)
        .v3(X_true.p)
        .v3(X_true.bias_omega())
        .v3(X_true.bias_accel());
    row.quat(X_hat.R)
        .v3(X_hat.v)
        .v3(X_hat.p)
        .v3(X_hat.bias_omega())
        .v3(X_hat.bias_accel());
    row.tangent<kDim>(filter.errorStateVector(X_true))
        .covarianceBlocks<kDim>(filter.covariance());
    row.end();
  }

 private:
  ImuNoise noise_;
};

inline RunSummary runScenario(const gtsam::Scenario& scenario,
                              const RunOptions& opts,
                              const std::string& scenario_name) {
  return imu_scenarios::runFilterScenario(ScenarioAdapter(opts), scenario, opts,
                                          scenario_name);
}

}  // namespace tfg::examples
