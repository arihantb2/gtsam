/**
 * @file  TGEqFScenarioExample.h
 * @brief TG-EqF adapter for the shared trajectory-example runner.
 *
 * The run loop, CLI, IMU simulation, CSV format and error reporting are shared
 * with the other filter examples (examples_common/FilterScenarioRunner.h); only
 * what is TG-EqF-specific lives here. Process noise comes from
 * examples_common/ImuProcessNoise.h and inputNoiseCov().
 */
#pragma once

#include <gtsam/navigation/NavState.h>
#include <gtsam/navigation/PreintegrationParams.h>
#include <gtsam/navigation/Scenario.h>
#include <gtsam_unstable/examples_common/FilterScenarioRunner.h>
#include <gtsam_unstable/examples_common/ImuProcessNoise.h>
#include <gtsam_unstable/examples_common/ScenarioHarness.h>
#include <gtsam_unstable/examples_common/TrajectoryCsv.h>
#include <gtsam_unstable/tg_eqf/EqF.h>

#include <ostream>
#include <string>

namespace tgeqf::examples {

// Run configuration and reporting are shared with the other filter examples.
// The bias sigma covers bg, ba and bv; bv has no physical truth and is not
// perturbed (see State::b_v).
using imu_scenarios::parseRunOptions;
using imu_scenarios::RunOptions;
using imu_scenarios::RunSummary;

/// Satisfies the adapter contract in examples_common/FilterScenarioRunner.h.
class ScenarioAdapter {
 public:
  static constexpr int kDim = 18;
  using Filter = TGEqF;
  using TrueState = State;
  using Covariance = TGEqF::Covariance18;

  explicit ScenarioAdapter(const RunOptions& opts)
      : noise_(imu_scenarios::toFilterImuNoise<ImuNoise>(
            imu_scenarios::resolvedProcessNoise(opts))),
        gravity_(
            gtsam::PreintegrationParams::MakeSharedU(imu_scenarios::kGravity)
                ->n_gravity) {}

  /// The virtual bias bv is estimate-only: it has no ground-truth counterpart.
  std::string csvHeader() const {
    return imu_scenarios::trajectoryCsvHeader(
        {"v", "p", "bg", "ba"}, {"v", "p", "bg", "ba", "bv"},
        {"R", "v", "p", "bg", "ba", "bv"});
  }

  /// Fix the origin xi0 at the scenario's nominal initial state with zero bias
  /// and start the observer at the identity, so the estimate starts at xi0.
  Filter makeFilter(const gtsam::NavState& gt0, const Covariance& P0) const {
    State xi0 = State::identity();
    xi0.R = gt0.attitude();
    xi0.v = gt0.velocity();
    xi0.p = gt0.position();
    return Filter(xi0, P0, TGElement::Identity());
  }

  TrueState trueState(const gtsam::NavState& gt, const Eigen::Vector3d& true_bg,
                      const Eigen::Vector3d& true_ba) const {
    State xi;
    xi.R = gt.attitude();
    xi.v = gt.velocity();
    xi.p = gt.position();
    xi.b_w = true_bg;
    xi.b_a = true_ba;
    xi.b_v = Eigen::Vector3d::Zero();  // virtual bias: no physical truth
    return xi;
  }

  void propagate(Filter& filter, const imu_scenarios::ImuMeasurement& imu,
                 double dt) const {
    filter.propagate(imu.omega, imu.accel, gravity_, noise_, dt);
  }

  void updatePosition(Filter& filter,
                      const imu_scenarios::PositionMeasurement& z) const {
    filter.update_position(z.position, z.covariance, /*use_Cstar=*/true);
  }

  void updateDvl(Filter& filter, const imu_scenarios::DvlMeasurement& z) const {
    filter.update_dvl(z.body_velocity, z.covariance);
  }

  /// Truth, estimate, tangent error and covariance. The error is
  /// eps = Local(xi_ref, phi(g^{-1}, xi_true)) and the covariance is the
  /// matching errorCovariance(), both in the fixed origin chart.
  void writeCsvRow(std::ostream& out, double t, const TrueState& xi_true,
                   const Filter& filter) const {
    const State& xi_hat = filter.state();
    imu_scenarios::CsvRow row(out, t);
    row.quat(xi_true.R)
        .v3(xi_true.v)
        .v3(xi_true.p)
        .v3(xi_true.b_w)
        .v3(xi_true.b_a);
    row.quat(xi_hat.R)
        .v3(xi_hat.v)
        .v3(xi_hat.p)
        .v3(xi_hat.b_w)
        .v3(xi_hat.b_a)
        .v3(xi_hat.b_v);
    row.tangent<kDim>(filter.errorStateVector(xi_true))
        .covarianceBlocks<kDim>(filter.errorCovariance());
    row.end();
  }

 private:
  ImuNoise noise_;
  gtsam::Vector3 gravity_;
};

inline RunSummary runScenario(const gtsam::Scenario& scenario,
                              const RunOptions& opts,
                              const std::string& scenario_name) {
  return imu_scenarios::runFilterScenario(ScenarioAdapter(opts), scenario, opts,
                                          scenario_name);
}

}  // namespace tgeqf::examples
