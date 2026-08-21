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

namespace gtsam::tgeqf::examples {

// Run configuration and reporting are shared with the other filter examples.
// The shared options only describe the physical state: bv has no physical
// truth, so the initial estimate leaves it at zero (see State::b_v) and its
// initial covariance comes from the filter itself.
using imu_scenarios::parseRunOptions;
using imu_scenarios::RunOptions;
using imu_scenarios::RunSummary;

/// Satisfies the adapter contract in examples_common/FilterScenarioRunner.h.
class ScenarioAdapter {
 public:
  static constexpr int kDim = 18;
  using Filter = TGEqF;
  using TrueState = State;

  explicit ScenarioAdapter(const RunOptions& opts)
      : noise_(imu_scenarios::toFilterImuNoise<ImuNoise>(
            imu_scenarios::resolvedProcessNoise(opts))),
        gravity_(
            gtsam::PreintegrationParams::MakeSharedU(imu_scenarios::kGravity)
                ->n_gravity),
        reset_step_(opts.tg_eqf_reset_step),
        depth_direct_(opts.tg_eqf_depth_direct) {}

  /// The virtual bias bv is estimate-only: it has no ground-truth counterpart.
  std::string csvHeader() const {
    return imu_scenarios::trajectoryCsvHeader(
        {"v", "p", "bg", "ba"}, {"v", "p", "bg", "ba", "bv"},
        {"R", "v", "p", "bg", "ba", "bv"});
  }

  /// Pin the origin xi0 at the manifold identity, so the error dynamics are
  /// always linearized about the same point, and carry the perturbed initial
  /// belief in the group element: phi(X0, xi0) is that belief. P0 covers the
  /// physical state only; TGEqF::initialCovariance() adds the virtual-bias
  /// block.
  Filter makeFilter(const imu_scenarios::InitialEstimate& initial,
                    const imu_scenarios::PhysicalStateCovariance& P0) const {
    State xi_hat0;
    xi_hat0.R = initial.nav.attitude();
    xi_hat0.v = initial.nav.velocity();
    xi_hat0.p = initial.nav.position();
    xi_hat0.b_w = initial.bias_gyro;
    xi_hat0.b_a = initial.bias_accel;
    // b_v stays zero: the virtual bias has no physical truth to perturb.
    const State xi0 = State::identity();
    Filter filter(xi0, Filter::initialCovariance(P0), phiInverse(xi0, xi_hat0));
    filter.set_reset_step(reset_step_);
    return filter;
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
    filter.update_position(z.position, z.covariance);
  }

  void updateDvl(Filter& filter, const imu_scenarios::DvlMeasurement& z) const {
    filter.update_dvl(z.body_velocity, z.covariance);
  }

  void updateDepth(Filter& filter,
                   const imu_scenarios::DepthMeasurement& z) const {
    if (depth_direct_) {
      filter.update_depth_direct(z.depth, z.covariance);
    } else {
      filter.update_depth(z.depth, z.covariance);
    }
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
  bool reset_step_;
  bool depth_direct_;
};

inline RunSummary runScenario(const gtsam::Scenario& scenario,
                              const RunOptions& opts,
                              const std::string& scenario_name) {
  return imu_scenarios::runFilterScenario(ScenarioAdapter(opts), scenario, opts,
                                          scenario_name);
}

}  // namespace gtsam::tgeqf::examples
