/**
 * @file  MEKFSinusoidExample.cpp
 * @brief Baseline MEKF, IMU-only, sinusoidal (weaving) trajectory.
 *
 * Forward drift at constant speed plus a lateral oscillation. No stock
 * gtsam::Scenario produces a sinusoid, so a small closed-form Scenario subclass
 * gives exact ground truth and IMU at every t.
 *
 * Trajectory (nav frame, identity attitude):
 *   p(t)   = ( v t,        A sin(w t),       0 )
 *   v_n(t) = ( v,          A w cos(w t),     0 )
 *   a_n(t) = ( 0,         -A w^2 sin(w t),   0 )
 *   omega_b = 0
 */
#include "MEKFScenarioExample.h"

#include <gtsam/geometry/Pose3.h>
#include <gtsam/navigation/Scenario.h>

#include <cmath>

using namespace gtsam;
using namespace mekf::examples;

namespace {

/// Closed-form lateral-sinusoid scenario (identity attitude, no rotation).
class SinusoidScenario : public Scenario {
 public:
  SinusoidScenario(double forward_speed, double amplitude, double omega)
      : v_(forward_speed), A_(amplitude), w_(omega) {}

  Pose3 pose(double t) const override {
    return Pose3(Rot3(), Point3(v_ * t, A_ * std::sin(w_ * t), 0.0));
  }
  Vector3 omega_b(double /*t*/) const override { return Vector3::Zero(); }
  Vector3 velocity_n(double t) const override {
    return Vector3(v_, A_ * w_ * std::cos(w_ * t), 0.0);
  }
  Vector3 acceleration_n(double t) const override {
    return Vector3(0.0, -A_ * w_ * w_ * std::sin(w_ * t), 0.0);
  }

 private:
  const double v_, A_, w_;
};

}  // namespace

int main(int argc, char* argv[]) {
  constexpr double kForwardSpeed = 1.0;  // m/s along nav x
  constexpr double kAmplitude = 2.0;     // m lateral (nav y)
  constexpr double kPeriod = 10.0;       // s per lateral cycle
  try {
    const RunOptions opts = parseRunOptions(argc, argv, "mekf_sinusoid.csv");
    const double omega = 2.0 * M_PI / kPeriod;
    SinusoidScenario scenario(kForwardSpeed, kAmplitude, omega);
    runScenario(scenario, opts, "Sinusoid");
  } catch (const std::exception& e) {
    std::cerr << "MEKFSinusoidExample failed: " << e.what() << '\n';
    return 2;
  }
  return 0;
}
