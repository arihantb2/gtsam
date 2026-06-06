/**
 * @file TGEqFSinusoidExample.cpp
 * @brief TG-EqF IMU-only propagation on a sinusoidal (weaving) trajectory.
 *
 * No built-in gtsam::Scenario produces a sinusoid (the stock ones are
 * constant-twist, constant-acceleration, or discrete-interpolated). This file
 * defines a small closed-form Scenario subclass instead, so the ground truth
 * and the synthesized IMU are exact at every t.
 *
 * Trajectory (navigation frame, identity attitude):
 *   p(t)   = ( v*t,           A sin(w t),        0 )
 *   v_n(t) = ( v,             A w cos(w t),      0 )
 *   a_n(t) = ( 0,            -A w^2 sin(w t),    0 )
 *   omega_b = 0
 *
 * Forward drift at constant speed with a lateral oscillation: time-varying
 * specific force exercises the filter's error dynamics periodically -- a good
 * consistency / NEES stressor that the constant-twist and constant-accel
 * scenarios cannot produce.
 */

#include "TGEqFScenarioExample.h"

#include <gtsam/geometry/Pose3.h>
#include <gtsam/navigation/Scenario.h>

#include <cmath>
#include <iostream>

using namespace gtsam;
using namespace tgeqf::examples;

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
  constexpr double kAmplitude = 2.0;       // m lateral (nav y)
  constexpr double kPeriod = 10.0;          // s per lateral cycle

  try {
    const RunOptions opts = parseRunOptions(argc, argv, "tg_eqf_sinusoid.csv");

    const double omega = 2.0 * M_PI / kPeriod;
    SinusoidScenario scenario(kForwardSpeed, kAmplitude, omega);

    runScenario(scenario, opts, "Sinusoid");
  } catch (const std::exception& e) {
    std::cerr << "TGEqFSinusoidExample failed: " << e.what() << '\n';
    return 2;
  }

  return 0;
}
