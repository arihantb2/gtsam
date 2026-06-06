/**
 * @file TGEqFConstantVelocityExample.cpp
 * @brief TG-EqF IMU-only propagation at constant linear velocity.
 *
 * Constant body velocity, zero rotation: straight-line motion with no linear
 * acceleration. Specific force is gravity only (as in the Static case), but the
 * body is moving -- isolates velocity/position propagation from any excitation.
 * Complements the Straight-line scenario, which adds constant acceleration.
 */

#include "TGEqFScenarioExample.h"

#include <gtsam/navigation/Scenario.h>

#include <iostream>

using namespace gtsam;
using namespace tgeqf::examples;

int main(int argc, char* argv[]) {
  constexpr double kSpeed = 1.0;  // m/s along body x

  try {
    const RunOptions opts =
        parseRunOptions(argc, argv, "tg_eqf_constant_velocity.csv");

    // Zero angular velocity, constant body velocity: a_b = w x v = 0, so the
    // path is a straight line at constant speed (body frame == nav frame).
    ConstantTwistScenario scenario(Vector3::Zero(), Vector3(kSpeed, 0, 0));

    runScenario(scenario, opts, "Constant-velocity");
  } catch (const std::exception& e) {
    std::cerr << "TGEqFConstantVelocityExample failed: " << e.what() << '\n';
    return 2;
  }

  return 0;
}
