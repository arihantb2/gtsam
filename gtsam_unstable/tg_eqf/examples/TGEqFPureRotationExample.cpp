/**
 * @file TGEqFPureRotationExample.cpp
 * @brief TG-EqF IMU-only propagation under pure rotation in place.
 *
 * Constant yaw rate, zero translational velocity: the body spins at a fixed
 * position. Isolates attitude propagation and gyro-bias excitation without
 * coupling to translation (complements the Circle scenario, which couples
 * rotation with motion).
 */

#include "TGEqFScenarioExample.h"

#include <gtsam/navigation/Scenario.h>

#include <iostream>

using namespace gtsam;
using namespace tgeqf::examples;

int main(int argc, char* argv[]) {
  constexpr double kYawRate = 0.5;  // rad/s about body z

  try {
    const RunOptions opts =
        parseRunOptions(argc, argv, "tg_eqf_pure_rotation.csv");

    // Nonzero angular velocity, zero body velocity -> rotation in place.
    ConstantTwistScenario scenario(Vector3(0, 0, kYawRate), Vector3::Zero());

    runScenario(scenario, opts, "Pure-rotation");
  } catch (const std::exception& e) {
    std::cerr << "TGEqFPureRotationExample failed: " << e.what() << '\n';
    return 2;
  }

  return 0;
}
