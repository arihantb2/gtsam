/**
 * @file  TFGInEKFConstantVelocityExample.cpp
 * @brief TFG-IEKF IMU-only, constant linear velocity.
 *
 * Constant body velocity, zero rotation: straight-line motion, no linear
 * acceleration (specific force is gravity only). Isolates velocity/position
 * propagation from any excitation.
 */
#include "TFGInEKFScenarioExample.h"

#include <gtsam/navigation/Scenario.h>

using namespace gtsam;
using namespace tfg::examples;

int main(int argc, char* argv[]) {
  constexpr double kSpeed = 1.0;  // m/s along body x
  try {
    const RunOptions opts =
        parseRunOptions(argc, argv, "tfg_inekf_constant_velocity.csv");
    // Zero angular velocity, constant body velocity (body frame == nav frame).
    ConstantTwistScenario scenario(Vector3::Zero(), Vector3(kSpeed, 0, 0));
    runScenario(scenario, opts, "Constant-velocity");
  } catch (const std::exception& e) {
    std::cerr << "TFGInEKFConstantVelocityExample failed: " << e.what() << '\n';
    return 2;
  }
  return 0;
}
