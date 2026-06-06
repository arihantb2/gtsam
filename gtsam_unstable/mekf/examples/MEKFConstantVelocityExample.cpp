/**
 * @file  MEKFConstantVelocityExample.cpp
 * @brief Baseline MEKF, IMU-only, constant linear velocity (straight line).
 */
#include "MEKFScenarioExample.h"

#include <gtsam/navigation/Scenario.h>

using namespace gtsam;
using namespace mekf::examples;

int main(int argc, char* argv[]) {
  constexpr double kSpeed = 1.0;  // m/s along body x
  try {
    const RunOptions opts =
        parseRunOptions(argc, argv, "mekf_constant_velocity.csv");
    // Zero angular velocity, constant body velocity (body frame == nav frame).
    ConstantTwistScenario scenario(Vector3::Zero(), Vector3(kSpeed, 0, 0));
    runScenario(scenario, opts, "Constant-velocity");
  } catch (const std::exception& e) {
    std::cerr << "MEKFConstantVelocityExample failed: " << e.what() << '\n';
    return 2;
  }
  return 0;
}
