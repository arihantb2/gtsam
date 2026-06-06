/**
 * @file  MEKFPureRotationExample.cpp
 * @brief Baseline MEKF, IMU-only, pure rotation (yaw rate, no translation).
 */
#include "MEKFScenarioExample.h"

#include <gtsam/navigation/Scenario.h>

using namespace gtsam;
using namespace mekf::examples;

int main(int argc, char* argv[]) {
  constexpr double kYawRate = 0.5;  // rad/s about body z
  try {
    const RunOptions opts =
        parseRunOptions(argc, argv, "mekf_pure_rotation.csv");
    ConstantTwistScenario scenario(Vector3(0, 0, kYawRate), Vector3::Zero());
    runScenario(scenario, opts, "Pure-rotation");
  } catch (const std::exception& e) {
    std::cerr << "MEKFPureRotationExample failed: " << e.what() << '\n';
    return 2;
  }
  return 0;
}
