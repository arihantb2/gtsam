/**
 * @file  TFGInEKFPureRotationExample.cpp
 * @brief TFG-IEKF IMU-only, pure rotation in place.
 *
 * Constant angular velocity about body z, zero linear velocity: the body spins
 * but does not translate. Specific force is gravity only. Isolates the rotation
 * propagation and its coupling into the (zero) velocity/position.
 */
#include "TFGInEKFScenarioExample.h"

#include <gtsam/navigation/Scenario.h>

using namespace gtsam;
using namespace tfg::examples;

int main(int argc, char* argv[]) {
  constexpr double kYawRate = 0.5;  // rad/s about body z
  try {
    const RunOptions opts =
        parseRunOptions(argc, argv, "tfg_inekf_pure_rotation.csv");
    ConstantTwistScenario scenario(Vector3(0, 0, kYawRate), Vector3::Zero());
    runScenario(scenario, opts, "Pure-rotation");
  } catch (const std::exception& e) {
    std::cerr << "TFGInEKFPureRotationExample failed: " << e.what() << '\n';
    return 2;
  }
  return 0;
}
