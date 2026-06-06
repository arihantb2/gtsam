/**
 * @file  MEKFCoordinatedTurnExample.cpp
 * @brief Baseline MEKF, IMU-only, coordinated turn (yaw rate + forward speed).
 */
#include "MEKFScenarioExample.h"

#include <gtsam/navigation/Scenario.h>

using namespace gtsam;
using namespace mekf::examples;

int main(int argc, char* argv[]) {
  constexpr double kYawRate = 0.3;  // rad/s about body z
  constexpr double kSpeed = 2.0;    // m/s along body x
  try {
    const RunOptions opts =
        parseRunOptions(argc, argv, "mekf_coordinated_turn.csv");
    // Constant twist: yaw rate + forward velocity => circle of radius v/omega.
    ConstantTwistScenario scenario(Vector3(0, 0, kYawRate),
                                   Vector3(kSpeed, 0, 0));
    runScenario(scenario, opts, "Coordinated-turn");
  } catch (const std::exception& e) {
    std::cerr << "MEKFCoordinatedTurnExample failed: " << e.what() << '\n';
    return 2;
  }
  return 0;
}
