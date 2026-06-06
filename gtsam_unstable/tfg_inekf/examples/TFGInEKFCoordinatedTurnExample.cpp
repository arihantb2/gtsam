/**
 * @file  TFGInEKFCoordinatedTurnExample.cpp
 * @brief TFG-IEKF IMU-only, coordinated turn (constant-rate circle).
 *
 * Constant angular velocity about body z plus constant forward body velocity:
 * a circular path. Excites both rotation and centripetal acceleration, so the
 * predict step sees a fully coupled SE_2(3) increment each step.
 */
#include "TFGInEKFScenarioExample.h"

#include <gtsam/navigation/Scenario.h>

using namespace gtsam;
using namespace tfg::examples;

int main(int argc, char* argv[]) {
  constexpr double kYawRate = 0.3;  // rad/s about body z
  constexpr double kSpeed = 2.0;    // m/s along body x
  try {
    const RunOptions opts =
        parseRunOptions(argc, argv, "tfg_inekf_coordinated_turn.csv");
    // Constant twist: yaw rate + forward velocity => circle of radius v/omega.
    ConstantTwistScenario scenario(Vector3(0, 0, kYawRate),
                                   Vector3(kSpeed, 0, 0));
    runScenario(scenario, opts, "Coordinated-turn");
  } catch (const std::exception& e) {
    std::cerr << "TFGInEKFCoordinatedTurnExample failed: " << e.what() << '\n';
    return 2;
  }
  return 0;
}
