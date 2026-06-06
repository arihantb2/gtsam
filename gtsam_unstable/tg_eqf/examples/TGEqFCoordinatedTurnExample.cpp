/**
 * @file TGEqFCoordinatedTurnExample.cpp
 * @brief TG-EqF IMU-only propagation on a climbing, turning trajectory.
 *
 * Combines a constant body yaw rate with a constant navigation-frame
 * acceleration (horizontal + vertical components). This is the cross-coupling
 * case the Circle (rotation, no linear acceleration) and Straight-line
 * (acceleration, no rotation) scenarios miss: it exercises the interaction of
 * attitude rate with specific force, where EqF and EKF behaviour diverge most.
 *
 * Note: a_n is constant in the navigation frame (not a perfect centripetal
 * circle); the goal is rich rotation x specific-force excitation, not a closed
 * path.
 */

#include "TGEqFScenarioExample.h"

#include <gtsam/navigation/Scenario.h>

#include <iostream>

using namespace gtsam;
using namespace tgeqf::examples;

int main(int argc, char* argv[]) {
  constexpr double kYawRate = 0.3;  // rad/s about body z
  constexpr double kSpeed0 = 1.0;     // m/s initial forward speed

  try {
    const RunOptions opts =
        parseRunOptions(argc, argv, "tg_eqf_coordinated_turn.csv");

    const Vector3 v0(kSpeed0, 0.0, 0.0);      // initial nav-frame velocity
    const Vector3 a_n(0.0, 0.5, 0.2);          // lateral + climb acceleration
    const Vector3 omega_b(0.0, 0.0, kYawRate); // body yaw rate

    AcceleratingScenario scenario(Rot3(), Point3(0, 0, 0), v0, a_n, omega_b);

    runScenario(scenario, opts, "Coordinated-turn");
  } catch (const std::exception& e) {
    std::cerr << "TGEqFCoordinatedTurnExample failed: " << e.what() << '\n';
    return 2;
  }

  return 0;
}
