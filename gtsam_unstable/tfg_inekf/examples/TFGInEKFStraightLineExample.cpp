/**
 * @file  TFGInEKFStraightLineExample.cpp
 * @brief TFG-IEKF IMU-only, oblique straight-line constant acceleration.
 *
 * Constant acceleration along a fixed oblique direction, zero rotation: the
 * body speeds up along a straight 3-D line. Adds linear acceleration on top of
 * the constant-velocity case, exercising all three velocity/position channels.
 */
#include "TFGInEKFScenarioExample.h"

#include <gtsam/navigation/Scenario.h>

using namespace gtsam;
using namespace tfg::examples;

int main(int argc, char* argv[]) {
  constexpr double kAccel = 0.1;   // m/s^2
  constexpr double kSpeed0 = 0.5;  // m/s
  try {
    const RunOptions opts =
        parseRunOptions(argc, argv, "tfg_inekf_straight_line.csv");
    const Vector3 dir = Vector3(2.0, 1.0, 0.5).normalized();
    const Vector3 v0 = kSpeed0 * dir;  // oblique initial velocity
    const Vector3 a_n = kAccel * dir;  // acceleration along the same line
    AcceleratingScenario scenario(Rot3(), Point3(0, 0, 0), v0, a_n,
                                  Vector3::Zero());
    runScenario(scenario, opts, "Straight-line");
  } catch (const std::exception& e) {
    std::cerr << "TFGInEKFStraightLineExample failed: " << e.what() << '\n';
    return 2;
  }
  return 0;
}
