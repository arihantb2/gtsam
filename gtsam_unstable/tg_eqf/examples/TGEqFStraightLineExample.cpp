/**
 * @file TGEqFStraightLineExample.cpp
 * @brief TG-EqF IMU-only propagation on a constant-acceleration straight-line path.
 */

#include "TGEqFScenarioExample.h"

#include <gtsam/navigation/Scenario.h>

#include <cmath>
#include <iostream>

using namespace gtsam;
using namespace tgeqf::examples;

int main(int argc, char* argv[]) {
  // Non-trivial straight line: motion along an oblique 3D direction with a
  // non-zero initial velocity. Velocity and acceleration are kept parallel so
  // the path stays straight, while exercising all three axes (and the filter
  // initialization from a non-rest start state).
  constexpr double kAccel = 0.1;  // m/s^2
  constexpr double kSpeed0 = 0.5;  // m/s

  try {
    const RunOptions opts =
        parseRunOptions(argc, argv, "tg_eqf_straight_line.csv");

    const Vector3 dir = Vector3(2.0, 1.0, 0.5).normalized();
    const Vector3 v0 = kSpeed0 * dir;     // oblique initial velocity
    const Vector3 a_n = kAccel * dir;     // acceleration along the same line

    AcceleratingScenario scenario(Rot3(), Point3(0, 0, 0), v0, a_n,
                                  Vector3::Zero());

    runScenario(scenario, opts, "Straight-line");
  } catch (const std::exception& e) {
    std::cerr << "TGEqFStraightLineExample failed: " << e.what() << '\n';
    return 2;
  }

  return 0;
}
