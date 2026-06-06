/**
 * @file TGEqFVerticalExample.cpp
 * @brief TG-EqF IMU-only propagation on a vertical (gravity-axis) trajectory.
 *
 * Constant acceleration along the navigation z-axis, no rotation. Stresses
 * accel-bias vs gravity separability along the gravity direction -- the axis
 * most relevant to the depth-aided INS design target.
 */

#include "TGEqFScenarioExample.h"

#include <gtsam/navigation/Scenario.h>

#include <iostream>

using namespace gtsam;
using namespace tgeqf::examples;

int main(int argc, char* argv[]) {
  constexpr double kAccel = 0.2;   // m/s^2 along nav z
  constexpr double kSpeed0 = 0.0;  // m/s initial vertical speed

  try {
    const RunOptions opts = parseRunOptions(argc, argv, "tg_eqf_vertical.csv");

    const Vector3 v0(0.0, 0.0, kSpeed0);
    const Vector3 a_n(0.0, 0.0, kAccel);  // pure vertical acceleration

    AcceleratingScenario scenario(Rot3(), Point3(0, 0, 0), v0, a_n,
                                  Vector3::Zero());

    runScenario(scenario, opts, "Vertical");
  } catch (const std::exception& e) {
    std::cerr << "TGEqFVerticalExample failed: " << e.what() << '\n';
    return 2;
  }

  return 0;
}
