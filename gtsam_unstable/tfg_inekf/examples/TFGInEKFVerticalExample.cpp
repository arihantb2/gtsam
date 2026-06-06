/**
 * @file  TFGInEKFVerticalExample.cpp
 * @brief TFG-IEKF IMU-only, vertical constant-acceleration climb.
 *
 * Pure vertical acceleration (nav z), zero rotation: the accelerometer reads
 * gravity plus the climb acceleration along z. Exercises the gravity-coupled
 * velocity/position integration in the vertical channel.
 */
#include "TFGInEKFScenarioExample.h"

#include <gtsam/navigation/Scenario.h>

using namespace gtsam;
using namespace tfg::examples;

int main(int argc, char* argv[]) {
  constexpr double kAccel = 0.2;   // m/s^2 along nav z
  constexpr double kSpeed0 = 0.0;  // m/s initial vertical speed
  try {
    const RunOptions opts =
        parseRunOptions(argc, argv, "tfg_inekf_vertical.csv");
    const Vector3 v0(0.0, 0.0, kSpeed0);
    const Vector3 a_n(0.0, 0.0, kAccel);
    AcceleratingScenario scenario(Rot3(), Point3(0, 0, 0), v0, a_n,
                                  Vector3::Zero());
    runScenario(scenario, opts, "Vertical");
  } catch (const std::exception& e) {
    std::cerr << "TFGInEKFVerticalExample failed: " << e.what() << '\n';
    return 2;
  }
  return 0;
}
