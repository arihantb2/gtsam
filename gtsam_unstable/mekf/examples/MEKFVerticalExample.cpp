/**
 * @file  MEKFVerticalExample.cpp
 * @brief Baseline MEKF, IMU-only, vertical acceleration.
 */
#include "MEKFScenarioExample.h"

#include <gtsam/navigation/Scenario.h>

using namespace gtsam;
using namespace mekf::examples;

int main(int argc, char* argv[]) {
  constexpr double kAccel = 0.2;   // m/s^2 along nav z
  constexpr double kSpeed0 = 0.0;  // m/s initial vertical speed
  try {
    const RunOptions opts = parseRunOptions(argc, argv, "mekf_vertical.csv");
    const Vector3 v0(0.0, 0.0, kSpeed0);
    const Vector3 a_n(0.0, 0.0, kAccel);
    AcceleratingScenario scenario(Rot3(), Point3(0, 0, 0), v0, a_n,
                                  Vector3::Zero());
    runScenario(scenario, opts, "Vertical");
  } catch (const std::exception& e) {
    std::cerr << "MEKFVerticalExample failed: " << e.what() << '\n';
    return 2;
  }
  return 0;
}
