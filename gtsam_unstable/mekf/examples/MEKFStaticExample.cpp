/**
 * @file  MEKFStaticExample.cpp
 * @brief Baseline MEKF, IMU-only, static (zero twist).
 */
#include "MEKFScenarioExample.h"

#include <gtsam/navigation/Scenario.h>

using namespace gtsam;
using namespace mekf::examples;

int main(int argc, char* argv[]) {
  try {
    const RunOptions opts = parseRunOptions(argc, argv, "mekf_static.csv");
    // Zero twist: no rotation, no translation.
    ConstantTwistScenario scenario(Vector3::Zero(), Vector3::Zero());
    runScenario(scenario, opts, "Static");
  } catch (const std::exception& e) {
    std::cerr << "MEKFStaticExample failed: " << e.what() << '\n';
    return 2;
  }
  return 0;
}
