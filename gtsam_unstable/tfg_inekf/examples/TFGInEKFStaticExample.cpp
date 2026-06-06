/**
 * @file  TFGInEKFStaticExample.cpp
 * @brief TFG-IEKF IMU-only, stationary vehicle.
 *
 * Body at rest: zero angular velocity, zero linear velocity. Specific force is
 * gravity reaction only. With an ideal IMU the estimate must stay exactly at
 * the initial pose; with bias/noise it drifts, exercising the predict step in
 * isolation (no position updates).
 */
#include "TFGInEKFScenarioExample.h"

#include <gtsam/navigation/Scenario.h>

using namespace gtsam;
using namespace tfg::examples;

int main(int argc, char* argv[]) {
  try {
    const RunOptions opts = parseRunOptions(argc, argv, "tfg_inekf_static.csv");
    // Zero twist: no rotation, no translation.
    ConstantTwistScenario scenario(Vector3::Zero(), Vector3::Zero());
    runScenario(scenario, opts, "Static");
  } catch (const std::exception& e) {
    std::cerr << "TFGInEKFStaticExample failed: " << e.what() << '\n';
    return 2;
  }
  return 0;
}
