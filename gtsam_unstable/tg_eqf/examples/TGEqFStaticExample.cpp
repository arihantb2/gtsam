/**
 * @file TGEqFStaticExample.cpp
 * @brief TG-EqF IMU-only propagation at rest (zero excitation).
 *
 * Static baseline: no rotation, no translation. The IMU senses only gravity
 * (and, when enabled, bias + noise). Isolates bias/gravity handling and gives
 * the cleanest covariance-growth and NEES baseline.
 */

#include "TGEqFScenarioExample.h"

#include <gtsam/navigation/Scenario.h>

#include <iostream>

using namespace gtsam;
using namespace tgeqf::examples;

int main(int argc, char* argv[]) {
  try {
    const RunOptions opts = parseRunOptions(argc, argv, "tg_eqf_static.csv");

    // Zero twist: omega_b = 0, v_b = 0 -> the body stays at the origin pose.
    ConstantTwistScenario scenario(Vector3::Zero(), Vector3::Zero());

    runScenario(scenario, opts, "Static");
  } catch (const std::exception& e) {
    std::cerr << "TGEqFStaticExample failed: " << e.what() << '\n';
    return 2;
  }

  return 0;
}
