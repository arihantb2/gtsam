/**
 * @file TGEqFCircleExample.cpp
 * @brief TG-EqF IMU-only propagation on a constant-twist circular path.
 */

#include "TGEqFScenarioExample.h"

#include <gtsam/navigation/Scenario.h>

#include <cmath>
#include <iostream>

using namespace gtsam;
using namespace tgeqf::examples;

int main(int argc, char* argv[]) {
  constexpr double kRadius = 10.0;  // m
  constexpr double kSpeed = 1.0;      // m/s
  const double omega = kSpeed / kRadius;

  try {
    const RunOptions opts = parseRunOptions(argc, argv, "tg_eqf_circle.csv");

    ConstantTwistScenario scenario(Vector3(0, 0, -omega),
                                   Vector3(kRadius * omega, 0, 0));

    runScenario(scenario, opts, "Circle");
  } catch (const std::exception& e) {
    std::cerr << "TGEqFCircleExample failed: " << e.what() << '\n';
    return 2;
  }

  return 0;
}
