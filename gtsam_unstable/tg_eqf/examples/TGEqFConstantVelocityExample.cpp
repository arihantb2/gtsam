/**
 * @file TGEqFConstantVelocityExample.cpp
 * @brief TG-EqF constant linear velocity; moving body, gravity-only specific
 * force.
 */
#include <gtsam_unstable/examples_common/IMUScenarios.h>
#include <gtsam_unstable/examples_common/ScenarioMain.h>

#include "TGEqFScenarioExample.h"

int main(int argc, char* argv[]) {
  return imu_scenarios::exampleMain(
      argc, argv, "TGEqFConstantVelocityExample",
      "tg_eqf_constant_velocity.csv", "Constant-velocity",
      imu_scenarios::constantVelocity, tgeqf::examples::runScenario);
}
