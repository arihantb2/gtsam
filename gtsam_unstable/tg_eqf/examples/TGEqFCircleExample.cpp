/**
 * @file TGEqFCircleExample.cpp
 * @brief TG-EqF IMU-only propagation on a constant-twist circular path.
 */
#include <gtsam_unstable/examples_common/IMUScenarios.h>
#include <gtsam_unstable/examples_common/ScenarioMain.h>

#include "TGEqFScenarioExample.h"

int main(int argc, char* argv[]) {
  return imu_scenarios::exampleMain(
      argc, argv, "TGEqFCircleExample", "tg_eqf_circle.csv", "Circle",
      imu_scenarios::circle, tgeqf::examples::runScenario);
}
