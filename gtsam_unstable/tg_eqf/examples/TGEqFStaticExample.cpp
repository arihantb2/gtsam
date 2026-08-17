/**
 * @file TGEqFStaticExample.cpp
 * @brief TG-EqF IMU-only propagation at rest. Cleanest NEES/covariance
 * baseline.
 */
#include <gtsam_unstable/examples_common/IMUScenarios.h>
#include <gtsam_unstable/examples_common/ScenarioMain.h>

#include "TGEqFScenarioExample.h"

int main(int argc, char* argv[]) {
  return imu_scenarios::exampleMain(
      argc, argv, "TGEqFStaticExample", "tg_eqf_static.csv", "Static",
      imu_scenarios::staticPose, gtsam::tgeqf::examples::runScenario);
}
