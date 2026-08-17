/**
 * @file TGEqFPureRotationExample.cpp
 * @brief TG-EqF pure rotation in place; attitude/gyro-bias without translation.
 */
#include <gtsam_unstable/examples_common/IMUScenarios.h>
#include <gtsam_unstable/examples_common/ScenarioMain.h>

#include "TGEqFScenarioExample.h"

int main(int argc, char* argv[]) {
  return imu_scenarios::exampleMain(argc, argv, "TGEqFPureRotationExample",
                                    "tg_eqf_pure_rotation.csv", "Pure-rotation",
                                    imu_scenarios::pureRotation,
                                    gtsam::tgeqf::examples::runScenario);
}
