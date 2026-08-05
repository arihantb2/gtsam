/**
 * @file TGEqFStraightLineExample.cpp
 * @brief TG-EqF constant-acceleration straight line along an oblique 3D
 * direction.
 */
#include <gtsam_unstable/examples_common/IMUScenarios.h>
#include <gtsam_unstable/examples_common/ScenarioMain.h>

#include "TGEqFScenarioExample.h"

int main(int argc, char* argv[]) {
  return imu_scenarios::exampleMain(argc, argv, "TGEqFStraightLineExample",
                                    "tg_eqf_straight_line.csv", "Straight-line",
                                    imu_scenarios::straightLine,
                                    tgeqf::examples::runScenario);
}
