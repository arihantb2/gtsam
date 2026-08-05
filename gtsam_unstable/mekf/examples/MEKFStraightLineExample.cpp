#include <gtsam_unstable/examples_common/IMUScenarios.h>
#include <gtsam_unstable/examples_common/ScenarioMain.h>

#include "MEKFScenarioExample.h"

int main(int argc, char* argv[]) {
  return imu_scenarios::exampleMain(argc, argv, "MEKFStraightLineExample",
                                    "mekf_straight_line.csv", "Straight-line",
                                    imu_scenarios::straightLine,
                                    mekf::examples::runScenario);
}
