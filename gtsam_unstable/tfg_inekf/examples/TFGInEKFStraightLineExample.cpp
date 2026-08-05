#include <gtsam_unstable/examples_common/IMUScenarios.h>
#include <gtsam_unstable/examples_common/ScenarioMain.h>

#include "TFGInEKFScenarioExample.h"

int main(int argc, char* argv[]) {
  return imu_scenarios::exampleMain(
      argc, argv, "TFGInEKFStraightLineExample", "tfg_inekf_straight_line.csv",
      "Straight-line", imu_scenarios::straightLine, tfg::examples::runScenario);
}
