#include <gtsam_unstable/examples_common/IMUScenarios.h>
#include <gtsam_unstable/examples_common/ScenarioMain.h>

#include "MEKFScenarioExample.h"

int main(int argc, char* argv[]) {
  return imu_scenarios::exampleMain(
      argc, argv, "MEKFCircleExample", "mekf_circle.csv", "Circle",
      imu_scenarios::circle, mekf::examples::runScenario);
}
