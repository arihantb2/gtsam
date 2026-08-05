#include <gtsam_unstable/examples_common/IMUScenarios.h>
#include <gtsam_unstable/examples_common/ScenarioMain.h>

#include "TFGInEKFScenarioExample.h"

int main(int argc, char* argv[]) {
  return imu_scenarios::exampleMain(
      argc, argv, "TFGInEKFPureRotationExample", "tfg_inekf_pure_rotation.csv",
      "Pure-rotation", imu_scenarios::pureRotation, tfg::examples::runScenario);
}
