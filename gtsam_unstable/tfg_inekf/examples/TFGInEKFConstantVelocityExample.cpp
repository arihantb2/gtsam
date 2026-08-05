#include <gtsam_unstable/examples_common/IMUScenarios.h>
#include <gtsam_unstable/examples_common/ScenarioMain.h>

#include "TFGInEKFScenarioExample.h"

int main(int argc, char* argv[]) {
  return imu_scenarios::exampleMain(
      argc, argv, "TFGInEKFConstantVelocityExample",
      "tfg_inekf_constant_velocity.csv", "Constant-velocity",
      imu_scenarios::constantVelocity, tfg::examples::runScenario);
}
