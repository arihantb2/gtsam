#include <gtsam_unstable/examples_common/IMUScenarios.h>
#include <gtsam_unstable/examples_common/ScenarioMain.h>

#include "MEKFScenarioExample.h"

int main(int argc, char* argv[]) {
  return imu_scenarios::exampleMain(
      argc, argv, "MEKFConstantVelocityExample", "mekf_constant_velocity.csv",
      "Constant-velocity", imu_scenarios::constantVelocity,
      mekf::examples::runScenario);
}
