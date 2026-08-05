#include <gtsam_unstable/examples_common/IMUScenarios.h>
#include <gtsam_unstable/examples_common/ScenarioMain.h>

#include "MEKFScenarioExample.h"

int main(int argc, char* argv[]) {
  return imu_scenarios::exampleMain(
      argc, argv, "MEKFSinusoidExample", "mekf_sinusoid.csv", "Sinusoid",
      imu_scenarios::sinusoid, mekf::examples::runScenario);
}
