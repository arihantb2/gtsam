#include <gtsam_unstable/examples_common/IMUScenarios.h>
#include <gtsam_unstable/examples_common/ScenarioMain.h>

#include "TFGInEKFScenarioExample.h"

int main(int argc, char* argv[]) {
  return imu_scenarios::exampleMain(
      argc, argv, "TFGInEKFSinusoidExample", "tfg_inekf_sinusoid.csv",
      "Sinusoid", imu_scenarios::sinusoid, tfg::examples::runScenario);
}
