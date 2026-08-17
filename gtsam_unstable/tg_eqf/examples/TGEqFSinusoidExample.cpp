/**
 * @file TGEqFSinusoidExample.cpp
 * @brief TG-EqF weaving trajectory; time-varying specific force NEES stressor.
 */
#include <gtsam_unstable/examples_common/IMUScenarios.h>
#include <gtsam_unstable/examples_common/ScenarioMain.h>

#include "TGEqFScenarioExample.h"

int main(int argc, char* argv[]) {
  return imu_scenarios::exampleMain(
      argc, argv, "TGEqFSinusoidExample", "tg_eqf_sinusoid.csv", "Sinusoid",
      imu_scenarios::sinusoid, gtsam::tgeqf::examples::runScenario);
}
