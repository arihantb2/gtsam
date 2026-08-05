#include <gtsam_unstable/examples_common/IMUScenarios.h>
#include <gtsam_unstable/examples_common/ScenarioMain.h>

#include "MEKFScenarioExample.h"

int main(int argc, char* argv[]) {
  return imu_scenarios::exampleMain(argc, argv, "MEKFPureRotationExample",
                                    "mekf_pure_rotation.csv", "Pure-rotation",
                                    imu_scenarios::pureRotation,
                                    mekf::examples::runScenario);
}
