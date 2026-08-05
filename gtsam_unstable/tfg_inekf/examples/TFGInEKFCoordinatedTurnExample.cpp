#include <gtsam_unstable/examples_common/IMUScenarios.h>
#include <gtsam_unstable/examples_common/ScenarioMain.h>

#include "TFGInEKFScenarioExample.h"

int main(int argc, char* argv[]) {
  return imu_scenarios::exampleMain(
      argc, argv, "TFGInEKFCoordinatedTurnExample",
      "tfg_inekf_coordinated_turn.csv", "Coordinated-turn",
      imu_scenarios::coordinatedTurn, tfg::examples::runScenario);
}
