#include <gtsam_unstable/examples_common/IMUScenarios.h>
#include <gtsam_unstable/examples_common/ScenarioMain.h>

#include "MEKFScenarioExample.h"

int main(int argc, char* argv[]) {
  return imu_scenarios::exampleMain(
      argc, argv, "MEKFCoordinatedTurnExample", "mekf_coordinated_turn.csv",
      "Coordinated-turn", imu_scenarios::coordinatedTurn,
      mekf::examples::runScenario);
}
