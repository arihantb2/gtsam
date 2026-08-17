/**
 * @file TGEqFCoordinatedTurnExample.cpp
 * @brief TG-EqF climbing turn; couples yaw rate with linear acceleration.
 */
#include <gtsam_unstable/examples_common/IMUScenarios.h>
#include <gtsam_unstable/examples_common/ScenarioMain.h>

#include "TGEqFScenarioExample.h"

int main(int argc, char* argv[]) {
  return imu_scenarios::exampleMain(
      argc, argv, "TGEqFCoordinatedTurnExample", "tg_eqf_coordinated_turn.csv",
      "Coordinated-turn", imu_scenarios::coordinatedTurn,
      gtsam::tgeqf::examples::runScenario);
}
