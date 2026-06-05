/**
 * @file testTGEqFExample.cpp
 * @brief Smoke compile/link check for the TG-EqF scaffold.
 *
 * Builds an executable that includes the EqF headers and links against
 * gtsam_unstable. No filter construction or propagation until implementations
 * are added.
 */

#include <gtsam_unstable/tg_eqf/EqF.h>

int main() {
  (void)sizeof(tgeqf::TGEqF);
  return 0;
}
