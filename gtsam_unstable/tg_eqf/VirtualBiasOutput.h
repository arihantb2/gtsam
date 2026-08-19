#pragma once
#include <gtsam_unstable/tg_eqf/Group.h>
#include <gtsam_unstable/tg_eqf/State.h>

#include <Eigen/Dense>

namespace gtsam {
namespace tgeqf {

/**
 * Virtual-velocity-bias pseudo-measurement h(xi) = b_v = 0.
 */
struct VirtualBiasMeasurement {
  static Eigen::Vector3d predict(const State& xi);

  /**
   * Output matrix C* in R^{3x18}, in error coordinates, Equ. (B.20)
   *
   *   C* = [ 0  0  0  -R_X^T p_X^  0  R_X^T ]
   *
   * The bias block of the state action is b = Ad_{A_X^{-1}}(b_ref + eps_b -
   * a_X), so b_v depends on the error only through the b_w and b_v blocks, and
   * does so affinely. This matrix is therefore exact at any error magnitude:
   * unlike the position and DVL outputs there is no linearization to improve
   * on, and no C0/C* distinction to make.
   */
  static Eigen::Matrix<double, 3, 18> jacobian_Cstar(const TGElement& g);
};

}  // namespace tgeqf
}  // namespace gtsam
