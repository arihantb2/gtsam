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
   * Output matrix C* in R^{3x18}, in error coordinates
   *
   *   C* = [ 0  0  0  -R_X^T p_X^  0  R_X^T ]
   *
   * The bias block of the state action is b = Ad_{A_X^{-1}}(b_ref + eps_b -
   * a_X), so b_v depends on the error affinely, through the b_w and b_v blocks
   * only. This matrix is therefore exact at any error magnitude. There is no
   * linearization to improve on, so this output has no C0/C* distinction.
   */
  static Eigen::Matrix<double, 3, 18> jacobian_Cstar(const TGElement& g);
};

}  // namespace tgeqf
}  // namespace gtsam
