#include <gtsam/geometry/Rot3.h>
#include <gtsam_unstable/tg_eqf/VirtualBiasOutput.h>

namespace gtsam {
namespace tgeqf {

Eigen::Vector3d VirtualBiasMeasurement::predict(const State& xi) {
  return xi.b_v;
}

Eigen::Matrix<double, 3, 18> VirtualBiasMeasurement::jacobian_Cstar(
    const TGElement& g) {
  // The b_v row of Ad_{A_X^{-1}}, placed in the b_w and b_v columns of the
  // [R, v, p, b_w, b_a, b_v] error chart.
  const Eigen::Matrix3d Rt = g.R.matrix().transpose();

  Eigen::Matrix<double, 3, 18> Cstar = Eigen::Matrix<double, 3, 18>::Zero();
  Cstar.block<3, 3>(0, 9) = -Rt * gtsam::skewSymmetric(g.p);
  Cstar.block<3, 3>(0, 15) = Rt;
  return Cstar;
}

Eigen::Vector3d VirtualBiasMeasurement::innovation(const State& xi_hat) {
  // Constraint target z = 0; residual is -h(xi_hat).
  return -predict(xi_hat);
}

}  // namespace tgeqf
}  // namespace gtsam
