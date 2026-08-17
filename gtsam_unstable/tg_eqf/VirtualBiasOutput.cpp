#include <gtsam/geometry/Rot3.h>
#include <gtsam_unstable/tg_eqf/VirtualBiasOutput.h>

namespace gtsam {
namespace tgeqf {

Eigen::Vector3d VirtualBiasMeasurement::predict(const State& xi) {
  return xi.b_v;
}

Eigen::Matrix<double, 3, 18> VirtualBiasMeasurement::stateJacobian(
    const State& /*xi_hat*/) {
  // b_v is a plain additive coordinate in the [R,v,p,b_w,b_a,b_v] Retract
  // chart, so the state-chart Jacobian is [0 0 0 0 0 I_3] wherever it is
  // evaluated. update_virtual_bias transports it to error coordinates, and that
  // product equals the literal Eq. (B.20) matrix.
  Eigen::Matrix<double, 3, 18> H = Eigen::Matrix<double, 3, 18>::Zero();
  H.block<3, 3>(0, 15) = Eigen::Matrix3d::Identity();
  return H;
}

Eigen::Vector3d VirtualBiasMeasurement::innovation(const State& xi_hat) {
  // Constraint target z = 0; residual is -h(xi_hat).
  return -predict(xi_hat);
}

}  // namespace tgeqf
}  // namespace gtsam
