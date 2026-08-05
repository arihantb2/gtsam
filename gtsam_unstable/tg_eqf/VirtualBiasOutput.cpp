#include <gtsam/geometry/Rot3.h>
#include <gtsam_unstable/tg_eqf/VirtualBiasOutput.h>

namespace tgeqf {

Eigen::Vector3d VirtualBiasMeasurement::predict(const State& xi) {
  return xi.b_v;
}

Eigen::Matrix<double, 3, 18> VirtualBiasMeasurement::jacobian_C0(
    const State& /*xi_hat*/) {
  // b_v is a plain additive coordinate in the [R,v,p,b_w,b_a,b_v] Retract
  // chart, so C0 = [0 0 0 0 0 I_3]. The filter consumes the origin-chart
  // Jacobian, which update_virtual_bias forms by composing this with the
  // transport Dphi_g; that product equals the literal Eq. (B.20) matrix.
  Eigen::Matrix<double, 3, 18> C0 = Eigen::Matrix<double, 3, 18>::Zero();
  C0.block<3, 3>(0, 15) = Eigen::Matrix3d::Identity();
  return C0;
}

Eigen::Vector3d VirtualBiasMeasurement::innovation(const State& xi_hat) {
  // Constraint target z = 0; residual is -h(xi_hat).
  return -predict(xi_hat);
}

}  // namespace tgeqf
