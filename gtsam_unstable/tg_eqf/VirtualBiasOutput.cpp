#include <gtsam_unstable/tg_eqf/VirtualBiasOutput.h>

#include <gtsam/geometry/Rot3.h>

namespace tgeqf {

Eigen::Vector3d VirtualBiasMeasurement::predict(const TGState& xi) {
  return xi.b_v;
}

Eigen::Matrix<double, 3, 18> VirtualBiasMeasurement::jacobian_C0(
    const TGState& /*xi_hat*/) {
  // Output Jacobian of h(xi) = b_v in the State::Retract chart at the current
  // estimate (same chart as the DVL/position output Jacobians, verified there
  // against finite differences). Since b_v is a plain additive coordinate, the
  // Jacobian is just +I in its own block:
  //   C0 = [0 0 0 0 0 I_3]  in the [R,v,p,b_w,b_a,b_v] tangent.
  //
  // This is NOT the literal Eq. (B.20) matrix. The filter consumes the
  // origin-chart Jacobian, which TGEqF::update_virtual_bias forms by composing
  // this C0 with the diffeomorphism transport Dphi_g (chain rule). That product
  // [0...I3]*Dphi_g equals the R_hat,p_hat-dependent matrix of Eq. (B.20) — the
  // R_hat,p_hat dependence there is genuine transport, not a chart relabeling
  // (CODE_REVIEW F1).
  Eigen::Matrix<double, 3, 18> C0 = Eigen::Matrix<double, 3, 18>::Zero();
  C0.block<3, 3>(0, 15) = Eigen::Matrix3d::Identity();
  return C0;
}

Eigen::Vector3d VirtualBiasMeasurement::innovation(const TGState& xi_hat) {
  // Constraint target z = 0; residual is -h(xi_hat).
  return -predict(xi_hat);
}

}  // namespace tgeqf
