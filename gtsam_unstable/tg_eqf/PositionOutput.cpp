#include <gtsam_unstable/tg_eqf/PositionOutput.h>

#include <gtsam/geometry/Rot3.h>

namespace tgeqf {

Eigen::Vector3d PositionMeasurement::predict(const TGState& xi,
                                              const Eigen::Vector3d& pi) {
  return xi.R.unrotate(pi - xi.p);
}

Eigen::Matrix<double, 3, 18> PositionMeasurement::jacobian_C0(
    const TGState& xi_ref, const Eigen::Vector3d& pi) {
  // Origin-chart first-order Jacobian of the equivariant position residual
  // (paper Sec. 2.2 / Eq. B.19 with the second argument at the origin output).
  // Tangent order [delta_R | delta_v | delta_p | delta_b]:
  //   C0 = [ y0^  0  -R0^T  0 ],   y0 = R0^T (pi - p0)   (origin output)
  // The y0^ rotation block is the limit of C* as the estimate reaches the
  // origin; the old [0|0|-I|0] dropped it (CODE_REVIEW F3).
  const Eigen::Vector3d y0 = xi_ref.R.unrotate(pi - xi_ref.p);
  Eigen::Matrix<double, 3, 18> C0 = Eigen::Matrix<double, 3, 18>::Zero();
  C0.block<3, 3>(0, 0) = gtsam::skewSymmetric(y0);
  C0.block<3, 3>(0, 6) = -xi_ref.R.matrix().transpose();
  return C0;
}

Eigen::Matrix<double, 3, 18> PositionMeasurement::jacobian_Cstar(
    const TGState& xi_ref, const TGGroupElement& g,
    const Eigen::Vector3d& pi) {
  // Paper-literal C* (Eq. B.19): average of D_E rho_E at the origin output y0
  // and at the back-transported measurement p_X = g.p (both global-frame).
  //   C* = [ 0.5 (y0 + p_X)^  0  -R0^T  0 ]
  // with y0 = R0^T (pi - p0). At identity origin this is
  //   [ 0.5 (pi + p_hat)^ | 0 | -I | 0 ];
  // at convergence (pi -> p_hat, y0 -> p_X) the rotation block tends to p_X^,
  // the consistent first-order value (the old 0.5 p_hat^ was half of it).
  const Eigen::Vector3d y0 = xi_ref.R.unrotate(pi - xi_ref.p);
  const Eigen::Vector3d vec = y0 + g.p;

  Eigen::Matrix<double, 3, 18> Cstar = Eigen::Matrix<double, 3, 18>::Zero();
  Cstar.block<3, 3>(0, 0) = 0.5 * gtsam::skewSymmetric(vec);
  Cstar.block<3, 3>(0, 6) = -xi_ref.R.matrix().transpose();
  return Cstar;
}

Eigen::Vector3d PositionMeasurement::innovation(
    const Eigen::Vector3d& pi, const TGState& xi_hat) {
  // Equivariant target is zero when pi == p; residual is -h'(xi_hat).
  return -predict(xi_hat, pi);
}

Eigen::Vector3d PositionMeasurement::output_action(
    const TGGroupElement& X, const Eigen::Vector3d& y) {
  return X.R.unrotate(y - X.p);
}

}  // namespace tgeqf
