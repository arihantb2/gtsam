#include <gtsam_unstable/tfg_inekf/PositionOutput.h>

#include <gtsam/base/Matrix.h>

// ============================================================================
//  Direct global position output (Sec. 7, Lemma 15).
//
//  Body-frame residual model (Eq. 34):  h(xi) = R^T (pi - p)
//  Output action (Eq. 35):              rho_X(y) = R_X^T (y - p_X)
//  Equivariance (verified):  h(xi * X) = rho_X(h(xi)).
//
//  Output Jacobian for the chart xi(eps) = xi_hat * Expmap(eps) (right retract):
//    C0    = [ y_hat^      0  -I  0 ],   y_hat = R_hat^T (pi - p_hat)
//    Cstar = [ 1/2 y_hat^  0  -I  0 ]    (midpoint-symmetrised)
//
//  Chart note: in this right-multiplicative chart the noise-free residual is
//  exactly y_hat = J_l(eps_theta) eps_p. Expanding J_l = I + 1/2 eps_theta^ +...
//  and substituting eps_p ~ y_hat collapses the attitude coupling y_hat^ to
//  1/2 y_hat^, giving cubic (O(||e||^3)) linearisation error. The paper's
//  Eq. (B.35), C* = [1/2 (pi + p_hat)^ 0 -I 0], is written in the paper's
//  global-frame EqF chart; transforming it into this body chart by
//  R_hat^T (.) Ad_{X_hat} yields exactly [1/2 y_hat^ 0 -I 0].
// ============================================================================

namespace tfg {

Eigen::Vector3d PositionOutput::predict(const TwoFrameGroup& xi,
                                        const Eigen::Vector3d& pi) {
  return xi.R.unrotate(pi - xi.p);  // R^T (pi - p)
}

Eigen::Vector3d PositionOutput::output_action(const TwoFrameGroup& X,
                                              const Eigen::Vector3d& y) {
  return X.R.unrotate(y - X.p);  // R_X^T (y - p_X)
}

Eigen::Matrix<double, 3, 15> PositionOutput::jacobian(
    const TwoFrameGroup& xi_hat, const Eigen::Vector3d& pi, Variant variant) {
  const Eigen::Vector3d y_hat = predict(xi_hat, pi);
  // d_theta coupling: y_hat^ (C0) or the midpoint-symmetrised 1/2 y_hat^ (C*).
  const Eigen::Vector3d coupling =
      (variant == Variant::Cstar) ? (0.5 * y_hat).eval() : y_hat;

  Eigen::Matrix<double, 3, 15> C = Eigen::Matrix<double, 3, 15>::Zero();
  C.block<3, 3>(0, 0) = gtsam::skewSymmetric(coupling);  // d_theta
  C.block<3, 3>(0, 6) = -Eigen::Matrix3d::Identity();    // d_p
  return C;
}

}  // namespace tfg
