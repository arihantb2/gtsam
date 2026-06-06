#include <gtsam_unstable/tfg_inekf/PositionOutput.h>

#include <gtsam/base/Matrix.h>

// ============================================================================
//  Direct global position output (Sec. 7, Lemma 15).
//
//  Body-frame residual model (Eq. 34):  h(xi) = R^T (pi - p)
//  Output action (Eq. 35):              rho_X(y) = R_X^T (y - p_X)
//  Equivariance (verified):  h(xi * X) = rho_X(h(xi)).
//
//  Output Jacobian for the chart xi(eps) = xi_hat * Expmap(eps):
//    C0    = [ y_hat^             0  -I  0 ],   y_hat = R_hat^T (pi - p_hat)
//    Cstar = [ 1/2 (y_hat+p_hat)^ 0  -I  0 ]    (midpoint-symmetrised, Eq. B.35)
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
  // d_theta coupling: y_hat^ (C0) or the midpoint 1/2 (y_hat + p_hat)^ (Cstar).
  const Eigen::Vector3d coupling =
      (variant == Variant::Cstar) ? 0.5 * (y_hat + xi_hat.p) : y_hat;

  Eigen::Matrix<double, 3, 15> C = Eigen::Matrix<double, 3, 15>::Zero();
  C.block<3, 3>(0, 0) = gtsam::skewSymmetric(coupling);  // d_theta
  C.block<3, 3>(0, 6) = -Eigen::Matrix3d::Identity();    // d_p
  return C;
}

Eigen::Vector3d PositionOutput::innovation(const TwoFrameGroup& xi_hat,
                                           const Eigen::Vector3d& pi) {
  return -predict(xi_hat, pi);  // z - h(xi_hat),  z = 0
}

}  // namespace tfg
