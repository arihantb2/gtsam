#include <gtsam/geometry/Rot3.h>
#include <gtsam_unstable/tg_eqf/PositionOutput.h>

namespace gtsam {
namespace tgeqf {

Vector3 PositionMeasurement::predict(const State& xi,
                                             const Vector3& pi) {
  return xi.R.unrotate(pi - xi.p);
}

Vector3 PositionMeasurement::predictAtOrigin(
    const State& xi_ref, const TGElement& g, const Vector3& pi) {
  const Vector3 y0 = xi_ref.R.unrotate(pi - xi_ref.p);
  return y0 - g.first.x(1);
}

Eigen::Matrix<double, 3, 18> PositionMeasurement::jacobian_C0(
    const State& xi_ref, const Vector3& pi) {
  const Vector3 y0 = xi_ref.R.unrotate(pi - xi_ref.p);
  Eigen::Matrix<double, 3, 18> C0 = Eigen::Matrix<double, 3, 18>::Zero();
  C0.block<3, 3>(0, 0) = gtsam::skewSymmetric(y0);
  C0.block<3, 3>(0, 6) = -Eigen::Matrix3d::Identity();
  return C0;
}

Eigen::Matrix<double, 3, 18> PositionMeasurement::jacobian_Cstar(
    const State& xi_ref, const TGElement& g, const Vector3& pi) {
  const Vector3 y0 = xi_ref.R.unrotate(pi - xi_ref.p);
  const Vector3 vec = y0 + g.first.x(1);

  Eigen::Matrix<double, 3, 18> Cstar = Eigen::Matrix<double, 3, 18>::Zero();
  Cstar.block<3, 3>(0, 0) = 0.5 * gtsam::skewSymmetric(vec);
  Cstar.block<3, 3>(0, 6) = -Eigen::Matrix3d::Identity();
  return Cstar;
}

Vector3 PositionMeasurement::output_action(const TGElement& X,
                                                   const Vector3& y) {
  return X.first.rotation().unrotate(y - X.first.x(1));
}

}  // namespace tgeqf
}  // namespace gtsam
