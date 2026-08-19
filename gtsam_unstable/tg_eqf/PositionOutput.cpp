#include <gtsam/geometry/Rot3.h>
#include <gtsam_unstable/tg_eqf/PositionOutput.h>

namespace gtsam {
namespace tgeqf {

Eigen::Vector3d PositionMeasurement::predict(const State& xi,
                                             const Eigen::Vector3d& pi) {
  return xi.R.unrotate(pi - xi.p);
}

Eigen::Matrix<double, 3, 18> PositionMeasurement::jacobian_C0(
    const State& xi_ref, const Eigen::Vector3d& pi) {
  const Eigen::Vector3d y0 = xi_ref.R.unrotate(pi - xi_ref.p);
  Eigen::Matrix<double, 3, 18> C0 = Eigen::Matrix<double, 3, 18>::Zero();
  C0.block<3, 3>(0, 0) = gtsam::skewSymmetric(y0);
  C0.block<3, 3>(0, 6) = -Eigen::Matrix3d::Identity();
  return C0;
}

Eigen::Matrix<double, 3, 18> PositionMeasurement::jacobian_Cstar(
    const State& xi_ref, const TGElement& g, const Eigen::Vector3d& pi) {
  const Eigen::Vector3d y0 = xi_ref.R.unrotate(pi - xi_ref.p);
  const Eigen::Vector3d vec = y0 + g.p;

  Eigen::Matrix<double, 3, 18> Cstar = Eigen::Matrix<double, 3, 18>::Zero();
  Cstar.block<3, 3>(0, 0) = 0.5 * gtsam::skewSymmetric(vec);
  Cstar.block<3, 3>(0, 6) = -Eigen::Matrix3d::Identity();
  return Cstar;
}

Eigen::Vector3d PositionMeasurement::output_action(const TGElement& X,
                                                   const Eigen::Vector3d& y) {
  return X.R.unrotate(y - X.p);
}

}  // namespace tgeqf
}  // namespace gtsam
