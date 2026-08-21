#include <gtsam/geometry/Rot3.h>
#include <gtsam_unstable/tg_eqf/DepthOutput.h>

namespace gtsam {
namespace tgeqf {

double DepthMeasurement::predict(const State& xi) { return xi.p.z(); }

Eigen::Matrix<double, 1, 18> DepthMeasurement::jacobian_C(
    const State& xi_ref, const TGElement& g) {
  const Eigen::Vector3d n = xi_ref.R.unrotate(Eigen::Vector3d::UnitZ());

  Eigen::Matrix<double, 1, 18> C = Eigen::Matrix<double, 1, 18>::Zero();
  C.block<1, 3>(0, 0) = g.p.cross(n).transpose();
  C.block<1, 3>(0, 6) = n.transpose();
  return C;
}

}  // namespace tgeqf
}  // namespace gtsam
