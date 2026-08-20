#include <gtsam/geometry/Rot3.h>
#include <gtsam_unstable/tg_eqf/VirtualBiasOutput.h>

namespace gtsam {
namespace tgeqf {

Eigen::Vector3d VirtualBiasMeasurement::predict(const State& xi) {
  return xi.b_v;
}

Eigen::Matrix<double, 3, 18> VirtualBiasMeasurement::jacobian_Cstar(
    const State& xi_ref, const TGElement& g) {
  // The b_v row of Ad_{A_X^{-1}}, placed in the b_w and b_v columns.
  const Eigen::Matrix3d Rt = g.R.matrix().transpose();

  Eigen::Matrix<double, 3, 9> M = Eigen::Matrix<double, 3, 9>::Zero();
  M.block<3, 3>(0, 0) = -Rt * gtsam::skewSymmetric(g.p);
  M.block<3, 3>(0, 6) = Rt;

  const se2_3 b_ref = {xi_ref.b_w, xi_ref.b_a, xi_ref.b_v};

  Eigen::Matrix<double, 3, 18> Cstar = Eigen::Matrix<double, 3, 18>::Zero();
  Cstar.block<3, 9>(0, 0) = M * detail::ad_se23(b_ref);
  Cstar.block<3, 9>(0, 9) = -M;
  return Cstar;
}

}  // namespace tgeqf
}  // namespace gtsam
