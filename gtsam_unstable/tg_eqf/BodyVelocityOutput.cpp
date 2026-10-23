#include <gtsam/geometry/Rot3.h>
#include <gtsam_unstable/tg_eqf/BodyVelocityOutput.h>

namespace gtsam {
namespace tgeqf {

Vector3 DVLMeasurement::predict(const State& xi) {
  return xi.R.unrotate(xi.v);
}

Eigen::Matrix<double, 3, 18> DVLMeasurement::jacobian_C0(const State& xi_ref) {
  const Vector3 y0 = predict(xi_ref);
  Eigen::Matrix<double, 3, 18> C0 = Eigen::Matrix<double, 3, 18>::Zero();
  C0.block<3, 3>(0, 0) = gtsam::skewSymmetric(y0);
  C0.block<3, 3>(0, 3) = Eigen::Matrix3d::Identity();
  return C0;
}

Eigen::Matrix<double, 3, 18> DVLMeasurement::jacobian_Cstar(
    const State& xi_ref, const TGElement& g, const Vector3& z_dvl) {
  const Vector3 y0 = predict(xi_ref);
  const Vector3 vec = y0 + inverse_output_action(g, z_dvl);

  Eigen::Matrix<double, 3, 18> Cstar = Eigen::Matrix<double, 3, 18>::Zero();
  Cstar.block<3, 3>(0, 0) = 0.5 * gtsam::skewSymmetric(vec);
  Cstar.block<3, 3>(0, 3) = Eigen::Matrix3d::Identity();
  return Cstar;
}

Vector3 DVLMeasurement::output_action(const TGElement& X,
                                              const Vector3& y) {
  return X.first.rotation().unrotate(y + X.first.x(0));
}

Vector3 DVLMeasurement::inverse_output_action(
    const TGElement& X, const Vector3& y) {
  return X.first.rotation().rotate(y) - X.first.x(0);
}

}  // namespace tgeqf
}  // namespace gtsam
