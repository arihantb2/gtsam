#include <gtsam_unstable/tg_eqf/DepthOutput.h>

#include <stdexcept>

namespace tgeqf {

double PressureMeasurement::depth_from_pressure(double, double, double, double) {
  throw std::runtime_error(
      "tgeqf::PressureMeasurement::depth_from_pressure not implemented");
}

Eigen::Vector3d PressureMeasurement::pseudo_position(const TGState&, double) {
  throw std::runtime_error(
      "tgeqf::PressureMeasurement::pseudo_position not implemented");
}

Eigen::Vector3d PressureMeasurement::predict(const TGState&,
                                             const Eigen::Vector3d&) {
  throw std::runtime_error("tgeqf::PressureMeasurement::predict not implemented");
}

Eigen::Matrix<double, 3, 18> PressureMeasurement::jacobian_C0(const TGState&) {
  throw std::runtime_error(
      "tgeqf::PressureMeasurement::jacobian_C0 not implemented");
}

Eigen::Matrix<double, 3, 18> PressureMeasurement::jacobian_Cstar(
    const TGState&, const Eigen::Vector3d&) {
  throw std::runtime_error(
      "tgeqf::PressureMeasurement::jacobian_Cstar not implemented");
}

Eigen::Matrix3d PressureMeasurement::noise_covariance(double, double) {
  throw std::runtime_error(
      "tgeqf::PressureMeasurement::noise_covariance not implemented");
}

Eigen::Vector3d PressureMeasurement::output_action(const TGGroupElement&,
                                                   const Eigen::Vector3d&) {
  throw std::runtime_error(
      "tgeqf::PressureMeasurement::output_action not implemented");
}

}  // namespace tgeqf
