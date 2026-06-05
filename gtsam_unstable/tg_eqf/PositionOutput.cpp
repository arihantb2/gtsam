#include <gtsam_unstable/tg_eqf/PositionOutput.h>

#include <stdexcept>

namespace tgeqf {

Eigen::Vector3d PositionMeasurement::predict(const TGState&,
                                              const Eigen::Vector3d&) {
  throw std::runtime_error("tgeqf::PositionMeasurement::predict not implemented");
}

Eigen::Matrix<double, 3, 18> PositionMeasurement::jacobian_C0(
    const TGState&) {
  throw std::runtime_error(
      "tgeqf::PositionMeasurement::jacobian_C0 not implemented");
}

Eigen::Matrix<double, 3, 18> PositionMeasurement::jacobian_Cstar(
    const TGState&, const Eigen::Vector3d&) {
  throw std::runtime_error(
      "tgeqf::PositionMeasurement::jacobian_Cstar not implemented");
}

Eigen::Vector3d PositionMeasurement::innovation(const Eigen::Vector3d&,
                                                 const TGState&) {
  throw std::runtime_error(
      "tgeqf::PositionMeasurement::innovation not implemented");
}

Eigen::Vector3d PositionMeasurement::output_action(const TGGroupElement&,
                                                    const Eigen::Vector3d&) {
  throw std::runtime_error(
      "tgeqf::PositionMeasurement::output_action not implemented");
}

}  // namespace tgeqf
