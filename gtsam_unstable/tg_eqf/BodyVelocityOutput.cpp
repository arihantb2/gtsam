#include <gtsam_unstable/tg_eqf/BodyVelocityOutput.h>

#include <stdexcept>

namespace tgeqf {

Eigen::Vector3d DVLMeasurement::predict(const TGState&) {
  throw std::runtime_error("tgeqf::DVLMeasurement::predict not implemented");
}

Eigen::Matrix<double, 3, 18> DVLMeasurement::jacobian(const TGState&) {
  throw std::runtime_error("tgeqf::DVLMeasurement::jacobian not implemented");
}

Eigen::Vector3d DVLMeasurement::innovation(const Eigen::Vector3d&,
                                           const TGState&) {
  throw std::runtime_error("tgeqf::DVLMeasurement::innovation not implemented");
}

}  // namespace tgeqf
