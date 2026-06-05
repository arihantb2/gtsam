#include <gtsam_unstable/tg_eqf/EqF.h>

#include <stdexcept>

namespace tgeqf {

TGEqF::TGEqF(const TGState& xi_ref, const Covariance18& Sigma0)
    : Base(xi_ref, Sigma0) {
  throw std::runtime_error("tgeqf::TGEqF::TGEqF not implemented");
}

void TGEqF::propagate(const Eigen::Vector3d&, const Eigen::Vector3d&,
                      const Eigen::Vector3d&, const Covariance18&, double) {
  throw std::runtime_error("tgeqf::TGEqF::propagate not implemented");
}

void TGEqF::update_dvl(const Eigen::Vector3d&, const Covariance3&) {
  throw std::runtime_error("tgeqf::TGEqF::update_dvl not implemented");
}

gtsam::Rot3 TGEqF::attitude() const {
  throw std::runtime_error("tgeqf::TGEqF::attitude not implemented");
}

Eigen::Vector3d TGEqF::position() const {
  throw std::runtime_error("tgeqf::TGEqF::position not implemented");
}

Eigen::Vector3d TGEqF::velocity() const {
  throw std::runtime_error("tgeqf::TGEqF::velocity not implemented");
}

Eigen::Vector3d TGEqF::bias_gyro() const {
  throw std::runtime_error("tgeqf::TGEqF::bias_gyro not implemented");
}

Eigen::Vector3d TGEqF::bias_accel() const {
  throw std::runtime_error("tgeqf::TGEqF::bias_accel not implemented");
}

Eigen::Vector3d TGEqF::bias_vel() const {
  throw std::runtime_error("tgeqf::TGEqF::bias_vel not implemented");
}

}  // namespace tgeqf
