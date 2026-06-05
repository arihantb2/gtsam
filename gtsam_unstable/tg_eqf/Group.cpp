#include <gtsam_unstable/tg_eqf/Group.h>

#include <stdexcept>

namespace tgeqf {

Eigen::Matrix<double, 9, 1> se2_3::vector() const {
  throw std::runtime_error("tgeqf::se2_3::vector not implemented");
}

se2_3 se2_3::from_vector(const Eigen::Matrix<double, 9, 1>&) {
  throw std::runtime_error("tgeqf::se2_3::from_vector not implemented");
}

Eigen::Matrix<double, 5, 5> se2_3::wedge() const {
  throw std::runtime_error("tgeqf::se2_3::wedge not implemented");
}

se2_3 se2_3::vee(const Eigen::Matrix<double, 5, 5>&) {
  throw std::runtime_error("tgeqf::se2_3::vee not implemented");
}

TGGroupElement TGGroupElement::Identity() {
  throw std::runtime_error("tgeqf::TGGroupElement::Identity not implemented");
}

TGGroupElement TGGroupElement::operator*(const TGGroupElement&) const {
  throw std::runtime_error("tgeqf::TGGroupElement::operator* not implemented");
}

TGGroupElement TGGroupElement::inverse() const {
  throw std::runtime_error("tgeqf::TGGroupElement::inverse not implemented");
}

se2_3 TGGroupElement::Ad(const se2_3&) const {
  throw std::runtime_error("tgeqf::TGGroupElement::Ad not implemented");
}

se2_3 TGGroupElement::Ad_AT_inv(const se2_3&) const {
  throw std::runtime_error("tgeqf::TGGroupElement::Ad_AT_inv not implemented");
}

TGGroupElement TGGroupElement::Expmap(const Eigen::Matrix<double, 21, 1>&) {
  throw std::runtime_error("tgeqf::TGGroupElement::Expmap not implemented");
}

Eigen::Matrix<double, 21, 1> TGGroupElement::Logmap() const {
  throw std::runtime_error("tgeqf::TGGroupElement::Logmap not implemented");
}

Eigen::Matrix<double, 5, 5> TGGroupElement::AT_matrix() const {
  throw std::runtime_error("tgeqf::TGGroupElement::AT_matrix not implemented");
}

}  // namespace tgeqf

namespace gtsam {

tgeqf::TGGroupElement traits<tgeqf::TGGroupElement>::Identity() {
  throw std::runtime_error(
      "gtsam::traits<tgeqf::TGGroupElement>::Identity not implemented");
}

tgeqf::TGGroupElement traits<tgeqf::TGGroupElement>::Compose(
    const tgeqf::TGGroupElement&, const tgeqf::TGGroupElement&) {
  throw std::runtime_error(
      "gtsam::traits<tgeqf::TGGroupElement>::Compose not implemented");
}

tgeqf::TGGroupElement traits<tgeqf::TGGroupElement>::Between(
    const tgeqf::TGGroupElement&, const tgeqf::TGGroupElement&) {
  throw std::runtime_error(
      "gtsam::traits<tgeqf::TGGroupElement>::Between not implemented");
}

tgeqf::TGGroupElement traits<tgeqf::TGGroupElement>::Expmap(
    const TangentVector&) {
  throw std::runtime_error(
      "gtsam::traits<tgeqf::TGGroupElement>::Expmap not implemented");
}

traits<tgeqf::TGGroupElement>::TangentVector
traits<tgeqf::TGGroupElement>::Logmap(const tgeqf::TGGroupElement&) {
  throw std::runtime_error(
      "gtsam::traits<tgeqf::TGGroupElement>::Logmap not implemented");
}

tgeqf::TGGroupElement traits<tgeqf::TGGroupElement>::Retract(
    const tgeqf::TGGroupElement&, const TangentVector&) {
  throw std::runtime_error(
      "gtsam::traits<tgeqf::TGGroupElement>::Retract not implemented");
}

traits<tgeqf::TGGroupElement>::TangentVector
traits<tgeqf::TGGroupElement>::Local(const tgeqf::TGGroupElement&,
                                     const tgeqf::TGGroupElement&) {
  throw std::runtime_error(
      "gtsam::traits<tgeqf::TGGroupElement>::Local not implemented");
}

}  // namespace gtsam
