#include <gtsam_unstable/tg_eqf/State.h>

#include <stdexcept>

namespace tgeqf {

TGState TGState::identity() {
  throw std::runtime_error("tgeqf::TGState::identity not implemented");
}

Eigen::Matrix<double, 5, 5> TGState::T_matrix() const {
  throw std::runtime_error("tgeqf::TGState::T_matrix not implemented");
}

Eigen::Matrix<double, 9, 1> TGState::bias_vector() const {
  throw std::runtime_error("tgeqf::TGState::bias_vector not implemented");
}

}  // namespace tgeqf

namespace gtsam {

tgeqf::TGState traits<tgeqf::TGState>::Retract(const tgeqf::TGState&,
                                               const TangentVector&) {
  throw std::runtime_error(
      "gtsam::traits<tgeqf::TGState>::Retract not implemented");
}

traits<tgeqf::TGState>::TangentVector traits<tgeqf::TGState>::Local(
    const tgeqf::TGState&, const tgeqf::TGState&) {
  throw std::runtime_error(
      "gtsam::traits<tgeqf::TGState>::Local not implemented");
}

bool traits<tgeqf::TGState>::Equals(const tgeqf::TGState&, const tgeqf::TGState&,
                                    double) {
  throw std::runtime_error(
      "gtsam::traits<tgeqf::TGState>::Equals not implemented");
}

void traits<tgeqf::TGState>::Print(const tgeqf::TGState&, const std::string&) {
  throw std::runtime_error(
      "gtsam::traits<tgeqf::TGState>::Print not implemented");
}

}  // namespace gtsam
