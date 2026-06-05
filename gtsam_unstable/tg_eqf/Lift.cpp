#include <gtsam_unstable/tg_eqf/Lift.h>

#include <stdexcept>

namespace tgeqf {

Eigen::Matrix<double, 21, 1> TGInput::vector() const {
  throw std::runtime_error("tgeqf::TGInput::vector not implemented");
}

TGInput TGInput::from_vector(const Eigen::Matrix<double, 21, 1>&) {
  throw std::runtime_error("tgeqf::TGInput::from_vector not implemented");
}

Lift::Lift(const TGInput& u) : u(u) {}

Eigen::Matrix<double, 21, 1> Lift::operator()(
    const TGState&, Eigen::Matrix<double, 21, 18>*) const {
  throw std::runtime_error("tgeqf::Lift::operator() not implemented");
}

Eigen::Matrix<double, 5, 5> Lift::W_matrix(const TGInput&) const {
  throw std::runtime_error("tgeqf::Lift::W_matrix not implemented");
}

Eigen::Matrix<double, 5, 5> Lift::B_matrix(const TGState&) const {
  throw std::runtime_error("tgeqf::Lift::B_matrix not implemented");
}

Eigen::Matrix<double, 5, 5> Lift::N_matrix() const {
  throw std::runtime_error("tgeqf::Lift::N_matrix not implemented");
}

Eigen::Matrix<double, 5, 5> Lift::G_matrix(const TGInput&) const {
  throw std::runtime_error("tgeqf::Lift::G_matrix not implemented");
}

InputOrbit::InputOrbit(const TGInput& u) : u(u) {}

TGInput InputOrbit::operator()(const TGGroupElement&) const {
  throw std::runtime_error("tgeqf::InputOrbit::operator() not implemented");
}

}  // namespace tgeqf
