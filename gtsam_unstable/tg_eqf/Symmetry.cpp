#include <gtsam_unstable/tg_eqf/Symmetry.h>

#include <stdexcept>

namespace tgeqf {

TGSymmetry::Orbit::Orbit(const TGState&) {}

TGState TGSymmetry::Orbit::operator()(const TGGroupElement&,
                                      Eigen::Matrix<double, 18, 21>*) const {
  throw std::runtime_error("tgeqf::TGSymmetry::Orbit::operator() not implemented");
}

TGSymmetry::Diffeomorphism::Diffeomorphism(const TGGroupElement&) {}

TGState TGSymmetry::Diffeomorphism::operator()(const TGState&,
                                               Eigen::Matrix<double, 18, 18>*) const {
  throw std::runtime_error(
      "tgeqf::TGSymmetry::Diffeomorphism::operator() not implemented");
}

TGState phi(const TGGroupElement&, const TGState&) {
  throw std::runtime_error("tgeqf::phi not implemented");
}

}  // namespace tgeqf
