#include <gtsam_unstable/tg_eqf/State.h>
#include <gtsam_unstable/tg_eqf/Symmetry.h>

#include <iostream>

namespace gtsam {
namespace tgeqf {

Eigen::Matrix<double, 5, 5> State::T_matrix() const {
  Eigen::Matrix<double, 5, 5> T = Eigen::Matrix<double, 5, 5>::Zero();
  T.block<3, 3>(0, 0) = R.matrix();
  T.block<3, 1>(0, 3) = v;
  T.block<3, 1>(0, 4) = p;
  T(3, 3) = 1.0;
  T(4, 4) = 1.0;
  return T;
}

Eigen::Matrix<double, 9, 1> State::bias_vector() const {
  Eigen::Matrix<double, 9, 1> b;
  b.segment<3>(0) = b_w;
  b.segment<3>(3) = b_a;
  b.segment<3>(6) = b_v;
  return b;
}

ExtendedPose3<2> State::extendedPose() const {
  Eigen::Matrix<double, 3, 2> vp;
  vp.col(0) = v;
  vp.col(1) = p;
  return ExtendedPose3<2>(R, vp);
}

// Chart: theta = log_G . phi_this^-1.
State State::retract(const Eigen::Matrix<double, 18, 1>& delta) const {
  return phi(TGElement::Expmap(delta), *this);
}

Eigen::Matrix<double, 18, 1> State::localCoordinates(const State& other) const {
  return phiInverse(*this, other).Logmap();
}

}  // namespace tgeqf
}  // namespace gtsam

namespace gtsam {

bool traits<tgeqf::State>::Equals(const tgeqf::State& a, const tgeqf::State& b,
                                  double tol) {
  return a.R.equals(b.R, tol) && (a.v - b.v).norm() < tol &&
         (a.p - b.p).norm() < tol && (a.b_w - b.b_w).norm() < tol &&
         (a.b_a - b.b_a).norm() < tol && (a.b_v - b.b_v).norm() < tol;
}

void traits<tgeqf::State>::Print(const tgeqf::State& xi,
                                 const std::string& str) {
  std::cout << str;
  xi.R.print("  R: ");
  std::cout << "  v:   " << xi.v.transpose() << "\n"
            << "  p:   " << xi.p.transpose() << "\n"
            << "  b_w: " << xi.b_w.transpose() << "\n"
            << "  b_a: " << xi.b_a.transpose() << "\n"
            << "  b_v: " << xi.b_v.transpose() << "\n";
}

}  // namespace gtsam
