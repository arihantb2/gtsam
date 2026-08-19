#include <gtsam_unstable/tg_eqf/State.h>

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

State State::retract(const Eigen::Matrix<double, 18, 1>& delta) const {
  const ExtendedPose3<2> T_new =
      extendedPose() * ExtendedPose3<2>::Expmap(delta.head<9>());

  State result;
  result.R = T_new.rotation();
  result.v = T_new.x(0);
  result.p = T_new.x(1);
  result.b_w = b_w + delta.segment<3>(9);
  result.b_a = b_a + delta.segment<3>(12);
  result.b_v = b_v + delta.segment<3>(15);
  return result;
}

Eigen::Matrix<double, 18, 1> State::localCoordinates(const State& other) const {
  Eigen::Matrix<double, 18, 1> delta;
  delta.head<9>() = ExtendedPose3<2>::Logmap(extendedPose().inverse() *
                                             other.extendedPose());
  delta.segment<3>(9) = other.b_w - b_w;
  delta.segment<3>(12) = other.b_a - b_a;
  delta.segment<3>(15) = other.b_v - b_v;
  return delta;
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
