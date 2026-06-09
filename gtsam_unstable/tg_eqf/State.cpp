#include <gtsam_unstable/tg_eqf/State.h>

#include <iostream>

namespace tgeqf {

TGState TGState::identity() {
  TGState xi;
  xi.R       = gtsam::Rot3::Identity();
  xi.v       = Eigen::Vector3d::Zero();
  xi.p       = Eigen::Vector3d::Zero();
  xi.b_w     = Eigen::Vector3d::Zero();
  xi.b_a     = Eigen::Vector3d::Zero();
  xi.b_v     = Eigen::Vector3d::Zero();
  return xi;
}

Eigen::Matrix<double, 5, 5> TGState::T_matrix() const {
  // T = [R, v, p; 0, 1, 0; 0, 0, 1]
  Eigen::Matrix<double, 5, 5> T = Eigen::Matrix<double, 5, 5>::Zero();
  T.block<3, 3>(0, 0) = R.matrix();
  T.block<3, 1>(0, 3) = v;
  T.block<3, 1>(0, 4) = p;
  T(3, 3) = 1.0;
  T(4, 4) = 1.0;
  return T;
}

Eigen::Matrix<double, 9, 1> TGState::bias_vector() const {
  // b = [b_w; b_a; b_v]
  Eigen::Matrix<double, 9, 1> b;
  b.segment<3>(0) = b_w;
  b.segment<3>(3) = b_a;
  b.segment<3>(6) = b_v;
  return b;
}

}  // namespace tgeqf

namespace gtsam {

tgeqf::TGState traits<tgeqf::TGState>::Retract(const tgeqf::TGState& xi,
                                               const TangentVector& delta) {
  tgeqf::TGState result;
  result.R       = xi.R       * Rot3::Expmap(delta.segment<3>(0));
  result.v       = xi.v       + delta.segment<3>(3);
  result.p       = xi.p       + delta.segment<3>(6);
  result.b_w     = xi.b_w     + delta.segment<3>(9);
  result.b_a     = xi.b_a     + delta.segment<3>(12);
  result.b_v     = xi.b_v     + delta.segment<3>(15); 
  return result;
}

traits<tgeqf::TGState>::TangentVector traits<tgeqf::TGState>::Local(
    const tgeqf::TGState& xi_ref, const tgeqf::TGState& xi_est) {
  TangentVector delta;
  delta.segment<3>(0)  = Rot3::Logmap(xi_ref.R.inverse() * xi_est.R);
  delta.segment<3>(3)  = xi_est.v       - xi_ref.v;
  delta.segment<3>(6)  = xi_est.p       - xi_ref.p;
  delta.segment<3>(9)  = xi_est.b_w     - xi_ref.b_w;
  delta.segment<3>(12) = xi_est.b_a     - xi_ref.b_a;
  delta.segment<3>(15) = xi_est.b_v     - xi_ref.b_v;
  return delta;
}

bool traits<tgeqf::TGState>::Equals(const tgeqf::TGState& a,
                                    const tgeqf::TGState& b, double tol) {
  return a.R.equals(b.R, tol) &&
         (a.v   - b.v).norm()       < tol &&
         (a.p   - b.p).norm()       < tol &&
         (a.b_w - b.b_w).norm() < tol &&
         (a.b_a - b.b_a).norm()     < tol &&
         (a.b_v - b.b_v).norm()     < tol;
}

void traits<tgeqf::TGState>::Print(const tgeqf::TGState& xi,
                                   const std::string& str) {
  std::cout << str;
  xi.R.print("  R: ");
  std::cout << "  v:   " << xi.v.transpose()       << "\n"
            << "  p:   " << xi.p.transpose()       << "\n"
            << "  b_w: " << xi.b_w.transpose() << "\n"
            << "  b_a: " << xi.b_a.transpose()     << "\n"
            << "  b_v: " << xi.b_v.transpose()     << "\n";
}

}  // namespace gtsam
