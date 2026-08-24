#pragma once
#include <gtsam/base/Lie.h>
#include <gtsam/geometry/ExtendedPose3.h>
#include <gtsam/geometry/Rot3.h>

#include <Eigen/Dense>

namespace gtsam {
namespace tgeqf {

/// Lie algebra element of se_2(3).
struct se2_3 {
  Eigen::Vector3d w = Eigen::Vector3d::Zero();
  Eigen::Vector3d a = Eigen::Vector3d::Zero();
  Eigen::Vector3d v = Eigen::Vector3d::Zero();

  Vector9 vector() const;
  static se2_3 from_vector(const Vector9& v);
  Matrix5 wedge() const;
  static se2_3 vee(const Matrix5& mat);
};

/// Group element X = [A, a] of G = SE_2(3) ⋉ se_2(3).
struct TGElement {
  gtsam::Rot3 R;
  Eigen::Vector3d v = Eigen::Vector3d::Zero();
  Eigen::Vector3d p = Eigen::Vector3d::Zero();
  se2_3 a;

  static constexpr int dimension = 18;

  /// Tangent vector of G, in the right chart.
  using TangentVector = Eigen::Matrix<double, 18, 1>;

  static TGElement Identity() { return {}; }

  TGElement operator*(const TGElement& Y) const;
  TGElement inverse() const;
  se2_3 Ad_A(const se2_3& xi) const;
  se2_3 Ad_A_inv(const se2_3& xi) const;

  /// Exponential map; fiber part uses Xi(tau) series (see Group.cpp).
  static TGElement Expmap(const TangentVector& xi);

  /// Logarithm map; exact inverse of Expmap.
  TangentVector Logmap() const;

  /// Right retract: Compose(*this, Expmap(xi)).
  TGElement retract(const TangentVector& xi) const;

  /// Right local coordinates: Logmap(inverse() * other).
  TangentVector localCoordinates(const TGElement& other) const;

  Matrix5 to_A_matrix() const;
};

}  // namespace tgeqf
}  // namespace gtsam

namespace gtsam {

template <>
struct traits<tgeqf::TGElement> {
  using ManifoldType = tgeqf::TGElement;
  using TangentVector = tgeqf::TGElement::TangentVector;
  using structure_category = lie_group_tag;
  using group_flavor = multiplicative_group_tag;
  using ChartJacobian = OptionalJacobian<18, 18>;

  static constexpr int dimension = 18;
  static int GetDimension(const tgeqf::TGElement&) { return dimension; }

  static bool Equals(const tgeqf::TGElement& X, const tgeqf::TGElement& Y,
                     double tol = 1e-9);
  static void Print(const tgeqf::TGElement& X, const std::string& str = "");

  static tgeqf::TGElement Identity();

  /// Hx = Ad_{Y^{-1}}, Hy = I (standard right-chart Compose Jacobians).
  static tgeqf::TGElement Compose(const tgeqf::TGElement& X,
                                  const tgeqf::TGElement& Y,
                                  ChartJacobian Hx = {}, ChartJacobian Hy = {});

  /// Hx = -Ad_{(X^{-1}Y)^{-1}}, Hy = I.
  static tgeqf::TGElement Between(const tgeqf::TGElement& X,
                                  const tgeqf::TGElement& Y,
                                  ChartJacobian Hx = {}, ChartJacobian Hy = {});

  /// H = -Ad_X.
  static tgeqf::TGElement Inverse(const tgeqf::TGElement& X,
                                  ChartJacobian H = {});

  /// Jacobian (dexp) not implemented; throws if H requested.
  static tgeqf::TGElement Expmap(const TangentVector& xi, ChartJacobian H = {});

  /// Jacobian (dlog) not implemented; throws if H requested.
  static TangentVector Logmap(const tgeqf::TGElement& X, ChartJacobian H = {});

  static Eigen::Matrix<double, 18, 18> AdjointMap(const tgeqf::TGElement& X);

  static tgeqf::TGElement Retract(const tgeqf::TGElement& X,
                                  const TangentVector& xi) {
    return X.retract(xi);
  }

  static TangentVector Local(const tgeqf::TGElement& X,
                             const tgeqf::TGElement& Y) {
    return X.localCoordinates(Y);
  }
};

template <>
struct traits<const tgeqf::TGElement> : traits<tgeqf::TGElement> {};

}  // namespace gtsam

namespace gtsam {
namespace tgeqf {
namespace detail {

using Se23 = gtsam::ExtendedPose3<2>;

inline Se23 toSe23(const tgeqf::TGElement& X) {
  Eigen::Matrix<double, 3, 2> vp;
  vp.col(0) = X.v;
  vp.col(1) = X.p;
  return Se23(X.R, vp);
}

// Adjoint Ad_A of an SE_2(3) element A = [R, v, p] acting on se_2(3):
//
//   Ad_A = [ R       0   0 ]
//          [ v^× R   R   0 ]
//          [ p^× R   0   R ]
inline Eigen::Matrix<double, 9, 9> Ad_SE23(const tgeqf::TGElement& X) {
  const Eigen::Matrix3d R = X.R.matrix();
  const Eigen::Matrix3d vR = gtsam::skewSymmetric(X.v) * R;
  const Eigen::Matrix3d pR = gtsam::skewSymmetric(X.p) * R;

  Eigen::Matrix<double, 9, 9> A = Eigen::Matrix<double, 9, 9>::Zero();
  A.block<3, 3>(0, 0) = R;
  A.block<3, 3>(3, 0) = vR;
  A.block<3, 3>(3, 3) = R;
  A.block<3, 3>(6, 0) = pR;
  A.block<3, 3>(6, 6) = R;
  return A;
}

// Inverse adjoint Ad_{A^{-1}} of an SE_2(3) element A = [R, v, p] on se_2(3):
//
//   Ad_{A^{-1}} = [  R^T         0    0   ]
//                 [ -R^T [v]^x   R^T  0   ]
//                 [ -R^T [p]^x   0    R^T ]
inline Eigen::Matrix<double, 9, 9> Ad_SE23_inv(const tgeqf::TGElement& X) {
  const Eigen::Matrix3d Rt = X.R.transpose();
  const Eigen::Matrix3d RtV = -Rt * gtsam::skewSymmetric(X.v);
  const Eigen::Matrix3d RtP = -Rt * gtsam::skewSymmetric(X.p);

  Eigen::Matrix<double, 9, 9> A = Eigen::Matrix<double, 9, 9>::Zero();
  A.block<3, 3>(0, 0) = Rt;
  A.block<3, 3>(3, 0) = RtV;
  A.block<3, 3>(3, 3) = Rt;
  A.block<3, 3>(6, 0) = RtP;
  A.block<3, 3>(6, 6) = Rt;
  return A;
}

// Algebra adjoint ad_a = [a, .] of a se_2(3) element a = [w; a; v]:
//
//   ad_a = [ w^×    0     0   ]
//          [ a^×    w^×   0   ]
//          [ v^×    0     w^× ]
inline Eigen::Matrix<double, 9, 9> ad_se23(const tgeqf::se2_3& a) {
  const Eigen::Matrix3d wx = gtsam::skewSymmetric(a.w);
  const Eigen::Matrix3d ax = gtsam::skewSymmetric(a.a);
  const Eigen::Matrix3d vx = gtsam::skewSymmetric(a.v);

  Eigen::Matrix<double, 9, 9> M = Eigen::Matrix<double, 9, 9>::Zero();
  M.block<3, 3>(0, 0) = wx;
  M.block<3, 3>(3, 0) = ax;
  M.block<3, 3>(3, 3) = wx;
  M.block<3, 3>(6, 0) = vx;
  M.block<3, 3>(6, 6) = wx;
  return M;
}

// Algebra adjoint ad_gamma of the full group G = SE_2(3) (x) se_2(3), gamma =
// (eta, zeta) in R^18. Not the group adjoint Ad_X of AdjointMap() below.
inline Eigen::Matrix<double, 18, 18> ad_G(
    const Eigen::Matrix<double, 18, 1>& gamma) {
  const tgeqf::se2_3 eta = tgeqf::se2_3::from_vector(gamma.head<9>());
  const tgeqf::se2_3 zeta = tgeqf::se2_3::from_vector(gamma.tail<9>());
  const Eigen::Matrix<double, 9, 9> ad_eta = ad_se23(eta);

  Eigen::Matrix<double, 18, 18> M = Eigen::Matrix<double, 18, 18>::Zero();
  M.block<9, 9>(0, 0) = ad_eta;
  M.block<9, 9>(9, 0) = ad_se23(zeta);
  M.block<9, 9>(9, 9) = ad_eta;
  return M;
}

// Left Jacobian of G, J_l(gamma) = sum_{k>=0} ad_gamma^k / (k+1)!; the 18x18
// analogue of Xi(tau) above.
inline Eigen::Matrix<double, 18, 18> LeftJacobianG(
    const Eigen::Matrix<double, 18, 1>& gamma) {
  const Eigen::Matrix<double, 18, 18> ad = ad_G(gamma);
  Eigen::Matrix<double, 18, 18> term = Eigen::Matrix<double, 18, 18>::Identity();
  Eigen::Matrix<double, 18, 18> sum = Eigen::Matrix<double, 18, 18>::Zero();
  double fact = 1.0;
  for (int k = 0; k < 28; ++k) {
    fact *= (k + 1);
    sum += term / fact;
    term = term * ad;
  }
  return sum;
}

}  // namespace detail
}  // namespace tgeqf
}  // namespace gtsam
