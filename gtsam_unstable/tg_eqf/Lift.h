#pragma once
#include <gtsam_unstable/tg_eqf/Group.h>
#include <gtsam_unstable/tg_eqf/State.h>

#include <Eigen/Dense>

namespace gtsam {
namespace tgeqf {

/// Extended input u in R^21.
struct Input {
  Eigen::Vector3d w = Eigen::Vector3d::Zero();
  Eigen::Vector3d a = Eigen::Vector3d::Zero();
  Eigen::Vector3d v = Eigen::Vector3d::Zero();  // virtual input \nu
  Eigen::Vector3d tau_w = Eigen::Vector3d::Zero();
  Eigen::Vector3d tau_a = Eigen::Vector3d::Zero();
  Eigen::Vector3d tau_v = Eigen::Vector3d::Zero();
  Eigen::Vector3d g_vec = Eigen::Vector3d::Zero();

  /// Coefficient vector of the extended input.
  using Vector21 = Eigen::Matrix<double, 21, 1>;

  Vector21 vector() const;
  static Input from_vector(const Vector21& v);
};

/**
 * Lift functor Lambda(xi, u) : M x L -> g.
 *
 * Expanding T^{-1}(G-N)T for T = [R, v, p] collapses the wedge/matrix algebra
 * to a closed form (the bottom-row coupling that N/-N introduce cancels the
 * homogeneous rows exactly, leaving only R^T acting on g and on v):
 *
 *   Lambda_1.w = w - b_w
 *   Lambda_1.a = a - b_a + R^T g
 *   Lambda_1.v = nu - b_v + R^T v
 *   Lambda_2   = ad_b[Lambda_1] - tau
 */
struct Lift {
  Input u;

  explicit Lift(const Input& u);

  /// Lambda(xi, u) with optional Jacobian d(Lambda)/d(xi) in R^{18 x 18}.
  Eigen::Matrix<double, 18, 1> operator()(
      const State& xi, Eigen::Matrix<double, 18, 18>* D_lift = nullptr) const;
};

/// Input orbit psi_u : G -> L.
struct InputOrbit {
  Input u;

  explicit InputOrbit(const Input& u);

  Input operator()(const TGElement& X) const;
};

}  // namespace tgeqf
}  // namespace gtsam
