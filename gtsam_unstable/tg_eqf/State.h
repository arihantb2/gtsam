#pragma once
#include <gtsam/base/Lie.h>
#include <gtsam/geometry/ExtendedPose3.h>
#include <gtsam/geometry/Rot3.h>

#include <Eigen/Dense>

namespace gtsam {
namespace tgeqf {

/// Biased INS manifold state xi = [R, v, p, b_w, b_a, b_v] on SE_2(3) x R^9
/// (dim 18).
struct State {
  gtsam::Rot3 R;
  Eigen::Vector3d v = Eigen::Vector3d::Zero();
  Eigen::Vector3d p = Eigen::Vector3d::Zero();
  Eigen::Vector3d b_w = Eigen::Vector3d::Zero();
  Eigen::Vector3d b_a = Eigen::Vector3d::Zero();
  Eigen::Vector3d b_v = Eigen::Vector3d::Zero();

  static constexpr int dimension = 18;

  static State identity() { return {}; }

  Eigen::Matrix<double, 5, 5> T_matrix() const;
  Eigen::Matrix<double, 9, 1> bias_vector() const;

  /// Extended pose T = [R, v, p] as an SE_2(3) group element.
  gtsam::ExtendedPose3<2> extendedPose() const;

  /**
   * Retract in the origin chart.
   *
   * The chart is the one the EqF design is posed in: the SE_2(3) exponential
   * on the navigation block and the identity on the biases,
   *
   *   Retract(xi, eps) = (T * Expmap_SE23(eps_T), b + eps_b).
   *
   * The navigation block is *not* the naive product chart. Its velocity and
   * position coordinates are SE_2(3) logarithm coordinates, which couple to
   * the rotation through the left Jacobian and are resolved in the body frame
   * of T; they coincide with global-frame offsets only when R is the identity.
   * This is what makes the navigation-state error dynamics exactly linear
   * rather than linear only to first order.
   */
  State retract(const Eigen::Matrix<double, 18, 1>& delta) const;

  /// Local coordinates in the origin chart; inverse of retract.
  Eigen::Matrix<double, 18, 1> localCoordinates(const State& other) const;
};

}  // namespace tgeqf
}  // namespace gtsam

namespace gtsam {

template <>
struct traits<tgeqf::State> {
  using ManifoldType = tgeqf::State;
  using TangentVector = Eigen::Matrix<double, 18, 1>;
  using Jacobian = Eigen::Matrix<double, 18, 18>;
  using structure_category = manifold_tag;

  static constexpr int dimension = 18;
  static int GetDimension(const tgeqf::State&) { return dimension; }

  /// Retract at xi_ref
  static tgeqf::State Retract(const tgeqf::State& xi,
                              const TangentVector& delta) {
    return xi.retract(delta);
  }

  /// Local(xi_ref, xi_est): origin-chart error; inverse of Retract.
  static TangentVector Local(const tgeqf::State& xi_ref,
                             const tgeqf::State& xi_est) {
    return xi_ref.localCoordinates(xi_est);
  }

  static tgeqf::State Identity() { return tgeqf::State::identity(); }

  static bool Equals(const tgeqf::State& a, const tgeqf::State& b,
                     double tol = 1e-9);

  static void Print(const tgeqf::State& xi, const std::string& str = "");
};

template <>
struct traits<const tgeqf::State> : traits<tgeqf::State> {};

}  // namespace gtsam
