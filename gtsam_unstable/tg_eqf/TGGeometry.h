#pragma once
#include <gtsam/base/GroupAction.h>
#include <gtsam/base/Lie.h>
#include <gtsam/base/SemidirectLieGroup.h>
#include <gtsam/base/TangentLieGroup.h>
#include <gtsam/geometry/ExtendedPose3.h>
#include <gtsam/geometry/Rot3.h>

#include <Eigen/Dense>

namespace gtsam {
namespace tgeqf {

/// Tangent group of SE_2(3) under its adjoint action: pairs (A, a) with
/// product (A1,a1)(A2,a2) = (A1 A2, a1 + Ad_{A1} a2).
using TGElement = gtsam::TangentLieGroup<gtsam::ExtendedPose3<2>>;

/**
 * Left action of the tangent group on R^3 through the attitude of A,
 * (A, a) . d = R_A d. It builds the magnetometer-bias factor of the symmetry
 * group below.
 */
struct MagBiasAction : public GroupAction<MagBiasAction, TGElement, Vector3> {
  static constexpr ActionType type = ActionType::Left;

  /// R_A d, with Jacobians for the right perturbation X Exp(delta) and for d.
  Vector3 operator()(const TGElement& X, const Vector3& d,
                     OptionalJacobian<3, 18> HX = {},
                     OptionalJacobian<3, 3> Hd = {}) const;

  /// Infinitesimal generator: the attitude component of u, as a 3x3 skew.
  static Matrix3 generator(const TGElement::TangentVector& u);
};

/**
 * Symmetry group G = TG ⋉ R^3 of the magnetometer-augmented INS: elements
 * X = (A, a, d) with product
 * (A1,a1,d1)(A2,a2,d2) = (A1 A2, a1 + Ad_{A1} a2, d1 + R_{A1} d2).
 * X.first is the tangent-group part (A, a) and X.second is d.
 */
using TGMagElement =
    gtsam::SemidirectLieGroup<TGElement, Vector3, MagBiasAction>;

/// Biased INS manifold state xi = (T, b, bm), T = [R, v, p] in SE_2(3),
/// b = [b_w; b_a; b_v] in R^9, and bm the magnetometer hard-iron bias in R^3
/// (dim 21).
struct State {
  gtsam::ExtendedPose3<2> T;
  Vector9 b = Vector9::Zero();
  /// Magnetometer hard-iron bias, body frame, in the units of the sensor.
  Vector3 bm = Vector3::Zero();

  static constexpr int dimension = 21;

  /// Tangent vector of the state manifold, in the origin chart.
  using TangentVector = Eigen::Matrix<double, 21, 1>;
  /// Jacobian of a map between tangent spaces of the state manifold.
  using Jacobian = Eigen::Matrix<double, 21, 21>;

  static State identity() { return {}; }

  Matrix5 T_matrix() const { return T.matrix(); }

  /// Retract in the origin chart: Retract(xi, eps) = phi(Expmap_G(eps), xi).
  /// The navigation block reads SE_2(3) logarithm coordinates in T's body
  /// frame; the bias blocks go through the group's fiber exponentials rather
  /// than a plain vector difference (to first order, bm moves by -eps_bm).
  State retract(const TangentVector& delta) const;

  /// Local coordinates in the origin chart; inverse of retract.
  TangentVector localCoordinates(const State& other) const;
};

}  // namespace tgeqf
}  // namespace gtsam

namespace gtsam {

template <>
struct traits<tgeqf::State> {
  using ManifoldType = tgeqf::State;
  using TangentVector = tgeqf::State::TangentVector;
  using Jacobian = tgeqf::State::Jacobian;
  using structure_category = manifold_tag;

  static constexpr int dimension = tgeqf::State::dimension;
  static int GetDimension(const tgeqf::State&) { return dimension; }

  static tgeqf::State Retract(const tgeqf::State& xi,
                              const TangentVector& delta) {
    return xi.retract(delta);
  }

  /// Inverse of Retract.
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

template <>
struct traits<const tgeqf::TGElement> : traits<tgeqf::TGElement> {};

template <>
struct traits<const tgeqf::TGMagElement> : traits<tgeqf::TGMagElement> {};

}  // namespace gtsam

namespace gtsam {
namespace tgeqf {

/**
 * Right group action phi : G x M -> M for TG-EqF,
 * phi(X, xi) = (T A, Ad_{A^-1}(b - a), R_A^T (bm - d)).
 */
struct TGSymmetry {
  using Group = TGMagElement;
  using Manifold = State;

  /// phi(X, phi(Y, xi)) == phi(Y * X, xi), so this is a right action.
  static constexpr ActionType type = ActionType::Right;

  /// Maps G -> M by acting on a fixed reference state (orbit functor).
  struct Orbit {
    State xi_ref;

    explicit Orbit(const State& xi_ref);

    /// phi(X, xi_ref) with optional Jacobian d(phi)/dX in R^{21 x 21}.
    State operator()(const TGMagElement& X, State::Jacobian* H = nullptr) const;
  };

  /// Maps M -> M by acting with a fixed X (diffeomorphism functor).
  struct Diffeomorphism {
    TGMagElement X;

    explicit Diffeomorphism(const TGMagElement& X);

    /// phi(X, xi) with optional Jacobian d(phi)/dxi in R^{21 x 21}.
    State operator()(const State& xi, State::Jacobian* H = nullptr) const;
  };
};

/// Standalone right group action phi(X, xi).
State phi(const TGMagElement& X, const State& xi);

/// The unique X with phi(X, xi_ref) == xi_est. Initializes an EqF whose
/// origin is fixed at xi_ref but whose estimate starts at xi_est.
TGMagElement phiInverse(const State& xi_ref, const State& xi_est);

}  // namespace tgeqf
}  // namespace gtsam
