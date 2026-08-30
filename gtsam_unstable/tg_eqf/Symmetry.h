#pragma once
#include <gtsam/base/GroupAction.h>
#include <gtsam_unstable/tg_eqf/Group.h>
#include <gtsam_unstable/tg_eqf/State.h>

namespace gtsam {
namespace tgeqf {

/// Right group action phi : G x M -> M for TG-EqF.
struct TGSymmetry {
  using Group = TGElement;
  using Manifold = State;

  /// phi(X, phi(Y, xi)) == phi(Y * X, xi), so this is a right action.
  static constexpr ActionType type = ActionType::Right;

  /// Maps G -> M by acting on a fixed reference state (orbit functor).
  struct Orbit {
    State xi_ref;

    explicit Orbit(const State& xi_ref);

    /// phi(X, xi_ref) with optional Jacobian d(phi)/dX in R^{18 x 18}.
    State operator()(const TGElement& X,
                     Eigen::Matrix<double, 18, 18>* H = nullptr) const;
  };

  /// Maps M -> M by acting with a fixed X (diffeomorphism functor).
  struct Diffeomorphism {
    TGElement X;

    explicit Diffeomorphism(const TGElement& X);

    /// phi(X, xi) with optional Jacobian d(phi)/dxi in R^{18 x 18}.
    State operator()(const State& xi,
                     Eigen::Matrix<double, 18, 18>* H = nullptr) const;
  };
};

/// Standalone right group action phi(X, xi).
State phi(const TGElement& X, const State& xi);

/// The unique group element taking xi_ref to xi_est, i.e. the X with
/// phi(X, xi_ref) == xi_est. Use it to initialize an EqF whose origin is fixed
/// at xi_ref but whose estimate should start at xi_est.
TGElement phiInverse(const State& xi_ref, const State& xi_est);

}  // namespace tgeqf
}  // namespace gtsam
