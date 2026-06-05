#pragma once
#include "State.h"
#include "Group.h"

namespace tgeqf {

/**
 * Symmetry functor for the TG-EqF.
 *
 * Encodes the right group action phi : G x M -> M
 *   phi(X, xi) = [T A_T, Ad_{A_T^{-1}}[b - a^vee], ...]
 *
 * Reference: proposal Eq. (13), Fornasier et al. [1] Lemma 4.1
 *
 * This struct satisfies the GTSAM EquivariantFilter Symmetry concept:
 *   - Group typedef
 *   - Orbit inner class (callable: G -> M, with optional Jacobian)
 *   - Diffeomorphism inner class (callable: M -> M, with optional Jacobian)
 */
struct TGSymmetry {

    using Group = TGGroupElement;

    /**
     * Orbit: functor that maps G -> M by acting on a fixed reference state.
     *
     * Constructed from xi_ref; operator()(X, H) returns phi(X, xi_ref)
     * and optionally the Jacobian dM/dG at the point X.
     *
     * Used by EquivariantFilter to:
     *   1. Recover state estimate: xi_hat = act_on_ref(g_)
     *   2. Compute Dphi0_ = d(phi)/dX |_{X=id} at construction time
     */
    struct Orbit {
        TGState xi_ref;

        explicit Orbit(const TGState& xi_ref);

        /// phi(X, xi_ref) with optional Jacobian d(phi)/dX in R^{18 x 21}
        TGState operator()(
            const TGGroupElement& X,
            Eigen::Matrix<double, 18, 21>* H = nullptr) const;
    };

    /**
     * Diffeomorphism: functor that maps M -> M by acting with a fixed X.
     *
     * Constructed from a group element X; operator()(xi, H) returns phi(X, xi)
     * and optionally the Jacobian d(phi)/dxi in R^{18 x 18}.
     *
     * Used by EquivariantFilter::covariance() to transport the covariance
     * from the origin tangent space to the current state tangent space.
     */
    struct Diffeomorphism {
        TGGroupElement X;

        explicit Diffeomorphism(const TGGroupElement& X);

        /// phi(X, xi) with optional Jacobian d(phi)/dxi in R^{18 x 18}
        TGState operator()(
            const TGState& xi,
            Eigen::Matrix<double, 18, 18>* H = nullptr) const;
    };
};

/**
 * Standalone right group action.
 *
 * phi(X, xi) = [T A_T, Ad_{A_T^{-1}}[b - a^vee]]
 *
 * Reference: proposal Eq. (13)
 */
TGState phi(const TGGroupElement& X, const TGState& xi);

} // namespace tgeqf