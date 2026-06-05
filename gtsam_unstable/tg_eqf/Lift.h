#pragma once
#include "State.h"
#include "Group.h"
#include <Eigen/Dense>

namespace tgeqf {

/**
 * Extended input vector u in L = R^21
 * u = [w, tau, g_vec]
 * w     = [omega, v_tilde, a_tilde] in R^9   (IMU + virtual inputs)
 * tau   = [tau_omega, tau_v, tau_a]  in R^9   (bias rate inputs)
 * g_vec = gravity vector             in R^3
 *
 * Reference: proposal Eq. (9), Section 6.3
 */
struct TGInput {
    Eigen::Vector3d omega;     // measured angular velocity
    Eigen::Vector3d v_tilde;   // virtual velocity input (set to b_v estimate)
    Eigen::Vector3d a_tilde;   // measured linear acceleration
    Eigen::Vector3d tau_omega; // bias rate for b_omega (usually zero)
    Eigen::Vector3d tau_v;     // bias rate for b_v     (usually zero)
    Eigen::Vector3d tau_a;     // bias rate for b_a     (usually zero)
    Eigen::Vector3d g_vec;     // gravity vector in global frame

    /// Pack to R^21
    Eigen::Matrix<double, 21, 1> vector() const;

    /// Unpack from R^21
    static TGInput from_vector(const Eigen::Matrix<double, 21, 1>& v);
};

/**
 * Lift functor Lambda(xi, u) : M x L -> g
 *
 * Lambda_1(xi, u) = (W - B + N) + T^{-1}(G - N)T      in se_2(3)
 * Lambda_2(xi, u) = ad_b[Lambda_1(xi, u)] - tau^      in se_2(3)
 *
 * Satisfies equivariance: Lambda(phi(X,xi), psi(X,u)) = Ad_{X^{-1}} Lambda(xi,u)
 *
 * Reference: Fornasier et al. [1] Theorem 5.5.2, proposal Section 6.5 (TODO)
 *
 * GTSAM concept requirement:
 *   Lift(u)(xi, D_lift) -> TangentG (R^21)
 *   where D_lift is OptionalJacobian<21, 18> = d(Lambda)/d(xi)
 */
struct Lift {
    TGInput u;

    explicit Lift(const TGInput& u);

    /// Lambda(xi, u) with optional Jacobian d(Lambda)/d(xi) in R^{18 x 18}
    Eigen::Matrix<double, 18, 1> operator()(
        const TGState& xi,
        Eigen::Matrix<double, 18, 18>* D_lift = nullptr) const;

private:
    /// Compute W, B, N, G matrices as defined in proposal Eq. (8)
    Eigen::Matrix<double, 5, 5> W_matrix(const TGInput& u) const;
    Eigen::Matrix<double, 5, 5> B_matrix(const TGState& xi) const;
    Eigen::Matrix<double, 5, 5> N_matrix() const;
    Eigen::Matrix<double, 5, 5> G_matrix(const TGInput& u) const;
};

/**
 * Input orbit functor psi_u : G -> L
 *
 * psi(X, u) maps the current input to the origin via the input symmetry action.
 * Used in EquivariantFilter::computeErrorDynamicsMatrix as:
 *   u_origin = psi_u(g_.inverse())
 *
 * Reference: van Goor et al. [6] Eq. (13), equivariance of the lift
 * Fornasier 2023 Lemma 8 Eq. (20) for G_TG = SE_2(3) ⋉ se_2(3):
 *   psi(X, u)_w   = Ad_{C^{-1}}(w - a^vee) + Omega(C^{-1})   in R^9
 *   psi(X, u)_tau = Ad_{C^{-1}}(tau)                          in R^9
 *   psi(X, u)_g   = g_vec                                      (unchanged)
 *
 * where C = SE_2(3) part of X, a = se_2(3) part of X, w and tau from u,
 * and Omega(C^{-1}) = (0, 0, a) places C^{-1}'s position in the a_tilde slot.
 */
struct InputOrbit {
    TGInput u;

    explicit InputOrbit(const TGInput& u);

    /// psi(X, u): maps group element X to transformed input in L
    TGInput operator()(const TGGroupElement& X) const;
};

} // namespace tgeqf