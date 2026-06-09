#pragma once
#include "State.h"
#include "Group.h"
#include <Eigen/Dense>

namespace tgeqf {

/**
 * Extended input vector u in L = R^21
 * u = [w, tau, g_vec]
 * w     = [w, a, v]             in R^9   (IMU + virtual velocity input)
 * tau   = [tau_w, tau_a, tau_v] in R^9   (bias rate inputs)
 * g_vec = [0, 0, g]             in R^3   (gravity vector in global frame)
 *
 * Reference: proposal Eq. (9), Section 6.3
 */
struct TGInput {
    Eigen::Vector3d w;     // measured angular velocity
    Eigen::Vector3d a;     // measured linear acceleration
    Eigen::Vector3d v;     // virtual velocity input (set to b_v estimate)
    Eigen::Vector3d tau_w; // bias rate for b_w
    Eigen::Vector3d tau_a; // bias rate for b_a
    Eigen::Vector3d tau_v; // bias rate for b_v (virtual; set to zero)
    Eigen::Vector3d g_vec; // gravity vector in global frame

    /// Pack to R^21 = [w, a, v, tau_w, tau_a, tau_v, g_vec]
    Eigen::Matrix<double, 21, 1> vector() const;

    /// Unpack from R^21
    static TGInput from_vector(const Eigen::Matrix<double, 21, 1>& v);
};

/**
 * Lift functor Lambda(xi, u) : M x L -> g
 *
 * System dynamics (Eq. 12a-b):
 *   Tdot = T(W - B + N) + (G - N)T
 *   bdot = tau
 *
 * Theorem 9 (Eq. 21-22) form:
 *   Lambda_1(xi, u) := (W - B + N) + T^{-1}(G - N)T            in se_2(3)
 *   Lambda_2(xi, u) := ad_b[Lambda_1(xi, u)] - tau^            in se_2(3)
 *
 * Satisfies equivariance: Lambda(phi(X,xi), psi(X,u)) = Ad_{X^{-1}} Lambda(xi,u)
 *
 * Reference: Fornasier et al. [1] Theorem 9 (Eq. 21-22), proposal Section 6.5 (TODO)
 *
 * GTSAM concept requirement:
 *   Lift(u)(xi, D_lift) -> TangentG (R^18, the Lie algebra g)
 *   where D_lift = d(Lambda)/d(xi) in R^{18 x 18} (dim g x dim M)
 */
struct Lift {
    TGInput u;

    explicit Lift(const TGInput& u);

    /// Lambda(xi, u) with optional Jacobian d(Lambda)/d(xi) in R^{18 x 18}
    Eigen::Matrix<double, 18, 1> operator()(
        const TGState& xi,
        Eigen::Matrix<double, 18, 18>* D_lift = nullptr) const;

private:
    /// W = wedge(w): input twist (Eq. 8)
    /// W = [wedge(w), a, v; 0_2x3, 0_2x1, 0_2x1]_5x5
    Eigen::Matrix<double, 5, 5> W_matrix(const TGInput& u) const;
    /// B = wedge(b): bias twist (Eq. 8)
    /// B = [wedge(b_w), b_a, b_v; 0_2x3, 0_2x1, 0_2x1]_5x5
    Eigen::Matrix<double, 5, 5> B_matrix(const TGState& xi) const;
    /// G = wedge(0,0,g): gravity in the acceleration slot (Eq. 8)
    /// G = [0_3x3, g, 0_3x1; 0_2x3, 0_2x1, 0_2x1]_5x5
    Eigen::Matrix<double, 5, 5> G_matrix(const TGInput& u) const;
    /// N = coupling matrix (Eq. 12c): pdot = v
    /// N = [0_3x3, 0_3x1, 0_3x1; 0_1x3, 0, 1; 0_1x3, 0, 0]_5x5
    Eigen::Matrix<double, 5, 5> N_matrix() const;
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
 *   psi(X, u)_w   = Ad_{A^{-1}}(w - a^vee) + Omega(A^{-1})   in R^9
 *   psi(X, u)_tau = Ad_{A^{-1}}(tau)                         in R^9
 *   psi(X, u)_g   = g_vec                                    in R^3
 *
 * where A = SE_2(3) part of X, a = se_2(3) part of X, w and tau from u,
 * Omega(A): SE_2(3) -> se_2(3)
 * Omega([R, v, p])= [0, 0, v], A = [R, v, p]
 */
struct InputOrbit {
    TGInput u;

    explicit InputOrbit(const TGInput& u);

    /// psi(X, u): maps group element X to transformed input in L
    TGInput operator()(const TGGroupElement& X) const;
};

} // namespace tgeqf