#pragma once
#include <gtsam/geometry/Rot3.h>
#include <gtsam/base/Lie.h>
#include <Eigen/Dense>

namespace tgeqf {

/**
 * Lie algebra element of SE_2(3): se_2(3)
 * Represented as a 5x5 matrix or as a 9-vector [omega; v_tilde; a_tilde]
 *
 * Reference: proposal Eq. (7b)
 */
struct se2_3 {
    Eigen::Vector3d omega;    // angular velocity component
    Eigen::Vector3d v_tilde;  // virtual velocity component
    Eigen::Vector3d a_tilde;  // acceleration component

    /// Pack to R^9 vector
    Eigen::Matrix<double, 9, 1> vector() const;

    /// Unpack from R^9 vector
    static se2_3 from_vector(const Eigen::Matrix<double, 9, 1>& v);

    /// Wedge map: se2_3 -> 5x5 matrix in se_2(3)
    /// Reference: proposal Eq. (7b)
    Eigen::Matrix<double, 5, 5> wedge() const;

    /// Vee map: 5x5 matrix -> se2_3
    static se2_3 vee(const Eigen::Matrix<double, 5, 5>& mat);
};

/**
 * Element of the symmetry group G = SE_2(3) ⋉ se_2(3)  (Tangent Group of SE_2(3))
 *
 * X = [A, a] where A in SE_2(3), a in se_2(3)
 *
 * Group product: XY = [A_X A_Y, a_X + Ad_{A_X}[a_Y]]    Eq. (12)
 * Inverse:       X^{-1} = [A^{-1}, Ad_{A^{-1}}[-a]]     Eq. (11)
 *
 * Reference: proposal Section 6.4, Fornasier et al. [1]
 */
struct TGGroupElement {
    // Navigation part: C in SE_2(3)
    gtsam::Rot3     R_X;   // rotation component
    Eigen::Vector3d p_X;   // position component
    Eigen::Vector3d v_X;   // velocity component

    // Bias part: gamma in se_2(3)
    se2_3 a;               // [a_omega, a_v, a_a]

    static constexpr int dimension = 18; // dim(G_TG) = dim(SE_2(3)) + dim(se_2(3)) = 9 + 9

    /// Identity element
    static TGGroupElement Identity();

    /// Group product XY
    /// Reference: proposal Eq. (12)
    TGGroupElement operator*(const TGGroupElement& Y) const;

    /// Group inverse X^{-1}
    /// Reference: proposal Eq. (11)
    TGGroupElement inverse() const;

    /// Ad_{C} : se_2(3) -> se_2(3)  (adjoint of the SE_2(3) part on its Lie algebra)
    /// Used in group product: (XY).a = a_X + Ad_{C_X}[a_Y]
    se2_3 Ad_AT(const se2_3& xi) const;

    /// Ad_{C^{-1}} : se_2(3) -> se_2(3)
    /// Used in group inverse: X^{-1}.a = -Ad_{C^{-1}}[a]
    /// and in InputOrbit: psi(X, u)_w = Ad_{C^{-1}}(w - a^vee)
    se2_3 Ad_AT_inv(const se2_3& xi) const;

    /// Exponential map: g -> G (from Lie algebra to group)
    static TGGroupElement Expmap(const Eigen::Matrix<double, 18, 1>& xi);

    /// Logarithm map: G -> g
    Eigen::Matrix<double, 18, 1> Logmap() const;

    /// A_T as a 5x5 SE_2(3) matrix
    Eigen::Matrix<double, 5, 5> AT_matrix() const;
};

} // namespace tgeqf

// ---------------------------------------------------------------------------
// GTSAM traits specialisation for TGGroupElement
// ---------------------------------------------------------------------------
namespace gtsam {

template <>
struct traits<tgeqf::TGGroupElement> {

    using ManifoldType       = tgeqf::TGGroupElement;
    using TangentVector      = Eigen::Matrix<double, 18, 1>;
    using structure_category = lie_group_tag;
    using group_flavor       = multiplicative_group_tag;
    using ChartJacobian      = OptionalJacobian<18, 18>;

    static constexpr int dimension = 18;
    static int GetDimension(const tgeqf::TGGroupElement&) { return dimension; }

    // --- Testable ---
    static bool Equals(const tgeqf::TGGroupElement& X,
                       const tgeqf::TGGroupElement& Y, double tol = 1e-9);
    static void Print(const tgeqf::TGGroupElement& X,
                      const std::string& str = "");

    // --- Group (no Jacobians) ---
    static tgeqf::TGGroupElement Identity();

    static tgeqf::TGGroupElement Compose(
        const tgeqf::TGGroupElement& X, const tgeqf::TGGroupElement& Y,
        ChartJacobian Hx = {}, ChartJacobian Hy = {});

    static tgeqf::TGGroupElement Between(
        const tgeqf::TGGroupElement& X, const tgeqf::TGGroupElement& Y,
        ChartJacobian Hx = {}, ChartJacobian Hy = {});

    static tgeqf::TGGroupElement Inverse(
        const tgeqf::TGGroupElement& X, ChartJacobian H = {});

    // --- Lie group ---
    static tgeqf::TGGroupElement Expmap(
        const TangentVector& xi, ChartJacobian H = {});

    static TangentVector Logmap(
        const tgeqf::TGGroupElement& X, ChartJacobian H = {});

    static Eigen::Matrix<double, 18, 18> AdjointMap(
        const tgeqf::TGGroupElement& X);

    // --- Manifold ---
    static tgeqf::TGGroupElement Retract(
        const tgeqf::TGGroupElement& X, const TangentVector& xi);

    static TangentVector Local(
        const tgeqf::TGGroupElement& X, const tgeqf::TGGroupElement& Y);
};

template <>
struct traits<const tgeqf::TGGroupElement> : traits<tgeqf::TGGroupElement> {};

} // namespace gtsam