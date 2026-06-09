#pragma once
#include <gtsam/geometry/Rot3.h>
#include <gtsam/base/Lie.h>
#include <Eigen/Dense>

#include <gtsam/geometry/ExtendedPose3.h>
#include <gtsam/geometry/ExtendedPose3-inl.h>

namespace tgeqf {

/**
 * Lie algebra element of SE_2(3): se_2(3)
 * Represented as a 5x5 matrix or as a 9-vector [w; a; v_]
 *
 * Reference: proposal Eq. (7b)
 */
struct se2_3 {
    // Bias block: [w; a; v]
    Eigen::Vector3d w;  // angular velocity component
    Eigen::Vector3d a;  // acceleration component
    Eigen::Vector3d v;  // velocity component

    /// Pack to R^9 vector
    Eigen::Matrix<double, 9, 1> vector() const;

    /// Unpack from R^9 vector
    static se2_3 from_vector(const Eigen::Matrix<double, 9, 1>& v);

    /// Wedge map: se2_3 -> 5x5 matrix in se_2(3)
    Eigen::Matrix<double, 5, 5> wedge() const;

    /// Vee map: 5x5 matrix -> se2_3
    static se2_3 vee(const Eigen::Matrix<double, 5, 5>& mat);
};

/**
 * Element of the symmetry group G = SE_2(3) ⋉ se_2(3)
 *
 * X = [A, a] where A in SE_2(3), a in se_2(3)
 *
 * Group product: XY = [A_X A_Y, a_X + Ad_{A_X}[a_Y]]    Eq. (12)
 * Inverse:       X^{-1} = [A^{-1}, Ad_{A^{-1}}[-a]]     Eq. (11)
 */
struct TGGroupElement {
    // Navigation block: A = [R, v, p] in SE_2(3)
    gtsam::Rot3     R;   // R: rotation component
    Eigen::Vector3d v;   // v: velocity component
    Eigen::Vector3d p;   // p: position component

    // Bias block: a = [w; a; v_] in se_2(3). This has the virtual velocity v_ as a component.
    // Note that this is semantically different from the lie algebra element of the navigation block {T_I}SE_2(3) = [w; a; v] in se_2(3)
    se2_3 a;               // [w, a, v_]

    static constexpr int dimension = 18; // dim(G_TG) = dim(SE_2(3)) + dim(se_2(3)) = 18

    /// Identity element
    static TGGroupElement Identity();

    /// Group product XY
    TGGroupElement operator*(const TGGroupElement& Y) const;

    /// Group inverse X^{-1}
    TGGroupElement inverse() const;

    /// Ad_{A_X} : se_2(3) -> se_2(3)
    /// Used in group product: (XY).a = a_X + Ad_{A_X}[a_Y]
    se2_3 Ad_A(const se2_3& xi) const;

    /// Ad_{A^{-1}} : se_2(3) -> se_2(3)
    /// Used in group inverse: X^{-1}.a = -Ad_{A^{-1}}[a]
    /// and in InputOrbit: psi(X, u)_w = Ad_{A^{-1}}(w - a^vee)
    se2_3 Ad_A_inv(const se2_3& xi) const;

    /// Exponential map: g -> G (from Lie algebra to group).
    static TGGroupElement Expmap(const Eigen::Matrix<double, 18, 1>& xi);

    /// Logarithm map: G -> g (from Lie group to algebra).
    Eigen::Matrix<double, 18, 1> Logmap() const;

    /// A = [R, v, p] as a 5x5 SE_2(3) matrix
    Eigen::Matrix<double, 5, 5> to_A_matrix() const;

    /// Unpack from 5x5 SE_2(3) matrix: [R, v, p]
    void A_from_matrix(const Eigen::Matrix<double, 5, 5>& A);
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

// ============================================================
// Helper functions
// ============================================================

namespace {

// SE_2(3) = ExtendedPose3<2>. Tangent vector is [omega; eta; alpha] in R^9.
using Se23 = gtsam::ExtendedPose3<2>;

inline Se23 toSe23(const tgeqf::TGGroupElement& X) {
  Eigen::Matrix<double, 3, 2> vp; // [v; p] in R^3x2
  vp.col(0) = X.v;
  vp.col(1) = X.p;
  return Se23(X.R, vp); // [R; v; p] in SE_2(3)
}

// Adjoint map of an SE_2(3) element acting on se_2(3):
// Ad_{(R,v,p)}(w, a, v) = (R*w, R*a + v×(R*w), R*v + p×(R*w))
//
//   Ad_A = [ R       0   0 ]
//          [ v^× R   R   0 ]
//          [ p^× R   0   R ]
inline Eigen::Matrix<double, 9, 9> Ad_SE23(const tgeqf::TGGroupElement& X) {
  // only uses the navigation block A = [R, v, p] in SE_2(3)
  const Eigen::Matrix3d R  = X.R.matrix();
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

// algebra adjoint map of a se_2(3) element:
// a = [w; a; v] in se_2(3)
// ad_a(xi) = [a, xi] = (a_w x xi_w,
//                       a_w x xi_v + a_v x xi_w,
//                       a_w x xi_a + a_a x xi_w)
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

}  // namespace

