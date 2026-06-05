#pragma once
#include <gtsam/geometry/Rot3.h>
#include <gtsam/base/Lie.h>
#include <Eigen/Dense>

namespace tgeqf {

/**
 * Physical manifold state for the TG-EqF biased INS.
 *
 * xi = [T, b] in M := SE_2(3) x R^9
 * T = [R, p, v] in SE_2(3)
 * b = [b_omega, b_v, b_a] in R^9
 *
 * dim(M) = 9 (SE_2(3)) + 9 (biases) = 18
 *
 * Reference: Fornasier et al. [1], Section 3 / proposal Section 6.3
 */
struct TGState {
    // Navigation states
    gtsam::Rot3     R;   // SO(3): body orientation
    Eigen::Vector3d p;   // R^3:   position in global frame
    Eigen::Vector3d v;   // R^3:   velocity in global frame

    // IMU bias states (body frame)
    Eigen::Vector3d b_omega;  // gyroscope bias
    Eigen::Vector3d b_v;      // virtual velocity bias (from TG extension)
    Eigen::Vector3d b_a;      // accelerometer bias

    static constexpr int dimension = 18;

    /// Identity/origin state: R=I, all vectors zero
    static TGState identity();

    /// Compose two states (used internally for retract)
    // NOTE: this is NOT standard group composition; it is manifold retraction
    // TGState operator*(const TGState& other) const;

    /// Serialize T as a 5x5 SE_2(3) matrix
    Eigen::Matrix<double, 5, 5> T_matrix() const;

    /// Pack bias vector b = [b_omega; b_v; b_a] in R^9
    Eigen::Matrix<double, 9, 1> bias_vector() const;
};

} // namespace tgeqf

// ---------------------------------------------------------------------------
// GTSAM traits specialisation for TGState
// ---------------------------------------------------------------------------
namespace gtsam {

template <>
struct traits<tgeqf::TGState> {

    // Required type aliases
    using ManifoldType = tgeqf::TGState;
    using TangentVector = Eigen::Matrix<double, 18, 1>;
    using Jacobian      = Eigen::Matrix<double, 18, 18>;

    static constexpr int dimension = 18;
    static int GetDimension(const tgeqf::TGState&) { return dimension; }

    /// xi_ref (+) delta -> new state on M
    /// Retraction: apply a tangent perturbation at identity origin
    /// delta = [delta_R (3); delta_p (3); delta_v (3); delta_b (9)]
    static tgeqf::TGState Retract(
        const tgeqf::TGState& xi,
        const TangentVector& delta);

    /// Local(xi_ref, xi_est) -> tangent vector (error coordinates)
    /// Inverse of Retract: error in the fixed origin tangent space
    static TangentVector Local(
        const tgeqf::TGState& xi_ref,
        const tgeqf::TGState& xi_est);

    static tgeqf::TGState Identity() { return tgeqf::TGState::identity(); }

    static bool Equals(
        const tgeqf::TGState& a,
        const tgeqf::TGState& b,
        double tol = 1e-9);

    static void Print(
        const tgeqf::TGState& xi,
        const std::string& str = "");
};

} // namespace gtsam