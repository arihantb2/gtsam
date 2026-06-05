#include <CppUnitLite/TestHarness.h>
#include <gtsam/base/Manifold.h>
#include <gtsam/base/Group.h>
#include <gtsam/base/Lie.h>
#include <gtsam_unstable/tg_eqf/State.h>
#include <gtsam_unstable/tg_eqf/Group.h>

// ---------------------------------------------------------------------------
// Compile-time concept checks.
// These expand to template class instantiations that trigger GTSAM_CONCEPT_USAGE
// bodies. A missing trait method causes a compile error here, not at runtime.
// ---------------------------------------------------------------------------

GTSAM_CONCEPT_MANIFOLD_INST(tgeqf::TGState)
GTSAM_CONCEPT_GROUP_INST(tgeqf::TGGroupElement)
GTSAM_CONCEPT_LIE_INST(tgeqf::TGGroupElement)

// ---------------------------------------------------------------------------
// Runtime dimension checks (fast sanity, no throws needed)
// ---------------------------------------------------------------------------

// Checks that traits<TGState>::dimension == 18 at compile time and that
// GetDimension() agrees at runtime; dim(M) = dim(SE_2(3)) + dim(R^9) = 9+9.
TEST(TGConcepts, StateDimension) {
  EXPECT_LONGS_EQUAL(18, gtsam::traits<tgeqf::TGState>::dimension);
  EXPECT_LONGS_EQUAL(18, gtsam::traits<tgeqf::TGState>::GetDimension(
                             tgeqf::TGState::identity()));
}

// Checks that traits<TGGroupElement>::dimension == 18;
// dim(G_TG) = dim(SE_2(3)) + dim(se_2(3)) = 9+9.
TEST(TGConcepts, GroupDimension) {
  EXPECT_LONGS_EQUAL(18, gtsam::traits<tgeqf::TGGroupElement>::dimension);
}

// Checks that TGState's tangent vector type has static compile-time size 18,
// ensuring fixed-size Eigen allocation in filter operations.
TEST(TGConcepts, StateTangentVectorSize) {
  using TV = gtsam::traits<tgeqf::TGState>::TangentVector;
  EXPECT_LONGS_EQUAL(18, TV::SizeAtCompileTime);
}

// Checks that TGGroupElement's tangent vector type has static compile-time size 18.
TEST(TGConcepts, GroupTangentVectorSize) {
  using TV = gtsam::traits<tgeqf::TGGroupElement>::TangentVector;
  EXPECT_LONGS_EQUAL(18, TV::SizeAtCompileTime);
}

// Checks that TGState's structure_category derives from manifold_tag,
// as required by the GTSAM Manifold concept for use as filter state space M.
TEST(TGConcepts, StateStructureCategory) {
  using cat = gtsam::traits<tgeqf::TGState>::structure_category;
  EXPECT(bool(std::is_base_of_v<gtsam::manifold_tag, cat>));
}

// Checks that TGGroupElement's structure_category derives from lie_group_tag,
// satisfying the GTSAM LieGroup concept for use as the symmetry group G.
TEST(TGConcepts, GroupStructureCategory) {
  using cat = gtsam::traits<tgeqf::TGGroupElement>::structure_category;
  EXPECT(bool(std::is_base_of_v<gtsam::lie_group_tag, cat>));
}

// Checks that TGGroupElement's group_flavor is multiplicative_group_tag,
// required so GTSAM's group utilities use operator* for composition.
TEST(TGConcepts, GroupFlavorIsMultiplicative) {
  using flavor = gtsam::traits<tgeqf::TGGroupElement>::group_flavor;
  EXPECT(bool(std::is_same_v<flavor, gtsam::multiplicative_group_tag>));
}

/* ************************************************************************* */
int main() {
  TestResult tr;
  return TestRegistry::runAllTests(tr);
}
