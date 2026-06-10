// Round-trip guard for the scenario CSV writer: the column labelled eps_pos_x
// must actually carry the position component of the origin-frame error eps, and
// P_pos_00 the position covariance — i.e. the CSV header labels match the raw
// tangent write order [att, vel, pos, bg, ba, bv] (CODE_REVIEW F2b).

#include <CppUnitLite/TestHarness.h>
#include <gtsam/base/TestableAssertions.h>
#include <gtsam_unstable/tg_eqf/examples/TGEqFScenarioExample.h>

#include <limits>
#include <sstream>
#include <string>
#include <vector>

using namespace tgeqf;
using namespace tgeqf::examples;
using namespace gtsam;

namespace {

std::vector<std::string> split(const std::string& line) {
  std::vector<std::string> out;
  std::stringstream ss(line);
  std::string tok;
  while (std::getline(ss, tok, ',')) {
    out.push_back(tok);
  }
  return out;
}

int columnIndex(const std::vector<std::string>& header, const std::string& name) {
  for (size_t i = 0; i < header.size(); ++i) {
    if (header[i] == name) return static_cast<int>(i);
  }
  return -1;
}

}  // namespace

// The eps_* and P_* columns must line up with the TGState tangent slots:
//   eps_pos_x = eps(6)  (position), eps_vel_x = eps(3)  (velocity),
//   eps_ba_x  = eps(12) (accel bias), eps_bv_x = eps(15) (virtual bias).
TEST(TGScenarioCsv, EpsAndCovColumnsMatchTangentSlots) {
  // Reference at identity; estimate stays at identity so eps = Local(id, xi_true).
  const TGState xi_ref = TGState::identity();
  TGEqF filter(xi_ref, 0.04 * TGEqF::Covariance18::Identity());

  // Ground truth with distinct, non-degenerate components per block so a
  // mislabelled column would be caught (velocity != position, etc.).
  const Rot3 R_gt = Rot3::RzRyRx(0.1, -0.2, 0.3);
  const Vector3 v_gt(4.0, 5.0, 6.0);
  const Vector3 p_gt(1.0, 2.0, 3.0);
  const NavState gt(R_gt, p_gt, v_gt);
  const Vector3 true_bg(0.01, -0.02, 0.03);
  const Vector3 true_ba(-0.04, 0.05, -0.06);

  // Independently recompute the origin-frame error the harness logs.
  const TGState xi_true = trueState(gt, true_bg, true_ba);
  const TGState e = phi(filter.groupEstimate().inverse(), xi_true);
  const Eigen::Matrix<double, 18, 1> eps =
      traits<TGState>::Local(xi_ref, e);
  const Eigen::Matrix<double, 18, 18>& P = filter.errorCovariance();

  std::ostringstream out;
  writeCsvRow(out, /*t=*/0.0, gt, filter, xi_ref, true_bg, true_ba);

  const std::vector<std::string> header = split(csvHeader());
  const std::vector<std::string> row = split(out.str());
  LONGS_EQUAL(header.size(), row.size());

  auto val = [&](const std::string& name) {
    const int j = columnIndex(header, name);
    return j >= 0 ? std::stod(row[j])
                  : std::numeric_limits<double>::quiet_NaN();
  };

  // eps columns vs the tangent slots they claim to be.
  DOUBLES_EQUAL(eps(0), val("eps_att_x"), 1e-9);
  DOUBLES_EQUAL(eps(3), val("eps_vel_x"), 1e-9);
  DOUBLES_EQUAL(eps(6), val("eps_pos_x"), 1e-9);
  DOUBLES_EQUAL(eps(9), val("eps_bg_x"), 1e-9);
  DOUBLES_EQUAL(eps(12), val("eps_ba_x"), 1e-9);
  DOUBLES_EQUAL(eps(15), val("eps_bv_x"), 1e-9);

  // Covariance diagonal vs the same slots.
  DOUBLES_EQUAL(P(3, 3), val("P_vel_00"), 1e-9);
  DOUBLES_EQUAL(P(6, 6), val("P_pos_00"), 1e-9);
  DOUBLES_EQUAL(P(12, 12), val("P_ba_00"), 1e-9);
  DOUBLES_EQUAL(P(15, 15), val("P_bv_00"), 1e-9);
}

int main() {
  TestResult tr;
  return TestRegistry::runAllTests(tr);
}
