#include "multiscope/clients/cpp/examples/scalar.h"

#include "testing/base/public/gunit.h"

namespace multiscope {
namespace {

TEST(ScalarTest, ScalarExampleRunsWithoutErrors) {
  ::multiscope::RunScalarExample(3,
                                           /* multiscope_strict_mode = */ true);
}

}  // namespace
}  // namespace multiscope

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
