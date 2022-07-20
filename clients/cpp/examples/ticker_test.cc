#include "multiscope/clients/cpp/examples/ticker.h"

#include "testing/base/public/gunit.h"

namespace multiscope {
namespace {

TEST(TickerTest, TickerExampleRunsWithoutErrors) {
  ::multiscope::RunTickerExample(10,
                                           /* multiscope_strict_mode = */ true);
}

}  // namespace
}  // namespace multiscope

int main(int argc, char **argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
