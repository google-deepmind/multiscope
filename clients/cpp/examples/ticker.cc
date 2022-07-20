#include "multiscope/clients/cpp/examples/ticker.h"

#include <cstdint>
#include <memory>
#include <string>

#include "base/integral_types.h"
#include "base/logging.h"
#include "multiscope/clients/cpp/multiscope.h"
#include "multiscope/clients/cpp/server.h"
#include "net/util/ports.h"
#include "third_party/absl/status/status.h"
#include "third_party/absl/time/time.h"
#include "util/task/status.h"

namespace multiscope {

/**
 * This example uses CHECK_OK and ValueOrDie everywhere, but in real application
 * code you probably want to ignore multiscope errors after logging at WARN
 * level instead.
 */
void RunTickerExample(int32_t max_steps, bool multiscope_strict_mode) {
  int32_t multiscope_web_port = net_util::PickUnusedPortOrDie();
  std::string socket_name =
      multiscope::StartServer(/*http_port=*/multiscope_web_port,
                              /*loupe_port=*/0, /*ice_port=*/0)
          .ValueOrDie();
  auto multiscope = Multiscope::Connect(socket_name, absl::Seconds(10),
                                        multiscope_strict_mode)
                        .ValueOrDie();

  auto ticker = multiscope->Ticker("main").ValueOrDie();

  for (int time = 0; time < max_steps; time++) {
    CHECK_OK(ticker->Tick());
  }
}

}  // namespace multiscope
