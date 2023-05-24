// Copyright 2023 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "multiscope/clients/cpp/examples/scalar.h"

#include <cmath>
#include <cstdint>
#include <string>
#include <unordered_map>

#include "multiscope/clients/cpp/multiscope.h"
#include "multiscope/clients/cpp/server.h"
#include "net/util/ports.h"
#include "third_party/absl/container/flat_hash_map.h"

namespace multiscope {

/**
 * This example uses CHECK_OK and ValueOrDie everywhere, but in real application
 * code you probably want to ignore multiscope errors after logging at WARN
 * level instead.
 */
void RunScalarExample(int32_t max_steps, bool multiscope_strict_mode) {
  constexpr auto chart_spec = R"CHART_SPEC(
    {
      "$schema": "https://vega.github.io/schema/vega-lite/v4.json",
      "height": 400,
      "width": 400,
      "mark": "bar",
      "encoding": {
        "x": {"field": "Label",
              "type": "ordinal"},
        "y": {"field": "Value",
              "type": "quantitative",
              "scale": {"domain": [-1,1]}}
      }
    }
  )CHART_SPEC";

  int32_t multiscope_web_port = net_util::PickUnusedPortOrDie();
  std::string socket_name =
      multiscope::StartServer(/*http_port=*/multiscope_web_port,
                              /*loupe_port=*/0, /*ice_port=*/0)
          .ValueOrDie();
  auto multiscope = Multiscope::Connect(socket_name, absl::Seconds(10),
                                        multiscope_strict_mode)
                        .ValueOrDie();

  auto ticker = multiscope->Ticker("main").ValueOrDie();
  auto scalar_writer = multiscope->ScalarWriter("main", *ticker).ValueOrDie();

  CHECK_OK(scalar_writer->SetHistoryLength(123));
  CHECK_OK(scalar_writer->SetSpec(chart_spec));

  absl::flat_hash_map<std::string, double> data;
  for (int time = 0; time < max_steps; time++) {
    // This is an example of how to do a conditional write to avoid overhead
    // when no one is looking at the multiscope UI.
    if (scalar_writer->ShouldWrite()) {
      data["sin"] = sin(time * M_PI / 180.0);
      data["cos"] = cos(time * M_PI / 180.0);
      CHECK_OK(scalar_writer->Write(data));
    }
    CHECK_OK(ticker->Tick());
    // TODO(abutcher): remove once setting period on the clock is
    // implemented
    absl::SleepFor(absl::Seconds(1));
  }
}

}  // namespace multiscope
