/*
 * Copyright 2023 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef MULTISCOPE_CLIENTS_CPP_TICKER_H_
#define MULTISCOPE_CLIENTS_CPP_TICKER_H_

#include <memory>

#include "multiscope/clients/cpp/control.h"
#include "multiscope/clients/cpp/stream_client.h"
#include "third_party/absl/status/statusor.h"
#include "third_party/absl/strings/string_view.h"

namespace multiscope {

// Ticker wraps a remote Multiscope Ticker.
class Ticker {
 public:
  // disable copy (and move)
  Ticker(const Ticker&) = delete;
  Ticker& operator=(const Ticker&) = delete;

  // Factory method returns a StatusOr indicating failure if the clock could not
  // be initialized remotely.
  // The StreamClient must outlive the Ticker.
  static absl::StatusOr<std::unique_ptr<Ticker>> New(
      internal::StreamClient* client, absl::string_view name);
  // Tick ticks the ticker to the next step.
  absl::Status Tick();
  // Get the node path for this Ticker.
  const golang::stream::NodePath& GetPath() const;

 private:
  Ticker(internal::StreamClient* client, golang::stream::NodePath path);

  internal::StreamClient* const client_;
  const golang::stream::NodePath path_;
};

}  // namespace multiscope

#endif  // MULTISCOPE_CLIENTS_CPP_TICKER_H_
