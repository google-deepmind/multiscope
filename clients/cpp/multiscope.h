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

#ifndef MULTISCOPE_CLIENTS_CPP_MULTISCOPE_H_
#define MULTISCOPE_CLIENTS_CPP_MULTISCOPE_H_

#include <memory>

#include "multiscope/clients/cpp/stream_client.h"
#include "multiscope/clients/cpp/ticker.h"
#include "multiscope/clients/cpp/writers/dm_env_rpc_writer.h"
#include "multiscope/clients/cpp/writers/image_writer.h"
#include "multiscope/clients/cpp/writers/scalar_writer.h"
#include "third_party/absl/status/statusor.h"

namespace multiscope {

// A Multiscope object represents a live connection to the Multiscope server.
// It's used to instantiate multiscope writers. This object must outlive any
// instantiated writers, e.g. ones created by ImageWriter(), ScalarWriter(),
// etc.
class Multiscope {
 public:
  virtual ~Multiscope() = default;

  // See Ticker::New.
  virtual absl::StatusOr<std::unique_ptr<Ticker>> Ticker(
      absl::string_view name) = 0;
  // See ScalarWriter::New.
  virtual absl::StatusOr<std::unique_ptr<class ScalarWriter>> ScalarWriter(
      absl::string_view name) = 0;
  virtual absl::StatusOr<std::unique_ptr<class ScalarWriter>> ScalarWriter(
      absl::string_view name, const class Ticker& parent) = 0;

  // Construct a Multiscope object from the URL where the multiscope gRPC
  // server is listening. Ensures the multiscope server is running and we
  // can connect to it within `timeout`.
  //
  // When `strict_mode` is true, performance and convenience features like
  // conditional writes and suppressing exceptions are disabled. This is
  // useful for debugging multiscope related issues, and in tests.
  static absl::StatusOr<std::unique_ptr<Multiscope>> Connect(
      absl::string_view url, absl::Duration timeout = absl::Seconds(10),
      bool strict_mode = false);
};

}  // namespace multiscope

#endif  // MULTISCOPE_CLIENTS_CPP_MULTISCOPE_H_
