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

#include "multiscope/clients/cpp/ticker.h"

#include <memory>
#include <utility>

#include "google/protobuf/any.proto.h"
#include "multiscope/protos/ticker.proto.h"
#include "multiscope/protos/tree.proto.h"
#include "multiscope/clients/cpp/utils.h"
#include "third_party/absl/status/statusor.h"
#include "util/gtl/any_span.h"
#include "util/task/status_macros.h"

namespace multiscope {

using deepmind::golang::stream::CreateNodeReply;
using deepmind::golang::stream::CreateNodeRequest;
using deepmind::golang::stream::NodePath;
using deepmind::golang::stream::PutNodeDataRequest;
using multiscope::ticker::TickerAction;
using multiscope::ticker::TickerAction_Tick;

Ticker::Ticker(internal::StreamClient* client, NodePath path)
    : client_(client), path_(std::move(path)) {}

absl::StatusOr<std::unique_ptr<Ticker>> Ticker::New(
    internal::StreamClient* client, absl::string_view name) {
  CreateNodeRequest request;
  request.mutable_path()->add_path(name);
  request.set_type(utils::GetWriters().ticker());
  ASSIGN_OR_RETURN(CreateNodeReply response, client->CreateNode(request));
  // Use new to get access to the private constructor.
  return absl::WrapUnique(new Ticker(client, response.path()));
}

absl::Status Ticker::Tick() {
  TickerAction ticker_action;
  // Set oneof `action` field as `Tick` action.
  *ticker_action.mutable_tick() = TickerAction_Tick();

  PutNodeDataRequest request;
  *request.mutable_data()->mutable_path() = path_;
  request.mutable_data()->mutable_pb()->PackFrom(ticker_action);

  return client_->PutNodeData(request).status();
}

const golang::stream::NodePath& Ticker::GetPath() const { return path_; }

}  // namespace multiscope
