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

#include "multiscope/clients/cpp/writers/scalar_writer.h"

#include <memory>
#include <string>
#include <string_view>
#include <utility>

#include "google/protobuf/any.proto.h"
#include "multiscope/clients/cpp/utils.h"
#include "third_party/absl/container/flat_hash_map.h"
#include "third_party/absl/functional/bind_front.h"
#include "third_party/absl/status/statusor.h"
#include "util/task/status_macros.h"

namespace multiscope {

using ::deepmind::golang::stream::CreateNodeReply;
using ::deepmind::golang::stream::CreateNodeRequest;
using ::deepmind::golang::stream::NodePath;
using ::deepmind::golang::stream::PutNodeDataRequest;
using ::multiscope::scalars::ScalarAction;

namespace {

constexpr absl::string_view kDataNodeName = "data";
constexpr absl::string_view kSpecNodeName = "specification";

NodePath Append(const NodePath& path, absl::string_view name) {
  NodePath result(path);
  result.add_path(name);
  return result;
}

}  // namespace

ScalarWriter::ScalarWriter(internal::StreamClient* client,
                           internal::Control* control, NodePath path)
    : client_(client),
      path_(std::move(path)),
      data_path_(Append(path_, kDataNodeName)),
      spec_path_(Append(path_, kSpecNodeName)),
      write_gate_(
          internal::WriteGate(control, {path_, data_path_, spec_path_})) {
}

absl::StatusOr<std::unique_ptr<ScalarWriter>> ScalarWriter::New(
    internal::StreamClient* client, internal::Control* control,
    absl::string_view name, const Ticker* ticker) {
  CreateNodeRequest request;
  if (ticker) {
    request.mutable_path()->MergeFrom(ticker->GetPath());
  }
  request.mutable_path()->add_path(name);
  request.set_type(utils::GetWriters().scalar_writer());
  ASSIGN_OR_RETURN(CreateNodeReply response, client->CreateNode(request));

  // Use new to get access to the private constructor.
  return absl::WrapUnique(new ScalarWriter(client, control, response.path()));
}

absl::Status ScalarWriter::Write(
    const absl::flat_hash_map<std::string, double>& data) {
  ScalarAction action;
  action.mutable_data()->mutable_data()->insert(data.cbegin(), data.cend());
  return SendAction(action);
}

absl::Status ScalarWriter::SetHistoryLength(int length) {
  ScalarAction action;
  action.mutable_set_history_length()->set_length(length);
  return SendAction(action);
}

absl::Status ScalarWriter::SetSpec(std::string_view utf8_json_spec) {
  PutNodeDataRequest request;
  request.mutable_data()->mutable_path()->MergeFrom(spec_path_);
  *request.mutable_data()->mutable_raw() = utf8_json_spec;
  return client_->PutNodeData(request).status();
}

absl::Status ScalarWriter::SendAction(const ScalarAction& action) {
  PutNodeDataRequest request;
  request.mutable_data()->mutable_pb()->PackFrom(action);
  *request.mutable_data()->mutable_path() = path_;
  return client_->PutNodeData(request).status();
}

bool ScalarWriter::ShouldWrite() { return write_gate_.ShouldWrite(); }

};  // namespace multiscope
