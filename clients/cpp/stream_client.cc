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

#include "multiscope/clients/cpp/stream_client.h"

#include <grpcpp/channel.h>
#include <grpcpp/client_context.h>

#include <memory>
#include <string>
#include <utility>

#include "multiscope/protos/tree.grpc.pb.h"
#include "net/grpc/public/include/grpcpp/credentials_google.h"
#include "net/grpc/public/include/grpcpp/server_credentials_google.h"
#include "net/grpc/public/include/grpcpp/support/time_google.h"
#include "third_party/absl/status/status.h"
#include "third_party/absl/status/statusor.h"
#include "third_party/absl/strings/str_cat.h"
#include "third_party/absl/strings/str_format.h"
#include "third_party/grpc/include/grpcpp/grpcpp.h"
#include "util/task/status_macros.h"
#include "util/time/clock.h"
#include "util/tuple/tuple.h"
#include "webutil/url/url.h"

namespace multiscope::internal {

absl::StatusOr<std::unique_ptr<internal::StreamClient>> StreamClient::Create(
    absl::string_view url, util::Clock* clock) {
  URL target_url(url);
  if (!target_url.is_valid()) {
    return absl::InvalidArgumentError(absl::StrCat("invalid URL: ", url));
  }
  std::shared_ptr<grpc::ChannelCredentials> creds;
  if (target_url.protocol_piece() == "unix") {
    creds = grpc::experimental::LocalCredentials(grpc_local_connect_type::UDS);
  } else {
    creds = grpc::Loas2Credentials(grpc::Loas2CredentialsOptions());
  }

  // private constructor
  return absl::WrapUnique(new StreamClient(
      CreateChannel(std::string(url), std::move(creds)), clock));
}

StreamClient::StreamClient(std::shared_ptr<grpc::Channel> channel,
                           util::Clock* clock)
    : channel_(std::move(channel)),
      stub_(Stream::NewStub(channel_)),
      dm_env_stub_(dmenv::DMEnv::NewStub(channel_)),
      clock_(clock) {}

absl::StatusOr<CreateNodeReply> StreamClient::CreateNode(
    const CreateNodeRequest& request) {
  grpc::ClientContext context;
  CreateNodeReply response;
  RETURN_IF_ERROR(stub_->CreateNode(&context, request, &response));
  return response;
}

absl::StatusOr<NodeDataReply> StreamClient::GetNodeData(
    const NodeDataRequest& request) const {
  grpc::ClientContext context;
  NodeDataReply response;
  RETURN_IF_ERROR(stub_->GetNodeData(&context, request, &response));
  return response;
}

absl::StatusOr<PutNodeDataReply> StreamClient::PutNodeData(
    const PutNodeDataRequest& request) {
  grpc::ClientContext context;
  PutNodeDataReply response;
  RETURN_IF_ERROR(stub_->PutNodeData(&context, request, &response));
  return response;
}

void StreamClient::ActivePathsAsync(
    grpc::ClientContext* context, ActivePathsRequest* request,
    grpc::ClientReadReactor<ActivePathsReply>* reactor) {
  stub_->async()->ActivePaths(context, request, reactor);
}

absl::Status StreamClient::WaitTillReady(absl::Duration timeout) const {
  grpc::ClientContext context;
  context.set_deadline(clock_->TimeNow() + timeout);
  context.set_wait_for_ready(true);
  NodeDataReply unused_response;
  return stub_->GetNodeData(&context, NodeDataRequest(), &unused_response);
}

multiscope::dmenv::DMEnv::Stub& StreamClient::DmEnv() const {
  return *dm_env_stub_;
}

}  // namespace multiscope::internal
