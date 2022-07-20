#ifndef MULTISCOPE_CLIENTS_CPP_STREAM_CLIENT_H_
#define MULTISCOPE_CLIENTS_CPP_STREAM_CLIENT_H_

#include <grpcpp/client_context.h>
#include <grpcpp/impl/codegen/client_callback.h>

#include <memory>

#include "multiscope/protos/tree.grpc.pb.h"
#include "multiscope/protos/tree.proto.h"
#include "third_party/absl/status/status.h"
#include "third_party/absl/status/statusor.h"
#include "third_party/absl/strings/string_view.h"
#include "util/time/clock.h"

namespace multiscope::internal {

using deepmind::golang::stream::ActivePathsReply;
using deepmind::golang::stream::ActivePathsRequest;
using deepmind::golang::stream::CreateNodeReply;
using deepmind::golang::stream::CreateNodeRequest;
using deepmind::golang::stream::NodeDataReply;
using deepmind::golang::stream::NodeDataRequest;
using deepmind::golang::stream::PutNodeDataReply;
using deepmind::golang::stream::PutNodeDataRequest;
using deepmind::golang::stream::grpc_gen::Stream;

// StreamClient wraps the multiscope stream API defined in
// multiscope/protos/tree.proto. Additionally it exposes typed
// clients for certain node types, e.g. DmEnv().
class StreamClient {
 public:
  // Construct a StreamClient from a URL where the multiscope gRPC server is
  // listening.
  static absl::StatusOr<std::unique_ptr<StreamClient>> Create(
      absl::string_view url, util::Clock* clock = util::Clock::RealClock());

  // Disable copy (and move).
  StreamClient(const StreamClient&) = delete;
  StreamClient& operator=(const StreamClient&) = delete;

  // Virtual for mocking.
  virtual ~StreamClient() = default;

  // See l/d/golang/stream/stream.proto for documentation for these methods.

  absl::StatusOr<CreateNodeReply> CreateNode(const CreateNodeRequest& request);

  absl::StatusOr<NodeDataReply> GetNodeData(
      const NodeDataRequest& request) const;

  absl::StatusOr<PutNodeDataReply> PutNodeData(
      const PutNodeDataRequest& request);

  void ActivePathsAsync(grpc::ClientContext* context,
                        ActivePathsRequest* request,
                        grpc::ClientReadReactor<ActivePathsReply>* reactor);

  // Ensures the server is running and we can successfully connect to
  // it.
  absl::Status WaitTillReady(absl::Duration timeout) const;

 private:
  StreamClient(std::shared_ptr<grpc::Channel> channel, util::Clock* clock);

  const std::shared_ptr<grpc::Channel> channel_;
  const std::unique_ptr<Stream::Stub> stub_;
  const std::unique_ptr<multiscope::dmenv::DMEnv::Stub> dm_env_stub_;
  util::Clock* const clock_;
};

}  // namespace multiscope::internal

#endif  // MULTISCOPE_CLIENTS_CPP_STREAM_CLIENT_H_
