#ifndef MULTISCOPE_CLIENTS_CPP_OBSERVED_CONTROL_H_
#define MULTISCOPE_CLIENTS_CPP_OBSERVED_CONTROL_H_

#include <grpc/grpc.h>

#include <functional>
#include <memory>
#include <vector>

#include "multiscope/clients/cpp/control.h"
#include "multiscope/clients/cpp/stream_client.h"
#include "net/proto2/contrib/hashcode/hashcode.h"
#include "third_party/absl/container/flat_hash_map.h"
#include "third_party/absl/container/flat_hash_set.h"
#include "util/gtl/any_span.h"

namespace multiscope::internal {

// ObservedPathsReactor calls the multiscope server to get a stream of observed
// node paths, and calls the provided callback for each message.
class ObservedPathsReactor
    : public grpc::ClientReadReactor<golang::stream::ActivePathsReply> {
 public:
  // The constructor makes a grpc request to fetch observed paths using
  // `client`. The StreamClient must outlive this object.
  ObservedPathsReactor(
      StreamClient* client,
      std::function<void(gtl::AnySpan<const golang::stream::NodePath>)>
          callback);
  // This object *must* outlive the grpc request, but there is no guaranteed way
  // to terminate the request, so the best we can do is call `TryCancel` and
  // then `WaitForDone` in the destructor.
  ~ObservedPathsReactor() override;
  // Waits indefinitely for the underlying grpc request to be completed.
  void WaitForDone();
  // Best-effort attempt to cancel the underlying grpc request. Succeeds if the
  // request is already complete.
  void TryCancel();

  void OnDone(const ::grpc::Status& status) override;
  void OnReadDone(bool ok) override;

 private:
  StreamClient* const client_;
  grpc::ClientContext context_;
  absl::Mutex mu_;
  bool cancelled_ ABSL_GUARDED_BY(mu_);
  bool done_ ABSL_GUARDED_BY(mu_);
  const std::function<void(gtl::AnySpan<const golang::stream::NodePath>)>
      callback_;
  golang::stream::ActivePathsRequest active_paths_request_;
  golang::stream::ActivePathsReply active_paths_reply_;
};

// ObservedControl implements controlling writer behaviour depending on which
// node paths are being observed in the multiscope UI.
class ObservedControl : public Control {
 public:
  // The StreamClient must outlive this ObservedControl.
  explicit ObservedControl(StreamClient* client);

  bool Write(const golang::stream::NodePath& path) const override;

  std::unique_ptr<Control::WriteRegistration> RegisterWriteCallback(
      gtl::AnySpan<const golang::stream::NodePath> paths,
      std::function<void(bool)> callback) override;

  // Store the latest set of observed paths.
  void Update(gtl::AnySpan<const golang::stream::NodePath> observed_paths);

  // ObservedControl is neither copyable nor movable.
  ObservedControl(const ObservedControl&) = delete;
  ObservedControl& operator=(const ObservedControl&) = delete;

 private:
  struct PathCallback {
    const std::vector<golang::stream::NodePath> paths;
    const std::function<void(bool)> callback;
  };

  class WriteRegistration : public Control::WriteRegistration {
   public:
    // The passed in ObservedControl must outlive this object.
    WriteRegistration(int registration_id, ObservedControl* control);

    // Unregisters the provided registration ID on destruction.
    ~WriteRegistration() override;

   private:
    const int registration_id_;
    ObservedControl* const control_;
  };

  absl::flat_hash_set<const golang::stream::NodePath,
                      proto2::contrib::hashcode::ProtoHash,
                      proto2::contrib::hashcode::ProtoEqual>
      observed_paths_ ABSL_GUARDED_BY(mu_);
  absl::flat_hash_map<int, PathCallback> write_callbacks_ ABSL_GUARDED_BY(mu_);
  int next_write_callback_id_ ABSL_GUARDED_BY(mu_);
  mutable absl::Mutex mu_;
  // Ensure this is declared last so it is destroyed first, and does not access
  // any other members after they have been destroyed.
  internal::ObservedPathsReactor reactor_;
};

}  // namespace multiscope::internal

#endif  // MULTISCOPE_CLIENTS_CPP_OBSERVED_CONTROL_H_
