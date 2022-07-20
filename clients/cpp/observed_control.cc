#include "multiscope/clients/cpp/observed_control.h"

#include <functional>
#include <memory>
#include <utility>

#include "base/logging.h"
#include "multiscope/protos/tree.proto.h"
#include "third_party/absl/functional/bind_front.h"

namespace multiscope::internal {

using ::deepmind::golang::stream::NodePath;

namespace {

bool ShouldWriteAny(gtl::AnySpan<const NodePath> paths,
                    gtl::SetView<const NodePath> observed_paths) {
  for (auto& path : paths) {
    if (observed_paths.contains(path)) {
      return true;
    }
  }
  return false;
}

}  // namespace

ObservedPathsReactor::ObservedPathsReactor(
    StreamClient* client,
    std::function<void(gtl::AnySpan<const NodePath>)> callback)
    : client_(client),
      cancelled_(false),
      done_(false),
      callback_(std::move(callback)) {
  client_->ActivePathsAsync(&context_, &active_paths_request_, this);
  StartRead(&active_paths_reply_);
}

ObservedPathsReactor::~ObservedPathsReactor() {
  TryCancel();
  WaitForDone();
}

void ObservedPathsReactor::WaitForDone() {
  absl::MutexLock lock(&mu_);
  while (!mu_.AwaitWithTimeout(absl::Condition(&done_), absl::Seconds(30))) {
    LOG(WARNING) << "ObservedPaths gRPC stream didn't complete within timeout; "
                    "continuing to wait";
  }
}

void ObservedPathsReactor::TryCancel() {
  absl::MutexLock lock(&mu_);
  cancelled_ = true;
  context_.TryCancel();
}

void ObservedPathsReactor::OnReadDone(bool ok) {
  if (!ok) {
    return;
  }
  callback_(active_paths_reply_.paths());
  StartRead(&active_paths_reply_);
}

void ObservedPathsReactor::OnDone(const grpc::Status& status) {
  absl::MutexLock lock(&mu_);
  done_ = true;
  // OnDone is called in 2 cases: either the multiscope server has disconnected
  // or this Control object is being destroyed. Disable all writes for the first
  // case, and in the second case all writers have been destroyed already.
  callback_({});
  if (cancelled_) {
    return;
  }
  if (!status.ok()) {
    LOG(ERROR) << "ObservedPaths request returned with error: " << status;
  }
  LOG(WARNING) << "ObservedPaths request unexpectedly done, disabling all "
                  "writes indefinitely.";
}

ObservedControl::WriteRegistration::WriteRegistration(
    int registration_id,
    multiscope::internal::ObservedControl* control)
    : registration_id_(registration_id), control_(control) {}

ObservedControl::WriteRegistration::~WriteRegistration() {
  absl::WriterMutexLock lock(&control_->mu_);
  control_->write_callbacks_.erase(registration_id_);
}

ObservedControl::ObservedControl(StreamClient* client)
    : next_write_callback_id_(0),
      reactor_(internal::ObservedPathsReactor(
          client, absl::bind_front(&ObservedControl::Update, this))) {
  reactor_.StartCall();
}

bool ObservedControl::Write(const NodePath& path) const {
  absl::ReaderMutexLock lock(&mu_);
  return observed_paths_.contains(path);
}

std::unique_ptr<Control::WriteRegistration>
ObservedControl::RegisterWriteCallback(gtl::AnySpan<const NodePath> paths,
                                       std::function<void(bool)> callback) {
  absl::WriterMutexLock lock(&mu_);
  int id = next_write_callback_id_++;
  // Ensure the newly registered callback is called with initial state.
  callback(ShouldWriteAny(paths, observed_paths_));
  write_callbacks_.insert(std::make_pair(
      id, PathCallback{std::vector<NodePath>(paths.cbegin(), paths.cend()),
                       std::move(callback)}));
  return std::make_unique<ObservedControl::WriteRegistration>(id, this);
}

void ObservedControl::Update(gtl::AnySpan<const NodePath> observed_paths) {
  absl::WriterMutexLock lock(&mu_);
  observed_paths_.clear();
  observed_paths_.insert(observed_paths.begin(), observed_paths.end());
  for (const auto& [_, cb] : write_callbacks_) {
    cb.callback(ShouldWriteAny(cb.paths, observed_paths_));
  }
}

}  // namespace multiscope::internal
