#include "multiscope/clients/cpp/multiscope.h"

#include <memory>
#include <utility>

#include "multiscope/clients/cpp/observed_control.h"
#include "multiscope/clients/cpp/stream_client.h"
#include "third_party/absl/status/statusor.h"
#include "util/task/status_macros.h"

namespace multiscope {
namespace {

// Holds singleton connection state, and must outlive all created writers.
class MultiscopeImpl final : public Multiscope {
 public:
  MultiscopeImpl(std::unique_ptr<internal::StreamClient> client,
                 std::unique_ptr<internal::Control> control)
      : client_(std::move(client)), control_(std::move(control)) {}
  // MultiscopeImpl is neither copyable nor movable.
  MultiscopeImpl(const MultiscopeImpl&) = delete;
  MultiscopeImpl& operator=(const MultiscopeImpl&) = delete;

  absl::StatusOr<std::unique_ptr<class Ticker>> Ticker(
      absl::string_view name) override {
    return Ticker::New(client_.get(), name);
  }

  absl::StatusOr<std::unique_ptr<class ScalarWriter>> ScalarWriter(
      absl::string_view name) override {
    return ScalarWriter::New(client_.get(), control_.get(), name);
  }
  absl::StatusOr<std::unique_ptr<class ScalarWriter>> ScalarWriter(
      absl::string_view name, const class Ticker& parent) override {
    return ScalarWriter::New(client_.get(), control_.get(), name, &parent);
  };

 private:
  // Ensure client_ is listed first so it's deleted last.
  const std::unique_ptr<internal::StreamClient> client_;
  const std::unique_ptr<internal::Control> control_;
};

}  // namespace

absl::StatusOr<std::unique_ptr<Multiscope>> Multiscope::Connect(
    absl::string_view url, absl::Duration timeout, bool strict_mode) {
  ASSIGN_OR_RETURN(std::unique_ptr<internal::StreamClient> client,
                   internal::StreamClient::Create(url));
  RETURN_IF_ERROR(client->WaitTillReady(timeout));
  std::unique_ptr<internal::Control> control;
  if (strict_mode) {
    control = internal::Control::AlwaysWrite();
  } else {
    control = std::make_unique<internal::ObservedControl>(client.get());
  }
  return std::make_unique<MultiscopeImpl>(std::move(client),
                                          std::move(control));
}

}  // namespace multiscope
