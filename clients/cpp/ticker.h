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
