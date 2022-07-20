#ifndef MULTISCOPE_CLIENTS_CPP_WRITERS_SCALAR_WRITER_H_
#define MULTISCOPE_CLIENTS_CPP_WRITERS_SCALAR_WRITER_H_

#include <memory>
#include <string>
#include <string_view>
#include <unordered_map>

#include "multiscope/protos/scalar.proto.h"
#include "multiscope/protos/tree.proto.h"
#include "multiscope/clients/cpp/stream_client.h"
#include "multiscope/clients/cpp/ticker.h"
#include "multiscope/clients/cpp/write_gate.h"
#include "third_party/absl/container/flat_hash_map.h"
#include "third_party/absl/status/statusor.h"

namespace multiscope {

using ::multiscope::scalars::ScalarAction;

// ScalarWriter is used to send scalar-valued data to a multiscope server for
// visualization. See http://go/multiscope.
class ScalarWriter {
 public:
  // Factory method returns a StatusOr indicating failure if the ScalarWriter
  // could not be initialized remotely.
  // The StreamClient and Control must outlive this ScalarWriter. The passed in
  // Ticker must persist for the duration of the call, but ScalarWriter doesn't
  // rely on it afterwards.
  static absl::StatusOr<std::unique_ptr<ScalarWriter>> New(
      internal::StreamClient* client, internal::Control* control,
      absl::string_view name, const Ticker* ticker = nullptr);

  // Write labeled data series to the multiscope server.
  absl::Status Write(const absl::flat_hash_map<std::string, double>& data);

  // Whether data should be written for this writer. Can be used to write
  // conditionally, determined by the internal::Control used during
  // construction.
  bool ShouldWrite();

  // Sets the number of samples that are retained and visualized
  // on the multiscope server.
  absl::Status SetHistoryLength(int length);

  // Controls the visualization of the data sent to the multiscope server.
  // utf8_json_spec is a UTF-8 encoded altair chart specification. The
  // json specification is sent to the multiscope server which updates the
  // visualization. See: http://go/multiscope#specifying-a-chart-specification.
  absl::Status SetSpec(std::string_view utf8_json_spec);

  ScalarWriter(const ScalarWriter&) = delete;
  ScalarWriter& operator=(const ScalarWriter&) = delete;

 private:
  ScalarWriter(internal::StreamClient* client, internal::Control* control,
               golang::stream::NodePath path);

  // Packs a ScalarAction into a multiscope PutNodeDataRequest and sends it to
  // the multsicope server.
  absl::Status SendAction(const ScalarAction& action);

  internal::StreamClient* const client_;
  const golang::stream::NodePath path_;
  const golang::stream::NodePath data_path_;
  const golang::stream::NodePath spec_path_;
  internal::WriteGate write_gate_;
};
}  // namespace multiscope

#endif  // MULTISCOPE_CLIENTS_CPP_WRITERS_SCALAR_WRITER_H_
