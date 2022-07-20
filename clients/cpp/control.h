#ifndef MULTISCOPE_CLIENTS_CPP_CONTROL_H_
#define MULTISCOPE_CLIENTS_CPP_CONTROL_H_

#include <functional>
#include <memory>

#include "multiscope/protos/tree.proto.h"
#include "third_party/absl/strings/string_view.h"
#include "third_party/absl/types/span.h"
#include "util/gtl/any_span.h"

namespace multiscope::internal {

// Interface for controlling multiscope writer behaviour. Named "control" as
// it's orthogonal to the "data" plane.
class Control {
 public:
  virtual ~Control() = default;

  // Whether to write any data for the given node path.
  virtual bool Write(const golang::stream::NodePath& path) const = 0;

  // Implements RAII for a callback registered via
  // `RegisterWriteCallback`. Unregisters the callback on destruction.
  class WriteRegistration {
   public:
    WriteRegistration() = default;
    virtual ~WriteRegistration() = 0;
    // Not copyable or movable.
    WriteRegistration(const WriteRegistration&) = delete;
    WriteRegistration& operator=(const WriteRegistration&) = delete;
  };

  // Calls `callback` with the logical OR of the result of `Write(path)`
  // whenever it changes for any path in `paths`. Guaranteed to be called at
  // least once per change, but may be called multiple times.
  //
  // Returns a WriteRegistration that unregisters the callback when destroyed.
  // Ensure this happens before releasing resources accessed by the callback.
  [[nodiscard]] virtual std::unique_ptr<WriteRegistration>
  RegisterWriteCallback(gtl::AnySpan<const golang::stream::NodePath> paths,
                        std::function<void(bool)> callback) = 0;

  // Returns a Control that effectively disables conditional writes by always
  // returning `true`.
  static std::unique_ptr<Control> AlwaysWrite();
};

}  // namespace multiscope::internal

#endif  // MULTISCOPE_CLIENTS_CPP_CONTROL_H_
