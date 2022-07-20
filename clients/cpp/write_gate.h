#ifndef MULTISCOPE_CLIENTS_CPP_WRITE_GATE_H_
#define MULTISCOPE_CLIENTS_CPP_WRITE_GATE_H_

#include <memory>

#include "multiscope/clients/cpp/control.h"

namespace multiscope::internal {

// WriteGate encapsulates the logic to do a fast boolean check to determine
// whether to write, and keep this updated by registering a write callback with
// a Control object.
class WriteGate {
 public:
  // `paths` are passed into the write callback registered with `control`.
  // `control` must outlive this WriteGate.
  WriteGate(Control* control,
            gtl::AnySpan<const golang::stream::NodePath> paths);

  // Whether to write to the `paths` provided in the constructor.
  bool ShouldWrite();

  // WriteGate is neither copyable nor movable.
  WriteGate(const WriteGate&) = delete;
  WriteGate& operator=(const WriteGate&) = delete;

 private:
  void SetShouldWrite(bool should_write);

  Control* const control_;
  absl::Mutex mu_;
  bool should_write_ ABSL_GUARDED_BY(mu_);
  // Ensure this is declared last so it's destroyed first and the write callback
  // is unregistered before destroying any other members.
  const std::unique_ptr<Control::WriteRegistration> write_registration_;
};

}  // namespace multiscope::internal

#endif  // MULTISCOPE_CLIENTS_CPP_WRITE_GATE_H_
