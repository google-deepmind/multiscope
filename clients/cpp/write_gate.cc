#include "multiscope/clients/cpp/write_gate.h"

#include "third_party/absl/functional/bind_front.h"

namespace multiscope::internal {

WriteGate::WriteGate(Control* control,
                     gtl::AnySpan<const golang::stream::NodePath> paths)
    : control_(control),
      should_write_(false),
      write_registration_(control_->RegisterWriteCallback(
          paths, absl::bind_front(&WriteGate::SetShouldWrite, this))) {}

bool WriteGate::ShouldWrite() {
  absl::ReaderMutexLock lock(&mu_);
  return should_write_;
}

void WriteGate::SetShouldWrite(bool should_write) {
  absl::WriterMutexLock lock(&mu_);
  should_write_ = should_write;
}

}  // namespace multiscope::internal
