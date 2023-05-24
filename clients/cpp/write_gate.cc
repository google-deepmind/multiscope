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
