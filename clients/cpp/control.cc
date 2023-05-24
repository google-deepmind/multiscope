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

#include "multiscope/clients/cpp/control.h"

#include <functional>
#include <memory>

namespace multiscope::internal {
namespace {

// Implements a Control that effectively disables conditional writes by always
// returning `true`.
class AlwaysWrite : public Control {
 public:
  bool Write(const golang::stream::NodePath& path) const override {
    return true;
  }

  std::unique_ptr<Control::WriteRegistration> RegisterWriteCallback(
      gtl::AnySpan<const golang::stream::NodePath> paths,
      std::function<void(bool)> callback) override {
    callback(true);
    return std::make_unique<NoopWriteRegistration>();
  }

 private:
  class NoopWriteRegistration : public Control::WriteRegistration {
   public:
    ~NoopWriteRegistration() override = default;
  };
};
}  // namespace

// Pure virtual destructors must still be defined.
Control::WriteRegistration::~WriteRegistration() = default;

std::unique_ptr<Control> Control::AlwaysWrite() {
  return std::make_unique<class AlwaysWrite>();
}

}  // namespace multiscope::internal
