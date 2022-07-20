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
