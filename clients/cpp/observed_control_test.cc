#include "multiscope/clients/cpp/observed_control.h"

#include <cstdint>
#include <memory>
#include <string>
#include <utility>

#include "multiscope/clients/cpp/multiscope.h"
#include "multiscope/clients/cpp/server.h"
#include "multiscope/clients/cpp/stream_client.h"
#include "net/util/ports.h"
#include "testing/base/public/gmock.h"
#include "testing/base/public/gunit.h"

namespace multiscope::internal {
namespace {

using ::deepmind::golang::stream::NodeDataRequest;

class ObservedControlTest : public testing::Test {
 protected:
  ObservedControlTest() {
    int32_t multiscope_web_port = net_util::PickUnusedPortOrDie();
    absl::StatusOr<std::string> url = multiscope::StartServer(
        /*http_port=*/multiscope_web_port, /*loupe_port=*/0, /*ice_port=*/0);
    CHECK_OK(url);
    absl::StatusOr<std::unique_ptr<internal::StreamClient>> client =
        internal::StreamClient::Create(*url);
    CHECK_OK(client);
    client_ = std::move(*client);
    CHECK_OK(client_->WaitTillReady(absl::Seconds(10)));
    control_ = std::make_unique<ObservedControl>(client_.get());
  }

  std::unique_ptr<internal::StreamClient> client_;
  std::unique_ptr<internal::Control> control_;
};

TEST_F(ObservedControlTest, TracksObservedPaths) {
  absl::Notification should_write_to_path;

  golang::stream::NodePath path;
  path.add_path("path");
  std::unique_ptr<internal::Control::WriteRegistration> registration =
      control_->RegisterWriteCallback({path}, [&](bool should_write) {
        if (should_write && !should_write_to_path.HasBeenNotified()) {
          should_write_to_path.Notify();
        } else if (!should_write) {  // should not stop writing once started
          EXPECT_FALSE(should_write_to_path.HasBeenNotified());
        }
      });
  EXPECT_FALSE(
      should_write_to_path.WaitForNotificationWithTimeout(absl::Seconds(1)));
  EXPECT_FALSE(control_->Write(path));

  NodeDataRequest request;
  *request.add_paths() = path;
  ASSERT_OK(client_->GetNodeData(request));
  EXPECT_TRUE(
      should_write_to_path.WaitForNotificationWithTimeout(absl::Seconds(10)));
  EXPECT_TRUE(control_->Write(path));
}

TEST_F(ObservedControlTest, CallsRegisteredCallbacksWhenTheyMatch) {
  absl::Notification should_write_to_path1, should_write_to_path2;

  golang::stream::NodePath path1, path2, path3;
  path1.add_path("path1");
  path2.add_path("path2");
  path3.add_path("path3");

  std::unique_ptr<internal::Control::WriteRegistration> registration1 =
      control_->RegisterWriteCallback({path1}, [&](bool should_write) {
        if (should_write && !should_write_to_path1.HasBeenNotified()) {
          should_write_to_path1.Notify();
        } else if (!should_write) {  // should not stop writing once started
          ASSERT_FALSE(should_write_to_path1.HasBeenNotified());
        }
      });

  std::unique_ptr<internal::Control::WriteRegistration> registration2 =
      control_->RegisterWriteCallback({path2, path3}, [&](bool should_write) {
        if (should_write && !should_write_to_path2.HasBeenNotified()) {
          should_write_to_path2.Notify();
        } else if (!should_write) {  // should not stop writing once started
          ASSERT_FALSE(should_write_to_path2.HasBeenNotified());
        }
      });

  {
    NodeDataRequest request;
    *request.add_paths() = path1;
    ASSERT_OK(client_->GetNodeData(request));
    EXPECT_TRUE(should_write_to_path1.WaitForNotificationWithTimeout(
        absl::Seconds(10)));
    EXPECT_FALSE(should_write_to_path2.HasBeenNotified());
  }

  {
    NodeDataRequest request;
    *request.add_paths() = path3;
    ASSERT_OK(client_->GetNodeData(request));
    EXPECT_TRUE(should_write_to_path2.WaitForNotificationWithTimeout(
        absl::Seconds(10)));
  }
}

}  // namespace
}  // namespace multiscope::internal
