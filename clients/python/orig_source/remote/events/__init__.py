# Copyright 2023 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Package to handle events with Multiscope."""
from multiscope.remote.events.events import EventSubscription
from multiscope.remote.events.events import register_callback
from multiscope.remote.events.events import register_gamepad_callback
from multiscope.remote.events.events import register_keyboard_callback
from multiscope.remote.events.events import register_mouse_callback
from multiscope.remote.events.events import register_webcam_callback
from multiscope.remote.events.events import subscribe_gamepad_events
from multiscope.remote.events.events import subscribe_keyboard_events
from multiscope.remote.events.events import subscribe_mouse_events
from multiscope.remote.events.events import subscribe_webcam_events
