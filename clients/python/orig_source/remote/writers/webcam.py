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

"""Multiscope webcam input."""

import io
import time
from typing import Callable, Optional, Text
from PIL import Image



from golang.multiscope.streams import writers_pb2 as writers_pb
from golang.stream import stream_pb2 as pb
from multiscope.proto import webcam_pb2 as webcam_pb
from multiscope.remote import control
from multiscope.remote import group
from multiscope.remote import stream_client
from multiscope.remote.events import events
from multiscope.remote.writers import base

DEFAULT_VIDEO_WIDTH = 640
DEFAULT_VIDEO_HEIGHT = 480
DEFAULT_ENCODING_QUALITY = 0.92


class Webcam(base.Writer):
  """Adds a webcam input to the multiscope dashboard."""

  @control.init
  def __init__(self, name: Text, parent: Optional[group.ParentNode] = None):

    path = group.join_path(parent, name)
    request = pb.CreateNodeRequest()
    request.path.path.extend(path)
    request.type = writers_pb.Writers().group
    request.mime = 'webcam/parent-node'
    resp = stream_client.CreateNode(request=request)
    super().__init__(path=tuple(resp.path.path))

    # Add a child ProtoWriter node for WebcamInfo
    request = pb.CreateNodeRequest()
    self.info_path = path + ('info',)
    request.path.path.extend(self.info_path)
    request.type = writers_pb.Writers().proto_writer
    stream_client.CreateNode(request=request)

    # Add a child ProtoWriter node for the config
    request = pb.CreateNodeRequest()
    self.config_path = path + ('config',)
    request.path.path.extend(self.config_path)
    request.type = writers_pb.Writers().proto_writer
    stream_client.CreateNode(request=request)

    self._set_display()
    self._info = webcam_pb.WebcamInfo()
    self._config = webcam_pb.WebcamConfig()
    self._config.max_height_pixels = DEFAULT_VIDEO_HEIGHT
    self._config.max_width_pixels = DEFAULT_VIDEO_WIDTH
    self._config.jpeg.quality = DEFAULT_ENCODING_QUALITY
    self._write_config()
    self._write_info()


  @control.method
  def write(self, latency_ms: float):
    self._info.latency_ms = latency_ms
    self._write_info()


  @control.method
  def set_input_size(self,
                     max_height_pixels: Optional[int] = None,
                     max_width_pixels: Optional[int] = None):
    if max_height_pixels is not None:
      self._config.max_height_pixels = max_height_pixels
    if max_width_pixels is not None:
      self._config.max_width_pixels = max_width_pixels
    self._write_config()


  @control.method
  def set_frame_rate(self, fps: float):
    self._config.fps = fps
    self._write_config()


  @control.method
  def set_encoding(self,
                   encoding: str,
                   quality: float = DEFAULT_ENCODING_QUALITY):
    if encoding == 'jpeg':
      self._config.jpeg.quality = quality
    elif encoding == 'png':
      self._config.png.SetInParent()
    else:
      raise ValueError(f'Unknown encoding provided: {encoding}')
    self._write_config()

  def _write_config(self):
    request = pb.PutNodeDataRequest()
    request.data.path.path.extend(self.config_path)
    request.data.pb.Pack(self._config)
    stream_client.PutNodeData(request=request)

  def _write_info(self):
    request = pb.PutNodeDataRequest()
    request.data.path.path.extend(self.info_path)
    request.data.pb.Pack(self._info)
    stream_client.PutNodeData(request=request)

  @control.method
  def register_frame_callback(self,
                              cb: Callable[[Image.Image, webcam_pb.WebcamEvent],
                                           None]):
    """Registers a callback function for every incoming webcam frame.

    Note: The first few frames from a webcam stream are often all zeros as the
    webcam stream in the browser initializes.

    Args:
      cb: Callback function.
    """

    ewma_latency_ms = 0
    last_latency_update = 0

    def wrapped_cb(_, event):
      nonlocal ewma_latency_ms, last_latency_update
      if event.HasField('frame'):
        now = time.time()
        frame_latency_ms = int(now * 1000) - event.frame.timestamp_ms
        ewma_latency_ms = ewma_latency_ms * 0.9 + frame_latency_ms * 0.1
        # Update latency only once every second because this write can take
        # ~1ms.
        if now - last_latency_update > 1:
          self.write(latency_ms=ewma_latency_ms)
          last_latency_update = now

        frame = Image.open(io.BytesIO(event.frame.frame))
        cb(frame, event)

    events.register_webcam_callback(wrapped_cb, self.path)
