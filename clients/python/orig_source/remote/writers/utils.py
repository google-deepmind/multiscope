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

"""Module defining common utilities."""
import json
from typing import Any

import altair as alt
import six


def extract_size(vega_spec: Any):
  if isinstance(vega_spec, alt.Chart):
    return vega_spec.height, vega_spec.width
  elif isinstance(vega_spec, six.string_types):
    vega_spec = json.loads(vega_spec)
  return vega_spec.get('height', None), vega_spec.get('width', None)
