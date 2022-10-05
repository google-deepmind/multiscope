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
