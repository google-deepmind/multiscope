# Copyright 2023 DeepMind Technologies Limited
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

# pylint: disable=g-bad-file-header

"""Install script for setuptools."""

from setuptools import find_packages
from setuptools import setup


setup(
    name="multiscope-client",
    version="0.1.0",
    description="A Python client to Multiscope.",
    author="DeepMind",
    license="BSD-3-Clause License",
    keywords="analysis visualization python reinforcement-learning machine learning",
    packages=find_packages(exclude=["examples"]),
    python_requires=">=3.7, <4",
    install_requires=[],
    tests_require=[],
    classifiers=[],
)
