# pylint: disable=g-bad-file-header
# TODO(dsz): add copyright and license info.

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
    install_requires=[
        # TODO(dsz): fill out later.
    ],
    tests_require=[
        # TODO(dsz): fill out later.
    ],
    classifiers=[
        # TODO(dsz): fill out later.
    ],
)
