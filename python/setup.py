"""
FastCuda Python package setup.

Build:
    pip install .
or:
    cmake -DFASTCUDA_BUILD_PYTHON=ON -S . -B build && cmake --build build
"""

from setuptools import setup

setup(
    name="fastcuda",
    version="0.2.0",
    description="Python bindings for FastCuda CUDA operator library",
    author="FastCuda contributors",
    license="Apache-2.0",
    python_requires=">=3.7",
    install_requires=["numpy"],
)
