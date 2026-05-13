"""Build script — invokes pybind11 with the vendored Eigen headers."""

from pathlib import Path

from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup

HERE = Path(__file__).parent.resolve()
CPP_INCLUDE = HERE.parent / "cpp" / "include"

ext_modules = [
    Pybind11Extension(
        "aompls._binding",
        sources=["src/aompls/_binding.cpp"],
        include_dirs=[str(CPP_INCLUDE)],
        cxx_std=17,
        define_macros=[("EIGEN_NO_DEBUG", "1"), ("EIGEN_DONT_PARALLELIZE", "1")],
        extra_compile_args=["-O3", "-Wno-maybe-uninitialized", "-Wno-deprecated-declarations"],
    ),
]

setup(
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
)
