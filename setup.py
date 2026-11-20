"""Builds the CPU kernel libraries into the wheel (one per instruction set; see src/vahana/csrc/build.py).

The libraries have a C ABI and no Python or torch dependency, so the wheel is tagged py3-none-<platform>: one wheel
per OS / architecture serves every Python >= 3.10 and every torch >= 2.4.
VAHANA_ISAS="avx2 scalar" limits the builds; VAHANA_SKIP_KERNEL=1 skips them (then build them yourself).
"""
import os
import subprocess
import sys

from setuptools import Distribution, setup
from setuptools.command.build_py import build_py

try:
    from setuptools.command.bdist_wheel import bdist_wheel
except ImportError:  # setuptools < 70
    from wheel.bdist_wheel import bdist_wheel


class BuildKernel(build_py):
    def run(self):
        if os.environ.get("VAHANA_SKIP_KERNEL") != "1":
            here = os.path.dirname(os.path.abspath(__file__))
            isas = os.environ.get("VAHANA_ISAS", "").split()
            subprocess.run([sys.executable, os.path.join(here, "src", "vahana", "csrc", "build.py"), *isas], check=True)
            self.data_files = self._get_data_files()          # include the libraries just built
        super().run()


class BinaryDistribution(Distribution):
    def has_ext_modules(self):          # the kernel libraries are binaries: platform wheel, files in platlib
        return True


class PlatformWheel(bdist_wheel):
    def finalize_options(self):
        super().finalize_options()
        self.root_is_pure = False

    def get_tag(self):
        _, _, plat = super().get_tag()
        return "py3", "none", plat


setup(distclass=BinaryDistribution, cmdclass={"build_py": BuildKernel, "bdist_wheel": PlatformWheel})
