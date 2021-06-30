import io
import os
import re

from setuptools import find_packages, setup


# From https://github.com/n2cholas/jax-resnet
def read(*names, **kwargs):
    with io.open(
        os.path.join(os.path.dirname(__file__), *names),
        encoding=kwargs.get("encoding", "utf8"),
    ) as fp:
        return fp.read()


def find_version(*file_paths):
    version_file = read(*file_paths)
    version_match = re.search(r'^__version__ = ["\']([^"\']*)["\']', version_file, re.M)
    if version_match:
        return version_match.group(1)
    raise RuntimeError("Unable to find version string.")


readme = read("README.md")
VERSION = find_version("mlpmixer_flax", "__init__.py")

setup(
    name="mlpmixer-flax",
    version=VERSION,
    author="Saurav Maheshkar",
    author_email="sauravvmaheshkar@gmail.com",
    url="https://github.com/SauravMaheshkar/MLP-Mixer",
    description="Flax implementation of MLP-Mixer from MLP-Mixer: An all-MLP Architecture for Vision",
    long_description_content_type="text/markdown",
    long_description=readme,
    license="MIT",
    packages=find_packages(
        exclude=(
            "tests",
            "tests.*",
        )
    ),
    zip_safe=True,
    install_requires=["einops>=0.3", "flax", "jax"],
)
