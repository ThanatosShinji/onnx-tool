#!/usr/bin/env python

from setuptools import setup, find_packages

readme = open("README.md").read()
VERSION = "0.6.3"


requirements = [
    "onnx",
    "numpy",
    'tabulate'
]

setup(
    # Metadata
    name="onnx-tool",
    version=VERSION,
    author="Luo Yu",
    author_email="luoyu888888@gmail.com",
    url="https://github.com/ThanatosShinji/onnx-tool",
    description="A tool for ONNX model:"
                "Rapid shape inference; "
                "Profile model; "
                "Compute Graph and Shape Engine; "
                "OPs fusion;"
                "Quantized models and sparse models are supported.",
    long_description=readme,
    long_description_content_type="text/markdown",
    license="MIT",
    # Package info
    packages=find_packages(),
    #
    zip_safe=True,
    install_requires=requirements,
    # Classifiers
    classifiers=[
        "Programming Language :: Python :: 3",
    ],
)
