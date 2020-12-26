#!/usr/bin/env python3

import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="hepaquery", # Replace with your own username
    version="0.0.1",
    author="almaaan",
    author_email="almaan@kth.se",
    description="Functions for spatial analysis of ST liver data",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/almaan/ankarliver",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3.7",
        "License :: OSI Approved :: MIT License",
        "Operating System :: Tested on Mac OS and Linux",
    ],
    python_requires='>=3.7',
)
