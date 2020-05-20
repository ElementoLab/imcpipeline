#! /usr/bin/env python

"""
Installer script for the ``imc`` library and the ``imcpipeline`` pipeline.

Install with ``pip install .``.
"""

import sys


def parse_requirements(req_file):
    """Parse requirements.txt files."""
    reqs = open(req_file).read().strip().split("\n")
    reqs = [r for r in reqs if not r.startswith("#")]
    return [r for r in reqs if "#egg=" not in r]


REQUIREMENTS_FILE = "requirements.txt"
README_FILE = "README.md"


# take care of extra required modules depending on Python version
extra = {}
try:
    from setuptools import setup, find_packages

    if sys.version_info < (2, 7):
        extra["install_requires"] = ["argparse"]
    if sys.version_info >= (3,):
        extra["use_2to3"] = True
except ImportError:
    from distutils.core import setup

    if sys.version_info < (2, 7):
        extra["dependencies"] = ["argparse"]

# Requirements
requirements = parse_requirements(REQUIREMENTS_FILE)
# requirements_test = parse_requirements("requirements/requirements.test.txt")
# requirements_docs = parse_requirements("requirements/requirements.docs.txt")

# Description
long_description = open(README_FILE).read()


# setup
setup(
    name="imc",
    packages=find_packages(),
    use_scm_version={
        'write_to': 'imc/_version.py',
        'write_to_template': '__version__ = "{version}"\n'
    },
    entry_points={
        "console_scripts": [
            "imcrunner = imcpipeline.runner:main",
            "imcpipeline = imcpipeline.pipeline:main"]
    },
    description="A pipeline and utils for IMC data analysis.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    classifiers=[
        "Programming Language :: Python :: 3 :: Only",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Development Status :: 3 - Alpha",
        "Typing :: Typed",
        "License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
    ],
    keywords=",".join([
        "computational biology",
        "bioinformatics",
        "imaging",
        "mass cytometry",
        "mass spectrometry"]),
    url="https://github.com/elementolab/hyperion-cytof",
    project_urls={
        "Bug Tracker": "https://github.com/elementolab/hyperion-cytof/issues",
        # "Documentation": "https://imc.readthedocs.io",
        "Source Code": "https://github.com/elementolab/hyperion-cytof",
    },
    author=u"Andre Rendeiro",
    author_email="andre.rendeiro@pm.me",
    license="GPL3",
    setup_requires=['setuptools_scm'],
    install_requires=requirements,
    # tests_require=requirements_test,
    # extras_require={
    #     "testing": requirements_test,
    #     "docs": requirements_docs},
    # package_data={"imc": ["config/*.yaml", "templates/*.html", "models/*"]},
    data_files=[
        REQUIREMENTS_FILE,
        # "requirements/requirements.test.txt",
    ],
    **extra
)
