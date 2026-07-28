"""Setup configuration for MecaPy."""

import os

from setuptools import setup, find_packages

HERE = os.path.abspath(os.path.dirname(__file__))

with open(os.path.join(HERE, "mecapy", "__init__.py")) as f:
    for line in f:
        if line.startswith("__version__"):
            version = line.split('"')[1]
            break

# README.md lives at the repository root, one level above this file.
long_description = "MecaPy - Python library for mechanical engineering calculations"
for candidate in (os.path.join(HERE, "README.md"), os.path.join(HERE, os.pardir, "README.md")):
    try:
        with open(candidate, "r", encoding="utf-8") as f:
            long_description = f.read()
        break
    except FileNotFoundError:
        continue

setup(
    name="mecapy",
    version=version,
    author="Piter-T-UC",
    author_email="pedrito00.taboada@gmail.com",
    license="MIT",
    license_files=["LICENSE"],
    description="Python library for mechanical engineering calculations",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Piter-T-UC/MecaPy",
    packages=find_packages(
        exclude=["tests", "tests.*", "examples", "examples.*", "sandbox", "sandbox.*", "docs", "docs.*"]
    ),
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.20.0",
        "sympy>=1.10",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0",
            "pytest-cov>=4.0",
            "black>=22.0",
            "flake8>=4.0",
            "mypy>=0.950",
            "pint>=0.20",
        ],
        "viz": [
            "matplotlib>=3.5.0",
        ],
        "docs": [
            "sphinx>=5.0",
            "sphinx-rtd-theme>=1.0",
            "matplotlib>=3.5.0",
        ],
        "units": [
            "pint>=0.20",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Education",
        "License :: OSI Approved :: MIT License",
        "Topic :: Scientific/Engineering",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
)
