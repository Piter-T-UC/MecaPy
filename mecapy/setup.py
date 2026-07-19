"""Setup configuration for MecaPy."""

from setuptools import setup, find_packages

with open("mecapy/__init__.py") as f:
    for line in f:
        if line.startswith("__version__"):
            version = line.split('"')[1]
            break

try:
    with open("README.md", "r", encoding="utf-8") as f:
        long_description = f.read()
except FileNotFoundError:
    long_description = "MecaPy - Python library for mechanical engineering calculations"

setup(
    name="mecapy",
    version=version,
    author="Piter-T-UC",
    author_email="pedrito00.taboada@gmail.com",
    license="MIT",
    description="Python library for mechanical engineering calculations",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Piter-T-UC/MecaPy",
    packages=find_packages(),
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
        ],
        "viz": [
            "matplotlib>=3.5.0",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Engineers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
)
