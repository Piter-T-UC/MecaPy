Installation
=============

Prerequisites
-------------

- Python 3.8 or higher
- pip (Python package manager)

Installing MecaPy
-----------------

From source (development installation)::

    git clone https://github.com/piter-t-uc/mecapy.git
    cd mecapy
    pip install -e .

Installing with development dependencies::

    pip install -e ".[dev]"

This will install MecaPy along with testing and documentation dependencies.

Verifying Installation
----------------------

To verify that MecaPy is installed correctly, run::

    python -c "import mecapy; print(mecapy.__version__)"

You should see the version number printed to the console.

Dependencies
------------

MecaPy requires:

- numpy >= 1.20
- scipy >= 1.7

Additional development dependencies:

- pytest >= 7.0
- pytest-cov >= 4.0
- sphinx >= 5.0
- sphinx-rtd-theme >= 1.0
