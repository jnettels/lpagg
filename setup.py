# MIT License

# Copyright (c) 2022 Joris Zimmermann

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to
# deal in the Software without restriction, including without limitation the
# rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
# sell copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

"""LPagg: Load profile aggregator for building simulations.

LPagg
=====
The load profile aggregator combines profiles for heat and power demand
of buildings from different sources.

Setup script
------------

Run one of the following commands to install into your Python environment:

.. code:: sh

    python setup.py install

    pip install -e <path to this folder>

"""
from setuptools import setup
from setuptools_scm import get_version


try:
    version = get_version(version_scheme='post-release')
except LookupError:
    version = '0.0.0'
    print('Warning: setuptools-scm requires an intact git repository to detect'
          ' the version number for this build.')

print('Building LPagg with version tag: ' + version)

# The setup function
setup(
    name='lpagg',
    version=version,
    description='Load profile aggregator for building simulations',
    long_description=open('README.md').read(),
    license='MIT',
    author='Joris Zimmermann',
    author_email='joris.zimmermann@siz-energieplus.de',
    url='https://github.com/jnettels/lpagg',
    python_requires='>=3.7',
    install_requires=[
        'pandas >= 1.2.1',
        'openpyxl >=3.0.3',
        'pyyaml >=5.1',
        'matplotlib',
        'scipy',
        'requests',
        'geopy',
        'holidays',
    ],
    extras_require={
        'simlty_GUI': ['PyQt5'],
    },
    packages=['lpagg', 'lpagg/examples', 'lpagg/resources_load',
              'lpagg/resources_weather'],
    package_data={'lpagg/examples': ['*.yaml'],
                  'lpagg/resources_load': ['*.xlsx'],
                  'lpagg/resources_weather': ['*_Jahr.dat', '*.geojson'],
                  'lpagg': ['lpagg.mplstyle'],
                  },
    entry_points={
        'console_scripts': ['lpagg = lpagg.__main__:main'],
        'gui_scripts': ['simlty_GUI = lpagg.simlty_GUI:main'],
        }
)
