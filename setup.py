# Copyright (C) 2020 Joris Zimmermann

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program. If not, see https://www.gnu.org/licenses/.

"""LPagg: Load profile aggregator for building simulations.

LPagg
=====
The load profile aggregator combines profiles for heat and power demand
of buildings from different sources.

Setup script
------------

Run the following command to install into your Python environment:

.. code:: sh

    python setup.py install


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
    license='GPL-3.0',
    author='Joris Nettelstroth',
    author_email='joris.nettelstroth@stw.de',
    url='https://github.com/jnettels/lpagg',
    install_requires=['pandas>=0.24.1', ],
    python_requires='>=3.7',
    packages=['lpagg', 'lpagg/examples', 'lpagg/resources_load',
              'lpagg/resources_weather'],
    package_data={'lpagg/examples': ['*.yaml'],
                  'lpagg/resources_load': ['*.xlsx'],
                  'lpagg/resources_weather': ['*_Jahr.dat'],
                  'lpagg': ['lpagg.mplstyle'],
                  },
    entry_points={
        'console_scripts': ['lpagg = lpagg.__main__:main'],
        'gui_scripts': ['simlty_GUI = lpagg.simlty_GUI:main'],
        }
)
