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

Run with the following command prompt to create a Windows executable:

.. code::

    python setup_exe.py build

To create a standalone installer:

.. code::

    python setup_exe.py bdist_msi


Troubleshooting

https://stackoverflow.com/questions/52376313/converting-py-file-to-exe-cannot-find-existing-pyqt5-plugin-directories

Last tested with cx_freeze 6.8.2

"""

from setuptools_scm import get_version
from cx_Freeze import setup, Executable
import os
import shutil
import sys


try:
    version = get_version(version_scheme='post-release')
except LookupError:
    version = '0.0.0'
    print('Warning: setuptools-scm requires an intact git repository to detect'
          ' the version number for this build.')

if 'g' in version:  # 'Dirty' version, does not fit to Windows' version scheme
    version_list = []
    for i in version.split('.'):
        try:  # Sort out all parts of the version name that are not integers
            version_list.append(str(int(i)))
        except Exception:
            pass
    if len(version_list) < 3:  # Version is X.Y -> make it X.Y.0.1
        version_list.append('1')  # Use this to mark as a dev build

    version = '.'.join(version_list)

print('Building with version tag: ' + version)

# These settings solved an error (set to folders in python directory)
os.environ['TCL_LIBRARY'] = os.path.join(sys.exec_prefix, r'tcl\tcl8.6')
os.environ['TK_LIBRARY'] = os.path.join(sys.exec_prefix, r'tcl\tk8.6')
mkl_dlls = os.path.join(sys.exec_prefix, r'Library\bin')

# We need to include a dll for Qt
# target = r'.\build\exe.win-amd64-3.7\platforms\qwindows.dll'
# if not os.path.exists(os.path.dirname(target)):
#     os.makedirs(os.path.dirname(target))
# shutil.copy2(os.path.join(mkl_dlls, '../plugins/platforms/qwindows.dll'),
#              target)

base = None  # None for cmd-line
if sys.platform == 'win32':
    base = 'Win32GUI'  # If only a GUI should be shown

# http://msdn.microsoft.com/en-us/library/windows/desktop/aa371847(v=vs.85).aspx
shortcut_table = [
    ("DesktopShortcut",        # Shortcut
     "DesktopFolder",          # Directory_
     "Gleichzeitigkeit",     # Name
     "TARGETDIR",              # Component_
     "[TARGETDIR]Gleichzeitigkeit.exe",   # Target
     None,                     # Arguments
     'Gleichzeitigkeit in Zeitreihen',  # Description
     None,                     # Hotkey
     None,                     # Icon
     None,                     # IconIndex
     None,                     # ShowCmd
     'TARGETDIR'               # WkDir
     ),
    ("ProgramMenuShortcut",        # Shortcut
     "ProgramMenuFolder",          # Directory_
     "Gleichzeitigkeit",     # Name
     "TARGETDIR",              # Component_
     "[TARGETDIR]Gleichzeitigkeit.exe",   # Target
     None,                     # Arguments
     'Gleichzeitigkeit in Zeitreihen',  # Description
     None,                     # Hotkey
     None,                     # Icon
     None,                     # IconIndex
     None,                     # ShowCmd
     'TARGETDIR'               # WkDir
     ),
     ]

author = 'Joris Zimmermann'
description = 'Load profile aggregator for building simulations'

# The setup function
setup(
    name='Gleichzeitigkeit',
    version=version,
    description=description,
    long_description=open('README.md').read(),
    license='GPL-3.0',
    author=author,
    author_email='joris.zimmermann@siz-energieplus.de',
    url='https://github.com/jnettels/lpagg',

    # Options for building the Windows .exe
    executables=[Executable(r'lpagg/simlty_GUI.py', base=base,
                            icon=r'./res/icon.ico',
                            targetName='Gleichzeitigkeit.exe',
                            shortcutName="Gleichzeitigkeit",
                            shortcutDir="ProgramMenuFolder",
                            )],
    options={'build_exe': {'packages': ['numpy', 'asyncio',
                                        'pandas.plotting._matplotlib',
                                        ],
                           # 'namespace_packages': ['mpl_toolkits'],
                           'zip_include_packages': ['*'],  # reduze file size
                           'zip_exclude_packages': [
                               # 'pandas', 'PyQt5',
                               # 'matplotlib'
                               ],
                           'includes': ['openpyxl',
                                        ],
                           'excludes': ['adodbapi',
                                        'alabaster'
                                        'asn1crypto',
                                        # 'asyncio',
                                        'atomicwrites',
                                        'attr',
                                        'babel',
                                        'backports',
                                        'bokeh',
                                        'bottleneck',
                                        'bs4',
                                        'certifi',
                                        'cffi',
                                        'chardet',
                                        'cloudpickle',
                                        'colorama',
                                        # 'collections',
                                        'concurrent',
                                        'cryptography',
                                        # 'ctypes',
                                        'curses',
                                        'Cython',
                                        'cytoolz',
                                        'dask',
                                        # 'et_xmlfile',  # for openpyxl
                                        'h5netcdf',
                                        'h5py',
                                        # 'html',
                                        'html5lib',
                                        'ipykernel',
                                        'IPython',
                                        'ipython_genutils',
                                        'jedi',
                                        'jinja2',
                                        'jupyter_client',
                                        'jupyter_core',
                                        'lib2to3',
                                        'lxml',
                                        'markupsafe',
                                        # 'matplotlib',
                                        'matplotlib.tests',
                                        'matplotlib.mpl-data',
                                        'msgpack',
                                        'nbconvert',
                                        'nbformat',
                                        'netCDF4',
                                        'nose',
                                        'notebook',
                                        'numexpr',
                                        'numpy.random._examples',
                                        # 'openpyxl',
                                        'OpenSSL',
                                        'pandas.tests',
                                        # 'PIL',
                                        # 'pkg_resources',
                                        'prompt_toolkit',
                                        'pycparser',
                                        'pydoc_data',
                                        'pygments',
                                        # 'PyQt5',
                                        'requests',
                                        'scipy',
                                        'seaborn',
                                        'setuptools',
                                        'sphinx',
                                        'sphinxcontrib',
                                        'sqlalchemy',
                                        'sqlite3',
                                        'tables',
                                        'testpath',
                                        'tornado',
                                        # 'tkinter',
                                        'traitlets',
                                        'wcwidth',
                                        'webencodings',
                                        'win32com',
                                        # 'xlwt',
                                        # 'xml',
                                        # 'xmlrpc',
                                        'zict',
                                        'zmq',
                                        '_pytest',
                                        ],
                           'include_files': [
                               # os.path.join(mkl_dlls, 'libiomp5md.dll'),
                               # os.path.join(mkl_dlls, 'mkl_core.dll'),
                               # os.path.join(mkl_dlls, 'mkl_def.dll'),
                               # os.path.join(mkl_dlls, 'mkl_intel_thread.dll'),
                               r'./lpagg/lpagg.mplstyle',
                               r'./res/icon.png',
                               r'./README.md',
                               ]

                           },
             'bdist_msi': {'data': {"Shortcut": shortcut_table},
                           'summary_data': {'author': author,
                                            'comments': description},
                           'install_icon': r'./res/icon.ico',
                           'upgrade_code':
                               '{4d27fdce-eca0-4f0a-bdf7-a06bd383351e}',
                           },
             },
)

# Remove some more specific folders:
# remove_folders = [
#         # r'.\build\exe.win-amd64-3.7\mpl-data',
#         # r'.\build\exe.win-amd64-3.7\tk',
#         # r'.\build\exe.win-amd64-3.7\tcl',
#         # r'.\build\exe.win-amd64-3.7\lib\pandas\tests',
#         r'.\build\exe.win-amd64-3.7\lib\lpagg\resources_weather',
#         ]
# for folder in remove_folders:
#     shutil.rmtree(folder, ignore_errors=True)

# Copy the README.md file to the build folder, changing extension to .txt
# shutil.copy2(r'.\README.md', r'.\build\exe.win-amd64-3.7\README.txt')
