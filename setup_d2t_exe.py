# Copyright (C) 2019 Joris Nettelstroth

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

r"""dwd2trnsys: Convert DWD weather data to TRNSYS.

dwd2trnsys
==========
Script for converting weather data downloaded from Deutscher Wetterdienst
(DWD) to the format required by the TRNSYS weather Type 99.

Setup script
------------

Run with the following command prompt to create a Windows executable:

.. code::

    python setup_d2t_exe.py build

To create a standalone installer:

.. code::

    python setup_d2t_exe.py bdist_msi

Known issues
------------

With cx_Freeze 6.1, an error appeared about not beeing able to find
the generated exe file at the path ``build/exe.win-amd64-3.7/dwd2trnsys.exe``.
Following the error trace and replacing ``exe.targetName`` with
``os.path.abspath(exe.targetName)`` solved the issue.
(site-packages\cx_Freeze\freezer.py line 205)

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

base = None
if sys.platform == 'win32':
    base = 'Win32GUI'

# http://msdn.microsoft.com/en-us/library/windows/desktop/aa371847(v=vs.85).aspx
shortcut_table = [
    ("DesktopShortcut",        # Shortcut
     "DesktopFolder",          # Directory_
     "dwd2trnsys",             # Name
     "TARGETDIR",              # Component_
     "[TARGETDIR]dwd2trnsys.exe",   # Target
     None,                     # Arguments
     'Convert DWD weather data to TRNSYS',  # Description
     None,                     # Hotkey
     None,                     # Icon
     None,                     # IconIndex
     None,                     # ShowCmd
     'TARGETDIR'               # WkDir
     ),
    ("ProgramMenuShortcut",    # Shortcut
     "ProgramMenuFolder",      # Directory_
     "dwd2trnsys",             # Name
     "TARGETDIR",              # Component_
     "[TARGETDIR]dwd2trnsys.exe",   # Target
     None,                     # Arguments
     'Convert DWD weather data to TRNSYS',  # Description
     None,                     # Hotkey
     None,                     # Icon
     None,                     # IconIndex
     None,                     # ShowCmd
     'TARGETDIR'               # WkDir
     ),
     ]

# The setup function
setup(
    name='dwd2trnsys',
    version=version,
    description='Convert DWD weather data to TRNSYS',
    long_description=open('README.md').read(),
    license='GPL-3.0',
    author='Joris Nettelstroth',
    author_email='joris.nettelstroth@stw.de',
    url='https://github.com/jnettels/lpagg',

    # Options for building the Windows .exe
    executables=[Executable(r'lpagg/dwd2trnsys.py',
                            # base=base,  # Removes the console (use GUI only)
                            icon=r'./res/icon_d2t.ico',
                            targetName='dwd2trnsys.exe',
                            shortcutName="dwd2trnsys",
                            shortcutDir="ProgramMenuFolder",
                            )],
    options={'build_exe': {'packages': ['numpy',
                                        'asyncio',
                                        'idna',
                                        'idna.idnadata',
                                        ],
                           'includes': [],
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
                                        # 'certifi',
                                        'cffi',
                                        # 'chardet',
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
                                        # 'et_xmlfile',
                                        'h5netcdf',
                                        'h5py',
                                        'html',
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
                                        'msgpack',
                                        'nbconvert',
                                        'nbformat',
                                        'netCDF4',
                                        'nose',
                                        'notebook',
                                        'numexpr',
                                        # 'openpyxl',
                                        'OpenSSL',
                                        'PIL',
                                        'pkg_resources',
                                        'prompt_toolkit',
                                        'pycparser',
                                        'pydoc_data',
                                        'pygments',
                                        # 'PyQt5',
                                        # 'requests',
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
                               os.path.join(mkl_dlls, 'libiomp5md.dll'),
                               os.path.join(mkl_dlls, 'mkl_core.dll'),
                               os.path.join(mkl_dlls, 'mkl_def.dll'),
                               os.path.join(mkl_dlls, 'mkl_vml_avx2.dll'),
                               os.path.join(mkl_dlls, 'mkl_vml_def.dll'),
                               os.path.join(mkl_dlls, 'mkl_intel_thread.dll'),
                               r'./res/icon_d2t.png',
                               ]
                           },
             'bdist_msi': {'data': {"Shortcut": shortcut_table},
                           'upgrade_code':
                               '{eaedb2b8-f535-439a-af23-4797a64c53c5}',
                           },
             },
)

# Remove some more specific folders:
remove_folders = [
        r'.\build\exe.win-amd64-3.7\mpl-data',
        r'.\build\exe.win-amd64-3.7\lib\pandas\tests',
        r'.\build\exe.win-amd64-3.7\lib\lpagg\resources_weather',
        ]
for folder in remove_folders:
    shutil.rmtree(folder, ignore_errors=True)

# Copy the README.md file to the build folder, changing extension to .txt
shutil.copy2(r'.\README.md', r'.\build\exe.win-amd64-3.7\README.txt')
