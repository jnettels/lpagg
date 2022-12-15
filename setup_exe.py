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

Run with the following command prompt to create a Windows executable:

.. code::

    python setup_exe.py build

To create a standalone installer:

.. code::

    python setup_exe.py bdist_msi


Troubleshooting

- Trouble with PyQt5?
  https://stackoverflow.com/questions/52376313/converting-py-file-to-exe-cannot-find-existing-pyqt5-plugin-directories

- When I click the exe-file, nothing happens...
    - Set "base = None" to get a console output, hopefully with error message
    - "mkl_intel_thread.1.dll" was missing from "include_files"

- "ValueError: FCI error 1"
    - Non-ascii characters in file names

- "DLL load failed while importing tf2font""
  https://stackoverflow.com/questions/64985612/how-to-include-a-microsoft-visual-c-redistributable-package-in-a-python-cx-fre
  freetype.dll was missing

- In general, if dll dependenices seem to be the problem, check dependencies
  with https://github.com/lucasg/Dependencies

- cx_freeze >= 6.12 recommended

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
# os.environ['TCL_LIBRARY'] = os.path.join(sys.exec_prefix, r'tcl\tcl8.6')
# os.environ['TK_LIBRARY'] = os.path.join(sys.exec_prefix, r'tcl\tk8.6')
dlls = os.path.join(sys.exec_prefix, 'Library', 'bin')

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
    license='MIT',
    author=author,
    author_email='joris.zimmermann@siz-energieplus.de',
    url='https://github.com/jnettels/lpagg',

    # Options for building the Windows .exe
    executables=[Executable(r'lpagg/simlty_GUI.py', base=base,
                            icon=r'./res/icon.ico',
                            target_name='Gleichzeitigkeit.exe',
                            shortcut_name="Gleichzeitigkeit",
                            shortcut_dir="ProgramMenuFolder",
                            )],
    options={'build_exe': {'packages': ['numpy',
                                        'asyncio',
                                        'pandas.plotting._matplotlib',
                                        'openpyxl',
                                        'xlsxwriter',
                                        ],
                           # 'namespace_packages': ['mpl_toolkits'],
                           'zip_include_packages': ['*'],  # reduze file size
                           'zip_exclude_packages': [
                               'lpagg',
                               'PIL',
                               # 'matplotlib',
                               # 'pandas', 'PyQt5',
                               ],
                           'includes': ['openpyxl',
                                        'xlsxwriter',
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
                               # os.path.join(dlls, 'libiomp5md.dll'),
                               # os.path.join(dlls, 'mkl_core.1.dll'),
                               # os.path.join(dlls, 'mkl_def.1.dll'),
                               os.path.join(dlls, 'mkl_avx2.1.dll'),
                               os.path.join(dlls, 'mkl_intel_thread.1.dll'),
                               os.path.join(dlls, 'mkl_sequential.1.dll'),
                               # os.path.join(dlls, 'vcruntime140_1.dll'),
                               os.path.join(dlls, 'msvcp140_1.dll'),
                               os.path.join(dlls, 'freetype.dll'),
                               r'./lpagg/lpagg.mplstyle',
                               r'./res/icon.png',
                               r'./README.md',
                               ],
                           # 'include_msvcr': True,  # Microsoft Visual C++

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
