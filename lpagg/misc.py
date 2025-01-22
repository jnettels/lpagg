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


Module misc
-----------
This is a collection of miscellaneous functions that are shared by the other
modules in the project.
"""

import time                      # Measure time
import multiprocessing           # Parallel (Multi-) Processing
import os                        # Operaing System
import logging
import pandas as pd              # Pandas
import matplotlib.pyplot as plt  # Plotting library
import matplotlib as mpl
import importlib.resources
from tkinter import Tk, filedialog

# Define the logging function
logger = logging.getLogger(__name__)


def setup():
    """Perform some basic setup."""
    multiprocessing.freeze_support()  # Needed for parallel processing

    # Global Pandas option for displaying terminal output
    pd.set_option('display.max_columns', 0)

    # Define the logging function
    logging.basicConfig(format='%(asctime)-15s %(levelname)-8s %(message)s')

    # Define style settings for the plots
    try:  # Try to load personalized matplotlib style file
        mpl.style.use(os.path.join(os.path.dirname(__file__),
                                   './lpagg.mplstyle'))
    except OSError as ex:
        logger.debug(ex)


def file_dialog(initialdir=os.getcwd(),
                title='Choose a file',
                filetypes=(('YAML File', '*.yaml'),)
                ):
    """Present a file dialog.

    Args:
        None

    Return:
        path (str): File path
    """
    root = Tk()  # Create window to use for the file dialog

    # Set a custom taskbar icon
    # For the 'noarch' conda build, access the file as resource object
    res_path = importlib.resources.files('lpagg').joinpath('./res/icon.ico')
    with importlib.resources.as_file(res_path) as resource:
        try:
            root.iconbitmap(resource)  # Set the custom taskbar icon
        except Exception:
            pass  # Does not work on linux

    root.title("lpagg")
    root.geometry("300x1")  # Show window only as title bar
    root.lift()  # Bring the window to the front

    # Open the file dialog window
    file = filedialog.askopenfilename(initialdir=initialdir, title=title,
                                      filetypes=filetypes)
    root.destroy()  # Destroy the root window after selection

    if file == '' or len(file) == 0:
        path = None
    else:
        path = os.path.abspath(file)
    return path


def multiprocessing_job(helper_function, work_list):
    """Generalize multiprocessing with integrated progress printing."""
    number_of_cores = min(multiprocessing.cpu_count()-1, len(work_list))
    logger.info('Parallel processing on '+str(number_of_cores)+' cores')
    pool = multiprocessing.Pool(number_of_cores)
    return_list = pool.map_async(helper_function, work_list)
    pool.close()

    total = return_list._number_left
    while return_list.ready() is False:
        remaining = return_list._number_left
        fraction = (total - remaining)/total
        print('\r{:5.1f}% done'.format(fraction*100), end='\r')  # progress
        time.sleep(1.0)

    pool.join()
    print('\r100.0% done', end='\r')

    return return_list


def resample_energy(df, freq):
    """Resample energy time series data to a new frequency.

    This keeps the total energy sum constant.

    - In case of downsampling (to longer time intervalls)

        - Simply take the sum

    - For upsampling (to shorter time intervalls)

        - Reindex with an index that is expanded at the beginning. For
          example, resampling from time step of 1h to 15 min
          needs 3 x 15min before the first time stamp)
        - Use backwards fill to fill resulting NaN
        - Divide all values by the fraction of new and old frequency

    Args:
        df (DataFrame): Pandas DataFrame with time index to resample.

        freq (Timedelta): New frequency to resample to.

    Returns:
        df (DataFrame): Resampled DataFrame.

    """
    from pandas.tseries.frequencies import to_offset

    freq_orig = pd.infer_freq(df.index)
    freq_orig = pd.to_timedelta(to_offset(freq_orig))
    f_freq = freq/freq_orig
    if f_freq < 1:
        # Reindex
        start = df.index[0] - (freq_orig - freq)
        end = df.index[-1]
        df = df.reindex(pd.date_range(start=start, end=end, freq=freq))

        # Fill all resulting missing values
        df.bfill(inplace=True)

        # Divide by factor f_freq to keep the total energy constant
        df *= f_freq

    elif f_freq > 1:
        df = df.resample(rule=freq, label='right', closed='right').sum()

    return df


def df_to_excel(df, path, sheet_names=[], merge_cells=False,
                check_permission=True, **kwargs):
    """Save one or multiple DataFrames to an Excel file.

    Wrapper around pandas' function ``DataFrame.to_excel()``, which creates
    the required directory.
    In case of a ``PermissionError`` (happens when the Excel file is currently
    opended), the file is instead saved with a time stamp.

    Additional keyword arguments are passed down to ``to_excel()``.
    Can save a single DataFrame to a single Excel file or multiple DataFrames
    to a combined Excel file.

    The function calls itself recursively to achieve those features.

    Args:
        df (DataFrame or list): Pandas DataFrame object(s) to save

        path (str): The full file path to save the DataFrame to

        sheet_names (list): List of sheet names to use when saving multiple
        DataFrames to the same Excel file

        merge_cells (boolean, optional): Write MultiIndex and Hierarchical
        Rows as merged cells. Default False.

        check_permission (boolean): If the file already exists, instead try
        to save with an appended time stamp.

        freeze_panes (tuple or boolean, optional): Per default, the sheet
        cells are frozen to always keep the index visible (by determining the
        correct coordinate ``tuple``). Use ``False`` to disable this.

    Returns:
        None
    """
    from collections.abc import Sequence
    import time

    if check_permission:
        try:
            # Try to complete the function without this permission check
            df_to_excel(df, path, sheet_names=sheet_names,
                        merge_cells=merge_cells, check_permission=False,
                        **kwargs)
            return  # Do not run the rest of the function
        except PermissionError as e:
            # If a PermissionError occurs, run the whole function again, but
            # with another file path (with appended time stamp)
            logger.critical(e)
            ts = time.localtime()
            ts = time.strftime('%Y-%m-%d_%H-%M-%S', ts)
            path_time = (os.path.splitext(path)[0] + '_' +
                         ts + os.path.splitext(path)[1])
            logger.critical('Writing instead to: %s', path_time)
            df_to_excel(df, path_time, sheet_names=sheet_names,
                        merge_cells=merge_cells, **kwargs)
            return  # Do not run the rest of the function

    # Here the 'actual' function content starts:
    if not os.path.exists(os.path.dirname(path)):
        logger.debug('Create directory %s', os.path.dirname(path))
        os.makedirs(os.path.dirname(path))

    if isinstance(df, Sequence) and not isinstance(df, str):
        # Save a list of DataFrame objects into a single Excel file
        with pd.ExcelWriter(path) as writer:
            for i, df_ in enumerate(df):
                try:  # Use given sheet name, or just an enumeration
                    sheet = sheet_names[i]
                except IndexError:
                    sheet = str(i)
                # Add current sheet to the ExcelWriter by calling this
                # function recursively
                df_to_excel(df=df_, path=writer, sheet_name=sheet,
                            merge_cells=merge_cells, **kwargs)

    else:
        # Per default, the sheet cells are frozen to keep the index visible
        if 'freeze_panes' not in kwargs or kwargs['freeze_panes'] is True:
            # Find the right cell to freeze in the Excel sheet
            if merge_cells:
                freeze_rows = len(df.columns.names) + 1
            else:
                freeze_rows = 1

            kwargs['freeze_panes'] = (freeze_rows, len(df.index.names))
        elif kwargs['freeze_panes'] is False:
            del kwargs['freeze_panes']

        # Save one DataFrame to one Excel file
        df.to_excel(path, merge_cells=merge_cells, **kwargs)


def get_TRY_polygons_GeoDataFrame(col_try="TRY_code"):
    """Return a GeoDataFrame with the 15 TRY regions.

    Can be used to test to which region a certain place belongs.
    """
    import geopandas as gpd
    filedir = os.path.dirname(__file__)
    TRY_polygons = gpd.read_file(
        os.path.join(filedir, 'resources_weather', 'TRY_polygons.geojson'))
    TRY_polygons.rename(columns={"TRY_code": col_try}, inplace=True)
    return TRY_polygons


def savefig_filetypes(save_folder, filename, filetypes=None, dpi=200):
    """Save active matplotlib figure to all given file types."""
    if save_folder is not None and filetypes is not None:
        # Make sure the save path exists
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)

        if 'png' in filetypes:
            plt.savefig(os.path.join(save_folder, filename+'.png'),
                        bbox_inches='tight', dpi=dpi)
        if 'svg' in filetypes:
            plt.savefig(os.path.join(save_folder, filename+'.svg'),
                        bbox_inches='tight')
        if 'pdf' in filetypes:
            plt.savefig(os.path.join(save_folder, filename+'.pdf'),
                        bbox_inches='tight')
