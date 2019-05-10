# -*- coding: utf-8 -*-
'''
**LPagg: Load profile aggregator for building simulations**

Copyright (C) 2019 Joris Nettelstroth

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program. If not, see https://www.gnu.org/licenses/.


LPagg
=====
The load profile aggregator combines profiles for heat and power demand
of buildings from different sources.


Module misc
-----------
This is a collection of miscellaneous functions that are shared by the other
modules in the project.

'''
import time                      # Measure time
import multiprocessing           # Parallel (Multi-) Processing
import os                        # Operaing System
import logging
import pandas as pd              # Pandas
import matplotlib as mpl
from tkinter import Tk, filedialog

# Define the logging function
logger = logging.getLogger(__name__)


def setup():
    '''Perform some basic setup.
    '''
    multiprocessing.freeze_support()  # Needed for parallel processing

    # Global Pandas option for displaying terminal output
    pd.set_option('display.max_columns', 0)

    # Define the logging function
    logging.basicConfig(format='%(asctime)-15s %(message)s')

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
    '''This function presents a file dialog.

    Args:
        None

    Return:
        path (str): File path
    '''
    root = Tk()
    root.withdraw()
    file = filedialog.askopenfilename(initialdir=initialdir, title=title,
                                      filetypes=filetypes)
    if file == '' or len(file)==0:
        path = None
    else:
        path = os.path.abspath(file)
    return path


def multiprocessing_job(helper_function, work_list):
    '''Generalization of multiprocessing with integrated progress printing.
    '''
    number_of_cores = min(multiprocessing.cpu_count()-1, len(work_list))
    logger.info('Parallel processing on '+str(number_of_cores)+' cores')
    pool = multiprocessing.Pool(number_of_cores)
    return_list = pool.map_async(helper_function, work_list)
    pool.close()

    total = return_list._number_left
    while return_list.ready() is False:
        remaining = return_list._number_left
        fraction = (total - remaining)/total
        print('{:5.1f}% done'.format(fraction*100), end='\r')  # print progress
        time.sleep(1.0)

    pool.join()
    print('100.0% done', end='\r')

    return return_list


def resample_energy(df, freq):
    '''Resample energy time series data to a new frequency, while keeping
    the total energy sum constant.

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

    '''
    from pandas.tseries.frequencies import to_offset

    freq_orig = pd.infer_freq(df.index, warn=True)
    freq_orig = pd.to_timedelta(to_offset(freq_orig))
    f_freq = freq/freq_orig
    if f_freq < 1:
        # Reindex
        start = df.index[0] - (freq_orig - freq)
        end = df.index[-1]
        df = df.reindex(pd.date_range(start=start, end=end, freq=freq))

        # Fill all resulting missing values
        df.fillna(method='backfill', inplace=True)

        # Divide by factor f_freq to keep the total energy constant
        df *= f_freq

    elif f_freq > 1:
        df = df.resample(rule=freq, label='right', closed='right').sum()

    return df


def df_to_excel(df, path, sheet_names=[], merge_cells=False,
                check_permission=True, **kwargs):
    '''Wrapper around pandas' function ``DataFrame.to_excel()``, which creates
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
    '''
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
            logger.critical('Writing instead to:  '+path_time)
            df_to_excel(df, path_time, sheet_names=sheet_names,
                        merge_cells=merge_cells, **kwargs)
            return  # Do not run the rest of the function

    # Here the 'actual' function content starts:
    if not os.path.exists(os.path.dirname(path)):
        logging.debug('Create directory ' + os.path.dirname(path))
        os.makedirs(os.path.dirname(path))

    if isinstance(df, Sequence) and not isinstance(df, str):
        # Save a list of DataFrame objects into a single Excel file
        writer = pd.ExcelWriter(path)
        for i, df_ in enumerate(df):
            try:  # Use given sheet name, or just an enumeration
                sheet = sheet_names[i]
            except IndexError:
                sheet = str(i)
            # Add current sheet to the ExcelWriter by calling this
            # function recursively
            df_to_excel(df=df_, path=writer, sheet_name=sheet,
                        merge_cells=merge_cells, **kwargs)
        writer.save()  # Save the actual Excel file

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
            del(kwargs['freeze_panes'])

        # Save one DataFrame to one Excel file
        df.to_excel(path, merge_cells=merge_cells, **kwargs)
