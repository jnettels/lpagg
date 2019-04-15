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
        mpl.style.use('../futureSuN.mplstyle')
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
    if file == '':
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
