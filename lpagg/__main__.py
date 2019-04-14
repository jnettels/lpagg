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


Module main
-----------
This script provides the entry point function for the load profile
aggregator program.

'''
import pandas as pd              # Pandas
import matplotlib as mpl
import matplotlib.pyplot as plt  # Plotting library
import time                      # Measure time
import multiprocessing           # Parallel (Multi-) Processing
import logging

# Import local modules from load profile aggregator project
import misc
import agg

# Define the logging function
logger = logging.getLogger(__name__)


def main():
    '''
    The following is the 'main' function, which contains the rest of the script
    '''
    setup()

    # --- Measure start time of this script -----------------------------------
    start_time = time.time()

    # --- Script options ------------------------------------------------------
    config_file = None
#    config_file = r'V:\MA\2_Projekte\SIZ10015_futureSuN\4_Bearbeitung\AP4_Transformation\AP404_Konzepte für zukünftige Systemlösungen\Lastprofile\VDI 4655\Berechnung\VDI_4655_config.yaml'
#    config_file = r'V:\MA\2_Projekte\SIZ10015_futureSuN\4_Bearbeitung\AP4_Transformation\AP401_Zukünftige Funktionen\Quellen\RH+TWE\VDI_4655_config.yaml'
#    config_file = r'C:\Trnsys17\Work\futureSuN\AP1\SB\Load\VDI_4655_config_Steinfurt_02.yaml'
#    config_file = r'C:\Trnsys17\Work\futureSuN\AP4\P2H_Quartier\Load\VDI_4655_config_P2HQuartier.yaml'
#    config_file = r'C:\Trnsys17\Work\futureSuN\AP4\P2H_Quartier\Load\VDI_4655_config_Hannover-Kronsberg.yaml'
#    config_file = r'C:\Trnsys17\Work\futureSuN\AP4\Referenz_Quartier_Neubau\Load\VDI_4655_config_Quartier_Neubau.yaml'
#    config_file = r'C:\Users\nettelstroth\Documents\02 Projekte - Auslagerung\SIZ10019_Quarree100_Heide\Load\VDI_4655_config.yaml'
#    config_file = r'V:\MA\2_Projekte\SIZ10015_futureSuN\4_Bearbeitung\AP4_Transformation\AP404_Konzepte für zukünftige Systemlösungen\03_Sonnenkamp\Lastprofile\VDI_4655_config_Sonnenkamp.yaml'
#    config_file = r'C:\Trnsys17\Work\SIZ055_Meldorf\Load\Meldorf_load_config.yaml'
#    config_file = r'C:\Trnsys17\Work\SIZ10022_Quarree100\Load\VDI_4655_Q100_Kataster.yaml'

    if config_file is None:  # show file dialog
        config_file = misc.file_dialog(title='Choose a yaml config file',
                                       filetypes=(('YAML File', '*.yaml'),))
        if config_file is None:
            logger.error('Empty selection. Exit program...')
            input('\nPress the enter key to exit.')
            raise SystemExit

    # Import the config YAML file and add the default settings
    cfg = agg.perform_configuration(config_file)

    # Read settings from the cfg
    settings = cfg['settings']
    logger.setLevel(level=settings['log_level'].upper())

    # Aggregate load profiles
    weather_data = agg.aggregator_run(cfg)
#    print(weather_data)

    # Plot & Print
    agg.plot_and_print(weather_data, cfg)

    # Print a final message with the required time
    script_time = pd.to_timedelta(time.time() - start_time, unit='s')
    logger.info('Finished script in time: %s' % (script_time))

    if settings.get('show_plot', False) is True:
        plt.show()  # Script is blocked until the user closes the plot window

    input('\nPress the enter key to exit.')


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


if __name__ == '__main__':
    '''This part is executed when the script is started directly with
    Python, not when it is loaded as a module.
    '''
    main()
