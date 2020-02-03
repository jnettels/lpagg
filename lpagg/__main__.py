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
import matplotlib.pyplot as plt  # Plotting library
import time                      # Measure time
import logging

# Import local modules from load profile aggregator project
import lpagg.misc
import lpagg.agg

# Define the logging function
logger = logging.getLogger(__name__)


def main():
    '''
    The following is the 'main' function, which contains the rest of the script
    '''
    lpagg.misc.setup()

    # Measure start time of this script
    start_time = time.time()

    #  Get user input
    args = run_OptionParser()

    # Import the config YAML file and add the default settings
    cfg = lpagg.agg.perform_configuration(args.file)

    # Read settings from the cfg
    settings = cfg['settings']
    logger.setLevel(level=settings.get('log_level', 'INFO').upper())

    # Aggregate load profiles
    weather_data = lpagg.agg.aggregator_run(cfg)
#    print(weather_data)

    # Plot & Print
    lpagg.agg.plot_and_print(weather_data, cfg)

    # Print a final message with the required time
    script_time = pd.to_timedelta(time.time() - start_time, unit='s')
    logger.info('Finished script in time: %s' % (script_time))

    if settings.get('show_plot', False) is True:
        plt.show()  # Script is blocked until the user closes the plot window

    input('\nPress the enter key to exit.')


def run_OptionParser():
    '''Define and run the argument parser. Return the collected arguments.
    '''
    import argparse
    description = 'The load profile aggregator combines profiles for heat '\
                  'and power demand of buildings from different sources.'
    parser = argparse.ArgumentParser(description=description,
                                     formatter_class=argparse.
                                     ArgumentDefaultsHelpFormatter)

    parser.add_argument('-f', '--file', dest='file', help='Path to a YAML '
                        'configuration file.', type=str, default=None)

    args = parser.parse_args()

    if args.file is None:
        args.file = lpagg.misc.file_dialog(title='Choose a yaml config file',
                                           filetypes=(('YAML File', '*.yaml'),)
                                           )
        if args.file is None:
            logger.info('Empty selection. Show help and exit program...')
            parser.print_help()
            input('\nPress the enter key to exit.')
            raise SystemExit

    return args


if __name__ == '__main__':
    '''This part is executed when the script is started directly with
    Python, not when it is loaded as a module.
    '''
    try:  # Wrap everything in a try-except to show exceptions with the logger
        main()
    except Exception as e:
        logger.exception(e)
        input('\nPress the enter key to exit.')  # Prevent console from closing
