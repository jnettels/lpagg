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


Module main
-----------
This script provides the entry point function for the load profile
aggregator program.
"""

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
    """Run the main function, which contains the rest of the script."""
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
    agg_dict = lpagg.agg.aggregator_run(cfg)
    weather_data = agg_dict['weather_data']
    # print(weather_data)

    # Plot & Print
    lpagg.agg.plot_and_print(weather_data, cfg)

    # Print a final message with the required time
    script_time = pd.to_timedelta(time.time() - start_time, unit='s')
    logger.info('Finished script in time: %s' % (script_time))

    if settings.get('show_plot', False) is True:
        plt.show()  # Script is blocked until the user closes the plot window

    if args.block:  # Prevent console from closing
        input('\nPress the enter key to exit.')


def run_OptionParser():
    """Define and run the argument parser. Return the collected arguments."""
    import argparse
    description = 'The load profile aggregator combines profiles for heat '\
                  'and power demand of buildings from different sources.'
    parser = argparse.ArgumentParser(description=description,
                                     formatter_class=argparse.
                                     ArgumentDefaultsHelpFormatter)

    parser.add_argument('-f', '--file', dest='file', help='Path to a YAML '
                        'configuration file.', type=str, default=None)
    parser.add_argument('-b', '--block', dest='block', help='If true, '
                        'require a user key press to finish the script.',
                        action=argparse.BooleanOptionalAction,
                        default=False)

    args = parser.parse_args()

    if args.file is None:
        args.file = lpagg.misc.file_dialog(title='Choose a yaml config file',
                                           filetypes=(('YAML File', '*.yaml'),)
                                           )
        if args.file is None:
            logger.info('Empty selection. Show help and exit program...')
            parser.print_help()

            if args.block:  # Prevent console from closing
                input('\nPress the enter key to exit.')
            raise SystemExit

    return args


if __name__ == '__main__':
    """This part is executed when the script is started directly with
    Python, not when it is loaded as a module."""
    try:  # Wrap everything in a try-except to show exceptions with the logger
        main()
    except Exception as e:
        logger.exception(e)
        # Prevent console from closing
        input('\nPress the enter key to exit.')
