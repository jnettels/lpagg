#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
**dwd2trnsys: Convert DWD weather data to TRNSYS**

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


dwd2trnsys
==========

Script for converting weather data downloaded from Deutscher Wetterdienst
(DWD) to the format required by the TRNSYS weather Type 99.
https://kunden.dwd.de/obt/

An internet connection is required to convert the coordinates.
'''

import pandas as pd               # Pandas
import os                         # Operaing System
import logging
import datetime
import lpagg.weather_converter as wc

# Define the logging function
logger = logging.getLogger(__name__)


def main():
    '''Read weather file downloaded from DWD and convert it to a TRNSYS
    weather file for Type 99.
    '''

    # Global Pandas option for displaying terminal output
    pd.set_option('display.max_columns', 0)

    # Create logger for this module
    logging.basicConfig(format='%(asctime)-15s %(levelname)-8s %(message)s')
#    log_level = 'DEBUG'
    log_level = 'INFO'
    logger.setLevel(level=log_level)
    logging.getLogger('lpagg.weather_converter').setLevel(level=log_level)

    ''' Script options
    '''
    datetime_start = datetime.datetime(2017, 1, 1, 00, 00, 00)
    datetime_end = datetime.datetime(2018, 1, 1, 00, 00, 00)
    interpolation_freq = pd.Timedelta('1 hours')

    bool_print_header = False
    bool_print_index = False
    remove_leapyear = False
    weather_data_type = 'DWD'

    weather_file_list = wc.run_OptionParser()
    base_folder = os.path.dirname(weather_file_list[0])

    for i, weather_file_path in enumerate(weather_file_list):
        print_folder = os.path.join(base_folder, 'Result')

        # Read and interpolate weather data:
        weather_data = wc.interpolate_weather_file(weather_file_path,
                                                   weather_data_type,
                                                   datetime_start,
                                                   datetime_end,
                                                   interpolation_freq,
                                                   remove_leapyear)

        # Analyse weather data
        weather_file = os.path.basename(weather_file_path)
        wc.analyse_weather_file(weather_data, interpolation_freq, weather_file,
                                print_folder)

        # Create header text for type99 weather file
        type99_header = wc.get_type99_header(weather_file_path,
                                             interpolation_freq)

        # Print the weather data file
        wc.print_IGS_weather_file(weather_data, print_folder, weather_file,
                                  bool_print_index, bool_print_header,
                                  type99_header=type99_header)


if __name__ == "__main__":
    try:
        main()
    except Exception as ex:
        logger.exception(ex)
    input('\nPress the enter key to exit.')
