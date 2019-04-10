#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
Implementation of VDI 4655

This script creates full year energy demand time series of domestic buildings
for use in TRNSYS. This is achieved by an implementation of the VDI 4655,
which gives sample energy demands for a number of typical days ('Typtage').
The energy demand contains heating, hot water and electricity.

For a given year, the typical days can be matched to the actual calendar days,
based on the following conditions:
    - Season: summer, winter or transition
    - Day: weekday or sunday (Or holiday, which counts as sunday)
    - Cloud coverage: cloudy or not cloudy
    - House type: single-family houses or multi-family houses (EFH or MFH)

The holidays are loaded from an Excel file, the weather conditions are loaded
from weather data (e.g. DWD TRY).
Most of the settings for this script are controlled with a configuration file
called 'config_file', the location of which is defined down below.
'''

# --- Imports -----------------------------------------------------------------
import pandas as pd              # Pandas
from pandas.tseries.frequencies import to_offset
import os                        # Operaing System
import yaml                      # Read YAML configuration files
import matplotlib as mpl
import matplotlib.pyplot as plt  # Plotting library
import time                      # Measure time
import multiprocessing           # Parallel (Multi-) Processing
import functools
import numpy as np
from tkinter import Tk, filedialog
from scipy import optimize
import logging
import pickle

# Import other user-made modules (which must exist in the same folder)
import weather_converter        # Script for interpolation of IGS weather files


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


def load_weather_file(settings):
    weather_file = settings['weather_file']
    weather_file_path = os.path.join(os.path.dirname(__file__),
                                     'resources_weather', weather_file)
    weather_data_type = settings['weather_data_type']
    datetime_start = pd.datetime(*settings['start'])  # * reads list as args
    datetime_end = pd.datetime(*settings['end'])
#    datetime_start = pd.datetime(2017,1,1,00,00,00) # Example
#    datetime_end = pd.datetime(2018,1,1,00,00,00)
    interpolation_freq = pd.Timedelta(settings['intervall'])
#    interpolation_freq = pd.Timedelta('14 minutes')
#    interpolation_freq = pd.Timedelta('1 hours')
    remove_leapyear = settings.get('remove_leapyear', False)

    settings['interpolation_freq'] = interpolation_freq
    """
    ---------------------------------------------------------------------------
    Read and interpolate "IGS Referenzklimaregion" files
    ---------------------------------------------------------------------------
    """
    logger.info('Read and interpolate the data in weather file '+weather_file)

    # Call external method in weather_converter.py:
    weather_data = weather_converter.interpolate_weather_file(
                                    weather_file_path,
                                    weather_data_type,
                                    datetime_start,
                                    datetime_end,
                                    interpolation_freq,
                                    remove_leapyear)

    # Analyse weather data
    if logger.isEnabledFor(logging.INFO):
        weather_converter.analyse_weather_file(
                weather_data, interpolation_freq, weather_file,
                print_folder=settings['print_folder'])
    weather_data.index.name = 'Time'
    return weather_data


def get_season_list_BDEW(weather_data):
    '''
    Winter:       01.11. to 20.03.
    Summer:       15.05. to 14.09.
    Transition:   21.03. to 14.05. and 15.09. to 31.10.
    '''
    season_list = []

    for j, date_obj in enumerate(weather_data.index):
        YEAR = date_obj.year
        if date_obj <= pd.datetime(YEAR, 3, 21, 00, 00, 00) or \
           date_obj > pd.datetime(YEAR, 10, 31, 00, 00, 00):
            season_list.append('Winter')  # Winter

        elif date_obj > pd.datetime(YEAR, 5, 15, 00, 00, 00) and \
                date_obj <= pd.datetime(YEAR, 9, 15, 00, 00, 00):
            season_list.append('Sommer')  # Summer

        else:
            season_list.append('Übergangszeit')  # Transition

    return season_list


def get_typical_days(weather_data, settings):
    # -------------------------------------------------------------------------
    # VDI 4655 - Step 1: Determine the "typtag" key for each timestep
    # -------------------------------------------------------------------------
    # Flag to determine if any holidays have been found:
    interpolation_freq = pd.Timedelta(settings['intervall'])
    flag_holidays_found = False

    # --- Season --------------------------------------------------------------
    # The 'season' (transition, summer or winter) is defined by the daily
    # average of the ambient temperature.

    # Resample ambient temperatures in DataFrame to days and take mean
    tamb_avg_list = weather_data['TAMB'].resample('D', label='right',
                                                  closed='right').mean()

    # Write the daily mean values to all original time steps
    tamb_avg_list = tamb_avg_list.reindex(weather_data.index)
    tamb_avg_list.fillna(method='backfill', inplace=True)

    season_list = []

    # The VDI 4655 default heat limit is 15°C (definition of summer days).
    # For low- and zero-energy houses, the average daily temperatures have
    # to be adapted to the actual conditions. (see VDI 4655, page 15)
    Tamb_heat_limit = settings.get('Tamb_heat_limit', 15)  # °C

    # Read through list of temperatures line by line and apply the definition
    for tamb_avg in tamb_avg_list:
        if tamb_avg < 5:
            season_list.append('W')  # Winter
        elif tamb_avg > Tamb_heat_limit:
            season_list.append('S')  # Summer
        else:
            season_list.append('U')  # Übergang (Transition)

    # Alternative season determination method:
    # From 'BDEW Standardlastprofile':
    season_list_BDEW = get_season_list_BDEW(weather_data)

    # Save the results in the weather_data DataFrame
    weather_data['TAMB_d'] = tamb_avg_list
    if use_BDEW_seasons is False:
        weather_data['season'] = season_list
    elif use_BDEW_seasons is True:
        weather_data['season'] = season_list_BDEW
        weather_data['season'].replace(to_replace={'Winter': 'W',
                                                   'Sommer': 'S',
                                                   'Übergangszeit': 'U'},
                                       inplace=True)

    # Store the BDEW seasons separately
    weather_data['season_BDEW'] = season_list_BDEW

    steps_per_day = 24 / (interpolation_freq.seconds / 3600.0)
    settings['steps_per_day'] = steps_per_day
    logger.debug('Number of days in winter:     ' +
                 str(season_list.count('W')/steps_per_day))
    logger.debug('Number of days in summer:     ' +
                 str(season_list.count('S')/steps_per_day))
    logger.debug('Number of days in transition: ' +
                 str(season_list.count('U')/steps_per_day))

    # --- Workdays, Sundays and public holidays -------------------------------
    holidays = pd.read_excel(open(holiday_file, 'rb'),
                             sheet_name='Feiertage',
                             index_col=[0])
    # Read through list of days line by line and see what kind of day they are.
    # Problem: In the weather data, the bins are labeled on the 'right'
    # (Each time stamp describes the interval before). Therefore the time stamp
    # midnight (00:00:00) describes the last interval of the day before.
    # However, asking for the weekday of a midnight time stamp gives the name
    # of the next day. Thus the resulting list of weekdays is shifted by one
    # time step.
    weekdays_list = []
    weekdays_list_BDEW = []
    for date_obj in weather_data.index:
        if date_obj.dayofweek == 6:  # 6 equals Sunday
            weekdays_list.append('S')
            weekdays_list_BDEW.append('Sonntag')
        elif date_obj.date() in holidays.index:
            weekdays_list.append('S')
            weekdays_list_BDEW.append('Sonntag')
            flag_holidays_found = True
        elif date_obj.dayofweek == 5:  # 5 equals Saturday
            weekdays_list.append('W')
            weekdays_list_BDEW.append('Samstag')
        else:
            weekdays_list.append('W')
            weekdays_list_BDEW.append('Werktag')

    # Solution to problem: We take the first list entry, then add the rest of
    # the list minus the very last entry.
    weather_data['weekday'] = [weekdays_list[0]] + weekdays_list[:-1]
    weather_data['weekday_BDEW'] = [weekdays_list_BDEW[0]] + \
        weekdays_list_BDEW[:-1]

    # Print a warning, if necessary
    if flag_holidays_found is False:
        logger.warning('Warning! No holidays were found for the chosen time!')

    # --- Cloud cover amount --------------------------------------------------
    ccover_avg_list = weather_data['CCOVER'].resample('D', label='right',
                                                      closed='right').mean()
    ccover_avg_list = ccover_avg_list.reindex(weather_data.index)
    ccover_avg_list.fillna(method='backfill', inplace=True)
    # The interpolation to 15min may cause a slight difference of daily means
    # compared to 60min, in rare cases shifting from >5.0 to <5.0.
    # Rounding to the first decimal place may prevent this issue.
    ccover_avg_list = ccover_avg_list.round(decimals=1)

    # Read through list of cloud cover line by line and apply the definition
    cloudy_list = []
    for ccover_avg in ccover_avg_list:
        if (ccover_avg < 5.0):
            cloudy_list.append('H')
        else:
            cloudy_list.append('B')

    weather_data['cloudy'] = cloudy_list

    # Combine the gathered information from season, weekday and cloudyness
    # into one 'typtag' key
    weather_data['typtag'] = weather_data['season'] + \
        weather_data['weekday'] + weather_data['cloudy']

    # For summer days, the VDI 4655 makes no distinction in terms of cloud
    # amount. So we need to replase 'heiter' and 'bewölkt' with 'X'
    typtage_replace = {'typtag':
                       {'SWH': 'SWX', 'SWB': 'SWX', 'SSH': 'SSX', 'SSB': 'SSX'}
                       }
    weather_data.replace(to_replace=typtage_replace, inplace=True)


def load_profile_factors(settings):
    '''VDI 4655 - Step 2:
    Match 'typtag' keys and reference load profile factors for each timestep
    (for each 'typtag' key, one load profile is defined by VDI 4655)
    '''
    interpolation_freq = settings['interpolation_freq']
    # Define all 'typtag' combinations and house types:
    typtage_combinations = ['UWH', 'UWB', 'USH', 'USB', 'SWX',
                            'SSX', 'WWH', 'WWB', 'WSH', 'WSB']
    house_types = ['EFH', 'MFH']
    settings['typtage_combinations'] = typtage_combinations

    energy_factor_types = ['F_Heiz_n_TT', 'F_el_n_TT', 'F_TWW_n_TT']
    settings['energy_factor_types'] = energy_factor_types

    if logger.isEnabledFor(logging.DEBUG):
        logger.debug('Number of typical days:')
        steps_per_day = settings['steps_per_day']
        N_typtage = weather_data['typtag'].value_counts()/steps_per_day
        for item in typtage_combinations:
            if item not in N_typtage.index:
                N_typtage[item] = 0  # Set the 'typtage' not found to zero
        print(pd.DataFrame(N_typtage).T.to_string(index=False,
              columns=typtage_combinations,  # fixed sorting of columns
              float_format='{:.0f}'.format))

    if settings.get('pickle_load_profile', False):
        # If the previous run had the same interpolation_freq, we do not need
        # to re-do the calculation of load_profile_df. Instead we load it
        # from a pickle file that was saved last time
        load_profile_pickle = 'load_profile_df.pkl'
        try:
            # Load an existing optimizer instance
            with open(load_profile_pickle, 'rb') as f:
                load_profile_df = pickle.load(f)
        except Exception:
            pass
        else:
            loaded_freq = pd.infer_freq(load_profile_df.index, warn=True)
            delta = pd.Timedelta(pd.tseries.frequencies.to_offset(loaded_freq))
            if delta == interpolation_freq:
                logger.info('Pickle: Loaded existing load_profile_df '
                            + load_profile_pickle)
                return load_profile_df

    # Load all 'typtag' load profile files into one DataFrame:
    # Please note:
    # These tables contain energy demand values for the period between the time
    # indicated in the time stamp column and the time in the next line.
    # They are labeled 'left', while the IGS-Weatherdata is labeled 'right'!

    # Read the excel workbook, which will return a dict of the sheets
    typtage_sheets_dict = pd.read_excel(open(typtage_file, 'rb'),
                                        sheet_name=None,
                                        index_col=[0, 1])
    # The DataFrame within every dict entry is combined to one large DataFrame
    typtage_df = pd.DataFrame()  # create a new DataFrame that is empty
    for sheet in typtage_sheets_dict:
        typtage_df_new = typtage_sheets_dict[sheet]
        typtage_df = pd.concat([typtage_df, typtage_df_new])

    # The column 'Zeit' is of the type datetime.time. It must be converted
    # to datetime.datetime by adding an arbitrary datetime.date object
    datetime_column = []
    for row, time_obj in enumerate(typtage_df['Zeit']):
        datetime_obj = pd.datetime.combine(pd.datetime(2017, 1, 1), time_obj)
        datetime_column.append(datetime_obj)
    typtage_df['Zeit'] = datetime_column
    # Now the column 'Zeit' can be added to the multiindex
    typtage_df.set_index('Zeit', drop=True, append=True, inplace=True)

    # EFH come in resolution of 1 min, MFH in 15 min. We need to get MFH down
    # to 1 min.
    # Unstack, so that only the time index remains. This creates NaNs for the
    # missing time stamps in MFH columns
    typtage_df = typtage_df.unstack(['Haus', 'typtag'])
    # Fill those NaN values with 'forward fill' method
    typtage_df.fillna(method='ffill', inplace=True)
    # Divide by factor 15 to keep the total energy demand constant
    typtage_df.loc[:, (slice(None), 'MFH',)] *= 1/15

    # In the next step, the values are summed up into the same time intervals
    # as the weather data, and the label is moved to the 'right'
    # (each time stamp now describes the data in the interval before)
    typtage_df = typtage_df.resample(rule=interpolation_freq, level='Zeit',
                                     label='right', closed='left').sum()
    typtage_df = typtage_df.stack(['Haus', 'typtag'])
    typtage_df = typtage_df.reorder_levels(['Haus', 'typtag', 'Zeit'])
    typtage_df = typtage_df.sort_index()

    # Create a new DataFrame with the load profile for the chosen time period,
    # using the correct house type, 'typtag' and time step

    # Contruct the two-dimensional multiindex of the new DataFrame
    # (One column for each energy_factor_type of each house)
    iterables = [energy_factor_types, house_types]
    multiindex = pd.MultiIndex.from_product(iterables,
                                            names=['energy', 'house'])
    load_profile_df = pd.DataFrame(index=weather_data.index,
                                   columns=multiindex)

    # Fill the load profile's time steps with the matching energy factors
    # Iterate over time slices of full days
    start = weather_data.index[0]
    while start < weather_data.index[-1]:
        end = start + pd.Timedelta('1 days') - interpolation_freq
        print('Progress: '+str(start), end='\r')  # print progress
        # Compare time stamps in typtage_df of the matching house and typtag
        typtag = weather_data.loc[start]['typtag']

        typtage_df.loc[house_types[0], typtag].index
        start_tt = pd.datetime.combine(pd.datetime(2017, 1, 1), start.time())
        end_tt = start_tt + pd.Timedelta('1 days') - interpolation_freq

        for energy in energy_factor_types:
            for house_type in house_types:
                load_profile_df.loc[start:end, (energy, house_type)] = \
                    typtage_df.loc[house_type, typtag,
                                   start_tt:end_tt][energy].values

        start = end + interpolation_freq

    # Debugging: The daily sum of each energy factor type must be '1':
#    if logger.isEnabledFor(logging.DEBUG):
#        print("The daily sum of each energy factor type must equal '1':")
#        print(load_profile_df.resample('D', label='left',
#                                       closed='right').sum())

    if settings.get('pickle_load_profile', False):
        with open(load_profile_pickle, 'wb') as f:
            pickle.dump(load_profile_df, f)

    return load_profile_df


def get_annual_energy_demand(settings):
    '''Read in houses and calculate their annual energy demand.

    VDI 4655 provides estimates for annual electrical and DHW energy demand
    (``W_a`` and ``Q_TWW_a``). ``Q_Heiz_TT`` cannot be estimated, but must
    be defined in the config file.
    If ``W_a`` or ``Q_TWW_a` are defined in the config file, their estimation
    is not used.
    '''
    # Get the dictionary of houses from the config_dict
    houses_dict = config_dict['houses']
    houses_list = sorted(houses_dict.keys())
    settings['houses_list'] = houses_list
    settings['houses_list_VDI'] = []
    settings['houses_list_BDEW'] = []

    # Calculate annual energy demand of houses
    # and store the result in the dict containing the house info
    for house_name in houses_list:
        house_type = houses_dict[house_name]['house_type']
        N_Pers = houses_dict[house_name].get('N_Pers', None)
        N_WE = houses_dict[house_name].get('N_WE', None)

        # Assign defaults if values are not defined
        if N_Pers is None:
            N_Pers = 3
            houses_dict[house_name]['N_Pers'] = N_Pers
            logger.warning('N_Pers not defined for ' + str(house_name)
                           + '. Using default ' + str(N_Pers))
        if N_WE is None:
            N_WE = 2
            houses_dict[house_name]['N_WE'] = N_WE
            logger.warning('N_WE not defined for ' + str(house_name)
                           + '. Using default ' + str(N_WE))

        # Implement the restrictions defined on page 3:
        if house_type == 'EFH' and N_Pers > 12:
            logger.warning('VDI 4655 is only defined for N_Pers <= 12. '
                           + str(house_name) + ' uses N_Pers = ' + str(N_Pers)
                           + '. Proceeding with your input...')
        if house_type == 'MFH' and N_WE > 40:
            logger.warning('VDI 4655 is only defined for N_WE <= 40. '
                           + str(house_name) + ' uses N_WE = ' + str(N_WE)
                           + '. Proceeding with your input...')

        # Calculate annual energy demand estimates
        if house_type == 'EFH':
            # (6.2.2) Calculate annual electrical energy demand of houses:
            if N_Pers < 3:
                W_a = N_Pers * 2000  # kWh
            elif N_Pers <= 6:
                W_a = N_Pers * 1750  # kWh
            else:
                W_a = N_Pers * 1500  # kWh

            # (6.2.3) Calculate annual DHW energy demand of houses:
            Q_TWW_a = N_Pers * 500  # kWh

            settings['houses_list_VDI'].append(house_name)

        elif house_type == 'MFH':
            # (6.2.2) Calculate annual electrical energy demand of houses:
            W_a = N_WE * 3000  # kWh

            # (6.2.3) Calculate annual DHW energy demand of houses:
            Q_TWW_a = N_WE * 1000  # kWh

            settings['houses_list_VDI'].append(house_name)

        else:
            # No house category given. Just use annual demand of 1 kWh
            W_a = 1
            Q_TWW_a = 1
            settings['houses_list_BDEW'].append(house_name)

        # If W_a and/or Q_TWW_a were already defined by the user in the yaml
        # file, we use those values instead of the calculated ones:
        W_a = houses_dict[house_name].get('W_a', W_a)
        Q_TWW_a = houses_dict[house_name].get('Q_TWW_a', Q_TWW_a)

        # Store the results in the dict
        houses_dict[house_name]['W_a'] = W_a
        houses_dict[house_name]['Q_TWW_a'] = Q_TWW_a

        # Assign defaults if values are not defined
        if houses_dict[house_name].get('Q_Heiz_a', None) is None:
            Q_Heiz_a = 1  # kWh
            houses_dict[house_name]['Q_Heiz_a'] = Q_Heiz_a
            logger.warning('Q_Heiz_a not defined for ' + house_name
                           + '. Using default ' + str(Q_Heiz_a) + ' kWh')

        # Apply the adjustment factors
        houses_dict[house_name]['Q_Heiz_a'] *= \
            config_dict.get('adjustment_factors', dict()).get('f_Q_Heiz', 1)

        houses_dict[house_name]['W_a'] *= \
            config_dict.get('adjustment_factors', dict()).get('f_W', 1)

        houses_dict[house_name]['Q_TWW_a'] *= \
            config_dict.get('adjustment_factors', dict()).get('f_Q_TWW', 1)

    return houses_dict


def get_daily_energy_demand_houses(houses_dict, settings):
    '''Determine the houses' energy demand values for each 'typtag'


    .. note::
        "The factors ``F_el_TT`` and ``F_TWW_TT`` are negative in some cases as
        they represent a variation from a one-year average. The values for the
        daily demand for electrical energy, ``W_TT``, and DHW energy,
        ``Q_TWW_TT``, usually remain positive. It is only in individual
        cases that the calculation for the typical-day category ``SWX``
        can yield a negative value of the DHW demand. In that case,
        assume ``F_TWW_SWX`` = 0." (VDI 4655, page 16)

        This occurs when ``N_Pers`` or ``N_WE`` are too large.

    '''
    typtage_combinations = settings['typtage_combinations']
    houses_list = settings['houses_list_VDI']

    # Load the file containing the energy factors of the different typical
    # radiation year (TRY) regions, house types and 'typtage'. In VDI 4655,
    # these are the tables 10 to 24.
    energy_factors_df = pd.read_excel(open(energy_factors_file, 'rb'),
                                      sheet_name='Faktoren',
                                      index_col=[0, 1, 2])

    if settings.get('zero_summer_heat_demand', None) is not None:
        # Reduze the value of 'F_Heiz_TT' to zero.
        # For modern houses, this eliminates the heat demand in summer
        energy_factors_df.loc[(slice(None), slice(None), 'F_Heiz_TT'),
                              ('SWX', 'SSX')] = 0

    # Create a new DataFrame with multiindex.
    # It has two levels of columns: houses and energy
    # The DataFrame stores the individual energy demands for each house in
    # each time step
    energy_demands_types = ['Q_Heiz_TT', 'W_TT', 'Q_TWW_TT']
    settings['energy_demands_types'] = energy_demands_types
    iterables = [houses_dict.keys(), energy_demands_types]
    multiindex = pd.MultiIndex.from_product(iterables, names=['house',
                                                              'energy'])
    daily_energy_demand_houses = pd.DataFrame(index=multiindex,
                                              columns=typtage_combinations)

    # Fill the DataFrame daily_energy_demand_houses
    for house_name in houses_list:
        house_type = houses_dict[house_name]['house_type']
        N_Pers = houses_dict[house_name]['N_Pers']
        N_WE = houses_dict[house_name]['N_WE']
        try:
            TRY = houses_dict[house_name]['TRY']
        except KeyError:
            raise KeyError('Key "TRY" (Region) missing from house '+house_name)

        # Savety check:
        if TRY not in energy_factors_df.index.get_level_values(0):
            logger.error('Error! TRY '+str(TRY)+' not contained in file ' +
                         energy_factors_file)
            logger.error('       Skipping house "'+house_name+'"!')
            continue  # 'Continue' skips the rest of the current for-loop

        # Get yearly energy demands
        Q_Heiz_a = houses_dict[house_name]['Q_Heiz_a']
        W_a = houses_dict[house_name]['W_a']
        Q_TWW_a = houses_dict[house_name]['Q_TWW_a']

        # (6.4) Do calculations according to VDI 4655 for each 'typtag'
        for typtag in typtage_combinations:
            F_Heiz_TT = energy_factors_df.loc[TRY, house_type,
                                              'F_Heiz_TT'][typtag]
            F_el_TT = energy_factors_df.loc[TRY, house_type, 'F_el_TT'][typtag]
            F_TWW_TT = energy_factors_df.loc[TRY, house_type,
                                             'F_TWW_TT'][typtag]

            Q_Heiz_TT = Q_Heiz_a * F_Heiz_TT

            if house_type == 'EFH':
                N_Pers_WE = N_Pers
            elif house_type == 'MFH':
                N_Pers_WE = N_WE

            W_TT = W_a * (1.0/365.0 + N_Pers_WE * F_el_TT)
            Q_TWW_TT = Q_TWW_a * (1.0/365.0 + N_Pers_WE * F_TWW_TT)

            if W_TT < 0:
                logger.warning('Warning:     W_TT for '+house_name+' and ' +
                               typtag + ' was negative, see VDI 4655 page 16')
                W_TT = W_a * (1.0/365.0 + N_Pers_WE * 0)

            if Q_TWW_TT < 0:
                logger.warning('Warning: Q_TWW_TT for '+house_name+' and ' +
                               typtag + ' was negative, see VDI 4655 page 16')
                Q_TWW_TT = Q_TWW_a * (1.0/365.0 + N_Pers_WE * 0)

            # Write values into DataFrame
            daily_energy_demand_houses.loc[house_name,
                                           'Q_Heiz_TT'][typtag] = Q_Heiz_TT
            daily_energy_demand_houses.loc[house_name,
                                           'W_TT'][typtag] = W_TT
            daily_energy_demand_houses.loc[house_name,
                                           'Q_TWW_TT'][typtag] = Q_TWW_TT

#    print(daily_energy_demand_houses)
    return daily_energy_demand_houses


def get_load_curve_houses(load_profile_df, houses_dict, settings):
    '''Generate the houses' energy demand values for each timestep
    '''
    energy_demands_types = settings['energy_demands_types']
    houses_list = settings['houses_list_VDI']
    energy_factor_types = settings['energy_factor_types']

    # Construct the DataFrame with multidimensional index that
    # contains the energy demand of each house for each time stamp
    iterables = [houses_list, energy_demands_types]
    multiindex = pd.MultiIndex.from_product(iterables,
                                            names=['house', 'energy'])
    load_curve_houses = pd.DataFrame(index=weather_data.index,
                                     columns=multiindex)

    # 'Partial' creates a function that only takes one argument. In our case
    # this is 'date_obj'. It will be given to the target function
    # 'get_energy_demand_values' as the last argument.
    helper_function = functools.partial(get_energy_demand_values,
                                        weather_data, houses_list, houses_dict,
                                        energy_factor_types,
                                        energy_demands_types,
                                        load_curve_houses, load_profile_df,
                                        daily_energy_demand_houses)

    work_list = weather_data.index
    return_list = multiprocessing_job(helper_function, work_list)

    # The 'pool' returns a list. Feed its contents to the DataFrame
    for returned_df in return_list.get():
        load_curve_houses.loc[returned_df.name] = returned_df

    load_curve_houses = load_curve_houses.astype('float')

    # For each time step, each house and each type of energy factor, we
    # multiply the energy factor with the daily energy demand. The result
    # is the energy demand of that time interval.
    # We save it to the load_curve_houses DataFrame.
    # total = float(len(weather_data.index))
    # for j, date_obj in enumerate(weather_data.index):
        # helper_function(date_obj)

        # print ('{:5.1f}% done'.format(j/total*100), end='\r')  # progress

    # The typical day calculation inherently does not add up to the
    # desired total energy demand of the full year. Here we fix that:
    for column in load_curve_houses.columns:
        if column[1] == 'Q_Heiz_TT':
            Q_a = houses_dict[column[0]]['Q_Heiz_a']
        elif column[1] == 'Q_TWW_TT':
            Q_a = houses_dict[column[0]]['Q_TWW_a']
        elif column[1] == 'W_TT':
            Q_a = houses_dict[column[0]]['W_a']
        sum_ = load_curve_houses[column].sum()
        if sum_ != 0:  # Would produce NaN otherwise
            load_curve_houses[column] = load_curve_houses[column]/sum_ * Q_a

    return load_curve_houses


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


def get_energy_demand_values(weather_data, houses_list, houses_dict,
                             energy_factor_types, energy_demands_types,
                             load_curve_houses, load_profile_df,
                             daily_energy_demand_houses, date_obj):
    '''
    This functions works through the lists houses_list and energy_factor_types
    for a given time step (=date_obj) and multiplies the current load profile
    value with the daily energy demand. It returns the result: the energy
    demand values for all houses and energy types (in kWh)
    '''

    typtag = weather_data.loc[date_obj]['typtag']
    for house_name in houses_list:
        house_type = houses_dict[house_name]['house_type']
        for i, energy_factor_type in enumerate(energy_factor_types):
            energy_demand_type = energy_demands_types[i]
            # Example: Q_Heiz_TT(t) = F_Heiz_TT(t) * Q_Heiz_TT
            load_curve_houses.loc[date_obj][house_name, energy_demand_type] =\
                load_profile_df.loc[date_obj][energy_factor_type,
                                              house_type] *\
                daily_energy_demand_houses.loc[house_name,
                                               energy_demand_type][typtag]

    return load_curve_houses.loc[date_obj]


def load_BDEW_style_profiles(source_file, weather_data, settings, houses_dict,
                             energy_type):
    '''Load energy profiles from files that are structured like BDEW profiles.
    Is used for BDEW profiles, and allows profiles from other sources to
    be integrated easily. For example, the U.S. Department of Energy (DOE)
    profiles for building types can manually be converted to the BDEW format,
    then loaded with this function.
    '''

    source_df = pd.read_excel(open(source_file, 'rb'), sheet_name=None,
                              skiprows=[0], header=[0, 1], index_col=[0],
                              skipfooter=1,
                              )

    weather_daily = weather_data.resample('D', label='right',
                                          closed='right').mean()
#    print(weather_daily)

    houses_list = settings['houses_list_BDEW']
    multiindex = pd.MultiIndex.from_product([houses_list, [energy_type]],
                                            names=['house', 'energy'])
    ret_profiles = pd.DataFrame(index=weather_data.index,
                                columns=multiindex)
    if len(houses_list) == 0:  # Skip
        return ret_profiles

    for house_name in houses_list:
        house_type = houses_dict[house_name]['house_type']
        if house_type not in source_df.keys():
            # Only use 'H0G', 'G0G', 'G1G', ...
            logger.warning('house_type "'+str(house_type)+'" not found in '
                           'profile sources: '+str(source_df.keys()))
            continue

        profile_year = pd.Series()  # Yearly profile for the current house
        for date in weather_daily.index:
            weekday = weather_data.loc[date]['weekday_BDEW']
            season = weather_data.loc[date]['season_BDEW']
            # Important: In order identify the weekday of the resampled days,
            # we labled them 'right'. From now on we need the label 'left',
            # so we substract '1 day' from each date:
            date -= pd.Timedelta('1 day')

            source_profile = source_df[house_type][season, weekday]

            # Combine date and time stamps
            profile_daily = source_profile.copy()
            index_new = []
            for time_idx in source_profile.index:
                try:
                    # Turn time stamp into a time difference (delta)
                    delta = pd.to_timedelta(str(time_idx))
                    if delta == pd.to_timedelta(0):
                        # Delta of zero must be a full day
                        delta = pd.to_timedelta('24 h')
                except Exception:
                    # The last entry of each profile ('0:00') is sometimes
                    # stored as a full date (1900-1-1 00:00) in Excel
                    delta = pd.to_timedelta('24 h')

                # Create full time stamp of date and time for the new index
                datetime_idx = date + delta
                index_new.append(datetime_idx)

            profile_daily.index = index_new
#            print(profile_daily)

            # Append to yearly profile
            profile_year = pd.concat([profile_year, profile_daily])

        # Store in DataFrame that will be returned
        ret_profiles[house_name, energy_type] = profile_year

    # Resample to the desired frequency (time intervall)
    source_freq = pd.infer_freq(profile_daily.index)
    source_freq = pd.to_timedelta(to_offset(source_freq))

    if source_freq < settings['interpolation_freq']:
        # In case of downsampling (to longer time intervalls) take the sum
        ret_profiles = ret_profiles.resample(settings['interpolation_freq'],
                                             label='right',
                                             closed='right').sum()
    elif source_freq > settings['interpolation_freq']:
        # In case of upsampling (to shorter time intervalls) use backwards fill
        ret_profiles.fillna(method='backfill', inplace=True)

    return ret_profiles


def load_BDEW_profiles(weather_data, settings, houses_dict):

    source_file = BDEW_file
    energy_type = 'W_TT'
    BDEW_profiles = load_BDEW_style_profiles(source_file, weather_data,
                                             settings, houses_dict,
                                             energy_type)

    # Rescale to the given yearly energy demand:
    for column in BDEW_profiles.columns:
        W_a = houses_dict[column[0]]['W_a']
        yearly_sum = BDEW_profiles[column].sum()
        BDEW_profiles[column] = BDEW_profiles[column]/yearly_sum * W_a

    return BDEW_profiles


def load_DOE_profiles(weather_data, settings, houses_dict):

    source_file = DOE_file
    energy_type = 'Q_TWW_TT'
    DOE_profiles = load_BDEW_style_profiles(source_file, weather_data,
                                            settings, houses_dict,
                                            energy_type)

    # Rescale to the given yearly energy demand:
    for column in DOE_profiles.columns:
        Q_TWW_a = houses_dict[column[0]]['Q_TWW_a']
        yearly_sum = DOE_profiles[column].sum()
        DOE_profiles[column] = DOE_profiles[column]/yearly_sum * Q_TWW_a

    return DOE_profiles


def load_futureSolar_profiles(weather_data, settings, houses_dict):

    houses_list = settings['houses_list_BDEW']
#    print(houses_list)
    # BDEW_file
    futureSolar_df = pd.read_excel(open(futureSolar_file, 'rb'), index_col=[0],
                                   sheet_name='Profile', header=[0, 1])
    futureSolar_df.index = pd.to_timedelta(futureSolar_df.index, unit='h')

    energy_types = ['Q_Heiz_TT', 'Q_Kalt_TT']
    multiindex = pd.MultiIndex.from_product([houses_list, energy_types],
                                            names=['house', 'energy'])
    futureSolar_profiles = pd.DataFrame(index=weather_data.index,
                                        columns=multiindex)
    if len(houses_list) == 0:  # Skip
        return futureSolar_profiles

#    print(futureSolar_df.keys())

    for shift_steps, date_obj in enumerate(weather_data.index):
        if date_obj.dayofweek == 1:  # 1 equals Tuesday
            first_tuesday = date_obj
#            print(shift_steps, date_obj)
            break

    futureSolar_df.index = first_tuesday + futureSolar_df.index

    futureSolar_df = futureSolar_df.resample(settings['interpolation_freq'],
                                             label='right',
                                             closed='right').sum()

    overlap = futureSolar_df[-(shift_steps + 1):]
    overlap.index = overlap.index - pd.Timedelta('365 days')
    futureSolar_df = overlap.append(futureSolar_df)

    for house_name in houses_list:
        house_type = houses_dict[house_name]['house_type']
        if house_type not in futureSolar_df.keys():
            # Only use 'G1G' and 'G4G'
            continue

        for energy_type in energy_types:
            profile_year = futureSolar_df[house_type, energy_type]
#            print(profile_year)
            futureSolar_profiles[house_name, energy_type] = profile_year

    # Rescale to the given yearly energy demand:
    for column in futureSolar_profiles.columns:
        if column[1] == 'Q_Heiz_TT':
            Q_a = houses_dict[column[0]].get('Q_Heiz_a', 1)
        elif column[1] == 'Q_Kalt_TT':
            Q_a = houses_dict[column[0]].get('Q_Kalt_a', 1)
        sum_ = futureSolar_profiles[column].sum()
        futureSolar_profiles[column] = futureSolar_profiles[column]/sum_ * Q_a

#    print(futureSolar_profiles)
    return futureSolar_profiles


def flatten_daily_TWE(load_curve_houses, settings):
    '''Flatten domestic hot water profile to a daily mean value.

    The domestic hot water demand profile represents what is actually
    used within the house. But the total demand of a house with hot
    water circulation is nearly constant for each day (from the
    perspective of a district heating system).
    '''

    if settings.get('flatten_daily_TWE', False):

        logger.info('Create (randomized) copies of the houses')

        # Resample TWW energy in DataFrame to days and take mean
        Q_TWW_avg_list = load_curve_houses.loc[
                :, (slice(None), slice(None), 'Q_TWW_TT')]
        Q_TWW_avg_list = Q_TWW_avg_list.resample('D', label='right',
                                                 closed='right').mean()
        # Write the daily mean values to all original time steps
        Q_TWW_avg_list = Q_TWW_avg_list.reindex(load_curve_houses.index)
        Q_TWW_avg_list.fillna(method='backfill', inplace=True)
        # Overwrite original DataFrame
        load_curve_houses.loc[
                :, (slice(None), slice(None), 'Q_TWW_TT')] = Q_TWW_avg_list
#        print(load_curve_houses)

    return load_curve_houses


def copy_and_randomize_houses(load_curve_houses, houses_dict, settings):
    '''Create copies of houses where needed. Apply a normal distribution to
    the copies, if a standard deviation ``sigma`` is given in the config.

    Remember: 68.3%, 95.5% and 99.7% of the values lie within one,
    two and three standard deviations of the mean.
    Example: With an interval of 15 min and a deviation of
    sigma = 2 time steps, 68% of profiles are shifted up to ±30 min (±1σ).
    27% of profiles are shifted ±30 to 60 min (±2σ) and another
    4% are shifted ±60 to 90 min (±3σ).

    This method decreases the maximum load and thereby creates a
    "simultaneity factor" (Gleichzeitigkeitsfaktor). It can be calculated by
    dividing the maximum loads with and without the normal distribution.

    For a plausibilty check see pages 3 and 12 of:
    Winter, Walter; Haslauer, Thomas; Obernberger, Ingwald (2001):
    Untersuchungen der Gleichzeitigkeit in kleinen und mittleren
    Nahwärmenetzen. In: Euroheat & Power (9/10). Online verfügbar unter
    http://www.bios-bioenergy.at/uploads/media/Paper-Winter-Gleichzeitigkeit-Euroheat-2001-09-02.pdf

    '''
    load_curve_houses = load_curve_houses.swaplevel('house', 'class', axis=1)
    load_curve_houses_ref = load_curve_houses.copy()  # Reference (no random)
    # Fix the 'randomness' (every run of the script generates the same results)
    np.random.seed(4)

    # Create a temporary dict with all the info needed for randomizer
    randomizer_dict = dict()
    for house_name in settings['houses_list']:
        copies = houses_dict[house_name].get('copies', 0)
        # Get standard deviation (spread or “width”) of the distribution:
        sigma = houses_dict[house_name].get('sigma', False)

        randomizer_dict[house_name] = dict({'copies': copies,
                                            'sigma': sigma})

    external_profiles = config_dict.get('external_profiles', dict())
    for house_name in external_profiles:
        copies = external_profiles[house_name].get('copies', 0)
        # Get standard deviation (spread or “width”) of the distribution:
        sigma = external_profiles[house_name].get('sigma', False)

        randomizer_dict[house_name] = dict({'copies': copies,
                                            'sigma': sigma})

    # Create copies for every house
    for house_name in randomizer_dict:
        copies = randomizer_dict[house_name]['copies']
        sigma = randomizer_dict[house_name]['sigma']
        randoms = np.random.normal(0, sigma, copies)  # Array of random values
        randoms_int = [int(value) for value in np.round(randoms, 0)]

        work_list = list(range(0, copies))
        if len(work_list) > 100:
            # Use multiprocessing to increase the speed
            f_help = functools.partial(mp_copy_and_randomize,
                                       load_curve_houses, house_name,
                                       randoms_int, sigma)
            return_list = multiprocessing_job(f_help, work_list)
            # Merge the existing and new dataframes
            df_list_tuples = return_list.get()

            df_list = [x[0] for x in df_list_tuples]
            df_list_ref = [x[1] for x in df_list_tuples]

            load_curve_houses = pd.concat([load_curve_houses]+df_list,
                                          axis=1, sort=False)
            load_curve_houses_ref = pd.concat([load_curve_houses_ref]
                                              + df_list_ref,
                                              axis=1, sort=False)
        else:
            # Implementation in serial
            for copy in range(0, copies):
                df_new, df_ref = mp_copy_and_randomize(load_curve_houses,
                                                       house_name,
                                                       randoms_int, sigma,
                                                       copy, b_print=True)
                # Merge the existing and new dataframes
                load_curve_houses = pd.concat([load_curve_houses, df_new],
                                              axis=1, sort=False)
                load_curve_houses_ref = pd.concat([load_curve_houses_ref,
                                                   df_ref],
                                                  axis=1, sort=False)

        if sigma and logger.isEnabledFor(logging.DEBUG):
            logger.debug('Interval shifts applied to copies of house '
                         + house_name + ':')
            print(randoms_int)
            mu = np.mean(randoms_int)
            sigma = np.std(randoms_int, ddof=1)
            text_mean_std = 'Mean = {:0.2f}, std = {:0.2f}'.format(mu, sigma)
            title_mu_std = r'$\mu={:0.3f},\ \sigma={:0.3f}$'.format(mu, sigma)
            print(text_mean_std)

            # Make sure the save path exists
            save_folder = os.path.join(base_folder, 'Result')
            if not os.path.exists(save_folder):
                os.makedirs(save_folder)

            # Create a histogram of the data
            limit = max(-1*min(randoms_int), max(randoms_int))
            bins = range(-limit, limit+2)
            plt.rcParams.update({'font.size': 16})
            fig = plt.figure()
            ax = fig.gca()
            ax.yaxis.grid(True)  # Activate grid on horizontal axis
            n, bins, patches = plt.hist(randoms_int, bins, align='left',
                                        rwidth=0.9)
            plt.title(str(copies)+' Kopien von Gebäude '+house_name +
                      ' ('+title_mu_std+')')
            plt.xlabel('Zeitschritte')
            plt.ylabel('Anzahl')
            plt.savefig(os.path.join(save_folder,
                                     'histogram_'+house_name+'.png'),
                        bbox_inches='tight', dpi=200)

            if settings.get('show_plot', False) is True:
                plt.show(block=False)  # Show plot without blocking the script

    # Calculate "simultaneity factor" (Gleichzeitigkeitsfaktor)
    calc_GLF(load_curve_houses, load_curve_houses_ref, settings)

    load_curve_houses = load_curve_houses.swaplevel('house', 'class', axis=1)
    return load_curve_houses


def mp_copy_and_randomize(load_curve_houses, house_name, randoms_int, sigma,
                          copy, b_print=False):
    copy_name = str(house_name) + '_c' + str(copy)
    if b_print and logger.isEnabledFor(logging.DEBUG):
        print('Copy (and randomize) house', copy_name, end='\r')  # status
    # Select the data of the house we want to copy
    df_new = load_curve_houses[house_name]
    # Rename the multiindex with the name of the copy
    df_new = pd.concat([df_new], keys=[copy_name], names=['house'],
                       axis=1)
    df_ref = df_new.copy()

    if sigma:  # Optional: Shift the rows
        shift_step = randoms_int[copy]
        if shift_step > 0:
            # Shifting forward in time pushes the last entries out of
            # the df and leaves the first entries empty. We take that
            # overlap and insert it at the beginning
            overlap = df_new[-shift_step:]
            df_shifted = df_new.shift(shift_step)
            df_shifted.dropna(inplace=True, how='all')
            overlap.index = df_new[:shift_step].index
            df_new = overlap.append(df_shifted)
        elif shift_step < 0:
            # Retrieve overlap from the beginning of the df, shift
            # backwards in time, paste overlap at end of the df
            overlap = df_new[:abs(shift_step)]
            df_shifted = df_new.shift(shift_step)
            df_shifted.dropna(inplace=True, how='all')
            overlap.index = df_new[shift_step:].index
            df_new = df_shifted.append(overlap)
        elif shift_step == 0:
            # No action required
            pass

    if b_print:
        print(' ', end='\r')  # overwrite last status with empty line
    return df_new, df_ref


def calc_GLF(load_curve_houses, load_curve_houses_ref, settings):
    '''Calculate "simultaneity factor" (Gleichzeitigkeitsfaktor)
    Uses a DataFrame with and one without randomness.
    '''
    load_curve_houses_ran = load_curve_houses.copy()
    load_ran = load_curve_houses_ran.groupby(level='energy', axis=1).sum()
    load_ref = load_curve_houses_ref.groupby(level='energy', axis=1).sum()

    hours = settings['interpolation_freq'].seconds / 3600.0        # h
    sf_df = pd.DataFrame(index=['P_max_kW', 'P_max_ref_kW', 'GLF'],
                         columns=['th_RH', 'th_TWE', 'th', 'el'])
    sf_df.loc['P_max_kW', 'th_RH'] = load_ran['Q_Heiz_TT'].max() / hours
    sf_df.loc['P_max_kW', 'th_TWE'] = load_ran['Q_TWW_TT'].max() / hours
    sf_df.loc['P_max_kW', 'th'] = (load_ran['Q_Heiz_TT']
                                   + load_ran['Q_TWW_TT']).max() / hours
    sf_df.loc['P_max_kW', 'el'] = load_ran['W_TT'].max() / hours

    sf_df.loc['P_max_ref_kW', 'th_RH'] = load_ref['Q_Heiz_TT'].max() / hours
    sf_df.loc['P_max_ref_kW', 'th_TWE'] = load_ref['Q_TWW_TT'].max() / hours
    sf_df.loc['P_max_ref_kW', 'th'] = (load_ref['Q_Heiz_TT']
                                       + load_ref['Q_TWW_TT']).max() / hours
    sf_df.loc['P_max_ref_kW', 'el'] = load_ref['W_TT'].max() / hours

    sf_df.loc['GLF'] = sf_df.loc['P_max_kW']/sf_df.loc['P_max_ref_kW']

    if logger.isEnabledFor(logging.INFO):
        logger.info('Simultaneity factors (Gleichzeitigkeitsfaktoren):')
        print(sf_df)
    # Make sure the save path exists and save the DataFrame
    save_folder = os.path.join(base_folder, 'Result')
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    sf_df.to_excel(os.path.join(save_folder, 'GLF.xlsx'))

    return None


def add_external_profiles(load_curve, settings):
    '''This allows to add additional external profiles to the calculated
    load curves.
    Currently, they need to have the same time step as all other data, no
    interpolation is performed.
    '''
    building_dict = config_dict.get('external_profiles', dict())
    for building in building_dict:
        filepath = building_dict[building]['file']
        class_ = building_dict[building]['class']
        rename_dict = building_dict[building]['rename_dict']
        read_excel_kwargs = building_dict[building]['read_excel_kwargs']
#        logger.debug(filepath)

        # Read data from file
        filetype = os.path.splitext(os.path.basename(filepath))[1]
        if filetype in ['.xlsx', '.xls']:
            # Excel can be read automatically with Pandas
            df = pd.read_excel(filepath, **read_excel_kwargs)
        elif filetype in ['.csv', '.dat']:
            # csv files can have different formats
            df = pd.read_csv(open(filepath, 'r'),
                             sep=None, engine='python',  # Guess separator
                             parse_dates=[0],  # Parse first column as date
                             infer_datetime_format=True)

        # Apply multiplication factor
        multiply_dict = building_dict[building].get('multiply_dict', dict())
        for key, value in multiply_dict.items():
            df[key] *= value

        # Rename to VDI 4655 standards
        df.rename(columns=rename_dict, inplace=True)

        freq = pd.infer_freq(df.index, warn=True)
        f_freq = settings['interpolation_freq']/freq
        if f_freq < 1:
            df = df.resample(rule=settings['interpolation_freq']).bfill()
            # Divide by factor f_freq to keep the total energy demand constant
            df *= f_freq
        elif f_freq > 1:
            df = df.resample(rule=settings['interpolation_freq'],
                             label='right', closed='right').sum()

        # Convert into a multi-index DataFrame
        multiindex = pd.MultiIndex.from_product(
            [[class_], [building], rename_dict.values()],
            names=['class', 'house', 'energy'])
        ext_profiles = pd.DataFrame(index=df.index, columns=multiindex)
        for col in rename_dict.values():
            ext_profiles[class_, building, col] = df[col]

        # Combine external profiles with existing profiles
        load_curve = pd.concat([load_curve, ext_profiles], axis=1)

        # Fill missing values (after resampling to a smaller timestep, the
        # beginning will have missing values that can be filled with backfill)
        load_curve.fillna(method='backfill', inplace=True)

    return load_curve


def sum_up_all_houses(load_curve_houses, weather_data):
    '''By grouping the 'energy' level, we can take the sum of all houses.
    Also renames the columns from VDI 4655 standard to 'E_th', 'E_el',
    as used in the project futureSuN.
    '''
    load_curve_houses = load_curve_houses.stack('class')
    load_curve_houses_sum = load_curve_houses.groupby(level='energy',
                                                      axis=1).sum()
    load_curve_houses_sum = load_curve_houses_sum.unstack('class')
    load_curve_houses_sum.rename(columns={'Q_Heiz_TT': 'E_th_RH',
                                          'Q_Kalt_TT': 'E_th_KL',
                                          'Q_TWW_TT': 'E_th_TWE',
                                          'Q_loss': 'E_th_loss',
                                          'W_TT': 'E_el'},
                                 inplace=True)
#    print(load_curve_houses_sum)

    # Flatten the heirarchical index
    settings['energy_demands_types'] = ['_'.join(col).strip() for col in
                                        load_curve_houses_sum.columns.values]

    load_curve_houses_sum.columns = settings['energy_demands_types']
    # Concatenate the weather_data and load_curve_houses_sum DataFrames
    weather_data = pd.concat([weather_data, load_curve_houses_sum], axis=1)
    return weather_data


def calc_heizkurve(weather_data, settings):
    '''Implementation 'Heizkurve (Vorlauf- und Rücklauftemperatur)'
    (Not part of the VDI 4655)

    Calculation according to:
    Knabe, Gottfried (1992): Gebäudeautomation. 1. Aufl.
    Berlin: Verl. für Bauwesen.
    Section 6.2.1., pages 267-268
    '''
    interpolation_freq = settings['interpolation_freq']

    if config_dict.get('Heizkurve', None) is not None:
        logger.info('Calculate heatcurve temperatures')

        T_VL_N = config_dict['Heizkurve']['T_VL_N']       # °C
        T_RL_N = config_dict['Heizkurve']['T_RL_N']       # °C
        T_i = config_dict['Heizkurve']['T_i']             # °C
        T_a_N = config_dict['Heizkurve']['T_a_N']         # °C
        m = config_dict['Heizkurve']['m']

        Q_loss_list = []
        T_VL_list = []
        T_RL_list = []
        M_dot_list = []

        total = float(len(weather_data.index))
        for j, date_obj in enumerate(weather_data.index):
            T_a = weather_data.loc[date_obj]['TAMB']           # °C
            hours = interpolation_freq.seconds / 3600.0        # h
            c_p = 4.18                                         # kJ/(kg*K)

            # Calculate temperatures and mass flow for heating
            if (T_a < T_i):  # only calculate if heating is necessary
                phi = (T_i - T_a) / (T_i - T_a_N)
                dT_N = (T_VL_N - T_RL_N)                                 # K
                dTm_N = (T_VL_N + T_RL_N)/2.0 - T_i                      # K

                T_VL_Heiz = phi**(1/(1+m)) * dTm_N + 0.5*phi*dT_N + T_i  # °C
                T_RL_Heiz = phi**(1/(1+m)) * dTm_N - 0.5*phi*dT_N + T_i  # °C

                Q_Heiz = 0
                try:  # Households may or may not have been defined
                    Q_Heiz += weather_data.loc[date_obj]['E_th_RH_HH']   # kWh
                except Exception:
                    pass
                try:  # GHD may or may not have been defined
                    Q_Heiz += weather_data.loc[date_obj]['E_th_RH_GHD']  # kWh
                except Exception:
                    pass

                Q_dot_Heiz = Q_Heiz / hours                              # kW
                M_dot_Heiz = Q_dot_Heiz/(c_p*(T_VL_Heiz-T_RL_Heiz))*3600
                # unit of M_dot_Heiz: kg/h

            else:  # heating is not necessary
                T_VL_Heiz = T_RL_N
                T_RL_Heiz = T_RL_N
                Q_dot_Heiz = 0
                M_dot_Heiz = 0

            # Calculate temperatures and mass flow for domestic hot water
            T_VL_TWW = T_VL_N
            T_RL_TWW = T_RL_N

            Q_TWW = 0
            try:  # Households may or may not have been defined
                Q_TWW += weather_data.loc[date_obj]['E_th_TWE_HH']    # kWh
            except Exception:
                pass
            try:  # GHD may or may not have been defined
                Q_TWW += weather_data.loc[date_obj]['E_th_TWE_GHD']   # kWh
            except Exception:
                pass
            Q_dot_TWW = Q_TWW / hours                                 # kW
            M_dot_TWW = Q_dot_TWW/(c_p*(T_VL_TWW - T_RL_TWW))*3600    # kg/h

            # Mix heating and domestic hot water
            M_dot = M_dot_Heiz + M_dot_TWW

            if (M_dot > 0):
                T_VL = (T_VL_Heiz*M_dot_Heiz + T_VL_TWW*M_dot_TWW) / M_dot
                T_RL = (T_RL_Heiz*M_dot_Heiz + T_RL_TWW*M_dot_TWW) / M_dot
            else:
                T_VL = T_RL_N
                T_RL = T_RL_N

            # Calculate the heat loss from the pipes to the environment
            # Careful: The heat loss calculation is far from perfect!
            # For instance, it is only used when a massflow occurs (M_dot>0).
            # As a result, smaller time step calculations will have much less
            # losses, since the massflow is zero more often.
            Q_loss = 0
            if (M_dot > 0) and (config_dict.get('Verteilnetz') is not None):
                length = config_dict['Verteilnetz']['length']            # m
                q_loss = config_dict['Verteilnetz']['loss_coefficient']
                # unit of q_loss: W/(m*K)

                dTm = (T_VL + T_RL)/2.0 - T_a                       # K
                Q_dot_loss = 2 * length * q_loss * dTm / 1000.0     # kW
                Q_loss = Q_dot_loss * hours                         # kWh

                # Calculate the increased mass flow based on the new heat flow
                Q_dot = Q_dot_Heiz + Q_dot_TWW + Q_dot_loss         # kW
                M_dot = Q_dot/(c_p*(T_VL - T_RL))*3600.0            # kg/h

    #            print(T_VL, T_RL, T_a, dTm)
    #            print(Q_dot_Heiz, Q_dot_TWW, Q_dot_loss, Q_loss)
    #            print(M_dot_Heiz + M_dot_TWW, M_dot)

            # Save calculation
            Q_loss_list.append(Q_loss)
            T_VL_list.append(T_VL)
            T_RL_list.append(T_RL)
            M_dot_list.append(M_dot)
            print('{:5.1f}% done'.format(j/total*100), end='\r')  # progress

        weather_data['E_th_loss'] = Q_loss_list
        weather_data['T_VL'] = T_VL_list
        weather_data['T_RL'] = T_RL_list
        weather_data['M_dot'] = M_dot_list

    else:  # Create dummy columns if no heatcurve calculation was performed
        weather_data['E_th_loss'] = 0
        weather_data['T_VL'] = 0
        weather_data['T_RL'] = 0
        weather_data['M_dot'] = 0

    # Add the loss to the energy demand types
    if 'E_th_loss' not in settings['energy_demands_types']:
        settings['energy_demands_types'].append('E_th_loss')


def fit_heizkurve(weather_data, config_dict):
    '''In some cases a certain total heat loss of the district heating pipes
    is desired, and the correct heat loss coefficient has to be determined.
    In this case (an optimization problem), we need to find the root of
    the function fit_pipe_heat_loss().

    If 'loss_total_kWh' is not defined in the config file, this whole process
    is skipped.

    Args:
        weather_data (DataFrame): Includes all weather and energy time series

        config_dict (Dict):  Configuration dict with initial loss_coefficient

    Returns:
        config_dict (Dict):  Configuration dict with updated loss_coefficient
    '''

    def fit_pipe_heat_loss(loss_coefficient):
        '''Helper function that is created as an input for SciPy's optimize
        function. Returns the difference between target and current heat loss.

        Args:
            loss_coefficient (Array): Careful! 'fsolve' will provide an array

        Returns:
            error (float): Deviation between set and current heat loss
        '''
        E_th_loss_set = config_dict['Verteilnetz']['loss_total_kWh']
        config_dict['Verteilnetz']['loss_coefficient'] = loss_coefficient[0]

        calc_heizkurve(weather_data, settings)
        E_th_loss = weather_data['E_th_loss'].sum()
        error = E_th_loss_set - E_th_loss
        print('Fitting E_th_loss_set:', E_th_loss_set, '... E_th_loss:',
              E_th_loss, 'loss_coefficient:', loss_coefficient[0])
        return error

    loss_kWh = config_dict.get('Verteilnetz', dict()).get('loss_total_kWh',
                                                          None)
    if loss_kWh is None:
        #  Skip the root finding procedure
        return config_dict  # return original config_dict
    else:
        func = fit_pipe_heat_loss
        x0 = config_dict['Verteilnetz']['loss_coefficient']
        root = optimize.fsolve(func, x0)  # Careful, returns an array!
        print("Solution: loss_coefficient = ", root, '[W/(m*K)]')
        config_dict['Verteilnetz']['loss_coefficient'] = root[0]
        return config_dict  # config with updated loss_coefficient


def normalize_energy(weather_data):
    '''Normalize results to a total of 1 kWh per year
    '''
    if config_dict.get('normalize', False) is True:
        logger.info('Normalize load profile')
        for column in settings['energy_demands_types']:
            yearly_sum = weather_data[column].sum()
            weather_data[column] = weather_data[column]/yearly_sum


if __name__ == '__main__':
    '''
    The following is the 'main' function, which contains the rest of the script
    '''
    multiprocessing.freeze_support()

    # --- Measure start time of this script -----------------------------------
    start_time = time.time()

    # Global Pandas option for displaying terminal output
    pd.set_option('display.max_columns', 0)

    # Define the logging function
    logging.basicConfig(format='%(asctime)-15s %(message)s')
    logger = logging.getLogger(__name__)

    # Define style settings for the plots
    try:  # Try to load personalized matplotlib style file
        mpl.style.use('../futureSuN.mplstyle')
    except OSError as ex:
        logger.warning(ex)

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
#    config_file = r'C:\Trnsys17\Work\SIZ10022_Quarree100\Load\VDI_4655_config_Quarree100_02.yaml'

    filedir = os.path.dirname(__file__)
    holiday_file = os.path.join(filedir, 'resources_load', 'Feiertage.xlsx')
    energy_factors_file = os.path.join(filedir, 'resources_load',
                                       'VDI 4655 Typtag-Faktoren.xlsx')
    typtage_file = os.path.join(filedir, 'resources_load',
                                'VDI 4655 Typtage.xlsx')
    BDEW_file = os.path.join(filedir, 'resources_load', 'BDEW Profile.xlsx')
    DOE_file = os.path.join(filedir, 'resources_load', 'DOE Profile TWE.xlsx')
    futureSolar_file = os.path.join(filedir, 'resources_load',
                                    'futureSolar Profile.xlsx')

    if config_file is None:
        config_file = file_dialog(
                title='Choose a yaml config file',
                filetypes=(('YAML File', '*.yaml'),),
                )  # show file dialog
        if config_file is None:
            logger.error('Empty selection. Exit program...')
            input('\nPress the enter key to exit.')
            raise SystemExit
    base_folder = os.path.dirname(config_file)

    # --- Import the config_dict from the YAML config_file --------------------
    config_dict = yaml.load(open(config_file, 'r'))

    # --- Read settings from the config_dict ----------------------------------
    settings = config_dict['settings']
    bool_print_header = settings.get('print_header', True)
    bool_print_index = settings.get('print_index', True)
    use_BDEW_seasons = settings.get('use_BDEW_seasons', False)

    print_file = settings['print_file']
    bool_show_plot = settings.get('show_plot', False)
    # Output folder is hardcoded here:
    print_folder = os.path.join(base_folder, 'Result')
    settings['print_folder'] = print_folder

    # Set logging level
    log_level = settings.get('log_level', 'WARNING')  # Use setting or default
    if settings.get('Debug', False):
        log_level = 'DEBUG'  # override with old 'DEBUG' key
    logger.setLevel(level=log_level.upper())
    logging.getLogger('weather_converter').setLevel(level=log_level.upper())

    logger.info('Using configuration file ' + config_file)

    # \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
    #                          VDI 4655 Implementation
    # \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
    weather_data = load_weather_file(settings)

    # -------------------------------------------------------------------------
    # VDI 4655 - Step 1: Determine the "typtag" key for each timestep
    # -------------------------------------------------------------------------
    logger.info('Determine "typtag" keys for each time step')
    get_typical_days(weather_data, settings)

    # -------------------------------------------------------------------------
    # VDI 4655 - Step 2:
    # Match 'typtag' keys and reference load profile factors for each timestep
    # (for each 'typtag' key, one load profile is defined by VDI 4655)
    # -------------------------------------------------------------------------
    logger.info('Read in reference load profile factors and match them to ' +
                '"typtag" keys for each timestep')
    load_profile_df = load_profile_factors(settings)

    # -------------------------------------------------------------------------
    # VDI 4655 - Step 3: Load houses and generate their load profiles
    #
    # In the end, we will multiply the energy factors contained in DataFrame
    # load_profile_df for each time step of the weather_data with the
    # corresponding daily energy demand of each house we want to simulate. This
    # will yield the DataFrame load_curve_houses, which contains the actual
    # energy demand of each house in each time step.
    # -------------------------------------------------------------------------

    # (6) Application of guideline:
    # -------------------------------------------------------------------------
    # The following numbered sections are the implementations of the
    # corresponding sections in the VDI 4655.

    # (6.1) Specification of building type:
    # -------------------------------------------------------------------------
    # Building type (EFH or MFH) is defined by user in YAML file.
    # - "single-family houses are residential buildings with up to
    #    three flasts and one common heating system"
    # - "multi-family houses are residential buildings with no less
    #    then four flasts and one common heating system"

    # (6.2) Specification of annual energy demand:
    # -------------------------------------------------------------------------
    logger.info('Read in houses and calculate their annual energy demand')
    houses_dict = get_annual_energy_demand(settings)

    # (6.3) Allocation of building site:
    # -------------------------------------------------------------------------
    # The user has to give the number of the TRY climat zone
    # in the yaml file. It is used in (6.4).

    # (6.4) Determination of the houses' energy demand values for each 'typtag'
    # -------------------------------------------------------------------------
    logger.info("Determine the houses' energy demand values for each typtag")
    daily_energy_demand_houses = get_daily_energy_demand_houses(houses_dict,
                                                                settings)

    # (6.5) Determination of a daily demand curve for each house:
    # -------------------------------------------------------------------------
    logger.info("Generate the houses' energy demand values for each timestep")
    load_curve_houses = get_load_curve_houses(load_profile_df, houses_dict,
                                              settings)

    # For the GHD building sector, combine profiles from various sources:
    # (not part of VDI 4655)
    # -------------------------------------------------------------------------
    BDEW_profiles = load_BDEW_profiles(weather_data, settings, houses_dict)
    DOE_profiles = load_DOE_profiles(weather_data, settings, houses_dict)
    futureSolar_profiles = load_futureSolar_profiles(weather_data, settings,
                                                     houses_dict)
    GHD_profiles = pd.concat([BDEW_profiles,
                              DOE_profiles,
                              futureSolar_profiles],
                             axis=1, sort=False)

    load_curve_houses = pd.concat([load_curve_houses, GHD_profiles],
                                  axis=1, sort=False,
                                  keys=['HH', 'GHD'], names=['class'])
#    print(load_curve_houses)

    # Flatten domestic hot water profile to a daily mean value
    # (Optional, not part of VDI 4655)
    # -------------------------------------------------------------------------
    load_curve_houses = flatten_daily_TWE(load_curve_houses, settings)
#    print(load_curve_houses.head())

    # Add external (complete) profiles
    # (Optional, not part of VDI 4655)
    # -------------------------------------------------------------------------
    load_curve_houses = add_external_profiles(load_curve_houses, settings)

    # Randomize the load profiles of identical houses
    # (Optional, not part of VDI 4655)
    # -------------------------------------------------------------------------
    logger.info('Create (randomized) copies of the houses')
    load_curve_houses = copy_and_randomize_houses(load_curve_houses,
                                                  houses_dict, settings)
#    print(load_curve_houses.head())

    # Debugging: Show the daily sum of each energy demand type:
#    print(load_curve_houses.resample('D', label='left', closed='right').sum())

    if logger.isEnabledFor(logging.DEBUG):
        print(load_curve_houses)
        try:
            load_curve_houses.to_csv(
                    os.path.join(print_folder, os.path.splitext(print_file)[0]
                                 + '_houses.dat'))
#            load_curve_houses.to_excel(
#                    os.path.join(print_folder, os.path.splitext(print_file)[0]
#                                 + '_houses.xlsx'))
        except Exception as ex:
            logger.debug(ex)

    # Sum up the energy demands of all houses, store result in weather_data
    logger.info('Sum up the energy demands of all houses')
    weather_data = sum_up_all_houses(load_curve_houses, weather_data)
#    print(weather_data)

    # \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
    #       Implementation 'Heizkurve (Vorlauf- und Rücklauftemperatur)'
    #                        (Not part of the VDI 4655)
    # \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
    config_dict = fit_heizkurve(weather_data, config_dict)  # Optional
    calc_heizkurve(weather_data, settings)

    # \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
    #                Normalize results to a total of 1 kWh per year
    # \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
    normalize_energy(weather_data)

#    print(weather_data)

    '''
    ---------------------------------------------------------------------------
                                   Plot & Print
    ---------------------------------------------------------------------------
    '''

    # Print a table of the energy sums to the console (monthly and annual)
    filter_sum = ['E_th_', 'E_el']
    filter_sum_heat = ['E_th_']
    sum_list = []
    sum_list_heat = []
    for column in weather_data.columns:
        for filter_ in filter_sum:
            if filter_ in column:
                sum_list.append(column)
        for filter_ in filter_sum_heat:
            if filter_ in column:
                sum_list_heat.append(column)

    # Set the number of decimal points for the following terminal output
    pd.set_option('precision', 2)

    weather_daily_sum = weather_data[sum_list].resample('D', label='left',
                                                        closed='right').sum()
    logger.info('Calculations completed')
    logger.info('Monthly energy sums in kWh:')
    weather_montly_sum = weather_daily_sum.resample('M', label='right',
                                                    closed='right').sum()
    if logger.isEnabledFor(logging.INFO):
        print(weather_montly_sum)
        print()
    logger.info('Annual energy sums in kWh:')
    weather_annual_sum = weather_montly_sum.resample('A', label='right',
                                                     closed='right').sum()
    if logger.isEnabledFor(logging.INFO):
        print(weather_annual_sum)
        print('Total heat energy demand is {:.2f} kWh.'.format(
            weather_annual_sum[sum_list_heat].sum(axis=1).sum()))
        print()

    pd.reset_option('precision')  # ...and reset the setting from above

    # Display a plot on screen for the user
    if bool_show_plot is True:
        logger.info('Showing plot of energy demand types...')
        fig = plt.figure()
        ax = fig.gca()
        for energy_demand_type in settings['energy_demands_types']:
            weather_data[energy_demand_type].plot(label=energy_demand_type)
        ax.yaxis.grid(True)  # Activate grid on horizontal axis
        plt.legend()
        plt.show(block=False)  # Show the plot, without blocking the script

    # Add a row at zero hours for the initialization in TRNSYS
    if settings.get('include_zero_row', False) is True:
        weather_data.loc[pd.datetime(*settings['start'])] = 0
        weather_data.sort_index(inplace=True)

    # Print the results file
    if settings.get('print_columns', None) is not None:
        # If defined, use only columns selected by the user
        for column in settings['print_columns']:
            if column not in weather_data.columns:
                # If a requested column is missing, create it
                weather_data[column] = 0
        try:
            weather_data = weather_data[settings['print_columns']]
        except Exception as ex:
            logger.exception(ex)

    # Call external method in weather_converter.py:
    weather_converter.print_IGS_weather_file(weather_data,
                                             print_folder,
                                             print_file,
                                             bool_print_index,
                                             bool_print_header)

    # Print a final message with the required time
    script_time = pd.to_timedelta(time.time() - start_time, unit='s')
    logger.info('Finished script in time: %s' % (script_time))

    if settings.get('show_plot', False) is True:
        plt.show()  # Script is blocked until the user closes the plot window

    input('\nPress the enter key to exit.')
