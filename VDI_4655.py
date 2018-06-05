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

# Import other user-made modules (which must exist in the same folder)
import zeitreihe_IGS_RKR        # Script for interpolation of IGS weather files


def load_weather_file(settings):
    weather_file = settings['weather_file']
    weather_file_path = os.path.join(base_folder, 'Wetter', weather_file)
    weather_data_type = settings['weather_data_type']
    datetime_start = pd.datetime(*settings['start'])  # * reads list as args
    datetime_end = pd.datetime(*settings['end'])
#    datetime_start = pd.datetime(2017,1,1,00,00,00) # Example
#    datetime_end = pd.datetime(2018,1,1,00,00,00)
    interpolation_freq = pd.Timedelta(settings['intervall'])
#    interpolation_freq = pd.Timedelta('14 minutes')
#    interpolation_freq = pd.Timedelta('1 hours')
    remove_leapyear = settings.get('remove_leapyear', False)

    # --- Savety check --------------------------------------------------------
    if interpolation_freq < pd.Timedelta('15 minutes'):
        print('Warning! Chosen interpolation intervall ' +
              str(interpolation_freq) +
              ' changed to 15 min. MFH load profiles are available in 15 ' +
              'min steps and interpolation to smaller time intervals is ' +
              'not (yet) implemented.')
        interpolation_freq = pd.Timedelta('15 minutes')
    settings['interpolation_freq'] = interpolation_freq
    """
    ---------------------------------------------------------------------------
    Read and interpolate "IGS Referenzklimaregion" files
    ---------------------------------------------------------------------------
    """
    print('Read and interpolate the data in weather file '+weather_file)

    # Call external method in zeitreihe_IGS_RKR.py:
    weather_data = zeitreihe_IGS_RKR.interpolate_weather_file(
                                    weather_file_path,
                                    weather_data_type,
                                    datetime_start,
                                    datetime_end,
                                    interpolation_freq,
                                    remove_leapyear)

    # Analyse weather data
    if DEBUG:
        zeitreihe_IGS_RKR.analyse_weather_file(weather_data,
                                               interpolation_freq,
                                               weather_file)
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
    # For modern building types, this can be set to a lower value
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
    if DEBUG:
        print('Number of days in winter:     ' +
              str(season_list.count('W')/steps_per_day))
        print('Number of days in summer:     ' +
              str(season_list.count('S')/steps_per_day))
        print('Number of days in transition: ' +
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
            weekdays_list.append('S')
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
        print('Warning! No holidays were found for the chosen time!')

    # --- Cloud cover amount --------------------------------------------------
    ccover_avg_list = weather_data['CCOVER'].resample('D', label='right',
                                                      closed='right').mean()
    ccover_avg_list = ccover_avg_list.reindex(weather_data.index)
    ccover_avg_list.fillna(method='backfill', inplace=True)

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

    if DEBUG:
        print('Number of typical days:')
        steps_per_day = settings['steps_per_day']
        N_typtage = weather_data['typtag'].value_counts()/steps_per_day
        for item in typtage_combinations:
            if item not in N_typtage.index:
                N_typtage[item] = 0  # Set the 'typtage' not found to zero
        print(pd.DataFrame(N_typtage).T.to_string(index=False,
              columns=typtage_combinations,  # fixed sorting of columns
              float_format='{:.0f}'.format))

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

    # In the next step, the values are summed up into the same time intervals
    # as the weather data, and the label is moved to the 'right'
    # (each time stamp now describes the data in the interval before)
    typtage_level_values = typtage_df.index.get_level_values
    typtage_df = typtage_df.groupby([typtage_level_values(i) for i in [0, 1]] +
                                    [pd.Grouper(freq=interpolation_freq,
                                                level=2, label='right',
                                                closed='left')]
                                    ).sum()

    # Create a new DataFrame with the load profile for the chosen time period,
    # using the correct house type, 'typtag' and time step
    energy_factor_types = ['F_Heiz_n_TT', 'F_el_n_TT', 'F_TWW_n_TT']
    settings['energy_factor_types'] = energy_factor_types

    # Contruct the two-dimensional multiindex of the new DataFrame
    # (One column for each energy_factor_type of each house)
    iterables = [energy_factor_types, house_types]
    multiindex = pd.MultiIndex.from_product(iterables,
                                            names=['energy', 'house'])
    load_profile_df = pd.DataFrame(index=weather_data.index,
                                   columns=multiindex)

    # Fill the load profile's time steps with the matching energy factors
    total = float(len(weather_data.index))
    for j, date_obj in enumerate(weather_data.index):
        # Compare time stamps in typtage_df of the matching house and typtag
        typtag = weather_data.loc[date_obj]['typtag']

        # The time index of all typtage_df levels should be the same, so we use
        # anyone of them to iterate over (here house_types[0] = 'EFH').
        for time_obj in typtage_df.loc[house_types[0], typtag].index:
            if (time_obj.time() == date_obj.time()):
                # Write each value from the current time stamp into its
                # corresponding field in the load_profile_df DataFrame:
                for energy in energy_factor_types:
                    for house_type in house_types:
                        load_profile_df.loc[date_obj][energy, house_type] = \
                          typtage_df.loc[house_type, typtag, time_obj][energy]

                break  # break for-loop (we don't need to keep searching)

        print('{:5.1f}% done'.format(j/total*100), end='\r')  # print progress

    # Debugging: The daily sum of each energy factor type must be '1':
#    if DEBUG:
#        print("The daily sum of each energy factor type must equal '1':")
#        print(load_profile_df.resample('D', label='left',
#                                       closed='right').sum())

    return load_profile_df


def get_annual_energy_demand(settings):
    '''Read in houses and calculate their annual energy demand
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

    return houses_dict


def get_daily_energy_demand_houses(houses_dict, settings):
    '''Determine the houses' energy demand values for each 'typtag'
    '''
    typtage_combinations = settings['typtage_combinations']
    houses_list = settings['houses_list_VDI']

    # Load the file containing the energy factors of the different typical
    # radiation year (TRY) regions, house types and 'typtage'. In VDI 4655,
    # these are the tables 10 to 24.
    energy_factors_df = pd.read_excel(open(energy_factors_file, 'rb'),
                                      sheet_name='Faktoren',
                                      index_col=[0, 1, 2])
#    print(energy_factors_df)

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
        TRY = houses_dict[house_name]['TRY']
        N_Pers = houses_dict[house_name]['N_Pers']
        N_WE = houses_dict[house_name]['N_WE']

        # Savety check:
        if TRY not in energy_factors_df.index.get_level_values(0):
            print('Error! TRY '+str(TRY)+' not contained in file ' +
                  energy_factors_file)
            print('       Skipping house "'+house_name+'"!')
            continue  # 'Continue' skips the rest of the current for-loop

        # (6.4) Do calculations according to VDI 4655 for each 'typtag'
        for typtag in typtage_combinations:
            F_Heiz_TT = energy_factors_df.loc[TRY, house_type,
                                              'F_Heiz_TT'][typtag]
            Q_Heiz_a = houses_dict[house_name]['Q_Heiz_a']
            Q_Heiz_a = Q_Heiz_a * config_dict.get('adjustment_factors',
                                                  dict()).get('f_Q_Heiz', 1)

            F_el_TT = energy_factors_df.loc[TRY, house_type, 'F_el_TT'][typtag]
            W_a = houses_dict[house_name]['W_a']
            W_a = W_a * config_dict.get('adjustment_factors',
                                        dict()).get('f_W', 1)

            F_TWW_TT = energy_factors_df.loc[TRY, house_type,
                                             'F_TWW_TT'][typtag]
            Q_TWW_a = houses_dict[house_name]['Q_TWW_a']
            Q_TWW_a = Q_TWW_a * config_dict.get('adjustment_factors',
                                                dict()).get('f_Q_TWW', 1)

            Q_Heiz_TT = Q_Heiz_a * F_Heiz_TT

            if house_type == 'EFH':
                W_TT = W_a * (1.0/365.0 + N_Pers * F_el_TT)
                Q_TWW_TT = Q_TWW_a * (1.0/365.0 + N_Pers * F_TWW_TT)
            elif house_type == 'MFH':
                W_TT = W_a * (1.0/365.0 + N_WE * F_el_TT)
                Q_TWW_TT = Q_TWW_a * (1.0/365.0 + N_WE * F_TWW_TT)

            # Write values into DataFrame
            daily_energy_demand_houses.loc[house_name,
                                           'Q_Heiz_TT'][typtag] = Q_Heiz_TT
            daily_energy_demand_houses.loc[house_name,
                                           'W_TT'][typtag] = W_TT
            daily_energy_demand_houses.loc[house_name,
                                           'Q_TWW_TT'][typtag] = Q_TWW_TT

    # print daily_energy_demand_houses
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
    # 'set_energy_demand_values' as the last argument.
    helper_function = functools.partial(get_energy_demand_values,
                                        weather_data, houses_list, houses_dict,
                                        energy_factor_types,
                                        energy_demands_types,
                                        load_curve_houses, load_profile_df,
                                        daily_energy_demand_houses)

    number_of_cores = multiprocessing.cpu_count() - 2
    print('Parallel processing on '+str(number_of_cores)+' cores')
    pool = multiprocessing.Pool(number_of_cores)
    work_list = weather_data.index
    return_list = pool.map_async(helper_function, work_list)
    pool.close()

    while return_list.ready() is False:
        remaining = return_list._number_left
        total = len(work_list)/float(return_list._chunksize)
        j = total - remaining
        print('{:5.1f}% done'.format(j/total*100), end='\r')  # print progress
        time.sleep(1.0)

    pool.join()
    print('100.0% done', end='\r')

    # The 'pool' returns a list. Feed its contents to the DataFrame
    for object in return_list.get():
        load_curve_houses.loc[object.name] = object

    # For each time step, each house and each type of energy factor, we
    # multiply the energy factor with the daily energy demand. The result
    # is the energy demand of that time interval.
    # We save it to the load_curve_houses DataFrame.
    # total = float(len(weather_data.index))
    # for j, date_obj in enumerate(weather_data.index):
        # helper_function(date_obj)

        # print ('{:5.1f}% done'.format(j/total*100), end='\r')  # progress

#    print(load_curve_houses)

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
        load_curve_houses[column] = load_curve_houses[column]/sum_ * Q_a

    return load_curve_houses


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

    for house_name in houses_list:
        house_type = houses_dict[house_name]['house_type']
        if house_type not in source_df.keys():
            # Only use 'H0G', 'G0G', 'G1G', ...
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

#    print(futureSolar_df.keys())

    for shift_steps, date_obj in enumerate(weather_data.index):
        if date_obj.dayofweek == 1:  # 1 equals Tuesday
#            print(shift_steps, date_obj)
            first_tuesday = date_obj
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


def copy_and_randomize_houses(load_curve_houses, houses_dict, settings):
    '''Create copies of houses where needed. Apply a normal distribution to
    the copies, if a standard deviation 'sigma' is given in the config.

    Remember: 68.3%, 95.5% and 99.7% of the values lie within one,
    two and three standard deviations of the mean.
    Example: With an interval of 15 min and a deviation of
    sigma = 2 time steps, 68% of profiles are shifted up to ±30 min (±1σ).
    27% of proflies are shifted ±30 to 60 min (±2σ) and another
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
    # Fix the 'randomness' (every run of the script generates the same results)
    np.random.seed(4)
    # Create copies for every house
    for house_name in settings['houses_list']:
        copies = houses_dict[house_name].get('copies', 0)
        # Get standard deviation (spread or “width”) of the distribution:
        sigma = houses_dict[house_name].get('sigma', False)
        randoms = np.random.normal(0, sigma, copies)  # Array of random values
        randoms_int = [int(value) for value in np.round(randoms, 0)]
        for copy in range(0, copies):
            copy_name = house_name + '_c' + str(copy)
            # Select the data of the house we want to copy
            df_new = load_curve_houses[house_name]
            # Rename the multiindex with the name of the copy
            df_new = pd.concat([df_new], keys=[copy_name], names=['house'],
                               axis=1)

            if sigma:  # Optional: Shift the rows
                shift_step = randoms_int[copy]
                if shift_step > 0:
                    # Shifting forward in time pushes the last entries out of
                    # the df and leaves the first entries empty. We take that
                    # overlap and insert it at the beginning
                    overlap = df_new[-shift_step:]
                    df_shifted = df_new.shift(shift_step)
                    df_shifted.dropna(inplace=True)
                    overlap.index = df_new[:shift_step].index
                    df_new = overlap.append(df_shifted)
                elif shift_step < 0:
                    # Retrieve overlap from the beginning of the df, shift
                    # backwards in time, paste overlap at end of the df
                    overlap = df_new[:abs(shift_step)]
                    df_shifted = df_new.shift(shift_step)
                    df_shifted.dropna(inplace=True)
                    overlap.index = df_new[shift_step:].index
                    df_new = df_shifted.append(overlap)
                elif shift_step == 0:
                    # No action required
                    pass

            # Merge the existing and new dataframes
            load_curve_houses = pd.concat([load_curve_houses, df_new],
                                          axis=1, sort=False)
        if sigma and DEBUG:
            print('Interval shifts applied to copies of house '+house_name+':')
            print(randoms_int)
            mu = np.mean(randoms_int)
            sigma = np.std(randoms_int, ddof=1)
            text_mean_std = 'Mean = {:0.2f}, std = {:0.2f}'.format(mu, sigma)
            title_mu_std = r'$\mu={:0.3f},\ \sigma={:0.3f}$'.format(mu, sigma)
            print(text_mean_std)

            if settings.get('show_plot', False) is True:
                # the histogram of the data
                bins = [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5]
                fig = plt.figure()
                ax = fig.gca()
                ax.yaxis.grid(True)  # Activate grid on horizontal axis
                n, bins, patches = plt.hist(randoms_int, bins, align='left',
                                            rwidth=0.9)
                plt.title(str(copies)+' Kopien von Gebäude '+house_name +
                          ' ('+title_mu_std+')')
                plt.xlabel('Zeitschritte')
                plt.ylabel('Anzahl')
                plt.rcParams.update({'font.size': 16})
                plt.savefig(os.path.join(base_folder, 'Result',
                                         'histogram_'+house_name+'.png'),
                            bbox_inches='tight', dpi=200)
                plt.show(block=False)  # Show plot without blocking the script
                pass

    load_curve_houses = load_curve_houses.swaplevel('house', 'class', axis=1)
    return load_curve_houses


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


def integrate_heizkurve(weather_data, settings):
    '''Implementation 'Heizkurve (Vorlauf- und Rücklauftemperatur)'
    (Not part of the VDI 4655)

    Calculation according to:
    Knabe, Gottfried (1992): Gebäudeautomation. 1. Aufl.
    Berlin: Verl. für Bauwesen.
    Section 6.2.1., pages 267-268
    '''
    interpolation_freq = settings['interpolation_freq']

    if config_dict.get('Heizkurve', None) is not None:
        print('Calculate heatcurve temperatures')

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
                Q_dot_loss = length * q_loss * dTm / 1000.0         # kW
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
    settings['energy_demands_types'].append('E_th_loss')


def normalize_energy(weather_data):
    '''Normalize results to a total of 1 kWh per year
    '''
    if config_dict.get('normalize', False) is True:
        print('Normalize load profile')
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

    # Define style settings for the plots
    mpl.style.use('./futureSuN.mplstyle')  # Personalized matplotlib style file

    # --- Script options ------------------------------------------------------
#    base_folder = r'V:\MA\2_Projekte\SIZ10015_futureSuN\4_Bearbeitung\AP4_Transformation\AP404_Konzepte für zukünftige Systemlösungen\Lastprofile\VDI 4655\Berechnung'
#    base_folder = r'V:\MA\2_Projekte\SIZ10015_futureSuN\4_Bearbeitung\AP4_Transformation\AP401_Zukünftige Funktionen\Quellen\RH+TWE'
#    base_folder = r'C:\Trnsys17\Work\futureSuN\SB\Load'
#    base_folder = r'C:\Trnsys17\Work\futureSuN\AP4\P2H_Quartier\Load'
    base_folder = r'C:\Trnsys17\Work\futureSuN\AP4\Referenz_Quartier_Neubau\Last'
#    base_folder = r'C:\Users\nettelstroth\Documents\02 Projekte - Auslagerung\SIZ10019_Quarree100_Heide\Load'

    holiday_file = os.path.join(base_folder, 'Typtage', 'Feiertage.xlsx')
    energy_factors_file = os.path.join(
        base_folder, 'Typtage', 'VDI 4655 Typtag-Faktoren.xlsx')
    typtage_file = os.path.join(base_folder, 'Typtage', 'VDI 4655 Typtage.xlsx')
    BDEW_file = os.path.join(base_folder, 'Typtage', 'BDEW Profile.xlsx')
    DOE_file = os.path.join(base_folder, 'Typtage', 'DOE Profile TWE.xlsx')
    futureSolar_file = os.path.join(base_folder, 'Typtage', 'futureSolar Profile.xlsx')
    config_file = os.path.join(base_folder, 'VDI_4655_config.yaml')

    # --- Import the config_dict from the YAML config_file --------------------
    config_dict = yaml.load(open(config_file, 'r'))

    # --- Read settings from the config_dict ----------------------------------
    settings = config_dict['settings']
    bool_print_header = settings.get('print_header', True)
    bool_print_index = settings.get('print_index', True)
    use_BDEW_seasons = settings.get('use_BDEW_seasons', False)

    print_file = settings['print_file']
    bool_show_plot = settings.get('show_plot', False)

    DEBUG = settings.get('Debug', False)

    # \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
    #                          VDI 4655 Implementation
    # \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
    weather_data = load_weather_file(settings)

    # -------------------------------------------------------------------------
    # VDI 4655 - Step 1: Determine the "typtag" key for each timestep
    # -------------------------------------------------------------------------
    print('Determine "typtag" keys for each time step')
    get_typical_days(weather_data, settings)

    # -------------------------------------------------------------------------
    # VDI 4655 - Step 2:
    # Match 'typtag' keys and reference load profile factors for each timestep
    # (for each 'typtag' key, one load profile is defined by VDI 4655)
    # -------------------------------------------------------------------------
    print('Read in reference load profile factors and match them to ' +
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
    print('Read in houses and calculate their annual energy demand')
    houses_dict = get_annual_energy_demand(settings)

    # (6.3) Allocation of building site:
    # -------------------------------------------------------------------------
    # The user has to give the number of the TRY climat zone
    # in the yaml file. It is used in (6.4).

    # (6.4) Determination of the houses' energy demand values for each 'typtag'
    # -------------------------------------------------------------------------
    print("Determine the houses' energy demand values for each 'typtag'")
    daily_energy_demand_houses = get_daily_energy_demand_houses(houses_dict,
                                                                settings)

    # (6.5) Determination of a daily demand curve for each house:
    # -------------------------------------------------------------------------
    print("Generate the houses' energy demand values for each timestep")
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

    # Randomize the
    # (Optional, not part of VDI 4655)
    # -------------------------------------------------------------------------
    print('Create (randomized) copies of the houses')
    load_curve_houses = copy_and_randomize_houses(load_curve_houses,
                                                  houses_dict, settings)

    # Debugging: Show the daily sum of each energy demand type:
#    print(load_curve_houses.resample('D', label='left', closed='right').sum())

    # Sum up the energy demands of all houses, store result in weather_data
    print('Sum up the energy demands of all houses')
    weather_data = sum_up_all_houses(load_curve_houses, weather_data)
#    print(weather_data)

    # \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
    #       Implementation 'Heizkurve (Vorlauf- und Rücklauftemperatur)'
    #                        (Not part of the VDI 4655)
    # \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
    integrate_heizkurve(weather_data, settings)

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
    print('Calculations completed')
    print('Monthly energy sums in kWh:')
    weather_montly_sum = weather_daily_sum.resample('M', label='right',
                                                    closed='right').sum()
    print(weather_montly_sum)
    print()
    print('Annual energy sums in kWh:')
    weather_annual_sum = weather_montly_sum.resample('A', label='right',
                                                     closed='right').sum()
    print(weather_annual_sum)
    # print weather_annual_sum.sum(axis=1,'Q_Heiz_TT', 'Q_TWW_TT', 'Q_loss')
    print('Total heat energy demand is {:.2f} kWh.'.format(
        weather_annual_sum[sum_list_heat].sum(axis=1).sum()))
    print()

    pd.reset_option('precision')  # ...and reset the setting from above

    # Display a plot on screen for the user
    if bool_show_plot is True:
        print('Showing plot of energy demand types...')
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
        weather_data = weather_data[settings['print_columns']]

    # Output folder is hardcoded here:
    print_folder = os.path.join(base_folder, 'Result')

    # Call external method in zeitreihe_IGS_RKR.py:
    zeitreihe_IGS_RKR.print_IGS_weather_file(weather_data,
                                             print_folder,
                                             print_file,
                                             bool_print_index,
                                             bool_print_header)

    # Print a final message with the required time
    script_time = pd.to_timedelta(time.time() - start_time, unit='s')
    print('Finished script in time: %s' % (script_time))

    plt.show()  # Script will be blocked until the user closes the plot window
