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


Module VDI 4655
---------------
This is an implementation of the calculation of economic efficiency
using the annuity method defined in the German VDI 4655.

    **VDI 4655**

    **Reference load profiles of single-family and
    multi-family houses for the use of CHP systems**

    *May 2008 (ICS 91.140.01)*

Copyright:

    *Verein Deutscher Ingenieure e.V.*

    *VDI Standards Department*

    *VDI-Platz 1, 40468 Duesseldorf, Germany*

Reproduced with the permission of the Verein Deutscher Ingenieure e.V.,
for non-commercial use only.

Notes
-----
This script creates full year energy demand time series of domestic buildings
for use in simulations. This is achieved by an implementation of the VDI 4655,
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
"""

import pandas as pd
import functools
import logging
import pickle
import datetime as dt
import holidays
import pkg_resources

# Import local modules from load profile aggregator project
import lpagg.misc


# Define the logging function
logger = logging.getLogger(__name__)


def run_demandlib(weather_data, cfg):
    """Get the VDI4655 profiles from the libary demandlib.

    This replaces the function ``run()``. The implementation of VDI4655
    was transfered to the library demandlib, to make it available to a
    broader audience.

    https://github.com/oemof/demandlib
    """
    from demandlib import vdi

    settings = cfg['settings']

    holidays_list = []
    if settings.get('holidays'):
        country = settings['holidays'].get('country', 'DE')
        province = settings['holidays'].get('province', None)
        holidays_list = holidays.country_holidays(country, subdiv=province)

    houses_dict = get_annual_energy_demand(cfg)
    my_houses = []
    for key, value in houses_dict.items():
        value['name'] = key
        my_houses.append(value)

    df_empty = pd.DataFrame(
        index=["house_type", "N_We", "N_Pers", "Q_Heiz_a", "Q_TWW_a", "W_a"])
    df_empty.columns.set_names('name', inplace=True)

    try:
        year = settings['start'].year  # A datetime object
    except (KeyError, AttributeError):
        year = settings['start'][0]  # A list, starting with the year

    # Define the region
    my_region = vdi.Region(
        year,
        holidays=holidays_list,
        try_region=my_houses[0]['TRY'],
        houses=my_houses,
        resample_rule=pd.Timedelta(settings.get('intervall', '1 hours')),
        file_weather=settings['weather_file'],
    )

    # Calculate load profiles
    logger.info('Calculate load profiles with demandlib')
    lc = my_region.get_load_curve_houses()

    # Demandlib uses a different time step notation then lpagg
    lc = lc.shift(periods=1, freq="infer").droplevel('house_type', axis='columns')

    lc.columns.set_names(['house', 'energy'], inplace=True)
    lc.index.set_names(['Time'], inplace=True)

    # This has nothing to do with demandlib, but for BDEW we currently need
    # more information added to weather_data
    # TODO: Move this to lpagg.BDEW
    get_typical_days(weather_data, cfg)

    return lc, houses_dict


def run(weather_data, cfg):
    """Run the VDI 4655 Implementation."""
    # -------------------------------------------------------------------------
    # VDI 4655 - Step 1: Determine the "typtag" key for each timestep
    # -------------------------------------------------------------------------
    logger.info('Determine "typtag" keys for each time step')
    get_typical_days(weather_data, cfg)

    # -------------------------------------------------------------------------
    # VDI 4655 - Step 2:
    # Match 'typtag' keys and reference load profile factors for each timestep
    # (for each 'typtag' key, one load profile is defined by VDI 4655)
    # -------------------------------------------------------------------------
    logger.info('Read in reference load profile factors and match them to ' +
                '"typtag" keys for each timestep')
    load_profile_df = load_profile_factors(weather_data, cfg)

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
    houses_dict = get_annual_energy_demand(cfg)

    # (6.3) Allocation of building site:
    # -------------------------------------------------------------------------
    # The user has to give the number of the TRY climate zone
    # in the yaml file. It is used in (6.4).

    # (6.4) Determination of the houses' energy demand values for each 'typtag'
    # -------------------------------------------------------------------------
    logger.info("Determine the houses' energy demand values for each typtag")
    daily_energy_demand_houses = get_daily_energy_demand_houses(houses_dict,
                                                                cfg)

    # (6.5) Determination of a daily demand curve for each house:
    # -------------------------------------------------------------------------
    logger.info("Generate the houses' energy demand values for each timestep")
    load_curve_houses = get_load_curve_houses(load_profile_df, houses_dict,
                                              weather_data, cfg,
                                              daily_energy_demand_houses)

    return load_curve_houses, houses_dict


def get_season_list_BDEW(weather_data):
    """Get a list of seasons as defined by BDEW for the BDEW profiles.

    Winter:       01.11. to 20.03.
    Summer:       15.05. to 14.09.
    Transition:   21.03. to 14.05. and 15.09. to 31.10.

    .. note::
        It might seem like the time comparisons are implemented incorrectly.
        ``date_obj <= 21.3. 00:00`` includes the 21.3. at 00:00 am, while
        winter is only supposed to include the 20.3. Should it not be
        ``date_obj < 21.3. 00:00`` instead?
        No, the implementation is correct, because of how the weather data
        is treated. In accordance to the DWD default, each timestamp
        represents the time step before.
        Thus the timestamp ``21.3. 00:00`` marks 20.3. 23:45 to 24:00 in
        a 15min interval which is still included in the winter season.

    """
    season_list = []

    for j, date_obj in enumerate(weather_data.index):
        YEAR = date_obj.year

        winter_end = dt.datetime(YEAR, 3, 21, 00, 00, 00)
        winter_start = dt.datetime(YEAR, 10, 31, 00, 00, 00)
        summer_start = dt.datetime(YEAR, 5, 15, 00, 00, 00)
        summer_end = dt.datetime(YEAR, 9, 15, 00, 00, 00)

        if date_obj <= winter_end or date_obj > winter_start:
            season_list.append('Winter')  # Winter

        elif date_obj > summer_start and date_obj <= summer_end:
            season_list.append('Sommer')  # Summer

        else:
            season_list.append('Übergangszeit')  # Transition

    return season_list


def get_typical_days(weather_data, cfg):
    """Run VDI 4655 - Step 1.

    Determine the "typtag" key for each timestep ().

    For the full "typtag", we need to identify:

        - Season
        - Workdays, Sundays and public holidays
        - Cloud cover amount

    """
    settings = cfg['settings']
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
    if settings.get('use_BDEW_seasons', False) is False:
        weather_data['season'] = season_list
    elif settings.get('use_BDEW_seasons', False) is True:
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

    # Use https://pypi.org/project/holidays/ for holiday-detection
    used_holidays = []
    if settings.get('holidays'):
        country = settings['holidays'].get('country', 'DE')
        province = settings['holidays'].get('province', None)
        used_holidays = holidays.country_holidays(country, subdiv=province)

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
        elif date_obj in used_holidays:
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
    # amount. So we need to replace 'heiter' and 'bewölkt' with 'X'
    typtage_replace = {'typtag':
                       {'SWH': 'SWX', 'SWB': 'SWX', 'SSH': 'SSX', 'SSB': 'SSX'}
                       }
    weather_data.replace(to_replace=typtage_replace, inplace=True)


def load_profile_factors(weather_data, cfg):
    """Run VDI 4655 - Step 2.

    Match 'typtag' keys and reference load profile factors for each timestep
    (for each 'typtag' key, one load profile is defined by VDI 4655)
    """
    settings = cfg['settings']
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
        print(pd.DataFrame(N_typtage).T.to_string(
            index=False,
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
            loaded_freq = pd.infer_freq(load_profile_df.index)
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
    # For the 'noarch' conda build, access the file as pkg resource object
    with pkg_resources.resource_stream('lpagg', cfg['data']['typtage']
                                       ) as resource:
        typtage_sheets_dict = pd.read_excel(resource,
                                            sheet_name=None,
                                            index_col=[0, 1])
    # The DataFrame within every dict entry is combined to one large DataFrame
    typtage_df = pd.DataFrame()  # create a new DataFrame that is empty
    for sheet in typtage_sheets_dict:
        typtage_df_new = typtage_sheets_dict[sheet]
        typtage_df = pd.concat([typtage_df, typtage_df_new])

    # The column 'Zeit' is of the type datetime.time. It must be converted
    # to dt.datetime by adding an arbitrary datetime.date object
    datetime_column = []
    for row, time_obj in enumerate(typtage_df['Zeit']):
        day = dt.datetime(2017, 1, 1)
        if type(time_obj) == type(day):
            time_obj = time_obj.time()
        datetime_obj = dt.datetime.combine(day, time_obj)
        datetime_column.append(datetime_obj)
    typtage_df['Zeit'] = datetime_column
    # Now the column 'Zeit' can be added to the multiindex
    typtage_df.set_index('Zeit', drop=True, append=True, inplace=True)
    typtage_df.columns.name = 'energy'
    typtage_df.index.set_names({'Haus': 'house'}, inplace=True)

    # EFH come in resolution of 1 min, MFH in 15 min. We need to get MFH down
    # to 1 min.
    # Unstack, so that only the time index remains. This creates NaNs for the
    # missing time stamps in MFH columns
    typtage_df = typtage_df.unstack(['house', 'typtag'])
    # Fill those NaN values with 'forward fill' method
    typtage_df.fillna(method='ffill', inplace=True)
    # Divide by factor 15 to keep the total energy demand constant
    typtage_df.loc[:, (slice(None), 'MFH',)] *= 1/15

    # In the next step, the values are summed up into the same time intervals
    # as the weather data, and the label is moved to the 'right'
    # (each time stamp now describes the data in the interval before)
    typtage_df = typtage_df.resample(rule=interpolation_freq, level='Zeit',
                                     label='right', closed='left').sum()
    typtage_df = typtage_df.stack(['house', 'typtag'])
    typtage_df = typtage_df.reorder_levels(['house', 'typtag', 'Zeit'])
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
    load_profile_df.fillna(0, inplace=True)

    # Fill the load profile's time steps with the matching energy factors
    # Iterate over time slices of full days
    start = weather_data.index[0]
    while start < weather_data.index[-1]:
        end = start + pd.Timedelta('1 days') - interpolation_freq
        if logger.isEnabledFor(logging.INFO):
            print('\rProgress: '+str(start), end='\r')  # print progress
        # Compare time stamps in typtage_df of the matching house and typtag
        typtag = weather_data.loc[start]['typtag']

        typtage_df.loc[house_types[0], typtag].index
        start_tt = dt.datetime.combine(dt.datetime(2017, 1, 1), start.time())
        end_tt = start_tt + pd.Timedelta('1 days') - interpolation_freq

        for energy in energy_factor_types:
            for house_type in house_types:
                load_profile_df.loc[start:end, (energy, house_type)] = \
                    typtage_df.loc[house_type, typtag,
                                   start_tt:end_tt][energy].values

        # pd.merge(left=load_profile_df.loc[start:end],
        #          right=typtage_df.unstack('house').xs(typtag, level='typtag'),
        #          how='left',
        #          left_index=True)

        # load_profile_df.columns
        # typtage_df.unstack('house').columns
        # test = load_profile_df.copy()
        # test.loc[start:end] += typtage_df.unstack('house').xs(typtag, level='typtag')
        # test2 = load_profile_df - test

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


def get_annual_energy_demand(cfg):
    """Read in houses and calculate their annual energy demand.

    VDI 4655 provides estimates for annual electrical and DHW energy demand
    (``W_a`` and ``Q_TWW_a``). ``Q_Heiz_TT`` cannot be estimated, but must
    be defined in the config file.
    If ``W_a`` or ``Q_TWW_a` are defined in the config file, their estimation
    is not used.
    """
    houses_dict = cfg['houses']
    houses_list = sorted(houses_dict.keys())

    # Calculate annual energy demand of houses
    # and store the result in the dict containing the house info
    for house_name in houses_list:
        house_type = houses_dict[house_name]['house_type']
        N_Pers = houses_dict[house_name].get('N_Pers', None)
        N_WE = houses_dict[house_name].get('N_WE', None)

        # Assign defaults if values are not defined
        if pd.isna(N_Pers):
            N_Pers = 3
            houses_dict[house_name]['N_Pers'] = N_Pers
            logger.warning('N_Pers not defined for ' + str(house_name)
                           + '. Using default ' + str(N_Pers))
        if pd.isna(N_WE):
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

        elif house_type == 'MFH':
            # (6.2.2) Calculate annual electrical energy demand of houses:
            W_a = N_WE * 3000  # kWh

            # (6.2.3) Calculate annual DHW energy demand of houses:
            Q_TWW_a = N_WE * 1000  # kWh

        else:
            # No house category given. Just use annual demand of 1 kWh
            W_a = 1
            Q_TWW_a = 1

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
            cfg.get('adjustment_factors', dict()).get('f_Q_Heiz', 1)

        houses_dict[house_name]['W_a'] *= \
            cfg.get('adjustment_factors', dict()).get('f_W', 1)

        houses_dict[house_name]['Q_TWW_a'] *= \
            cfg.get('adjustment_factors', dict()).get('f_Q_TWW', 1)

    return houses_dict


def get_daily_energy_demand_houses(houses_dict, cfg):
    """Determine the houses' energy demand values for each 'typtag'.

    .. note::
        "The factors ``F_el_TT`` and ``F_TWW_TT`` are negative in some cases as
        they represent a variation from a one-year average. The values for the
        daily demand for electrical energy, ``W_TT``, and DHW energy,
        ``Q_TWW_TT``, usually remain positive. It is only in individual
        cases that the calculation for the typical-day category ``SWX``
        can yield a negative value of the DHW demand. In that case,
        assume ``F_TWW_SWX`` = 0." (VDI 4655, page 16)

        This occurs when ``N_Pers`` or ``N_WE`` are too large.

    """
    settings = cfg['settings']
    typtage_combinations = settings['typtage_combinations']
    houses_list = settings['houses_list_VDI']

    # Load the file containing the energy factors of the different typical
    # radiation year (TRY) regions, house types and 'typtage'. In VDI 4655,
    # these are the tables 10 to 24.
    # For the 'noarch' conda build, access the file as pkg resource object
    with pkg_resources.resource_stream('lpagg', cfg['data']['energy_factors']
                                       ) as resource:
        energy_factors_df = pd.read_excel(resource,
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
                         cfg['data']['energy_factors'])
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


def get_load_curve_houses(load_profile_df, houses_dict, weather_data, cfg,
                          daily_energy_demand_houses):
    """Generate the houses' energy demand values for each timestep."""
    settings = cfg['settings']
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

    if settings.get('run_in_parallel', False):
        # 'Partial' creates a function that only takes one argument. In our
        # case this is 'date_obj'. It will be given to the target function
        # 'get_energy_demand_values' as the last argument.
        helper_func = functools.partial(get_energy_demand_values,
                                        weather_data, houses_list,
                                        houses_dict, energy_factor_types,
                                        energy_demands_types,
                                        load_curve_houses, load_profile_df,
                                        daily_energy_demand_houses)

        work_list = weather_data.index
        return_list = lpagg.misc.multiprocessing_job(helper_func, work_list)

        # The 'pool' returns a list. Feed its contents to the DataFrame
        for returned_df in return_list.get():
            load_curve_houses.loc[returned_df.name] = returned_df

        load_curve_houses = load_curve_houses.astype('float')

    else:  # Run in serial (default behaviour)
        load_curve_houses = get_energy_demand_values_day(
                weather_data, houses_list, houses_dict,
                energy_factor_types, energy_demands_types,
                load_curve_houses, load_profile_df,
                daily_energy_demand_houses)

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


def get_energy_demand_values(weather_data, houses_list, houses_dict,
                             energy_factor_types, energy_demands_types,
                             load_curve_houses, load_profile_df,
                             daily_energy_demand_houses, date_obj):
    """Get the energy demand values for all time steps.

    This functions works through the lists houses_list and energy_factor_types
    for a given time step (=date_obj) and multiplies the current load profile
    value with the daily energy demand. It returns the result: the energy
    demand values for all houses and energy types (in kWh)
    """
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


def get_energy_demand_values_day(weather_data, houses_list, houses_dict,
                                 energy_factor_types, energy_demands_types,
                                 load_curve_houses, load_profile_df,
                                 daily_energy_demand_houses):
    """Get the energy demand values for all days.

    This functions works through the lists houses_list and energy_factor_types
    day by day and multiplies the current load profile
    value with the daily energy demand. It returns the result: the energy
    demand values for all houses and energy types (in kWh)
    """
    start = weather_data.index[0]
    while start < weather_data.index[-1]:
        end = start + pd.Timedelta('1 days')
        if logger.isEnabledFor(logging.INFO):
            print('\rProgress: '+str(start), end='\r')  # print progress
        typtag = weather_data.loc[start]['typtag']
        for house_name in houses_list:
            house_type = houses_dict[house_name]['house_type']
            for i, energy_factor_type in enumerate(energy_factor_types):
                energy_demand_type = energy_demands_types[i]
                # Example: Q_Heiz_TT(t) = F_Heiz_TT(t) * Q_Heiz_TT
                load_curve_houses.loc[start:end, (house_name,
                                                  energy_demand_type)] =\
                    load_profile_df.loc[start:end, (energy_factor_type,
                                                    house_type)] *\
                    daily_energy_demand_houses.loc[(house_name,
                                                   energy_demand_type), typtag]
#                print(load_curve_houses.loc[start:end])
        start = end

    if logger.isEnabledFor(logging.INFO):
        # overwrite last status with empty line
        print('\r', end='\r')

    return load_curve_houses


def main():
    """Run the main method (unused and redundant)."""
    pass


if __name__ == '__main__':
    '''This part is executed when the script is started directly with
    Python, not when it is loaded as a module.
    '''
    main()
