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


Module BDEW
-----------
This module combines profiles for commercial building types from different
sources.
"""

import pandas as pd
from pandas.tseries.frequencies import to_offset
import logging

# Import local modules from load profile aggregator project
import lpagg.misc

# Define the logging function
logger = logging.getLogger(__name__)


def get_GHD_profiles(weather_data, cfg, houses_dict):
    """Combine profiles from various sources for the GHD building sector.

    (not part of VDI 4655)
    """
    BDEW_profiles = load_BDEW_profiles(weather_data, cfg, houses_dict)
    DOE_profiles = load_DOE_profiles(weather_data, cfg, houses_dict)
    futureSolar_profiles = load_futureSolar_profiles(weather_data, cfg,
                                                     houses_dict)
    GHD_profiles = pd.concat([BDEW_profiles,
                              DOE_profiles,
                              futureSolar_profiles],
                             axis=1, sort=False)

    return GHD_profiles


def load_BDEW_style_profiles(source_file, weather_data, cfg, houses_dict,
                             energy_type):
    """Load energy profiles from files that are structured like BDEW profiles.

    Is used for BDEW profiles, and allows profiles from other sources to
    be integrated easily. For example, the U.S. Department of Energy (DOE)
    profiles for building types can manually be converted to the BDEW format,
    then loaded with this function.

    .. note::
        The BDEW profiles use the unit 'W', while the VDI 4655 profiles
        come in 'kWh'. The BDEW profiles are converted, even though that does
        not have any effect (since they are normalized anyway).

    """
    settings = cfg['settings']
    source_df = pd.read_excel(source_file, sheet_name=None,
                              skiprows=[0], header=[0, 1], index_col=[0],
                              skipfooter=1,
                              )

    weather_daily = weather_data.resample('D', label='right',
                                          closed='right').mean()
    # print(weather_daily)

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

            # Append to yearly profile
            profile_year = pd.concat([profile_year, profile_daily])

        # Convert unit from 'W' to 'kWh'
        freq = pd.infer_freq(profile_year.index, warn=True)
        freq = pd.to_timedelta(to_offset(freq))
        profile_year *= freq / (pd.Timedelta('1 hours') * 1000)  # W to kWh

        # Resample to the desired frequency (time intervall)
        profile_year = lpagg.misc.resample_energy(
                profile_year, settings['interpolation_freq'])

        # Store in DataFrame that will be returned
        ret_profiles[house_name, energy_type] = profile_year

    return ret_profiles


def load_BDEW_profiles(weather_data, cfg, houses_dict):
    """Load profiles for GHD from BDEW."""
    source_file = cfg['data']['BDEW']
    energy_type = 'W_TT'
    BDEW_profiles = load_BDEW_style_profiles(source_file, weather_data,
                                             cfg, houses_dict,
                                             energy_type)

    # Rescale to the given yearly energy demand:
    for column in BDEW_profiles.columns:
        W_a = houses_dict[column[0]]['W_a']
        yearly_sum = BDEW_profiles[column].sum()
        if W_a >= 0:  # Only rescale if W_a has a meaningful value
            BDEW_profiles[column] = BDEW_profiles[column]/yearly_sum * W_a

    return BDEW_profiles


def load_DOE_profiles(weather_data, cfg, houses_dict):
    """Load profiles for GHD from DOE (departement of energy)."""
    source_file = cfg['data']['DOE']
    energy_type = 'Q_TWW_TT'
    DOE_profiles = load_BDEW_style_profiles(source_file, weather_data,
                                            cfg, houses_dict,
                                            energy_type)

    # Rescale to the given yearly energy demand:
    for column in DOE_profiles.columns:
        Q_TWW_a = houses_dict[column[0]]['Q_TWW_a']
        yearly_sum = DOE_profiles[column].sum()
        if yearly_sum > 0:
            DOE_profiles[column] = DOE_profiles[column]/yearly_sum * Q_TWW_a

    return DOE_profiles


def load_futureSolar_profiles(weather_data, cfg, houses_dict):
    """Load profiles for commerical buildings from project 'futureSolar'.

    These heating and cooling profiles were simulated by IGS TU Braunschweig.

    In order to NOT use these profiles, set ``Q_Heiz_a`` and/or ``Q_Kalt_a``
    to ``None`` for all houses in your list of houses.

    These profiles start on a tuesday. Therefore we first have to align them
    with the calendar of the aggregated profiles.
    """
    settings = cfg['settings']
    houses_list = settings['houses_list_BDEW']

    futureSolar_df = pd.read_excel(cfg['data']['futureSolar'],
                                   index_col=[0],
                                   sheet_name='Profile', header=[0, 1])
    futureSolar_df.index = pd.to_timedelta(futureSolar_df.index, unit='h')

    energy_types = ['Q_Heiz_TT', 'Q_Kalt_TT']
    multiindex = pd.MultiIndex.from_product([houses_list, energy_types],
                                            names=['house', 'energy'])
    futureSolar_profiles = pd.DataFrame(index=weather_data.index,
                                        columns=multiindex)
    if len(houses_list) == 0:  # Skip
        return futureSolar_profiles

    for shift_steps, date_obj in enumerate(weather_data.index):
        if date_obj.dayofweek == 1:  # 1 equals Tuesday
            first_tuesday = date_obj
#            print(shift_steps, date_obj)
            break

    futureSolar_df.index = first_tuesday + futureSolar_df.index

    # Resample to target frequency
    futureSolar_df = lpagg.misc.resample_energy(futureSolar_df,
                                                settings['interpolation_freq'])

    # Finish aligning the calendar days
    overlap = futureSolar_df[-(shift_steps + 1):]
    overlap.index = overlap.index - pd.Timedelta('365 days')
    futureSolar_df = overlap.append(futureSolar_df)

    for house_name in houses_list:
        house_type = houses_dict[house_name]['house_type']
        if house_type not in futureSolar_df.keys():
            # Only use 'G1G' and 'G4G'
            logger.warning('house_type "'+str(house_type)+'" not found in '
                           'futureSolar profiles: '+str(futureSolar_df.keys()))
            continue

        for energy_type in energy_types:
            profile_year = futureSolar_df[house_type, energy_type]
#            print(profile_year)
            futureSolar_profiles[house_name, energy_type] = profile_year

    # Rescale to the given yearly energy demand:
    for column in futureSolar_profiles.columns:
        if column[1] == 'Q_Heiz_TT':
            Q_a = houses_dict[column[0]].get('Q_Heiz_a', None)
        elif column[1] == 'Q_Kalt_TT':
            Q_a = houses_dict[column[0]].get('Q_Kalt_a', None)
        sum_ = futureSolar_profiles[column].sum()

        if Q_a is None:  # Do not use the loaded profiles (and remove them)
            futureSolar_profiles.drop(columns=column, inplace=True)
        elif sum_ > 0:  # Use the loaded profiles
            futureSolar_profiles[column] *= Q_a / sum_

#    print(futureSolar_profiles)
    return futureSolar_profiles
