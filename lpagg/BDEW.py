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

import re
import numpy as np
import pandas as pd
from pandas.tseries.frequencies import to_offset
import logging
import importlib.resources

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
    energy_types_annual = {'Q_TWW_TT': 'Q_TWW_a', 'W_TT': 'W_a'}
    energy_type_annual_str = energy_types_annual[energy_type]

    settings = cfg['settings']
    # For the 'noarch' conda build, access the file as resource object
    res_path = importlib.resources.files('lpagg').joinpath(source_file)
    with importlib.resources.as_file(res_path) as resource:
        source_df = pd.read_excel(resource, sheet_name=None,
                                  skiprows=[0], header=[0, 1], index_col=[0],
                                  skipfooter=1,
                                  )
    weather_daily = (weather_data.resample('D', label='right', closed='right')
                     .mean(numeric_only=True))
    # print(weather_daily)

    houses_list = settings['houses_list_BDEW']
    multiindex = pd.MultiIndex.from_product([houses_list, [energy_type]],
                                            names=['house', 'energy'])

    ret_profiles = pd.DataFrame(index=weather_data.index,
                                columns=multiindex,
                                dtype='float')
    if len(houses_list) == 0:  # Skip
        return ret_profiles

    for house_name in houses_list:
        if pd.isna(houses_dict[house_name].get(energy_type_annual_str,
                                               np.nan)):
            continue
        elif houses_dict[house_name][energy_type_annual_str] == 0:
            continue

        house_type = houses_dict[house_name]['house_type']
        pattern = r"(?i)[GLH]\d+"  # Test for G0, G1, ... G7, L0, L1, L2, H0
        matches = re.findall(pattern, house_type)
        if len(matches) == 1:
            house_type = matches[0] + 'G'  # Construct 'H0G', 'G0G', 'G1G', ...
        if house_type not in source_df.keys():
            # Only use 'H0G', 'G0G', 'G1G', ...
            logger.warning('house_type "%s" not found in profile sources: '
                           '%s', house_type, source_df.keys())
            continue

        # Create the yearly profile for the current house
        profile_year = pd.Series(dtype='float')
        profiles_daily = []
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

            # Append to list of dayly profiles
            profiles_daily.append(profile_daily)

        # Combine to yearly profile
        profile_year = pd.concat(profiles_daily)

        # Convert unit from 'W' to 'kWh'
        freq = pd.infer_freq(profile_year.index)
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
        W_a = houses_dict[column[0]].get('W_a', np.nan)
        yearly_sum = BDEW_profiles[column].sum()
        if W_a >= 0:  # Only rescale if W_a has a meaningful value
            BDEW_profiles[column] = BDEW_profiles[column]/yearly_sum * W_a
        elif pd.isna(W_a):
            BDEW_profiles[column] = np.nan

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
        Q_TWW_a = houses_dict[column[0]].get('Q_TWW_a', np.nan)
        yearly_sum = DOE_profiles[column].sum()
        if yearly_sum > 0:
            DOE_profiles[column] = DOE_profiles[column]/yearly_sum * Q_TWW_a
        elif pd.isna(Q_TWW_a):
            DOE_profiles[column] = np.nan

    return DOE_profiles


def load_futureSolar_profiles(weather_data, cfg, houses_dict,
                              energy_types=['Q_Heiz_TT', 'Q_Kalt_TT']):
    """Load profiles for commerical buildings from project 'futureSolar'.

    These heating and cooling profiles were simulated by IGS TU Braunschweig.

    In order to NOT use these profiles, set ``Q_Heiz_a`` and/or ``Q_Kalt_a``
    to ``None`` for all houses in your list of houses.

    These profiles start on a tuesday. Therefore we first have to align them
    with the calendar of the aggregated profiles.
    """
    settings = cfg['settings']
    houses_list = settings['houses_list_BDEW']
    # For the 'noarch' conda build, access the file as resource object
    res_path = importlib.resources.files('lpagg').joinpath(
        cfg['data']['futureSolar'])
    with importlib.resources.as_file(res_path) as resource:
        futureSolar_df = pd.read_excel(resource,
                                       index_col=[0],
                                       sheet_name='Profile', header=[0, 1])
    futureSolar_df.index = pd.to_timedelta(futureSolar_df.index, unit='h')

    multiindex = pd.MultiIndex.from_product([houses_list, energy_types],
                                            names=['house', 'energy'])
    futureSolar_profiles = pd.DataFrame(index=weather_data.index,
                                        columns=multiindex,
                                        dtype='float')
    if len(houses_list) == 0:  # Skip
        return futureSolar_profiles

    for shift_steps, date_obj in enumerate(weather_data.index):
        if date_obj.dayofweek == 1:  # 1 equals Tuesday
            first_tuesday = date_obj
            # print(shift_steps, date_obj)
            break

    futureSolar_df.index = first_tuesday + futureSolar_df.index

    # Resample to target frequency
    futureSolar_df = lpagg.misc.resample_energy(futureSolar_df,
                                                settings['interpolation_freq'])

    # Finish aligning the calendar days
    overlap = futureSolar_df[-(shift_steps + 1):]
    overlap.index = overlap.index - pd.Timedelta('365 days')
    futureSolar_df = pd.concat([overlap, futureSolar_df])

    for house_name in houses_list:
        house_type = houses_dict[house_name]['house_type']
        pattern = r"(?i)[GLH]\d+"  # Test for G0, G1, ... G7, L0, L1, L2, H0
        matches = re.findall(pattern, house_type)
        if len(matches) == 1:
            house_type = matches[0] + 'G'
        if house_type not in futureSolar_df.keys():
            # Only use 'G1G' and 'G4G'
            logger.warning('house_type "'+str(house_type)+'" not found in '
                           'futureSolar profiles: '+str(futureSolar_df.keys()))
            continue

        for energy_type in energy_types:
            profile_year = futureSolar_df[house_type, energy_type]
            # print(profile_year)
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


def run_demandlib(weather_data, cfg, houses_dict):
    """Generate commercial profiles from BDEW with demandlib.

    Demandlib is used for heating, hot water and electricity profiles.
    Generalized cooling profiles are still generated with lpagg.
    """
    # Use demandlib for heating, hot water and electricity profiles
    demandlib_profiles = get_demandlib_profiles(weather_data, cfg, houses_dict)

    # But lpagg also provides the option to generate a (generalized) cooling
    # profile. This is calculated internally by lpagg and combined with the
    # demandlib results
    futureSolar_profiles = load_futureSolar_profiles(
        weather_data, cfg, houses_dict,
        energy_types=['Q_Kalt_TT'])

    GHD_profiles = pd.concat([demandlib_profiles,
                              futureSolar_profiles],
                             axis=1, sort=False)
    GHD_profiles = GHD_profiles.sort_index(axis='columns')

    return GHD_profiles


def get_demandlib_profiles(weather_data, cfg, houses_dict):
    """Generate commercial profiles from BDEW with demandlib.

    House types are expected to come in the form of e.g."GHD/G1"
    One part denotes the type of thermal profile, the other the eletrical

    Thermal profiles:
        "GMF", "GPD", "GHD", "GWA", "GGB", "EFH", "GKO",
        "MFH", "GBD", "GBA", "GMK", "GBH", "GGA", "GHA"
    Electrical profile:
        g0, g1, ... l0, l1, h0

    For descriptions about the profiles refer to the demandlib docs:
    https://demandlib.readthedocs.io/en/latest/bdew.html

    The implementation in demandlib is based on the following sources:

    https://mediatum.ub.tum.de/doc/601557/601557.pdf

    https://www.avacon-netz.de/content/dam/revu-global/avacon-netz/documents/Energie_anschliessen/netzzugang-gas/Leitfaden_20180329_Abwicklung-Standardlastprofile-Gas.pdf
    """
    import holidays
    from demandlib import bdew  # Requires demandlib v0.2.1

    # Demandlib uses a different time step notation then lpagg
    # (In demandlib, time Label describes the beginning of the time step)
    weather_data_dl = weather_data.shift(periods=-1, freq="infer")

    settings = cfg['settings']
    houses_list = settings['houses_list_BDEW']
    energy_types = ['Q_Heiz_TT', 'Q_TWW_TT', 'W_TT']
    energy_types_annual = {'Q_Heiz_TT': 'Q_Heiz_a',
                           'Q_TWW_TT': 'Q_TWW_a',
                           'W_TT': 'W_a'}
    multiindex = pd.MultiIndex.from_product([houses_list, energy_types],
                                            names=['house', 'energy'])

    ret_profiles = pd.DataFrame(index=weather_data_dl.index,
                                columns=multiindex,
                                dtype='float')

    if len(houses_list) == 0:  # Skip
        return ret_profiles

    year = int(weather_data.index[0].year)

    holidays_dict = dict()
    if settings.get('holidays'):
        country = settings['holidays'].get('country', 'DE')
        province = settings['holidays'].get('province', None)
        holidays_dict = holidays.country_holidays(
            country, subdiv=province, years=year)

    house_name = houses_list[0]
    # Potential for optimization: Group buildings with the same settings
    # and scale them afterwards
    logger.info("Create %s BDEW profiles with demandlib", len(houses_list))
    for house_name in houses_list:
        # breakpoint()
        house_type = houses_dict[house_name]['house_type']

        # Valid house types for heating profiles
        valid_types = ["GMF", "GPD", "GHD", "GWA", "GGB", "EFH", "GKO",
                       "MFH", "GBD", "GBA", "GMK", "GBH", "GGA", "GHA"]

        matches = [valid_type for valid_type in valid_types
                   if re.search(rf'(?i)\b{valid_type}\b', house_type)]
        if len(matches) == 0:
            logger.error("No valid BDEW heating profile type detected "
                         "for building '%s' with given type '%s'. "
                         "Allowed types are: %s", house_name, house_type,
                         valid_types)
            house_type_heat = None
        elif len(matches) > 1:
            logger.error("Too many BDEW heating profile type detected "
                         "for building '%s' with given type '%s'. "
                         "Allowed types are: %s", house_name, house_type,
                         valid_types)
            house_type_heat = None
        else:
            house_type_heat = matches[0].upper()

        if house_type_heat is not None:
            # Get load profile for energy demand for heating
            Q_heat = houses_dict[house_name].get(
                energy_types_annual['Q_Heiz_TT'], float('NaN'))
            if Q_heat >= 0:
                heatBuilding = bdew.heat_building.HeatBuilding(
                    df_index=weather_data_dl.index,
                    temperature=weather_data_dl['TAMB'],
                    year=year,
                    holidays=holidays_dict,
                    name=house_name,
                    annual_heat_demand=Q_heat,
                    shlp_type=house_type_heat,
                    building_class=0,
                    wind_class=0,
                    ww_incl=False,
                    )

                lp_heat = heatBuilding.get_bdew_profile()
                # Rescale to the given yearly energy demand:
                lp_heat = lp_heat/lp_heat.sum() * Q_heat
                # Store the generated profile
                ret_profiles[house_name, 'Q_Heiz_TT'] = lp_heat

            # Get load profile for energy demand including domestic hot water
            Q_DHW = houses_dict[house_name].get(
                energy_types_annual['Q_TWW_TT'], float('NaN'))
            if Q_DHW >= 0:
                heatBuilding = bdew.heat_building.HeatBuilding(
                    df_index=weather_data_dl.index,
                    temperature=weather_data_dl['TAMB'],
                    year=year,
                    holidays=holidays_dict,
                    name=house_name,
                    annual_heat_demand=Q_DHW,
                    shlp_type=house_type_heat,
                    building_class=0,
                    wind_class=0,
                    ww_only=True,
                    )
                lp_DHW = heatBuilding.get_bdew_profile()
                # Rescale to the given yearly energy demand:
                lp_DHW = lp_DHW/lp_DHW.sum() * Q_DHW
                # Store the generated profile
                ret_profiles[house_name, 'Q_TWW_TT'] = lp_DHW

        # Get electricity profile
        W_a = houses_dict[house_name].get(energy_types_annual['W_TT'],
                                          float('NaN'))
        if W_a >= 0:
            # house types are expected to come in the form of e.g."GHD/G1G"
            # We need to find matches of electrical profile descriptors
            # like g0, g1, ... l0, l1, h0
            pattern = r"(?i)[glh]\d+"
            matches = re.findall(pattern, house_type)
            if len(matches) == 0:
                logger.error("No valid BDEW electrical profile type detected "
                             "for building '%s' with given type '%s'. "
                             "Allowed types: g0, g1, ... g7, l0, l1, l2, h0",
                             house_name, house_type)
                house_type_el = None
            elif len(matches) > 1:
                logger.error("Too many BDEW electrical profile type detected "
                             "for building '%s' with given type '%s'. "
                             "Allowed types: g0, g1, ... g7, l0, l1, l2, h0",
                             house_name, house_type)
                house_type_el = None
            else:
                house_type_el = matches[0].lower()

            if house_type_el is not None:
                elec_slp = bdew.elec_slp.ElecSlp(
                    year=year,
                    holidays=holidays_dict,
                    )
                lp_W_el = elec_slp.get_profile({house_type_el: W_a})
                # Convert unit "W" (per 15min) to "kWh" (per 15min)
                lp_W_el *= 1/4
                # Resample to the desired frequency (time intervall)
                lp_W_el = lpagg.misc.resample_energy(
                        lp_W_el.shift(periods=1, freq="infer"),
                        settings['interpolation_freq']
                        ).shift(periods=-1, freq="infer")
                # Rescale to the given yearly energy demand:
                lp_W_el = lp_W_el/lp_W_el.sum() * W_a
                # Store the generated profile
                ret_profiles[house_name, 'W_TT'] = lp_W_el

    # Covert the time index back to the lpagg notation
    ret_profiles = ret_profiles.shift(periods=1, freq="infer")

    return ret_profiles
