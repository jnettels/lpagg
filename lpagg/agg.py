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


Module agg
----------
The aggregator module is the core of the load profile aggregator project.

'''
import pandas as pd              # Pandas
import os                        # Operaing System
import matplotlib.pyplot as plt  # Plotting library
import yaml                      # Read YAML configuration files
from scipy import optimize
import logging

# Import local modules from load profile aggregator project
import lpagg.weather_converter   # Script for interpolation of weather files
import lpagg.VDI4655
import lpagg.BDEW
import lpagg.simultaneity

# Define the logging function
logger = logging.getLogger(__name__)


def perform_configuration(config_file, ignore_errors=False):
    '''Import the cfg dictionary from the YAML config_file and set defaults.

    Certain tasks must be performed, or else the program will not run
    correctly. However, the user may choose to interfer and perform these
    tasks manually. If 'ignore_errors' is true, this is possible.
    Example: The dict 'houses' is not defined in the cfg and is constructed
    from a different source instead.
    '''
    logger.info('Using configuration file ' + config_file)

    with open(config_file, 'r') as file:
        cfg = yaml.load(file, Loader=yaml.UnsafeLoader)

    # Read settings from the cfg
    settings = cfg['settings']

    # Set logging level
    log_level = settings.get('log_level', 'WARNING').upper()
    if settings.get('Debug', False):
        log_level = 'DEBUG'  # override with old 'DEBUG' key
    logger.setLevel(level=log_level)
    # Set levels for all imported modules
    logging.getLogger('lpagg.agg').setLevel(level=log_level)
    logging.getLogger('lpagg.misc').setLevel(level=log_level)
    logging.getLogger('lpagg.weather_converter').setLevel(level=log_level)
    logging.getLogger('lpagg.VDI4655').setLevel(level=log_level)
    logging.getLogger('lpagg.BDEW').setLevel(level=log_level)
    logging.getLogger('lpagg.simultaneity').setLevel(level=log_level)

    # Define the file paths
    filedir = os.path.dirname(__file__)
    cfg.setdefault('data', dict())
    cfg['data'].setdefault('holiday', os.path.join(filedir, 'resources_load',
                                                   'Feiertage.xlsx'))
    cfg['data'].setdefault('energy_factors',
                           os.path.join(filedir, 'resources_load',
                                        'VDI 4655 Typtag-Faktoren.xlsx'))
    cfg['data'].setdefault('typtage', os.path.join(filedir, 'resources_load',
                                                   'VDI 4655 Typtage.xlsx'))
    cfg['data'].setdefault('BDEW', os.path.join(filedir, 'resources_load',
                                                'BDEW Profile.xlsx'))
    cfg['data'].setdefault('DOE', os.path.join(filedir, 'resources_load',
                                               'DOE Profile TWE.xlsx'))
    cfg['data'].setdefault('futureSolar',
                           os.path.join(filedir, 'resources_load',
                                        'futureSolar Profile.xlsx'))

    # Derive additional settings from user input
    cfg['base_folder'] = os.path.dirname(config_file)
    cfg['print_folder'] = os.path.join(cfg['base_folder'],
                                       cfg.get('result_folder', 'Result'))

    # Certain tasks must be performed, or else the program will not run
    try:
        cfg = houses_sort(cfg)
    except Exception:
        if ignore_errors:
            pass
        else:
            raise

    return cfg


def houses_sort(cfg):
    '''Get the dictionary of houses from the cfg.
    Buildings are separated into VDI4655 for households and BDEW for
    commercial types.
    '''
    houses_dict = cfg['houses']
    houses_list = sorted(houses_dict.keys())
    settings = cfg['settings']
    settings['houses_list'] = houses_list
    settings['houses_list_VDI'] = []
    settings['houses_list_BDEW'] = []

    for house_name in houses_list:
        house_type = houses_dict[house_name]['house_type']
        if house_type == 'EFH' or house_type == 'MFH':
            settings['houses_list_VDI'].append(house_name)

        else:
            settings['houses_list_BDEW'].append(house_name)

    return cfg


def aggregator_run(cfg):
    '''
    Aggregate load profiles
    '''
    settings = cfg['settings']
    weather_data = load_weather_file(cfg)

    # For households, use the VDI 4655
    load_curve_houses, houses_dict = lpagg.VDI4655.run(weather_data, cfg)

    # For the GHD building sector, combine profiles from various sources:
    # (not part of VDI 4655)
    GHD_profiles = lpagg.BDEW.get_GHD_profiles(weather_data, cfg, houses_dict)

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
    load_curve_houses = add_external_profiles(load_curve_houses, cfg)

    # Randomize the load profiles of identical houses
    # (Optional, not part of VDI 4655)
    # -------------------------------------------------------------------------
    load_curve_houses = lpagg.simultaneity.copy_and_randomize_houses(
            load_curve_houses, houses_dict, cfg)

    # Debugging: Show the daily sum of each energy demand type:
#    print(load_curve_houses.resample('D', label='left', closed='right').sum())

    if logger.isEnabledFor(logging.DEBUG):
        load_curve_houses.sort_index(axis=1, inplace=True)

        logger.info('Printing *_houses.dat file')
        print(load_curve_houses.head())
        load_curve_houses.to_csv(
                os.path.join(cfg['print_folder'],
                             os.path.splitext(settings['print_file'])[0]
                             + '_houses.dat'))

    if settings.get('print_houses_xlsx', False):
        logger.info('Printing *_houses.xlsx file')
        # Be careful, can create huge file sizes
        load_curve_houses.to_excel(
                os.path.join(cfg['print_folder'],
                             os.path.splitext(settings['print_file'])[0]
                             + '_houses.xlsx'))

    # Sum up the energy demands of all houses, store result in weather_data
    logger.info('Sum up the energy demands of all houses')
    weather_data = sum_up_all_houses(load_curve_houses, weather_data, cfg)
#    print(weather_data)

    # \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
    #       Implementation 'Heizkurve (Vorlauf- und Rücklauftemperatur)'
    #                        (Not part of the VDI 4655)
    # \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
    cfg = fit_heizkurve(weather_data, cfg)  # Optional
    calc_heizkurve(weather_data, cfg)

    # \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
    #                Normalize results to a total of 1 kWh per year
    # \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
    normalize_energy(weather_data, cfg)  # Optional

    return weather_data


def load_weather_file(cfg):
    """Read and interpolate weather data files.
    """
    settings = cfg['settings']
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
    logger.info('Read and interpolate the data in weather file '+weather_file)

    # Call external method in weather_converter.py:
    weather_data = lpagg.weather_converter.interpolate_weather_file(
                                    weather_file_path,
                                    weather_data_type,
                                    datetime_start,
                                    datetime_end,
                                    interpolation_freq,
                                    remove_leapyear)

    # Analyse weather data
    if logger.isEnabledFor(logging.INFO):
        lpagg.weather_converter.analyse_weather_file(
                weather_data, interpolation_freq, weather_file,
                print_folder=cfg['print_folder'])
    weather_data.index.name = 'Time'
    return weather_data


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


def load_excel_or_csv(filepath, **read_excel_kwargs):
    '''Read data from file
    '''
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
    return df


def add_external_profiles(load_curve_houses, cfg):
    '''This allows to add additional external profiles to the calculated
    load curves.
    Interpolation to the set frequency is performed.

    If house names in existing and external profiles are identical, their
    energies are summed up.
    '''
    settings = cfg['settings']
    if cfg.get('external_profiles', False):
        logger.info('Add external profiles: Loading...')
    else:
        return load_curve_houses

    ext_profiles = pd.DataFrame()
    building_dict = cfg.get('external_profiles', dict())

    df_last = pd.DataFrame()  # For skipping reloading files repeatedly
    path_last = None  # For skipping reloading files repeatedly
    kwargs_last = dict()  # For skipping reloading files repeatedly

    for i, building in enumerate(building_dict):
        fraction = (i+1) / len(building_dict)
        if logger.isEnabledFor(logging.INFO):  # print progress
            print('\r{:5.1f}% done'.format(fraction*100), end='\r')

        filepath = building_dict[building]['file']
        class_ = building_dict[building]['class']
        rename_dict = building_dict[building]['rename_dict']
        read_excel_kwargs = building_dict[building]['read_excel_kwargs']
#        logger.debug(filepath)

        if (filepath == path_last) and (read_excel_kwargs == kwargs_last):
            # We have read this file before with the same settings
            df = df_last.copy()
#            print('skipping', i)
            pass
        else:
            path_last = filepath
            kwargs_last = read_excel_kwargs
            # First load of a file with given settings
            df = load_excel_or_csv(filepath, **read_excel_kwargs)

            # Slice with time selection
            df = df.loc[pd.datetime(*settings['start']):
                        pd.datetime(*settings['end'])]

            # Resample to the desired frequency (time intervall)
            df = lpagg.misc.resample_energy(df, settings['interpolation_freq'])

            df_last = df.copy()  # Save for next loop

        # Apply multiplication factors to columns
        multiply_dict = building_dict[building].get('multiply_dict', dict())
        for key, value in multiply_dict.items():
            try:
                df[key] *= value
            except KeyError:
                logging.error('Key "'+str(key)+'" not found in external '
                              'profile '+str(filepath))
                raise

        # Rename to VDI 4655 standards
        df.rename(columns=rename_dict, inplace=True)

        # Convert into a multi-index DataFrame
        multiindex = pd.MultiIndex.from_product(
            [[class_], [building], rename_dict.values()],
            names=['class', 'house', 'energy'])
        ext_profile = pd.DataFrame(index=load_curve_houses.index,
                                   columns=multiindex)
        for col in rename_dict.values():
            ext_profile[class_, building, col] = df[col]

        # Collect all new external profiles
        ext_profiles = pd.concat([ext_profiles, ext_profile], axis=1)

    # Combine external profiles with existing profiles
    # At this point the DataFrame can get quite large, too large for pandas
    # to handle the stacking
    logger.info('Add external profiles: Combining...')
    load_curve_houses = pd.concat([load_curve_houses, ext_profiles], axis=1,
                                  keys=['int', 'ext'], names=['source'])
    load_curve_houses = load_curve_houses.sort_index(axis=1)

    del df
    del df_last
    del ext_profile
    del ext_profiles

    # TODO: The following line should be removed someday. See Pandas issue:
    # https://github.com/pandas-dev/pandas/issues/24671
    load_curve_houses.fillna(0, axis=1, inplace=True)

    logger.debug('Add external profiles: Grouping...')
    load_curve_houses = load_curve_houses.groupby(
            level=['class', 'house', 'energy'], axis=1).sum()
#    print(load_curve_houses)

    return load_curve_houses


def sum_up_all_houses(load_curve_houses, weather_data, cfg):
    '''By grouping with the levels ``energy`` and ``class``, we can take
    the sum of all houses. These level names are joined into a flat index.
    Finally, the DataFrames ``load_curve_houses`` and ``weather_data`` are
    joined and ``weather_data`` is returned.

    Optionally rename the columns from VDI 4655 standard to a user definition
    (if a dict ``rename_columns`` is defined in ``settings``).
    '''
    settings = cfg['settings']
    # TODO: The following line should be removed someday. See Pandas issue:
    # https://github.com/pandas-dev/pandas/issues/24671
    load_curve_houses.fillna(0, axis=1, inplace=True)

    load_curve_houses_sum = load_curve_houses.groupby(
            level=['energy', 'class'], axis=1).sum()

    rename_dict = settings.get('rename_columns', dict())
    load_curve_houses_sum.rename(columns=rename_dict, inplace=True)
#    print(load_curve_houses_sum)

    # Flatten the heirarchical index
    settings['energy_demands_types'] = ['_'.join(col).strip() for col in
                                        load_curve_houses_sum.columns.values]

    load_curve_houses_sum.columns = settings['energy_demands_types']
    # Concatenate the weather_data and load_curve_houses_sum DataFrames
    weather_data = pd.concat([weather_data, load_curve_houses_sum], axis=1)
    return weather_data


def calc_heizkurve(weather_data, cfg):
    '''Implementation 'Heizkurve (Vorlauf- und Rücklauftemperatur)'
    (Not part of the VDI 4655)

    Calculation according to:
    Knabe, Gottfried (1992): Gebäudeautomation. 1. Aufl.
    Berlin: Verl. für Bauwesen.
    Section 6.2.1., pages 267-268
    '''
    settings = cfg['settings']
    interpolation_freq = settings['interpolation_freq']

    if cfg.get('Heizkurve', None) is not None:
        logger.info('Calculate heatcurve temperatures')

        T_VL_N = cfg['Heizkurve']['T_VL_N']       # °C
        T_RL_N = cfg['Heizkurve']['T_RL_N']       # °C
        T_i = cfg['Heizkurve']['T_i']             # °C
        T_a_N = cfg['Heizkurve']['T_a_N']         # °C
        m = cfg['Heizkurve']['m']

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
            if (M_dot > 0) and (cfg.get('Verteilnetz') is not None):
                length = cfg['Verteilnetz']['length']            # m
                q_loss = cfg['Verteilnetz']['loss_coefficient']
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
            if logger.isEnabledFor(logging.INFO):  # print progress
                print('\r{:5.1f}% done'.format(j/total*100), end='\r')

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


def fit_heizkurve(weather_data, cfg):
    '''In some cases a certain total heat loss of the district heating pipes
    is desired, and the correct heat loss coefficient has to be determined.
    In this case (an optimization problem), we need to find the root of
    the function fit_pipe_heat_loss().

    If 'loss_total_kWh' is not defined in the config file, this whole process
    is skipped.

    Args:
        weather_data (DataFrame): Includes all weather and energy time series

        cfg (Dict):  Configuration dict with initial loss_coefficient

    Returns:
        cfg (Dict):  Configuration dict with updated loss_coefficient
    '''
    settings = cfg['settings']

    def fit_pipe_heat_loss(loss_coefficient):
        '''Helper function that is created as an input for SciPy's optimize
        function. Returns the difference between target and current heat loss.

        Args:
            loss_coefficient (Array): Careful! 'fsolve' will provide an array

        Returns:
            error (float): Deviation between set and current heat loss
        '''
        E_th_loss_set = cfg['Verteilnetz']['loss_total_kWh']
        cfg['Verteilnetz']['loss_coefficient'] = loss_coefficient[0]

        calc_heizkurve(weather_data, settings)
        E_th_loss = weather_data['E_th_loss'].sum()
        error = E_th_loss_set - E_th_loss
        print('Fitting E_th_loss_set:', E_th_loss_set, '... E_th_loss:',
              E_th_loss, 'loss_coefficient:', loss_coefficient[0])
        return error

    loss_kWh = cfg.get('Verteilnetz', dict()).get('loss_total_kWh', None)
    if loss_kWh is None:
        #  Skip the root finding procedure
        return cfg  # return original cfg
    else:
        func = fit_pipe_heat_loss
        x0 = cfg['Verteilnetz']['loss_coefficient']
        root = optimize.fsolve(func, x0)  # Careful, returns an array!
        print("Solution: loss_coefficient = ", root, '[W/(m*K)]')
        cfg['Verteilnetz']['loss_coefficient'] = root[0]
        return cfg  # config with updated loss_coefficient


def normalize_energy(weather_data, cfg):
    '''Normalize results to a total of 1 kWh per year
    '''
    settings = cfg['settings']
    if cfg.get('normalize', False) is True:
        logger.info('Normalize load profile')
        for column in settings['energy_demands_types']:
            yearly_sum = weather_data[column].sum()
            weather_data[column] = weather_data[column]/yearly_sum


def plot_and_print(weather_data, cfg):
    '''Plot & Print
    '''
    settings = cfg['settings']
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
    if settings.get('show_plot', False) is True:
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
    lpagg.weather_converter.print_IGS_weather_file(
            weather_data,
            cfg['print_folder'],
            settings['print_file'],
            settings.get('print_index', True),
            settings.get('print_header', True))
