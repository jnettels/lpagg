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


Module agg
----------
The aggregator module is the core of the load profile aggregator project.

"""
import locale
import numpy as np
import pandas as pd              # Pandas
import os                        # Operaing System
import matplotlib.pyplot as plt  # Plotting library
import yaml                      # Read YAML configuration files
from scipy import optimize
import logging
import datetime
# from memory_profiler import profile

# Import local modules from load profile aggregator project
import lpagg.weather_converter   # Script for interpolation of weather files
import lpagg.VDI4655
import lpagg.BDEW
import lpagg.simultaneity

# Define the logging function
logger = logging.getLogger(__name__)


def perform_configuration(config_file='', cfg=None, ignore_errors=False):
    """Import the cfg dictionary from the YAML config_file and set defaults.

    Certain tasks must be performed, or else the program will not run
    correctly. However, the user may choose to interfer and perform these
    tasks manually. If 'ignore_errors' is true, this is possible.
    Example: The dict 'houses' is not defined in the cfg and is constructed
    from a different source instead.


    Parameters
    ----------
    config_file : String, optional
        Path to a YAML configuration file. The default is ''.
    cfg : dict, optional
        A dictionary, to be used instead of a YAML file. The default is None.
    ignore_errors : bool, optional
        Ignore errors on some preperation steps. The default is False.

    Returns
    -------
    cfg : dict
        Configuration dictionary.

    """
    logger.info('Using configuration file ' + config_file)

    if cfg is None:
        with open(config_file, 'r', encoding='utf8') as file:
            cfg = yaml.load(file, Loader=yaml.UnsafeLoader)

    # Read settings from the cfg
    settings = cfg['settings']

    settings.setdefault('print_file', 'lpagg_load.dat')
    settings.setdefault('intervall', '1 hours')

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
    cfg.setdefault('data', dict())
    cfg['data'].setdefault('energy_factors',
                           os.path.join('resources_load',
                                        'VDI 4655 Typtag-Faktoren.xlsx'))
    cfg['data'].setdefault('typtage',
                           os.path.join('resources_load',
                                        'VDI 4655 Typtage.xlsx'))
    cfg['data'].setdefault('BDEW',
                           os.path.join('resources_load',
                                        'BDEW Profile.xlsx'))
    cfg['data'].setdefault('DOE',
                           os.path.join('resources_load',
                                        'DOE Profile TWE.xlsx'))
    cfg['data'].setdefault('futureSolar',
                           os.path.join('resources_load',
                                        'futureSolar Profile.xlsx'))

    # Derive additional settings from user input
    cfg['base_folder'] = os.path.dirname(config_file)
    cfg.setdefault('print_folder',
                   os.path.join(cfg['base_folder'],
                                cfg['settings'].get('result_folder',
                                                    'Result')))

    weather_file = os.path.join(cfg['base_folder'], settings['weather_file'])
    cfg['settings']['weather_file'] = os.path.abspath(weather_file)

    language = cfg['settings'].get('language', 'de')
    if language == 'de':
        # Define language for axis labels (month names)
        locale.setlocale(locale.LC_ALL, 'de_DE.UTF-8')

    # Certain tasks must be performed, or else the program will not run
    try:
        cfg = get_houses_from_table_file(cfg)
        cfg = houses_sort(cfg)
    except Exception:
        if ignore_errors:
            pass
        else:
            raise

    return cfg


def get_houses_from_table_file(cfg):
    """Get building information from a table file.

    In addition to defining a dictionary with all houses in the yaml config,
    load the list of houses from an Excel table (if given).
    """
    if cfg.get('houses_table_file'):
        # File path is interpreted relative to yaml config file
        file = os.path.join(cfg['base_folder'],
                            cfg['houses_table_file'].get('file', ''))

        # Optional keyword arguments can be used to customize read_excel()
        kwargs = cfg['houses_table_file'].get('kwargs', dict())
        kwargs.setdefault('index_col', 0)
        df = pd.read_excel(file, **kwargs)
        if cfg['houses_table_file'].get('transpose'):
            df = df.T
        df_dict = df.to_dict()  # Convert DataFrame to dictionary

        cfg.setdefault('houses', dict())  # define, if it does not exist
        # Add each house to the existing 'houses' dict
        for house in df_dict.keys():
            # Issue a warning, if necessary:
            if cfg['houses'].get(house) is not None:
                logger.warning('You are overwriting the data of house {} '
                               'defined in the YAML file with the data '
                               'defined in the Excel file.'.format(house))

            # Now we can add the house to the dict
            cfg['houses'][house] = df_dict[house]

    return cfg


def houses_sort(cfg):
    """Get the dictionary of houses from the cfg.

    Buildings are separated into VDI4655 for households and BDEW for
    commercial types.
    """
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
    """Run the aggregator to create the load profiles.

    Args:
        cfg (dict): A dictionary containing the dicts 'settings' and 'houses'.

    Returns:
        result_dict (dict): A dictionary containing the resulting DataFrames.
        ``weather_data`` contains the input weather data combined with the
        aggregated load profiles. ``load_curve_houses`` contains all individual
        profiles of the houses. ``P_max_houses`` contains the peak power
        of each house.

    """
    if cfg['settings'].get('unique_profile_workflow', True):
        cfg = preprocess_unique_profiles(cfg)

        # Now the "sorting" of houses has to be triggered manually
        cfg = lpagg.agg.houses_sort(cfg)

        # Aggregate load profiles
        result_dict = _aggregator_run(cfg)

        # Run the post-processing (assign unique profiles to all buildings)
        result_dict = postprocess_unique_profiles(result_dict, cfg)

    else:
        result_dict = _aggregator_run(cfg)

    return result_dict


def _aggregator_run(cfg):
    """Run the aggregator to create the load profiles.

    Args:
        cfg (dict): A dictionary containing the dicts 'settings' and 'houses'.

    Returns:
        result_dict (dict): A dictionary containing the resulting DataFrames.
        ``weather_data`` contains the input weather data combined with the
        aggregated load profiles. ``load_curve_houses`` contains all individual
        profiles of the houses. ``P_max_houses`` contains the peak power
        of each house.

    """
    settings = cfg['settings']
    weather_data = load_weather_file(cfg)

    # For households, use the VDI 4655
    # If the python package 'demandlib' is installed, it can be used
    # for the VDI4655 calculation with improved performance. It might fail,
    # since that implementation is currently in progress.
    # https://github.com/oemof/demandlib
    if cfg['settings'].get('use_demandlib', False):
        load_curve_houses, houses_dict = lpagg.VDI4655.run_demandlib(
            weather_data, cfg)
    else:
        load_curve_houses, houses_dict = lpagg.VDI4655.run(weather_data, cfg)

    # For the GHD building sector, combine profiles from various sources:
    # (not part of VDI 4655)
    logger.debug('Load commercial profiles')
    if cfg['settings'].get('use_demandlib', False):
        GHD_profiles = lpagg.BDEW.run_demandlib(weather_data, cfg, houses_dict)
    else:
        # Identify seasons and weekdays for BDEW
        lpagg.VDI4655.get_typical_days(weather_data, cfg)
        # Get the profiles
        GHD_profiles = lpagg.BDEW.get_GHD_profiles(weather_data, cfg,
                                                   houses_dict)
    logger.debug('Combine residential and commercial profiles')
    load_curve_houses = pd.concat([load_curve_houses, GHD_profiles],
                                  axis=1, sort=False,
                                  keys=['HH', 'GHD'], names=['class'])

    if settings.get('apply_DST', True):
        # Shift the profiles according to daylight saving time
        # (Optional, not part of VDI 4655)
        logger.debug('Apply daylight saving time')
        load_curve_houses = apply_DST(load_curve_houses)

    # Flatten domestic hot water profile to a daily mean value
    # (Optional, not part of VDI 4655)
    # -------------------------------------------------------------------------
    load_curve_houses = flatten_daily_TWE(load_curve_houses, settings)
    # print(load_curve_houses.head())

    # Add external (complete) profiles
    # (Optional, not part of VDI 4655)
    # -------------------------------------------------------------------------
    load_curve_houses = add_external_profiles(load_curve_houses, cfg)

    # Randomize the load profiles of identical houses
    # (Optional, not part of VDI 4655)
    # -------------------------------------------------------------------------
    load_curve_houses = lpagg.simultaneity.copy_and_randomize_houses(
            load_curve_houses, houses_dict, cfg)

    # TODO temporary solution!!!
    # load_curve_houses, df_ref = lpagg.simultaneity.create_simultaneity(
    #     load_curve_houses,
    #     sigma=settings['sigma'],
    #     copies=0,
    #     seed=4)

    # Debugging: Show the daily sum of each energy demand type:
    # print(load_curve_houses.resample('D', label='left', closed='right').sum())

    # Save some intermediate result files
    P_max_houses = intermediate_printing(load_curve_houses, cfg)

    # Sum up the energy demands of all houses, store result in weather_data
    logger.info('Sum up the energy demands of all houses')
    weather_data = sum_up_all_houses(load_curve_houses, weather_data, cfg)
    # print(weather_data)

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

    result_dict = {'weather_data': weather_data,
                   'load_curve_houses': load_curve_houses,
                   'P_max_houses': P_max_houses,
                   }

    return result_dict


def preprocess_unique_profiles(cfg):
    """Perform pre-processing of workflow with unique profiles."""
    logger.info("Perform pre-processing of workflow with unique profiles")

    # VDI4655 defines default values for annual eletricity and domestic
    # hot water energy demand. Fill in those values where appropriate
    lpagg.VDI4655.get_annual_energy_demand(cfg)  # updates cfg['houses']
    # Create DataFrame from dict for easier workflow
    df = pd.DataFrame.from_dict(cfg['houses'], orient='index')
    df_unique = df.copy()
    # Set all not-na values to 1 (normalized total energy demand of 1 kWh)
    cols_energy = ['Q_Heiz_a', 'Q_Kalt_a', 'Q_TWW_a', 'W_a']
    cols_energy = [c for c in cols_energy if c in df_unique.columns]
    df_unique[cols_energy] = df_unique[cols_energy] * 0 + 1
    # Disable sigma for unique profiles (set not-na to 0)
    df_unique['sigma'] *= 0

    if 'copies' in df_unique.columns:  # Not supported yet
        # Set number of copies to default value '1'
        df_unique['copies'] = df_unique['copies'] * 0 + 1

    df_unique = df_unique.drop_duplicates().reset_index(drop=True)
    df_unique = df_unique.reset_index().rename(columns={'index': 'id_unique'})

    df = df.merge(df_unique
                  .drop(columns=cols_energy + ['sigma', 'copies'])
                  ).set_index(df.index)

    # Replace the regular 'houses' dict with only the unique buildings
    cfg['houses'] = df_unique.drop(columns='id_unique').to_dict(orient='index')
    # Store an additional entry in the config, which will later be used
    # to assign the unique profiles to each building
    cfg['houses_id_unique'] = df.to_dict(orient='index')

    # For settings where the default is True, set them specifically
    cfg['settings']['calc_P_max'] = True

    # Create a backup of original configuration settings
    cfg['settings_bak'] = cfg['settings'].copy()
    # Set all settings to False which need to be disabled for the
    # run with the unique profiles
    cfg['settings']['print_houses_dat'] = False
    cfg['settings']['print_houses_xlsx'] = False
    cfg['settings']['calc_P_max'] = False
    cfg['settings']['print_P_max'] = False
    cfg['settings']['print_GLF_stats'] = False
    cfg['settings']['show_plot'] = False

    logger.info('Number of unique profiles: %s', len(df_unique))

    return cfg


# @profile  # Decorator for memory monitoring
def postprocess_unique_profiles(agg_dict, cfg,
                                dtype='auto',
                                length_enable_float32=10000,
                                workflow='A'):
    """Perform post-processing of workflow with unique profiles."""
    logger.info("Perform post-processing of workflow with unique profiles")

    # Process load curve DataFrame
    df_lc = agg_dict['load_curve_houses']
    # all-na columns are not part of the desired output and need to be removed
    df_lc = df_lc.dropna(axis='columns', how='all')
    df_lc = df_lc.rename_axis(columns={'house': 'id_unique'})

    # Process unique houses DataFrame
    df_unique = pd.DataFrame.from_dict(cfg['houses_id_unique'], orient='index')
    df_unique = df_unique.rename_axis(index=['house'])
    cols_keep = ['Q_Heiz_a', 'Q_Kalt_a', 'Q_TWW_a', 'W_a', 'id_unique']
    cols_keep = [c for c in cols_keep if c in df_unique]
    df_unique = df_unique[cols_keep]
    df_unique = df_unique.set_index(['id_unique'], append=True)
    df_unique = df_unique.rename(columns={'Q_Heiz_a': 'Q_Heiz_TT',
                                          'Q_Kalt_a': 'Q_Kalt_TT',
                                          'Q_TWW_a': 'Q_TWW_TT',
                                          'W_a': 'W_TT'})
    df_unique = df_unique.rename_axis(columns=['energy'])
    df_unique = df_unique.stack(future_stack=True)  # Keep NA values

    # The next step is probably the most memory-intensive operation.
    # For scenarios with >10000 buildings it might cause problems for
    # systems with e.g. 16GB RAM. Changing the dtype to float32 helped in
    # certain cases.
    if dtype == 'auto':
        if len(df_unique) > length_enable_float32:
            dtype = 'float32'
            df_lc = df_lc.astype(dtype)
            df_unique = df_unique.astype(dtype)

    # Match the annual energy sum of each building with its unique profile.
    # During tests with very large datasets, different workflows were
    # explored.
    if workflow == "A":
        logger.info("Match unique profiles (Workflow %s)", workflow)
        df_lc = (
            pd.DataFrame(index=df_lc.index, columns=df_unique.index,
                         dtype=df_lc.dtypes.values[0])
            .fillna(1.0)
            .mul(df_unique)
            .mul(df_lc, axis='index')  # Multiply matching parts of multiindex
            .dropna(axis='columns', how='all')
            )

    elif workflow == "B":
        logger.info("Match unique profiles (Workflow %s)", workflow)
        df_lc = df_lc.T.multiply(df_unique, axis=0).T
        df_lc = df_lc.dropna(axis='columns', how='all')

    elif workflow == "dask":  # Does not work (yet)!
        logger.info("Match unique profiles (Workflow %s)", workflow)
        raise NotImplementedError("Using dask for large datasets should be "
                                  "possible, but does not work yet.")
        # import dask.dataframe as dd
        # df_lc.columns = df_lc.columns.to_flat_index()
        # df_unique.index = df_unique.index.to_flat_index()

        # df_res = (
        #     pd.DataFrame(index=df_lc.index, columns=df_unique.index,
        #                  dtype=df_lc.dtypes.values[0])
        #     .fillna(1.0)
        #     .mul(df_unique)
        #     )

        # # Convert Pandas DataFrames to Dask DataFrames
        # dd_res = dd.from_pandas(df_res, npartitions=8)
        # dd_lc = dd.from_pandas(df_lc, npartitions=8)
        # dd_unique = dd.from_pandas(df_unique, npartitions=8)

        # dd_lc = (
        #     dd_res
        #     # .mul(dd_unique)
        #     .mul(dd_lc, axis='index')
        #     )
        # df_lc = df_lc.compute()

    # Adapt result DataFrame to expected output

    # Restore the original sorting of the houses from df_unique.
    # To do that, match their column index contents and level order
    idx = df_unique.index.to_frame(index=False)
    idx = pd.merge(idx,
                   df_lc.columns.to_frame(index=False),
                   on=idx.columns.to_list())
    df_unique = df_unique.reindex(pd.MultiIndex.from_frame(idx))

    # Match levels and their order in df_lc to load_curve_houses
    df_lc = df_lc.droplevel('id_unique', axis='columns')
    df_lc = df_lc.reorder_levels(agg_dict['load_curve_houses'].columns.names,
                                 axis='columns')
    # Match levels and their order in df_unique to df_lc
    df_unique = (df_unique
                 .droplevel('id_unique')
                 .reorder_levels(df_lc.columns.names)
                 )
    # Now the actual sorting of the column index can happen
    df_lc = df_lc.reindex(df_unique.index, axis='columns')  # remove?

    # Test success by comparing the annual sum of each building
    df_lc_sum = df_lc.sum(min_count=1)
    df_unique_compare = df_unique.dropna().astype(df_lc_sum.dtype)
    if not df_lc_sum.round(4).equals(df_unique_compare.round(4)):
        if dtype == 'float32':
            txt = ("Annual sums of individual profiles do not "
                   "match the input sums. When using dtype 'float32', "
                   "this can be expected to occur.")
            if df_unique.sum() == df_lc.sum().sum():
                txt += (" However, the total annual sum of all profiles is "
                        "correct.")
            logger.warning(txt)
        else:
            raise ValueError("Mismatch of sums in profile post-processing")

    # Restore settings
    for key, value in cfg['settings'].items():
        cfg['settings_bak'].setdefault(key, value)
    cfg['settings'] = cfg['settings_bak']

    # Now repeat the remaining steps of _aggregator_run with the full dataset

    # Randomize the load profiles of identical houses
    df_lc = lpagg.simultaneity.copy_and_randomize_houses(
            df_lc, cfg['houses_id_unique'], cfg)

    # Save some intermediate result files
    P_max_houses = lpagg.agg.intermediate_printing(df_lc, cfg)

    weather_data = agg_dict['weather_data']

    # Remove all columns from weather data that need to be updated
    weather_data = weather_data.drop(
        columns=(cfg['settings']['energy_demands_types']
                 + ['T_VL', 'T_RL', 'M_dot']),
        errors='ignore')
    weather_data = lpagg.agg.sum_up_all_houses(df_lc, weather_data, cfg)

    # Implementation 'Heizkurve (Vorlauf- und Rücklauftemperatur)'
    cfg = fit_heizkurve(weather_data, cfg)  # Optional
    calc_heizkurve(weather_data, cfg)

    # Normalize results to a total of 1 kWh per year
    normalize_energy(weather_data, cfg)  # Optional

    # Update the results in the aggregation dictionary
    agg_dict['load_curve_houses'] = df_lc
    agg_dict['P_max_houses'] = P_max_houses
    agg_dict['weather_data'] = weather_data

    return agg_dict


def load_weather_file(cfg):
    """Read and interpolate weather data files."""
    settings = cfg['settings']
    weather_file = settings['weather_file']

    weather_data_type = settings['weather_data_type']
    try:
        # * read as args, when defined in yaml file
        datetime_start = datetime.datetime(*settings['start'])
        datetime_end = datetime.datetime(*settings['end'])
    except TypeError:  # Read as Timestamp objects otherwise
        datetime_start = settings['start']
        datetime_end = settings['end']
#    datetime_start = datetime.datetime(2017,1,1,00,00,00) # Example
#    datetime_end = datetime.datetime(2018,1,1,00,00,00)
    interpolation_freq = pd.Timedelta(settings['intervall'])
#    interpolation_freq = pd.Timedelta('14 minutes')
#    interpolation_freq = pd.Timedelta('1 hours')
    remove_leapyear = settings.get('remove_leapyear', False)

    settings['interpolation_freq'] = interpolation_freq
    logger.info('Read and interpolate the data in weather file '+weather_file)

    # Call external method in weather_converter.py:
    weather_data = lpagg.weather_converter.interpolate_weather_file(
                                    weather_file,
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
    """Flatten domestic hot water profile to a daily mean value.

    The domestic hot water demand profile represents what is actually
    used within the house. But the total demand of a house with hot
    water circulation is nearly constant for each day (from the
    perspective of a district heating system).
    """
    if settings.get('flatten_daily_TWE', False):

        logger.info('Flatten daily domestic hot water profile')

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
    """Read data from file."""
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
    """Add additional external profiles to the calculated load curves.

    Interpolation to the set frequency is performed. If house names in existing
    and external profiles are identical, their energies are summed up.
    """
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
        read_excel_kwargs = building_dict[building].get('read_excel_kwargs',
                                                        dict())
        # logger.debug(filepath)

        if (filepath == path_last) and (read_excel_kwargs == kwargs_last):
            # We have read this file before with the same settings
            df = df_last.copy()
            # print('skipping', i)
            pass
        else:
            path_last = filepath
            kwargs_last = read_excel_kwargs
            # First load of a file with given settings
            df = load_excel_or_csv(filepath, **read_excel_kwargs)

            # Slice with time selection
            try:
                df = df.loc[datetime.datetime(*settings['start']):
                            datetime.datetime(*settings['end'])]
            except Exception:
                logger.error('Error while slicing the external profile: {}\n{}'
                             .format(filepath, df.to_string(max_rows=10)))
                raise

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
                continue

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
    load_curve_houses = load_curve_houses.T.groupby(
            level=['class', 'house', 'energy']).sum().T
#    print(load_curve_houses)

    return load_curve_houses


def intermediate_printing(load_curve_houses, cfg):
    """Perform some intermediate printing tasks."""
    settings = cfg['settings']

    # TODO: The following line should be removed someday. See Pandas issue:
    # https://github.com/pandas-dev/pandas/issues/24671
    load_curve_houses.fillna(0, axis=1, inplace=True)

    # Print load profile for each house (creates large file sizes!)
    load_curve_houses_tmp = (load_curve_houses
                             .T.groupby(level=['house', 'energy']).sum().T
                             .sort_index(axis=1))

    if logger.isEnabledFor(logging.DEBUG):
        logger.info('Printing *_houses.dat file')
        print(load_curve_houses_tmp.head())
        load_curve_houses_tmp.to_csv(
                os.path.join(cfg['print_folder'],
                             os.path.splitext(settings['print_file'])[0]
                             + '_houses.dat'))

    if settings.get('print_houses_xlsx', False):
        logger.info('Printing *_houses.xlsx file')
        df_H = load_curve_houses_tmp
        df_D = df_H.resample('D', label='left', closed='right').sum()
        df_W = df_D.resample('W', label='right', closed='right').sum()
        df_M = df_D.resample('ME', label='right', closed='right').sum()
        df_A = df_M.resample('YE', label='right', closed='right').sum()
        print(df_M)

        # Be careful, can create huge file sizes
        df_to_excel(df=[df_H, df_D, df_W, df_M, df_A],
                    sheet_names=['Hour', 'Day', 'Week', 'Month', 'Year'],
                    path=os.path.join(
                        cfg['print_folder'],
                        os.path.splitext(settings['print_file'])[0]
                        + '_houses.xlsx'),
                    merge_cells=True,
                    )

    # Print peak power
    hours = settings['interpolation_freq'].seconds / 3600.0  # h
    P_max_houses = load_curve_houses.copy()
    P_max_houses = P_max_houses.stack(["class", "house"])
    P_max_houses["Q_th"] = (P_max_houses["Q_Heiz_TT"]
                            + P_max_houses["Q_TWW_TT"])
    P_max_houses = P_max_houses.unstack(["class", "house"])
    P_max_houses = (P_max_houses
                    .T.groupby(level=['house', 'energy']).sum().T
                    .max(axis=0)
                    .unstack()
                    .sort_index()
                    .rename(columns={'Q_Heiz_TT': 'P_th_RH',
                                     'Q_TWW_TT': 'P_th_TWE',
                                     'W_TT': 'P_el',
                                     'Q_th': 'P_th'})
                    )

    P_max_houses = P_max_houses / hours  # Convert kWh to kW
    if settings.get('print_P_max', False):
        logger.info('Printing *_P_max.dat and .xlsx files')
        P_max_houses.to_csv(os.path.join(cfg['print_folder'], os.path.splitext(
            settings['print_file'])[0] + '_P_max.dat'))
        df_to_excel(df=[P_max_houses], sheet_names=['P_max (kW)'],
                    path=os.path.join(cfg['print_folder'], os.path.splitext(
                        settings['print_file'])[0] + '_P_max.xlsx'))

    return P_max_houses


def sum_up_all_houses(load_curve_houses, weather_data, cfg):
    """Sum up the energies of all houses.

    By grouping with the levels ``energy`` and ``class``, we can take
    the sum of all houses. These level names are joined into a flat index.
    Finally, the DataFrames ``load_curve_houses`` and ``weather_data`` are
    joined and ``weather_data`` is returned.

    Optionally rename the columns from VDI 4655 standard to a user definition
    (if a dict ``rename_columns`` is defined in ``settings``).
    """
    settings = cfg['settings']
    # TODO: The following line should be removed someday. See Pandas issue:
    # https://github.com/pandas-dev/pandas/issues/24671
    load_curve_houses.fillna(0, axis=1, inplace=True)

    load_curve_houses_sum = load_curve_houses.T.groupby(
            level=['energy', 'class']).sum().T

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
    """Calculate a 'Heizkurve (Vorlauf- und Rücklauftemperatur)'.

    (Not part of the VDI 4655)

    Calculation according to:
    Knabe, Gottfried (1992): Gebäudeautomation. 1. Aufl.
    Berlin: Verl. für Bauwesen.
    Section 6.2.1., pages 267-268
    """
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
    """Fit heat loss coefficient of heatcurve to a given total heat loss.

    In some cases a certain total heat loss of the district heating pipes
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
    """
    settings = cfg['settings']

    def fit_pipe_heat_loss(loss_coefficient):
        """Help with fitting the heat loss.

        Helper function that is created as an input for SciPy's optimize
        function. Returns the difference between target and current heat loss.

        Args:
            loss_coefficient (Array): Careful! 'fsolve' will provide an array

        Returns:
            error (float): Deviation between set and current heat loss
        """
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
    """Normalize results to a total of 1 kWh per year."""
    settings = cfg['settings']
    if cfg.get('normalize', False) is True:
        logger.info('Normalize load profile')
        for column in settings['energy_demands_types']:
            yearly_sum = weather_data[column].sum()
            weather_data[column] = weather_data[column]/yearly_sum


def plot_and_print(weather_data, cfg):
    """Plot & print the results."""
    settings = cfg['settings']
    # Print a table of the energy sums to the console (monthly and annual)
    filter_sum = ['E_th_', 'E_el']
    filter_sum_th = ['E_th_']
    filter_sum_el = ['E_el_']
    sum_list = []
    sum_list_th = []
    sum_list_el = []
    for column in weather_data.columns:
        for filter_ in filter_sum:
            if filter_ in column:
                sum_list.append(column)
        for filter_ in filter_sum_th:
            if filter_ in column:
                sum_list_th.append(column)
        for filter_ in filter_sum_el:
            if filter_ in column:
                sum_list_el.append(column)

    # Set the number of decimal points for the following terminal output
    pd.set_option('display.precision', 2)

    weather_daily_sum = weather_data[sum_list].resample('D', label='left',
                                                        closed='right').sum()
    logger.info('Calculations completed')
    logger.info('Monthly energy sums in kWh:')
    weather_montly_sum = weather_daily_sum.resample('ME', label='right',
                                                    closed='right').sum()
    if logger.isEnabledFor(logging.INFO):
        print(weather_montly_sum)
        print()
    logger.info('Annual energy sums in kWh:')
    weather_annual_sum = weather_montly_sum.resample('YE', label='right',
                                                     closed='right').sum()
    if logger.isEnabledFor(logging.INFO):
        print(weather_annual_sum)
        print('Total heat energy demand is {:.2f} kWh.'.format(
            weather_annual_sum[sum_list_th].sum(axis=1).sum()))
        print('Total electricity demand is {:.2f} kWh.'.format(
            weather_annual_sum[sum_list_el].sum(axis=1).sum()))
        print('     Total energy demand is {:.2f} kWh.'.format(
            weather_annual_sum.sum(axis=1).sum()))
        print()

    pd.reset_option('display.precision')  # ...and reset the setting from above

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
            settings.get('print_header', True),
            sheet_name='kWh')


def apply_DST(df, tz_default='Etc/GMT-1', tz_DST='CET', normalize=True):
    """Apply daylight saving time to a DataFrame.

    Assumes that the values in the given DataFrame are meant to be in timezone
    tz_DST, while the current index is formated as timezone tz_default.
    After fitting the data to an index with tz_DST, the timezone is converted
    to tz_default before returning it. Effectively, this shifts the profiles
    -1 hour during summer.

    See https://en.wikipedia.org/wiki/List_of_tz_database_time_zones

    Args:
        df (DataFrame): A Pandas DataFrame with datetime index.

        tz_default (str, optional): The timezone of the index. Defaults
        to 'Etc/GMT-1' (for central europe, but without summer time).

        tz_DST (TYPE, optional): The timezone of the data. Defaults to 'CET'
        (Central European Time, which includes daylight saving time).

    Returns:
        df_DST (DataFrame): DataFrame indexed with tz_default.

    """
    # Localize the DataFrame with the default timezone, to make it tz-aware
    df_default = df.tz_localize(tz_default)
    # Remove localization from the temporary DataFrame that we are manipulating
    df_DST = df_default.tz_localize(None)
    # Localize the temporary df to the DST timezone. Ambiguous is a list of
    # True for all steps to tell that all steps are meant as local time.
    # This will cause some steps to be nonexistent, which we fill with 'NaT'.
    df_DST = df_DST.tz_localize(tz_DST, ambiguous=[True]*len(df_DST.index),
                                nonexistent='NaT')
    # Convert from local timezone (with DST) to default (without DST)
    df_DST = df_DST.tz_convert(tz_default)
    # Drop the empty time stamp(s) from March (where the hour we skipped in DST
    # was deleted).
    df_DST = df_DST[df_DST.index.notnull()]
    # In October, the repeated DST-hour is not in the index yet.
    # Reindex with the initial index. This will cause missing rows in October.
    # We fill them with method forward fill.
    df_DST = df_DST.reindex_like(df_default, method='ffill')
    # Finally, make the DataFrame timezone-unaware before returning it.
    df_DST = df_DST.tz_localize(None)

    # Shifting the time steps in this manner slightly alters the yearly sums.
    # They have to stay constant, since they are normalized to a certain sum.
    if normalize:
        logger.debug('Normalize after applying DST')
        df_DST = df_DST.div(df_DST.sum()).mul(df_default.sum())
        # Columns with a sum of zero produce NaN, so we fill these
        df_DST.fillna(0, inplace=True)

    return df_DST
