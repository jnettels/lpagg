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


Module simultaneity
-------------------
Copies of a given time series are created and, if a standard deviation
``sigma`` is given, a time shift is applied to the copies.

This method decreases the maximum load and thereby creates the effect of a
"simultaneity factor" (Gleichzeitigkeitsfaktor). It can be calculated by
dividing the maximum loads with and without the normal distribution.

- Primarily, this module is part of the automated load profile generator
  program.

- However, it also serves as a standalone script for applying a random time
  shift to the columns in a given Excel file. To see the help, call:

  .. code::

      python simultaneity.py --help

.. note::

    Remember: 68.3%, 95.5% and 99.7% of the values lie within one,
    two and three standard deviations of the mean.
    Example: With an interval of 15 min and a deviation of
    sigma = 2 time steps, 68% of profiles are shifted up to ±30 min (±1σ).
    27% of profiles are shifted ±30 to 60 min (±2σ) and another
    4% are shifted ±60 to 90 min (±3σ).

'''

import os
import logging
import pandas as pd
import functools
import numpy as np
import matplotlib.pyplot as plt  # Plotting library
import matplotlib as mpl

# Import local modules from load profile aggregator project
import lpagg.misc
import lpagg.DIN4708

# Define the logging function
logger = logging.getLogger(__name__)


def main():
    '''Read user input and run the script.
    '''
    setup()  # Set some basic settings

    args = run_OptionParser()  # Read user options

    run(args.sigma, args.copies, args.seed, args.file,
        set_hist=dict(PNG=True, PDF=False))


def run(sigma, copies, seed, file, set_hist=dict(), show_plot=False):
    '''Perform task and produce the output.
    '''
    # 1) Read in data as Pandas DataFrame from the given file
    df = pd.read_excel(file, header=0, index_col=0)
#    print(df)  # Show imported DataFrame on screen

    # 2) Create the simultaneity effect
    df, df_ref = create_simultaneity(df, sigma, copies, seed,
                                     os.path.dirname(file), set_hist)
#    print(df)  # Show results on screen

    df_sum = pd.DataFrame(data={'Shift': df.sum(axis=1),
                                'Reference': df_ref.sum(axis=1)})
    GLF = df_sum['Shift'].max() / df_sum['Reference'].max()

    if logger.isEnabledFor(logging.DEBUG):
        print(df_sum)

    if show_plot:
        logger.debug('Showing plot figure with matplotlib...')
        try:  # Show a plot of the aggregated profiles
            plt.close('all')
            ax = df_sum.plot()
            ax.yaxis.grid(True)  # Activate grid on horizontal axis
            plt.show()
        except Exception as e:
            logger.exception(e)

    # 3) Print results to an Excel spreadsheet
    output = os.path.splitext(file)[0]+'_c'+str(copies)+'_s'+str(sigma)+'.xlsx'
    lpagg.misc.df_to_excel([df, df_ref, df_sum],
                           sheet_names=['Shift', 'Reference', 'Sum'],
                           path=output)
    result = dict({'output': output, 'GLF': GLF, 'df_sum': df_sum})
    # Done!
    return result


def setup():
    '''Basic setup of logger.
    '''
    # Define the logging function
    log_level = 'DEBUG'
#    log_level = 'INFO'
    logger.setLevel(level=log_level.upper())  # Logger for this module
    logging.basicConfig(format='%(asctime)-15s %(message)s')

    # Define style settings for the plots
    try:  # Try to load personalized matplotlib style file
        mpl.style.use(os.path.join(os.path.dirname(__file__),
                                   './lpagg.mplstyle'))
    except OSError as ex:
        logger.debug(ex)


def run_OptionParser():
    '''Define and run the argument parser. Return the collected arguments.
    '''
    import argparse
    description = 'simultaneity.py: Create simultaneity effects in time series'
    parser = argparse.ArgumentParser(description=description,
                                     formatter_class=argparse.
                                     ArgumentDefaultsHelpFormatter)

    parser.add_argument('-f', '--file', dest='file', help='Path to an Excel '
                        'spreadsheet.', type=str, default=None)
    parser.add_argument('-s', '--sigma', dest='sigma', help='Standard '
                        'deviation of random distribution.', type=float,
                        default=10)
    parser.add_argument('-c', '--copies', dest='copies', help='Number '
                        'of (randomized) copies for each column.', type=int,
                        default=2)
    parser.add_argument('--seed', dest='seed', help='Seed for Numpy '
                        'random generator.', type=int, default=4)
    args = parser.parse_args()

    if args.file is None:
        args.file = lpagg.misc.file_dialog(title='Choose an Excel file',
                                           filetypes=(('Excel File', '*.xlsx'),
                                                      ))
        if args.file is None:
            logger.info('Empty selection. Show help and exit program...')
            parser.print_help()
            input('\nPress the enter key to exit.')
            raise SystemExit

    return args


def create_simultaneity(df, sigma, copies, seed, save_folder, set_hist):
    '''Apply the simultaneity effect to the columns in a given DataFrame.
    This function is called by ``main()`` in the standalone script mode.
    It is the minimalist equivalent to ``copy_and_randomize_houses()``, which
    is used in the load profile aggregator program. Both call
    ``copy_and_randomize()`` to perform the actual randomization.

    Args:
        df (Pandas DataFrame): Time series data to use for simultaneity.

        sigma (float): The standard deviation of the normal distribution from
        which the time shifts are randomly drawn.

        copies (int): The number of copies to create for each column in ``df``.

        seed (int): Seed for random generator. Set to a fixed value to keep
        the results persistent. Set to ``None`` to get different results for
        each run of the script.

    Returns:
        df (Pandas DataFrame): The input data, combined with the new data
    '''
    np.random.seed(seed)  # Fixing the seed makes the results persistent
    df_refs = df.copy()

    for col in df.columns:
        # Create a list of random values for all copies of the current column
        randoms = np.random.normal(0, sigma, copies)  # Array of random values
        randoms_int = [int(value) for value in np.round(randoms, 0)]

        for copy in range(copies):
            # Create all copies of the current column
            df_new, df_ref = copy_and_randomize(df, col, randoms_int,
                                                sigma, copy)
            # Combine the existing DataFrame and the last copy
            df = pd.concat([df, df_new], axis=1, sort=False,
                           verify_integrity=True)
            df_refs = pd.concat([df_refs, df_ref], axis=1, sort=False,
                                verify_integrity=True)
            df.sort_index(axis=1, inplace=True)  # Sort the column names
            df_refs.sort_index(axis=1, inplace=True)  # Sort the column names

        if copies == 0:
            # No copies are requested. Here we do not add new copies,
            # but instead overwrite the original profile
            randoms = np.random.normal(0, sigma, 1)  # Array of randoms
            randoms_int = [int(value) for value in np.round(randoms, 0)]
            df_new, df_ref = copy_and_randomize(df, col, randoms_int,
                                                sigma, copy=0)
            df.loc[:, col] = df_new.values
            df_refs.loc[:, col] = df_ref.values

        # Save a histogram plot
        plot_normal_histogram(col, randoms_int, save_folder, set_hist)

    return df, df_refs


def copy_and_randomize(load_curve_houses, house_name, randoms_int, sigma,
                       copy, b_print=False):
    '''Apply the simultaneity effect to a column in a given DataFrame.

    Args:
        load_curve_houses (DataFrame): Time series data to use.

        house_name (string): Name of the column to use.

        randoms_int (List): List of integer values (the random number of
        steps to use for the time shift).

        sigma (float): The standard deviation of the normal distribution from
        which the time shifts are randomly drawn.

        copy (int): Number of the copy we are currently working on.

        b_print (bool): Optionally print status message to the screen.

    Returns:
        df_new (DataFrame): The new data

        df_ref (DataFrame): Reference data (copies without time shift)
    '''
    copy_name = str(house_name) + '_c' + str(copy)
    if b_print and logger.isEnabledFor(logging.DEBUG):
        print('\rCopy (and randomize) house', copy_name, end='\r')  # status
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
        print('\r ', end='\r')  # overwrite last status with empty line
    return df_new, df_ref


def copy_and_randomize_houses(load_curve_houses, houses_dict, cfg):
    '''Create copies of houses where needed. Apply a normal distribution to
    the copies, if a standard deviation ``sigma`` is given in the config.
    This function is an internal part of the load profile aggregator program.

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

    Args:
        load_curve_houses (DataFrame): Time series data to use.

        houses_dict (dict): Dictionary with all the houses

        cfg (dict): The configuration for the load profile aggregator

    Returns:
        load_curve_houses (DataFrame): Manipulatted time series data
    '''
    settings = cfg['settings']
    if settings.get('GLF_on', True) is False:
        return load_curve_houses

    logger.info('Create (randomized) copies of the houses')

    load_curve_houses = load_curve_houses.swaplevel('house', 'class', axis=1)
    load_curve_houses_ref = load_curve_houses.copy()  # Reference (no random)
    # Fix the 'randomness' (every run of the script generates the same results)
    np.random.seed(4)
    randoms_all = []
    sigma_used = False  # Was a sigma > 0 used for any building?

    # Create a temporary dict with all the info needed for randomizer
    randomizer_dict = dict()
    for house_name in settings['houses_list']:
        copies = houses_dict[house_name].get('copies', 0)
        # Get standard deviation (spread or “width”) of the distribution:
        sigma = houses_dict[house_name].get('sigma', False)
        if sigma > 0:
            sigma_used = True

        randomizer_dict[house_name] = dict({'copies': copies,
                                            'sigma': sigma})

    external_profiles = cfg.get('external_profiles', dict())
    for house_name in external_profiles:
        copies = external_profiles[house_name].get('copies', 0)
        # Get standard deviation (spread or “width”) of the distribution:
        sigma = external_profiles[house_name].get('sigma', False)
        if sigma > 0:
            sigma_used = True

        randomizer_dict[house_name] = dict({'copies': copies,
                                            'sigma': sigma})

    # Create copies for every house
    for i, house_name in enumerate(randomizer_dict):
        fraction = (i+1) / len(randomizer_dict)
        if logger.isEnabledFor(logging.INFO):  # print progress
            print('\r{:5.1f}% done'.format(fraction*100), end='\r')

        copies = randomizer_dict[house_name]['copies']
        sigma = randomizer_dict[house_name]['sigma']
        randoms = np.random.normal(0, sigma, copies)  # Array of random values
        randoms_int = [int(value) for value in np.round(randoms, 0)]

        work_list = list(range(0, copies))
        if len(work_list) > 100 and settings.get('run_in_parallel', False):
            randoms_all += randoms_int
            # Use multiprocessing to increase the speed
            f_help = functools.partial(copy_and_randomize,
                                       load_curve_houses, house_name,
                                       randoms_int, sigma)
            return_list = lpagg.misc.multiprocessing_job(f_help, work_list)
            # Merge the existing and new dataframes
            df_list_tuples = return_list.get()

            df_list = [x[0] for x in df_list_tuples]
            df_list_ref = [x[1] for x in df_list_tuples]

            load_curve_houses = pd.concat([load_curve_houses]+df_list,
                                          axis=1, sort=False)
            load_curve_houses_ref = pd.concat([load_curve_houses_ref]
                                              + df_list_ref,
                                              axis=1, sort=False)
            if sigma and logger.isEnabledFor(logging.DEBUG):
                debug_plot_normal_histogram(house_name, randoms_int, cfg)
        elif len(work_list) > 0:
            # Implementation in serial
            randoms_all += randoms_int
            for copy in range(0, copies):
                df_new, df_ref = copy_and_randomize(load_curve_houses,
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
                debug_plot_normal_histogram(house_name, randoms_int, cfg)
        else:
            # The building exists only once. Here we do not add new copies,
            # but instead overwrite the reference profile
            if sigma is not False:
                randoms = np.random.normal(0, sigma, 1)  # Array of randoms
                randoms_int = [int(value) for value in np.round(randoms, 0)]
                randoms_all += randoms_int
                df_new, df_ref = copy_and_randomize(load_curve_houses,
                                                    house_name,
                                                    randoms_int, sigma,
                                                    0, b_print=True)
                load_curve_houses.loc[:, house_name] = df_new.values

    if sigma_used and logger.isEnabledFor(logging.DEBUG):
        debug_plot_normal_histogram('Gebäude gesamt', randoms_all, cfg)

    # Calculate "simultaneity factor" (Gleichzeitigkeitsfaktor)
    calc_GLF(load_curve_houses, load_curve_houses_ref, cfg)

    if cfg['settings'].get('show_plot', False) is True:
        # Plot lineplots of shifted and reference load profiles
        plot_shifted_lineplots(load_curve_houses, load_curve_houses_ref, cfg)

    load_curve_houses = load_curve_houses.swaplevel('house', 'class', axis=1)
    return load_curve_houses


def plot_shifted_lineplots(df_shift, df_ref, cfg):
    """Plot lineplot of sum of shifted and reference load profiles.

    This helps to visualize the quality of the simultaneity shift.

    Args:
        df_shift (DataFrame): Shifted load profiles.

        df_ref (DataFrame): Reference load profiles.

        cfg (dict): The configuration for the load profile aggregator.

    Returns:
        None.

    """
    import matplotlib.dates as mdates

    # Group by energy to sum up all classes and houses
    # and convert from kWh to kW
    hours = cfg['settings']['interpolation_freq'].seconds / 3600.0  # h
    load_shift = df_shift.groupby(level='energy', axis=1).sum() / hours
    load_ref = df_ref.groupby(level='energy', axis=1).sum() / hours

    # Separate the energies into thermal and electrical
    load_th_shift = load_shift[['Q_Heiz_TT', 'Q_TWW_TT']].sum(axis=1)
    load_el_shift = load_shift['W_TT']
    load_th_ref = load_ref[['Q_Heiz_TT', 'Q_TWW_TT']].sum(axis=1)
    load_el_ref = load_ref['W_TT']

    # Create two plot figures for thermal and electrical
    for load_shift, load_ref, ylabel in zip(
            [load_th_shift, load_el_shift],
            [load_th_ref, load_el_ref],
            ['thermische Leistung in [kW]', 'elektrische Leistung in [kW]']
            ):

        fig = plt.figure()
        ax = fig.gca()
        plt.plot(load_shift, label='Shift')
        plt.plot(load_ref, '--', label='Referenz')
        plt.axhline(load_shift.max(), linestyle='-.',
                    label='max(Shift)', color='#e8d654')
        plt.axhline(load_ref.max(), linestyle='-.',
                    label='max(Referenz)', color='#5eccf3')
        plt.legend(loc='lower center', ncol=5, bbox_to_anchor=(0.5, 1.0))
        plt.ylabel(ylabel)
        ax.yaxis.grid(True)  # Activate grid on horizontal axis
        ax.xaxis.set_tick_params(rotation=30)  # rotation is useful sometimes
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
        plt.tight_layout()  # Fit plot within figure cleanly
        plt.show(block=False)  # Show plot without blocking the script


def debug_plot_normal_histogram(house_name, randoms_int, cfg):
    '''Wrapper around ``plot_normal_histogram()`` used by the load profile
    aggregator program.
    '''
    settings = cfg['settings']
    save_folder = cfg['print_folder']
    plot_normal_histogram(house_name, randoms_int, save_folder,
                          set_hist=dict({'PNG': True, 'PDF': True}))

    if settings.get('show_plot', False) is True:
        plt.show(block=False)  # Show plot without blocking the script


def plot_normal_histogram(house_name, randoms_int, save_folder=None,
                          set_hist=dict()):
    '''Save a histogram of the values in ``randoms_int`` to a .png file.
    '''
    logger.debug('Interval shifts applied to ' + str(house_name) + ':')
    logger.debug(randoms_int)
    mu = np.mean(randoms_int)
    sigma = np.std(randoms_int, ddof=1)
    text_mean_std = 'Mean = {:0.2f}, std = {:0.2f}'.format(mu, sigma)
    title_mu_std = r'$\mu={:0.3f},\ \sigma={:0.3f}$'.format(mu, sigma)
    logger.debug(text_mean_std)

    # Create a histogram of the data
    limit = max(-1*min(randoms_int), max(randoms_int))
    bins = range(-limit, limit+2)
    default_font = plt.rcParams.get('font.size')
    plt.rcParams.update({'font.size': 16})
    fig = plt.figure()
    ax = fig.gca()
    ax.yaxis.grid(True)  # Activate grid on horizontal axis
    n, bins, patches = plt.hist(randoms_int, bins, align='left',
                                rwidth=0.9)
    plt.title(str(len(randoms_int))+' '+str(house_name)+' ('+title_mu_std+')')
    plt.xlabel('Zeitschritte')
    plt.ylabel('Häufigkeit')

    if save_folder is not None:
        # Make sure the save path exists
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)

        if set_hist.get('PNG', False):
            plt.savefig(os.path.join(save_folder,
                                     'histogram_'+str(house_name)+'.png'),
                        bbox_inches='tight', dpi=200)
        if set_hist.get('PDF', False):
            plt.savefig(os.path.join(save_folder,
                                     'histogram_'+str(house_name)+'.pdf'),
                        bbox_inches='tight')

    # Reset settings
    plt.rcParams.update({'font.size': default_font})
    return ax


def calc_GLF(load_curve_houses, load_curve_houses_ref, cfg):
    '''Calculate "simultaneity factor" (Gleichzeitigkeitsfaktor)
    Uses a DataFrame with and one without randomness.
    '''
    settings = cfg['settings']

    # Some GHD energies may be split into different columns
    load_curve_houses.sort_index(axis=1, inplace=True)

    load_curve_houses_ran = load_curve_houses.copy()
    try:
        load_ran = load_curve_houses_ran.groupby(level='energy', axis=1).sum()
        load_ref = load_curve_houses_ref.groupby(level='energy', axis=1).sum()
    except KeyError:
        # TODO: This error appeared first in pandas 0.25.0
        # Swapping the levels before grouping seems to fix it, though.
        load_curve_houses_ran = load_curve_houses_ran.swaplevel(
                i='energy', j='house', axis=1)
        load_curve_houses_ref = load_curve_houses_ref.swaplevel(
                i='energy', j='house', axis=1)
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

    for col in ['th_RH', 'th_TWE', 'th', 'el']:
        try:
            sf_df.loc['GLF', col] = (sf_df.loc['P_max_kW', col]
                                     / sf_df.loc['P_max_ref_kW', col])
        except ZeroDivisionError:
            sf_df.loc['GLF', col] = float('NaN')

    # Calculate a reference simultaneity factor with DIN 4708
    homes_count = 0
    buildings_count = 0
    for house in cfg['houses']:
        homes = cfg['houses'][house].get('N_WE', 0)
        buildings = (1 + cfg['houses'][house].get('copies', 0))
        buildings_count += buildings
        homes *= buildings
        homes_count += homes

    GLF_DIN4708 = lpagg.DIN4708.calc_GLF(homes_count)
    DIN4708_df = pd.Series(data=[buildings_count, homes_count, GLF_DIN4708],
                           index=['Buildings', 'Homes', 'GLF'])
    logger.info('Reference simultaneity factor from DIN 4708 for {:d} homes '
                'in {:d} buildings is {:.2f}%.'
                .format(homes_count, buildings_count, GLF_DIN4708*100))

    if logger.isEnabledFor(logging.INFO):
        logger.info('Simultaneity factors (Gleichzeitigkeitsfaktoren):')
        print(sf_df)
    # Make sure the save path exists and save the DataFrame
    save_folder = cfg['print_folder']
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    writer = pd.ExcelWriter(os.path.join(save_folder, 'GLF.xlsx'))
    sf_df.to_excel(writer, sheet_name='GLF')
    DIN4708_df.to_excel(writer, sheet_name='DIN4708')
    writer.save()  # Save the actual Excel file

    return None


if __name__ == '__main__':
    '''This code is executed when the script is started
    '''
    try:  # Wrap everything in a try-except to show exceptions with the logger
        main()
    except Exception as e:
        logger.exception(e)
        input('\nPress the enter key to exit.')  # Prevent console from closing
