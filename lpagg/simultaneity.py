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

"""

import os
import logging
import pandas as pd
import numpy as np
import scipy.stats
import matplotlib.pyplot as plt  # Plotting library
import matplotlib as mpl

# Import local modules from load profile aggregator project
import lpagg.misc
import lpagg.DIN4708

# Define the logging function
logger = logging.getLogger(__name__)


def main():
    """Read user input and run the script."""
    setup()  # Set some basic settings

    args = run_OptionParser()  # Read user options

    run(args.sigma, args.copies, args.seed, args.file,
        set_hist=dict(PNG=True, PDF=False))


def run(sigma, copies, seed, file, set_hist=dict(), show_plot=False):
    """Perform task and produce the output."""
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
    """Set up the logger."""
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
    """Define and run the argument parser. Return the collected arguments."""
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


def create_simultaneity(df, sigma, copies, seed, save_folder=None,
                        set_hist=dict()):
    """Apply the simultaneity effect to the columns in a given DataFrame.

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
    """
    np.random.seed(seed)  # Fixing the seed makes the results persistent

    shift_list = []
    for col in df.columns:
        # Create a list of random values for all copies of the current column
        randoms = np.random.normal(0, sigma, copies)  # Array of random values
        randoms_int = [int(value) for value in np.round(randoms, 0)]
        shift_list += randoms_int

        # Save a histogram plot
        plot_normal_histogram(col, randoms_int, save_folder, set_hist)

    # Each column gets same number of copies
    copies_list = [copies] * len(df.columns)
    # Create the copied columns
    df = copy_df_columns(df, copies_list)
    # Store a reference that will not be randomized
    df_refs = df.copy()
    # Shift all the columns
    df = shift_columns(df, shift_list)

    return df, df_refs


def copy_df_columns(df, copies_list, level=None):
    """Copy each column in df the number of times given in copies_list.

    If name of a multiindex level is given, the resulting groups are copied.
    Make sure the order in copies_list actually fits the DataFrame columns.
    """
    if (pd.Series(copies_list) <= 1).all():
        return df  # If all columns need one copy, just return the original

    if level is None:
        columns = df.columns
    else:
        columns = df.columns.unique(level=level)

    df_new = pd.DataFrame()
    for col, copies in zip(columns, copies_list):
        for copy in range(0, copies):
            copy_name = str(col) + '_c{0:0{width}}'.format(
                copy, width=len(str(copies)))

            if level is None:
                df_tmp = df[[col]].rename(columns={col: copy_name})
            else:
                df_tmp = pd.concat([df[col]], keys=[copy_name],
                                   names=[level], axis=1)
            df_new = pd.concat([df_new, df_tmp], axis='columns')
    return df_new


def shift_columns(df, shift_list, level=None, sort_shifts=True):
    """Shift each column in df by the value in shift_list.

    Elements that are shifted beyond the end are added at the beginning.

    df (DataFrame): The DataFrame with columns to shift along the index axis.

    shift_list (list): A list of integers defining the number of rows each
    column in df is shifted. Make sure the order in shift_list actually fits
    the DataFrame columns.

    level (str): If name of a multiindex level is given, the unique columns
    in that level are treated as groups with one shift

    sort_shifts (bool): If True, all columns with equal shift value are
    shifted together, which increases the speed significantly.
    """
    if level is None:
        columns = df.columns
    else:
        columns = df.columns.unique(level=level)

    if sort_shifts:
        df_shifts = pd.DataFrame(data=zip(columns, shift_list),
                                 columns=['name', 'shift'])
        for shift in df_shifts['shift'].unique():
            if shift == 0:
                continue
            # For each unique shift, get all columns and shift them together
            shift_cols = df_shifts[df_shifts['shift'] == shift]['name']
            df.loc[:, pd.IndexSlice[shift_cols]] = np.roll(
                df.loc[:, pd.IndexSlice[shift_cols]].to_numpy(),
                shift, axis=0)
    else:  # Should be slower in most cases, but may use less memory
        for col, shift in zip(columns, shift_list):
            if shift != 0:
                df[col] = np.roll(df[col].to_numpy(), shift, axis=0)

    return df


def copy_and_randomize(load_curve_houses, house_name, randoms_int, sigma,
                       copy, b_print=False, keep_reference=True):
    """Apply the simultaneity effect to a column in a given DataFrame.

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
    """
    copy_name = str(house_name) + '_c' + str(copy)
    if b_print and logger.isEnabledFor(logging.DEBUG):
        print('\r Copy (and randomize) house', copy_name, '   ', end='\r')
    # Select the data of the house we want to copy
    df_new = load_curve_houses[house_name]
    # Rename the multiindex with the name of the copy
    df_new = pd.concat([df_new], keys=[copy_name], names=['house'], axis=1)

    if keep_reference:
        df_ref = df_new.copy()
    else:  # This saves on used memory, but both returns will be the same
        df_ref = df_new

    if sigma:  # Optional: Shift the rows
        shift_step = randoms_int[copy]
        df_new = pd.DataFrame(
            index=df_new.index,
            columns=df_new.columns,
            data=np.roll(df_new.to_numpy(), shift_step, axis=0))

    if b_print:
        # overwrite last status with empty line
        print('\r', end='\r')
    return df_new, df_ref


def copy_and_randomize_houses(load_curve_houses, houses_dict, cfg):
    """Create copies of houses and randomize load where needed.

    Apply a normal distribution to the houses, if a standard deviation
    ``sigma`` is given in the config. This function is an internal part
    of the load profile aggregator program.

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
    """
    settings = cfg['settings']
    if settings.get('GLF_on', True) is False:
        return load_curve_houses

    logger.info('Create (randomized) copies of the houses')

    memory = load_curve_houses.memory_usage().sum() / 10**6  # MB
    logger.debug('Memory usage: {} MB'.format(memory))

    # Code requires house as the top column level
    if 'class' in load_curve_houses.columns.names:
        load_curve_houses = load_curve_houses.swaplevel('house', 'class',
                                                        axis=1)

    # Fix the 'randomness' (every run of the script generates the same results)
    np.random.seed(4)
    sigma_used = False  # Take note if a sigma > 0 used for any building

    # Create a temporary dict with all the info needed for randomizer
    randomizer_dict = dict()
    for house_name in houses_dict.keys():
        copies = houses_dict[house_name].get('copies', 1)
        # Get standard deviation (spread or “width”) of the distribution:
        sigma = houses_dict[house_name].get('sigma', False)
        randomizer_dict[house_name] = dict({'copies': copies, 'sigma': sigma})
        if sigma > 0:
            sigma_used = True

    external_profiles = cfg.get('external_profiles', dict())
    for house_name in external_profiles:
        copies = external_profiles[house_name].get('copies', 1)
        # Get standard deviation (spread or “width”) of the distribution:
        sigma = external_profiles[house_name].get('sigma', False)
        randomizer_dict[house_name] = dict({'copies': copies, 'sigma': sigma})
        if sigma > 0:
            sigma_used = True

    # Draw random numbers for each house. Needs to use the same order as
    # the columns in load_curve_houses
    shift_list = []  # Is ordered like load_curve_houses.columns
    copies_list = []  # Is ordered like load_curve_houses.columns
    houses = load_curve_houses.columns.unique(level='house')
    for house_name in houses:
        copies = randomizer_dict[house_name]['copies']
        sigma = randomizer_dict[house_name]['sigma']
        randoms = np.random.normal(0, sigma, copies)  # Array of random values
        randoms_int = [int(value) for value in np.round(randoms, 0)]
        randomizer_dict[house_name]['shifts'] = randoms_int
        shift_list += randoms_int
        copies_list.append(copies)

    # Create copies for every house
    logger.debug('Copy houses')
    load_curve_houses = copy_df_columns(
        load_curve_houses, copies_list, level='house')

    if sigma_used:  # This part is only required if sigma is actually used
        # Store a reference that will not be randomized
        load_curve_houses_ref = load_curve_houses.copy()
        # Randomize all the houses
        logger.debug('Randomize houses')
        shift_columns(load_curve_houses, shift_list, level='house',
                      sort_shifts=settings.get('sort_shifts', True))
        # Calculate "simultaneity factor" (Gleichzeitigkeitsfaktor)
        calc_GLF(load_curve_houses, load_curve_houses_ref, cfg)

        # Perform some plotting
        if (cfg['settings'].get('show_plot', False)
           or cfg['settings'].get('save_plot_filetypes', None)):

            # Plot histograms of all buildings and each building individually
            for house_name in houses:
                randoms_int = randomizer_dict[house_name]['shifts']
                if len(randoms_int) > 1:  # Only when using multiple copies
                    debug_plot_normal_histogram(house_name, randoms_int, cfg)

            language = cfg.get('settings', {}).get('language', 'de')
            if language == 'en':
                txt_title = 'buildings'
            else:
                txt_title = 'Gebäude gesamt'
            debug_plot_normal_histogram(txt_title, shift_list, cfg)

            # Plot lineplots of shifted and reference load profiles
            plot_shifted_lineplots(load_curve_houses, load_curve_houses_ref,
                                   cfg)

    # Prepare result for return
    if 'class' in load_curve_houses.columns.names:
        load_curve_houses = load_curve_houses.swaplevel('house', 'class',
                                                        axis=1)
    # Store the shift list for further usage
    cfg['settings']['shift_list'] = shift_list
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
    language = cfg['settings'].get('language', 'de')
    if language == 'en':
        txt_P_th = 'thermal power in [kW]'
        txt_P_el = 'electrical power in [kW]'
        txt_shift = 'shift'
        txt_ref = 'reference'
        txt_shift_max = 'max(shift)'
        txt_ref_max = 'max(reference)'
    else:
        txt_P_th = 'thermische Leistung in [kW]'
        txt_P_el = 'elektrische Leistung in [kW]'
        txt_shift = 'Shift'
        txt_ref = 'Referenz'
        txt_shift_max = 'max(Shift)'
        txt_ref_max = 'max(Referenz)'

    # Group by energy to sum up all classes and houses
    # and convert from kWh to kW
    hours = cfg['settings']['interpolation_freq'].seconds / 3600.0  # h
    load_shift = df_shift.T.groupby(level='energy').sum().T / hours
    load_ref = df_ref.T.groupby(level='energy').sum().T / hours

    # Separate the energies into thermal and electrical
    cols_th = ['Q_Heiz_TT', 'Q_TWW_TT']
    cols_el = ['W_TT']
    cols_th = [c for c in cols_th if c in load_shift.columns]
    cols_el = [c for c in cols_el if c in load_shift.columns]
    load_th_shift = load_shift[cols_th].sum(min_count=1, axis=1)
    load_el_shift = load_shift[cols_el].sum(min_count=1, axis=1)
    load_th_ref = load_ref[cols_th].sum(min_count=1, axis=1)
    load_el_ref = load_ref[cols_el].sum(min_count=1, axis=1)

    # Create two plot figures for thermal and electrical
    for load_shift, load_ref, ylabel in zip(
            [load_th_shift, load_el_shift],
            [load_th_ref, load_el_ref],
            [txt_P_th, txt_P_el]
            ):
        if load_shift.dropna().empty:
            continue  # skip emtpy plots

        fig = plt.figure(figsize=[10, 5])
        ax = fig.gca()
        load_ref.shift(periods=-1, freq="infer").plot(
            ax=ax, label=txt_ref, style='--', color='tab:red')
        load_shift.shift(periods=-1, freq="infer").plot(
            ax=ax, label=txt_shift, color='tab:blue')
        plt.axhline(load_ref.max(), linestyle='-.',
                    label=txt_ref_max, color='tab:orange')
        plt.axhline(load_shift.max(), linestyle='-.',
                    label=txt_shift_max, color='tab:cyan')
        plt.legend(loc='lower center', ncol=5, bbox_to_anchor=(0.5, 1.0))
        plt.ylabel(ylabel)
        plt.xlabel("")
        ax.yaxis.grid(True)  # Activate grid on horizontal axis
        plt.tight_layout()  # Fit plot within figure cleanly

        # Save plot
        lpagg.misc.savefig_filetypes(
            cfg['print_folder'],
            filename=ylabel + ' (shift)',
            filetypes=cfg['settings'].get('save_plot_filetypes', None),
            dpi=400)

        if cfg['settings'].get('show_plot', False) is True:
            plt.show(block=False)  # Show plot without blocking the script
        else:
            plt.close()

        # Save raw data
        df_excel = pd.concat([load_shift, load_ref], axis='columns',
                             keys=[txt_shift, txt_ref])
        df_excel.to_excel(os.path.join(cfg['print_folder'],
                                       ylabel + ' (shift)'+'.xlsx'))


def debug_plot_normal_histogram(house_name, randoms_int, cfg):
    """Plot a histogram for debugging purposes.

    Wrapper around ``plot_normal_histogram()`` used by the load profile
    aggregator program.
    """
    settings = cfg['settings']
    save_folder = cfg['print_folder']
    plot_normal_histogram(house_name, randoms_int, save_folder, cfg=cfg,
                          filetypes=['png', 'svg', 'pdf'])

    if settings.get('show_plot', False) is True:
        plt.show(block=False)  # Show plot without blocking the script
    else:
        plt.close()


def plot_normal_histogram(house_name, randoms_int, save_folder=None,
                          filetypes=None, cfg=dict()):
    """Save a histogram of the values in ``randoms_int`` to a .png file."""
    language = cfg.get('settings', {}).get('language', 'de')
    if language == 'en':
        txt_xlabel = 'shifted time steps'
        txt_ylabel = 'frequency'
        txt_label_norm = 'normal distribution'
        txt_label_data = 'buildings'
    else:
        txt_xlabel = 'Verschobene Zeitschritte'
        txt_ylabel = 'Häufigkeit'
        txt_label_norm = 'Normalverteilung'
        txt_label_data = 'Gebäude'

    logger.debug('Interval shifts applied to ' + str(house_name) + ':')
    logger.debug(randoms_int)
    mu = np.mean(randoms_int)
    sigma = np.std(randoms_int, ddof=1)
    text_mean_std = 'Mean = {:0.2f}, std = {:0.2f}'.format(mu, sigma)
    title_mu_std = r'$\mu={:0.3f},\ \sigma={:0.3f}$'.format(mu, sigma)
    logger.debug(text_mean_std)

    # Prepare a histogram of the data
    limit = max(-1*min(randoms_int), max(randoms_int))
    bins = np.arange(-limit-0.5, limit+1.5)

    # Create a histogram of the data
    default_font = plt.rcParams.get('font.size')
    plt.rcParams.update({'font.size': 16})
    fig = plt.figure()
    ax = fig.gca()
    ax.yaxis.grid(True)  # Activate grid on horizontal axis
    n, bins, patches = plt.hist(randoms_int, bins, align='mid',
                                rwidth=0.9, label=txt_label_data)
    if cfg['settings'].get('GLF_stats_include_normal', True):
        # Create an ideal normal distribution for reference
        x_norm = np.linspace(min(bins), max(bins), 100)
        y_norm = scipy.stats.norm.pdf(x_norm, mu, sigma)
        y_norm_scaled = y_norm/max(y_norm)*max(n)
        ax.plot(x_norm, y_norm_scaled, label=txt_label_norm)
    plt.title(str(len(randoms_int))+' '+str(house_name)+' ('+title_mu_std+')')
    plt.xlabel(txt_xlabel)
    plt.ylabel(txt_ylabel)
    if cfg['settings'].get('GLF_stats_include_normal', True):
        plt.legend()

    lpagg.misc.savefig_filetypes(
        save_folder,
        filename='Histogram '+str(house_name),
        filetypes=cfg['settings'].get('save_plot_filetypes', None))

    # Reset settings
    plt.rcParams.update({'font.size': default_font})
    return ax


def calc_GLF(load_curve_houses, load_curve_houses_ref, cfg):
    """Calculate "simultaneity factor" (Gleichzeitigkeitsfaktor).

    Uses a DataFrame with and one without randomness.
    """
    settings = cfg['settings']

    # Some GHD energies may be split into different columns
    load_curve_houses.sort_index(axis=1, inplace=True)

    try:
        load_ran = load_curve_houses.T.groupby(level='energy').sum().T
        load_ref = load_curve_houses_ref.T.groupby(level='energy').sum().T
    except KeyError:
        # This error appeared first in pandas 0.25.0. Swapping the levels
        # before grouping fixed it. In pandas >= 2.0 the issue is solved,
        # but the workaroud is kept for backwards compatibility
        load_curve_houses_ran = load_curve_houses.copy()
        load_curve_houses_ran = load_curve_houses_ran.swaplevel(
                i='energy', j='house', axis=1)
        load_curve_houses_ref = load_curve_houses_ref.swaplevel(
                i='energy', j='house', axis=1)
        load_ran = load_curve_houses_ran.T.groupby(level='energy').sum().T
        load_ref = load_curve_houses_ref.T.groupby(level='energy').sum().T

    hours = settings['interpolation_freq'].seconds / 3600.0        # h

    # Calculate sum of thermal powers (heating and hot water)
    th_sum_cols = ['Q_Heiz_TT', 'Q_TWW_TT']
    th_sum_cols = [c for c in th_sum_cols if c in load_ran.columns]
    load_ran['th'] = load_ran[th_sum_cols].sum(axis='columns')
    load_ref['th'] = load_ref[th_sum_cols].sum(axis='columns')

    # P_max_kW: Calculate maximum power of the randomized load profiles
    # P_max_ref_kW: Calculate reference power, without artificial
    # simultaneity introduced by time shift. However, some simultaneity
    # effects might still be included if different original load profiles
    # were used, with power peaks at different times of the year
    sf_df = pd.concat([(load_ran.max() / hours),
                       (load_ref.max() / hours)],
                      axis='columns',
                      keys=['P_max_kW', 'P_max_ref_kW']
                      ).T

    sf_df = sf_df.rename(columns={'Q_Heiz_TT': 'th_RH', 'Q_TWW_TT': 'th_TWE'})

    # Calculate a simultaneity factor that shows the impact of time shift
    sf_df.loc['GLF_timeshift'] = \
        sf_df.loc['P_max_kW'].div(sf_df.loc['P_max_ref_kW'])

    # Calculate a 'static' refrence power value. This is just the sum of
    # the maximum power for each house, regardless of shape of profile
    if 'Q_Heiz_TT' in load_ran.columns:
        sf_df.loc['P_max_ref_static_kW', 'th_RH'] = (
            load_curve_houses_ref.xs('Q_Heiz_TT', level='energy', axis=1)
            .max().sum() / hours)
    if 'Q_TWW_TT' in load_ran.columns:
        sf_df.loc['P_max_ref_static_kW', 'th_TWE'] = (
            load_curve_houses_ref.xs('Q_TWW_TT', level='energy', axis=1)
            .max().sum() / hours)
    if 'Q_Heiz_TT' in load_ran.columns and 'Q_TWW_TT' in load_ran.columns:
        sf_df.loc['P_max_ref_static_kW', 'th'] = (
            (load_curve_houses_ref.xs('Q_Heiz_TT', level='energy', axis=1)
             + load_curve_houses_ref.xs('Q_TWW_TT', level='energy', axis=1))
            .max().sum() / hours)
    if 'W_TT' in load_ran.columns:
        sf_df.loc['P_max_ref_static_kW', 'el'] = (
            load_curve_houses_ref.xs('W_TT', level='energy', axis=1)
            .max().sum() / hours)
    if 'Q_Kalt_TT' in load_ran.columns:
        sf_df.loc['P_max_ref_static_kW', 'Q_Kalt_TT'] = (
            load_curve_houses_ref.xs('Q_Kalt_TT', level='energy', axis=1)
            .max().sum() / hours)
    # Calculate overall simultaneity factor
    sf_df.loc['GLF'] = sf_df.loc['P_max_kW'].div(
        sf_df.loc['P_max_ref_static_kW'])

    sf_df_wo_comment = sf_df.copy()  # Save a copy without comments for print

    sf_df['Comment'] = [
        "The maximum load in the resulting profiles at one point in time, "
        "after application of time shift (due to definition of standard "
        "deviation 'sigma')",
        "The maximum load in the input profiles at one point in time "
        "(without time shift)",
        "GLF_timeshift = P_max_kW / P_max_ref_kW (this simultaneity factor "
        "is useful for measuring the effect of the time shift)",
        "The 'static' sum of the maxima of each individual input profile, "
        "regardless of time",
        "GLF = P_max_kW / P_max_ref_static_kW (this is the relevant "
        "overall simultaneity factor)",
        ]

    # Calculate a reference simultaneity factor with DIN 4708
    homes_count = 0
    buildings_count = 0
    for house in cfg['houses']:
        homes = cfg['houses'][house].get('N_WE', 0)
        if pd.isna(homes):
            homes = 0
        buildings = cfg['houses'][house].get('copies', 1)
        buildings_count += buildings
        homes *= buildings
        homes_count += homes

    GLF_DIN4708 = lpagg.DIN4708.calc_GLF(homes_count)
    DIN4708_df = pd.Series(data=[buildings_count, homes_count, GLF_DIN4708],
                           index=['Buildings', 'Homes', 'GLF'])
    logger.info('Reference simultaneity factor from DIN 4708 for {:.0f} homes'
                ' in {:.0f} buildings is {:.2f}%.'
                .format(homes_count, buildings_count, GLF_DIN4708*100))

    if logger.isEnabledFor(logging.INFO):
        logger.info('Simultaneity factors (Gleichzeitigkeitsfaktoren):')
        print(sf_df_wo_comment)

    if settings.get('print_GLF_stats', False):
        # Make sure the save path exists and save the DataFrame
        save_folder = cfg['print_folder']
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
        with pd.ExcelWriter(os.path.join(save_folder, 'GLF.xlsx')) as writer:
            sf_df.to_excel(writer, sheet_name='GLF')
            DIN4708_df.to_excel(writer, sheet_name='DIN4708')

    # Store the table with the simultaneity factor results in the cfg dict
    cfg['simultaneity_factor_results'] = sf_df.to_dict(orient='index')

    return None


def run_simul_lpagg_post(cfg):
    """Run simultaneity in postprocessing to LPagg."""
    csv_file = os.path.join(
        cfg['print_folder'],
        os.path.splitext(cfg['settings']['print_file'])[0] + '_houses.dat')

    load_curve_houses = pd.read_csv(csv_file,
                                    low_memory=False,
                                    header=[0, 1],
                                    index_col=0)

    df, df_ref = create_simultaneity(load_curve_houses,
                                     sigma=8,
                                     copies=1,
                                     seed=4,
                                     save_folder=cfg['print_folder'],
                                     set_hist=dict(PNG=True, PDF=False))

    logger.info('Printing *_houses_GLF.dat file')
    df.to_csv(os.path.join(cfg['print_folder'],
                           os.path.splitext(cfg['settings']['print_file'])[0]
                           + '_houses.dat'))
    return df


if __name__ == '__main__':
    """This code is executed when the script is started."""
    try:  # Wrap everything in a try-except to show exceptions with the logger
        main()
    except Exception as e:
        logger.exception(e)
        input('\nPress the enter key to exit.')  # Prevent console from closing
