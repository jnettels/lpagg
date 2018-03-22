#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
Script for interpolating "IGS Referenzklimaregion" files to new time steps
Can also deal with weather data from Deutscher Wetterdienst (DWD) in form
TRY2010 (data in 15 climate regions) and TRY2015 (released in 2017)

Important definition by Markus Peter for TRNSYS time stamps:

    "Column value is a mean value related to the time interval delta t
    ending at the time corresponding to actual weather_data line."

This script uses regular expressions. For help, see:
https://regex101.com/#python
'''

import pandas as pd               # Pandas
import numpy as np                # Numpy
import os                         # Operaing System
import matplotlib.pyplot as plt   # Plotting library
import calendar                   # Calendar
import re                         # Regular expressions
import requests
import json

# Global Pandas option for displaying terminal output
pd.set_option('display.max_columns', 0)


''' Script options
'''
datetime_start = pd.datetime(2014, 1, 1, 00, 00, 00)
#datetime_end = pd.datetime(2017, 2, 1, 00, 00, 00)
datetime_end = pd.datetime(2015, 1, 1, 00, 00, 00)
interpolation_freq = pd.Timedelta('15 minutes')
#interpolation_freq = pd.Timedelta('1 hours')

bool_print_header = True
#bool_print_header = False
bool_print_type99_head = True
#bool_print_type99_head = False
bool_print_index = True
#bool_print_index = False
remove_leapyear = False

'''
base_folder = u'V:\\MA\\2_Projekte\\SIZ10015_futureSuN\\4_Bearbeitung\\AP4_Transformation\\AP404_Konzepte für zukünftige Systemlösungen\\Lastprofile\\VDI 4655\\Berechnung\\Wetter'
base_folder = u'C:\\Trnsys17\\Work\\futureSuN\\SB\Weather\\TRY_38265002816500'
base_folder = u'C:\\Trnsys17\\Work\\futureSuN\\HK\Weather\\TRY2010_03'
base_folder = r'V:\MA\2_Projekte\SIZ10015_futureSuN\4_Bearbeitung\AP4_Transformation\AP404_Konzepte für zukünftige Systemlösungen\Wetterdaten\DWD TRY 2011\Daten\TRY-Daten'
base_folder = r'V:\MA\2_Projekte\SIZ10002_Hydraulik\5_Archiv\Backups Mani\Dissertation\3_Trnsys Modell\3_Wetterdaten\TRY Ortsgenau DWD\TRY_42255002859500'
'''
base_folder = r'C:\Users\nettelstroth\Downloads\Wetter'

'''
Ein Datentyp für die Wetterdaten muss definiert werden, damit die Werte
im korrekten Formate eingelesen werden können. Unterschieden werden
Daten im Format für TRNSYS (=IGS) und Daten vom DWD.
Über die regex unten kann der Typ auch im Pfad gesucht werden und muss dann
in der Gruppe 'ztype' gespeichert werden.
'''
#weather_data_type = 'TRNSYS'
#weather_data_type = 'IGS'
weather_data_type = 'DWD'

weather_file_list = [
#    'IGS_Referenzklimaregion_05.dat',
#    'IGS_Referenzklimaregion_06.dat',
#    'TRY2015_38265002816500_Jahr.dat',
#    'TRY2015_38265002816500_Somm.dat',
#    'TRY2015_38265002816500_Wint.dat',
#    'TRY2045_38265002816500_Jahr.dat',
#    'TRY2045_38265002816500_Somm.dat',
#    'TRY2045_38265002816500_Wint.dat',
    'TRY2010_03_Jahr.dat',
                    ]

# Joing base_folder and weather files:
weather_file_list = [os.path.join(base_folder, f) for f in weather_file_list]

bool_regex = False
if bool_regex:
    # folder_regex = 'TRY_2016_(?P<region>.+)\\TRY(?P<year>.+)_\d{14}_Jahr\.dat'
    # folder_regex = '.+(?P<aname>TRY_2016)_(?P<bregion>\d{2})\\\\TRY2015_\d{14}_Jahr\.dat'
    # folder_regex = '.+TRY_2016_(?P<bregion>\d{2})\\\\(?P<aname>TRY2015)_(?P<ccoor>\d{14})_(?P<dtyp>Jahr)\.dat'
    # folder_regex = '.+(?P<aname>TRY_2016)_(?P<bregion>01)\\\\TRY2015_\d{14}_Jahr\.dat'
    # folder_regex = '.+TRY_2016_01\\\\TRY2015_\d{14}_Jahr\.dat'
    folder_regex = '.+(?P<ztype>DWD|TRNSYS|IGS).+(?P<byear>2010|2015|Referenzklimaregion)_(01|39095002965500)(_Jahr|)\.dat'
    # folder_regex = '.+(?P<ztype>DWD|TRNSYS|IGS).+(?P<byear>2010|2015|Referenzklimaregion)_(03|40005002975500)(_Jahr|)\.dat'
    #folder_regex = '.+(?P<ztype>DWD|TRNSYS|IGS).+(?P<byear>2010|x2015|xReferenzklimaregion)_(07|39625002724500)(_Jahr|)\.dat'

    weather_file_list = []
    matchlist = []
    for root, dirs, files in os.walk(base_folder):
        regex_compiled = re.compile(r''+folder_regex)
        for file in files:
            path = os.path.join(root, file)
            regex_match = re.match(regex_compiled, path)

            if regex_match:
                # print path
                matchlist.append(regex_match.groupdict())
                weather_file_list.append(path)
            else:
                print('Folder did not match the regex, skipping...', root)
                continue

    if weather_file_list == []:
        print('Error! The regular expression did not match '
              'any folder within the top folder: ')
        print('      regex: '+folder_regex)
        print(' top folder: '+base_folder+'\n')
        exit()

    # Then sort the both matchlist and weather_file_list by weather_file_list
    weather_file_list, matchlist = (list(x) for x in
                                    zip(*sorted(zip(weather_file_list,
                                                    matchlist),
                                                key=lambda pair: pair[0])))

#    print(matchlist)
#    print(weather_file_list)


def read_IGS_weather_file(weather_file_path):
    '''Read and interpolate "IGS Referenzklimaregion" files
    '''

    # Read the file and store it in a DataFrame
    weather_data = pd.read_csv(
        open(weather_file_path, 'r'),
        delim_whitespace=True,
        names=['HOUR', 'IBEAM_H', 'IDIFF_H', 'TAMB', 'WSPEED',
               'RHUM', 'WDIR', 'CCOVER', 'PAMB'],
        comment='<',
        )

    # Plausibility check:
    if weather_data.isnull().values.any():
        print(weather_data)
        print('Error: There are "NaN" values in your weather data. '
              'Is the type "IGS" correct? Exiting...')
        print('File is: '+weather_file_path)
        exit()
    return weather_data


def read_DWD_weather_file(weather_file_path):
    '''Read and interpolate "DWD Testreferenzjahr" files
    '''
    # The comments in DWD files before the header are not commented out.
    # Thus we have to search for the line with the header information:
    header_row = None
    with open(weather_file_path, 'r') as rows:
        for number, row in enumerate(rows, 1):
            # The header is the row before the appearance of '***'
            if '***' in row:
                header_row = number - 1
                break

    # Plausibility check:
    if header_row is None:
        print('Error: Header row not found in weather file. '
              'Is the data type "DWD" correct? Exiting...')
        print('File is: ' + weather_file_path)
        exit()

    # Read the file and store it in a DataFrame
    weather_data = pd.read_csv(
        open(weather_file_path, 'r'),
        delim_whitespace=True,
        skiprows=header_row-1,
        index_col=['MM', 'DD', 'HH'],
        usecols=['MM', 'DD', 'HH', 'B', 'D', 't', 'WG', 'RF', 'WR', 'N', 'p'],
        comment='*',
        )

    # Rename the columns to the TRNSYS standard:
    weather_data.rename(columns={'B': 'IBEAM_H',
                                 'D': 'IDIFF_H',
                                 't': 'TAMB',
                                 'WG': 'WSPEED',
                                 'RF': 'RHUM',
                                 'WR': 'WDIR',
                                 'N': 'CCOVER',
                                 'p': 'PAMB'},
                        inplace=True)

    # Add an 'HOUR' column:
    weather_data['HOUR'] = range(1, 8761)

    # Make sure all columns are in the correct order
    weather_data = weather_data.reindex(columns=['HOUR', 'IBEAM_H', 'IDIFF_H',
                                                 'TAMB', 'WSPEED', 'RHUM',
                                                 'WDIR', 'CCOVER', 'PAMB'])

    # print weather_data
    return weather_data


def get_TRNSYS_coordinates(weather_file_path):
    '''Attempts to get the coordinates required for the TRNSYS Type99 input.

    If values for 'Rechtswert' and 'Hochwert' are found (in DWD 2016 files),
    they are converted from the "Lambert conformal conic projection" to the
    TRNSYS format:
        - decimal longitude and latitude
        - 'east of greenwich' is defined 'negative'
    by using the website http://epsg.io.

    Args:
        weather_file_path (str): path to a weather file

    Returns:
        TRNcoords (dict): Dictionary with longitude and latitude
    '''
    with open(weather_file_path, 'r') as weather_file:
        regex = r'Rechtswert\s*:\s(?P<x>\d*).*\nHochwert\s*:\s(?P<y>\d*)'
        match = re.search(regex, weather_file.read())
        if match:  # Matches of the regular expression were found
            url = 'http://epsg.io/trans?s_srs=3034&t_srs=4326'
            response = requests.get(url, params=match.groupdict())
            coords = response.json()  # Create dict from json object
            TRNcoords = {'longitude': float(coords['x'])*-1,  # negative
                         'latitude': float(coords['y'])}
            print('Coordinates for TRNSYS: '+str(TRNcoords))
            return TRNcoords

        else:
            return dict()


def get_type99_header(weather_file_path, interpolate_freq):
    '''Create the header for Type99 weather files.
    '''

    type99_header = '''<userdefined>
 <longitude>   -0.000  ! east of greenwich: negative
 <latitude>     0.000  !
 <gmt>             1   ! time shift from GMT, east: positive (hours)
 <interval>        1   ! Data file time interval between consecutive lines
 <firsttime>       1   ! Time corresponding to first data line (hours)
 <var> IBEAM_H <col> 2 <interp> 0 <add> 0 <mult> 1 <samp> -1 !...to read radiation in [W/m²]
 <var> IDIFF_H <col> 3 <interp> 0 <add> 0 <mult> 1 <samp> -1 !...to read radiation in [W/m²]
 <var> TAMB    <col> 4 <interp> 2 <add> 0 <mult> 1 <samp> -1 !...to read ambient temperature in [°C]
 <var> WSPEED  <col> 5 <interp> 1 <add> 0 <mult> 1 <samp> -1 !...to read wind speed in [m/s]
 <var> RHUM    <col> 6 <interp> 1 <add> 0 <mult> 1 <samp> -1 !...to read relative humidity in [%]
 <var> WDIR    <col> 7 <interp> 1 <add> 0 <mult> 1 <samp> -1 !...to read wind direction in [degree] (north=0°/360°; east=90°; south=180°: west=270°)
 <var> CCOVER  <col> 8 <interp> 1 <add> 0 <mult> 1 <samp> -1 !...to read cloud cover in [octas] (Bedeckungsgrad in Achtel)
 <var> PAMB    <col> 9 <interp> 1 <add> 0 <mult> 1 <samp> -1 !...to read ambient air pressure in [hPa]
<data>
   '''

    replace_dict = get_TRNSYS_coordinates(weather_file_path)
    replace_dict['interval'] = interpolation_freq.seconds / 3600.0  # h
    replace_dict['firsttime'] = interpolation_freq.seconds / 3600.0  # h
    replace_dict['gmt'] = 1  # currently fixed

    for key, value in replace_dict.items():
        re_find = r'<'+key+'>\s*(.*)\s!'
        re_replace = r'<'+key+'>  '+str(value)+'  !'
        type99_header = re.sub(re_find, re_replace, type99_header)

    return type99_header


def interpolate_weather_file(weather_file_path,
                             weather_data_type,
                             datetime_start,
                             datetime_end,
                             interpolation_freq,
                             remove_leapyear):

    debug_plotting = False  # Show a plot to check the interpolation result
#    debug_plotting = True  # Show a plot to check the interpolation result

#    plot_value = 'IBEAM_H'
#    plot_value = 'IDIFF_H'
    plot_value = 'TAMB'
#    plot_value = 'WSPEED'
#    plot_value = 'RHUM'
#    plot_value = 'WDIR'
#    plot_value = 'CCOVER'
#    plot_value = 'PAMB'

    weather_file = os.path.basename(weather_file_path)

    # Read the file and store it in a DataFrame
    if weather_data_type == 'IGS' or weather_data_type == 'TRNSYS':
        weather_data = read_IGS_weather_file(weather_file_path)
    elif weather_data_type == 'DWD':
        weather_data = read_DWD_weather_file(weather_file_path)
    else:
        print('Error: Weather data type "'+weather_data_type+'" unknown!')
        exit()

    # Assumption: The IGS weather files always start at January 01.
    current_year = datetime_start.year
    newyear = pd.datetime(current_year, 1, 1)
    # Convert hours of year to DateTime and make that the index of DataFrame
    weather_data.index = pd.to_timedelta(weather_data['HOUR'],
                                         unit='h') + newyear

    # Infer the time frequency of the original data
    original_freq = pd.infer_freq(weather_data.index, warn=True)
    original_freq = pd.to_timedelta(1, unit=original_freq)
    # print 'Inferred freqency = '+str(original_freq)

    if debug_plotting is True:  # Plot the original data (Ambient temperature)
        fig = plt.figure()
        fig.suptitle(weather_file)
        weather_data[plot_value].plot(marker='.', label=plot_value+' orig')

    if interpolation_freq != original_freq:
        # Perform interpolation to new index of hours
        # Definition:
        # "Column value is a mean value related to the time interval delta t
        # ending at the time corresponding to actual weather_data line."
        #
        # Thus the first index must be the start time plus interpolation_freq

        # If the new frequency is larger (i.e. we are downsampling the data),
        # we need to use 'resample' to take the mean of the time intervals we
        # combine
        if interpolation_freq > original_freq:
            weather_data = weather_data.resample(interpolation_freq,
                                                 label='right',
                                                 closed='right').mean()
        # Now we can do the interpolation (upsampling). If we downsampled
        # before, this now only affects the start and end of the data
        interpolate_start = datetime_start + pd.Timedelta(interpolation_freq)
        interpolate_index = pd.DatetimeIndex(start=interpolate_start,
                                             end=datetime_end,
                                             freq=interpolation_freq)
        weather_data = weather_data.reindex(interpolate_index).interpolate(
                method='time')

        # The interpolation will generate NaN on the lines before the first
        # original line (hours = 1). Fill those NaN 'backwards' with the last
        # valid values:
        weather_data.fillna(method='backfill', inplace=True)

        # Cloud cover is given in integers, so interpolated values need to be
        # rounded
        weather_data['CCOVER'] = weather_data['CCOVER'].round(decimals=0)

        # Convert DateTime index to hours of the year
        weather_data['HOUR'] = (weather_data.index - datetime_start) / \
            np.timedelta64(1, 'h')

        if debug_plotting is True:  # Plot the interpolated data
            weather_data[plot_value].plot(marker='x',
                                          label=plot_value+' intpl.')

    # Remove leapyear from DataFrame (optional)
    if calendar.isleap(current_year) is True:
        print('Warning: '+str(current_year)+' is a leap year. Be careful!')
    if remove_leapyear is True:
        weather_data = weather_data[~((weather_data.index.month == 2) &
                                      (weather_data.index.day == 29))]

    # Now show the plots, including their legend
    if debug_plotting is True:
        plt.legend()
        plt.show(block=False)

    return weather_data


def analyse_weather_file(weather_data, interpolation_freq, weather_file):
    '''Analyse weather data for key values.

     Degree days (Gradtage) according to:

     a) VDI 3807-1, 2013: Verbrauchskennwerte für Gebäude - Grundlagen
        Uses a fixed indoor reference temperature of 20°C:
        G = (20 - t_m) if t_m < 15

     b) VDI 4710-2:
        G = sum (from n = 1 to z) of (T_g - T_m,n)
        where T_g = heating temperature limit
        and T_m = average temperature of day n
        z = number of heating days

     C) VDI 4710-3:
        In the Excel file available for download, the 'heating hours' are
        calculated from hourly values (not daily averages!), and the result
        is devided by 24 to gain 'heating days'. This gives, of course,
        very different results than method VDI 4710-2
    '''
    hours = interpolation_freq.seconds / 3600.0  # h
    # print weather_data['TAMB']

    # Resample ambient temperatures in DataFrame to days and take mean
    tamb_avg_list = weather_data['TAMB'].resample('D', label='left',
                                                  closed='right').mean()
#    tamb_avg_list = weather_data['TAMB']
#    print(tamb_avg_list)

    #  Generate histogram information: (number of days with certain tamb_avg)
#    step = 1
#    min = round(tamb_avg_list.min()-1, 0)
#    max = round(tamb_avg_list.max()+1, 0)
#    bin_range = np.arange(min, max+step, step)
#    out, bins = pd.cut(tamb_avg_list, bins=bin_range,
#                       include_lowest=True, right=False, retbins=True)
#    print(out.value_counts(sort=False))

    # Define new DataFrame with the desired columns
    weather_data_daily = pd.DataFrame(data=tamb_avg_list,
                                      columns=['TAMB'])
#                                      columns=['TAMB', 'G20', 'heating days'])

    t_heat = 15  # °C heating temperature limit: t_m < t_heat
#    t_heat = 10  # °C heating temperature limit: t_m < t_heat
    for j, date_obj in enumerate(weather_data_daily.index):
        t_m = weather_data_daily.loc[date_obj]['TAMB']  # °C
        if t_m < t_heat:
            G = (20 - t_m)  # VDI 3807-1: Fixed indoor reference temperature
#            G = (t_heat - t_m)  # VDI 4710-2: Flexible calculation
            d_heat = 1.0
        else:
            G = 0
            d_heat = 0.0
        weather_data_daily.loc[date_obj, 'G20/15'] = G
        weather_data_daily.loc[date_obj, 'Heating days'] = d_heat

    tamb_avg = weather_data['TAMB'].mean()                        # °C
    IBEAM_H_sum = hours/1000*weather_data['IBEAM_H'].sum()        # kWh/m²
    IDIFF_H_sum = hours/1000*weather_data['IDIFF_H'].sum()        # kWh/m²
    G_sum = weather_data_daily['G20/15'].sum()                         # K*d
    d_heat_sum = weather_data_daily['Heating days'].sum()          # d
    print('Year statistics for: '+weather_file)
    print('   T_amb average = {:6.1f} °C'.format(tamb_avg))
    print('   I_beam,h      = {:6.1f} kWh/m^2'.format(IBEAM_H_sum))
    print('   I_diff,h      = {:6.1f} kWh/m^2'.format(IDIFF_H_sum))
    print('   G20/{:2.0f}        = {:6.1f} K*d'.format(t_heat, G_sum))
    print('   Heating days  = {:6.1f} d'.format(d_heat_sum))

    # Print table of montly sum / mean values
    wd_sum = weather_data_daily[['G20/15', 'Heating days']].resample('M').sum()
    wd_mean = weather_data_daily['TAMB'].resample('M').mean()
    weather_data_monthly = pd.concat([wd_sum, wd_mean], axis=1)
    print(weather_data_monthly)

    return True


def print_IGS_weather_file(weather_data, print_folder, print_file,
                           bool_print_index, bool_print_header,
                           type99_header=None):
    '''Print the results to a file
    '''

    if not os.path.exists(print_folder):
        os.makedirs(print_folder)

    print_path = os.path.join(print_folder, print_file)
    print('Printing to '+print_path)

    print_file = open(print_path, 'w')
    if type99_header is not None:
        print_file.write(type99_header)

    weather_data.to_string(print_file,
                           index=bool_print_index,
                           header=bool_print_header)
    print_file.close()

    return True


if __name__ == "__main__":
    '''Execute the methods defined above.
    Only do this, if the script is executed directly (in contrast to imported)
    '''

    for i, weather_file_path in enumerate(weather_file_list):
        if bool_regex:
            weather_data_type = matchlist[i].get('ztype', weather_data_type)

        # Read and interpolate weather data:
        weather_data = interpolate_weather_file(weather_file_path,
                                                weather_data_type,
                                                datetime_start,
                                                datetime_end,
                                                interpolation_freq,
                                                remove_leapyear)

        # Analyse weather data
        weather_file = os.path.basename(weather_file_path)
        analyse_weather_file(weather_data, interpolation_freq, weather_file)

        # Print the weather data file
        if bool_regex is False:
            print_file = weather_file
        else:
            match_dict = matchlist[i]
            if match_dict == dict():
                # Just use the name of the original file if the dict is empty
                print_file = weather_file
            else:
                # Else, use the contents of the match_dict to name the file:
                entry_list = []
                for entry in sorted(match_dict):
                    entry_list.append(match_dict[entry])
                print_file = '_'.join(entry_list)+'.dat'

        if bool_print_type99_head:
            type99_header = get_type99_header(weather_file_path,
                                              interpolation_freq)

        print_folder = os.path.join(base_folder, 'Result')
        print_IGS_weather_file(weather_data, print_folder, print_file,
                               bool_print_index, bool_print_header,
                               type99_header=type99_header)

    # The script will be blocked until the user closes the plot window
    plt.show()
