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


Module DIN4708
--------------
Implementation of DIN 4708 "Central heat-water-installations"

Implementierung der DIN 4708 "Zentrale Wassererwärmungsanlagen"

This allows an estimation of the simultaneity factor for domestic hot water,
by determining the water-heat-demand in one and multiple reference buildings.

'''
import logging
from math import sqrt, pi, factorial

# Define the logging function
logger = logging.getLogger(__name__)


def main():
    '''Example usage for calculation of different heat demands.
    '''
    for N in range(1, 20):
        GLF = calc_GLF(N)
        W = W_z(N)
        print(N, W, GLF)

    N = 10
    print('W_z:', W_z(N))
    print('W_1:', W_1(N))
    print('W_p:', W_p(N))


def calc_GLF(N):
    '''Calculate a simultaneity factor. Divide the heat demand of a single
    reference home by the heat demand of ``N`` reference homes. Strictly
    speaking, this is not part of DIN 4708.

    Args:
        N (int): Number of reference homes

    Returns:
        GLF (float): simultaneity factor
    '''
    try:
        GLF = W_z() / W_z(N)
    except ZeroDivisionError:
        GLF = 0
    return GLF


def calc_N():
    '''The demand indicator ("Bedarfskennzahl") N for a reference home
    ("Einheitswohnung") is defined as ``N=1``. For homes not identical to
    the reference home, it can be calculated with DIN 4708 part 2.
    '''
    raise NotImplementedError('Calculation of demand indicator N '
                              '("Bedarfskennzahl") with DIN 4708-2 '
                              'not implemented.')


def W_1(N=1):
    '''Hourly heat demand ("Stundenwärmebedarf") W_1.0

    Args:
        N (int): Number of reference homes

    Returns:
        W_z (float): Heat demand in [Wh]
    '''
    return W_z(N, z=1.0)


def W_2TN(N=1):
    '''Periodic heat demand ("Periodenwärmebedarf")

    Args:
        N (int): Number of reference homes

    Returns:
        W_z (float): Heat demand in [Wh]
    '''
    return W_p(N)


def W_p(N=1):
    '''Periodic heat demand ("Periodenwärmebedarf")

    Args:
        N (int): Number of reference homes

    Returns:
        W_z (float): Heat demand in [Wh]
    '''
    Wb = 5820  # [Wh] specific heat demand for reference home
    W = Wb * N * (1 + sqrt(N))/sqrt(N)
    return W


def W_z(N=1, z=1/6):
    '''Calculate heat demand ``W_z`` during demand duration ``z`` for
    the demand indicator ``N`` (number of reference homes).
    The default duration ``z=zB`` calculates the peak heat demand ``W_zB``.

    Args:
        N (int): Number of reference homes

        z (float): Demand duration in [h]. Default: 1/6h = 10min (Bathtub)

    Returns:
        W_z (float): Heat demand in [Wh]
    '''
    Wb = 5820  # [Wh] spezifischer Wärmebedarf für Einheitswohnung
    u1 = z * 0.244 * (1 + sqrt(N))/sqrt(N)
    u2 = z * 3.599 * (1 + sqrt(N))/sqrt(N)
    W_z = Wb * (N * K(u1) + sqrt(N) * K(u2))
    return W_z


def K(u):
    '''Calculate result of K(u)-function with gaussian normal distribution.

    Args:
        u (float): Time function

    Returns:
        K (float): Integral value
    '''
    if u >= 1.81:
        K = 1.0

    else:
        SUM = 0
        for k in range(1, 50):
            SUM += (pow(-1, k) * pow(u, 2*k + 1)) / (factorial(k) * (2*k + 1))

        K = 2/sqrt(pi) * (u + SUM)

    return K


if __name__ == '__main__':
    '''This part is executed when the script is started directly with
    Python, not when it is loaded as a module.
    '''
    main()
