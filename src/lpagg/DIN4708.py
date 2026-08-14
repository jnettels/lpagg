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


Module DIN4708
--------------
Implementation of DIN 4708 "Central heat-water-installations"

Implementierung der DIN 4708 "Zentrale Wassererwärmungsanlagen"

This allows an estimation of the simultaneity factor for domestic hot water,
by determining the water-heat-demand in one and multiple reference buildings.
"""

import logging
from math import sqrt, pi, factorial

# Define the logging function
logger = logging.getLogger(__name__)


def main():
    """Run example usage for calculation of different heat demands."""
    for N in range(1, 20):
        GLF = calc_GLF(N)
        W = W_z(N)
        print(N, W, GLF)

    N = 10
    print('W_z:', W_z(N))
    print('W_1:', W_1(N))
    print('W_p:', W_p(N))


def calc_GLF(N):
    """Calculate a simultaneity factor.

    Divide the heat demand of a single reference home by the heat demand
    of ``N`` reference homes. Strictly speaking, this is not part of DIN 4708.

    Args:
        N (int): Number of reference homes

    Returns:
        GLF (float): simultaneity factor
    """
    try:
        GLF = W_z() / W_z(N)
    except ZeroDivisionError:
        GLF = 0
    return GLF


def calc_N():
    """Calculate the demand indicator ("Bedarfskennzahl") N.

    The demand indicator ("Bedarfskennzahl") N for a reference home
    ("Einheitswohnung") is defined as ``N=1``. For homes not identical to
    the reference home, it can be calculated with DIN 4708 part 2.
    """
    raise NotImplementedError('Calculation of demand indicator N '
                              '("Bedarfskennzahl") with DIN 4708-2 '
                              'not implemented.')


def W_1(N=1):
    """Return hourly heat demand ("Stundenwärmebedarf") W_1.0.

    Args:
        N (int): Number of reference homes

    Returns:
        W_z (float): Heat demand in [Wh]
    """
    return W_z(N, z=1.0)


def W_2TN(N=1):
    """Return periodic heat demand ("Periodenwärmebedarf").

    Args:
        N (int): Number of reference homes

    Returns:
        W_z (float): Heat demand in [Wh]
    """
    return W_p(N)


def W_p(N=1):
    """Return periodic heat demand ("Periodenwärmebedarf").

    Args:
        N (int): Number of reference homes

    Returns:
        W_z (float): Heat demand in [Wh]
    """
    Wb = 5820  # [Wh] specific heat demand for reference home
    W = Wb * N * (1 + sqrt(N))/sqrt(N)
    return W


def W_z(N=1, z=1/6):
    """Calculate heat demand ``W_z``.

    Calculate heat demand ``W_z`` during demand duration ``z`` for
    the demand indicator ``N`` (number of reference homes).
    The default duration ``z=zB`` calculates the peak heat demand ``W_zB``.

    Args:
        N (int): Number of reference homes

        z (float): Demand duration in [h]. Default: 1/6h = 10min (Bathtub)

    Returns:
        W_z (float): Heat demand in [Wh]
    """
    Wb = 5820  # [Wh] spezifischer Wärmebedarf für Einheitswohnung
    u1 = z * 0.244 * (1 + sqrt(N))/sqrt(N)
    u2 = z * 3.599 * (1 + sqrt(N))/sqrt(N)
    W_z = Wb * (N * K(u1) + sqrt(N) * K(u2))
    return W_z


def K(u):
    """Calculate result of K(u)-function with gaussian normal distribution.

    Args:
        u (float): Time function

    Returns:
        K (float): Integral value
    """
    if u >= 1.81:
        K = 1.0

    else:
        SUM = 0
        for k in range(1, 50):
            SUM += (pow(-1, k) * pow(u, 2*k + 1)) / (factorial(k) * (2*k + 1))

        K = 2/sqrt(pi) * (u + SUM)

    return K


if __name__ == '__main__':
    """This part is executed when the script is started directly with
    Python, not when it is loaded as a module.
    """
    main()
