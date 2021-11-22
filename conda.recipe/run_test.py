# Copyright (C) 2020 Joris Zimmermann

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program. If not, see https://www.gnu.org/licenses/.

"""LPagg: Load profile aggregator for building simulations.

LPagg
=====
The load profile aggregator combines profiles for heat and power demand
of buildings from different sources.


Module run_test
---------------
Tests to run during build process.
"""

import unittest
import os
import logging
import lpagg.misc
import lpagg.agg


def main_test():
    """Run test function."""
    lpagg.misc.setup()

    #  Get user input
    file = os.path.join(os.path.dirname(__file__),
                        r'./lpagg/examples/VDI_4655_config_example.yaml')

    # Import the config YAML file and add the default settings
    cfg = lpagg.agg.perform_configuration(file)
    log_level = 'ERROR'
    logging.getLogger('lpagg.agg').setLevel(level=log_level)
    logging.getLogger('lpagg.misc').setLevel(level=log_level)
    logging.getLogger('lpagg.weather_converter').setLevel(level=log_level)
    logging.getLogger('lpagg.VDI4655').setLevel(level=log_level)
    logging.getLogger('lpagg.BDEW').setLevel(level=log_level)
    logging.getLogger('lpagg.simultaneity').setLevel(level=log_level)

    # Aggregate load profiles
    weather_data = lpagg.agg.aggregator_run(cfg)

    # Evalute results
    cols = [x for x in weather_data.columns if 'E_' in x]
    weather_daily_sum = weather_data[cols].resample('D', label='left',
                                                    closed='right').sum()
    weather_montly_sum = weather_daily_sum.resample('M', label='right',
                                                    closed='right').sum()
    weather_annual_sum = weather_montly_sum.resample('A', label='right',
                                                     closed='right').sum()
    result = weather_annual_sum.sum(axis=1).sum()
    return result


class TestMethods(unittest.TestCase):
    """Defines tests."""

    def test(self):
        """Test the calculated total energy demand in kWh."""
        self.assertAlmostEqual(main_test(), 7674902.833019385)


if __name__ == '__main__':
    unittest.main()
