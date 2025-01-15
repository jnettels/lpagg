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


def main_test(use_demandlib=False, unique_profile_workflow=True):
    """Run test function."""
    lpagg.misc.setup()

    # Get example user input. Directory is different in conda test environment
    file = os.path.join(os.path.dirname(__file__),
                        r'./lpagg/examples/VDI_4655_config_example.yaml')
    if not os.path.exists(file):
        file = os.path.join(os.path.dirname(__file__),
                            r'../lpagg/examples/VDI_4655_config_example.yaml')

    # Import the config YAML file and add the default settings
    cfg = lpagg.agg.perform_configuration(file)

    # Change the configuration to run with different test settings
    cfg['settings']['use_demandlib'] = use_demandlib
    cfg['settings']['unique_profile_workflow'] = unique_profile_workflow

    log_level = 'ERROR'
    logging.getLogger('lpagg.agg').setLevel(level=log_level)
    logging.getLogger('lpagg.misc').setLevel(level=log_level)
    logging.getLogger('lpagg.weather_converter').setLevel(level=log_level)
    logging.getLogger('lpagg.VDI4655').setLevel(level=log_level)
    logging.getLogger('lpagg.BDEW').setLevel(level=log_level)
    logging.getLogger('lpagg.simultaneity').setLevel(level=log_level)

    # Aggregate load profiles
    agg_dict = lpagg.agg.aggregator_run(cfg)
    weather_data = agg_dict['weather_data']

    # Evalute results
    cols = [x for x in weather_data.columns if 'E_' in x]
    weather_daily_sum = weather_data[cols].resample('D', label='left',
                                                    closed='right').sum()
    weather_montly_sum = weather_daily_sum.resample('ME', label='right',
                                                    closed='right').sum()
    weather_annual_sum = weather_montly_sum.resample('YE', label='right',
                                                     closed='right').sum()
    result = weather_annual_sum.sum(axis=1).sum()
    return result


class TestMethods(unittest.TestCase):
    """Defines tests."""

    def test_lpagg_internal(self):
        """Test the calculated total energy demand in kWh."""
        self.assertAlmostEqual(
            main_test(use_demandlib=False, unique_profile_workflow=False),
            7674663.094635218)

    def test_lpagg_internal_with_unique_profile(self):
        """Test the calculated total energy demand in kWh."""
        self.assertAlmostEqual(
            main_test(use_demandlib=False, unique_profile_workflow=True),
            7674663.094635218)

    def test_demandlib(self):
        """Test the calculated total energy demand in kWh."""
        try:  # if import is successfull, use demandlib
            from demandlib import vdi
        except ImportError:  # if demandlib is not available, do not test
            self.assertEqual(0, 0)
        else:  # demandlib with vdi is available, so test it
            # The total energy differs between 'lpagg internal' and
            # 'demandlib', because a heat loss calculation is included.
            # The shape of the profiles differ, so heat losses are not the
            # same for each time step.
            self.assertAlmostEqual(
                main_test(use_demandlib=True, unique_profile_workflow=False),
                7467881.584333282)

    def test_demandlib_with_unique_profile(self):
        """Test the calculated total energy demand in kWh."""
        try:  # if import is successfull, use demandlib
            from demandlib import vdi
        except ImportError:  # if demandlib is not available, do not test
            self.assertEqual(0, 0)
        else:  # demandlib with vdi is available, so test it
            # The total energy differs between 'lpagg internal' and
            # 'demandlib', because a heat loss calculation is included.
            # The shape of the profiles differ, so heat losses are not the
            # same for each time step.
            self.assertAlmostEqual(
                main_test(use_demandlib=True, unique_profile_workflow=True),
                7467881.584333282)


if __name__ == '__main__':
    unittest.main()
