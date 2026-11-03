"""Check boundary interpolation and preservation of physical gas inputs."""

from pathlib import Path
import tempfile
import unittest
import warnings

import h5py
import numpy as np

from Pipeline.scr.tables import cloudy_hdf5


class TestCloudyHDF5Clipping(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.filename = Path(self.directory.name) / 'table.hdf5'
        with h5py.File(self.filename, 'w') as table:
            table.attrs['schema'] = cloudy_hdf5.SCHEMA
            table.attrs['abundance_model'] = 'fixed test abundances'
            table['axes/log_temperature_K'] = [6.0, 7.0]
            table['axes/log_nH_cm3'] = [-5.0, -4.0]
            table.create_dataset('lines/labels', data=['O  8 18.9689A'],
                                 dtype=h5py.string_dtype())
            # log emissivity = log T + 2*log nH - 26.
            values = 10.0 ** np.array([[-30.0, -28.0], [-29.0, -27.0]])
            emissivity = table.create_dataset('lines/emissivity_erg_s_cm3', data=values[:, :, None])
            emissivity.attrs['unit'] = 'erg s^-1 cm^-3'
            emissivity.attrs['axis_order'] = 'log_temperature_K, log_nH_cm3, line'

    def test_both_axes_clip_at_both_limits_without_mutating_gas(self):
        gas = {'T_K': 10.0 ** np.array([5.0, 8.0, 6.5, 5.0, 6.0]),
               'nH_cm3': 10.0 ** np.array([-6.0, -3.0, -4.5, -4.5, -5.0]),
               'volume_cm3': np.array([2.0, 3.0, 4.0, 5.0, 6.0])}
        original = {name: values.copy() for name, values in gas.items()}
        with warnings.catch_warnings(record=True) as messages:
            warnings.simplefilter('always')
            result = cloudy_hdf5.calculate_particle_line_luminosity(
                gas, table_filename=self.filename, redshift=0, line_label='O  8 18.9689A',
            )
        expected = np.array([1e-30, 1e-27, (1e-30 + 1e-28 + 1e-29 + 1e-27)/4,
                             (1e-30 + 1e-28)/2, 1e-30])
        np.testing.assert_allclose(result['log_volume_emissivity'], np.log10(expected))
        np.testing.assert_allclose(result['line_luminosity_erg_s'],
                                   expected * gas['volume_cm3'],
                                   rtol=1e-12, atol=0)
        self.assertEqual(result['clipped_particle_count'], 3)
        text = [str(message.message) for message in messages]
        self.assertTrue(any('log_temperature_K clipping: 2 below, 1 above' in message for message in text))
        self.assertTrue(any('log_nH_cm3 clipping: 1 below, 1 above' in message for message in text))
        for name in gas:
            np.testing.assert_array_equal(gas[name], original[name])

    def test_in_range_gas_is_unchanged_without_clipping_warning(self):
        with warnings.catch_warnings(record=True) as messages:
            warnings.simplefilter('always')
            result = cloudy_hdf5.calculate_particle_line_luminosity(
                {'T_K': np.array([10.0**6.5]), 'nH_cm3': np.array([10.0**-4.5]),
                 'volume_cm3': np.array([2.0])},
                table_filename=self.filename, redshift=0, line_label='O  8 18.9689A',
            )
        np.testing.assert_allclose(result['log_volume_emissivity'], [np.log10((1e-30+1e-28+1e-29+1e-27)/4)])
        self.assertEqual(result['clipped_particle_count'], 0)
        self.assertFalse(any('clipping:' in str(message.message) for message in messages))

    def test_invalid_physical_values_still_raise(self):
        for name, value in [('T_K', np.nan), ('T_K', -1), ('nH_cm3', np.inf),
                            ('nH_cm3', 0), ('volume_cm3', -1)]:
            with self.subTest(name=name, value=value):
                gas = {'T_K': np.array([1e6]), 'nH_cm3': np.array([1e-5]),
                       'volume_cm3': np.array([1.0])}
                gas[name] = np.array([value])
                with self.assertRaises(ValueError):
                    cloudy_hdf5.calculate_particle_line_luminosity(
                        gas, table_filename=self.filename, redshift=0, line_label='O  8 18.9689A',
                    )


if __name__ == '__main__':
    unittest.main()
