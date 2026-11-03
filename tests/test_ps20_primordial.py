"""Check the PS20 special primordial bin with a tiny synthetic table."""

from pathlib import Path
import tempfile
import unittest

import h5py
import numpy as np

from Pipeline.scr.emissivity_tables import calculate_particle_line_luminosity
from Pipeline.scr.tables import ps20


LINE_LABEL = 'H  1      6562.81A'


class TestPS20Primordial(unittest.TestCase):
    def test_zero_and_positive_metallicity_use_separate_table_slices(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            table_path = Path(temporary_directory) / 'tiny_ps20.hdf5'
            with h5py.File(table_path, 'w') as table:
                bins = table.create_group('TableBins')
                bins.create_dataset('RedshiftBins', data=[0.0, 1.0])
                bins.create_dataset('TemperatureBins', data=[3.0, 4.0])
                bins.create_dataset('MetallicityBins', data=[-50.0, -1.0, 0.0])
                bins.create_dataset('DensityBins', data=[-3.0, -2.0])
                table.create_dataset('IdentifierLines', data=[LINE_LABEL.encode()])
                emissivity = np.empty((2, 2, 3, 2, 1), dtype=np.float64)
                emissivity[0, :, 0, :, 0] = 1.0
                emissivity[1, :, 0, :, 0] = 3.0
                emissivity[0, :, 1, :, 0] = 4.0
                emissivity[1, :, 1, :, 0] = 6.0
                emissivity[0, :, 2, :, 0] = 6.0
                emissivity[1, :, 2, :, 0] = 8.0
                table.create_dataset('Tdep/EmissivitiesVol', data=emissivity)

            gas_fields = {
                'nH_cm3': np.array([10.0 ** -2.5, 10.0 ** -2.5]),
                'T_K': np.array([10.0 ** 3.5, 10.0 ** 3.5]),
                'Z_Zsun': np.array([0.0, 0.1]),
                'volume_cm3': np.array([2.0, 3.0]),
            }
            result = ps20.calculate_particle_line_luminosity(
                gas_fields,
                table_filename=str(table_path),
                redshift=0.5,
                line_label=LINE_LABEL,
            )

        np.testing.assert_allclose(result['log_volume_emissivity'], [2.0, 5.0])
        np.testing.assert_allclose(result['line_luminosity_erg_s'], [200.0, 300000.0])

    def test_negative_metallicity_is_not_silently_changed(self):
        gas_fields = {
            'nH_cm3': np.array([1.0]),
            'T_K': np.array([10000.0]),
            'Z_Zsun': np.array([-0.1]),
            'volume_cm3': np.array([1.0]),
        }
        with self.assertRaises(ValueError):
            ps20.calculate_particle_line_luminosity(
                gas_fields,
                table_filename='unused.hdf5',
                redshift=0.5,
                line_label=LINE_LABEL,
            )

    def test_generic_dispatcher_calls_ps20_module(self):
        gas_fields = {
            'nH_cm3': np.array([1.0]),
            'T_K': np.array([10000.0]),
            'Z_Zsun': np.array([0.0]),
            'volume_cm3': np.array([1.0]),
        }
        from unittest.mock import patch
        expected = {'line_luminosity_erg_s': np.array([7.0])}
        with tempfile.NamedTemporaryFile(suffix='.hdf5') as table, patch.object(ps20, 'calculate_particle_line_luminosity', return_value=expected) as call:
            result = calculate_particle_line_luminosity(
                gas_fields, table_type='ps20', table_filename=table.name,
                redshift=0.5, line_label=LINE_LABEL,
            )
        self.assertIs(result, expected)
        call.assert_called_once_with(
            gas_fields, table_filename=table.name, redshift=0.5,
            line_label=LINE_LABEL,
        )


if __name__ == '__main__':
    unittest.main()
