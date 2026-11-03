"""Check the current public emissivity dispatcher and PPV cube entry point."""

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from Pipeline.scr import emissivity_tables, ppv_cube
from Pipeline.scr.tables import ps20


class TestEmissivityDispatcher(unittest.TestCase):
    def test_missing_file_or_directory_raises_before_adapter(self):
        with tempfile.TemporaryDirectory() as directory:
            for filename in [directory, str(Path(directory) / 'missing.hdf5')]:
                with self.subTest(filename=filename), patch.object(ps20, 'calculate_particle_line_luminosity') as adapter:
                    with self.assertRaisesRegex(AssertionError, 'Emissivity table file not found'):
                        emissivity_tables.calculate_particle_line_luminosity(
                            {}, table_type='ps20', table_filename=filename,
                            redshift=0, line_label='test',
                        )
                    adapter.assert_not_called()

    def test_existing_file_dispatches_fields_and_settings_unchanged(self):
        with tempfile.NamedTemporaryFile(suffix='.hdf5') as table:
            expected = {'line_luminosity_erg_s': np.array([2.0])}
            gas = {'example': np.array([1.0])}
            with patch.object(ps20, 'calculate_particle_line_luminosity', return_value=expected) as adapter:
                result = emissivity_tables.calculate_particle_line_luminosity(
                    gas, table_type='PS20', table_filename=table.name,
                    redshift=0, line_label='test',
                )
            self.assertIs(result, expected)
            self.assertIs(adapter.call_args.args[0], gas)
            adapter.assert_called_once_with(
                gas, table_filename=table.name, redshift=0, line_label='test',
            )

    def test_invalid_model_and_path_inputs_raise_clear_errors(self):
        for model in [None, 42, '', 'unknown']:
            with self.subTest(model=model), patch.object(ps20, 'calculate_particle_line_luminosity') as adapter:
                with self.assertRaises(ValueError):
                    emissivity_tables.calculate_particle_line_luminosity(
                        {}, table_type=model, table_filename='unused.hdf5',
                        redshift=0, line_label='test',
                    )
                adapter.assert_not_called()
        with self.assertRaisesRegex(TypeError, 'table_filename'):
            emissivity_tables.calculate_particle_line_luminosity(
                {}, table_type='ps20', table_filename=None, redshift=0, line_label='test',
            )

    def test_raw_cloudy_model_is_rejected_before_opening_a_table(self):
        with patch.object(ps20, 'calculate_particle_line_luminosity') as adapter:
            with self.assertRaisesRegex(ValueError, 'Unknown emissivity table type'):
                emissivity_tables.calculate_particle_line_luminosity(
                    {}, table_type='cloudy_mvp', table_filename='unused.emis',
                    redshift=0, line_label='test',
                )
            adapter.assert_not_called()


class TestPPVCube(unittest.TestCase):
    def test_public_entry_point_preserves_axes_and_luminosity(self):
        for los_axis in ['x', 'y', 'z']:
            with self.subTest(los_axis=los_axis):
                gas = {f'{axis}_kpc': np.array([0.0, 0.5]) for axis in 'xyz'}
                gas.update({f'v_{axis}_kms': np.array([-5.0, 5.0]) for axis in 'xyz'})
                result = ppv_cube.make_ppv_cube(
                    gas, np.array([2.0, 3.0]), [-1, 1], [-1, 1], [-10, 0, 10],
                    los_axis=los_axis,
                )
                np.testing.assert_array_equal(result['cube_erg_s'], [[[2.0, 3.0]]])
                self.assertEqual(result['cube_axis_order'], 'x,y,velocity')
                image_axes = {'x': ('y', 'z'), 'y': ('x', 'z'), 'z': ('x', 'y')}
                self.assertEqual((result['image_x_axis'], result['image_y_axis']), image_axes[los_axis])
                self.assertEqual(result['cube_erg_s'].sum(), 5.0)


if __name__ == '__main__':
    unittest.main()
