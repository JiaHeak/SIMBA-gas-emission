"""Keep offline Cloudy text conversion usable, but reject it as a pipeline model."""

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from Pipeline.scr.emissivity_tables import calculate_particle_line_luminosity
from Pipeline.scr.tables import cloudy_mvp


LINE = 'O  8 18.9689A'


def make_fixture(directory):
    """Write a tiny complete grid with coordinates deliberately out of axis order.

    Returns: paths (tuple[Path, Path]): emissivity and grid filenames.
    """
    emis = Path(directory) / 'mvp.emis'
    grid = Path(directory) / 'actual_coordinates.grid'
    points = [(7.0, -4.0), (6.8, -5.0), (7.0, -5.0), (6.8, -4.0)]
    header = '#Index\tFailure?\tWarnings?\tExit code\t#rank\t#seq\tCOROnal e\tHDEN=%f L\tgrid parameter string\n'
    grid.write_text(header + ''.join(
        f'{i:09d}\tF\tF\tok\t0\t{i}\t{t}\t{n}\t{t}, {n}\n'
        for i, (t, n) in enumerate(points)
    ), encoding='utf-8')
    # This analytic log emissivity is exactly linear in log T and log nH.
    emis.write_text('#depth\t' + LINE + ' \n' + ''.join(
        f'0.5\t{10.0 ** (t + 2*n - 30):.12e}\n' for t, n in points
    ), encoding='utf-8')
    return emis, grid


class TestCloudyMVP(unittest.TestCase):
    def test_dispatcher_rejects_raw_cloudy_model(self):
        with self.assertRaisesRegex(ValueError, 'Unknown emissivity table type'):
            calculate_particle_line_luminosity(
                {}, table_type='cloudy_mvp', table_filename='unused.emis',
                redshift=0, line_label=LINE,
            )

    def test_raw_parser_remains_available_for_hdf5_conversion(self):
        from Pipeline.scr.tables.build_cloudy_hdf5 import build_table

        with tempfile.TemporaryDirectory() as directory:
            emis, grid = make_fixture(directory)
            commands = Path(directory) / 'model.in'
            commands.write_text('coronal 6.8 vary\nhden -5 vary\nstop zone 1\n')
            output = Path(directory) / 'table.hdf5'
            build_table(commands, emis, grid, output)
            from Pipeline.scr.tables.cloudy_hdf5 import read_cloudy_hdf5_table
            table = read_cloudy_hdf5_table(output)
            self.assertEqual(table['line_labels'], [LINE])
            self.assertEqual(table['volume_emissivity_erg_s_cm3'].shape, (2, 2, 1))

    def test_coordinate_mapping_log_interpolation_and_particle_luminosity(self):
        with tempfile.TemporaryDirectory() as directory:
            emis, grid = make_fixture(directory)
            table = cloudy_mvp.read_cloudy_mvp_table(emis, grid)
            self.assertEqual(table['line_labels'], [LINE])
            self.assertEqual(table['volume_emissivity_erg_s_cm3'].shape, (2, 2, 1))
            self.assertAlmostEqual(np.log10(table['volume_emissivity_erg_s_cm3'][0, 0, 0]), -33.2)
            result = cloudy_mvp.interpolate_log_emissivity(emis, -4.5, 6.9, LINE, grid_filename=grid)
            self.assertEqual(result.shape, ())
            self.assertAlmostEqual(float(result), -32.1)
            fields = {'T_K': np.array([10.0**6.9]), 'nH_cm3': np.array([10.0**-4.5]),
                      'volume_cm3': np.array([3.0]), 'Z_Zsun': np.array([0.0])}
            with self.assertWarnsRegex(RuntimeWarning, 'fixed Cloudy default abundances'):
                emission = cloudy_mvp.calculate_particle_line_luminosity(
                    fields, table_filename=emis, grid_filename=grid,
                    redshift=0, line_label=LINE,
                )
            np.testing.assert_allclose(emission['line_luminosity_erg_s'], [3*10.0**-32.1],
                                       rtol=1e-10, atol=0)
            np.testing.assert_array_equal(fields['Z_Zsun'], [0.0])

    def test_out_of_bounds_missing_lines_and_invalid_models_raise(self):
        with tempfile.TemporaryDirectory() as directory:
            emis, grid = make_fixture(directory)
            with self.assertRaisesRegex(ValueError, 'outside'):
                cloudy_mvp.interpolate_log_emissivity(emis, -4.5, 6.0, LINE, grid_filename=grid)
            with self.assertRaisesRegex(ValueError, 'absent'):
                cloudy_mvp.interpolate_log_emissivity(emis, -4.5, 6.9, 'unknown', grid_filename=grid)
            original_grid = grid.read_text()
            grid.write_text(original_grid.replace('\tF\tF\tok', '\tT\tF\tok', 1))
            with self.assertRaisesRegex(ValueError, 'failed'):
                cloudy_mvp.read_cloudy_mvp_table(emis, grid)
            grid.write_text(original_grid)
            original_emis = emis.read_text()
            emis.write_text(original_emis + '0.5\t1e-30\n')
            with self.assertRaisesRegex(ValueError, 'one .emis row'):
                cloudy_mvp.read_cloudy_mvp_table(emis, grid)
            emis.write_text(original_emis)
            grid.write_text(original_grid.replace('6.8\t-5.0', '7.0\t-4.0', 1))
            with self.assertRaisesRegex(ValueError, 'unique pairs'):
                cloudy_mvp.read_cloudy_mvp_table(emis, grid)


if __name__ == '__main__':
    unittest.main()
