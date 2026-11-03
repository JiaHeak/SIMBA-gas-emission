"""Small multi-line converter checks, including explicit parallel alignment."""

from pathlib import Path
import tempfile
import unittest
import warnings

import h5py
import numpy as np

from Pipeline.scr.tables.build_cloudy_hdf5 import build_table
from Pipeline.scr.tables.cloudy_hdf5 import read_cloudy_hdf5_table, calculate_particle_line_luminosity


class TestMultilineConversion(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.input = self.root / 'model.in'
        self.grid = self.root / '.grd'
        self.emis = self.root / '.emis'
        self.output = self.root / 'table.hdf5'
        self.input.write_text('coronal 6.8 vary\nhden -5 vary\nstop zone 1\nset dr 0\n')
        self.points = [(7.0, -4.0), (6.8, -5.0), (7.0, -5.0), (6.8, -4.0)]
        self.grid.write_text(
            '#Index\tFailure?\tWarnings?\tExit code\t#rank\t#seq\tCOROnal e\tHDEN=%f L\tgrid parameter string\n'
            + ''.join(f'{i}\tF\tF\tok\t{i%2}\t{3-i}\t{t}\t{n}\t{t}, {n}\n'
                      for i, (t, n) in enumerate(self.points)))
        self.labels = ['O  8 18.9689A', 'Ne10 12.1300A']
        self.emis.write_text('#depth\t' + '\t'.join(self.labels) + '\n' + ''.join(
            f'0.5\t{10.0**(t+2*n-26):.12e}\t{3*10.0**(t+2*n-26):.12e}\n'
            for t, n in self.points))

    def test_multiline_parallel_round_trip_and_existing_pipeline_reader(self):
        build_table(self.input, self.emis, self.grid, self.output,
                    model_order='cloudy-c25-index', resolved_wavelengths=[18.9689, 12.1321],
                    line_match_note='Synthetic fixture; not measured atomic data.')
        table = read_cloudy_hdf5_table(self.output)
        self.assertEqual(table['line_labels'], self.labels)
        self.assertEqual(table['volume_emissivity_erg_s_cm3'].shape, (2, 2, 2))
        with h5py.File(self.output) as saved:
            self.assertEqual(saved['provenance/input_text'].asstr()[()], self.input.read_text())
            np.testing.assert_array_equal(saved['models/rank'][...], [0, 1, 0, 1])
            np.testing.assert_array_equal(saved['lines/resolved_wavelength_A'][...], [18.9689, 12.1321])
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            for j, label in enumerate(self.labels):
                emission = calculate_particle_line_luminosity(
                    {'T_K': np.array([1e7]), 'nH_cm3': np.array([1e-4]),
                     'volume_cm3': np.array([2.0])}, table_filename=self.output,
                    redshift=0, line_label=label)
                np.testing.assert_allclose(emission['line_luminosity_erg_s'],
                                           [2 * (1 if j == 0 else 3) * 1e-27], rtol=1e-10, atol=0)
        with self.assertRaises(FileExistsError):
            build_table(self.input, self.emis, self.grid, self.output)

    def test_parallel_requires_explicit_opt_in_and_never_sorts_grid_alone(self):
        with self.assertRaisesRegex(ValueError, 'sequential'):
            build_table(self.input, self.emis, self.grid, self.output)
        rows = self.grid.read_text().splitlines()
        rows[1], rows[2] = rows[2], rows[1]
        self.grid.write_text('\n'.join(rows) + '\n')
        with self.assertRaisesRegex(ValueError, 'model-index order'):
            build_table(self.input, self.emis, self.grid, self.output, model_order='cloudy-c25-index')

    def test_blank_input_incomplete_emission_and_unproven_wavelengths_raise(self):
        original = self.input.read_text()
        self.input.write_text(original + '\n')
        with self.assertRaisesRegex(ValueError, 'Blank'):
            build_table(self.input, self.emis, self.grid, self.output, model_order='cloudy-c25-index')
        self.input.write_text(original)
        with self.assertRaisesRegex(ValueError, 'provenance note'):
            build_table(self.input, self.emis, self.grid, self.output,
                        model_order='cloudy-c25-index', resolved_wavelengths=[18.9689, 12.1321])
        self.emis.write_text('\n'.join(self.emis.read_text().splitlines()[:-1]) + '\n')
        with self.assertRaisesRegex(ValueError, 'one .emis row'):
            build_table(self.input, self.emis, self.grid, self.output, model_order='cloudy-c25-index')


if __name__ == '__main__':
    unittest.main()
