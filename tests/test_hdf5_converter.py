"""Small scientific-format checks for the N-axis Cloudy converter."""

import itertools
import json
from pathlib import Path
import tempfile
import unittest
import warnings

import h5py
import numpy as np

from Pipeline.scr.tables.hdf5_converter import (
    SCHEMA, convert_to_hdf5, read_cloudy_input,
)


class TestCloudyConverter(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.specifications = {
            'temperature': ('coronal 4 log vary', 'grid range from 4 to 4.2 step 0.2',
                            'COROnal e', np.array([4.0, 4.2])),
            'density': ('hden -5 vary', 'grid range from -5 to -4 step 0.5',
                        'HDEN=%f L', np.array([-5.0, -4.5, -4.0])),
            'metals': ('metals -2 log vary', 'grid range from -2 to -0.5 step 0.5',
                       'METALS= %f', np.array([-2., -1.5, -1., -0.5])),
            'ionization': ('ionization parameter -3 vary', 'grid range from -3 to -2 step 1',
                           'IONIzation', np.array([-3., -2.])),
        }
        self.labels = ['O  8 18.9689A', 'O  3 5006.84A']

    def fixture(self, order=('temperature', 'density', 'metals'), *, parallel=False):
        name = '_'.join(order)
        input_path, grid_path, emis_path = [self.root / (name + suffix) for suffix in ('.in', '.grd', '.emis')]
        commands = ['title Synthetic format fixture', 'abundances GASS10 no grains']
        for key in order:
            commands.extend(self.specifications[key][:2])
        commands.extend(['table HM12 redshift 0', 'CMB redshift 0', 'set dr 0',
                         'stop zone 1', 'iterate to convergence',
                         'save grid "fixture.grd" last no hash',
                         'save lines emissivity "fixture.emis" last no hash',
                         *self.labels, 'end of lines',
                         '##### Your Notes #####', '# Synthetic values; not measured emissivities.'])
        input_path.write_text('\n'.join(commands) + '\n')
        points = list(itertools.product(*(self.specifications[key][3] for key in order)))[::-1]
        grid_header = ['#Index', 'Failure?', 'Warnings?', 'Exit code', '#rank', '#seq',
                       *(self.specifications[key][2] for key in order), 'grid parameter string']
        grid_rows, emis_rows = [], []
        expected = []
        for index, point in enumerate(points):
            coordinates = dict(zip(order, point))
            # An independently defined value distinguishes axis permutations.
            luminosity = (100 * coordinates['temperature'] + 10 * coordinates['density']
                          + coordinates.get('metals', 0) + 0.1 * coordinates.get('ionization', 0))
            values = [luminosity * 1e-30, (luminosity + 7) * 1e-30]
            rank, seq = (index % 2, index // 2) if parallel else (0, index)
            grid_rows.append([str(index), 'F', 'F', 'ok', str(rank), str(seq),
                              *(f'{value:.6f}' for value in point), ', '.join(map(str, point))])
            emis_rows.append(['0.5', *(f'{value:.16e}' for value in values)])
            expected.append(values)
        grid_path.write_text('\t'.join(grid_header) + '\n' + '\n'.join('\t'.join(row) for row in grid_rows) + '\n')
        emis_path.write_text('#depth\t' + '\t'.join(self.labels) + '\n'
                             + '\n'.join('\t'.join(row) for row in emis_rows) + '\n')
        return input_path, grid_path, emis_path, np.array(expected)

    def test_current_sample_input_and_notes(self):
        sample = Path(__file__).resolve().parents[2] / 'Cloudy/simba_cgm_lines.in'
        model = read_cloudy_input(sample)
        self.assertEqual([len(axis['values']) for axis in model['axes']], [33, 23, 8])
        self.assertEqual(len(model['requested_lines']), 12)
        self.assertIn('You may write about the physics', model['user_notes'])
        self.assertEqual([record['component'] for record in model['physics'] if record['kind'] == 'radiation'], ['HM12', 'CMB'])

    def test_three_axis_permutation_preserves_values_and_provenance(self):
        stored = []
        for order in [('temperature', 'density', 'metals'), ('metals', 'density', 'temperature')]:
            input_path, grid, emis, expected = self.fixture(order, parallel=True)
            output = self.root / ('_'.join(order) + '.hdf5')
            convert_to_hdf5(input_path, grid, emis, output,
                             model_order='cloudy-c25-index', cloudy_version='25.00')
            with h5py.File(output) as table:
                self.assertEqual(table.attrs['schema'], SCHEMA)
                self.assertTrue(table.attrs['complete'])
                self.assertEqual(table['lines/emissivity_erg_s_cm3'].shape, (2, 3, 4, 2))
                actual = table['lines/emissivity_erg_s_cm3'][...]
                indices = table['models/storage_indices'][...]
                np.testing.assert_array_equal(actual[tuple(indices.T)], expected)
                np.testing.assert_array_equal(table['models/sequence'][...], np.arange(24) // 2)
                self.assertEqual(table['provenance/input_text'].asstr()[()], input_path.read_text())
                self.assertIn('# Synthetic values', table['provenance/user_notes'].asstr()[()])
                self.assertEqual(list(table['provenance/input_axis_order'].asstr()[...]),
                                 [read_cloudy_input(input_path)['axes'][i]['name'] for i in range(3)])
                physics = json.loads(table['physics/parsed_commands_json'].asstr()[()])
                self.assertEqual(next(record['reference'] for record in physics if record['kind'] == 'abundances'), 'GASS10')
                self.assertEqual([dimension.label for dimension in table['lines/emissivity_erg_s_cm3'].dims],
                                 ['log_temperature_K', 'log_nH_cm3', 'log_metal_scale', 'line'])
                stored.append(actual)
            with self.assertRaises(FileExistsError):
                convert_to_hdf5(input_path, grid, emis, output)
        np.testing.assert_array_equal(*stored)

    def test_two_and_four_axes_with_explicit_extra_axis(self):
        extra = {'ionization parameter': {
            'name': 'log_ionization_parameter', 'unit': 'dex', 'scale': 'log10',
            'grid_header': '^IONI', 'definition': 'Synthetic log10 ionization parameter.'}}
        for order, shape, definitions in [
            (('density', 'temperature'), (2, 3, 2), None),
            (('ionization', 'metals', 'temperature', 'density'), (2, 3, 4, 2, 2), extra),
        ]:
            input_path, grid, emis, expected = self.fixture(order)
            output = self.root / ('_'.join(order) + '.hdf5')
            convert_to_hdf5(input_path, grid, emis, output, axis_definitions=definitions)
            with h5py.File(output) as table:
                values = table['lines/emissivity_erg_s_cm3'][...]
                self.assertEqual(values.shape, shape)
                np.testing.assert_array_equal(values[tuple(table['models/storage_indices'][...].T)], expected)
                self.assertEqual(table.attrs['cloudy_version'], 'unconfirmed')

    def test_zeros_preserved_and_unknown_physics_not_promoted_from_notes(self):
        input_path, grid, emis, _ = self.fixture()
        input_path.write_text(input_path.read_text() + '# No radiation, regardless of commands above.\n'
                              'print short\n')
        rows = emis.read_text().splitlines()
        cells = rows[1].split('\t')
        cells[1] = '0'
        rows[1] = '\t'.join(cells)
        emis.write_text('\n'.join(rows) + '\n')
        output = self.root / 'zero.hdf5'
        with self.assertWarnsRegex(RuntimeWarning, 'zero emissivities'):
            convert_to_hdf5(input_path, grid, emis, output)
        with h5py.File(output) as table:
            self.assertEqual(table.attrs['zero_emissivity_count'], 1)
            self.assertFalse(table.attrs['log_interpolation_ready'])
            self.assertIn('# No radiation', table['provenance/user_notes'].asstr()[()])
            self.assertNotIn('print short', table['provenance/user_notes'].asstr()[()])
            physics = json.loads(table['physics/parsed_commands_json'].asstr()[()])
            self.assertIn('HM12', [record.get('component') for record in physics])
            self.assertEqual(physics[-1]['kind'], 'uninterpreted')

    def test_invalid_outputs_do_not_publish(self):
        input_path, grid, emis, _ = self.fixture(parallel=True)
        original_grid, original_emis = grid.read_text(), emis.read_text()
        for failure in ('parallel', 'missing', 'duplicate', 'failed', 'line', 'negative'):
            with self.subTest(failure=failure):
                grid.write_text(original_grid)
                emis.write_text(original_emis)
                grid_rows = original_grid.splitlines()
                emis_rows = original_emis.splitlines()
                if failure == 'missing':
                    grid.write_text('\n'.join(grid_rows[:-1]) + '\n')
                elif failure == 'duplicate':
                    a, b = grid_rows[1].split('\t'), grid_rows[2].split('\t')
                    b[6:] = a[6:]
                    grid_rows[2] = '\t'.join(b)
                    grid.write_text('\n'.join(grid_rows) + '\n')
                elif failure == 'failed':
                    grid.write_text(original_grid.replace('\tF\tF\tok\t', '\tT\tF\tfailed\t', 1))
                elif failure == 'line':
                    emis.write_text(original_emis.replace('O  8 18.9689A', 'O  7 18.9689A', 1))
                elif failure == 'negative':
                    cells = emis_rows[1].split('\t')
                    cells[1] = '-1e-30'
                    emis_rows[1] = '\t'.join(cells)
                    emis.write_text('\n'.join(emis_rows) + '\n')
                output = self.root / (failure + '.hdf5')
                with self.assertRaises(ValueError):
                    convert_to_hdf5(input_path, grid, emis, output, cloudy_version='25.00',
                                     model_order='sequential' if failure == 'parallel' else 'cloudy-c25-index')
                self.assertFalse(output.exists())

    def test_real_baseline_values_are_unchanged(self):
        root = Path(__file__).resolve().parents[2] / 'Cloudy'
        raw = root / 'results/simba_hot_cgm_multiline_492637'
        output = self.root / 'real_baseline.hdf5'
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            convert_to_hdf5(raw / 'simba_hot_cgm_multiline.in', raw / '.grd', raw / '.emis', output,
                             model_order='cloudy-c25-index', cloudy_version='25.00')
        with h5py.File(output) as converted, h5py.File(root / 'simba_hot_cgm_multiline.hdf5') as baseline:
            np.testing.assert_array_equal(converted['lines/emissivity_erg_s_cm3'][...],
                                          baseline['lines/emissivity_erg_s_cm3'][...])
            self.assertEqual(list(converted['lines/labels'].asstr()[...]), list(baseline['lines/labels'].asstr()[...]))

    def test_c25_float32_coordinates_preserve_raw_values_and_reject_mismatches(self):
        self.specifications['temperature'] = (
            'coronal 2.2 log vary', 'grid range from 2.2 to 8.6 step 0.2',
            'COROnal e', 2.2 + np.arange(33) * .2)
        input_path, grid, emis, expected = self.fixture(('temperature', 'density'))
        original = grid.read_text()
        grid.write_text(original.replace('\t8.400000\t', '\t8.400001\t'))
        output = self.root / 'float32.hdf5'
        with self.assertWarnsRegex(RuntimeWarning, 'float32 range arithmetic'):
            convert_to_hdf5(input_path, grid, emis, output)
        with h5py.File(output) as table:
            raw = table['models/coordinates'][...]
            self.assertIn(8.400001, raw[:, 0])
            np.testing.assert_array_equal(table['lines/emissivity_erg_s_cm3'][...][
                tuple(table['models/storage_indices'][...].T)], expected)
            reports = json.loads(table['provenance/coordinate_match_reports_json'].asstr()[()])
            self.assertEqual(reports[0]['float32_only_row_count'], 3)
        grid.write_text(original.replace('\t8.400000\t', '\t8.400004\t'))
        with self.assertRaisesRegex(ValueError, 'coordinates disagree'):
            convert_to_hdf5(input_path, grid, emis, self.root / 'mismatched.hdf5')

    def test_log_version_and_unproven_wavelengths(self):
        input_path, grid, emis, _ = self.fixture(parallel=True)
        log = self.root / 'model.out'
        log.write_text('Cloudy 25.00\nSynthetic test log; not a Cloudy execution.\n')
        output = self.root / 'version.hdf5'
        convert_to_hdf5(input_path, grid, emis, output, out_filename=log,
                         model_order='cloudy-c25-index')
        with h5py.File(output) as table:
            self.assertEqual(table.attrs['cloudy_version'], '25.00')
            self.assertEqual(table.attrs['cloudy_version_source'], 'cloudy_output')
            self.assertEqual(table['provenance/sources/cloudy_output/text'].asstr()[()], log.read_text())
            self.assertIn('Unconfirmed', table['lines'].attrs['resolution_status'])
        with self.assertRaisesRegex(ValueError, 'Conflicting'):
            convert_to_hdf5(input_path, grid, emis, self.root / 'bad_version.hdf5',
                             out_filename=log, cloudy_version='23.01', model_order='cloudy-c25-index')
        with self.assertRaisesRegex(ValueError, 'provenance note'):
            convert_to_hdf5(input_path, grid, emis, self.root / 'bad_lines.hdf5',
                             model_order='cloudy-c25-index', cloudy_version='25.00',
                             resolved_wavelengths=[18.9689, 5006.84])


if __name__ == '__main__':
    unittest.main()
