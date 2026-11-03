"""Check v2 producer/writer/reader agreement using unequal x/y/third sizes."""

from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from Pipeline.scr import pipeline, output
from Pipeline.scr.ppp_cube import make_ppp_cube
from Reader_Object import read_DataCube
from Reader_Object.reader import SCHEMA as READER_SCHEMA


class TestAxisV2RoundTrip(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.x = np.array([0., 1., 4.])
        self.y = np.array([0., 2., 5., 9.])
        self.third = np.array([-3., -1., 0., 2., 6.])
        self.image_x = np.array([0.5, 2., 0.5, 9.])
        self.image_y = np.array([1., 7., 3., 1.])
        self.depth = np.array([-2., 4., 1., -2.])
        self.luminosity = np.array([2., 7., 3., 5.])
        self.labels = ['line A / exact', 'line B']
        self.expected = np.zeros((2, 3, 4))
        self.expected[0, 0, 0] = 2
        self.expected[1, 2, 3] = 7
        self.expected[0, 1, 2] = 3
        self.table = {'table_type': 'cloudy_hdf5', 'table_filename': 'unused.hdf5',
                      'line_labels': self.labels, 'data': {'axes': {
                          'log_temperature_K': [6., 7.], 'log_nH_cm3': [-5., -4.],
                      }}}
        self.parameters = {
            'config_filename': 'unused.yaml',
            'source': {'snapshot_filename': 'unused.hdf5', 'catalogue_filename': 'unused.hdf5'},
            'selection': {'group_type': 'halo', 'target_index': 0}, 'unit_factors': {},
            'emissivity': {'type': 'cloudy_hdf5', 'table_filename': 'unused.hdf5', 'line_labels': self.labels},
            'output': {'filename': str(self.root / 'cubes.hdf5')},
        }

    def test_full_round_trip_and_projection_for_ppp_and_all_ppv_directions(self):
        self.assertEqual(output.SCHEMA, 'codex_simba_line_cubes_v2')
        self.assertEqual(output.SCHEMA, READER_SCHEMA)
        for mode, los in (('ppp', 'z'), ('ppv', 'z'), ('ppv', 'y'), ('ppv', 'x')):
            with self.subTest(mode=mode, los=los):
                gas = {f'{axis}_kpc': np.zeros(4) for axis in 'xyz'}
                if mode == 'ppp':
                    gas.update(x_kpc=self.image_x, y_kpc=self.image_y, z_kpc=self.depth)
                else:
                    image_axes = {'z': ('x', 'y'), 'y': ('x', 'z'), 'x': ('y', 'z')}[los]
                    gas[f'{image_axes[0]}_kpc'] = self.image_x
                    gas[f'{image_axes[1]}_kpc'] = self.image_y
                    gas[f'v_{los}_kms'] = self.depth + 10.
                parameters = deepcopy(self.parameters)
                parameters['output']['filename'] = str(self.root / f'{mode}_{los}.hdf5')
                emissions = [dict(line_label=label, line_luminosity_erg_s=self.luminosity * factor)
                             for label, factor in zip(self.labels, (1, 3))]
                with (patch.object(pipeline.simba, 'load_physical_gas_fields', return_value=deepcopy(gas)),
                      patch.object(pipeline, 'iter_particle_line_luminosities', return_value=emissions)):
                    results = pipeline.run_line_emission_pipeline(
                        'unused', np.arange(4), emissivity_table=self.table,
                        redshift=0, unit_factors={}, center_kpc=[0, 0, 0],
                        x_edges_kpc=self.x, y_edges_kpc=self.y, z_edges_kpc=self.third,
                        velocity_edges_kms=self.third, cube_type=mode, los_axis=los,
                        systemic_velocity_kms=10.,
                    )
                    path = output.save_result(results, parameters)
                reader = read_DataCube(path)
                expected = self.expected
                if mode == 'ppp':
                    # Independent reflecting transition matrices for each axis.
                    matrices = []
                    for size in self.expected.shape:
                        matrix = np.diag(np.full(size, .75))
                        matrix += np.diag(np.full(size - 1, .125), 1)
                        matrix += np.diag(np.full(size - 1, .125), -1)
                        matrix[0, 0] += .125
                        matrix[-1, -1] += .125
                        matrices.append(matrix)
                    expected = np.einsum('ai,bj,ck,ijk->abc', *matrices, self.expected)
                expected_order = 'x,y,z' if mode == 'ppp' else 'x,y,velocity'
                self.assertEqual(reader.shape, (2, 3, 4))
                self.assertEqual(reader.axis_order, expected_order)
                for index, factor in enumerate((1, 3)):
                    values = reader.get_fields('luminosity', lines=index)
                    np.testing.assert_array_equal(values, expected * factor)
                    np.testing.assert_array_equal(
                        reader.get_fields('luminosity', lines=index, selection=(slice(None), slice(None), 3)),
                        expected[:, :, 3] * factor,
                    )
                    self.assertEqual(reader.line_metadata[self.labels[index]]['included_luminosity_erg_s'], 12 * factor)
                    self.assertEqual(reader.line_metadata[self.labels[index]]['excluded_luminosity_erg_s'], 5 * factor)
                with h5py.File(path) as saved:
                    self.assertEqual(saved.attrs['schema'], READER_SCHEMA)
                    self.assertEqual(saved['lines/line_0000/cube_erg_s'].attrs['axis_order'], expected_order)
                    if mode == 'ppp':
                        self.assertTrue(saved.attrs['smoothing_enabled'])
                        self.assertEqual(saved.attrs['smoothing_sigma_voxels'], .5)
                        np.testing.assert_array_equal(saved.attrs['smoothing_kernel_1d'], [.125, .75, .125])
                        self.assertIn('smoothing', saved.attrs['deposition'])
                        volume = (np.diff(self.x)[:, None, None] * np.diff(self.y)[None, :, None]
                                  * np.diff(self.third)[None, None, :] * (3.0856775814913673e21 ** 3))
                        np.testing.assert_allclose(saved['geometry/voxel_volume_cm3'][...], volume)
                        self.assertEqual(saved['geometry/voxel_volume_cm3'].attrs['axis_order'], 'x,y,z')
                        np.testing.assert_allclose(reader.get_fields('emission', lines=0) * volume,
                                                   expected, rtol=1e-12, atol=0)
                    else:
                        self.assertNotIn('smoothing_enabled', saved.attrs)
                        self.assertIn('no smoothing', saved.attrs['deposition'])
                fig, panels = reader.plot(lines=0, log=False, show=False)
                try:
                    mesh = panels[0].collections[0]
                    np.testing.assert_array_equal(np.asarray(mesh.get_array()).reshape(3, 2), expected.sum(axis=2).T)
                    np.testing.assert_array_equal(mesh.get_coordinates()[0, :, 0], self.x)
                    np.testing.assert_array_equal(mesh.get_coordinates()[:, 0, 1], self.y)
                finally:
                    plt.close(fig)

    def test_writer_rejects_old_order_and_mismatched_volume_without_publishing(self):
        gas = dict(x_kpc=self.image_x, y_kpc=self.image_y, z_kpc=self.depth)
        result = make_ppp_cube(gas, self.luminosity, self.x, self.y, self.third)
        result.update(cube_type='ppp', line_label=self.labels[0])
        result['cube_axis_order'] = 'z,y,x'
        with self.assertRaisesRegex(ValueError, 'Version 2 requires'):
            output.save_result(iter([result]), self.parameters)
        result['cube_axis_order'] = 'x,y,z'
        result['voxel_volume_cm3'] = result['voxel_volume_cm3'].transpose(2, 1, 0)
        with self.assertRaisesRegex(ValueError, 'voxel volumes must match'):
            output.save_result(iter([result]), self.parameters)
        self.assertFalse(Path(self.parameters['output']['filename']).exists())
        self.assertFalse(list(self.root.glob('.line_cubes_*')))


if __name__ == '__main__':
    unittest.main()
