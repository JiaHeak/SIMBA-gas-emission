"""Check multi-line ordering, clipping/coverage, and complete HDF5 publication."""

from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import warnings

import h5py
import numpy as np
import yaml

from Pipeline.scr import input as settings, pipeline, output
from Pipeline.scr.tables import cloudy_hdf5
from Pipeline.tests.test_read_parameters import example_parameters


class TestMultilineCubeOutput(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name).resolve()
        self.labels = ['line A / exact label', 'line B']
        self.table_path = self.root / 'table.hdf5'
        with h5py.File(self.table_path, 'w') as table:
            table.attrs['schema'] = cloudy_hdf5.SCHEMA
            table.attrs['abundance_model'] = 'fixed test abundances'
            table['axes/log_temperature_K'] = [6.0, 7.0]
            table['axes/log_nH_cm3'] = [-5.0, -4.0]
            table.create_dataset('lines/labels', data=self.labels, dtype=h5py.string_dtype())
            base = 10.0 ** np.array([[-30.0, -28.0], [-29.0, -27.0]])
            data = table.create_dataset('lines/emissivity_erg_s_cm3', data=np.stack((base, base * 3), axis=-1))
            data.attrs['unit'] = 'erg s^-1 cm^-3'
            data.attrs['axis_order'] = 'log_temperature_K, log_nH_cm3, line'
        self.config = example_parameters()
        self.config['emissivity'].update(table_filename=str(self.table_path),
                                         line_labels=list(reversed(self.labels)))
        self.config['cube'] = {'type': 'ppv', 'center_kpc': [0, 0, 0],
                               'map_half_width_kpc': 1, 'spatial_bins': 1,
                               'velocity_half_width_kms': 1, 'velocity_bins': 1}
        self.config['output'] = {'filename': str(self.root / 'cubes.hdf5')}
        self.config_path = self.root / 'config.yaml'
        self.config_path.write_text(yaml.safe_dump(self.config))
        self.parameters = settings.read_parameters(self.config_path)
        # Physical gas loading is mocked below; record the corresponding fixture units.
        self.parameters['physical_unit_factors'] = {}
        self.table = pipeline.load_emissivity_table('cloudy_hdf5', str(self.table_path), self.config['emissivity']['line_labels'])
        self.gas = {
            'T_K': 10.0 ** np.array([5.0, 8.0, 6.5]),
            'nH_cm3': 10.0 ** np.array([-6.0, -3.0, -4.5]),
            'volume_cm3': np.array([2.0, 3.0, 4.0]),
            'x_kpc': np.array([0.5, -0.5, 9.0]),
            'y_kpc': np.zeros(3), 'z_kpc': np.zeros(3), 'v_z_kms': np.zeros(3),
        }

    def cubes(self, mode='ppv'):
        """Run real emission and binning with only physical gas loading mocked.

        Returns: cubes (collections.abc.Iterator): two ordered cube results.
        """
        with patch.object(pipeline.simba, 'load_physical_gas_fields', return_value=deepcopy(self.gas)):
            yield from pipeline.run_line_emission_pipeline(
                'unused', np.array([2, 0, 1]), emissivity_table=self.table,
                redshift=0, unit_factors={}, center_kpc=[0, 0, 0],
                x_edges_kpc=[-1, 1], y_edges_kpc=[-1, 1],
                cube_type=mode, z_edges_kpc=[-1, 1], velocity_edges_kms=[-1, 1],
            )

    def test_order_units_clipping_and_luminosity_accounting_for_both_modes(self):
        for mode in ('ppv', 'ppp'):
            with self.subTest(mode=mode):
                parameters = deepcopy(self.parameters)
                parameters['cube']['cube_type'] = mode
                parameters['output']['filename'] = str(self.root / f'{mode}.hdf5')
                with warnings.catch_warnings(record=True) as messages:
                    warnings.simplefilter('always')
                    path = output.save_result(self.cubes(mode), parameters)
                self.assertEqual(sum('log_temperature_K clipping:' in str(m.message) for m in messages), 1)
                self.assertEqual(sum('log_nH_cm3 clipping:' in str(m.message) for m in messages), 1)
                with h5py.File(path) as saved:
                    self.assertEqual(saved.attrs['schema'], output.SCHEMA)
                    self.assertEqual(list(saved['lines/labels'].asstr()[...]), list(reversed(self.labels)))
                    for index, factor in enumerate((3, 1)):
                        line = saved[f'lines/line_{index:04d}']
                        self.assertEqual(line.attrs['line_label'], list(reversed(self.labels))[index])
                        self.assertEqual(line.attrs['clipped_particle_count'], 2)
                        self.assertEqual(line.attrs['included_particle_count'], 2)
                        self.assertEqual(line.attrs['excluded_particle_count'], 1)
                        included = factor * (2e-30 + 3e-27)
                        excluded = factor * 4 * 10.0 ** -28.5
                        np.testing.assert_allclose(line['cube_erg_s'][...].sum(), included, rtol=1e-12, atol=0)
                        np.testing.assert_allclose(line.attrs['excluded_luminosity_erg_s'], excluded, rtol=1e-12, atol=0)
                        np.testing.assert_allclose(line.attrs['input_luminosity_erg_s'], included + excluded, rtol=1e-12, atol=0)
                        self.assertIn('abundance_model', line.attrs)
                        if mode == 'ppp':
                            np.testing.assert_allclose(line['volume_emissivity_erg_s_cm3'][...]
                                                       * saved['geometry/voxel_volume_cm3'][...],
                                                       line['cube_erg_s'][...], rtol=1e-12, atol=0)
                self.assertFalse(list(self.root.glob('*.npz')))

    def test_missing_line_fails_before_snapshot_or_catalogue_selection(self):
        self.config['emissivity']['line_labels'] = ['missing line']
        self.config_path.write_text(yaml.safe_dump(self.config))
        with (patch.object(pipeline.simba, 'get_redshift') as read_snapshot,
              patch.object(pipeline.simba, 'select_gas_rows') as select):
            with self.assertRaisesRegex(ValueError, 'Requested lines absent'):
                pipeline.main(self.config_path)
        read_snapshot.assert_not_called()
        select.assert_not_called()
        self.assertFalse(Path(self.parameters['output']['filename']).exists())

    def test_failed_or_incomplete_run_leaves_no_final_file(self):
        def fail_after_first():
            """Simulate interruption after one successfully calculated line.

            Returns: cubes (collections.abc.Iterator): one cube followed by an error.
            """
            yield next(self.cubes())
            raise RuntimeError('simulated interruption')

        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            with self.assertRaisesRegex(RuntimeError, 'simulated interruption'):
                output.save_result(fail_after_first(), self.parameters)
            with self.assertRaisesRegex(ValueError, 'all requested line cubes'):
                output.save_result(iter([next(self.cubes())]), self.parameters)
        self.assertFalse(Path(self.parameters['output']['filename']).exists())
        self.assertFalse(list(self.root.glob('.line_cubes_*')))

    def test_existing_output_is_not_overwritten(self):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            path = output.save_result(self.cubes(), self.parameters)
        before = path.read_bytes()
        with self.assertRaises(FileExistsError):
            output.save_result(self.cubes(), self.parameters)
        self.assertEqual(path.read_bytes(), before)


if __name__ == '__main__':
    unittest.main()
