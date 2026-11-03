"""Check grouped YAML inputs and a tiny snapshot-to-Cloudy-HDF5 run."""

from copy import deepcopy
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

import h5py
import numpy as np
import yaml

from Pipeline.scr import input as settings, pipeline, simba
from Pipeline.scr.tables import cloudy_hdf5


def example_parameters():
    """Provide a small grouped configuration with inspectable unit constants.

    Returns: config (dict): editable YAML-compatible settings.
    """
    return {
        'source': {'snapshot_filename': 'snapshot.hdf5', 'catalogue_filename': 'catalogue.hdf5'},
        'selection': {'group_type': 'halo', 'target_index': 0},
        'unit_factors': dict(solar_mass_g=1.988416e33, kpc_to_cm=3.0856775814913673e21,
                             specific_energy_to_erg_g=1e10, z_sun_mass_fraction=0.0134),
        'emissivity': {'type': 'cloudy_hdf5', 'table_filename': 'table.hdf5',
                       'line_labels': ['O  8 18.9689A']},
    }


class TestReadParameters(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.directory = Path(directory.name).resolve()
        self.filename = self.directory / 'run.yaml'

    def read(self, config):
        """Write a temporary YAML fixture and read the production parameters.

        Returns: parameters (dict): normalized runtime input groups.
        """
        self.filename.write_text(yaml.safe_dump(config), encoding='utf-8')
        return settings.read_parameters(self.filename)

    def test_defaults_relative_paths_and_auto_output_need_no_simulation_files(self):
        config = example_parameters()
        with patch.object(settings.h5py, 'File') as reader:
            parameters = self.read(config)
        reader.assert_not_called()
        self.assertEqual(parameters['source']['type'], 'simba')
        self.assertEqual(parameters['source']['redshift'], 'auto')
        self.assertEqual(parameters['source']['snapshot_filename'], str(self.directory / 'snapshot.hdf5'))
        self.assertEqual(parameters['emissivity']['line_labels'], ['O  8 18.9689A'])
        self.assertNotIn('grid_filename', parameters['emissivity'])
        self.assertEqual(parameters['cube']['center_kpc'], 'auto')
        self.assertEqual(parameters['cube']['cube_type'], 'ppv')
        self.assertAlmostEqual(parameters['unit_factors']['gamma'], 5 / 3)
        self.assertEqual(parameters['output']['filename'], str(self.directory / 'line_cubes_ppv_halo_0.hdf5'))

    def test_explicit_paths_numeric_redshift_edges_and_output(self):
        config = example_parameters()
        config['source']['redshift'] = 0.5
        config['cube'] = {'type': 'ppp', 'center_kpc': [1, 2, 3],
                          'x_edges_kpc': [-1, 1], 'map_half_width_kpc': 4, 'spatial_bins': 2}
        config['output'] = {'filename': 'products/cube'}
        parameters = self.read(config)
        self.assertEqual(parameters['source']['redshift'], 0.5)
        self.assertEqual(parameters['output']['filename'], str(self.directory / 'products/cube.hdf5'))
        cube = settings.get_cube_parameters(parameters['cube'])
        np.testing.assert_array_equal(cube['x_edges_kpc'], [-1, 1])
        np.testing.assert_array_equal(cube['z_edges_kpc'], [-4, 0, 4])
        self.assertNotIn('velocity_edges_kms', cube)

    def test_only_hdf5_models_are_accepted_and_grid_key_is_rejected(self):
        config = example_parameters()
        for model in ('ps20', 'cloudy_hdf5'):
            config['emissivity']['type'] = model
            self.assertEqual(self.read(config)['emissivity']['type'], model)
        config['emissivity']['type'] = 'cloudy_mvp'
        with self.assertRaisesRegex(ValueError, 'HDF5 tables only'):
            self.read(config)
        config['emissivity']['type'] = 'cloudy_hdf5'
        config['emissivity']['grid_filename'] = 'obsolete.grid'
        with self.assertRaisesRegex(ValueError, 'unknown keys.*grid_filename'):
            self.read(config)

    def test_invalid_or_unknown_settings_raise(self):
        for section, key, value in (
            ('selection', 'target_index', True), ('selection', 'group_type', 'galaxy'),
            ('unit_factors', 'solar_mass_g', -1), ('unit_factors', 'kpc_to_cm', '1e-28'),
            ('source', 'redshift', float('nan')), ('cube', 'periodic', 'false'),
            ('cube', 'type', 'image'), ('cube', 'map_half_wdith_kpc', 100),
            ('cube', 'center_kpc', [0, 1]), ('emissivity', 'line_labels', ''),
            ('emissivity', 'line_labels', []), ('emissivity', 'line_labels', ['same', 'same']),
            ('emissivity', 'line_labels', ['valid', '']), ('output', 'filename', 'old.npz'),
        ):
            with self.subTest(section=section, key=key):
                config = example_parameters()
                config.setdefault(section, {})[key] = value
                with self.assertRaises(ValueError):
                    self.read(config)
        config = example_parameters()
        del config['source']['snapshot_filename']
        with self.assertRaisesRegex(ValueError, 'missing keys'):
            self.read(config)
        with self.assertRaisesRegex(ValueError, 'grouped YAML'):
            self.read({'snapshot_filename': 'old.hdf5', 'cube_type': 'ppv'})

    def test_real_snapshot_conversion_cloudy_interpolation_and_both_cube_modes(self):
        sample = Path(__file__).parent / 'data/sample_snapshot.hdf5'
        with h5py.File(self.directory / 'table.hdf5', 'w') as table:
            table.attrs['schema'] = cloudy_hdf5.SCHEMA
            table.attrs['abundance_model'] = 'fixed test abundances'
            table['axes/log_temperature_K'] = [0, 12]
            table['axes/log_nH_cm3'] = [-20, 10]
            labels = ['O  8 18.9689A', 'Ne10 12.1300A']
            table.create_dataset('lines/labels', data=labels, dtype=h5py.string_dtype())
            data = table.create_dataset('lines/emissivity_erg_s_cm3', data=np.tile([1e-30, 3e-30], (2, 2, 1)))
            data.attrs['unit'] = 'erg s^-1 cm^-3'
            data.attrs['axis_order'] = 'log_temperature_K, log_nH_cm3, line'
        catalogue = MagicMock()
        catalogue.__enter__.return_value = catalogue
        catalogue.redshift = 0
        catalogue.haloes.__len__.return_value = 1
        catalogue.haloes.get_particle_indices.return_value = np.array([1, 0])
        catalogue.haloes.get_dataset.side_effect = lambda name, **kwargs: {
            'r200m': np.array([1e6]), 'minpot_pos': np.array([[0, 0, 0]]),
        }[name]
        octavius = SimpleNamespace(load_catalogue=lambda filename: catalogue)
        config = example_parameters()
        config['emissivity']['line_labels'] = list(reversed(labels))
        config['source']['snapshot_filename'] = str(sample.resolve())
        for mode in ('ppv', 'ppp'):
            with self.subTest(mode=mode):
                case = deepcopy(config)
                case['cube'] = {'type': mode, 'center_kpc': [0, 0, 0], 'map_half_width_kpc': 1e6,
                                'spatial_bins': 1, 'velocity_half_width_kms': 1e6, 'velocity_bins': 1}
                case['output'] = {'filename': f'cube_{mode}.hdf5'}
                parameters = self.read(case)
                with (patch.dict('sys.modules', {'octavius': octavius}),
                      patch.object(cloudy_hdf5, 'read_cloudy_hdf5_table', wraps=cloudy_hdf5.read_cloudy_hdf5_table) as table_reader,
                      patch.object(simba, 'load_physical_gas_fields', wraps=simba.load_physical_gas_fields) as gas_reader,
                      self.assertWarnsRegex(RuntimeWarning, 'fixed Cloudy default abundances')):
                    pipeline.main(self.filename)
                table_reader.assert_called_once()
                gas_reader.assert_called_once()
                units = simba.compute_unit_factors(str(sample), parameters['unit_factors'])
                physical = simba.load_physical_gas_fields(str(sample), np.array([1, 0]), units)
                with h5py.File(parameters['output']['filename'], 'r') as result:
                    self.assertEqual(result.attrs['selected_particle_count'], 2)
                    self.assertEqual(result.attrs['line_count'], 2)
                    self.assertTrue(result.attrs['complete'])
                    self.assertEqual(list(result['lines/labels'].asstr()[...]), list(reversed(labels)))
                    self.assertEqual(result['provenance'].attrs['config_filename'], str(self.filename))
                    for index, factor in enumerate((3, 1)):
                        line = result[f'lines/line_{index:04d}']
                        cube = line['cube_erg_s'][...]
                        self.assertEqual(cube.shape, (1, 1, 1))
                        self.assertEqual(line['cube_erg_s'].attrs['unit'], 'erg s^-1')
                        np.testing.assert_allclose(cube.sum(), physical['volume_cm3'].sum() * factor * 1e-30)
                        self.assertEqual(line.attrs['clipped_particle_count'], 0)
                        if mode == 'ppp':
                            np.testing.assert_allclose(line['volume_emissivity_erg_s_cm3'][...]
                                                       * result['geometry/voxel_volume_cm3'][...], cube)


if __name__ == '__main__':
    unittest.main()
