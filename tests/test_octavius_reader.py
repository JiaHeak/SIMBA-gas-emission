"""Check Octavius membership selection and its controller wiring."""

import importlib.util
import yaml
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, Mock, patch

import h5py
import numpy as np

from Pipeline.scr import pipeline, simba
from Pipeline.tests.cube_fixtures import mock_emissivity_table


SAMPLE_SNAPSHOT = Path(__file__).resolve().parent / 'data' / 'sample_snapshot.hdf5'
OCTAVIUS_SPEC = importlib.util.find_spec('octavius')
HAS_OCTAVIUS = OCTAVIUS_SPEC is not None and OCTAVIUS_SPEC.loader is not None


class TestOctaviusSelection(unittest.TestCase):
    def test_all_selection_helpers_reject_galaxy_before_reading_data(self):
        loader = Mock()
        with patch.dict('sys.modules', {'octavius': SimpleNamespace(load_catalogue=loader)}):
            for function, extra in (
                (simba.select_gas_rows, {}),
                (simba.get_reference_frame, {}),
                (simba.select_cgm_gas_rows, {
                    'snapshot_filename': 'unused.hdf5',
                    'gas_rows': np.array([0]), 'unit_factors': {},
                }),
            ):
                with self.subTest(function=function.__name__):
                    with self.assertRaisesRegex(ValueError, 'group_type must be halo'):
                        function(catalogue_filename='unused.hdf5', target_index=0,
                                 group_type='galaxy', **extra)
        loader.assert_not_called()

    def test_invalid_group_and_index_raise_before_catalogue_read(self):
        loader = Mock()
        module = SimpleNamespace(load_catalogue=loader)
        with patch.dict('sys.modules', {'octavius': module}):
            for group_type, index in [('unknown', 0), ('halo', -1), ('galaxy', True), ('halo', 0.5)]:
                with self.subTest(group_type=group_type, index=index):
                    with self.assertRaises(ValueError):
                        simba.select_gas_rows('catalogue.hdf5', index, group_type)
        loader.assert_not_called()

    def test_halo_membership_keeps_snapshot_row_order_and_closes_file(self):
        catalogue = MagicMock()
        catalogue.__enter__.return_value = catalogue
        catalogue.haloes.__len__.return_value = 2
        catalogue.haloes.get_particle_indices.return_value = np.array([6, 1])
        loader = Mock(return_value=catalogue)
        with patch.dict('sys.modules', {'octavius': SimpleNamespace(load_catalogue=loader)}):
            rows = simba.select_gas_rows('mock_octavius.hdf5', 1)

        np.testing.assert_array_equal(rows, [6, 1])
        loader.assert_called_once_with(Path('mock_octavius.hdf5'))
        catalogue.haloes.get_particle_indices.assert_called_once_with(
            group_index=1, ptype='gas',
        )
        catalogue.__exit__.assert_called_once()

    def test_repeated_membership_rows_raise(self):
        catalogue = MagicMock()
        catalogue.__enter__.return_value = catalogue
        catalogue.haloes.__len__.return_value = 1
        catalogue.haloes.get_particle_indices.return_value = np.array([1, 1])
        module = SimpleNamespace(load_catalogue=Mock(return_value=catalogue))
        with patch.dict('sys.modules', {'octavius': module}):
            with self.assertRaises(AssertionError):
                simba.select_gas_rows('mock_octavius.hdf5', 0)

    @unittest.skipUnless(HAS_OCTAVIUS, 'Real Octavius reader is not installed in this environment.')
    def test_installed_octavius_reads_tiny_hdf5_membership(self):
        with tempfile.TemporaryDirectory() as directory:
            filename = Path(directory) / 'catalogue.hdf5'
            with h5py.File(filename, 'w') as table:
                header = table.create_group('header')
                header['redshift'] = 0.0
                header['scale_factor'] = 1.0
                header['boxsize'] = 100.0
                metadata = table.create_group('metadata')
                metadata.attrs['simulation_type'] = 'SIMBA'
                metadata.attrs['octavius_version'] = 'test'
                metadata.attrs['timestamp'] = 'test'
                metadata.create_group('config_parameters').attrs['halo_id_source'] = 'SNAPSHOT'
                halo = table.create_group('halo_data')
                halo['HaloID'] = [0, 1]
                halo.create_group('properties')
                membership = halo.create_group('membership')
                membership['gas_offsets'] = [0, 2, 3]
                membership['gas_indices'] = [6, 1, 3]

            rows = simba.select_gas_rows(str(filename), 0)
            np.testing.assert_array_equal(rows, [6, 1])
            self.assertEqual(simba.get_catalogue_redshift(str(filename)), 0.0)
            with self.assertRaises(IndexError):
                simba.select_gas_rows(str(filename), 2)


class TestOctaviusController(unittest.TestCase):
    def test_legacy_selection_schema_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            config_path = Path(directory) / 'config.yaml'
            config_path.write_text(yaml.safe_dump({
                'snapshot_filename': 'unused.hdf5',
                'selection': {'type': 'legacy', 'catalogue_filename': 'unused.hdf5'},
            }), encoding='utf-8')
            with patch('sys.argv', ['pipeline.py', str(config_path)]):
                with self.assertRaisesRegex(ValueError, 'grouped YAML'):
                    pipeline.main()

    def run_controller(self, directory, catalogue_redshift):
        """Run the controller with mocked catalogue membership and line emission.

        Returns: result (tuple): output path and the halo-selector mock.
        """
        directory = Path(directory)
        config = {
            'source': {'snapshot_filename': str(SAMPLE_SNAPSHOT),
                       'catalogue_filename': 'octavius.hdf5'},
            'selection': {
                'group_type': 'halo',
                'target_index': 2,
            },
            'emissivity': {
                'type': 'ps20', 'table_filename': 'ps20.hdf5',
                'line_labels': ['H  1      6562.81A'],
            },
            'unit_factors': {
                'solar_mass_g': 1.988416e33, 'kpc_to_cm': 3.0856775814913673e21,
                'specific_energy_to_erg_g': 1.0, 'z_sun_mass_fraction': 0.02,
            },
            'cube': {'center_kpc': [0, 0, 0], 'x_edges_kpc': [-100, 100],
                     'y_edges_kpc': [-100, 100], 'velocity_edges_kms': [-1000, 1000]},
            'output': {'filename': 'cube.hdf5'},
        }
        config_path = directory / 'config.yaml'
        config_path.write_text(yaml.safe_dump(config), encoding='utf-8')
        with (
            patch.object(pipeline, 'load_emissivity_table', side_effect=mock_emissivity_table),
            patch.object(simba, 'select_gas_rows', return_value=np.array([1, 6])) as select,
            patch.object(simba, 'select_cgm_gas_rows', side_effect=lambda **kw: kw['gas_rows']),
            patch.object(simba, 'get_catalogue_redshift', return_value=catalogue_redshift),
            patch.object(pipeline, 'iter_particle_line_luminosities', return_value=[{
                'line_label': 'H  1      6562.81A', 'line_luminosity_erg_s': np.array([2.0, 3.0]),
            }]),
            patch('sys.argv', ['pipeline.py', str(config_path)]),
        ):
            pipeline.main()
        return directory / 'cube.hdf5', select

    def test_controller_uses_halo_index_and_saves_cube(self):
        with tempfile.TemporaryDirectory() as directory:
            output, select = self.run_controller(directory, 0.0)
            select.assert_called_once_with(catalogue_filename=str((Path(directory) / 'octavius.hdf5').resolve()), target_index=2, group_type='halo')
            with h5py.File(output) as data:
                self.assertEqual(data['lines/line_0000/cube_erg_s'].shape, (1, 1, 1))
                self.assertAlmostEqual(float(data['lines/line_0000/cube_erg_s'][...].sum()), 5.0)

    def test_catalogue_snapshot_redshift_mismatch_raises(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, 'different redshifts'):
                self.run_controller(directory, 8.0)
            self.assertFalse((Path(directory) / 'cube.hdf5').exists())


if __name__ == '__main__':
    unittest.main()
