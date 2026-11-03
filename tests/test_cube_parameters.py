"""Check automatic catalogue reference frames, grid settings, and periodic offsets."""

import yaml
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import h5py

from Pipeline.scr import pipeline, simba
from Pipeline.scr import input as pipeline_input
from Pipeline.tests.cube_fixtures import mock_emissivity_table


class TestCubeParameters(unittest.TestCase):
    def test_physical_box_size_uses_coordinate_conversion(self):
        with tempfile.TemporaryDirectory() as directory:
            filename = Path(directory) / 'snapshot.hdf5'
            with h5py.File(filename, 'w') as snapshot:
                snapshot.create_group('Header').attrs['BoxSize'] = 100
            self.assertEqual(pipeline_input.get_physical_box_size(filename, 0.5), 50)
            with self.assertRaises(ValueError):
                pipeline_input.get_physical_box_size(filename, -1)

    def test_automatic_settings_require_a_reference_frame(self):
        with self.assertRaisesRegex(ValueError, 'reference frame'):
            pipeline_input.get_cube_parameters({'center_kpc': 'auto'})

    def test_reference_frame_uses_selected_halo_and_physical_units(self):
        catalogue = MagicMock()
        catalogue.__enter__.return_value = catalogue
        catalogue.haloes.__len__.return_value = 2
        catalogue.haloes.get_dataset.side_effect = [
            np.array([[1, 2, 3]]), np.array([[10, 20, 30]]),
        ]
        loader = Mock(return_value=catalogue)
        with patch.dict('sys.modules', {'octavius': SimpleNamespace(load_catalogue=loader)}):
            frame = simba.get_reference_frame('catalogue.hdf5', 1, 'halo')
        np.testing.assert_array_equal(frame['center_kpc'], [1, 2, 3])
        np.testing.assert_array_equal(frame['bulk_velocity_kms'], [10, 20, 30])
        calls = catalogue.haloes.get_dataset.call_args_list
        self.assertEqual(calls[0].args, ('minpot_pos',))
        self.assertEqual(calls[0].kwargs['to_units'], 'kpc')
        self.assertTrue(calls[0].kwargs['to_physical'])
        np.testing.assert_array_equal(calls[0].kwargs['mask'], [1])
        self.assertEqual(calls[1].args, ('com_vel_total',))
        self.assertEqual(calls[1].kwargs['to_units'], 'km/s')
        catalogue.__exit__.assert_called_once()

    def test_generated_bins_and_velocity_component(self):
        frame = {'center_kpc': [1, 2, 3], 'bulk_velocity_kms': [10, 20, 30]}
        parameters = pipeline_input.get_cube_parameters({
            'center_kpc': 'auto', 'systemic_velocity_kms': 'auto',
            'los_axis': 'y', 'map_half_width_kpc': 100, 'spatial_bins': 2,
            'velocity_half_width_kms': 300, 'velocity_bins': 2,
        }, frame)
        np.testing.assert_array_equal(parameters['center_kpc'], [1, 2, 3])
        self.assertEqual(parameters['systemic_velocity_kms'], 20)
        np.testing.assert_array_equal(parameters['x_edges_kpc'], [-100, 0, 100])
        np.testing.assert_array_equal(parameters['velocity_edges_kms'], [-300, 0, 300])

    def test_explicit_settings_override_automatic_values(self):
        parameters = pipeline_input.get_cube_parameters({
            'center_kpc': [4, 5, 6], 'systemic_velocity_kms': 7,
            'x_edges_kpc': [-2, 0, 1], 'y_edges_kpc': [-3, 3],
            'velocity_edges_kms': [-10, 10], 'spatial_bins': 0,
        })
        np.testing.assert_array_equal(parameters['x_edges_kpc'], [-2, 0, 1])
        self.assertEqual(parameters['systemic_velocity_kms'], 7)

    def test_periodic_offsets_keep_gas_across_box_boundary(self):
        def gas_fields(*args):
            return {
                'x_kpc': np.array([0.5, 98.5]),
                'y_kpc': np.array([10.0, 10.0]),
                'z_kpc': np.array([10.0, 10.0]),
                'v_z_kms': np.array([5.0, 5.0]),
            }
        with (
            patch.object(simba, 'load_physical_gas_fields', side_effect=gas_fields),
            patch.object(pipeline, 'iter_particle_line_luminosities', return_value=[{
                'line_label': 'line', 'line_luminosity_erg_s': np.array([2.0, 3.0]),
            }]),
        ):
            result = next(pipeline.run_line_emission_pipeline(
                'snapshot.hdf5', [0, 1], emissivity_table=mock_emissivity_table('ps20', 'unused', ['line']),
                redshift=0, unit_factors={},
                center_kpc=[99.5, 10, 10], x_edges_kpc=[-2, 0, 2],
                y_edges_kpc=[-2, 2], velocity_edges_kms=[-1, 1],
                systemic_velocity_kms=5, box_size_kpc=100,
            ))
        np.testing.assert_allclose(result['cube_erg_s'], [[[3.0]], [[2.0]]])

    def test_halo_controller_resolves_automatic_values_and_output_name(self):
        snapshot = Path(__file__).resolve().parent / 'data' / 'sample_snapshot.hdf5'
        with tempfile.TemporaryDirectory() as directory:
            config = {
                'source': {'snapshot_filename': str(snapshot), 'catalogue_filename': 'catalogue.hdf5'},
                'selection': {'group_type': 'halo', 'target_index': 0},
                'emissivity': {'type': 'ps20', 'table_filename': 'table.hdf5',
                               'line_labels': ['H  1      6562.81A']},
                'unit_factors': dict(solar_mass_g=1.988416e33, kpc_to_cm=3.0856775814913673e21,
                                     specific_energy_to_erg_g=1, z_sun_mass_fraction=0.02),
                'cube': {'center_kpc': 'auto', 'systemic_velocity_kms': 'auto'},
                'output': {'filename': 'auto'},
            }
            path = Path(directory) / 'config.yaml'
            path.write_text(yaml.safe_dump(config), encoding='utf-8')
            with (
                patch.object(pipeline, 'load_emissivity_table', side_effect=mock_emissivity_table),
                patch.object(simba, 'get_catalogue_redshift', return_value=0),
                patch.object(simba, 'select_gas_rows', return_value=np.array([1, 6])) as select,
                patch.object(simba, 'select_cgm_gas_rows', side_effect=lambda **kw: kw['gas_rows']),
                patch.object(simba, 'get_reference_frame', return_value={
                    'center_kpc': np.array([0, 0, 0]), 'bulk_velocity_kms': [1, 2, 3],
                }) as frame,
                patch.object(pipeline, 'iter_particle_line_luminosities', return_value=[{
                    'line_label': 'H  1      6562.81A', 'line_luminosity_erg_s': np.array([2.0, 3.0]),
                }]),
                patch('sys.argv', ['pipeline.py', str(path)]),
            ):
                pipeline.main()
            select.assert_called_once_with(catalogue_filename=str((Path(directory) / 'catalogue.hdf5').resolve()), target_index=0, group_type='halo')
            self.assertEqual(frame.call_args.kwargs['group_type'], 'halo')
            output = Path(directory) / 'line_cubes_ppv_halo_0.hdf5'
            with h5py.File(output) as result:
                self.assertEqual(float(result['lines/line_0000/cube_erg_s'][...].sum()), 5.0)
                self.assertEqual(result.attrs['systemic_velocity_kms'], 3.0)


if __name__ == '__main__':
    unittest.main()
