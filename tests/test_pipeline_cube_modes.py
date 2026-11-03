"""Focused checks of YAML-to-cube routing and the shared physical frame."""

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import h5py
import yaml

from Pipeline.scr import input as settings
from Pipeline.scr import pipeline
from Pipeline.tests.cube_fixtures import mock_emissivity_table


class TestCubeModes(unittest.TestCase):
    def test_controller_saves_both_modes_with_shared_centering_and_periodicity(self):
        for mode in ('ppv', 'ppp'):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as directory:
                config = {
                    'source': {'snapshot_filename': 'snapshot.hdf5',
                               'catalogue_filename': 'catalogue.hdf5'},
                    'selection': {'group_type': 'halo', 'target_index': 0},
                    'emissivity': {'type': 'ps20', 'table_filename': 'table.hdf5',
                                   'line_labels': ['Halpha']},
                    'unit_factors': dict(solar_mass_g=1.988416e33, kpc_to_cm=3.0856775814913673e21,
                                         specific_energy_to_erg_g=1, z_sun_mass_fraction=0.02),
                    'cube': {'type': mode, 'center_kpc': 'auto', 'systemic_velocity_kms': 'auto',
                    'x_edges_kpc': [-2, 0, 2], 'y_edges_kpc': [-2, 2],
                    'z_edges_kpc': [-2, 2], 'velocity_edges_kms': [-2, 2],
                    'periodic': True}, 'output': {'filename': 'auto'},
                }
                path = Path(directory) / 'run.yaml'
                path.write_text(yaml.safe_dump(config), encoding='utf-8')
                gas = {'x_kpc': np.array([0.5, 98.5]),
                       'y_kpc': np.array([10.0, 10.0]),
                       'z_kpc': np.array([10.0, 15.0]),
                       'v_z_kms': np.array([1.0, 1.0])}
                with (
                    patch.object(pipeline, 'load_emissivity_table', side_effect=mock_emissivity_table),
                    patch.object(pipeline.simba, 'get_catalogue_redshift', return_value=0),
                    patch.object(pipeline.simba, 'get_redshift', return_value=0),
                    patch.object(pipeline.simba, 'compute_unit_factors', return_value={
                        'position_to_physical_kpc': 1, 'density_to_g_cm3': 1,
                    }),
                    patch.object(pipeline.simba, 'select_gas_rows', return_value=np.array([1, 6])),
                    patch.object(pipeline.simba, 'select_cgm_gas_rows', side_effect=lambda **kw: kw['gas_rows']),
                    patch.object(pipeline.simba, 'get_reference_frame', return_value={
                        'center_kpc': [99.5, 10, 10], 'bulk_velocity_kms': [0, 0, 1],
                    }),
                    patch.object(settings, 'get_physical_box_size', return_value=100),
                    patch.object(pipeline.simba, 'load_physical_gas_fields', return_value=gas) as loader,
                    patch.object(pipeline, 'iter_particle_line_luminosities', return_value=[{
                        'line_label': 'Halpha', 'line_luminosity_erg_s': np.array([2.0, 3.0]),
                    }]) as emission,
                    patch('sys.argv', ['pipeline.py', str(path)]),
                ):
                    pipeline.main()
                loader.assert_called_once()
                emission.assert_called_once()
                with h5py.File(Path(directory) / f'line_cubes_{mode}_halo_0.hdf5') as result:
                    self.assertEqual(result.attrs['cube_type'], mode)
                    self.assertEqual(result.attrs['selected_particle_count'], 2)
                    self.assertEqual(result['provenance'].attrs['config_filename'], str(path.resolve()))
                    line = result['lines/line_0000']
                    if mode == 'ppv':
                        np.testing.assert_allclose(line['cube_erg_s'][...], [[[3]], [[2]]])
                        self.assertEqual(result.attrs['cube_axis_order'], 'x,y,velocity')
                    else:
                        np.testing.assert_allclose(line['cube_erg_s'][...], [[[.25]], [[1.75]]])
                        self.assertEqual(result.attrs['cube_axis_order'], 'x,y,z')
                        self.assertEqual(line.attrs['excluded_particle_count'], 1)
                        np.testing.assert_allclose(
                            line['volume_emissivity_erg_s_cm3'][...] * result['geometry/voxel_volume_cm3'][...],
                            line['cube_erg_s'][...],
                        )
                        self.assertNotIn('velocity_edges_kms', result['axes'])

    def test_ppp_parameters_generate_depth_without_velocity_settings(self):
        config = {'cube_type': 'ppp', 'center_kpc': [0, 0, 0],
                  'map_half_width_kpc': 200, 'spatial_bins': 4,
                  'systemic_velocity_kms': 'auto'}
        result = settings.get_cube_parameters(config)
        np.testing.assert_array_equal(result['z_edges_kpc'], [-200, -100, 0, 100, 200])
        self.assertNotIn('systemic_velocity_kms', result)
        config['z_edges_kpc'] = [-20, 10, 30]
        np.testing.assert_array_equal(settings.get_cube_parameters(config)['z_edges_kpc'], [-20, 10, 30])

    def test_invalid_mode_or_missing_depth_fails_before_loading_snapshot(self):
        with patch.object(pipeline.simba, 'load_physical_gas_fields') as loader:
            for mode in ('invalid', 'ppp'):
                with self.subTest(mode=mode), self.assertRaises(ValueError):
                    list(pipeline.run_line_emission_pipeline(
                        'unused', [0], emissivity_table={},
                        redshift=0, unit_factors={},
                        center_kpc=[0, 0, 0], x_edges_kpc=[0, 1], y_edges_kpc=[0, 1],
                        cube_type=mode,
                    ))
            loader.assert_not_called()


if __name__ == '__main__':
    unittest.main()
