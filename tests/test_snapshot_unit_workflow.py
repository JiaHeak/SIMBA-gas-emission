"""Focused checks of header-derived SIMBA units and their pipeline propagation."""

from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch
import warnings

import h5py
import numpy as np
import yaml

from Pipeline.scr import input as settings, pipeline, simba
from Pipeline.scr.tables import cloudy_hdf5


class TestSnapshotUnitWorkflow(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.snapshot = self.root / 'snapshot.hdf5'
        # Deliberately simple synthetic constants expose accidental hard-coded units.
        self.constants = dict(solar_mass_g=2e33, kpc_to_cm=3e21,
                              specific_energy_to_erg_g=1e10, z_sun_mass_fraction=0.02)

    def write_header(self, a, h):
        """Write metadata without any particle arrays.

        Returns: result (None): replaces the temporary test header.
        """
        with h5py.File(self.snapshot, 'w') as snapshot:
            snapshot.create_group('Header').attrs.update(Time=a, HubbleParam=h,
                                                         Redshift=1 / a - 1, BoxSize=10.)

    def test_header_factors_and_volume_consistency(self):
        for a, h in [(1., 0.68), (0.25, 0.5)]:
            with self.subTest(a=a, h=h):
                self.write_header(a, h)
                units = simba.compute_unit_factors(self.snapshot, self.constants)
                self.assertAlmostEqual(units['position_to_physical_kpc'], a / h)
                self.assertAlmostEqual(units['velocity_to_kms'], np.sqrt(a))
                self.assertEqual(units['specific_energy_to_erg_g'], 1e10)
                np.testing.assert_allclose(units['mass_to_g'], 2e43 / h)
                np.testing.assert_allclose(units['density_to_g_cm3'], 2e43 / (3e21)**3 * h**2 / a**3)
                np.testing.assert_allclose(units['mass_to_g'] / units['density_to_g_cm3'],
                                           (a / h * self.constants['kpc_to_cm'])**3)
                self.assertEqual(units['scale_factor'], a)
                self.assertEqual(units['hubble_param'], h)

    def test_invalid_header_and_mixed_final_factors_raise(self):
        for a, h in [(1., 0.), (float('nan'), 0.68)]:
            self.write_header(1., 0.68)
            with h5py.File(self.snapshot, 'a') as snapshot:
                snapshot['Header'].attrs.update(Time=a, HubbleParam=h)
            with self.assertRaisesRegex(ValueError, 'scale factor'):
                simba.compute_unit_factors(self.snapshot, self.constants)
        with self.assertRaisesRegex(ValueError, 'unknown'):
            simba.compute_unit_factors(self.snapshot, {**self.constants, 'density_to_g_cm3': 1.})

    def test_cosmological_and_unit_stages_can_be_selected_independently(self):
        self.write_header(0.25, 0.5)
        expected_cosmology = dict(position=0.5, mass=2., density=16., velocity=0.5, specific_energy=1.)
        expected_units = dict(position=1., mass=2e43, density=2e43 / (3e21)**3,
                              velocity=1., specific_energy=1e10)
        for physical in (False, True):
            for units in (False, True):
                with self.subTest(to_physical=physical, to_units=units):
                    stages = simba.compute_conversion_factors(
                        self.snapshot, self.constants, to_physical=physical, to_units=units)
                    self.assertEqual(stages['cosmological_factors'], expected_cosmology)
                    self.assertEqual(stages['unit_conversions'].keys(), expected_units.keys())
                    for quantity in expected_units:
                        np.testing.assert_allclose(stages['unit_conversions'][quantity], expected_units[quantity], rtol=1e-14, atol=0)
                    self.assertIs(stages['to_physical'], physical)
                    self.assertIs(stages['to_units'], units)
                    for quantity, factor in stages['factors'].items():
                        expected = (expected_cosmology[quantity] if physical else 1.) * (expected_units[quantity] if units else 1.)
                        np.testing.assert_allclose(factor, expected, rtol=1e-14, atol=0)
        # Ordinary unit conversions must not change when a/h change.
        self.write_header(1., 0.68)
        other_epoch = simba.compute_conversion_factors(self.snapshot, self.constants)
        self.assertEqual(other_epoch['unit_conversions'], stages['unit_conversions'])
        with self.assertRaisesRegex(ValueError, 'booleans'):
            simba.compute_conversion_factors(self.snapshot, self.constants, to_units='g')

    def test_end_to_end_shared_units_for_selection_gas_geometry_and_provenance(self):
        self.write_header(0.25, 0.5)
        metal = np.zeros((2, 11))
        metal[:, :2] = [0.02, 0.24]
        with h5py.File(self.snapshot, 'a') as snapshot:
            gas = snapshot.create_group('PartType0')
            for name, values in {
                'ParticleIDs': [1, 2], 'Coordinates': [[0.5, 0, 0], [1, 0, 0]],
                'Velocities': [[2, 0, 0], [2, 0, 0]], 'Masses': [1., 2.],
                'Density': [1e-8, 2e-8], 'InternalEnergy': [100., 100.],
                'Metallicity': metal, 'ElectronAbundance': [0., 0.],
            }.items():
                gas[name] = values
        table_path = self.root / 'table.hdf5'
        with h5py.File(table_path, 'w') as table:
            table.attrs.update(schema=cloudy_hdf5.SCHEMA, abundance_model='synthetic fixed test composition')
            table['axes/log_temperature_K'] = [0., 12.]
            table['axes/log_nH_cm3'] = [-30., 20.]
            table.create_dataset('lines/labels', data=['test line'], dtype=h5py.string_dtype())
            values = table.create_dataset('lines/emissivity_erg_s_cm3', data=np.full((2, 2, 1), 1e-30))
            values.attrs.update(unit='erg s^-1 cm^-3', axis_order='log_temperature_K, log_nH_cm3, line')
        catalogue = MagicMock()
        catalogue.__enter__.return_value = catalogue
        catalogue.redshift = 3.
        catalogue.haloes.__len__.return_value = 1
        catalogue.haloes.get_particle_indices.return_value = np.array([0, 1])
        catalogue.haloes.get_dataset.side_effect = lambda name, **kwargs: {
            'minpot_pos': np.array([[0., 0., 0.]]), 'com_vel_total': np.zeros((1, 3)),
            'r200m': np.array([0.4]),
        }[name]
        octavius = SimpleNamespace(load_catalogue=lambda filename: catalogue)
        for mode in ('ppp', 'ppv'):
            with self.subTest(mode=mode):
                config = {
                    'source': {'snapshot_filename': str(self.snapshot), 'catalogue_filename': 'catalogue.hdf5'},
                    'selection': {'group_type': 'halo', 'target_index': 0},
                    'unit_factors': dict(self.constants),
                    'emissivity': {'type': 'cloudy_hdf5', 'table_filename': str(table_path), 'line_labels': ['test line']},
                    'cube': {'type': mode, 'center_kpc': 'auto', 'periodic': True,
                             'map_half_width_kpc': 1., 'spatial_bins': 1, 'los_axis': 'x',
                             'systemic_velocity_kms': 'auto', 'velocity_edges_kms': [0.5, 1.5]},
                    'output': {'filename': str(self.root / f'{mode}.hdf5')},
                }
                config_path = self.root / f'{mode}.yaml'
                config_path.write_text(yaml.safe_dump(config))
                with (patch.dict('sys.modules', {'octavius': octavius}),
                      patch.object(simba, 'compute_unit_factors', wraps=simba.compute_unit_factors) as compute,
                      patch.object(simba, 'select_cgm_gas_rows', wraps=simba.select_cgm_gas_rows) as select,
                      patch.object(simba, 'load_physical_gas_fields', wraps=simba.load_physical_gas_fields) as load,
                      warnings.catch_warnings()):
                    warnings.simplefilter('ignore', RuntimeWarning)
                    pipeline.main(config_path)
                compute.assert_called_once()
                units = select.call_args.kwargs['unit_factors']
                self.assertIs(load.call_args.args[2], units)
                gas = simba.load_physical_gas_fields(self.snapshot, np.array([0]), units)
                np.testing.assert_allclose(gas['x_kpc'], [0.25])
                np.testing.assert_allclose(gas['v_x_kms'], [1.])
                self.assertEqual(units['specific_energy_to_erg_g'], 1e10)
                with h5py.File(config['output']['filename']) as output:
                    self.assertEqual(output.attrs['selected_particle_count'], 1)
                    self.assertEqual(output['geometry'].attrs['box_size_kpc'], 5.)
                    np.testing.assert_allclose(output['lines/line_0000/cube_erg_s'][...].sum(),
                                               gas['volume_cm3'].sum() * 1e-30)
                    self.assertEqual(output['provenance/unit_inputs'].attrs['solar_mass_g'], 2e33)
                    self.assertEqual(output['provenance/unit_factors'].attrs['scale_factor'], 0.25)
                    self.assertEqual(output['provenance/unit_factors'].attrs['hubble_param'], 0.5)
                    self.assertEqual(output['provenance/unit_factors'].attrs['kpc_to_cm'], 3e21)
                    self.assertEqual(output['provenance/cosmological_factors'].attrs['mass'], 2.)
                    self.assertEqual(output['provenance/cosmological_factors'].attrs['density'], 16.)
                    np.testing.assert_allclose(output['provenance/unit_conversions'].attrs['mass'], 2e43, rtol=1e-14)
                    np.testing.assert_allclose(output['provenance/unit_factors'].attrs['mass_to_g'],
                                               output['provenance/cosmological_factors'].attrs['mass']
                                               * output['provenance/unit_conversions'].attrs['mass'])
                    if mode == 'ppp':
                        np.testing.assert_allclose(output['geometry/voxel_volume_cm3'][...], (2 * 3e21)**3)


if __name__ == '__main__':
    unittest.main()
