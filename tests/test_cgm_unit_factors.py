"""Focused checks that CGM selection and gas conversion use the same units."""

from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from Pipeline.scr import simba
from Pipeline.tests.cube_fixtures import mock_emissivity_table


class Snapshot(dict):
    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


class TestCGMUnitFactors(unittest.TestCase):
    def test_configured_factors_control_radius_and_density_consistently(self):
        metal = np.zeros((4, 11))
        metal[:, 0] = 0.02
        metal[:, 1] = 0.24
        raw = {
            'PartType0/Coordinates': np.array([[0.4, 0, 0], [0.8, 0, 0],
                                              [9.4, 0, 0], [0.1, 0, 0]]),
            'PartType0/Density': np.array([0.1, 0.1, 0.1, 0.3]),
            'PartType0/Metallicity': metal,
            'PartType0/ParticleIDs': np.arange(4),
            'PartType0/Velocities': np.zeros((4, 3)),
            'PartType0/Masses': np.ones(4),
            'PartType0/InternalEnergy': np.ones(4),
            'PartType0/ElectronAbundance': np.zeros(4),
        }
        snapshot = Snapshot(raw, Header=SimpleNamespace(attrs={'BoxSize': 10.0}))
        groups = SimpleNamespace(get_dataset=lambda name, **kwargs: {
            'r200m': np.array([1.0]), 'minpot_pos': np.zeros((1, 3)),
        }[name])
        module = SimpleNamespace(load_catalogue=lambda filename: SimpleNamespace(haloes=groups))
        rows = np.array([2, 0, 3, 1])
        for position, density_multiple, expected in [(1.0, 1.0, [2, 0, 1]),
                                                       (2.0, 1.0, [0]),
                                                       (1.0, 2.0, [])]:
            with self.subTest(position=position, density_multiple=density_multiple):
                units = dict(position_to_physical_kpc=position, velocity_to_kms=1.0,
                             mass_to_g=1.0, density_to_g_cm3=simba.PROTON_MASS_G * density_multiple,
                             specific_energy_to_erg_g=1.0, z_sun_mass_fraction=0.02)
                with patch.dict('sys.modules', {'octavius': module}), patch.object(simba.h5py, 'File', return_value=snapshot):
                    selected = simba.select_cgm_gas_rows(
                        'mock', 'mock', rows, 0, unit_factors=units,
                    )
                np.testing.assert_array_equal(selected, expected)
                physical = simba.convert_gas_fields(raw, **units)
                box = 10 * position
                x = physical['x_kpc'] - box * np.rint(physical['x_kpc'] / box)
                keep = (np.abs(x) <= 1.0) & (physical['nH_cm3'] <= 0.13)
                np.testing.assert_array_equal(selected, rows[keep[rows]])

    def test_controller_passes_the_same_unit_dictionary(self):
        from Pipeline.scr import pipeline

        units = {'position_to_physical_kpc': 0.5, 'density_to_g_cm3': 2e-22}
        config = {'source': {'snapshot_filename': 'mock.hdf5',
                             'catalogue_filename': 'mock_catalogue.hdf5', 'redshift': 'auto'},
                  'unit_factors': units,
                  'selection': {'group_type': 'halo', 'target_index': 2,
                                },
                  'cube': {'cube_type': 'ppp', 'center_kpc': [0, 0, 0],
                           'systemic_velocity_kms': 0, 'periodic': False},
                  'emissivity': {'type': 'ps20', 'table_filename': 'mock.hdf5', 'line_labels': ['mock']}}
        rows = np.array([3, 1])
        class StopAfterSelection(Exception):
            pass
        with (
            patch.object(pipeline.input, 'read_parameters', return_value=config),
            patch.object(simba, 'compute_unit_factors', return_value=units) as compute,
            patch.object(pipeline, 'load_emissivity_table', side_effect=mock_emissivity_table),
            patch.object(simba, 'get_catalogue_redshift', return_value=0),
            patch.object(simba, 'get_redshift', return_value=0),
            patch.object(simba, 'select_gas_rows', return_value=rows),
            patch.object(simba, 'select_cgm_gas_rows', side_effect=StopAfterSelection) as select,
            patch('sys.argv', ['pipeline.py', 'mock.yaml']),
        ):
            with self.assertRaises(StopAfterSelection):
                pipeline.main()
        self.assertIs(select.call_args.kwargs['unit_factors'], units)
        self.assertIs(select.call_args.kwargs['gas_rows'], rows)
        self.assertEqual(select.call_args.kwargs['group_type'], 'halo')
        compute.assert_called_once_with('mock.hdf5', config['unit_factors'])
        self.assertIs(config['physical_unit_factors'], units)


if __name__ == '__main__':
    unittest.main()
