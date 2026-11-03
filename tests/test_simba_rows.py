"""Tests for the active catalogue-row selection and pipeline path."""

import yaml
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import h5py
import numpy as np

from Pipeline.scr import pipeline, simba
from Pipeline.tests.cube_fixtures import mock_emissivity_table


SAMPLE_SNAPSHOT = Path(__file__).resolve().parent / 'data' / 'sample_snapshot.hdf5'


class TestGasRowSelection(unittest.TestCase):
    def test_row_reader_preserves_order_and_field_shapes(self):
        fields = simba.get_gas_fields(
            SAMPLE_SNAPSHOT,
            [6, 1],
            ['PartType0/Coordinates', 'PartType0/Metallicity'],
        )

        np.testing.assert_array_equal(fields['PartType0/ParticleIDs'], [1007, 1002])
        self.assertEqual(fields['PartType0/Coordinates'].shape, (2, 3))
        self.assertEqual(fields['PartType0/Metallicity'].shape, (2, 11))

    def test_row_reader_does_not_drop_rows_with_the_same_id(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            snapshot_path = Path(temporary_directory) / 'duplicate_ids.hdf5'
            with h5py.File(snapshot_path, 'w') as snapshot:
                gas = snapshot.create_group('PartType0')
                gas.create_dataset('ParticleIDs', data=[42, 42, 99])
                gas.create_dataset('Coordinates', data=[[1, 0, 0], [2, 0, 0], [3, 0, 0]])

            fields = simba.get_gas_fields(
                snapshot_path, [1, 0], ['PartType0/Coordinates']
            )

        np.testing.assert_array_equal(fields['PartType0/ParticleIDs'], [42, 42])
        np.testing.assert_array_equal(
            fields['PartType0/Coordinates'], [[2, 0, 0], [1, 0, 0]]
        )

    def test_repeated_or_out_of_range_rows_raise(self):
        with self.assertRaises(AssertionError):
            simba.get_gas_fields(SAMPLE_SNAPSHOT, [1, 1])
        with self.assertRaises(IndexError):
            simba.get_gas_fields(SAMPLE_SNAPSHOT, [8])


class TestRowPipelineController(unittest.TestCase):
    def test_snapshot_to_ps20_to_cube_without_mocked_emissivity(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            table_path = Path(temporary_directory) / 'tiny_ps20.hdf5'
            with h5py.File(table_path, 'w') as table:
                bins = table.create_group('TableBins')
                bins.create_dataset('RedshiftBins', data=[0.0, 1.0])
                bins.create_dataset('TemperatureBins', data=[-6.0, -5.0])
                bins.create_dataset('MetallicityBins', data=[-50.0, -1.0, 0.0])
                bins.create_dataset('DensityBins', data=[18.0, 22.0])
                table.create_dataset('IdentifierLines', data=[b'H  1      6562.81A'])
                table.create_dataset('Tdep/EmissivitiesVol', data=np.zeros((2, 2, 3, 2, 1)))

            unit_factors = {
                'position_to_physical_kpc': 1.0,
                'velocity_to_kms': 1.0,
                'mass_to_g': 1.0,
                'density_to_g_cm3': 1.0,
                'specific_energy_to_erg_g': 1.0,
                'z_sun_mass_fraction': 0.02,
            }
            result = next(pipeline.run_line_emission_pipeline(
                SAMPLE_SNAPSHOT, np.array([1, 6]),
                emissivity_table=pipeline.load_emissivity_table('ps20', str(table_path), ['H  1      6562.81A']),
                redshift=0.0,
                unit_factors=unit_factors, center_kpc=[0, 0, 0],
                x_edges_kpc=[-100, 100], y_edges_kpc=[-100, 100],
                velocity_edges_kms=[-1000, 1000],
            ))
            gas = simba.load_physical_gas_fields(SAMPLE_SNAPSHOT, [1, 6], unit_factors)

        self.assertEqual(result['cube_erg_s'].shape, (1, 1, 1))
        self.assertAlmostEqual(
            float(result['cube_erg_s'].sum()), float(gas['volume_cm3'].sum())
        )

    def test_row_controller_saves_cube(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            temp_path = Path(temporary_directory)
            output_path = temp_path / 'line_cube.hdf5'
            config_path = temp_path / 'config.yaml'
            config = {
                'source': {'type': 'simba', 'snapshot_filename': str(SAMPLE_SNAPSHOT),
                           'catalogue_filename': 'mock_catalogue.hdf5'},
                'selection': {
                    'group_type': 'halo',
                    'target_index': 0,
                },
                'emissivity': {
                    'type': 'ps20',
                    'table_filename': 'mock_ps20.hdf5',
                    'line_labels': ['H  1      6562.81A'],
                },
                'unit_factors': {
                    'solar_mass_g': 1.988416e33,
                    'kpc_to_cm': 3.0856775814913673e21,
                    'specific_energy_to_erg_g': 1.0,
                    'z_sun_mass_fraction': 0.02,
                },
                'cube': {'center_kpc': [0.0, 0.0, 0.0],
                'x_edges_kpc': [-100.0, 100.0],
                'y_edges_kpc': [-100.0, 100.0],
                'velocity_edges_kms': [-1000.0, 1000.0]},
                'output': {'filename': str(output_path)},
            }
            config_path.write_text(yaml.safe_dump(config), encoding='utf-8')

            with (
                patch.object(pipeline, 'load_emissivity_table', side_effect=mock_emissivity_table),
                patch.object(simba, 'get_catalogue_redshift', return_value=0.0),
                patch.object(simba, 'select_cgm_gas_rows', side_effect=lambda **kw: kw['gas_rows']),
                patch.object(
                    simba, 'select_gas_rows', return_value=np.array([1, 6])
                ) as select_rows,
                patch.object(
                    pipeline, 'iter_particle_line_luminosities',
                    return_value=[{'line_label': 'H  1      6562.81A', 'line_luminosity_erg_s': np.array([2.0, 3.0])}],
                ),
                patch('sys.argv', ['pipeline.py', str(config_path)]),
            ):
                pipeline.main()

            select_rows.assert_called_once_with(
                catalogue_filename=str((temp_path / 'mock_catalogue.hdf5').resolve()),
                target_index=0,
                group_type='halo',
            )
            with h5py.File(output_path) as result:
                self.assertEqual(result['lines/line_0000/cube_erg_s'].shape, (1, 1, 1))
                self.assertAlmostEqual(float(result['lines/line_0000/cube_erg_s'][...].sum()), 5.0)
                self.assertEqual(result.attrs['selected_particle_count'], 2)


if __name__ == '__main__':
    unittest.main()
