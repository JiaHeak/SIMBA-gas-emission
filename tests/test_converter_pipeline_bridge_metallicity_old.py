"""Check raw Cloudy conversion through physical gas, cubes and cube reading."""

from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch
import warnings

import h5py
import numpy as np
import yaml

from Pipeline.scr import pipeline, simba
from Pipeline.scr.tables import cloudy_hdf5, hdf5_converter
from Reader_Object import read_DataCube


class TestConverterPipelineBridge(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.labels = ['O  8 18.9689A', 'O  3 5006.84A']

    def convert_fixture(self, order=('density', 'temperature')):
        """Convert a complete shuffled synthetic grid with known emissivities.

        Returns: filename (pathlib.Path): new converter-format HDF5 table.
        """
        axes = {
            'temperature': ('coronal 4 log vary', 'grid range from 4 to 4.2 step 0.2',
                            'COROnal e', [4., 4.2]),
            'density': ('hden -5 vary', 'grid range from -5 to -4 step 0.5',
                        'HDEN=%f L', [-5., -4.5, -4.]),
        }
        prefix = self.root / '_'.join(order)
        source, grid, emis = [prefix.with_suffix(suffix) for suffix in ('.in', '.grd', '.emis')]
        commands = ['title Synthetic bridge test', 'abundances GASS10 no grains']
        for name in order:
            commands.extend(axes[name][:2])
        commands += ['table HM12 redshift 0.5', 'CMB redshift 0.5',
                     'set dr 0', 'stop zone 1', 'save grid "test.grd" last no hash',
                     'save lines emissivity "test.emis" last no hash',
                     *self.labels, 'end of lines', '##### Your Notes #####',
                     '# Synthetic fixture: not a Cloudy calculation.']
        source.write_text('\n'.join(commands) + '\n')
        # Model order is deliberately neither coordinate-sorted nor reshape order.
        points = [(4.2, -4.5), (4., -5.), (4.2, -4.),
                  (4., -4.), (4.2, -5.), (4., -4.5)]
        rows, emission_rows = [], []
        for index, (temperature, density) in enumerate(points):
            coordinates = {'temperature': temperature, 'density': density}
            rows.append('\t'.join([str(index), 'F', 'F', 'ok', '0', str(index),
                                   *(str(coordinates[name]) for name in order), 'fixture']))
            epsilon = 10.**(temperature + 2*density - 26)
            emission_rows.append(f'0.5\t{epsilon:.16e}\t{3*epsilon:.16e}')
        grid.write_text('\t'.join(['#Index', 'Failure?', 'Warnings?', 'Exit code', '#rank', '#seq',
                                   *(axes[name][2] for name in order), 'grid parameter string'])
                        + '\n' + '\n'.join(rows) + '\n')
        emis.write_text('#depth\t' + '\t'.join(self.labels) + '\n'
                        + '\n'.join(emission_rows) + '\n')
        return hdf5_converter.convert_to_hdf5(source, grid, emis, prefix.with_suffix('.hdf5'))

    def test_two_axis_conversion_to_multiline_ppp_ppv_and_reader(self):
        snapshot = self.root / 'snapshot.hdf5'
        constants = dict(solar_mass_g=2e33, kpc_to_cm=3e21,
                         specific_energy_to_erg_g=1e10, z_sun_mass_fraction=0.02)
        with h5py.File(snapshot, 'w') as source:
            source.create_group('Header').attrs.update(Time=2/3, HubbleParam=0.5,
                                                       Redshift=0.5, BoxSize=10.)
        units = simba.compute_unit_factors(snapshot, constants)
        temperature = 10.**np.array([4., 4.1])
        nH = 10.**np.array([-5., -4.5])
        hydrogen = 1 - 0.02 - 0.24
        molecular_weight = 1 / (hydrogen * (1 + 0.24 / (4*hydrogen)))
        energy = temperature * 1.380649e-16 / ((5/3 - 1)*molecular_weight*simba.PROTON_MASS_G)
        density = nH * simba.PROTON_MASS_G / hydrogen
        metals = np.zeros((2, 11))
        metals[:, :2] = [0.02, 0.24]
        with h5py.File(snapshot, 'a') as source:
            gas = source.create_group('PartType0')
            for name, values in {
                'ParticleIDs': [11, 22], 'Coordinates': [[0.1, 0, 0], [0.2, 0, 0]],
                'Velocities': np.zeros((2, 3)), 'Masses': [1., 2.],
                'Density': density / units['density_to_g_cm3'],
                'InternalEnergy': energy / units['specific_energy_to_erg_g'],
                'Metallicity': metals, 'ElectronAbundance': [0., 0.],
            }.items():
                gas[name] = values
        catalogue = MagicMock()
        catalogue.__enter__.return_value = catalogue
        catalogue.redshift = 0.5
        catalogue.haloes.__len__.return_value = 1
        catalogue.haloes.get_particle_indices.return_value = np.array([1, 0])
        catalogue.haloes.get_dataset.side_effect = lambda name, **kw: {
            'r200m': np.array([1.]), 'minpot_pos': np.zeros((1, 3)),
            'com_vel_total': np.zeros((1, 3)),
        }[name]
        octavius = SimpleNamespace(load_catalogue=lambda filename: catalogue)
        luminosity = 10.**(np.log10(temperature) + 2*np.log10(nH) - 26) * (
            np.array([1., 2.])*units['mass_to_g']/density)
        for order in [('density', 'temperature'), ('temperature', 'density')]:
            table = self.convert_fixture(order)
            for mode in ('ppp', 'ppv'):
                with self.subTest(order=order, mode=mode):
                    output = self.root / ('_'.join(order) + '_' + mode + '_cubes.hdf5')
                    config = {
                        'source': {'snapshot_filename': str(snapshot), 'catalogue_filename': 'mock'},
                        'selection': {'group_type': 'halo', 'target_index': 0},
                        'unit_factors': constants,
                        'emissivity': {'type': 'cloudy_hdf5', 'table_filename': str(table),
                                       'line_labels': list(reversed(self.labels))},
                        'cube': {'type': mode, 'center_kpc': 'auto', 'periodic': True,
                                 'map_half_width_kpc': 1., 'spatial_bins': 2,
                                 'systemic_velocity_kms': 'auto'},
                        'output': {'filename': str(output)},
                    }
                    config_path = self.root / 'run.yaml'
                    config_path.write_text(yaml.safe_dump(config))
                    with patch.dict('sys.modules', {'octavius': octavius}), warnings.catch_warnings():
                        warnings.simplefilter('ignore', RuntimeWarning)
                        pipeline.main(config_path)
                    reader = read_DataCube(output)
                    self.assertEqual(reader.line_labels, tuple(reversed(self.labels)))
                    for index, scale in enumerate((3., 1.)):
                        np.testing.assert_allclose(reader.get_fields('luminosity', lines=index).sum(),
                                                   scale*luminosity.sum(), rtol=1e-12, atol=0)
                    with h5py.File(output) as saved:
                        self.assertEqual(saved.attrs['schema'], 'codex_simba_line_cubes_v2')
                        provenance = saved['provenance/emissivity_table']
                        self.assertEqual(provenance.attrs['schema'], hdf5_converter.SCHEMA)
                        self.assertIn('GASS10', provenance.attrs['abundance_model'])
                        self.assertIn('HM12', provenance['physics_commands_json'].asstr()[()])
                        self.assertEqual(saved['provenance/table_axes/log_nH_cm3'].attrs['scale'], 'log10')
                        if mode == 'ppp':
                            np.testing.assert_allclose(reader.get_fields('emission', lines=0)
                                                       * saved['geometry/voxel_volume_cm3'][...],
                                                       reader.get_fields('luminosity', lines=0), atol=0)

    def test_extra_axes_and_invalid_runtime_tables_fail_explicitly(self):
        filename = self.convert_fixture()
        # No data repair, zero floor, implicit axis dropping or incomplete reads.
        for condition, message in [('extra_axis', 'Unsupported Cloudy axes'),
                                   ('zero', 'zeros need an explicit'),
                                   ('incomplete', 'not marked complete'),
                                   ('wrong_units', 'unit/scale')]:
            with self.subTest(condition=condition):
                invalid = self.root / (condition + '.hdf5')
                invalid.write_bytes(filename.read_bytes())
                with h5py.File(invalid, 'a') as table:
                    if condition == 'extra_axis':
                        table['axes/log_metal_scale'] = [-1., 0.]
                        table['lines/emissivity_erg_s_cm3'].attrs['axis_order'] = (
                            'log_temperature_K, log_nH_cm3, log_metal_scale, line')
                    elif condition == 'zero':
                        table['lines/emissivity_erg_s_cm3'][0, 0, 0] = 0.
                    elif condition == 'incomplete':
                        table.attrs['complete'] = False
                    else:
                        table['axes/log_nH_cm3'].attrs['unit'] = 'incorrect'
                with self.assertRaisesRegex(ValueError, message):
                    cloudy_hdf5.read_cloudy_hdf5_table(invalid)

    def test_fixed_radiation_redshift_must_match_run(self):
        table = cloudy_hdf5.read_cloudy_hdf5_table(self.convert_fixture())
        gas = dict(T_K=np.array([1e4]), nH_cm3=np.array([1e-5]), volume_cm3=np.array([2.]))
        with self.assertRaisesRegex(ValueError, 'differs from fixed Cloudy radiation'):
            next(cloudy_hdf5.iter_particle_line_luminosities(
                gas, table=table, redshift=0., line_labels=self.labels))
        with warnings.catch_warnings(record=True) as messages:
            warnings.simplefilter('always')
            emission = next(cloudy_hdf5.iter_particle_line_luminosities(
                gas, table=table, redshift=0.5, line_labels=self.labels))
        np.testing.assert_allclose(emission['line_luminosity_erg_s'], [2e-32], rtol=1e-12, atol=0)
        self.assertTrue(any('GASS10' in str(message.message) for message in messages))
        self.assertFalse(any('default abundances' in str(message.message) for message in messages))

    def test_declared_storage_order_drives_interpolation(self):
        filename = self.convert_fixture()
        # This is a valid alternative storage order, not a relabelled tensor.
        with h5py.File(filename, 'a') as table:
            dataset = table['lines/emissivity_erg_s_cm3']
            values = dataset[...].transpose(1, 0, 2)
            attributes = {key: value for key, value in dataset.attrs.items()
                          if key not in ('DIMENSION_LIST', 'DIMENSION_LABELS')}
            del table['lines/emissivity_erg_s_cm3']
            dataset = table.create_dataset('lines/emissivity_erg_s_cm3', data=values)
            dataset.attrs.update(attributes)
            order = 'log_nH_cm3, log_temperature_K, line'
            dataset.attrs['axis_order'] = order
            table.attrs['axis_order'] = order
            names = ['log_nH_cm3', 'log_temperature_K']
            for index, name in enumerate(names + ['line']):
                dataset.dims[index].label = name
                dataset.dims[index].attach_scale(table['lines/labels' if name == 'line' else 'axes/' + name])
            table['provenance/storage_axis_order'][...] = names
            indices = table['models/storage_indices']
            indices[...] = indices[...][:, ::-1]
        table = cloudy_hdf5.read_cloudy_hdf5_table(filename)
        gas = dict(T_K=np.array([10.**4.1]), nH_cm3=np.array([10.**-4.5]),
                   volume_cm3=np.array([2.]))
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            emissions = list(cloudy_hdf5.iter_particle_line_luminosities(
                gas, table=table, redshift=0.5, line_labels=self.labels))
        for emission, factor in zip(emissions, (1., 3.)):
            np.testing.assert_allclose(emission['line_luminosity_erg_s'],
                                       [2*factor*10.**(4.1 - 9 - 26)], rtol=1e-12, atol=0)


if __name__ == '__main__':
    unittest.main()
