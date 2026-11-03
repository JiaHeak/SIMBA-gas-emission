"""Small three-axis/zero/clipping checks, including a real cube writer round-trip."""

import itertools
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import warnings

import h5py
import numpy as np

from Pipeline.scr import pipeline, output, simba
from Pipeline.scr.tables import cloudy_hdf5, hdf5_converter


class TestCloudyMetalScale(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.labels = ['O  8 18.9689A', 'H  1 6562.81A']
        source, grid, emis = [self.root / ('fixture' + suffix) for suffix in ('.in', '.grd', '.emis')]
        source.write_text('\n'.join([
            'title Synthetic metal mapping fixture', 'abundances GASS10 no grains',
            'coronal 4 log vary', 'grid range from 4 to 4.2 step 0.2',
            'hden -5 vary', 'grid range from -5 to -4 step 1',
            'metals -2 log vary', 'grid range from -2 to -1 step 1',
            'table HM12 redshift 0.5', 'CMB redshift 0.5',
            'set dr 0', 'stop zone 1', 'save grid "fixture.grd" last no hash',
            'save lines emissivity "fixture.emis" last no hash',
            *self.labels, 'end of lines', '']) )
        rows, values = [], []
        for index, (ti, ni, mi) in enumerate(itertools.product(range(2), repeat=3)):
            t, n, metal = 4 + .2*ti, -5 + ni, -2 + mi
            epsilon = (ti + 2*ni + 4*mi) * 1e-30
            rows.append(f'{index}\tF\tF\tok\t0\t{index}\t{t}\t{n}\t{metal}\tfixture')
            values.append(f'0.5\t{epsilon:.16e}\t{2*epsilon+1e-30:.16e}')
        grid.write_text('#Index\tFailure?\tWarnings?\tExit code\t#rank\t#seq\tCOROnal e\tHDEN=%f L\tMETALS= %f\tgrid parameter string\n'
                        + '\n'.join(rows) + '\n')
        emis.write_text('#depth\t' + '\t'.join(self.labels) + '\n' + '\n'.join(values) + '\n')
        self.filename = self.root / 'table.hdf5'
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            hdf5_converter.convert_to_hdf5(source, grid, emis, self.filename, cloudy_version='25.00')
        self.table = cloudy_hdf5.read_cloudy_hdf5_table(self.filename)

    def gas(self):
        """Create aligned particles probing interior, boundaries and metal clipping.

        Returns: gas_fields (dict[str, numpy.ndarray]): synthetic physical fields.
        """
        reference = self.table['metal_reference']['metal_to_hydrogen_mass_ratio']
        return {
            'T_K': 10.**np.array([4.1, 4.1, 4.1, 4.1, 4.]),
            'nH_cm3': 10.**np.array([-4.5, -4.5, -4.5, -4.5, -5.]),
            'volume_cm3': np.full(5, 2.), 'X_mass_fraction': np.full(5, .7),
            'Z_mass_fraction': .7 * reference * np.array([10.**-1.5, 0., 1e-5, 1., .01]),
            'x_kpc': np.zeros(5), 'y_kpc': np.zeros(5), 'z_kpc': np.zeros(5),
        }

    def test_linear_epsilon_zeros_clipping_and_original_fractions_preserved(self):
        gas = self.gas()
        original = {k: v.copy() for k, v in gas.items()}
        with warnings.catch_warnings(record=True) as messages:
            warnings.simplefilter('always')
            emissions = list(cloudy_hdf5.iter_particle_line_luminosities(
                gas, table=self.table, redshift=.5, line_labels=self.labels))
        expected = np.array([3.5, 1.5, 1.5, 5.5, 0.]) * 1e-30
        for index, emission in enumerate(emissions):
            epsilon = expected if index == 0 else 2*expected+1e-30
            np.testing.assert_allclose(emission['volume_emissivity_erg_s_cm3'], epsilon, atol=1e-44, rtol=1e-12)
            np.testing.assert_allclose(emission['line_luminosity_erg_s'], 2*epsilon, atol=1e-44, rtol=1e-12)
            self.assertEqual(emission['clipped_particle_count'], 3)
            self.assertEqual(emission['metallicity_clipped_particle_count'], 3)
            self.assertEqual(emission['zero_metallicity_particle_count'], 1)
            self.assertEqual(emission['clipping_counts']['log_metal_scale'], {'below': 2, 'above': 1})
        self.assertEqual(emissions[0]['volume_emissivity_erg_s_cm3'][-1], 0.)
        self.assertTrue(np.isneginf(emissions[0]['log_volume_emissivity'][-1]))
        self.assertEqual(sum('CAVEAT:' in str(x.message) for x in messages), 1)
        for key in gas:
            np.testing.assert_array_equal(gas[key], original[key])

    def test_mass_fraction_validation_and_reference_rejection(self):
        for name, value in [('Z_mass_fraction', -1.), ('Z_mass_fraction', np.nan),
                            ('X_mass_fraction', 0.), ('X_mass_fraction', 1.1)]:
            gas = self.gas()
            gas[name][0] = value
            with self.subTest(name=name, value=value), self.assertRaises(ValueError):
                next(cloudy_hdf5.iter_particle_line_luminosities(
                    gas, table=self.table, redshift=.5, line_labels=self.labels))
        gas = self.gas()
        del gas['Z_mass_fraction']
        with self.assertRaisesRegex(KeyError, 'Z_mass_fraction'):
            next(cloudy_hdf5.iter_particle_line_luminosities(
                gas, table=self.table, redshift=.5, line_labels=self.labels))
        with h5py.File(self.filename, 'r+') as f:
            records = json.loads(f['physics/parsed_commands_json'].asstr()[()])
            records[0]['reference'] = 'UNKNOWN'
            f['physics/parsed_commands_json'][()] = json.dumps(records)
        with self.assertRaisesRegex(ValueError, 'GASS10'):
            cloudy_hdf5.read_cloudy_hdf5_table(self.filename)

    def test_raw_Z_X_are_passed_by_SIMBA_conversion(self):
        raw = {'PartType0/Coordinates': np.zeros((2, 3)), 'PartType0/Velocities': np.zeros((2, 3)),
               'PartType0/Masses': np.ones(2), 'PartType0/Density': np.ones(2),
               'PartType0/InternalEnergy': np.ones(2), 'PartType0/ParticleIDs': np.array([22, 11]),
               'PartType0/ElectronAbundance': np.zeros(2),
               'PartType0/Metallicity': np.array([[0., .24, 0., 0., 0.], [.02, .25, .001, 0., .002]])}
        gas = simba.convert_gas_fields(raw, position_to_physical_kpc=1, velocity_to_kms=1,
                                      mass_to_g=1, density_to_g_cm3=1,
                                      specific_energy_to_erg_g=1, z_sun_mass_fraction=.02)
        np.testing.assert_array_equal(gas['Z_mass_fraction'], [0., .02])
        np.testing.assert_allclose(gas['X_mass_fraction'], [.76, .73])
        np.testing.assert_array_equal(gas['particle_id'], [22, 11])

    def test_controller_and_saved_cube_keep_the_scientific_caveat(self):
        gas = self.gas()
        loaded = {'table_type': 'cloudy_hdf5', 'table_filename': str(self.filename),
                  'line_labels': self.labels, 'data': self.table}
        with patch.object(pipeline.simba, 'load_physical_gas_fields', return_value=gas), warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            results = list(pipeline.run_line_emission_pipeline(
                'synthetic', np.arange(5), emissivity_table=loaded, redshift=.5, unit_factors={},
                center_kpc=[0, 0, 0], x_edges_kpc=[-1, 1], y_edges_kpc=[-1, 1],
                cube_type='ppp', z_edges_kpc=[-1, 1], velocity_edges_kms=None))
        parameters = {'output': {'filename': str(self.root/'cube.hdf5')},
                      'emissivity': {'line_labels': self.labels, 'table_filename': str(self.filename)},
                      'source': {'snapshot_filename': 'synthetic', 'catalogue_filename': 'synthetic'},
                      'selection': {'group_type': 'halo', 'target_index': 0},
                      'unit_factors': {}, 'config_filename': 'synthetic'}
        path = output.save_result(iter(results), parameters)
        with h5py.File(path) as f:
            meta = f['provenance/emissivity_table']
            self.assertIn('linear emissivity', meta.attrs['interpolation_method'])
            self.assertIn('NOT a primordial', meta.attrs['metallicity_clipping_caveat'])
            mapping = json.loads(meta['metallicity_mapping_json'].asstr()[()])
            self.assertAlmostEqual(mapping['Z_ref']/mapping['X_ref'],
                                   self.table['metal_reference']['metal_to_hydrogen_mass_ratio'])
            self.assertEqual(f['lines/line_0000'].attrs['zero_metallicity_particle_count'], 1)
            self.assertEqual(f['lines/line_0000'].attrs['metallicity_clipped_particle_count'], 3)
            self.assertAlmostEqual(float(f['lines/line_0000/cube_erg_s'][...].sum())/1e-30, 24.)


if __name__ == '__main__':
    unittest.main()
