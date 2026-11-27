"""V2 particle export and chunked rendering checks; no HPC or real Octavius needed."""
import tempfile
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch
import unittest
import warnings
import h5py
import numpy as np
import yaml
from scr import pipeline, input as settings
from scr.particle_output import _json, save_particle_catalogue
from Reader_Object import read_EmissionCatalogue, read_RawSnapshot


class TestParticlePipeline(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.parameters = dict(output={'filename': str(self.root/'particles.hdf5')},
            source={'snapshot_filename':'synthetic', 'catalogue_filename':'synthetic'},
            selection={'target_index':0}, emissivity={'line_labels':['A','B']})
        self.run = dict(redshift=0., unit_factors={},
            frame=dict(center_kpc=[9.5,0,0], bulk_velocity_kms=[1,2,10], box_size_kpc=10., periodic=True,
                       halo_center_kpc=[4.5,0,0], halo_bulk_velocity_kms=[0,0,0]),
            table=dict(table_type='synthetic',table_filename='synthetic',data={'axes':{}}))
        self.gas = dict(snapshot_row=np.array([9,2,5]), particle_id=np.array([2**63+3,7,7],dtype=np.uint64),
            x_kpc=np.array([9.7,.2,4.5]),y_kpc=np.array([0.,.2,0.]),z_kpc=np.zeros(3),
            v_x_kms=np.array([1.,2.,3.]),v_y_kms=np.array([2.,3.,4.]),v_z_kms=np.array([5.,15.,35.]),
            volume_cm3=np.array([2.,3.,5.]), smoothing_length_kpc=np.array([.1,.2,.3]))

    def emissions(self):
        for label, factor in [('A',1.),('B',3.)]:
            yield dict(line_label=label,volume_emissivity_erg_s_cm3=np.full(3,factor),
                       line_luminosity_erg_s=factor*self.gas['volume_cm3'])

    def test_roundtrip_render_conservation_chunks_and_frames(self):
        path=save_particle_catalogue(self.gas,self.emissions(),self.parameters,self.run)
        reader=read_EmissionCatalogue(path)
        np.testing.assert_array_equal(reader.get_fields('particle_id'),self.gas['particle_id'])
        np.testing.assert_array_equal(reader.get_fields('snapshot_row'),[9,2,5])
        for size in (1,2,100):
            image=reader.make_map('A',half_width_kpc=1,bins=4,chunk_size=size)
            self.assertEqual(image['values'].sum(),5.)
            self.assertEqual(image['excluded_luminosity_erg_s'],5.)
            spectrum=reader.make_spectrum('B',[-10,0,10],aperture_radius_kpc=1,chunk_size=size)
            np.testing.assert_array_equal(spectrum['values'],[6,9])
            self.assertEqual(spectrum['excluded_luminosity_erg_s'],15.)
        self.assertEqual(reader.make_map('A',half_width_kpc=1,periodic=False)['values'].sum(),2.)
        for los in 'xyz':
            self.assertEqual(reader.make_spectrum('A',[-100,100],los_axis=los)['values'].sum(),10.)
        self.assertEqual(reader.make_map('A',half_width_kpc=1,center=[4.5,0,0])['values'].sum(),5.)
        self.assertEqual(reader.make_map('A',half_width_kpc=1,center='halo')['values'].sum(),5.)
        np.testing.assert_array_equal(reader.make_spectrum('A',[0,10,20],velocity_frame='halo')['values'],[2.,3.])
        with self.assertRaises(FileExistsError):
            save_particle_catalogue(self.gas,self.emissions(),self.parameters,self.run)
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        reader.plot_map(image, show=False); reader.plot_spectrum(spectrum, show=False); plt.close('all')

    def test_wavelength_spectrum_uses_halo_velocity_and_preserves_luminosity(self):
        """Check Doppler binning against known particle velocities and luminosities.

        Returns: result (NoneType): verifies wavelength and velocity render products.
        """
        self.run['redshift'] = 0.6
        self.run['frame']['halo_bulk_velocity_kms'] = [0, 0, 5]
        label = 'O  3 0.5m'
        self.parameters['emissivity']['line_labels'][0] = label
        emissions = list(self.emissions())
        emissions[0]['line_label'] = label
        reader = read_EmissionCatalogue(
            save_particle_catalogue(self.gas, emissions, self.parameters, self.run))
        wavelength_edges = [4999.9, 5000.1, 5000.3]
        # Halo-frame speeds 0, 10, 30 km/s put the first two particles in
        # separate bins and the third at about 5000.50 Å, outside the range.
        wave = reader.make_spectrum(label, wavelength_edges_angstrom=wavelength_edges, chunk_size=1)
        np.testing.assert_array_equal(wave['values'], [2., 3.])
        np.testing.assert_array_equal(wave['edges'][0], wavelength_edges)
        self.assertEqual(wave['excluded_luminosity_erg_s'], 5.)
        self.assertEqual(wave['wavelength_frame'], 'halo')
        self.assertEqual(wave['rest_wavelength_source'], 'line_label')
        self.assertEqual(wave['rest_wavelength_angstrom'], 5000.)
        override = reader.make_spectrum(
            label, wavelength_edges_angstrom=[5999.9, 6000.1, 6000.3], rest_wavelength_angstrom=6000.)
        np.testing.assert_array_equal(override['values'], wave['values'])
        with self.assertRaisesRegex(ValueError, 'exactly one'):
            reader.make_spectrum(label, [-10, 10], wavelength_edges_angstrom=wavelength_edges)
        with self.assertRaisesRegex(ValueError, 'rest_wavelength_angstrom'):
            reader.make_spectrum('B', wavelength_edges_angstrom=wavelength_edges)
        with self.assertRaisesRegex(ValueError, 'positive'):
            reader.make_spectrum(label, wavelength_edges_angstrom=[-1, 0])
        with self.assertRaisesRegex(ValueError, 'velocity_frame'):
            reader.make_spectrum(label, wavelength_edges_angstrom=wavelength_edges, velocity_frame=[0, 0, 0])
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        wave['values'] *= 2
        ax = reader.plot_spectrum(wave, show=False)
        np.testing.assert_array_equal(ax.patches[0].get_data().values, [4., 6.])
        self.assertEqual(ax.get_xlabel(), 'Wavelength [Å]')
        plt.close(ax.figure)

    def test_radial_shell_profiles_and_plot_edited_values(self):
        """Check periodic shell assignments and plotting of edited results.

        Returns: result (NoneType): verifies shell sums, coverage and plot data.
        """
        path = save_particle_catalogue(self.gas, self.emissions(), self.parameters, self.run)
        reader = read_EmissionCatalogue(path)
        edges = [0., .5, 1., 2., 5.]
        for chunk_size in (1, 100):
            profile = reader.make_radial_profile(['B', 'A'], edges, chunk_size=chunk_size)
            self.assertEqual(list(profile['values']), ['B', 'A'])
            np.testing.assert_array_equal(profile['particle_count'], [1, 1, 0, 1])
            np.testing.assert_array_equal(profile['values']['A'], [2., 3., 0., 5.])
            np.testing.assert_array_equal(profile['values']['B'], [6., 9., 0., 15.])
            self.assertEqual(profile['included_luminosity_erg_s'], {'B': 30., 'A': 10.})
        cumulative = reader.make_radial_profile('A', [.5, 1., 2.], quantity='cumulative_luminosity')
        np.testing.assert_array_equal(cumulative['values']['A'], [3., 3.])
        self.assertEqual(cumulative['excluded_luminosity_erg_s']['A'], 7.)
        emission = reader.make_radial_profile('B', edges, quantity='emission')
        np.testing.assert_allclose(emission['values']['B'], [3., 3., np.nan, 3.])
        unwrapped = reader.make_radial_profile('A', edges, periodic=False)
        self.assertEqual(unwrapped['excluded_luminosity_erg_s']['A'], 3.)
        automatic = reader.make_radial_profile('A', 2, chunk_size=1)
        np.testing.assert_allclose(automatic['edges'][0], [0., 2.5, 5.])
        np.testing.assert_array_equal(automatic['values']['A'], [5., 5.])
        with self.assertRaisesRegex(ValueError, 'nonnegative'):
            reader.make_radial_profile('A', [-1., 1.])
        profile['values']['A'] *= 2
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        with patch('Reader_Object.particle_catalogue.h5py.File', side_effect=AssertionError('unexpected read')):
            ax = reader.plot_profile(profile, show=False)
        np.testing.assert_array_equal(ax.patches[1].get_data().values, [4., 6., 0., 10.])
        plt.close(ax.figure)

    def test_prepared_radial_and_map_line_ratios(self):
        """Check sums, undefined bins, map alignment and editable plots.

        Returns: result (NoneType): verifies both ratio products.
        """
        emissions = list(self.emissions())
        for emission, luminosity in zip(emissions, ([0., 6., 5.], [2., 1., 0.])):
            emission['line_luminosity_erg_s'] = np.array(luminosity)
            emission['volume_emissivity_erg_s_cm3'] = luminosity / self.gas['volume_cm3']
        reader = read_EmissionCatalogue(
            save_particle_catalogue(self.gas, emissions, self.parameters, self.run))

        profile = reader.make_radial_profile(['A', 'B'], [0., 1., 2., 5.])
        radial = reader.make_line_ratio(profile, numerator='A', denominator='B')
        np.testing.assert_allclose(radial['values']['A / B'], [2., np.nan, np.nan])
        np.testing.assert_array_equal(radial['valid_mask'], [True, False, False])
        self.assertEqual(radial['unit'], 'dimensionless')
        profile['values']['A'][0] = 9.
        edited = reader.make_line_ratio(profile, numerator='A', denominator='B')
        self.assertEqual(edited['values']['A / B'][0], 3.)
        self.assertEqual(radial['values']['A / B'][0], 2.)
        with self.assertRaisesRegex(ValueError, 'Both line labels'):
            reader.make_line_ratio(profile, numerator='C', denominator='B')

        settings = dict(x_edges_kpc=[-1., .5, 1.], y_edges_kpc=[-1., .1, 1.])
        numerator = reader.make_map('A', **settings)
        denominator = reader.make_map('B', **settings)
        image = reader.make_line_ratio_map(numerator, denominator)
        np.testing.assert_allclose(image['values'], [[0., np.nan], [np.nan, 6.]])
        np.testing.assert_array_equal(image['valid_mask'], [[True, False], [False, True]])
        thresholded = reader.make_line_ratio_map(numerator, denominator, min_denominator=2.)
        np.testing.assert_allclose(thresholded['values'], [[0., np.nan], [np.nan, np.nan]])
        with self.assertRaisesRegex(ValueError, 'min_denominator'):
            reader.make_line_ratio_map(numerator, denominator, min_denominator=-1.)
        different_grid = reader.make_map('B', x_edges_kpc=[-1., 0., 1.], y_edges_kpc=settings['y_edges_kpc'])
        with self.assertRaisesRegex(ValueError, 'pixel edges'):
            reader.make_line_ratio_map(numerator, different_grid)
        smoothed = reader.make_map('B', **settings, smoothing=True)
        with self.assertRaisesRegex(ValueError, 'smoothing'):
            reader.make_line_ratio_map(numerator, smoothed)
        with self.assertRaisesRegex(ValueError, 'center_kpc'):
            reader.make_line_ratio_map(numerator, dict(denominator, center_kpc=[0., 0., 0.]))

        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        self.addCleanup(plt.close, 'all')
        radial['values']['A / B'][0] = 4.
        image['values'][1, 1] = 5.
        with patch('Reader_Object.particle_catalogue.h5py.File', side_effect=AssertionError('unexpected read')):
            radial_ax = reader.plot_profile(radial, show=False)
            map_ax = reader.plot_map(image, log=False, show=False)
            log_ax = reader.plot_map(image, log=True, show=False)
        np.testing.assert_allclose(radial_ax.patches[0].get_data().values, [4., np.nan, np.nan])
        drawn = np.ma.asarray(map_ax.collections[0].get_array()).reshape(2, 2)
        np.testing.assert_allclose(drawn.filled(np.nan), [[0., np.nan], [np.nan, 5.]])
        log_drawn = np.ma.asarray(log_ax.collections[0].get_array()).reshape(2, 2)
        np.testing.assert_array_equal(np.ma.getmaskarray(log_drawn), [[True, True], [True, False]])
        self.assertIn('dimensionless', map_ax.figure.axes[-1].get_ylabel())

    def test_weighted_maps_read_only_in_frame_snapshot_rows(self):
        """Use independent known masses and temperatures in a periodic projection.

        Returns: result (NoneType): checks row alignment, weights and snapshot provenance.
        """
        snapshot = self.root / 'snapshot.hdf5'
        rows = self.gas['snapshot_row']
        ids = np.zeros(10, dtype=np.uint64)
        ids[rows] = self.gas['particle_id']
        mass = np.zeros(10)
        mass[rows] = [1., 3., 5.]
        energy = np.zeros(10)
        energy[rows] = [100., 300., 400.]
        with h5py.File(snapshot, 'w') as source:
            source.create_group('Header').attrs.update(Time=.25, HubbleParam=.5)
            gas = source.create_group('PartType0')
            gas['ParticleIDs'] = ids
            gas['Masses'] = mass
            gas['InternalEnergy'] = energy
            gas['Metallicity'] = np.zeros((10, 2))
            gas['ElectronAbundance'] = np.zeros(10)
        self.run['unit_factors'] = dict(
            scale_factor=.25, hubble_param=.5, mass_to_g=2., gamma=5/3,
            specific_energy_to_erg_g=1.5 * 1.380649e-16 / 1.67262192369e-24)
        path = save_particle_catalogue(
            dict(self.gas, mass_g=np.ones(3), T_K=np.ones(3)), self.emissions(), self.parameters, self.run)
        with h5py.File(path, 'r') as source:
            self.assertNotIn('mass_g', source['particles'])
            self.assertNotIn('T_K', source['particles'])
        reader = read_EmissionCatalogue(path)
        raw = read_RawSnapshot(snapshot)
        with patch.object(raw, 'read_gas_fields', wraps=raw.read_gas_fields) as read:
            image = reader.make_weighted_map(snapshot=raw, half_width_kpc=1, bins=1)
        read.assert_called_once()
        np.testing.assert_array_equal(read.call_args.args[0], [9, 2])
        np.testing.assert_allclose(image['values'], [[250.]])
        np.testing.assert_allclose(image['weighted_sum'], [[2000.]])
        np.testing.assert_array_equal(image['weight_sum'], [[8.]])
        self.assertEqual(image['unit'], 'K')
        self.assertEqual(image['weight_unit'], 'g')
        self.assertEqual(image['snapshot_filename'], str(snapshot.resolve()))
        self.assertEqual(image['excluded_particle_count'], 1)

        luminous = reader.make_weighted_map('A', 'z', 1, snapshot=raw, weighted_by='luminosity',
                                            bins=1, chunk_size=1)
        np.testing.assert_allclose(luminous['values'], [[220.]])
        self.assertEqual(luminous['weight_unit'], 'erg s^-1')
        separate_pixels = dict(x_edges_kpc=[-1., .5, 1.], y_edges_kpc=[-1., .1, 1.])
        sparse = reader.make_weighted_map(snapshot=raw, **separate_pixels)
        np.testing.assert_allclose(sparse['values'], [[100., np.nan], [np.nan, 300.]])
        np.testing.assert_array_equal(sparse['valid_mask'], [[True, False], [False, True]])
        smoothed = reader.make_weighted_map(snapshot=raw, smoothing=True, **separate_pixels)
        self.assertAlmostEqual(smoothed['values'][0, 1], 250.)
        self.assertAlmostEqual(smoothed['weight_sum'].sum(), 8.)
        self.assertAlmostEqual(smoothed['weighted_sum'].sum(), 2000.)

        with self.assertRaisesRegex(TypeError, 'read_RawSnapshot'):
            reader.make_weighted_map()
        with h5py.File(snapshot, 'r+') as source:
            source['Header'].attrs['Time'] = .5
        with self.assertRaisesRegex(ValueError, 'scale factor'):
            reader.make_weighted_map(snapshot=read_RawSnapshot(snapshot))
        with h5py.File(snapshot, 'r+') as source:
            source['Header'].attrs['Time'] = .25
            source['PartType0/ParticleIDs'][2] = 42
        with self.assertRaisesRegex(ValueError, 'ParticleIDs'):
            reader.make_weighted_map(snapshot=read_RawSnapshot(snapshot))

    def test_luminosity_weighted_velocity_uses_catalogue_and_plots_signed_values(self):
        """Project peculiar velocities in the saved frame and plot edited values.

        Returns: result (NoneType): checks signed means, frame changes and plot data.
        """
        reader = read_EmissionCatalogue(
            save_particle_catalogue(self.gas, self.emissions(), self.parameters, self.run))
        settings = dict(quantity='velocity', weighted_by='luminosity',
                        x_edges_kpc=[-1., .5, 1.], y_edges_kpc=[-1., .1, 1.])
        image = reader.make_weighted_map('A', **settings)
        np.testing.assert_allclose(image['values'], [[-5., np.nan], [np.nan, 5.]])
        halo = reader.make_weighted_map('A', velocity_frame='halo', **settings)
        np.testing.assert_allclose(halo['values'], [[5., np.nan], [np.nan, 15.]])
        with self.assertRaisesRegex(ValueError, 'line label'):
            reader.make_weighted_map(**settings)
        with self.assertRaisesRegex(ValueError, 'log=False'):
            reader.plot_map(image, log=True, show=False)

        import matplotlib.pyplot as plt
        image['values'][0, 0] = -10.
        with patch('Reader_Object.particle_catalogue.h5py.File', side_effect=AssertionError('unexpected read')):
            ax = reader.plot_map(image, show=False)
        self.addCleanup(plt.close, ax.figure)
        drawn = np.ma.asarray(ax.collections[0].get_array()).reshape(2, 2)
        np.testing.assert_allclose(drawn.filled(np.nan), [[-10., np.nan], [np.nan, 5.]])
        self.assertIn('km s^-1', ax.figure.axes[-1].get_ylabel())

    def test_metadata_bytes_arrays_and_hdf5_dimension_scales(self):
        """Write actual HDF5 scale attributes without losing text or reference paths.

        Returns: result (NoneType): verifies portable JSON and unchanged emission data.
        """
        table = self.root/'scales.hdf5'
        with h5py.File(table, 'w') as source:
            axis = source.create_dataset('axes/log_temperature_K', data=[4., 5.])
            axis.make_scale('log_temperature_K')
            axis.attrs['unit'] = 'log10(K)'
            dataset = source.create_dataset('lines/emissivity_erg_s_cm3', data=[1., 2.])
            dataset.dims[0].attach_scale(axis)
            metadata = dict(axis.attrs)
        self.run['table']['table_filename'] = str(table)
        self.run['table']['data']['axis_metadata'] = {'log_temperature_K': metadata}
        self.run['table']['data']['example'] = {
            'text': np.bytes_('CGM'), 'names': np.array([b'A', b'B']),
            'number': np.int64(3), 'flag': np.bool_(True), 'scalar_array': np.array(2.5),
        }
        path = save_particle_catalogue(self.gas, self.emissions(), self.parameters, self.run)
        with h5py.File(path) as output:
            saved = json.loads(output['provenance/table_metadata_json'].asstr()[()])
            self.assertTrue(output.attrs['complete'])
            np.testing.assert_array_equal(output['lines/line_0000/luminosity_erg_s'], self.gas['volume_cm3'])
        axis_metadata = saved['axis_metadata']['log_temperature_K']
        self.assertEqual(axis_metadata['CLASS'], 'DIMENSION_SCALE')
        self.assertEqual(axis_metadata['NAME'], 'log_temperature_K')
        self.assertEqual(axis_metadata['REFERENCE_LIST'], [
            {'dataset': {'hdf5_object_path': '/lines/emissivity_erg_s_cm3'}, 'dimension': 0}])
        self.assertEqual(saved['example'], {'text': 'CGM', 'names': ['A', 'B'],
                                          'number': 3, 'flag': True, 'scalar_array': 2.5})

    def test_metadata_invalid_values_still_raise_and_cleanup(self):
        """Reject unsupported/binary/nonfinite metadata rather than silently repair it.

        Returns: result (NoneType): checks serializer errors and atomic-output cleanup.
        """
        for value in (float('nan'), np.float32('inf'), np.array([np.nan])):
            with self.assertRaises(ValueError):
                _json({'value': value})
        with self.assertRaises(UnicodeDecodeError):
            _json(b'\xff')
        with self.assertRaisesRegex(TypeError, 'Unsupported JSON metadata'):
            _json(object())
        with self.assertRaisesRegex(TypeError, 'hdf5_filename'):
            _json(h5py.Reference())
        with self.assertRaisesRegex(TypeError, 'region references'):
            _json(h5py.RegionReference())
        self.run['table']['data']['bad'] = object()
        with self.assertRaises(TypeError):
            save_particle_catalogue(self.gas, self.emissions(), self.parameters, self.run)
        self.assertFalse(Path(self.parameters['output']['filename']).exists())
        self.assertFalse(list(self.root.glob('.particle_emission_*')))

    def test_line_computation_counts_include_zeros_and_clipped_results(self):
        """Count finite zeros as completed and retain separate clipping diagnostics.

        Returns: result (NoneType): verifies saved/read counts for each emission line.
        """
        emissions = []
        for label, epsilon in [('A', np.array([0., 1., 0.])), ('B', np.zeros(3))]:
            emissions.append(dict(line_label=label, volume_emissivity_erg_s_cm3=epsilon,
                                  line_luminosity_erg_s=epsilon*self.gas['volume_cm3'],
                                  clipped_particle_count=2,
                                  zero_emissivity_particle_count=int(np.count_nonzero(epsilon == 0))))
        path = save_particle_catalogue(self.gas, iter(emissions), self.parameters, self.run)
        reader = read_EmissionCatalogue(path)
        for label, positive, zero in [('A', 1, 2), ('B', 0, 3)]:
            info = reader.diagnostics[label]
            self.assertEqual(info['input_particle_count'], 3)
            self.assertEqual(info['computed_particle_count'], 3)
            self.assertEqual(info['failed_particle_count'], 0)
            self.assertEqual(info['positive_emissivity_particle_count'], positive)
            self.assertEqual(info['zero_emissivity_particle_count'], zero)
            self.assertEqual(info['clipped_particle_count'], 2)
            self.assertEqual(info['computation_counts_source'], 'writer_validation')
            self.assertIn('physical accuracy', info['computation_counts_note'])
        with h5py.File(path) as source:
            saved = json.loads(source['lines/line_0000'].attrs['diagnostics_json'])
        self.assertEqual(saved['computed_particle_count'], 3)
        self.assertEqual(reader.diagnostics['A'], saved)

    def test_invalid_emission_reports_failed_particle_count_and_no_output(self):
        """Count the union of numerical failures and keep atomic failure behavior.

        Returns: result (NoneType): checks failure accounting and no published output.
        """
        emission = dict(line_label='A', volume_emissivity_erg_s_cm3=np.array([np.nan, -1., 1.]),
                        line_luminosity_erg_s=np.array([np.nan, -3., 100.]))
        with self.assertRaisesRegex(ValueError, '3 failed / 3 particles'):
            save_particle_catalogue(self.gas, iter([emission]), self.parameters, self.run)
        self.assertFalse(Path(self.parameters['output']['filename']).exists())
        self.assertFalse(list(self.root.glob('.particle_emission_*')))
        emission.update(volume_emissivity_erg_s_cm3=np.ones(3),
                        line_luminosity_erg_s=self.gas['volume_cm3'], zero_emissivity_particle_count=3)
        with self.assertRaisesRegex(ValueError, 'Inconsistent zero_emissivity_particle_count'):
            save_particle_catalogue(self.gas, iter([emission]), self.parameters, self.run)
        self.assertFalse(Path(self.parameters['output']['filename']).exists())

    def test_failed_and_empty_outputs(self):
        with self.assertRaisesRegex(ValueError,'Missing requested'):
            save_particle_catalogue(self.gas,iter([]),self.parameters,self.run)
        self.assertFalse(Path(self.parameters['output']['filename']).exists())
        self.assertFalse(list(self.root.glob('.particle_emission_*')))
        empty={name:values[:0] for name,values in self.gas.items()}
        emissions=(dict(line_label=label,volume_emissivity_erg_s_cm3=np.array([]),
                        line_luminosity_erg_s=np.array([])) for label in ('A','B'))
        path=save_particle_catalogue(empty,emissions,self.parameters,self.run)
        reader = read_EmissionCatalogue(path)
        self.assertEqual(reader.make_map('A')['values'].sum(),0.)
        for label in reader.line_labels:
            for name in ('input_particle_count', 'computed_particle_count', 'failed_particle_count',
                         'zero_emissivity_particle_count', 'positive_emissivity_particle_count'):
                self.assertEqual(reader.diagnostics[label][name], 0)

    def test_main_with_real_snapshot_fields_and_table(self):
        snapshot=self.root/'snapshot.hdf5'; table=self.root/'table.hdf5'
        with h5py.File(snapshot,'w') as f:
            f.create_group('Header').attrs.update(Time=.25,HubbleParam=.5,Redshift=3.,BoxSize=10.)
            g=f.create_group('PartType0'); metals=np.zeros((2,11));metals[:,:2]=[.02,.24]
            for name,values in dict(ParticleIDs=np.array([8,8],dtype=np.uint64),
                Coordinates=[[.2,0,0],[.4,0,0]],Velocities=[[2,4,6],[4,6,8]],
                Masses=[1.,2.],Density=[1e-8,2e-8],InternalEnergy=[100.,100.],
                Metallicity=metals,ElectronAbundance=[0.,0.],SmoothingLength=[.1,.2]).items():g[name]=values
        with h5py.File(table,'w') as f:
            f.attrs.update(schema='codex_simba_cloudy_mvp_v1',abundance_model='synthetic')
            f['axes/log_temperature_K']=[0.,12.];f['axes/log_nH_cm3']=[-30.,20.]
            f.create_dataset('lines/labels',data=['A','B'],dtype=h5py.string_dtype())
            d=f.create_dataset('lines/emissivity_erg_s_cm3',data=np.ones((2,2,2))*1e-30)
            d.attrs.update(unit='erg s^-1 cm^-3',axis_order='log_temperature_K, log_nH_cm3, line')
        config=dict(source=dict(snapshot_filename=str(snapshot),catalogue_filename='fake'),
            selection=dict(group_type='halo',target_index=0),
            unit_factors=dict(solar_mass_g=2e33,kpc_to_cm=3e21,specific_energy_to_erg_g=1e10,z_sun_mass_fraction=.02),
            emissivity=dict(type='cloudy_hdf5',table_filename=str(table),line_labels=['B','A']),
            output=dict(filename=str(self.root/'main.hdf5')))
        config_path=self.root/'run.yaml';config_path.write_text(yaml.safe_dump(config))
        cat=MagicMock();cat.__enter__.return_value=cat;cat.redshift=3.;cat.haloes.__len__.return_value=1
        cat.haloes.get_particle_indices.return_value=np.array([1,0])
        cat.haloes.get_dataset.side_effect=lambda name,**kw:dict(r200m=np.array([1.]),
            minpot_pos=np.zeros((1,3)),com_vel_total=np.zeros((1,3)))[name]
        with patch.dict('sys.modules',{'octavius':SimpleNamespace(load_catalogue=lambda filename:cat)}),warnings.catch_warnings():
            warnings.simplefilter('ignore',RuntimeWarning)
            path=pipeline.main(config_path)
        reader=read_EmissionCatalogue(path)
        self.assertEqual(reader.line_labels,('B','A'))
        np.testing.assert_array_equal(reader.get_fields('snapshot_row'),[1,0])
        np.testing.assert_allclose(reader.get_fields('smoothing_length_kpc'),[.1,.05])
        np.testing.assert_allclose(reader.get_fields('position_kpc')[:,0],[.2,.1])
        np.testing.assert_allclose(reader.get_fields('velocity_kms')[:,2],[4.,3.])
        epsilon=reader.get_fields('volume_emissivity_erg_s_cm3',line='A')
        luminosity=reader.get_fields('luminosity_erg_s',line='A')
        np.testing.assert_allclose(epsilon*reader.get_fields('volume_cm3'),luminosity,atol=0)
        with h5py.File(path) as f:self.assertIsNotNone(f['particles/position_kpc'].chunks)
        cat.haloes.get_particle_indices.return_value = np.array([], dtype=np.int64)
        config['output']['filename'] = str(self.root/'empty_main.hdf5')
        config_path.write_text(yaml.safe_dump(config))
        with patch.dict('sys.modules',{'octavius':SimpleNamespace(load_catalogue=lambda filename:cat)}),warnings.catch_warnings():
            warnings.simplefilter('ignore',RuntimeWarning)
            empty = read_EmissionCatalogue(pipeline.main(config_path))
        self.assertEqual(empty.particle_count, 0)
        self.assertEqual(empty.make_spectrum('A', [-10,10])['values'].sum(), 0.)
        config['cube']={'type':'ppp'};config_path.write_text(yaml.safe_dump(config))
        with self.assertRaises(ValueError):settings.read_parameters(config_path)

if __name__=='__main__':unittest.main()
