"""Focused numerical/plot separation checks for the particle map API."""

import unittest
from unittest.mock import patch
import warnings

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np

import test_particle_pipeline as fixtures
from scr.particle_output import save_particle_catalogue
from Reader_Object import read_EmissionCatalogue
from Reader_Object.map_tools import smooth_map


class TestParticleMaps(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.TestParticlePipeline()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.addCleanup(plt.close, 'all')
        self.fixture.run['redshift'] = 1.
        path = save_particle_catalogue(self.fixture.gas, self.fixture.emissions(),
                                       self.fixture.parameters, self.fixture.run)
        self.reader = read_EmissionCatalogue(path)

    def test_make_map_conserves_luminosity_selection_and_chunk_independence(self):
        raw = self.reader.make_map('A', half_width_kpc=1, bins=8)
        self.assertFalse(raw['smoothing_enabled'])
        self.assertEqual(raw['values'].sum(), 5.)
        reference = self.reader.make_map('A', half_width_kpc=1, bins=8, smoothing=True)
        for chunk_size in (1, 2, 100):
            image = self.reader.make_map('A', half_width_kpc=1, bins=8, chunk_size=chunk_size, smoothing=True)
            np.testing.assert_array_equal(image['values'], reference['values'])
            self.assertEqual(image['values'].sum(), 5.)
            self.assertEqual(image['excluded_luminosity_erg_s'], 5.)
            self.assertEqual(image['included_particle_count'], 2)
            self.assertEqual(image['product_type'], 'map')
            self.assertEqual(image['smoothing_sigma_pixels'], .5)
        self.assertGreater(np.count_nonzero(reference['values']), np.count_nonzero(raw['values']))
        # LOS-x uses image y,z, retaining all three fixture particles.
        custom = self.reader.make_map('A', los_axis='x',
                    x_edges_kpc=[-1, 0, .1, 1], y_edges_kpc=[-1, .5, 1])
        self.assertEqual(custom['axis_names'], ['y', 'z'])
        self.assertEqual(custom['values'].shape, (3, 2))
        self.assertEqual(custom['values'].sum(), 10.)
        with self.assertRaisesRegex(ValueError, 'smoothing'):
            self.reader.make_map('A', smoothing='yes')

    def test_smoothing_impulses_boundaries_and_no_mutation(self):
        values = np.zeros((5, 5)); values[2, 2] = 8.
        raw = dict(values=values, included_luminosity_erg_s=8.)
        weights = np.array([0., .125, .75, .125, 0.])
        result = smooth_map(raw)
        np.testing.assert_array_equal(result['values'], 8*np.outer(weights, weights))
        self.assertEqual(np.count_nonzero(values), 1)
        self.assertEqual(result['values'].sum(), 8.)
        values[2, 2] = 0.; values[0, 0] = 8.
        weights = np.array([.875, .125, 0., 0., 0.])
        result = smooth_map(raw)
        np.testing.assert_array_equal(result['values'], 8*np.outer(weights, weights))
        singleton = smooth_map(dict(values=np.array([[8.]]), included_luminosity_erg_s=8.))
        self.assertEqual(singleton['values'][0, 0], 8.)

    def test_surface_brightness_uses_pixel_areas_and_catalogue_redshift(self):
        """Check the conversion and luminosity accounting on an unequal rectangular grid.

        Returns: result (NoneType): verifies area, redshift, axis order and smoothing.
        """
        settings = dict(x_edges_kpc=[-1., 0., .5, 1.5], y_edges_kpc=[-.5, .1, 1.3])
        luminosity = self.reader.make_map(line='A', field='luminosity', **settings)
        np.testing.assert_array_equal(self.reader.make_map('A', **settings)['values'], luminosity['values'])
        image = self.reader.make_map(line='A', field='surface_brightness', **settings)
        expected_area = np.array([[.6, 1.2], [.3, .6], [.6, 1.2]])
        np.testing.assert_allclose(image['pixel_area_kpc2'], expected_area)
        self.assertEqual(image['unit'], 'erg s^-1 cm^-2 arcsec^-2')
        self.assertEqual(image['redshift'], 1.)
        self.assertEqual(image['included_luminosity_erg_s'], 5.)
        self.assertEqual(image['excluded_luminosity_erg_s'], 5.)

        # At z=1, D_L/D_A=4. Compute flux and angular pixel area independently.
        distance_kpc = 1e5
        flux = np.array([[0., 0.], [2., 0.], [0., 3.]]) / (4 * np.pi * (4 * distance_kpc * 3.08567758149e21)**2)
        angular_area = expected_area * (206264.806247 / distance_kpc)**2
        np.testing.assert_allclose(image['values'], flux / angular_area, rtol=1e-10, atol=0)
        undimmed = self.reader.make_map('A', field='surface_brightness', redshift=0., **settings)
        np.testing.assert_allclose(undimmed['values'], 16 * image['values'], atol=0)

        smoothed = self.reader.make_map('A', field='surface_brightness', smoothing=True, **settings)
        smoothed_luminosity = self.reader.make_map('A', smoothing=True, **settings)
        conversion = (flux / angular_area)[1, 0] * expected_area[1, 0] / 2
        np.testing.assert_allclose(smoothed['values'] * expected_area,
                                   smoothed_luminosity['values'] * conversion, rtol=1e-10, atol=0)

    def test_surface_brightness_plots_prepared_values_and_validates_before_rendering(self):
        """Check the display metadata and rejection of invalid requests.

        Returns: result (NoneType): verifies display units, redshift and argument errors.
        """
        image = self.reader.make_map('A', field='surface_brightness', half_width_kpc=1, bins=2)
        image['values'] *= 2
        with patch.object(self.reader, '_render', side_effect=AssertionError('unexpected render')):
            ax = self.reader.plot_map(image, log=False, show=False)
            with self.assertRaisesRegex(ValueError, 'field must be'):
                self.reader.make_map('A', field='flux')
            with self.assertRaisesRegex(ValueError, 'applies only'):
                self.reader.make_map('A', redshift=1.)
            for redshift in (-1., np.nan, np.inf, True):
                with self.subTest(redshift=redshift), self.assertRaisesRegex(ValueError, 'redshift'):
                    self.reader.make_map('A', field='surface_brightness', redshift=redshift)
        drawn = np.ma.asarray(ax.collections[0].get_array()).reshape(2, 2)
        np.testing.assert_array_equal(drawn, image['values'].T)
        self.assertIn('arcsec^-2', ax.figure.axes[-1].get_ylabel())
        self.assertIn('z = 1', ax.get_title())

    def test_plot_uses_edited_map_values_log_masks_orientation_and_show(self):
        options = dict(los_axis='x', half_width_kpc=1, bins=8, center='halo')
        image = self.reader.make_map('A', **options)
        image['values'] *= 2
        _, axes = plt.subplots()
        with patch('matplotlib.pyplot.show') as show, patch.object(
                self.reader, '_render', side_effect=AssertionError('unexpected render')):
            ax = self.reader.plot_map(image, cmap='viridis', ax=axes)
        self.assertIs(ax, axes)
        show.assert_called_once_with()
        mesh = ax.collections[0]
        self.assertIsInstance(mesh.norm, LogNorm)
        self.assertEqual(mesh.get_cmap().name, 'viridis')
        plotted = np.ma.asarray(mesh.get_array()).reshape(image['values'].T.shape)
        np.testing.assert_array_equal(plotted.filled(0), image['values'].T)
        np.testing.assert_array_equal(np.ma.getmaskarray(plotted), image['values'].T == 0)
        self.assertEqual(ax.get_xlabel(), 'y offset [kpc]')
        self.assertEqual(ax.get_ylabel(), 'z offset [kpc]')
        with patch('matplotlib.pyplot.show') as show:
            linear = self.reader.plot_map(image, log=False, show=False)
        show.assert_not_called()
        self.assertNotIsInstance(linear.collections[0].norm, LogNorm)
        np.testing.assert_array_equal(self.reader.get_fields('particle_id'), self.fixture.gas['particle_id'])

    def test_empty_map_fallback_and_spectrum_remain_usable(self):
        image = self.reader.make_map('A', center=[4.5, 4.5, 4.5], half_width_kpc=.01, bins=8)
        with warnings.catch_warnings(record=True) as messages:
            warnings.simplefilter('always')
            ax = self.reader.plot_map(image, show=False)
        self.assertTrue(any('All-zero' in str(item.message) for item in messages))
        self.assertNotIsInstance(ax.collections[0].norm, LogNorm)
        spectrum = self.reader.make_spectrum('A', [-100, 100])
        self.assertEqual(spectrum['product_type'], 'spectrum')
        self.assertEqual(spectrum['values'].sum(), 10.)
        self.assertNotIn('smoothing_enabled', spectrum)
        self.reader.plot_spectrum(spectrum, show=False)

    def test_gaussian_impulse_width_and_large_kernel_reflection(self):
        """Check sampled Gaussian weights and conservation on large/small images.

        Returns: result (NoneType): validates impulse response and reflected flux.
        """
        for sigma in (.5, 1., 2.):
            values = np.zeros((21, 21)); values[10, 10] = 8.
            radius = int(np.ceil(4*sigma))
            offsets = np.arange(-radius, radius+1)
            weights = np.exp(-offsets**2/(2*sigma**2)); weights /= weights.sum()
            expected = np.zeros_like(values)
            expected[10-radius:11+radius, 10-radius:11+radius] = 8*np.outer(weights, weights)
            result = smooth_map(dict(values=values, included_luminosity_erg_s=8.),
                                enabled='gaussian', sigma_pixels=sigma)
            np.testing.assert_allclose(result['values'], expected, rtol=1e-14, atol=0)
            self.assertAlmostEqual(result['values'].sum(), 8.)
            self.assertEqual(result['smoothing_method'], 'gaussian')
            self.assertEqual(result['smoothing_sigma_pixels'], sigma)
            self.assertEqual(result['smoothing_radius_pixels'], radius)
            self.assertAlmostEqual(result['smoothing_effective_sigma_pixels'],
                                   np.sqrt(np.sum(weights*offsets**2)))
            self.assertEqual(np.count_nonzero(values), 1)
        # Gaussian support can exceed the whole image; reflect without losing flux.
        for shape in ((1, 1), (1, 3), (2, 3), (9, 9)):
            values = np.zeros(shape); values[0, 0] = 8.
            result = smooth_map(dict(values=values, included_luminosity_erg_s=8.),
                                enabled='gaussian', sigma_pixels=2.)
            self.assertAlmostEqual(result['values'].sum(), 8.)
            self.assertTrue(np.all(result['values'] >= 0))
            if shape == (9, 9):
                self.assertGreater(result['values'][0, 0], result['values'][-1, -1])

    def test_gaussian_public_api_matches_log_plot_and_preserves_exclusions(self):
        """Check that plotting preserves the prepared Gaussian map and its selection.

        Returns: result (NoneType): verifies plot data, metadata and selection.
        """
        options = dict(half_width_kpc=1, bins=8, smoothing='gaussian', sigma_pixels=.8)
        image = self.reader.make_map('A', **options)
        self.assertAlmostEqual(image['values'].sum(), 5.)
        self.assertEqual(image['excluded_luminosity_erg_s'], 5.)
        with patch('matplotlib.pyplot.show') as show:
            ax = self.reader.plot_map(image)
        show.assert_called_once_with()
        self.assertIsInstance(ax.collections[0].norm, LogNorm)
        plotted = np.ma.asarray(ax.collections[0].get_array()).reshape(image['values'].T.shape)
        np.testing.assert_allclose(plotted.filled(0), image['values'].T)
        self.assertIn('sigma 0.8', ax.get_title())
        compact = self.reader.make_map('A', smoothing='compact')
        enabled = self.reader.make_map('A', smoothing=True)
        np.testing.assert_array_equal(compact['values'], enabled['values'])

    def test_invalid_gaussian_options_fail_before_particle_reading(self):
        """Reject invalid widths and avoid silently using them for other kernels.

        Returns: result (NoneType): checks validation before expensive rendering.
        """
        with patch.object(self.reader, '_render') as render:
            for sigma in (0, -1, np.nan, np.inf, True, '0.5', [.5, .5]):
                with self.subTest(sigma=sigma), self.assertRaisesRegex(ValueError, 'sigma_pixels'):
                    self.reader.make_map('A', smoothing='gaussian', sigma_pixels=sigma)
            for mode in (True, False, 'compact'):
                with self.assertRaisesRegex(ValueError, 'only with'):
                    self.reader.make_map('A', smoothing=mode, sigma_pixels=2.)
            render.assert_not_called()


if __name__ == '__main__':
    unittest.main()
