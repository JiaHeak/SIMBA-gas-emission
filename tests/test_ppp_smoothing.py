"""Small checks of conservative smoothing after PPP binning."""

import unittest

import numpy as np

from Pipeline.scr.ppp_cube import make_ppp_cube


class TestPPPSmoothing(unittest.TestCase):
    def test_center_impulse_kernel_and_width(self):
        gas = {f'{axis}_kpc': [2.5] for axis in 'xyz'}
        edges = np.arange(6.)
        result = make_ppp_cube(gas, [8.], edges, edges, edges)
        weights = np.array([0., .125, .75, .125, 0.])
        expected = 8 * np.einsum('i,j,k->ijk', weights, weights, weights)
        np.testing.assert_array_equal(result['cube_erg_s'], expected)
        self.assertEqual(expected.sum(), 8.)
        self.assertEqual(np.count_nonzero(expected), 27)
        self.assertAlmostEqual(np.sum(expected * (np.arange(5)[:, None, None] - 2)**2) / 8, .25)
        self.assertEqual(result['smoothing_sigma_voxels'], .5)

    def test_boundary_conservation_exclusions_and_no_wrap(self):
        gas = {f'{axis}_kpc': [.5, -.5] for axis in 'xyz'}
        edges = np.arange(5.)
        result = make_ppp_cube(gas, [8., 1.e6], edges, edges, edges)
        weights = np.array([.875, .125, 0., 0.])
        np.testing.assert_array_equal(result['cube_erg_s'], 8 * np.einsum('i,j,k->ijk', weights, weights, weights))
        self.assertEqual(result['cube_erg_s'].sum(), 8.)
        self.assertEqual(result['excluded_luminosity_erg_s'], 1.e6)
        self.assertEqual(result['excluded_particle_count'], 1)

    def test_unequal_volumes_disable_and_singleton_axes(self):
        gas = {'x_kpc': [.5], 'y_kpc': [.5], 'z_kpc': [.5]}
        edges = ([0, 1, 4], [0, 1], [0, 1])
        result = make_ppp_cube(gas, [8.], *edges, kpc_to_cm=1.)
        np.testing.assert_array_equal(result['cube_erg_s'][:, 0, 0], [7., 1.])
        np.testing.assert_allclose(result['volume_emissivity_erg_s_cm3'][:, 0, 0], [7., 1./3])
        np.testing.assert_allclose(result['volume_emissivity_erg_s_cm3'] * result['voxel_volume_cm3'], result['cube_erg_s'])
        raw = make_ppp_cube(gas, [8.], *edges, smoothing=False)
        np.testing.assert_array_equal(raw['cube_erg_s'][:, 0, 0], [8., 0.])
        self.assertFalse(raw['smoothing_enabled'])
        for bad in (0, 'yes', None):
            with self.assertRaisesRegex(ValueError, 'smoothing'):
                make_ppp_cube(gas, [8.], *edges, smoothing=bad)


if __name__ == '__main__':
    unittest.main()
