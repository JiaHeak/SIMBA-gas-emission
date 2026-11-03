"""Check spatial cube orientation, physical normalization, and exclusions."""

import unittest

import numpy as np

from Pipeline.scr.ppp_cube import make_ppp_cube


class TestPPPCube(unittest.TestCase):
    def test_asymmetric_axes_and_nonuniform_voxel_volumes(self):
        gas = {'x_kpc': np.array([0.5, 2.0]),
               'y_kpc': np.array([1.0, 3.0]),
               'z_kpc': np.array([5.0, 25.0])}
        result = make_ppp_cube(gas, [2.0, 6.0], [0, 1, 3], [0, 2, 4], [0, 10, 20, 30], smoothing=False)
        expected = np.zeros((2, 2, 3))
        expected[0, 0, 0] = 2
        expected[1, 1, 2] = 6
        np.testing.assert_array_equal(result['cube_erg_s'], expected)
        self.assertEqual(result['cube_axis_order'], 'x,y,z')
        self.assertAlmostEqual(result['voxel_volume_cm3'][0, 0, 0] / (3.0856775814913673e21 ** 3), 20)
        self.assertAlmostEqual(result['voxel_volume_cm3'][1, 1, 2] / (3.0856775814913673e21 ** 3), 40)
        np.testing.assert_allclose(
            result['volume_emissivity_erg_s_cm3'] * result['voxel_volume_cm3'], expected,
        )
        self.assertEqual(result['cube_erg_s'].sum(), 8)
        self.assertEqual(result['excluded_particle_count'], 0)

    def test_exclusions_and_final_upper_edges_are_explicit(self):
        gas = {'x_kpc': [-1, 0, 1, 2], 'y_kpc': [0, 0, 1, 0], 'z_kpc': [0, 0, 1, 0]}
        result = make_ppp_cube(gas, [10, 2, 3, 20], [0, 1], [0, 1], [0, 1])
        self.assertEqual(result['input_particle_count'], 4)
        self.assertEqual(result['included_particle_count'], 2)
        self.assertEqual(result['excluded_particle_count'], 2)
        self.assertEqual(result['cube_erg_s'].sum(), 5)
        self.assertEqual(result['included_luminosity_erg_s'], 5)
        self.assertEqual(result['excluded_luminosity_erg_s'], 30)
        self.assertEqual(result['input_luminosity_erg_s'], 35)

    def test_empty_input_returns_zero_cube(self):
        result = make_ppp_cube({f'{axis}_kpc': [] for axis in 'xyz'}, [], [0, 1], [0, 1], [0, 1])
        self.assertEqual(result['cube_erg_s'].shape, (1, 1, 1))
        self.assertEqual(result['cube_erg_s'].sum(), 0)
        self.assertEqual(result['volume_emissivity_erg_s_cm3'].sum(), 0)
        self.assertEqual(result['input_particle_count'], 0)

    def test_invalid_edges_raise(self):
        gas = {f'{axis}_kpc': [0.0] for axis in 'xyz'}
        for bad in ([], [0], [0, 0], [1, 0], [0, np.nan], [[0, 1]]):
            for axis_index in range(3):
                with self.subTest(edges=bad, axis=axis_index):
                    edges = [[0, 1], [0, 1], [0, 1]]
                    edges[axis_index] = bad
                    with self.assertRaises(ValueError):
                        make_ppp_cube(gas, [1.0], *edges)

    def test_invalid_particle_arrays_raise(self):
        for gas, luminosity in [
            ({'x_kpc': [0, 1], 'y_kpc': [0], 'z_kpc': [0]}, [1]),
            ({'x_kpc': [np.nan], 'y_kpc': [0], 'z_kpc': [0]}, [1]),
            ({'x_kpc': [[0]], 'y_kpc': [0], 'z_kpc': [0]}, [1]),
            ({'x_kpc': [0], 'y_kpc': [0], 'z_kpc': [0]}, [-1]),
            ({'x_kpc': [0], 'y_kpc': [0], 'z_kpc': [0]}, [np.inf]),
        ]:
            with self.subTest(gas=gas, luminosity=luminosity):
                with self.assertRaises(ValueError):
                    make_ppp_cube(gas, luminosity, [0, 1], [0, 1], [0, 1])
        with self.assertRaises(KeyError):
            make_ppp_cube({}, [], [0, 1], [0, 1], [0, 1])

    def test_inputs_are_not_modified(self):
        gas = {f'{axis}_kpc': np.array([0.5]) for axis in 'xyz'}
        luminosity = np.array([2.0])
        edges = np.array([0.0, 1.0])
        make_ppp_cube(gas, luminosity, edges, edges, edges)
        for values in gas.values():
            np.testing.assert_array_equal(values, [0.5])
        np.testing.assert_array_equal(luminosity, [2.0])
        np.testing.assert_array_equal(edges, [0.0, 1.0])

    def test_multiple_particles_sum_before_voxel_emissivity_conversion(self):
        gas = {f'{axis}_kpc': [0.25, 0.75] for axis in 'xyz'}
        result = make_ppp_cube(gas, [2.0, 6.0], [0, 1], [0, 1], [0, 1])
        self.assertEqual(result['cube_erg_s'][0, 0, 0], 8.0)
        self.assertAlmostEqual(
            result['volume_emissivity_erg_s_cm3'][0, 0, 0] * (3.0856775814913673e21 ** 3),
            8.0,
        )


if __name__ == '__main__':
    unittest.main()
