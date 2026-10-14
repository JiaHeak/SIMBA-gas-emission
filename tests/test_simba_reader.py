"""Small local fixtures: no HPC snapshot or CAESAR installation required."""
import importlib.util
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import h5py
import numpy as np


# Temporary import arrangement while the project is not yet an installed package.
spec = importlib.util.spec_from_file_location(
    'simba_reader', Path(__file__).resolve().parents[1] / 'scr' / 'simba.py'
)
simba = importlib.util.module_from_spec(spec)
spec.loader.exec_module(simba)


class TestGetComponent(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / 'tiny_snapshot.hdf5'
        with h5py.File(self.path, 'w') as f:
            f['PartType0/Density'] = [10., 20., 30.]
            f['PartType0/Coordinates'] = np.arange(9).reshape(3, 3)
            f['PartType0/ParticleIDs'] = np.array([101, 205, 309], dtype=np.uint64)
        self.fields = ['PartType0/Density', 'PartType0/Coordinates']

    def test_multiple_fields_and_full_read(self):
        result = simba.get_component(self.path, self.fields)
        self.assertEqual(set(result), set(self.fields))
        np.testing.assert_array_equal(result[self.fields[0]], [10, 20, 30])
        self.assertEqual(result[self.fields[1]].shape, (3, 3))

    def test_unsorted_repeated_indices_preserve_alignment(self):
        result = simba.get_component(self.path, self.fields, [2, 0, 2])
        np.testing.assert_array_equal(result[self.fields[0]], [30, 10, 30])
        np.testing.assert_array_equal(result[self.fields[1]], [[6, 7, 8], [0, 1, 2], [6, 7, 8]])

    def test_empty_selection_preserves_shape(self):
        result = simba.get_component(self.path, self.fields, [])
        self.assertEqual(result[self.fields[0]].shape, (0,))
        self.assertEqual(result[self.fields[1]].shape, (0, 3))

    def test_particle_ids_are_not_row_indices(self):
        with self.assertRaises(IndexError):
            simba.get_component(self.path, self.fields, [101])

    def test_invalid_inputs_raise(self):
        cases = [(['missing'], None, KeyError),
                 (['PartType0'], None, TypeError),
                 (self.fields, [0.5], TypeError),
                 (self.fields, [-1], IndexError),
                 (self.fields, [[0]], ValueError)]
        for fields, indices, error in cases:
            with self.subTest(fields=fields, indices=indices):
                with self.assertRaises(error):
                    simba.get_component(self.path, fields, indices)


class TestSavedSampleSnapshot(unittest.TestCase):
    def test_reads_synthetic_snapshot_fixture(self):
        path = Path(__file__).resolve().parent / "data" / "sample_snapshot.hdf5"
        fields = [
            "PartType0/Coordinates",
            "PartType0/Density",
            "PartType0/Metallicity",
            "PartType0/ParticleIDs",
        ]
        result = simba.get_component(path, fields, [6, 1, 6])

        self.assertEqual(result[fields[0]].shape, (3, 3))
        self.assertEqual(result[fields[1]].shape, (3,))
        self.assertEqual(result[fields[2]].shape, (3, 11))
        np.testing.assert_array_equal(result[fields[3]], [1007, 1002, 1007])

    def test_select_halo_particle_ids_preserves_caesar_order(self):
        path = Path(__file__).resolve().parent / 'data' / 'sample_snapshot.hdf5'
        galaxy = SimpleNamespace(
            glist=[6, 1, 6],
            halo=SimpleNamespace(glist=[1, 6]),
        )
        catalogue = SimpleNamespace(galaxies=[galaxy])
        with patch.object(simba, 'load', return_value=catalogue):
            galaxy_ids = simba.select_halo_particle_ids(path, 'mock_caesar.hdf5', 0)
            halo_ids = simba.select_halo_particle_ids(
                path, 'mock_caesar.hdf5', 0, find_host=True
            )
        np.testing.assert_array_equal(galaxy_ids, [1007, 1002, 1007])
        np.testing.assert_array_equal(halo_ids, [1002, 1007])

    def test_get_gas_fields_by_id_keeps_vector_fields_aligned(self):
        path = Path(__file__).resolve().parent / 'data' / 'sample_snapshot.hdf5'
        components = ['PartType0/Coordinates', 'PartType0/Metallicity']
        fields = simba.get_gas_fields_by_id(path, [1007, 1002, 1007], components)
        np.testing.assert_array_equal(fields['PartType0/ParticleIDs'], [1007, 1002, 1007])
        self.assertEqual(fields['PartType0/Coordinates'].shape, (3, 3))
        self.assertEqual(fields['PartType0/Metallicity'].shape, (3, 11))
        np.testing.assert_array_equal(
            fields['PartType0/Coordinates'][0], fields['PartType0/Coordinates'][2]
        )


if __name__ == '__main__':
    unittest.main()
