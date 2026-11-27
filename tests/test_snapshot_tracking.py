"""Small synthetic snapshots exercise identity matching and epoch-specific units."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import h5py
import numpy as np

from Reader_Object import read_RawSnapshots, read_RawSnapshot


class TestSnapshotTracking(unittest.TestCase):
    def setUp(self):
        """Create shuffled snapshots including a gap and IDs above signed int64.

        Returns: result (NoneType): initializes temporary test files.
        """
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.big = 2**63 + 17
        self.current = self.write_snapshot('snapshot_120.hdf5', 1., [11, self.big, 33, 44])
        self.middle = self.write_snapshot('snapshot_118.hdf5', .5, [44, 11, 999])
        self.early = self.write_snapshot('snapshot_116.hdf5', .25, [33, self.big, 11, 44])
        self.reader = read_RawSnapshots([self.early, self.current, self.middle])

    def write_snapshot(self, name, a, ids):
        """Write distinct per-row vectors and minimal physical gas state.

        Returns: path (pathlib.Path): synthetic single-file snapshot.
        """
        path = self.root / name
        n = len(ids)
        with h5py.File(path, 'w') as output:
            output.create_group('Header').attrs.update(
                Time=a, HubbleParam=.5, BoxSize=100., NumFilesPerSnapshot=1,
                NumPart_ThisFile=[n, 0, 0, 0, 0, 0])
            gas = output.create_group('PartType0')
            gas['ParticleIDs'] = np.array(ids, dtype=np.uint64)
            gas['Coordinates'] = np.arange(n * 3, dtype=float).reshape(n, 3)
            gas['Velocities'] = np.full((n, 3), 10.)
            gas['Masses'] = np.arange(n, dtype=float) + 1
            gas['InternalEnergy'] = np.full(n, 100.)
            gas['Metallicity'] = np.tile([.02, .24], (n, 1))
            gas['ElectronAbundance'] = np.zeros(n)
        return path

    def track(self, **overrides):
        """Track an unsorted source selection using catalogue IDs for verification.

        Returns: tracks (ParticleTracks): synthetic matches.
        """
        options = dict(selected_rows=np.array([3, 1, 0]), current_snapshot=self.current,
                       target_snapshots=[self.middle, self.early],
                       expected_particle_ids=np.array([44, self.big, 11], dtype=np.uint64))
        options.update(overrides)
        return self.reader.track_backward(**options)

    def test_matching_preserves_order_uint64_and_recovers_after_gap(self):
        """Verify direct source matching preserves a particle missing in the middle.

        Returns: result (NoneType): checks lookup and cache behavior.
        """
        self.assertEqual(self.reader._cache, {})
        tracks = self.track()
        np.testing.assert_array_equal(tracks.rows[self.middle.name], [0, -1, 1])
        np.testing.assert_array_equal(tracks.rows[self.early.name], [3, 1, 2])
        self.assertEqual(int(tracks.particle_ids[1]), self.big)
        self.assertEqual(len(self.reader._cache), 3)
        cached = self.reader._cache[self.middle.name]
        self.track()
        self.assertIs(self.reader._cache[self.middle.name], cached)
        self.reader.clear_cache()
        self.assertEqual(self.reader._cache, {})
        with patch('builtins.print'):
            self.assertEqual(tracks.describe()[self.middle.name]['missing'], 1)

    def test_fields_use_target_units_mask_missing_and_preserve_vector_alignment(self):
        """Check physical coordinates, peculiar velocities, mass and temperature.

        Returns: result (NoneType): verifies fields against independent expectations.
        """
        tracks = self.track()
        result = tracks.get_fields(snapshot=self.middle.name,
                                   fields=['position_kpc', 'velocity_kms', 'mass_g', 'T_K'], chunk_size=1)
        np.testing.assert_array_equal(result.found, [True, False, True])
        np.testing.assert_array_equal(np.ma.getmaskarray(result.values['position_kpc']),
                                      [[False]*3, [True]*3, [False]*3])
        np.testing.assert_allclose(result.values['position_kpc'][[0, 2]], [[0, 1, 2], [3, 4, 5]])
        np.testing.assert_allclose(result.values['velocity_kms'][[0, 2]], 10 * np.sqrt(.5))
        np.testing.assert_allclose(result.values['mass_g'][[0, 2]], np.array([1., 2.])*1.e10*1.988416e33/.5)
        expected_temperature = (2./3.)*100*1.e10/(.74 + .24/4)*1.67262192369e-24/1.380649e-16
        np.testing.assert_allclose(result.values['T_K'][[0, 2]], expected_temperature)
        earlier = tracks.get_fields(snapshot=self.early, fields='position_kpc')
        np.testing.assert_allclose(earlier.values['position_kpc'],
                                  np.array([[9, 10, 11], [3, 4, 5], [6, 7, 8]])*.5)

    def test_missing_and_empty_selections_never_read_minus_one(self):
        """Cover no matches, absent gas groups, and a genuinely empty selection.

        Returns: result (NoneType): verifies missing and empty read behavior.
        """
        empty = self.write_snapshot('empty.hdf5', .1, [])
        with h5py.File(empty, 'a') as source:
            del source['PartType0']
        reader = read_RawSnapshots([self.current, empty])
        tracks = reader.track_backward(selected_rows=np.array([0]), current_snapshot=self.current,
                                        target_snapshots=[empty])
        with patch.object(read_RawSnapshot(self.current).__class__, 'read_gas_fields',
                          side_effect=AssertionError('Must not read unmatched rows')):
            values = tracks.get_fields(snapshot=empty, fields=['T_K', 'position_kpc'])
        self.assertTrue(values.values['T_K'].mask.all())
        np.testing.assert_array_equal(values.found, [False])
        none = self.track(selected_rows=np.array([], dtype=int), expected_particle_ids=np.array([], dtype=np.uint64))
        self.assertEqual(none.get_fields(snapshot=self.early, fields='position_kpc').values['position_kpc'].shape, (0, 3))

    def test_invalid_rows_ids_direction_and_ptype_raise(self):
        """Reject wrong source identity, unsupported types and malformed rows.

        Returns: result (NoneType): verifies explicit input failures.
        """
        for changes in [dict(selected_rows=[1., 2.]), dict(selected_rows=[-1]),
                        dict(selected_rows=[1, 1]), dict(expected_particle_ids=[44, 1, 11]),
                        dict(target_snapshots=[self.current]), dict(ptype='stars'),
                        dict(current_snapshot=self.early, target_snapshots=[self.middle])]:
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                self.track(**changes)
        with self.assertRaises(IndexError):
            self.track(selected_rows=[4])
        with self.assertRaises(KeyError):
            self.track(current_snapshot=self.root/'wrong'/'snapshot_120.hdf5')

    def test_duplicate_ids_in_source_or_target_raise(self):
        """Never choose one of several gas rows with the same ID.

        Returns: result (NoneType): verifies ambiguity detection in both directions.
        """
        for filename, changed_row in ((self.current, 2), (self.middle, 2)):
            with self.subTest(filename=filename):
                with h5py.File(filename, 'a') as source:
                    original = source['PartType0/ParticleIDs'][:]
                    source['PartType0/ParticleIDs'][changed_row] = original[0]
                self.reader.clear_cache()
                with self.assertRaisesRegex(ValueError, rf'{filename.name}.*1 selected ID'):
                    self.track()
                with h5py.File(filename, 'a') as source:
                    source['PartType0/ParticleIDs'][:] = original

    def test_duplicates_outside_selection_allow_tracking(self):
        """Keep unambiguous tracks when other gas IDs occur more than once.

        Returns: result (NoneType): checks that unrelated duplicates leave matches intact.
        """
        with h5py.File(self.current, 'a') as source:
            source['PartType0/ParticleIDs'][2] = 11
        with h5py.File(self.middle, 'a') as source:
            source['PartType0/ParticleIDs'][1] = 999
        tracks = self.track(selected_rows=[3, 1],
                            expected_particle_ids=np.array([44, self.big], dtype=np.uint64))
        np.testing.assert_array_equal(tracks.rows[self.current.name], [3, 1])
        np.testing.assert_array_equal(tracks.rows[self.middle.name], [0, -1])
        np.testing.assert_array_equal(tracks.rows[self.early.name], [3, 1])

    def test_metadata_and_wrong_epoch_units_raise(self):
        """Reject incompatible cosmology, split snapshots and mixed epoch factors.

        Returns: result (NoneType): verifies metadata and conversion validation.
        """
        for key, value, error in [('BoxSize', 200., 'BoxSize'), ('NumFilesPerSnapshot', 2, 'split')]:
            with h5py.File(self.early, 'a') as source:
                old = source['Header'].attrs[key]
                source['Header'].attrs[key] = value
            with self.assertRaisesRegex(ValueError, error):
                read_RawSnapshots([self.current, self.early])
            with h5py.File(self.early, 'a') as source:
                source['Header'].attrs[key] = old
        with self.assertRaisesRegex(ValueError, 'wrong scale_factor'):
            read_RawSnapshot(self.early).read_gas_fields(np.array([0]), fields='mass_g',
                                                       unit_factors=read_RawSnapshot(self.current).unit_factors)
        custom = dict(read_RawSnapshot(self.early).unit_inputs, solar_mass_g=2.e33)
        values = read_RawSnapshot(self.early, unit_inputs=custom).read_gas_fields(np.array([0]), fields='mass_g')
        np.testing.assert_allclose(values['mass_g'], [4.e43])


if __name__ == '__main__':
    unittest.main()
