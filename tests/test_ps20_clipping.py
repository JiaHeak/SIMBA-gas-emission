"""Check PS20 metallicity clipping while preserving the primordial slice."""

from pathlib import Path
import tempfile
import unittest
import warnings

import h5py
import numpy as np

from Pipeline.scr.tables import ps20


LINE_LABEL = 'H  1      6562.81A'


class TestPS20Clipping(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.table_path = Path(self.directory.name) / 'tiny_ps20.hdf5'
        with h5py.File(self.table_path, 'w') as table:
            bins = table.create_group('TableBins')
            bins['RedshiftBins'] = [0.0, 1.0]
            bins['TemperatureBins'] = [3.0, 4.0]
            bins['MetallicityBins'] = [-50.0, -1.0, 0.0]
            bins['DensityBins'] = [-3.0, -2.0]
            table['IdentifierLines'] = [LINE_LABEL.encode()]
            emissivity = np.empty((2, 2, 3, 2, 1))
            for redshift_index in range(2):
                emissivity[redshift_index, :, 0, :, 0] = 1 + 2 * redshift_index
                emissivity[redshift_index, :, 1, :, 0] = 4 + 2 * redshift_index
                emissivity[redshift_index, :, 2, :, 0] = 6 + 2 * redshift_index
            table['Tdep/EmissivitiesVol'] = emissivity

    def test_both_limits_clip_but_zero_and_input_arrays_are_preserved(self):
        metallicity = np.array([0.0, 1e-20, 0.2, 9.0, 0.1, 1.0])
        original = metallicity.copy()
        gas = {
            'nH_cm3': np.full(6, 10.0 ** -2.5),
            'T_K': np.full(6, 10.0 ** 3.5),
            'Z_Zsun': metallicity,
            'volume_cm3': np.ones(6),
        }
        with self.assertWarnsRegex(RuntimeWarning, '1 below, 1 above'):
            result = ps20.calculate_particle_line_luminosity(
                gas, table_filename=str(self.table_path), redshift=0.5,
                line_label=LINE_LABEL,
            )
        expected = [2.0, 5.0, 7 + 2 * np.log10(0.2), 7.0, 5.0, 7.0]
        np.testing.assert_allclose(result['log_volume_emissivity'], expected)
        np.testing.assert_allclose(result['line_luminosity_erg_s'], 10.0 ** np.array(expected))
        np.testing.assert_array_equal(metallicity, original)

    def test_direct_interpolation_preserves_broadcast_shape_and_inputs(self):
        log_Z = np.array([-3.0, 2.0])
        with self.assertWarnsRegex(RuntimeWarning, '2 below, 2 above'):
            result = ps20.interpolate_log_emissivity(
                str(self.table_path), log_nH=-2.5,
                log_T=np.array([[3.2], [3.8]]), log_Z=log_Z,
                redshift=0.5, line_label=LINE_LABEL,
            )
        np.testing.assert_allclose(result, [[5, 7], [5, 7]])
        np.testing.assert_array_equal(log_Z, [-3, 2])

    def test_in_range_scalar_is_unchanged_without_warning(self):
        with warnings.catch_warnings(record=True) as messages:
            warnings.simplefilter('always')
            result = ps20.interpolate_log_emissivity(
                str(self.table_path), -2.5, 3.5, -0.5, 0.5, LINE_LABEL,
            )
        self.assertEqual(messages, [])
        self.assertEqual(result.shape, ())
        self.assertAlmostEqual(float(result), 6.0)

    def test_other_axes_are_not_clipped(self):
        for log_nH, log_T, redshift in ((-4, 3.5, .5), (-2.5, 2, .5), (-2.5, 3.5, 2)):
            with self.subTest(log_nH=log_nH, log_T=log_T, redshift=redshift):
                with self.assertRaises(ValueError):
                    ps20.interpolate_log_emissivity(
                        str(self.table_path), log_nH, log_T, -.5, redshift, LINE_LABEL,
                    )

    def test_nonfinite_metallicity_is_rejected_before_clipping(self):
        for log_Z in (np.nan, np.inf, -np.inf):
            with self.subTest(log_Z=log_Z):
                with self.assertRaisesRegex(ValueError, 'finite values before clipping'):
                    ps20.interpolate_log_emissivity(
                        str(self.table_path), -2.5, 3.5, log_Z, .5, LINE_LABEL,
                    )


if __name__ == '__main__':
    unittest.main()
