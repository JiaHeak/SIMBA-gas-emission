"""Check that the same project schema accepts one-line and multi-line tables."""

from pathlib import Path
import tempfile
import unittest

import h5py
import numpy as np

from Pipeline.scr.tables import cloudy_hdf5


class TestCloudyHDF5Schema(unittest.TestCase):
    def test_one_and_six_lines_share_the_same_schema(self):
        with tempfile.TemporaryDirectory() as directory:
            filename = Path(directory) / 'table.hdf5'
            for line_count in (1, 6):
                with self.subTest(line_count=line_count):
                    with h5py.File(filename, 'w') as table:
                        table.attrs['schema'] = cloudy_hdf5.SCHEMA
                        table.attrs['abundance_model'] = 'fixed test abundances'
                        table['axes/log_temperature_K'] = [6.0, 7.0]
                        table['axes/log_nH_cm3'] = [-5.0, -4.0]
                        table.create_dataset('lines/labels', data=[f'line {i}' for i in range(line_count)],
                                             dtype=h5py.string_dtype())
                        values = table.create_dataset('lines/emissivity_erg_s_cm3',
                                                      data=np.full((2, 2, line_count), 1e-30))
                        values.attrs['unit'] = 'erg s^-1 cm^-3'
                        values.attrs['axis_order'] = 'log_temperature_K, log_nH_cm3, line'
                    result = cloudy_hdf5.read_cloudy_hdf5_table(filename)
                    self.assertEqual(len(result['line_labels']), line_count)
                    self.assertEqual(result['volume_emissivity_erg_s_cm3'].shape, (2, 2, line_count))

    def test_missing_or_unknown_schema_reports_actual_value(self):
        with tempfile.TemporaryDirectory() as directory:
            filename = Path(directory) / 'table.hdf5'
            for schema in (None, 'different_layout_v2'):
                with self.subTest(schema=schema):
                    with h5py.File(filename, 'w') as table:
                        if schema is not None:
                            table.attrs['schema'] = schema
                    with self.assertRaises(ValueError) as raised:
                        cloudy_hdf5.read_cloudy_hdf5_table(filename)
                    self.assertIn(repr(schema), str(raised.exception))
                    self.assertIn(cloudy_hdf5.SCHEMA, str(raised.exception))

    def test_schema_tag_does_not_hide_missing_datasets(self):
        with tempfile.TemporaryDirectory() as directory:
            filename = Path(directory) / 'incomplete.hdf5'
            with h5py.File(filename, 'w') as table:
                table.attrs['schema'] = cloudy_hdf5.SCHEMA
            with self.assertRaisesRegex(ValueError, 'missing required datasets.*lines/labels'):
                cloudy_hdf5.read_cloudy_hdf5_table(filename)


if __name__ == '__main__':
    unittest.main()
