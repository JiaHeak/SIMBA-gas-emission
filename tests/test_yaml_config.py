"""Check commented YAML settings and their numerical types."""

import math
from pathlib import Path
import tempfile
import unittest

import yaml

from Pipeline.scr import input as pipeline_input


class TestYamlConfig(unittest.TestCase):
    def test_comments_nested_settings_arrays_and_numbers(self):
        with tempfile.TemporaryDirectory() as directory:
            filename = Path(directory) / 'config.yaml'
            filename.write_text(
                '# A user comment\n'
                'selection:\n'
                '  group_type: halo  # Inline comment\n'
                '  target_index: 0\n'
                'line_label: "H  1      6562.81A"\n'
                'mass_to_g: 2.9241321628e+43\n'
                'periodic: true\n'
                'center_kpc: auto\n'
                'x_edges_kpc: [-100, 0, 100]\n',
                encoding='utf-8',
            )
            config = pipeline_input.load_config(filename)
        self.assertEqual(config['selection'], {'group_type': 'halo', 'target_index': 0})
        self.assertEqual(config['line_label'], 'H  1      6562.81A')
        self.assertIsInstance(config['mass_to_g'], float)
        self.assertIs(config['periodic'], True)
        self.assertEqual(config['center_kpc'], 'auto')
        self.assertEqual(config['x_edges_kpc'], [-100, 0, 100])

    def test_example_has_usable_settings_and_numeric_units(self):
        root = Path(__file__).resolve().parents[1]
        config = pipeline_input.read_parameters(root / 'config.example.yaml')
        # The YAML is editable: it need not match the historical JSON example.
        self.assertEqual(config['source']['type'], 'simba')
        self.assertEqual(config['selection']['group_type'], 'halo')
        index = config['selection']['target_index']
        self.assertIsInstance(index, int)
        self.assertNotIsInstance(index, bool)
        self.assertGreaterEqual(index, 0)
        self.assertTrue(config['source']['catalogue_filename'])
        self.assertTrue(config['source']['snapshot_filename'])
        self.assertIn(config['emissivity']['type'], ('ps20', 'cloudy_hdf5'))
        self.assertTrue(config['emissivity']['table_filename'])
        self.assertIsInstance(config['emissivity']['line_labels'], list)
        self.assertTrue(config['emissivity']['line_labels'])
        required_units = {
            'solar_mass_g', 'kpc_to_cm', 'specific_energy_to_erg_g', 'z_sun_mass_fraction',
        }
        self.assertTrue(required_units <= config['unit_factors'].keys())
        for name in required_units:
            value = config['unit_factors'][name]
            self.assertIsInstance(value, (int, float))
            self.assertNotIsInstance(value, bool)
            self.assertTrue(math.isfinite(value))
            self.assertGreater(value, 0)

    def test_empty_or_non_mapping_document_raises(self):
        for content in ('', '# Comments only\n', '[1, 2]', 'some text'):
            with self.subTest(content=content), tempfile.TemporaryDirectory() as directory:
                filename = Path(directory) / 'invalid.yaml'
                filename.write_text(content, encoding='utf-8')
                with self.assertRaisesRegex(ValueError, 'YAML mapping'):
                    pipeline_input.load_config(filename)

    def test_unsafe_python_tags_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            filename = Path(directory) / 'unsafe.yaml'
            filename.write_text('value: !!python/tuple [1, 2]\n', encoding='utf-8')
            with self.assertRaises(yaml.YAMLError):
                pipeline_input.load_config(filename)


if __name__ == '__main__':
    unittest.main()
