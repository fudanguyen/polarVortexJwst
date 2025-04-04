import unittest
from src.config.config_loader import ConfigLoader

class TestConfigLoader(unittest.TestCase):

    def setUp(self):
        self.config_loader = ConfigLoader('path/to/config.txt')

    def test_load_config(self):
        config = self.config_loader.load_config()
        self.assertIsInstance(config, dict)
        self.assertIn('parameter1', config)
        self.assertIn('parameter2', config)

    def test_invalid_file(self):
        invalid_loader = ConfigLoader('invalid/path/to/config.txt')
        with self.assertRaises(FileNotFoundError):
            invalid_loader.load_config()

    def test_default_values(self):
        config = self.config_loader.load_config()
        self.assertEqual(config.get('parameter1'), 'default_value1')
        self.assertEqual(config.get('parameter2'), 'default_value2')

if __name__ == '__main__':
    unittest.main()