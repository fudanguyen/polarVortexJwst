class ConfigLoader:
    def __init__(self, config_file):
        self.config_file = config_file
        self.configurations = {}

    def load_config(self):
        try:
            with open(self.config_file, 'r') as file:
                for line in file:
                    if line.strip() and not line.startswith('#'):
                        key, value = line.split('=')
                        self.configurations[key.strip()] = self._parse_value(value.strip())
        except FileNotFoundError:
            raise Exception(f"Configuration file '{self.config_file}' not found.")
        except Exception as e:
            raise Exception(f"Error loading configuration: {e}")

    def _parse_value(self, value):
        if value.isdigit():
            return int(value)
        try:
            return float(value)
        except ValueError:
            return value  # Return as string if not a number

    def get(self, key, default=None):
        return self.configurations.get(key, default)