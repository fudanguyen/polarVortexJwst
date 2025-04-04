from src.config.config_loader import ConfigLoader
from src.rendering.atmosphere_generator import AtmosphereGenerator
from src.rendering.visualization import Visualization
from datetime import datetime

def main():
    # Load configuration settings
    config_loader = ConfigLoader('path/to/config.txt')
    config = config_loader.load_config()

    # Initialize atmosphere generator
    atmosphere_generator = AtmosphereGenerator(config)

    # Start the atmosphere rendering process
    start_time = datetime.now()
    print("Starting atmosphere rendering...")

    # Generate the atmosphere
    atmosphere_data = atmosphere_generator.generate_atmosphere()

    # Visualize the atmosphere
    visualizer = Visualization(atmosphere_data)
    visualizer.render()

    print("Atmosphere rendering completed.")
    print("Elapsed Time:", datetime.now() - start_time)

if __name__ == "__main__":
    main()