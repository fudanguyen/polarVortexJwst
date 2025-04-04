from scipy.ndimage import gaussian_filter
import numpy as np
from tqdm import tqdm
from PIL import Image

class Photometry:
    def __init__(self, atmosphere_data, inclination):
        self.atmosphere_data = atmosphere_data
        self.inclination = inclination
        self.photometry_results = []

    def calculate_photometry(self, time_array):
        for t in tqdm(time_array, desc='Calculating Photometry'):
            frame_data = self.render_frame(t)
            self.photometry_results.append(frame_data)
        return self.photometry_results

    def render_frame(self, t):
        # Placeholder for actual rendering logic
        # This should include the logic to generate the photometric data for the frame at time t
        frame_data = self.atmosphere_data * np.sin(t)  # Example operation
        return frame_data

    def save_image_with_cmap(self, data, filename, cmap='inferno', dpi=300):
        vmin, vmax = np.min(data), np.max(data)
        normalized = (np.clip(data, vmin, vmax) - vmin) / (vmax - vmin)
        cmap = plt.cm.get_cmap(cmap)
        rgba = cmap(normalized)
        img = Image.fromarray((rgba[..., :3] * 255).astype(np.uint8))
        img.save(filename, dpi=(dpi, dpi), format='PNG')

    def generate_photometry_images(self, output_path):
        for idx, frame in enumerate(self.photometry_results):
            filename = f"{output_path}/photometry_frame_{idx}.png"
            self.save_image_with_cmap(frame, filename)