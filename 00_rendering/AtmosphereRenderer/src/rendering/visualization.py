from tqdm import tqdm
import numpy as np
import pyvista as pv
from datetime import datetime
from PIL import Image
import os

class AtmosphereVisualizer:
    def __init__(self, plot_path, inline_plot=False):
        self.plot_path = plot_path
        self.inline_plot = inline_plot

    def save_image_with_cmap(self, data, filename, cmap='inferno', dpi=300):
        vmin, vmax = np.min(data), np.max(data)
        normalized = (np.clip(data, vmin, vmax) - vmin) / (vmax - vmin)
        cmap = plt.cm.get_cmap(cmap)
        rgba = cmap(normalized)
        img = Image.fromarray((rgba[..., :3] * 255).astype(np.uint8))
        img.save(filename, dpi=(dpi, dpi), format='PNG')

    def render_frame(self, plotter, grid, scalars):
        plotter.clear()
        plotter.add_mesh(grid, scalars=scalars.ravel(order='F'), cmap='inferno', clim=[0, 1], show_scalar_bar=False)
        plotter.render()
        return plotter.screenshot()

    def generate_spectral_map(self, x, y, z, config, inclin, metadata):
        plotter = self.configure_pyvista_plotter(inclin=inclin, zoom_factor=1.57)
        grid = pv.StructuredGrid(x, y, z)
        _, s = atmos_mesh(x, config, spec=True)
        grid.point_data['scalars'] = s.ravel(order='F')
        plotter.add_mesh(grid, scalars='scalars', cmap='plasma', clim=[0, 1], show_scalar_bar=False)
        plotter.camera_set = True
        plotter.render()
        specmap = plotter.screenshot()
        plotter.close()
        return self.process_spectral_map(specmap, metadata)

    def process_spectral_map(self, specmap, metadata):
        specgray = specmap[:, :, 0] * 0.2989 + specmap[:, :, 1] * 0.587 + specmap[:, :, 2] * 0.114
        con_amb = [0.2, 0.3]
        is_amb = ((specgray >= con_amb[0]) & (specgray < con_amb[1])).astype(int)
        con_band = [0.5, 0.6]
        is_band = ((specgray >= con_band[0]) & (specgray < con_band[1])).astype(int)
        con_pol = [0.6, 0.75]
        is_pol = ((specgray >= con_pol[0]) & (specgray < con_pol[1])).astype(int)
        total_count = is_amb.sum() + is_band.sum() + is_pol.sum()

        metadata.update({
            'specmap': specgray,
            'speckey': {'A': 0.25, 'B': 0.58, 'P': 0.75},
            'cond_is_amb': con_amb,
            'cond_is_band': con_band,
            'cond_is_pol': con_pol,
            'total_count': total_count,
            'is_amb': is_amb,
            'is_band': is_band,
            'is_pol': is_pol
        })
        return metadata

    def configure_pyvista_plotter(self, inclin=0, zoom_factor=1.01):
        plotter = pv.Plotter(off_screen=True, window_size=(500, 500))
        plotter.background_color = 'black'
        plotter.camera.SetParallelProjection(True)
        plotter.camera.elevation = inclin + 56
        return plotter

    def generate_photometry(self, x, y, z, config, inclin, time_array):
        plotter = self.configure_pyvista_plotter(inclin=inclin)
        grid = pv.StructuredGrid(x, y, z)

        imarray = []
        fluxarray = []
        gray_array = []
        limb_dark_mask = None

        progress = tqdm(enumerate(time_array), total=len(time_array), desc=f'Rendering i={inclin}')

        for frame_idx, ti in progress:
            # Implement the rendering logic here
            pass

        plotter.close()
        return {
            'imarray': imarray,
            'fluxarray': fluxarray,
            'gray_array': gray_array,
            'limb_mask': limb_dark_mask
        }