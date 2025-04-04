from datetime import datetime
import numpy as np
import os
import h5py
import pickle
from tqdm import tqdm
import pyvista as pv
import warnings

class AtmosphereGenerator:
    def __init__(self, config):
        self.config = config
        self.plot_path = self.config['plot_path']
        self.xsize = self.config['xsize']
        self.ysize = self.config['ysize']
        self.Fambient = self.config['Fambient']
        self.Fband = self.config['Fband']
        self.Fpolar = self.config['Fpolar']
        self.Pband = self.config['Pband']
        self.Ppol = self.config['Ppol']
        self.inclination = self.config['inclination']
        self.frame_no = self.config['frame_no']
        self.t0 = self.config['t0']
        self.t1 = self.config['t1']
        self.modelname = self.config['modelname']
        self.modu_config = self.config['modu_config']
        self.metadata = {}

    def create_spherical_mesh(self):
        r = 1
        phi0 = np.linspace(0, np.pi, 200)
        theta0 = np.linspace(0, 2 * np.pi, 200)
        phi, theta = np.meshgrid(phi0, theta0)

        x = r * np.sin(phi) * np.cos(theta)
        y = r * np.sin(phi) * np.sin(theta)
        z = r * np.cos(phi)

        return x, y, z

    def generate_atmospheric_features(self, recmap, t):
        # Implement atmospheric feature generation logic here
        pass

    def run_simulation(self):
        start_time = datetime.now()
        for inclin in self.inclination:
            print(f'Running simulation for inclination: {inclin}')
            # Initialize the atmospheric map
            recmap = np.full((self.xsize, self.ysize), self.Fambient, dtype=np.float32)

            # Generate atmospheric features
            self.generate_atmospheric_features(recmap, t=0)

            # Save results
            self.save_results(recmap, inclin)

        print('Elapsed Time: ')
        print(datetime.now() - start_time)

    def save_results(self, recmap, inclin):
        folder_model = f'dataCube[{self.modelname}][{self.modu_config}][{inclin}]'
        folder_model_path = os.path.join(self.plot_path, folder_model)
        os.makedirs(folder_model_path, exist_ok=True)

        gray_array_path = os.path.join(folder_model_path, f'dataCube[{self.modu_config}][{self.modelname}][{inclin}].h5')
        with h5py.File(gray_array_path, 'w') as file:
            file.create_dataset('dataset', data=recmap)

        # Save metadata
        meta_path = os.path.join(folder_model_path, f'meta_dataCube[{self.modu_config}][{self.modelname}][{inclin}].pkl')
        with open(meta_path, 'wb') as file:
            pickle.dump(self.metadata, file)

    def load_metadata(self):
        # Implement metadata loading logic here
        pass

    def visualize(self):
        # Implement visualization logic here
        pass

    def generate_photometry(self):
        # Implement photometry generation logic here
        pass

    def limb_darkening(self, arr, u_coefficient=0.75):
        xlen, ylen = arr.shape
        if xlen != ylen:
            raise ValueError('Array must be square')
        xcen, ycen = xlen // 2, ylen // 2
        boundary_pixel = np.where(arr[:, ycen] > 0.)[0][0] - 1
        radius = xcen - boundary_pixel
        y, x = np.ogrid[:xlen, :ylen]
        distance_from_center = np.sqrt((x - xcen) ** 2 + (y - ycen) ** 2)
        mask = np.zeros((xlen, ylen))
        inside_circle = distance_from_center <= radius
        mask[inside_circle] = (1 - u_coefficient * (1 - np.sqrt((radius ** 2 - distance_from_center[inside_circle] ** 2) / (radius ** 2))))
        return mask

    # Additional methods for atmospheric feature generation can be added here.