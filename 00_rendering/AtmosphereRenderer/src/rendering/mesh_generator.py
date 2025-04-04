from numpy import pi, sin, cos, meshgrid, sqrt, linspace, full, zeros, where, array
import os

class MeshGenerator:
    def __init__(self, radius=1, resolution=200):
        self.radius = radius
        self.resolution = resolution
        self.phi = linspace(0, pi, self.resolution)
        self.theta = linspace(0, 2 * pi, self.resolution)
        self.x, self.y, self.z = self._generate_mesh()

    def _generate_mesh(self):
        phi, theta = meshgrid(self.phi, self.theta)
        x = self.radius * sin(phi) * cos(theta)
        y = self.radius * sin(phi) * sin(theta)
        z = self.radius * cos(phi)
        return x, y, z

    def get_mesh(self):
        return self.x, self.y, self.z

    @staticmethod
    def create_folder(folder_path):
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
            print(f"Folder '{folder_path}' created successfully.")