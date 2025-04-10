"""
Created on April 1 2025

Upgrade from AtmosphereGenerator.py:
- significant improvement in speed via vectorization of routines
- gpu acceleration with pyvista but mainly cpu computation
- added multi-pressure capability

@author: nguyendat
"""
# IMPORT LIBRARIES
import h5py
import pickle
import os
# =============================================================================
# Enable vtk GPU-backend 
import vtk
# Set debug flags for VTK/OpenGL
os.environ["VTK_DEBUG_OPENGL"] = "1"
os.environ["VTK_REPORT_OPENGL_ERRORS"] = "1"
print("VTK Version:", vtk.vtkVersion.GetVTKVersion())
print("OpenGL2 Enabled:", hasattr(vtk, 'vtkOpenGLRenderWindow'))
# =============================================================================
import pyvista as pv
import tqdm
import numpy as np
from numba import jit
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.animation as animation
import warnings
from scipy.ndimage import gaussian_filter
import pandas as pd
from sklearn.decomposition import PCA
from datetime import datetime
import imageio
import cupy as cp
from PIL import Image
# =============================================================================
### Path management
import os
from os.path import join
folderarray = os.path.abspath('').split('/')
homedir = '/'
for i in range(len(folderarray)):
   homedir = join(homedir, folderarray[i])
plotPath = join(homedir, 'plot/')
# ==============================================================================
# Handles spherical mesh generation and geometric calculations
# =============================================================================
class SphericalMesh:
    def __init__(self, resolution=200, radius=1):
        self.radius = radius
        self.resolution = resolution
        self.phi, self.theta = None, None
        self.x, self.y, self.z = None, None, None
        self.generate_mesh()
        
    def generate_mesh(self):
        """Create spherical grid coordinates"""
        phi = np.linspace(0, np.pi, self.resolution)
        theta = np.linspace(0, 2*np.pi, self.resolution)
        self.phi, self.theta = np.meshgrid(phi, theta)
        
        self.x = self.radius * np.sin(self.phi) * np.cos(self.theta)
        self.y = self.radius * np.sin(self.phi) * np.sin(self.theta)
        self.z = self.radius * np.cos(self.phi)
    
    @property
    def shape(self):
        return self.x.shape
# ==============================================================================
# Manage parameters for atmospheric features
# =============================================================================
class AtmosphericConfig:
    def __init__(self, config, modu_config, modelname):
        self.config = config  # Band/pole parameters
        self.modu_config = modu_config  # Modulation type
        self.modelname = modelname
        self._validate_config()
        
    def _validate_config(self):
        """Ensure configuration has valid parameters"""
        required_keys = ['lat2', 'lat1', 'amp', 'typ', 'phase', 'period']
        for entry in self.config:
            if len(entry) != len(required_keys):
                raise ValueError("Invalid config entry")
# ==============================================================================
# Core atmospheric simulation logic
# =============================================================================
class AtmosphericModel:
    def __init__(self, mesh, config, speckey=None):
        """
        Args:
            mesh: SphericalMesh object (provides x, y, z coordinates)
            config: AtmosphericConfig object (simulation parameters)
            speckey: Dict mapping region types to spectral values
        """
        self.mesh = mesh
        self.config = config
        self.speckey = speckey or {'A': 0.25, 'B': 0.58, 'P': 0.75}
        
        # Derived properties from mesh
        self.xsize, self.ysize = self.mesh.shape
        self.xx, self.yy = np.meshgrid(np.arange(self.xsize), np.arange(self.ysize), indexing='ij')
        
        # Precompute latitude grid (vectorized)
        self.lat_grid = np.abs(self.yy - 90) / 180 * self.ysize  # From lat() function
        
    def generate_atmosphere(self, t, spec=False):
        """
        Generate atmospheric map at time `t`.
        
        Args:
            t: Current timestep
            spec: If True, return spectral map alongside flux map
            
        Returns:
            im (flux map) or (im, sm) tuple if spec=True
        """
        # Initialize base maps
        im = np.full((self.xsize, self.ysize), self.config.Fambient, dtype=np.float32)
        sm = np.full_like(im, self.speckey['A']) if spec else None
        
        # Apply all configured atmospheric features
        for group in self.config.config:
            lat2, lat1, amp, typ, phase, period = group
            lat_px1 = self._lat_px(lat1)
            lat_px2 = self._lat_px(lat2)
            
            # Vectorized latitude mask
            mask = (self.lat_grid >= lat_px2) & (self.lat_grid <= lat_px1)
            
            if typ.upper() == 'B':  # Band
                im = self._apply_planetary_wave(im, mask, t, amp, phase, period)
                
            elif typ.upper() == 'P':  # Polar
                im = self._apply_polar_effect(im, mask, t, amp, phase, period)
                
            if spec:
                sm[mask] = self.speckey[typ.upper()]
                
        # Apply vortices if needed
        if self.config.modu_config in ['polarStatic', 'polarDynamic']:
            im = self._apply_vortices(im, t)
            
        return (im, sm) if spec else im
    
    def _lat_px(self, lat_deg):
        """Convert latitude to pixel coordinate (vectorized)"""
        return np.abs(lat_deg - 90) / 180 * self.ysize
    
    def _apply_planetary_wave(self, im, mask, t, amp, phase, period):
        """Vectorized planetary wave implementation"""
        # Spatial frequency (1/wavelength)
        w = self.xsize  # Full circumference resolution
        sine_wave = 0.1 * amp * np.sin(
            2 * np.pi / w * (self.xx + (t / period) * w) + phase * np.pi / 180
        )
        im[mask] += amp + sine_wave[mask]
        return im
    
    def _apply_polar_effect(self, im, mask, t, amp, phase, period):
        """Polar cap modulation (vectorized)"""
        flux = amp + 0.01 * amp * np.sin(2 * np.pi / period * t + phase * np.pi / 180)
        im[mask] = flux
        return im
    
    def _apply_vortices(self, im, t):
        """Vectorized vortices implementation"""
        # Get polar regions from config
        polar_groups = [g for g in self.config.config if g[3].upper() == 'P']
        
        for group in polar_groups:
            lat2, lat1, *_ = group
            im = self._circle_vortice_vectorized(im, lat1, lat2, t)
            
        return im
    
    def _circle_vortice_vectorized(self, im, lat1, lat2, t):
        """Vectorized vortex generator"""
        # Vortex properties (from config)
        radius_frac = 0.3
        a, b = 0.75, 0.25  # Ellipse axes
        
        # Center coordinates
        center_lat = (lat1 + lat2) / 2
        center_px = self._lat_px(center_lat)
        
        # Vortex radius calculation
        area_cap = 2 * np.pi * (np.sin(np.radians(lat2)) - np.sin(np.radians(lat1)))
        r_vortice = np.sqrt(radius_frac * area_cap) * (self.xsize / np.pi)
        ar, br = a * r_vortice, b * r_vortice
        
        # Time-dependent longitudinal positions
        long_positions = self._equidistant_longitudes(t, group[-1])  # Rotation period
        
        # Vectorized mask for all vortices
        for long_px in long_positions:
            ellipse_mask = (
                ((self.xx - long_px) ** 2 / ar ** 2) +
                ((self.yy - center_px) ** 2 / br ** 2)
            ) <= 1
            
            lat_mask = (self.lat_grid >= self._lat_px(lat2)) & \
                       (self.lat_grid <= self._lat_px(lat1))
            
            im[ellipse_mask & lat_mask] += 0.2  # Flux enhancement
            
        return im
    
    def _equidistant_longitudes(self, t, rotation_period):
        """Calculate vortex positions (vectorized)"""
        n_vortices = 5  # From original code
        base_pos = np.linspace(0, self.xsize, n_vortices + 1)[:-1]
        drift = (t / rotation_period) * self.xsize
        return (base_pos + drift) % self.xsize
# ==============================================================================
# Visualization of atmospheric data using PyVista
# =============================================================================
class AtmosphereVisualizer:
    def __init__(self, mesh, inclination=0):
        self.mesh = mesh
        self.inclination = inclination
        self.plotter = self._create_plotter()
    
    def _create_plotter(self):
        """Configure PyVista plotter"""
        plotter = pv.Plotter(off_screen=True)
        plotter.background_color = 'black'
        plotter.camera.elevation = self.inclination + 56
        return plotter
    
    def render_frame(self, atmospheric_data):
        """Render single timestep"""
        self.plotter.clear()
        grid = pv.StructuredGrid(self.mesh.x, self.mesh.y, self.mesh.z)
        grid.point_data['scalars'] = atmospheric_data.ravel(order='F')
        self.plotter.add_mesh(grid, cmap='inferno')
        return self.plotter.screenshot()
# ==============================================================================
# Run the simulation and visualization
#===============================================================================
class SimulationRunner:
    def __init__(self, config_params, inclinations):
        self.mesh = SphericalMesh()
        self.config = AtmosphericConfig(**config_params)
        self.inclinations = inclinations
        self.results = {}
    
    def run_simulation(self, t0=0, t1=60, frames=60):
        time_array = np.linspace(t0, t1, frames)
        
        for inclin in self.inclinations:
            visualizer = AtmosphereVisualizer(self.mesh, inclin)
            model = AtmosphericModel(self.mesh, self.config)
            
            results = {
                'gray_array': [],
                'metadata': self.config.__dict__
            }
            
            for t in time_array:
                atmospheric_data = model.generate_atmosphere(t)
                frame = visualizer.render_frame(atmospheric_data)
                results['gray_array'].append(frame)
            
            self.results[inclin] = results
# ============================================================================
# Input ouput handler and data management
# ============================================================================
class DataManager:
    def __init__(self, base_path='output/'):
        self.base_path = base_path
        
    def save_simulation(self, results, prefix):
        with h5py.File(f'{self.base_path}/{prefix}.h5', 'w') as f:
            for inclin, data in results.items():
                f.create_dataset(f'{inclin}/gray', data=data['gray_array'])
                f.create_dataset(f'{inclin}/meta', data=str(data['metadata']))
                
# ==============================================================================                
# ==============================================================================
# Set up configurations and test call
# =============================================================================

# Example configuration
config = [
    # [lat2, lat1, amplitude, type, phase, period]
    [90, 65, 0.7, 'P', 0, 60],    # Polar cap
    [45, 38, 0.6, 'B', 10, 2.5],  # Mid-latitude band
    [-20, -40, 0.6, 'B', -26, 5]  # Southern band
]

atmo_config = AtmosphericConfig(
    config=config,
    modu_config='polarStatic',
    modelname='production1',
    Fambient=0.5,
    speckey={'A': 0.25, 'B': 0.58, 'P': 0.75}
)

# Initialize components
mesh = SphericalMesh(resolution=200)
config = AtmosphericConfig(...)
model = AtmosphericModel(mesh, config)

# Generate atmosphere at t=30
flux_map, spectral_map = model.generate_atmosphere(t=30, spec=True)

# Animate over time
for t in range(0, 2):
    flux_map = model.generate_atmosphere(t)
    # Render with AtmosphereVisualizer
    
runner = SimulationRunner(
    config_params=config_params,
    inclinations=[-30, 0, 30]
)

runner.run_simulation(t0=0, t1=60, frames=2)

DataManager().save_simulation(runner.results, 'simulation_run1')