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
# Manage config and parameters
# =============================================================================
class TimeConfig:
    """Centralized management of temporal parameters"""
    def __init__(self, t0=0, t1=60, frames=60):
        """
        Args:
            t0: Start time (hours)
            t1: End time (hours)
            frames: Number of animation frames
        """
        self.t0 = t0
        self.t1 = t1
        self.frames = frames
        
        # Derived properties
        self.time_array = np.linspace(t0, t1, frames)
        self.dt = (t1 - t0) / frames  # Time step
        
    def __repr__(self):
        return f"TimeConfig(t0={self.t0}, t1={self.t1}, frames={self.frames})"
# =============================================================================
class AtmosphericConfig:
    """Combined atmospheric and temporal configuration"""
    def __init__(self, 
                 band_config: list,
                 modu_config: str,
                 modelname: str,
                 time_config: TimeConfig,
                 Fband: float = 0.6,
                 Fambient: float  = 0.5,
                 Fpolar: float  = 0.7,
                 Pband: float  = 5.0,
                 Ppol: float  = 60.0,
                 speckey: dict = None):
        """
        Args:
            band_config: Atmospheric band parameters
            modu_config: Modulation type ('polarStatic' etc)
            modelname: Simulation identifier
            time_config: TimeConfig object
            Fambient/band/pole: Ambient/band/pole base contrast value
            Pband/pole: Band/pole period (in hours)
            speckey: Spectral value mapping
        """
        self.band_config = band_config
        self.modu_config = modu_config
        self.modelname = modelname
        self.time_config = time_config
        self.Fambient = Fambient
        self.Fband = Fband
        self.Fpolar = Fpolar
        self.Pband = Pband
        self.Ppol = Ppol
        self.speckey = speckey or {'A': 0.25, 'B': 0.58, 'P': 0.75}
        
        self._validate_config()

    def _validate_config(self):
        """Sanity checks for configuration"""
        if not isinstance(self.time_config, TimeConfig):
            raise TypeError("time_config must be TimeConfig instance")
        
        required_band_keys = ['lat2', 'lat1', 'amp', 'typ', 'phase', 'period']
        for band in self.band_config:
            if len(band) != len(required_band_keys):
                raise ValueError("Invalid band configuration")
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
        for group in self.config.band_config:
            lat2, lat1, amp, typ, phase, period = group
            lat_px1 = self._lat_px(lat1)
            lat_px2 = self._lat_px(lat2)
            
            # Vectorized latitude mask
            mask = (self.yy >= lat_px2) & (self.yy <= lat_px1)
            
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
        sine_wave = amp * np.sin(
            2 * np.pi / w * (self.xx + (t / period) * w) + phase * np.pi / 180
        )
        im[mask] += sine_wave[mask]
        return im
    
    def _apply_polar_effect(self, im, mask, t, amp, phase, period):
        """Polar cap modulation (vectorized)"""
        flux = amp * np.sin(2 * np.pi / period * t + phase * np.pi / 180)
        im[mask] += flux
        return im
    
    def _apply_vortices(self, im, t):
        """Vectorized vortices implementation"""
        # Get polar regions from config
        polar_groups = [g for g in self.config.band_config if g[3].upper() == 'P']
        
        for group in polar_groups:
            lat2, lat1, *_ = group
            im = self._circle_vortice_vectorized(im, lat1, lat2, t, group)  # Pass full group
            
        return im
    
    def _circle_vortice_vectorized(self, im, lat1, lat2, t, group):
        """Vectorized vortex generator (corrected)"""
        # Sort latitudes (lat1 > lat2)
        lat1, lat2 = sorted([lat1, lat2], reverse=True)
        center_lat = (lat1 + lat2) / 2
        center_px = self._lat_px(center_lat)
        lat_px1 = self._lat_px(lat1)
        lat_px2 = self._lat_px(lat2)
        
        # Vortex properties (from config)
        radius_frac = 0.3
        a, b = 0.75, 0.25  # Ellipse axes
        # Make sure area_cap doesn't become negative 
        # when lat1 < lat2 (e.g., southern hemisphere).
        area_cap = 2 * np.pi * abs(np.sin(np.radians(lat1)) - np.sin(np.radians(lat2)))
        r_vortice = np.sqrt(radius_frac * area_cap) * (self.xsize / np.pi)
        ar, br = a * r_vortice, b * r_vortice
        
        # Time-dependent longitudinal positions (corrected drift)
        rotation_period = group[5]
        long_positions = self._equidistant_longitudes(t, rotation_period)
        
        # Latitude mask
        lat_mask = (self.yy >= lat_px2) & (self.yy <= lat_px1)
        
        # Amplitude from polar region
        amplitude = im[int(self.xsize / 2), self.ysize - 1]  # Central pixel
        
        # Generate grid
        xx, yy = np.meshgrid(np.arange(self.xsize), np.arange(self.ysize), indexing='ij')
        
        for long_px in long_positions:
            xi = int(long_px)
            ellipse_mask = (
                ((xx - xi) ** 2 / ar ** 2) + 
                ((yy - center_px) ** 2 / br ** 2)
            ) <= 1
            
            im[ellipse_mask & lat_mask] += 0.2  # Dynamic amplitude + flux
        
        return im
    
    def _equidistant_longitudes(self, t, rotation_period):
        """Calculate vortex positions (vectorized)"""
        n_vortices = 5  # From original code
        base_pos = np.linspace(0, self.xsize, n_vortices + 1)[:-1]
        # drift = (t / rotation_period) * self.xsize
        drift = ( (-t % rotation_period) / rotation_period ) * self.xsize  
        # Negative drift + modulo
        
        return (base_pos + drift) % self.xsize
# ==============================================================================
# Visualization of atmospheric data using PyVista
# =============================================================================
class AtmosphereVisualizer:
    def __init__(self, mesh, inclination=0):
        self.mesh = mesh
        self.inclination = inclination
        self.plotter = self._create_plotter()
        self.plotter.camera.parallel_scale = 0.99  # For photometry

    def _create_plotter(self):
        """Configure PyVista plotter"""
        """Configure PyVista plotter with proper camera setup"""
        plotter = pv.Plotter(
            off_screen=True,
            window_size=(1024, 1024),  # Explicit size helps consistency
            lighting="three lights"  # Better depth perception
        )
        plotter.background_color = 'black'
        return plotter
    
    def render_frame(self, atmospheric_data):
        """Render single timestep with full sphere visible"""
        self.plotter.clear()
        grid = pv.StructuredGrid(self.mesh.x, self.mesh.y, self.mesh.z)
        grid.point_data['scalars'] = atmospheric_data.ravel(order='F')
        
        self.plotter.add_mesh(grid, cmap='inferno', show_scalar_bar=False)
        # Set camera to show full sphere
        self.plotter.camera_position = [
            (0, 0, 4),  # Camera position (2.5 radii away from origin)
            (0, 0, 0),    # Focal point (center of sphere)
            (0, 1, 0)     # View-up vector (keep north at top)
        ]
        # Apply inclination adjustment
        self.plotter.camera.elevation = self.inclination
        # Ensure entire mesh fits in view
        self.plotter.reset_camera_clipping_range()
        return self.plotter.screenshot()
# ==============================================================================
# Run the simulation and visualization
#===============================================================================
class SimulationRunner:
    def __init__(self, config, inclinations):
        self.mesh = SphericalMesh()
        self.config = config  # Directly use the provided AtmosphericConfig instance
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
# Input output handler and data management
# ============================================================================
class DataManager:
    def __init__(self, base_path='output'):
        # Get current script directory and create full output path
        current_dir = os.path.dirname(os.path.abspath(__file__))
        self.base_path = os.path.join(current_dir, base_path)
        
        # Create directory if it doesn't exist
        os.makedirs(self.base_path, exist_ok=True)
        
    def save_simulation(self, results, prefix):
        output_path = os.path.join(self.base_path, f'{prefix}.h5')
        with h5py.File(output_path, 'w') as f:
            for inclin, data in results.items():
                f.create_dataset(f'{inclin}/gray', data=np.array(data['gray_array']))
                f.create_dataset(f'{inclin}/meta', data=str(data['metadata']))
                
# ==============================================================================
# Set up configurations and test call
# =============================================================================

# Example configuration
Ppol, Pband = 60, 5  # Periods in hours
Fpolar, Fband, Fambient = 0.7, 0.6, 0.5
bandConfig = [
    # [lat2, lat1, amplitude, type, phase, period]
    [90, 65, Fpolar, 'P', 0, Ppol],
    [45, 38., Fband, 'B', 10, Pband/2],
    [25, 15, Fband, 'B', 150, Pband], 
    [-10, -20, Fband, 'B', -26, Pband],
    [-33, -40, Fband, 'B', 135, Pband/2],
    [-65, -90, Fpolar, 'P', 0, Ppol]
]

# When initializing the config:
atmo_config = AtmosphericConfig(
    band_config=bandConfig,  # This is your band configuration list
    modu_config='polarStatic',
    modelname='production1',
    time_config=TimeConfig(t0=0, t1=60, frames=60),
    Fambient=Fambient,  # This will be accessible as config.Fambient
    Fband=Fband,
    Fpolar=Fpolar,
    speckey={'A': 0.25, 'B': 0.58, 'P': 0.75}
)

# Initialize components
mesh = SphericalMesh(resolution=200)
model = AtmosphericModel(mesh, atmo_config)

runner = SimulationRunner(
    config=atmo_config,
    inclinations=[-90, -60, -30, 0, 30, 60, 60]
)

runner.run_simulation(t0=0, t1=60, frames=2)

DataManager().save_simulation(runner.results, 'simulation_run1')