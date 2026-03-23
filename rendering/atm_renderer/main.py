"""
Created on April 1 2025

Upgrade from AtmosphereGenerator.py:
- significant improvement in speed via vectorization of routines
- gpu acceleration with pyvista but mainly cpu computation
- added multi-pressure capability

@author: nguyendat
"""
#%%
# =============================================================================
# IMPORT LIBRARIES

# import sys, os
# # Prepend the correct site-packages folder
# sys.path.insert(0, os.path.join(os.environ['CONDA_PREFIX'], 'Lib', 'site-packages'))

import vtk
# print(vtk.vtkVersion.GetVTKVersion())

import h5py
import pickle
import os
import json

import matplotlib.pyplot as plt
from matplotlib.patches import Circle
# =============================================================================
# Enable vtk GPU-backend 
import importlib.util

vtk_path = importlib.util.find_spec("vtk").origin
netcdf_path = importlib.util.find_spec("vtkmodules.vtkIONetCDF").origin

# print("vtk.py path:", vtk_path)
# print("vtkIONetCDF path:", netcdf_path)
# print("\nDirectory contents of vtk folder:")
# print(os.listdir(os.path.dirname(vtk_path)))

# Set debug flags for VTK/OpenGL
os.environ["VTK_DEBUG_OPENGL"] = "1"
os.environ["VTK_REPORT_OPENGL_ERRORS"] = "1"
# =============================================================================
import pyvista as pv
import cv2
from tqdm import tqdm
import numpy as np
from numba import jit
from matplotlib import cm, colors

import warnings
from scipy.ndimage import gaussian_filter
import pandas as pd
from sklearn.decomposition import PCA
from datetime import datetime
import time
import imageio
from PIL import Image
from sklearn.cluster import KMeans
import gc
# =============================================================================
### Path management
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
    """
    Creates object representing spherical mesh array.

    Attributes:
        radius (float): radius of the object
        resolution (int): resolution of array
        phi (array): values along phi in spherical coordinates
        theta (array): values along theta in spherical coordinates
        x (array): values along x in Cartesian coordinates
        y (array): values along y in Cartesian coordinates
        z (array): values along z in Cartesian coordinates
    
    Methods:
        generate_mesh(self): creates spherical grid coordinates
    """
    def __init__(self, resolution=400, radius=1):
        self.radius = radius
        self.resolution = resolution
        self.phi, self.theta = None, None
        self.x, self.y, self.z = None, None, None
        self.generate_mesh()
        
    def generate_mesh(self):
        """
        Create spherical grid coordinate array transformed to Cartesian coordinates.
        
        Parameters: None

        Returns: None
        """
        phi = np.linspace(0, np.pi, self.resolution)
        # phi = np.arccos(np.linspace(1, -1, self.resolution))  # Cosine spacing
        theta = np.linspace(0, 2*np.pi, self.resolution)
        self.phi, self.theta = np.meshgrid(phi, theta)
        
        self.x = self.radius * np.sin(self.phi) * np.cos(self.theta)
        self.y = self.radius * np.sin(self.phi) * np.sin(self.theta)
        self.z = self.radius * np.cos(self.phi)
    
    @property
    def shape(self):
        """
        Returns the shape of the coordinate array.
        """
        return self.x.shape
# ==============================================================================
# Manage config and parameters
# =============================================================================
class TimeConfig:
    """
    Manage configuration of temporal parameters.

    Attributes:
        t0 (float): start time (hours)
        t1 (float): end time (hours)
        frames (int): number of animation frames
        time_array (array): array of equidistant time increments from t0 to t1
        dt (float): time step size
    
    Methods:
        repr(): prints out configured time parameters

    # can add "incomplete" time array to account for gaps in observation
    # ex. compare JWST sparse vs full time array setup:
        import matplotlib.pyplot as plt
        
        # 120 frames over 60 hours
        t_full = TimeConfig(t0=0, t1=60, frames=120).time_array 

        # 360 frames, 180 hours of observed time over 3 segments
        # gaps of 60 hours in between, so total time is 180 + 60*2 = 300 hours
        t_jwst = TimeConfig(t0=0, t1=180, frames=360, option='jwst').time_array
        # t_jwst = TimeConfig(t0=0, t1=90, frames=120, option='jwst').time_array # or shorter version

        plt.figure(figsize=(2,10))
        plt.scatter(0.5*np.ones(len(t_full)), t_full, s=0.4)
        plt.scatter(1.5*np.ones(len(t_jwst)), t_jwst, s=0.4)
        plt.xlim(0,2)

    """
    def __init__(self, t0=0, t1=60, frames=60, option='full', jwst_setup={'gap':60, 'segments':3}):
        """
        Args:
            t0: Start time (hours)
            t1: End time (hours)
            frames: Number of animation frames
            option: 'full' for full time array, 'sparse' for sparse sampling, 'jwst' for JWST-like gaps
            jwst_setup: dict with 'gap' (hours) and 'segments' (int) for JWST mode
        """
        self.t0 = t0
        self.t1 = t1
        self.frames = frames
        self.option = option
        
        # Derived properties
        if option == 'full':
            self.time_array = np.linspace(t0, t1, frames)
            self.dt = (t1 - t0) / frames  # Time step

        if option == 'sparse':
            s = 5
            # Sparse sample for polar dynamics
            self.time_array = np.arange(t0, t1*s, s)
            self.dt = s # Time step
            print(f"Sparse mode: every 5 hours over {t1*s} hours; frames ignored")
            self.frames = len(self.time_array)

        if option == 'jwst':
            # devide t0-t1 into 3 segments with gaps of 60 hours in between
            # preserving total no of frames
            gap = jwst_setup['gap'] # hours
            segment_frames = frames // jwst_setup['segments']  # frames per segment
            segments = []
            for i in range(jwst_setup['segments']):
                start_time = t0 + i * ((t1 - t0) / jwst_setup['segments']) + i * gap
                end_time = start_time + ((t1 - t0) / jwst_setup['segments'])
                if i == jwst_setup['segments'] - 1:  # Last segment takes remaining frames
                    segment_frames = frames - len(segments) * segment_frames
                segments.append(np.linspace(start_time, end_time, segment_frames, endpoint=False))
            self.time_array = np.concatenate(segments)
            self.dt = (t1 - t0 + 2 * gap) / frames  # Average time step

            print(f"JWST mode: 3 segments ({t1-t0} hours)  with gaps of {gap} hours; total time {t1 - t0 + 2 * gap} hours")

    def __repr__(self):
        return f"TimeConfig(t0={self.t0}, t1={self.t1}, frames={self.frames}, format={self.option})"

    def _to_dict(self):
        """Convert to JSON-serializable dictionary"""
        return {
            't0': self.t0,
            't1': self.t1,
            'frames': self.frames,
            'dt': self.dt,
            'option': self.option
        }
    
    @classmethod
    def from_dict(cls, d):
        """Reconstruct from dictionary"""
        return cls(t0=d['t0'], t1=d['t1'], frames=d['frames'])
# =============================================================================
class AtmosphericConfig:
    """
    Manages configuration of atmospheric and temporal parameters.

    Attributes:
        band_config (list): band configuration parameters
        modu_config (str): module configuration
        modelname (str): name of selected model
        time_config (class instance): TimeConfig object with chosen temporal parameters
        Fambient (float): base ambient contrast value
        Fambient_var (float): ambient variability value
        Fband (float): base band contrast value
        Fband_var (float): band variability value
        Fpolar (float): base polar contrast value
        Fpolar_var (float): polar variability value
        Pband (float): band variation period
        Ppol (float): pole variation period
        Prot (float): planetary rotation period
        speckey (dict): spectral value mapping
    
    Methods:
        _validate_config(): validates configuration parameters
    """
    def __init__(self, 
                 band_config: list,
                 modu_config: str,
                 modelname: str,
                 time_config: TimeConfig,
                 Fband: float = 1,
                 Fambient: float  = 1,
                 Fpolar: float  = 1,
                 Pband: float  = 5.0,
                 Ppol: float  = 60.0,
                 Prot: float = 5.0,
                 Fambient_var: float = 0.0,
                 Fpolar_var: float = 0.05,
                 Fband_var: float = 0.05,
                 speckey: dict = None,
                 colorlim: list = [0.5, 1.5]):
        """
        Args:
            band_config: Atmospheric band parameters
            modu_config: Modulation type ('polarStatic' etc)
            modelname: Simulation identifier
            time_config: TimeConfig object
            Fambient/band/pole: Ambient/band/pole base contrast value (amp)
            Fambient_var/band_var/polar_var: Variability value (variab) 
            Pband/pole: Band/pole period (in hours)
            speckey: Spectral value mapping
        """
        self.band_config = band_config
        self.modu_config = modu_config
        self.modelname = modelname
        self.time_config = time_config
        # Base contrast values
        self.Fambient = Fambient
        self.Fband = Fband
        self.Fpolar = Fpolar
        # Variability values
        self.Fband_var = Fband_var
        self.Fpolar_var = Fpolar_var
        self.Fambient_var = Fambient_var
        # Periods in hours
        self.Pband = Pband
        self.Ppol = Ppol
        self.Prot = Prot
        self.speckey = speckey or {'BG':0, 'A': 150, 'B': 200, 'P': 250}
        self.colorlim = colorlim
        
        self._validate_config()

    def _validate_config(self):
        """
        Sanity checks for configuration
        """
        if not isinstance(self.time_config, TimeConfig):
            raise TypeError("time_config must be TimeConfig instance")
        
        required_band_keys = ['lat2', 'lat1', 'amp', 'typ', 'phase', 'period', 'planet_period', 'variab']
        for band in self.band_config:
            if len(band) != len(required_band_keys):
                raise ValueError("Invalid band configuration")

    def _to_dict(self):
        """Convert to JSON-serializable dictionary"""
        return {
            'band_config': self.band_config,
            'modu_config': self.modu_config,
            'modelname': self.modelname,
            'time_config': self.time_config._to_dict(),  # Nested serialization
            'Fambient': self.Fambient,
            'Fband': self.Fband,
            'Fpolar': self.Fpolar,
            'Pband': self.Pband,
            'Ppol': self.Ppol,
            'Fambient_var': self.Fambient_var,
            'Fband_var': self.Fband_var,
            'Fpolar_var': self.Fpolar_var,
            'speckey': self.speckey,
            'colorlim': self.colorlim
        }
    
    @classmethod
    def from_dict(cls, d):
        """Reconstruct from dictionary"""
        time_config = TimeConfig.from_dict(d['time_config'])
        return cls(
            band_config=d['band_config'],
            modu_config=d['modu_config'],
            modelname=d['modelname'],
            time_config=time_config,
            Fambient=d['Fambient'],
            Fband=d['Fband'],
            Fpolar=d['Fpolar'],
            Pband=d['Pband'],
            Ppol=d['Ppol'],
            Fambient_var=d['Fambient_var'],
            Fband_var=d['Fband_var'],
            Fpolar_var=d['Fpolar_var'],
            speckey=d['speckey']
        )
    
    def __repr__(self):
        return f"AtmosphericConfig(model={self.modelname}, time={self.time_config})"

# ==============================================================================
# Core atmospheric simulation logic
# =============================================================================
class AtmosphericModel:
    """
    Creates atmospheric model object with all features and parameters applied.

    Attributes:
        mesh (class instance): SphericalMesh object (provides x, y, z coordinates)
        config (class instance): AtmosphericConfig object (simulation parameters)
        speckey (dict): maps region types to spectral values
        xsize, ysize (int): x, y shape of mesh
        xx, yy (array): 
        lat_grid (array): vectorized latitude grid
    
    Methods:
        generate_specmap(): generate spectral map based on speckey config
        generate_atmosphere(): generate atmospheric map at time = t
        _lat_px(lat_deg): convert latitude (vectorized) to pixel coordinates
        _apply_planetary_wave(im, mask, t, amp, phase, period, variab): apply planetary wave feature to given region (bands)
        _apply_polar_effect(im, mask, t, amp, phase, period, variab): apply polar cap modulation effects
        _circle_vortice_vectorized(im, lat1, lat2, t, group, modu_config): generates equally spaced vortices in polar cap with configured features
        _long_px(long_deg): convert longitude (vectorized) to pixel coordinates
        _equidistant_longitudes(t, rotation_period): find equidistant longitudes of vortex centers
    """
    def __init__(self, mesh, band_config, vortex_config):
        """
        Args:
            mesh: SphericalMesh object (provides x, y, z coordinates)
            config: AtmosphericConfig object (simulation parameters)
            speckey: Dict mapping region types to spectral values
        """
        self.mesh = mesh
        self.config = band_config
        self.speckey = band_config.speckey
        self.vortex_config = vortex_config
        
        # Derived properties from mesh
        self.xsize, self.ysize = self.mesh.shape
        self.xx, self.yy = np.meshgrid(np.arange(self.xsize), np.arange(self.ysize), indexing='ij')
        
        # Precompute latitude grid (vectorized)
        self.lat_grid = np.abs(self.yy - 90) / 180 * self.ysize  # From lat() function

        if vortex_config is not None:
            self.n_vortice = vortex_config[0]
            self.radius_frac = vortex_config[1]
            self.drift = vortex_config[2]
            self.center_lat = vortex_config[3] if vortex_config[3] is not None else 82.5
            self.vortex_amp = vortex_config[4] if vortex_config[4] is not None else 0.2

            # Create random motion grids
            rng = np.random.default_rng(11)
            self.drift_angles = rng.uniform(0, 2 * np.pi, size=(self.n_vortice, 120))
            self.drift_positions = None
            
            print(f"Drift angles shape: {self.drift_angles.shape}")
            print(f"Sample angles: {self.drift_angles[0, :5]}")  # First vortex, first 5 steps

    def add_drift(self, step_size, max_drift_radius, center_lat):
        drift_positions = np.zeros((self.n_vortice, 120, 2))

        ang_step_size = (step_size / self.ysize) * 180
        max_ang_drift = (max_drift_radius / self.ysize) * 180

        center_lat_rad = np.radians(center_lat)

        step_lon = ang_step_size * np.cos(self.drift_angles) / np.cos(center_lat_rad)
        step_lat = ang_step_size * np.sin(self.drift_angles)

        cumulative_lon = np.zeros((self.n_vortice, 120))
        cumulative_lat = np.zeros((self.n_vortice, 120))

        for t_idx in range(120):
            if t_idx == 0:
                proposed_lon = step_lon[:, 0]
                proposed_lat = step_lat[:, 0]
            else:
                proposed_lon = cumulative_lon[:, t_idx - 1] + step_lon[:, t_idx]
                proposed_lat = cumulative_lat[:, t_idx - 1] + step_lat[:, t_idx]
            
            ang_dist = np.sqrt(proposed_lat**2 + (proposed_lon * np.cos(center_lat_rad))**2)

            within_bounds = ang_dist <= max_ang_drift
            
            if t_idx == 0:
                cumulative_lon[:, t_idx] = np.where(within_bounds, proposed_lon, 0.0)
                cumulative_lat[:, t_idx] = np.where(within_bounds, proposed_lat, 0.0)
            else:
                cumulative_lon[:, t_idx] = np.where(
                    within_bounds, 
                    proposed_lon, 
                    cumulative_lon[:, t_idx - 1]
                )
                cumulative_lat[:, t_idx] = np.where(
                    within_bounds, 
                    proposed_lat, 
                    cumulative_lat[:, t_idx - 1]
                )
        
        # Package results: shape (n_vortice, 120, 2)
        drift_positions = np.zeros((self.n_vortice, 120, 2))
        drift_positions[:, :, 0] = cumulative_lon  # longitude degrees
        drift_positions[:, :, 1] = cumulative_lat  # latitude degrees
        
        return drift_positions

    def generate_specmap(self):
        """
        Generate a spectral mask based on the speckey configuration.
        
        Returns:
            sm: Spectral mask array with shape (xsize, ysize)
        """
        sm = np.full((self.xsize, self.ysize), self.speckey['A'], dtype=np.float32)

        for group in self.config.band_config:
            lat2, lat1, amp, typ, phase, period, planet_period, variab = group
            lat_px1 = self._lat_px(lat1)
            lat_px2 = self._lat_px(lat2)
            
            # Vectorized latitude mask
            mask = (self.yy >= lat_px2) & (self.yy <= lat_px1)
            sm[mask] = self.speckey[typ.upper()]
        
        return sm

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
            lat2, lat1, amp, typ, phase, period, planet_period, variab = group
            lat_px1 = self._lat_px(lat1)
            lat_px2 = self._lat_px(lat2)
            wavenumber = self.config.Pband / period
            
            # Vectorized latitude mask
            mask = (self.yy >= lat_px2) & (self.yy <= lat_px1)
            im[mask] = amp
            
            if typ.upper() == 'B':  # Band
                # im = self._apply_discrete_planetary_wave(im, mask, t, amp, phase, period, variab)
                im = self._apply_planetary_wave(im, mask, t, amp, phase, period, planet_period, variab, wavenumber)

            elif typ.upper() == 'P':  # Polar
                im = self._apply_polar_effect(im, mask, t, amp, phase, period, planet_period, variab)
                
            if spec:
                sm[mask] = self.speckey[typ.upper()]
                
        # Apply vortices if needed
        if self.config.modu_config in ['polarStatic', 'polarDynamic'] and self.vortex_config is not None:
            im = self._apply_vortices(im, t, self.config.modu_config)
            
        return (im, sm) if spec else im
    
    def _lat_px(self, lat_deg):
        """
        Convert latitude to pixel coordinate (vectorized)

        Parameters:
            lat_deg (float): latitude in degrees
        
        Returns:
            corresponding pixel coordinates
        """
        return np.abs(lat_deg - 90) / 180 * self.ysize
    
    def _apply_planetary_wave(self, im, mask, t, amp, phase, period, planet_period, variab, wavenumber):
        """
        Vectorized planetary wave implementation.

        Parameters:
            im (array): flux map
            mask (array): mask selecting region on map for planetary waves
            t (float): time
            amp (float): planetary wave amplitude
            phase (float): phase shift of planetary wave
            period (float): planetary wave period
            variab (float): variability
        
        Returns:
            im (array): updated flux map with planetary wave features
        """
        # Spatial frequency (1/wavelength)
        w = self.xsize / wavenumber  # Full circumference resolution
        sine_wave = variab * np.sin(
            2 * np.pi / w * (self.xx + (t / period) * w) + phase * np.pi / 180
        )
        im[mask] += sine_wave[mask]
        return im
    
    def _apply_discrete_planetary_wave(self, im, mask, t, amp, phase, period, planet_period, variab):
        w = self.xsize  # longitudinal resolution
        # Generate continuous sine
        sine_wave = np.sin(
            2 * np.pi / w * (self.xx + (t / period) * w) + phase * np.pi / 180)
        # Convert to discrete (±1)
        discrete_wave = np.where(sine_wave >= 0, 1.0, -1.0)
        # Scale by variab (half amplitude span)
        flux = variab * discrete_wave
        # Apply only inside band mask
        im[mask] += flux[mask]
        return im
    
    def _apply_polar_effect(self, im, mask, t, amp, phase, period, planet_period, variab):
        """
        Polar cap modulation (vectorized)

        Parameters:
            im (array): flux map
            mask (array): mask selecting region on map for planetary waves
            t (float): time
            amp (float): planetary wave amplitude
            phase (float): phase shift of planetary wave
            period (float): planetary wave period
            variab (float): variability
        
        Returns:
            im (array): updated flux map with polar cap modulation features
        """
        flux = variab * np.sin(2 * np.pi / period * t + phase * np.pi / 180)
        im[mask] += flux
        return im
    
    def _apply_vortices(self, im, t, modu_config):
        """Vectorized vortices implementation"""
        # Get polar regions from config
        polar_groups = [g for g in self.config.band_config if g[3].upper() == 'P']
        
        for group in polar_groups:
            lat2, lat1 = group[0], group[1]
            im = self._circle_vortice_vectorized(im, lat1, lat2, t, group, modu_config,
                                                self.center_lat, self.vortex_amp)  # Pass full group
            
        return im
    
    def _latlon_to_stereographic_vectorized(self, lat, lon, lat0, lon0):
        """
        Vectorized conversion of lat/lon to stereographic projection coordinates.
        Handles broadcasting for multiple vortex centers.
        
        Parameters:
            lat (array): latitude in degrees (can be multi-dimensional)
            lon (array): longitude in degrees (can be multi-dimensional)
            lat0 (array): center latitude for projection (can be multi-dimensional)
            lon0 (array): center longitude for projection (can be multi-dimensional)
        
        Returns:
            X, Y (arrays): stereographic coordinates (same shape as input)
        """
        # Convert to radians
        lat_rad = np.radians(lat)
        lon_rad = np.radians(lon)
        lat0_rad = np.radians(lat0)
        lon0_rad = np.radians(lon0)
        
        # Handle longitude wrapping properly (normalize to [-pi, pi])
        dlon = lon_rad - lon0_rad
        # Ensure dlon is in range [-pi, pi] for proper wrapping
        dlon = np.mod(dlon + np.pi, 2 * np.pi) - np.pi
        
        # Stereographic projection formulas (fully vectorized)
        k = 2 / (1 + np.sin(lat0_rad) * np.sin(lat_rad) + 
                np.cos(lat0_rad) * np.cos(lat_rad) * np.cos(dlon))
        
        X = k * np.cos(lat_rad) * np.sin(dlon)
        Y = k * (np.cos(lat0_rad) * np.sin(lat_rad) - 
                np.sin(lat0_rad) * np.cos(lat_rad) * np.cos(dlon))
        
        return X, Y
    
    def _circle_vortice_vectorized(self, im, lat1, lat2, t, group, modu_config, #band_config
                                   center_lat=82.5, vortex_amp=0.4 #polar_config
                                   ):
        """
        Fully vectorized vortex generator using stereographic projection.
        
        Parameters:
            im (array): flux map
            lat1 (float): first latitude boundary in degrees
            lat2 (float): second latitude boundary in degrees
            t (float): time
            group (tuple): configuration details
            modu_config (str): module type ('noPolar', 'polarStatic', 'polarDynamic')
        
        Returns:
            im (array): flux map with vortex features added
        """
        # Sort latitudes (lat1 > lat2)
        lat1, lat2 = sorted([lat1, lat2], reverse=True)

        amp = group[2]
        phase = group[4]
        period = group[5]
        planet_period = group[6]
        variab = group[7]
        
        # Vortex properties
        
        area_cap = 2 * np.pi * abs(np.sin(np.radians(lat1)) - np.sin(np.radians(lat2)))
        r_vortice = np.sqrt(self.radius_frac * area_cap) * (self.xsize / np.pi)
        
        # Time-dependent longitudinal positions
        long_positions = self._equidistant_longitudes_degrees(t, planet_period)
        step_size = r_vortice / 2
        max_drift_radius = r_vortice / 2
        
        # Use actual number of available positions (in case fewer than n_vortice)
        n_vortice = min(self.n_vortice, len(long_positions))
        
        # Vortex centers (shape: n_vortice x 2 for [lon, lat])
        vortex_lons = long_positions[:self.n_vortice]  # shape: (n_vortice,)
        vortex_lats = np.full(self.n_vortice, center_lat)  # shape: (n_vortice,)

        lon_grid = (self.xx / self.xsize) * 360
        lat_grid = 90 - (self.yy / self.ysize) * 180

        if self.drift == True:
            if self.drift_positions is None:
                self.drift_positions = self.add_drift(step_size, max_drift_radius, center_lat)
            
            t_idx = int(t) % 120

            drift_lon_deg = self.drift_positions[:self.n_vortice, t_idx, 0]
            drift_lat_deg = self.drift_positions[:self.n_vortice, t_idx, 1]

            vortex_lons = vortex_lons + drift_lon_deg
            vortex_lats = vortex_lats + drift_lat_deg

            vortex_lons = vortex_lons % 360
            vortex_lats = np.clip(vortex_lats, -90, 90)

        # Latitude mask for the band
        lat_mask = (lat_grid >= lat2) & (lat_grid <= lat1)
        polar_indices = np.where(lat_mask)

        lat_polar = lat_grid[polar_indices]
        lon_polar = lon_grid[polar_indices]
 
        # Amplitude from polar region
        polar_background = im[0,0]
        
        # Handle variable flux based on mode (ensure it's a NumPy array)
        if modu_config == 'polarDynamic':
            
            ### for test and jwst
            # phase_values = np.array([0.1, -0.2, 0.4, -0.1, 0.3, 0.2, 0.3, -0.1, -0.4, -0.3, 0.1,])

            ### for test_onlyVortex
            # phase_values = 3*np.array([0.1, -0.2, 0.4, -0.1, 0.3, 0.2, 0.3, -0.1, -0.4, -0.3, 0.1,])
            phase_values = 3*np.array([1.3, 1.0, 0.7, 0.1, 0.9, 0.4, 1.2, 0.4, 1.3, 0.9, 0.5,])

            variableflux = np.array(vortex_amp * np.sin(2 * np.pi / period * t + phase 
                                                        + phase_values[:self.n_vortice]))
        elif modu_config == 'polarStatic':
            variableflux = np.array(vortex_amp * np.sin(2 * np.pi / period * t + phase))
        else:
            variableflux = np.zeros(self.n_vortice)
        
        # Ensure variableflux is a proper NumPy array
        variableflux = np.atleast_1d(variableflux)
        
        # Convert angular radius to stereographic distance
        angular_radius_deg = (r_vortice / self.ysize) * 180
        r_stereo = 2 * np.tan(np.radians(angular_radius_deg) / 2)
        
        # Vectorized stereographic projection for all vortices at once
        # Expand dimensions: grid is (H, W), vortex centers are (n_vortice,)
        # Result will be (n_vortice, H, W)
        lat_expanded = lat_polar[np.newaxis, :]  # (1, H, W)
        lon_expanded = lon_polar[np.newaxis, :]  # (1, H, W)
        vortex_lats_expanded = vortex_lats[:, np.newaxis]  # (n_vortice, 1, 1)
        vortex_lons_expanded = vortex_lons[:, np.newaxis]  # (n_vortice, 1, 1)
        
        # Compute stereographic projection for all vortices simultaneously
        X_grids, Y_grids = self._latlon_to_stereographic_vectorized(
            lat_expanded, lon_expanded, 
            vortex_lats_expanded, vortex_lons_expanded)
        
        # Distance from each vortex center (shape: n_vortice, H, W)
        distance_stereo = np.sqrt(X_grids**2 + Y_grids**2)
        
        # Create circular masks for all vortices (shape: n_vortice, H, W)
        circle_masks = distance_stereo <= r_stereo
        
        # Apply latitude mask to all vortices
        #circle_masks = circle_masks & lat_mask[np.newaxis, :, :]
        
        # Compute flux contributions for each vortex (shape: n_vortice, H, W)
        # flux_contributions = np.abs((polar_background + variableflux[:, np.newaxis])) * circle_masks
        flux_contributions = (variableflux[:, np.newaxis]) * circle_masks

        # Sum all vortex contributions and add to image
        im[polar_indices] += np.sum(flux_contributions, axis=0)
        return im

    def _equidistant_longitudes_degrees(self, t, rotation_period):
        """
        Calculate equidistant longitudes for vortice centers in degrees.

        Parameters:
            n_vortices (int): number of vortices
            t (float): time
            rotation_period (float): rotation period of object
        
        Returns:
            long_positions (array): array of longitude positions in degrees [0, 360)
        """
        # Generate evenly spaced base positions in degrees
        base_pos_deg = np.linspace(0, 360, self.n_vortice + 1)[:-1]
        
        # Calculate drift in degrees
        drift_deg = ((-t % rotation_period) / rotation_period) * 360
        
        # Apply drift and wrap to [0, 360)
        long_positions_deg = (base_pos_deg + drift_deg) % 360
        
        return long_positions_deg

    def track_and_plot_vortex_paths(self, t_range, save_path=None, verbose=False):
        """
        Track vortex positions over time and plot their paths on a polar cap map.
        
        Args:
            t_range: Array of time steps to track (e.g., np.arange(0, 120, 1))
            save_path: Optional path to save the figure
        """
        # Get polar cap configuration
        polar_groups = [g for g in self.config.band_config if g[3].upper() == 'P']
        if not polar_groups:
            print("No polar regions found")
            return
        
        group = polar_groups[0]
        lat2, lat1, amp, typ, phase, period, planet_period, variab = group
        center_lat = 82.5

        if lat1 < lat2:
            lat1, lat2 = lat2, lat1
        
        if verbose:
            print(f"Polar cap: lat1={lat1}, lat2={lat2}, center={center_lat}")
            print(f"Period: {period}, Time range: {t_range[0]} to {t_range[-1]}")
        
        # Make sure drift positions exist
        if self.drift_positions is None:
            area_cap = 2 * np.pi * abs(np.sin(np.radians(lat1)) - np.sin(np.radians(lat2)))
            r_vortice = np.sqrt(0.02 * area_cap) * (self.xsize / np.pi)
            step_size = r_vortice / 2
            max_drift_radius = r_vortice / 2
            self.drift_positions = self.add_drift(step_size, max_drift_radius, center_lat)
            print(f"Generated drift positions: {self.drift_positions.shape}")
        
        if verbose: 
            print(f"Drift positions shape: {self.drift_positions.shape}")
        
        # Extract paths for the requested time range
        vortex_paths = {i: {'lons': [], 'lats': []} for i in range(self.n_vortice)}
        
        for t in t_range:
            # Get base longitude positions (rotation only)
            long_positions = self._equidistant_longitudes_degrees(t, planet_period)
            
            # Get drift offsets for this timestep
            t_idx = int(t) % 120
            dx_px = self.drift_positions[:self.n_vortice, t_idx, 0]
            dy_px = self.drift_positions[:self.n_vortice, t_idx, 1]
            
            # Convert drift from pixels to degrees
            drift_lon_deg = (dx_px / self.xsize) * 360
            drift_lat_deg = (dy_px / self.ysize) * 180
            
            # Calculate final positions
            vortex_lons = (long_positions + drift_lon_deg) % 360
            vortex_lats = np.clip(center_lat + drift_lat_deg, -90, 90)
            
            # Store positions
            for i in range(self.n_vortice):
                vortex_paths[i]['lons'].append(vortex_lons[i])
                vortex_paths[i]['lats'].append(vortex_lats[i])
        
        # Print debug info for first vortex
        if verbose:
            print(f"\nVortex 0 path info:")
            print(f"  Number of timesteps: {len(vortex_paths[0]['lons'])}")
            print(f"  Lon range: {min(vortex_paths[0]['lons']):.2f} to {max(vortex_paths[0]['lons']):.2f}")
            print(f"  Lat range: {min(vortex_paths[0]['lats']):.2f} to {max(vortex_paths[0]['lats']):.2f}")
            print(f"  First 5 positions:")
            for j in range(min(5, len(vortex_paths[0]['lons']))):
                print(f"    t={t_range[j]}: lon={vortex_paths[0]['lons'][j]:.2f}, lat={vortex_paths[0]['lats'][j]:.2f}")
        
        # Create visualization
        fig, axes = plt.subplots(1, 2, figsize=(16, 7))
        
        # Plot 1: Rectangular projection
        ax1 = axes[0]
        
        # Draw polar cap boundary
        ax1.axhline(y=lat1, color='gray', ls='--', alpha=0.5, linewidth=2)
        ax1.axhline(y=lat2, color='gray', ls='--', alpha=0.5, linewidth=2)
        ax1.fill_between([0, 360], lat2, lat1, alpha=0.15, color='lightblue')
        
        colors = plt.cm.tab10(np.linspace(0, 1, self.n_vortice))
        
        for i in range(self.n_vortice):
            lons = np.array(vortex_paths[i]['lons'])
            lats = np.array(vortex_paths[i]['lats'])
            
            if verbose:
                print(f"\nPlotting vortex {i}:")
                print(f"  Arrays have {len(lons)} points")
                print(f"  Lon: min={lons.min():.2f}, max={lons.max():.2f}, mean={lons.mean():.2f}")
                print(f"  Lat: min={lats.min():.2f}, max={lats.max():.2f}, mean={lats.mean():.2f}")
            
            # Plot path with thicker line
            line = ax1.plot(lons, lats, '-', color=colors[i], alpha=0.8, linewidth=3, 
                    label=f'Vortex {i+1}', zorder=5)
            if verbose: print(f"  Plotted line: {line}")
            
            # Start marker
            ax1.plot(lons[0], lats[0], 'o', color=colors[i], markersize=12, 
                    markeredgecolor='black', markeredgewidth=2, zorder=10)
            
            # End marker
            ax1.plot(lons[-1], lats[-1], 's', color=colors[i], markersize=12,
                    markeredgecolor='black', markeredgewidth=2, zorder=10)
        
        ax1.set_xlim(0, 360)
        ax1.set_ylim(lat2 - 2, lat1 + 2)
        ax1.set_xlabel('Longitude (degrees)', fontsize=12)
        ax1.set_ylabel('Latitude (degrees)', fontsize=12)
        ax1.set_title(f'Vortex Paths - Rectangular (t={t_range[0]} to {t_range[-1]})', fontsize=14)
        ax1.legend(loc='best', fontsize=9)
        ax1.grid(True, alpha=0.3)
        
        if verbose:
            print(f"\nPlot 1 limits: x={ax1.get_xlim()}, y={ax1.get_ylim()}")
        
        # Plot 2: Polar view
        ax2 = axes[1]
        ax2.set_aspect('equal')
        
        # Polar cap boundaries
        r_inner = 90 - center_lat
        r_outer = 90 - lat2
        
        if verbose:
            print(f"\nPolar view:")
            print(f"  r_inner (lat {lat1}°) = {r_inner:.2f}")
            print(f"  r_outer (lat {lat2}°) = {r_outer:.2f}")
        
        for i in range(self.n_vortice):
            lons = np.array(vortex_paths[i]['lons'])
            lats = np.array(vortex_paths[i]['lats'])
            
            # Convert to polar coordinates
            r = 90 - lats
            theta = np.radians(lons)
            
            # Convert to Cartesian
            x = r * np.sin(theta)
            y = r * np.cos(theta)
            
            if verbose:
                print(f"\nVortex {i} in polar coords:")
                print(f"  r: min={r.min():.2f}, max={r.max():.2f}")
                print(f"  x: min={x.min():.2f}, max={x.max():.2f}")
                print(f"  y: min={y.min():.2f}, max={y.max():.2f}")
            
            # Plot path
            ax2.plot(x, y, '-', color=colors[i], alpha=0.8, 
                    ls='', marker='o', markersize=10,
                    label=f'Vortex {i+1}', zorder=5)
            
            # Markers
            ax2.plot(x[0], y[0], 'o', color=colors[i], markersize=13,
                    markeredgecolor='black', markeredgewidth=2, zorder=10)
            ax2.plot(x[-1], y[-1], 'o', color=colors[i], markersize=13,
                    markeredgecolor='black', markeredgewidth=2, zorder=10)
        
        # Draw circles
        circle_inner = Circle((0, 0), r_inner, fill=False, edgecolor='gray',
                            ls='--', linewidth=2)
        circle_outer = Circle((0, 0), r_outer, fill=False, edgecolor='gray',
                            ls='--', linewidth=2)
        ax2.add_patch(circle_inner)
        ax2.add_patch(circle_outer)
        
        # Pole
        ax2.plot(0, 0, 'k*', markersize=20, zorder=15)
        
        # Longitude lines
        for lon in [0, 90, 180, 270]:
            theta_rad = np.radians(lon)
            x_line = r_outer * np.sin(theta_rad)
            y_line = r_outer * np.cos(theta_rad)
            ax2.plot([0, x_line], [0, y_line], 'k:', alpha=0.3, linewidth=1)
            
            label_r = r_outer + 1.5
            ax2.text(label_r * np.sin(theta_rad), label_r * np.cos(theta_rad), 
                    f'{lon}°', ha='center', va='center', fontsize=10)
        
        padding = 2
        ax2.set_xlim(-r_outer-padding, r_outer+padding)
        ax2.set_ylim(-r_outer-padding, r_outer+padding)
        ax2.set_xlabel('East ←→ West', fontsize=12)
        ax2.set_ylabel('South ←→ North', fontsize=12)
        ax2.set_title(f'Vortex Paths - Polar View (t={t_range[0]} to {t_range[-1]})', fontsize=14)
        ax2.legend(loc='upper right', fontsize=9)
        ax2.grid(True, alpha=0.3)
        
        if verbose:
            print(f"\nPlot 2 limits: x={ax2.get_xlim()}, y={ax2.get_ylim()}")
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"\nSaved to {save_path}")
        else:
            plt.show()
        
        return fig, vortex_paths

# ==============================================================================
# Visualization of atmospheric data using PyVista
# =============================================================================
class AtmosphereVisualizer:
    """
    Visualizes planet flux and spectral maps with given parameters/configurations.

    Attributes:
        mesh (class instance): SphericalMesh object (provides x, y, z coordinates)
        inclination (list): list of inclinations to be visualized
        speckey (dict): maps regions to spectral values
        imsize (list): length and width values in pixels defining map size
        plotter (class instance): specific plotter object
    """
    def __init__(self, mesh, speckey, config, imsize=(300, 300), inclination=0):
        self.mesh = mesh
        self.inclination = inclination
        self.speckey = speckey
        self.config = config
        self.imsize = imsize
        self.plotter = None
        self._limb_mask_cache = None  # Cache for limb darkening mask

    def configure_plotter(self, zoom_factor=1.01):
        """
        Configure PyVista plotter with proper camera setup.

        Parameters:
            zoom_factor (float): value of zoom factor
        
        Returns:
            self.plotter (class instance): configured plotter object
        """
        # Close existing plotter if it exists
        # Force cleanup of previous plotter
        if self.plotter is not None:
            try:
                self.plotter.close()
                del self.plotter
            except:
                pass
            gc.collect()  # Force garbage collection
            
        self.plotter = pv.Plotter(
            off_screen=True,
            window_size=self.imsize,  # Explicit size helps consistency
            lighting='none'  # Remove anisotropy
        )
        
        self.plotter.camera.SetParallelProjection(True)  # Set parallel projection for photometry
        self.plotter.camera.elevation = self.inclination + 56  # Adjust for default value
        self.plotter.background_color = 'black'  # Set background color to black

        # Fine-tune the field of view with parallel_scale
        self.plotter.camera.parallel_scale = zoom_factor  # Uncommented this line
        
        return self.plotter

    def _apply_limb_darkening(self, grayscale, u_coefficient=0.1):
        """
        Apply limb darkening to grayscale image with caching.
        
        Args:
            grayscale: 2D grayscale image array
            u_coefficient: Limb-darkening coefficient (0.5-0.9 typical range)
        
        Returns:
            grayscale_darkened: Image with limb darkening applied
        """
        # Compute limb darkening mask (cached based on shape and u_coefficient)
        cache_key = (grayscale.shape, u_coefficient)
        if self._limb_mask_cache is None or self._limb_mask_cache[0] != cache_key:
            # Detect actual sphere boundary from the rendered image
            xlen, ylen = grayscale.shape
            xcen, ycen = xlen // 2, ylen // 2
            
            # Find radius by detecting first non-black pixel from center outward
            boundary_pixel = np.where(grayscale[:, ycen] > 0.)[0]
            if len(boundary_pixel) > 0:
                radius = xcen - boundary_pixel[0]
            else:
                # Fallback: use half the image size
                radius = min(xcen, ycen)
            
            # Compute the limb darkening mask
            y, x = np.ogrid[:xlen, :ylen]
            distance_from_center = np.sqrt((x - xcen) ** 2 + (y - ycen) ** 2)
            
            mask = np.ones((xlen, ylen), dtype=np.float32)
            inside_circle = distance_from_center <= radius
            
            # Limb darkening formula: I(μ) = I₀[1 - u(1 - μ)]
            # where μ = cos(θ) = sqrt(1 - (r/R)²)
            r_normalized = distance_from_center[inside_circle] / radius
            mu = np.sqrt(np.maximum(0, 1 - r_normalized ** 2))
            mask[inside_circle] = 1 - u_coefficient * (1 - mu)
            
            # Cache the mask
            self._limb_mask_cache = (cache_key, mask)
        
        # Apply the cached mask
        _, mask = self._limb_mask_cache
        return grayscale * mask

    # def im_posterize(self, img, tol=15, n_clusters=4, min_count=20):
    #     """Posterize grayscale image using KMeans clustering and remap to speckey values"""
    #     target_values = np.array(list(self.speckey.values()), dtype=np.uint8)

    #     # Flatten to 1D array of intensities
    #     pixels = img.ravel()

    #     # Filter out rare/noisy intensities
    #     unique, counts = np.unique(pixels, return_counts=True)
    #     valid = unique[counts >= min_count]
    #     filtered_pixels = pixels[np.isin(pixels, valid)]

    #     if filtered_pixels.size == 0:
    #         return np.zeros_like(img, dtype=np.uint8), {}

    #     # Run KMeans on filtered values
    #     X = filtered_pixels.reshape(-1, 1)
    #     kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init=10)
    #     kmeans.fit(X)

    #     # Round centroids to nearest integers and sort them
    #     centroids = np.sort(np.rint(kmeans.cluster_centers_.flatten()).astype(int))

    #     # Map sorted centroids to speckey values (smallest->0, next->80, etc.)
    #     centroid_map = dict(zip(centroids, target_values))

    #     # Create output image initialized to 0
    #     output_img = np.zeros_like(img, dtype=np.uint8)

    #     # Apply mapping: within tolerance of centroid -> mapped speckey value
    #     for c, mapped_val in centroid_map.items():
    #         mask = np.abs(img.astype(int) - int(c)) <= tol
    #         output_img[mask] = mapped_val

    #     return output_img #, centroid_map

    def im_posterize(self, img):
        """Posterize grayscale image by mapping to nearest speckey value"""
        target_values = np.array(sorted(self.speckey.values()), dtype=np.uint8)  # [0, 150, 200, 250]
        
        # For each pixel, find the nearest target value
        img_flat = img.ravel().astype(float)
        
        # Vectorized nearest-neighbor assignment
        distances = np.abs(img_flat[:, np.newaxis] - target_values)
        nearest_indices = np.argmin(distances, axis=1)
        output_img = target_values[nearest_indices]
        
        return np.array(output_img.reshape(img.shape).astype(np.uint8))

    def render_specmask(self, specmap, posterize=False):
        """
        Render spectral mask with full sphere visible.

        Parameters:
            specmap (array): spectral map
            posterize (bool): whether posterization of image is necessary
        
        Returns:
            specmap_clean (array): rendered specmap, posterized if posterize=True
        """
        self.configure_plotter()
        
        # Validate mesh dimensions
        if not hasattr(self.mesh, 'x') or not hasattr(self.mesh, 'y') or not hasattr(self.mesh, 'z'):
            raise AttributeError("Mesh must have x, y, z attributes")
            
        grid = pv.StructuredGrid(self.mesh.x, self.mesh.y, self.mesh.z)
        
        # Validate specmap dimensions
        expected_points = grid.n_points
        if specmap.size != expected_points:
            print(f"Warning: specmap size {specmap.size} doesn't match grid points {expected_points}")
            
        grid.point_data['scalars'] = specmap.ravel(order='F')
        
        # set color limits based on specmap range
        clim = [0,255]

        # Add mesh to plotter
        self.plotter.add_mesh(grid, show_scalar_bar=False, interpolate_before_map=True,
                              cmap='gray', clim=clim)
        self.plotter.camera_set = True  # Lock camera after initial setup

        # Return grayscale screenshot
        screenshot = self.plotter.screenshot()
        if screenshot is None or screenshot.size == 0:
            raise RuntimeError("Screenshot failed - empty or None result")
            
        specmask_clean = screenshot[..., 0]
        
        # Clean up
        self.plotter.close()
        self.plotter = None

        if posterize: 
            return self.im_posterize(specmask_clean)
        else:
            return specmask_clean

    def render_frame(self, atmospheric_data, colorlim=[0.0, 1.0], 
                        apply_limb_darkening=False, u_coefficient=0.3):
        """
        Render single timestep with full sphere visible.
        
        Args:
            atmospheric_data: Atmospheric intensity data
            colorlim: Color limits for mapping
            apply_limb_darkening: Whether to apply limb darkening effect
            u_coefficient: Limb darkening coefficient (0.5-0.9 typical)
        
        Returns:
            grayscale: Rendered frame with optional limb darkening applied
        """
        # Don't reconfigure plotter if it already exists
        if self.plotter is None:
            self.configure_plotter()
            
        grid = pv.StructuredGrid(self.mesh.x, self.mesh.y, self.mesh.z)
        
        # Validate atmospheric_data dimensions
        expected_points = grid.n_points
        if atmospheric_data.size != expected_points:
            print(f"Warning: atmospheric_data size {atmospheric_data.size} doesn't match grid points {expected_points}")
            
        grid.point_data['scalars'] = atmospheric_data.ravel(order='F')

        self.plotter.add_mesh(grid, cmap='gray', show_scalar_bar=False,
                              clim=colorlim, interpolate_before_map=True)
        
        if not hasattr(self.plotter, 'camera_set') or not self.plotter.camera_set:
            self.plotter.camera_set = True  # Lock camera after initial setup
        
        screenshot = self.plotter.screenshot()
        if screenshot is None or screenshot.size == 0:
            raise RuntimeError("Screenshot failed - empty or None result")
            
        grayscale = np.dot(screenshot[..., :3], [0.2989, 0.5870, 0.1140])
        # Apply limb darkening if requested
        if apply_limb_darkening:
            grayscale = self._apply_limb_darkening(grayscale, u_coefficient)
        else:
            # run only once to cache the mask
            if self._limb_mask_cache is None:
                _ = self._apply_limb_darkening(grayscale, u_coefficient)

        return grayscale

    def digitizer(self, img, bins):
        return None
    
    def photometry(self, config, model, inclin):
        """
        Generate photometry images over time.
        
        Parameters:
            config (class instance): AtmosphereConfig instance with configured parameters
            model (class instance): AtomsphereModel instance with configured parameters
            inclin (list): list of inclination values
            colorlim (list): list of color limits
        
        Returns:
            photometry_array (array): array containing photometry data
        """
        colorlim = config.colorlim
        photometry_array = np.empty((config.time_config.frames, 
                                    self.imsize[0], self.imsize[1]), dtype=np.float32)
        time_array = config.time_config.time_array
        self.configure_plotter()

        try:
            for i, t in enumerate(tqdm(time_array, desc=f"Inclination {inclin}")):
                self.plotter.clear()  # Clear previous meshes
                atmospheric_data = model.generate_atmosphere(t)
                frame = self.render_frame(atmospheric_data, colorlim)
                photometry_array[i] = frame
        finally:
            if self.plotter is not None:
                self.plotter.close()
                self.plotter = None

        return photometry_array
    
    def __del__(self):
        """Ensure plotter is closed when object is deleted"""
        if hasattr(self, 'plotter') and self.plotter is not None:
            self.plotter.close()

# ==============================================================================
# Light curve generation and plotting
# ==============================================================================
class LightcurveGenerator:
    """
    Compute photometric lightcurves from gray_array and specmask,
    with vectorized flux calculation for efficiency.
    """
    def __init__(self, results: dict):
        self.results = results

    def generate_all(self):
        """Run flux generation for all inclinations in results."""
        for inclination, data in self.results.items():
            self.results[inclination]['flux'] = self._generate_flux(data)

    def _generate_flux(self, data: dict, verbose=False) -> dict:
        """Compute normalized fluxes and area fractions for a single inclination."""
        # Convert gray_array (list of RGB frames) → ndarray (t, h, w, 3), float32
        """Compute normalized fluxes and area fractions for a single inclination."""
        if verbose:
            print(f"Debug - type of sum function: {type(sum)}")
            print(f"Debug - sum function: {sum}")
        gray_array = np.array(data['gray_array'], dtype=np.float32)   # shape (t, h, w, 3)
        time_array = np.array(data['time_array'])   # (t,)
        specmask = data['specmask']
        speckey = data['metadata']['speckey']
        # Handle different gray_array formats
        if len(gray_array.shape) == 4:  # (t, h, w, 3) - RGB
            # Convert RGB to grayscale
            gray_array = np.dot(gray_array[..., :3], [0.2989, 0.5870, 0.1140])
        elif len(gray_array.shape) == 3:  # (t, h, w) - already grayscale
            pass
        else:
            raise ValueError(f"Unexpected gray_array shape: {gray_array.shape}")
        frame_height, frame_width = gray_array.shape[1:3]
        norm_const = frame_height * frame_width
        # CRITICAL FIX: Use tolerance-based matching for specmask values
        # The grayscale conversion changes exact spectral values
        indices = {}
        tolerance = 50  # Allow some tolerance for matching
        
        for region, target_value in speckey.items():
            if region == 'BG':
                continue
                
            # Find pixels close to the target spectral value
            mask = np.abs(specmask - target_value) <= tolerance
            idx = np.where(mask)
            
            if len(idx[0]) > 0:
                indices[region] = idx
                if verbose: print(f"Debug - Found {len(idx[0])} pixels for region '{region}' (target: {target_value})")
            else:
                if verbose: print(f"Warning: No pixels found for region '{region}' with target value {target_value}")
                # Try finding the closest values
                unique_vals = np.unique(specmask)
                closest_val = unique_vals[np.argmin(np.abs(unique_vals - target_value))]
                if verbose: print(f"  Closest value in specmask: {closest_val}")
        if not indices:
            print("ERROR: No spectral regions found! This indicates a problem with specmask generation.")
            return {
                'area_fractions': {},
                'time': time_array,
                'fluxtotal': np.zeros(len(time_array))
            }
        # Calculate area fractions
        pixel_counts = {region: len(idx[0]) for region, idx in indices.items()}
        total_area = __builtins__.sum(pixel_counts.values())  # Explicitly convert to list if needed
        total_area = int(total_area)  # Force it to be an integer
        
        if total_area == 0:
            area_fractions = {region: 0.0 for region in indices.keys()}
        else:
            area_fractions = {region: count / total_area for region, count in pixel_counts.items()}
        
        if verbose:
            print(f"Debug - pixel_counts: {pixel_counts}")
            print(f"Debug - pixel_counts.values(): {pixel_counts.values()}")
            print(f"Debug - type of total_area: {type(total_area)}, value: {total_area}")
        # Vectorized flux computation
        fluxes = {}
        for region, idx in indices.items():
            # Extract pixel values across all time steps
            region_pixels = gray_array[:, idx[0], idx[1]]  # Shape: (time, n_pixels)
            flux_region = region_pixels.mean(axis=1)  # Average flux per timestep
            fluxes[f"flux{region}"] = flux_region / norm_const * len(idx[0])  # Scale by region size
        # Total flux
        if fluxes:
            fluxtotal = __builtins__.sum(fluxes.values())
        else:
            fluxtotal = np.zeros(len(time_array))
        if verbose: print(f"Debug - Generated fluxes for regions: {list(fluxes.keys())}")
        return {
            'area_fractions': area_fractions,
            'time': time_array,
            **fluxes,
            'fluxtotal': fluxtotal
        }
    
    def plot_all_inclinations(self, flux_type='fluxtotal', normalize=True, 
                                figsize=(10, 6), alpha=0.8, verbose=False):
            """
            Plot lightcurves for all inclinations on the same plot
            Parameters
            ----------
            flux_type : str, optional
                Which flux to plot. Options: 'fluxtotal', 'fluxA', 'fluxB', 'fluxP', etc.
                Default is 'fluxtotal' to show total flux from all regions.
            normalize : bool, optional
                If True, normalize each lightcurve by its own maximum value.
            figsize : tuple, optional
                Figure size (width, height) in inches.
            alpha : float, optional
                Line transparency (0-1).
            """            
            # Check if flux data exists
            flux_data_available = {inc: self.results[inc].get('flux', None) 
                                for inc in self.results.keys()}
            
            missing_flux = [inc for inc, data in flux_data_available.items() if data is None]
            if missing_flux:
                raise ValueError(f"No flux data found for inclinations {missing_flux}. "
                                "Run generate_all() first.")
            # Determine available flux types from first inclination
            first_flux_data = list(flux_data_available.values())[0]
            available_flux_types = [k for k in first_flux_data.keys() 
                                if k.startswith('flux') or k == 'fluxtotal']
            
            if flux_type not in available_flux_types:
                raise ValueError(f"Flux type '{flux_type}' not found. "
                                f"Available types: {available_flux_types}")
            plt.figure(figsize=figsize)
            
            # Sort inclinations for consistent color progression
            sorted_inclinations = sorted(self.results.keys())
            
            for inclination in sorted_inclinations:
                flux_data = self.results[inclination]['flux']
                time = flux_data['time']
                flux = flux_data[flux_type]
                
                # Apply baseline correction (shift to same baseline)
                flux_shifted = 1 + (flux-np.mean(flux))/np.max(flux)
                
                # Normalize by maximum average flux across all inclinations
                flux_final = flux_shifted
                
                plt.plot(time, flux_final, label=f'{inclination}°', 
                        alpha=alpha, linewidth=1.5)
            plt.xlabel("Time (hours)")
            ylabel = "Normalized Intensity" if normalize else "Intensity"
            plt.ylabel(ylabel)
            plt.title(f"Lightcurves for All Inclinations: [{flux_type}]")
            
            # Create a nice legend
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', 
                    title='Inclination')
            
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.show()

#=============================================================================
# Multiprocessing helper function: Needs to be placed outside class
# ============================================================================

def process_single_inclination(inclin, config, mesh, model, colorlim):
    """Top-level function for multiprocessing"""
    visualizer = AtmosphereVisualizer(mesh=mesh, 
                                      speckey=config.speckey,
                                      config=config,
                                      imsize=[300, 300], 
                                      inclination=inclin)
    
    gray_array = visualizer.photometry(config, model, inclin)
    specmap = model.generate_specmap()
    specmask = visualizer.render_specmask(specmap, posterize=True)
    
    return {
        'gray_array': gray_array,
        'time_array': config.time_config.time_array,
        'metadata': config._to_dict(),
        'specmask': specmask,
        'limb_mask_cache': visualizer._limb_mask_cache
    }

def process_time_chunk(inclin, time_chunk, config, mesh, model, colorlim):
    """
    Process a chunk of timesteps for a given inclination.
    This reuses the plotter across multiple frames.
    
    Parameters:
        inclin: inclination angle
        time_chunk: list/array of time points to process
        config: configuration object
        mesh: mesh object
        model: atmospheric model
        colorlim: color limits
    
    Returns:
        dict with frames for this chunk
    """
    visualizer = AtmosphereVisualizer(mesh=mesh, 
                                      speckey=config.speckey,
                                      config=config,
                                      imsize=[300, 300], 
                                      inclination=inclin)
    
    visualizer.configure_plotter()
    frames = []
    
    try:
        for t in time_chunk:
            visualizer.plotter.clear()
            atmospheric_data = model.generate_atmosphere(t)
            frame = visualizer.render_frame(atmospheric_data, colorlim)
            frames.append(frame)
    finally:
        if visualizer.plotter is not None:
            visualizer.plotter.close()
            visualizer.plotter = None
    
    return {
        'inclin': inclin,
        'times': time_chunk,
        'frames': np.array(frames)
    }
# ==============================================================================
# Run the simulation and visualization
#===============================================================================
class SimulationRunner:
    def __init__(self, mesh, config, model, inclinations, base_path='output'):
        self.mesh = mesh
        self.config = config  # Directly use the provided AtmosphericConfig instance
        self.model = model
        self.inclinations = inclinations

        # Get current script directory and create full output path
        current_dir = os.path.dirname(os.path.abspath(__file__))
        self.base_path = os.path.join(current_dir, base_path)
        
        # Create directory if it doesn't exist
        os.makedirs(self.base_path, exist_ok=True)

        self.results = {}

    ### Running the simulation with multi-inclination processing
    def run_simulation(self, colorlim=[0.5, 1.5], n_workers=4):
        from multiprocessing import Pool
        start = time.perf_counter()
        
        args_list = [(inclin, self.config, self.mesh, self.model, colorlim)
                    for inclin in self.inclinations]
        
        with Pool(processes=n_workers) as pool:
            results_list = pool.starmap(process_single_inclination, args_list)
        
        self.results = dict(zip(self.inclinations, results_list))
        
        end = time.perf_counter()
        print(f"Simulation completed in {end - start:.2f} seconds.")
    
        return self.results

    # ### Running the simulation with multi-chunk processing
    # def run_simulation(self, colorlim=[0.0, 1.0], n_workers=6, chunk_size=40):
    #     """
    #     Run simulation with chunk-based parallelization.
    #     Each worker processes multiple frames to amortize plotter setup cost.
        
    #     Parameters:
    #         colorlim: color limits for visualization
    #         n_workers: number of parallel workers (defaults to cpu_count - 1)
    #         chunk_size: number of frames per chunk (tune this for performance)
        
    #     Returns:
    #         results dictionary organized by inclination
    #     """
        
    #     if n_workers is None:
    #         n_workers = max(1, os.cpu_count() - 1)
        
    #     start = time.perf_counter()
        
    #     time_array = self.config.time_config.time_array
        
    #     # Split time array into chunks
    #     time_chunks = [time_array[i:i+chunk_size] for i in range(0, len(time_array), chunk_size)]
        
    #     # Create tasks: (inclination, time_chunk) pairs
    #     args_list = [
    #         (inclin, chunk, self.config, self.mesh, self.model)
    #         for inclin in self.inclinations
    #         for chunk in time_chunks
    #     ]
        
    #     n_chunks = len(time_chunks)
    #     n_tasks = len(args_list)
    #     print(f"Processing {len(time_array)} frames in {n_chunks} chunks of ~{chunk_size} frames")
    #     print(f"Total tasks: {n_tasks} across {n_workers} workers...")
        
    #     # Process chunks in parallel
    #     from multiprocessing import Pool
    #     with Pool(processes=n_workers) as pool:
    #         chunk_results = pool.starmap(process_time_chunk, args_list)
        
    #     # Reorganize results by inclination
    #     self.results = {}
        
    #     for inclin in self.inclinations:
    #         # Get all chunks for this inclination
    #         inclin_chunks = [r for r in chunk_results if r['inclin'] == inclin]
            
    #         # Sort by first time in chunk to maintain temporal order
    #         inclin_chunks.sort(key=lambda x: x['times'][0])
            
    #         # Concatenate all frames
    #         all_frames = np.concatenate([chunk['frames'] for chunk in inclin_chunks], axis=0)
            
    #         # Generate specmap (only once per inclination)
    #         specmap = self.model.generate_specmap()
    #         visualizer = AtmosphereVisualizer(mesh=self.mesh, 
    #                                         speckey=self.config.speckey,
    #                                         config=self.config,
    #                                         imsize=[300, 300], 
    #                                         inclination=inclin)
    #         specmask = visualizer.render_specmask(specmap, posterize=False)
            
    #         # Generate limb mask by rendering a dummy frame
    #         dummy_atm = self.model.generate_atmosphere(0)
    #         _ = visualizer.render_frame(dummy_atm, self.config.colorlim, apply_limb_darkening=True)
    #         limb_mask_cache = visualizer._limb_mask_cache

    #         self.results[inclin] = {
    #             'gray_array': all_frames,
    #             'time_array': time_array,
    #             'metadata': self.config._to_dict(),
    #             'specmask': specmask,
    #             'limb_mask_cache': limb_mask_cache,
    #         }
        
    #     end = time.perf_counter()
    #     print(f"Simulation completed in {end - start:.2f} seconds.")

    #     return self.results
            
# ============================================================================
# Input output handler and data management
# ============================================================================
    def save_simulation(self, prefix, compression='gzip'):
        # compression: gzip much smaller filesize than without compression
        results = self.results
        output_path = os.path.join(self.base_path, f'{prefix}.h5')
        with h5py.File(output_path, 'w') as f:
            for inclin, data in results.items():
                gray_data = data['gray_array']
                if not isinstance(gray_data, np.ndarray):
                    gray_data = np.array(gray_data, dtype=np.float32)
                f.create_dataset(f'{inclin}/gray_array', 
                                data=gray_data, chunks=True, compression=compression)
                f.create_dataset(f'{inclin}/specmask', data=data['specmask'])
                f.create_dataset(f'{inclin}/time_array', data=data['time_array'])
                # Save metadata as JSON string
                metadata_json = json.dumps(data['metadata'])
                f.create_dataset(f'{inclin}/metadata', data=metadata_json)
                # Save limb_mask_cache if exists
                if data.get('limb_mask_cache') is not None:
                    _, limb_cache_mask = data['limb_mask_cache']
                    f.create_dataset(f'{inclin}/limb_mask_cache_mask', data=limb_cache_mask)
                # f.create_dataset(f'{inclin}/centroids_specmask', data=str(data['centroids_specmask']))

    # ===================================
    # Convert gray_array to video
    # ===================================
    @staticmethod
    def save_video_from_array(gray_array, filepath, fps=30, cmap='inferno', quality='high', clim=None):
        """
        Save grayscale array as video with colormap applied.
        
        Parameters:
        -----------
        gray_array : ndarray
            Shape (n_frames, height, width) - grayscale values 0-255
        filepath : str
            Output video path
        fps : int
            Frames per second
        cmap : str
            Matplotlib colormap name (default: 'plasma')
        quality : str
            'high' or 'medium' - affects bitrate
        clim : list or tuple, optional
            [vmin, vmax] to clip values before normalizing (default: None uses full range)
        """
        import matplotlib.pyplot as plt
        
        # Apply clim if provided
        if clim is not None:
            gray_array = np.clip(gray_array, clim[0], clim[1])
            # Normalize to 0-255 range based on clim
            frames_uint8 = ((gray_array - clim[0]) / (clim[1] - clim[0]) * 255).astype(np.uint8)
        else:
            # Ensure frames are uint8
            if gray_array.dtype != np.uint8:
                frames_uint8 = gray_array.astype(np.uint8)
            else:
                frames_uint8 = gray_array
        
        # Get colormap from matplotlib
        colormap = plt.get_cmap(cmap)
        
        # Apply colormap to each frame
        frames_colored = []
        for frame in frames_uint8:
            # Normalize to 0-1 for colormap
            frame_norm = frame / 255.0
            # Apply colormap (returns RGBA)
            frame_colored = colormap(frame_norm)
            # Convert to RGB (0-255) and drop alpha channel
            frame_rgb = (frame_colored[:, :, :3] * 255).astype(np.uint8)
            # Convert RGB to BGR for OpenCV
            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            frames_colored.append(frame_bgr)
        
        height, width, _ = frames_colored[0].shape
        
        # Remove existing file if it exists
        if os.path.exists(filepath):
            os.remove(filepath)
        
        # Use H.264 codec for better compression
        fourcc = cv2.VideoWriter_fourcc(*'avc1')  # or try 'H264', 'X264'
        out = cv2.VideoWriter(filepath, fourcc, fps, (width, height))
        
        if not out.isOpened():
            # Fallback to mp4v if H.264 not available
            print("H.264 codec not available, falling back to mp4v")
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(filepath, fourcc, fps, (width, height))
        
        if not out.isOpened():
            raise RuntimeError(f"Failed to open VideoWriter: {filepath}")
        
        for frame in frames_colored:
            out.write(frame)
        
        out.release()
        print(f"Video saved: {filepath}")

    def create_videos_from_h5(self, prefix, fps=30, clim=None, apply_limb_darkening=True, n_workers=4):
        """
        Create grayscale videos for each inclination stored in an HDF5 file.
        
        Parameters:
        -----------
        prefix : str
            HDF5 file prefix
        fps : int
            Frames per second
        clim : list or tuple, optional
            [vmin, vmax] to clip values before normalizing (default: None uses full range)
        apply_limb_darkening : bool
            Whether to apply limb darkening mask
        n_workers : int, optional
            Number of parallel workers (default: None uses CPU count)
        """
        from multiprocessing import Pool, cpu_count
        
        h5_file_path = os.path.join(self.base_path, f'{prefix}.h5')
        base_name = os.path.splitext(os.path.basename(h5_file_path))[0]
        output_folder = os.path.join(self.base_path, f"{base_name}_video")
        os.makedirs(output_folder, exist_ok=True)

        # Prepare arguments for parallel processing
        args_list = []
        with h5py.File(h5_file_path, 'r') as f:
            for inclin in f.keys():
                video_path = os.path.join(output_folder, f"{base_name}_inclin={inclin}.mp4")
                args_list.append((h5_file_path, inclin, video_path, fps, clim, apply_limb_darkening))
        
        # Process videos in parallel
        n_workers = n_workers or min(cpu_count(), len(args_list))
        if n_workers > 1 and len(args_list) > 1:
            print(f"Creating {len(args_list)} videos using {n_workers} workers...")
            with Pool(processes=n_workers) as pool:
                pool.starmap(self._create_single_video, args_list)
        else:
            # Single-threaded for small jobs
            for args in args_list:
                self._create_single_video(*args)
    
    @staticmethod
    def _create_single_video(h5_file_path, inclin, video_path, fps, clim, apply_limb_darkening):
        """Helper function to create a single video (for parallel processing)."""
        with h5py.File(h5_file_path, 'r') as f:
            gray_array = f[f'{inclin}/gray_array'][:]
            
            if apply_limb_darkening:
                # Check if limb mask exists before trying to use it
                limb_mask_key = f'{inclin}/limb_mask_cache_mask'
                if limb_mask_key in f:
                    limb_mask = f[limb_mask_key][:]
                    gray_array = limb_mask * gray_array
                else:
                    print(f"Warning: limb_mask_cache_mask not found for inclination {inclin}, skipping limb darkening")
        
        SimulationRunner.save_video_from_array(gray_array, video_path, fps=fps, clim=clim)

#%%
# ==============================================================================
# Set up configurations and test call
# =============================================================================

if __name__ == "__main__":
    # VTK and OpenGL check
    print("VTK Version:", vtk.vtkVersion.GetVTKVersion())
    print("OpenGL2 Enabled:", hasattr(vtk, 'vtkOpenGLRenderWindow'))

    # runName = 'test_polarMonitor_static'  # Simulation identifier
    # runName = 'test_polarMonitor_dynamic'  # Simulation identifier

    # runName = 'test_polar_v0_static'
    # runName = 'test_polar_v0_dynamic'

    # runName = 'test_polar_v1_static'
    runName = 'test_polar_v1_dynamic'

    # runName = 'test_polar_v2_static'
    # runName = 'test_polar_v2_dynamic'

    # runName = 'test_onlyVortex_v0_dynamic1'
    # runName = 'test_onlyVortex_v0_dynamic2'
    # runName = 'test_onlyVortex_v0_static'

    # runName = 'polar_v1_static_baseline120'
    # runName = 'polar_v1_dynamic_baseline240'

    # runName = 'jwst_v0_static'
    # runName = 'jwst_v0_dynamic'

    # Set up band_config: latitudinal features
    Ppol, Pband, Prot = 60, 5, 5  # Periods in hours
    Fpolar, Fband, Fambient = 0.98, 1, 1 # amp
    Fpolar_var, Fband_var, Fambient_var = 0.05, 0.15, 0.00 # variab
    # variability: amp + variab * sin(...)
    ''' 
    bandConfig has to be in this format:
    [latUp, latDown, brightness, type, phase, period, variability] 
    '''

    ### For test and jwst 
    bandConfig = [
        # [lat2, lat1, amplitude, type, phase, period]
        [90, 65, Fpolar, 'P', 0, Ppol, Prot, Fpolar_var],
        [15, 5, Fband, 'B', 0, Pband, Prot, Fband_var], 
        [-5, -15, Fband, 'B', 0, Pband, Prot, Fband_var],
        [-65, -90, Fpolar, 'P', 0, Ppol, Prot, Fpolar_var]]

    ### For test_onlyVortex
    if 'test_onlyVortex' in runName:
        Fpolar_var, Fband_var, Fambient_var = 0.00, 0.00, 0.00 # variab
        Fpolar, Fband, Fambient = 1, 1, 1 # amp
        bandConfig = [
            # [lat2, lat1, amplitude, type, phase, period]
            [90, 65, Fpolar, 'P', 0, Ppol, Prot, Fpolar_var],
            [-65, -90, Fpolar, 'P', 0, Ppol, Prot, Fpolar_var]]

    if 'test_polar_v1' in runName:
        Fpolar, Fband, Fambient = 1, 1, 1 # amp
        Fpolar_var, Fband_var, Fambient_var = 0.02, 0.15, 0.00 # variab
        bandConfig = [
            # [lat2, lat1, amplitude, type, phase, period]
            [90, 65, Fpolar, 'P', 0, Ppol, Prot, Fpolar_var],
            [17, 10, Fband, 'B', 0, Pband/2, Prot, Fband_var],
            [8, 2, Fband, 'B', 10, Pband, Prot, Fband_var], 
            [-2, -8, Fband, 'B', 10, Pband, Prot, Fband_var],
            [-10, -17, Fband, 'B', 0, Pband/2, Prot, Fband_var],
            [-65, -90, Fpolar, 'P', 0, Ppol, Prot, Fpolar_var]]

    if 'test_polar_v2' in runName:
        Fpolar, Fband, Fambient = 1, 1, 1 # amp
        Fpolar_var, Fband_var, Fambient_var = 0.02, 0.15, 0.00 # variab
        bandConfig = [
            # [lat2, lat1, amplitude, type, phase, period]
            [90, 65, Fpolar, 'P', 0, Ppol, Prot, Fpolar_var],
            [28, 21, Fband, 'B', 0, Pband/2, Prot, Fband_var],
            [12, 5, Fband, 'B', 10, Pband, Prot, Fband_var], 
            [-5, -12, Fband, 'B', 10, Pband, Prot, Fband_var],
            [-21, -28, Fband, 'B', 0, Pband/2, Prot, Fband_var],
            [-65, -90, Fpolar, 'P', 0, Ppol, Prot, Fpolar_var]]


    n_vortice = 5 # number of vortices
    radius_frac = 0.0075 # in unit of polar cap area
    drift = True # whether the vortex drifts
    centerLat = 80 # great circle line 
    ampVortex = 0.15 # amplitude in vortex/polar cap unit at time t
    vortexConfig = [n_vortice, radius_frac, drift, centerLat, ampVortex]
    # vortexConfig = None

    # Set up modu_config: static or dynamic polar monitor
    if 'static' in runName:
        moduConfig = 'polarStatic'
    elif 'dynamic' in runName:
        moduConfig = 'polarDynamic'
    else:
        moduConfig = 'polarStatic'

    if 'jwst' in runName:
        # Set up time_config: JWST-like observation
        timeConfigVar = TimeConfig(t0=0, t1=180, frames=360, option='jwst', jwst_setup={'gap':60, 'segments':3})
    elif 'test' in runName:
        timeConfigVar = TimeConfig(t0=0, t1=60, frames=240, option='full')
    else:
        timeConfigVar = TimeConfig(t0=0, t1=60, frames=120, option='full')

    ### test_onlyVortex
    if 'test_onlyVortex' in runName:
        timeConfigVar = TimeConfig(t0=0, t1=180, frames=360, option='full')

    # Set up atmosphere config: the rest of the simulation
    atmo_config = AtmosphericConfig(
        band_config=bandConfig,  # This is your band configuration list
        # modu_config='polarStatic',
        modu_config=moduConfig,
        modelname='production1',
        time_config=timeConfigVar,
        Fambient=Fambient,  # This will be accessible as config.Fambient
        Fband=Fband,
        Fpolar=Fpolar,
        Fambient_var=Fambient_var,
        Fband_var=Fband_var,
        Fpolar_var=Fpolar_var,
        Prot=5.0,
        speckey= {'BG':0, 'A': 150, 'B': 200, 'P': 250},
        colorlim=[0.0, 2.0] # color_lim sets the mapping range from amplitude to [0.255]
    )

    # Set up the spherical mesh, initialization
    mesh = SphericalMesh(resolution=400)
    model = AtmosphericModel(mesh, atmo_config, vortexConfig)

    # incli_array = [40] # List of inclinations to simulate
    # incli_array = [0, 50, 90]
    incli_array = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90]
    if 'test_onlyVortex' in runName: incli_array = [0, 30, 60, 90]
    # incli_array = [0]
    # Set up the inclination configuration
    runner = SimulationRunner(
        mesh=mesh,
        config=atmo_config,
        model=model,
        inclinations=incli_array
    )

    # Run the simulation for a specific time range and number of frames
    results = runner.run_simulation() 
    
    # Usage example:
    fig, vortexPaths = model.track_and_plot_vortex_paths(np.arange(0, 130, 10), save_path=False)
    # # # export vortexPaths dictionary to pickle file
    # import pickle
    # vortex_path_file = os.path.join(runner.base_path, f'default120_vortex_paths.pkl')
    # with open(vortex_path_file, 'wb') as f:
    #     pickle.dump(vortexPaths, f)
    # print(f'Vortex paths saved to {vortex_path_file}')

    # Save the simulation results
    runner.save_simulation(runName)

    # Save a video of simulation results
    runner.create_videos_from_h5(runName, fps=6, clim=[0,150])

    #%% Binned image generator and plotter
    def generate_bins(a, b, nbin, type='linear', power=2):
        """
        Generate bins with power-law spacing.
        """
        if type == 'linear':
            bins = np.linspace(a, b, nbin + 1)
        elif type == 'power':
            # Generate interior points only (excluding 0 and 1)
            t = np.linspace(0, 1, nbin)  # Exclude endpoints
            normalized = t ** power
            bins = a + normalized * (b - a)
        return [0] + bins.tolist() + [255]

    # bins = generate_bins(106.25, 143.75, nbin=10, type='power', power=0.9)
    bins = generate_bins(106.25, 143.75, nbin=18, type='linear')

    def plot_frames(h5_path, inclination, t=0, handle='gray', plot_discrete=True, bins=None):
        with h5py.File(h5_path, 'r') as f:
            data = f[f'{inclination}/gray_array'][t]  # Frame at time t
            limb_mask = f[f'{inclination}/limb_mask_cache_mask']
            spec = f[f'{inclination}/specmask']
            fig, axes = plt.subplots(1,3, figsize=(15,5))
            # Original data
            axes[0].imshow(limb_mask*data, vmin=0, vmax=155, cmap='inferno')
            # Binned image
            binned = np.digitize(np.array(data), bins, right=True)
            axes[1].imshow(binned, vmin=0, vmax=20, cmap='viridis')
            # Specmask
            axes[2].imshow(spec, cmap='viridis')
            plt.tight_layout()

            save_dir = 'output/10in_drift'
            os.makedirs(save_dir, exist_ok=True)
            filename = f'frame_incl{inclination}_t{t:03d}.png'
            filepath = os.path.join(save_dir, filename)
            # plt.savefig(filepath, dpi=150, bbox_inches='tight')
            # print(f'Saved: {filepath}')

            plt.show()
            plt.close()
        return binned

    if True:
        filepath = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output', runName+'.h5')
        for inc in incli_array:
            for t in range(2):
                binned = plot_frames(filepath, inclination=inc, t=3*t, bins=bins)

    ### Plot horizontal colorbar
    # Normalize bins for colormap

    # Create a colormap
    cmap = cm.viridis
    # Create a normalization based on bins
    norm = colors.Normalize(vmin=0, vmax=150)
    # Create a ScalarMappable for the colorbar
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    # Plot colorbar
    fig, ax = plt.subplots(figsize=(1,8))
    cbar = plt.colorbar(sm, cax=ax, orientation='vertical', ticks=bins, boundaries=bins)
    cbar.ax.set_yticklabels([f'{b:.2f}' for b in bins])
    plt.show()

    #%% LIGHT CURVE GENERATOR

    ### After running your simulation
    start = time.perf_counter()
    lc_generator = LightcurveGenerator(results)
    lc_generator.generate_all()
    end = time.perf_counter()
    print(f"Lightcurve generated in {end - start:.2f} seconds.")

    ### Plot total flux for all inclinations
    lc_generator.plot_all_inclinations()

    ### Plot regional flux (e.g., bands) for all inclinations  
    # lc_generator.plot_all_inclinations(flux_type='fluxA', normalize=False)
    # lc_generator.plot_all_inclinations(flux_type='fluxB', normalize=False)
    # lc_generator.plot_all_inclinations(flux_type='fluxP', normalize=False)

    # Compare flux types for specific inclination
    # lc_generator.plot_flux_comparison(inclination=40)
# %%
