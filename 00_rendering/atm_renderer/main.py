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
from tqdm import tqdm
import numpy as np
import numba
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
