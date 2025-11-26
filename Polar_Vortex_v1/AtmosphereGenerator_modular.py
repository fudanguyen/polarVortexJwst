#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on March 1 2025

Upgrade from AtmosphereGenerator.py:
- significant improvement in speed via vectorization of routines
- gpu acceleration with pyvista but mainly cpu computation
- added multi-pressure capability

@author: nguyendat
"""
#%% IMPORT LIBRARIES
import h5py
import pickle
# from mayavi import mlab
# =============================================================================
# Enable vtk-m GPU-backend 
from vtkmodules.vtkRenderingCore import vtkRenderWindow
# Create a render window
render_window = vtkRenderWindow()

# Check the rendering backend
print("Rendering Backend:", render_window.GetRenderingBackend())

# # Check GPU support
# print("GPU Acceleration Supported:", render_window.GetGPUSupported())

# # Check GPU details (if supported)
# if render_window.GetGPUSupported():
#     print("GPU Vendor:", render_window.GetGPUVendor())
#     print("GPU Renderer:", render_window.GetGPUName())
#     print("GPU Version:", render_window.GetGLVersion())
# =============================================================================
import pyvista as pv
from tqdm import tqdm
import numpy as np
# import numba
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.animation as animation
# from PIL import Image, ImageDraw
import warnings
from scipy.ndimage.filters import gaussian_filter
import pandas as pd
from sklearn.decomposition import PCA
from datetime import datetime

# new imports
# import imageio
# import cupy as cp
from PIL import Image

### Path management
import os
from os.path import join
folderarray = os.path.abspath('').split('/')
homedir = '/'
for i in range(len(folderarray)):
   homedir = join(homedir, folderarray[i])
plotPath = join(homedir, 'plot/')

########## Construct spherical mesh #######################
### Create a sphere
r = 1
pi = np.pi
cos = np.cos
sin = np.sin

### Construct the spherical mesh grid
phi0 = np.linspace(0, np.pi, 200)
theta0 = np.linspace(0, 2*np.pi, 200)
phi, theta = np.meshgrid(phi0, theta0)

x = r * sin(phi) * cos(theta)
y = r * sin(phi) * sin(theta)
z = r * cos(phi)

xsize, ysize = x.shape[0], x.shape[1]

##### Create new folder function ########
def create_folder(folder_path):
    # Check if the folder exists
    if not os.path.exists(folder_path):
        # If it doesn't exist, create the folder
        os.makedirs(folder_path)
        print(f"Folder '{folder_path}' created successfully.")
    # else:
        # print(f"Folder '{folder_path}' already exists.")

# # Example usage:
# folder_name = "my_folder"
# create_folder(folder_name)

#################### FUNCTIONS FOR CREATING ATMOSPHERIC FEATURES ##########

############## Create Planetary-scale waves ##############################
### unit: hour, rotational period: 5 hour
def plWave(x, value, t, w=xsize, f=0, amplitude=0.4, base=0.5, RP=5): # planetary scale waves
    """
    Creates planetary scale sine wave for bands that aligns with rotation period.

    Parameters:
        x (array): spatial position array
        value (float): base value/array to which sine modulation is added
        t (float): current time step
        w (int): spatial wavelength of sine wave
        f (float): phase offset of the sine wave
        amplitude (float): amplitude of sine wave
        base (int): base value around which sine wave oscillates
        RP (float): rotational period of object

    Returns:
        value + sine (array): combined pattern from adding time-varying sine wave and base value
    """
    # create the sine-wave modulation, such that the sine wave looks the same after
    # every rotational period RP
    # w is the spatial period equals to the pixel size
    sine = amplitude*np.sin(2*np.pi/w * (x + (t/RP)*w) + f*np.pi/180) + base
    # return the original value + sine wave
    return value + sine

############## Create uniform time-variant polar-layer ##############################
def polar_change(value=0, amplitude=0.25, t=0, f=0, RP=60):
    """
    Creates uniform time-variant polar layer by establishing flux value for the layer.

    Parameters:
        value (float): base value for flux
        amplitude (float): amplitude of the time-varying sine modulation of flux
        t (float): current time step
        f (float): phase offset for sine wave
        RP (float): modulation period of the flux

    Returns:
        flux (float): flux value at given time step
    """
    # value: polar cap base flux value
    flux = value + amplitude*np.sin(2*np.pi/RP*t + f*np.pi/180)
    return flux

############## QUALITY OF LIFE ##############
##### Limb-darkening mask-array for a square image
## STATUS: OPTIMIZED
def limb_darkening(arr, u_coefficient=0.75):
    """
    Creates limb darkening mask by identifying darkening boundaries in pixels.

    Parameters:
        arr (array): pixel array of first rendered frame
        u_coefficient (float): limb-darkening coefficient

    Returns:
        mask (array): new pixel array where inner circle has accordingly adjusted (limb-darkened) values
    """
    ## u_coefficient limb-darkening coefficient, assume a low limb darkening 
    ## (decreases with increasing wavelength, with decreasing temperature)
    xlen, ylen = arr.shape[0], arr.shape[1]
    if xlen != ylen:
        raise ValueError('Has to be square array')
    else:
        xcen, ycen = int(xlen / 2), int(ylen / 2)
        boundary_pixel = np.where(arr[:, ycen] > 0.)[0][0] - 1
        radius = xcen - boundary_pixel
        y, x = np.ogrid[:xlen, :ylen]
        distance_from_center = np.sqrt((x - xcen) ** 2 + (y - ycen) ** 2)
        mask = np.zeros((xlen, ylen))
        inside_circle = distance_from_center <= radius
        mask[inside_circle] = (1 - u_coefficient * (1 - np.sqrt((radius ** 2 - distance_from_center[inside_circle] ** 2) / (radius ** 2))))
        return mask
    
##### Create equi-distant list for vortices center coordinate ######
def equidistant_values(X0, X1, n):
    """
    Creates equi-distant list of center coordinates between two endpoints for the vortices.

    Parameters:
        X0 (float): starting interval coordinate
        X1 (float): ending interval coordinate
        n (int): number of coordinates to be returned

    Returns:
        array of n equally spaced center coordinates between X0 and X1
    """
    N = n+1
    if N < 1:
        raise ValueError("N must be greater than or equal to 1")
    if N == 1:
        return [X0]
    step = (X1 - X0) / (N - 1)
    equidistant_list = [X0 + i*step + step/2 for i in range(N)]
    return np.array(equidistant_list[:-1])

####### convert latitude to pixel location ############
def lat(coord,n=ysize):
    """
    Converts latitude to pixel location.
    
    Parameters:
        coord (float): latitude on object
        n (int): height of pixel array
    
    Returns:
        pixel location along the y-axis
    """
    return abs(coord-90)/180*n # pixel location

####### convert longitude to pixel location ############
def long(coord,n=xsize):
    """
    Converts longitude to pixel location.
    
    Parameters:
        coord (float): longitude on object
        n (int): width of pixel array
    
    Returns:
        pixel location along the x-axis
    """
    return abs(coord)/360*n # pixel location

####### area of region bounded by two latitudes #######
def area_bounded_latitudes(lat1, lat2, radius=1):
    """
    Determines area of region on sphere bounded by two latitudes.
    
    Parameters:
        lat1 (float): first latitude value
        lat2 (float): second latitude value
        radius (float): radius of sphere
    
    Returns:
        area on sphere between latitudes 1 and 2
    """
    ## lat1 > lat2, lat in degree
    return abs(2*np.pi * radius * (np.sin(lat2*np.pi/180)-np.sin(lat1*np.pi/180)))

####### Circle Vortices generator
def circle_vortice(recmap, coord, t=0, number=5, variable_vortice=False, rotation_period=30):
    # draw circular patch near the polar region
    # coord: [[lat1, lat2], ...]
    # spacing: equidistance between [number] of circular vortices
    # radius such that area_vortice / area_cap = percentage
    
    ### Elliptical vortices prop:
    # Setup each vortice to be individually variable (each with phase offset)
    radiusfrac = 0.3
    a, b = 0.75, 0.25

    if variable_vortice:
        phaseValue = np.array([0, -2, 5, -4, 8, 2, 4, -6, -8])
    else:
        phaseValue = np.zeros(9)
    
    for array in coord:
        lat1, lat2 = array[0], array[1]
        # center coordinates longitudinal, add one extra value so vortices reflow
        center_coord_long = equidistant_values(0, 360, n=number)
        center_coord_lat = (lat1 + lat2) / 2
        centerlat = int(lat(center_coord_lat))
        
        # Variable vortices
        amplitude = recmap[int(xsize / 2), ysize - 1]
        variableflux = 0.2 * np.sin(2 * np.pi / rotation_period * t - phaseValue) if variable_vortice else np.full(number, 0.2)
        
        # Set up elliptical vortices
        r_vortice = np.sqrt(radiusfrac * area_bounded_latitudes(lat1, lat2, radius=1)) * (xsize / np.pi)
        ar, br = a * r_vortice, b * r_vortice
        
        # Set up time variability for the polar spots such that they overflows
        revised_center_coord_long_px = long(center_coord_long) + (xsize) * (-t % rotation_period / rotation_period)
        revised_center_coord_long_px = np.where(revised_center_coord_long_px > xsize - 1, revised_center_coord_long_px - (xsize - 1), revised_center_coord_long_px)
        revised_center_coord_long_px = np.where(revised_center_coord_long_px < 0, revised_center_coord_long_px + (xsize - 1), revised_center_coord_long_px)
        
        dxCen = np.diff(long(center_coord_long))[0]
        if revised_center_coord_long_px.min() < long(center_coord_long).min():
            revised_center_coord_long_px = np.append(revised_center_coord_long_px, revised_center_coord_long_px.max() + dxCen)
        elif revised_center_coord_long_px.max() > long(center_coord_long).max():
            revised_center_coord_long_px = np.append(revised_center_coord_long_px, revised_center_coord_long_px.min() - dxCen)
        
        if len(center_coord_long) < len(revised_center_coord_long_px):
            variableflux = np.append(variableflux, variableflux[0])
        
        xx, yy = np.meshgrid(np.arange(xsize), np.arange(ysize), indexing='ij')
        for i, xi in enumerate(revised_center_coord_long_px):
            mask = ((xx - int(xi)) ** 2 / ar ** 2 + (yy - centerlat) ** 2 / br ** 2 <= 1) & (lat(lat2) > yy) & (yy > lat(lat1))
            recmap[mask] = amplitude + variableflux[i]
    
    return recmap

def circle_vortice_vectorized(recmap, coord, t=0, number=5, variable_vortice=False, rotation_period=30):
    """
    Generates vectorized circular vortices - circular patches in polar regions.

    Parameters:
        recmap (array): array representing planetary map
        coord (array): latitudinal boundaries for polar bands
        t (float): time step
        number (float): number of vortex center coordinates to be generated
        variable_vortice (bool): whether the vortex is variable
        rotation_period (float): rotation period of object
    
    Returns:
        recmap (array): modified planetary map with elliptical vortices with new flux values added
    """
    radiusfrac = 0.3
    #a, b = 0.7, 0.20

    phaseValue = np.zeros(9) if not variable_vortice else np.array([0, -2, 5, -4, 8, 2, 4, -6, -8])
    
    for array in coord:
        lat1, lat2 = array[0], array[1]
        center_coord_long = equidistant_values(0, 360, n=number)
        centerlong = long(center_coord_long) + xsize * (-t % rotation_period / rotation_period)
        centerlong %= xsize

        center_coord_lat = (lat1 + lat2) / 2
        centerlat = int(lat(center_coord_lat))

        area_cap = 2 * np.pi * abs(np.sin(np.radians(lat1)) - np.sin(np.radians(lat2)))
        r_vortice = np.sqrt(radiusfrac * area_cap) * (xsize / np.pi)

        rng = np.random.default_rng(11)
        max_drift_radius = r_vortice / 2

        theta = rng.uniform(0, 2 * np.pi, size=(5, 120))
        r = rng.uniform(0, max_drift_radius, size=(5, 120))
        drift_x_matrix = r * np.cos(theta)
        drift_y_matrix = r * np.sin(theta)

        cos_lat = np.cos(np.radians(center_coord_lat))
        radius_y = r_vortice
        radius_x = r_vortice / cos_lat
         
        amplitude = recmap[int(xsize / 2), ysize - 1]
        variableflux = 0.2 * np.sin(2 * np.pi / rotation_period * t - phaseValue) if variable_vortice else np.full(number, 0.2)
        
        # r_vortice = np.sqrt(radiusfrac * area_bounded_latitudes(lat1, lat2, radius=1)) * (xsize / np.pi)
        # ar, br = a * r_vortice, b * r_vortice
        
        # # Vectorized coordinate adjustments
        # revised_center_coord_long_px = long(center_coord_long) + (xsize) * (-t % rotation_period / rotation_period)
        # revised_center_coord_long_px = np.where(
        #     revised_center_coord_long_px > xsize - 1,
        #     revised_center_coord_long_px - (xsize - 1),
        #     revised_center_coord_long_px
        # )
        # revised_center_coord_long_px = np.where(
        #     revised_center_coord_long_px < 0,
        #     revised_center_coord_long_px + (xsize - 1),
        #     revised_center_coord_long_px
        # )
        
        # dxCen = np.diff(long(center_coord_long))[0]
        # if revised_center_coord_long_px.min() < long(center_coord_long).min():
        #     revised_center_coord_long_px = np.append(
        #         revised_center_coord_long_px,
        #         revised_center_coord_long_px.max() + dxCen
        #     )
        # elif revised_center_coord_long_px.max() > long(center_coord_long).max():
        #     revised_center_coord_long_px = np.append(
        #         revised_center_coord_long_px, 
        #         revised_center_coord_long_px.min() - dxCen
        #     )

        # # make sure flux array matches
        # if revised_center_coord_long_px.size > variableflux.size:
        #     variableflux = np.append(variableflux, variableflux[0])

        # Vectorized mask for vortices
        xx, yy = np.meshgrid(np.arange(xsize), np.arange(ysize), indexing='ij')
        for i, xi in enumerate(centerlong):
            dx = drift_x_matrix[i, t % 120]
            dy = drift_y_matrix[i, t % 120]

            lon_px_int = int((xi + float(dx) % xsize))
            lat_px_int = int(np.clip(float(centerlong + dy), 0, ysize - 1))

            mask = (
                (((xx - lon_px_int) / radius_x) ** 2 + ((yy - lat_px_int) / radius_y) ** 2) <= 1) & (yy <= lat(lat1)) & (yy >= lat(lat2))

            recmap[mask] = amplitude + variableflux[i % len(variableflux)]
            # xi_int = int(xi)
            # mask = (
            #     ((xx - xi_int) / radius_x) ** 2 + 
            #     ((yy - centerlat) / radius_y) ** 2 <= 1) & \
            #     (yy <= lat(lat1)) & (yy >= lat(lat2))
            # recmap[mask] = amplitude + variableflux[i % len(variableflux)]
    
    return recmap

#%% RUN THE ATMOSPHERE EVOLUTION 3D MODEL

# =============================================================================
# Create 2d map, wrap around 3d Mayavi projection, rotate the atmosphere
# and animate photometry frame.
# ===================   ==========================================================
startTime = datetime.now()

###############################
warnings.filterwarnings("ignore")
###############################

#### SAVE options
# TEST = True
TEST = False

#### INLINE PLOTTING: TURN OFF FOR FASTER COMPUTATION
inlinePlot = True
# inlinePlot = False

if TEST: 
    plotPath = join(homedir, 'plot/', 'test/')
else:
    plotPath = join(homedir, 'plot/')

####### draw the 3D animation; always true #######
draw_animation = True
# draw_animation = False

######## save animation as gif ################### 
# save_animation = True
save_animation = False

######## save the first still frame ##############
# save_still = True 
save_still = False 

######### save the spectral type map #############
# save_specmap = True
save_specmap = False
###############################

### save or run command 
if TEST == True:
    run_visualization_again, save_output_array = True, False
elif TEST == False:
    run_visualization_again, save_output_array = True, True
# run_visualization_again, save_output_array = False, False

# =============================================================================
# RUN THE VISUALIZATION OR READ PAST RESULTS
# =============================================================================
#### Flux + Period Value
### Sine Amplitutde

### ============= production1 =========================
Fband= 0.6
Fambient = 0.5
Fpolar = 0.7
### periods
Pband = 5.0
Ppol = 60.0

# =============================================================================
# Polar vs NoPolar Modulation config
# =============================================================================
## Polar color are different from band color, but the question is
## where the long-term modulation comes from band or pole.
## Three cases to solve this problem.

#### Case A: Polar cap with circle vortices that do not change individually.
modu_config = 'polarStatic'

# ##### Case B: Polar cap with circle vortices that change individually.
# modu_config = 'polarDynamic' # NOT USED IN FUDA & APAI 2024

##### Case C: Contains polar cap, but long-term change comes from bands.
# modu_config = 'noPolar'

# =============================================================================
# TEST ONLY
######### Configuration: Comment out to choose a config
### Simple, all planetary-scale sinusoid models. Contain a polar cap that changes over time.
###### modu_config = 'all_sinusoid' TEST ONLY
# =============================================================================

# =============================================================================
# Lookup-Spectra-Map Dict-Key: This value indicate which 'color' in the lookup
# map correspond to which spectras. A: Ambient, B: Bands, P: Polar.
# =============================================================================
speckey = {'A':0.25, 'B':0.58, 'P':0.75}
### Observation run 

# =============================================================================
# Time Array Construction: Decide photometry length, number of frames, 
# continuous or non-continous time-array.
# =============================================================================
option_time_array = 'simple'# simple: generate continuous evenly-spaced-time-array
# option_time_array = 'multivisit'# simple: generate continuous evenly-spaced-time-array'
### Color mapping
cmap = 'inferno'

frame_no = 60
t0, t1 = 0,60

### Other options for TESTING
# frame_no = 30
# t0, t1 = 0,60

# frame_no = 90
# t0, t1 = 0,30

# frame_no = 2
# t0, t1 = 29,30
# frame_no = 10

# =============================================================================
#  Inclination Value of the models.
# =============================================================================
# inclination = [-90, -80, -70, -60, -50, -40, -30, -20, -10, 0.]
inclination = [90., 80., 70., 60., 50., 40., 30., 20., 10., 0.]
# inclination = [-60, -10]
# inclination =[60]
# inclination = [-90]

# =============================================================================
# Read Pre-ran Models Instead of Running New Instances
# =============================================================================
print('###### Module=[%s], t=[%i, %i], frame_no=%i'%(modu_config, t0, t1, frame_no))
for counter, inclin in enumerate(inclination):
    if not run_visualization_again:
        modelname = input('modelname:')
        inclin = float(input('inclination:'))
        grayArrayPath = homedir+'/output/dataCube[%s][%s][%i][%i][%i][%i].h5'%(modu_config, modelname, inclin, t0, t1, frame_no)
        metaPath = homedir+'/output/[meta]dataCube[%s][%s][%i][%i][%i][%i].pkl'%(modu_config, modelname, inclin, t0, t1, frame_no)
        with h5py.File(grayArrayPath, 'r') as file:
            gray_array = file['dataset'][:]
        with open(metaPath, 'rb') as file:
            metadata = pickle.load(file)
# =============================================================================
# Run New Models: 2d Map construction starts here
# =============================================================================
    else:   
        print('  #### Run:[i=%i], %i/%i ####'%(inclin, counter+1, len(inclination)))
        ### model name  
        modelname = 'production3'
        # config = [[90, 65, Fpolar, 'P', 20, Ppol],
        #           [45, 38., Fband, 'B', 10, Pband/2],
        #           [25, 15, Fband, 'B', 150, Pband], 
        #           [-10, -20, Fband, 'B', -26, Pband],
        #           [-33, -40, Fband, 'B', 135, Pband/2],
        #           [-65, -90, Fpolar, 'P', 0, Ppol]]

        config = [[90, 65, Fpolar, 'P', 20, Ppol],
                  [50, 20, Fband, 'B', 60, Pband],
                  [-20, -50, Fband, 'B', 135, Pband],
                  [-65, -90, Fpolar, 'P', 0, Ppol]]
        
        ### Build metadata list
        metadata = {}
        metadata['modu_config'] = modu_config
        metadata['modelname'], metadata['inclin'] = modelname, inclin
        metadata['Fband'], metadata['Fambient'] = Fband, Fambient
        metadata['Pband'], metadata['Ppol'] = Pband, Ppol
        metadata['config_columns'] = ['lat2', 'lat1', 'amp', 'typ', 'phase', 'period']
        metadata['config'] = config
        
        # =====================================================================
        # Create Rectangular Map + 3D Mesh
        # =====================================================================
        ### Edit the property of bands here!
        def atmos_mesh(x, config, t=0, spec=False):
            """
            Constructs full planetary map with the respective features in each band.

            Parameters:
                x (array): pixel array representing blank planetary map
                config (array): array of configuration parameters for each band and its respective features
                t (float): current time step
                spec (bool): 
            
            Returns:
                if spec = TRUE - (im, sm) (tuple of arrays): planetary and spectral maps with new features in respective bands
                if spec = FALSE - im (array): planetary map with new features in respective
            """
            im = np.full(x.shape, Fambient, dtype=np.float32)  # Vectorized initialization
            sm = np.full(x.shape, speckey['A'], dtype=np.float32)  # Vectorized initialization
            
            # Create latitude grid (vectorized)
            yy = np.arange(ysize)  # Shape: (ysize,)
            
            for group in config:
                lat2, lat1, amp, typ, phase, period = group
                lat_px1 = lat(lat1)  # Scalar
                lat_px2 = lat(lat2)  # Scalar
                
                # Vectorized mask (shape: (ysize,))
                mask = (yy >= lat_px2) & (yy <= lat_px1)
                
                if typ in ['B', 'b']:
                    # Vectorized planetary wave (x-axis: xsize)
                    xx_grid = np.arange(xsize)[:, np.newaxis]  # Shape: (xsize, 1)
                    wave = plWave(x=xx_grid, value=0.1*amp, f=phase, t=t, base=0, RP=period)
                    im[:, mask] = amp + wave  # Assign to entire band
                    
                    if spec:
                        sm[:, mask] = speckey[typ]
                    
                    if modu_config == 'noPolar':
                        polar_mod = 0.35 * polar_change(value=0, t=t, f=0 if lat2 < 0 else 30, RP=Ppol)
                        im[:, mask] += polar_mod
                
                elif typ in ['P', 'p']:
                    im[:, mask] = amp if modu_config == 'noPolar' else amp + polar_change(value=0.01*amp, t=t, f=phase, RP=period)
                    if spec:
                        sm[:, mask] = speckey[typ]
    
            # Vectorize vortices (see Step 2)
            if modu_config in ['polarDynamic', 'polarStatic']:
                polar_lats = [[g[0], g[1]] for g in config if g[3] in ['P', 'p']]
                # im = circle_vortice_(im, polar_lats, t=t, rotation_period=period)
                im = circle_vortice_vectorized(im, polar_lats, t=t, rotation_period=period)
            
            return (im, sm) if spec else im
            
        # =====================================================================
        ###################### GENERATE SPECTRAL LOOKUP MAPS ##################
        # =====================================================================

        ### Set up pyvista zoom function
        def configure_pyvista_plotter(inclin=0, zoom_factor=1.01):
            plotter = pv.Plotter(off_screen=True,
                                window_size=(300, 300),  # Explicit size helps consistency
                                lighting='none')
            plotter.background_color = 'black'
            plotter.camera.SetParallelProjection(True)
            
             # Adjust camera elevation for inclination (polar viewing angle)
            plotter.camera.elevation = inclin+56 # Critical fix, cancel out default elevation
            
            # # Equivalent to Mayavi's zoom(xx) for parallel projection
            # plotter.camera.parallel_scale = 1.0 / zoom_factor  # Inverse relationship
            
            return plotter

        ### spectral lookup map; Run once only
        ### SPECTRA WINDOW
        def generate_spectral_map_pv(x, y, z, config, inclin, metadata):
            plotter = configure_pyvista_plotter(inclin=inclin)
            
            # Create structured grid and add to plotter
            grid = pv.StructuredGrid(x, y, z)
            _, s = atmos_mesh(x, config, spec=True)
            grid.point_data['scalars'] = s.ravel(order='F')
            
            # Set camera position and render
            plotter.add_mesh(grid, scalars='scalars', cmap='plasma', clim=[0, 1], show_scalar_bar=False)
            plotter.camera_set = True  # Lock camera after initial setup
            plotter.render()
            
            # Capture and process image
            specmap = plotter.screenshot()
            specgray = specmap[:,:,0] * 0.2989 + specmap[:,:,1] * 0.587 + specmap[:,:,2] * 0.114
            
            # Region classification
            con_amb = [50, 75]
            is_amb = ((specgray >= con_amb[0]) & (specgray < con_amb[1])).astype(int)
            con_band = [120, 150]
            is_band = ((specgray >= con_band[0]) & (specgray < con_band[1])).astype(int)
            con_pol = [151, 175]
            is_pol = ((specgray >= con_pol[0]) & (specgray < con_pol[1])).astype(int)
            total_count = is_amb.sum() + is_band.sum() + is_pol.sum()
            
            # Update metadata
            metadata.update({
                'specmap': specgray,
                'speckey': speckey,
                'cond_is_amb': con_amb,
                'cond_is_band': con_band,
                'cond_is_pol': con_pol,
                'total_count': total_count,
                'is_amb': is_amb,
                'is_band': is_band, 
                'is_pol': is_pol
            })
            
            plotter.close()
            return metadata

        # =====================================================================
        ########################## CREATE PHOTOMETRY ##########################
        # =====================================================================
        def generate_photometry_pv(x, y, z, config, inclin, time_array):
            plotter = configure_pyvista_plotter(inclin=inclin)
            grid = pv.StructuredGrid(x, y, z)
            
            fluxarray = []
            gray_array = []
            limb_dark_mask = None
            
            progress = tqdm(enumerate(time_array), total=len(time_array), desc=f'Rendering i={inclin}')

            for frame_idx, ti in progress:
                # Generate atmospheric map and update mesh
                s = atmos_mesh(x, config, t=ti)
                grid.point_data['scalars'] = s.ravel(order='F')
                
                # Render and capture frame
                imgray = render_frame_pv(plotter, grid, s)
                
                # Compute limb darkening mask from FIRST RENDERED FRAME
                if frame_idx == 0:
                    # Use ACTUAL RENDERED IMAGE (square screenshot)
                    limb_dark_mask = limb_darkening(imgray, u_coefficient=0.4)
                    
                    # Optional: Save mask for inspection
                    # plt.imsave(f'limb_mask_i{inclin}.png', limb_dark_mask, cmap='gray')
                
                # Apply mask to all frames including the first
                if limb_dark_mask is not None:
                    imgray = imgray * limb_dark_mask  # Element-wise multiplication
                    
                # Store processed frame
                gray_array.append(imgray)
                fluxarray.append(s.copy())  # Preserve original atmospheric data
                
                # Matplotlib animation handling

                # # Progress reporting
                # if (frame_idx + 1) % 10 == 0:
                #     print(f'Frame {frame_idx+1}/{len(time_array)} processed')

            plotter.close()
            return {
                'fluxarray': fluxarray,
                'gray_array': gray_array,
                'limb_mask': limb_dark_mask  # Optional return
            }

        # =====================================================================
        ################# RENDERS, CAPTURE FRAMES AND CONVERT GRAYSCALE #######
        # =====================================================================

        def render_frame_pv(plotter, grid, scalars):
            plotter.clear()
            plotter.add_mesh(grid, scalars=scalars.ravel(order='F'), 
                            cmap='inferno', clim=[0, 1], show_scalar_bar=False)
            plotter.render()
            imgmap = plotter.screenshot()
            return imgmap[:,:,0] * 0.2989 + imgmap[:,:,1] * 0.587 + imgmap[:,:,2] * 0.114
        
        # =====================================================================
        # ############### Generate Photometry and Configure PyVista Plotter ###    
        # =====================================================================
        # Generate spectral map with metadata
        
        metadata = generate_spectral_map_pv(x, y, z, config, inclin, metadata)
        
        # Generate photometry data
        time_array = np.linspace(t0, t1, frame_no)
        photometry_data = generate_photometry_pv(x, y, z, config, inclin, time_array)
        
        # Merge results
        
        gray_array = photometry_data['gray_array']

    # =========================================================================
    #     ##### OUTPUT HANDLING AND METADATA WRITER
    # ================================geo=========================================
    # 1. write output image cube file
    # 2. write output FLUX file
    # 3. write model metadata: numbers of period bands, periods, location, types, fambient, fband, fpole, 
    #########################################
    ## WRITE PICKLE OUTPUT FOR GRAY_ARRAY AND METADATA
    if save_output_array:
        folderModel = 'dataCube[%s][%s][%i][%i][%i]'%(modelname, modu_config, t0, t1, frame_no)
        folderModel_path = join(homedir, 'output', folderModel)
        ### Create new output folder if not existed
        create_folder(folderModel_path)
        grayArrayPath = join(folderModel_path, 'dataCube[%s][%s][%i][%i][%i][%i].h5'%(modu_config, modelname, inclin, t0, t1, frame_no))
        metaPath = join(folderModel_path, '[meta]dataCube[%s][%s][%i][%i][%i][%i].pkl'%(modu_config, modelname, inclin, t0, t1, frame_no))
        ### Write file
        with h5py.File(grayArrayPath, 'w') as file:
            file.create_dataset('dataset', data=gray_array, compression='gzip')
        with open(metaPath, 'wb') as file2:
            pickle.dump(metadata, file2)
    # =========================================================================
    
    ### Plot specmap
    specmap = metadata['specmap']
    speckey = metadata['speckey']
    total_count = metadata['total_count']
    is_amb = metadata['is_amb']
    is_pol = metadata['is_pol']
    is_band = metadata['is_band']
    
    #### Definition: apply colormap and save 2 frames: photometry and specmap
    def save_image_with_cmap(data, filename, cmap='inferno', dpi=300):
        """Matplotlib-equivalent save function using imageio"""
        # 1. Normalize data
        vmin, vmax = np.nanmin(data), np.nanmax(data)
        normalized = (np.clip(data, vmin, vmax) - vmin) / (vmax - vmin)
        
        # 2. Apply colormap
        cmap = plt.cm.get_cmap(cmap)
        rgba = cmap(normalized)
        
        # 3. Convert to PIL Image for DPI control
        img = Image.fromarray((rgba[..., :3] * 255).astype(np.uint8))
        
        # 4. Save with metadata (equivalent to matplotlib's dpi)
        img.save(filename, dpi=(dpi, dpi), format='PNG')

    ## Photometry map
    save_image_with_cmap(
        gray_array[0], f'{plotPath}[{modu_config}][{modelname}]_[phot_at_t0_i={inclin}]_[still].png',
        cmap='inferno', dpi=300)
    
    ## Specmap
    save_image_with_cmap(
        specmap, f'{plotPath}[{modu_config}][{modelname}]_[spectraCoverageMap]_i={inclin}.png',
        cmap='viridis', dpi=300)
    
    ### Plot and save the first photometry frame
    if inlinePlot: 
        plt.figure(dpi=300), plt.tight_layout()
        plt.imshow(gray_array[1], cmap='inferno')
        plt.title('Photometry at t0 [%s][%s][i=%i]:'%(modu_config, modelname, inclin))  
        plt.show()
        
        plt.figure(dpi=300), plt.tight_layout()
        plt.imshow(specmap, cmap='viridis')
        fracA, fracP, fracB = is_amb.sum()/total_count, is_pol.sum()/total_count, is_band.sum()/total_count
        plt.title('Spectra Area Coverage, i=%i: [A]=%.2f, [P]=%.2f, [B]=%.2f'%(inclin, fracA, fracP, fracB))
        plt.show(), plt.close()
        
print('Elapsed Time: ')
print(datetime.now() - startTime)
plt.close('all')

#  # %%%
# plt.close()
# for t in range (10):
#     plt.figure(), plt.tight_layout()
#     plt.imshow(gray_array[t*3], cmap='inferno')

# %%
