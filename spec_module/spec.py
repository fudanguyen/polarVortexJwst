# -*- coding: utf-8 -*-
"""
#### Created on Mon Aug 11 2025 
# Name: polar_vortex v2
# Author: Fuda
# Description: This code combine the spectral 
module and the spatiotemporal module, and 
provide built in analysis capability.

# Modules: 

# Workflow:
- Spectral module: 
    - Input: physical prop., cloud config
    - Initialize PICASO model
    - Output contribution function

- Spatiotemporal module:
    - Input: spatial prop., temporal prop.
    - Run photometry cube
    - Save full cube, spectral light curve.

# Usages: 

"""
# ============= Imports =============

import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd # type: ignore
import astropy.units as u # type: ignore
import matplotlib.pyplot as plt # type: ignore
import seaborn as sns # type: ignore

import os

#picaso
from picaso import justdoit as jdi  # type: ignore
from picaso import justplotit as jpi # type: ignore
#plotting
from bokeh.io import output_notebook # type: ignore

output_notebook()
from bokeh.plotting import show,figure # type: ignore

# ========================= PICASO Sonora Class ==========================

class picasoSonora:
    """
    A class to handle picaso custom cloud config and spectra calculation.
    with Sonora Bobcat specifically.
    
    Attributes
    ----------
    input_spectral : dict
        Dictionary containing picaso input parameters.
    cloud_config : dict
        Dictionary containing cloud configuration parameters.
        
    Methods
    -------
    set_environment(picaso_refdata, pysyn_cdbs)
        Set environment variables for PICASO reference data.

    initialize_picaso()
        Initialize PICASO model with input_spectral parameters.

    configure_clouds(cloud_config, add_cloud=True)
        Configure multi-layer clouds in PICASO.
    
    run_spectrum(R)
        Run spectrum calculation, do conversion of units,
        and return full dataframe with regridded results.

    compute_contribution_function(df, plot_cf=True, save_cf=False, savepath=None)
        Compute and optionally plot/save the contribution function.

    integrate_spectrum(CF, pressure_list, wave_list, plevels,
                       title=None, plotting=True)
        Integrate the contribution function up to specified pressure levels.
    
    run_with_cloud_config(cloud_config, add_cloud=True, plot_cf=True, save_cf=False, cloudTop=True)
        Full run, getting pressure-dependent spectrum for a given cloud_config.
    
    """
    
    def __init__(self, input_spectral):
        """
        Initialize Brown Dwarf spectral pipeline with fixed input_spectral.
        Cloud configurations will be passed per run.
        """
        self.input_spectral = input_spectral # dict of picaso input parameters
        self.opa, self.bd = self.initialize_picaso() # opacity and atm. model
    
    def initialize_picaso(self):
        """Initialize PICASO model with input_spectral parameters."""
        wave_range = self.input_spectral['wave_range']
        opa = jdi.opannection(wave_range=wave_range)
        bd = jdi.inputs(calculation='browndwarf')
        bd.gravity(gravity=self.input_spectral['gravity'], gravity_unit=u.Unit('m/s**2'))
        # this function create opacity grid from sonora profile onto PICASO 
        bd.phase_angle(self.input_spectral['phase_angle'])
        bd.sonora(self.input_spectral['database'], self.input_spectral['teff'])
        # initialize sonora profile from given gravity and temp.
        return opa, bd
    
    def configure_clouds(self, cloud_config, add_cloud=True):
        """Configure multi-layer clouds in PICASO."""
        p, dp = cloud_config['p'], cloud_config['dp']
        N_layers = len(p) # number of layers

        # check if cloud config is valid: all params must have the same length
        if not (N_layers == len(cloud_config['dp']) == len(cloud_config['asymetry']) == 
                len(cloud_config['scattering']) == len(cloud_config['tau'])):
            raise ValueError("Cloud config must have the same length for all parameters.")

        # g0: asymetry factor, w0: single scattering albedo, opd: optical depth
        # provide option to null out clouds
        if add_cloud:
            self.bd.clouds(g0=cloud_config['asymetry'], w0=cloud_config['scattering'], 
                           opd=cloud_config['tau'], p=cloud_config['p'], dp=cloud_config['dp'])  
        else:
            self.bd.clouds(g0=[0], w0=[0], opd=[0], p=[-1], dp=[1])

        cloud_tops = [10**(p[i] - dp[i]) for i in range(N_layers)]
        return cloud_tops
    
    def run_spectrum(self):
        """Run spectrum calculation, do conversion of units,
        and return full dataframe with regridded results."""
        df = self.bd.spectrum(self.opa, full_output=True)
        R = self.input_spectral['r_resolution']

        # Convert to Fnu
        x, y = df['wavenumber'], df['thermal'] #units of erg/cm2/s/cm
        x_um = 1e4 / x # to micron conversion
        y_flam = y * 1e-8 #per anstrom instead of per cm-1
        sp = jdi.psyn.ArraySpectrum(x_um, y_flam, waveunits='um', fluxunits='FLAM')   
        sp.convert("um") #micron
        sp.convert('Fnu') #erg/cm2/s/Hz

        df['fluxnu'] = sp.flux
        x, y = jdi.mean_regrid(1e4 / x, sp.flux, R=R) 
        df['regridy'] = y
        df['regridx'] = x
        return df, sp

    def compute_contribution_function(self, df, plot_cf=True, save_cf=False, savepath=None):
        """Compute and optionally plot/save the contribution function."""
        R = self.input_spectral['r_resolution']
        # use picaso thermal contribution function
        if plot_cf:
            fig, ax, CF = jpi.thermal_contribution(
                df['full_output'], R=R, norm=jpi.colors.LogNorm(vmin=1e7, vmax=1e11)
            )
        else:
            _, _, CF = jpi.thermal_contribution(df['full_output'], R=R, norm=None)
            plt.close('all')

        # get grid of pressures and wavelengths
        pressure_list = df['full_output']['layer']['pressure'][:-1]
        wave_list = jdi.mean_regrid(df['regridx'], df['regridx'], R=R)[0]

        if save_cf and savepath:
            np.savez(savepath, wave=wave_list, pressure=pressure_list, cf=CF)

        return CF, pressure_list, wave_list

    def integrate_spectrum(self, CF, pressure_list, wave_list, plevels, 
                       title=None, plotting=True):
        """Integrate the contribution function up to specified pressure levels."""
        # plevels taken from cloud_tops or any pressure levels you want to integrate up to
        if not isinstance(plevels, list):
            raise ValueError("plevels must be a list of pressure levels.")
        if not all(isinstance(p, (int, float)) for p in plevels):
            raise ValueError("All elements in plevels must be numeric.")
        
        # find the closest pressure levels in the pressure_list
        closest_id = [np.searchsorted(pressure_list, p) for p in plevels]
        CF_copy = np.flip(CF, axis=1)
        dlnP = np.diff(np.log(pressure_list)).mean()
        # integrate contribution function to get p-dependent spectrum
        # but integration order is from bottom to top pressure
        cumspec = np.flip(np.cumsum(np.flip(CF_copy, axis=0), axis=0) * dlnP, axis=0)
        cumspec_list = [cumspec[idx, :] for idx in closest_id]
        # get the wavelength-out median flux
        cumspec_median = [np.median(cumspec[idx, :]) for idx in closest_id]

        if plotting:
            plt.figure(figsize=(8, 6))
            for i, level in enumerate(plevels):
                plt.semilogy(wave_list, cumspec_list[i], lw=0.75, label=f'P = {level:.1e} bar')
            plt.xlabel('Wavelength (um)')
            plt.ylabel('Flux')
            plt.title(title if title else 'Cumulative Spectrum Up to Given P')
            plt.legend()

        return cumspec_list, cumspec_median

    def run_with_cloud_config(self, cloud_config, pressure_custom_level=[0.01, 0.1, 1],
                          add_cloud=True, plot_cf=False, save_cf=False, cloudTop=False):
        """Full run, getting pressure-dependent spectrum for a given cloud_config."""

        cloud_tops = self.configure_clouds(cloud_config, add_cloud=add_cloud)
        df, sp = self.run_spectrum()

        savepath = os.path.join(
            self.input_spectral['database'],
            'contribution_function',
            f"sonora_t{self.input_spectral['teff']}g{self.input_spectral['gravity']}_R{self.input_spectral['r_resolution']}_cf.npz"
        )

        CF, pressure_list, wave_list = self.compute_contribution_function(
            df, plot_cf=plot_cf, save_cf=save_cf, savepath=savepath
        )

        plevels = cloud_tops if cloudTop else pressure_custom_level
        cumspec_list, cumspec_median = self.integrate_spectrum(
            CF, pressure_list, wave_list,
            plevels=plevels,
            title=f"[Cloud Config] TEFF={self.input_spectral['teff']}K",
            plotting=False
        )
        return CF, pressure_list, wave_list, cumspec_list, cumspec_median
# ========================= End of PICASO Sonora Class ==========================

#%% ============= Main Polar Vortex v2 Code Testing =============

# Set environment variables for PICASO reference data    
picaso_refdata = "/Users/nguyendat/Documents/GitHub/picaso/reference/"
pysyn_cdbs = "/Users/nguyendat/Documents/GitHub/picaso/reference/stellar_spectra/grp/redcat/trds"

"""Set environment variables for PICASO reference data."""
os.environ['picaso_refdata'] = picaso_refdata
os.environ['PYSYN_CDBS'] = pysyn_cdbs

# Sonora bobcat database path
sonora_profile_db = "/Users/nguyendat/Documents/GitHub/picaso/data/sonora_profile/"

# physical input
physics1 = {
    'gravity': 100, # m/s^2
    'teff': 900, # K effective temp
    'phase_angle': 0, # degrees
    'wave_range': [1, 4], # microns
    'r_resolution': 2000, # spectral resolution
    'model': 'sonora_bobcat',
    'database': sonora_profile_db,
}

#%% ======= Setup 1: Same thickness cloud configs with difference in elevation =======

# Initialize physical properties for spectral model
bd_model = picasoSonora(physics1)
# Pressure levels for cumulative spectra
pressure_layers = [0.01, 0.1, 1]   

cloud_configs = {
    'config1': {
        'p': [-1, 0, 1], # this means 0.1, 1, 10 bar
        'dp': [np.log10(2)] * 3,
        'asymetry': [0.9, 0.9, 0.9],
        'scattering': [0.4, 0.4, 0.4],
        'tau': [0.1, 0.2, 0.3],
    },
    'config2': {
        'p': [-1.25, 0.25, 1.25], # this cloud deck is slightly more elevated.
        'dp': [np.log10(2)] * 3,
        'asymetry': [0.9, 0.9, 0.9],
        'scattering': [0.4, 0.4, 0.4],
        'tau': [0.1, 0.2, 0.3],
    }
}

# Loop through configs and store outputs
cumspec_results = []
for config_name, cfg in cloud_configs.items():
    print(f"\n=== Running cloud config {config_name} ===")
    results = bd_model.run_with_cloud_config(
        cfg, pressure_custom_level=pressure_layers, 
        add_cloud=True, plot_cf=True, save_cf=False, cloudTop=False)
    
    cumspec_results.append({
        "config_name": config_name,
        "CF": results[0],
        "pressure_list": results[1],
        "wave_list": results[2],
        "cumspec_list": results[3],
        "cumspec_median": results[4]
    })

# function for quick comparison of cloud config results
def plot_cloud_config_results(cumspec_results, pressure_layers, height=5):

    """
    Plot cloud configuration results in side-by-side plots.

    Parameters:
    - cumspec_results: List of dictionaries containing cloud configuration results.
    - pressure_layers: List of pressure levels used for cumulative spectra.
    - height: Height of each subplot (default is 5).
    """
    num_configs = len(cumspec_results)
    fig, axes = plt.subplots(1, num_configs, figsize=(height * num_configs, height), constrained_layout=True)

    # Determine global y-axis limits for consistent scaling
    global_ymin = float('inf')
    global_ymax = float('-inf')

    # First pass to find global y-axis limits
    for result in cumspec_results:
        for spectrum in result["cumspec_list"]:
            global_ymin = min(global_ymin, spectrum.min())
            global_ymax = max(global_ymax, spectrum.max())

    # Second pass to plot with consistent y-axis scaling
    for i, result in enumerate(cumspec_results):
        # Extract data for the current configuration
        wave_list = result["wave_list"]
        cumspec_list = result["cumspec_list"]
        config_name = result["config_name"]

        # Plot spectra as a function of pressure levels
        for j, spectrum in enumerate(cumspec_list):
            # Plot the spectrum
            axes[i].semilogy(wave_list, spectrum, lw=0.75, label=f'P = {pressure_layers[j]:.1e} bar')

        # Set plot properties
        axes[i].set_title(f"Spectra: {config_name}")
        axes[i].set_xlabel("Wavelength (µm)")
        axes[i].set_ylabel("Flux")
        axes[i].set_ylim(global_ymin, global_ymax)  # Apply consistent y-axis limits
        axes[i].legend()

    plt.show()

    # Print out cumspec_median and plevels for each cloud config
    for result in cumspec_results:
        config_name = result["config_name"]
        cumspec_median = result["cumspec_median"]

        print(f"Config: {config_name}")
        for p, median in zip(pressure_layers, cumspec_median):
            print(f"  P = {p:.1e} bar, Median Flux = {median:.3e}")

plot_cloud_config_results(cumspec_results, pressure_layers)

#%% ======= Setup 2: Thicker cloud configs starting at the same elevation =======

bd_model = []
# Initialize physical properties for spectral model
bd_model = picasoSonora(physics1)
pressure_layers = [0.01, 0.1, 1]  # Example pressure levels for cumulative spectra 

cloud_configs2 = {
    'config3': {
        'p': [-1, 0, 1], # this means 0.1, 1, 10 bar
        'dp': [np.log10(2)] * 3,
        'asymetry': [0.9, 0.9, 0.9],
        'scattering': [0.4, 0.4, 0.4],
        'tau': [0.1, 0.2, 0.3],
    },
    'config4': {
        'p': [-1, 0, 1], 
        'dp': [np.log10(4)] * 3, # this cloud deck is twice as thick.
        'asymetry': [0.9, 0.9, 0.9],
        'scattering': [0.4, 0.4, 0.4],
        'tau': [0.1, 0.2, 0.3],
    }
}

# Loop through configs and store outputs
cumspec_results2 = []
for config_name, cfg in cloud_configs2.items():
    print(f"\n=== Running cloud config {config_name} ===")
    results = bd_model.run_with_cloud_config(
        cfg, pressure_custom_level=pressure_layers, 
        add_cloud=True, plot_cf=True, save_cf=False, cloudTop=False)
    
    cumspec_results2.append({
        "config_name": config_name,
        "CF": results[0],
        "pressure_list": results[1],
        "wave_list": results[2],
        "cumspec_list": results[3],
        "cumspec_median": results[4]
    })

plot_cloud_config_results(cumspec_results2, pressure_layers)
# %%
