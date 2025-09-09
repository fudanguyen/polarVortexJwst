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
import matplotlib.patches as patches # type: ignore
import seaborn as sns # type: ignore

import os

#picaso
from picaso import justdoit as jdi  # type: ignore
from picaso import justplotit as jpi # type: ignore
#plotting
from bokeh.io import output_notebook # type: ignore

output_notebook()
from bokeh.plotting import show,figure # type: ignore
from bokeh.models import HoverTool
from bokeh.palettes import Category10

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
    
    def run_spectrum(self, plot=True):
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
        sp_regrid = np.array([x,y])

        if plot:
            show(jpi.spectrum(x, y, plot_width=500, y_axis_type='log',
                              title=f"Sonora Spectrum: Teff={self.input_spectral['teff']}K, g={self.input_spectral['gravity']}m/s², R={R}",
                              x_axis_label='Wavelength (micron)', y_axis_label='Fnu (erg/cm²/s/Hz)'),
                              wave_range=self.input_spectral['wave_range'])
        return df, sp, sp_regrid

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

    def run_with_cloud_config(self, cloud_config, pressure_custom_level=[0.01, 0.1, 1],
                          add_cloud=True, calc_cf=False, save_cf=False, cloudTop=False):
        """Full run, getting pressure-dependent spectrum for a given cloud_config."""

        cloud_tops = self.configure_clouds(cloud_config, add_cloud=add_cloud)
        df, sp, sp_regrid = self.run_spectrum(plot=False)

        savepath = os.path.join(
            self.input_spectral['database'],
            'contribution_function',
            f"sonora_t{self.input_spectral['teff']}g{self.input_spectral['gravity']}_R{self.input_spectral['r_resolution']}_cf.npz"
        )

        if calc_cf:
            CF, pressure_list, wave_list = self.compute_contribution_function(
                df, plot_cf=True, save_cf=save_cf, savepath=savepath)
        else:
            CF, pressure_list, wave_list = None, None, None

        results = {'full': df, 'sp': sp, 'sp_regrid': sp_regrid,
                   'CF': CF, 'pressure_list': pressure_list, 'wave_list': wave_list}

        return results
    
    def plot_cloud_configs(cloud_configs):
        fig, ax = plt.subplots(figsize=(6, 8))

        # Pick distinct colors from a colormap
        cmap = plt.cm.viridis
        colors = [cmap(i / len(cloud_configs)) for i in range(len(cloud_configs))]

        for (i, (name, cfg)) in enumerate(cloud_configs.items()):
            p = cfg['p'][0]
            dp = cfg['dp'][0]

            # Cloud boundaries
            p_bottom = 10**p
            p_top = 10**(p - dp)

            # Draw rectangle: x spans [i, i+1] just for separation
            rect = patches.Rectangle(
                (i, p_top),         # (x, y) lower-left corner
                0.8,                # width
                p_bottom - p_top,   # height
                facecolor=colors[i],
                alpha=0.6,
                label=name
            )
            ax.add_patch(rect)

        # Set log scale for y-axis
        ax.set_yscale("log")
        ax.invert_yaxis()  # high pressure at bottom

        ax.set_ylabel("Pressure [bar]")
        ax.set_xlabel("Cloud configuration")
        ax.set_title("Cloud Decks")

        # Use config names as xticks
        ax.set_xticks([i + 0.4 for i in range(len(cloud_configs))])
        ax.set_xticklabels(cloud_configs.keys(), rotation=45, ha="right")

        plt.tight_layout()
        plt.show()

    def plot_allspec(cumspec_results):
        fig, ax = plt.subplots(figsize=(8, 6))

        for name, result in cumspec_results.items():
            sp = result['sp_regrid']
            wavelength = sp[0]
            intensity = sp[1]

            ax.plot(wavelength, intensity, label=name)

        ax.set_xlabel("Wavelength")
        ax.set_ylabel("Intensity")
        ax.set_title("Cumulative Spectra (sp_regrid)")
        ax.legend()
        plt.tight_layout()
        plt.show()

    def plot_allspec_bokeh(cumspec_results):
        # Make a Bokeh figure
        p = figure(
            width=800, height=500,
            title="Cumulative Spectra (sp_regrid)",
            x_axis_label="Wavelength (µm)",
            y_axis_label="Intensity (Fν)",
            tools="pan,wheel_zoom,box_zoom,reset,save",
            active_drag="pan",
            active_scroll="wheel_zoom"
        )
        
        # Color palette
        colors = Category10[10] if len(cumspec_results) <= 10 else None
        
        for i, (name, result) in enumerate(cumspec_results.items()):
            sp = result['sp_regrid']
            wavelength = sp[0].flatten()
            intensity = sp[1].flatten()
            
            color = colors[i % 10] if colors else None
            p.line(wavelength, intensity, line_width=2, legend_label=name, color=color)
        
        # Add hover tooltip
        hover = HoverTool(
            tooltips=[
                ("Config", "$name"),
                ("Wavelength (µm)", "$x"),
                ("Intensity", "$y")
            ],
            mode="vline"
        )
        p.add_tools(hover)

        # Legend settings
        p.legend.click_policy = "hide"  # click to hide/show spectra
        p.legend.label_text_font_size = "8pt"

        show(p)


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
# Pressure levels for output spectra
pressure_layers = [0.01, 0.1, 1]   

# Generate 20 values between 10^-1 and 10^1 in log space,
# then take log10 so values are between -1 and 1
p_values = np.linspace(-2, 1, 5)

cloud_configs = {}

for i, p in enumerate(p_values, start=1):
    cloud_configs[f'config{i}'] = {
        'p': [p],
        'dp': [np.log10(2)],
        'asymetry': [0.9],
        'scattering': [0.4],
        'tau': [0.1],
    }

picasoSonora.plot_cloud_configs(cloud_configs)

# Loop through configs and store outputs
cumspec_results = {}
for config_name, cfg in cloud_configs.items():
    print(f"\n=== Running cloud config {config_name} ===")
    cumspec_results[config_name] = bd_model.run_with_cloud_config(
        cfg, pressure_custom_level=pressure_layers, calc_cf=False,
        add_cloud=True, save_cf=False, cloudTop=False)

#%%
picasoSonora.plot_allspec(cumspec_results)