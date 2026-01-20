"""
PICASO Multiprocessing Grid Runner
Runs PICASO atmosphere models in parallel across a parameter grid
"""

import os
import sys

# Set environment variables FIRST before any other imports
picaso_refdata = "/Users/nguyendat/Documents/GitHub/picaso/reference/"
pysyn_cdbs = "/Users/nguyendat/Documents/GitHub/picaso/reference/stellar_spectra/grp/redcat/trds"
picaso_home = '/Users/nguyendat/Documents/GitHub/picaso/'

os.environ['picaso_refdata'] = picaso_refdata
os.environ['PYSYN_CDBS'] = pysyn_cdbs

# Database paths
sonora_profile_db = '/Users/nguyendat/Documents/GitHub/picaso/data/sonora_profile/'
virga_directory = '/Users/nguyendat/Documents/GitHub/picaso/data/virga/'

# Now import everything else
import numpy as np
import pandas as pd
import pickle
from multiprocessing import Pool, cpu_count
from itertools import product
from datetime import datetime
import astropy.units as u
from picaso import justdoit as jdi
from virga import justdoit as vjdi

# ============================================================================
def configure_atm(C_to_O=0.55, mh=1.0, Teff=1200, gravity=10000, 
                  kzz=1e7, phase=0, wave_range=[1,6], eq=False, excluded_mol=None):
    """Configure brown dwarf atmosphere."""
    opa = jdi.opannection(wave_range=wave_range)
    bd = jdi.inputs(calculation='browndwarf')
    bd.phase_angle(phase)
    bd.gravity(gravity=gravity, gravity_unit=u.Unit('m/(s**2)'))
    bd.sonora(sonora_profile_db, Teff)
    
    if eq:
        bd.chemeq_visscher(C_to_O, mh)
    
    profile = bd.inputs['atmosphere']['profile']
    profile['kzz'] = np.ones(profile['pressure'].shape[0]) * kzz
    profile['kz'] = profile['kzz'].copy()
    bd.atmosphere(df=profile, exclude_mol=excluded_mol)
    
    return bd, opa, profile

# ===========================================================================
def convert_and_regrid(df, R=500):
    """Convert spectrum to different units and regrid."""
    x, y = df['wavenumber'], df['thermal']
    xmicron = 1e4/x
    flamy = y*1e-8
    sp = jdi.psyn.ArraySpectrum(xmicron, flamy, waveunits='um', fluxunits='FLAM')
    sp.convert("um")
    sp.convert('Fnu')
    
    x = sp.wave
    y = sp.flux
    df['fluxnu'] = y
    x, y = jdi.mean_regrid(x, y, R=R)
    df['regridy'] = y
    df['regridx'] = x
    
    return df

# ===========================================================================
def configure_cloud(df, profile, opacity, mh=1, mmw=2.2, fsed=1, R=300,
                    gases=['Fe', 'MgSiO3', 'Mg2SiO4', 'Al2O3']):
    """Configure clouds in atmosphere."""
    virga_available_condensates = ['ZnS', 'TiO2', 'NH3', 'Na2S', 'MnS', 
                                   'MgSiO3', 'Mg2SiO4', 'KCl', 'H2O', 
                                   'Fe', 'Cr', 'CH4', 'CaTiO3', 'Al2O3']
    
    if gases == 'recommended':
        recommended_gases = vjdi.recommend_gas(profile['pressure'], 
                                              profile['temperature'], 
                                              mh=mh, mmw=mmw, plot=False)
        gases = [g for g in recommended_gases if g in virga_available_condensates]
        print(f"Using recommended gases: {gases}")
    else:
        gases = [g for g in gases if g in virga_available_condensates]
        print(f"Using specified gases (Virga-compatible only): {gases}")
        if len(gases) == 0:
            print("WARNING: No valid condensate gases specified for Virga. No clouds will be added.")
    
    cld_out = df.virga(gases, virga_directory, fsed=fsed, mmw=mmw)
    df_out = df.spectrum(opacity, full_output=True)
    df_out = convert_and_regrid(df_out, R=R)
    
    return df_out['regridx'], df_out['regridy'], cld_out, df_out


def run_single_model(params):
    """
    Run a single PICASO model with given parameters.
    
    Parameters:
    -----------
    params : dict
        Dictionary containing all model parameters including 'output_dir'
    """
    try:
        # Unpack parameters
        atm_params = {k: params[k] for k in ['C_to_O', 'mh', 'Teff', 'gravity', 
                                               'kzz', 'phase', 'wave_range', 'eq', 'excluded_mol']}
        cloud_params = {k: params[k] for k in ['mh', 'mmw', 'fsed', 'R', 'gases']}
        
        output_dir = params['output_dir']
        cloudfree = params.get('cloudfree', False)
        
        print(f"Starting model: Teff={atm_params['Teff']}, "
              f"fsed={cloud_params['fsed']}, gravity={atm_params['gravity']}")
        
        # Configure atmosphere
        bd, opa, profile = configure_atm(**atm_params)

        # Initialize result dictionary WITHOUT any file references
        result = {
            'model_id': None,
            'params': params,
        }
        
        if cloudfree:
            # Run cloud-free spectrum
            print("Running cloud-free model ONLY")
            clf = bd.spectrum(opa, full_output=True)
            clf = convert_and_regrid(clf, R=cloud_params['R'])
            
            # Create descriptive filename for cloud-free model
            # Format: bd_[Teff]_[gravity]_[kzz]_cloudfree.nc
            filename = f"bd_{atm_params['Teff']:.0f}_{atm_params['gravity']:.0f}_{atm_params['kzz']:.1E}_cloudfree.nc"
            specname = f"spec_{atm_params['Teff']:.0f}_{atm_params['gravity']:.0f}_{atm_params['kzz']:.1E}_cloudfree.csv"
            wave, thermal = clf['regridx'], clf['regridy']

            # Add to result AFTER creating the file
            result.update({
                'model_id': filename[:-3],  # Remove .nc for model_id
                'specfile': clf,
                'wave': wave,
                'thermal': thermal
            })
        else: 
            # Run cloudy spectrum 
            print("Running cloudy model ONLY")
            
            x_cldy, y_cldy, cld_out, df_out = configure_cloud(
                bd, profile, opa, 
                mh=cloud_params['mh'],
                mmw=cloud_params['mmw'],
                fsed=cloud_params['fsed'],
                R=cloud_params['R'],
                gases=cloud_params['gases']
            )
            
            # Create descriptive filename for cloudy model
            # Format: bd_[Teff]_[gravity]_[kzz]_[fsed]_cloudy.nc
            filename = f"bd_{atm_params['Teff']:.0f}_{atm_params['gravity']:.0f}_{atm_params['kzz']:.1E}_{cloud_params['fsed']:.2f}_cloudy.nc"
            specname = f"spec_{atm_params['Teff']:.0f}_{atm_params['gravity']:.0f}_{atm_params['kzz']:.1E}_{cloud_params['fsed']:.2f}_cloudy.csv"
            virganame = f"virga_{atm_params['Teff']:.0f}_{atm_params['gravity']:.0f}_{atm_params['kzz']:.1E}_{cloud_params['fsed']:.2f}_cloudy.pkl"
            wave, thermal = x_cldy, y_cldy

            # Add to result AFTER creating the file
            result.update({
                'model_id': filename[:-3],  # Remove .nc for model_id
                'specfile': cld_out,
                'wave': x_cldy,
                'thermal': y_cldy
            })

        # output spectra
        spec_file = os.path.join(output_dir, specname)
        np.savetxt(spec_file, np.column_stack((wave, thermal)), delimiter=",", header="wavelength_um, flux_erg/cm2/s/hz", comments="")

        # output .nc model file
        model_file = os.path.join(output_dir, filename)
        jdi.output_xarray(df_out, bd, savefile=model_file)

        # bundle cloud_optics as .pkl file
        if not cloudfree:
            results_file = os.path.join(output_dir, virganame)
            with open(results_file, 'wb') as f:
                pickle.dump(cld_out, f)
        
        return result
        
    except Exception as e:
        import traceback
        error_msg = f"{str(e)}\n{traceback.format_exc()}"
        print(f"Error in model {params.get('model_id', 'unknown')}: {error_msg}")
        return {'model_id': params.get('model_id'), 'error': error_msg, 'params': params}
    
# ===========================================================================
def create_parameter_grid(
    Teff_list=[1200],
    C_to_O_list=[0.55],
    mh_list=[1.0],
    gravity_list=[10000],
    kzz_list=[1e7],
    phase_list=[0],
    wave_range_list=[[1, 6]],
    eq_list=[False],
    excluded_mol_list=[None],
    mmw_list=[2.2],
    fsed_list=[1],
    R_list=[500],
    gases_list=[['Fe', 'MgSiO3', 'Mg2SiO4', 'Al2O3', 'Na2S']],
    cloudfree=False
):
    """
    Create a grid of parameter combinations.
    
    Returns:
    --------
    list of dicts : Each dict contains one parameter combination
    """
    param_grid = []
    
    for combo in product(Teff_list, C_to_O_list, mh_list, gravity_list, kzz_list,
                        phase_list, wave_range_list, eq_list, excluded_mol_list,
                        mmw_list, fsed_list, R_list, gases_list):
        
        params = {
            'Teff': combo[0],
            'C_to_O': combo[1],
            'mh': combo[2],
            'gravity': combo[3],
            'kzz': combo[4],
            'phase': combo[5],
            'wave_range': combo[6],
            'eq': combo[7],
            'excluded_mol': combo[8],
            'mmw': combo[9],
            'fsed': combo[10],
            'R': combo[11],
            'gases': combo[12],
            'cloudfree': cloudfree
        }
        param_grid.append(params)
    
    return param_grid

def run_parallel_grid(parameter_grid, runname='picaso_run', n_processes=None):
    """
    Run PICASO models in parallel across parameter grid.
    
    Parameters:
    -----------
    parameter_grid : list of dicts
        Parameter combinations to run
    runname : str
        Base name for the run directory
    n_processes : int, optional
        Number of parallel processes (default: CPU count - 1)
    
    Returns:
    --------
    results : list
        List of result dictionaries
    output_dir : str
        Path to output directory
    """
    time0 = datetime.now()
    if n_processes is None:
        n_processes = max(1, cpu_count() - 1)
    
    # Create output directory with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = f"{runname}_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Output directory: {output_dir}")
    print(f"Running {len(parameter_grid)} models on {n_processes} processes...")
    print(f"CPU count: {cpu_count()}")
    
    # Add output_dir to each parameter set
    for params in parameter_grid:
        params['output_dir'] = output_dir
    
    # Run models in parallel
    print("Starting parallel pool...")
    with Pool(n_processes) as pool:
        results = pool.map(run_single_model, parameter_grid)
    
    print("All models completed, writing config file...")
    
    # Write configuration file
    config_file = os.path.join(output_dir, 'config.txt')
    write_config_file(config_file, results)
    
    # Save results summary as pickle
    results_file = os.path.join(output_dir, 'results_summary.pkl')
    with open(results_file, 'wb') as f:
        pickle.dump(results, f)

    print(f"\nResults saved to {output_dir}")
    print(f"Configuration saved to {config_file}")
    
    # Print summary
    successful = sum(1 for r in results if 'error' not in r)
    print(f"Completed: {successful}/{len(results)} models successful")
    stop = datetime.now()
    print(f"Total time: {stop - time0}")

    return results, output_dir

def write_config_file(filename, results):
    """
    Write configuration file with model parameters.
    
    Parameters:
    -----------
    filename : str
        Path to config.txt file
    results : list
        List of result dictionaries
    """
    with open(filename, 'w') as f:
        for result in results:
            model_id = result['model_id']
            params = result['params']
            
            # Write model ID
            f.write(f"model{model_id}:\n")
            
            # Write parameters
            f.write(f"  Atmosphere:\n")
            f.write(f"    Teff       : {params['Teff']} K\n")
            f.write(f"    C_to_O     : {params['C_to_O']}\n")
            f.write(f"    mh         : {params['mh']}\n")
            f.write(f"    gravity    : {params['gravity']} m/s^2\n")
            f.write(f"    kzz        : {params['kzz']} cm^2/s\n")
            f.write(f"    phase      : {params['phase']} rad\n")
            f.write(f"    wave_range : {params['wave_range']} micron\n")
            f.write(f"    eq         : {params['eq']}\n")
            f.write(f"    excluded_mol: {params['excluded_mol']}\n")
            
            f.write(f"  Clouds:\n")
            f.write(f"    fsed       : {params['fsed']}\n")
            f.write(f"    mmw        : {params['mmw']}\n")
            f.write(f"    gases      : {params['gases']}\n")
            f.write(f"    R          : {params['R']}\n")
            
            # Write output files
            f.write(f"  Output:\n")
            if 'cloudfree_file' in result:
                f.write(f"    cloudfree  : {os.path.basename(result['cloudfree_file'])}\n")
            if 'cloudy_file' in result:
                f.write(f"    cloudy     : {os.path.basename(result['cloudy_file'])}\n")
            if 'error' in result:
                f.write(f"    ERROR      : {result['error']}\n")
            
            f.write("\n" + "-"*100 + "\n\n")
    
    print(f"Configuration written to {filename}")

# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    
    print("Setting up parameter grid...")
    
    ### Define parameter grid # HAS TO BE LIST
    ### Different cloud thickness, all condensates
    # param_grid = create_parameter_grid(
    #     Teff_list=[1200],
    #     C_to_O_list=[0.55],
    #     wave_range_list=[[1.5, 5]],
    #     R_list=[700],
    #     gravity_list=[10000],
    #     fsed_list=[1., 1.02, 1.04, 1.06, 1.08, 1.1 , 1.12, 1.14, 1.16, 1.18],
    #     excluded_mol_list=[None],
    #     gases_list=[['Fe', 'Al2O3', 'MgSiO3', 'Mg2SiO4', 'Na2S']],
    #     kzz_list=[1e7],
    #     cloudfree=False  # Set to True for cloud-free models only
    # )

    ### Alternate grid playing with clouds condensates option
    ### No SiO3
    # param_grid = create_parameter_grid(
    #     Teff_list=[1200],
    #     C_to_O_list=[0.55],
    #     wave_range_list=[[1.5, 5]],
    #     R_list=[700],
    #     gravity_list=[10000],
    #     fsed_list=[1., 1.02, 1.04, 1.06, 1.08, 1.1 , 1.12, 1.14, 1.16, 1.18],
    #     excluded_mol_list=[None],
    #     gases_list=[['Fe', 'Al2O3', 'Mg2SiO4', 'Na2S']],  # Exclude Mg2SiO4
    #     kzz_list=[1e7],
    #     cloudfree=False  # Set to True for cloud-free models only
    # )

    ### No Na2S
    # param_grid = create_parameter_grid(
    #     Teff_list=[1200],
    #     C_to_O_list=[0.55],
    #     wave_range_list=[[1.5, 5]],
    #     R_list=[700],
    #     gravity_list=[10000],
    #     fsed_list=[1., 1.02, 1.04, 1.06, 1.08, 1.1 , 1.12, 1.14, 1.16, 1.18],
    #     excluded_mol_list=[None],
    #     gases_list=[['Fe', 'Al2O3', 'MgSiO3', 'Mg2SiO4']],  # Exclude Mg2SiO4
    #     kzz_list=[1e7],
    #     cloudfree=False  # Set to True for cloud-free models only
    # )
    
    ### Simple 3 clouds
    fsed_list = sorted(set([1., 1.02, 1.04, 1.06, 1.08, 1.1, 1.12, 1.14, 1.16, 1.18] + 
                           [1.01, 1.03, 1.05, 1.07, 1.09, 1.11, 1.13, 1.15, 1.17, 1.19]))

    param_grid = create_parameter_grid(
        Teff_list=[1200],
        C_to_O_list=[0.],
        wave_range_list=[[1, 6]],
        R_list=[700],
        gravity_list=[10000],
        fsed_list=fsed_list,
        excluded_mol_list=['CH4'],
        gases_list=[['Fe', 'MgSiO3', 'Na2S']],
        kzz_list=[1e7],
        cloudfree=False  # Set to True for cloud-free models only
    )

    print(f"Total models to run: {len(param_grid)}")
    
    # Run the grid
    results, output_dir = run_parallel_grid(
        param_grid, 
        runname='bd_grid_noCH4',  # Creates directory like "bd_grid_20241016_143022"
        n_processes=4  # Adjust based on your CPU
    )
    
    # Example: Load a specific model later
    print("\n" + "="*60)
    print("Example - Loading a saved model:")
    print("="*60)
    
#%% Loading results 

    import matplotlib.pyplot as plt
    import glob
    from virga import justplotit as vjpi
    from picaso import justplotit as jpi
    import xarray 

    ## Plot all spectra from results
    plt.figure(figsize=(12, 6), dpi=300)
    for model in results:
        model_id = model['model_id']
        x = model['wave']
        y = model['thermal']
        plt.plot(x, y, label=model_id)

    plt.xlabel("Wavelength (µm)")
    plt.ylabel("Flux (erg/cm²/s/Hz)")
    plt.title("Spectra from Results")
    # plt.yscale('log')
    plt.legend(fontsize='small')
    plt.grid()
    plt.tight_layout()
    plt.show()

    ## Plot virga cloud optics
    from bokeh.plotting import show, figure
    from bokeh.io import output_notebook 
    output_notebook()

    for model in results[:2]:  # Plot first 2 models only
        virgadf = model['specfile']
        print(f"\nPlotting model: {model['model_id']}, fsed={model['params']['fsed']}")
        p = show(vjpi.all_optics(virgadf))

    ## Load a specific model file, reload and reuse
    model_file = glob.glob(os.path.join(output_dir, '*.nc'))[0]
    ds = xarray.load_dataset(model_file)
    print(f"Dataset dimensions: {ds.dims}")
    print(f"Available variables: {list(ds.data_vars)}")
    opa_reload = jdi.opannection(wave_range=[1, 10])
    reuse = jdi.input_xarray(ds, opa_reload, calculation='browndwarf')
    ## run gravity
    gravity_value = os.path.basename(model_file).split('_')[2]
    reuse.gravity(gravity=float(gravity_value), gravity_unit=u.Unit('m/(s**2)'))
    print("Model successfully reloaded for new calculations!")
    sp = reuse.spectrum(opa_reload)
    sp = convert_and_regrid(sp, R=500)

    p = show(jpi.spectrum(1e4/sp['regridx'], sp['regridy'],
        plot_height=400, plot_width=700, title="Reloaded Spectrum: "+model_file))
# %%
