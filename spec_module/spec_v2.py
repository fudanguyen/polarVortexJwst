"""
PICASO Multiprocessing Grid Runner
Runs PICASO atmosphere models in parallel across a parameter grid

UPDATED (merged from sio2_picaso_virga_aug16.py, Aug 16 base code):
  - Sonora Diamondback PT-profile support (background='diamondback')
  - SiO2 / SiO2_amorph VIRGA condensates (monkey-patched gas_properties/pvaps)
  - Corrected Fnu unit conversion via convert_unit() (pysynphot-based),
    replacing the old direct jdi.mean_regrid(wavenumber, thermal) call.
    NOTE: output spectrum unit changed from Flam (erg/cm2/s/cm) to
    Fnu (erg/cm2/s/Hz). See "BREAKING CHANGES" note near convert_unit().
"""

import os
import sys
import re
import math
from io import StringIO

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Set environment variables FIRST before any other imports
picaso_refdata = "/Users/nguyendat/Documents/GitHub/picaso/reference/"
pysyn_cdbs = "/Users/nguyendat/Documents/GitHub/picaso/reference/stellar_spectra/grp/redcat/trds"
picaso_home = '/Users/nguyendat/Documents/GitHub/picaso/'

os.environ['picaso_refdata'] = picaso_refdata
os.environ['PYSYN_CDBS'] = pysyn_cdbs

# Database paths
sonora_profile_db = '/Users/nguyendat/Documents/GitHub/picaso/data/sonora_profile/'
# NEW: Sonora Diamondback climate-converged PT profile grid (Morley et al. 2024)
sonora_diamondback_profile_db = '/Users/nguyendat/Documents/GitHub/picaso/data/sonora_diamondback_profile/'
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
# NEW: needed to monkey-patch in SiO2_amorph support
import virga.gas_properties as gas_properties
import virga.pvaps as pvaps

# ============================================================================
# NEW: SiO2_amorph condensate registration
# ============================================================================
# The default VIRGA 'SiO2' entry is SiO2_alpha (crystalline quartz); its .mieff
# is identical to 'SiO2_alpha'. This block adds 'SiO2_amorph' (amorphous silica,
# Moran+2024) as a separate condensate: same saturation vapor pressure curve as
# SiO2, different density. Runs once at import time.
optical_prop_dir = virga_directory
if not os.path.exists(os.path.join(optical_prop_dir, 'SiO2_amorph.mieff')):
    print("SiO2_amorph.mieff file not found. Calculating...")
    qext, qscat, g_qscat, radii, wave = vjdi.calc_mie_db(
        ['SiO2_amorph'],
        optical_prop_dir,
        optical_prop_dir
    )

def SiO2_amorph_virga(mw_atmos, mh=1, gas_mmr=None):
    """Amorphous silica: same vapor budget/condensation curve as SiO2
    (Virga treats all SiO2 polymorphs as sharing one saturation vapor
    pressure curve, per Moran+2024 Sec 3.1), but with amorphous density."""
    if gas_mmr is None:
        gas_mmr = 60.3e-6 * mh
    gas_mw = 60
    gas_mmr = gas_mmr * (gas_mw / mw_atmos)
    rho_p = 2.20  # Moran+2024 Table 1
    return gas_mw, gas_mmr, rho_p

def SiO2_amorph(temp, mh=1):
    """Saturation vapor pressure for amorphous SiO2.

    Parameters
    ----------
    temp : float or ndarray
        Temperature (K)
    mh : float
        Metallicity relative to solar (1 = 1X solar).

    Returns
    -------
    float or ndarray
        Vapor pressure in dyn/cm^2.
    """
    mh = np.log10(mh)
    pvap_sio2 = 10.0**(13.168 - 28265.0 / temp - mh)
    return 1e6 * pvap_sio2  # bar -> dyn/cm^2

# monkey-patch so Atmosphere()/compute() can find SiO2_amorph by name
gas_properties.SiO2_amorph = SiO2_amorph_virga
pvaps.SiO2_amorph = SiO2_amorph

# ============================================================================
# NEW: Sonora Diamondback PT-profile reader
# ============================================================================
def read_sonora_diamondback(bd, sonora_diamondback_path, teff, fsed=1, mh=0.0, co=1.0,
                             drop_cols=('n', 'e-', 'H+', 'H-')):
    """
    Loads a climate-converged Sonora Diamondback P-T + chemistry profile
    (Morley et al. 2024) directly into a PICASO `inputs` bundle, as a
    drop-in analogue to `bd.sonora()` -- but for the cloudy, RC-converged
    Diamondback grid instead of the cloud-free Bobcat grid.

    File naming convention: t{Teff}g{gravity, m/s2}f{fsed}_m{[M/H]}_co{C/O, xsolar}
    e.g. t900g31f1_m-0.5_co1.0

    File structure (confirmed from an actual .pt file):
      line 1: whitespace-delimited column header
              n P T e- H2 H H+ H- VO TiO CO2 He H2O CH4 CO NH3 N2 PH3 H2S Fe Na K
      line 2: a single scalar -- the gravity in m/s2, redundant with the filename.
              NOT a data row, but has the same number of leading-whitespace-stripped
              tokens as a short/blank line would, so a naive header=0 read silently
              misaligns every subsequent row by one and NaN-pads the rest of that row.
      lines 3+: data, P in bar, T in K, rest are volume mixing ratios.

    This function detects and skips ANY line whose token count doesn't match the
    header's column count (not just line 2 specifically), so it's robust if a
    given file has extra metadata/footer lines elsewhere too.

    Selection logic: nearest available fsed -> nearest available [M/H] -> nearest
    available C/O narrow the candidate files down to one (Teff, gravity) sub-grid,
    then nearest-neighbor in (Teff, gravity) picks the final file. Gravity is
    pulled from whatever `bd.gravity(...)` was already set to (must be called first).

    Parameters
    ----------
    bd : picaso.justdoit.inputs
        Bundle object; bd.gravity(...) must already have been called.
    sonora_diamondback_path : str
        Path to the untarred pressure-temperature_profiles directory.
    teff : float
        Target Teff (K). Grid: 900-2400 K in 100 K steps.
    fsed : float
        Target *background* cloud sedimentation efficiency used to anchor the
        climate-converged P-T structure (grid: 1, 2, 3, 4, 8).
    mh : float
        Target [M/H] (grid: -0.5, 0.0, 0.5).
    co : float
        Target C/O relative to solar (grid currently only has 1.0 = solar).
    drop_cols : tuple of str
        Columns to drop from the Diamondback file before handing off to
        PICASO (level index + charged/ionic species PICASO doesn't carry
        opacities for).

    Returns
    -------
    bd : picaso.justdoit.inputs
        Same bundle, mutated in place (profile, nlevel, sonora_filename set).
    profile : pandas.DataFrame
        The loaded pressure/temperature/chemistry profile.
    """
    try:
        g_mks = bd.inputs['planet']['gravity'] / 100  # cgs (cm/s2) -> m/s2
    except (AttributeError, KeyError):
        raise Exception("Gravity has not been set. Please run bd.gravity(...) first.")

    flist = os.listdir(sonora_diamondback_path)
    flist = [f for f in flist if not f.startswith('._')]

    pattern = re.compile(r'^t(\d+)g(\d+)f(\d+)_m([+-]?\d+(?:\.\d+)?)_co(\d+(?:\.\d+)?)')

    parsed = []
    for f in flist:
        m = pattern.match(f)
        if m:
            t, g, fs, m_h, c_o = m.groups()
            parsed.append({'file': f, 'teff': float(t), 'grav': float(g),
                            'fsed': float(fs), 'mh': float(m_h), 'co': float(c_o)})

    if not parsed:
        raise Exception(f"No Sonora Diamondback profile files could be parsed in "
                         f"{sonora_diamondback_path}. Expected filenames like "
                         f"'t900g31f1_m-0.5_co1.0'. Check the directory contents.")

    grid = pd.DataFrame(parsed)

    for target, col in [(fsed, 'fsed'), (mh, 'mh'), (co, 'co')]:
        avail = grid[col].unique()
        nearest_val = avail[np.argmin(np.abs(avail - target))]
        grid = grid[np.isclose(grid[col], nearest_val)]

    dists = grid.apply(lambda r: math.hypot(r['teff'] - teff, r['grav'] - g_mks), axis=1)
    best = grid.loc[dists.idxmin()]

    filepath = os.path.join(sonora_diamondback_path, best['file'])

    with open(filepath, 'r') as fh:
        raw_lines = fh.readlines()

    header_tokens = raw_lines[0].split()
    ncols = len(header_tokens)

    data_lines, skipped_meta = [], []
    for ln in raw_lines[1:]:
        if not ln.strip():
            continue
        if len(ln.split()) == ncols:
            data_lines.append(ln)
        else:
            skipped_meta.append(ln.strip())

    if skipped_meta:
        print(f"NOTE: skipped {len(skipped_meta)} non-data line(s) in {best['file']} "
              f"whose token count didn't match the {ncols}-column header: {skipped_meta}")
        for s in skipped_meta:
            try:
                sval = float(s)
                if not np.isclose(sval, best['grav'], rtol=0.05):
                    print(f"  WARNING: skipped value {sval} doesn't closely match "
                          f"the filename-implied gravity ({best['grav']} m/s2) -- "
                          f"double check this isn't real data being dropped.")
            except ValueError:
                pass

    # Fix Fortran-style 'D' exponents (e.g. 1.23D-05 -> 1.23E-05) as a safety net
    raw_text_fixed = re.sub(r'(\d)[dD]([+-]?\d)', r'\1E\2', ''.join(data_lines))
    prof = pd.read_csv(StringIO(raw_text_fixed), sep=r'\s+', header=None, names=header_tokens)
    prof = prof.rename(columns={'P': 'pressure', 'T': 'temperature'})
    prof = prof.drop(columns=[c for c in drop_cols if c in prof.columns])

    for c in prof.columns:
        prof[c] = pd.to_numeric(prof[c], errors='coerce')

    n_nan = int(prof.isna().sum().sum())
    if n_nan > 0:
        bad_cols = prof.columns[prof.isna().any()].tolist()
        print(f"WARNING: {n_nan} NaN value(s) remain after parsing {best['file']}, "
              f"in column(s): {bad_cols}. Inspect the raw file before proceeding.")

    if not (prof['pressure'].is_monotonic_increasing or prof['pressure'].is_monotonic_decreasing):
        print(f"WARNING: pressure column in {best['file']} is not monotonic -- "
              f"check the file wasn't truncated or mis-delimited.")

    bd.inputs['atmosphere']['profile'] = prof
    bd.nlevel = prof.shape[0]
    bd.inputs['atmosphere']['sonora_filename'] = best['file']

    print(f"Loaded Sonora Diamondback profile: {best['file']} "
          f"(Teff={best['teff']:.0f} K, g={best['grav']:.0f} m/s2, "
          f"fsed={best['fsed']:.0f}, [M/H]={best['mh']:+.1f}, C/O={best['co']:.2f}x solar), "
          f"{prof.shape[0]} levels, P range [{prof['pressure'].min():.2e}, {prof['pressure'].max():.2e}] bar")

    return bd, prof

# ============================================================================
# NEW: correct Flam -> Fnu unit conversion + regrid
# ============================================================================
# BREAKING CHANGE: this replaces the old pattern of calling
#   jdi.mean_regrid(wavenumber, thermal, R=R)   # thermal in erg/cm2/s/cm (Flam-like)
# directly on PICASO's raw output. That old pattern is the source of the
# "precise factor-of-2 truncation" bug (mean_regrid expects wavelength in
# um, not wavenumber in cm^-1, for some internal binning paths -- passing
# cm^-1 silently truncates the range). convert_unit() instead:
#   1. converts PICASO's raw Flam-like output to true Fnu via pysynphot
#      (f_nu = f_lambda * lambda^2 / c), which is unit-safe regardless of
#      how mean_regrid is later called, and
#   2. regrids in wavenumber space (ascending, explicitly sorted) at R.
# Downstream spectra are now Fnu (erg/cm2/s/Hz), NOT Flam (erg/cm2/s/cm).
def convert_unit(df, R=500):
    x, y = df['wavenumber'], df['thermal']  # wavenumber (cm^-1), erg/cm2/s/cm

    xmicron = 1e4 / x       # wavelength in microns, for pysynphot
    flamy = y * 1e-8        # per Angstrom instead of per cm

    sp = jdi.psyn.ArraySpectrum(xmicron, flamy, waveunits='um', fluxunits='FLAM')
    sp.convert('um')
    sp.convert('Fnu')       # erg/cm2/s/Hz

    wave_um = np.asarray(sp.wave)    # microns -- pysynphot may resort these, don't
    flux_nu = np.asarray(sp.flux)    # assume they're still in the original row order

    wavenum = 1e4 / wave_um          # back to wavenumber (cm^-1), paired correctly with flux_nu

    # sort explicitly by wavenumber ascending -- guarantees x/y stay paired and
    # gives mean_regrid a properly monotonic array regardless of pysynphot's internal order
    order = np.argsort(wavenum)
    wavenum_sorted = wavenum[order]
    flux_nu_sorted = flux_nu[order]

    xg, yg = jdi.mean_regrid(wavenum_sorted, flux_nu_sorted, R=R)  # wavenumber (cm^-1), erg/cm2/s/Hz

    # stash results in extra keys rather than overwriting core columns -- xg/yg are a
    # *different length* than df (mean_regrid downsamples), so they can never safely
    # be real columns
    df['fluxnu_wavenumber'] = wavenum_sorted
    df['fluxnu'] = flux_nu_sorted
    df['regridx'] = xg
    df['regridy'] = yg

    return df  # unchanged core columns ('wavenumber', 'thermal', etc.) -- anything else
               # indexing out['wavenumber'] keeps working as before

# ============================================================================
def configure_atm(C_to_O=0.55, mh=1.0, Teff=1400, gravity=10000,
                  kzz=1e7, phase=0, wave_range=[1,6], eq=False, excluded_mol=None,
                  background='bobcat', diamondback_fsed=1):
    """Configure brown dwarf atmosphere.

    NEW params (default to old bobcat-only behavior, so existing calls are
    unaffected unless you opt in):
    background : 'bobcat' (default, unchanged behavior) or 'diamondback'.
        'diamondback' loads a climate-converged (RC-equilibrium) Sonora
        Diamondback P-T + chemistry structure anchored at `diamondback_fsed`
        -- use this if you plan to sweep VIRGA's own cloud fsed over a
        narrow range around that anchor, so the background thermal
        structure is self-consistent with the cloud regime being modeled.
    diamondback_fsed : float, only used if background='diamondback'. Which
        converged Diamondback cloud structure to anchor to (grid: 1, 2, 3, 4, 8).
        NOTE: this is a *different knob* from the VIRGA cloud fsed used in
        configure_cloud() -- don't conflate the two when building the grid.
    """
    opa = jdi.opannection(wave_range=wave_range)
    bd = jdi.inputs(calculation='browndwarf')
    bd.phase_angle(phase)
    bd.gravity(gravity=gravity, gravity_unit=u.Unit('cm/(s**2)'))  # FIX: gravity is passed as cgs (cm/s2)
                                                                     # everywhere in this codebase (see e.g.
                                                                     # g_cgs_list below); tagging it m/s2 here
                                                                     # silently inflated internal gravity by 100x
                                                                     # and broke Diamondback g-matching downstream.

    if background == 'diamondback':
        # Diamondback filenames encode [M/H] in dex (log10 relative to solar), but this
        # function's `mh` argument follows PICASO's linear-ratio-to-solar convention
        # (mh=1.0 -> solar), same as C_to_O. Convert before matching against the grid.
        if mh <= 0:
            raise ValueError("mh must be > 0 (it's a linear ratio to solar; log10(mh) "
                            "is taken to match Diamondback's dex-based [M/H] grid).")
        bd, profile = read_sonora_diamondback(bd, sonora_diamondback_profile_db, Teff,
                                           fsed=diamondback_fsed, mh=np.log10(mh), co=C_to_O)
    elif background == 'bobcat':
        bd.sonora(sonora_profile_db, Teff)
        if eq:
            bd.chemeq_visscher(C_to_O, mh)
        profile = bd.inputs['atmosphere']['profile']
    else:
        raise ValueError("background must be 'diamondback' or 'bobcat'")

    profile['kzz'] = np.ones(profile['pressure'].shape[0]) * kzz
    profile['kz'] = profile['kzz'].copy()
    bd.atmosphere(df=profile, exclude_mol=excluded_mol)

    return bd, opa, profile

# ===========================================================================
def configure_cloud(df, profile, opacity, mh=1, mmw=2.2, fsed=1, R=300,
                    gases=['Fe', 'MgSiO3', 'Mg2SiO4', 'Al2O3'],
                    directory=virga_directory):
    """Configure clouds in atmosphere.

    CHANGED:
      - virga_available_condensates now includes 'SiO2' and 'SiO2_amorph'.
      - output spectrum is now converted to Fnu via convert_unit() instead of
        a raw jdi.mean_regrid(wavenumber, thermal) call -- see convert_unit()
        docstring for why. df_out now carries 'regridx'/'regridy' (Fnu,
        erg/cm2/s/Hz) in addition to the original raw columns.
    """
    virga_available_condensates = ['ZnS', 'TiO2', 'NH3', 'Na2S', 'MnS',
                                   'MgSiO3', 'Mg2SiO4', 'KCl', 'H2O',
                                   'Fe', 'Cr', 'CH4', 'CaTiO3', 'Al2O3',
                                   'SiO2', 'SiO2_amorph']  # NEW: SiO2, SiO2_amorph

    if gases == 'recommended':
        recommended_gases = vjdi.recommend_gas(profile['pressure'],
                                              profile['temperature'],
                                              mh=mh, mmw=mmw, plot=False)
        # NEW: CaAl12O19 has no mieff file in the virga database -- drop it
        recommended_gases = [g for g in recommended_gases if g != 'CaAl12O19']
        gases = [g for g in recommended_gases if g in virga_available_condensates]
        print(f"Using recommended gases: {gases}")
    else:
        gases = [g for g in gases if g in virga_available_condensates]
        print(f"Using specified gases (Virga-compatible only): {gases}")
        if len(gases) == 0:
            print("WARNING: No valid condensate gases specified for Virga. No clouds will be added.")

    cld_out = df.virga(gases, directory, fsed=fsed, mmw=mmw)
    df_out = df.spectrum(opacity, full_output=True)
    df_out = convert_unit(df_out, R=R)  # NEW: correct Flam -> Fnu conversion + regrid

    return cld_out, df_out

def _fsed_to_str(fsed):
    """Return a filename-safe string for either a float or per-species dict fsed."""
    if isinstance(fsed, dict):
        # e.g. {'Fe': 14, 'MgSiO3': 3.1, 'Na2S': 4.6}  ->  'Fe14.00-MgSiO33.10-Na2S4.60'
        return '-'.join(f"{sp}{v:.4f}" for sp, v in fsed.items())
    return f"{fsed:.4f}"

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
        # NEW: 'background' and 'diamondback_fsed' added to atm_params
        atm_params = {k: params[k] for k in ['C_to_O', 'mh', 'Teff', 'gravity',
                                               'kzz', 'phase', 'wave_range', 'eq', 'excluded_mol',
                                               'background', 'diamondback_fsed']}
        cloud_params = {k: params[k] for k in ['mh', 'mmw', 'fsed', 'R', 'gases']}

        output_dir = params['output_dir']
        cloudfree = params.get('cloudfree', False)

        print(f"Starting model: Teff={atm_params['Teff']}, "
              f"fsed={cloud_params['fsed']}, gravity={atm_params['gravity']}, "
              f"background={atm_params['background']}")

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
            clf = convert_unit(clf, R=cloud_params['R'])  # NEW: Flam -> Fnu conversion + regrid

            # Create descriptive filename for cloud-free model
            filename = f"bd_{atm_params['Teff']:.0f}_{atm_params['gravity']:.0f}_{atm_params['kzz']:.1E}_cloudfree.nc"
            specname = f"spec_{atm_params['Teff']:.0f}_{atm_params['gravity']:.0f}_{atm_params['kzz']:.3E}_cloudfree.csv"
            wavenumber_um = clf['regridx']          # NEW: from convert_unit, cm^-1
            fnu = clf['regridy']                    # NEW: Fnu, erg/cm2/s/Hz
            wavelength_um = 1e4 / wavenumber_um

            # Add to result AFTER creating the file
            result.update({
                'model_id': filename[:-3],  # Remove .nc for model_id
                'wavelength': wavelength_um,
                'thermal': fnu,  # NEW: Fnu, erg/cm2/s/Hz (was Flam, erg/cm2/s/cm)
            })
        else:
            # Run cloudy spectrum
            print("Running cloudy model ONLY")

            cld_out, df_out = configure_cloud(
                bd, profile, opa,
                mh=cloud_params['mh'],
                mmw=cloud_params['mmw'],
                fsed=cloud_params['fsed'],
                R=cloud_params['R'],
                gases=cloud_params['gases'],
                directory=virga_directory,
            )

            # Create descriptive filename for cloudy model
            _fsed_str = _fsed_to_str(cloud_params['fsed'])
            filename  = f"bd_{atm_params['Teff']:.0f}_{atm_params['gravity']:.0f}_{atm_params['kzz']:.4E}_{_fsed_str}_cloudy.nc"
            specname  = f"spec_{atm_params['Teff']:.0f}_{atm_params['gravity']:.0f}_{atm_params['kzz']:.4E}_{_fsed_str}_cloudy.csv"
            virganame = f"virga_{atm_params['Teff']:.0f}_{atm_params['gravity']:.0f}_{atm_params['kzz']:.4E}_{_fsed_str}_cloudy.pkl"
            wavenumber_um = df_out['regridx']       # NEW: from convert_unit, cm^-1
            fnu = df_out['regridy']                 # NEW: Fnu, erg/cm2/s/Hz
            wavelength_um = 1e4 / wavenumber_um

            # Add to result AFTER creating the file
            result.update({
                'model_id': filename[:-3],  # Remove .nc for model_id
                'cloud_optics': cld_out,
                'wavelength': wavelength_um,
                'thermal': fnu,  # NEW: Fnu, erg/cm2/s/Hz (was Flam, erg/cm2/s/cm)
            })

        # After the if/else block, resolve which output object to use:
        spectrum_output = clf if cloudfree else df_out

        # output spectra
        # NEW: header reflects Fnu units now (was flux_erg/cm2/s/cm)
        spec_file = os.path.join(output_dir, specname)
        np.savetxt(spec_file, np.column_stack((wavelength_um, fnu)), delimiter=",",
                   header="wavelength_um, flux_erg/cm2/s/Hz", comments="")

        # output .nc model file
        model_file = os.path.join(output_dir, filename)
        jdi.output_xarray(spectrum_output, bd, savefile=model_file)

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
def generate_fsed(
    mode='simple',
    # --- Simple mode ---
    start=1.0,
    stop=1.20,
    total=20,
    # --- Complex mode ---
    species_params=None,
    n_steps=5,
    gases=None,
):
    """
    Generate fsed parameter lists for PICASO/VIRGA models.

    Parameters
    ----------
    mode : str
        'simple'  - one shared fsed value per model (returns list of floats)
        'complex' - per-species fsed values per model (returns list of dicts)

    Simple mode
    -----------
    start : float   Starting fsed value (inclusive)
    stop  : float   Ending fsed value   (exclusive)
    total : int     Number of evenly-spaced values to generate
                    Step is computed as (stop - start) / total,
                    decimal precision is inferred from the step magnitude.
    Example
    -------
    generate_fsed('simple', start=1.0, stop=1.20, total=20)
    -> [1.0, 1.01, 1.02, ..., 1.19]   # step = 0.01, 20 values

    Complex mode
    ------------
    species_params : dict
        Maps each gas species to either:
          - a single float  -> held constant across all steps
          - a dict {'start': x, 'stop': y} -> linearly varied over n_steps
            direction is automatic (start > stop -> decreasing)
    n_steps : int
        Number of fsed combinations to generate.
    gases : list, optional
        The gases list from create_parameter_grid() - used for validation.
        Raises ValueError if there is a mismatch between the species in
        species_params and the gases list.

    Returns
    -------
    list
        Simple  -> list of floats
        Complex -> list of dicts  {species: fsed_value}
    """
    if mode == 'simple':
        if total <= 0:
            raise ValueError(f"total must be > 0, got {total}")

        step = (stop - start) / total

        if step != 0:
            magnitude = math.floor(math.log10(abs(step)))
            decimals = max(0, -magnitude + 3)
        else:
            decimals = 10

        fsed_list = [round(start + i * step, decimals) for i in range(total)]
        return fsed_list

    elif mode == 'complex':
        if species_params is None:
            raise ValueError("species_params must be provided for complex mode.")
        if n_steps <= 0:
            raise ValueError(f"n_steps must be > 0, got {n_steps}")

        if gases is not None:
            param_species = set(species_params.keys())
            grid_species  = set(gases)
            missing = param_species - grid_species
            extra   = grid_species  - param_species
            errors  = []
            if missing:
                errors.append(f"  species in species_params but NOT in gases: {sorted(missing)}")
            if extra:
                errors.append(f"  species in gases but NOT in species_params: {sorted(extra)}")
            if errors:
                raise ValueError("Gas species mismatch:\n" + "\n".join(errors))

        species_arrays = {}
        for species, cfg in species_params.items():
            if isinstance(cfg, (int, float)):
                species_arrays[species] = [float(cfg)] * n_steps
            elif isinstance(cfg, dict):
                s, e = cfg['start'], cfg['stop']
                vals = list(np.linspace(s, e, n_steps))
                species_arrays[species] = vals
            else:
                raise ValueError(
                    f"Invalid config for '{species}': expected float or "
                    f"{{'start': x, 'stop': y}}, got {type(cfg)}"
                )

        fsed_list = [
            {sp: species_arrays[sp][i] for sp in species_params}
            for i in range(n_steps)
        ]
        return fsed_list

    else:
        raise ValueError(f"mode must be 'simple' or 'complex', got '{mode!r}'")

# ===========================================================================
def create_parameter_grid(
    Teff_list=[100],
    C_to_O_list=[0.55],
    mh_list=[1.0],
    gravity_list=[10000],
    kzz_list=[1e7],
    phase_list=[0],
    wave_range_list=[[1, 10]],
    eq_list=[False],
    excluded_mol_list=[None],
    mmw_list=[2.2],
    fsed_list=[1],
    R_list=[500],
    gases_list=[['Fe', 'MgSiO3', 'Mg2SiO4', 'Al2O3', 'Na2S']],
    background_list=['bobcat'],          # NEW
    diamondback_fsed_list=[1],           # NEW
    cloudfree=False
):
    """
    Create a grid of parameter combinations.

    NEW: background_list / diamondback_fsed_list control the Sonora background
    used by configure_atm(). Defaults ('bobcat', 1) reproduce the old,
    pre-diamondback behavior exactly.

    Returns:
    --------
    list of dicts : Each dict contains one parameter combination
    """
    param_grid = []

    for combo in product(Teff_list, C_to_O_list, mh_list, gravity_list, kzz_list,
                        phase_list, wave_range_list, eq_list, excluded_mol_list,
                        mmw_list, fsed_list, R_list, gases_list,
                        background_list, diamondback_fsed_list):

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
            'background': combo[13],           # NEW
            'diamondback_fsed': combo[14],     # NEW
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

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    base_dir =  SCRIPT_DIR
    output_dir = os.path.join(base_dir, f"{runname}_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)

    print(f"Output directory: {output_dir}")
    print(f"Running {len(parameter_grid)} models on {n_processes} processes...")
    print(f"CPU count: {cpu_count()}")

    for params in parameter_grid:
        params['output_dir'] = output_dir

    print("Starting parallel pool...")
    with Pool(n_processes) as pool:
        results = pool.map(run_single_model, parameter_grid)

    print("All models completed, writing config file...")

    config_file = os.path.join(output_dir, 'config.txt')
    write_config_file(config_file, results)

    results_file = os.path.join(output_dir, 'results_summary.pkl')
    with open(results_file, 'wb') as f:
        pickle.dump(results, f)

    print(f"\nResults saved to {output_dir}")
    print(f"Configuration saved to {config_file}")

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

            f.write(f"model{model_id}:\n")

            f.write(f"  Atmosphere:\n")
            f.write(f"    Teff       : {params['Teff']} K\n")
            f.write(f"    C_to_O     : {params['C_to_O']}\n")
            f.write(f"    mh         : {params['mh']}\n")
            f.write(f"    gravity    : {params['gravity']} cm/s^2 (cgs)\n")
            f.write(f"    kzz        : {params['kzz']} cm^2/s\n")
            f.write(f"    phase      : {params['phase']} rad\n")
            f.write(f"    wave_range : {params['wave_range']} micron\n")
            f.write(f"    eq         : {params['eq']}\n")
            f.write(f"    excluded_mol: {params['excluded_mol']}\n")
            f.write(f"    background : {params.get('background', 'bobcat')}\n")           # NEW
            if params.get('background') == 'diamondback':                                  # NEW
                f.write(f"    diamondback_fsed: {params.get('diamondback_fsed', 1)}\n")     # NEW

            f.write(f"  Clouds:\n")
            fsed = params['fsed']
            if isinstance(fsed, dict):
                f.write(f"    fsed       :\n")
                for sp, val in fsed.items():
                    f.write(f"      {sp:<12}: {val}\n")
            else:
                f.write(f"    fsed       : {fsed}\n")
            f.write(f"    mmw        : {params['mmw']}\n")
            f.write(f"    gases      : {params['gases']}\n")
            f.write(f"    R          : {params['R']}\n")

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
    import re
    # runnerName = 'bd_v3_grid_Teff=1300K_kzz=1e7_withCH4_fsed=1.0-1.50_n=50_R=300'

    # runnerName = 'bd_v3_grid_Teff=1400K_kzz=1e7_withCH4_fsed=2.0-2.50_n=20_R=300'
    # runnerName = 'bd_v3_grid_Teff=1400K_kzz=1e6_withCH4_fsed=2.0-2.50_n=20_R=300'
    # runnerName = 'bd_v3_grid_Teff=1400K_kzz=1e6_withCH4_fsed=2.0-3.5_n=20_R=300'

    # runnerName = 'bd_v4_grid_Teff=1400K_kzz=7e9_withCH4_fsed=1.0-1.25_n=50_R=300'
    # runnerName = 'bd_v4_grid_Teff=1400K_kzz=3e9_withCH4_fsed=1.0-1.25_n=50_R=300'
    # runnerName = 'bd_v4_grid_Teff=1400K_kzz=1e9_withCH4_fsed=1.0-1.25_n=50_R=300'

    # runnerName = 'bd_v4_grid_Teff=1400K_kzz=7e9_withCH4_fsed=1.0-1.25_n=20_R=300'
    # runnerName = 'bd_v4_grid_Teff=1400K_kzz=3e9_withCH4_fsed=1.0-1.25_n=20_R=300'
    # runnerName = 'bd_v4_grid_Teff=1400K_kzz=1e9_withCH4_fsed=1.3-1.5_n=20_R=300'
    # runnerName = 'bd_v4_grid_Teff=1400K_kzz=1e9_withCH4_fsed=2.0-2.4_n=20_R=300'
    # runnerName = 'bd_v4_grid_Teff=1400K_kzz=1e9_withCH4_fsed=3.0-3.4_n=20_R=300'
    # runnerName = 'bd_v4_grid_Teff=1400K_kzz=5e8_withCH4_fsed=3.0-3.6_n=20_R=300'
    # runnerName = 'bd_v4_grid_Teff=1400K_kzz=1e8_withCH4_fsed=3.0-3.6_n=20_R=300'
    # runnerName = 'bd_v4_grid_Teff=1400K_kzz=1e7_withCH4_fsed=2.0-2.5_n=20_R=300'
    # runnerName = 'bd_v4_grid_Teff=1400K_kzz=1e6_withCH4_fsed=2.5-3.5_n=20_R=300'
    # runnerName = 'bd_v4_grid_Teff=1400K_kzz=1e8_withCH4_fsed=1.0-1.50_n=20_R=300'

    # runnerName = 'bd_v4_grid_Teff=1400K_kzz=1e7_withCH4_fsed=1.0-1.5_n=20_R=300'
    # runnerName = 'bd_v4_grid_Teff=1400K_kzz=1e7_withCH4_fsed=3.0-3.5_n=20_R=300'
    # runnerName = 'bd_v4_grid_Teff=1400K_kzz=1e6_withCH4_fsed=1.0-1.5_n=20_R=300'
    runnerName = 'bd_v4_grid_Teff=1400K_kzz=1e6_withCH4_fsed=2.0-2.5_n=20_R=300'
    runnerName = 'bd_v4_grid_Teff=1400K_kzz=1e6_withCH4_fsed=2.0-2.5_n=20_R=300_CO=2'

    Teff = int(re.search(r'Teff=(\d+)K', runnerName).group(1))
    kzz = float(re.search(r'kzz=([0-9.eE+-]+)', runnerName).group(1))
    match = re.search(r'CO=([0-9.eE+-]+)', runnerName)
    C_O = float(match.group(1)) if match else None

    # Teff_list=[100],
    # C_to_O_list=[0.55],
    # mh_list=[1.0],
    # gravity_list=[10000],
    # kzz_list=[1e7],
    # phase_list=[0],
    # wave_range_list=[[1, 10]],
    # eq_list=[False],
    # excluded_mol_list=[None],
    # mmw_list=[2.2],
    # fsed_list=[1],
    # R_list=[500],
    # gases_list=[['Fe', 'MgSiO3', 'Mg2SiO4', 'Al2O3', 'Na2S']],
    # background_list=['bobcat'],          # NEW
    # diamondback_fsed_list=[1],           # NEW
    # cloudfree=False

    g_cgs_list = [3e4]
    wave_rangeList = [[0.7, 15.0]]
    Teff_list = [Teff]
    C_to_O_list = [1.0 if C_O is None else C_O]
    mh_list = [1.0]
    kzz_list = [kzz]

    fsed_match = re.search(r'fsed=([0-9.eE+-]+)-([0-9.eE+-]+)', runnerName)
    if fsed_match:
        fsed_start, fsed_stop = float(fsed_match.group(1)), float(fsed_match.group(2))
    else:
        # fall back to single-value form, e.g. 'fsed=2.0_n=20' (no dash = no range)
        single_match = re.search(r'fsed=([0-9.eE+-]+)(?:_|$)', runnerName)
        if single_match:
            fsed_start = fsed_stop = float(single_match.group(1))
        else:
            raise ValueError(f"Could not parse fsed range from runnerName: {runnerName!r}")

    # NEW: opt into the Diamondback climate-converged background here. Leave
    # as 'bobcat'/[1] to reproduce old behavior exactly.
    # background_list = ['bobcat']
    background_list = ['diamondback']
    gases = ['Fe', 'Al2O3', 'MgSiO3', 'Mg2SiO4', 'SiO2']  # same gases for every run now
    diamondback_fsed_list = [np.floor(fsed_stop)]

    R = int(runnerName.split('_R=')[-1])
    mode = 'complex' if 'complex' in runnerName else 'simple'
    fsed_list = []
    n_levels = int(runnerName.split('_n=')[-1].split('_')[0])

    excluded_mol = 'CH4' if 'noCH4' in runnerName else None
    if mode == 'simple':
        fsed_list = generate_fsed(
            mode='simple',
            start=fsed_start,
            stop=fsed_stop,
            total=n_levels,
        )
    elif mode == 'complex':
        iterationName = "fSed_test_fe[14]_mgsiO3[3.1-3.9]_na2s[4.6-2.8]"
        runnerName = f"{runnerName}_{iterationName}"
        fsed_list = generate_fsed(
            mode='complex',
            species_params={
                'Fe':     14,
                'MgSiO3': {'start': 3.1, 'stop': 3.9},
                'Na2S':   {'start': 4.6, 'stop': 2.8},},
            n_steps=n_levels,
            gases=gases,
        )

    param_grid = create_parameter_grid(
        Teff_list=Teff_list,
        C_to_O_list=C_to_O_list,
        mh_list=mh_list,
        wave_range_list=wave_rangeList,
        R_list=[R],
        gravity_list=g_cgs_list,
        fsed_list=fsed_list,
        excluded_mol_list=[excluded_mol],
        gases_list=[gases],
        kzz_list=kzz_list,
        background_list=background_list,             # NEW
        diamondback_fsed_list=diamondback_fsed_list,  # NEW
        cloudfree=False
    )

    print(f"Total models to run: {len(param_grid)}")

    results, output_dir = run_parallel_grid(
        param_grid,
        runname=runnerName,
        n_processes=4
    )

    print("\n" + "="*60)
    print("Example - Loading a saved model:")
    print("="*60)

#%% Loading results

    import matplotlib.pyplot as plt
    import glob
    from virga import justplotit as vjpi
    from picaso import justplotit as jpi
    import xarray

    from bokeh.plotting import show, figure
    from bokeh.io import output_notebook
    output_notebook()

    # Plot all models in results on the same plot for comparison
    plt.figure(figsize=(15, 10))
    from matplotlib.lines import Line2D
    import matplotlib.colors as mcolors
    import matplotlib.cm as cm
    cmap = cm.get_cmap('bwr_r', len(results))
    handles = []
    for i, model in enumerate(results):
        if 'error' in model:
            continue
        fsed = model['params']['fsed']
        label = f"fsed={fsed}" if not isinstance(fsed, dict) else "fsed=" + "-".join(f"{sp}{v}" for sp, v in fsed.items())
        plt.plot(model['wavelength'], model['thermal'], color=cmap(i), label=label)
        handles.append(Line2D([0], [0], color=cmap(i), lw=2, label=label))
    plt.xlabel('Wavelength (micron)')
    plt.ylabel('Flux (erg/s/cm^2/Hz)')  # NEW: Fnu, not Flam
    plt.title('Comparison of Spectra from All Models')
    plt.yscale('linear')
    plt.legend(handles=handles, title="Model Parameters", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid()

    plt.savefig(os.path.join(output_dir, 'all_models_comparison.png'), bbox_inches='tight')
    plt.show()
    plt.close()
# %%