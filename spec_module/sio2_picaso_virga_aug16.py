# Set environment variables for PICASO reference data    
picaso_refdata = "/Users/nguyendat/Documents/GitHub/picaso/reference/"
pysyn_cdbs = "/Users/nguyendat/Documents/GitHub/picaso/reference/stellar_spectra/grp/redcat/trds"
picaso_home = '/Users/nguyendat/Documents/GitHub/picaso/'

import os 
"""Set environment variables for PICASO reference data."""
os.environ['picaso_refdata'] = picaso_refdata
os.environ['PYSYN_CDBS'] = pysyn_cdbs

from picaso import justdoit as jdi
from virga import justdoit as vjdi

import importlib
importlib.reload(vjdi)

#plot tools
from picaso import justplotit as jpi
from virga import justplotit as vjpi

import astropy.units as u
import pandas as pd
from bokeh.plotting import show, figure
from bokeh.io import output_notebook 
output_notebook()

import pickle as pickle
import numpy as np

### Function to read Diamondback PT-profile: get closest physical properties.

import re
import math
from io import StringIO

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
        # sanity-check: the gravity metadata line, if present, should roughly
        # match the gravity implied by the filename
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

    ### Declare SiO2_amorph
### The default VIRGA call is SiO2 is SiO2_alpha (the SiO2_alpha and SiO2 mieff files are identical)
import virga.gas_properties as gas_properties
import virga.pvaps as pvaps

optical_prop_dir = os.path.join(picaso_home, 'data', 'virga')   # contains virga's .refrind and .mieff files
# check if SiO2_amorph exist, if not, calculate it
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

    # bar -> dyn/cm^2
    return 1e6 * pvap_sio2

# monkey-patch so Atmosphere()/compute() can find SiO2_amorph by name
gas_properties.SiO2_amorph = SiO2_amorph_virga
pvaps.SiO2_amorph = SiO2_amorph

### Configure Picaso atmosphere case function

#point to where you untared your sonora profiles
sonora_profile_db = '/Users/nguyendat/Documents/GitHub/picaso/data/sonora_profile/'
# point to the untarred "pressure-temperature_profiles" folder from the Sonora
sonora_diamondback_profile_db = '/Users/nguyendat/Documents/GitHub/picaso/data/sonora_diamondback_profile/'

### setup for w1741: 
# T=1450K, logg=4.65, R=1.2Rjup, 
# M=27.2MJup, L7 spectral type, C/O=0, M/H=1.0, kzz=5e5

def configure_atm(C_to_O=1.00, mh=1.0, Teff=1400,
                  gravity=10**4.65, kzz=5e5, phase=0, wave_range=[0.3,15], 
                  background='diamondback', fsed=1, eq=True, excluded_mol=None):
    """
    This function configures a brown dwarf atmosphere with the following inputs:
    C_to_O: float, carbon to oxygen ratio, relative to solar, default: 1.0, solar
    mh: float, metallicity in [M/H] relative to solar, default: 1.0, solar.  VIRGA BROWN DWARF WILL GIVE ERROR FOR MH.
    Teff: int, effective temperature in Kelvin
    gravity: float, surface gravity in m/s^2
    kzz: float, vertical mixing coefficient in cm^2/s
    phase: float, phase angle in radians
    wave_range: list, wavelength range in microns [min, max]
    background: str, 'diamondback' (default) or 'bobcat'.
        'diamondback' loads a climate-converged (RC-equilibrium) Sonora
        Diamondback P-T + chemistry structure anchored at `fsed` -- use this
        if you plan to sweep VIRGA's own fsed over a narrow range around that
        anchor, so the background thermal structure is self-consistent with
        the cloud regime you're actually modeling.
        'bobcat' reproduces the original cloud-free Sonora Bobcat behavior
        (nearest Teff/gravity neighbor), with chemistry from either
        chemeq_visscher (eq=True) or the packaged low-T grid (eq=False).
    fsed: float, only used if background='diamondback'. Which converged
        Diamondback cloud structure to anchor to (grid: 1, 2, 3, 4, 8).
    eq: chem_eq visscher, chem-equilibrium if true
    """
    
    opa = jdi.opannection(wave_range=wave_range) # define opacity range
    bd = jdi.inputs(calculation='browndwarf') # define brown dwarf model/ use no star
    bd.phase_angle(phase) #radians #phase angle of 0 is full phase
    bd.gravity(gravity=gravity, gravity_unit=u.Unit('m/(s**2)')) #any astropy units available

    if background == 'diamondback':
        # Diamondback filenames encode [M/H] in dex (log10 relative to solar), but this
        # function's `mh` argument follows PICASO's linear-ratio-to-solar convention
        # (mh=1.0 -> solar), same as C_to_O. Convert before matching against the grid.
        if mh <= 0:
            raise ValueError("mh must be > 0 (it's a linear ratio to solar; log10(mh) "
                            "is taken to match Diamondback's dex-based [M/H] grid).")
        bd, profile = read_sonora_diamondback(bd, sonora_diamondback_profile_db, Teff,
                                           fsed=fsed, mh=np.log10(mh), co=C_to_O)
    elif background == 'bobcat':
        #this function will grab the gravity you have input above and find the nearest neighbor with the 
        #note sonora chemistry grid is on the same opacity grid as our opacities (1060). 
        bd.sonora(sonora_profile_db, Teff) # get sonora profile closest to Teff and input gravity
        if eq: # if using chemical equilibrium
            bd.chemeq_visscher(C_to_O, mh) # compute chemeq with visscher grid
        profile = bd.inputs['atmosphere']['profile']
    else:
        raise ValueError("background must be 'diamondback' or 'bobcat'")

    # Add kzz (vertical mixing) to the profile
    profile['kzz'] = np.ones(profile['pressure'].shape[0])*[kzz]
    profile['kz'] = profile['kzz'].copy() # virga wants it named 'kz' so add it again
    
    bd.atmosphere(df=profile, exclude_mol=excluded_mol) # input the profile

    print('====== Configured a brown dwarf atmosphere with: ======')
    print(f'Background: {background}' + (f' (anchor fsed={fsed})' if background == 'diamondback' else ''))
    print(f'Requested Teff: {Teff} K, log(g): {np.log10(gravity)}, g: {gravity} m/s^2, C/O: {C_to_O}, [M/H]: {mh}, kzz: {kzz} cm^2/s')
    
    return bd, opa, profile

# let's try to make an atmosphere at 1450K, anchored to the climate-converged
# Sonora Diamondback fsed structure
bd, opa, profile = configure_atm(fsed=4, eq=True)
print(profile.isna().sum())
print(profile.describe())

fig=figure(y_axis_type="log",x_axis_type='log', y_range=(1e3, 1e-6))

# Example: plot the Sonora profile gases for CO and CO2 with pressure on y-axis
p = profile['pressure']

print('Profile Gases Molecules:', bd.inputs['atmosphere']['profile'].keys())

fig.line(profile['CO'], p, legend_label='CO', line_width=2, color='red')
fig.line(profile['H2O'], p, legend_label='H2O', line_width=2, color='blue')
fig.line(profile['CH4'], p, legend_label='CH4', line_width=2, color='green')
fig.line(profile['CO2'], p, legend_label='CO2', line_width=2, color='orange')
fig.line(profile['NH3'], p, legend_label='NH3', line_width=2, color='purple')
fig.xaxis.axis_label = "Concentration"
fig.yaxis.axis_label = "Pressure (bar)"
show(fig)

#get pyeddy recommendation for which gases to run
metallicity = 1.0
mean_molecular_weight = 2.2
recommended_gases = vjdi.recommend_gas(profile['pressure'], profile['temperature'],
                                     mh=metallicity, mmw=mean_molecular_weight, 
                                     plot=True, plot_width=800, plot_height=600)
print('Recommended gases to run (raw virga output):\n', recommended_gases, '\n')
recommended_gases = [item for item in recommended_gases if item != 'CaAl12O19'] # remove CaAl12O19 as mieff is not in the virga database

# Make sure to check available condensates in virga

# Only these gases have virga Mieff files
# SiO2_alpha is alpha crystalline quartz
available_condensates = ['ZnS', 'TiO2', 'NH3', 'Na2S', 'MnS', 
                        'MgSiO3', 'Mg2SiO4', 'KCl', 'H2O', 'Fe', 
                        'Cr', 'CH4', 'CaTiO3', 'Al2O3','SiO2', 
                        'SiO2_amorph'] 
print('Condensates with mieff:\n', available_condensates, '\n')

# Only keep recommended gases that are also available in virga
available_condensates = [item for item in recommended_gases if item in available_condensates]
print('Filtered recommended condensates with mieff:\n', available_condensates, '\n')

#let's get the cloud free spectrum for reference
clf = bd.spectrum(opa, full_output=True)

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

    # stash results in .attrs rather than as columns -- xg/yg are a *different length*
    # than df (mean_regrid downsamples), so they can never safely be real columns
    df['fluxnu_wavenumber'] = wavenum_sorted
    df['fluxnu'] = flux_nu_sorted
    df['regridx'] = xg
    df['regridy'] = yg

    return df # unchanged core columns ('wavenumber', 'thermal', etc.) -- configure_cloud
               # and anything else indexing out['wavenumber'] keeps working as before

df = convert_unit(clf)
xg, yg = df['regridx'], df['regridy']
xclf, yclf = jdi.mean_regrid(clf['wavenumber'], clf['thermal'], R=300)

show(jpi.spectrum(xclf, yclf, plot_width=1000, plot_height=300,
                     y_axis_label='Flux (erg/s/cm^2/cm)',
                     x_axis_label='Wavelength (micron)',
                     y_axis_type='log', x_axis_type='log',
                     title='Diamondback fsed 1 spectrum in original units (erg/s/cm^2/cm)'))
show(jpi.spectrum(xg, yg, plot_width=1000, plot_height=300,
                     y_axis_label='Flux (erg/s/cm^2/Hz)',
                     x_axis_label='Wavelength (micron)',
                     y_axis_type='log', x_axis_type='log',
                     title='Cloud-free spectrum converted to Fnu (erg/s/cm^2/Hz)'))

### Plot condensation temperature of some species on top of the P-T profile
from bokeh.palettes import Category10_10  # or Category10_6 for 6 colors

fig = figure(y_axis_type="log",x_axis_type='log', y_range=(1e2, 1e-3), 
             width=800, height=600,
             x_axis_label='Temperature (K)',
             y_axis_label='Pressure (bar)')

gases = ['Cr', 'Fe', 'Al2O3', 'MgSiO3', 'Mg2SiO4', 'SiO2', 'KCl', 'MnS', 'Na2S']
colors = Category10_10[:len(gases)] # get tableau colors up to number of gases

# plot the P-T profile
fig.line(profile['temperature'], profile['pressure'], 
         legend_label='P-T Profile', line_width=2, color='black',
         line_dash='dashed')

for i, gasName in enumerate(gases):
    #grab p,t from eddysed
    pgas, tgas = vjdi.condensation_t(gas_name=gasName, mh=1, mmw=2.2)
    fig.line(tgas, pgas, legend_label=gasName, line_width=2, 
             color=colors[i])

# Make legend interactive
fig.legend.click_policy = "mute"  # or "hide" to hide out instead

show(fig)
    
def configure_cloud(df, opacity, mmw=2.2, gases=['MgSiO3','Fe','Mg2SiO4', 'SiO2'],
                    fsed=1, directory='/Users/nguyendat/Documents/GitHub/picaso/data/virga/', R=300):
    """
    This function configures clouds in a brown dwarf atmosphere with the following inputs:
    df: picaso atmosphere object
    opacity: picaso opacity object
    mmw: float, mean molecular weight
    gases: list, list of cloud species to include
    fsed: float, sedimentation efficiency parameter
    directory: str, directory to save cloud files
    """
    if gases == 'recommended':
        gases = available_condensates

    print('Using cloud species:', gases)
    # call virga to compute cloud profile and optical depth
    cld_out = df.virga(gases, directory, fsed = fsed, mmw = mmw)
    # VIRGA BROWN DWARF WILL GIVE ERROR FOR MH metalicity

    out = df.spectrum(opacity, full_output=True)
    out = convert_unit(out, R=R)
    # x_cldy, y_cldy = jdi.mean_regrid(out['wavenumber'], out['thermal'], R=R)

    return out['regridx'], out['regridy'], cld_out, out

fsed_list = {'Fe': 1, 'MgSiO3': 1, 'SiO2': 1}
gas = ['Fe', 'MgSiO3', 'SiO2']
x_cldy, y_cldy, cld_out, out = configure_cloud(bd, opa, gases=gas, fsed=fsed_list)

fsed_list1 = {'Fe': 1.2, 'MgSiO3': 1.2, 'SiO2': 1.2}
gas1 = ['Fe', 'MgSiO3', 'SiO2']
x_cldy1, y_cldy1, cld_out1, out1 = configure_cloud(bd, opa, gases=gas1, fsed=fsed_list1)

fsed_list2 = {'Fe': 1.5, 'MgSiO3': 1.5, 'SiO2': 1.5}
gas2 = ['Fe', 'MgSiO3', 'SiO2']
x_cldy2, y_cldy2, cld_out2, out2 = configure_cloud(bd, opa, gases=gas2, fsed=fsed_list2)