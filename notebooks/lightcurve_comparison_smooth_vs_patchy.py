"""
lightcurve_comparison.py
========================
Generates and compares three brown-dwarf surface models:

  Case A  –  smooth longitudinal sine wave
  Case B  –  coarse random patches with fill-fraction modulation
  Case C  –  anisotropic Gaussian random field with fill-fraction modulation

Produces a four-panel figure: maps for A/B/C and a shared disk-integrated
light curve.  Equatorial viewing geometry (inclination = 90°) is assumed.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.ndimage import gaussian_filter1d

# ── Grid constants ─────────────────────────────────────────────────────────────
NLON     = 180   # longitudinal pixels  (360° / NLON = 2° per pixel)
NLAT     = 90    # latitudinal pixels   (180° / NLAT = 2° per pixel)
N_PHASES = 90    # number of light-curve phase bins


# ══════════════════════════════════════════════════════════════════════════════
# Surface generators
# ══════════════════════════════════════════════════════════════════════════════

def gen_case_a(amplitude, nlon=NLON, nlat=NLAT):
    """
    Case A: perfectly smooth sinusoidal longitudinal wave.

    Parameters
    ----------
    amplitude : float
        Half-amplitude of the wave.  Map spans (1 - amplitude) to (1 + amplitude).

    Returns
    -------
    ndarray (nlat, nlon)
    """
    lons = np.linspace(-np.pi, np.pi, nlon, endpoint=False) + np.pi / nlon
    return np.ones((nlat, nlon)) + amplitude * np.sin(lons)[np.newaxis, :]


def gen_case_b(p, b, f0, df, patch_size, n_levels=2, seed=42,
               nlon=NLON, nlat=NLAT):
    """
    Case B: coarse random patches with sinusoidally modulated fill fraction.

    The surface is tiled with rectangular coarse cells.  Each cell is
    independently assigned a discrete intensity level; the probability of
    landing on a bright level is set by the local fill fraction f(lon).

    Parameters
    ----------
    p, b        : float  –  bright-feature and background intensity
    f0          : float  –  mean fill fraction (fraction of bright pixels)
    df          : float  –  fill-fraction modulation amplitude
    patch_size  : float  –  coarse-cell size as a divisor of NLON/NLAT
                            (patch_size ≈ patch width in pixels;
                             1 pixel ≈ 2° for the default NLON=180)
    n_levels    : int    –  number of discrete intensity steps between b and p
    seed        : int

    Returns
    -------
    ndarray (nlat, nlon)
    """
    rng = np.random.default_rng(seed)
    cn  = max(4, round(nlon / patch_size))   # coarse cells in longitude
    cm  = max(2, round(nlat / patch_size))   # coarse cells in latitude

    # Fill fraction as a function of coarse longitude index
    coarse_lons = (np.arange(cn) + 0.5) / cn * 2 * np.pi - np.pi
    f_coarse    = np.clip(f0 + df * np.sin(coarse_lons), 0.0, 1.0)  # (cn,)

    # Random draw per cell, shifted toward higher values when f_lon > f0
    u        = rng.random((cm, cn))
    u_biased = np.clip(u + (f_coarse[np.newaxis, :] - f0), 0.0, 1.0)

    # Quantise to n_levels and map to intensity
    levels = np.clip((u_biased * n_levels).astype(int), 0, n_levels - 1)
    coarse = b + levels / max(1, n_levels - 1) * (p - b)   # (cm, cn)

    # Nearest-neighbour upsample to full map
    row_idx = np.minimum(cm - 1, (np.arange(nlat) / nlat * cm).astype(int))
    col_idx = np.minimum(cn - 1, (np.arange(nlon) / nlon * cn).astype(int))
    return coarse[np.ix_(row_idx, col_idx)]


def gen_case_c(p, b, f0, df, blob_size_deg, aspect_ratio, seed=42,
               nlon=NLON, nlat=NLAT):
    """
    Case C: anisotropic Gaussian random field, thresholded per-column to
    achieve a sinusoidally modulated fill fraction.

    White noise is blurred with a separable Gaussian kernel whose standard
    deviations are:
        σ_lon  =  blob_size_deg / (360 / nlon)     [pixels]
        σ_lat  =  blob_size_deg * aspect_ratio / (180 / nlat)  [pixels]

    Each longitude column is then independently thresholded so that the
    fraction of bright pixels equals f(lon) = f0 + df·sin(lon).

    Parameters
    ----------
    p, b            : float  –  bright-feature and background intensity
    f0              : float  –  mean fill fraction
    df              : float  –  fill-fraction modulation amplitude
    blob_size_deg   : float  –  Gaussian σ in longitude (degrees)
    aspect_ratio    : float  –  σ_lat / σ_lon  (>1 → vertically elongated)
    seed            : int

    Returns
    -------
    ndarray (nlat, nlon)
    """
    rng = np.random.default_rng(seed)
    noise = rng.random((nlat, nlon))

    # Degrees-to-pixel conversion
    deg_per_px_lon = 360.0 / nlon
    deg_per_px_lat = 180.0 / nlat
    sigma_x = max(0.5, blob_size_deg / deg_per_px_lon)
    sigma_y = max(0.5, blob_size_deg * aspect_ratio / deg_per_px_lat)

    # Separable anisotropic blur
    noise = gaussian_filter1d(noise, sigma=sigma_x, axis=1, mode='wrap')     # longitude — wrap
    noise = gaussian_filter1d(noise, sigma=sigma_y, axis=0, mode='nearest')  # latitude  — clamp at poles

    # Per-column threshold to enforce f(lon)
    lons  = np.linspace(-np.pi, np.pi, nlon, endpoint=False) + np.pi / nlon
    f_lon = np.clip(f0 + df * np.sin(lons), 0.0, 1.0)

    m = np.empty((nlat, nlon))
    for i in range(nlon):
        col         = noise[:, i]
        thresh_idx  = int(np.clip((1.0 - f_lon[i]) * nlat, 0, nlat - 1))
        thresh      = np.sort(col)[thresh_idx]
        m[:, i]     = np.where(col >= thresh, p, b)

    return m


# ══════════════════════════════════════════════════════════════════════════════
# Light-curve integrator
# ══════════════════════════════════════════════════════════════════════════════

def compute_light_curve(surface_map, n_phases=N_PHASES):
    """
    Disk-integrate surface_map over the visible hemisphere at each rotation
    phase, assuming equatorial viewing (inclination = 90°).

    Each pixel is weighted by:
        w(i, j)  =  max(0, cos Δlon)  ×  cos(lat)

    where Δlon is the angular separation between the pixel and the
    sub-observer longitude, and cos(lat) is the spherical area element.

    Parameters
    ----------
    surface_map : ndarray (nlat, nlon)
    n_phases    : int  –  number of evenly-spaced phase bins

    Returns
    -------
    phases : ndarray (n_phases,)  –  rotation phase in degrees [0, 360)
    flux   : ndarray (n_phases,)  –  disk-averaged intensity
    """
    nlat, nlon = surface_map.shape

    lats    = np.linspace(np.pi / 2, -np.pi / 2, nlat, endpoint=False) \
              + np.pi / (2 * nlat)
    cos_lat = np.cos(lats)                                        # (nlat,)

    lons   = np.linspace(-np.pi, np.pi, nlon, endpoint=False) + np.pi / nlon
    phases = np.linspace(0, 2 * np.pi, n_phases, endpoint=False)

    flux = np.empty(n_phases)
    for t, phi_c in enumerate(phases):
        d_lon  = ((lons - phi_c + 3 * np.pi) % (2 * np.pi)) - np.pi  # wrap to [-π, π]
        mu     = np.maximum(np.cos(d_lon), 0.0)                       # (nlon,)
        W      = mu[np.newaxis, :] * cos_lat[:, np.newaxis]           # (nlat, nlon)
        flux[t] = np.sum(surface_map * W) / np.sum(W)

    return np.degrees(phases), flux


# ══════════════════════════════════════════════════════════════════════════════
# Figure
# ══════════════════════════════════════════════════════════════════════════════

def plot_comparison(map_A, map_B, map_C,
                    lc_phases, lc_A, lc_B, lc_C,
                    vmin, vmax,
                    savepath=None):
    """
    Four-panel figure: three surface maps (top row) + shared light curve (bottom).
    """
    AMBER  = '#BA7517'
    PURPLE = '#534AB7'
    TEAL   = '#1D9E75'

    fig = plt.figure(figsize=(12, 4))
    gs  = gridspec.GridSpec(
        2, 3, figure=fig,
        height_ratios=[0.5, 1],
        hspace=0.42, wspace=0.06,
    )

    ax_A  = fig.add_subplot(gs[0, 0])
    ax_B  = fig.add_subplot(gs[0, 1])
    ax_C  = fig.add_subplot(gs[0, 2])
    ax_lc = fig.add_subplot(gs[1, :])

    map_axes = [ax_A, ax_B, ax_C]
    titles   = ['Case A — smooth sinusoidal',
                'Case B — pixel level',
                'Case C — Gaussian patches']
    case_colors = [AMBER, PURPLE, TEAL]
    surface_maps = [map_A, map_B, map_C]

    # ── Map panels ────────────────────────────────────────────────────────────
    im = None
    for ax, title, color, m in zip(map_axes, titles, case_colors, surface_maps):
        vmin, vmax = m.min(), m.max()  # individual color scaling

        if ax is not ax_A: 
            cmap = 'plasma'
            # increase vmin/vmax contrast for better visibility of discrete levels
            vmin, vmax = vmin*0.75, vmax*1.2
        else:          
            cmap = 'plasma'
            vmin, vmax = vmin*0.85, vmax*1.15 

        im = ax.imshow(
            m, origin='upper', aspect='auto',
            extent=[-180, 180, -90, 90],
            vmin=vmin, vmax=vmax, cmap=cmap,
        )
        ax.set_title(title, color=color, fontsize=9, fontweight='bold', pad=5)
        ax.set_xlabel('Longitude (°)', fontsize=8)
        ax.tick_params(labelsize=7)
        ax.axhline(0, color='white', lw=0.6, ls='--', alpha=0.5)   # equator
        if ax is ax_C:
            ax.set_xticks([]),
            ax.set_xticks([-90, 0, 90, 180])
        else: ax.set_xticks([-180, -90, 0, 90, 180])
        ax.set_yticks([-90, -45, 0, 45, 90])

        if ax is ax_A:
            ax.set_ylabel('Latitude (°)', fontsize=8)
            c = plt.colorbar(im, ax=ax, shrink=0.88)  # individual colorbars
            # c.set_label('Intensity', fontsize=8)
            c.ax.tick_params(labelsize=7)
        else:
            ax.set_yticklabels([])

    # Shared colorbar for ax_B and ax_C if they have the same scaling
    if (map_B.min(), map_B.max()) == (map_C.min(), map_C.max()):
        cbar = fig.colorbar(im, ax=map_axes[1:3], shrink=0.88, pad=0.015, fraction=0.018)
        cbar.set_label('Intensity', fontsize=6)
        cbar.ax.tick_params(labelsize=6)

    # ── Light curve panel ─────────────────────────────────────────────────────
    lc_data   = [lc_A,   lc_B,   lc_C]
    lc_styles = [
        dict(ls='-',      lw=2.0, zorder=3),
        dict(ls='--',     lw=1.6, zorder=2),
        dict(ls=(0,(3,1,1,1)), lw=1.6, zorder=2),  # dash-dot
    ]

    for lc, color, style, label in zip(lc_data, case_colors, lc_styles, titles):
        ax_lc.plot(lc_phases, lc, color=color, label=label, **style)

    amp_A = (lc_A.max() - lc_A.min())
    amp_B = (lc_B.max() - lc_B.min())
    amp_C = (lc_C.max() - lc_C.min())

    ax_lc.set_xlabel('Rotation phase (°)', fontsize=9)
    ax_lc.set_ylabel('Normalized flux',    fontsize=9)
    ax_lc.tick_params(labelsize=7)
    ax_lc.set_xlim(0, 360)
    ax_lc.set_ylim(0.8, 1.2)
    ax_lc.xaxis.set_major_locator(plt.MultipleLocator(45))
    ax_lc.grid(axis='x', lw=0.4, alpha=0.4)
    ax_lc.grid(axis='y', lw=0.4, alpha=0.4)

    legend = ax_lc.legend(fontsize=8, framealpha=0.5, loc='upper right')

    # Amplitude annotation
    ann_text = (f"LC amplitudes:   "
                f"A ±{amp_A*100:.3f}%   "
                f"B ±{amp_B*100:.3f}%   "
                f"C ±{amp_C*100:.3f}%")
    ax_lc.text(0.02, 0.04, ann_text, transform=ax_lc.transAxes,
               fontsize=9, color='0.4', va='bottom')
    plt.tight_layout()

    if savepath:
        fig.savefig(savepath, dpi=300, bbox_inches='tight')
        print(f"Saved → {savepath}")

    return fig

# ══════════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':

    # ── Parameters ────────────────────────────────────────────────────────────
    amplitude  = 0.11    # Case A half-amplitude  → map range  0.9–1.1
    p          = 1.45    # bright-feature intensity  (B & C)
    b          = 0.50    # background intensity       (B & C)
    f0         = 0.55    # mean fill fraction          (B & C)
    df         = 0.11    # fill-fraction modulation amplitude
    # Case B
    patch_size = 4       # coarse cell divisor  (cell ≈ 8° for NLON=180)
    n_levels   = 2       # intensity levels
    # Case C
    blob_size  = 3.0     # Gaussian σ in longitude (degrees)
    aspect     = 3.5     # σ_lat / σ_lon
    SEED       = 42

    # ── Generate surfaces ─────────────────────────────────────────────────────
    map_A = gen_case_a(amplitude)
    map_B = gen_case_b(p, b, f0, df, patch_size, n_levels, seed=SEED)
    map_C = gen_case_c(p, b, f0, df, blob_size,  aspect,   seed=SEED)

    # ── Compute light curves ──────────────────────────────────────────────────
    phases, lc_A = compute_light_curve(map_A)
    _,      lc_B = compute_light_curve(map_B)
    _,      lc_C = compute_light_curve(map_C)

    # ── Shared colour scale ───────────────────────────────────────────────────
    vmin = min(b, 1.0 - amplitude)
    vmax = max(p, 1.0 + amplitude)

    # ── Diagnostics ───────────────────────────────────────────────────────────
    print(f"Feature contrast  p - b   = {p - b:.2f}")
    print(f"Δf needed to match A      = {amplitude / (p - b):.4f}  (current Δf = {df})")
    print(f"LC amplitude  A : {(lc_A.max()-lc_A.min())*100:.2f}%")
    print(f"LC amplitude  B : {(lc_B.max()-lc_B.min())*100:.2f}%")
    print(f"LC amplitude  C : {(lc_C.max()-lc_C.min())*100:.2f}%")

    # ── Plot ──────────────────────────────────────────────────────────────────
    fig = plot_comparison(
        map_A, map_B, map_C,
        phases, lc_A, lc_B, lc_C,
        vmin=vmin, vmax=vmax,
        savepath='phasecurve_comparison_smooth_vs_patchy_sinusoidal_maps.pdf',
    )
    plt.show()
