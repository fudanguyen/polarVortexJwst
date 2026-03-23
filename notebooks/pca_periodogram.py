"""
PCA on Periodograms & Bandpass-Filtered Light Curves
=====================================================
Strategy:
  1. Normalize each frame by the median spectrum.
  2. Compute a Lomb-Scargle periodogram for every wavelength bin
     → matrix (1255 wave × N_periods).  PCA on this tells you which
     *period* components have the strongest spectral color.
  3. (Fallback) If the two sinusoids are spectrally degenerate in the
     periodogram PCA, bandpass-filter each wavelength light curve into
     a "5-hr band" and a "60-hr band", then run a separate PCA on each
     filtered datacube.  This forces temporal separation before PCA and
     lets you recover the spectral shape of each mode independently.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from astropy.timeseries import LombScargle
from sklearn.decomposition import PCA
from scipy.signal import butter, filtfilt

# ─── 1. LOAD & NORMALIZE ─────────────────────────────────────────────────────
df   = pd.read_csv("df_plot.csv", header=0, index_col=None)
wave = df.iloc[0, :].values.astype(float)          # 1255 wavelength bins (µm)
flux = df.iloc[1:, :].values.astype(float)         # 120 frames × 1255 wave

n_frames, n_wave = flux.shape
t_hours = np.linspace(0, 60, n_frames)             # uniform 0.5-hr cadence

# Divide every frame by the time-median spectrum → residuals ≈ 1
med_spec = np.median(flux, axis=0)
norm_flux = flux / med_spec[np.newaxis, :]         # shape: (120, 1255)

# ─── 2. PERIODOGRAM FOR EVERY WAVELENGTH BIN ─────────────────────────────────
# Period grid from ~0.5 hr (Nyquist) out to 100 hr
periods = np.linspace(0.6, 100, 2000)              # hours
freqs   = 1.0 / periods                            # cycles / hour

power_matrix = np.zeros((n_wave, len(periods)))    # (1255, 2000)

for i in range(n_wave):
    lc = norm_flux[:, i] - 1.0                     # subtract baseline
    ls = LombScargle(t_hours, lc)
    power_matrix[i, :] = ls.power(freqs)

# ─── 3. PCA ON PERIODOGRAM MATRIX ────────────────────────────────────────────
# Each row = one wavelength's periodogram.  PCA finds the period-space
# components with the most spectral variation.
pca_pg = PCA(n_components=5)
pg_scores = pca_pg.fit_transform(power_matrix)     # (1255, 5)

print("=== PCA on Periodogram matrix ===")
for k in range(5):
    print(f"  PC{k+1}: {pca_pg.explained_variance_ratio_[k]*100:.3f}%")

# ─── 4. BANDPASS FILTER + PCA (fallback if periodogram PCA is degenerate) ────
dt   = t_hours[1] - t_hours[0]                     # 0.5 hr
fs   = 1.0 / dt                                    # 2 cycles/hr (sampling freq)
nyq  = fs / 2.0                                    # 1 cycle/hr

def bandpass(data_1d, lo_period, hi_period, fs, nyq, order=2):
    """Butter bandpass on a single light curve. lo/hi are in hours (period)."""
    lo_freq = 1.0 / hi_period   # lower freq edge
    hi_freq = 1.0 / lo_period   # upper freq edge
    b, a = butter(order, [lo_freq / nyq, hi_freq / nyq], btype='band')
    return filtfilt(b, a, data_1d)

# 5-hr mode: keep periods 3–8 hr  (centred on 5 hr)
# 60-hr mode: keep periods 20–100 hr  (centred on 60 hr, upper = Nyquist-ish)
filtered_5h  = np.zeros_like(norm_flux)            # (120, 1255)
filtered_60h = np.zeros_like(norm_flux)

for i in range(n_wave):
    lc = norm_flux[:, i] - 1.0
    filtered_5h[:, i]  = bandpass(lc,  3,   8,  fs, nyq)
    filtered_60h[:, i] = bandpass(lc, 20, 100,  fs, nyq)

# PCA on each filtered cube  (rows = frames, cols = wave → transpose for "spectral PCA")
# We want to find the *spectral shape* of each mode, so we PCA across wavelength.
# Transpose: (1255 wave × 120 frames) so each row is a wavelength's filtered LC.
pca_5h  = PCA(n_components=3)
scores_5h  = pca_5h.fit_transform(filtered_5h.T)   # (1255, 3)

pca_60h = PCA(n_components=3)
scores_60h = pca_60h.fit_transform(filtered_60h.T) # (1255, 3)

print("\n=== PCA on 5-hr bandpass cube ===")
for k in range(3):
    print(f"  PC{k+1}: {pca_5h.explained_variance_ratio_[k]*100:.3f}%")
print("=== PCA on 60-hr bandpass cube ===")
for k in range(3):
    print(f"  PC{k+1}: {pca_60h.explained_variance_ratio_[k]*100:.3f}%")

# ─── 5. PLOTTING ──────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(16, 20))
gs  = GridSpec(4, 2, figure=fig, hspace=0.35, wspace=0.3)

# --- Panel A: example periodograms at 3 wavelengths ---
ax = fig.add_subplot(gs[0, :])
test_waves = [1.35, 2.45, 4.9]   # µm — match user's stated bins
colors = ["#e74c3c", "#3498db", "#2ecc71"]
for wt, col in zip(test_waves, colors):
    idx = np.argmin(np.abs(wave - wt))
    ax.plot(periods, power_matrix[idx, :], color=col, label=f"{wave[idx]:.2f} µm", lw=1.2)
ax.set_xlabel("Period (hours)", fontsize=12)
ax.set_ylabel("Lomb-Scargle Power", fontsize=12)
ax.set_title("A — Periodograms at representative wavelengths", fontsize=13)
ax.axvline(5, color="k", ls="--", lw=0.8, alpha=0.5)
ax.axvline(60, color="k", ls="--", lw=0.8, alpha=0.5)
ax.text(5.5, ax.get_ylim()[1]*0.9, "5 hr", fontsize=10)
ax.text(61, ax.get_ylim()[1]*0.9, "60 hr", fontsize=10)
ax.legend(fontsize=10)
ax.set_xlim(0, 100)

# --- Panel B: PCA scores from periodogram (spectral color of each PC) ---
ax = fig.add_subplot(gs[1, 0])
for k in range(3):
    ax.plot(wave, pg_scores[:, k], label=f"PC{k+1} ({pca_pg.explained_variance_ratio_[k]*100:.2f}%)", lw=1)
ax.set_xlabel("Wavelength (µm)", fontsize=11)
ax.set_ylabel("PC Score", fontsize=11)
ax.set_title("B — Periodogram PCA: spectral loading", fontsize=12)
ax.legend(fontsize=9)

# --- Panel C: PCA components (period-space) ---
ax = fig.add_subplot(gs[1, 1])
for k in range(3):
    ax.plot(periods, pca_pg.components_[k, :], label=f"PC{k+1}", lw=1)
ax.set_xlabel("Period (hours)", fontsize=11)
ax.set_ylabel("Component weight", fontsize=11)
ax.set_title("C — Periodogram PCA: period-space components", fontsize=12)
ax.axvline(5, color="k", ls="--", lw=0.8, alpha=0.4)
ax.axvline(60, color="k", ls="--", lw=0.8, alpha=0.4)
ax.set_xlim(0, 100)
ax.legend(fontsize=9)

# --- Panel D: Bandpass-filtered light curves (one example wavelength) ---
ax = fig.add_subplot(gs[2, 0])
idx_ex = np.argmin(np.abs(wave - 1.35))
ax.plot(t_hours, norm_flux[:, idx_ex] - 1, color="gray",  lw=0.8, alpha=0.6, label="Full residual")
ax.plot(t_hours, filtered_5h[:, idx_ex],  color="#e74c3c", lw=1.2, label="3–8 hr band")
ax.plot(t_hours, filtered_60h[:, idx_ex], color="#3498db", lw=1.2, label="20–100 hr band")
ax.set_xlabel("Time (hours)", fontsize=11)
ax.set_ylabel("Residual flux", fontsize=11)
ax.set_title(f"D — Filtered LCs at {wave[idx_ex]:.2f} µm", fontsize=12)
ax.legend(fontsize=9)

# --- Panel E: 5-hr PCA spectral scores ---
ax = fig.add_subplot(gs[2, 1])
for k in range(3):
    ax.plot(wave, scores_5h[:, k], label=f"PC{k+1} ({pca_5h.explained_variance_ratio_[k]*100:.1f}%)", lw=1)
ax.set_xlabel("Wavelength (µm)", fontsize=11)
ax.set_ylabel("PC Score", fontsize=11)
ax.set_title("E — 5-hr band PCA: spectral shape", fontsize=12)
ax.legend(fontsize=9)

# --- Panel F: 60-hr PCA spectral scores ---
ax = fig.add_subplot(gs[3, 0])
for k in range(3):
    ax.plot(wave, scores_60h[:, k], label=f"PC{k+1} ({pca_60h.explained_variance_ratio_[k]*100:.1f}%)", lw=1)
ax.set_xlabel("Wavelength (µm)", fontsize=11)
ax.set_ylabel("PC Score", fontsize=11)
ax.set_title("F — 60-hr band PCA: spectral shape", fontsize=12)
ax.legend(fontsize=9)

# --- Panel G: Reconstructed time series from bandpass PCs ---
ax = fig.add_subplot(gs[3, 1])
# PCA was fit on (1255 wave × 120 frames), so components[k] IS the k-th time series
recon_5h_pc1  = pca_5h.components_[0, :]    # (120,)
recon_60h_pc1 = pca_60h.components_[0, :]   # (120,)
recon_5h_pc1  /= np.std(recon_5h_pc1)
recon_60h_pc1 /= np.std(recon_60h_pc1)
ax.plot(t_hours, recon_5h_pc1,  color="#e74c3c", lw=1.2, label="5-hr band PC1")
ax.plot(t_hours, recon_60h_pc1, color="#3498db", lw=1.2, label="60-hr band PC1")
ax.set_xlabel("Time (hours)", fontsize=11)
ax.set_ylabel("Normalised PC1 projection", fontsize=11)
ax.set_title("G — PC1 time series (bandpass separated)", fontsize=12)
ax.legend(fontsize=9)

plt.savefig("/mnt/user-data/outputs/pca_periodogram_output.png", dpi=150, bbox_inches="tight")
plt.show()
print("\nDone. Figure saved to pca_periodogram_output.png")
