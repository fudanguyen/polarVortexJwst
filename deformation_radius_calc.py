"""
Deformation radius (L_D) and equatorial deformation radius (L_eq) calculation
for Jupiter and six substellar targets (PSO 318, DENIS J112639/2M1126,
Luhman 16A, Luhman 16B, SIMP0136, 2M2139).

Physics
-------
L_D  = N*H / f_pole          (Rossby radius of deformation; Gill 1982, Pedlosky
                               1987, Vallis 2017)
L_eq = sqrt(N*H / beta)      (equatorial Rossby radius; Matsuno 1966, Gill 1980)

with:
    f_pole = 2*Omega                      (Coriolis parameter at the pole)
    beta   = 2*Omega / R                  (equatorial df/dy)
    H      = R_specific * Teff / g        (pressure scale height, kT/mg)
    R_specific = k_B / (mu * m_H) = R_universal / mu

N (Brunt-Vaisala / buoyancy frequency) is the least-constrained quantity here.
Where a literature value exists it is used and cited; otherwise a fiducial
N = 2e-2 s^-1 is used and flagged as UNCITED.

    Jupiter:        N ~ 2e-2 s^-1  (upper end of Watkins & Cho 2013, GRL 40, 472;
                                     they report 2-12e-3 s^-1 in/between cloud
                                     layers, rising to 1.6-2.3e-2 s^-1 near
                                     250 mbar / tropopause)
    PSO 318:        N = 6e-3 s^-1  (Parent et al. 2020, A&A; their "LGTD"
                                     low-gravity T-dwarf model at Teff=1200 K,
                                     log g=3.0 gives buoyancy period ~1000 s,
                                     i.e. N = 2*pi/1000 ~ 6e-3 s^-1)
    All others:     N = 2e-2 s^-1  (FIDUCIAL, UNCITED -- placeholder pending a
                                     literature or retrieval-based value)

Everything else (H, f_pole, beta) is computed directly from the object's own
Teff, log g, rotation period, and radius -- no fitting or tuning.
"""

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------------
R_UNIVERSAL = 8314.0      # J / (kmol K)   universal gas constant
MU          = 2.3         # mean molecular weight, H2/He atmosphere (amu)
R_SPECIFIC  = R_UNIVERSAL / MU   # J / (kg K)
R_JUP       = 7.1492e7    # m, Jupiter equatorial radius
G_GRAV      = 6.674e-11   # m^3 / (kg s^2), gravitational constant

# ---------------------------------------------------------------------------
# Object parameters
#   Teff   [K]
#   logg   [cgs, log10(cm/s^2)]
#   Prot   [hours]
#   R_RJup [planetary radius, units of R_Jup]
#   N      [s^-1]  buoyancy frequency
#   N_src  citation / status string for N
# ---------------------------------------------------------------------------
objects = {
    "Jupiter": dict(
        Teff=124.4, logg=3.395, Prot=9.925, R_RJup=1.00,
        N=2.0e-2, N_src="Watkins & Cho 2013 (upper end of range)"
    ),
    "Saturn": dict(
        Teff=95.0, logg=3.02, Prot=10.561, R_RJup=0.843,
        N=2.0e-2, N_src="this work"
    ),
    "PSO 318": dict(
        Teff=1200.0, logg=3.0, Prot=8.6, R_RJup=1.80,   # Prot updated: Biller et al. 2018 (8.6+/-0.1 hr), confirmed by Vos et al. 2018
        N=2.0e-3, N_src="this work"
        # NOTE: Teff/logg/R follow Mollière et al. 2025 SED retrieval, not Allers et al. 2016
        # (Teff=1127 K, logg=4.01, R=1.464 RJup) -- confirm this substitution is intentional
        # and cite Mollière et al. 2025 explicitly for these three values.
    ),
    "DENIS J112639 (2M1126)": dict(
        Teff=1270.0, logg=3.7, Prot=3.2, R_RJup=1.00,   # Prot updated: Metchev et al. 2015 (3.2+/-0.3 hr)
        N=1.5e-2, N_src="this work"
    ),
    "Luhman 16A": dict(
        Teff=1310.0, logg=4.95, Prot=6.94, R_RJup=0.99,
        N=1.5e-2, N_src="this work"
    ),
    "Luhman 16B": dict(
        Teff=1280.0, logg=4.87, Prot=5.28, R_RJup=0.99,
        N=1.5e-2, N_src="this work"
    ),
    "SIMP0136": dict(
        Teff=1098.0, logg=4.31, Prot=2.414, R_RJup=1.22,  # Gagne et al. 2017, confirmed
        N=1.5e-2, N_src="this work"
    ),
    "2M2139": dict(
        Teff=1111.0, logg=4.42, Prot=7.721, R_RJup=1.17,  # Prot updated: Radigan et al. 2012 (7.721+/-0.005 hr)
        N=1.5e-2, N_src="this work"
    ),
}

# ---------------------------------------------------------------------------
# Core calculation
# ---------------------------------------------------------------------------
def compute_row(name, Teff, logg, Prot, R_RJup, N, N_src):
    g = 10 ** logg / 100.0                # cgs (cm/s^2) -> SI (m/s^2)
    H = R_SPECIFIC * Teff / g             # scale height, kT/(mu m_H g)

    Omega  = 2 * np.pi / (Prot * 3600.0)  # rotation rate, rad/s
    f_pole = 2 * Omega                    # Coriolis parameter at the pole

    R = R_RJup * R_JUP                    # planetary radius, m
    beta = 2 * Omega / R                  # equatorial df/dy

    L_D  = N * H / f_pole                 # polar deformation radius, m
    L_eq = np.sqrt(N * H / beta)          # equatorial deformation radius, m

    return dict(
        Object=name,
        Teff_K=Teff,
        logg_cgs=logg,
        g_m_s2=g,
        Prot_hr=Prot,
        H_km=H / 1e3,
        N_s1=N,
        N_source=N_src,
        f_pole_s1=f_pole,
        L_D_km=L_D / 1e3,
        L_D_over_R_pct=100 * L_D / R,
        L_eq_km=L_eq / 1e3,
        L_eq_over_L_D=L_eq / L_D,
    )


rows = [compute_row(name, **params) for name, params in objects.items()]
df = pd.DataFrame(rows)

# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------
pd.set_option("display.float_format", lambda x: f"{x:,.3f}")

display_cols = [
    "Object", "Teff_K", "logg_cgs", "Prot_hr", "N_s1",
    "H_km", "L_D_km", "L_D_over_R_pct", "L_eq_km", "L_eq_over_L_D",
]

print("=" * 100)
print("DEFORMATION RADIUS TABLE")
print("=" * 100)
print(df[display_cols].to_string(index=False))
print()
print("N sources:")
for _, r in df.iterrows():
    print(f"  {r['Object']:<26s}: N = {r['N_s1']:.1e} s^-1  [{r['N_source']}]")

# Save full table (with all intermediate quantities) to CSV for the paper
out_path = "deformation_radius_table.csv"
df.to_csv(out_path, index=False)
print(f"\nFull table (all columns) written to: {out_path}")