def apply_KL_modulation(sim,
                    i0_deg=60.0,
                    e_min=None,
                    e_max=0.9,
                    T_KL=None,
                    include_ecc_boost=True):

ts  = np.asarray(sim["ts"])
hp  = np.asarray(sim["hps"])
hx  = np.asarray(sim["hxs"])
e_series = np.asarray(sim["e"])

if ts.size == 0:
    raise ValueError("apply_KL_modulation: empty time array in sim['ts'].")

e0 = float(e_series[0])
if e_min is None:
    e_min = e0
if e_max <= e_min:
    raise ValueError("apply_KL_modulation: require e_max > e_min.")

# KL period: default to the full simulation duration
if T_KL is None:
    T_KL = ts[-1] - ts[0]
    if T_KL <= 0.0:
        raise ValueError("apply_KL_modulation: non-positive total duration.")

# Initial inclination (radians) and Kozai constant
i0 = np.deg2rad(i0_deg)
C  = np.sqrt(max(1.0 - e0*e0, 0.0)) * np.cos(i0)

# --- Simple sinusoidal KL eccentricity between e_min and e_max ---
e_mid = 0.5 * (e_min + e_max)
amp_e = 0.5 * (e_max - e_min)

phase = 2.0 * np.pi * (ts - ts[0]) / T_KL
e_KL  = e_mid + amp_e * np.sin(phase)
e_KL  = np.clip(e_KL, 0.0, 0.999)

# --- Inclination from Kozai constant: C = sqrt(1-e^2) * cos(i) ---
denom = np.sqrt(np.maximum(1.0 - e_KL*e_KL, 1e-6))
cos_i = C / denom
cos_i = np.clip(cos_i, -1.0, 1.0)
i_KL  = np.arccos(cos_i)

# --- Geometric beaming for + and x polarisations ---
# h+ ∝ (1 + cos^2 i),  h× ∝ 2 cos i
Fp0 = 1.0 + np.cos(i0)**2
Fx0 = 2.0 * np.cos(i0)

Fp_t = 1.0 + cos_i**2
Fx_t = 2.0 * cos_i

geom_plus  = Fp_t / Fp0
if abs(Fx0) > 1e-6:
    geom_cross = Fx_t / Fx0
else:
    geom_cross = np.ones_like(Fx_t)

# --- Eccentricity-dependent amplitude boost ---
if include_ecc_boost:
    Se0 = 1.0 / (1.0 - e0*e0 + 1e-6)
    Se_t = 1.0 / (1.0 - e_KL*e_KL + 1e-6)
    ecc_amp = Se_t / Se0
else:
    ecc_amp = 1.0

envelope = ecc_amp

hp_KL = hp * geom_plus  * envelope
hx_KL = hx * geom_cross * envelope

sim_mod = dict(sim)
sim_mod["hps_KL"]      = hp_KL
sim_mod["hxs_KL"]      = hx_KL
sim_mod["e_KL"]        = e_KL
sim_mod["i_KL"]        = i_KL
sim_mod["KL_envelope"] = geom_plus * envelope

return sim_mod
