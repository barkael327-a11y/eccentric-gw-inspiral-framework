# src/gw_inspiral.py

import math
import numpy as np

G = 6.67430e-11
c = 2.99792458e8

# ==== helpers ====
def integration_profile(e):
    """
    Dynamic integrator settings by current eccentricity.
    Designed so e~0.4 has enough resolution near pericenter,
    while low-e runs stay fast.
    """
    if e < 0.30:
        return dict(steps_per_orbit=220, peri_dt_factor=0.10, safety=6.0, record_every=5)
    if e < 0.50:   # e ~ 0.4 lands here
        return dict(steps_per_orbit=500, peri_dt_factor=0.05, safety=8.0, record_every=3)
    if e < 0.75:
        return dict(steps_per_orbit=700, peri_dt_factor=0.035, safety=9.0, record_every=2)
    return dict(steps_per_orbit=950, peri_dt_factor=0.025, safety=10.0, record_every=1)

def isco_radius(m1, m2):
    """Schwarzschild ISCO for total mass m1+m2."""
    return 6.0 * G * (m1 + m2) / c**2

# ==== integrator ====
def rk4_step(fun, t, y, dt):
    k1 = fun(t, y)
    k2 = fun(t + 0.5*dt, y + 0.5*dt*k1)
    k3 = fun(t + 0.5*dt, y + 0.5*dt*k2)
    k4 = fun(t + dt,     y + dt*k3)
    return y + (dt/6.0)*(k1 + 2*k2 + 2*k3 + k4)

# ==== dynamics helpers ====
def init_circular_xy(a0, m1, m2):
    M  = m1 + m2
    r0 = np.array([a0, 0.0], dtype=float)
    v0 = np.array([0.0, math.sqrt(G*M/a0)], dtype=float)
    return r0, v0

def init_elliptic(a0, e0, f0, m1, m2):
    """Eccentric Keplerian initial conditions in the orbital plane (x–y)."""
    if not (0.0 <= e0 < 1.0):
        raise ValueError("eccentricity e0 must be in [0,1)")
    M = m1 + m2
    p  = a0 * (1.0 - e0*e0)                    # semi-latus rectum
    r  = p / (1.0 + e0 * math.cos(f0))         # separation
    h  = math.sqrt(G * M * a0 * (1.0 - e0*e0)) # specific angular momentum
    cf, sf = math.cos(f0), math.sin(f0)
    r_vec  = np.array([r*cf, r*sf], dtype=float)
    v_vec  = (G*M / h) * np.array([-sf, e0 + cf], dtype=float)
    return r_vec, v_vec

def accel_newton(r, m1, m2):
    M  = m1 + m2
    r2 = float(r[0]*r[0] + r[1]*r[1])
    if r2 == 0.0:
        return np.zeros_like(r)
    r3 = r2**1.5
    return -G*M * r / r3

# --- osculating (a,e) from instantaneous r,v (2D) ---
def osculating_elements(r, v, M):
    """
    ε = v^2/2 - GM/r,  h = |r×v| (z-component in 2D)
    e = sqrt(1 + 2 ε h^2 /(G^2 M^2)),  a = -GM/(2ε) (if ε<0 else inf)
    """
    r = np.asarray(r, dtype=float)
    v = np.asarray(v, dtype=float)
    r_mag = float(np.linalg.norm(r))
    v_mag = float(np.linalg.norm(v))
    if r_mag == 0.0:
        return np.inf, np.nan
    eps = 0.5 * v_mag**2 - G*M / r_mag
    h   = abs(r[0]*v[1] - r[1]*v[0])
    e_sq = 1.0 + 2.0 * eps * h**2 / (G**2 * M**2)
    e = math.sqrt(max(e_sq, 0.0))
    a = (-G*M / (2.0*eps)) if eps < 0.0 else np.inf
    return a, e

# --- Peters RR using osculating (a,e); loss mapped along v ---
def accel_rr_peters_ecc_safe(r, v, m1, m2):
    M  = m1 + m2
    mu = m1*m2 / M
    r_mag = float(np.linalg.norm(r))
    v2    = float(np.dot(v, v))
    if r_mag == 0.0 or v2 == 0.0:
        return np.zeros_like(r)

    a_oscul, e_oscul = osculating_elements(r, v, M)
    e = float(np.clip(e_oscul, 0.0, 0.999))  # avoid e→1 singularity
    a = float(max(a_oscul, r_mag))           # avoid tiny/negative a
    one_me2 = max(1.0 - e*e, 1e-6)

    F_e = (1 + 73*e*e/24 + 37*e**4/96) / (one_me2**3.5)   # (1-e^2)^(7/2)
    dE_dt = -(32.0/5.0) * G**4 * mu**2 * M**3 / (c**5 * a**5) * F_e

    # (mu a_rr)·v = dE/dt  =>  a_rr parallel to v
    a_rr = (dE_dt / (mu * max(v2, 1e-20))) * v

    # cap RR for stability
    aN = np.linalg.norm(accel_newton(r, m1, m2)) + 1e-30
    a_rr_norm = np.linalg.norm(a_rr)
    if a_rr_norm > 1e-3 * aN:
        a_rr *= (1e-3 * aN) / a_rr_norm
    return a_rr

def rhs_cartesian_total(t, y, m1, m2):
    x, y_, vx, vy = y
    r = np.array([x, y_], dtype=float)
    v = np.array([vx, vy], dtype=float)
    a = accel_newton(r, m1, m2) + accel_rr_peters_ecc_safe(r, v, m1, m2)
    return np.array([v[0], v[1], a[0], a[1]], dtype=float)

# --- Quadrupole (analytic ddQ) and polarizations (observer +z) ---
def quadrupole_ddot(r, v, a, mu):
    x, y = float(r[0]), float(r[1])
    vx, vy = float(v[0]), float(v[1])
    ax, ay = float(a[0]), float(a[1])
    v2 = vx*vx + vy*vy
    rdota = x*ax + y*ay
    d2Q_xx = mu*(ax*x + x*ax + 2*vx*vx) - (2.0/3.0)*mu*(v2 + rdota)
    d2Q_yy = mu*(ay*y + y*ay + 2*vy*vy) - (2.0/3.0)*mu*(v2 + rdota)
    d2Q_xy = mu*(ax*y + x*ay + 2*vx*vy)
    return d2Q_xx, d2Q_yy, d2Q_xy

def polarisations_from_ddQ(r, v, a, m1, m2, R):
    M  = m1 + m2
    mu = m1*m2 / M
    d2Q_xx, d2Q_yy, d2Q_xy = quadrupole_ddot(r, v, a, mu)
    h_plus  = (G/(c**4 * R)) * (d2Q_xx - d2Q_yy)
    h_cross = (2.0*G/(c**4 * R)) * d2Q_xy
    return h_plus, h_cross

# ==== single-pass simulation (adaptive; no plotting) ====
def simulate(m1, m2, a0, R, e0=0.0, f0=0.0,
             f_gw_max=1024.0, steps_per_orbit=600, max_steps=2_000_000):
    """
    Adaptive inspiral with eccentricity-aware step control.
    Returns dict with time series (decimated as per 'record_every').
    """
    M = m1 + m2
    a_ISCO = isco_radius(m1, m2)

    # initial profile from requested e0; scale by user 'steps_per_orbit'
    prof = integration_profile(float(e0))
    prof['steps_per_orbit'] = max(50, int(prof['steps_per_orbit'] * (steps_per_orbit/600.0)))
    record_every = int(prof['record_every'])

    # ensure pericenter safely above ISCO for high e0
    if e0 > 0.0:
        e_safe = min(e0, 0.999)
        a0_min = (prof['safety'] * a_ISCO) / (1.0 - e_safe)
        if a0 < a0_min:
            print(f"[note] a0 raised {a0:.3e} -> {a0_min:.3e} so r_p ≥ {prof['safety']} r_ISCO for e0={e0:.2f}")
            a0 = a0_min

    # initialize state
    if e0 > 0.0:
        r, v = init_elliptic(a0, e0, f0, m1, m2)
    else:
        r, v = init_circular_xy(a0, m1, m2)
    y = np.array([r[0], r[1], v[0], v[1]], dtype=float)
    t = 0.0

    # initial dt from P(a0) and a pericenter timescale
    omega0 = math.sqrt(G * M / a0**3)
    P0 = 2.0 * math.pi / omega0
    r_peri0 = a0 * (1.0 - e0) if e0 > 0.0 else a0
    omega_peri0 = math.sqrt(G * M / max(r_peri0, 1.1*a_ISCO)**3)
    dt = min(P0 / prof['steps_per_orbit'], prof['peri_dt_factor'] / omega_peri0)

    # storage
    ts, hps, hxs, fgw = [], [], [], []
    a_hist, e_hist = [], []
    rec_stride = []
    it = 0

    # main loop
    while it < max_steps:
        x, y_, vx, vy = y
        r = np.array([x, y_], dtype=float)
        v = np.array([vx, vy], dtype=float)

        rn = float(np.linalg.norm(r))
        if rn == 0.0:
            break

        # frequency proxy (2 f_orb)
        omega = math.sqrt(G * M / (rn**3))
        f_inst = omega / math.pi

        # termination conditions
        if rn <= a_ISCO or f_inst >= f_gw_max:
            break

        # total acceleration
        a_vec = accel_newton(r, m1, m2) + accel_rr_peters_ecc_safe(r, v, m1, m2)

        # osculating elements for logging & control
        a_oscul, e_oscul = osculating_elements(r, v, M)
        e_now = float(np.clip(e_oscul, 0.0, 0.999))

        # record (decimated)
        if it % record_every == 0:
            hp, hx = polarisations_from_ddQ(r, v, a_vec, m1, m2, R)
            ts.append(t); hps.append(hp); hxs.append(hx); fgw.append(f_inst)
            a_hist.append(a_oscul); e_hist.append(e_now)
            rec_stride.append(record_every)

        # occasionally refresh the profile if e-regime changed
        if it % 50 == 0:
            new_prof = integration_profile(e_now)
            new_prof['steps_per_orbit'] = max(50, int(new_prof['steps_per_orbit'] * (steps_per_orbit/600.0)))
            if (new_prof['steps_per_orbit'] != prof['steps_per_orbit'] or
                new_prof['peri_dt_factor']  != prof['peri_dt_factor']):
                prof = new_prof
                record_every = int(prof['record_every'])

        # adaptive timestep
        P = 2.0 * math.pi / omega
        local_dyn = rn / (np.linalg.norm(v) + 1e-30)
        dt_target = min(P/prof['steps_per_orbit'], prof['peri_dt_factor'] * local_dyn)

        SPO_min, SPO_target = 120.0, 350.0
        dt_spo_min = P / SPO_min
        dt_spo_tgt = P / SPO_target
        dt_target  = min(dt_target, dt_spo_tgt)
        dt_floor   = min(dt_spo_min, dt_target)

        if np.isfinite(a_oscul) and a_oscul > 0.0:
            rp_est = a_oscul * max(1e-6, 1.0 - e_now)
            if rn < 1.5 * rp_est:
                dt_target = min(dt_target, 0.02 * P)
                dt_floor  = min(dt_floor,  0.01 * P)

        dt = max(min(dt_target, dt * 1.25), 0.5 * dt_floor)

        # advance
        y = rk4_step(lambda T, Y: rhs_cartesian_total(T, Y, m1, m2), t, y, dt)
        t += dt
        it += 1

def apply_KL_modulation(sim,
                    i0_deg=60.0,
                    e_min=None,
                    e_max=0.8,
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


    return {
        "ts":  np.array(ts),
        "hps": np.array(hps),
        "hxs": np.array(hxs),
        "fgw": np.array(fgw),
        "a":   np.array(a_hist),
        "e":   np.array(e_hist),
        "rec_stride": np.array(rec_stride),
    }
