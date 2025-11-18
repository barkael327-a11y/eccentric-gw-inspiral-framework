# examples/run_default_sim.py

import numpy as np
from src.gw_inspiral import simulate

#---Set mass, eccentricity, and semi-major axis parameters here---
#NOTE: I defaulted to component masses of 10 and 20 solar masses for my research paper.

def main():
    m_sun = 1.98847e30
    m1 = 10.0 * m_sun
    m2 = 20.0 * m_sun
    a0 = 1.0e6
    e0 = 0.4
    f0 = 0.3
    R  = 500e6 * 3.085677581e16  # 500 Mpc
    f_gw_max = 1024.0
    steps_per_orbit = 600
    max_steps = 2_000_000

    results = simulate(
        m1, m2, a0, R,
        e0=e0, f0=f0,
        f_gw_max=f_gw_max,
        steps_per_orbit=steps_per_orbit,
        max_steps=max_steps,
    )

    if len(results["ts"]) < 5:
        raise RuntimeError("Run ended too quickly. Try larger a0 or lower f_gw_max.")

    np.savez("default_e04_run.npz", **results)
    print("Saved default_e04_run.npz with keys:", list(results.keys()))

if __name__ == "__main__":
    main()
