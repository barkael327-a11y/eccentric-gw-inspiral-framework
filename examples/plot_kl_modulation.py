from src.gw_inspiral import simulate
from src.kl_modulation import apply_KL_modulation


def main():
    # ---- load base two-body run ----
    data = np.load("default_e04_run.npz")

    sim = {
        "ts":  data["ts"],
        "hps": data["hps"],
        "hxs": data["hxs"],
        "e":   data["e"],
    }

    # ---- impose Kozai–Lidov modulation ----
    res_KL = apply_KL_modulation(
        sim,
        i0_deg=60.0,     # initial inclination (deg)
        e_min=0.4,       # min eccentricity during KL cycle
        e_max=0.75,       # max eccentricity during KL cycle
        T_KL=None,       # one KL cycle over full simulation duration
        include_ecc_boost=True,
    )

    t   = sim["ts"]
    hp  = sim["hps"]
    hx  = sim["hxs"]
    hpK = res_KL["hps_KL"]
    hxK = res_KL["hxs_KL"]

    # ---- zoom to last 0.25 s with a shared time axis ----
    mask   = t > (t[-1] - 0.25)
    t_zoom = t[mask] - t[mask][0]

    plt.figure(figsize=(9, 3))
    # two-body eccentric waveform
    plt.plot(t_zoom, hp[mask],  label="h+ (two-body)", lw=1.2)
    plt.plot(t_zoom, hx[mask],  label="hx (two-body)", lw=1.2, alpha=0.7)

    # KL–modulated waveform
    plt.plot(t_zoom, hpK[mask], label="h+ (KL-modulated)", ls="--", lw=1.2)
    plt.plot(t_zoom, hxK[mask], label="hx (KL-modulated)", ls="--", lw=1.2, alpha=0.7)

    plt.xlabel("time (s)")
    plt.ylabel("strain")
    plt.title("Eccentric vs KL–modulated waveform (last 0.25 s)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
