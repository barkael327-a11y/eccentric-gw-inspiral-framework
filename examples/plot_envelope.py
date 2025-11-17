import numpy as np
import matplotlib.pyplot as plt

def plot_envelope_binned(t, h, bins=400, last_seconds=None,
                         title="Gravitational-wave amplitude envelope"):
    t_arr = np.asarray(t)
    h_abs = np.abs(np.asarray(h))
    if len(t_arr) == 0:
        return

    if last_seconds is not None:
        tmin = max(t_arr[0], t_arr[-1] - float(last_seconds))
        mask = t_arr >= tmin
        t_arr = t_arr[mask]
        h_abs = h_abs[mask]
        if len(t_arr) == 0:
            return

    edges = np.linspace(t_arr[0], t_arr[-1], bins+1)
    idx   = np.searchsorted(t_arr, edges)
    idx   = np.clip(idx, 0, len(t_arr))
    t_bin, h_bin = [], []

    for i in range(len(edges)-1):
        lo, hi = idx[i], idx[i+1]
        if hi > lo:
            t_bin.append(0.5*(edges[i] + edges[i+1]))
            h_bin.append(h_abs[lo:hi].max())

    t_bin = np.asarray(t_bin)
    h_bin = np.asarray(h_bin)

    plt.figure(figsize=(10, 4))
    plt.plot(t_bin, h_bin)
    plt.xlabel("time (s)")
    plt.ylabel("|h₊| (envelope, binned max)")
    plt.title(title)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

def main():
    data = np.load("default_e04_run.npz")
    t = data["ts"]
    hp = data["hps"]
    plot_envelope_binned(
        t, hp,
        bins=500,
        last_seconds=5.0,
        title="Gravitational-wave amplitude envelope (eccentric RR)",
    )

if __name__ == "__main__":
    main()
