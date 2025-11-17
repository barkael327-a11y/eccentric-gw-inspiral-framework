import numpy as np
import matplotlib.pyplot as plt

def main():
    data = np.load("default_e04_run.npz")
    t  = data["ts"]
    hp = data["hps"]
    hx = data["hxs"]

    mask = t > (t[-1] - 0.25)
    t_zoom = t[mask] - t[mask][0]

    plt.figure(figsize=(9,3))
    plt.plot(t_zoom, hp[mask], label="h+")
    plt.plot(t_zoom, hx[mask], label="hx", alpha=0.85)
    plt.xlabel("time (s)")
    plt.ylabel("strain")
    plt.title("Waveform (last 0.25 s)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
