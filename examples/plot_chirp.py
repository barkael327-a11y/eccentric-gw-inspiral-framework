import numpy as np
import matplotlib.pyplot as plt

def main():
    data = np.load("default_e04_run.npz")
    t   = data["ts"]
    fgw = data["fgw"]

    plt.figure(figsize=(10, 3))
    plt.plot(t, fgw)
    plt.xlabel("time (s)")
    plt.ylabel("f_gw (Hz)")
    plt.title("GW frequency track (chirp)")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
