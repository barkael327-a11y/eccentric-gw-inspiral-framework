import numpy as np
import matplotlib.pyplot as plt
from src.gw_analysis import fft_asd

def main():
    data = np.load("default_e04_run.npz")
    t  = data["ts"]
    hp = data["hps"]

    fs = 512.0
    f, A = fft_asd(t, hp, fs=fs)

    plt.figure(figsize=(8, 4))
    plt.loglog(f[1:], A[1:])
    plt.xlabel("frequency (Hz)")
    plt.ylabel(r"ASD $|\tilde h(f)|/\sqrt{\mathrm{Hz}}$")
    plt.title("ASD of h+")
    plt.grid(which="both", alpha=0.3)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
