import numpy as np
import matplotlib.pyplot as plt
from src.gw_analysis import spectrogram

def main():
    data = np.load("default_e04_run.npz")
    t  = data["ts"]
    hp = data["hps"]

    fs = 512.0
    nperseg, noverlap = 4096, 3072

    F, T, S = spectrogram(t, hp, fs=fs, nperseg=nperseg, noverlap=noverlap)

    plt.figure(figsize=(9, 4))
    plt.pcolormesh(T, F, 20*np.log10(S + 1e-20),
                   shading="auto", cmap="viridis")
    plt.ylim(0, fs/16)
    plt.xlabel("time (s)")
    plt.ylabel("frequency (Hz)")
    plt.title("Spectrogram of h+ (dB)")
    plt.colorbar(label="amplitude [dB]")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
