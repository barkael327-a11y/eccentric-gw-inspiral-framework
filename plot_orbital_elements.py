import numpy as np
import matplotlib.pyplot as plt

def main():
    data = np.load("default_e04_run.npz")
    t = data["ts"]
    a = data["a"]
    e = data["e"].astype(float)

    # a(t)
    plt.figure(figsize=(9, 4))
    plt.plot(t, a, label="a(t) (osculating)")
    plt.xlabel("time (s)")
    plt.ylabel("semi-major axis a(t) [m]")
    plt.title("Semi-major axis decay (osculating a(t))")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # e(t) with cleanup
    e = np.nan_to_num(e, nan=0.0, posinf=0.0, neginf=0.0)
    e = np.clip(e, 0.0, 0.999999)

    plt.figure(figsize=(9, 4))
    plt.plot(t, e, label="e(t) (osculating)")
    plt.xlabel("time (s)")
    plt.ylabel("eccentricity e(t)")
    plt.title("Circularization: osculating e(t)")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
