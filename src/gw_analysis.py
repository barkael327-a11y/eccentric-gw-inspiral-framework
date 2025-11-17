# src/gw_analysis.py

import math
import numpy as np

def resample_uniform(t, x, fs):
    """Linear-resample irregular (t,x) to uniform grid at sample rate fs."""
    t = np.asarray(t, dtype=float)
    x = np.asarray(x, dtype=float)
    if t.ndim != 1 or x.ndim != 1 or len(t) != len(x) or len(t) < 2:
        raise ValueError("resample_uniform: t and x must be 1D, same length, len>=2")
    t0, t1 = float(t[0]), float(t[-1])
    if not np.isfinite(fs) or fs <= 0:
        raise ValueError("resample_uniform: fs must be positive")
    N = int(max(2, math.floor((t1 - t0) * fs) + 1))
    tu = t0 + np.arange(N) / fs
    xu = np.interp(tu, t, x)
    return tu, xu

def fft_asd(t, x, fs=8192.0, window="hann"):
    """
    One-sided amplitude spectral density ASD(f) of x(t).
    Returns (f, asd).
    """
    tu, xu = resample_uniform(t, x, fs)

    if window == "hann":
        w = np.hanning(len(xu))
    elif window in (None, "rect"):
        w = np.ones_like(xu)
    else:
        raise ValueError("fft_asd: window must be 'hann', 'rect', or None")

    win_rms = np.sqrt(np.mean(w**2))
    X = np.fft.rfft(xu * w)
    f = np.fft.rfftfreq(len(xu), d=1.0/fs)

    # One-sided ASD (per √Hz), with window RMS correction
    asd = (np.abs(X) / (len(xu) * win_rms)) * np.sqrt(2.0 / fs)
    return f, asd

def spectrogram(t, x, fs, nperseg=2048, noverlap=1536):
    """
    Simple STFT spectrogram using Hann windows.
    Returns (F, T, S) with S(f,t) magnitude.
    """
    tu, xu = resample_uniform(t, x, fs)
    if nperseg <= 1 or noverlap >= nperseg:
        raise ValueError("spectrogram: require nperseg>1 and noverlap<nperseg")

    step = nperseg - noverlap
    w = np.hanning(nperseg)
    win_rms = np.sqrt(np.mean(w**2))

    cols = max(1, (len(xu) - nperseg) // step + 1)
    Scols, Tm = [], []
    for k in range(cols):
        i0 = k * step
        seg = xu[i0:i0 + nperseg]
        if len(seg) < nperseg:
            break
        X = np.fft.rfft(seg * w)
        Scols.append(np.abs(X) / (nperseg * win_rms))
        Tm.append(tu[i0 + nperseg//2])

    S = np.array(Scols, dtype=float).T  # shape: (freq, time)
    F = np.fft.rfftfreq(nperseg, d=1.0/fs)
    T = np.array(Tm, dtype=float)
    if T.size:
        T = T - T[0]
    return F, T, S
