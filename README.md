# eccentric-gw-inspiral-framework
See README for brief description.

# Eccentric GW Inspiral Framework

A lightweight numerical framework for evolving **eccentric compact binaries** under
Newtonian orbital dynamics with **Peters–Mathews radiation reaction** and generating
gravitational-wave **quadrupole waveforms**.  
The code uses an **adaptive RK4 integrator** with eccentricity-aware time-stepping to
resolve rapid pericenter motion while keeping low-eccentricity inspirals fast.

This project supports studies of:
- Eccentric binary inspirals
- Time–frequency GW morphology (amplitude evolution, chirp structure, spectrograms)
- Comparisons between numerical waveforms and analytic template-based waveforms
- KL-driven binaries injected from hierarchical triple dynamics

---

## Features

- **Adaptive RK4 integrator** with dynamic time-stepping based on eccentricity.
- **Newtonian orbital dynamics** with stable simulations down to the Schwarzschild ISCO.
- **Peters–Mathews radiation reaction** implemented via an effective drag force along the velocity.
- **Quadrupole GW polarizations** (h₊, h×) for an observer along the +z axis.
- **Automatic extraction of osculating orbital elements** (a(t), e(t)).
- **Waveform tools**:
  - Time-domain waveform generation
  - Amplitude envelopes
  - ASD (amplitude spectral density)
  - Spectrograms and instantaneous frequency tracks
- **Supports eccentric initial conditions** (a₀, e₀, f₀).

---

## Quick Start

```bash
git clone https://github.com/barkael327-a11y/eccentric-gw-inspiral-framework.git
cd eccentric-gw-inspiral-framework
