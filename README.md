# personal-site

A simple personal site built using [mvp.css](https://andybrewer.github.io/mvp/).

To see a live example of the site using the code in this repo please take a look [here](https://radekosmulski.github.io/personal-site/).

To build your own:

1. Fork this repository
2. In the fork, go to settings > options > GitHub pages and enable it

You can find the site I built for myself [here](https://www.radekosmulski.com/).

## THERML Ising Model Implementation

This repository includes a simple implementation of an Ising model inspired by [THERML](https://github.com/extropic-ai/thrml), a library for probabilistic computing with energy-based models.

### Files

- **`therml_simple.py`** - Python implementation of the simplest Ising model with Gibbs sampling
  - Binary spins {-1, +1}
  - Energy-based probability: P(x) ∝ e^(-βE(x))
  - Hardware-friendly Gibbs sampling
  - Examples: chain and fully-connected topologies

- **`therml_demo.html`** - Interactive browser-based demo
  - Real-time visualization of Gibbs sampling
  - Adjustable parameters (number of spins, temperature, speed)
  - Energy tracking and magnetization display

### Running the Demo

**Python version:**
```bash
python therml_simple.py
```

**Browser version:**
Open `therml_demo.html` in your web browser.

### Key Concepts

- **Energy-Based Models**: Probability distribution defined by energy function
- **Gibbs Sampling**: Sequential spin updates based on local field (hardware-friendly)
- **Temperature Control**: β (inverse temperature) controls determinism vs randomness
- **Binary Spins**: Simple {-1, +1} representation suitable for hardware implementation

Based on the spin models example from [THERML](https://github.com/extropic-ai/thrml/blob/main/examples/02_spin_models.ipynb).
