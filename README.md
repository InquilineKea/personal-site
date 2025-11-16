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

- **`therml_demo.html`** - Simple interactive browser-based demo
  - Real-time visualization of Gibbs sampling
  - Adjustable parameters (number of spins, temperature, speed)
  - Energy tracking and magnetization display

- **`therml_full.html`** - Complete THRML implementation in the browser
  - Multiple graph topologies (grid, ring, random, complete)
  - Graph coloring for parallel Gibbs sampling (DSATUR algorithm)
  - Sampling schedules with warmup, thinning, and sample collection
  - Visual representation of graph structure with color-coded blocks
  - Real-time energy evolution tracking
  - Sample statistics and logging

### Running the Demo

**Python version:**
```bash
python therml_simple.py
```

**Browser version (simple):**
Open `therml_demo.html` in your web browser.

**Browser version (full THRML implementation):**
Open `therml_full.html` in your web browser.

### Key Concepts

- **Energy-Based Models**: Probability distribution defined by energy function P(x) ∝ e^(-βE(x))
- **Gibbs Sampling**: Sequential spin updates based on local field (hardware-friendly)
- **Temperature Control**: β (inverse temperature) controls determinism vs randomness
- **Binary Spins**: Simple {-1, +1} representation suitable for hardware implementation
- **Graph Coloring**: Identifies independent sets of nodes that can be updated in parallel
- **Sampling Schedule**: Warmup period + sample collection with thinning to reduce correlation

### Key THRML Features Implemented

The full browser implementation (`therml_full.html`) includes the core concepts from the THRML notebook:

1. **Graph-Based Spin Models**: Multiple topologies (grid, ring, random, complete graph)
2. **Greedy Graph Coloring**: DSATUR-inspired algorithm to identify parallel update blocks
3. **Block-Based Sampling**: Nodes with same color updated in parallel (hardware-efficient!)
4. **Sampling Schedules**: Configurable warmup, sample count, and thinning parameters
5. **Energy Tracking**: Real-time visualization of energy evolution during sampling
6. **Visual Feedback**: Graph visualization showing node colors and current spin states

Based on the spin models example from [THRML](https://github.com/extropic-ai/thrml/blob/main/examples/02_spin_models.ipynb).
