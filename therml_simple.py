"""
Simplest THERML Implementation - Ising Model
Based on: https://github.com/extropic-ai/thrml/blob/main/examples/02_spin_models.ipynb

This is a minimal implementation of an Ising model (Boltzmann machine) using
energy-based probabilistic modeling, inspired by THERML.

THERML Core Concept:
- Binary spins with values {-1, 1}
- Energy function: E(x) = -sum(biases * spins) - sum(weights * spin_i * spin_j)
- Probability: P(x) ∝ e^(-β*E(x)) where β is inverse temperature
- Sampling: Gibbs sampling (hardware-friendly)
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple


class SimpleIsingModel:
    """
    Simplest Ising model with biases and pairwise interactions.

    Energy: E(x) = -sum_i(bias_i * x_i) - sum_{i,j}(weight_{ij} * x_i * x_j)
    Probability: P(x) ∝ exp(-β * E(x))
    """

    def __init__(self, n_spins: int, edges: List[Tuple[int, int]],
                 biases: np.ndarray = None, weights: np.ndarray = None,
                 beta: float = 1.0):
        """
        Args:
            n_spins: Number of spin variables
            edges: List of (i, j) tuples defining connections
            biases: Bias for each spin (default: random)
            weights: Weight for each edge (default: random)
            beta: Inverse temperature (higher = more deterministic)
        """
        self.n_spins = n_spins
        self.edges = edges
        self.beta = beta

        # Initialize parameters randomly if not provided
        if biases is None:
            biases = np.random.randn(n_spins) * 0.5
        if weights is None:
            weights = np.random.randn(len(edges)) * 0.5

        self.biases = biases
        self.weights = weights

    def energy(self, state: np.ndarray) -> float:
        """Calculate energy of a state."""
        # Bias term
        bias_energy = -np.dot(self.biases, state)

        # Interaction term
        interaction_energy = 0.0
        for (i, j), w in zip(self.edges, self.weights):
            interaction_energy -= w * state[i] * state[j]

        return bias_energy + interaction_energy

    def local_field(self, state: np.ndarray, spin_idx: int) -> float:
        """Calculate local field at a spin (sum of neighbor influences)."""
        field = self.biases[spin_idx]

        for (i, j), w in zip(self.edges, self.weights):
            if i == spin_idx:
                field += w * state[j]
            elif j == spin_idx:
                field += w * state[i]

        return field

    def gibbs_step(self, state: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """
        One Gibbs sampling step - update each spin sequentially.
        This is hardware-friendly as it only requires:
        1. Computing local field (sum of neighbor states)
        2. Generating biased random bit
        """
        new_state = state.copy()

        for i in range(self.n_spins):
            # Calculate probability of spin being +1
            h = self.local_field(new_state, i)
            prob_up = 1.0 / (1.0 + np.exp(-2 * self.beta * h))

            # Sample new spin value
            new_state[i] = 1 if rng.random() < prob_up else -1

        return new_state

    def sample(self, n_samples: int, n_burnin: int = 100,
               n_thin: int = 10, seed: int = 42) -> np.ndarray:
        """
        Generate samples using Gibbs sampling.

        Args:
            n_samples: Number of samples to generate
            n_burnin: Number of initial steps to discard
            n_thin: Keep every n_thin'th sample
            seed: Random seed

        Returns:
            Array of shape (n_samples, n_spins)
        """
        rng = np.random.default_rng(seed)

        # Random initialization
        state = rng.choice([-1, 1], size=self.n_spins)

        # Burn-in
        for _ in range(n_burnin):
            state = self.gibbs_step(state, rng)

        # Collect samples
        samples = []
        for _ in range(n_samples * n_thin):
            state = self.gibbs_step(state, rng)
            if len(samples) % n_thin == 0:
                samples.append(state.copy())

        return np.array(samples[:n_samples])


def create_chain_model(n_spins: int = 5, beta: float = 1.0) -> SimpleIsingModel:
    """Create a simple chain Ising model (1D lattice)."""
    edges = [(i, i+1) for i in range(n_spins - 1)]
    return SimpleIsingModel(n_spins, edges, beta=beta)


def create_fully_connected_model(n_spins: int = 4, beta: float = 1.0) -> SimpleIsingModel:
    """Create a fully connected Ising model."""
    edges = [(i, j) for i in range(n_spins) for j in range(i+1, n_spins)]
    return SimpleIsingModel(n_spins, edges, beta=beta)


def visualize_samples(samples: np.ndarray, title: str = "Ising Model Samples"):
    """Visualize spin samples as a heatmap."""
    plt.figure(figsize=(12, 4))
    plt.imshow(samples.T, aspect='auto', cmap='RdBu', vmin=-1, vmax=1)
    plt.colorbar(label='Spin value')
    plt.xlabel('Sample index')
    plt.ylabel('Spin index')
    plt.title(title)
    plt.tight_layout()
    return plt


if __name__ == "__main__":
    print("=" * 60)
    print("Simplest THERML Implementation - Ising Model Demo")
    print("=" * 60)

    # Example 1: Simple chain
    print("\n1. Chain Model (5 spins in a line)")
    print("-" * 40)
    model = create_chain_model(n_spins=5, beta=2.0)
    print(f"Biases: {model.biases}")
    print(f"Weights: {model.weights}")
    print(f"Edges: {model.edges}")

    # Sample from the model
    samples = model.sample(n_samples=100, seed=42)
    print(f"\nGenerated {len(samples)} samples")
    print(f"Sample mean magnetization: {samples.mean(axis=0)}")

    # Example state and its energy
    example_state = samples[0]
    print(f"\nExample state: {example_state}")
    print(f"Energy: {model.energy(example_state):.3f}")

    # Example 2: Fully connected
    print("\n\n2. Fully Connected Model (4 spins)")
    print("-" * 40)
    model2 = create_fully_connected_model(n_spins=4, beta=1.5)
    print(f"Number of edges: {len(model2.edges)}")

    samples2 = model2.sample(n_samples=100, seed=43)
    print(f"Sample mean magnetization: {samples2.mean(axis=0)}")

    # Demonstrate temperature effect
    print("\n\n3. Temperature Effect")
    print("-" * 40)
    for beta in [0.1, 1.0, 5.0]:
        model_temp = create_chain_model(n_spins=5, beta=beta)
        # Use same parameters for fair comparison
        model_temp.biases = np.ones(5) * 0.5
        model_temp.weights = np.ones(4) * 0.3

        samples_temp = model_temp.sample(n_samples=200, seed=44)
        avg_mag = np.abs(samples_temp.mean())

        print(f"β={beta:.1f} (T={1/beta:.1f}): avg |magnetization|={avg_mag:.3f}")

    print("\n" + "=" * 60)
    print("Key THERML Concepts Demonstrated:")
    print("  • Energy-based model: P(x) ∝ exp(-β*E(x))")
    print("  • Gibbs sampling (hardware-friendly)")
    print("  • Binary spins {-1, 1}")
    print("  • Biases and pairwise interactions")
    print("  • Temperature control via β")
    print("=" * 60)
