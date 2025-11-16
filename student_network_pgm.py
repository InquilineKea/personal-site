"""
Student Network - A Classic PGM Example from Daphne Koller's Textbook

This implements the Student Network, a canonical example from "Probabilistic Graphical
Models: Principles and Techniques" by Daphne Koller and Nir Friedman.

Network Structure:
    Difficulty (D) ──┐
                     ├──> Grade (G) ──> Letter (L)
    Intelligence (I) ┘       │
                             └──> SAT (S)

Variables (represented as binary spins: -1, +1):
- Difficulty (D): Course difficulty (Easy: -1, Hard: +1)
- Intelligence (I): Student intelligence (Low: -1, High: +1)
- Grade (G): Final grade (Low: -1, High: +1)
- SAT (S): SAT score (Low: -1, High: +1)
- Letter (L): Recommendation letter quality (Weak: -1, Strong: +1)

This implementation uses thrml's energy-based model framework with Ising-like
spin interactions to represent the conditional probability distributions.
"""

import jax
import jax.numpy as jnp
import networkx as nx
from thrml import SpinNode
from thrml.block_management import Block
from thrml.block_sampling import sample_states, SamplingSchedule
from thrml.models.ising import IsingEBM, IsingSamplingProgram, hinton_init


def create_student_network():
    """
    Creates the Student Network graphical model using thrml.

    Returns:
        tuple: (ebm, graph, node_labels, node_list, edge_list) where:
            - ebm: IsingEBM configured with the network structure
            - graph: NetworkX graph representing the network
            - node_labels: Dictionary mapping node indices to variable names
            - node_list: List of SpinNode objects
            - edge_list: List of edge tuples
    """

    # Create the graph structure
    # Nodes: 0=Difficulty, 1=Intelligence, 2=Grade, 3=SAT, 4=Letter
    G = nx.DiGraph()
    node_names = ['Difficulty', 'Intelligence', 'Grade', 'SAT', 'Letter']
    node_map = {name: idx for idx, name in enumerate(node_names)}

    # Create SpinNode objects
    n_nodes = len(node_names)
    spin_nodes = [SpinNode() for _ in range(n_nodes)]

    # Add nodes to graph
    for idx in range(n_nodes):
        G.add_node(idx)

    # Add edges representing dependencies
    # Note: For Ising model, we need undirected edges
    edges_directed = [
        (node_map['Difficulty'], node_map['Grade']),
        (node_map['Intelligence'], node_map['Grade']),
        (node_map['Grade'], node_map['SAT']),
        (node_map['Grade'], node_map['Letter']),
    ]
    G.add_edges_from(edges_directed)

    # Create edge list using SpinNode objects
    edge_list = [(spin_nodes[i], spin_nodes[j]) for i, j in edges_directed]

    # Define bias (field) parameters for each node
    # Positive bias encourages spin to be +1, negative encourages -1
    biases = jnp.array([
        0.0,   # Difficulty: neutral prior
        0.3,   # Intelligence: slight bias toward high intelligence
        0.0,   # Grade: neutral (depends heavily on D and I)
        0.2,   # SAT: slight bias toward higher scores
        0.0,   # Letter: neutral (depends on grade)
    ])

    # Define pairwise interaction weights (coupling strengths)
    # These encode the conditional probability relationships
    # Positive weight: nodes prefer to align, Negative: prefer opposite
    weights = jnp.array([
        -0.8,  # Difficulty -> Grade: Hard courses lead to lower grades
        0.9,   # Intelligence -> Grade: High intelligence leads to higher grades
        0.7,   # Grade -> SAT: Higher grades correlate with higher SAT
        0.85,  # Grade -> Letter: Higher grades lead to stronger letters
    ])

    # Temperature parameter (inverse temperature beta = 1/T)
    # Higher beta = lower temperature = sharper distribution
    beta = jnp.array(1.0)

    # Create the Ising EBM
    ebm = IsingEBM(
        nodes=spin_nodes,
        edges=edge_list,
        biases=biases,
        weights=weights,
        beta=beta
    )

    node_labels = {idx: name for idx, name in enumerate(node_names)}

    return ebm, G, node_labels, spin_nodes, edge_list


def setup_sampling_blocks(spin_nodes):
    """
    Sets up block Gibbs sampling configuration.

    For efficient sampling, we partition nodes into blocks that can be
    sampled in parallel. Here we use a simple two-coloring scheme.

    Args:
        spin_nodes: List of SpinNode objects

    Returns:
        tuple: (free_blocks, clamped_blocks) for block Gibbs sampling
    """
    # For this small network, we can use a simple alternating block structure
    # Block 0: Difficulty, Grade, Letter (indices 0, 2, 4)
    # Block 1: Intelligence, SAT (indices 1, 3)

    block_0_nodes = [spin_nodes[i] for i in [0, 2, 4]]
    block_1_nodes = [spin_nodes[i] for i in [1, 3]]

    free_blocks = [Block(block_0_nodes), Block(block_1_nodes)]
    clamped_blocks = []

    return free_blocks, clamped_blocks


def sample_network(ebm, spin_nodes, n_samples=1000, n_warmup=200, steps_per_sample=5,
                   n_chains=1, seed=42):
    """
    Samples from the Student Network using block Gibbs sampling.

    Args:
        ebm: The IsingEBM representing the network
        spin_nodes: List of SpinNode objects
        n_samples: Number of samples to collect
        n_warmup: Number of warmup iterations
        steps_per_sample: Number of Gibbs steps between samples
        n_chains: Number of parallel chains to run
        seed: Random seed

    Returns:
        dict: Contains samples and statistics
    """

    # Create sampling blocks
    free_blocks, clamped_blocks = setup_sampling_blocks(spin_nodes)

    # Create sampling program
    prog = IsingSamplingProgram(
        ebm=ebm,
        free_blocks=free_blocks,
        clamped_blocks=clamped_blocks
    )

    # Initialize state using Hinton initialization
    key = jax.random.PRNGKey(seed)
    init_key, sample_key = jax.random.split(key)

    init_state = hinton_init(init_key, ebm, free_blocks, (n_chains,))

    # Define sampling schedule
    schedule = SamplingSchedule(
        n_warmup=n_warmup,
        n_samples=n_samples,
        steps_per_sample=steps_per_sample
    )

    # Run sampling
    # sample_states expects: (key, prog, schedule, init_free_state, init_clamped_state, observe_blocks)
    # We want to observe all nodes
    observe_blocks = [Block(spin_nodes)]

    samples = sample_states(
        sample_key,
        prog,
        schedule,
        [x[0] for x in init_state],  # Extract first chain from each block
        [],  # No clamped states
        observe_blocks
    )

    return {
        'samples': samples,
        'n_chains': n_chains
    }


def analyze_results(results, node_labels, spin_nodes):
    """
    Analyzes and prints statistics from sampling results.

    Args:
        results: Dictionary containing samples from sampling
        node_labels: Dictionary mapping node indices to variable names
        spin_nodes: List of SpinNode objects
    """

    # samples is a list containing one element (the observe block)
    # Shape: (n_samples, n_nodes)
    samples_data = results['samples'][0]

    if len(samples_data) == 0:
        print("No samples collected!")
        return

    # samples_data is a JAX array of shape (n_samples, n_nodes)
    # where each value is either False or True (representing spins -1 or +1)
    # Convert to spin format: False -> -1, True -> +1
    samples_array = jnp.where(samples_data, 1, -1)

    print("\n" + "="*60)
    print("STUDENT NETWORK - SAMPLING RESULTS")
    print("="*60)

    # Compute marginal probabilities for each variable
    print("\nMarginal Probabilities:")
    print("-" * 60)
    print(f"{'Variable':<15} {'P(+1)':<10} {'P(-1)':<10} {'Mean':<10}")
    print("-" * 60)

    for idx, name in node_labels.items():
        # Get samples for this variable
        var_samples = samples_array[:, idx]
        mean_spin = float(jnp.mean(var_samples))
        prob_plus = float(jnp.mean(var_samples == 1))
        prob_minus = float(jnp.mean(var_samples == -1))

        print(f"{name:<15} {prob_plus:>8.3f}  {prob_minus:>8.3f}  {mean_spin:>8.3f}")

    # Compute some conditional statistics
    print("\n" + "="*60)
    print("CONDITIONAL PROBABILITIES")
    print("="*60)

    # P(Grade=High | Intelligence=High)
    intelligence_high = samples_array[:, 1] == 1
    if jnp.sum(intelligence_high) > 0:
        grade_high_given_intel_high = jnp.mean(
            samples_array[intelligence_high, 2] == 1
        )
        print(f"\nP(Grade=High | Intelligence=High) = {grade_high_given_intel_high:.3f}")

    # P(Grade=High | Intelligence=Low)
    intelligence_low = samples_array[:, 1] == -1
    if jnp.sum(intelligence_low) > 0:
        grade_high_given_intel_low = jnp.mean(
            samples_array[intelligence_low, 2] == 1
        )
        print(f"P(Grade=High | Intelligence=Low)  = {grade_high_given_intel_low:.3f}")

    # P(Letter=Strong | Grade=High)
    grade_high = samples_array[:, 2] == 1
    if jnp.sum(grade_high) > 0:
        letter_strong_given_grade_high = jnp.mean(
            samples_array[grade_high, 4] == 1
        )
        print(f"\nP(Letter=Strong | Grade=High) = {letter_strong_given_grade_high:.3f}")

    # P(Letter=Strong | Grade=Low)
    grade_low = samples_array[:, 2] == -1
    if jnp.sum(grade_low) > 0:
        letter_strong_given_grade_low = jnp.mean(
            samples_array[grade_low, 4] == 1
        )
        print(f"P(Letter=Strong | Grade=Low)  = {letter_strong_given_grade_low:.3f}")

    print("\n" + "="*60)


def main():
    """
    Main function demonstrating the Student Network PGM.
    """

    print("\n" + "="*60)
    print("STUDENT NETWORK - PROBABILISTIC GRAPHICAL MODEL")
    print("From: Probabilistic Graphical Models by Daphne Koller")
    print("="*60)

    # Create the network
    print("\n[1] Creating Student Network structure...")
    ebm, graph, node_labels, spin_nodes, edge_list = create_student_network()

    print(f"    Network has {len(spin_nodes)} nodes and {len(edge_list)} edges")
    print(f"    Nodes: {', '.join(node_labels.values())}")

    # Print network structure
    print("\n[2] Network Dependencies:")
    for edge in graph.edges():
        src_name = node_labels[edge[0]]
        dst_name = node_labels[edge[1]]
        print(f"    {src_name} -> {dst_name}")

    # Sample from the network
    print("\n[3] Sampling from the network using Block Gibbs sampling...")
    print("    (This may take a moment...)")

    results = sample_network(
        ebm,
        spin_nodes,
        n_samples=5000,
        n_warmup=500,
        steps_per_sample=3,
        n_chains=1,
        seed=42
    )

    print(f"    Collected {len(results['samples'][0])} samples")

    # Analyze results
    print("\n[4] Analyzing results...")
    analyze_results(results, node_labels, spin_nodes)

    print("\n✓ Analysis complete!\n")


if __name__ == "__main__":
    main()
