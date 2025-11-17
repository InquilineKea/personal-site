"""
Spin Models in THRML - Example from Extropic AI
Adapted from: https://github.com/extropic-ai/thrml/blob/main/examples/02_spin_models.ipynb

This demonstrates using thrml to build and sample from Ising models (binary PGMs).
"""

import time
import jax
import dwave_networkx
import jax.numpy as jnp
import jax.random
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from thrml.block_management import Block
from thrml.block_sampling import sample_states, SamplingSchedule
from thrml.models.discrete_ebm import SpinEBMFactor
from thrml.models.ising import (
    estimate_kl_grad,
    hinton_init,
    IsingEBM,
    IsingSamplingProgram,
    IsingTrainingSpec,
)
from thrml.pgm import SpinNode

print("=" * 80)
print("THRML Spin Models Example - Ising Model on Pegasus Graph")
print("=" * 80)

# ============================================================
# 1. Create the graph using DWave's Pegasus topology
# ============================================================
print("\n📊 Creating Pegasus graph...")
graph = dwave_networkx.pegasus_graph(14)
coord_to_node = {coord: SpinNode() for coord in graph.nodes}
nx.relabel_nodes(graph, coord_to_node, copy=False)

print(f"   Nodes: {len(graph.nodes)}")
print(f"   Edges: {len(graph.edges)}")

# ============================================================
# 2. Define the Ising model with random biases and weights
# ============================================================
print("\n🎲 Initializing Ising model with random parameters...")
nodes = list(graph.nodes)
edges = list(graph.edges)

seed = 4242
key = jax.random.key(seed)

key, subkey = jax.random.split(key, 2)
biases = jax.random.normal(subkey, (len(nodes),))

key, subkey = jax.random.split(key, 2)
weights = jax.random.normal(subkey, (len(edges),))

beta = jnp.array(1.0)

model = IsingEBM(nodes, edges, biases, weights, beta)

print(f"   Model created with {len(model.factors)} factors")
print(f"   Factor types: {[x.__class__.__name__ for x in model.factors]}")

# ============================================================
# 3. Set up training configuration
# ============================================================
print("\n🔧 Setting up training configuration...")

# Choose random subset of nodes to represent "data" (visible variables)
n_data = 500
np.random.seed(seed)
data_inds = np.random.choice(len(graph.nodes), n_data, replace=False)
data_nodes = [nodes[x] for x in data_inds]

print(f"   Data nodes: {n_data}")
print(f"   Latent nodes: {len(nodes) - n_data}")

# Compute minimum coloring for unclamped (free) sampling
print("\n🎨 Computing graph coloring...")
coloring = nx.coloring.greedy_color(graph, strategy="DSATUR")
n_colors = max(coloring.values()) + 1
free_coloring = [[] for _ in range(n_colors)]
for node in graph.nodes:
    free_coloring[coloring[node]].append(node)

free_blocks = [Block(x) for x in free_coloring]
print(f"   Colors needed: {n_colors}")

# Coloring for clamped sampling (with data nodes fixed)
graph_copy = graph.copy()
graph_copy.remove_nodes_from(data_nodes)

clamped_coloring = [[] for _ in range(n_colors)]
for node in graph_copy.nodes:
    clamped_coloring[coloring[node]].append(node)

clamped_blocks = [Block(x) for x in clamped_coloring]

# ============================================================
# 4. Generate random "data" and estimate gradients
# ============================================================
print("\n📈 Setting up gradient estimation...")

# Random binary data (in real ML this could be images, text, etc.)
data_batch_size = 50
key, subkey = jax.random.split(key, 2)
data = jax.random.bernoulli(subkey, 0.5, (data_batch_size, len(data_nodes))).astype(jnp.bool_)

# Sampling schedule: (num_sweeps, sweeps_between_samples, num_samples)
schedule = SamplingSchedule(5, 100, 5)

# Training specification
training_spec = IsingTrainingSpec(
    model,
    [Block(data_nodes)],
    [],
    clamped_blocks,
    free_blocks,
    schedule,
    schedule
)

# Number of parallel sampling chains
n_chains_free = data_batch_size
n_chains_clamped = 1

# Initial states using Hinton initialization
key, subkey = jax.random.split(key, 2)
init_state_free = hinton_init(subkey, model, free_blocks, (n_chains_free,))
key, subkey = jax.random.split(key, 2)
init_state_clamped = hinton_init(subkey, model, clamped_blocks, (n_chains_clamped, data_batch_size))

print(f"   Data batch size: {data_batch_size}")
print(f"   Free chains: {n_chains_free}")
print(f"   Clamped chains: {n_chains_clamped}")

# ============================================================
# 5. Estimate KL divergence gradients
# ============================================================
print("\n🔥 Estimating KL divergence gradients...")
start_time = time.time()

key, subkey = jax.random.split(key, 2)
weight_grads, bias_grads, clamped_moments, free_moments = estimate_kl_grad(
    subkey,
    training_spec,
    nodes,
    edges,
    [data],
    [],
    init_state_clamped,
    init_state_free,
)

end_time = time.time()

print(f"   Gradient estimation completed in {end_time - start_time:.2f} seconds")
print(f"\n   Weight gradients (first 10): {weight_grads[:10]}")
print(f"   Bias gradients (first 10): {bias_grads[:10]}")
print(f"\n   Weight grad stats: mean={jnp.mean(weight_grads):.4f}, std={jnp.std(weight_grads):.4f}")
print(f"   Bias grad stats: mean={jnp.mean(bias_grads):.4f}, std={jnp.std(bias_grads):.4f}")

# ============================================================
# 6. Demonstrate higher-order interactions
# ============================================================
print("\n🎯 Demonstrating higher-order spin interactions...")
print("   Creating cubic interaction (s_1 * s_2 * s_3) between 30 nodes:")

key, subkey = jax.random.split(key, 2)
cubic_factor = SpinEBMFactor(
    [Block(nodes[:10]), Block(nodes[10:20]), Block(nodes[20:30])],
    jax.random.normal(subkey, (10,))
)

print(f"   Factor type: {cubic_factor.__class__.__name__}")
print(f"   Number of node groups: {len(cubic_factor.node_groups)}")
print(f"   Interaction order: 3 (cubic)")

# ============================================================
# 7. Simple sampling benchmark (single GPU/CPU)
# ============================================================
print("\n⚡ Running simple sampling benchmark...")
print("   (Note: Full benchmark requires 8 GPUs - we'll do a simplified version)")

timing_program = IsingSamplingProgram(model, free_blocks, [])
timing_chain_len = 100
small_batch_size = 100

schedule = SamplingSchedule(timing_chain_len, 1, 1)

# Create JIT-compiled sampling function
call_f = jax.jit(
    jax.vmap(lambda k: sample_states(
        k,
        timing_program,
        schedule,
        [x[0] for x in init_state_free],
        [],
        [Block(nodes)]
    ))
)

key, subkey = jax.random.split(key, 2)
keys = jax.random.split(subkey, small_batch_size)

# Warmup
print("   Warming up JIT compilation...")
_ = jax.block_until_ready(call_f(keys))

# Actual timing
print("   Running benchmark...")
start_time = time.time()
_ = jax.block_until_ready(call_f(keys))
end_time = time.time()

elapsed = end_time - start_time
total_flips = timing_chain_len * len(nodes) * small_batch_size
flips_per_second = total_flips / elapsed
flips_per_ns = total_flips / (elapsed * 1e9)

print(f"\n   Batch size: {small_batch_size}")
print(f"   Chain length: {timing_chain_len}")
print(f"   Total flips: {total_flips:,}")
print(f"   Time: {elapsed:.3f} seconds")
print(f"   Performance: {flips_per_second:,.0f} flips/second")
print(f"   Performance: {flips_per_ns:.2f} flips/nanosecond")
print(f"\n   (Compare to FPGA implementation: ~60 flips/ns)")

# ============================================================
# Summary
# ============================================================
print("\n" + "=" * 80)
print("✅ THRML Spin Models Example Complete!")
print("=" * 80)
print("\nKey takeaways:")
print("  • Created Ising model on Pegasus graph with 14x14 topology")
print("  • Estimated KL divergence gradients for training")
print("  • Demonstrated higher-order (cubic) spin interactions")
print("  • Benchmarked Gibbs sampling performance")
print("\nThis demonstrates heterogeneous PGM capabilities:")
print("  • Binary (spin) variables: {-1, +1}")
print("  • Polynomial interactions (quadratic, cubic, etc.)")
print("  • Efficient GPU-accelerated sampling with JAX")
print("  • Training via gradient estimation")
print("=" * 80)
