"""
Basic Ising Model Example with THRML
Demonstrates creating and sampling from a simple 5-node Ising chain.
"""

import jax
import jax.numpy as jnp
import thrml as th
from thrml.models import IsingEBM

print("=" * 60)
print("THRML Basic Ising Model Example")
print("=" * 60)

# Step 1: Create 5 spin nodes
print("\n[Step 1] Creating 5 spin nodes for an Ising chain...")
nodes = [th.SpinNode() for _ in range(5)]
print(f"✓ Created {len(nodes)} spin nodes")

# Step 2: Define alternating blocks for efficient sampling
print("\n[Step 2] Defining alternating color blocks...")
blocks = [
    th.Block([nodes[i] for i in range(5) if i % 2 == 0]),  # Even nodes: 0, 2, 4
    th.Block([nodes[i] for i in range(5) if i % 2 == 1])   # Odd nodes: 1, 3
]
print(f"✓ Created {len(blocks)} blocks:")
print(f"  - Block 0 (even): {len(blocks[0].nodes)} nodes")
print(f"  - Block 1 (odd): {len(blocks[1].nodes)} nodes")

# Step 3: Configure the Ising model
print("\n[Step 3] Initializing Ising EBM...")

# Create chain edges: 0-1, 1-2, 2-3, 3-4
edges = [(nodes[i], nodes[i+1]) for i in range(len(nodes)-1)]
print(f"  - Created {len(edges)} edges for chain topology")

# Initialize biases and weights
key = jax.random.key(42)
key_b, key_w = jax.random.split(key)

biases = jax.random.normal(key_b, shape=(len(nodes),)) * 0.1
weights = jax.random.normal(key_w, shape=(len(edges),)) * 1.0 + 1.0  # Mean 1.0
beta = jnp.array(1.0)  # Inverse temperature

ebm = IsingEBM(nodes, edges, biases, weights, beta)
print("✓ Ising EBM initialized")
print(f"  - Biases shape: {biases.shape}")
print(f"  - Weights shape: {weights.shape}")
print(f"  - Beta (inverse temp): {beta}")

# Step 4: Create sampling program
print("\n[Step 4] Creating sampling program...")
# For unclamped sampling, pass empty list for clamped_blocks
program = th.models.IsingSamplingProgram(ebm, blocks, clamped_blocks=[])
print("✓ Sampling program created")

# Step 5: Set up sampling schedule
print("\n[Step 5] Setting up sampling schedule...")
schedule = th.SamplingSchedule(
    n_warmup=100,
    n_samples=1000,
    steps_per_sample=1
)
print(f"✓ Sampling schedule created:")
print(f"  - Warmup steps: {schedule.n_warmup}")
print(f"  - Number of samples: {schedule.n_samples}")
print(f"  - Steps per sample: {schedule.steps_per_sample}")

# Step 6: Run sampling
print("\n[Step 6] Running Block Gibbs sampling...")

# Initialize random states for all nodes (SpinNode uses boolean values)
key = jax.random.key(123)
init_states = []
for block in blocks:
    block_size = len(block.nodes)
    init_states.append(jax.random.choice(key, jnp.array([False, True]), shape=(block_size,)))
    key, _ = jax.random.split(key)

# Sample using the program
samples = th.sample_states(
    key,
    program,
    schedule,
    init_state_free=init_states,
    state_clamp=[],
    nodes_to_sample=blocks
)

print(f"✓ Sampling complete!")
print(f"  - Number of block sample sets: {len(samples)}")
for i, sample_set in enumerate(samples):
    print(f"  - Block {i} shape: {sample_set.shape}")

# Step 7: Combine samples from blocks
print("\n[Step 7] Analyzing samples...")
# Reconstruct full configurations by interleaving block samples
all_samples = []
n_samples = samples[0].shape[0]

for s in range(n_samples):
    config = jnp.zeros(len(nodes))
    # Even block (0, 2, 4)
    for idx, node_idx in enumerate([0, 2, 4]):
        config = config.at[node_idx].set(samples[0][s, idx])
    # Odd block (1, 3)
    for idx, node_idx in enumerate([1, 3]):
        config = config.at[node_idx].set(samples[1][s, idx])
    all_samples.append(config)

all_samples = jnp.stack(all_samples)
print(f"  - Reconstructed full configurations: {all_samples.shape}")

# Calculate mean magnetization
mean_magnetization = jnp.mean(all_samples, axis=0)
print(f"\n✓ Mean magnetization per node:")
for i, mag in enumerate(mean_magnetization):
    print(f"  - Node {i}: {mag:.4f}")

# Show a few sample configurations
print("\n[Step 8] Sample spin configurations (first 5):")
for i in range(min(5, len(all_samples))):
    config = all_samples[i]
    spin_str = " ".join(["+1" if bool(s) else "-1" for s in config])
    print(f"  - Config {i}: [{spin_str}]")

print("\n" + "=" * 60)
print("✓ Basic Ising model example completed successfully!")
print("=" * 60)
