"""
Advanced Spin Models with THRML
Demonstrates:
- Larger grid models
- Energy computation
- Performance benchmarking
"""

import jax
import jax.numpy as jnp
import thrml as th
from thrml.models import IsingEBM
import time

print("=" * 70)
print("THRML Advanced Spin Models Example")
print("=" * 70)

# Step 1: Create a 2D grid (4x4 = 16 nodes)
print("\n[Step 1] Creating a 4x4 grid of spin nodes (16 total)...")
grid_size = 4
num_nodes = grid_size * grid_size
nodes = [th.SpinNode() for _ in range(num_nodes)]
print(f"✓ Created {num_nodes} spin nodes in a {grid_size}x{grid_size} grid")

# Step 2: Create checkerboard blocking
print("\n[Step 2] Creating checkerboard blocks for parallel sampling...")
block_0 = []
block_1 = []
for i in range(num_nodes):
    row = i // grid_size
    col = i % grid_size
    if (row + col) % 2 == 0:
        block_0.append(nodes[i])
    else:
        block_1.append(nodes[i])

blocks = [th.Block(block_0), th.Block(block_1)]
print(f"✓ Created checkerboard blocks:")
print(f"  - Block 0 (white squares): {len(blocks[0].nodes)} nodes")
print(f"  - Block 1 (black squares): {len(blocks[1].nodes)} nodes")

# Step 3: Initialize Ising model
print("\n[Step 3] Initializing Ising model with 2D grid edges...")

# Create edges for 2D grid (4-connectivity: right, down)
edges = []
for i in range(num_nodes):
    row = i // grid_size
    col = i % grid_size
    # Right neighbor
    if col < grid_size - 1:
        edges.append((nodes[i], nodes[i + 1]))
    # Down neighbor
    if row < grid_size - 1:
        edges.append((nodes[i], nodes[i + grid_size]))

print(f"  - Created {len(edges)} edges for {grid_size}x{grid_size} grid")

# Initialize parameters
key = jax.random.key(2025)
biases = jnp.zeros(num_nodes)  # No external field
weights = jnp.ones(len(edges)) * 2.0  # Strong ferromagnetic coupling
beta = jnp.array(1.0)  # Inverse temperature

ebm = IsingEBM(nodes, edges, biases, weights, beta)
print("✓ Ising EBM initialized:")
print(f"  - Biases: {biases.shape} (all zeros)")
print(f"  - Weights: {weights.shape} (all 2.0)")
print(f"  - Beta: {beta}")

# Step 4: Create sampling program
print("\n[Step 4] Creating sampling program...")
program = th.models.IsingSamplingProgram(ebm, blocks, clamped_blocks=[])
print("✓ Sampling program created")

# Step 5: Run sampling with performance measurement
print("\n[Step 5] Running sampling with performance measurement...")
schedule = th.SamplingSchedule(
    n_warmup=200,
    n_samples=500,
    steps_per_sample=1
)

# Initialize random states (SpinNode uses boolean values)
key = jax.random.key(int(time.time() * 1000) % 2**32)
init_states = []
for block in blocks:
    block_size = len(block.nodes)
    init_states.append(jax.random.choice(key, jnp.array([False, True]), shape=(block_size,)))
    key, _ = jax.random.split(key)

# Warmup JIT compilation
print("  - JIT compiling (warmup run)...")
_ = th.sample_states(key, program, schedule, init_states, [], blocks)

# Timed run
print("  - Running timed sampling...")
start_time = time.time()
samples = th.sample_states(key, program, schedule, init_states, [], blocks)
elapsed = time.time() - start_time

print(f"✓ Sampling complete in {elapsed:.4f}s")
print(f"  - Block 0 samples shape: {samples[0].shape}")
print(f"  - Block 1 samples shape: {samples[1].shape}")
print(f"  - Total samples per block: {schedule.n_samples}")

# Step 6: Reconstruct full configurations
print("\n[Step 6] Reconstructing full spin configurations...")
n_samples = samples[0].shape[0]

# Create mapping from block position to grid position
block_0_indices = [i for i in range(num_nodes)
                   if ((i // grid_size) + (i % grid_size)) % 2 == 0]
block_1_indices = [i for i in range(num_nodes)
                   if ((i // grid_size) + (i % grid_size)) % 2 == 1]

# Reconstruct configurations
all_configs = []
for s in range(n_samples):
    config = jnp.zeros(num_nodes)
    # Place block 0 spins
    for idx, grid_idx in enumerate(block_0_indices):
        config = config.at[grid_idx].set(samples[0][s, idx])
    # Place block 1 spins
    for idx, grid_idx in enumerate(block_1_indices):
        config = config.at[grid_idx].set(samples[1][s, idx])
    all_configs.append(config)

all_configs = jnp.stack(all_configs)
print(f"✓ Reconstructed {all_configs.shape[0]} configurations")
print(f"  - Configuration shape: {all_configs.shape}")

# Step 7: Statistical analysis
print("\n[Step 7] Statistical analysis...")
total_magnetization = jnp.mean(all_configs)
print(f"  - Mean magnetization: {total_magnetization:.4f}")

# Count fully aligned states
final_configs = all_configs[-100:]  # Last 100 samples
fully_up = jnp.sum(jnp.all(final_configs == 1, axis=1))
fully_down = jnp.sum(jnp.all(final_configs == -1, axis=1))
mixed = 100 - fully_up - fully_down

print(f"  - Analysis of final 100 samples:")
print(f"    Fully spin-up: {fully_up}/100")
print(f"    Fully spin-down: {fully_down}/100")
print(f"    Mixed states: {mixed}/100")

# Step 8: Visualize sample configurations
print("\n[Step 8] Sample grid configurations (4x4):")

def get_2d_coords(idx, grid_size):
    return idx // grid_size, idx % grid_size

for sample_idx in range(min(3, n_samples)):
    config = all_configs[sample_idx]
    print(f"\n  Configuration {sample_idx}:")
    for row in range(grid_size):
        row_str = "    "
        for col in range(grid_size):
            idx = row * grid_size + col
            spin = config[idx]
            symbol = "+" if bool(spin) else "-"
            row_str += f"{symbol} "
        print(row_str)

# Step 9: Performance summary
print("\n[Step 9] Performance Summary:")
total_spin_samples = schedule.n_samples * num_nodes
samples_per_sec = total_spin_samples / elapsed
print(f"  - Total runtime: {elapsed:.4f}s")
print(f"  - Samples per spin: {schedule.n_samples}")
print(f"  - Total spin-samples: {total_spin_samples}")
print(f"  - Throughput: {samples_per_sec:.1f} spin-samples/sec")

print("\n" + "=" * 70)
print("✓ Advanced spin models example completed successfully!")
print("=" * 70)
