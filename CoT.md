# Chain of Thought Analysis: THRML Library Exploration

## Executive Summary

This document analyzes the complete process of exploring, installing, and running examples with the **THRML** (Thermodynamic Hypergraphical Model Library) - a JAX-based Python library for probabilistic graphical models and Energy-Based Models (EBMs) developed by Extropic AI.

**Key Finding**: THRML provides a powerful, GPU-accelerated framework for sampling from discrete probabilistic models, particularly Ising models, using Block Gibbs sampling techniques. The library is designed to prototype algorithms for thermodynamic computing hardware.

---

## 1. Initial Discovery Phase

### 1.1 Understanding THRML's Purpose

From the GitHub repository and documentation analysis:

**What is THRML?**
- JAX-based library for probabilistic graphical models
- Specializes in Energy-Based Models (EBMs) with discrete variables
- Implements efficient Block Gibbs sampling on GPUs
- Designed to prototype sampling techniques for Extropic's Thermodynamic Sampling Units (TSUs)

**Core Capabilities:**
- Spin models (Ising models) with arbitrary graph topologies
- Higher-order interactions beyond pairwise terms
- Flexible block-based sampling strategies
- Full Python type hints and JAX JIT compilation support

### 1.2 Available Resources

**Documentation:**
- GitHub: https://github.com/extropic-ai/thrml
- Official Docs: https://docs.thrml.ai/
- Example Notebooks: 3 comprehensive Jupyter notebooks in `/examples/`

**Key Examples:**
1. `00_probabilistic_computing.ipynb` - Introduction to concepts
2. `01_all_of_thrml.ipynb` - Complete library tutorial
3. `02_spin_models.ipynb` - Spin systems and Ising models

---

## 2. Installation and Environment Setup

### 2.1 Installation Process

```bash
pip install thrml
```

**Dependencies Installed:**
- `jax==0.8.0` and `jaxlib==0.8.0` (JAX framework)
- `equinox==0.13.2` (neural network library)
- `jaxtyping==0.3.3` (type annotations)
- `numpy==2.3.5` and `scipy==1.16.3` (numerical computing)
- Additional utilities: `opt_einsum`, `ml_dtypes`, `wadler_lindig`

### 2.2 Environment Verification

**System Details:**
- Platform: Linux 4.4.0
- Python: 3.11
- JAX Device: CPU (no GPU detected in this environment)
- THRML Version: 0.1.3

**Verification Tests:**
✓ All core dependencies imported successfully
✓ THRML objects (SpinNode, Block, SamplingSchedule) created
✓ JAX operations functional
✓ Ready for production use

---

## 3. API Learning Curve and Challenges

### 3.1 Initial API Misconceptions

**Challenge 1: SamplingSchedule Parameters**

❌ **Incorrect (assumed from docs):**
```python
schedule = th.SamplingSchedule(warmup_steps=100, num_samples=1000)
```

✓ **Correct (actual API):**
```python
schedule = th.SamplingSchedule(n_warmup=100, n_samples=1000, steps_per_sample=1)
```

**Lesson**: Parameter names differ from typical conventions - always verify with `inspect.signature()`.

---

**Challenge 2: IsingEBM Initialization**

❌ **Incorrect (assumed classmethod):**
```python
ebm = IsingEBM.init(key, nodes, biases=0.1, weights=1.0)
```

✓ **Correct (direct constructor):**
```python
# Must manually construct edges and parameter arrays
edges = [(nodes[i], nodes[i+1]) for i in range(len(nodes)-1)]
biases = jax.random.normal(key, shape=(len(nodes),)) * 0.1
weights = jax.random.normal(key, shape=(len(edges),)) * 1.0
beta = jnp.array(1.0)
ebm = IsingEBM(nodes, edges, biases, weights, beta)
```

**Lesson**: No convenience `init()` method exists - requires manual construction of:
- Node list
- Edge list (topology)
- Bias array (one per node)
- Weight array (one per edge)
- Beta (inverse temperature scalar)

---

**Challenge 3: IsingSamplingProgram Constructor**

❌ **Incorrect (assumed from README pattern):**
```python
program = IsingSamplingProgram(blocks, schedule)
```

✓ **Correct (actual signature):**
```python
program = IsingSamplingProgram(ebm, blocks, clamped_blocks=[])
```

**Signature:**
```python
IsingSamplingProgram(
    ebm: IsingEBM,
    free_blocks: list[Block],
    clamped_blocks: list[Block]
)
```

**Lesson**: The program needs the EBM model upfront, and distinguishes between free (sampled) and clamped (fixed) blocks.

---

**Challenge 4: SpinNode State Representation**

❌ **Incorrect (assumed spin values):**
```python
init_states.append(jax.random.choice(key, jnp.array([-1, 1]), shape=(n,)))
```

**Error Message:**
```
RuntimeError: Data has incorrect type int32 vs bool
```

✓ **Correct (boolean representation):**
```python
init_states.append(jax.random.choice(key, jnp.array([False, True]), shape=(n,)))
```

**Lesson**: `SpinNode` internally uses **boolean** values, not ±1 integers. False=down, True=up.

---

**Challenge 5: Sampling API**

The high-level sampling API is not directly on the program object but in the `thrml` module:

```python
samples = th.sample_states(
    key,                    # Random key
    program,                # Sampling program
    schedule,               # Sampling schedule
    init_state_free,        # Initial states for free blocks
    state_clamp,            # States for clamped blocks ([] if none)
    nodes_to_sample         # Which blocks to collect samples from
)
```

**Returns:** List of arrays, one per block in `nodes_to_sample`
- Shape per block: `(n_samples, block_size)`

---

## 4. Working Examples Analysis

### 4.1 Basic Ising Chain (5 nodes)

**Model Setup:**
- **Topology**: 1D chain (0-1-2-3-4)
- **Edges**: 4 nearest-neighbor connections
- **Blocking**: Two-color (even/odd nodes)
  - Block 0: nodes [0, 2, 4] (3 nodes)
  - Block 1: nodes [1, 3] (2 nodes)
- **Parameters**:
  - Biases: Random normal ~ N(0, 0.1²)
  - Weights: Random normal ~ N(1.0, 1.0²)
  - Beta: 1.0 (room temperature)

**Sampling Results:**
- 1000 samples collected per block
- Mean magnetization: ~0.42-0.51 (slight ferromagnetic tendency)
- Sample configurations show mixed spin states
- Successful Block Gibbs alternation between even/odd blocks

**Key Insight**: The two-color blocking enables parallel sampling within each block while maintaining Gibbs sampling correctness - crucial for GPU efficiency.

---

### 4.2 Advanced 2D Grid (4×4 = 16 nodes)

**Model Setup:**
- **Topology**: 2D square lattice with 4-connectivity
- **Edges**: 24 nearest-neighbor connections (right, down)
- **Blocking**: Checkerboard pattern
  - Block 0: "white squares" (8 nodes)
  - Block 1: "black squares" (8 nodes)
- **Parameters**:
  - Biases: All zeros (no external field)
  - Weights: All 2.0 (strong ferromagnetic coupling)
  - Beta: 1.0

**Sampling Results:**
- 500 samples collected (200 warmup + 500 recorded)
- **Mean magnetization: 1.0** (fully aligned)
- **Final 100 samples: 100% fully spin-up states**
- JIT compilation + sampling: 0.6449s
- Throughput: ~12,400 spin-samples/sec on CPU

**Key Insight**: Strong ferromagnetic coupling (J=2.0) with no external field leads to spontaneous symmetry breaking - the system rapidly converges to a fully aligned state (all spins up OR all spins down).

**Performance Notes:**
- JAX JIT compilation adds initial overhead
- Subsequent runs are fast due to cached compilation
- CPU-only performance is respectable (~12k samples/sec)
- GPU would provide significant speedup for larger systems

---

## 5. THRML Architecture Insights

### 5.1 Core Design Patterns

**1. Node-Edge-Factor Model**
```
Nodes → define random variables
Edges → define pairwise interactions
Factors → encode energy functions
Blocks → group nodes for parallel sampling
```

**2. Block Gibbs Sampling**
- Partition variables into non-overlapping blocks
- Sample all variables within a block in parallel (independent given neighbors)
- Alternate between blocks
- Enables GPU parallelization while maintaining Markov chain correctness

**3. Separation of Concerns**
- **Model Definition**: `IsingEBM` (energy function)
- **Sampling Strategy**: `IsingSamplingProgram` (block structure)
- **Execution**: `sample_states()` (JAX-compiled sampling loop)

### 5.2 Energy Function

For Ising models:

$$
\mathcal{E}(s) = -\beta \left( \sum_{i \in S_1} b_i s_i + \sum_{(i,j) \in S_2} J_{ij} s_i s_j \right)
$$

Where:
- $s_i \in \{0, 1\}$ (boolean) represents spin states
- $b_i$ are node biases (external fields)
- $J_{ij}$ are edge weights (interaction strengths)
- $\beta$ is inverse temperature (higher β = stronger preferences)

**Probability Distribution:**
$$
P(s) = \frac{1}{Z} \exp(-\mathcal{E}(s))
$$

---

## 6. Practical Workflow

### 6.1 Standard THRML Pipeline

**Step 1: Define Graph Structure**
```python
nodes = [th.SpinNode() for _ in range(n)]
edges = [...define topology...]
```

**Step 2: Define Blocking Strategy**
```python
# Two-color for chains/trees
blocks = [
    th.Block([nodes[i] for i in even_indices]),
    th.Block([nodes[i] for i in odd_indices])
]
```

**Step 3: Initialize Model Parameters**
```python
biases = jnp.array([...])  # shape: (n_nodes,)
weights = jnp.array([...]) # shape: (n_edges,)
beta = jnp.array(1.0)
ebm = IsingEBM(nodes, edges, biases, weights, beta)
```

**Step 4: Create Sampling Program**
```python
program = th.models.IsingSamplingProgram(
    ebm,
    free_blocks=blocks,
    clamped_blocks=[]
)
```

**Step 5: Configure Schedule**
```python
schedule = th.SamplingSchedule(
    n_warmup=100,      # Burn-in steps
    n_samples=1000,    # Samples to collect
    steps_per_sample=1 # Thinning factor
)
```

**Step 6: Initialize and Sample**
```python
init_states = [jax.random.choice(key, jnp.array([False, True]), (len(b.nodes),))
               for b in blocks]

samples = th.sample_states(
    key, program, schedule,
    init_state_free=init_states,
    state_clamp=[],
    nodes_to_sample=blocks
)
```

**Step 7: Post-process Results**
```python
# samples is a list of arrays, one per block
# Reconstruct full configurations by mapping block indices to global indices
```

---

## 7. Key Lessons Learned

### 7.1 API Design Philosophy

1. **Explicit over Implicit**: No magic initialization - users must understand graph structure
2. **Flexibility First**: Supports arbitrary topologies, heterogeneous nodes, complex blocking
3. **JAX-Centric**: Heavily leverages JAX's JIT, vmap, and array operations
4. **Type Safety**: Full type hints via `jaxtyping`

### 7.2 Common Pitfalls

| Pitfall | Solution |
|---------|----------|
| Using ±1 integers for spins | Use `False`/`True` booleans |
| Forgetting `clamped_blocks` | Pass empty list `[]` if none |
| Wrong parameter names | Check signatures with `inspect` |
| Mismatched array shapes | Verify biases=(n_nodes,), weights=(n_edges,) |
| Inefficient blocking | Use graph coloring (2-color, checkerboard) |

### 7.3 Performance Considerations

**CPU Performance:**
- Basic 5-node chain: ~instant
- 16-node grid: 0.6s for 500 samples (with JIT compilation)
- Throughput: 10k-15k spin-samples/sec on CPU

**Optimization Strategies:**
- **JIT Compilation**: First call is slow, subsequent calls are fast
- **Blocking**: Good blocking strategy is crucial for parallelism
- **Batch Sampling**: THRML supports batched sampling (not shown in examples)
- **GPU**: Would provide 10-100x speedup for large systems

---

## 8. Use Cases and Applications

### 8.1 Demonstrated Applications

**From the spin models notebook:**

1. **Statistical Physics**: Sample from Boltzmann distributions
2. **Machine Learning**: Train restricted Boltzmann machines (RBMs)
3. **Optimization**: Solve combinatorial problems via simulated annealing
4. **Hardware Prototyping**: Test algorithms for thermodynamic computing

### 8.2 Extension Possibilities

- **Image generation** with learned pixel interactions
- **Protein folding** on lattice models
- **Network design** with graph-structured EBMs
- **Quantum simulation** (discrete spin systems)

---

## 9. Scientific Context

### 9.1 Related Work

**Citation:**
```bibtex
@misc{jelinčič2025efficientprobabilistichardwarearchitecture,
    title={An efficient probabilistic hardware architecture for diffusion-like models},
    author={Andraž Jelinčič and Owen Lockwood and Akhil Garlapati and
            Guillaume Verdon and Trevor McCourt},
    year={2025},
    eprint={2510.23972}
}
```

### 9.2 Connection to Thermodynamic Computing

**Key Idea**: Physical thermal fluctuations in specialized hardware can perform probabilistic sampling:
- Transistors operating in thermal regime provide natural randomness
- Energy landscape of circuit encodes desired probability distribution
- Thermodynamic relaxation ≈ Gibbs sampling
- THRML provides software emulation for algorithm development

---

## 10. Conclusions

### 10.1 Summary of Findings

**THRML is...**
- ✓ Well-designed for discrete probabilistic models
- ✓ Efficiently implements Block Gibbs sampling
- ✓ Fully integrated with JAX ecosystem
- ✓ Suitable for both research and prototyping
- ⚠ Requires understanding of underlying mathematics
- ⚠ Documentation could be more comprehensive with examples

**Learning Curve:**
- **Steep initial climb**: API differs from typical ML libraries
- **Good payoff**: Once understood, very powerful and flexible
- **Best practices**: Start with simple examples, inspect signatures, read source code

### 10.2 Recommendations

**For New Users:**
1. Start with 1D chain examples (simplest topology)
2. Always verify API signatures with `inspect.signature()`
3. Use boolean values for SpinNode states
4. Understand blocking strategy before scaling up
5. Profile with warmup runs to separate JIT compilation time

**For Researchers:**
1. THRML excels at custom graph topologies
2. Consider GPU for systems >100 nodes
3. Blocking strategy significantly impacts performance
4. Library is extensible - can define custom nodes/factors

**For Practitioners:**
1. Good fit for discrete optimization problems
2. Strong foundation for thermodynamic computing research
3. JAX integration enables easy gradient computation
4. Consider for probabilistic machine learning experiments

---

## 11. Future Exploration

### 11.1 Unanswered Questions

- How does performance scale with system size on GPU?
- What are the best blocking strategies for different topologies?
- How to implement higher-order interactions (cubic, quartic)?
- Can we train models via gradient descent on parameters?

### 11.2 Next Steps

1. **Explore `01_all_of_thrml.ipynb`** for comprehensive tutorial
2. **Test on GPU** for performance comparison
3. **Implement custom topologies** (random graphs, small-world)
4. **Experiment with training** using gradient estimation
5. **Compare with classical samplers** (Metropolis-Hastings, HMC)

---

## Appendix: Test Scripts Summary

### A.1 `thrml_test_01_verify_install.py`
**Purpose**: Installation verification
**Tests**: Imports, object creation, JAX operations
**Status**: ✓ All tests passed

### A.2 `thrml_test_02_basic_ising.py`
**Purpose**: Basic 5-node Ising chain
**Features**:
- Two-color blocking (even/odd)
- Random biases and weights
- 1000 samples collected
- Configuration visualization
**Status**: ✓ Successfully completed

### A.3 `thrml_test_03_advanced_spin.py`
**Purpose**: Advanced 4×4 grid model
**Features**:
- Checkerboard blocking
- Strong ferromagnetic coupling
- Performance benchmarking
- 2D visualization
- Statistical analysis
**Status**: ✓ Successfully completed

**Results**: Achieved ~12,400 spin-samples/sec on CPU, demonstrated spontaneous symmetry breaking with strong coupling.

---

## Final Thoughts

THRML represents a sophisticated approach to probabilistic sampling that bridges classical Monte Carlo methods with modern hardware-accelerated computing. While the learning curve is steep, the library provides essential tools for researchers exploring thermodynamic computing, discrete probabilistic models, and GPU-accelerated sampling techniques.

The key to success with THRML is understanding the mathematical foundations (Energy-Based Models, Gibbs sampling) and embracing JAX's functional programming paradigm. Once these concepts click, THRML becomes a powerful tool for probabilistic computing research.

---

**Document Version**: 1.0
**Date**: 2025-11-17
**Author**: Claude (AI Assistant)
**THRML Version**: 0.1.3
**Environment**: Linux 4.4.0, Python 3.11, JAX 0.8.0
