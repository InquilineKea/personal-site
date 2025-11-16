# Probabilistic Graphical Model Implementation using THRML

This repository contains an implementation of the classic **Student Network** from Daphne Koller's textbook *"Probabilistic Graphical Models: Principles and Techniques"* using the [thrml](https://github.com/extropic-ai/thrml) library.

## About

### The Student Network

The Student Network is a canonical example used in PGM textbooks to demonstrate Bayesian networks and conditional independence relationships. It models the relationships between:

- **Difficulty (D)**: Course difficulty
- **Intelligence (I)**: Student intelligence level
- **Grade (G)**: Final grade (depends on both difficulty and intelligence)
- **SAT (S)**: SAT score (depends on intelligence)
- **Letter (L)**: Recommendation letter quality (depends on grade)

### Network Structure

```
    Difficulty (D) ──┐
                     ├──> Grade (G) ──> Letter (L)
    Intelligence (I) ┘       │
                             └──> SAT (S)
```

This structure encodes several important conditional independence relationships typical of real-world scenarios:
- A student's grade depends on both their intelligence and the course difficulty
- SAT scores correlate with intelligence but are independent of course difficulty given intelligence
- Letter quality depends primarily on the grade received

## Implementation Details

### Energy-Based Model Representation

This implementation uses thrml's **Ising Energy-Based Model (EBM)** framework to represent the Student Network. Each variable is represented as a binary spin:
- **-1**: Low/Easy/Weak state
- **+1**: High/Hard/Strong state

The joint probability distribution is defined through an energy function:

```
E(x) = -β(Σᵢ hᵢxᵢ + Σ<i,j> Wᵢⱼxᵢxⱼ)

P(x) ∝ exp(-E(x))
```

Where:
- `hᵢ` are bias (field) terms encoding prior preferences
- `Wᵢⱼ` are pairwise weights encoding conditional relationships
- `β` is the inverse temperature parameter

### Parameter Configuration

The model is configured with:

**Biases** (prior preferences):
```python
Difficulty:    0.0   # Neutral prior
Intelligence:  0.3   # Slight bias toward high intelligence
Grade:         0.0   # Neutral (depends heavily on D and I)
SAT:           0.2   # Slight bias toward higher scores
Letter:        0.0   # Neutral (depends on grade)
```

**Pairwise Weights** (conditional relationships):
```python
Difficulty -> Grade:    -0.8   # Hard courses → lower grades
Intelligence -> Grade:   0.9   # High intelligence → higher grades
Grade -> SAT:            0.7   # High grades correlate with high SAT
Grade -> Letter:         0.85  # High grades → strong letters
```

Positive weights encourage variables to align (both high or both low), while negative weights encourage opposition.

### Block Gibbs Sampling

The implementation uses **block Gibbs sampling** for efficient sampling:

1. **Block Partitioning**: Nodes are partitioned into two blocks that can be updated in parallel:
   - Block 0: Difficulty, Grade, Letter
   - Block 1: Intelligence, SAT

2. **Sampling Schedule**:
   - 500 warmup iterations (burn-in)
   - 5000 samples collected
   - 3 Gibbs steps between each sample

3. **Initialization**: Uses Hinton initialization, where each spin is sampled according to its marginal bias

## Installation

```bash
# Install required dependencies
pip install -r requirements.txt
```

Or manually:
```bash
pip install jax jaxlib thrml numpy networkx matplotlib
```

## Usage

### Running the Example

```bash
python3 student_network_pgm.py
```

### Expected Output

The program will:
1. Create the Student Network structure
2. Sample 5000 configurations using block Gibbs sampling
3. Compute and display:
   - Marginal probabilities for each variable
   - Conditional probabilities demonstrating the network relationships

Example output:
```
Marginal Probabilities:
Variable        P(+1)      P(-1)      Mean
------------------------------------------------------------
Difficulty         0.382     0.618    -0.236
Intelligence       0.698     0.302     0.396
Grade              0.677     0.323     0.354
SAT                0.660     0.340     0.321
Letter             0.628     0.372     0.256

CONDITIONAL PROBABILITIES
P(Grade=High | Intelligence=High) = 0.892
P(Grade=High | Intelligence=Low)  = 0.178

P(Letter=Strong | Grade=High) = 0.783
P(Letter=Strong | Grade=Low)  = 0.304
```

### Interpreting Results

The results demonstrate the network's conditional dependencies:

1. **Intelligence Effect**: Students with high intelligence have a much higher probability of getting good grades (89.2%) compared to those with low intelligence (17.8%)

2. **Grade Effect**: Students with high grades are much more likely to receive strong recommendation letters (78.3%) compared to those with low grades (30.4%)

3. **Bias Effects**: The marginal probabilities reflect the bias parameters:
   - Intelligence leans toward high (~70%) due to positive bias
   - SAT scores similarly lean positive due to correlation with intelligence

## Code Structure

### Main Components

1. **`create_student_network()`**
   - Constructs the network graph structure
   - Defines nodes, edges, biases, and weights
   - Returns an IsingEBM model

2. **`setup_sampling_blocks()`**
   - Partitions nodes for efficient block Gibbs sampling
   - Uses a two-coloring scheme

3. **`sample_network()`**
   - Initializes the sampling state
   - Runs block Gibbs sampling
   - Returns collected samples

4. **`analyze_results()`**
   - Computes marginal and conditional probabilities
   - Displays statistics about the learned distribution

### Customization

You can modify the network by adjusting:

**Network structure**:
```python
edges_directed = [
    (node_map['Difficulty'], node_map['Grade']),
    (node_map['Intelligence'], node_map['Grade']),
    # Add more edges here
]
```

**Bias parameters**:
```python
biases = jnp.array([
    0.0,   # Difficulty
    0.3,   # Intelligence (adjust this value)
    # ...
])
```

**Interaction weights**:
```python
weights = jnp.array([
    -0.8,  # Difficulty -> Grade (adjust strength)
    0.9,   # Intelligence -> Grade
    # ...
])
```

## References

1. **Koller, D., & Friedman, N.** (2009). *Probabilistic Graphical Models: Principles and Techniques*. MIT Press.

2. **THRML Library**: [https://github.com/extropic-ai/thrml](https://github.com/extropic-ai/thrml)

3. **Hinton, G. E.** (2012). *A Practical Guide to Training Restricted Boltzmann Machines*. In Neural Networks: Tricks of the Trade.

## About THRML

THRML is a JAX-based library for building and sampling probabilistic graphical models, developed by [Extropic AI](https://github.com/extropic-ai). It emphasizes:
- Efficient block Gibbs sampling
- GPU acceleration via JAX
- Energy-based model utilities
- Support for heterogeneous graphical models

This implementation demonstrates THRML's capabilities for educational purposes and showcases how classical PGM examples can be implemented using modern probabilistic programming frameworks.

## License

This implementation is provided for educational purposes. Please refer to the THRML library's license for usage restrictions.
