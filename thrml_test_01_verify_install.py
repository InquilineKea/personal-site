"""
THRML Installation Verification Script
Tests that thrml and all dependencies are properly installed.
"""

import sys
print("=" * 60)
print("THRML Installation Verification")
print("=" * 60)

# Test 1: Import core dependencies
print("\n[Test 1] Importing core dependencies...")
try:
    import jax
    import jax.numpy as jnp
    import equinox
    import jaxtyping
    print("✓ JAX, Equinox, and JaxTyping imported successfully")
    print(f"  - JAX version: {jax.__version__}")
    print(f"  - Equinox version: {equinox.__version__}")
except ImportError as e:
    print(f"✗ Failed to import dependencies: {e}")
    sys.exit(1)

# Test 2: Import thrml
print("\n[Test 2] Importing thrml...")
try:
    import thrml as th
    from thrml.models import IsingEBM, IsingSamplingProgram
    print("✓ THRML imported successfully")
    print(f"  - THRML version: {th.__version__}")
except ImportError as e:
    print(f"✗ Failed to import thrml: {e}")
    sys.exit(1)

# Test 3: Create basic objects
print("\n[Test 3] Creating basic THRML objects...")
try:
    node = th.SpinNode()
    block = th.Block([node])
    schedule = th.SamplingSchedule(n_warmup=10, n_samples=100, steps_per_sample=1)
    print("✓ Created SpinNode, Block, and SamplingSchedule")
except Exception as e:
    print(f"✗ Failed to create objects: {e}")
    sys.exit(1)

# Test 4: JAX GPU/CPU detection
print("\n[Test 4] JAX device detection...")
try:
    devices = jax.devices()
    print(f"✓ Found {len(devices)} JAX device(s):")
    for i, device in enumerate(devices):
        print(f"  - Device {i}: {device.device_kind} ({device.platform})")
except Exception as e:
    print(f"✗ Device detection failed: {e}")

# Test 5: Simple JAX operation
print("\n[Test 5] Running simple JAX operation...")
try:
    key = jax.random.key(42)
    random_array = jax.random.normal(key, shape=(3, 3))
    print("✓ JAX random number generation works")
    print(f"  Sample output:\n{random_array}")
except Exception as e:
    print(f"✗ JAX operation failed: {e}")
    sys.exit(1)

print("\n" + "=" * 60)
print("✓ All tests passed! THRML is ready to use.")
print("=" * 60)
