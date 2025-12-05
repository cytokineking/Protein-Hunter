"""
Test functions for validating Modal infrastructure.

This module contains test functions to verify GPU availability,
image configuration, and weight availability.
"""

import subprocess

from modal_boltz_ph.app import app, af3_weights_volume
from modal_boltz_ph.images import image, af3_image


@app.function(image=image, gpu="T4", timeout=60)
def _test_gpu() -> str:
    """
    Test GPU availability.
    
    Runs nvidia-smi to verify GPU is accessible.
    
    Returns:
        nvidia-smi output
    """
    result = subprocess.run(["nvidia-smi"], capture_output=True, text=True)
    return result.stdout


@app.function(
    image=af3_image,
    gpu="A100-80GB",
    timeout=300,
    volumes={"/af3_weights": af3_weights_volume}
)
def test_af3_image() -> str:
    """
    Test that the AF3 image is correctly configured.
    
    Performs comprehensive checks:
    - Python version
    - AlphaFold3 import
    - JAX/jaxtyping
    - GPU availability
    - HMMER installation
    - AF3 weights volume
    - AF3 attention module
    
    Returns:
        Formatted string with test results
    """
    import os
    import subprocess
    results = []
    
    # Test 1: Check Python version
    results.append("=== Python Version ===")
    result = subprocess.run(["python", "--version"], capture_output=True, text=True)
    results.append(result.stdout + result.stderr)
    
    # Test 2: Check if AF3 can be imported
    results.append("\n=== AF3 Import Test ===")
    try:
        import alphafold3
        results.append(f"✓ alphafold3 imported successfully (version: {getattr(alphafold3, '__version__', 'unknown')})")
    except Exception as e:
        results.append(f"✗ Failed to import alphafold3: {e}")
    
    # Test 3: Check jaxtyping
    results.append("\n=== JAX/jaxtyping Test ===")
    try:
        import jax
        import jaxtyping  # noqa: F401
        results.append(f"✓ jax version: {jax.__version__}")
        results.append("✓ jaxtyping imported successfully")
    except Exception as e:
        results.append(f"✗ JAX import error: {e}")
    
    # Test 4: Check GPU
    results.append("\n=== GPU Test ===")
    try:
        import jax
        devices = jax.devices()
        results.append(f"✓ JAX devices: {devices}")
    except Exception as e:
        results.append(f"✗ GPU test failed: {e}")
    
    # Test 5: Check HMMER
    results.append("\n=== HMMER Test ===")
    result = subprocess.run(["jackhmmer", "-h"], capture_output=True, text=True)
    if result.returncode == 0:
        results.append("✓ jackhmmer available")
    else:
        results.append(f"✗ jackhmmer not found: {result.stderr}")
    
    # Test 6: Check AF3 weights volume
    results.append("\n=== AF3 Weights ===")
    weights_path = "/af3_weights/af3.bin"
    if os.path.exists(weights_path):
        size_gb = os.path.getsize(weights_path) / (1024**3)
        results.append(f"✓ Weights found: {weights_path} ({size_gb:.2f} GB)")
    else:
        results.append(f"✗ Weights not found at {weights_path}")
    
    # Test 7: Try importing the AF3 run script
    results.append("\n=== AF3 Run Script ===")
    try:
        from alphafold3.jax.attention import attention  # noqa: F401
        results.append("✓ alphafold3.jax.attention imported successfully")
    except Exception as e:
        results.append(f"✗ Failed: {e}")
    
    return "\n".join(results)

