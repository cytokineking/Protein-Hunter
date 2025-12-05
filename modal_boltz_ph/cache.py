"""
Cache initialization and weight management.

This module handles downloading and caching model weights (Boltz, LigandMPNN)
and managing AlphaFold3 weights uploads.
"""

import subprocess
from pathlib import Path

from modal_boltz_ph.app import app, cache_volume, af3_weights_volume
from modal_boltz_ph.images import image


@app.function(
    image=image,
    gpu="T4",  # Use cheap GPU for downloads
    timeout=3600,
    volumes={"/cache": cache_volume},
)
def initialize_cache() -> str:
    """
    Download and cache Boltz model weights, CCD data, and LigandMPNN models.
    
    This function should be run once before using the pipeline to ensure
    all required model weights are available in the Modal volume.
    
    Returns:
        Status message indicating success or failure
    """
    from boltz.main import download_boltz2
    
    cache_dir = Path("/cache/boltz")
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("INITIALIZING PROTEIN HUNTER CACHE")
    print("=" * 60)
    
    # Download Boltz weights
    print("\n1. Downloading Boltz2 weights and CCD data...")
    print("   This may take 10-20 minutes on first run.")
    
    try:
        download_boltz2(cache_dir)
        print("   ✓ Boltz2 downloaded successfully")
    except Exception as e:
        print(f"   ✗ Error downloading Boltz2: {e}")
        return f"Error: {e}"
    
    # Download LigandMPNN weights
    print("\n2. Downloading LigandMPNN model weights...")
    mpnn_dir = Path("/cache/ligandmpnn")
    mpnn_dir.mkdir(parents=True, exist_ok=True)
    
    mpnn_models = [
        ("proteinmpnn_v_48_020.pt", "https://files.ipd.uw.edu/pub/ligandmpnn/proteinmpnn_v_48_020.pt"),
        ("ligandmpnn_v_32_010_25.pt", "https://files.ipd.uw.edu/pub/ligandmpnn/ligandmpnn_v_32_010_25.pt"),
        ("solublempnn_v_48_020.pt", "https://files.ipd.uw.edu/pub/ligandmpnn/solublempnn_v_48_020.pt"),
    ]
    
    for model_name, url in mpnn_models:
        model_path = mpnn_dir / model_name
        if not model_path.exists():
            print(f"   Downloading {model_name}...")
            result = subprocess.run(["wget", "-q", url, "-O", str(model_path)], capture_output=True)
            if result.returncode == 0:
                print(f"   ✓ {model_name}")
            else:
                print(f"   ✗ Failed to download {model_name}")
        else:
            print(f"   ✓ {model_name} (cached)")
    
    # Commit volume
    cache_volume.commit()
    
    # Report cache contents
    print("\n3. Cache contents:")
    for subdir in [cache_dir, mpnn_dir]:
        if subdir.exists():
            files = list(subdir.rglob("*"))
            total_size = sum(f.stat().st_size for f in files if f.is_file())
            print(f"   {subdir}: {len(files)} files, {total_size / 1e9:.2f} GB")
    
    return "Cache initialized successfully!"


@app.function(
    image=image,  # Use base image for upload
    volumes={"/af3_weights": af3_weights_volume},
    timeout=600,
)
def _upload_af3_weights_impl(weights_bytes: bytes, filename: str) -> str:
    """
    Save AF3 weights to Modal volume.
    
    This is called by the local entrypoint after decompression.
    
    Args:
        weights_bytes: Raw bytes of the AF3 weights file
        filename: Filename to save as (e.g., "af3.bin")
    
    Returns:
        Status message
    """
    weights_path = Path("/af3_weights") / filename
    weights_path.parent.mkdir(parents=True, exist_ok=True)
    weights_path.write_bytes(weights_bytes)
    
    print(f"Wrote {len(weights_bytes) / 1e9:.2f} GB to {weights_path}")
    
    af3_weights_volume.commit()
    
    # List contents
    print("\nAF3 weights volume contents:")
    for f in Path("/af3_weights").rglob("*"):
        if f.is_file():
            print(f"  {f}: {f.stat().st_size / 1e9:.2f} GB")
    
    return "AF3 weights uploaded successfully!"

