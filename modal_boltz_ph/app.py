"""
Modal App definition and shared infrastructure.

This module contains the core Modal app and shared resources that other modules depend on.
"""

import modal

# =============================================================================
# MODAL APP DEFINITION
# =============================================================================

app = modal.App("protein-hunter-boltz")

# =============================================================================
# VOLUMES
# =============================================================================

# Volume for caching model weights (Boltz, LigandMPNN)
cache_volume = modal.Volume.from_name("protein-hunter-cache", create_if_missing=True)

# Volume for AlphaFold3 weights (user uploads their own)
af3_weights_volume = modal.Volume.from_name("af3-weights", create_if_missing=True)

# Volume for Protenix weights (auto-downloaded on first use)
protenix_weights_volume = modal.Volume.from_name("protenix-weights", create_if_missing=True)

# Volume for OpenFold3 weights (auto-downloaded on first use)
openfold3_weights_volume = modal.Volume.from_name("protein-hunter-openfold3-weights", create_if_missing=True)

# =============================================================================
# SHARED DICT FOR RESULT STREAMING
# =============================================================================

# Dict for real-time result streaming between Modal containers and local sync
results_dict = modal.Dict.from_name("protein-hunter-results", create_if_missing=True)

# =============================================================================
# GPU CONFIGURATION
# =============================================================================

# Supported GPU types with descriptions and pricing
GPU_TYPES = {
    "T4": "16GB - $0.59/h",
    "L4": "24GB - $0.80/h",
    "A10G": "24GB - $1.10/h",
    "L40S": "48GB - $1.95/h",
    "A100-40GB": "40GB - $2.10/h",
    "A100-80GB": "80GB - $2.50/h",
    "H100": "80GB - $3.95/h (RECOMMENDED)",
}

DEFAULT_GPU = "H100"

