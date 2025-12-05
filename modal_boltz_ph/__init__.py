"""
Modal Boltz Protein Hunter - Modular Package

This package provides serverless GPU execution of the Protein Hunter design pipeline
on Modal's cloud infrastructure.

Modules:
    app: Modal App definition and shared infrastructure
    images: Docker image definitions for Boltz, AF3, and PyRosetta
    helpers: Shared utility functions
    design: Core design implementation and GPU wrappers
    validation_af3: AlphaFold3 validation functions
    scoring_pyrosetta: PyRosetta interface analysis
    sync: Result streaming and synchronization
    cache: Cache initialization and weight management
    tests: Test functions for validating infrastructure

Usage:
    # Initialize cache (run once)
    modal run modal_boltz_ph_cli.py::init_cache

    # Run design pipeline
    modal run modal_boltz_ph_cli.py::run_pipeline \\
        --name "PDL1_binder" \\
        --target-seq "AFTVTVPK..." \\
        --num-designs 5
"""

from modal_boltz_ph.app import (
    app,
    cache_volume,
    af3_weights_volume,
    results_dict,
    GPU_TYPES,
    DEFAULT_GPU,
)

__all__ = [
    "app",
    "cache_volume",
    "af3_weights_volume",
    "results_dict",
    "GPU_TYPES",
    "DEFAULT_GPU",
]

