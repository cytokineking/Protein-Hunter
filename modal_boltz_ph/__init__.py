"""
Modal Boltz Protein Hunter - Modular Package

This package provides serverless GPU execution of the Protein Hunter design pipeline
on Modal's cloud infrastructure.

Modules:
    app: Modal App definition and shared infrastructure
    images: Docker image definitions for Boltz, AF3, Protenix, and PyRosetta
    helpers: Shared utility functions
    design: Core design implementation and GPU wrappers
    validation_af3: AlphaFold3 validation functions
    validation_protenix: Protenix (open-source AF3) validation functions
    validation_base: Shared validation utilities
    validation: Validation dispatcher and orchestration
    scoring_pyrosetta: PyRosetta interface analysis
    scoring_opensource: Open-source interface scoring (OpenMM + FreeSASA)
    sync: Result streaming and synchronization
    cache: Cache initialization and weight management
    tests: Test functions for validating infrastructure

Usage:
    # Initialize cache (run once)
    modal run modal_boltz_ph_cli.py::init_cache

    # Run design pipeline (design only)
    modal run modal_boltz_ph_cli.py::run_pipeline \\
        --name "PDL1_binder" \\
        --protein-seqs "AFTVTVPK..." \\
        --num-designs 5

    # Run with AF3 validation
    modal run modal_boltz_ph_cli.py::run_pipeline \\
        --name "PDL1_af3" \\
        --protein-seqs "AFTVTVPK..." \\
        --validation-model af3 \\
        --scoring-method pyrosetta

    # Run fully open-source (Protenix + OpenMM)
    modal run modal_boltz_ph_cli.py::run_pipeline \\
        --name "PDL1_open" \\
        --protein-seqs "AFTVTVPK..." \\
        --validation-model protenix \\
        --scoring-method opensource
"""

from modal_boltz_ph.app import (
    app,
    cache_volume,
    af3_weights_volume,
    protenix_weights_volume,
    results_dict,
    GPU_TYPES,
    DEFAULT_GPU,
)

from modal_boltz_ph.validation import (
    get_validation_function,
    get_default_validation_gpu,
    validate_model_gpu_combination,
)

from modal_boltz_ph.validation_protenix import (
    PROTENIX_GPU_FUNCTIONS,
    DEFAULT_PROTENIX_GPU,
)

__all__ = [
    # App and infrastructure
    "app",
    "cache_volume",
    "af3_weights_volume",
    "protenix_weights_volume",
    "results_dict",
    "GPU_TYPES",
    "DEFAULT_GPU",
    # Validation
    "get_validation_function",
    "get_default_validation_gpu",
    "validate_model_gpu_combination",
    "PROTENIX_GPU_FUNCTIONS",
    "DEFAULT_PROTENIX_GPU",
]

