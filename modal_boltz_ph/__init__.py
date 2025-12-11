"""
Modal Boltz Protein Hunter - Modular Package

This package provides serverless GPU execution of the Protein Hunter design pipeline
on Modal's cloud infrastructure.

Package Structure:
    app: Modal App definition and shared infrastructure
    images: Docker image definitions for Boltz, AF3, Protenix, OpenFold3, and scoring
    helpers: Shared utility functions
    design: Core Boltz design implementation and GPU wrappers
    cache: Cache initialization and weight management
    sync: Result streaming and synchronization
    
    validation/: Structure validation subpackage
        base: Shared validation utilities (ipSAE, pLDDT normalization)
        af3: AlphaFold3 validation functions
        protenix: Protenix (open-source AF3) validation functions
        openfold3: OpenFold3 validation functions
    
    scoring/: Interface scoring subpackage
        opensource: Open-source scoring (OpenMM + FreeSASA + sc-rs)
        pyrosetta: PyRosetta interface analysis
    
    utils/: Shared utilities subpackage
        logging: Verbose logging utilities
        weights: Model weight download helpers
    
    tests/: Test harnesses for pipeline components

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
    openfold3_weights_volume,
    results_dict,
    GPU_TYPES,
    DEFAULT_GPU,
)

from modal_boltz_ph.validation import (
    get_default_validation_gpu,
    validate_model_gpu_combination,
    PROTENIX_GPU_FUNCTIONS,
    DEFAULT_PROTENIX_GPU,
    OPENFOLD3_GPU_FUNCTIONS,
    DEFAULT_OPENFOLD3_GPU,
    AF3_GPU_FUNCTIONS,
    AF3_APO_GPU_FUNCTIONS,
)

from modal_boltz_ph.scoring import (
    OPENSOURCE_SCORING_GPU_FUNCTIONS,
    DEFAULT_OPENSOURCE_GPU,
)

__all__ = [
    # App and infrastructure
    "app",
    "cache_volume",
    "af3_weights_volume",
    "protenix_weights_volume",
    "openfold3_weights_volume",
    "results_dict",
    "GPU_TYPES",
    "DEFAULT_GPU",
    # Validation
    "get_default_validation_gpu",
    "validate_model_gpu_combination",
    "PROTENIX_GPU_FUNCTIONS",
    "DEFAULT_PROTENIX_GPU",
    "OPENFOLD3_GPU_FUNCTIONS",
    "DEFAULT_OPENFOLD3_GPU",
    "AF3_GPU_FUNCTIONS",
    "AF3_APO_GPU_FUNCTIONS",
    # Scoring
    "OPENSOURCE_SCORING_GPU_FUNCTIONS",
    "DEFAULT_OPENSOURCE_GPU",
]
