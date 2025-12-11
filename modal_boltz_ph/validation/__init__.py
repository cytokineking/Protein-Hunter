"""
Structure validation package.

This package provides validation functions for different structure prediction
models (AF3, Protenix, OpenFold3) with a unified interface.

Modules:
    base: Shared utilities (ipSAE calculation, chain conversion, pLDDT normalization)
    af3: AlphaFold3 validation (requires proprietary weights)
    protenix: Protenix validation (open-source, Apache 2.0)
    openfold3: OpenFold3 validation (open-source, Apache 2.0)
"""

from typing import List, Optional, Tuple

# Import GPU function mappings for configuration helpers
from modal_boltz_ph.validation.af3 import (
    AF3_GPU_FUNCTIONS,
    AF3_APO_GPU_FUNCTIONS,
    run_af3_single_A100_80GB,
    run_af3_apo_A100_80GB,
)
from modal_boltz_ph.validation.protenix import (
    PROTENIX_GPU_FUNCTIONS,
    DEFAULT_PROTENIX_GPU,
    run_protenix_validation_A100,
)
from modal_boltz_ph.validation.openfold3 import (
    OPENFOLD3_GPU_FUNCTIONS,
    DEFAULT_OPENFOLD3_GPU,
    run_openfold3_validation_A100,
)

# Re-export base utilities
from modal_boltz_ph.validation.base import (
    calculate_ipsae_from_pae,
    convert_chain_indices_to_letters,
    normalize_plddt_scale,
)


# =============================================================================
# CONFIGURATION HELPERS
# =============================================================================

def get_default_validation_gpu(validation_model: str) -> Optional[str]:
    """
    Get the recommended default GPU for a validation model.
    
    Args:
        validation_model: One of "af3", "protenix", "openfold3", or "none"
    
    Returns:
        Recommended GPU type or None if validation disabled
    """
    defaults = {
        "af3": "A100-80GB",
        "protenix": "A100",
        "openfold3": "A100",
        "none": None,
    }
    return defaults.get(validation_model, "A100")


def get_supported_gpus(validation_model: str) -> List[str]:
    """
    Get list of supported GPUs for a validation model.
    
    Args:
        validation_model: One of "af3", "protenix", "openfold3", or "none"
    
    Returns:
        List of supported GPU types
    """
    if validation_model == "af3":
        return list(AF3_GPU_FUNCTIONS.keys())
    elif validation_model == "protenix":
        return list(PROTENIX_GPU_FUNCTIONS.keys())
    elif validation_model == "openfold3":
        return list(OPENFOLD3_GPU_FUNCTIONS.keys())
    else:
        return []


def validate_model_gpu_combination(
    validation_model: str,
    validation_gpu: str,
) -> Tuple[bool, Optional[str]]:
    """
    Validate that the GPU is supported for the given validation model.
    
    Args:
        validation_model: One of "af3", "protenix", "openfold3", or "none"
        validation_gpu: GPU type to validate
    
    Returns:
        Tuple of (is_valid, error_message or None)
    """
    if validation_model == "none":
        return True, None
    
    supported = get_supported_gpus(validation_model)
    if not supported:
        return False, f"Unknown validation model: {validation_model}"
    
    if validation_gpu not in supported:
        return False, (
            f"GPU '{validation_gpu}' not supported for {validation_model}. "
            f"Supported: {', '.join(supported)}"
        )
    
    return True, None


__all__ = [
    # Configuration helpers
    "get_default_validation_gpu",
    "get_supported_gpus",
    "validate_model_gpu_combination",
    # AF3
    "AF3_GPU_FUNCTIONS",
    "AF3_APO_GPU_FUNCTIONS",
    "run_af3_single_A100_80GB",
    "run_af3_apo_A100_80GB",
    # Protenix
    "PROTENIX_GPU_FUNCTIONS",
    "DEFAULT_PROTENIX_GPU",
    "run_protenix_validation_A100",
    # OpenFold3
    "OPENFOLD3_GPU_FUNCTIONS",
    "DEFAULT_OPENFOLD3_GPU",
    "run_openfold3_validation_A100",
    # Base utilities
    "calculate_ipsae_from_pae",
    "convert_chain_indices_to_letters",
    "normalize_plddt_scale",
]
