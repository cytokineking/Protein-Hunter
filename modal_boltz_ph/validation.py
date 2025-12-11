"""
Validation dispatcher and orchestration.

This module provides a unified interface for selecting and running
structure validation with different models (AF3, Protenix, OpenFold3) and
scoring methods (PyRosetta, open-source).
"""

from typing import Any, Callable, Dict, Optional

from modal_boltz_ph.validation_af3 import (
    AF3_GPU_FUNCTIONS,
    AF3_APO_GPU_FUNCTIONS,
    run_af3_single_A100_80GB,
    run_af3_apo_A100_80GB,
)
from modal_boltz_ph.validation_protenix import (
    PROTENIX_GPU_FUNCTIONS,
    DEFAULT_PROTENIX_GPU,
    run_protenix_validation_A100,
)
from modal_boltz_ph.validation_openfold3 import (
    OPENFOLD3_GPU_FUNCTIONS,
    DEFAULT_OPENFOLD3_GPU,
    run_openfold3_validation_A100,
)
from modal_boltz_ph.scoring_pyrosetta import run_pyrosetta_single
from modal_boltz_ph.scoring_opensource import (
    OPENSOURCE_SCORING_GPU_FUNCTIONS,
    DEFAULT_OPENSOURCE_GPU,
)


# =============================================================================
# VALIDATION MODEL SELECTION
# =============================================================================

def get_validation_function(
    validation_model: str,
    scoring_method: str,
    validation_gpu: str = "A100",
    scoring_gpu: str = DEFAULT_OPENSOURCE_GPU,
) -> Optional[Callable]:
    """
    Return the appropriate validation function based on user selection.
    
    This function selects the right combination of structure prediction
    and scoring based on CLI arguments.
    
    Args:
        validation_model: "none", "af3", or "protenix"
        scoring_method: "pyrosetta" or "opensource"
        validation_gpu: GPU type for validation container
        scoring_gpu: GPU type for scoring container (if separate)
    
    Returns:
        Validation function or None if validation disabled
    """
    if validation_model == "none":
        return None
    
    elif validation_model == "af3":
        # AF3 uses separate containers for prediction and scoring
        af3_fn = AF3_GPU_FUNCTIONS.get(validation_gpu, run_af3_single_A100_80GB)
        af3_apo_fn = AF3_APO_GPU_FUNCTIONS.get(validation_gpu, run_af3_apo_A100_80GB)
        
        if scoring_method == "pyrosetta":
            return _create_af3_pyrosetta_pipeline(af3_fn, af3_apo_fn)
        else:
            scoring_fn = OPENSOURCE_SCORING_GPU_FUNCTIONS.get(
                scoring_gpu, 
                OPENSOURCE_SCORING_GPU_FUNCTIONS[DEFAULT_OPENSOURCE_GPU]
            )
            return _create_af3_opensource_pipeline(af3_fn, af3_apo_fn, scoring_fn)
    
    elif validation_model == "protenix":
        if scoring_method == "opensource":
            # Single container - most efficient
            # Protenix validation includes bundled open-source scoring
            return PROTENIX_GPU_FUNCTIONS.get(validation_gpu, run_protenix_validation_A100)
        else:
            # Protenix → PyRosetta (separate containers)
            protenix_fn = PROTENIX_GPU_FUNCTIONS.get(validation_gpu, run_protenix_validation_A100)
            return _create_protenix_pyrosetta_pipeline(protenix_fn)
    
    elif validation_model == "openfold3":
        if scoring_method == "opensource":
            # Single container - bundled scoring
            return OPENFOLD3_GPU_FUNCTIONS.get(validation_gpu, run_openfold3_validation_A100)
        else:
            # OpenFold3 → PyRosetta (separate containers)
            of3_fn = OPENFOLD3_GPU_FUNCTIONS.get(validation_gpu, run_openfold3_validation_A100)
            return _create_openfold3_pyrosetta_pipeline(of3_fn)
    
    else:
        raise ValueError(f"Unknown validation model: {validation_model}")


def _create_af3_pyrosetta_pipeline(af3_fn, af3_apo_fn) -> Callable:
    """Create a pipeline function that runs AF3 → APO → PyRosetta."""
    
    def pipeline(
        design_id: str,
        binder_seq: str,
        target_seq: str,
        target_msas: Optional[Dict[str, str]] = None,
        af3_msa_mode: str = "reuse",
        template_content: Optional[str] = None,
        template_chain_ids: Optional[str] = None,
        target_type: str = "protein",
        verbose: bool = False,
    ) -> Dict[str, Any]:
        """Run AF3 validation with PyRosetta scoring."""
        
        # AF3 HOLO
        af3_result = af3_fn.remote(
            design_id, binder_seq, target_seq,
            "A", "B",
            target_msas,
            af3_msa_mode,
            template_content,
            template_chain_ids,
        )
        
        if not af3_result.get("af3_structure"):
            return af3_result
        
        # AF3 APO
        apo_result = af3_apo_fn.remote(design_id, binder_seq, "A")
        apo_structure = apo_result.get("apo_structure")
        
        # PyRosetta scoring
        if target_type == "protein":
            scoring_result = run_pyrosetta_single.remote(
                design_id,
                af3_result.get("af3_structure"),
                af3_result.get("af3_iptm", 0),
                af3_result.get("af3_ptm", 0),
                af3_result.get("af3_plddt", 0),
                "A", "B",
                apo_structure,
                af3_result.get("af3_confidence_json"),
                target_type,
            )
        else:
            scoring_result = {"accepted": True}
        
        return {
            **af3_result,
            "apo_structure": apo_structure,
            **scoring_result,
        }
    
    return pipeline


def _create_af3_opensource_pipeline(af3_fn, af3_apo_fn, scoring_fn) -> Callable:
    """Create a pipeline function that runs AF3 → APO → OpenSource scoring."""
    
    def pipeline(
        design_id: str,
        binder_seq: str,
        target_seq: str,
        target_msas: Optional[Dict[str, str]] = None,
        af3_msa_mode: str = "reuse",
        template_content: Optional[str] = None,
        template_chain_ids: Optional[str] = None,
        target_type: str = "protein",
        verbose: bool = False,
    ) -> Dict[str, Any]:
        """Run AF3 validation with open-source scoring."""
        
        # AF3 HOLO
        af3_result = af3_fn.remote(
            design_id, binder_seq, target_seq,
            "A", "B",
            target_msas,
            af3_msa_mode,
            template_content,
            template_chain_ids,
        )
        
        if not af3_result.get("af3_structure"):
            return af3_result
        
        # AF3 APO
        apo_result = af3_apo_fn.remote(design_id, binder_seq, "A")
        apo_structure = apo_result.get("apo_structure")
        
        # Open-source scoring
        if target_type == "protein":
            scoring_result = scoring_fn.remote(
                design_id,
                af3_result.get("af3_structure"),
                af3_result.get("af3_iptm", 0),
                af3_result.get("af3_ptm", 0),
                af3_result.get("af3_plddt", 0),
                "A", "B",
                apo_structure,
                af3_result.get("af3_confidence_json"),
                target_type,
                verbose,
            )
        else:
            scoring_result = {"accepted": True}
        
        return {
            **af3_result,
            "apo_structure": apo_structure,
            **scoring_result,
        }
    
    return pipeline


def _create_protenix_pyrosetta_pipeline(protenix_fn) -> Callable:
    """Create a pipeline function that runs Protenix → PyRosetta scoring."""
    
    def pipeline(
        design_id: str,
        binder_seq: str,
        target_seq: str,
        target_msas: Optional[Dict[str, str]] = None,
        af3_msa_mode: str = "reuse",  # Unused but kept for interface compatibility
        template_content: Optional[str] = None,  # Unused
        template_chain_ids: Optional[str] = None,  # Unused
        target_type: str = "protein",
        verbose: bool = False,
    ) -> Dict[str, Any]:
        """Run Protenix validation with PyRosetta scoring."""
        
        # Protenix HOLO + APO (without bundled scoring)
        protenix_result = protenix_fn.remote(
            design_id, binder_seq, target_seq,
            target_msas,
            run_scoring=False,  # Don't run bundled scoring
            verbose=verbose,
        )
        
        if not protenix_result.get("af3_structure"):
            return protenix_result
        
        # PyRosetta scoring
        if target_type == "protein":
            scoring_result = run_pyrosetta_single.remote(
                design_id,
                protenix_result.get("af3_structure"),
                protenix_result.get("af3_iptm", 0),
                protenix_result.get("af3_ptm", 0),
                protenix_result.get("af3_plddt", 0),
                "A", "B",
                protenix_result.get("apo_structure"),
                protenix_result.get("af3_confidence_json"),
                target_type,
            )
        else:
            scoring_result = {"accepted": True}
        
        return {
            **protenix_result,
            **scoring_result,
        }
    
    return pipeline


def _create_openfold3_pyrosetta_pipeline(of3_fn) -> Callable:
    """Create a pipeline function that runs OpenFold3 → PyRosetta scoring."""
    
    def pipeline(
        design_id: str,
        binder_seq: str,
        target_seq: str,
        target_msas: Optional[Dict[str, str]] = None,
        af3_msa_mode: str = "reuse",  # Unused but kept for interface compatibility
        template_content: Optional[str] = None,  # Unused
        template_chain_ids: Optional[str] = None,  # Unused
        target_type: str = "protein",
        verbose: bool = False,
    ) -> Dict[str, Any]:
        """Run OpenFold3 validation with PyRosetta scoring."""
        
        # OpenFold3 HOLO + APO (without bundled scoring)
        of3_result = of3_fn.remote(
            design_id, binder_seq, target_seq,
            target_msas,
            run_scoring=False,  # Don't run bundled scoring
            verbose=verbose,
        )
        
        if not of3_result.get("af3_structure"):
            return of3_result
        
        # PyRosetta scoring
        if target_type == "protein":
            scoring_result = run_pyrosetta_single.remote(
                design_id,
                of3_result.get("af3_structure"),
                of3_result.get("af3_iptm", 0),
                of3_result.get("af3_ptm", 0),
                of3_result.get("af3_plddt", 0),
                "A", "B",
                of3_result.get("apo_structure"),
                of3_result.get("af3_confidence_json"),
                target_type,
            )
        else:
            scoring_result = {"accepted": True}
        
        return {
            **of3_result,
            **scoring_result,
        }
    
    return pipeline


# =============================================================================
# CONFIGURATION HELPERS
# =============================================================================

def get_default_validation_gpu(validation_model: str) -> str:
    """Get the recommended default GPU for a validation model."""
    defaults = {
        "af3": "A100-80GB",
        "protenix": "A100",
        "openfold3": "A100",
        "none": None,
    }
    return defaults.get(validation_model, "A100")


def get_supported_gpus(validation_model: str) -> list:
    """Get list of supported GPUs for a validation model."""
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
) -> tuple:
    """
    Validate that the GPU is supported for the given validation model.
    
    Returns:
        Tuple of (is_valid: bool, error_message: str or None)
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
