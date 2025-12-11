"""
Base classes and utilities for structure validation.

This module provides shared utilities used by both AF3 and Protenix validation.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np


@dataclass
class ValidationInput:
    """Standard input format for all validation models."""
    design_id: str
    binder_seq: str
    target_seq: str  # Colon-separated for multi-chain
    target_msas: Optional[Dict[str, str]] = None  # chain_id -> A3M content
    template_content: Optional[str] = None  # Base64-encoded PDB/CIF


@dataclass
class ValidationOutput:
    """Standard output format for all validation models."""
    design_id: str
    iptm: float
    ptm: float
    plddt: float
    ipsae: Optional[float] = None
    chain_pair_iptm: Optional[Dict[str, float]] = None
    structure_cif: Optional[str] = None
    confidence_json: Optional[str] = None
    error: Optional[str] = None


def calculate_ipsae_from_pae(
    pae_matrix: np.ndarray,
    binder_length: int,
    target_length: int,
    pae_cutoff: float = 10.0,
) -> Dict[str, float]:
    """
    Calculate ipSAE (interface predicted Squared Aligned Error) from PAE matrix.
    
    Shared implementation for AF3 and Protenix.
    
    This uses the same algorithm as Boltz ipSAE calculation - a PTM-like
    transformation applied to interface PAE values.
    
    Args:
        pae_matrix: 2D numpy array of PAE values (N x N)
        binder_length: Number of residues in binder (first chain)
        target_length: Number of residues in target (subsequent chains)
        pae_cutoff: PAE cutoff for considering residue pairs (default 10.0 Å)
    
    Returns:
        dict with:
            - 'ipsae': max of binder and target direction scores
            - 'ipsae_binder_to_target': per-residue max from binder → target
            - 'ipsae_target_to_binder': per-residue max from target → binder
    """
    result = {
        'ipsae': 0.0,
        'ipsae_binder_to_target': 0.0,
        'ipsae_target_to_binder': 0.0,
    }
    
    if pae_matrix is None or len(pae_matrix) == 0:
        return result
    
    try:
        total_length = binder_length + target_length
        
        # Validate matrix dimensions
        if pae_matrix.ndim == 1:
            expected_size = total_length * total_length
            if len(pae_matrix) != expected_size:
                return result
            pae_matrix = pae_matrix.reshape(total_length, total_length)
        
        if pae_matrix.shape != (total_length, total_length):
            return result
        
        # Define indices
        binder_indices = np.arange(binder_length)
        target_indices = np.arange(binder_length, total_length)
        
        # PTM-style transformation functions
        def ptm_func(x: np.ndarray, d0: float) -> np.ndarray:
            return 1.0 / (1.0 + (x / d0) ** 2.0)
        
        def calc_d0(L: int) -> float:
            L = float(max(L, 27))
            d0 = 1.24 * (L - 15) ** (1.0 / 3.0) - 1.8
            return max(1.0, d0)
        
        # Binder → Target direction
        interface_pae = pae_matrix[np.ix_(binder_indices, target_indices)]
        valid_mask = interface_pae < pae_cutoff
        
        ipsae_byres_binder = []
        for i in range(binder_length):
            valid = valid_mask[i]
            if valid.any():
                n0res = valid.sum()
                d0res = calc_d0(n0res)
                ptm_vals = ptm_func(interface_pae[i][valid], d0res)
                ipsae_byres_binder.append(ptm_vals.mean())
            else:
                ipsae_byres_binder.append(0.0)
        
        ipsae_byres_binder = np.array(ipsae_byres_binder)
        ipsae_binder_max = float(ipsae_byres_binder.max()) if len(ipsae_byres_binder) > 0 else 0.0
        
        # Target → Binder direction
        interface_pae_rev = pae_matrix[np.ix_(target_indices, binder_indices)]
        valid_mask_rev = interface_pae_rev < pae_cutoff
        
        ipsae_byres_target = []
        for i in range(target_length):
            valid = valid_mask_rev[i]
            if valid.any():
                n0res = valid.sum()
                d0res = calc_d0(n0res)
                ptm_vals = ptm_func(interface_pae_rev[i][valid], d0res)
                ipsae_byres_target.append(ptm_vals.mean())
            else:
                ipsae_byres_target.append(0.0)
        
        ipsae_byres_target = np.array(ipsae_byres_target)
        ipsae_target_max = float(ipsae_byres_target.max()) if len(ipsae_byres_target) > 0 else 0.0
        
        # Take max of both directions
        ipsae = max(ipsae_binder_max, ipsae_target_max)
        
        result['ipsae'] = round(ipsae, 4)
        result['ipsae_binder_to_target'] = round(ipsae_binder_max, 4)
        result['ipsae_target_to_binder'] = round(ipsae_target_max, 4)
        
    except Exception:
        pass
    
    return result


def convert_chain_indices_to_letters(
    chain_data: Dict[str, Any],
    key_pattern: str = "_",
) -> Dict[str, Any]:
    """
    Convert Protenix numeric chain indices to letter-based chain IDs.
    
    Protenix outputs chain indices as "0", "1", "2", etc.
    AF3 and the rest of the pipeline use "A", "B", "C", etc.
    
    Args:
        chain_data: Dict with numeric string keys (e.g., {"0": 0.85, "1": 0.72})
        key_pattern: Separator for chain pair keys (e.g., "0_1" -> "A_B")
    
    Returns:
        Dict with letter-based keys (e.g., {"A": 0.85, "B": 0.72})
    """
    converted = {}
    
    for key, value in chain_data.items():
        if key_pattern in key:
            # Chain pair key (e.g., "0_1")
            parts = key.split(key_pattern)
            new_parts = []
            for p in parts:
                try:
                    idx = int(p)
                    new_parts.append(chr(ord('A') + idx))
                except ValueError:
                    new_parts.append(p)
            new_key = key_pattern.join(new_parts)
        else:
            # Single chain key (e.g., "0")
            try:
                idx = int(key)
                new_key = chr(ord('A') + idx)
            except ValueError:
                new_key = key
        
        converted[new_key] = value
    
    return converted


def normalize_plddt_scale(plddt: float, from_scale: str = "0-1") -> float:
    """
    Normalize pLDDT to 0-100 scale for consistency.
    
    AF3 outputs pLDDT in 0-100 scale.
    Protenix outputs pLDDT in 0-1 scale (needs *100).
    
    Args:
        plddt: pLDDT value
        from_scale: "0-1" or "0-100"
    
    Returns:
        pLDDT in 0-100 scale
    """
    if from_scale == "0-1":
        return plddt * 100.0
    return plddt


def get_validation_model_info(model_name: str) -> Dict[str, Any]:
    """
    Get information about a validation model.
    
    Args:
        model_name: "af3", "protenix", or "openfold3"
    
    Returns:
        Dict with model information
    """
    models = {
        "af3": {
            "name": "AlphaFold3",
            "license": "Proprietary (research only)",
            "weights_required": True,
            "gpu_recommended": "A100-80GB",
            "description": "Google DeepMind's AlphaFold 3 - requires proprietary weights",
        },
        "protenix": {
            "name": "Protenix",
            "license": "Apache 2.0",
            "weights_required": False,  # Auto-downloaded
            "gpu_recommended": "A100",
            "description": "ByteDance's open-source AF3 reproduction",
        },
        "openfold3": {
            "name": "OpenFold3",
            "license": "Apache 2.0",
            "weights_required": False,  # Auto-downloaded
            "gpu_recommended": "A100",
            "description": "AlQuraishi Lab's open-source AF3 reproduction",
        },
        "none": {
            "name": "None",
            "license": "N/A",
            "weights_required": False,
            "gpu_recommended": None,
            "description": "No structure validation (design only)",
        },
    }
    
    return models.get(model_name.lower(), models["none"])
