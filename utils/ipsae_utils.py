"""
ipSAE scoring utilities for Protein-Hunter with Boltz

Based on: https://www.biorxiv.org/content/10.1101/2025.02.10.637595v1
Original implementation by Roland Dunbrack, Fox Chase Cancer Center

This module provides functions to calculate ipSAE (interface predicted 
Structural Alignment Error) scores from Boltz PAE matrices.

Key feature: Correctly handles multi-chain targets by only computing
binder ↔ target interface scores, excluding intra-target chain interactions.

MIT license: script can be modified and redistributed for non-commercial 
and commercial use, as long as this information is reproduced.
"""

import numpy as np
import torch
from typing import Dict, Optional, Set

# Nucleic acid residue types (DNA and RNA)
NUC_RESIDUE_SET: Set[str] = {"DA", "DC", "DT", "DG", "A", "C", "U", "G"}


def ptm_func(x: np.ndarray, d0: float) -> np.ndarray:
    """
    Core PTM-style transformation for ipSAE calculation.
    
    Args:
        x: PAE value(s) - numpy array or scalar
        d0: Normalization constant based on alignment length
    
    Returns:
        Score in range [0, 1]
    """
    return 1.0 / (1.0 + (x / d0) ** 2.0)


def calc_d0(L: int, pair_type: str = 'protein') -> float:
    """
    Calculate d0 based on alignment length.
    From Yang and Skolnick, PROTEINS 57:702–710 (2004)
    
    Args:
        L: Number of residues in the alignment
        pair_type: 'protein' or 'nucleic_acid'. Nucleic acid pairs use
                   min_value=2.0 (approximately 21 base pairs equivalent)
    
    Returns:
        d0 value (minimum 1.0 for protein, 2.0 for nucleic_acid)
    """
    L = float(max(L, 27))
    min_value = 2.0 if pair_type == 'nucleic_acid' else 1.0
    d0 = 1.24 * (L - 15) ** (1.0 / 3.0) - 1.8
    return max(min_value, d0)


def is_nucleic_acid_residue(residue_type: str) -> bool:
    """Check if a residue type is a nucleic acid."""
    return residue_type in NUC_RESIDUE_SET


def classify_chain_type(residue_types: np.ndarray) -> str:
    """
    Classify a chain as 'nucleic_acid' or 'protein' based on its residue types.
    
    A chain is classified as nucleic_acid if ANY of its residues are nucleic acid.
    
    Args:
        residue_types: Array of residue type strings for residues in the chain
    
    Returns:
        'nucleic_acid' or 'protein'
    """
    for res_type in residue_types:
        if res_type in NUC_RESIDUE_SET:
            return 'nucleic_acid'
    return 'protein'


def get_pair_type(
    binder_residue_types: Optional[np.ndarray],
    target_residue_types: Optional[np.ndarray],
) -> str:
    """
    Determine the pair type for d0 calculation.
    
    If EITHER the binder or target contains nucleic acid residues,
    the pair type is 'nucleic_acid' (following original implementation).
    
    Args:
        binder_residue_types: Array of residue type strings for binder
        target_residue_types: Array of residue type strings for target
    
    Returns:
        'nucleic_acid' or 'protein'
    """
    if binder_residue_types is None or target_residue_types is None:
        return 'protein'  # Default to protein if not provided
    
    binder_type = classify_chain_type(binder_residue_types)
    target_type = classify_chain_type(target_residue_types)
    
    # If either chain is nucleic acid, use nucleic_acid pair type
    if binder_type == 'nucleic_acid' or target_type == 'nucleic_acid':
        return 'nucleic_acid'
    return 'protein'


def calculate_ipsae_from_boltz_output(
    output: Dict,
    feats: Dict,
    binder_chain_idx: int = 0,
    pae_cutoff: float = 10.0,
    residue_types: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """
    Calculate ipSAE score from Boltz model output for BINDER-to-TARGET interface only.
    
    This correctly handles multi-chain targets by:
    - Including: binder ↔ all target chains
    - Excluding: target chain A ↔ target chain B (no intra-target scoring)
    
    Args:
        output: Boltz model output dictionary containing 'pae' (shape [B, N, N])
        feats: Boltz features dict containing 'asym_id' and 'token_pad_mask'
        binder_chain_idx: Chain index of the binder (default 0 = chain A)
        pae_cutoff: PAE cutoff for considering residue pairs (default 10.0 Å)
        residue_types: Optional array of residue type strings (e.g., 'ALA', 'DA')
                       for nucleic acid detection. If None, assumes all protein.
    
    Returns:
        dict with:
            - 'ipSAE': max of binder and target direction scores (primary metric)
            - 'ipSAE_binder_to_target': per-residue max from binder → target
            - 'ipSAE_target_to_binder': per-residue max from target → binder  
            - 'n0dom': number of residues with good PAE values
            - 'pair_type': 'protein' or 'nucleic_acid'
    """
    # Check if PAE is available
    if 'pae' not in output or output['pae'] is None:
        return {
            'ipSAE': 0.0,
            'ipSAE_binder_to_target': 0.0,
            'ipSAE_target_to_binder': 0.0,
            'n0dom': 0,
            'pair_type': 'protein',
        }
    
    # Extract PAE matrix and chain info
    pae_matrix = output['pae']
    
    # Handle batched vs unbatched
    if isinstance(pae_matrix, torch.Tensor):
        pae_matrix = pae_matrix.detach().cpu().numpy()
    
    if pae_matrix.ndim == 3:
        pae_matrix = pae_matrix[0]  # Take first sample if batched
    
    # Get chain assignments
    asym_id = feats['asym_id']
    if isinstance(asym_id, torch.Tensor):
        asym_id = asym_id.detach().cpu().numpy()
    if asym_id.ndim == 2:
        asym_id = asym_id[0]
    
    # Get padding mask
    pad_mask = feats['token_pad_mask']
    if isinstance(pad_mask, torch.Tensor):
        pad_mask = pad_mask.detach().cpu().numpy()
    if pad_mask.ndim == 2:
        pad_mask = pad_mask[0]
    
    # Identify binder and target residue indices
    binder_mask = (asym_id == binder_chain_idx) & (pad_mask > 0)
    target_mask = (asym_id != binder_chain_idx) & (pad_mask > 0)
    
    binder_indices = np.where(binder_mask)[0]
    target_indices = np.where(target_mask)[0]
    
    binder_len = len(binder_indices)
    target_len = len(target_indices)
    
    if binder_len == 0 or target_len == 0:
        return {
            'ipSAE': 0.0,
            'ipSAE_binder_to_target': 0.0,
            'ipSAE_target_to_binder': 0.0,
            'n0dom': 0,
            'pair_type': 'protein',
        }
    
    # Determine pair type for d0 calculation (nucleic acid vs protein)
    if residue_types is not None:
        if isinstance(residue_types, torch.Tensor):
            residue_types = residue_types.detach().cpu().numpy()
        if residue_types.ndim == 2:
            residue_types = residue_types[0]
        binder_res_types = residue_types[binder_indices]
        target_res_types = residue_types[target_indices]
        pair_type = get_pair_type(binder_res_types, target_res_types)
    else:
        pair_type = 'protein'
    
    # Extract interface PAE: binder rows → target columns
    # This is ONLY binder ↔ target, NOT target-chain ↔ target-chain
    interface_pae = pae_matrix[np.ix_(binder_indices, target_indices)]  # [binder_len, target_len]
    
    # Apply PAE cutoff mask
    valid_mask = interface_pae < pae_cutoff
    
    # Count residues with good PAE values (for n0dom)
    binder_good_residues = np.any(valid_mask, axis=1).sum()
    target_good_residues = np.any(valid_mask, axis=0).sum()
    n0dom = int(binder_good_residues + target_good_residues)
    
    # === Calculate per-binder-residue ipSAE scores (binder → target direction) ===
    ipsae_byres_binder = []
    for i in range(binder_len):
        valid = valid_mask[i]  # Which target residues have good PAE with this binder residue
        if valid.any():
            n0res = valid.sum()
            d0res = calc_d0(n0res, pair_type)
            ptm_vals = ptm_func(interface_pae[i][valid], d0res)
            ipsae_byres_binder.append(ptm_vals.mean())
        else:
            ipsae_byres_binder.append(0.0)
    
    ipsae_byres_binder = np.array(ipsae_byres_binder)
    ipsae_binder_max = float(ipsae_byres_binder.max()) if len(ipsae_byres_binder) > 0 else 0.0
    
    # === Calculate reverse direction: target → binder ===
    interface_pae_rev = pae_matrix[np.ix_(target_indices, binder_indices)]  # [target_len, binder_len]
    valid_mask_rev = interface_pae_rev < pae_cutoff
    
    ipsae_byres_target = []
    for i in range(target_len):
        valid = valid_mask_rev[i]
        if valid.any():
            n0res = valid.sum()
            d0res = calc_d0(n0res, pair_type)
            ptm_vals = ptm_func(interface_pae_rev[i][valid], d0res)
            ipsae_byres_target.append(ptm_vals.mean())
        else:
            ipsae_byres_target.append(0.0)
    
    ipsae_byres_target = np.array(ipsae_byres_target)
    ipsae_target_max = float(ipsae_byres_target.max()) if len(ipsae_byres_target) > 0 else 0.0
    
    # Take max of both directions (as in original ipSAE paper)
    ipsae = max(ipsae_binder_max, ipsae_target_max)
    
    return {
        'ipSAE': round(ipsae, 4),
        'ipSAE_binder_to_target': round(ipsae_binder_max, 4),
        'ipSAE_target_to_binder': round(ipsae_target_max, 4),
        'n0dom': n0dom,
        'pair_type': pair_type,
    }


def calculate_ipsae_from_pae_matrix(
    pae_matrix: np.ndarray,
    asym_id: np.ndarray,
    binder_chain_idx: int = 0,
    pae_cutoff: float = 10.0,
    pad_mask: Optional[np.ndarray] = None,
    residue_types: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """
    Calculate ipSAE from raw PAE matrix and chain IDs.
    Convenience wrapper for when you have numpy arrays directly.
    
    Args:
        pae_matrix: PAE matrix of shape (N, N)
        asym_id: Array of chain indices for each residue (N,)
        binder_chain_idx: Index of binder chain (default 0)
        pae_cutoff: Cutoff for good PAE values
        pad_mask: Optional padding mask (N,). If None, assumes all valid.
        residue_types: Optional array of residue type strings for nucleic acid detection.
    
    Returns:
        Same dict as calculate_ipsae_from_boltz_output
    """
    # Create mock feats dict
    if pad_mask is None:
        pad_mask = np.ones_like(asym_id, dtype=np.float32)
    
    feats = {
        'asym_id': asym_id,
        'token_pad_mask': pad_mask,
    }
    output = {'pae': pae_matrix}
    
    return calculate_ipsae_from_boltz_output(
        output, feats, binder_chain_idx, pae_cutoff, residue_types
    )


def compute_ipsae_from_batch(
    batch: Dict,
    output: Dict,
    binder_chain_idx: int = 0,
    pae_cutoff: float = 10.0,
    residue_types: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """
    Compute ipSAE directly from Boltz batch and output dictionaries.
    
    This is the primary interface for use in the pipeline.
    
    Args:
        batch: Boltz batch dictionary (contains 'asym_id', 'token_pad_mask')
        output: Boltz model output dictionary (contains 'pae')
        binder_chain_idx: Chain index of binder
        pae_cutoff: PAE cutoff threshold
        residue_types: Optional array of residue type strings for nucleic acid detection.
                       Can also check batch for 'res_type' or similar field.
    
    Returns:
        dict with ipSAE metrics
    """
    # Extract features from batch
    feats = {
        'asym_id': batch.get('asym_id'),
        'token_pad_mask': batch.get('token_pad_mask'),
    }
    
    # Try to get residue types from batch if not provided
    if residue_types is None:
        residue_types = batch.get('res_type') or batch.get('residue_types')
    
    return calculate_ipsae_from_boltz_output(
        output, feats, binder_chain_idx, pae_cutoff, residue_types
    )
