"""
Protenix validation functions.

This module provides open-source structure validation using Protenix,
a trainable PyTorch reproduction of AlphaFold 3 from ByteDance.

Protenix is bundled with open-source scoring (OpenMM, FreeSASA, sc-rs)
in a single container for efficiency.

Reference: https://github.com/bytedance/Protenix
"""

import json
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from modal_boltz_ph.app import app, cache_volume, protenix_weights_volume
from modal_boltz_ph.images import protenix_validation_image
from modal_boltz_ph.validation.base import (
    calculate_ipsae_from_pae,
    convert_chain_indices_to_letters,
    normalize_plddt_scale,
)

# Default Protenix model
DEFAULT_PROTENIX_MODEL = "protenix_base_default_v0.5.0"
PROTENIX_WEIGHTS_PATH = "/protenix_weights"

# Global verbose flag (set by _run_protenix_validation_impl)
_PROTENIX_VERBOSE = False

# Protenix model weight URLs (from protenix/web_service/dependency_url.py)
PROTENIX_MODEL_URLS = {
    "protenix_base_default_v0.5.0": "https://af3-dev.tos-cn-beijing.volces.com/release_model/protenix_base_default_v0.5.0.pt",
    "protenix_base_constraint_v0.5.0": "https://af3-dev.tos-cn-beijing.volces.com/release_model/protenix_base_constraint_v0.5.0.pt",
    "protenix_mini_default_v0.5.0": "https://af3-dev.tos-cn-beijing.volces.com/release_model/protenix_mini_default_v0.5.0.pt",
    "protenix_tiny_default_v0.5.0": "https://af3-dev.tos-cn-beijing.volces.com/release_model/protenix_tiny_default_v0.5.0.pt",
}

# CCD cache URLs
PROTENIX_DATA_URLS = {
    "ccd_components_file": "https://af3-dev.tos-cn-beijing.volces.com/release_data/components.v20240608.cif",
    "ccd_components_rdkit_mol_file": "https://af3-dev.tos-cn-beijing.volces.com/release_data/components.v20240608.cif.rdkit_mol.pkl",
    "pdb_cluster_file": "https://af3-dev.tos-cn-beijing.volces.com/release_data/clusters-by-entity-40.txt",
}


# =============================================================================
# PROTENIX WEIGHTS MANAGEMENT
# =============================================================================

from modal_boltz_ph.utils.weights import download_with_progress


# Expected file sizes for verification (approximate, in bytes)
PROTENIX_MODEL_SIZES = {
    "protenix_base_default_v0.5.0": 1.4 * 1024**3,  # ~1.4 GB
    "protenix_mini_default_v0.5.0": 0.5 * 1024**3,  # ~0.5 GB (estimate)
    "protenix_tiny_default_v0.5.0": 0.2 * 1024**3,  # ~0.2 GB (estimate)
}


def ensure_protenix_weights(model_name: str = DEFAULT_PROTENIX_MODEL) -> Path:
    """
    Ensure Protenix model weights are available on the volume.
    
    Downloads weights on first use and caches them on the Modal volume.
    Subsequent calls return immediately if weights exist.
    
    Args:
        model_name: Protenix model name (default: protenix_base_default_v0.5.0)
    
    Returns:
        Path to the weights directory
    """
    import torch
    
    weights_dir = Path(PROTENIX_WEIGHTS_PATH)
    weights_dir.mkdir(parents=True, exist_ok=True)
    
    checkpoint_path = weights_dir / f"{model_name}.pt"
    
    # Check if weights already exist on volume
    if checkpoint_path.exists():
        size_bytes = checkpoint_path.stat().st_size
        size_gb = size_bytes / (1024**3)
        
        # Verify file size is reasonable (at least 90% of expected)
        expected_size = PROTENIX_MODEL_SIZES.get(model_name, 1.0 * 1024**3)
        if size_bytes >= expected_size * 0.9:
            # Also try to verify it's a valid checkpoint
            try:
                ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
                if "model" in ckpt:
                    del ckpt
                    print(f"  ✓ Protenix weights verified: {checkpoint_path} ({size_gb:.2f} GB)")
                    return weights_dir
            except Exception as e:
                print(f"  ⚠ Checkpoint verification failed: {e}")
        
        # File is incomplete or corrupt - delete and re-download
        print(f"  ⚠ Incomplete weights found ({size_gb:.2f} GB, expected ~{expected_size/(1024**3):.2f} GB)")
        print(f"  Deleting partial download and re-downloading...")
        checkpoint_path.unlink()
    
    # Download weights
    print(f"  Downloading Protenix weights ({model_name})...")
    print(f"  This is a one-time operation - weights will be cached on the volume.")
    
    if model_name not in PROTENIX_MODEL_URLS:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(PROTENIX_MODEL_URLS.keys())}")
    
    url = PROTENIX_MODEL_URLS[model_name]
    
    try:
        download_with_progress(url, checkpoint_path)
        
        # Verify the download
        ckpt = torch.load(checkpoint_path, map_location="cpu")
        del ckpt
        
        size_gb = checkpoint_path.stat().st_size / (1024**3)
        print(f"  ✓ Weights downloaded and verified: {checkpoint_path} ({size_gb:.2f} GB)")
        
        # Commit volume changes
        protenix_weights_volume.commit()
        
        return weights_dir
        
    except Exception as e:
        # Clean up partial download
        if checkpoint_path.exists():
            checkpoint_path.unlink()
        print(f"  ✗ Failed to download weights: {e}")
        raise RuntimeError(
            f"Failed to download Protenix weights. "
            f"You can manually download from {url} and upload to the volume."
        )


def _setup_protenix_env(model_name: str = DEFAULT_PROTENIX_MODEL) -> Path:
    """
    Set up Protenix environment with weights from volume.
    
    Ensures weights are downloaded and returns the checkpoint directory path.
    
    Returns:
        Path to the checkpoint directory (for use with load_checkpoint_dir)
    """
    import os
    
    weights_dir = ensure_protenix_weights(model_name)
    
    # Protenix expects checkpoint at {load_checkpoint_dir}/{model_name}.pt
    # Our weights are at /protenix_weights/{model_name}.pt
    # So load_checkpoint_dir = /protenix_weights
    
    return weights_dir


# =============================================================================
# PROTENIX INPUT/OUTPUT UTILITIES
# =============================================================================

def _build_protenix_input(
    design_id: str,
    binder_seq: str,
    target_seq: str,
    msa_dir: Optional[Path] = None,
    work_dir: Optional[Path] = None,
) -> Path:
    """
    Build Protenix JSON input file.
    
    Protenix uses a list-based JSON format with sequences array.
    Chain order: binder first (entity 0), then target chain(s) (entities 1, 2, ...).
    
    Args:
        design_id: Unique identifier for this prediction
        binder_seq: Designed binder sequence
        target_seq: Target sequence(s), colon-separated for multi-chain
        msa_dir: Path to MSA directory (e.g., results_name/msas/)
                 Expected structure: msa_dir/chain_B/msa.a3m, msa_dir/chain_C/msa.a3m, etc.
        work_dir: Working directory for output file (uses tempdir if None)
    
    Returns:
        Path to the generated JSON input file
    """
    if work_dir is None:
        work_dir = Path(tempfile.mkdtemp())
    
    sequences = []
    
    # Binder chain (entity 0) - create single-sequence "MSA" for designed sequence
    # De novo designed proteins have no evolutionary history, so the MSA is just
    # the sequence itself. This tells Protenix "no homologs found" which is correct.
    binder_entry = {
        "proteinChain": {
            "sequence": binder_seq,
            "count": 1,
        }
    }
    
    # Create dummy MSA directory for binder if msa_dir is provided
    if msa_dir:
        binder_msa_dir = msa_dir / "chain_A"
        binder_msa_dir.mkdir(parents=True, exist_ok=True)
        binder_msa_file = binder_msa_dir / "non_pairing.a3m"
        if not binder_msa_file.exists():
            # Single-sequence A3M: just the query sequence with a header
            binder_msa_file.write_text(f">binder\n{binder_seq}\n")
        binder_entry["proteinChain"]["msa"] = {
            "precomputed_msa_dir": str(binder_msa_dir),
            "pairing_db": "uniref100",
        }
    
    sequences.append(binder_entry)
    
    # Target chain(s) (entities 1, 2, ...)
    target_chains = target_seq.split(":") if target_seq else []
    for i, seq in enumerate(target_chains):
        chain_id = chr(ord('B') + i)  # B, C, D, ...
        
        entry = {
            "proteinChain": {
                "sequence": seq,
                "count": 1,
            }
        }
        
        # Check if MSA exists for this chain
        # Protenix expects non_pairing.a3m or pairing.a3m
        if msa_dir:
            chain_msa_dir = msa_dir / f"chain_{chain_id}"
            msa_file = chain_msa_dir / "non_pairing.a3m"
            if chain_msa_dir.exists() and msa_file.exists():
                entry["proteinChain"]["msa"] = {
                    "precomputed_msa_dir": str(chain_msa_dir),
                    "pairing_db": "uniref100",
                }
        
        sequences.append(entry)
    
    # Build Protenix input format (list wrapper)
    protenix_input = [{
        "name": design_id,
        "sequences": sequences,
    }]
    
    json_path = work_dir / "input.json"
    json_str = json.dumps(protenix_input, indent=2)
    json_path.write_text(json_str)
    
    return json_path


def convert_colabfold_msa_to_protenix_format(msa_content: str) -> str:
    """
    Convert ColabFold API MSA format to Protenix-compatible format.
    
    ColabFold API returns MSAs with headers like:
        >101
        SEQUENCE...
        >UniRef100_A0A0A0A0A0/1-100
        SEQUENCE...
    
    Protenix expects headers with pseudo taxonomy IDs for pairing:
        >query
        SEQUENCE...
        >UniRef100_{identifier}_{taxonomy_id}/{range}
        SEQUENCE...
    
    This function adds pseudo taxonomy IDs to enable Protenix's MSA pairing
    data pipeline to work correctly with ColabFold-sourced MSAs.
    
    For single-chain targets (monomers), pairing is not used anyway,
    but the format must still be correct for Protenix to parse the MSA.
    
    Args:
        msa_content: A3M content from ColabFold API
        
    Returns:
        A3M content with Protenix-compatible headers
    """
    lines = msa_content.strip().split('\n')
    converted_lines = []
    
    # Counter for generating pseudo taxonomy IDs
    pseudo_tax_id = 0
    
    for line in lines:
        if line.startswith('>'):
            header = line[1:].strip()
            
            # First sequence is typically ">101" or ">query" - rename to ">query"
            if header.isdigit() or header == 'query':
                converted_lines.append('>query')
            # Already has UniRef100 format with taxonomy ID
            elif header.startswith('UniRef100_') and '/' in header:
                first_part = header.split('/')[0]
                # Check if it already has proper format: UniRef100_xxx_taxid
                if first_part.count('_') >= 2:
                    # Already properly formatted
                    converted_lines.append(line)
                else:
                    # Missing taxonomy ID - add pseudo ID
                    pseudo_tax_id += 1
                    # Insert pseudo taxonomy ID before the /range part
                    if '/' in header:
                        base, range_part = header.rsplit('/', 1)
                        new_header = f">{base}_{pseudo_tax_id}/{range_part}"
                    else:
                        new_header = f">{header}_{pseudo_tax_id}"
                    converted_lines.append(new_header)
            # Handle other UniRef formats without proper structure
            elif 'UniRef' in header or 'UniProt' in header or header.startswith('tr|') or header.startswith('sp|'):
                pseudo_tax_id += 1
                # Create Protenix-compatible header
                # Extract an identifier from the header
                parts = header.split()
                identifier = parts[0].replace('|', '_').replace('/', '_')[:30]  # Truncate long IDs
                converted_lines.append(f">UniRef100_{identifier}_{pseudo_tax_id}/1-1000")
            else:
                # Generic fallback: create a UniRef100 format header
                pseudo_tax_id += 1
                # Clean up the header for use as identifier
                clean_id = ''.join(c if c.isalnum() else '_' for c in header[:20])
                converted_lines.append(f">UniRef100_{clean_id}_{pseudo_tax_id}/1-1000")
        else:
            # Sequence line - keep as is
            converted_lines.append(line)
    
    return '\n'.join(converted_lines)


def ensure_msa_files(
    msa_dir: Path,
    target_msas: Dict[str, str],
) -> None:
    """
    Write MSA content to files if they don't already exist.
    
    Called once at the start of validation to persist MSAs to disk.
    Subsequent validations will find the files and skip regeneration.
    
    Protenix expects MSA files named:
      - non_pairing.a3m (or pairing.a3m for paired MSAs)
    
    The MSA content is post-processed to add pseudo taxonomy IDs
    for Protenix compatibility (see convert_colabfold_msa_to_protenix_format).
    
    Args:
        msa_dir: Base MSA directory (e.g., results_name/msas/)
        target_msas: Dict mapping chain_id -> A3M content
    """
    for chain_id, msa_content in target_msas.items():
        chain_msa_dir = msa_dir / f"chain_{chain_id}"
        # Protenix expects non_pairing.a3m (or pairing.a3m)
        msa_file = chain_msa_dir / "non_pairing.a3m"
        
        # Skip if already exists
        if msa_file.exists():
            continue
        
        # Convert ColabFold MSA format to Protenix-compatible format
        protenix_msa = convert_colabfold_msa_to_protenix_format(msa_content)
        
        # Create directory and write MSA
        chain_msa_dir.mkdir(parents=True, exist_ok=True)
        msa_file.write_text(protenix_msa)


def _parse_protenix_output(
    output_dir: Path,
    design_id: str,
    seed: int = 101,
    sample: int = 0,
) -> Dict[str, Any]:
    """
    Parse Protenix output files.
    
    Output structure:
        output_dir/<name>/<seed>/<name>_<seed>_sample_<n>.cif
        output_dir/<name>/<seed>/<name>_<seed>_summary_confidence_sample_<n>.json
    
    Args:
        output_dir: Base output directory from Protenix
        design_id: Name used in the input JSON
        seed: Random seed used (default 101)
        sample: Sample index (default 0)
    
    Returns:
        Dict with parsed metrics and structure
    """
    # Try multiple directory structures (Protenix output format varies by version)
    # Format 1: output_dir/design_id/seed_XXX/predictions/
    # Format 2: output_dir/design_id/XXX/
    result_dir = output_dir / design_id / f"seed_{seed}" / "predictions"
    if not result_dir.exists():
        result_dir = output_dir / design_id / str(seed)
    
    # Confidence JSON - try multiple naming patterns
    conf_file = result_dir / f"{design_id}_summary_confidence_sample_{sample}.json"
    if not conf_file.exists():
        conf_file = result_dir / f"{design_id}_{seed}_summary_confidence_sample_{sample}.json"
    if not conf_file.exists():
        # Try alternative naming patterns
        alt_patterns = [
            f"{design_id}_{seed}_summary_confidence.json",
            f"summary_confidence_sample_{sample}.json",
            "summary_confidence.json",
        ]
        for pattern in alt_patterns:
            alt_file = result_dir / pattern
            if alt_file.exists():
                conf_file = alt_file
                break
        else:
            # List what's actually in the output directory for debugging
            import subprocess
            try:
                find_result = subprocess.run(
                    ["find", str(output_dir), "-type", "f", "-name", "*.json"],
                    capture_output=True, text=True, timeout=10
                )
                found_files = find_result.stdout.strip() if find_result.returncode == 0 else "Could not list files"
            except Exception:
                found_files = "Could not list files"
            
            raise FileNotFoundError(
                f"Confidence file not found in {result_dir}\n"
                f"Expected one of: {[str(conf_file)] + alt_patterns}\n"
                f"JSON files found in output_dir:\n{found_files}"
            )
    
    confidence = json.loads(conf_file.read_text())
    
    # Structure CIF - try multiple naming patterns
    structure_file = result_dir / f"{design_id}_sample_{sample}.cif"
    if not structure_file.exists():
        structure_file = result_dir / f"{design_id}_{seed}_sample_{sample}.cif"
    if not structure_file.exists():
        # Try alternative patterns
        alt_patterns = [
            f"{design_id}_{seed}.cif",
            f"{design_id}.cif",
            f"sample_{sample}.cif",
            "model.cif",
        ]
        for pattern in alt_patterns:
            alt_file = result_dir / pattern
            if alt_file.exists():
                structure_file = alt_file
                break
    
    structure_cif = structure_file.read_text() if structure_file.exists() else None
    
    # Extract metrics
    # Note: Protenix uses 0-indexed chain IDs in output (0, 1, 2...)
    # We convert to letter-based (A, B, C...) for consistency with AF3
    
    def _list_to_dict(data):
        """Convert list to dict with string indices, or return dict as-is."""
        if isinstance(data, list):
            return {str(i): v for i, v in enumerate(data)}
        return data if data else {}
    
    chain_plddt = _list_to_dict(confidence.get("chain_plddt"))
    chain_ptm = _list_to_dict(confidence.get("chain_ptm"))
    chain_pair_iptm = _list_to_dict(confidence.get("chain_pair_iptm"))
    
    # Protenix returns plddt in 0-100 scale (like AF3), not 0-1 scale
    plddt_raw = confidence.get("plddt", 0.0)
    
    # Full data JSON contains PAE matrix (token_pair_pae)
    # Try to find full_data_sample file for PAE
    full_data_json = None
    full_data_patterns = [
        f"{design_id}_full_data_sample_{sample}.json",
        f"{design_id}_{seed}_full_data_sample_{sample}.json",
        f"full_data_sample_{sample}.json",
        # More patterns based on Protenix output naming conventions
        f"{design_id}_seed_{seed}_full_data_sample_{sample}.json",
        "*full_data*.json",
    ]
    
    for pattern in full_data_patterns:
        if "*" in pattern:
            # Glob pattern
            matches = list(result_dir.glob(pattern))
            if matches:
                full_data_file = matches[0]
                full_data_json = full_data_file.read_text()
                break
        else:
            full_data_file = result_dir / pattern
            if full_data_file.exists():
                full_data_json = full_data_file.read_text()
                break
    
    return {
        "iptm": confidence.get("iptm", 0.0),
        "ptm": confidence.get("ptm", 0.0),
        "plddt": plddt_raw,  # Already 0-100 scale from Protenix
        "chain_plddt": convert_chain_indices_to_letters(chain_plddt),
        "chain_ptm": convert_chain_indices_to_letters(chain_ptm),
        "chain_pair_iptm": convert_chain_indices_to_letters(chain_pair_iptm),
        "ranking_score": confidence.get("ranking_score", 0.0),
        "has_clash": confidence.get("has_clash", False),
        "structure_cif": structure_cif,
        "confidence_json": conf_file.read_text(),
        "full_data_json": full_data_json,  # Contains token_pair_pae for ipSAE
    }


def _run_protenix_prediction(
    input_json_path: Path,
    output_dir: Path,
    seed: int = 101,
    model_name: str = "protenix_base_default_v0.5.0",
    n_sample: int = 1,
    n_cycle: int = 10,
    n_step: int = 200,
    use_msa: bool = True,
    timeout: int = 1800,
    checkpoint_dir: str = PROTENIX_WEIGHTS_PATH,
) -> subprocess.CompletedProcess:
    """
    Run Protenix prediction via subprocess.
    
    Args:
        input_json_path: Path to input JSON file
        output_dir: Output directory for results
        seed: Random seed
        model_name: Protenix model variant
        n_sample: Number of diffusion samples (1 for validation)
        n_cycle: Number of recycling cycles
        n_step: Number of diffusion steps
        use_msa: Whether to use MSA features
        timeout: Timeout in seconds
        checkpoint_dir: Directory containing model weights
    
    Returns:
        subprocess.CompletedProcess result
    """
    import os
    
    # Use python runner directly for more control
    cmd = [
        "python", "-m", "runner.inference",
        "--input_json_path", str(input_json_path),
        "--dump_dir", str(output_dir),
        "--seeds", str(seed),
        "--model_name", model_name,
        "--load_checkpoint_dir", checkpoint_dir,
        "--sample_diffusion.N_sample", str(n_sample),
        "--model.N_cycle", str(n_cycle),
        "--sample_diffusion.N_step", str(n_step),
        # v0.7.0 optimizations for faster inference
        "--enable_tf32", "true",
        "--enable_efficient_fusion", "true",
        "--enable_diffusion_shared_vars_cache", "true",
        # Enable full_data output with PAE matrix (token_pair_pae) for ipSAE calculation
        # This causes Protenix to save full_data_sample_*.json in addition to summary_confidence
        "--need_atom_confidence", "true",
        # MSA usage - disable for APO predictions (binder-only, no MSA available)
        "--use_msa", str(use_msa).lower(),
    ]
    
    # Environment setup for optimized kernels
    env = os.environ.copy()
    # Use torch layernorm - fast_layernorm requires CUDA dev headers for JIT compilation
    # which are not available in the conda environment (only runtime libs)
    # The v0.7.0 flags (tf32, efficient_fusion, shared_vars_cache) still provide speedup
    env["LAYERNORM_TYPE"] = "torch"
    
    # Run prediction
    proc = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=timeout,
        env=env,
    )
    
    return proc


# =============================================================================
# PROTENIX VALIDATION IMPLEMENTATIONS
# =============================================================================

def _run_protenix_holo_impl(
    design_id: str,
    binder_seq: str,
    target_seq: str,
    msa_dir: Optional[Path] = None,
    model_name: str = "protenix_base_default_v0.5.0",
) -> Dict[str, Any]:
    """
    Run Protenix HOLO prediction (binder + target complex).
    
    Args:
        design_id: Unique identifier for this design
        binder_seq: Designed binder sequence
        target_seq: Target sequence(s), colon-separated for multi-chain
        msa_dir: Path to directory containing MSA files for target chains
        model_name: Protenix model variant to use
    
    Returns:
        Dict with prediction results or error
    """
    work_dir = Path(tempfile.mkdtemp())
    output_dir = work_dir / "output"
    output_dir.mkdir()
    
    holo_name = f"{design_id}_holo"
    
    # Build input JSON
    input_json = _build_protenix_input(
        design_id=holo_name,
        binder_seq=binder_seq,
        target_seq=target_seq,
        msa_dir=msa_dir,
        work_dir=work_dir,
    )
    
    print(f"  [{design_id}] Running Protenix HOLO prediction...")
    t0 = time.time()
    
    # Run prediction
    try:
        proc = _run_protenix_prediction(
            input_json_path=input_json,
            output_dir=output_dir,
            model_name=model_name,
            n_sample=1,  # Single sample for validation
        )
    except subprocess.TimeoutExpired:
        return {"error": "Protenix HOLO timed out after 30 minutes"}
    except Exception as e:
        return {"error": f"Protenix HOLO failed: {e}"}
    
    elapsed = time.time() - t0
    
    # Log subprocess output on failure (always) or verbose mode
    # Note: verbose is checked via global VERBOSE flag set in _run_protenix_validation_impl
    show_output = proc.returncode != 0 or _PROTENIX_VERBOSE
    if show_output:
        if proc.stdout:
            stdout_tail = proc.stdout[-1500:] if len(proc.stdout) > 1500 else proc.stdout
            print(f"  [{design_id}] Protenix stdout (last 1500 chars):\n{stdout_tail}")
        if proc.stderr:
            stderr_tail = proc.stderr[-1500:] if len(proc.stderr) > 1500 else proc.stderr
            print(f"  [{design_id}] Protenix stderr (last 1500 chars):\n{stderr_tail}")
    
    if proc.returncode != 0:
        return {
            "error": f"Protenix HOLO failed with code {proc.returncode}",
            "stdout": proc.stdout,
            "stderr": proc.stderr,
        }
    
    # Parse output
    try:
        result = _parse_protenix_output(output_dir, holo_name)
        result["design_id"] = design_id
        result["prediction_time"] = round(elapsed, 1)
        print(f"  [{design_id}] Protenix HOLO complete: ipTM={result['iptm']:.3f}, pLDDT={result['plddt']:.3f} ({elapsed:.1f}s)")
        return result
    except Exception as e:
        return {"error": f"Failed to parse Protenix HOLO output: {e}"}


def _run_protenix_apo_impl(
    design_id: str,
    binder_seq: str,
    model_name: str = "protenix_base_default_v0.5.0",
) -> Dict[str, Any]:
    """
    Run Protenix APO prediction (binder alone).
    
    This predicts the binder structure in isolation to compare against
    the holo (bound) state for conformational stability analysis.
    
    Args:
        design_id: Unique identifier for this design
        binder_seq: Binder protein sequence
        model_name: Protenix model variant to use
    
    Returns:
        Dict with APO structure and metrics or error
    """
    work_dir = Path(tempfile.mkdtemp())
    output_dir = work_dir / "output"
    output_dir.mkdir()
    
    apo_name = f"{design_id}_apo"
    
    # Build input JSON (binder only, no target)
    sequences = [{
        "proteinChain": {
            "sequence": binder_seq,
            "count": 1,
        }
    }]
    
    protenix_input = [{
        "name": apo_name,
        "sequences": sequences,
    }]
    
    input_json = work_dir / "input.json"
    input_json.write_text(json.dumps(protenix_input, indent=2))
    
    print(f"  [{design_id}] Running Protenix APO prediction...")
    t0 = time.time()
    
    # Run prediction (no MSA needed for binder-only)
    try:
        proc = _run_protenix_prediction(
            input_json_path=input_json,
            output_dir=output_dir,
            model_name=model_name,
            n_sample=1,
            use_msa=False,  # Binder is hallucinated
        )
    except subprocess.TimeoutExpired:
        return {"error": "Protenix APO timed out"}
    except Exception as e:
        return {"error": f"Protenix APO failed: {e}"}
    
    elapsed = time.time() - t0
    
    # Log subprocess output on failure (always) or verbose mode
    show_output = proc.returncode != 0 or _PROTENIX_VERBOSE
    if show_output:
        if proc.stdout:
            stdout_tail = proc.stdout[-1500:] if len(proc.stdout) > 1500 else proc.stdout
            print(f"  [{design_id}] Protenix APO stdout (last 1500 chars):\n{stdout_tail}")
        if proc.stderr:
            stderr_tail = proc.stderr[-1500:] if len(proc.stderr) > 1500 else proc.stderr
            print(f"  [{design_id}] Protenix APO stderr (last 1500 chars):\n{stderr_tail}")
    
    if proc.returncode != 0:
        return {
            "error": f"Protenix APO failed with code {proc.returncode}",
            "stdout": proc.stdout,
            "stderr": proc.stderr,
        }
    
    # Parse output
    try:
        result = _parse_protenix_output(output_dir, apo_name)
        result["prediction_time"] = round(elapsed, 1)
        print(f"  [{design_id}] Protenix APO complete: pLDDT={result['plddt']:.3f} ({elapsed:.1f}s)")
        return result
    except Exception as e:
        return {"error": f"Failed to parse Protenix APO output: {e}"}


def calculate_protenix_ipsae(
    full_data_json: str,
    binder_length: int,
    target_length: int,
) -> Dict[str, float]:
    """
    Calculate ipSAE from Protenix full_data JSON.
    
    Note: Protenix stores the PAE matrix in full_data_sample*.json under 
    the key 'token_pair_pae', NOT in summary_confidence*.json.
    
    Args:
        full_data_json: Raw JSON string from Protenix full_data_sample output
        binder_length: Number of residues in binder
        target_length: Total number of residues in target chain(s)
    
    Returns:
        Dict with ipSAE values
    """
    result = {'protenix_ipsae': 0.0}
    
    if not full_data_json:
        return result
    
    try:
        full_data = json.loads(full_data_json)
        # Protenix uses 'token_pair_pae' key for the PAE matrix
        pae_data = full_data.get("token_pair_pae", [])
        
        if not pae_data:
            # Also try 'pae' as fallback
            pae_data = full_data.get("pae", [])
        
        if not pae_data:
            return result
        
        pae_matrix = np.array(pae_data)
        
        ipsae_result = calculate_ipsae_from_pae(
            pae_matrix,
            binder_length=binder_length,
            target_length=target_length,
        )
        
        result['protenix_ipsae'] = ipsae_result.get('ipsae', 0.0)
        result['protenix_ipsae_binder_to_target'] = ipsae_result.get('ipsae_binder_to_target', 0.0)
        result['protenix_ipsae_target_to_binder'] = ipsae_result.get('ipsae_target_to_binder', 0.0)
        
    except Exception as e:
        print(f"  Warning: Protenix ipSAE calculation failed: {e}")
    
    return result


# =============================================================================
# MODAL GPU FUNCTIONS
# =============================================================================

@app.function(
    image=protenix_validation_image,
    gpu="A100",
    timeout=3600,
    volumes={"/cache": cache_volume, PROTENIX_WEIGHTS_PATH: protenix_weights_volume},
    max_containers=20,
)
def run_protenix_validation_A100(
    design_id: str,
    binder_seq: str,
    target_seq: str,
    target_msas: Optional[Dict[str, str]] = None,
    run_scoring: bool = True,
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    Run complete Protenix validation: HOLO + APO + scoring (A100 GPU).
    
    All steps run in a single container for efficiency.
    """
    return _run_protenix_validation_impl(
        design_id, binder_seq, target_seq, target_msas, run_scoring, verbose
    )


@app.function(
    image=protenix_validation_image,
    gpu="A100-80GB",
    timeout=3600,
    volumes={"/cache": cache_volume, PROTENIX_WEIGHTS_PATH: protenix_weights_volume},
    max_containers=20,
)
def run_protenix_validation_A100_80GB(
    design_id: str,
    binder_seq: str,
    target_seq: str,
    target_msas: Optional[Dict[str, str]] = None,
    run_scoring: bool = True,
    verbose: bool = False,
) -> Dict[str, Any]:
    """Run complete Protenix validation on A100-80GB GPU."""
    return _run_protenix_validation_impl(
        design_id, binder_seq, target_seq, target_msas, run_scoring, verbose
    )


@app.function(
    image=protenix_validation_image,
    gpu="H100",
    timeout=3600,
    volumes={"/cache": cache_volume, PROTENIX_WEIGHTS_PATH: protenix_weights_volume},
    max_containers=20,
)
def run_protenix_validation_H100(
    design_id: str,
    binder_seq: str,
    target_seq: str,
    target_msas: Optional[Dict[str, str]] = None,
    run_scoring: bool = True,
    verbose: bool = False,
) -> Dict[str, Any]:
    """Run complete Protenix validation on H100 GPU."""
    return _run_protenix_validation_impl(
        design_id, binder_seq, target_seq, target_msas, run_scoring, verbose
    )


@app.function(
    image=protenix_validation_image,
    gpu="L40S",
    timeout=3600,
    volumes={"/cache": cache_volume, PROTENIX_WEIGHTS_PATH: protenix_weights_volume},
    max_containers=20,
)
def run_protenix_validation_L40S(
    design_id: str,
    binder_seq: str,
    target_seq: str,
    target_msas: Optional[Dict[str, str]] = None,
    run_scoring: bool = True,
    verbose: bool = False,
) -> Dict[str, Any]:
    """Run complete Protenix validation on L40S GPU."""
    return _run_protenix_validation_impl(
        design_id, binder_seq, target_seq, target_msas, run_scoring, verbose
    )


def _run_protenix_validation_impl(
    design_id: str,
    binder_seq: str,
    target_seq: str,
    target_msas: Optional[Dict[str, str]] = None,
    run_scoring: bool = True,
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    Core implementation for Protenix validation.
    
    Runs HOLO + APO predictions, calculates ipSAE, and optionally runs
    open-source interface scoring.
    
    Args:
        design_id: Unique identifier for this design
        binder_seq: Designed binder sequence
        target_seq: Target sequence(s), colon-separated for multi-chain
        target_msas: Dict mapping chain_id -> A3M content for target chains
        run_scoring: Whether to run open-source interface scoring
        verbose: Enable verbose logging
    
    Returns:
        Dict with validation metrics and structures
    """
    from modal_boltz_ph.scoring.opensource import (
        _run_opensource_scoring_impl,
        configure_verbose,
    )
    
    configure_verbose(verbose)
    
    # Set global verbose flag for subprocess output logging
    global _PROTENIX_VERBOSE
    _PROTENIX_VERBOSE = verbose
    
    # Ensure Protenix weights are available
    _setup_protenix_env()
    
    result = {
        "design_id": design_id,
        "validation_model": "protenix",
    }
    
    # Setup MSA directory if MSAs provided
    msa_dir = None
    if target_msas:
        msa_dir = Path(tempfile.mkdtemp()) / "msas"
        msa_dir.mkdir(parents=True)
        if verbose:
            print(f"  [Protenix] MSA setup: {len(target_msas)} target chain(s)")
        ensure_msa_files(msa_dir, target_msas)
    
    # Calculate target length for ipSAE
    target_chains = target_seq.split(":") if target_seq else []
    total_target_length = sum(len(seq) for seq in target_chains)
    
    # ========== 1. HOLO PREDICTION (binder + target complex) ==========
    holo_result = _run_protenix_holo_impl(
        design_id, binder_seq, target_seq, msa_dir
    )
    
    if "error" in holo_result:
        result["error"] = holo_result["error"]
        return result
    
    # pLDDT is already in 0-100 scale from Protenix (like AF3)
    holo_plddt_scaled = holo_result["plddt"]
    
    # Calculate ipSAE from HOLO full_data (contains token_pair_pae)
    ipsae_result = {}
    if holo_result.get("full_data_json"):
        ipsae_result = calculate_protenix_ipsae(
            holo_result["full_data_json"],
            binder_length=len(binder_seq),
            target_length=total_target_length,
        )
    
    result.update({
        # Core metrics (prefixed with validation model for clarity in mixed pipelines)
        "protenix_iptm": holo_result["iptm"],
        "protenix_ptm": holo_result["ptm"],
        "protenix_plddt": holo_plddt_scaled,
        "protenix_ipsae": ipsae_result.get("protenix_ipsae", 0.0),
        # Also expose as generic af3_* keys for pipeline compatibility
        "af3_iptm": holo_result["iptm"],
        "af3_ptm": holo_result["ptm"],
        "af3_plddt": holo_plddt_scaled,
        "af3_ipsae": ipsae_result.get("protenix_ipsae", 0.0),
        # Additional Protenix-specific metrics
        "chain_pair_iptm": holo_result.get("chain_pair_iptm", {}),
        "ranking_score": holo_result.get("ranking_score", 0.0),
        "has_clash": holo_result.get("has_clash", False),
        # Structure outputs
        "holo_structure": holo_result.get("structure_cif"),
        "af3_structure": holo_result.get("structure_cif"),  # Alias for compatibility
        "holo_confidence_json": holo_result.get("confidence_json"),
        "af3_confidence_json": holo_result.get("confidence_json"),  # Alias
        # Full data JSON with PAE matrix for debugging/analysis
        "holo_full_data_json": holo_result.get("full_data_json"),
    })
    
    # ========== 2. APO PREDICTION (binder alone) ==========
    apo_result = _run_protenix_apo_impl(design_id, binder_seq)
    
    if "error" not in apo_result:
        result["apo_structure"] = apo_result.get("structure_cif")
        result["apo_plddt"] = apo_result.get("plddt", 0)  # Already 0-100 scale
    
    # ========== 3. OPEN-SOURCE SCORING (if enabled) ==========
    if run_scoring and result.get("holo_structure"):
        print(f"  [{design_id}] Running open-source interface scoring...")
        
        try:
            scoring_result = _run_opensource_scoring_impl(
                design_id=design_id,
                af3_structure=result["holo_structure"],
                af3_iptm=result["af3_iptm"],
                af3_ptm=result["af3_ptm"],
                af3_plddt=result["af3_plddt"],
                binder_chain="A",
                target_chain="B",
                apo_structure=result.get("apo_structure"),
                af3_confidence_json=result.get("holo_confidence_json"),
                target_type="protein",
                verbose=verbose,
            )
            
            # Merge scoring results
            result.update(scoring_result)
            
            status_str = "ACCEPTED" if scoring_result.get("accepted") else f"REJECTED ({scoring_result.get('rejection_reason', 'unknown')})"
            print(f"  [{design_id}] Scoring: SC={scoring_result.get('interface_sc', 0):.2f}, "
                  f"dSASA={scoring_result.get('interface_dSASA', 0):.1f} → {status_str}")
            
        except Exception as e:
            print(f"  [{design_id}] Open-source scoring failed: {e}")
            result["scoring_error"] = str(e)
            result["accepted"] = False
            result["rejection_reason"] = f"Scoring error: {e}"
    
    return result


# =============================================================================
# STANDALONE PREDICTION FUNCTIONS (for use with separate scoring containers)
# =============================================================================

@app.function(
    image=protenix_validation_image,
    gpu="A100",
    timeout=1800,
    volumes={"/cache": cache_volume, PROTENIX_WEIGHTS_PATH: protenix_weights_volume},
    max_containers=20,
)
def run_protenix_holo(
    design_id: str,
    binder_seq: str,
    target_seq: str,
    target_msas: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """Run Protenix HOLO prediction only (for use with separate scoring)."""
    _setup_protenix_env()
    
    msa_dir = None
    if target_msas:
        msa_dir = Path(tempfile.mkdtemp()) / "msas"
        msa_dir.mkdir(parents=True)
        ensure_msa_files(msa_dir, target_msas)
    
    return _run_protenix_holo_impl(design_id, binder_seq, target_seq, msa_dir)


@app.function(
    image=protenix_validation_image,
    gpu="A100",
    timeout=1800,
    volumes={"/cache": cache_volume, PROTENIX_WEIGHTS_PATH: protenix_weights_volume},
    max_containers=20,
)
def run_protenix_apo(
    design_id: str,
    binder_seq: str,
) -> Dict[str, Any]:
    """Run Protenix APO prediction only."""
    _setup_protenix_env()
    return _run_protenix_apo_impl(design_id, binder_seq)


# =============================================================================
# GPU FUNCTION MAPPING
# =============================================================================

PROTENIX_GPU_FUNCTIONS = {
    "A100": run_protenix_validation_A100,
    "A100-40GB": run_protenix_validation_A100,
    "A100-80GB": run_protenix_validation_A100_80GB,
    "H100": run_protenix_validation_H100,
    "L40S": run_protenix_validation_L40S,
}

DEFAULT_PROTENIX_GPU = "A100"
