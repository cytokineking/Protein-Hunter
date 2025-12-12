"""
Local Protenix validation utilities.

This module provides Protenix-based structure validation for the local Boltz pipeline.
It uses PersistentProtenixRunner to run Protenix directly via Python API (not subprocess),
keeping the model loaded in GPU memory across multiple predictions for ~4x faster throughput.

Key features:
  - Direct Python API integration with Protenix (no subprocess overhead)
  - Persistent model loading via singleton PersistentProtenixRunner
  - HOLO (binder+target) and APO (binder-only) predictions
  - ipSAE calculation from PAE matrices
  - Results returned using unified `val_*` schema (val_iptm, val_plddt, etc.)

Weight management:
  Protenix weights are normally downloaded during setup.sh installation to
  ~/.protein-hunter/protenix_weights/. The ensure_protenix_weights() function
  provides a fallback for environments where setup.sh wasn't run or weights
  need re-downloading.

See also:
  - boltz_ph/validation/protenix_runner.py: PersistentProtenixRunner implementation
  - modal_boltz_ph/validation/protenix.py: Modal cloud version (uses subprocess CLI)
"""

from __future__ import annotations

import json
import os
import tempfile
import time
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from utils.convert import download_with_progress


# =============================================================================
# CONSTANTS
# =============================================================================

DEFAULT_PROTENIX_MODEL = "protenix_base_default_v0.5.0"

# Protenix model weight URLs (from Protenix official dependency_url.py)
PROTENIX_MODEL_URLS = {
    "protenix_base_default_v0.5.0": "https://af3-dev.tos-cn-beijing.volces.com/release_model/protenix_base_default_v0.5.0.pt",
    "protenix_base_constraint_v0.5.0": "https://af3-dev.tos-cn-beijing.volces.com/release_model/protenix_base_constraint_v0.5.0.pt",
    "protenix_mini_default_v0.5.0": "https://af3-dev.tos-cn-beijing.volces.com/release_model/protenix_mini_default_v0.5.0.pt",
    "protenix_tiny_default_v0.5.0": "https://af3-dev.tos-cn-beijing.volces.com/release_model/protenix_tiny_default_v0.5.0.pt",
}

# Approximate expected sizes for verification
PROTENIX_MODEL_SIZES = {
    "protenix_base_default_v0.5.0": 1.4 * 1024**3,
    "protenix_mini_default_v0.5.0": 0.5 * 1024**3,
    "protenix_tiny_default_v0.5.0": 0.2 * 1024**3,
}

# Local repo root for Protenix
PROTENIX_REPO_ROOT = Path(__file__).resolve().parents[2] / "Protenix"

# Global verbose toggle set by run_protenix_validation_persistent
_PROTENIX_VERBOSE = False


# =============================================================================
# SHARED VALIDATION UTILITIES
# =============================================================================

def calculate_ipsae_from_pae(
    pae_matrix: np.ndarray,
    binder_length: int,
    target_length: int,
    pae_cutoff: float = 10.0,
) -> Dict[str, float]:
    """
    Calculate ipSAE from PAE matrix. Ported from Modal shared validation base.
    """
    result = {
        "ipsae": 0.0,
        "ipsae_binder_to_target": 0.0,
        "ipsae_target_to_binder": 0.0,
    }
    if pae_matrix is None or len(pae_matrix) == 0:
        return result

    try:
        total_length = binder_length + target_length
        if pae_matrix.ndim == 1:
            expected_size = total_length * total_length
            if len(pae_matrix) != expected_size:
                return result
            pae_matrix = pae_matrix.reshape(total_length, total_length)

        if pae_matrix.shape != (total_length, total_length):
            return result

        binder_indices = np.arange(binder_length)
        target_indices = np.arange(binder_length, total_length)

        def ptm_func(x: np.ndarray, d0: float) -> np.ndarray:
            return 1.0 / (1.0 + (x / d0) ** 2.0)

        def calc_d0(L: int) -> float:
            L = float(max(L, 27))
            d0 = 1.24 * (L - 15) ** (1.0 / 3.0) - 1.8
            return max(1.0, d0)

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

        ipsae = max(ipsae_binder_max, ipsae_target_max)
        result["ipsae"] = round(ipsae, 4)
        result["ipsae_binder_to_target"] = round(ipsae_binder_max, 4)
        result["ipsae_target_to_binder"] = round(ipsae_target_max, 4)
    except Exception:
        pass

    return result


# =============================================================================
# WEIGHTS MANAGEMENT
# =============================================================================

def _get_weights_dir() -> Path:
    env_dir = os.environ.get("PROTENIX_WEIGHTS_DIR")
    if env_dir:
        return Path(env_dir).expanduser()
    return Path("~/.protein-hunter/protenix_weights").expanduser()


def ensure_protenix_weights(model_name: str = DEFAULT_PROTENIX_MODEL) -> Path:
    """
    Ensure Protenix weights are present locally. Downloads on first use.
    """
    import torch

    weights_dir = _get_weights_dir()
    weights_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = weights_dir / f"{model_name}.pt"
    if checkpoint_path.exists():
        size_bytes = checkpoint_path.stat().st_size
        expected_size = PROTENIX_MODEL_SIZES.get(model_name, 1.0 * 1024**3)
        if size_bytes >= expected_size * 0.9:
            try:
                ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
                if isinstance(ckpt, dict) and "model" in ckpt:
                    del ckpt
                    if _PROTENIX_VERBOSE:
                        print(f"  ✓ Protenix weights verified: {checkpoint_path}")
                    return weights_dir
            except Exception:
                pass
        print("  ⚠ Incomplete/corrupt Protenix weights found; re-downloading.")
        try:
            checkpoint_path.unlink()
        except Exception:
            pass

    if model_name not in PROTENIX_MODEL_URLS:
        raise ValueError(f"Unknown Protenix model '{model_name}'. Available: {list(PROTENIX_MODEL_URLS)}")

    url = PROTENIX_MODEL_URLS[model_name]
    print(f"  Downloading Protenix weights ({model_name}) to {checkpoint_path}...")
    ok = download_with_progress(url, str(checkpoint_path))
    if not ok:
        raise RuntimeError(f"Failed to download Protenix weights from {url}")

    # Basic verification
    try:
        _ = torch.load(checkpoint_path, map_location="cpu")
    except Exception as e:
        raise RuntimeError(f"Downloaded weights appear invalid: {e}")

    return weights_dir


# =============================================================================
# MSA HANDLING
# =============================================================================

def convert_colabfold_msa_to_protenix_format(msa_content: str) -> str:
    """Ported from Modal. Adds pseudo taxonomy IDs to ColabFold MSAs."""
    lines = msa_content.strip().split("\n")
    converted_lines: list[str] = []
    pseudo_tax_id = 0

    for line in lines:
        if line.startswith(">"):
            header = line[1:].strip()
            if header.isdigit() or header == "query":
                converted_lines.append(">query")
            elif header.startswith("UniRef100_") and "/" in header:
                first_part = header.split("/")[0]
                if first_part.count("_") >= 2:
                    converted_lines.append(line)
                else:
                    pseudo_tax_id += 1
                    base, range_part = header.rsplit("/", 1)
                    converted_lines.append(f">{base}_{pseudo_tax_id}/{range_part}")
            elif (
                "UniRef" in header
                or "UniProt" in header
                or header.startswith("tr|")
                or header.startswith("sp|")
            ):
                pseudo_tax_id += 1
                parts = header.split()
                identifier = parts[0].replace("|", "_").replace("/", "_")[:30]
                converted_lines.append(f">UniRef100_{identifier}_{pseudo_tax_id}/1-1000")
            else:
                pseudo_tax_id += 1
                clean_id = "".join(c if c.isalnum() else "_" for c in header[:20])
                converted_lines.append(f">UniRef100_{clean_id}_{pseudo_tax_id}/1-1000")
        else:
            converted_lines.append(line)

    return "\n".join(converted_lines)


def ensure_msa_files(msa_dir: Path, target_msas: Dict[str, str]) -> None:
    """
    Persist target MSAs to disk in Protenix expected layout.
    """
    for chain_id, msa_content in target_msas.items():
        chain_msa_dir = msa_dir / f"chain_{chain_id}"
        msa_file = chain_msa_dir / "non_pairing.a3m"
        if msa_file.exists():
            continue
        protenix_msa = convert_colabfold_msa_to_protenix_format(msa_content)
        chain_msa_dir.mkdir(parents=True, exist_ok=True)
        msa_file.write_text(protenix_msa)


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
    Build Protenix JSON input file for HOLO predictions.

    Binder is entity 0 (chain A), target chains follow (B/C/...).
    """
    if work_dir is None:
        work_dir = Path(tempfile.mkdtemp())

    sequences = []

    binder_entry: Dict[str, Any] = {
        "proteinChain": {"sequence": binder_seq, "count": 1}
    }

    if msa_dir:
        binder_msa_dir = msa_dir / "chain_A"
        binder_msa_dir.mkdir(parents=True, exist_ok=True)
        binder_msa_file = binder_msa_dir / "non_pairing.a3m"
        if not binder_msa_file.exists():
            binder_msa_file.write_text(f">binder\n{binder_seq}\n")
        binder_entry["proteinChain"]["msa"] = {
            "precomputed_msa_dir": str(binder_msa_dir),
            "pairing_db": "uniref100",
        }

    sequences.append(binder_entry)

    target_chains = target_seq.split(":") if target_seq else []
    for i, seq in enumerate(target_chains):
        chain_id = chr(ord("B") + i)
        target_entry: Dict[str, Any] = {
            "proteinChain": {"sequence": seq, "count": 1}
        }
        if msa_dir:
            chain_msa_dir = msa_dir / f"chain_{chain_id}"
            if chain_msa_dir.exists():
                target_entry["proteinChain"]["msa"] = {
                    "precomputed_msa_dir": str(chain_msa_dir),
                    "pairing_db": "uniref100",
                }
        sequences.append(target_entry)

    protenix_input = [{
        "name": design_id,
        "sequences": sequences,
        "use_msa": True,
    }]

    json_path = work_dir / "query.json"
    json_path.write_text(json.dumps(protenix_input, indent=2))
    return json_path


def _parse_protenix_output(
    output_dir: Path,
    design_id: str,
    seed: int = 101,
    sample: int = 0,
) -> Dict[str, Any]:
    """
    Parse Protenix output for a single prediction name.
    """
    result_dir = output_dir / design_id / f"seed_{seed}" / "predictions"
    if not result_dir.exists():
        result_dir = output_dir / design_id / str(seed)

    conf_file = result_dir / f"{design_id}_summary_confidence_sample_{sample}.json"
    if not conf_file.exists():
        conf_file = result_dir / f"{design_id}_{seed}_summary_confidence_sample_{sample}.json"
    if not conf_file.exists():
        for pattern in [
            f"{design_id}_{seed}_summary_confidence.json",
            f"summary_confidence_sample_{sample}.json",
            "summary_confidence.json",
        ]:
            alt_file = result_dir / pattern
            if alt_file.exists():
                conf_file = alt_file
                break

    cif_file = result_dir / f"{design_id}_{seed}_sample_{sample}.cif"
    if not cif_file.exists():
        cif_file = result_dir / f"{design_id}_seed_{seed}_sample_{sample}.cif"
    if not cif_file.exists():
        for pattern in [
            f"{design_id}_sample_{sample}.cif",
            f"{design_id}.cif",
            "*.cif",
        ]:
            matches = list(result_dir.glob(pattern))
            if matches:
                cif_file = matches[0]
                break

    if not conf_file.exists() or not cif_file.exists():
        raise FileNotFoundError(f"Missing Protenix outputs in {result_dir}")

    confidence = json.loads(conf_file.read_text())
    structure_cif = cif_file.read_text()

    plddt_raw = float(confidence.get("plddt", 0.0))

    # Optional chain-level metrics
    chain_plddt = confidence.get("chain_plddt", {})
    chain_ptm = confidence.get("chain_ptm", {})
    chain_pair_iptm = confidence.get("chain_pair_iptm", {})

    full_data_json = ""
    for pattern in [
        f"full_data_sample_{sample}.json",
        f"{design_id}_full_data_sample_{sample}.json",
        "full_data.json",
    ]:
        full_data_file = result_dir / pattern
        if full_data_file.exists():
            full_data_json = full_data_file.read_text()
            break

    return {
        "iptm": confidence.get("iptm", 0.0),
        "ptm": confidence.get("ptm", 0.0),
        "plddt": plddt_raw,
        "chain_plddt": chain_plddt,
        "chain_ptm": chain_ptm,
        "chain_pair_iptm": chain_pair_iptm,
        "ranking_score": confidence.get("ranking_score", 0.0),
        "has_clash": confidence.get("has_clash", False),
        "structure_cif": structure_cif,
        "confidence_json": conf_file.read_text(),
        "full_data_json": full_data_json,
    }




def calculate_protenix_ipsae(
    full_data_json: str,
    binder_length: int,
    target_length: int,
) -> Dict[str, float]:
    """Calculate ipSAE from Protenix full_data JSON."""
    result: Dict[str, float] = {"protenix_ipsae": 0.0}
    if not full_data_json:
        return result
    try:
        full_data = json.loads(full_data_json)
        pae_data = full_data.get("token_pair_pae") or full_data.get("pae") or []
        if not pae_data:
            return result
        pae_matrix = np.array(pae_data)
        ipsae_result = calculate_ipsae_from_pae(
            pae_matrix,
            binder_length=binder_length,
            target_length=target_length,
        )
        result["protenix_ipsae"] = ipsae_result.get("ipsae", 0.0)
        result["protenix_ipsae_binder_to_target"] = ipsae_result.get("ipsae_binder_to_target", 0.0)
        result["protenix_ipsae_target_to_binder"] = ipsae_result.get("ipsae_target_to_binder", 0.0)
    except Exception as e:
        print(f"  Warning: Protenix ipSAE calculation failed: {e}")
    return result


# =============================================================================
# PUBLIC ENTRY POINT (Persistent Runner - Model Stays Loaded, ~6x Faster)
# =============================================================================

def run_protenix_validation_persistent(
    design_id: str,
    binder_seq: str,
    target_seq: str,
    target_msas: Optional[Dict[str, str]] = None,
    verbose: bool = False,
    model_name: str = DEFAULT_PROTENIX_MODEL,
) -> Dict[str, Any]:
    """
    Run Protenix HOLO (+ APO) validation using persistent runner.
    
    This keeps the Protenix model loaded in GPU memory across multiple
    predictions, reducing per-design time from ~105s to ~25s after the
    first load (~70s one-time cost).
    
    Args:
        design_id: Unique identifier for this design
        binder_seq: Binder sequence (chain A)
        target_seq: Target sequence(s), colon-separated for multi-chain
        target_msas: Optional dict mapping chain_id -> MSA content
        verbose: Print detailed timing info
        model_name: Protenix model name (default: protenix_base_default_v0.5.0)
    
    Returns:
        Dict compatible with unified val_* schema
    """
    global _PROTENIX_VERBOSE
    _PROTENIX_VERBOSE = verbose
    
    # Initialize result with defaults
    result: Dict[str, Any] = {
        "val_iptm": 0.0,
        "val_ipsae": 0.0,
        "val_ptm": 0.0,
        "val_plddt": 0.0,
        "val_structure": None,
        "val_confidence_json": None,
        "apo_structure": None,
    }
    
    # Ensure weights are available
    try:
        ensure_protenix_weights(model_name)
    except Exception as e:
        return {**result, "error": f"Protenix weights unavailable: {e}"}
    
    # Get/create the persistent runner
    try:
        from boltz_ph.validation.protenix_runner import PersistentProtenixRunner
        
        runner = PersistentProtenixRunner.get_instance()
        load_time = runner.ensure_loaded()
        
        if load_time > 0:
            print(f"  ✓ Protenix model loaded in {load_time:.1f}s (one-time cost)")
        
    except Exception as e:
        return {**result, "error": f"Failed to initialize persistent runner: {e}"}
    
    # Calculate target length for ipSAE
    target_chains = target_seq.split(":") if target_seq else []
    total_target_length = sum(len(seq) for seq in target_chains)
    
    # Run HOLO prediction
    try:
        holo_start = time.time()
        holo_result = runner.predict_holo(
            design_id=design_id,
            binder_seq=binder_seq,
            target_seq=target_seq,
            target_msas=target_msas,
        )
        holo_elapsed = time.time() - holo_start
        
        if verbose:
            print(f"  HOLO prediction: {holo_elapsed:.1f}s")
        
        if holo_result.get("error"):
            return {**result, "error": holo_result["error"]}
        
    except Exception as e:
        return {**result, "error": f"HOLO prediction failed: {e}"}
    
    # Calculate ipSAE from PAE matrix
    ipsae_result: Dict[str, float] = {}
    if holo_result.get("full_data_json"):
        ipsae_result = calculate_protenix_ipsae(
            holo_result["full_data_json"],
            binder_length=len(binder_seq),
            target_length=total_target_length,
        )
    
    # Build unified result
    result.update({
        "protenix_iptm": holo_result.get("iptm", 0.0),
        "protenix_ptm": holo_result.get("ptm", 0.0),
        "protenix_plddt": holo_result.get("plddt", 0.0),
        "protenix_ipsae": ipsae_result.get("protenix_ipsae", 0.0),
        "val_iptm": holo_result.get("iptm", 0.0),
        "val_ptm": holo_result.get("ptm", 0.0),
        "val_plddt": holo_result.get("plddt", 0.0),
        "val_ipsae": ipsae_result.get("protenix_ipsae", 0.0),
        "val_structure": holo_result.get("structure_cif"),
        "val_confidence_json": holo_result.get("confidence_json"),
        "chain_pair_iptm": holo_result.get("chain_pair_iptm", {}),
        "ranking_score": holo_result.get("ranking_score", 0.0),
        "has_clash": holo_result.get("has_clash", False),
        "holo_time": holo_result.get("total_time", 0.0),
        "holo_forward_time": holo_result.get("forward_time", 0.0),
    })
    
    # Run APO prediction (binder only)
    try:
        apo_start = time.time()
        apo_result = runner.predict_apo(
            design_id=design_id,
            binder_seq=binder_seq,
        )
        apo_elapsed = time.time() - apo_start
        
        if verbose:
            print(f"  APO prediction: {apo_elapsed:.1f}s")
        
        if "error" not in apo_result:
            result["apo_structure"] = apo_result.get("structure_cif")
            result["apo_plddt"] = apo_result.get("plddt", 0.0)
            result["apo_time"] = apo_result.get("total_time", 0.0)
        
    except Exception as e:
        # APO failure is non-fatal
        if verbose:
            print(f"  Warning: APO prediction failed: {e}")
    
    return result


def shutdown_persistent_runner() -> None:
    """
    Explicitly shutdown the persistent Protenix runner and free GPU memory.
    
    Call this at the end of validation or when switching to another model.
    """
    try:
        from boltz_ph.validation.protenix_runner import PersistentProtenixRunner
        PersistentProtenixRunner.shutdown()
    except Exception:
        pass
