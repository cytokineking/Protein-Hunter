"""
AlphaFold3 validation functions.

This module provides AF3 validation for designed binder structures,
including both holo (bound) and apo (unbound) predictions.
"""

import json
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional

from modal_boltz_ph.app import app, cache_volume, af3_weights_volume
from modal_boltz_ph.images import af3_image
from modal_boltz_ph.helpers import get_cif_alignment_json


def _run_af3_single_impl(
    design_id: str,
    binder_seq: str,
    target_seq: str,
    binder_chain: str = "A",
    target_chain: str = "B",
    target_msa: Optional[str] = None,
    af3_msa_mode: str = "none",
    template_path: Optional[str] = None,
    template_chain_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Run AF3 validation on a SINGLE design (holo state).
    
    This predicts the binder-target complex structure using AlphaFold3.
    Supports template structures for improved predictions.
    
    Args:
        design_id: Unique identifier for this design
        binder_seq: Binder protein sequence
        target_seq: Target protein sequence
        binder_chain: Chain ID for binder (default "A")
        target_chain: Chain ID for target (default "B")
        target_msa: Optional MSA content for target (A3M format)
        af3_msa_mode: MSA mode ("none", "reuse")
        template_path: Optional path to template structure file
        template_chain_id: Optional chain ID in template file
    
    Returns:
        Dict with af3_iptm, af3_ptm, af3_plddt, af3_structure, af3_confidence_json
    """
    work_dir = Path(tempfile.mkdtemp())
    af_input_dir = work_dir / "af_input"
    af_output_dir = work_dir / "af_output"
    af_input_dir.mkdir()
    af_output_dir.mkdir()
    
    result = {
        "design_id": design_id,
        "af3_iptm": 0,
        "af3_ptm": 0,
        "af3_plddt": 0,
        "af3_structure": None,
        "af3_confidence_json": None,  # For i_pae calculation in PyRosetta
    }
    
    # Check for AF3 weights
    weights_path = Path("/af3_weights/af3.bin")
    if not weights_path.exists():
        result["error"] = "AF3 weights not found"
        return result
    
    # Process template if provided
    target_template_json = []
    if template_path and Path(template_path).exists():
        try:
            template_alignment = get_cif_alignment_json(
                target_seq, template_path, template_chain_id
            )
            target_template_json = [template_alignment]
            print(f"  {design_id}: Using template from {template_path}")
        except Exception as e:
            print(f"  {design_id}: Template processing failed: {e}")
    
    # Build AF3 JSON input
    af3_input = {
        "name": design_id,
        "modelSeeds": [1],
        "dialect": "alphafold3",
        "version": 1,
        "sequences": []
    }
    
    # BINDER: Always query-only MSA (hallucinated)
    af3_input["sequences"].append({
        "protein": {
            "id": binder_chain,
            "sequence": binder_seq,
            "unpairedMsa": f">query\n{binder_seq}\n",
            "pairedMsa": f">query\n{binder_seq}\n",
            "templates": [],
        }
    })
    
    # TARGET: Handle MSA and templates based on mode
    target_entry = {
        "protein": {
            "id": target_chain,
            "sequence": target_seq,
            "templates": target_template_json,
        }
    }
    
    if af3_msa_mode == "reuse" and target_msa:
        target_entry["protein"]["unpairedMsa"] = target_msa
        target_entry["protein"]["pairedMsa"] = target_msa
    else:
        target_entry["protein"]["unpairedMsa"] = f">query\n{target_seq}\n"
        target_entry["protein"]["pairedMsa"] = f">query\n{target_seq}\n"
    
    af3_input["sequences"].append(target_entry)
    
    # Write JSON
    json_path = af_input_dir / f"{design_id}.json"
    json_path.write_text(json.dumps(af3_input, indent=2))
    
    # Run AF3
    try:
        subprocess.run([
            "python", "/app/alphafold/run_alphafold.py",
            f"--json_path={json_path}",
            "--model_dir=/af3_weights",
            f"--output_dir={af_output_dir}",
            "--run_data_pipeline=false",
            "--run_inference=true",
        ], capture_output=True, text=True, timeout=1800, cwd="/app/alphafold")
    except Exception as e:
        result["error"] = str(e)
        return result
    
    # Read results
    output_subdir = af_output_dir / design_id
    if output_subdir.exists():
        # Try both naming conventions
        confidence_file = output_subdir / f"{design_id}_confidences.json"
        if not confidence_file.exists():
            confidence_file = output_subdir / "confidence.json"
        
        structure_file = output_subdir / f"{design_id}_model.cif"
        if not structure_file.exists():
            structure_file = output_subdir / "model.cif"
        
        if confidence_file.exists():
            try:
                confidence_text = confidence_file.read_text()
                confidence = json.loads(confidence_text)
                
                # Store raw confidence JSON for i_pae calculation in PyRosetta
                result["af3_confidence_json"] = confidence_text
                
                # Get pLDDT - average of atom_plddts
                atom_plddts = confidence.get("atom_plddts", [])
                if atom_plddts:
                    result["af3_plddt"] = sum(atom_plddts) / len(atom_plddts)
                
                # Check for summary file with ipTM
                summary_file = output_subdir / f"{design_id}_summary_confidences.json"
                if summary_file.exists():
                    summary = json.loads(summary_file.read_text())
                    result["af3_iptm"] = summary.get("iptm", summary.get("ptm", 0))
                    result["af3_ptm"] = summary.get("ptm", 0)
                else:
                    result["af3_iptm"] = confidence.get("iptm", confidence.get("ranking_score", 0))
                    result["af3_ptm"] = confidence.get("ptm", 0)
            except Exception as e:
                result["error"] = f"Error reading confidence: {e}"
        
        if structure_file.exists():
            result["af3_structure"] = structure_file.read_text()
    
    return result


def _run_af3_apo_impl(
    design_id: str,
    binder_seq: str,
    binder_chain: str = "A",
) -> Dict[str, Any]:
    """
    Run AF3 on binder ALONE (APO state) for holo-apo RMSD calculation.
    
    This predicts the binder structure in isolation to compare against
    the holo (bound) state for conformational stability analysis.
    
    Args:
        design_id: Unique identifier for this design
        binder_seq: Binder protein sequence
        binder_chain: Chain ID for binder (default "A")
    
    Returns:
        Dict with apo_structure (CIF text) and any errors
    """
    work_dir = Path(tempfile.mkdtemp())
    af_input_dir = work_dir / "af_input"
    af_output_dir = work_dir / "af_output"
    af_input_dir.mkdir()
    af_output_dir.mkdir()
    
    result = {
        "design_id": design_id,
        "apo_structure": None,
        "error": None,
    }
    
    # Check for AF3 weights
    weights_path = Path("/af3_weights/af3.bin")
    if not weights_path.exists():
        result["error"] = "AF3 weights not found"
        return result
    
    # Build AF3 JSON input - BINDER ONLY (APO state)
    apo_name = f"{design_id}_apo"
    af3_input = {
        "name": apo_name,
        "modelSeeds": [1],
        "dialect": "alphafold3",
        "version": 1,
        "sequences": [{
            "protein": {
                "id": binder_chain,
                "sequence": binder_seq,
                "unpairedMsa": f">query\n{binder_seq}\n",
                "pairedMsa": f">query\n{binder_seq}\n",
                "templates": [],
            }
        }]
    }
    
    json_path = af_input_dir / f"{apo_name}.json"
    json_path.write_text(json.dumps(af3_input, indent=2))
    
    # Run AF3
    try:
        subprocess.run([
            "python", "/app/alphafold/run_alphafold.py",
            f"--json_path={json_path}",
            "--model_dir=/af3_weights",
            f"--output_dir={af_output_dir}",
            "--run_data_pipeline=false",
            "--run_inference=true",
        ], capture_output=True, text=True, timeout=1800, cwd="/app/alphafold")
    except subprocess.TimeoutExpired:
        result["error"] = "AF3 APO prediction timed out"
        return result
    except Exception as e:
        result["error"] = str(e)
        return result
    
    # Read APO structure
    output_subdir = af_output_dir / apo_name
    if output_subdir.exists():
        # Try both naming conventions
        structure_file = output_subdir / f"{apo_name}_model.cif"
        if not structure_file.exists():
            structure_file = output_subdir / "model.cif"
        
        if structure_file.exists():
            result["apo_structure"] = structure_file.read_text()
            print(f"  ✓ APO structure generated for {design_id}")
        else:
            result["error"] = "APO structure file not found"
    else:
        result["error"] = f"Output directory not found: {output_subdir}"
    
    return result


# =============================================================================
# GPU-SPECIFIC AF3 HOLO VALIDATION FUNCTIONS
# =============================================================================

@app.function(
    image=af3_image,
    gpu="H100",
    timeout=1800,
    volumes={"/cache": cache_volume, "/af3_weights": af3_weights_volume},
    max_containers=20
)
def run_af3_single_H100(
    design_id: str,
    binder_seq: str,
    target_seq: str,
    binder_chain: str = "A",
    target_chain: str = "B",
    target_msa: Optional[str] = None,
    af3_msa_mode: str = "none",
    template_path: Optional[str] = None,
    template_chain_id: Optional[str] = None
) -> Dict[str, Any]:
    """Run AF3 validation on H100 GPU (80GB VRAM)."""
    return _run_af3_single_impl(
        design_id, binder_seq, target_seq, binder_chain, target_chain,
        target_msa, af3_msa_mode, template_path, template_chain_id
    )

@app.function(
    image=af3_image,
    gpu="A100-80GB",
    timeout=1800,
    volumes={"/cache": cache_volume, "/af3_weights": af3_weights_volume},
    max_containers=20
)
def run_af3_single_A100_80GB(
    design_id: str,
    binder_seq: str,
    target_seq: str,
    binder_chain: str = "A",
    target_chain: str = "B",
    target_msa: Optional[str] = None,
    af3_msa_mode: str = "none",
    template_path: Optional[str] = None,
    template_chain_id: Optional[str] = None
) -> Dict[str, Any]:
    """Run AF3 validation on A100 80GB GPU."""
    return _run_af3_single_impl(
        design_id, binder_seq, target_seq, binder_chain, target_chain,
        target_msa, af3_msa_mode, template_path, template_chain_id
    )

@app.function(
    image=af3_image,
    gpu="A100",
    timeout=1800,
    volumes={"/cache": cache_volume, "/af3_weights": af3_weights_volume},
    max_containers=20
)
def run_af3_single_A100_40GB(
    design_id: str,
    binder_seq: str,
    target_seq: str,
    binder_chain: str = "A",
    target_chain: str = "B",
    target_msa: Optional[str] = None,
    af3_msa_mode: str = "none",
    template_path: Optional[str] = None,
    template_chain_id: Optional[str] = None
) -> Dict[str, Any]:
    """Run AF3 validation on A100 40GB GPU."""
    return _run_af3_single_impl(
        design_id, binder_seq, target_seq, binder_chain, target_chain,
        target_msa, af3_msa_mode, template_path, template_chain_id
    )


# AF3 GPU function mapping for dynamic selection
AF3_GPU_FUNCTIONS = {
    "H100": run_af3_single_H100,
    "A100-80GB": run_af3_single_A100_80GB,
    "A100": run_af3_single_A100_40GB,
    "A100-40GB": run_af3_single_A100_40GB,
}


# =============================================================================
# GPU-SPECIFIC AF3 APO PREDICTION FUNCTIONS
# =============================================================================

@app.function(
    image=af3_image,
    gpu="H100",
    timeout=1800,
    volumes={"/cache": cache_volume, "/af3_weights": af3_weights_volume},
    max_containers=20
)
def run_af3_apo_H100(
    design_id: str,
    binder_seq: str,
    binder_chain: str = "A"
) -> Dict[str, Any]:
    """Run AF3 APO prediction on H100 GPU."""
    return _run_af3_apo_impl(design_id, binder_seq, binder_chain)

@app.function(
    image=af3_image,
    gpu="A100-80GB",
    timeout=1800,
    volumes={"/cache": cache_volume, "/af3_weights": af3_weights_volume},
    max_containers=20
)
def run_af3_apo_A100_80GB(
    design_id: str,
    binder_seq: str,
    binder_chain: str = "A"
) -> Dict[str, Any]:
    """Run AF3 APO prediction on A100 80GB GPU."""
    return _run_af3_apo_impl(design_id, binder_seq, binder_chain)

@app.function(
    image=af3_image,
    gpu="A100",
    timeout=1800,
    volumes={"/cache": cache_volume, "/af3_weights": af3_weights_volume},
    max_containers=20
)
def run_af3_apo_A100_40GB(
    design_id: str,
    binder_seq: str,
    binder_chain: str = "A"
) -> Dict[str, Any]:
    """Run AF3 APO prediction on A100 40GB GPU."""
    return _run_af3_apo_impl(design_id, binder_seq, binder_chain)


# AF3 APO GPU function mapping for dynamic selection
AF3_APO_GPU_FUNCTIONS = {
    "H100": run_af3_apo_H100,
    "A100-80GB": run_af3_apo_A100_80GB,
    "A100": run_af3_apo_A100_40GB,
    "A100-40GB": run_af3_apo_A100_40GB,
}

