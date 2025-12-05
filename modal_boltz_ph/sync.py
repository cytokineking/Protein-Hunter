"""
Result streaming and synchronization utilities.

This module handles streaming results from Modal GPU containers to the local filesystem
via the Modal Dict for real-time progress monitoring.
"""

import threading
import time
from pathlib import Path
from typing import Dict

from modal_boltz_ph.app import results_dict


def _stream_result(
    run_id: str,
    design_idx: int,
    cycle: int,
    iptm: float,
    ipsae: float,
    plddt: float,
    iplddt: float,
    seq: str,
    pdb_file: Path,
    binder_name: str,
    target_seqs: str = "",
    contact_residues: str = "",
    msa_mode: str = "empty",
    cyclic: bool = False,
    alanine_count: int = 0,
    # AF3 reconstruction fields
    ligand_smiles: str = "",
    ligand_ccd: str = "",
    nucleic_seq: str = "",
    nucleic_type: str = "",
    template_path: str = "",
    template_mapping: str = "",
):
    """
    Stream a cycle result to the Modal Dict with full config for CSV.
    
    This function is called from within Modal containers to publish results
    that can be polled by the local sync worker.
    
    Args:
        run_id: Unique identifier for the run
        design_idx: Index of the design within the run
        cycle: Cycle number (0 for initial, 1+ for optimization cycles)
        iptm: Interface pTM score
        ipsae: Interface pSAE score
        plddt: Complex pLDDT score
        iplddt: Interface pLDDT score
        seq: Binder sequence
        pdb_file: Path to the PDB file
        binder_name: Name of the binder design
        target_seqs: Target protein sequences
        contact_residues: Contact residue specification
        msa_mode: MSA mode (empty or mmseqs)
        cyclic: Whether binder is cyclic
        alanine_count: Number of alanines in sequence
        ligand_smiles: SMILES string for ligand (if any)
        ligand_ccd: CCD code for ligand (if any)
        nucleic_seq: Nucleic acid sequence (if any)
        nucleic_type: Type of nucleic acid (dna/rna)
        template_path: Path to template structure
        template_mapping: Template chain mapping
    """
    try:
        key = f"{run_id}:d{design_idx}:c{cycle}"
        results_dict[key] = {
            "run_id": run_id,
            "design_idx": design_idx,
            "cycle": cycle,
            "iptm": iptm,
            "ipsae": ipsae,
            "plddt": plddt,
            "iplddt": iplddt,
            "seq": seq,
            "pdb": pdb_file.read_text() if pdb_file.exists() else None,
            "timestamp": time.time(),
            # Additional config for reproducibility
            "binder_name": binder_name,
            "target_seqs": target_seqs,
            "contact_residues": contact_residues,
            "msa_mode": msa_mode,
            "cyclic": cyclic,
            "alanine_count": alanine_count,
            # AF3 reconstruction fields
            "ligand_smiles": ligand_smiles,
            "ligand_ccd": ligand_ccd,
            "nucleic_seq": nucleic_seq,
            "nucleic_type": nucleic_type,
            "template_path": template_path,
            "template_mapping": template_mapping,
        }
    except Exception as e:
        print(f"  Stream error: {e}")


def _sync_worker(
    run_id: str,
    output_path: Path,
    stop_event: threading.Event,
    interval: float
):
    """
    Background worker that polls Modal Dict and saves results locally.
    
    This runs in a daemon thread on the local machine, periodically checking
    for new results in the Modal Dict and saving them to the local filesystem.
    
    Args:
        run_id: Unique identifier for the run to sync
        output_path: Local directory to save results
        stop_event: Threading event to signal worker to stop
        interval: Polling interval in seconds
    """
    synced_keys = set()
    
    while not stop_event.is_set():
        try:
            all_keys = [k for k in results_dict.keys() if k.startswith(f"{run_id}:")]
            new_keys = [k for k in all_keys if k not in synced_keys]
            
            for key in new_keys:
                try:
                    result = results_dict[key]
                    _save_synced_result(output_path, result, key)
                    synced_keys.add(key)
                except Exception:
                    pass  # Ignore individual sync errors
        except Exception:
            pass  # Ignore Dict access errors
        
        stop_event.wait(timeout=interval)
    
    # Final sync after stop signal
    try:
        all_keys = [k for k in results_dict.keys() if k.startswith(f"{run_id}:")]
        new_keys = [k for k in all_keys if k not in synced_keys]
        for key in new_keys:
            try:
                result = results_dict[key]
                _save_synced_result(output_path, result, key)
                synced_keys.add(key)
            except Exception:
                pass
    except Exception:
        pass


def _save_synced_result(output_path: Path, result: Dict, key: str):
    """
    Save a synced result from Dict to local filesystem with flat structure.
    
    Creates the designs/ directory structure and appends to design_stats.csv.
    
    Args:
        output_path: Base output directory
        result: Result dictionary from Modal Dict
        key: Dict key (used for logging only)
    """
    from utils.csv_utils import append_to_csv_safe
    import datetime as dt

    design_idx = result.get("design_idx", 0)
    cycle = result.get("cycle", 0)
    binder_name = result.get("binder_name", "design")

    # Create flat designs directory
    designs_dir = output_path / "designs"
    designs_dir.mkdir(parents=True, exist_ok=True)

    # Build design_id with descriptive naming: {name}_d{N}_c{M}
    design_id = f"{binder_name}_d{design_idx}_c{cycle}"

    # Save PDB with new naming convention
    if result.get("pdb"):
        pdb_file = designs_dir / f"{design_id}.pdb"
        pdb_file.write_text(result["pdb"])

    # Calculate alanine percentage
    seq = result.get("seq", "")
    binder_length = len(seq) if seq else 0
    alanine_count = result.get("alanine_count", seq.count("A") if seq else 0)
    alanine_pct = (alanine_count / binder_length * 100) if binder_length > 0 else 0.0

    # Append to design_stats.csv with all columns for reproducibility
    csv_file = designs_dir / "design_stats.csv"
    row = {
        # Core identification
        "design_id": design_id,
        "design_num": design_idx,
        "cycle": cycle,
        # Designed binder
        "binder_sequence": seq,
        "binder_length": binder_length,
        "cyclic": result.get("cyclic", False),
        # Prediction metrics
        "iptm": result.get("iptm", 0.0),
        "ipsae": result.get("ipsae", 0.0),
        "plddt": result.get("plddt", 0.0),
        "iplddt": result.get("iplddt", 0.0),
        "alanine_count": alanine_count,
        "alanine_pct": round(alanine_pct, 2),
        # Job configuration (enables reproducibility without YAML files)
        "target_seqs": result.get("target_seqs", ""),
        "contact_residues": result.get("contact_residues", ""),
        "msa_mode": result.get("msa_mode", "empty"),
        # AF3 reconstruction fields
        "ligand_smiles": result.get("ligand_smiles", ""),
        "ligand_ccd": result.get("ligand_ccd", ""),
        "nucleic_seq": result.get("nucleic_seq", ""),
        "nucleic_type": result.get("nucleic_type", ""),
        "template_path": result.get("template_path", ""),
        "template_mapping": result.get("template_mapping", ""),
        # Run metadata
        "timestamp": dt.datetime.fromtimestamp(result.get("timestamp", time.time())).isoformat(),
    }
    append_to_csv_safe(csv_file, row)

