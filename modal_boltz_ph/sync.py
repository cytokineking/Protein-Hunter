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
    contact_residues_auth: str = "",       # Hotspots in auth/PDB numbering (for traceability)
    template_first_residue: str = "",      # Pre-formatted "A:69,B:1" string
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
        contact_residues: Contact residue specification (canonical numbering)
        contact_residues_auth: Contact residues in auth/PDB numbering (for traceability)
        template_first_residue: First auth residue per chain ("A:69,B:1")
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
            "result_type": "cycle",
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
            "contact_residues_auth": contact_residues_auth,      # Auth/PDB numbering
            "template_first_residue": template_first_residue,    # "A:69,B:1" format
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


def _stream_best_design(
    run_id: str,
    design_idx: int,
    design_id: str,
    best_cycle: int,
    best_seq: str,
    best_pdb: str,
    metrics: Dict,
):
    """
    Stream best design result after Boltz completes.
    
    This function is called after the design phase to publish the best
    design from all cycles for real-time syncing.
    
    Args:
        run_id: Unique identifier for the run
        design_idx: Index of the design within the run
        design_id: Full design identifier (e.g., "PDL1_d0_c3")
        best_cycle: Cycle number of the best design
        best_seq: Best binder sequence
        best_pdb: Best PDB content as string
        metrics: Dict with iptm, ipsae, plddt, iplddt, alanine_count, cyclic
    """
    try:
        key = f"{run_id}:d{design_idx}:best"
        results_dict[key] = {
            "result_type": "best",
            "design_id": design_id,
            "design_idx": design_idx,
            "best_cycle": best_cycle,
            "best_seq": best_seq,
            "best_pdb": best_pdb,
            **metrics,
            "timestamp": time.time(),
        }
    except Exception as e:
        print(f"  Stream best error: {e}")


def _stream_af3_result(
    run_id: str,
    design_idx: int,
    design_id: str,
    af3_iptm: float,
    af3_ipsae: float,
    af3_ptm: float,
    af3_plddt: float,
    af3_structure: str,
):
    """
    Stream AF3 validation result.
    
    This function is called after AF3 validation completes for real-time syncing.
    
    Args:
        run_id: Unique identifier for the run
        design_idx: Index of the design within the run
        design_id: Full design identifier
        af3_iptm: AF3 interface pTM score
        af3_ipsae: AF3 interface pSAE score
        af3_ptm: AF3 pTM score
        af3_plddt: AF3 pLDDT score
        af3_structure: AF3 structure as CIF text
    """
    try:
        key = f"{run_id}:d{design_idx}:af3"
        results_dict[key] = {
            "result_type": "af3",
            "design_id": design_id,
            "design_idx": design_idx,
            "af3_iptm": af3_iptm,
            "af3_ipsae": af3_ipsae,
            "af3_ptm": af3_ptm,
            "af3_plddt": af3_plddt,
            "af3_structure": af3_structure,
            "timestamp": time.time(),
        }
    except Exception as e:
        print(f"  Stream AF3 error: {e}")


def _stream_final_result(
    run_id: str,
    design_idx: int,
    design_id: str,
    accepted: bool,
    rejection_reason: str,
    metrics: Dict,
    relaxed_pdb: str,
):
    """
    Stream final PyRosetta result.
    
    This function is called after PyRosetta scoring completes for real-time syncing.
    
    Args:
        run_id: Unique identifier for the run
        design_idx: Index of the design within the run
        design_id: Full design identifier
        accepted: Whether the design passed filters
        rejection_reason: Reason for rejection (if rejected)
        metrics: Dict with interface_dG, interface_sc, interface_nres, etc.
        relaxed_pdb: Relaxed PDB content as string
    """
    try:
        key = f"{run_id}:d{design_idx}:final"
        results_dict[key] = {
            "result_type": "final",
            "design_id": design_id,
            "design_idx": design_idx,
            "accepted": accepted,
            "rejection_reason": rejection_reason,
            "relaxed_pdb": relaxed_pdb,
            **metrics,
            "timestamp": time.time(),
        }
    except Exception as e:
        print(f"  Stream final error: {e}")


def _extract_scoring_metrics(validation_result: Dict) -> Dict:
    """
    Extract scoring metrics from a validation result dict.
    
    This helper consolidates the metric extraction logic used when streaming
    bundled validation+scoring results from Protenix or OpenFold3.
    
    Args:
        validation_result: Dict containing validation and scoring results
    
    Returns:
        Dict with extracted scoring metrics
    """
    return {
        "interface_dG": validation_result.get("interface_dG", 0.0),
        "interface_sc": validation_result.get("interface_sc", 0.0),
        "interface_nres": validation_result.get("interface_nres", 0),
        "interface_dSASA": validation_result.get("interface_dSASA", 0.0),
        "interface_packstat": validation_result.get("interface_packstat", 0.0),
        "interface_dG_SASA_ratio": validation_result.get("interface_dG_SASA_ratio", 0.0),
        "interface_interface_hbonds": validation_result.get("interface_interface_hbonds", 0),
        "interface_delta_unsat_hbonds": validation_result.get("interface_delta_unsat_hbonds", 0),
        "interface_hydrophobicity": validation_result.get("interface_hydrophobicity", 0.0),
        "surface_hydrophobicity": validation_result.get("surface_hydrophobicity", 0.0),
        "binder_sasa": validation_result.get("binder_sasa", 0.0),
        "interface_fraction": validation_result.get("interface_fraction", 0.0),
        "interface_hbond_percentage": validation_result.get("interface_hbond_percentage", 0.0),
        "interface_delta_unsat_hbonds_percentage": validation_result.get("interface_delta_unsat_hbonds_percentage", 0.0),
        "apo_holo_rmsd": validation_result.get("apo_holo_rmsd"),
        "i_pae": validation_result.get("i_pae"),
        "rg": validation_result.get("rg"),
    }


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
    Save a synced result from Dict to local filesystem.
    
    Routes to appropriate handler based on result_type.
    
    Args:
        output_path: Base output directory
        result: Result dictionary from Modal Dict
        key: Dict key (used for logging only)
    """
    result_type = result.get("result_type", "cycle")
    
    if result_type == "cycle":
        _save_cycle_result(output_path, result)
    elif result_type == "best":
        _save_best_result(output_path, result)
    elif result_type == "af3":
        _save_af3_result(output_path, result)
    elif result_type == "final":
        _save_final_result(output_path, result)


def _save_cycle_result(output_path: Path, result: Dict):
    """
    Save a cycle result to designs/ directory.
    
    Args:
        output_path: Base output directory
        result: Result dictionary from Modal Dict
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
        "contact_residues_auth": result.get("contact_residues_auth", ""),      # Auth/PDB numbering
        "template_first_residue": result.get("template_first_residue", ""),    # "A:69,B:1" format
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


def _save_best_result(output_path: Path, result: Dict):
    """
    Save best design result to best_designs/ directory.
    
    Creates initial row in best_designs.csv with Boltz metrics.
    AF3 and PyRosetta stages will update this row via update_csv_row.
    
    Args:
        output_path: Base output directory
        result: Result dictionary from Modal Dict
    """
    from utils.csv_utils import append_to_csv_safe
    import datetime as dt

    design_id = result.get("design_id", "unknown")

    # Create best_designs directory
    best_dir = output_path / "best_designs"
    best_dir.mkdir(parents=True, exist_ok=True)

    # Save PDB
    if result.get("best_pdb"):
        pdb_file = best_dir / f"{design_id}.pdb"
        pdb_file.write_text(result["best_pdb"])

    # Calculate alanine percentage
    seq = result.get("best_seq", "")
    binder_length = len(seq) if seq else 0
    alanine_count = result.get("alanine_count", seq.count("A") if seq else 0)
    alanine_pct = (alanine_count / binder_length * 100) if binder_length > 0 else 0.0

    # Create initial row with Boltz metrics (AF3 and PyRosetta will update later)
    csv_file = best_dir / "best_designs.csv"
    row = {
        # Identity
        "design_id": design_id,
        "design_num": result.get("design_idx", 0),
        "cycle": result.get("best_cycle", 0),
        # Binder info
        "binder_sequence": seq,
        "binder_length": binder_length,
        "cyclic": result.get("cyclic", False),
        "alanine_count": alanine_count,
        "alanine_pct": round(alanine_pct, 2),
        # Boltz design metrics (prefixed)
        "boltz_iptm": result.get("boltz_iptm", result.get("iptm", 0.0)),
        "boltz_ipsae": result.get("boltz_ipsae", result.get("ipsae", 0.0)),
        "boltz_plddt": result.get("boltz_plddt", result.get("plddt", 0.0)),
        "boltz_iplddt": result.get("boltz_iplddt", result.get("iplddt", 0.0)),
        # Placeholder columns for AF3 (will be updated by _save_af3_result)
        "af3_iptm": None,
        "af3_ipsae": None,
        "af3_ptm": None,
        "af3_plddt": None,
        # Placeholder columns for PyRosetta (will be updated by _save_final_result)
        "interface_dG": None,
        "interface_sc": None,
        "interface_nres": None,
        "interface_dSASA": None,
        "interface_packstat": None,
        "interface_hbonds": None,
        "interface_delta_unsat_hbonds": None,
        "apo_holo_rmsd": None,
        "i_pae": None,
        "rg": None,
        "accepted": None,
        "rejection_reason": None,
        # Hotspot/template traceability
        "contact_residues": result.get("contact_residues", ""),
        "contact_residues_auth": result.get("contact_residues_auth", ""),
        "template_first_residue": result.get("template_first_residue", ""),
        "timestamp": dt.datetime.fromtimestamp(result.get("timestamp", time.time())).isoformat(),
    }
    append_to_csv_safe(csv_file, row)


def _save_af3_result(output_path: Path, result: Dict):
    """
    Save validation result to refolded/ directory.
    
    Also updates the existing row in best_designs.csv with validation metrics.
    
    Args:
        output_path: Base output directory
        result: Result dictionary from Modal Dict
    """
    from utils.csv_utils import append_to_csv_safe, update_csv_row
    import datetime as dt

    design_id = result.get("design_id", "unknown")

    # Create refolded directory
    refolded_dir = output_path / "refolded"
    refolded_dir.mkdir(parents=True, exist_ok=True)

    # Save CIF structure
    if result.get("af3_structure"):
        cif_file = refolded_dir / f"{design_id}_refolded.cif"
        cif_file.write_text(result["af3_structure"])

    # Append to validation_results.csv (includes af3_ipsae)
    csv_file = refolded_dir / "validation_results.csv"
    row = {
        "design_id": design_id,
        "af3_iptm": result.get("af3_iptm", 0.0),
        "af3_ipsae": result.get("af3_ipsae", 0.0),
        "af3_ptm": result.get("af3_ptm", 0.0),
        "af3_plddt": result.get("af3_plddt", 0.0),
        "timestamp": dt.datetime.fromtimestamp(result.get("timestamp", time.time())).isoformat(),
    }
    append_to_csv_safe(csv_file, row)
    
    # Update existing row in best_designs.csv with AF3 metrics
    best_csv = output_path / "best_designs" / "best_designs.csv"
    if best_csv.exists():
        af3_update = {
            "af3_iptm": result.get("af3_iptm", 0.0),
            "af3_ipsae": result.get("af3_ipsae", 0.0),
            "af3_ptm": result.get("af3_ptm", 0.0),
            "af3_plddt": result.get("af3_plddt", 0.0),
        }
        update_csv_row(best_csv, key_col="design_id", key_val=design_id, update_data=af3_update)


def _save_final_result(output_path: Path, result: Dict):
    """
    Save final PyRosetta result to accepted_designs/ or rejected/ directory.
    
    Also updates the existing row in best_designs.csv with PyRosetta metrics
    and acceptance status.
    
    Args:
        output_path: Base output directory
        result: Result dictionary from Modal Dict
    """
    from utils.csv_utils import append_to_csv_safe, update_csv_row
    import datetime as dt

    design_id = result.get("design_id", "unknown")
    accepted = result.get("accepted", False)

    # Determine target directory
    if accepted:
        target_dir = output_path / "accepted_designs"
    else:
        target_dir = output_path / "rejected"
    target_dir.mkdir(parents=True, exist_ok=True)

    # Save relaxed PDB
    if result.get("relaxed_pdb"):
        pdb_file = target_dir / f"{design_id}_relaxed.pdb"
        pdb_file.write_text(result["relaxed_pdb"])

    # Build stats row with all interface metrics
    csv_file = target_dir / ("accepted_stats.csv" if accepted else "rejected_stats.csv")
    row = {
        "design_id": design_id,
        "accepted": accepted,
        "rejection_reason": result.get("rejection_reason", ""),
        # Interface metrics from PyRosetta
        "interface_dG": result.get("interface_dG", 0.0),
        "interface_sc": result.get("interface_sc", 0.0),
        "interface_nres": result.get("interface_nres", 0),
        "interface_dSASA": result.get("interface_dSASA", 0.0),
        "interface_packstat": result.get("interface_packstat", 0.0),
        "interface_dG_SASA_ratio": result.get("interface_dG_SASA_ratio", 0.0),
        "interface_interface_hbonds": result.get("interface_interface_hbonds", 0),
        "interface_delta_unsat_hbonds": result.get("interface_delta_unsat_hbonds", 0),
        "interface_hydrophobicity": result.get("interface_hydrophobicity", 0.0),
        "surface_hydrophobicity": result.get("surface_hydrophobicity", 0.0),
        "binder_sasa": result.get("binder_sasa", 0.0),
        "interface_fraction": result.get("interface_fraction", 0.0),
        "interface_hbond_percentage": result.get("interface_hbond_percentage", 0.0),
        "interface_delta_unsat_hbonds_percentage": result.get("interface_delta_unsat_hbonds_percentage", 0.0),
        # Secondary metrics
        "apo_holo_rmsd": result.get("apo_holo_rmsd"),
        "i_pae": result.get("i_pae"),
        "rg": result.get("rg"),
        "timestamp": dt.datetime.fromtimestamp(result.get("timestamp", time.time())).isoformat(),
    }
    append_to_csv_safe(csv_file, row)
    
    # Update existing row in best_designs.csv with PyRosetta metrics and acceptance status
    best_csv = output_path / "best_designs" / "best_designs.csv"
    if best_csv.exists():
        pyrosetta_update = {
            "interface_dG": result.get("interface_dG", 0.0),
            "interface_sc": result.get("interface_sc", 0.0),
            "interface_nres": result.get("interface_nres", 0),
            "interface_dSASA": result.get("interface_dSASA", 0.0),
            "interface_packstat": result.get("interface_packstat", 0.0),
            "interface_hbonds": result.get("interface_interface_hbonds", 0),
            "interface_delta_unsat_hbonds": result.get("interface_delta_unsat_hbonds", 0),
            "apo_holo_rmsd": result.get("apo_holo_rmsd"),
            "i_pae": result.get("i_pae"),
            "rg": result.get("rg"),
            "accepted": accepted,
            "rejection_reason": result.get("rejection_reason", ""),
        }
        update_csv_row(best_csv, key_col="design_id", key_val=design_id, update_data=pyrosetta_update)

