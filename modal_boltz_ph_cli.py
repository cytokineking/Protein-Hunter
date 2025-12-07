#!/usr/bin/env python3
"""
Modal Boltz Protein Hunter CLI

This is the main entry point for running the Protein Hunter design pipeline on Modal.

Usage:
    # Initialize cache (run once)
    modal run modal_boltz_ph_cli.py::init_cache
    
    # Upload AF3 weights (if using AF3 validation)
    modal run modal_boltz_ph_cli.py::upload_af3_weights --weights-path ~/AF3/af3.bin.zst
    
    # Run design pipeline
    modal run modal_boltz_ph_cli.py::run_pipeline \\
        --name "PDL1_binder" \\
        --protein-seqs "AFTVTVPK..." \\
        --num-designs 5
    
    # List available GPUs
    modal run modal_boltz_ph_cli.py::list_gpus
    
    # Test connection
    modal run modal_boltz_ph_cli.py::test_connection
"""

import datetime
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional


def str2bool(v):
    """Convert string to boolean, matching local pipeline behavior."""
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', '0'):
        return False
    else:
        raise ValueError(f'Boolean value expected, got: {v}')


# Import Modal app and shared resources
from modal_boltz_ph.app import app, GPU_TYPES, DEFAULT_GPU

# Import functions from modules
from modal_boltz_ph.design import GPU_FUNCTIONS
from modal_boltz_ph.validation_af3 import (
    AF3_GPU_FUNCTIONS,
    AF3_APO_GPU_FUNCTIONS,
    run_af3_single_A100_80GB,
    run_af3_apo_A100_80GB,
)
from modal_boltz_ph.scoring_pyrosetta import run_pyrosetta_single
from modal_boltz_ph.cache import initialize_cache, _upload_af3_weights_impl
from modal_boltz_ph.sync import (
    _sync_worker,
    _stream_best_design,
    _stream_af3_result,
    _stream_final_result,
)
from modal_boltz_ph.tests import test_af3_image, _test_gpu


# =============================================================================
# CSV COLUMN DEFINITIONS
# =============================================================================

# Fields to exclude from CSV outputs (large data or internal fields)
CSV_EXCLUDE = {
    "relaxed_pdb", "_target_msas", "af3_confidence_json", "af3_structure",
    "apo_structure", "best_pdb", "cycles", "target_msas"
}

# Unified column schema for best_designs, accepted_stats, and rejected_stats CSVs
# These three CSVs use identical columns for consistency
UNIFIED_DESIGN_COLUMNS = [
    # Identity
    "design_id", "design_num", "cycle",
    # Binder info
    "binder_sequence", "binder_length", "cyclic", "alanine_count", "alanine_pct",
    # Boltz design metrics (prefixed for clarity)
    "boltz_iptm", "boltz_ipsae", "boltz_plddt", "boltz_iplddt",
    # AF3 validation metrics
    "af3_iptm", "af3_ipsae", "af3_ptm", "af3_plddt",
    # PyRosetta interface metrics
    "interface_dG", "interface_sc", "interface_nres", "interface_dSASA",
    "interface_packstat", "interface_hbonds", "interface_delta_unsat_hbonds",
    # Secondary quality metrics
    "apo_holo_rmsd", "i_pae", "rg",
    # Acceptance status
    "accepted", "rejection_reason",
]


# =============================================================================
# LOCAL ENTRYPOINTS
# =============================================================================

@app.local_entrypoint()
def upload_af3_weights(weights_path: str):
    """
    Upload AlphaFold3 weights to Modal volume.
    
    Usage:
        modal run modal_boltz_ph_cli.py::upload_af3_weights --weights-path ~/AF3/af3.bin.zst
    """
    import subprocess as sp
    
    weights_file = Path(weights_path).expanduser()
    
    if not weights_file.exists():
        print(f"Error: Weights file not found: {weights_file}")
        return
    
    file_size = weights_file.stat().st_size
    print(f"AF3 weights: {weights_file}")
    print(f"File size: {file_size / 1e9:.2f} GB")
    
    # Decompress locally if needed
    upload_file = weights_file
    upload_name = weights_file.name
    temp_file = None
    
    if weights_file.suffix == ".zst":
        print("Decompressing locally (this may take a minute)...")
        temp_file = Path(tempfile.mktemp(suffix=".bin"))
        result = sp.run(["zstd", "-d", str(weights_file), "-o", str(temp_file)], capture_output=True, text=True)
        if result.returncode != 0:
            print(f"Error decompressing: {result.stderr}")
            return
        upload_file = temp_file
        upload_name = "af3.bin"
        print(f"Decompressed to {temp_file} ({temp_file.stat().st_size / 1e9:.2f} GB)")
    
    print("Uploading to Modal volume... (this may take a few minutes)")
    
    # Read file and upload
    weights_bytes = upload_file.read_bytes()
    result = _upload_af3_weights_impl.remote(weights_bytes, upload_name)
    print(result)
    
    # Cleanup temp file
    if temp_file and temp_file.exists():
        temp_file.unlink()


@app.local_entrypoint()
def init_cache():
    """
    Initialize the cache (download model weights).
    Run this ONCE before using the pipeline.
    
    Usage:
        modal run modal_boltz_ph_cli.py::init_cache
    """
    print("Initializing Protein Hunter cache...")
    result = initialize_cache.remote()
    print(result)


@app.local_entrypoint()
def run_pipeline(
    # Job identity
    name: str = "protein_hunter_run",
    # Target specification
    protein_seqs: Optional[str] = None,
    ligand_ccd: Optional[str] = None,
    ligand_smiles: Optional[str] = None,
    nucleic_seq: Optional[str] = None,
    nucleic_type: str = "dna",
    # Template
    template_path: Optional[str] = None,
    template_cif_chain_id: Optional[str] = None,
    # Binder configuration
    seq: Optional[str] = None,
    min_protein_length: int = 100,
    max_protein_length: int = 150,
    percent_x: int = 90,
    cyclic: str = "false",
    exclude_p: str = "false",
    # Design parameters
    num_designs: int = 50,
    num_cycles: int = 5,
    contact_residues: Optional[str] = None,
    temperature: float = 0.1,
    omit_aa: str = "C",
    alanine_bias: str = "true",
    alanine_bias_start: float = -0.5,
    alanine_bias_end: float = -0.1,
    high_iptm_threshold: float = 0.8,
    high_plddt_threshold: float = 0.8,
    # Contact filtering
    no_contact_filter: str = "false",
    max_contact_filter_retries: int = 6,
    contact_cutoff: float = 15.0,
    # MSA options
    msa_mode: str = "mmseqs",  # "single" or "mmseqs"
    # Model parameters
    diffuse_steps: int = 200,
    recycling_steps: int = 3,
    randomly_kill_helix_feature: str = "false",
    negative_helix_constant: float = 0.2,
    grad_enabled: str = "false",
    logmd: str = "false",
    # Execution
    gpu: str = DEFAULT_GPU,
    max_concurrent: int = 0,  # 0 = unlimited
    output_dir: Optional[str] = None,
    no_stream: str = "false",
    sync_interval: float = 5.0,
    # AF3 Validation (optional)
    use_alphafold3_validation: str = "false",
    use_msa_for_af3: str = "true",
    af3_gpu: str = "A100-80GB",
):
    """
    Run the Protein Hunter design pipeline on Modal.
    
    Examples:
        # Basic protein binder design (alanine_bias is ON by default)
        modal run modal_boltz_ph_cli.py::run_pipeline \\
            --name "PDL1_binder" \\
            --protein-seqs "AFTVTVPK..." \\
            --num-designs 5 \\
            --num-cycles 7
        
        # With AF3 validation (PyRosetta runs automatically for protein targets)
        modal run modal_boltz_ph_cli.py::run_pipeline \\
            --name "PDL1_validated" \\
            --protein-seqs "AFTVTVPK..." \\
            --num-designs 3 \\
            --use-alphafold3-validation
        
        # With hotspots
        modal run modal_boltz_ph_cli.py::run_pipeline \\
            --name "PDL1_hotspot" \\
            --protein-seqs "AFTVTVPK..." \\
            --contact-residues "54,56,115" \\
            --num-designs 3
        
        # Small molecule binder
        modal run modal_boltz_ph_cli.py::run_pipeline \\
            --name "SAM_binder" \\
            --ligand-ccd "SAM" \\
            --num-designs 5
        
        # Disable alanine bias (rare)
        modal run modal_boltz_ph_cli.py::run_pipeline \\
            --name "test" \\
            --protein-seqs "AFTVTVPK..." \\
            --alanine-bias=false
    
    AF3 Validation:
        First upload weights: modal run modal_boltz_ph_cli.py::upload_af3_weights --weights-path ~/AF3/af3.bin.zst
        Then use --use-alphafold3-validation flag
        
        MSA reuse (default: True):
            --use-msa-for-af3=true   Reuse MSAs from design phase (recommended)
            --use-msa-for-af3=false  Query-only for all chains (faster, less accurate)
        
        PyRosetta filtering runs automatically for protein targets when AF3 validation is enabled.
    """
    import base64
    import pandas as pd
    
    # Convert string boolean parameters to actual booleans
    cyclic = str2bool(cyclic)
    exclude_p = str2bool(exclude_p)
    alanine_bias = str2bool(alanine_bias)
    no_contact_filter = str2bool(no_contact_filter)
    randomly_kill_helix_feature = str2bool(randomly_kill_helix_feature)
    grad_enabled = str2bool(grad_enabled)
    logmd = str2bool(logmd)
    no_stream = str2bool(no_stream)
    use_alphafold3_validation = str2bool(use_alphafold3_validation)
    use_msa_for_af3 = str2bool(use_msa_for_af3)
    
    stream = not no_stream
    run_id = f"{name}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # Validate inputs
    if not any([protein_seqs, ligand_ccd, ligand_smiles, nucleic_seq]):
        print("Error: Must provide at least one target (--protein-seqs, --ligand-ccd, --ligand-smiles, or --nucleic-seq)")
        return
    
    # Auto-derive target_type from inputs (replaces manual flag)
    if nucleic_seq:
        target_type = "nucleic"
    elif ligand_ccd or ligand_smiles:
        target_type = "small_molecule"
    else:
        target_type = "protein"
    
    # Read and encode template file if provided
    template_content = ""
    if template_path:
        template_file = Path(template_path)
        if template_file.exists():
            template_content = base64.b64encode(template_file.read_bytes()).decode('utf-8')
            print(f"Loaded template: {template_path} ({len(template_content)} bytes encoded)")
        else:
            print(f"Warning: Template file not found: {template_path}")
    
    # Setup output directory
    output_path = Path(output_dir) if output_dir else Path(f"./results_{name}")
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Print configuration
    print(f"\n{'='*70}")
    print("PROTEIN HUNTER (Modal)")
    print(f"{'='*70}")
    print(f"Run ID: {run_id}")
    print(f"Target: {protein_seqs[:50] + '...' if protein_seqs and len(protein_seqs) > 50 else protein_seqs or ligand_ccd or nucleic_seq}")
    print(f"Target type: {target_type}")
    print(f"Num designs: {num_designs}")
    print(f"Num cycles: {num_cycles}")
    print(f"GPU: {gpu}")
    print(f"Max concurrent: {max_concurrent if max_concurrent > 0 else 'unlimited'}")
    print(f"Output: {output_path}")
    if template_path:
        print(f"Template: {template_path}")
        print(f"Template chains: {template_cif_chain_id}")
    if contact_residues:
        print(f"Hotspots: {contact_residues}")
    print(f"MSA mode: {msa_mode}")
    print(f"Alanine bias: {alanine_bias}")
    print(f"{'='*70}\n")
    
    # Build tasks
    tasks = []
    for i in range(num_designs):
        task = {
            "run_id": run_id,
            "design_idx": i,
            "total_designs": num_designs,
            "stream_to_dict": stream,
            # Target
            "protein_seqs": protein_seqs or "",
            "ligand_ccd": ligand_ccd or "",
            "ligand_smiles": ligand_smiles or "",
            "nucleic_seq": nucleic_seq or "",
            "nucleic_type": nucleic_type,
            "template_content": template_content,
            "template_chain_ids": template_cif_chain_id or "",
            "msa_mode": msa_mode,
            # Binder
            "seq": seq or "",
            "min_protein_length": min_protein_length,
            "max_protein_length": max_protein_length,
            "percent_X": percent_x,
            "cyclic": cyclic,
            "exclude_P": exclude_p,
            # Design
            "num_cycles": num_cycles,
            "contact_residues": contact_residues or "",
            "temperature": temperature,
            "omit_AA": omit_aa,
            "alanine_bias": alanine_bias,
            "alanine_bias_start": alanine_bias_start,
            "alanine_bias_end": alanine_bias_end,
            "high_iptm_threshold": high_iptm_threshold,
            "high_plddt_threshold": high_plddt_threshold,
            # Contact filtering
            "no_contact_filter": no_contact_filter,
            "max_contact_filter_retries": max_contact_filter_retries,
            "contact_cutoff": contact_cutoff,
            # Model
            "diffuse_steps": diffuse_steps,
            "recycling_steps": recycling_steps,
            "randomly_kill_helix_feature": randomly_kill_helix_feature,
            "negative_helix_constant": negative_helix_constant,
            "grad_enabled": grad_enabled,
            "logmd": logmd,
        }
        tasks.append(task)
    
    # Select GPU function
    if gpu not in GPU_FUNCTIONS:
        print(f"Error: Unknown GPU '{gpu}'. Available: {', '.join(GPU_FUNCTIONS.keys())}")
        return
    
    gpu_fn = GPU_FUNCTIONS[gpu]
    
    # Start background sync thread
    sync_thread = None
    stop_sync = None
    
    if stream:
        stop_sync = threading.Event()
        sync_thread = threading.Thread(
            target=_sync_worker,
            args=(run_id, output_path, stop_sync, sync_interval),
            daemon=True,
        )
        sync_thread.start()
        print(f"Background sync started (polling every {sync_interval}s)\n")
    
    # ==========================================================================
    # END-TO-END PER-DESIGN ORCHESTRATION
    # Each design completes: Boltz → AF3 → APO → PyRosetta before next slot freed
    # ==========================================================================
    
    # Select AF3 functions (if needed)
    af3_gpu_to_use = gpu if gpu in AF3_GPU_FUNCTIONS else "A100-80GB"
    af3_fn = AF3_GPU_FUNCTIONS.get(af3_gpu_to_use, run_af3_single_A100_80GB)
    af3_apo_fn = AF3_APO_GPU_FUNCTIONS.get(af3_gpu_to_use, run_af3_apo_A100_80GB)
    
    def _run_full_pipeline_for_design(task_input: dict) -> dict:
        """
        Run complete pipeline for one design: Boltz → AF3 → APO → PyRosetta.
        
        All stages run sequentially for this design before the thread is freed.
        """
        design_idx = task_input.get("design_idx", 0)
        
        # ========== STAGE 1: BOLTZ DESIGN ==========
        try:
            design_result = gpu_fn.remote(task_input)
        except Exception as e:
            return {
                "status": "error",
                "error": f"Design failed: {e}",
                "design_idx": design_idx,
                "stage_failed": "design",
                "best_iptm": 0,
            }
        
        if design_result.get("status") != "success":
            return {**design_result, "stage_failed": "design"}
        
        # Stream best design result
        best_cycle = design_result.get("best_cycle", 0)
        design_id = f"{name}_d{design_idx}_c{best_cycle}"
        best_seq = design_result.get("best_seq", "")
        best_pdb = design_result.get("best_pdb", "")
        
        # Get metrics from best cycle
        best_cycle_data = None
        for cycle_data in design_result.get("cycles", []):
            if cycle_data.get("cycle") == best_cycle:
                best_cycle_data = cycle_data
                break
        
        if stream and best_seq:
            _stream_best_design(
                run_id=run_id,
                design_idx=design_idx,
                design_id=design_id,
                best_cycle=best_cycle,
                best_seq=best_seq,
                best_pdb=best_pdb,
                metrics={
                    "iptm": design_result.get("best_iptm", 0.0),
                    "ipsae": best_cycle_data.get("ipsae", 0.0) if best_cycle_data else 0.0,
                    "plddt": best_cycle_data.get("plddt", 0.0) if best_cycle_data else 0.0,
                    "iplddt": best_cycle_data.get("iplddt", 0.0) if best_cycle_data else 0.0,
                    "alanine_count": best_cycle_data.get("alanine_count", 0) if best_cycle_data else 0,
                    "cyclic": task_input.get("cyclic", False),
                },
            )
        
        # If no AF3 validation requested, return design result only
        if not use_alphafold3_validation:
            return design_result
        
        # ========== STAGE 2: AF3 VALIDATION ==========
        binder_seq = best_seq
        target_seq = task_input.get("protein_seqs", "")
        target_msas = design_result.get("target_msas", {})
        target_msa = target_msas.get("B") if use_msa_for_af3 else None
        
        print(f"  [{design_id}] Starting AF3 validation...")
        
        try:
            af3_result = af3_fn.remote(
                design_id, binder_seq, target_seq,
                "A", "B",  # binder_chain, target_chain
                target_msa,
                "reuse" if use_msa_for_af3 else "none",
                None, None  # template_path, template_chain_id
            )
        except Exception as e:
            print(f"  [{design_id}] AF3 failed: {e}")
            return {**design_result, "design_id": design_id, "af3_error": str(e), "stage_failed": "af3"}
        
        if not af3_result.get("af3_structure"):
            return {**design_result, "design_id": design_id, **af3_result, "stage_failed": "af3"}
        
        print(f"  [{design_id}] AF3 complete: ipTM={af3_result.get('af3_iptm', 0):.3f}, ipSAE={af3_result.get('af3_ipsae', 0):.3f}")
        
        # Stream AF3 result
        if stream:
            _stream_af3_result(
                run_id=run_id,
                design_idx=design_idx,
                design_id=design_id,
                af3_iptm=af3_result.get("af3_iptm", 0.0),
                af3_ipsae=af3_result.get("af3_ipsae", 0.0),
                af3_ptm=af3_result.get("af3_ptm", 0.0),
                af3_plddt=af3_result.get("af3_plddt", 0.0),
                af3_structure=af3_result.get("af3_structure", ""),
            )
        
        # ========== STAGE 3: APO PREDICTION (protein targets only) ==========
        apo_structure = None
        if target_type == "protein":
            try:
                apo_result = af3_apo_fn.remote(design_id, binder_seq, "A")
                apo_structure = apo_result.get("apo_structure")
                if apo_structure:
                    print(f"  [{design_id}] APO structure generated")
            except Exception as e:
                # APO failure is non-fatal, continue without RMSD
                print(f"  [{design_id}] APO prediction failed (non-fatal): {e}")
        
        # ========== STAGE 4: PYROSETTA SCORING (protein targets only) ==========
        pr_result = {"accepted": True}  # Default for non-protein targets
        if target_type == "protein":
            print(f"  [{design_id}] Running PyRosetta scoring...")
            try:
                pr_result = run_pyrosetta_single.remote(
                    design_id,
                    af3_result.get("af3_structure"),
                    af3_result.get("af3_iptm", 0),
                    af3_result.get("af3_ptm", 0),
                    af3_result.get("af3_plddt", 0),
                    "A", "B",  # binder_chain, target_chain
                    apo_structure,
                    af3_result.get("af3_confidence_json"),
                    target_type,
                )
                status_str = "ACCEPTED" if pr_result.get("accepted") else f"REJECTED ({pr_result.get('rejection_reason', 'unknown')})"
                print(f"  [{design_id}] PyRosetta: dG={pr_result.get('interface_dG', 0):.1f}, SC={pr_result.get('interface_sc', 0):.2f} → {status_str}")
            except Exception as e:
                print(f"  [{design_id}] PyRosetta failed: {e}")
                pr_result = {"error": str(e), "accepted": False, "rejection_reason": f"PyRosetta error: {e}"}
        
        # Stream final result (after PyRosetta scoring)
        if stream and target_type == "protein":
            _stream_final_result(
                run_id=run_id,
                design_idx=design_idx,
                design_id=design_id,
                accepted=pr_result.get("accepted", False),
                rejection_reason=pr_result.get("rejection_reason", ""),
                metrics={
                    "interface_dG": pr_result.get("interface_dG", 0.0),
                    "interface_sc": pr_result.get("interface_sc", 0.0),
                    "interface_nres": pr_result.get("interface_nres", 0),
                    "interface_dSASA": pr_result.get("interface_dSASA", 0.0),
                    "interface_packstat": pr_result.get("interface_packstat", 0.0),
                    "interface_dG_SASA_ratio": pr_result.get("interface_dG_SASA_ratio", 0.0),
                    "interface_interface_hbonds": pr_result.get("interface_interface_hbonds", 0),
                    "interface_delta_unsat_hbonds": pr_result.get("interface_delta_unsat_hbonds", 0),
                    "interface_hydrophobicity": pr_result.get("interface_hydrophobicity", 0.0),
                    "surface_hydrophobicity": pr_result.get("surface_hydrophobicity", 0.0),
                    "binder_sasa": pr_result.get("binder_sasa", 0.0),
                    "interface_fraction": pr_result.get("interface_fraction", 0.0),
                    "interface_hbond_percentage": pr_result.get("interface_hbond_percentage", 0.0),
                    "interface_delta_unsat_hbonds_percentage": pr_result.get("interface_delta_unsat_hbonds_percentage", 0.0),
                    "apo_holo_rmsd": pr_result.get("apo_holo_rmsd"),
                    "i_pae": pr_result.get("i_pae"),
                    "rg": pr_result.get("rg"),
                },
                relaxed_pdb=pr_result.get("relaxed_pdb", ""),
            )
        
        # ========== COMBINE ALL RESULTS ==========
        
        return {
            # Design results
            **design_result,
            # Identity
            "design_id": design_id,
            "design_num": design_idx,
            "cycle": best_cycle,
            # AF3 results
            "af3_iptm": af3_result.get("af3_iptm", 0),
            "af3_ipsae": af3_result.get("af3_ipsae", 0),
            "af3_ptm": af3_result.get("af3_ptm", 0),
            "af3_plddt": af3_result.get("af3_plddt", 0),
            "af3_structure": af3_result.get("af3_structure"),
            "af3_confidence_json": af3_result.get("af3_confidence_json"),
            # APO results
            "apo_structure": apo_structure,
            # PyRosetta results (merge all keys except design_id)
            **{k: v for k, v in pr_result.items() if k != "design_id"},
            # Binder metadata
            "binder_sequence": binder_seq,
            "binder_length": len(binder_seq) if binder_seq else 0,
            "ipsae": best_cycle_data.get("ipsae", 0.0) if best_cycle_data else 0.0,
            "plddt": best_cycle_data.get("plddt", 0.0) if best_cycle_data else 0.0,
            "iplddt": best_cycle_data.get("iplddt", 0.0) if best_cycle_data else 0.0,
        }
    
    # Execute with concurrency limit
    all_results = []
    pipeline_mode = "full pipeline (Boltz→AF3→PyRosetta)" if use_alphafold3_validation else "design only"
    
    if max_concurrent > 0:
        print(f"Executing {len(tasks)} tasks [{pipeline_mode}] (concurrency: {max_concurrent} GPUs)...\n")
        
        with ThreadPoolExecutor(max_workers=max_concurrent) as executor:
            futures = {executor.submit(_run_full_pipeline_for_design, t): t for t in tasks}
            
            for i, future in enumerate(as_completed(futures)):
                result = future.result()
                all_results.append(result)
                
                # Print completion status
                status = "✓" if result.get("status") == "success" else "✗"
                design_idx = result.get("design_idx", "?")
                
                if use_alphafold3_validation and result.get("af3_iptm") is not None:
                    accepted = "✓ ACCEPTED" if result.get("accepted") else "✗ REJECTED"
                    print(f"\n[{i+1}/{len(tasks)}] {status} Design {design_idx} COMPLETE: "
                          f"Boltz={result.get('best_iptm', 0):.3f}, AF3={result.get('af3_iptm', 0):.3f}, "
                          f"dG={result.get('interface_dG', 0):.1f} → {accepted}\n")
                else:
                    print(f"[{i+1}/{len(tasks)}] {status} Design {design_idx}: ipTM={result.get('best_iptm', 0):.3f}")
    else:
        # Unlimited concurrency (design only mode)
        print(f"Executing {len(tasks)} tasks [{pipeline_mode}] (unlimited concurrency)...")
        for i, result in enumerate(gpu_fn.map(tasks)):
            all_results.append(result)
            status = "✓" if result.get("status") == "success" else "✗"
            iptm = result.get("best_iptm", 0)
            print(f"[{i+1}/{len(tasks)}] {status} Design {result.get('design_idx')}: ipTM={iptm:.3f}")
    
    # Stop sync thread
    if sync_thread:
        print("\nStopping background sync...")
        stop_sync.set()
        sync_thread.join(timeout=30)
    
    # ==========================================================================
    # UNIFIED RESULT SAVING
    # Handles both design-only and full pipeline (AF3+PyRosetta) results
    # ==========================================================================
    print(f"\nSaving results to {output_path}...")
    
    # Create all directories
    designs_dir = output_path / "designs"
    best_dir = output_path / "best_designs"
    af3_dir = output_path / "af3_validation"
    accepted_dir = output_path / "accepted_designs"
    rejected_dir = output_path / "rejected"
    
    designs_dir.mkdir(parents=True, exist_ok=True)
    best_dir.mkdir(exist_ok=True)
    
    if use_alphafold3_validation:
        af3_dir.mkdir(exist_ok=True)
        accepted_dir.mkdir(exist_ok=True)
        rejected_dir.mkdir(exist_ok=True)
    
    # Process results
    best_rows = []
    accepted = []
    rejected = []
    
    for r in all_results:
        if r.get("status") != "success":
            continue
        
        design_idx = r.get("design_idx", 0)
        best_cycle = r.get("best_cycle", 0)
        design_id = r.get("design_id", f"{name}_d{design_idx}_c{best_cycle}")
        
        # Save best design PDB (from Boltz)
        if r.get("best_pdb"):
            (best_dir / f"{design_id}.pdb").write_text(r["best_pdb"])
        
        # Save AF3 structure (if available)
        if r.get("af3_structure"):
            (af3_dir / f"{design_id}_af3.cif").write_text(r["af3_structure"])
        
        # Save relaxed PDB to accepted/rejected (if PyRosetta ran)
        if use_alphafold3_validation and r.get("relaxed_pdb"):
            if r.get("accepted"):
                (accepted_dir / f"{design_id}_relaxed.pdb").write_text(r["relaxed_pdb"])
                accepted.append(r)
            else:
                (rejected_dir / f"{design_id}_relaxed.pdb").write_text(r["relaxed_pdb"])
                rejected.append(r)
        elif use_alphafold3_validation and r.get("af3_structure"):
            # No relaxed PDB but have AF3 structure - still classify
            if r.get("accepted"):
                accepted.append(r)
            else:
                rejected.append(r)
        
        # Get cycle data for metrics
        best_cycle_data = None
        for cycle_data in r.get("cycles", []):
            if cycle_data.get("cycle") == best_cycle:
                best_cycle_data = cycle_data
                break
        
        seq = r.get("best_seq", "")
        binder_length = len(seq) if seq else 0
        alanine_count = best_cycle_data.get("alanine_count", 0) if best_cycle_data else 0
        alanine_pct = (alanine_count / binder_length * 100) if binder_length > 0 else 0.0
        
        # Build unified row with boltz_ prefix for design-stage metrics
        best_rows.append({
            # Identity
            "design_id": design_id,
            "design_num": design_idx,
            "cycle": best_cycle,
            # Binder info
            "binder_sequence": seq,
            "binder_length": binder_length,
            "cyclic": cyclic,
            "alanine_count": alanine_count,
            "alanine_pct": round(alanine_pct, 2),
            # Boltz design metrics (prefixed)
            "boltz_iptm": r.get("best_iptm", 0.0),
            "boltz_ipsae": r.get("ipsae", best_cycle_data.get("ipsae", 0.0) if best_cycle_data else 0.0),
            "boltz_plddt": r.get("plddt", best_cycle_data.get("plddt", 0.0) if best_cycle_data else 0.0),
            "boltz_iplddt": r.get("iplddt", best_cycle_data.get("iplddt", 0.0) if best_cycle_data else 0.0),
            # AF3 validation metrics
            "af3_iptm": r.get("af3_iptm"),
            "af3_ipsae": r.get("af3_ipsae"),
            "af3_ptm": r.get("af3_ptm"),
            "af3_plddt": r.get("af3_plddt"),
            # PyRosetta interface metrics
            "interface_dG": r.get("interface_dG"),
            "interface_sc": r.get("interface_sc"),
            "interface_nres": r.get("interface_nres"),
            "interface_dSASA": r.get("interface_dSASA"),
            "interface_packstat": r.get("interface_packstat"),
            "interface_hbonds": r.get("interface_interface_hbonds"),
            "interface_delta_unsat_hbonds": r.get("interface_delta_unsat_hbonds"),
            # Secondary quality metrics
            "apo_holo_rmsd": r.get("apo_holo_rmsd"),
            "i_pae": r.get("i_pae"),
            "rg": r.get("rg"),
            # Acceptance status
            "accepted": r.get("accepted"),
            "rejection_reason": r.get("rejection_reason"),
        })
    
    # Helper function to reorder columns using unified schema
    def reorder_columns(df: pd.DataFrame) -> pd.DataFrame:
        """Reorder DataFrame columns to match UNIFIED_DESIGN_COLUMNS."""
        ordered_cols = [c for c in UNIFIED_DESIGN_COLUMNS if c in df.columns]
        extra_cols = [c for c in df.columns if c not in UNIFIED_DESIGN_COLUMNS]
        return df[ordered_cols + extra_cols]
    
    # Save best_designs CSV with consistent column ordering
    if best_rows:
        best_df = pd.DataFrame(best_rows)
        best_df = reorder_columns(best_df)
        # Remove None columns for cleaner CSV
        best_df = best_df.dropna(axis=1, how='all')
        best_df.to_csv(best_dir / "best_designs.csv", index=False)
        print(f"  ✓ best_designs/ ({len(best_rows)} PDBs + best_designs.csv)")
    
    design_pdbs = list(designs_dir.glob("*.pdb"))
    print(f"  ✓ designs/ ({len(design_pdbs)} PDBs)")
    
    # Save AF3 results CSV (if AF3 ran) - includes af3_ipsae
    if use_alphafold3_validation:
        af3_rows = [{"design_id": r.get("design_id"), "af3_iptm": r.get("af3_iptm"),
                     "af3_ipsae": r.get("af3_ipsae"), "af3_ptm": r.get("af3_ptm"),
                     "af3_plddt": r.get("af3_plddt")}
                    for r in all_results if r.get("af3_iptm") is not None]
        if af3_rows:
            pd.DataFrame(af3_rows).to_csv(af3_dir / "af3_results.csv", index=False)
            print(f"  ✓ af3_validation/ ({len(af3_rows)} structures)")
        
        # Save accepted/rejected CSVs using unified schema (filtered from best_rows)
        # This ensures identical columns across best_designs, accepted_stats, rejected_stats
        accepted_rows = [row for row in best_rows if row.get("accepted") is True]
        rejected_rows = [row for row in best_rows if row.get("accepted") is False]
        
        if accepted_rows:
            accepted_df = pd.DataFrame(accepted_rows)
            accepted_df = reorder_columns(accepted_df)
            accepted_df = accepted_df.dropna(axis=1, how='all')
            accepted_df.to_csv(accepted_dir / "accepted_stats.csv", index=False)
            print(f"  ✓ accepted_designs/ ({len(accepted_rows)} designs)")
        
        if rejected_rows:
            rejected_df = pd.DataFrame(rejected_rows)
            rejected_df = reorder_columns(rejected_df)
            rejected_df = rejected_df.dropna(axis=1, how='all')
            rejected_df.to_csv(rejected_dir / "rejected_stats.csv", index=False)
            print(f"  ✓ rejected/ ({len(rejected_rows)} designs)")
    
    # ==========================================================================
    # PRINT SUMMARY
    # ==========================================================================
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    
    successful = [r for r in all_results if r.get("status") == "success"]
    print(f"Successful designs: {len(successful)}/{len(all_results)}")
    
    if successful:
        best_overall = max(successful, key=lambda r: r.get("best_iptm", 0))
        print(f"Best Boltz ipTM: {name}_d{best_overall['design_idx']} = {best_overall['best_iptm']:.3f}")
    
    if use_alphafold3_validation:
        af3_results = [r for r in successful if r.get("af3_iptm") is not None]
        if af3_results:
            best_af3 = max(af3_results, key=lambda r: r.get("af3_iptm", 0))
            print(f"Best AF3 ipTM: {best_af3.get('design_id')} = {best_af3.get('af3_iptm', 0):.3f}")
        
        print("\nPyRosetta Filtering:")
        print(f"  Accepted: {len(accepted)}")
        print(f"  Rejected: {len(rejected)}")
        
        if accepted:
            print("\n  Accepted designs:")
            for r in accepted:
                print(f"    {r.get('design_id')}: dG={r.get('interface_dG', 0):.1f}, "
                      f"SC={r.get('interface_sc', 0):.2f}, "
                      f"RMSD={r.get('apo_holo_rmsd', 'N/A')}")
    
    print(f"\nOutput: {output_path}/")
    if use_alphafold3_validation:
        print("  ├── designs/           # All cycles from Boltz")
        print("  ├── best_designs/      # Best cycle per design (Boltz PDBs)")
        print("  ├── af3_validation/    # AF3 predicted structures")
        print("  ├── accepted_designs/  # Passed PyRosetta filters (relaxed PDBs)")
        print("  └── rejected/          # Failed PyRosetta filters")
    else:
        print("  ├── designs/           # All cycles")
        print("  └── best_designs/      # Best cycle per design")


@app.local_entrypoint()
def list_gpus():
    """List available GPU types."""
    print("\nAvailable GPU types:")
    print("-" * 40)
    for gpu, desc in GPU_TYPES.items():
        default = " (DEFAULT)" if gpu == DEFAULT_GPU else ""
        # Remove "(RECOMMENDED)" from description, keep just the specs
        desc_clean = desc.replace(" (RECOMMENDED)", "")
        print(f"  {gpu}: {desc_clean}{default}")
    print("\nUsage: --gpu H100")


@app.local_entrypoint()
def test_connection(gpu: str = DEFAULT_GPU):
    """Test Modal connection and GPU."""
    print(f"Testing Modal connection with GPU: {gpu}...")
    result = _test_gpu.remote()
    print(f"\n{result}")


@app.local_entrypoint()
def test_af3():
    """Test that the AF3 image is correctly configured."""
    print("Testing AF3 image configuration...")
    print("=" * 60)
    result = test_af3_image.remote()
    print(result)
    print("=" * 60)


if __name__ == "__main__":
    print("Use 'modal run modal_boltz_ph_cli.py::<entrypoint>' to execute")
    print("\nAvailable entrypoints:")
    print("  - init_cache          Initialize model weights cache")
    print("  - upload_af3_weights  Upload AlphaFold3 weights")
    print("  - run_pipeline        Run the design pipeline")
    print("  - list_gpus           List available GPU types")
    print("  - test_connection     Test Modal connection")
    print("  - test_af3            Test AF3 image configuration")

