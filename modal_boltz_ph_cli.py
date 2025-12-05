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
        --target-seq "AFTVTVPK..." \\
        --num-designs 5
    
    # List available GPUs
    modal run modal_boltz_ph_cli.py::list_gpus
    
    # Test connection
    modal run modal_boltz_ph_cli.py::test_connection
"""

import datetime
import shutil
import tempfile
import threading
from pathlib import Path
from typing import Optional

# Import Modal app and shared resources
from modal_boltz_ph.app import app, results_dict, GPU_TYPES, DEFAULT_GPU

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
from modal_boltz_ph.sync import _sync_worker
from modal_boltz_ph.tests import test_af3_image, _test_gpu


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
    target_seq: Optional[str] = None,
    ligand_ccd: Optional[str] = None,
    ligand_smiles: Optional[str] = None,
    nucleic_seq: Optional[str] = None,
    nucleic_type: str = "dna",
    # Template
    template_path: Optional[str] = None,
    template_chain_id: Optional[str] = None,
    # Binder configuration
    seq: Optional[str] = None,
    min_protein_length: int = 100,
    max_protein_length: int = 150,
    percent_x: int = 90,
    cyclic: bool = False,
    exclude_p: bool = False,
    # Design parameters
    num_designs: int = 50,
    num_cycles: int = 5,
    contact_residues: Optional[str] = None,
    temperature: float = 0.1,
    omit_aa: str = "C",
    alanine_bias: bool = False,
    alanine_bias_start: float = -0.5,
    alanine_bias_end: float = -0.1,
    high_iptm_threshold: float = 0.8,
    high_plddt_threshold: float = 0.8,
    # Contact filtering
    no_contact_filter: bool = False,
    max_contact_filter_retries: int = 6,
    contact_cutoff: float = 15.0,
    # MSA options
    msa_mode: str = "mmseqs",  # "single" or "mmseqs"
    # Model parameters
    diffuse_steps: int = 200,
    recycling_steps: int = 3,
    randomly_kill_helix_feature: bool = False,
    negative_helix_constant: float = 0.2,
    grad_enabled: bool = False,
    logmd: bool = False,
    # Execution
    gpu: str = DEFAULT_GPU,
    max_concurrent: int = 0,  # 0 = unlimited
    output_dir: Optional[str] = None,
    no_stream: bool = False,
    sync_interval: float = 5.0,
    # AF3 Validation (optional)
    enable_af3_validation: bool = False,
    af3_msa_mode: str = "reuse",  # "none", "reuse", or "generate"
    af3_gpu: str = "A100-80GB",
    # PyRosetta Filtering (optional)
    enable_pyrosetta: bool = False,
    # Target type (affects filtering thresholds)
    target_type: str = "protein",  # "protein", "peptide", "small_molecule", "nucleic"
):
    """
    Run the Protein Hunter design pipeline on Modal.
    
    Examples:
        # Basic protein binder design
        modal run modal_boltz_ph_cli.py::run_pipeline \\
            --name "PDL1_binder" \\
            --target-seq "AFTVTVPK..." \\
            --num-designs 5 \\
            --num-cycles 7
        
        # With AF3 validation and PyRosetta filtering
        modal run modal_boltz_ph_cli.py::run_pipeline \\
            --name "PDL1_validated" \\
            --target-seq "AFTVTVPK..." \\
            --num-designs 3 \\
            --enable-af3-validation \\
            --enable-pyrosetta
        
        # With hotspots
        modal run modal_boltz_ph_cli.py::run_pipeline \\
            --name "PDL1_hotspot" \\
            --target-seq "AFTVTVPK..." \\
            --contact-residues "54,56,115" \\
            --num-designs 3
        
        # Small molecule binder
        modal run modal_boltz_ph_cli.py::run_pipeline \\
            --name "SAM_binder" \\
            --ligand-ccd "SAM" \\
            --num-designs 5
    
    AF3 Validation:
        First upload weights: modal run modal_boltz_ph_cli.py::upload_af3_weights --weights-path ~/AF3/af3.bin.zst
        Then use --enable-af3-validation flag
        
        MSA modes:
            --af3-msa-mode=none     Query-only for all chains (fast, less accurate)
            --af3-msa-mode=reuse    Reuse MSAs from design phase (recommended)
            --af3-msa-mode=generate Generate fresh MSAs for targets (slow, most accurate)
    
    PyRosetta Filtering:
        Use --enable-pyrosetta flag to run interface analysis on AF3 results.
        
        Target type controls filtering thresholds:
            --target-type=protein   Default: interface_nres > 7, BUNS < 4
            --target-type=peptide   Stricter: interface_nres > 4, BUNS < 2
    """
    import base64
    import pandas as pd
    
    stream = not no_stream
    run_id = f"{name}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # Validate inputs
    if not any([target_seq, ligand_ccd, ligand_smiles, nucleic_seq]):
        print("Error: Must provide at least one target (--target-seq, --ligand-ccd, --ligand-smiles, or --nucleic-seq)")
        return
    
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
    print(f"Target: {target_seq[:50] + '...' if target_seq and len(target_seq) > 50 else target_seq or ligand_ccd or nucleic_seq}")
    print(f"Num designs: {num_designs}")
    print(f"Num cycles: {num_cycles}")
    print(f"GPU: {gpu}")
    print(f"Max concurrent: {max_concurrent if max_concurrent > 0 else 'unlimited'}")
    print(f"Output: {output_path}")
    if template_path:
        print(f"Template: {template_path}")
        print(f"Template chains: {template_chain_id}")
    if contact_residues:
        print(f"Hotspots: {contact_residues}")
    print(f"MSA mode: {msa_mode}")
    if enable_pyrosetta:
        print(f"Target type: {target_type} (affects PyRosetta thresholds)")
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
            "protein_seqs": target_seq or "",
            "ligand_ccd": ligand_ccd or "",
            "ligand_smiles": ligand_smiles or "",
            "nucleic_seq": nucleic_seq or "",
            "nucleic_type": nucleic_type,
            "template_content": template_content,
            "template_chain_ids": template_chain_id or "",
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
    
    # Execute tasks in parallel
    if max_concurrent > 0:
        print(f"Submitting {len(tasks)} design task(s) to Modal (max {max_concurrent} concurrent GPUs)...")
    else:
        print(f"Submitting {len(tasks)} design task(s) to Modal (unlimited concurrency)...")
    
    all_results = []
    completed = 0
    
    if max_concurrent > 0 and max_concurrent < len(tasks):
        # Batched execution
        def batch_tasks(task_list, batch_size):
            for i in range(0, len(task_list), batch_size):
                yield task_list[i:i + batch_size]
        
        for batch_idx, batch in enumerate(batch_tasks(tasks, max_concurrent)):
            batch_start = batch_idx * max_concurrent
            print(f"\n--- Batch {batch_idx + 1}: designs {batch_start}-{batch_start + len(batch) - 1} ---")
            
            for result in gpu_fn.map(batch):
                all_results.append(result)
                completed += 1
                status = "✓" if result.get("status") == "success" else "✗"
                iptm = result.get("best_iptm", 0)
                print(f"[{completed}/{len(tasks)}] {status} Design {result.get('design_idx')}: best ipTM={iptm:.3f}")
    else:
        # Unlimited concurrency
        for i, result in enumerate(gpu_fn.map(tasks)):
            all_results.append(result)
            status = "✓" if result.get("status") == "success" else "✗"
            iptm = result.get("best_iptm", 0)
            print(f"[{i+1}/{len(tasks)}] {status} Design {result.get('design_idx')}: best ipTM={iptm:.3f}")
    
    # Stop sync thread
    if sync_thread:
        print("\nStopping background sync...")
        stop_sync.set()
        sync_thread.join(timeout=30)
    
    # Save results
    print(f"\nSaving results to {output_path}...")

    designs_dir = output_path / "designs"
    designs_dir.mkdir(parents=True, exist_ok=True)

    # Save best designs
    best_dir = output_path / "best_designs"
    best_dir.mkdir(exist_ok=True)
    best_rows = []

    for r in all_results:
        if r.get("best_pdb") and r.get("best_seq"):
            design_idx = r.get("design_idx", 0)
            best_cycle = r.get("best_cycle", 0)
            design_id = f"{name}_d{design_idx}_c{best_cycle}"

            pdb_file = best_dir / f"{design_id}.pdb"
            pdb_file.write_text(r["best_pdb"])

            # Find best cycle metrics
            best_cycle_data = None
            for cycle_data in r.get("cycles", []):
                if cycle_data.get("cycle") == best_cycle:
                    best_cycle_data = cycle_data
                    break

            seq = r.get("best_seq", "")
            binder_length = len(seq) if seq else 0
            alanine_count = best_cycle_data.get("alanine_count", 0) if best_cycle_data else 0
            alanine_pct = (alanine_count / binder_length * 100) if binder_length > 0 else 0.0

            best_rows.append({
                "design_id": design_id,
                "design_num": design_idx,
                "cycle": best_cycle,
                "binder_sequence": seq,
                "binder_length": binder_length,
                "cyclic": cyclic,
                "iptm": r.get("best_iptm", 0.0),
                "ipsae": best_cycle_data.get("ipsae", 0.0) if best_cycle_data else 0.0,
                "plddt": best_cycle_data.get("plddt", 0.0) if best_cycle_data else 0.0,
                "iplddt": best_cycle_data.get("iplddt", 0.0) if best_cycle_data else 0.0,
                "alanine_count": alanine_count,
                "alanine_pct": round(alanine_pct, 2),
                "target_seqs": target_seq or "",
                "contact_residues": contact_residues or "",
                "msa_mode": msa_mode,
                "ligand_smiles": ligand_smiles or "",
                "ligand_ccd": ligand_ccd or "",
                "nucleic_seq": nucleic_seq or "",
                "nucleic_type": nucleic_type or "",
                "template_path": template_path or "",
                "template_mapping": template_chain_id or "",
                # MSA content for AF3 validation (not saved to CSV)
                "_target_msas": r.get("target_msas", {}),
            })

    if best_rows:
        best_df = pd.DataFrame(best_rows)
        csv_columns = [c for c in best_df.columns if not c.startswith("_")]
        best_df[csv_columns].to_csv(best_dir / "best_designs.csv", index=False)
        print(f"  ✓ best_designs/ ({len(best_rows)} PDBs + best_designs.csv)")

    design_pdbs = list(designs_dir.glob("*.pdb"))
    print("  ✓ designs/ ({} PDBs + design_stats.csv)".format(len(design_pdbs)))

    # Print summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    successful = [r for r in all_results if r.get("status") == "success"]
    print(f"Successful: {len(successful)}/{len(all_results)}")
    if successful:
        best_overall = max(successful, key=lambda r: r.get("best_iptm", 0))
        print(f"Best overall: {name}_d{best_overall['design_idx']} with ipTM={best_overall['best_iptm']:.3f}")
    print(f"Total cycles saved: {len(design_pdbs)}")
    print(f"Best designs: {len(best_rows)}")
    print("\nOutput structure:")
    print(f"  {output_path}/")
    print("  ├── designs/           # ALL cycles (PDBs + design_stats.csv)")
    print("  └── best_designs/      # Best cycle per design run")
    print(f"\nResults saved to: {output_path}")
    
    # ===========================================
    # OPTIONAL: AF3 VALIDATION
    # ===========================================
    if enable_af3_validation and best_rows:
        print(f"\n{'='*70}")
        print("ALPHAFOLD3 VALIDATION (Parallel)")
        print(f"{'='*70}")
        
        af3_gpu_to_use = gpu if gpu in AF3_GPU_FUNCTIONS else "A100-80GB"
        af3_fn = AF3_GPU_FUNCTIONS.get(af3_gpu_to_use, run_af3_single_A100_80GB)
        
        # Build AF3 tasks
        af3_tasks = []
        for row in best_rows:
            target_msas = row.get("_target_msas", {})
            target_msa = target_msas.get("B")
            
            if target_msa and af3_msa_mode == "reuse":
                print(f"  {row['design_id']}: Using MSA ({len(target_msa)} chars)")
            elif af3_msa_mode == "reuse":
                print(f"  {row['design_id']}: No MSA available, using query-only")
            
            row_template_path = row.get("template_path", "") or template_path or ""
            row_template_chain = row.get("template_mapping", "") or template_chain_id or ""
            
            af3_tasks.append({
                "design_id": row["design_id"],
                "binder_seq": row["binder_sequence"],
                "target_seq": row["target_seqs"],
                "binder_chain": "A",
                "target_chain": "B",
                "target_msa": target_msa if af3_msa_mode == "reuse" else None,
                "af3_msa_mode": af3_msa_mode,
                "template_path": row_template_path if row_template_path else None,
                "template_chain_id": row_template_chain if row_template_chain else None,
            })
        
        print(f"Submitting {len(af3_tasks)} designs for AF3 validation...")
        print(f"GPU: {af3_gpu_to_use}")
        print(f"MSA mode: {af3_msa_mode}")
        if template_path:
            print(f"Template: {template_path} (chain {template_chain_id})")
        
        try:
            af3_results_list = list(af3_fn.starmap([
                (t["design_id"], t["binder_seq"], t["target_seq"], t["binder_chain"], t["target_chain"], 
                 t["target_msa"], t["af3_msa_mode"], t["template_path"], t["template_chain_id"])
                for t in af3_tasks
            ]))
            
            af3_results = {r["design_id"]: r for r in af3_results_list}
            
            # Save AF3 results
            af3_dir = output_path / "af3_validation"
            af3_dir.mkdir(exist_ok=True)
            
            for design_id, result in af3_results.items():
                if result.get("af3_structure"):
                    cif_file = af3_dir / f"{design_id}_af3.cif"
                    cif_file.write_text(result["af3_structure"])
            
            af3_rows = []
            for design_id, result in af3_results.items():
                af3_rows.append({
                    "design_id": design_id,
                    "af3_iptm": result.get("af3_iptm", 0),
                    "af3_ptm": result.get("af3_ptm", 0),
                    "af3_plddt": result.get("af3_plddt", 0),
                })
            
            if af3_rows:
                af3_df = pd.DataFrame(af3_rows)
                af3_df.to_csv(af3_dir / "af3_results.csv", index=False)
            
            print(f"\n✓ AF3 validation complete: {len(af3_results)} structures")
            print(f"  Results saved to: {af3_dir}/")
            
            # ===========================================
            # AF3 APO PREDICTIONS (for RMSD)
            # ===========================================
            apo_results = {}
            if enable_pyrosetta:
                print(f"\n{'='*70}")
                print("ALPHAFOLD3 APO PREDICTIONS")
                print(f"{'='*70}")
                
                af3_apo_fn = AF3_APO_GPU_FUNCTIONS.get(af3_gpu_to_use, run_af3_apo_A100_80GB)
                
                apo_tasks = []
                for row in best_rows:
                    design_id = row["design_id"]
                    if design_id in af3_results and af3_results[design_id].get("af3_structure"):
                        apo_tasks.append({
                            "design_id": design_id,
                            "binder_seq": row["binder_sequence"],
                            "binder_chain": "A",
                        })
                
                print(f"Submitting {len(apo_tasks)} APO predictions...")
                
                try:
                    apo_results_list = list(af3_apo_fn.starmap([
                        (t["design_id"], t["binder_seq"], t["binder_chain"])
                        for t in apo_tasks
                    ]))
                    apo_results = {r["design_id"]: r for r in apo_results_list}
                    
                    apo_success = sum(1 for r in apo_results.values() if r.get("apo_structure"))
                    print(f"\n✓ APO predictions complete: {apo_success}/{len(apo_tasks)} structures")
                except Exception as e:
                    print(f"⚠ APO predictions error: {e}")
            
            # ===========================================
            # PYROSETTA FILTERING
            # ===========================================
            if enable_pyrosetta:
                print(f"\n{'='*70}")
                print("PYROSETTA FILTERING (Parallel)")
                print(f"{'='*70}")
                
                pr_tasks = []
                for design_id, result in af3_results.items():
                    if result.get("af3_structure"):
                        pr_tasks.append({
                            "design_id": design_id,
                            "af3_structure": result["af3_structure"],
                            "af3_iptm": result.get("af3_iptm", 0),
                            "af3_plddt": result.get("af3_plddt", 0),
                            "binder_chain": "A",
                            "target_chain": "B",
                            "apo_structure": apo_results.get(design_id, {}).get("apo_structure"),
                            "af3_confidence_json": result.get("af3_confidence_json"),
                            "target_type": target_type,
                        })
                
                print(f"Submitting {len(pr_tasks)} structures for PyRosetta analysis...")
                print("  - FastRelax + InterfaceAnalyzer + APO-HOLO RMSD")
                
                try:
                    pr_results_list = list(run_pyrosetta_single.starmap([
                        (t["design_id"], t["af3_structure"], t["af3_iptm"], t["af3_plddt"], 
                         t["binder_chain"], t["target_chain"], t["apo_structure"], t["af3_confidence_json"],
                         t["target_type"])
                        for t in pr_tasks
                    ]))
                    
                    accepted = [r for r in pr_results_list if r.get("accepted")]
                    rejected = [r for r in pr_results_list if not r.get("accepted")]
                    
                    for r in pr_results_list:
                        status = "✓" if r.get("accepted") else "✗"
                        reason = f" ({r.get('rejection_reason')})" if r.get("rejection_reason") else ""
                        rmsd_str = f", RMSD={r.get('apo_holo_rmsd', 'N/A')}" if r.get('apo_holo_rmsd') is not None else ""
                        rg_str = f", rg={r.get('rg', 'N/A')}" if r.get('rg') is not None else ""
                        ipae_str = f", iPAE={r.get('i_pae', 'N/A')}" if r.get('i_pae') is not None else ""
                        metrics = f"dG={r.get('interface_dG', 0):.1f}, SC={r.get('interface_sc', 0):.2f}, nres={r.get('interface_nres', 0)}{rmsd_str}{rg_str}{ipae_str}"
                        print(f"  {status} {r['design_id']}: {metrics}{reason}")
                    
                    # Save results
                    accepted_dir = output_path / "accepted_designs"
                    rejected_dir = output_path / "rejected"
                    accepted_dir.mkdir(exist_ok=True)
                    rejected_dir.mkdir(exist_ok=True)
                    
                    csv_exclude = {"relaxed_pdb", "_target_msas", "af3_confidence_json"}
                    
                    if accepted:
                        accepted_csv_data = [
                            {k: v for k, v in r.items() if k not in csv_exclude}
                            for r in accepted
                        ]
                        accepted_df = pd.DataFrame(accepted_csv_data)
                        accepted_df.to_csv(accepted_dir / "accepted_stats.csv", index=False)
                        
                        for entry in accepted:
                            design_id = entry["design_id"]
                            relaxed_pdb = entry.get("relaxed_pdb")
                            if relaxed_pdb:
                                pdb_path = accepted_dir / f"{design_id}_relaxed.pdb"
                                pdb_path.write_text(relaxed_pdb)
                            else:
                                src_cif = af3_dir / f"{design_id}_af3.cif"
                                if src_cif.exists():
                                    shutil.copy(src_cif, accepted_dir / f"{design_id}_af3.cif")
                        
                        print(f"  ✓ Saved {len(accepted)} relaxed PDBs to accepted_designs/")
                    
                    if rejected:
                        rejected_csv_data = [
                            {k: v for k, v in r.items() if k not in csv_exclude}
                            for r in rejected
                        ]
                        rejected_df = pd.DataFrame(rejected_csv_data)
                        rejected_df.to_csv(rejected_dir / "rejected_stats.csv", index=False)
                        
                        for entry in rejected:
                            design_id = entry["design_id"]
                            relaxed_pdb = entry.get("relaxed_pdb")
                            if relaxed_pdb:
                                pdb_path = rejected_dir / f"{design_id}_relaxed.pdb"
                                pdb_path.write_text(relaxed_pdb)
                        
                        print(f"  ✓ Saved {len(rejected)} relaxed PDBs to rejected/")
                    
                    print("\n✓ PyRosetta filtering complete")
                    print(f"  Accepted: {len(accepted)}")
                    print(f"  Rejected: {len(rejected)}")
                    
                    if accepted:
                        print("\n  Interface Metrics (accepted):")
                        for r in accepted:
                            print(f"    {r['design_id']}:")
                            print(f"      interface_dG: {r.get('interface_dG', 0):.2f}")
                            print(f"      interface_sc: {r.get('interface_sc', 0):.3f}")
                            print(f"      interface_dSASA: {r.get('interface_dSASA', 0):.2f}")
                            print(f"      interface_nres: {r.get('interface_nres', 0)}")
                            print(f"      binder_sasa: {r.get('binder_sasa', 0):.2f}")
                            print(f"      interface_fraction: {r.get('interface_fraction', 0):.2f}%")
                            print(f"      interface_hbond_pct: {r.get('interface_hbond_percentage', 0):.2f}%")
                            if r.get('apo_holo_rmsd') is not None:
                                print(f"      apo_holo_rmsd: {r.get('apo_holo_rmsd'):.2f}")
                            if r.get('rg') is not None:
                                print(f"      rg: {r.get('rg'):.2f}")
                            if r.get('i_pae') is not None:
                                print(f"      i_pae: {r.get('i_pae'):.2f}")
                
                except Exception as e:
                    print(f"⚠ PyRosetta error: {e}")
        
        except Exception as e:
            print(f"⚠ AF3 validation error: {e}")
    
    elif enable_af3_validation:
        print("\n⚠ No best designs found for AF3 validation")


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

