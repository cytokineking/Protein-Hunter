#!/usr/bin/env python3
"""
Modal deployment for Protein Hunter (Boltz Edition).

This module provides serverless GPU execution of the Protein Hunter design pipeline
on Modal's cloud infrastructure.

KEY DESIGN:
- Each design run executes as an independent Modal function
- Results stream to Modal Dict for real-time local sync
- Boltz weights and LigandMPNN models cached in persistent volume

Usage:
    # Initialize cache (run once)
    modal run modal_protein_hunter.py::init_cache

    # Run design pipeline
    modal run modal_protein_hunter.py::run_pipeline \
        --name "PDL1_binder" \
        --target-seq "AFTVTVPKDLYVVEYGSNMTIECKFPVEKQLDLAALIVYWEMEDKNIIQFVHGEEDLKVQHSSYRQRARLLKDQLSLGNAALQITDVKLQDAGVYRCMISYGGADYKRITVKVNAPYAAALE" \
        --num-designs 5 \
        --num-cycles 7 \
        --gpu H100

    # With hotspots and template
    modal run modal_protein_hunter.py::run_pipeline \
        --name "PDL1_hotspot" \
        --target-seq "AFTVTVPK..." \
        --contact-residues "54,56,115" \
        --num-designs 3 \
        --output-dir ./results
"""

import datetime
import gc
import json
import os
import random
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, Dict, Optional

import modal

app = modal.App("protein-hunter-boltz")

# Volume for caching model weights
cache_volume = modal.Volume.from_name("protein-hunter-cache", create_if_missing=True)

# Dict for real-time result streaming
results_dict = modal.Dict.from_name("protein-hunter-results", create_if_missing=True)

# Supported GPU types
GPU_TYPES = {
    "T4": "16GB - $0.59/h",
    "L4": "24GB - $0.80/h",
    "A10G": "24GB - $1.10/h",
    "L40S": "48GB - $1.95/h",
    "A100-40GB": "40GB - $2.10/h",
    "A100-80GB": "80GB - $2.50/h",
    "H100": "80GB - $3.95/h (RECOMMENDED)",
}

DEFAULT_GPU = "H100"

# =============================================================================
# MODAL IMAGE DEFINITION
# =============================================================================

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "wget", "build-essential")
    .pip_install(
        # Core dependencies (DON'T install boltz from PyPI - we use local fork)
        "torch>=2.2",
        "numpy>=1.24,<2.0",
        "pandas>=2.0",
        "pyyaml>=6.0",
        "biopython>=1.83",
        "gemmi>=0.6.3",
        "prody>=2.4",
        "matplotlib>=3.7",
        "rdkit>=2024.3.1",
        # ML dependencies
        "ml-collections>=0.1.1",
        "dm-tree>=0.1.8",
        "einops>=0.7",
        "scipy>=1.12",
        # Visualization (needed by model_utils.py imports)
        "py3Dmol",
        # Additional deps for custom boltz fork
        "hydra-core==1.3.2",
        "pytorch-lightning==2.5.0",
        "einx==0.3.0",
        "fairscale==0.4.13",
        "mashumaro==3.14",
        "modelcif==1.2",
        "wandb==0.18.7",
        "click==8.1.7",
        "numba>=0.60",
        "scikit-learn>=1.3",
        "chembl_structure_pipeline>=1.2",
    )
    # Add protein-hunter code to image with copy=True so we can run commands after
    .add_local_dir("boltz_ph", "/root/protein_hunter/boltz_ph", copy=True)
    .add_local_dir("LigandMPNN", "/root/protein_hunter/LigandMPNN", copy=True)
    .add_local_dir("utils", "/root/protein_hunter/utils", copy=True)
    # Install the local boltz fork from boltz_ph
    .run_commands(
        "cd /root/protein_hunter/boltz_ph && pip install -e .",
        # Handle optional cuequivariance
        "pip install cuequivariance-torch || pip install cuequivariance_torch || echo 'cuequivariance not available'",
    )
)


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def sample_seq(length: int, exclude_P: bool = False, frac_X: float = 0.5) -> str:
    """Generate a random sequence with specified fraction of X residues."""
    aa_list = list("ARNDCEQGHILKMFPSTWYV")
    if exclude_P:
        aa_list.remove("P")
    
    seq = []
    for _ in range(length):
        if random.random() < frac_X:
            seq.append("X")
        else:
            seq.append(random.choice(aa_list))
    return "".join(seq)


def shallow_copy_tensor_dict(d: Dict) -> Dict:
    """Create a shallow copy of a dict with tensors."""
    import torch
    result = {}
    for k, v in d.items():
        if isinstance(v, torch.Tensor):
            result[k] = v.detach().clone()
        elif isinstance(v, dict):
            result[k] = shallow_copy_tensor_dict(v)
        elif isinstance(v, list):
            result[k] = [x.detach().clone() if isinstance(x, torch.Tensor) else x for x in v]
        else:
            result[k] = v
    return result


# =============================================================================
# CORE DESIGN FUNCTION
# =============================================================================

def _run_design_impl(task_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Run a single design trajectory (all cycles) for one binder.
    
    This is the core implementation that runs inside Modal containers.
    """
    import numpy as np
    import torch
    import yaml
    
    # Add protein hunter to path
    sys.path.insert(0, "/root/protein_hunter")
    
    from boltz_ph.constants import CHAIN_TO_NUMBER
    from boltz_ph.model_utils import (
        get_boltz_model,
        load_canonicals,
        run_prediction,
        save_pdb,
        design_sequence,
        clean_memory,
        sample_seq as ph_sample_seq,
        binder_binds_contacts,
    )
    from utils.ipsae_utils import calculate_ipsae_from_boltz_output
    from LigandMPNN.wrapper import LigandMPNNWrapper
    
    # Extract task parameters
    run_id = task_dict["run_id"]
    design_idx = task_dict["design_idx"]
    total_designs = task_dict["total_designs"]
    stream_to_dict = task_dict.get("stream_to_dict", True)
    
    # Design parameters
    protein_seqs = task_dict.get("protein_seqs", "")
    ligand_ccd = task_dict.get("ligand_ccd", "")
    ligand_smiles = task_dict.get("ligand_smiles", "")
    nucleic_seq = task_dict.get("nucleic_seq", "")
    nucleic_type = task_dict.get("nucleic_type", "dna")
    template_content = task_dict.get("template_content", "")  # Base64-encoded PDB content
    template_chain_ids = task_dict.get("template_chain_ids", "")  # Chain IDs from template file
    msa_mode = task_dict.get("msa_mode", "single")
    
    # Binder parameters
    starting_seq = task_dict.get("seq", "")
    min_length = task_dict.get("min_protein_length", 100)
    max_length = task_dict.get("max_protein_length", 150)
    percent_X = task_dict.get("percent_X", 50)
    cyclic = task_dict.get("cyclic", False)
    exclude_P = task_dict.get("exclude_P", False)
    
    # Optimization parameters
    num_cycles = task_dict.get("num_cycles", 7)
    contact_residues = task_dict.get("contact_residues", "")
    temperature = task_dict.get("temperature", 0.1)
    omit_AA = task_dict.get("omit_AA", "C")
    alanine_bias = task_dict.get("alanine_bias", True)
    alanine_bias_start = task_dict.get("alanine_bias_start", -0.5)
    alanine_bias_end = task_dict.get("alanine_bias_end", -0.1)
    high_iptm_threshold = task_dict.get("high_iptm_threshold", 0.7)
    high_plddt_threshold = task_dict.get("high_plddt_threshold", 0.7)
    
    # Contact filtering
    no_contact_filter = task_dict.get("no_contact_filter", False)
    contact_cutoff = task_dict.get("contact_cutoff", 15.0)
    max_contact_filter_retries = task_dict.get("max_contact_filter_retries", 6)
    
    # Model parameters
    diffuse_steps = task_dict.get("diffuse_steps", 200)
    recycling_steps = task_dict.get("recycling_steps", 3)
    boltz_model_version = task_dict.get("boltz_model_version", "boltz2")
    randomly_kill_helix_feature = task_dict.get("randomly_kill_helix_feature", False)
    negative_helix_constant = task_dict.get("negative_helix_constant", 0.2)
    grad_enabled = task_dict.get("grad_enabled", False)
    logmd = task_dict.get("logmd", False)
    
    # Setup
    device = "cuda" if torch.cuda.is_available() else "cpu"
    work_dir = Path(tempfile.mkdtemp())
    binder_chain = "A"
    
    # Results container
    result = {
        "status": "pending",
        "run_id": run_id,
        "design_idx": design_idx,
        "cycles": [],
        "best_iptm": 0.0,
        "best_cycle": -1,
        "best_seq": None,
        "best_pdb": None,
        "error": None,
    }
    
    try:
        print(f"\n{'='*60}")
        print(f"Design {design_idx + 1}/{total_designs} for run {run_id}")
        print(f"{'='*60}")
        
        # Setup paths
        cache_dir = Path("/cache/boltz")
        ccd_path = cache_dir / "mols"
        model_path = cache_dir / "boltz2_conf.ckpt"
        
        # Symlink LigandMPNN model params from cache to expected location
        mpnn_cache = Path("/cache/ligandmpnn")
        mpnn_dest = Path("/root/protein_hunter/LigandMPNN/model_params")
        if mpnn_cache.exists() and not mpnn_dest.exists():
            mpnn_dest.mkdir(parents=True, exist_ok=True)
            for model_file in mpnn_cache.glob("*.pt"):
                dest_file = mpnn_dest / model_file.name
                if not dest_file.exists():
                    os.symlink(model_file, dest_file)
            print("Symlinked LigandMPNN models from cache")
        
        # Load CCD library
        print("Loading CCD library...")
        ccd_lib = load_canonicals(str(ccd_path))
        
        # Load Boltz model
        print("Loading Boltz model...")
        predict_args = {
            "recycling_steps": recycling_steps,
            "sampling_steps": diffuse_steps,
            "diffusion_samples": 1,
            "write_confidence_summary": True,
            "write_full_pae": True,  # Enable for ipSAE calculation
            "write_full_pde": False,
            "max_parallel_samples": 1,
        }
        boltz_model = get_boltz_model(
            checkpoint=str(model_path),
            predict_args=predict_args,
            device=device,
            model_version=boltz_model_version,
            no_potentials=not bool(contact_residues),
            grad_enabled=grad_enabled,
        )
        
        # Initialize LigandMPNN
        designer = LigandMPNNWrapper("/root/protein_hunter/LigandMPNN/run.py")
        
        # Build input data
        print("Building input data...")
        data = _build_input_data(
            protein_seqs=protein_seqs,
            ligand_ccd=ligand_ccd,
            ligand_smiles=ligand_smiles,
            nucleic_seq=nucleic_seq,
            nucleic_type=nucleic_type,
            msa_mode=msa_mode,
            work_dir=work_dir,
            cyclic=cyclic,
            contact_residues=contact_residues,
            template_content=template_content,
            template_chain_ids=template_chain_ids,
        )
        
        pocket_conditioning = bool(contact_residues and contact_residues.strip())
        
        # Determine protein chain IDs for contact checking
        protein_chain_ids = []
        if protein_seqs:
            seqs = protein_seqs.split(":") if ":" in protein_seqs else [protein_seqs]
            protein_chain_ids = [chr(ord('B') + i) for i in range(len(seqs))]
        
        # Initialize binder sequence
        if starting_seq:
            binder_length = len(starting_seq)
            initial_seq = starting_seq
        else:
            binder_length = random.randint(min_length, max_length)
            initial_seq = ph_sample_seq(binder_length, exclude_P=exclude_P, frac_X=percent_X/100)
        
        # Update binder sequence in data
        for seq_entry in data["sequences"]:
            if "protein" in seq_entry and binder_chain in seq_entry["protein"]["id"]:
                seq_entry["protein"]["sequence"] = initial_seq
        
        print(f"Binder length: {binder_length}, Initial seq sample: {initial_seq[:20]}...")
        
        # Tracking variables
        best_iptm = float("-inf")
        best_seq = None
        best_pdb_content = None
        best_cycle_idx = -1
        
        # ===== CYCLE 0: Initial prediction with contact filtering =====
        print("\n--- Cycle 0: Initial structure prediction ---")
        
        contact_filter_attempt = 0
        pdb_file = work_dir / "cycle_0.pdb"
        batch_feats = None  # Will be set in the loop
        
        while True:
            output, structure, batch_feats = run_prediction(
                data,
                binder_chain,
                randomly_kill_helix_feature=randomly_kill_helix_feature,
                negative_helix_constant=negative_helix_constant,
                boltz_model=boltz_model,
                ccd_lib=ccd_lib,
                ccd_path=ccd_path,
                logmd=logmd,
                device=device,
                boltz_model_version=boltz_model_version,
                pocket_conditioning=pocket_conditioning,
                return_feats=True,
            )
            
            plddts = output["plddt"].detach().cpu().numpy()[0]
            save_pdb(structure, output["coords"], plddts, str(pdb_file))
            
            # Contact filtering check
            contact_check_okay = True
            if contact_residues and contact_residues.strip() and not no_contact_filter:
                try:
                    # Check if binder contacts required residues
                    binds = all([
                        binder_binds_contacts(
                            str(pdb_file),
                            binder_chain,
                            protein_chain_ids[i] if i < len(protein_chain_ids) else protein_chain_ids[0],
                            contact_res,
                            cutoff=contact_cutoff,
                        )
                        for i, contact_res in enumerate(contact_residues.split("|"))
                    ])
                    if not binds:
                        print("  ❌ Binder does NOT contact required residues. Retrying...")
                        contact_check_okay = False
                except Exception as e:
                    print(f"  WARNING: Could not perform contact check: {e}")
                    contact_check_okay = True  # Fail open
            
            if contact_check_okay:
                break
            
            contact_filter_attempt += 1
            if contact_filter_attempt >= max_contact_filter_retries:
                print(f"  WARNING: Max retries ({max_contact_filter_retries}) reached. Proceeding anyway.")
                break
            
            # Resample initial sequence and try again
            initial_seq = ph_sample_seq(binder_length, exclude_P=exclude_P, frac_X=percent_X/100)
            for seq_entry in data["sequences"]:
                if "protein" in seq_entry and binder_chain in seq_entry["protein"]["id"]:
                    seq_entry["protein"]["sequence"] = initial_seq
            clean_memory()
        
        clean_memory()
        
        # Calculate cycle 0 metrics
        binder_chain_idx = CHAIN_TO_NUMBER[binder_chain]
        pair_chains = output["pair_chains_iptm"]
        
        if len(pair_chains) > 1:
            values = [
                (pair_chains[binder_chain_idx][i].detach().cpu().numpy() +
                 pair_chains[i][binder_chain_idx].detach().cpu().numpy()) / 2.0
                for i in range(len(pair_chains)) if i != binder_chain_idx
            ]
            cycle_0_iptm = float(np.mean(values) if values else 0.0)
        else:
            cycle_0_iptm = 0.0
        
        cycle_0_plddt = float(output.get("complex_plddt", torch.tensor([0.0])).detach().cpu().numpy()[0])
        cycle_0_iplddt = float(output.get("complex_iplddt", torch.tensor([0.0])).detach().cpu().numpy()[0])
        
        # Calculate ipSAE for cycle 0
        ipsae_result = calculate_ipsae_from_boltz_output(
            output, batch_feats, binder_chain_idx=binder_chain_idx
        )
        cycle_0_ipsae = ipsae_result['ipSAE']
        
        result["cycles"].append({
            "cycle": 0,
            "iptm": cycle_0_iptm,
            "ipsae": cycle_0_ipsae,
            "plddt": cycle_0_plddt,
            "iplddt": cycle_0_iplddt,
            "alanine_count": 0,
            "seq": initial_seq,
        })
        
        print(f"  Cycle 0: ipTM={cycle_0_iptm:.3f}, ipSAE={cycle_0_ipsae:.3f}, pLDDT={cycle_0_plddt:.1f}, iPLDDT={cycle_0_iplddt:.1f}")

        # Get binder name from run_id (format: name_timestamp)
        binder_name = "_".join(run_id.split("_")[:-2]) if "_" in run_id else run_id

        # Stream cycle 0 results
        if stream_to_dict:
            _stream_result(
                run_id=run_id,
                design_idx=design_idx,
                cycle=0,
                iptm=cycle_0_iptm,
                ipsae=cycle_0_ipsae,
                plddt=cycle_0_plddt,
                iplddt=cycle_0_iplddt,
                seq=initial_seq,
                pdb_file=pdb_file,
                binder_name=binder_name,
                target_seqs=protein_seqs,
                contact_residues=contact_residues,
                msa_mode=msa_mode,
                cyclic=cyclic,
                alanine_count=0,
            )
        
        clean_memory()
        
        # ===== OPTIMIZATION CYCLES =====
        current_pdb_file = pdb_file
        
        for cycle in range(num_cycles):
            print(f"\n--- Cycle {cycle + 1}/{num_cycles} ---")
            
            try:
                # Calculate alanine bias for this cycle
                cycle_norm = (cycle / (num_cycles - 1)) if num_cycles > 1 else 0.0
                alpha = alanine_bias_start - cycle_norm * (alanine_bias_start - alanine_bias_end)
                
                # Design sequence
                model_type = "ligand_mpnn" if (ligand_smiles or ligand_ccd or nucleic_seq) else "soluble_mpnn"
                design_kwargs = {
                    "pdb_file": str(current_pdb_file),
                    "temperature": temperature,
                    "chains_to_design": binder_chain,
                    "omit_AA": f"{omit_AA},P" if cycle == 0 else omit_AA,
                    "bias_AA": f"A:{alpha}" if alanine_bias else "",
                }
                
                seq_str, _ = design_sequence(designer, model_type, **design_kwargs)
                seq = seq_str.split(":")[CHAIN_TO_NUMBER[binder_chain]]
                
                # Update data with new sequence
                for seq_entry in data["sequences"]:
                    if "protein" in seq_entry and binder_chain in seq_entry["protein"]["id"]:
                        seq_entry["protein"]["sequence"] = seq
                
                # Calculate alanine percentage
                alanine_count = seq.count("A")
                alanine_pct = alanine_count / binder_length if binder_length > 0 else 0.0
                
                # Predict structure
                output, structure, batch_feats = run_prediction(
                    data,
                    binder_chain,
                    seq=seq,
                    randomly_kill_helix_feature=False,
                    negative_helix_constant=0.0,
                    boltz_model=boltz_model,
                    ccd_lib=ccd_lib,
                    ccd_path=ccd_path,
                    logmd=False,
                    device=device,
                    return_feats=True,
                )
                
                # Save PDB
                pdb_file = work_dir / f"cycle_{cycle + 1}.pdb"
                plddts = output["plddt"].detach().cpu().numpy()[0]
                save_pdb(structure, output["coords"], plddts, str(pdb_file))
                current_pdb_file = pdb_file
                
                # Calculate ipTM
                pair_chains = output["pair_chains_iptm"]
                if len(pair_chains) > 1:
                    values = [
                        (pair_chains[binder_chain_idx][i].detach().cpu().numpy() +
                         pair_chains[i][binder_chain_idx].detach().cpu().numpy()) / 2.0
                        for i in range(len(pair_chains)) if i != binder_chain_idx
                    ]
                    current_iptm = float(np.mean(values) if values else 0.0)
                else:
                    current_iptm = 0.0
                
                # Calculate ipSAE
                ipsae_result = calculate_ipsae_from_boltz_output(
                    output, batch_feats, binder_chain_idx=binder_chain_idx
                )
                current_ipsae = ipsae_result['ipSAE']
                
                current_plddt = float(output.get("complex_plddt", torch.tensor([0.0])).detach().cpu().numpy()[0])
                current_iplddt = float(output.get("complex_iplddt", torch.tensor([0.0])).detach().cpu().numpy()[0])
                
                print(f"  ipTM={current_iptm:.3f}, ipSAE={current_ipsae:.3f}, pLDDT={current_plddt:.1f}, iPLDDT={current_iplddt:.1f}, Ala={alanine_count} ({alanine_pct*100:.1f}%)")
                
                # Store cycle results
                result["cycles"].append({
                    "cycle": cycle + 1,
                    "iptm": current_iptm,
                    "ipsae": current_ipsae,
                    "plddt": current_plddt,
                    "iplddt": current_iplddt,
                    "alanine_count": alanine_count,
                    "seq": seq,
                })
                
                # Stream cycle results
                if stream_to_dict:
                    _stream_result(
                        run_id=run_id,
                        design_idx=design_idx,
                        cycle=cycle + 1,
                        iptm=current_iptm,
                        ipsae=current_ipsae,
                        plddt=current_plddt,
                        iplddt=current_iplddt,
                        seq=seq,
                        pdb_file=pdb_file,
                        binder_name=binder_name,
                        target_seqs=protein_seqs,
                        contact_residues=contact_residues,
                        msa_mode=msa_mode,
                        cyclic=cyclic,
                        alanine_count=alanine_count,
                    )
                
                # Update best if acceptable (low alanine %)
                if alanine_pct <= 0.20 and current_iptm > best_iptm:
                    best_iptm = current_iptm
                    best_seq = seq
                    best_pdb_content = pdb_file.read_text()
                    best_cycle_idx = cycle + 1

                clean_memory()
                
            except Exception as cycle_error:
                print(f"  ⚠ Cycle {cycle + 1} error: {cycle_error}")
                result["cycles"].append({
                    "cycle": cycle + 1,
                    "iptm": 0.0,
                    "plddt": 0.0,
                    "alanine_count": 0,
                    "seq": "",
                    "error": str(cycle_error),
                })
                continue
        
        # Finalize results
        result["status"] = "success"
        result["best_iptm"] = best_iptm if best_iptm > float("-inf") else 0.0
        result["best_cycle"] = best_cycle_idx
        result["best_seq"] = best_seq
        result["best_pdb"] = best_pdb_content
        
        print(f"\n✓ Design {design_idx} complete: best ipTM={result['best_iptm']:.3f} at cycle {best_cycle_idx}")
        
    except Exception as e:
        result["status"] = "error"
        result["error"] = str(e)
        import traceback
        print(f"Error in design {design_idx}: {traceback.format_exc()}")
    
    finally:
        # Cleanup
        shutil.rmtree(work_dir, ignore_errors=True)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    return result


def _build_input_data(
    protein_seqs: str,
    ligand_ccd: str,
    ligand_smiles: str,
    nucleic_seq: str,
    nucleic_type: str,
    msa_mode: str,
    work_dir: Path,
    cyclic: bool,
    contact_residues: str,
    template_content: str = "",
    template_chain_ids: str = "",
) -> Dict:
    """Build the input data dictionary for Boltz prediction.
    
    Args:
        template_content: Base64-encoded PDB/CIF file content (if using template)
        template_chain_ids: Comma-separated chain IDs from template file to use for each target protein
                           e.g., "A,B,C" means use chain A for first target, B for second, C for third
    """
    import base64
    from boltz_ph.model_utils import process_msa
    
    sequences = []
    
    # Parse protein sequences
    protein_seqs_list = protein_seqs.split(":") if protein_seqs else []
    protein_chain_ids = [chr(ord('B') + i) for i in range(len(protein_seqs_list))]
    
    next_chain_idx = len(protein_chain_ids)
    ligand_chain_id = None
    nucleic_chain_id = None
    
    if ligand_smiles or ligand_ccd:
        ligand_chain_id = chr(ord('B') + next_chain_idx)
        next_chain_idx += 1
    
    if nucleic_seq:
        nucleic_chain_id = chr(ord('B') + next_chain_idx)
        next_chain_idx += 1
    
    # Track unique sequences for MSA deduplication
    seq_to_msa_path = {}
    
    # Add target protein chains
    for i, seq in enumerate(protein_seqs_list):
        if not seq:
            continue
        
        # Determine MSA value based on mode
        if msa_mode == "mmseqs":
            # Use cached MSA if same sequence was already processed
            if seq in seq_to_msa_path:
                msa_value = str(seq_to_msa_path[seq])
            else:
                print(f"  Generating MSA for chain {protein_chain_ids[i]}...")
                msa_path = process_msa(protein_chain_ids[i], seq, work_dir)
                seq_to_msa_path[seq] = msa_path
                msa_value = str(msa_path)
        else:
            msa_value = "empty"
        
        sequences.append({
            "protein": {
                "id": [protein_chain_ids[i]],
                "sequence": seq,
                "msa": msa_value,
            }
        })
    
    # Add binder chain (placeholder sequence)
    sequences.append({
        "protein": {
            "id": ["A"],
            "sequence": "X",  # Will be updated
            "msa": "empty",
            "cyclic": cyclic,
        }
    })
    
    # Add ligand
    if ligand_smiles:
        sequences.append({"ligand": {"id": [ligand_chain_id], "smiles": ligand_smiles}})
    elif ligand_ccd:
        sequences.append({"ligand": {"id": [ligand_chain_id], "ccd": ligand_ccd}})
    
    # Add nucleic acid
    if nucleic_seq:
        sequences.append({nucleic_type: {"id": [nucleic_chain_id], "sequence": nucleic_seq}})
    
    data = {"sequences": sorted(sequences, key=lambda e: list(e.values())[0]["id"][0])}
    
    # Add templates if provided
    if template_content and template_chain_ids:
        # Decode and write template to temp file
        template_bytes = base64.b64decode(template_content)
        template_file = work_dir / "template.pdb"
        template_file.write_bytes(template_bytes)
        
        # Parse template chain IDs
        template_chain_list = [c.strip() for c in template_chain_ids.split(",") if c.strip()]
        
        # Build template entries - one per target protein chain
        templates = []
        for i, target_chain_id in enumerate(protein_chain_ids):
            if i < len(template_chain_list):
                cif_chain = template_chain_list[i]
                templates.append({
                    "pdb": str(template_file),
                    "chain_id": target_chain_id,  # Target chain in our design (B, C, D, ...)
                    "cif_chain_id": cif_chain,    # Chain from template file (A, B, C, ...)
                })
        
        if templates:
            data["templates"] = templates
            print(f"  Added {len(templates)} template(s) from PDB file")
    
    # Add constraints for contact residues
    if contact_residues and contact_residues.strip():
        contacts = []
        for i, res_str in enumerate(contact_residues.split("|")):
            for res in res_str.split(","):
                if res.strip():
                    contacts.append([protein_chain_ids[i], int(res.strip())])
        if contacts:
            data["constraints"] = [{"pocket": {"binder": "A", "contacts": contacts}}]
    
    return data


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
):
    """Stream a cycle result to the Modal Dict with full config for CSV."""
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
        }
    except Exception as e:
        print(f"  Stream error: {e}")




# =============================================================================
# GPU-SPECIFIC MODAL FUNCTIONS
# =============================================================================

@app.function(image=image, gpu="T4", timeout=7200, volumes={"/cache": cache_volume}, max_containers=20)
def run_design_T4(task_dict: Dict[str, Any]) -> Dict[str, Any]:
    return _run_design_impl(task_dict)

@app.function(image=image, gpu="L4", timeout=7200, volumes={"/cache": cache_volume}, max_containers=20)
def run_design_L4(task_dict: Dict[str, Any]) -> Dict[str, Any]:
    return _run_design_impl(task_dict)

@app.function(image=image, gpu="A10G", timeout=7200, volumes={"/cache": cache_volume}, max_containers=20)
def run_design_A10G(task_dict: Dict[str, Any]) -> Dict[str, Any]:
    return _run_design_impl(task_dict)

@app.function(image=image, gpu="L40S", timeout=7200, volumes={"/cache": cache_volume}, max_containers=20)
def run_design_L40S(task_dict: Dict[str, Any]) -> Dict[str, Any]:
    return _run_design_impl(task_dict)

@app.function(image=image, gpu="A100", timeout=7200, volumes={"/cache": cache_volume}, max_containers=20)
def run_design_A100_40GB(task_dict: Dict[str, Any]) -> Dict[str, Any]:
    return _run_design_impl(task_dict)

@app.function(image=image, gpu="A100-80GB", timeout=7200, volumes={"/cache": cache_volume}, max_containers=20)
def run_design_A100_80GB(task_dict: Dict[str, Any]) -> Dict[str, Any]:
    return _run_design_impl(task_dict)

@app.function(image=image, gpu="H100", timeout=7200, volumes={"/cache": cache_volume}, max_containers=20)
def run_design_H100(task_dict: Dict[str, Any]) -> Dict[str, Any]:
    return _run_design_impl(task_dict)

GPU_FUNCTIONS = {
    "T4": run_design_T4,
    "L4": run_design_L4,
    "A10G": run_design_A10G,
    "L40S": run_design_L40S,
    "A100": run_design_A100_40GB,
    "A100-40GB": run_design_A100_40GB,
    "A100-80GB": run_design_A100_80GB,
    "H100": run_design_H100,
}


# =============================================================================
# CACHE INITIALIZATION
# =============================================================================

@app.function(
    image=image,
    gpu="T4",  # Use cheap GPU for downloads
    timeout=3600,
    volumes={"/cache": cache_volume},
)
def initialize_cache() -> str:
    """Download and cache Boltz model weights, CCD data, and LigandMPNN models."""
    from boltz.main import download_boltz2
    
    cache_dir = Path("/cache/boltz")
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("INITIALIZING PROTEIN HUNTER CACHE")
    print("=" * 60)
    
    # Download Boltz weights
    print("\n1. Downloading Boltz2 weights and CCD data...")
    print("   This may take 10-20 minutes on first run.")
    
    try:
        download_boltz2(cache_dir)
        print("   ✓ Boltz2 downloaded successfully")
    except Exception as e:
        print(f"   ✗ Error downloading Boltz2: {e}")
        return f"Error: {e}"
    
    # Download LigandMPNN weights
    print("\n2. Downloading LigandMPNN model weights...")
    mpnn_dir = Path("/cache/ligandmpnn")
    mpnn_dir.mkdir(parents=True, exist_ok=True)
    
    mpnn_models = [
        ("proteinmpnn_v_48_020.pt", "https://files.ipd.uw.edu/pub/ligandmpnn/proteinmpnn_v_48_020.pt"),
        ("ligandmpnn_v_32_010_25.pt", "https://files.ipd.uw.edu/pub/ligandmpnn/ligandmpnn_v_32_010_25.pt"),
        ("solublempnn_v_48_020.pt", "https://files.ipd.uw.edu/pub/ligandmpnn/solublempnn_v_48_020.pt"),
    ]
    
    for model_name, url in mpnn_models:
        model_path = mpnn_dir / model_name
        if not model_path.exists():
            print(f"   Downloading {model_name}...")
            result = subprocess.run(["wget", "-q", url, "-O", str(model_path)], capture_output=True)
            if result.returncode == 0:
                print(f"   ✓ {model_name}")
            else:
                print(f"   ✗ Failed to download {model_name}")
        else:
            print(f"   ✓ {model_name} (cached)")
    
    # Commit volume
    cache_volume.commit()
    
    # Report cache contents
    print("\n3. Cache contents:")
    for subdir in [cache_dir, mpnn_dir]:
        if subdir.exists():
            files = list(subdir.rglob("*"))
            total_size = sum(f.stat().st_size for f in files if f.is_file())
            print(f"   {subdir}: {len(files)} files, {total_size / 1e9:.2f} GB")
    
    return "Cache initialized successfully!"


# =============================================================================
# LOCAL ENTRYPOINTS
# =============================================================================

@app.local_entrypoint()
def init_cache():
    """
    Initialize the cache (download model weights).
    Run this ONCE before using the pipeline.
    
    Usage:
        modal run modal_protein_hunter.py::init_cache
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
    max_concurrent: int = 0,  # 0 = unlimited, otherwise limit concurrent GPUs
    output_dir: Optional[str] = None,
    no_stream: bool = False,
    sync_interval: float = 5.0,
):
    """
    Run the Protein Hunter design pipeline on Modal.
    
    Examples:
        # Basic protein binder design
        modal run modal_protein_hunter.py::run_pipeline \\
            --name "PDL1_binder" \\
            --target-seq "AFTVTVPKDLYVVEYGSNMTIECKFPVEKQLDLAALIVYWEMEDKNIIQFVHGEEDLKVQHSSYRQRARLLKDQLSLGNAALQITDVKLQDAGVYRCMISYGGADYKRITVKVNAPYAAALE" \\
            --num-designs 5 \\
            --num-cycles 7
        
        # With hotspots
        modal run modal_protein_hunter.py::run_pipeline \\
            --name "PDL1_hotspot" \\
            --target-seq "AFTVTVPK..." \\
            --contact-residues "54,56,115" \\
            --num-designs 3
        
        # Small molecule binder
        modal run modal_protein_hunter.py::run_pipeline \\
            --name "SAM_binder" \\
            --ligand-ccd "SAM" \\
            --num-designs 5
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
    
    # Setup output directory with new naming convention
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
            "template_content": template_content,  # Base64-encoded PDB content
            "template_chain_ids": template_chain_id or "",  # Chain IDs from template file
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
    
    # Execute tasks in parallel (with optional concurrency limit)
    if max_concurrent > 0:
        print(f"Submitting {len(tasks)} design task(s) to Modal (max {max_concurrent} concurrent GPUs)...")
    else:
        print(f"Submitting {len(tasks)} design task(s) to Modal (unlimited concurrency)...")
    
    all_results = []
    completed = 0
    
    if max_concurrent > 0 and max_concurrent < len(tasks):
        # Batched execution: run tasks in chunks to limit concurrency
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
        # Unlimited concurrency: submit all at once
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
    
    # Save results with new flat structure
    print(f"\nSaving results to {output_path}...")

    # Ensure designs folder exists (sync worker may have created it already)
    designs_dir = output_path / "designs"
    designs_dir.mkdir(parents=True, exist_ok=True)

    # Save best designs (best cycle per design run)
    best_dir = output_path / "best_designs"
    best_dir.mkdir(exist_ok=True)
    best_rows = []

    for r in all_results:
        if r.get("best_pdb") and r.get("best_seq"):
            design_idx = r.get("design_idx", 0)
            best_cycle = r.get("best_cycle", 0)
            design_id = f"{name}_d{design_idx}_c{best_cycle}"

            # Save best PDB with new naming
            pdb_file = best_dir / f"{design_id}.pdb"
            pdb_file.write_text(r["best_pdb"])

            # Find best cycle metrics
            best_cycle_data = None
            for cycle_data in r.get("cycles", []):
                if cycle_data.get("cycle") == best_cycle:
                    best_cycle_data = cycle_data
                    break

            # Build row for best_designs.csv
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
            })

    if best_rows:
        best_df = pd.DataFrame(best_rows)
        best_df.to_csv(best_dir / "best_designs.csv", index=False)
        print(f"  ✓ best_designs/ ({len(best_rows)} PDBs + best_designs.csv)")

    # Count designs in designs folder
    design_pdbs = list(designs_dir.glob("*.pdb"))
    design_csv = designs_dir / "design_stats.csv"
    print(f"  ✓ designs/ ({len(design_pdbs)} PDBs + design_stats.csv)")

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
    print(f"\nOutput structure:")
    print(f"  {output_path}/")
    print(f"  ├── designs/           # ALL cycles (PDBs + design_stats.csv)")
    print(f"  └── best_designs/      # Best cycle per design run")
    print(f"\nResults saved to: {output_path}")


def _sync_worker(run_id: str, output_path: Path, stop_event: threading.Event, interval: float):
    """Background worker that polls Modal Dict and saves results locally."""
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
    
    # Final sync
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
    """Save a synced result from Dict to local filesystem with new flat structure."""
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
        # Run metadata
        "timestamp": dt.datetime.fromtimestamp(result.get("timestamp", time.time())).isoformat(),
    }
    append_to_csv_safe(csv_file, row)


@app.local_entrypoint()
def list_gpus():
    """List available GPU types."""
    print("\nAvailable GPU Types:")
    print("=" * 50)
    for gpu, desc in GPU_TYPES.items():
        marker = " (DEFAULT)" if gpu == DEFAULT_GPU else ""
        print(f"  {gpu:15s} - {desc}{marker}")
    print("\nUsage: --gpu H100")


@app.local_entrypoint()
def test_connection(gpu: str = DEFAULT_GPU):
    """Test Modal connection and GPU."""
    print(f"Testing Modal connection with GPU: {gpu}...")
    result = _test_gpu.remote()
    print(f"\n{result}")


@app.function(image=image, gpu="T4", timeout=60)
def _test_gpu() -> str:
    """Test GPU availability."""
    result = subprocess.run(["nvidia-smi"], capture_output=True, text=True)
    return result.stdout


if __name__ == "__main__":
    print("Use 'modal run modal_protein_hunter.py::run_pipeline' to execute")

