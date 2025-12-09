"""
Core design implementation for Boltz structure prediction.

This module contains the main design loop that runs inside Modal containers,
including the _run_design_impl function and GPU-specific wrappers.
"""

import gc
import os
import random
import shutil
import sys
import tempfile
import time
import warnings
from pathlib import Path
from typing import Any, Dict

from modal_boltz_ph.app import app, cache_volume, results_dict
from modal_boltz_ph.images import image


def _run_design_impl(task_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Run a single design trajectory (all cycles) for one binder.
    
    This is the core implementation that runs inside Modal containers.
    It handles:
    - Loading Boltz model and CCD library
    - Building input data from task parameters
    - Running cycle 0 with contact filtering
    - Running optimization cycles with LigandMPNN
    - Streaming results to Modal Dict
    
    Args:
        task_dict: Dictionary containing all design parameters including:
            - run_id, design_idx, total_designs
            - protein_seqs, ligand_ccd, ligand_smiles, etc.
            - num_cycles, temperature, contact_residues
            - Model parameters (diffuse_steps, recycling_steps, etc.)
    
    Returns:
        Dictionary with status, cycles data, best_iptm, best_seq, best_pdb
    """
    import numpy as np
    import torch
    
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
    
    # Import stream function (defined in sync.py but needs results_dict from container)
    from modal_boltz_ph.sync import _stream_result
    
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
    precomputed_msas = task_dict.get("precomputed_msas", {})  # Pre-computed MSAs (seq -> a3m_content)
    
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
    _high_iptm_threshold = task_dict.get("high_iptm_threshold", 0.7)  # Reserved for future use
    _high_plddt_threshold = task_dict.get("high_plddt_threshold", 0.7)  # Reserved for future use
    
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
        # Suppress Lightning checkpoint version warning (benign minor version mismatch)
        warnings.filterwarnings(
            "ignore",
            message="The loaded checkpoint was produced with Lightning",
            category=UserWarning,
            module="pytorch_lightning"
        )
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
        
        # Build input data (returns data dict + MSA content for AF3 reuse)
        print("Building input data...")
        data, target_msas = _build_input_data(
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
            precomputed_msas=precomputed_msas,
        )
        
        # Store MSAs in result for AF3 validation
        result["target_msas"] = target_msas
        
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
                # AF3 reconstruction fields
                ligand_smiles=ligand_smiles,
                ligand_ccd=ligand_ccd,
                nucleic_seq=nucleic_seq,
                nucleic_type=nucleic_type,
                template_path="",  # Template content is passed separately
                template_mapping=template_chain_ids or "",
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
                        # AF3 reconstruction fields
                        ligand_smiles=ligand_smiles,
                        ligand_ccd=ligand_ccd,
                        nucleic_seq=nucleic_seq,
                        nucleic_type=nucleic_type,
                        template_path="",
                        template_mapping=template_chain_ids or "",
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
    precomputed_msas: dict = None,
) -> tuple:
    """
    Build the input data dictionary for Boltz prediction.
    
    Args:
        protein_seqs: Colon-separated target protein sequences
        ligand_ccd: CCD code for ligand (optional)
        ligand_smiles: SMILES string for ligand (optional)
        nucleic_seq: Nucleic acid sequence (optional)
        nucleic_type: Type of nucleic acid (dna/rna)
        msa_mode: MSA mode ("empty" or "mmseqs")
        work_dir: Working directory for temporary files
        cyclic: Whether binder is cyclic
        contact_residues: Contact residue specification
        template_content: Base64-encoded PDB/CIF file content (if using template)
        template_chain_ids: Comma-separated chain IDs from template file
        precomputed_msas: Dict mapping sequence -> a3m_content (pre-computed MSAs to avoid re-fetching)
    
    Returns:
        Tuple of (data_dict, target_msas) where target_msas is {chain_id: a3m_content}
    """
    import base64
    from boltz_ph.model_utils import process_msa
    from boltz.data.parse.a3m import parse_a3m
    
    sequences = []
    target_msas = {}  # Store MSA content for AF3 reuse
    precomputed_msas = precomputed_msas or {}
    
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
        
        chain_id = protein_chain_ids[i]
        
        # Determine MSA value based on mode
        if msa_mode == "mmseqs":
            # Check for pre-computed MSA first (avoids hitting ColabFold API)
            if seq in precomputed_msas:
                print(f"  Using pre-computed MSA for chain {chain_id}")
                msa_content = precomputed_msas[seq]
                target_msas[chain_id] = msa_content
                
                # Write A3M to work_dir and convert to NPZ for Boltz
                msa_chain_dir = work_dir / f"{chain_id}_env"
                msa_chain_dir.mkdir(exist_ok=True)
                msa_a3m_path = msa_chain_dir / "msa.a3m"
                msa_a3m_path.write_text(msa_content)
                
                # Convert to NPZ format
                msa_npz_path = msa_chain_dir / "msa.npz"
                if not msa_npz_path.exists():
                    msa = parse_a3m(msa_a3m_path, taxonomy=None, max_seqs=4096)
                    msa.dump(msa_npz_path)
                
                msa_value = str(msa_npz_path)
                print(f"    MSA for chain {chain_id}: {len(msa_content)} chars, {msa_content.count('>')} sequences")
            # Use cached MSA if same sequence was already processed in this run
            elif seq in seq_to_msa_path:
                msa_path = seq_to_msa_path[seq]
                msa_value = str(msa_path)
            else:
                # Fetch from ColabFold API
                print(f"  Generating MSA for chain {chain_id}...")
                msa_path = process_msa(chain_id, seq, work_dir)
                seq_to_msa_path[seq] = msa_path
                msa_value = str(msa_path)
                
                # Read A3M text file for AF3 reuse (not the .npz binary)
                msa_npz = Path(msa_path)
                msa_a3m = msa_npz.parent / "msa.a3m"
                if msa_a3m.exists():
                    msa_content = msa_a3m.read_text()
                    target_msas[chain_id] = msa_content
                    print(f"    MSA for chain {chain_id}: {len(msa_content)} chars, {msa_content.count('>')} sequences")
                else:
                    print(f"    Warning: A3M file not found at {msa_a3m}")
        else:
            msa_value = "empty"
        
        sequences.append({
            "protein": {
                "id": [chain_id],
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
    
    return data, target_msas


# =============================================================================
# GPU-SPECIFIC MODAL FUNCTIONS
# =============================================================================

@app.function(image=image, gpu="T4", timeout=7200, volumes={"/cache": cache_volume}, max_containers=20)
def run_design_T4(task_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Run design on T4 GPU (16GB VRAM)."""
    return _run_design_impl(task_dict)

@app.function(image=image, gpu="L4", timeout=7200, volumes={"/cache": cache_volume}, max_containers=20)
def run_design_L4(task_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Run design on L4 GPU (24GB VRAM)."""
    return _run_design_impl(task_dict)

@app.function(image=image, gpu="A10G", timeout=7200, volumes={"/cache": cache_volume}, max_containers=20)
def run_design_A10G(task_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Run design on A10G GPU (24GB VRAM)."""
    return _run_design_impl(task_dict)

@app.function(image=image, gpu="L40S", timeout=7200, volumes={"/cache": cache_volume}, max_containers=20)
def run_design_L40S(task_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Run design on L40S GPU (48GB VRAM)."""
    return _run_design_impl(task_dict)

@app.function(image=image, gpu="A100", timeout=7200, volumes={"/cache": cache_volume}, max_containers=20)
def run_design_A100_40GB(task_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Run design on A100 40GB GPU."""
    return _run_design_impl(task_dict)

@app.function(image=image, gpu="A100-80GB", timeout=7200, volumes={"/cache": cache_volume}, max_containers=20)
def run_design_A100_80GB(task_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Run design on A100 80GB GPU."""
    return _run_design_impl(task_dict)

@app.function(image=image, gpu="H100", timeout=7200, volumes={"/cache": cache_volume}, max_containers=20)
def run_design_H100(task_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Run design on H100 GPU (80GB VRAM) - RECOMMENDED."""
    return _run_design_impl(task_dict)


# GPU function mapping for dynamic selection
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

