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

# Volume for AlphaFold3 weights (user uploads their own)
af3_weights_volume = modal.Volume.from_name("af3-weights", create_if_missing=True)

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
# ALPHAFOLD3 IMAGE (for validation)
# =============================================================================
# Builds AF3 from source at image build time (no local AF3 repo needed)
# Replicates the official Dockerfile: https://github.com/google-deepmind/alphafold3

af3_image = (
    modal.Image.debian_slim(python_version="3.11")
    # Install system dependencies (same pattern as main Boltz image)
    .apt_install(
        "git", "wget", "gcc", "g++", "make", "zlib1g-dev", "zstd",
    )
    # Install HMMER from source with seq_limit patch
    .run_commands(
        # Download HMMER
        "mkdir -p /hmmer_build /hmmer",
        "wget http://eddylab.org/software/hmmer/hmmer-3.4.tar.gz -P /hmmer_build",
        "cd /hmmer_build && echo 'ca70d94fd0cf271bd7063423aabb116d42de533117343a9b27a65c17ff06fbf3 hmmer-3.4.tar.gz' | sha256sum --check",
        "cd /hmmer_build && tar zxf hmmer-3.4.tar.gz && rm hmmer-3.4.tar.gz",
        # Clone AF3 to get the patch file
        "git clone --depth 1 https://github.com/google-deepmind/alphafold3.git /app/alphafold",
        # Apply HMMER patch
        "cp /app/alphafold/docker/jackhmmer_seq_limit.patch /hmmer_build/",
        "cd /hmmer_build && patch -p0 < jackhmmer_seq_limit.patch",
        # Build HMMER
        "cd /hmmer_build/hmmer-3.4 && ./configure --prefix /hmmer",
        "cd /hmmer_build/hmmer-3.4 && make -j$(nproc)",
        "cd /hmmer_build/hmmer-3.4 && make install",
        "cd /hmmer_build/hmmer-3.4/easel && make install",
        "rm -rf /hmmer_build",
    )
    # Install AF3 Python dependencies (versions from AF3 pyproject.toml)
    .pip_install(
        # JAX with CUDA - MUST match AF3 requirements exactly
        "jax==0.4.34",
        "jax[cuda12]==0.4.34",
        "jax-triton==0.2.0",
        "triton==3.1.0",
        # AF3 core dependencies from pyproject.toml
        "absl-py",
        "dm-haiku==0.0.13",
        "dm-tree",
        "jaxtyping==0.2.34",
        "typeguard==2.13.3",
        "numpy",
        "rdkit==2024.3.5",
        "tqdm",
        "zstandard",
        # Additional deps for full functionality
        "ml-collections",
        "pandas",
        "scipy",
        "biopython>=1.83",
        "gemmi>=0.6.3",
    )
    # Install AF3 package
    .run_commands(
        "pip install --no-deps /app/alphafold",
        # Build chemical components database
        "build_data || echo 'build_data may require GPU, skipping for now'",
    )
    # Set environment variables (from official Dockerfile)
    .env({
        "PATH": "/hmmer/bin:$PATH",
        "XLA_FLAGS": "--xla_gpu_enable_triton_gemm=false",
        "XLA_PYTHON_CLIENT_PREALLOCATE": "true",
        "XLA_CLIENT_MEM_FRACTION": "0.95",
    })
    # Add protein-hunter utils
    .add_local_dir("utils", "/root/protein_hunter/utils", copy=True)
)

# =============================================================================
# PYROSETTA IMAGE (for interface analysis)
# =============================================================================
# PyRosetta installed from conda-forge channel (same approach as BindCraft)
# Users are responsible for their own license compliance

pyrosetta_image = (
    modal.Image.from_registry("continuumio/miniconda3:latest")
    .apt_install("libgfortran5")  # Required by DAlphaBall for BUNS calculation
    .run_commands(
        # Install PyRosetta from rosettacommons conda channel (like BindCraft does)
        "conda install -y -c https://conda.rosettacommons.org pyrosetta",
        # Install other dependencies (matching local pipeline)
        # - biopython: CIF->PDB conversion (MMCIFParser + PDBIO)
        # - scipy: KD-tree for hotspot_residues
        "pip install pandas numpy biopython scipy",
    )
    .add_local_dir("utils", "/root/protein_hunter/utils", copy=True)
    .run_commands(
        # Make DAlphaBall executable (needed for BUNS calculation)
        "chmod +x /root/protein_hunter/utils/DAlphaBall.gcc",
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


def get_cif_alignment_json(query_seq: str, cif_path: str, chain_id: Optional[str] = None) -> Dict[str, Any]:
    """
    Build AF3 template JSON with sequence alignment indices.
    
    Phase 5: Template integration for AF3 validation (matching local pipeline).
    
    Args:
        query_seq: The query protein sequence to align against
        cif_path: Path to the template CIF/PDB file
        chain_id: Optional chain ID to extract from template (default: first chain)
    
    Returns:
        Dict with 'mmcif', 'queryIndices', and 'templateIndices' for AF3 JSON
    """
    from Bio.Align import PairwiseAligner
    from Bio.Data import IUPACData
    from Bio.PDB import MMCIFParser, PDBParser
    
    # Determine parser based on file extension
    if cif_path.lower().endswith('.cif'):
        parser = MMCIFParser(QUIET=True)
    else:
        parser = PDBParser(QUIET=True)
    
    structure = parser.get_structure("template", cif_path)
    
    # Get the specified chain or first chain
    if chain_id:
        chain = next((ch for ch in structure.get_chains() if ch.id == chain_id), None)
        if chain is None:
            raise ValueError(f"Chain {chain_id} not found in {cif_path}")
    else:
        chain = next(structure.get_chains())
    
    # Extract template sequence using 3-letter to 1-letter mapping
    three_to_one = IUPACData.protein_letters_3to1
    template_seq = "".join(
        three_to_one.get(residue.resname.capitalize(), "X")
        for residue in chain.get_residues() if residue.id[0] == " "
    )
    
    # Perform global sequence alignment using PairwiseAligner (replaces deprecated pairwise2)
    aligner = PairwiseAligner()
    aligner.mode = 'global'
    aligner.match_score = 1
    aligner.mismatch_score = 0
    aligner.open_gap_score = 0
    aligner.extend_gap_score = 0
    alignments = list(aligner.align(query_seq, template_seq))
    if not alignments:
        raise ValueError(f"Could not align query sequence to template from {cif_path}")
    
    alignment = alignments[0]
    
    # Extract index mappings using alignment.indices
    # indices[0] = query indices (-1 for gaps), indices[1] = target indices (-1 for gaps)
    indices = alignment.indices
    query_indices = []
    template_indices = []
    for col in range(indices.shape[1]):
        q_idx = indices[0, col]
        t_idx = indices[1, col]
        if q_idx >= 0 and t_idx >= 0:
            query_indices.append(int(q_idx))
            template_indices.append(int(t_idx))
    
    # Read mmCIF/PDB text
    with open(cif_path) as f:
        structure_text = f.read()
    
    return {
        "mmcif": structure_text,
        "queryIndices": query_indices,
        "templateIndices": template_indices,
    }


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
) -> tuple:
    """Build the input data dictionary for Boltz prediction.
    
    Args:
        template_content: Base64-encoded PDB/CIF file content (if using template)
        template_chain_ids: Comma-separated chain IDs from template file to use for each target protein
                           e.g., "A,B,C" means use chain A for first target, B for second, C for third
    
    Returns:
        Tuple of (data_dict, target_msas) where target_msas is {chain_id: a3m_content}
    """
    import base64
    from boltz_ph.model_utils import process_msa
    
    sequences = []
    target_msas = {}  # Store MSA content for AF3 reuse
    
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
            # Use cached MSA if same sequence was already processed
            if seq in seq_to_msa_path:
                msa_path = seq_to_msa_path[seq]
                msa_value = str(msa_path)
            else:
                print(f"  Generating MSA for chain {chain_id}...")
                msa_path = process_msa(chain_id, seq, work_dir)
                seq_to_msa_path[seq] = msa_path
                msa_value = str(msa_path)
            
            # Read A3M text file for AF3 reuse (not the .npz binary)
            # The A3M file is stored alongside the NPZ file
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
# SINGLE-DESIGN AF3 VALIDATION FUNCTIONS
# =============================================================================

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
    Run AF3 validation on a SINGLE design.
    
    Phase 5: Now supports template structures for improved AF3 predictions.
    
    Args:
        design_id: Unique identifier for this design
        binder_seq: Binder protein sequence
        target_seq: Target protein sequence
        binder_chain: Chain ID for binder (default "A")
        target_chain: Chain ID for target (default "B")
        target_msa: Optional MSA content for target (A3M format)
        af3_msa_mode: MSA mode ("none", "reuse")
        template_path: Optional path to template structure file (Phase 5)
        template_chain_id: Optional chain ID in template file (Phase 5)
    
    Returns:
        Dict with af3_iptm, af3_ptm, af3_plddt, af3_structure, af3_confidence_json
    """
    import json
    import subprocess
    import tempfile
    from pathlib import Path
    
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
    
    # Phase 5: Process template if provided
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
            "templates": target_template_json,  # Phase 5: Include templates
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


# Per-GPU AF3 functions (same pattern as design)
@app.function(image=af3_image, gpu="H100", timeout=1800, volumes={"/cache": cache_volume, "/af3_weights": af3_weights_volume}, max_containers=20)
def run_af3_single_H100(design_id: str, binder_seq: str, target_seq: str, binder_chain: str = "A", target_chain: str = "B", target_msa: Optional[str] = None, af3_msa_mode: str = "none", template_path: Optional[str] = None, template_chain_id: Optional[str] = None) -> Dict[str, Any]:
    return _run_af3_single_impl(design_id, binder_seq, target_seq, binder_chain, target_chain, target_msa, af3_msa_mode, template_path, template_chain_id)

@app.function(image=af3_image, gpu="A100-80GB", timeout=1800, volumes={"/cache": cache_volume, "/af3_weights": af3_weights_volume}, max_containers=20)
def run_af3_single_A100_80GB(design_id: str, binder_seq: str, target_seq: str, binder_chain: str = "A", target_chain: str = "B", target_msa: Optional[str] = None, af3_msa_mode: str = "none", template_path: Optional[str] = None, template_chain_id: Optional[str] = None) -> Dict[str, Any]:
    return _run_af3_single_impl(design_id, binder_seq, target_seq, binder_chain, target_chain, target_msa, af3_msa_mode, template_path, template_chain_id)

@app.function(image=af3_image, gpu="A100", timeout=1800, volumes={"/cache": cache_volume, "/af3_weights": af3_weights_volume}, max_containers=20)
def run_af3_single_A100_40GB(design_id: str, binder_seq: str, target_seq: str, binder_chain: str = "A", target_chain: str = "B", target_msa: Optional[str] = None, af3_msa_mode: str = "none", template_path: Optional[str] = None, template_chain_id: Optional[str] = None) -> Dict[str, Any]:
    return _run_af3_single_impl(design_id, binder_seq, target_seq, binder_chain, target_chain, target_msa, af3_msa_mode, template_path, template_chain_id)

AF3_GPU_FUNCTIONS = {
    "H100": run_af3_single_H100,
    "A100-80GB": run_af3_single_A100_80GB,
    "A100": run_af3_single_A100_40GB,
    "A100-40GB": run_af3_single_A100_40GB,
}


# =============================================================================
# PHASE 2: AF3 APO PREDICTION (for holo-apo RMSD calculation)
# =============================================================================

def _run_af3_apo_impl(
    design_id: str,
    binder_seq: str,
    binder_chain: str = "A",
) -> Dict[str, Any]:
    """
    Run AF3 on binder ALONE (APO state) for holo-apo RMSD calculation.
    
    This predicts the binder structure in isolation to compare against
    the holo (bound) state for conformational stability analysis.
    
    Returns:
        Dict with apo_structure (CIF text) and any errors
    """
    import json
    import subprocess
    import tempfile
    from pathlib import Path
    
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


# Per-GPU AF3 APO functions
@app.function(image=af3_image, gpu="H100", timeout=1800, volumes={"/cache": cache_volume, "/af3_weights": af3_weights_volume}, max_containers=20)
def run_af3_apo_H100(design_id: str, binder_seq: str, binder_chain: str = "A") -> Dict[str, Any]:
    return _run_af3_apo_impl(design_id, binder_seq, binder_chain)

@app.function(image=af3_image, gpu="A100-80GB", timeout=1800, volumes={"/cache": cache_volume, "/af3_weights": af3_weights_volume}, max_containers=20)
def run_af3_apo_A100_80GB(design_id: str, binder_seq: str, binder_chain: str = "A") -> Dict[str, Any]:
    return _run_af3_apo_impl(design_id, binder_seq, binder_chain)

@app.function(image=af3_image, gpu="A100", timeout=1800, volumes={"/cache": cache_volume, "/af3_weights": af3_weights_volume}, max_containers=20)
def run_af3_apo_A100_40GB(design_id: str, binder_seq: str, binder_chain: str = "A") -> Dict[str, Any]:
    return _run_af3_apo_impl(design_id, binder_seq, binder_chain)

AF3_APO_GPU_FUNCTIONS = {
    "H100": run_af3_apo_H100,
    "A100-80GB": run_af3_apo_A100_80GB,
    "A100": run_af3_apo_A100_40GB,
    "A100-40GB": run_af3_apo_A100_40GB,
}


# =============================================================================
# SINGLE-DESIGN PYROSETTA FUNCTION (Full Implementation - Matching Local Pipeline)
# =============================================================================

@app.function(image=pyrosetta_image, cpu=4, timeout=1800, max_containers=20)
def run_pyrosetta_single(
    design_id: str,
    af3_structure: str,
    af3_iptm: float,
    af3_plddt: float,
    binder_chain: str = "A",
    target_chain: str = "B",
    apo_structure: Optional[str] = None,
    af3_confidence_json: Optional[str] = None,
    target_type: str = "protein",
) -> Dict[str, Any]:
    """
    Run full PyRosetta analysis on a SINGLE design (matching local pipeline).
    
    Includes:
    - CIF→PDB conversion using BioPython (matching local)
    - Multi-chain collapse for >2 chain complexes
    - FastRelax with proper settings
    - InterfaceAnalyzerMover for detailed interface metrics
    - hotspot_residues for interface_nres calculation (matching local)
    - BUNS calculation with DAlphaBall
    - APO-HOLO RMSD calculation (if apo_structure provided)
    - i_pae calculation (if af3_confidence_json provided)
    - Radius of gyration calculation
    - Returns relaxed PDB content
    
    Args:
        design_id: Unique identifier for this design
        af3_structure: CIF text of AF3 holo structure
        af3_iptm: ipTM score from AF3
        af3_plddt: pLDDT score from AF3
        binder_chain: Chain ID of the binder (default "A")
        target_chain: Chain ID of the target (default "B")
        apo_structure: Optional CIF text of AF3 apo (binder-only) structure
        af3_confidence_json: Optional JSON text of AF3 confidence data (for i_pae)
        target_type: Type of target - "protein", "peptide", "small_molecule", or "nucleic"
                     Affects filtering thresholds (peptide uses stricter BUNS, looser nres)
    
    Returns:
        Dict with acceptance status, interface scores, and relaxed_pdb content
    """
    import json
    import tempfile
    import sys
    from pathlib import Path
    
    # Add utils to path for imports
    sys.path.insert(0, "/root/protein_hunter")
    
    result = {
        "design_id": design_id,
        "af3_iptm": float(af3_iptm),
        "af3_plddt": float(af3_plddt),
        "accepted": False,
        "rejection_reason": None,
        "relaxed_pdb": None,
        # Interface metrics (will be populated)
        # Note: interface_interface_hbonds matches local column naming
        "binder_score": 0.0,
        "total_score": 0.0,
        "interface_sc": 0.0,
        "interface_packstat": 0.0,
        "interface_dG": 0.0,
        "interface_dSASA": 0.0,
        "interface_dG_SASA_ratio": 0.0,
        "interface_nres": 0,
        "interface_interface_hbonds": 0,
        "interface_delta_unsat_hbonds": 0,
        "interface_hydrophobicity": 0.0,
        "surface_hydrophobicity": 0.0,
        # Missing metrics from local pipeline (Phase 1)
        "binder_sasa": 0.0,
        "interface_fraction": 0.0,
        "interface_hbond_percentage": 0.0,
        "interface_delta_unsat_hbonds_percentage": 0.0,
        # Secondary quality metrics (Phase 3)
        "apo_holo_rmsd": None,
        "i_pae": None,
        "rg": None,
    }
    
    if not af3_structure:
        result["rejection_reason"] = "No AF3 structure"
        return result
    
    work_dir = Path(tempfile.mkdtemp())
    
    # ========================================
    # HELPER: Multi-chain collapse (Phase 4 - matching local pipeline)
    # ========================================
    def collapse_multiple_chains(pdb_in: str, pdb_out: str, binder_chain: str = "A", collapse_target: str = "B"):
        """
        Collapse all non-binder chains into a single target chain.
        Matching local pyrosetta_utils.py logic.
        """
        with open(pdb_in, "r") as f:
            lines = f.readlines()

        atom_indices = []
        chain_list = []
        for i, line in enumerate(lines):
            if line.startswith(("ATOM  ", "HETATM")):
                atom_indices.append(i)
                chain_list.append(line[21])

        all_chains = sorted(set(chain_list))
        collapse_chains = [c for c in all_chains if c != binder_chain]

        # Detect transitions
        transitions = []
        for (idx1, c1), (idx2, c2) in zip(
                zip(atom_indices, chain_list),
                zip(atom_indices[1:], chain_list[1:])):
            if c1 != c2:
                transitions.append((idx1, c1, c2))

        ter_after = set()
        seen_binder_to_collapsed = False
        for idx, c1, c2 in transitions:
            if c1 == binder_chain and c2 in collapse_chains:
                if not seen_binder_to_collapsed:
                    ter_after.add(idx)
                    seen_binder_to_collapsed = True

        # Find last collapsed atom
        last_collapsed_idx = None
        for i, line in enumerate(lines):
            if line.startswith(("ATOM  ", "HETATM")) and line[21] in collapse_chains:
                last_collapsed_idx = i
        if last_collapsed_idx is not None:
            ter_after.add(last_collapsed_idx)

        # Write output without existing TERs
        temp_out = []
        for i, line in enumerate(lines):
            if line.startswith(("ATOM  ", "HETATM")):
                temp_out.append(line)
                if i in ter_after:
                    temp_out.append("TER\n")
            elif not line.startswith("TER"):
                temp_out.append(line)

        # Collapse chain IDs
        final_out = []
        for line in temp_out:
            if line.startswith(("ATOM  ", "HETATM")) and line[21] in collapse_chains:
                line = line[:21] + collapse_target + line[22:]
            final_out.append(line)

        with open(pdb_out, "w") as f:
            f.writelines(final_out)
    
    try:
        # ========================================
        # CIF to PDB CONVERSION (using BioPython - matching local)
        # ========================================
        from Bio.PDB import MMCIFParser, PDBIO, PDBParser, Selection
        
        # Write CIF structure to temp file
        cif_file = work_dir / f"{design_id}.cif"
        cif_file.write_text(af3_structure)
        
        # Convert CIF to PDB using BioPython (matching local convert.py)
        pdb_file = work_dir / f"{design_id}.pdb"
        try:
            parser = MMCIFParser(QUIET=True)
            structure = parser.get_structure(design_id, str(cif_file))
            
            # Rename long chain IDs to single letters if needed
            next_chain_idx = 0
            def int_to_chain(i):
                if i < 26:
                    return chr(ord("A") + i)
                elif i < 52:
                    return chr(ord("a") + i - 26)
                else:
                    return chr(ord("0") + i - 52)
            
            chainmap = {}
            for chain in structure.get_chains():
                if len(chain.id) != 1:
                    while True:
                        c = int_to_chain(next_chain_idx)
                        if c not in chainmap:
                            chainmap[c] = chain.id
                            chain.id = c
                            break
                        next_chain_idx += 1
            
            # Truncate long residue names
            for model in structure:
                for chain in model:
                    for residue in chain:
                        if len(residue.resname) > 3:
                            residue.resname = residue.resname[:3]
            
            io = PDBIO()
            io.set_structure(structure)
            io.save(str(pdb_file))
            print(f"  Converted CIF to PDB for {design_id}")
        except Exception as e:
            result["rejection_reason"] = f"CIF to PDB conversion failed: {e}"
            return result
        
        # ========================================
        # MULTI-CHAIN COLLAPSE (Phase 4)
        # ========================================
        # Check if we need to collapse chains (>2 chains)
        pdb_parser = PDBParser(QUIET=True)
        temp_structure = pdb_parser.get_structure("check", str(pdb_file))
        total_chains = [chain.id for model in temp_structure for chain in model]
        
        if len(total_chains) > 2:
            collapsed_pdb = work_dir / f"{design_id}_collapsed.pdb"
            collapse_multiple_chains(str(pdb_file), str(collapsed_pdb), binder_chain, "B")
            pdb_file = collapsed_pdb
            print(f"  Collapsed {len(total_chains)} chains to 2 for interface analysis")
        
        # ========================================
        # PYROSETTA INITIALIZATION (with DAlphaBall)
        # ========================================
        import pyrosetta as pr
        from pyrosetta.rosetta.core.kinematics import MoveMap
        from pyrosetta.rosetta.protocols.relax import FastRelax
        from pyrosetta.rosetta.protocols.analysis import InterfaceAnalyzerMover
        from pyrosetta.rosetta.core.select.residue_selector import ChainSelector
        from pyrosetta.rosetta.protocols.simple_moves import AlignChainMover
        from pyrosetta.rosetta.protocols.rosetta_scripts import XmlObjects
        
        # DAlphaBall path (matching local pipeline)
        dalphaball_path = "/root/protein_hunter/utils/DAlphaBall.gcc"
        pr.init(f"-ignore_unrecognized_res -ignore_zero_occupancy -mute all -holes:dalphaball {dalphaball_path} -corrections::beta_nov16 true -relax:default_repeats 1")
        
        # Load pose from PDB
        pose = pr.pose_from_file(str(pdb_file))
        start_pose = pose.clone()
        
        # Get score function
        sfxn = pr.get_fa_scorefxn()
        
        # ========================================
        # FAST RELAX (matching local pipeline)
        # ========================================
        mmf = MoveMap()
        mmf.set_chi(True)
        mmf.set_bb(True)
        mmf.set_jump(False)
        
        fastrelax = FastRelax()
        fastrelax.set_scorefxn(sfxn)
        fastrelax.set_movemap(mmf)
        fastrelax.max_iter(200)
        fastrelax.min_type("lbfgs_armijo_nonmonotone")
        fastrelax.constrain_relax_to_start_coords(True)
        fastrelax.apply(pose)
        
        # Align relaxed structure to original
        align = AlignChainMover()
        align.source_chain(0)
        align.target_chain(0)
        align.pose(start_pose)
        align.apply(pose)
        
        # Copy B factors from start_pose for visualization
        for resid in range(1, pose.total_residue() + 1):
            if pose.residue(resid).is_protein():
                try:
                    bfactor = start_pose.pdb_info().bfactor(resid, 1)
                    for atom_id in range(1, pose.residue(resid).natoms() + 1):
                        pose.pdb_info().bfactor(resid, atom_id, bfactor)
                except Exception:
                    pass
        
        # Save relaxed PDB
        relaxed_pdb_path = work_dir / f"{design_id}_relaxed.pdb"
        pose.dump_pdb(str(relaxed_pdb_path))
        
        # Clean PDB (remove non-standard lines) - matching local clean_pdb()
        with open(relaxed_pdb_path) as f_in:
            relevant_lines = [
                line for line in f_in
                if line.startswith(("ATOM", "HETATM", "MODEL", "TER", "END"))
            ]
        with open(relaxed_pdb_path, "w") as f_out:
            f_out.writelines(relevant_lines)
        
        result["relaxed_pdb"] = relaxed_pdb_path.read_text()
        result["total_score"] = float(sfxn(pose))
        
        # ========================================
        # HOTSPOT RESIDUES (matching local pipeline - inline to avoid import issues)
        # ========================================
        # PDBParser and Selection already imported above
        from scipy.spatial import cKDTree
        import numpy as np
        
        def _hotspot_residues(pdb_path, binder_chain, target_chain, atom_distance_cutoff=4.0):
            """Identify interface residues (matching local pipeline)."""
            aa3to1_map = {
                "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
                "GLU": "E", "GLN": "Q", "GLY": "G", "HIS": "H", "ILE": "I",
                "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
                "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V"
            }
            parser = PDBParser(QUIET=True)
            try:
                structure = parser.get_structure("complex", pdb_path)
            except Exception as e:
                print(f"[ERROR] Could not parse PDB: {e}")
                return {}
            
            model = structure[0]
            if binder_chain not in model:
                print(f"[WARNING] Binder chain '{binder_chain}' not found.")
                return {}
            
            binder_atoms = Selection.unfold_entities(model[binder_chain], "A")
            if len(binder_atoms) == 0:
                return {}
            
            # Target = all non-binder chains
            target_atoms = []
            for chain in model:
                if chain.id != binder_chain:
                    target_atoms.extend(Selection.unfold_entities(chain, "A"))
            
            if len(target_atoms) == 0:
                return {}
            
            # KD-tree for fast contact search
            binder_coords = np.array([a.coord for a in binder_atoms])
            target_coords = np.array([a.coord for a in target_atoms])
            
            binder_tree = cKDTree(binder_coords)
            target_tree = cKDTree(target_coords)
            pairs = binder_tree.query_ball_tree(target_tree, atom_distance_cutoff)
            
            interacting = {}
            for binder_idx, close_list in enumerate(pairs):
                if not close_list:
                    continue
                atom = binder_atoms[binder_idx]
                residue = atom.get_parent()
                resnum = residue.id[1]
                res3 = residue.get_resname().upper()
                aa1 = aa3to1_map.get(res3, "X")
                interacting[resnum] = aa1
            
            return interacting
        
        # Use relaxed PDB for interface analysis
        interface_residues_set = _hotspot_residues(
            str(relaxed_pdb_path), 
            binder_chain=binder_chain, 
            target_chain=target_chain,
            atom_distance_cutoff=4.0
        )
        interface_nres = len(interface_residues_set)
        result["interface_nres"] = int(interface_nres)
        
        # Calculate interface hydrophobicity (matching local)
        HYDROPHOBIC_AA = set("AILMFVWY")
        hydrophobic_count = sum(1 for aa in interface_residues_set.values() if aa in HYDROPHOBIC_AA)
        result["interface_hydrophobicity"] = round((hydrophobic_count / interface_nres) * 100, 2) if interface_nres > 0 else 0.0
        
        # ========================================
        # INTERFACE ANALYSIS (matching local pipeline)
        # ========================================
        interface_string = f"{binder_chain}_{target_chain}"
        
        iam = InterfaceAnalyzerMover()
        iam.set_interface(interface_string)
        iam.set_scorefunction(sfxn)
        iam.set_compute_packstat(True)
        iam.set_compute_interface_energy(True)
        iam.set_calc_dSASA(True)
        iam.set_calc_hbond_sasaE(True)
        iam.set_compute_interface_sc(True)
        iam.set_pack_separated(True)
        iam.apply(pose)
        
        # Get interface scores (convert to native Python types)
        interface_data = iam.get_all_data()
        result["interface_sc"] = round(float(interface_data.sc_value), 3)
        result["interface_interface_hbonds"] = int(interface_data.interface_hbonds)
        result["interface_dG"] = round(float(iam.get_interface_dG()), 2)
        result["interface_dSASA"] = round(float(iam.get_interface_delta_sasa()), 2)
        result["interface_packstat"] = round(float(iam.get_interface_packstat()), 3)
        result["interface_dG_SASA_ratio"] = round(float(interface_data.dG_dSASA_ratio) * 100, 2)
        
        # ========================================
        # BUNS (Buried Unsatisfied Hbonds) - with DAlphaBall
        # ========================================
        try:
            buns_filter = XmlObjects.static_get_filter(
                '<BuriedUnsatHbonds report_all_heavy_atom_unsats="true" scorefxn="scorefxn" ignore_surface_res="false" use_ddG_style="true" dalphaball_sasa="1" probe_radius="1.1" burial_cutoff_apo="0.2" confidence="0" />'
            )
            result["interface_delta_unsat_hbonds"] = int(buns_filter.report_sm(pose))
        except Exception as e:
            print(f"  BUNS calculation failed: {e}")
            result["interface_delta_unsat_hbonds"] = 0
        
        # ========================================
        # BINDER ENERGY AND SASA (matching local pipeline)
        # ========================================
        chain_selector = ChainSelector(binder_chain)
        tem = pr.rosetta.core.simple_metrics.metrics.TotalEnergyMetric()
        tem.set_scorefunction(sfxn)
        tem.set_residue_selector(chain_selector)
        result["binder_score"] = round(float(tem.calculate(pose)), 2)
        
        # Phase 1: Add binder SASA (missing in Modal)
        bsasa = pr.rosetta.core.simple_metrics.metrics.SasaMetric()
        bsasa.set_residue_selector(chain_selector)
        binder_sasa = float(bsasa.calculate(pose))
        result["binder_sasa"] = round(binder_sasa, 2)
        
        # Phase 1: Calculate interface fraction
        result["interface_fraction"] = round(
            (result["interface_dSASA"] / binder_sasa) * 100 if binder_sasa > 0 else 0, 2
        )
        
        # Phase 1: Calculate interface hbond percentage
        result["interface_hbond_percentage"] = round(
            (result["interface_interface_hbonds"] / interface_nres) * 100 if interface_nres > 0 else 0, 2
        )
        
        # Phase 1: Calculate BUNS percentage
        result["interface_delta_unsat_hbonds_percentage"] = round(
            (result["interface_delta_unsat_hbonds"] / interface_nres) * 100 if interface_nres > 0 else 0, 2
        )
        
        # ========================================
        # SURFACE HYDROPHOBICITY (matching local pipeline)
        # ========================================
        binder_pose = None
        try:
            for i in range(1, pose.num_chains() + 1):
                chain_begin = pose.conformation().chain_begin(i)
                chain_id = pose.pdb_info().chain(chain_begin)
                if chain_id == binder_chain:
                    binder_pose = pose.split_by_chain()[i]
                    break
            
            if binder_pose:
                layer_sel = pr.rosetta.core.select.residue_selector.LayerSelector()
                layer_sel.set_layers(pick_core=False, pick_boundary=False, pick_surface=True)
                surface_res = layer_sel.apply(binder_pose)
                
                exp_apol_count = 0
                total_count = 0
                for i in range(1, binder_pose.total_residue() + 1):
                    if surface_res[i]:
                        res = binder_pose.residue(i)
                        if res.is_apolar() or res.name() in ["PHE", "TRP", "TYR"]:
                            exp_apol_count += 1
                        total_count += 1
                
                result["surface_hydrophobicity"] = round(exp_apol_count / total_count, 3) if total_count > 0 else 0.0
        except Exception as e:
            print(f"  Surface hydrophobicity calculation failed: {e}")
            result["surface_hydrophobicity"] = 0.0
        
        # ========================================
        # RADIUS OF GYRATION (Phase 3 - matching local pipeline)
        # ========================================
        try:
            if binder_pose:
                # Get CA coordinates for binder
                ca_coords = []
                for i in range(1, binder_pose.total_residue() + 1):
                    if binder_pose.residue(i).has("CA"):
                        ca_atom = binder_pose.residue(i).xyz("CA")
                        ca_coords.append([ca_atom.x, ca_atom.y, ca_atom.z])
                
                if ca_coords:
                    coords = np.array(ca_coords)
                    centroid = coords.mean(axis=0)
                    rg = np.sqrt(np.mean(np.sum((coords - centroid)**2, axis=1)))
                    result["rg"] = round(float(rg), 2)
        except Exception as e:
            print(f"  Radius of gyration calculation failed: {e}")
            result["rg"] = None
        
        # ========================================
        # i_pae CALCULATION (Phase 3 - from AF3 confidence)
        # ========================================
        if af3_confidence_json:
            try:
                confidence = json.loads(af3_confidence_json)
                pae_matrix = np.array(confidence.get('pae', []))
                
                if len(pae_matrix) > 0:
                    # Get binder length - count residues in binder chain
                    binder_len = 0
                    for resid in range(1, pose.total_residue() + 1):
                        if pose.pdb_info().chain(resid) == binder_chain:
                            binder_len += 1
                    
                    if binder_len > 0 and pae_matrix.shape[0] > binder_len:
                        # i_pae = mean of off-diagonal blocks (binder ↔ target)
                        interface_pae1 = np.mean(pae_matrix[:binder_len, binder_len:])
                        interface_pae2 = np.mean(pae_matrix[binder_len:, :binder_len])
                        result["i_pae"] = round((interface_pae1 + interface_pae2) / 2, 2)
            except Exception as e:
                print(f"  i_pae calculation failed: {e}")
                result["i_pae"] = None
        
        # ========================================
        # APO-HOLO RMSD (Phase 2 - conformational stability)
        # ========================================
        if apo_structure:
            try:
                from scipy.spatial.transform import Rotation
                
                # Write and convert APO structure
                apo_cif_file = work_dir / f"{design_id}_apo.cif"
                apo_cif_file.write_text(apo_structure)
                
                apo_pdb_file = work_dir / f"{design_id}_apo.pdb"
                apo_parser = MMCIFParser(QUIET=True)
                apo_struct = apo_parser.get_structure(f"{design_id}_apo", str(apo_cif_file))
                
                # Rename chains if needed
                for chain in apo_struct.get_chains():
                    if len(chain.id) != 1:
                        chain.id = "A"  # APO is always single chain
                
                apo_io = PDBIO()
                apo_io.set_structure(apo_struct)
                apo_io.save(str(apo_pdb_file))
                
                # Get CA coordinates from holo (binder chain) and apo
                def get_ca_coords_from_pdb(pdb_path, chain_id):
                    pdb_parser = PDBParser(QUIET=True)
                    struct = pdb_parser.get_structure("s", str(pdb_path))
                    coords = []
                    for model in struct:
                        for chain in model:
                            if chain.id == chain_id:
                                for residue in chain:
                                    if 'CA' in residue:
                                        coords.append(residue['CA'].coord)
                    return np.array(coords)
                
                def np_rmsd(xyz1, xyz2):
                    """Kabsch-aligned RMSD (matching local pipeline)"""
                    if len(xyz1) != len(xyz2) or len(xyz1) == 0:
                        return None
                    centroid1 = xyz1.mean(axis=0)
                    centroid2 = xyz2.mean(axis=0)
                    xyz1_centered = xyz1 - centroid1
                    xyz2_centered = xyz2 - centroid2
                    rotation, _ = Rotation.align_vectors(xyz1_centered, xyz2_centered)
                    xyz2_rotated = rotation.apply(xyz2_centered)
                    return float(np.sqrt(np.mean(np.sum((xyz1_centered - xyz2_rotated)**2, axis=1))))
                
                holo_coords = get_ca_coords_from_pdb(relaxed_pdb_path, binder_chain)
                apo_coords = get_ca_coords_from_pdb(apo_pdb_file, "A")  # APO is always chain A
                
                if len(holo_coords) == len(apo_coords) and len(holo_coords) > 0:
                    result["apo_holo_rmsd"] = round(np_rmsd(holo_coords, apo_coords), 2)
                else:
                    print(f"  Warning: Chain length mismatch for RMSD: holo={len(holo_coords)}, apo={len(apo_coords)}")
                    result["apo_holo_rmsd"] = None
            except Exception as e:
                print(f"  APO-HOLO RMSD calculation failed: {e}")
                result["apo_holo_rmsd"] = None
        
        # ========================================
        # ACCEPTANCE CRITERIA (matching local pipeline)
        # ========================================
        rejection_reasons = []
        
        # Determine target-specific thresholds (matching local pyrosetta_utils.py)
        # Peptide targets use stricter BUNS and looser nres thresholds
        if target_type == "peptide":
            nres_threshold = 4
            buns_threshold = 2
        else:  # protein, small_molecule, nucleic
            nres_threshold = 7
            buns_threshold = 4
        
        # Primary interface quality filters (from local measure_rosetta_energy)
        if af3_iptm < 0.7:
            rejection_reasons.append(f"Low AF3 ipTM: {af3_iptm:.3f}")
        
        # Secondary pLDDT filter (from local get_metrics - was MISSING)
        if af3_plddt < 80:
            rejection_reasons.append(f"Low AF3 pLDDT: {af3_plddt:.1f}")
        
        if result["binder_score"] >= 0:
            rejection_reasons.append(f"binder_score >= 0: {result['binder_score']}")
        
        if result["surface_hydrophobicity"] >= 0.35:
            rejection_reasons.append(f"surface_hydrophobicity >= 0.35: {result['surface_hydrophobicity']}")
        
        if result["interface_sc"] <= 0.55:
            rejection_reasons.append(f"interface_sc <= 0.55: {result['interface_sc']}")
        
        if result["interface_packstat"] <= 0:
            rejection_reasons.append(f"interface_packstat <= 0: {result['interface_packstat']}")
        
        if result["interface_dG"] >= 0:
            rejection_reasons.append(f"interface_dG >= 0: {result['interface_dG']}")
        
        if result["interface_dSASA"] <= 1:
            rejection_reasons.append(f"interface_dSASA <= 1: {result['interface_dSASA']}")
        
        if result["interface_dG_SASA_ratio"] >= 0:
            rejection_reasons.append(f"interface_dG_SASA_ratio >= 0: {result['interface_dG_SASA_ratio']}")
        
        # Target-type-specific threshold for interface_nres
        if result["interface_nres"] <= nres_threshold:
            rejection_reasons.append(f"interface_nres <= {nres_threshold}: {result['interface_nres']}")
        
        if result["interface_interface_hbonds"] <= 3:
            rejection_reasons.append(f"interface_interface_hbonds <= 3: {result['interface_interface_hbonds']}")
        
        if result["interface_hbond_percentage"] <= 0:
            rejection_reasons.append(f"interface_hbond_percentage <= 0: {result['interface_hbond_percentage']}")
        
        # Target-type-specific threshold for BUNS
        if result["interface_delta_unsat_hbonds"] >= buns_threshold:
            rejection_reasons.append(f"interface_delta_unsat_hbonds >= {buns_threshold}: {result['interface_delta_unsat_hbonds']}")
        
        # Secondary quality filters (from local get_metrics)
        if result.get("i_pae") is not None and result["i_pae"] >= 15:
            rejection_reasons.append(f"i_pae >= 15: {result['i_pae']}")
        
        if result.get("rg") is not None and result["rg"] >= 17:
            rejection_reasons.append(f"rg >= 17: {result['rg']}")
        
        if result.get("apo_holo_rmsd") is not None and result["apo_holo_rmsd"] >= 3.5:
            rejection_reasons.append(f"apo_holo_rmsd >= 3.5: {result['apo_holo_rmsd']}")
        
        if rejection_reasons:
            result["accepted"] = False
            result["rejection_reason"] = "; ".join(rejection_reasons)
        else:
            result["accepted"] = True
            
    except Exception as e:
        import traceback
        result["rejection_reason"] = f"PyRosetta error: {str(e)[:200]}"
        print(f"PyRosetta error for {design_id}: {traceback.format_exc()}")
    
    return result


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
# ALPHAFOLD3 VALIDATION FUNCTIONS
# =============================================================================

@app.function(
    image=image,  # Use base image for upload
    volumes={"/af3_weights": af3_weights_volume},
    timeout=600,
)
def _upload_af3_weights_impl(weights_bytes: bytes, filename: str) -> str:
    """Save AF3 weights to Modal volume."""
    weights_path = Path("/af3_weights") / filename
    weights_path.parent.mkdir(parents=True, exist_ok=True)
    weights_path.write_bytes(weights_bytes)
    
    print(f"Wrote {len(weights_bytes) / 1e9:.2f} GB to {weights_path}")
    
    af3_weights_volume.commit()
    
    # List contents
    print("\nAF3 weights volume contents:")
    for f in Path("/af3_weights").rglob("*"):
        if f.is_file():
            print(f"  {f}: {f.stat().st_size / 1e9:.2f} GB")
    
    return "AF3 weights uploaded successfully!"


@app.function(
    image=af3_image,
    gpu="A100-80GB",  # AF3 needs significant VRAM
    timeout=7200,
    volumes={
        "/cache": cache_volume,
        "/af3_weights": af3_weights_volume,
    },
)
def run_af3_validation(
    design_csv_content: str,
    design_pdbs: Dict[str, str],
    target_msas: Optional[Dict[str, str]] = None,
    binder_chain: str = "A",
    af3_msa_mode: str = "reuse",
) -> Dict[str, Any]:
    """
    Run AlphaFold3 validation on designed structures.
    
    Args:
        design_csv_content: Contents of best_designs.csv
        design_pdbs: Dict of {filename: pdb_content}
        target_msas: Dict of {chain_id: a3m_content} for target proteins
        binder_chain: Chain ID of the designed binder
        af3_msa_mode: "none" (query-only), "reuse" (from design), or "generate"
    
    Returns:
        Dict of {design_id: {iptm, ptm, plddt, pdb_content}}
    """
    import json
    import pandas as pd
    
    work_dir = Path(tempfile.mkdtemp())
    af_input_dir = work_dir / "af_input"
    af_output_dir = work_dir / "af_output"
    af_input_dir.mkdir()
    af_output_dir.mkdir()
    
    # Check for AF3 weights
    weights_path = Path("/af3_weights/af3.bin")
    if not weights_path.exists():
        return {"error": "AF3 weights not found. Run: modal run modal_protein_hunter.py::upload_af3_weights --weights-path /path/to/af3.bin.zst"}
    
    print(f"AF3 weights found: {weights_path} ({weights_path.stat().st_size / 1e9:.2f} GB)")
    
    # Parse CSV
    df = pd.read_csv(pd.io.common.StringIO(design_csv_content))
    print(f"Validating {len(df)} designs...")
    
    results = {}
    
    for _, row in df.iterrows():
        design_id = row["design_id"]
        print(f"\nProcessing {design_id}...")
        
        # Build AF3 JSON input
        af3_input = {
            "name": design_id,
            "modelSeeds": [1],
            "dialect": "alphafold3",
            "version": 1,
            "sequences": []
        }
        
        # BINDER: Always query-only MSA (hallucinated, no evolutionary data)
        # Empty templates required for inference-only mode
        binder_seq = row["binder_sequence"]
        af3_input["sequences"].append({
            "protein": {
                "id": binder_chain,
                "sequence": binder_seq,
                "unpairedMsa": f">query\n{binder_seq}\n",
                "pairedMsa": f">query\n{binder_seq}\n",
                "templates": [],
            }
        })
        
        # TARGETS: Handle MSA based on mode
        target_seqs_raw = row.get("target_seqs", "")
        if target_seqs_raw and isinstance(target_seqs_raw, str) and target_seqs_raw.strip():
            # Parse target sequences - could be JSON dict or colon-separated
            if target_seqs_raw.startswith("{"):
                target_seqs = json.loads(target_seqs_raw)
            else:
                # Colon-separated format: SEQ1:SEQ2:SEQ3
                seqs = target_seqs_raw.split(":")
                target_seqs = {chr(ord('B') + i): seq for i, seq in enumerate(seqs)}
            
            for chain_id, seq in target_seqs.items():
                target_entry = {
                    "protein": {
                        "id": chain_id,
                        "sequence": seq,
                        "templates": [],  # Required for inference-only mode
                    }
                }
                
                if af3_msa_mode == "none":
                    # Query-only for targets
                    target_entry["protein"]["unpairedMsa"] = f">query\n{seq}\n"
                    target_entry["protein"]["pairedMsa"] = f">query\n{seq}\n"
                elif af3_msa_mode == "reuse" and target_msas and chain_id in target_msas:
                    # Reuse MSAs from design phase
                    target_entry["protein"]["unpairedMsa"] = target_msas[chain_id]
                    target_entry["protein"]["pairedMsa"] = target_msas[chain_id]
                else:
                    # Fallback: query-only
                    target_entry["protein"]["unpairedMsa"] = f">query\n{seq}\n"
                    target_entry["protein"]["pairedMsa"] = f">query\n{seq}\n"
                
                af3_input["sequences"].append(target_entry)
        
        # Add ligand if present (careful: CSV reads can turn empty strings into nan)
        def _is_valid_str(val) -> bool:
            """Check if value is a valid non-empty string (not NaN)."""
            if val is None:
                return False
            if isinstance(val, float):
                import math
                return not math.isnan(val)
            return bool(str(val).strip()) and str(val).strip().lower() != "nan"
        
        ligand_smiles = row.get("ligand_smiles")
        ligand_ccd = row.get("ligand_ccd")
        
        if _is_valid_str(ligand_smiles):
            af3_input["sequences"].append({
                "ligand": {"id": "L", "smiles": str(ligand_smiles)}
            })
        elif _is_valid_str(ligand_ccd):
            af3_input["sequences"].append({
                "ligand": {"id": "L", "ccdCodes": [str(ligand_ccd)]}
            })
        
        # Write JSON
        json_path = af_input_dir / f"{design_id}.json"
        json_path.write_text(json.dumps(af3_input, indent=2))
    
    # Run AF3 inference for each design
    print("\n" + "="*60)
    print("Running AlphaFold3 inference...")
    print("="*60)
    
    # Get all JSON input files
    json_files = list(af_input_dir.glob("*.json"))
    print(f"Processing {len(json_files)} design(s)...")
    
    for json_file in json_files:
        design_id = json_file.stem
        print(f"\n  Running AF3 on {design_id}...")
        
        try:
            # AF3 is installed in system Python, source at /app/alphafold
            # Note: AF3 expects --json_path to be a FILE, not a directory
            result = subprocess.run([
                "python", "/app/alphafold/run_alphafold.py",
                f"--json_path={json_file}",
                "--model_dir=/af3_weights",
                f"--output_dir={af_output_dir}",
                "--run_data_pipeline=false",  # Skip MSA generation - inference only
                "--run_inference=true",
            ], capture_output=True, text=True, timeout=1800, cwd="/app/alphafold")
            
            if result.returncode != 0:
                print(f"    AF3 error for {design_id}: {result.stderr[-500:] if result.stderr else 'unknown'}")
            else:
                print(f"    ✓ {design_id} complete")
        except subprocess.TimeoutExpired:
            print(f"    AF3 timed out for {design_id}")
        except Exception as e:
            print(f"    AF3 error for {design_id}: {e}")
    
    # Collect results
    print("\nCollecting results...")
    for output_subdir in af_output_dir.iterdir():
        if output_subdir.is_dir():
            design_id = output_subdir.name
            
            # AF3 outputs files with job_name prefix: {job_name}_confidences.json, {job_name}_model.cif
            confidence_file = output_subdir / f"{design_id}_confidences.json"
            structure_file = output_subdir / f"{design_id}_model.cif"
            
            # Fallback to old naming if new naming doesn't exist
            if not confidence_file.exists():
                confidence_file = output_subdir / "confidence.json"
            if not structure_file.exists():
                structure_file = output_subdir / "model.cif"
            
            result_entry = {"design_id": design_id}
            
            if confidence_file.exists():
                try:
                    confidence = json.loads(confidence_file.read_text())
                    # Debug: print actual keys in confidence file
                    print(f"  {design_id}: Confidence keys: {list(confidence.keys())[:10]}")
                    
                    # AF3 confidences.json has different structure:
                    # - "atom_chain_ids", "atom_plddts", "contact_probs", "pae", "token_chain_ids", "token_res_ids"
                    # For interface metrics, need to compute from pae or check summary files
                    
                    # Get pLDDT - average of atom_plddts
                    atom_plddts = confidence.get("atom_plddts", [])
                    if atom_plddts:
                        result_entry["af3_plddt"] = sum(atom_plddts) / len(atom_plddts)
                    else:
                        result_entry["af3_plddt"] = 0
                    
                    # Check for ranking/summary file which has ipTM
                    summary_file = output_subdir / f"{design_id}_summary_confidences.json"
                    if summary_file.exists():
                        summary = json.loads(summary_file.read_text())
                        result_entry["af3_iptm"] = summary.get("iptm", summary.get("ptm", 0))
                        result_entry["af3_ptm"] = summary.get("ptm", 0)
                        print(f"  {design_id}: Summary keys: {list(summary.keys())}")
                    else:
                        # Try to get from main confidence file (some versions)
                        result_entry["af3_iptm"] = confidence.get("iptm", confidence.get("ranking_score", 0))
                        result_entry["af3_ptm"] = confidence.get("ptm", 0)
                    
                    print(f"  {design_id}: ipTM={result_entry['af3_iptm']:.3f}, pLDDT={result_entry['af3_plddt']:.1f}")
                except Exception as e:
                    print(f"  {design_id}: Error reading confidence: {e}")
            else:
                # List what files ARE in the directory for debugging
                files_in_dir = list(output_subdir.glob("*"))
                print(f"  {design_id}: No confidence file found. Files: {[f.name for f in files_in_dir[:5]]}")
            
            if structure_file.exists():
                result_entry["af3_structure"] = structure_file.read_text()
            
            results[design_id] = result_entry
    
    print(f"\n✓ AF3 validation complete: {len(results)} structures processed")
    return results


@app.function(
    image=pyrosetta_image,
    cpu=4,  # PyRosetta is CPU-only
    timeout=3600,
)
def run_pyrosetta_filtering(
    af3_results: Dict[str, Any],
    design_csv_content: str,
    binder_chain: str = "A",
) -> Dict[str, Any]:
    """
    Run PyRosetta interface analysis on validated structures.
    
    Returns:
        Dict with accepted/rejected designs and their scores
    """
    import tempfile
    from pathlib import Path
    
    try:
        import pyrosetta
        pyrosetta.init("-mute all")
    except ImportError:
        return {"error": "PyRosetta not available in image."}
    
    import pandas as pd
    
    work_dir = Path(tempfile.mkdtemp())
    
    results = {
        "accepted": [],
        "rejected": [],
    }
    
    df = pd.read_csv(pd.io.common.StringIO(design_csv_content))
    
    for _, row in df.iterrows():
        design_id = row["design_id"]
        
        if design_id not in af3_results:
            continue
        
        af3_result = af3_results[design_id]
        structure_content = af3_result.get("af3_structure")
        
        if not structure_content:
            continue
        
        try:
            # Write structure to temp file
            cif_file = work_dir / f"{design_id}.cif"
            cif_file.write_text(structure_content)
            
            # Load pose
            pose = pyrosetta.pose_from_file(str(cif_file))
            
            # Score
            sfxn = pyrosetta.get_fa_scorefxn()
            total_score = sfxn(pose)
            
            # Interface analysis (simplified)
            # In full implementation, use InterfaceAnalyzerMover
            
            result_entry = {
                "design_id": design_id,
                "total_score": total_score,
                "af3_iptm": af3_result.get("af3_iptm", 0),
                "af3_plddt": af3_result.get("af3_plddt", 0),
            }
            
            # Simple acceptance criteria
            if af3_result.get("af3_iptm", 0) > 0.7:
                results["accepted"].append(result_entry)
            else:
                result_entry["rejection_reason"] = f"Low ipTM: {af3_result.get('af3_iptm', 0):.3f}"
                results["rejected"].append(result_entry)
                
        except Exception as e:
            print(f"Error processing {design_id}: {e}")
            results["rejected"].append({
                "design_id": design_id,
                "rejection_reason": str(e),
            })
    
    print(f"Accepted: {len(results['accepted'])}, Rejected: {len(results['rejected'])}")
    return results


# =============================================================================
# LOCAL ENTRYPOINTS
# =============================================================================

@app.local_entrypoint()
def upload_af3_weights(weights_path: str):
    """
    Upload AlphaFold3 weights to Modal volume.
    
    Usage:
        modal run modal_protein_hunter.py::upload_af3_weights --weights-path ~/AF3/af3.bin.zst
    """
    import subprocess as sp
    import tempfile
    
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
        modal run modal_protein_hunter.py::run_pipeline \\
            --name "PDL1_binder" \\
            --target-seq "AFTVTVPKDLYVVEYGSNMTIECKFPVEKQLDLAALIVYWEMEDKNIIQFVHGEEDLKVQHSSYRQRARLLKDQLSLGNAALQITDVKLQDAGVYRCMISYGGADYKRITVKVNAPYAAALE" \\
            --num-designs 5 \\
            --num-cycles 7
        
        # With AF3 validation and PyRosetta filtering
        modal run modal_protein_hunter.py::run_pipeline \\
            --name "PDL1_validated" \\
            --target-seq "AFTVTVPK..." \\
            --num-designs 3 \\
            --enable-af3-validation \\
            --enable-pyrosetta
        
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
    
    AF3 Validation:
        First upload weights: modal run modal_protein_hunter.py::upload_af3_weights --weights-path ~/AF3/af3.bin.zst
        Then use --enable-af3-validation flag
        
        MSA modes:
            --af3-msa-mode=none     Query-only for all chains (fast, less accurate)
            --af3-msa-mode=reuse    Reuse MSAs from design phase (recommended)
            --af3-msa-mode=generate Generate fresh MSAs for targets (slow, most accurate)
    
    PyRosetta Filtering:
        Use --enable-pyrosetta flag to run interface analysis on AF3 results.
        PyRosetta is installed from conda-forge. Users are responsible for license compliance.
        
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
                # Fields for AF3 reconstruction
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
        # Exclude internal columns from CSV output (MSAs are for internal AF3 use only)
        csv_columns = [c for c in best_df.columns if not c.startswith("_")]
        best_df[csv_columns].to_csv(best_dir / "best_designs.csv", index=False)
        print(f"  ✓ best_designs/ ({len(best_rows)} PDBs + best_designs.csv)")

    # Count designs in designs folder
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
    # OPTIONAL: AF3 VALIDATION (PARALLELIZED)
    # ===========================================
    if enable_af3_validation and best_rows:
        print(f"\n{'='*70}")
        print("ALPHAFOLD3 VALIDATION (Parallel)")
        print(f"{'='*70}")
        
        # Select AF3 GPU function (use same GPU type as design, or A100-80GB as fallback)
        af3_gpu_to_use = gpu if gpu in AF3_GPU_FUNCTIONS else "A100-80GB"
        af3_fn = AF3_GPU_FUNCTIONS.get(af3_gpu_to_use, run_af3_single_A100_80GB)
        
        # Build AF3 tasks from best_rows
        af3_tasks = []
        for row in best_rows:
            # Get target MSA for chain B (first target protein)
            target_msas = row.get("_target_msas", {})
            target_msa = target_msas.get("B")  # Chain B is the first target
            
            if target_msa and af3_msa_mode == "reuse":
                print(f"  {row['design_id']}: Using MSA ({len(target_msa)} chars)")
            elif af3_msa_mode == "reuse":
                print(f"  {row['design_id']}: No MSA available, using query-only")
            
            # Phase 5: Get template information if available
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
                # Phase 5: Template information
                "template_path": row_template_path if row_template_path else None,
                "template_chain_id": row_template_chain if row_template_chain else None,
            })
        
        print(f"Submitting {len(af3_tasks)} designs for AF3 validation...")
        print(f"GPU: {af3_gpu_to_use} (parallel, max {max_concurrent if max_concurrent > 0 else 'unlimited'} concurrent)")
        print(f"MSA mode: {af3_msa_mode}")
        if template_path:
            print(f"Template: {template_path} (chain {template_chain_id})")
        
        try:
            # Run AF3 in parallel using .starmap() - Phase 5: now includes template params
            af3_results_list = list(af3_fn.starmap([
                (t["design_id"], t["binder_seq"], t["target_seq"], t["binder_chain"], t["target_chain"], 
                 t["target_msa"], t["af3_msa_mode"], t["template_path"], t["template_chain_id"])
                for t in af3_tasks
            ]))
            
            # Convert list to dict for compatibility
            af3_results = {r["design_id"]: r for r in af3_results_list}
            
            # Save AF3 results
            af3_dir = output_path / "af3_validation"
            af3_dir.mkdir(exist_ok=True)
            
            # Save AF3 structures
            for design_id, result in af3_results.items():
                if result.get("af3_structure"):
                    cif_file = af3_dir / f"{design_id}_af3.cif"
                    cif_file.write_text(result["af3_structure"])
            
            # Save AF3 metrics to CSV
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
            # PHASE 2: AF3 APO PREDICTIONS (for holo-apo RMSD)
            # ===========================================
            apo_results = {}
            if enable_pyrosetta:
                print(f"\n{'='*70}")
                print("ALPHAFOLD3 APO PREDICTIONS (for conformational stability)")
                print(f"{'='*70}")
                
                # Select APO GPU function
                af3_apo_fn = AF3_APO_GPU_FUNCTIONS.get(af3_gpu_to_use, run_af3_apo_A100_80GB)
                
                # Build APO tasks - one per design with valid holo structure
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
                    
                    # Count successful APO predictions
                    apo_success = sum(1 for r in apo_results.values() if r.get("apo_structure"))
                    print(f"\n✓ APO predictions complete: {apo_success}/{len(apo_tasks)} structures")
                except Exception as e:
                    print(f"⚠ APO predictions error: {e}")
            
            # ===========================================
            # OPTIONAL: PYROSETTA FILTERING (PARALLELIZED)
            # ===========================================
            if enable_pyrosetta:
                print(f"\n{'='*70}")
                print("PYROSETTA FILTERING (Parallel)")
                print(f"{'='*70}")
                
                # Build PyRosetta tasks from AF3 results, including APO structures and confidence JSON
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
                            # Phase 2: Add APO structure for holo-apo RMSD
                            "apo_structure": apo_results.get(design_id, {}).get("apo_structure"),
                            # Phase 3: Add confidence JSON for i_pae calculation
                            "af3_confidence_json": result.get("af3_confidence_json"),
                            # Target type for filtering thresholds (peptide vs protein)
                            "target_type": target_type,
                        })
                
                print(f"Submitting {len(pr_tasks)} structures for PyRosetta analysis...")
                print("  - FastRelax + InterfaceAnalyzer + APO-HOLO RMSD (may take ~5-10 min per design)")
                
                try:
                    # Run PyRosetta in parallel using .starmap() with new parameters
                    pr_results_list = list(run_pyrosetta_single.starmap([
                        (t["design_id"], t["af3_structure"], t["af3_iptm"], t["af3_plddt"], 
                         t["binder_chain"], t["target_chain"], t["apo_structure"], t["af3_confidence_json"],
                         t["target_type"])
                        for t in pr_tasks
                    ]))
                    
                    # Separate accepted and rejected
                    accepted = [r for r in pr_results_list if r.get("accepted")]
                    rejected = [r for r in pr_results_list if not r.get("accepted")]
                    
                    for r in pr_results_list:
                        status = "✓" if r.get("accepted") else "✗"
                        reason = f" ({r.get('rejection_reason')})" if r.get("rejection_reason") else ""
                        # Show key interface metrics including new ones
                        rmsd_str = f", RMSD={r.get('apo_holo_rmsd', 'N/A')}" if r.get('apo_holo_rmsd') is not None else ""
                        rg_str = f", rg={r.get('rg', 'N/A')}" if r.get('rg') is not None else ""
                        ipae_str = f", iPAE={r.get('i_pae', 'N/A')}" if r.get('i_pae') is not None else ""
                        metrics = f"dG={r.get('interface_dG', 0):.1f}, SC={r.get('interface_sc', 0):.2f}, nres={r.get('interface_nres', 0)}{rmsd_str}{rg_str}{ipae_str}"
                        print(f"  {status} {r['design_id']}: {metrics}{reason}")
                    
                    # Save accepted/rejected designs with RELAXED structures
                    accepted_dir = output_path / "accepted_designs"
                    rejected_dir = output_path / "rejected"
                    accepted_dir.mkdir(exist_ok=True)
                    rejected_dir.mkdir(exist_ok=True)
                    
                    # Columns to exclude from CSV (don't save large text fields)
                    csv_exclude = {"relaxed_pdb", "_target_msas", "af3_confidence_json"}
                    
                    if accepted:
                        # Save CSV (without relaxed_pdb column)
                        accepted_csv_data = [
                            {k: v for k, v in r.items() if k not in csv_exclude}
                            for r in accepted
                        ]
                        accepted_df = pd.DataFrame(accepted_csv_data)
                        accepted_df.to_csv(accepted_dir / "accepted_stats.csv", index=False)
                        
                        # Save RELAXED PDB structures (not AF3 CIF)
                        for entry in accepted:
                            design_id = entry["design_id"]
                            relaxed_pdb = entry.get("relaxed_pdb")
                            if relaxed_pdb:
                                pdb_path = accepted_dir / f"{design_id}_relaxed.pdb"
                                pdb_path.write_text(relaxed_pdb)
                            else:
                                # Fallback to AF3 CIF if no relaxed structure
                                src_cif = af3_dir / f"{design_id}_af3.cif"
                                if src_cif.exists():
                                    shutil.copy(src_cif, accepted_dir / f"{design_id}_af3.cif")
                        
                        print(f"  ✓ Saved {len(accepted)} relaxed PDBs to accepted_designs/")
                    
                    if rejected:
                        # Save CSV (without relaxed_pdb column)
                        rejected_csv_data = [
                            {k: v for k, v in r.items() if k not in csv_exclude}
                            for r in rejected
                        ]
                        rejected_df = pd.DataFrame(rejected_csv_data)
                        rejected_df.to_csv(rejected_dir / "rejected_stats.csv", index=False)
                        
                        # Save relaxed structures for rejected designs too
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
                    
                    # Print summary of interface metrics for accepted designs
                    if accepted:
                        print("\n  Interface Metrics (accepted):")
                        for r in accepted:
                            print(f"    {r['design_id']}:")
                            print(f"      interface_dG: {r.get('interface_dG', 0):.2f}")
                            print(f"      interface_sc: {r.get('interface_sc', 0):.3f}")
                            print(f"      interface_dSASA: {r.get('interface_dSASA', 0):.2f}")
                            print(f"      interface_nres: {r.get('interface_nres', 0)}")
                            # Phase 1: New metrics
                            print(f"      binder_sasa: {r.get('binder_sasa', 0):.2f}")
                            print(f"      interface_fraction: {r.get('interface_fraction', 0):.2f}%")
                            print(f"      interface_hbond_pct: {r.get('interface_hbond_percentage', 0):.2f}%")
                            # Phase 3: Secondary quality metrics
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


@app.function(image=af3_image, gpu="A100-80GB", timeout=300, volumes={"/af3_weights": af3_weights_volume})
def test_af3_image() -> str:
    """Test that the AF3 image is correctly configured."""
    import subprocess
    results = []
    
    # Test 1: Check Python version
    results.append("=== Python Version ===")
    result = subprocess.run(["python", "--version"], capture_output=True, text=True)
    results.append(result.stdout + result.stderr)
    
    # Test 2: Check if AF3 can be imported
    results.append("\n=== AF3 Import Test ===")
    try:
        import alphafold3
        results.append(f"✓ alphafold3 imported successfully (version: {getattr(alphafold3, '__version__', 'unknown')})")
    except Exception as e:
        results.append(f"✗ Failed to import alphafold3: {e}")
    
    # Test 3: Check jaxtyping
    results.append("\n=== JAX/jaxtyping Test ===")
    try:
        import jax
        import jaxtyping  # noqa: F401
        results.append(f"✓ jax version: {jax.__version__}")
        results.append("✓ jaxtyping imported successfully")
    except Exception as e:
        results.append(f"✗ JAX import error: {e}")
    
    # Test 4: Check GPU
    results.append("\n=== GPU Test ===")
    try:
        import jax
        devices = jax.devices()
        results.append(f"✓ JAX devices: {devices}")
    except Exception as e:
        results.append(f"✗ GPU test failed: {e}")
    
    # Test 5: Check HMMER
    results.append("\n=== HMMER Test ===")
    result = subprocess.run(["jackhmmer", "-h"], capture_output=True, text=True)
    if result.returncode == 0:
        results.append("✓ jackhmmer available")
    else:
        results.append(f"✗ jackhmmer not found: {result.stderr}")
    
    # Test 6: Check AF3 weights volume
    results.append("\n=== AF3 Weights ===")
    import os
    weights_path = "/af3_weights/af3.bin"
    if os.path.exists(weights_path):
        size_gb = os.path.getsize(weights_path) / (1024**3)
        results.append(f"✓ Weights found: {weights_path} ({size_gb:.2f} GB)")
    else:
        results.append(f"✗ Weights not found at {weights_path}")
    
    # Test 7: Try importing the AF3 run script
    results.append("\n=== AF3 Run Script ===")
    try:
        from alphafold3.jax.attention import attention  # noqa: F401
        results.append("✓ alphafold3.jax.attention imported successfully")
    except Exception as e:
        results.append(f"✗ Failed: {e}")
    
    return "\n".join(results)


@app.local_entrypoint()
def test_af3():
    """Test that the AF3 image is correctly configured."""
    print("Testing AF3 image configuration...")
    print("=" * 60)
    result = test_af3_image.remote()
    print(result)
    print("=" * 60)


if __name__ == "__main__":
    print("Use 'modal run modal_protein_hunter.py::run_pipeline' to execute")

