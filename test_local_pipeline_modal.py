#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════
TEST HARNESS FOR LOCAL boltz_ph PIPELINE
═══════════════════════════════════════════════════════════════════════════════

This runs the LOCAL boltz_ph/pipeline.py code on Modal GPUs for testing.
Use this to verify the HPC/cluster-compatible code works correctly.

This is NOT the production Modal pipeline. For production runs with 
parallelization and streaming, use:
    modal run modal_boltz_ph_cli.py::run_pipeline

This harness is equivalent to running on an HPC cluster:
    python boltz_ph/design.py --name test --protein_seqs "..."

Usage:
    # Basic test (Boltz design only)
    modal run test_local_pipeline_modal.py --protein-seqs "YOUR_TARGET"
    
    # Full pipeline with AF3 + PyRosetta (protein targets only)
    modal run test_local_pipeline_modal.py --protein-seqs "YOUR_TARGET" --use-alphafold3-validation
    
    # Custom parameters
    modal run test_local_pipeline_modal.py --name my_test --num-designs 2 --num-cycles 3
"""

import modal
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional

# =============================================================================
# APP AND VOLUMES
# =============================================================================
# Use the SAME app name and volume names as modal_boltz_ph.
# This allows calling modal_boltz_ph AF3 functions via lazy import in local entrypoint.
# The app/volume definitions must be at module level for Modal to find them,
# but must NOT import from modal_boltz_ph (which isn't installed in containers).

app = modal.App("protein-hunter-boltz")  # Same name as modal_boltz_ph.app

# Shared volumes (same names as modal_boltz_ph)
cache_volume = modal.Volume.from_name("protein-hunter-cache", create_if_missing=True)
af3_weights_volume = modal.Volume.from_name("af3-weights", create_if_missing=True)

# =============================================================================
# IMAGES
# =============================================================================

# Boltz design image
boltz_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "wget", "build-essential")
    .pip_install(
        "torch>=2.2",
        "numpy>=1.24,<2.0",
        "pandas>=2.0",
        "pyyaml>=6.0",
        "biopython>=1.83",
        "gemmi>=0.6.3",
        "prody>=2.4",
        "matplotlib>=3.7",
        "rdkit>=2024.3.1",
        "ml-collections>=0.1.1",
        "dm-tree>=0.1.8",
        "einops>=0.7",
        "scipy>=1.12",
        "py3Dmol",
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
    .add_local_dir("boltz_ph", "/root/protein_hunter/boltz_ph", copy=True)
    .add_local_dir("LigandMPNN", "/root/protein_hunter/LigandMPNN", copy=True)
    .add_local_dir("utils", "/root/protein_hunter/utils", copy=True)
    .run_commands(
        "cd /root/protein_hunter/boltz_ph && pip install -e .",
        "pip install cuequivariance-torch || pip install cuequivariance_torch || echo 'cuequivariance not available'",
    )
)

# AlphaFold3 image for validation (self-contained, same as modal_boltz_ph/images.py)
af3_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install(
        "git", "wget", "gcc", "g++", "make", "zlib1g-dev", "zstd",
    )
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
    .pip_install(
        "jax==0.4.34",
        "jax[cuda12]==0.4.34",
        "jax-triton==0.2.0",
        "triton==3.1.0",
        "absl-py",
        "dm-haiku==0.0.13",
        "dm-tree",
        "jaxtyping==0.2.34",
        "typeguard==2.13.3",
        "numpy",
        "rdkit==2024.3.5",
        "tqdm",
        "zstandard",
        "ml-collections",
        "pandas",
        "scipy",
        "biopython>=1.83",
        "gemmi>=0.6.3",
    )
    .run_commands(
        "pip install --no-deps /app/alphafold",
        "build_data || echo 'build_data may require GPU, skipping for now'",
    )
    .env({
        "PATH": "/hmmer/bin:$PATH",
        "XLA_FLAGS": "--xla_gpu_enable_triton_gemm=false",
        "XLA_PYTHON_CLIENT_PREALLOCATE": "true",
        "XLA_CLIENT_MEM_FRACTION": "0.95",
    })
    .add_local_dir("utils", "/root/protein_hunter/utils", copy=True)
)

# PyRosetta image for interface scoring
pyrosetta_image = (
    modal.Image.from_registry("continuumio/miniconda3:latest")
    .apt_install("libgfortran5")
    .run_commands(
        "conda install -y -c https://conda.rosettacommons.org pyrosetta",
        # Install standard deps from PyPI, torch CPU-only from extra index
        "pip install pandas numpy biopython scipy",
        "pip install torch --extra-index-url https://download.pytorch.org/whl/cpu",
    )
    .add_local_dir("utils", "/root/protein_hunter/utils", copy=True)
    .add_local_dir("boltz_ph", "/root/protein_hunter/boltz_ph", copy=True)
    .run_commands(
        "chmod +x /root/protein_hunter/utils/DAlphaBall.gcc",
    )
)

# =============================================================================
# STAGE 1: LOCAL BOLTZ PIPELINE
# =============================================================================

# GPU type mapping for dispatch
GPU_FUNCTIONS = {}


def _run_local_pipeline_impl(
    # Core settings
    name: str,
    num_designs: int = 1,
    num_cycles: int = 5,
    # Target specification
    protein_seqs: str = "",
    ligand_smiles: str = "",
    ligand_ccd: str = "",
    nucleic_seq: str = "",
    nucleic_type: str = "dna",
    # Template settings
    template_path: str = "",
    template_cif_chain_id: str = "",
    msa_mode: str = "mmseqs",
    # Binder sequence settings
    seq: str = "",
    min_protein_length: int = 100,
    max_protein_length: int = 150,
    percent_X: int = 90,
    cyclic: bool = False,
    exclude_P: bool = False,
    # Hotspot/contact settings
    contact_residues: str = "",
    contact_cutoff: float = 15.0,
    max_contact_filter_retries: int = 6,
    no_contact_filter: bool = False,
    # Sequence design (MPNN) settings
    temperature: float = 0.1,
    omit_AA: str = "C",
    alanine_bias: bool = False,
    alanine_bias_start: float = -0.5,
    alanine_bias_end: float = -0.1,
    # Quality thresholds
    high_iptm_threshold: float = 0.8,
    high_plddt_threshold: float = 0.8,
    # Model settings
    diffuse_steps: int = 200,
    recycling_steps: int = 3,
) -> dict:
    """Run the local Boltz pipeline (boltz_ph/pipeline.py) on Modal."""
    import argparse

    sys.path.insert(0, "/root/protein_hunter")
    os.chdir("/root/protein_hunter")

    # Symlink model weights from cache
    cache_dir = Path("/cache/boltz")
    mpnn_cache = Path("/cache/ligandmpnn")
    mpnn_dest = Path("/root/protein_hunter/LigandMPNN/model_params")

    if mpnn_cache.exists() and not mpnn_dest.exists():
        mpnn_dest.mkdir(parents=True, exist_ok=True)
        for model_file in mpnn_cache.glob("*.pt"):
            dest_file = mpnn_dest / model_file.name
            if not dest_file.exists():
                os.symlink(model_file, dest_file)
        print("Symlinked LigandMPNN models from cache")

    from boltz_ph.pipeline import ProteinHunter_Boltz

    args = argparse.Namespace(
        # Core
        name=name,
        save_dir=f"/root/protein_hunter/results_{name}",
        num_designs=num_designs,
        num_cycles=num_cycles,
        mode="conditional",
        # Target
        protein_seqs=protein_seqs,
        ligand_smiles=ligand_smiles,
        ligand_ccd=ligand_ccd,
        nucleic_seq=nucleic_seq,
        nucleic_type=nucleic_type,
        # Template
        template_path=template_path,
        template_cif_chain_id=template_cif_chain_id,
        template=template_path,  # Alias for CSV logging
        template_mapping=template_cif_chain_id,  # Alias for CSV logging
        msa_mode=msa_mode,
        # Binder
        seq=seq,
        min_protein_length=min_protein_length,
        max_protein_length=max_protein_length,
        percent_X=percent_X,
        cyclic=cyclic,
        exclude_P=exclude_P,
        # Hotspot
        contact_residues=contact_residues,
        contact_cutoff=contact_cutoff,
        max_contact_filter_retries=max_contact_filter_retries,
        no_contact_filter=no_contact_filter,
        # MPNN
        temperature=temperature,
        omit_AA=omit_AA,
        alanine_bias=alanine_bias,
        alanine_bias_start=alanine_bias_start,
        alanine_bias_end=alanine_bias_end,
        # Quality
        high_iptm_threshold=high_iptm_threshold,
        high_plddt_threshold=high_plddt_threshold,
        # Model
        gpu_id=0,
        ccd_path=str(cache_dir / "mols"),
        boltz_model_path=str(cache_dir / "boltz2_conf.ckpt"),
        boltz_model_version="boltz2",
        diffuse_steps=diffuse_steps,
        recycling_steps=recycling_steps,
        randomly_kill_helix_feature=False,
        negative_helix_constant=0.2,
        grad_enabled=False,
        logmd=False,
        # Validation (disabled - we handle separately)
        use_alphafold3_validation=False,
        alphafold_dir="",
        af3_docker_name="",
        af3_database_settings="",
        hmmer_path="",
        work_dir="",
        use_msa_for_af3=False,
        # Output
        plot=False,
    )

    print(f"\n{'='*60}")
    print("STAGE 1: BOLTZ DESIGN")
    print(f"{'='*60}")
    print(f"Name: {name}")
    print(f"Target: {protein_seqs[:50]}..." if len(protein_seqs) > 50 else f"Target: {protein_seqs}")
    print(f"Designs: {num_designs}, Cycles: {num_cycles}")
    print(f"{'='*60}\n")

    pipeline = ProteinHunter_Boltz(args)
    pipeline.run_pipeline()

    # Collect results
    results_dir = Path(args.save_dir)
    designs_dir = results_dir / "designs"
    best_dir = results_dir / "best_designs"

    result = {
        "status": "success",
        "save_dir": str(results_dir),
        "target_seq": protein_seqs,
        "designs": {},
        "best_designs": {},
        "design_stats_csv": None,
        "best_designs_csv": None,
    }

    csv_file = designs_dir / "design_stats.csv"
    if csv_file.exists():
        result["design_stats_csv"] = csv_file.read_text()

    for pdb in designs_dir.glob("*.pdb"):
        result["designs"][pdb.name] = pdb.read_text()

    best_csv = best_dir / "best_designs.csv"
    if best_csv.exists():
        result["best_designs_csv"] = best_csv.read_text()

    if best_dir.exists():
        for pdb in best_dir.glob("*.pdb"):
            result["best_designs"][pdb.name] = pdb.read_text()

    print(f"\nCollected {len(result['designs'])} design PDBs")
    print(f"Collected {len(result['best_designs'])} best design PDBs")

    return result


# GPU-specific wrapper functions (Modal decorators are compile-time)
@app.function(image=boltz_image, gpu="L4", timeout=3600, volumes={"/cache": cache_volume})
def run_local_pipeline_L4(**kwargs) -> dict:
    """Run Boltz pipeline on L4 GPU (24GB VRAM, budget option)."""
    return _run_local_pipeline_impl(**kwargs)

@app.function(image=boltz_image, gpu="A10G", timeout=3600, volumes={"/cache": cache_volume})
def run_local_pipeline_A10G(**kwargs) -> dict:
    """Run Boltz pipeline on A10G GPU (24GB VRAM)."""
    return _run_local_pipeline_impl(**kwargs)

@app.function(image=boltz_image, gpu="A100-40GB", timeout=3600, volumes={"/cache": cache_volume})
def run_local_pipeline_A100_40GB(**kwargs) -> dict:
    """Run Boltz pipeline on A100-40GB GPU."""
    return _run_local_pipeline_impl(**kwargs)

@app.function(image=boltz_image, gpu="A100-80GB", timeout=3600, volumes={"/cache": cache_volume})
def run_local_pipeline_A100_80GB(**kwargs) -> dict:
    """Run Boltz pipeline on A100-80GB GPU."""
    return _run_local_pipeline_impl(**kwargs)

@app.function(image=boltz_image, gpu="H100", timeout=3600, volumes={"/cache": cache_volume})
def run_local_pipeline_H100(**kwargs) -> dict:
    """Run Boltz pipeline on H100 GPU (80GB VRAM, fastest)."""
    return _run_local_pipeline_impl(**kwargs)

# Register GPU functions for dispatch
GPU_FUNCTIONS["L4"] = run_local_pipeline_L4
GPU_FUNCTIONS["A10G"] = run_local_pipeline_A10G
GPU_FUNCTIONS["A100-40GB"] = run_local_pipeline_A100_40GB
GPU_FUNCTIONS["A100-80GB"] = run_local_pipeline_A100_80GB
GPU_FUNCTIONS["H100"] = run_local_pipeline_H100


# =============================================================================
# STAGE 2: AF3 VALIDATION (self-contained, same logic as modal_boltz_ph)
# =============================================================================

def _run_af3_single_impl(
    design_id: str,
    binder_seq: str,
    target_seq: str,
    binder_chain: str = "A",
    target_chain: str = "B",
) -> Dict[str, Any]:
    """
    Run AF3 validation on a SINGLE design (holo state).
    """
    import json
    import subprocess
    import tempfile
    
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
        "af3_confidence_json": None,
    }
    
    # Check for AF3 weights
    weights_path = Path("/af3_weights/af3.bin")
    if not weights_path.exists():
        result["error"] = "AF3 weights not found"
        return result
    
    # Build AF3 JSON input
    af3_input = {
        "name": design_id,
        "modelSeeds": [1],
        "dialect": "alphafold3",
        "version": 1,
        "sequences": [
            {
                "protein": {
                    "id": binder_chain,
                    "sequence": binder_seq,
                    "unpairedMsa": f">query\n{binder_seq}\n",
                    "pairedMsa": f">query\n{binder_seq}\n",
                    "templates": [],
                }
            },
            {
                "protein": {
                    "id": target_chain,
                    "sequence": target_seq,
                    "unpairedMsa": f">query\n{target_seq}\n",
                    "pairedMsa": f">query\n{target_seq}\n",
                    "templates": [],
                }
            }
        ]
    }
    
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
                result["af3_confidence_json"] = confidence_text
                
                atom_plddts = confidence.get("atom_plddts", [])
                if atom_plddts:
                    result["af3_plddt"] = sum(atom_plddts) / len(atom_plddts)
                
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
    """
    import json
    import subprocess
    import tempfile
    
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
    
    weights_path = Path("/af3_weights/af3.bin")
    if not weights_path.exists():
        result["error"] = "AF3 weights not found"
        return result
    
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
    
    output_subdir = af_output_dir / apo_name
    if output_subdir.exists():
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


# AF3 GPU function mappings (AF3 requires 40GB+ VRAM)
AF3_HOLO_GPU_FUNCTIONS = {}
AF3_APO_GPU_FUNCTIONS = {}


# AF3 HOLO functions for different GPU types
@app.function(image=af3_image, gpu="A100", timeout=1800, volumes={"/cache": cache_volume, "/af3_weights": af3_weights_volume})
def run_af3_holo_A100_40GB(design_id: str, binder_seq: str, target_seq: str, binder_chain: str = "A", target_chain: str = "B") -> Dict[str, Any]:
    """Run AF3 HOLO on A100-40GB."""
    return _run_af3_single_impl(design_id, binder_seq, target_seq, binder_chain, target_chain)

@app.function(image=af3_image, gpu="A100-80GB", timeout=1800, volumes={"/cache": cache_volume, "/af3_weights": af3_weights_volume})
def run_af3_holo_A100_80GB(design_id: str, binder_seq: str, target_seq: str, binder_chain: str = "A", target_chain: str = "B") -> Dict[str, Any]:
    """Run AF3 HOLO on A100-80GB."""
    return _run_af3_single_impl(design_id, binder_seq, target_seq, binder_chain, target_chain)

@app.function(image=af3_image, gpu="H100", timeout=1800, volumes={"/cache": cache_volume, "/af3_weights": af3_weights_volume})
def run_af3_holo_H100(design_id: str, binder_seq: str, target_seq: str, binder_chain: str = "A", target_chain: str = "B") -> Dict[str, Any]:
    """Run AF3 HOLO on H100."""
    return _run_af3_single_impl(design_id, binder_seq, target_seq, binder_chain, target_chain)


# AF3 APO functions for different GPU types
@app.function(image=af3_image, gpu="A100", timeout=1800, volumes={"/cache": cache_volume, "/af3_weights": af3_weights_volume})
def run_af3_apo_A100_40GB(design_id: str, binder_seq: str, binder_chain: str = "A") -> Dict[str, Any]:
    """Run AF3 APO on A100-40GB."""
    return _run_af3_apo_impl(design_id, binder_seq, binder_chain)

@app.function(image=af3_image, gpu="A100-80GB", timeout=1800, volumes={"/cache": cache_volume, "/af3_weights": af3_weights_volume})
def run_af3_apo_A100_80GB(design_id: str, binder_seq: str, binder_chain: str = "A") -> Dict[str, Any]:
    """Run AF3 APO on A100-80GB."""
    return _run_af3_apo_impl(design_id, binder_seq, binder_chain)

@app.function(image=af3_image, gpu="H100", timeout=1800, volumes={"/cache": cache_volume, "/af3_weights": af3_weights_volume})
def run_af3_apo_H100(design_id: str, binder_seq: str, binder_chain: str = "A") -> Dict[str, Any]:
    """Run AF3 APO on H100."""
    return _run_af3_apo_impl(design_id, binder_seq, binder_chain)


# Register AF3 GPU functions (AF3 requires 40GB+ VRAM, so L4/A10G fallback to A100-40GB)
AF3_HOLO_GPU_FUNCTIONS["L4"] = run_af3_holo_A100_40GB  # L4 too small, use A100
AF3_HOLO_GPU_FUNCTIONS["A10G"] = run_af3_holo_A100_40GB  # A10G too small, use A100
AF3_HOLO_GPU_FUNCTIONS["A100-40GB"] = run_af3_holo_A100_40GB
AF3_HOLO_GPU_FUNCTIONS["A100-80GB"] = run_af3_holo_A100_80GB
AF3_HOLO_GPU_FUNCTIONS["H100"] = run_af3_holo_H100

AF3_APO_GPU_FUNCTIONS["L4"] = run_af3_apo_A100_40GB
AF3_APO_GPU_FUNCTIONS["A10G"] = run_af3_apo_A100_40GB
AF3_APO_GPU_FUNCTIONS["A100-40GB"] = run_af3_apo_A100_40GB
AF3_APO_GPU_FUNCTIONS["A100-80GB"] = run_af3_apo_A100_80GB
AF3_APO_GPU_FUNCTIONS["H100"] = run_af3_apo_H100


# =============================================================================
# STAGE 3: PYROSETTA SCORING (wraps utils/pyrosetta_utils.py)
# =============================================================================

@app.function(
    image=pyrosetta_image,
    cpu=4,
    timeout=1800,
)
def run_pyrosetta_scoring(
    design_id: str,
    af3_structure: str,
    af3_iptm: float,
    af3_ptm: float,
    af3_plddt: float,
    binder_chain: str = "A",
    target_chain: str = "B",
    apo_structure: Optional[str] = None,
    af3_confidence_json: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Run PyRosetta interface analysis using utils/pyrosetta_utils.py.
    
    This wraps the local PyRosetta functions (pr_relax, score_interface, etc.)
    with the Modal PyRosetta container.
    """
    import json
    import tempfile
    import numpy as np
    
    sys.path.insert(0, "/root/protein_hunter")
    
    from Bio.PDB import MMCIFParser, PDBIO, PDBParser
    
    result = {
        "design_id": design_id,
        "af3_iptm": float(af3_iptm),
        "af3_ptm": float(af3_ptm),
        "af3_plddt": float(af3_plddt),
        "accepted": False,
        "rejection_reason": None,
        "relaxed_pdb": None,
        # Interface metrics
        "binder_score": 0.0,
        "total_score": 0.0,
        "surface_hydrophobicity": 0.0,
        "interface_sc": 0.0,
        "interface_packstat": 0.0,
        "interface_dG": 0.0,
        "interface_dSASA": 0.0,
        "interface_dG_SASA_ratio": 0.0,
        "interface_nres": 0,
        "interface_interface_hbonds": 0,
        "interface_hbond_percentage": 0.0,
        "interface_delta_unsat_hbonds": 0,
        "interface_delta_unsat_hbonds_percentage": 0.0,
        "interface_hydrophobicity": 0.0,
        "binder_sasa": 0.0,
        "interface_fraction": 0.0,
        # Secondary metrics
        "apo_holo_rmsd": None,
        "i_pae": None,
        "rg": None,
    }
    
    if not af3_structure:
        result["rejection_reason"] = "No AF3 structure"
        return result
    
    work_dir = Path(tempfile.mkdtemp())
    
    print(f"  Running PyRosetta scoring for {design_id}...")
    
    try:
        # Convert CIF to PDB
        cif_file = work_dir / f"{design_id}.cif"
        cif_file.write_text(af3_structure)
        
        pdb_file = work_dir / f"{design_id}.pdb"
        parser = MMCIFParser(QUIET=True)
        structure = parser.get_structure(design_id, str(cif_file))
        
        # Fix chain IDs if needed (CIF may have long chain IDs)
        next_chain_idx = 0
        def int_to_chain(i):
            if i < 26:
                return chr(ord("A") + i)
            elif i < 52:
                return chr(ord("a") + i - 26)
            else:
                return chr(ord("0") + i - 52)
        
        for chain in structure.get_chains():
            if len(chain.id) != 1:
                while True:
                    c = int_to_chain(next_chain_idx)
                    next_chain_idx += 1
                    if c not in [ch.id for ch in structure.get_chains()]:
                        chain.id = c
                        break
        
        io = PDBIO()
        io.set_structure(structure)
        io.save(str(pdb_file))
        
        # Check for multi-chain collapse
        pdb_parser = PDBParser(QUIET=True)
        temp_structure = pdb_parser.get_structure("check", str(pdb_file))
        total_chains = [chain.id for model in temp_structure for chain in model]
        
        # Import local PyRosetta utils
        from utils.pyrosetta_utils import pr_relax, score_interface, collapse_multiple_chains
        
        # Collapse chains if needed
        if len(total_chains) > 2:
            collapsed_pdb = work_dir / f"{design_id}_collapsed.pdb"
            collapse_multiple_chains(str(pdb_file), str(collapsed_pdb), binder_chain, "B")
            pdb_for_scoring = str(collapsed_pdb)
            print(f"    Collapsed {len(total_chains)} chains to 2 for interface analysis")
        else:
            pdb_for_scoring = str(pdb_file)
        
        # Relax structure
        relaxed_pdb = work_dir / f"{design_id}_relaxed.pdb"
        pr_relax(pdb_for_scoring, str(relaxed_pdb))
        
        if relaxed_pdb.exists():
            result["relaxed_pdb"] = relaxed_pdb.read_text()
        
        # Calculate total_score and binder_sasa (not returned by score_interface)
        try:
            import pyrosetta as pr
            from pyrosetta.rosetta.core.select.residue_selector import ChainSelector
            
            # Initialize if not already done
            if not pr.rosetta.basic.was_init_called():
                pr.init("-ignore_unrecognized_res -ignore_zero_occupancy -mute all")
            
            pose = pr.pose_from_file(str(relaxed_pdb))
            sfxn = pr.get_fa_scorefxn()
            
            # Total score
            result["total_score"] = round(float(sfxn(pose)), 2)
            
            # Binder SASA
            chain_selector = ChainSelector(binder_chain)
            bsasa = pr.rosetta.core.simple_metrics.metrics.SasaMetric()
            bsasa.set_residue_selector(chain_selector)
            result["binder_sasa"] = round(float(bsasa.calculate(pose)), 2)
        except Exception as e:
            print(f"    Warning: total_score/binder_sasa calculation failed: {e}")
        
        # Score interface using local utils function
        interface_scores, interface_AA, interface_residues = score_interface(
            str(relaxed_pdb), pdb_for_scoring, binder_chain=binder_chain, target_chain="B"
        )
        
        # Copy scores to result
        for key, value in interface_scores.items():
            if key in result:
                result[key] = value
        
        # Calculate i_pae from confidence JSON
        if af3_confidence_json:
            try:
                confidence = json.loads(af3_confidence_json)
                pae_matrix = np.array(confidence.get('pae', []))
                
                if len(pae_matrix) > 0:
                    # Get binder length from sequence
                    binder_len = len([r for r in structure.get_residues() 
                                     if r.get_parent().id == binder_chain])
                    
                    if binder_len > 0 and pae_matrix.shape[0] > binder_len:
                        interface_pae1 = np.mean(pae_matrix[:binder_len, binder_len:])
                        interface_pae2 = np.mean(pae_matrix[binder_len:, :binder_len])
                        result["i_pae"] = round((interface_pae1 + interface_pae2) / 2, 2)
            except Exception as e:
                print(f"    Warning: i_pae calculation failed: {e}")
        
        # Calculate radius of gyration
        try:
            from utils.metrics import radius_of_gyration
            rg, _ = radius_of_gyration(str(relaxed_pdb), chain_id=binder_chain)
            result["rg"] = round(rg, 2) if rg else None
        except Exception as e:
            print(f"    Warning: rg calculation failed: {e}")
        
        # Calculate APO-HOLO RMSD if apo structure provided
        if apo_structure:
            try:
                from utils.metrics import get_CA_and_sequence, np_rmsd
                
                apo_cif = work_dir / f"{design_id}_apo.cif"
                apo_cif.write_text(apo_structure)
                
                apo_pdb = work_dir / f"{design_id}_apo.pdb"
                apo_struct = parser.get_structure(f"{design_id}_apo", str(apo_cif))
                for chain in apo_struct.get_chains():
                    if len(chain.id) != 1:
                        chain.id = "A"
                io.set_structure(apo_struct)
                io.save(str(apo_pdb))
                
                xyz_holo, _ = get_CA_and_sequence(str(relaxed_pdb), chain_id=binder_chain)
                xyz_apo, _ = get_CA_and_sequence(str(apo_pdb), chain_id="A")
                
                if len(xyz_holo) == len(xyz_apo):
                    rmsd = np_rmsd(xyz_holo, xyz_apo)
                    result["apo_holo_rmsd"] = round(rmsd, 2) if rmsd else None
            except Exception as e:
                print(f"    Warning: APO-HOLO RMSD calculation failed: {e}")
        
        # Acceptance criteria
        rejection_reasons = []
        
        if af3_iptm < 0.7:
            rejection_reasons.append(f"Low AF3 ipTM: {af3_iptm:.3f}")
        if af3_plddt < 80:
            rejection_reasons.append(f"Low AF3 pLDDT: {af3_plddt:.1f}")
        if result["binder_score"] >= 0:
            rejection_reasons.append(f"binder_score >= 0: {result['binder_score']}")
        if result["surface_hydrophobicity"] >= 0.35:
            rejection_reasons.append(f"surface_hydrophobicity >= 0.35: {result['surface_hydrophobicity']}")
        if result["interface_sc"] <= 0.55:
            rejection_reasons.append(f"interface_sc <= 0.55: {result['interface_sc']}")
        if result["interface_dG"] >= 0:
            rejection_reasons.append(f"interface_dG >= 0: {result['interface_dG']}")
        if result["interface_nres"] <= 7:
            rejection_reasons.append(f"interface_nres <= 7: {result['interface_nres']}")
        if result["interface_delta_unsat_hbonds"] >= 4:
            rejection_reasons.append(f"BUNS >= 4: {result['interface_delta_unsat_hbonds']}")
        
        # Secondary filters
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
        print(f"    Error: {traceback.format_exc()}")
    
    return result


# =============================================================================
# LOCAL ENTRYPOINT WITH FULL PIPELINE
# =============================================================================

@app.local_entrypoint()
def main(
    # Core settings
    name: str = "test_run",
    num_designs: int = 1,
    num_cycles: int = 5,
    gpu: str = "H100",
    # Target specification
    protein_seqs: str = "",
    ligand_smiles: str = "",
    ligand_ccd: str = "",
    nucleic_seq: str = "",
    nucleic_type: str = "dna",
    # Template settings
    template_path: str = "",
    template_cif_chain_id: str = "",
    msa_mode: str = "mmseqs",
    # Binder sequence settings
    seq: str = "",
    min_protein_length: int = 100,
    max_protein_length: int = 150,
    percent_x: int = 90,
    cyclic: bool = False,
    exclude_p: bool = False,
    # Hotspot/contact settings
    contact_residues: str = "",
    contact_cutoff: float = 15.0,
    max_contact_filter_retries: int = 6,
    no_contact_filter: bool = False,
    # Sequence design (MPNN) settings
    temperature: float = 0.1,
    omit_aa: str = "C",
    alanine_bias: bool = False,
    alanine_bias_start: float = -0.5,
    alanine_bias_end: float = -0.1,
    # Quality thresholds
    high_iptm_threshold: float = 0.8,
    high_plddt_threshold: float = 0.8,
    # Model settings
    diffuse_steps: int = 200,
    recycling_steps: int = 3,
    # Validation (matches main script)
    use_alphafold3_validation: bool = False,
):
    """
    Test the local Boltz pipeline on Modal with optional AF3/PyRosetta validation.
    
    Arguments mirror the main boltz_ph/design.py script for consistency.
    Use --use-alphafold3-validation to enable downstream validation (AF3 + PyRosetta for protein targets).
    
    Examples:
        # Basic test with PDL1 target
        modal run test_local_pipeline_modal.py --protein-seqs "AFTVTVPK..."
        
        # Full pipeline with AF3 + PyRosetta validation
        modal run test_local_pipeline_modal.py \\
            --name my_test \\
            --protein-seqs "TARGET_SEQ" \\
            --num-designs 3 \\
            --num-cycles 5 \\
            --alanine-bias \\
            --use-alphafold3-validation
        
        # With hotspots
        modal run test_local_pipeline_modal.py \\
            --protein-seqs "TARGET_SEQ" \\
            --contact-residues "54,56,66" \\
            --use-alphafold3-validation
    """
    import csv
    from io import StringIO
    
    # Validate GPU type
    valid_gpus = ["L4", "A10G", "A100-40GB", "A100-80GB", "H100"]
    if gpu not in valid_gpus:
        print(f"❌ Invalid GPU type: {gpu}")
        print(f"   Valid options: {', '.join(valid_gpus)}")
        return
    
    # Use PDL1 as default target if none provided
    target_seq = protein_seqs if protein_seqs else "AFTVTVPKDLYVVEYGSNMTIECKFPVEKQLDLAALIVYWEMEDKNIIQFVHGEEDLKVQHSSYRQRARLLKDQLSLGNAALQITDVKLQDAGVYRCMISYGGADYKRITVKVNAPYAAALE"
    
    # Determine target type (matches boltz_ph/pipeline.py logic)
    any_ligand_or_nucleic = ligand_smiles or ligand_ccd or nucleic_seq
    if nucleic_type.strip() and nucleic_seq.strip():
        target_type = "nucleic"
    elif any_ligand_or_nucleic:
        target_type = "small_molecule"
    else:
        target_type = "protein"

    print(f"\n{'='*70}")
    print("LOCAL PIPELINE TEST WITH MODAL")
    print(f"{'='*70}")
    print(f"Name: {name}")
    print(f"GPU: {gpu}")
    print(f"Designs: {num_designs}, Cycles: {num_cycles}")
    print(f"Target: {target_seq[:50]}..." if len(target_seq) > 50 else f"Target: {target_seq}")
    print(f"Target type: {target_type}")
    if ligand_ccd:
        print(f"Ligand CCD: {ligand_ccd}")
    if ligand_smiles:
        print(f"Ligand SMILES: {ligand_smiles[:30]}...")
    if contact_residues:
        print(f"Hotspots: {contact_residues}")
    print(f"MSA mode: {msa_mode}")
    print(f"Alanine bias: {alanine_bias}")
    print(f"AF3 validation: {use_alphafold3_validation}")
    if use_alphafold3_validation:
        print(f"PyRosetta scoring: {target_type == 'protein'} (only for protein targets)")
    print(f"{'='*70}\n")

    # =========================================================================
    # STAGE 1: Boltz Design
    # =========================================================================
    print("\n" + "="*60)
    print(f"STAGE 1: BOLTZ DESIGN (GPU: {gpu})")
    print("="*60)
    
    # Dispatch to GPU-specific function
    pipeline_func = GPU_FUNCTIONS[gpu]
    
    boltz_result = pipeline_func.remote(
        # Core
        name=name,
        num_designs=num_designs,
        num_cycles=num_cycles,
        # Target
        protein_seqs=target_seq,
        ligand_smiles=ligand_smiles,
        ligand_ccd=ligand_ccd,
        nucleic_seq=nucleic_seq,
        nucleic_type=nucleic_type,
        # Template
        template_path=template_path,
        template_cif_chain_id=template_cif_chain_id,
        msa_mode=msa_mode,
        # Binder
        seq=seq,
        min_protein_length=min_protein_length,
        max_protein_length=max_protein_length,
        percent_X=percent_x,
        cyclic=cyclic,
        exclude_P=exclude_p,
        # Hotspot
        contact_residues=contact_residues,
        contact_cutoff=contact_cutoff,
        max_contact_filter_retries=max_contact_filter_retries,
        no_contact_filter=no_contact_filter,
        # MPNN
        temperature=temperature,
        omit_AA=omit_aa,
        alanine_bias=alanine_bias,
        alanine_bias_start=alanine_bias_start,
        alanine_bias_end=alanine_bias_end,
        # Quality
        high_iptm_threshold=high_iptm_threshold,
        high_plddt_threshold=high_plddt_threshold,
        # Model
        diffuse_steps=diffuse_steps,
        recycling_steps=recycling_steps,
    )

    # Save Boltz results locally
    output_dir = Path(f"./results_{name}")
    output_dir.mkdir(exist_ok=True)

    designs_dir = output_dir / "designs"
    designs_dir.mkdir(exist_ok=True)

    best_dir = output_dir / "best_designs"
    best_dir.mkdir(exist_ok=True)

    if boltz_result.get("design_stats_csv"):
        (designs_dir / "design_stats.csv").write_text(boltz_result["design_stats_csv"])
        print("✓ Saved designs/design_stats.csv")

    for pdb_name, pdb_content in boltz_result.get("designs", {}).items():
        (designs_dir / pdb_name).write_text(pdb_content)
    print(f"✓ Saved {len(boltz_result.get('designs', {}))} PDBs to designs/")

    if boltz_result.get("best_designs_csv"):
        (best_dir / "best_designs.csv").write_text(boltz_result["best_designs_csv"])
        print("✓ Saved best_designs/best_designs.csv")

    for pdb_name, pdb_content in boltz_result.get("best_designs", {}).items():
        (best_dir / pdb_name).write_text(pdb_content)
    print(f"✓ Saved {len(boltz_result.get('best_designs', {}))} PDBs to best_designs/")

    # =========================================================================
    # STAGE 2 & 3: AF3 + PyRosetta (if enabled)
    # =========================================================================
    if not use_alphafold3_validation:
        print(f"\n{'='*60}")
        print("TEST COMPLETE (Boltz only)")
        print(f"{'='*60}")
        print(f"Results saved to: {output_dir}")
        return

    # Parse best designs CSV to get sequences
    if not boltz_result.get("best_designs_csv"):
        print("\n⚠ No best designs found. Skipping AF3/PyRosetta.")
        return
    
    csv_reader = csv.DictReader(StringIO(boltz_result["best_designs_csv"]))
    best_designs = list(csv_reader)
    
    # PyRosetta scoring only applies to protein targets (matches main script)
    run_pyrosetta = target_type == "protein"
    
    # Determine AF3 GPU (falls back to A100-40GB for L4/A10G)
    af3_gpu = gpu if gpu in ["A100-40GB", "A100-80GB", "H100"] else "A100-40GB"
    
    print(f"\n{'='*60}")
    print(f"STAGE 2: AF3 VALIDATION (GPU: {af3_gpu})")
    print(f"{'='*60}")
    print(f"Validating {len(best_designs)} best designs with AF3...")
    
    # Create af3_validation directory
    af3_dir = output_dir / "af3_validation"
    af3_dir.mkdir(exist_ok=True)
    
    af3_results = []
    
    # Run AF3 HOLO and APO for each design
    for design in best_designs:
        design_id = design.get("design_id", "unknown")
        binder_seq = design.get("binder_sequence", "")
        
        if not binder_seq:
            print(f"  ⚠ Skipping {design_id}: no binder sequence")
            continue
        
        print(f"\n  Processing {design_id}...")
        
        # Run HOLO (complex) using GPU-specific AF3 function
        af3_holo_func = AF3_HOLO_GPU_FUNCTIONS[gpu]
        holo_result = af3_holo_func.remote(
            design_id=design_id,
            binder_seq=binder_seq,
            target_seq=target_seq,
        )
        
        # Run APO (binder only) for PyRosetta RMSD calculation (protein targets only)
        apo_result = None
        if run_pyrosetta:
            af3_apo_func = AF3_APO_GPU_FUNCTIONS[gpu]
            apo_result = af3_apo_func.remote(
                design_id=design_id,
                binder_seq=binder_seq,
            )
        
        af3_results.append({
            "design": design,
            "holo": holo_result,
            "apo": apo_result,
        })
    
    # Save AF3 results CSV
    af3_csv_rows = []
    for af3_data in af3_results:
        holo = af3_data["holo"]
        af3_csv_rows.append({
            "design_id": holo.get("design_id", "unknown"),
            "af3_iptm": holo.get("af3_iptm", 0),
            "af3_ptm": holo.get("af3_ptm", 0),
            "af3_plddt": holo.get("af3_plddt", 0),
        })
    
    if af3_csv_rows:
        import pandas as pd
        af3_csv_df = pd.DataFrame(af3_csv_rows)
        af3_csv_df.to_csv(af3_dir / "af3_results.csv", index=False)
        print("✓ Saved af3_validation/af3_results.csv")
    
    # =========================================================================
    # STAGE 3: PyRosetta Scoring (protein targets only)
    # =========================================================================
    if not run_pyrosetta:
        # Just save AF3 results (non-protein targets skip PyRosetta)
        print(f"\n{'='*60}")
        print("AF3 VALIDATION RESULTS")
        print(f"{'='*60}")
        print(f"(PyRosetta skipped for {target_type} targets)")
        
        for af3_data in af3_results:
            holo = af3_data["holo"]
            design_id = holo.get("design_id", "unknown")
            
            print(f"  {design_id}:")
            print(f"    ipTM: {holo.get('af3_iptm', 0):.3f}")
            print(f"    ptm: {holo.get('af3_ptm', 0):.3f}")
            print(f"    pLDDT: {holo.get('af3_plddt', 0):.1f}")
            
            if holo.get("af3_structure"):
                (af3_dir / f"{design_id}_af3.cif").write_text(holo["af3_structure"])
            
            if holo.get("error"):
                print(f"    Error: {holo['error']}")
        
        print(f"\n✓ AF3 structures saved to {af3_dir}/")
        print(f"\n{'='*60}")
        print("TEST COMPLETE (Boltz + AF3)")
        print(f"{'='*60}")
        return
    
    print(f"\n{'='*60}")
    print("STAGE 3: PYROSETTA SCORING")
    print(f"{'='*60}")
    
    # Create accepted and rejected directories (replaces pyrosetta_scored)
    accepted_dir = output_dir / "accepted_designs"
    rejected_dir = output_dir / "rejected"
    accepted_dir.mkdir(exist_ok=True)
    rejected_dir.mkdir(exist_ok=True)
    
    final_results = []
    
    for af3_data in af3_results:
        holo = af3_data["holo"]
        apo = af3_data["apo"]
        design = af3_data["design"]
        design_id = holo.get("design_id", "unknown")
        
        if not holo.get("af3_structure"):
            print(f"  ⚠ Skipping {design_id}: no AF3 structure")
            continue
        
        print(f"\n  Scoring {design_id}...")
        
        # Run PyRosetta scoring
        pyrosetta_result = run_pyrosetta_scoring.remote(
            design_id=design_id,
            af3_structure=holo.get("af3_structure"),
            af3_iptm=holo.get("af3_iptm", 0),
            af3_ptm=holo.get("af3_ptm", 0),
            af3_plddt=holo.get("af3_plddt", 0),
            apo_structure=apo.get("apo_structure") if apo else None,
            af3_confidence_json=holo.get("af3_confidence_json"),
        )
        
        final_results.append({
            "design": design,
            "af3": holo,
            "pyrosetta": pyrosetta_result,
        })
    
    # =========================================================================
    # FINAL OUTPUT
    # =========================================================================
    print(f"\n{'='*60}")
    print("FINAL RESULTS")
    print(f"{'='*60}")
    
    accepted_rows = []
    rejected_rows = []
    
    for result_data in final_results:
        design = result_data["design"]
        af3 = result_data["af3"]
        pr = result_data["pyrosetta"]
        
        design_id = af3.get("design_id", "unknown")
        accepted = pr.get("accepted", False)
        
        status = "✓ ACCEPTED" if accepted else "✗ REJECTED"
        print(f"\n  {design_id}: {status}")
        print(f"    AF3 ipTM: {af3.get('af3_iptm', 0):.3f}, ptm: {af3.get('af3_ptm', 0):.3f}, pLDDT: {af3.get('af3_plddt', 0):.1f}")
        print(f"    interface_dG: {pr.get('interface_dG', 0):.1f}, interface_sc: {pr.get('interface_sc', 0):.3f}")
        print(f"    BUNS: {pr.get('interface_delta_unsat_hbonds', 0)}, interface_nres: {pr.get('interface_nres', 0)}")
        
        if pr.get("i_pae"):
            print(f"    i_pae: {pr['i_pae']:.1f}")
        if pr.get("rg"):
            print(f"    rg: {pr['rg']:.1f}")
        if pr.get("apo_holo_rmsd"):
            print(f"    apo_holo_rmsd: {pr['apo_holo_rmsd']:.2f}")
        
        if not accepted:
            print(f"    Rejection: {pr.get('rejection_reason', 'Unknown')}")
        
        # Build full row with design metadata
        row = {
            "design_id": design_id,
            "design_num": design.get("design_num", 0),
            "cycle": design.get("cycle", 0),
            "binder_sequence": design.get("binder_sequence", ""),
            "binder_length": design.get("binder_length", 0),
            "af3_iptm": pr.get("af3_iptm", 0),
            "af3_ptm": pr.get("af3_ptm", 0),
            "af3_plddt": pr.get("af3_plddt", 0),
            "accepted": pr.get("accepted", False),
            "rejection_reason": pr.get("rejection_reason", ""),
            # PyRosetta metrics
            "binder_score": pr.get("binder_score", 0),
            "total_score": pr.get("total_score", 0),
            "interface_sc": pr.get("interface_sc", 0),
            "interface_packstat": pr.get("interface_packstat", 0),
            "interface_dG": pr.get("interface_dG", 0),
            "interface_dSASA": pr.get("interface_dSASA", 0),
            "interface_dG_SASA_ratio": pr.get("interface_dG_SASA_ratio", 0),
            "interface_nres": pr.get("interface_nres", 0),
            "interface_interface_hbonds": pr.get("interface_interface_hbonds", 0),
            "interface_hbond_percentage": pr.get("interface_hbond_percentage", 0),
            "interface_delta_unsat_hbonds": pr.get("interface_delta_unsat_hbonds", 0),
            "interface_delta_unsat_hbonds_percentage": pr.get("interface_delta_unsat_hbonds_percentage", 0),
            "interface_hydrophobicity": pr.get("interface_hydrophobicity", 0),
            "surface_hydrophobicity": pr.get("surface_hydrophobicity", 0),
            "binder_sasa": pr.get("binder_sasa", 0),
            "interface_fraction": pr.get("interface_fraction", 0),
            # Secondary metrics
            "apo_holo_rmsd": pr.get("apo_holo_rmsd"),
            "i_pae": pr.get("i_pae"),
            "rg": pr.get("rg"),
        }
        
        # Save to appropriate directory and list
        if accepted:
            accepted_rows.append(row)
            if pr.get("relaxed_pdb"):
                (accepted_dir / f"{design_id}_relaxed.pdb").write_text(pr["relaxed_pdb"])
        else:
            rejected_rows.append(row)
            if pr.get("relaxed_pdb"):
                (rejected_dir / f"{design_id}_relaxed.pdb").write_text(pr["relaxed_pdb"])
        
        # Save AF3 structure with correct naming (*_af3.cif)
        if af3.get("af3_structure"):
            (af3_dir / f"{design_id}_af3.cif").write_text(af3["af3_structure"])
    
    # Save accepted/rejected CSVs
    import pandas as pd
    
    if accepted_rows:
        accepted_df = pd.DataFrame(accepted_rows)
        accepted_df.to_csv(accepted_dir / "accepted_stats.csv", index=False)
        print(f"\n✓ Saved accepted_designs/accepted_stats.csv ({len(accepted_rows)} designs)")
    
    if rejected_rows:
        rejected_df = pd.DataFrame(rejected_rows)
        rejected_df.to_csv(rejected_dir / "rejected_stats.csv", index=False)
        print(f"✓ Saved rejected/rejected_stats.csv ({len(rejected_rows)} designs)")
    
    print(f"\n{'='*60}")
    print("TEST COMPLETE (Full Pipeline)")
    print(f"{'='*60}")
    print(f"Accepted: {len(accepted_rows)}")
    print(f"Rejected: {len(rejected_rows)}")
    print("\nOutput structure:")
    print(f"  {output_dir}/")
    print("  ├── designs/              # All cycles")
    print("  ├── best_designs/         # Best cycle per design")
    print("  ├── af3_validation/       # AF3 structures + metrics")
    print("  │   ├── af3_results.csv")
    print("  │   └── *_af3.cif")
    print("  ├── accepted_designs/     # Passed filters")
    print("  │   ├── accepted_stats.csv")
    print("  │   └── *_relaxed.pdb")
    print("  └── rejected/             # Failed filters")
    print("      ├── rejected_stats.csv")
    print("      └── *_relaxed.pdb")
    print(f"\nResults saved to: {output_dir}")
