#!/usr/bin/env python3
"""
Simple Modal harness to test the local Boltz pipeline (boltz_ph/pipeline.py).

No streaming, no parallelization - just runs the pipeline on Modal's GPU
and downloads results.

Usage:
    modal run test_local_pipeline_modal.py
"""

import modal
import os
import sys
from pathlib import Path

app = modal.App("test-local-pipeline")

# Volume for caching model weights (reuse from main script)
cache_volume = modal.Volume.from_name("protein-hunter-cache", create_if_missing=True)

# Same image as main modal script
image = (
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


@app.function(
    image=image,
    gpu="L4",
    timeout=3600,
    volumes={"/cache": cache_volume},
)
def run_local_pipeline(
    name: str,
    target_seq: str,
    num_designs: int = 1,
    num_cycles: int = 2,
    min_length: int = 100,
    max_length: int = 150,
    msa_mode: str = "single",  # Use "single" (empty MSA) to avoid ColabFold API rate limits
) -> dict:
    """Run the local Boltz pipeline on Modal."""
    import argparse

    # Add protein hunter to path
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

    # Import pipeline
    from boltz_ph.pipeline import ProteinHunter_Boltz

    # Create args namespace (mimicking argparse)
    args = argparse.Namespace(
        # Job identity
        name=name,
        save_dir=f"/root/protein_hunter/results_{name}",
        # Target
        protein_seqs=target_seq,
        ligand_smiles="",
        ligand_ccd="",
        nucleic_seq="",
        nucleic_type="dna",
        # Template
        template_path="",
        template_cif_chain_id="",
        # Binder
        seq="",
        min_protein_length=min_length,
        max_protein_length=max_length,
        percent_X=90,
        cyclic=False,
        exclude_P=False,
        # Design
        num_designs=num_designs,
        num_cycles=num_cycles,
        contact_residues="",
        temperature=0.1,
        omit_AA="C",
        alanine_bias=False,
        alanine_bias_start=-0.5,
        alanine_bias_end=-0.1,
        high_iptm_threshold=0.8,
        high_plddt_threshold=0.8,
        # Contact filtering
        no_contact_filter=True,
        max_contact_filter_retries=6,
        contact_cutoff=15.0,
        # MSA
        msa_mode=msa_mode,
        # Model
        gpu_id=0,
        ccd_path=str(cache_dir / "mols"),
        boltz_model_path=str(cache_dir / "boltz2_conf.ckpt"),
        boltz_model_version="boltz2",
        diffuse_steps=200,
        recycling_steps=3,
        randomly_kill_helix_feature=False,
        negative_helix_constant=0.2,
        grad_enabled=False,
        logmd=False,
        # Validation (disabled)
        use_alphafold3_validation=False,
        alphafold_dir="",
        af3_docker_name="",
        af3_database_settings="",
        hmmer_path="",
        work_dir="",
        use_msa_for_af3=False,
        # Visualization
        plot=False,
        mode="conditional",
    )

    print(f"\n{'='*60}")
    print("TESTING LOCAL PIPELINE ON MODAL")
    print(f"{'='*60}")
    print(f"Name: {name}")
    print(f"Target: {target_seq[:50]}...")
    print(f"Designs: {num_designs}, Cycles: {num_cycles}")
    print(f"{'='*60}\n")

    # Run pipeline
    pipeline = ProteinHunter_Boltz(args)
    pipeline.run_pipeline()

    # Collect results
    results_dir = Path(args.save_dir)
    designs_dir = results_dir / "designs"
    best_dir = results_dir / "best_designs"

    result = {
        "status": "success",
        "save_dir": str(results_dir),
        "designs": {},
        "best_designs": {},
        "design_stats_csv": None,
        "best_designs_csv": None,
    }

    # Read design_stats.csv
    csv_file = designs_dir / "design_stats.csv"
    if csv_file.exists():
        result["design_stats_csv"] = csv_file.read_text()

    # Read PDBs from designs/
    for pdb in designs_dir.glob("*.pdb"):
        result["designs"][pdb.name] = pdb.read_text()

    # Read best_designs.csv
    best_csv = best_dir / "best_designs.csv"
    if best_csv.exists():
        result["best_designs_csv"] = best_csv.read_text()

    # Read PDBs from best_designs/
    if best_dir.exists():
        for pdb in best_dir.glob("*.pdb"):
            result["best_designs"][pdb.name] = pdb.read_text()

    print(f"\nCollected {len(result['designs'])} design PDBs")
    print(f"Collected {len(result['best_designs'])} best design PDBs")

    return result


@app.local_entrypoint()
def main(
    name: str = "pdl1_test",
    num_designs: int = 1,
    num_cycles: int = 2,
):
    """Test the local Boltz pipeline on Modal."""
    # PDL1 target sequence
    target_seq = "AFTVTVPKDLYVVEYGSNMTIECKFPVEKQLDLAALIVYWEMEDKNIIQFVHGEEDLKVQHSSYRQRARLLKDQLSLGNAALQITDVKLQDAGVYRCMISYGGADYKRITVKVNAPYAAALE"

    print(f"Running local pipeline test: {name}")
    print(f"  Designs: {num_designs}, Cycles: {num_cycles}")

    result = run_local_pipeline.remote(
        name=name,
        target_seq=target_seq,
        num_designs=num_designs,
        num_cycles=num_cycles,
    )

    # Save results locally
    output_dir = Path(f"./results_{name}")
    output_dir.mkdir(exist_ok=True)

    designs_dir = output_dir / "designs"
    designs_dir.mkdir(exist_ok=True)

    best_dir = output_dir / "best_designs"
    best_dir.mkdir(exist_ok=True)

    # Save design_stats.csv
    if result.get("design_stats_csv"):
        (designs_dir / "design_stats.csv").write_text(result["design_stats_csv"])
        print("Saved designs/design_stats.csv")

    # Save design PDBs
    for pdb_name, pdb_content in result.get("designs", {}).items():
        (designs_dir / pdb_name).write_text(pdb_content)
    print(f"Saved {len(result.get('designs', {}))} PDBs to designs/")

    # Save best_designs.csv
    if result.get("best_designs_csv"):
        (best_dir / "best_designs.csv").write_text(result["best_designs_csv"])
        print("Saved best_designs/best_designs.csv")

    # Save best design PDBs
    for pdb_name, pdb_content in result.get("best_designs", {}).items():
        (best_dir / pdb_name).write_text(pdb_content)
    print(f"Saved {len(result.get('best_designs', {}))} PDBs to best_designs/")

    print(f"\n{'='*60}")
    print("TEST COMPLETE")
    print(f"{'='*60}")
    print(f"Results saved to: {output_dir}")

    # Show CSV contents
    if result.get("design_stats_csv"):
        print("\ndesign_stats.csv preview:")
        lines = result["design_stats_csv"].strip().split("\n")
        for line in lines[:5]:  # Show header + first few rows
            print(f"  {line[:100]}...")
