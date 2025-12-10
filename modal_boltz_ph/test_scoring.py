"""
Test harness for comparing open-source and PyRosetta scoring methods.

This script allows isolated testing of the scoring functions using existing
AF3 validation outputs, without running the full Boltz → AF3 pipeline.

Usage:
    modal run modal_boltz_ph/test_scoring.py::test_scoring \
        --input-dir ./modal_results/PDL1_open_test3/af3_validation \
        --compare-pyrosetta true \
        --output-dir ./scoring_test_results

Input format:
    The input directory should contain:
    - af3_results.csv: CSV with columns design_id, af3_iptm, af3_ptm, af3_plddt
    - *_af3.cif: AF3-generated CIF files for each design

Output:
    - comparison_results.csv: Side-by-side metrics comparison
    - opensource_relaxed/: Relaxed PDBs from open-source scoring (OpenMM)
    - pyrosetta_relaxed/: Relaxed PDBs from PyRosetta scoring (if enabled)
"""

import csv
import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import modal

from modal_boltz_ph.app import app
from modal_boltz_ph.images import opensource_scoring_image, pyrosetta_image
from modal_boltz_ph.scoring_opensource import (
    OPENSOURCE_SCORING_GPU_FUNCTIONS,
    DEFAULT_OPENSOURCE_GPU,
    compare_sasa_methods,
)
from modal_boltz_ph.scoring_pyrosetta import run_pyrosetta_single


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def load_af3_results(input_dir: str) -> List[Dict[str, Any]]:
    """
    Load AF3 results from a validation directory.
    
    Args:
        input_dir: Path to af3_validation directory
        
    Returns:
        List of dicts with design_id, af3_iptm, af3_ptm, af3_plddt, cif_path
    """
    input_path = Path(input_dir)
    results_csv = input_path / "af3_results.csv"
    
    if not results_csv.exists():
        raise FileNotFoundError(f"AF3 results CSV not found: {results_csv}")
    
    designs = []
    with open(results_csv, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            design_id = row["design_id"]
            cif_file = input_path / f"{design_id}_af3.cif"
            
            if not cif_file.exists():
                print(f"  Warning: CIF file not found for {design_id}, skipping")
                continue
            
            designs.append({
                "design_id": design_id,
                "af3_iptm": float(row.get("af3_iptm", 0)),
                "af3_ipsae": float(row.get("af3_ipsae", 0)),
                "af3_ptm": float(row.get("af3_ptm", 0)),
                "af3_plddt": float(row.get("af3_plddt", 0)),
                "cif_path": str(cif_file),
            })
    
    return designs


def format_comparison_row(
    design_id: str,
    metric: str,
    opensource_value: Any,
    pyrosetta_value: Any,
) -> Dict[str, Any]:
    """Format a comparison row with diff calculation."""
    diff = None
    if opensource_value is not None and pyrosetta_value is not None:
        try:
            diff = float(opensource_value) - float(pyrosetta_value)
        except (ValueError, TypeError):
            pass
    
    return {
        "design_id": design_id,
        "metric": metric,
        "opensource": opensource_value,
        "pyrosetta": pyrosetta_value,
        "diff": diff,
    }


# =============================================================================
# MODAL LOCAL ENTRYPOINT
# =============================================================================

@app.local_entrypoint()
def test_scoring(
    input_dir: str,
    output_dir: str = "./scoring_test_results",
    compare_pyrosetta: bool = True,
    open_scoring_gpu: str = "A10G",
    target_type: str = "protein",
    binder_chain: str = "A",
    target_chain: str = "B",
    max_designs: int = 0,
):
    """
    Test scoring methods on existing AF3 structures.
    
    This harness allows isolated testing of the open-source scoring pipeline
    (OpenMM relaxation + FreeSASA + sc-rs) against PyRosetta scoring, using
    existing AF3 validation outputs from prior runs.
    
    Args:
        input_dir: Path to af3_validation directory from a prior run
        output_dir: Where to save comparison results and relaxed PDBs
        compare_pyrosetta: Also run PyRosetta scoring for comparison
        open_scoring_gpu: GPU type for open-source scoring (default: A10G)
        target_type: Target type for filtering thresholds
        binder_chain: Chain ID of binder (default: A)
        target_chain: Chain ID of target (default: B)
        max_designs: Maximum designs to process (0 = all)
    
    Example:
        modal run modal_boltz_ph/test_scoring.py::test_scoring \\
            --input-dir ./modal_results/PDL1_open_test3/af3_validation \\
            --compare-pyrosetta true \\
            --output-dir ./scoring_test_results
    """
    print("\n" + "=" * 70)
    print("SCORING TEST HARNESS")
    print("=" * 70)
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Compare PyRosetta: {compare_pyrosetta}")
    print(f"Open-source GPU: {open_scoring_gpu}")
    print(f"Target type: {target_type}")
    print("=" * 70 + "\n")
    
    # Create output directories
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    opensource_pdb_dir = output_path / "opensource_relaxed"
    opensource_pdb_dir.mkdir(exist_ok=True)
    
    if compare_pyrosetta:
        pyrosetta_pdb_dir = output_path / "pyrosetta_relaxed"
        pyrosetta_pdb_dir.mkdir(exist_ok=True)
    
    # Load AF3 results
    print("Loading AF3 results...")
    designs = load_af3_results(input_dir)
    
    if max_designs > 0:
        designs = designs[:max_designs]
    
    print(f"Found {len(designs)} designs to process\n")
    
    # Get scoring functions
    opensource_fn = OPENSOURCE_SCORING_GPU_FUNCTIONS.get(
        open_scoring_gpu,
        OPENSOURCE_SCORING_GPU_FUNCTIONS[DEFAULT_OPENSOURCE_GPU]
    )
    
    # Metrics to compare
    METRICS = [
        "accepted",
        "rejection_reason",
        "interface_sc",
        "interface_dG",
        "interface_dSASA",
        "interface_nres",
        "interface_packstat",
        "interface_hbonds",
        "interface_delta_unsat_hbonds",
        "interface_hydrophobicity",
        "surface_hydrophobicity",
        "binder_sasa",
        "interface_fraction",
        "interface_hbond_percentage",
        "interface_delta_unsat_hbonds_percentage",
        "interface_dG_SASA_ratio",
        "binder_score",
        "total_score",
        "apo_holo_rmsd",
        "i_pae",
        "rg",
    ]
    
    # Process each design
    all_results = []
    comparison_rows = []
    
    for i, design in enumerate(designs):
        design_id = design["design_id"]
        print(f"\n[{i+1}/{len(designs)}] Processing {design_id}")
        print("-" * 50)
        
        # Load CIF content
        with open(design["cif_path"], "r") as f:
            cif_content = f.read()
        
        # ========== OPEN-SOURCE SCORING ==========
        print(f"  Running open-source scoring ({open_scoring_gpu})...")
        t0 = time.time()
        try:
            opensource_result = opensource_fn.remote(
                design_id=design_id,
                af3_structure=cif_content,
                af3_iptm=design["af3_iptm"],
                af3_ptm=design["af3_ptm"],
                af3_plddt=design["af3_plddt"],
                binder_chain=binder_chain,
                target_chain=target_chain,
                apo_structure=None,  # No APO for this test
                af3_confidence_json=None,  # No confidence data
                target_type=target_type,
            )
            opensource_time = time.time() - t0
            
            # Save relaxed PDB
            if opensource_result.get("relaxed_pdb"):
                pdb_path = opensource_pdb_dir / f"{design_id}_opensource_relaxed.pdb"
                with open(pdb_path, "w") as f:
                    f.write(opensource_result["relaxed_pdb"])
            
            status = "ACCEPTED" if opensource_result.get("accepted") else "REJECTED"
            print(f"  ✓ Open-source ({opensource_time:.1f}s): SC={opensource_result.get('interface_sc', 0):.3f}, "
                  f"dSASA={opensource_result.get('interface_dSASA', 0):.1f} → {status}")
            
        except Exception as e:
            print(f"  ✗ Open-source failed: {e}")
            opensource_result = {"error": str(e)}
            opensource_time = time.time() - t0
        
        # ========== PYROSETTA SCORING (optional) ==========
        pyrosetta_result = None
        pyrosetta_time = 0
        
        if compare_pyrosetta:
            print(f"  Running PyRosetta scoring...")
            t0 = time.time()
            try:
                pyrosetta_result = run_pyrosetta_single.remote(
                    design_id=design_id,
                    af3_structure=cif_content,
                    af3_iptm=design["af3_iptm"],
                    af3_ptm=design["af3_ptm"],
                    af3_plddt=design["af3_plddt"],
                    binder_chain=binder_chain,
                    target_chain=target_chain,
                    apo_structure=None,
                    af3_confidence_json=None,
                    target_type=target_type,
                )
                pyrosetta_time = time.time() - t0
                
                # Save relaxed PDB
                if pyrosetta_result.get("relaxed_pdb"):
                    pdb_path = pyrosetta_pdb_dir / f"{design_id}_pyrosetta_relaxed.pdb"
                    with open(pdb_path, "w") as f:
                        f.write(pyrosetta_result["relaxed_pdb"])
                
                status = "ACCEPTED" if pyrosetta_result.get("accepted") else "REJECTED"
                print(f"  ✓ PyRosetta ({pyrosetta_time:.1f}s): dG={pyrosetta_result.get('interface_dG', 0):.1f}, "
                      f"SC={pyrosetta_result.get('interface_sc', 0):.3f} → {status}")
                
            except Exception as e:
                print(f"  ✗ PyRosetta failed: {e}")
                pyrosetta_result = {"error": str(e)}
                pyrosetta_time = time.time() - t0
        
        # ========== BUILD COMPARISON ==========
        result_entry = {
            "design_id": design_id,
            "af3_iptm": design["af3_iptm"],
            "af3_ptm": design["af3_ptm"],
            "af3_plddt": design["af3_plddt"],
            "opensource_time": opensource_time,
            "pyrosetta_time": pyrosetta_time if compare_pyrosetta else None,
        }
        
        for metric in METRICS:
            os_val = opensource_result.get(metric)
            pr_val = pyrosetta_result.get(metric) if pyrosetta_result else None
            
            result_entry[f"opensource_{metric}"] = os_val
            if compare_pyrosetta:
                result_entry[f"pyrosetta_{metric}"] = pr_val
            
            # Add to row-based comparison
            comparison_rows.append(format_comparison_row(
                design_id, metric, os_val, pr_val
            ))
        
        all_results.append(result_entry)
    
    # ========== SAVE RESULTS ==========
    print("\n" + "=" * 70)
    print("SAVING RESULTS")
    print("=" * 70)
    
    # Save detailed results CSV
    if all_results:
        results_csv = output_path / "detailed_results.csv"
        fieldnames = list(all_results[0].keys())
        with open(results_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_results)
        print(f"  Saved: {results_csv}")
    
    # Save comparison CSV (row-based for easy analysis)
    if comparison_rows:
        comparison_csv = output_path / "comparison_results.csv"
        fieldnames = ["design_id", "metric", "opensource", "pyrosetta", "diff"]
        with open(comparison_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(comparison_rows)
        print(f"  Saved: {comparison_csv}")
    
    # ========== SUMMARY ==========
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    # Count acceptances
    os_accepted = sum(1 for r in all_results if r.get("opensource_accepted"))
    os_rejected = len(all_results) - os_accepted
    
    print(f"\nOpen-source scoring:")
    print(f"  Accepted: {os_accepted}")
    print(f"  Rejected: {os_rejected}")
    
    if compare_pyrosetta:
        pr_accepted = sum(1 for r in all_results if r.get("pyrosetta_accepted"))
        pr_rejected = len(all_results) - pr_accepted
        
        print(f"\nPyRosetta scoring:")
        print(f"  Accepted: {pr_accepted}")
        print(f"  Rejected: {pr_rejected}")
        
        # Agreement stats
        agree = sum(
            1 for r in all_results 
            if r.get("opensource_accepted") == r.get("pyrosetta_accepted")
        )
        print(f"\nAgreement: {agree}/{len(all_results)} ({100*agree/len(all_results):.1f}%)")
    
    # Average timing
    avg_os_time = sum(r.get("opensource_time", 0) for r in all_results) / len(all_results)
    print(f"\nAverage open-source time: {avg_os_time:.1f}s")
    
    if compare_pyrosetta:
        avg_pr_time = sum(r.get("pyrosetta_time", 0) or 0 for r in all_results) / len(all_results)
        print(f"Average PyRosetta time: {avg_pr_time:.1f}s")
    
    print("\n" + "=" * 70)
    print(f"Results saved to: {output_path}")
    print("=" * 70 + "\n")


# =============================================================================
# QUICK TEST FUNCTION (for single structure)
# =============================================================================

@app.local_entrypoint()
def test_single(
    cif_path: str,
    af3_iptm: float = 0.8,
    af3_ptm: float = 0.8,
    af3_plddt: float = 85.0,
    open_scoring_gpu: str = "A10G",
    compare_pyrosetta: bool = True,
):
    """
    Quick test of scoring on a single CIF file.
    
    Example:
        modal run modal_boltz_ph/test_scoring.py::test_single \\
            --cif-path ./modal_results/PDL1_open_test3/af3_validation/PDL1_open_test3_d1_c5_af3.cif \\
            --compare-pyrosetta true
    """
    print("\n" + "=" * 70)
    print("SINGLE STRUCTURE SCORING TEST")
    print("=" * 70)
    
    cif_file = Path(cif_path)
    if not cif_file.exists():
        raise FileNotFoundError(f"CIF file not found: {cif_path}")
    
    design_id = cif_file.stem.replace("_af3", "")
    
    with open(cif_file, "r") as f:
        cif_content = f.read()
    
    print(f"Design ID: {design_id}")
    print(f"AF3 ipTM: {af3_iptm}")
    print(f"AF3 pTM: {af3_ptm}")
    print(f"AF3 pLDDT: {af3_plddt}")
    print("=" * 70)
    
    # Get scoring function
    opensource_fn = OPENSOURCE_SCORING_GPU_FUNCTIONS.get(
        open_scoring_gpu,
        OPENSOURCE_SCORING_GPU_FUNCTIONS[DEFAULT_OPENSOURCE_GPU]
    )
    
    # Run open-source scoring
    print(f"\nRunning open-source scoring ({open_scoring_gpu})...")
    t0 = time.time()
    opensource_result = opensource_fn.remote(
        design_id=design_id,
        af3_structure=cif_content,
        af3_iptm=af3_iptm,
        af3_ptm=af3_ptm,
        af3_plddt=af3_plddt,
        binder_chain="A",
        target_chain="B",
        apo_structure=None,
        af3_confidence_json=None,
        target_type="protein",
    )
    opensource_time = time.time() - t0
    
    print(f"\n--- Open-source Results ({opensource_time:.1f}s) ---")
    for key in ["accepted", "rejection_reason", "interface_sc", "interface_dG", 
                "interface_dSASA", "interface_nres", "surface_hydrophobicity", 
                "interface_hydrophobicity", "binder_score", "rg"]:
        print(f"  {key}: {opensource_result.get(key)}")
    
    # Run PyRosetta scoring (if requested)
    if compare_pyrosetta:
        print(f"\nRunning PyRosetta scoring...")
        t0 = time.time()
        pyrosetta_result = run_pyrosetta_single.remote(
            design_id=design_id,
            af3_structure=cif_content,
            af3_iptm=af3_iptm,
            af3_ptm=af3_ptm,
            af3_plddt=af3_plddt,
            binder_chain="A",
            target_chain="B",
            apo_structure=None,
            af3_confidence_json=None,
            target_type="protein",
        )
        pyrosetta_time = time.time() - t0
        
        print(f"\n--- PyRosetta Results ({pyrosetta_time:.1f}s) ---")
        for key in ["accepted", "rejection_reason", "interface_sc", "interface_dG", 
                    "interface_dSASA", "interface_nres", "surface_hydrophobicity", 
                    "interface_hydrophobicity", "binder_score", "rg"]:
            print(f"  {key}: {pyrosetta_result.get(key)}")
        
        # Quick comparison
        print("\n--- Comparison ---")
        for key in ["interface_sc", "interface_dG", "interface_dSASA", "interface_nres"]:
            os_val = opensource_result.get(key)
            pr_val = pyrosetta_result.get(key)
            diff = None
            if os_val is not None and pr_val is not None:
                try:
                    diff = float(os_val) - float(pr_val)
                except (ValueError, TypeError):
                    pass
            print(f"  {key}: opensource={os_val}, pyrosetta={pr_val}, diff={diff}")
    
    print("\n" + "=" * 70 + "\n")


# =============================================================================
# SASA COMPARISON TEST
# =============================================================================

@app.local_entrypoint()
def test_sasa(
    input_dir: str,
    output_dir: str = "./sasa_comparison_results",
    binder_chain: str = "A",
    target_chain: str = "B",
    max_designs: int = 0,
):
    """
    Compare Biopython vs FreeSASA SASA calculations.
    
    This entrypoint runs both SASA methods on relaxed structures and outputs
    a detailed comparison to help validate/debug SASA calculations.
    
    Args:
        input_dir: Path to af3_validation directory from a prior run
        output_dir: Where to save comparison results
        binder_chain: Chain ID of binder (default: A)
        target_chain: Chain ID of target (default: B)
        max_designs: Maximum designs to process (0 = all)
    
    Example:
        modal run modal_boltz_ph/test_scoring.py::test_sasa \\
            --input-dir ./modal_results/PDL1_open_test5/af3_validation \\
            --output-dir ./sasa_comparison_results
    """
    print("\n" + "=" * 70)
    print("SASA COMPARISON: Biopython vs FreeSASA")
    print("=" * 70)
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Binder chain: {binder_chain}")
    print(f"Target chain: {target_chain}")
    print("=" * 70 + "\n")
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load AF3 results
    print("Loading AF3 results...")
    designs = load_af3_results(input_dir)
    
    if max_designs > 0:
        designs = designs[:max_designs]
    
    print(f"Found {len(designs)} designs to process\n")
    
    all_results = []
    
    for i, design in enumerate(designs):
        design_id = design["design_id"]
        print(f"\n[{i+1}/{len(designs)}] Processing {design_id}")
        print("-" * 50)
        
        # Load CIF content
        with open(design["cif_path"], "r") as f:
            cif_content = f.read()
        
        t0 = time.time()
        try:
            result = compare_sasa_methods.remote(
                design_id=design_id,
                af3_structure=cif_content,
                binder_chain=binder_chain,
                target_chain=target_chain,
            )
            elapsed = time.time() - t0
            
            if "error" in result:
                print(f"  ✗ Error: {result['error']}")
                continue
            
            bp = result.get("biopython", {})
            fs = result.get("freesasa", {})
            comp = result.get("comparison", {})
            
            print(f"  ✓ Completed in {elapsed:.1f}s (relax: {result.get('relax_time', 0):.1f}s)")
            print(f"  ")
            print(f"  {'Metric':<25} {'Biopython':>12} {'FreeSASA':>12} {'Diff':>12} {'%Diff':>10}")
            print(f"  {'-'*71}")
            print(f"  {'surface_hydrophobicity':<25} {bp.get('surface_hydrophobicity', 'N/A'):>12} {fs.get('surface_hydrophobicity', 'N/A'):>12} {comp.get('surface_hydrophobicity_diff', 'N/A'):>12}")
            print(f"  {'binder_sasa_mono':<25} {bp.get('binder_sasa_mono', 'N/A'):>12.1f} {fs.get('binder_sasa_mono', 'N/A'):>12.1f} {comp.get('binder_sasa_mono_diff', 'N/A'):>12.1f} {comp.get('binder_sasa_mono_pct_diff', 'N/A'):>9.1f}%")
            print(f"  {'target_sasa_mono':<25} {bp.get('target_sasa_mono', 'N/A'):>12.1f} {fs.get('target_sasa_mono', 'N/A'):>12.1f} {comp.get('target_sasa_mono_diff', 'N/A'):>12.1f}")
            print(f"  {'total_dSASA':<25} {bp.get('total_dSASA', 'N/A'):>12.1f} {fs.get('total_dSASA', 'N/A'):>12.1f} {comp.get('total_dSASA_diff', 'N/A'):>12.1f} {comp.get('total_dSASA_pct_diff', 'N/A'):>9.1f}%")
            
            # Flatten for CSV
            row = {"design_id": design_id, "relax_time": result.get("relax_time")}
            for key, val in bp.items():
                row[f"biopython_{key}"] = val
            for key, val in fs.items():
                row[f"freesasa_{key}"] = val
            for key, val in comp.items():
                row[f"comp_{key}"] = val
            all_results.append(row)
            
        except Exception as e:
            print(f"  ✗ Failed: {e}")
    
    # Save results
    if all_results:
        results_csv = output_path / "sasa_comparison.csv"
        fieldnames = list(all_results[0].keys())
        with open(results_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_results)
        print(f"\n\nSaved: {results_csv}")
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    if all_results:
        avg_binder_diff = sum(r.get("comp_binder_sasa_mono_pct_diff", 0) or 0 for r in all_results) / len(all_results)
        avg_dsasa_diff = sum(r.get("comp_total_dSASA_pct_diff", 0) or 0 for r in all_results) / len(all_results)
        
        print(f"\nAverage binder_sasa_mono % difference: {avg_binder_diff:.2f}%")
        print(f"Average total_dSASA % difference: {avg_dsasa_diff:.2f}%")
    
    print("\n" + "=" * 70 + "\n")
