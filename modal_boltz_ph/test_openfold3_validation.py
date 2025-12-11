"""
Test harness for OpenFold3 validation on existing design structures.

This script allows isolated testing of the OpenFold3 validation pipeline using
existing Boltz design outputs, without running the full design pipeline.

OpenFold3 is an Apache 2.0 licensed reproduction of AlphaFold3 from the
AlQuraishi Lab at Columbia University.

Usage:
    # Test a single PDB file
    modal run modal_boltz_ph/test_openfold3_validation.py::test_single \
        --pdb-path ./modal_results/test_run/best_designs/design_001.pdb \
        --validation-gpu A100

    # Test all designs in a folder
    modal run modal_boltz_ph/test_openfold3_validation.py::test_folder \
        --input-dir ./modal_results/test_run/best_designs \
        --output-dir ./openfold3_validation_test \
        --validation-gpu A100

Input format:
    - PDB files with designed binder (chain A) and target chains (B, C, D, ...)
    - Target MSAs will be automatically fetched and cached

Output:
    - validation_results.csv: OpenFold3 validation metrics for each design
    - msas/: Cached MSA files for target sequences (reused on subsequent runs)
    - structures/: Predicted structures from OpenFold3 (PDB format)
"""

import csv
import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import modal

from modal_boltz_ph.app import app
from modal_boltz_ph.cache import precompute_msas
from modal_boltz_ph.validation_openfold3 import (
    OPENFOLD3_GPU_FUNCTIONS,
    DEFAULT_OPENFOLD3_GPU,
)

# Reuse utilities from test_protenix_validation (avoid code duplication)
from modal_boltz_ph.test_protenix_validation import (
    extract_sequences_from_pdb,
    parse_design_pdbs,
    load_cached_msas,
    save_cached_msas,
    get_or_fetch_msas,
)


# =============================================================================
# MODAL LOCAL ENTRYPOINTS
# =============================================================================

@app.local_entrypoint()
def test_single_of3(
    pdb_path: str,
    validation_gpu: str = "A100",
    output_dir: str = "./openfold3_single_test",
    run_scoring: bool = True,
    verbose: bool = True,
):
    """
    Test OpenFold3 validation on a single PDB file.
    
    Chain A is treated as the binder (no MSA).
    Chains B, C, D, etc. are treated as targets (MSAs fetched).
    
    Args:
        pdb_path: Path to the design PDB file
        validation_gpu: GPU type for OpenFold3 validation
        output_dir: Where to save results and MSA cache
        run_scoring: Also run open-source scoring (default: True)
        verbose: Enable detailed output
    
    Example:
        modal run modal_boltz_ph/test_openfold3_validation.py::test_single_of3 \\
            --pdb-path ./modal_results/test_run/best_designs/design_001.pdb
    """
    print("\n" + "=" * 70)
    print("OPENFOLD3 VALIDATION - SINGLE DESIGN TEST")
    print("=" * 70)
    
    pdb_file = Path(pdb_path)
    if not pdb_file.exists():
        raise FileNotFoundError(f"PDB file not found: {pdb_path}")
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    msa_cache_dir = output_path / "msas"
    
    # Extract sequences from PDB
    print(f"\nPDB file: {pdb_path}")
    sequences = extract_sequences_from_pdb(pdb_path)
    
    if 'A' not in sequences:
        raise ValueError("No chain A (binder) found in PDB")
    
    binder_seq = sequences['A']
    target_chains = [(cid, seq) for cid, seq in sorted(sequences.items()) if cid != 'A']
    target_seq = ":".join(seq for _, seq in target_chains)
    
    print(f"\nChains detected:")
    print(f"  A (binder): {len(binder_seq)} residues")
    for chain_id, seq in target_chains:
        print(f"  {chain_id} (target): {len(seq)} residues")
    
    print(f"\nGPU: {validation_gpu}")
    print(f"Run scoring: {run_scoring}")
    print("=" * 70)
    
    # Get MSAs for target sequences
    unique_target_seqs = list(set(seq for _, seq in target_chains))
    seq_to_msa = get_or_fetch_msas(unique_target_seqs, msa_cache_dir)
    
    # Convert sequence-keyed MSAs to chain_id-keyed MSAs
    target_msas = {}
    for chain_id, seq in target_chains:
        if seq in seq_to_msa:
            target_msas[chain_id] = seq_to_msa[seq]
    
    # Get validation function
    validation_fn = OPENFOLD3_GPU_FUNCTIONS.get(
        validation_gpu,
        OPENFOLD3_GPU_FUNCTIONS[DEFAULT_OPENFOLD3_GPU]
    )
    
    # Run validation
    design_id = pdb_file.stem
    print(f"\nRunning OpenFold3 validation for {design_id}...")
    
    t0 = time.time()
    result = validation_fn.remote(
        design_id=design_id,
        binder_seq=binder_seq,
        target_seq=target_seq,
        target_msas=target_msas,
        run_scoring=run_scoring,
        verbose=verbose,
    )
    elapsed = time.time() - t0
    
    # Display results
    print("\n" + "=" * 70)
    print(f"RESULTS ({elapsed:.1f}s)")
    print("=" * 70)
    
    if "error" in result:
        print(f"\n✗ Error: {result['error']}")
        return
    
    # Core metrics
    print(f"\n--- OpenFold3 Confidence ---")
    print(f"  ipTM: {result.get('of3_iptm', 0):.4f}")
    print(f"  pTM: {result.get('of3_ptm', 0):.4f}")
    print(f"  pLDDT: {result.get('of3_plddt', 0):.2f}")
    print(f"  ipSAE: {result.get('of3_ipsae', 0):.4f}")
    print(f"  gPDE: {result.get('of3_gpde', 0):.4f}")
    print(f"  Ranking score: {result.get('ranking_score', 0):.4f}")
    print(f"  Has clash: {result.get('has_clash', 'N/A')}")
    
    if run_scoring and result.get('interface_sc') is not None:
        print(f"\n--- Interface Scoring ---")
        print(f"  Shape Comp (SC): {result.get('interface_sc', 0):.4f}")
        print(f"  dG: {result.get('interface_dG', 0):.2f}")
        print(f"  dSASA: {result.get('interface_dSASA', 0):.1f}")
        print(f"  Interface residues: {result.get('interface_nres', 0)}")
        print(f"  Surface hydrophobicity: {result.get('surface_hydrophobicity', 0):.4f}")
        
        status = "ACCEPTED" if result.get('accepted') else f"REJECTED ({result.get('rejection_reason', 'unknown')})"
        print(f"\n  Status: {status}")
    
    # Save structure (OpenFold3 outputs CIF format)
    if result.get('holo_structure'):
        struct_path = output_path / f"{design_id}_openfold3.cif"
        struct_path.write_text(result['holo_structure'])
        print(f"\n✓ Saved structure: {struct_path}")
    
    # Save results JSON
    results_json = output_path / f"{design_id}_results.json"
    # Remove large structure strings for JSON
    json_result = {k: v for k, v in result.items() 
                   if not k.endswith('_structure') and not k.endswith('_json')}
    with open(results_json, 'w') as f:
        json.dump(json_result, f, indent=2)
    print(f"✓ Saved metrics: {results_json}")
    
    print("\n" + "=" * 70 + "\n")


@app.local_entrypoint()
def test_folder_of3(
    input_dir: str,
    output_dir: str = "./openfold3_validation_test",
    validation_gpu: str = "A100",
    run_scoring: bool = True,
    max_designs: int = 0,
    verbose: bool = False,
):
    """
    Test OpenFold3 validation on all PDB files in a folder.
    
    Chain A is treated as the binder (no MSA).
    Chains B, C, D, etc. are treated as targets (MSAs fetched once and cached).
    
    Args:
        input_dir: Directory containing design PDB files
        output_dir: Where to save results and MSA cache
        validation_gpu: GPU type for OpenFold3 validation
        run_scoring: Also run open-source scoring (default: True)
        max_designs: Maximum designs to process (0 = all)
        verbose: Enable detailed output per design
    
    Example:
        modal run modal_boltz_ph/test_openfold3_validation.py::test_folder_of3 \\
            --input-dir ./modal_results/test_run/best_designs \\
            --output-dir ./openfold3_validation_test \\
            --validation-gpu A100
    """
    print("\n" + "=" * 70)
    print("OPENFOLD3 VALIDATION - FOLDER TEST")
    print("=" * 70)
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Validation GPU: {validation_gpu}")
    print(f"Run scoring: {run_scoring}")
    print(f"Max designs: {max_designs if max_designs > 0 else 'all'}")
    print("=" * 70)
    
    # Setup output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    msa_cache_dir = output_path / "msas"
    structures_dir = output_path / "structures"
    structures_dir.mkdir(exist_ok=True)
    
    # Parse all designs
    print("\nParsing design PDB files...")
    designs = parse_design_pdbs(input_dir)
    
    if max_designs > 0:
        designs = designs[:max_designs]
    
    print(f"Found {len(designs)} design(s) to process\n")
    
    if not designs:
        print("No valid designs found!")
        return
    
    # Collect all unique target sequences for MSA fetching
    all_target_seqs = set()
    for design in designs:
        for _, seq in design["target_chains"]:
            all_target_seqs.add(seq)
    
    print(f"Target sequences: {len(all_target_seqs)} unique sequence(s)")
    for i, seq in enumerate(all_target_seqs):
        print(f"  Target {i+1}: {len(seq)} residues")
    
    # Fetch all MSAs upfront (keyed by sequence)
    seq_to_msa = get_or_fetch_msas(list(all_target_seqs), msa_cache_dir)
    
    # Get validation function
    validation_fn = OPENFOLD3_GPU_FUNCTIONS.get(
        validation_gpu,
        OPENFOLD3_GPU_FUNCTIONS[DEFAULT_OPENFOLD3_GPU]
    )
    
    # Process each design
    all_results = []
    
    for i, design in enumerate(designs):
        design_id = design["design_id"]
        print(f"\n[{i+1}/{len(designs)}] Processing {design_id}")
        print("-" * 50)
        print(f"  Binder: {len(design['binder_seq'])} residues")
        print(f"  Targets: {', '.join(f'{cid}={len(seq)}' for cid, seq in design['target_chains'])}")
        
        # Convert sequence-keyed MSAs to chain_id-keyed MSAs for this design
        design_target_msas = {}
        for chain_id, seq in design["target_chains"]:
            if seq in seq_to_msa:
                design_target_msas[chain_id] = seq_to_msa[seq]
        
        t0 = time.time()
        try:
            result = validation_fn.remote(
                design_id=design_id,
                binder_seq=design["binder_seq"],
                target_seq=design["target_seq"],
                target_msas=design_target_msas,
                run_scoring=run_scoring,
                verbose=verbose,
            )
            elapsed = time.time() - t0
            
            if "error" in result:
                print(f"  ✗ Error ({elapsed:.1f}s): {result['error']}")
                result_entry = {
                    "design_id": design_id,
                    "binder_length": len(design["binder_seq"]),
                    "error": result["error"],
                }
            else:
                # Success
                iptm = result.get('of3_iptm', 0)
                plddt = result.get('of3_plddt', 0)
                sc = result.get('interface_sc', 0) if run_scoring else None
                
                status_str = ""
                if run_scoring:
                    status_str = " → ACCEPTED" if result.get('accepted') else f" → REJECTED ({result.get('rejection_reason', '?')})"
                
                print(f"  ✓ Success ({elapsed:.1f}s): ipTM={iptm:.3f}, pLDDT={plddt:.1f}" +
                      (f", SC={sc:.3f}" if sc else "") + status_str)
                
                # Save structure (OpenFold3 outputs CIF format)
                if result.get('holo_structure'):
                    struct_path = structures_dir / f"{design_id}_openfold3.cif"
                    struct_path.write_text(result['holo_structure'])
                
                result_entry = {
                    "design_id": design_id,
                    "binder_length": len(design["binder_seq"]),
                    "binder_seq": design["binder_seq"],
                    "of3_iptm": result.get('of3_iptm'),
                    "of3_ptm": result.get('of3_ptm'),
                    "of3_plddt": result.get('of3_plddt'),
                    "of3_ipsae": result.get('of3_ipsae'),
                    "of3_gpde": result.get('of3_gpde'),
                    "ranking_score": result.get('ranking_score'),
                    "has_clash": result.get('has_clash'),
                    "elapsed_time": elapsed,
                }
                
                if run_scoring:
                    result_entry.update({
                        "accepted": result.get('accepted'),
                        "rejection_reason": result.get('rejection_reason'),
                        "interface_sc": result.get('interface_sc'),
                        "interface_dG": result.get('interface_dG'),
                        "interface_dSASA": result.get('interface_dSASA'),
                        "interface_nres": result.get('interface_nres'),
                        "interface_hbonds": result.get('interface_hbonds'),
                        "surface_hydrophobicity": result.get('surface_hydrophobicity'),
                        "binder_score": result.get('binder_score'),
                        "apo_holo_rmsd": result.get('apo_holo_rmsd'),
                    })
            
            all_results.append(result_entry)
            
        except Exception as e:
            elapsed = time.time() - t0
            print(f"  ✗ Exception ({elapsed:.1f}s): {e}")
            all_results.append({
                "design_id": design_id,
                "binder_length": len(design["binder_seq"]),
                "error": str(e),
            })
    
    # Save results CSV
    print("\n" + "=" * 70)
    print("SAVING RESULTS")
    print("=" * 70)
    
    if all_results:
        results_csv = output_path / "validation_results.csv"
        fieldnames = list(all_results[0].keys())
        # Ensure consistent columns even if some entries have more fields
        for r in all_results:
            for k in r.keys():
                if k not in fieldnames:
                    fieldnames.append(k)
        
        with open(results_csv, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_results)
        print(f"  ✓ Saved: {results_csv}")
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    successful = [r for r in all_results if 'of3_iptm' in r and r.get('of3_iptm') is not None]
    failed = len(all_results) - len(successful)
    
    print(f"\nTotal designs: {len(all_results)}")
    print(f"Successful: {len(successful)}")
    print(f"Failed: {failed}")
    
    if successful:
        avg_iptm = sum(r['of3_iptm'] for r in successful) / len(successful)
        avg_plddt = sum(r['of3_plddt'] for r in successful) / len(successful)
        avg_time = sum(r.get('elapsed_time', 0) for r in successful) / len(successful)
        
        print(f"\nAverage metrics:")
        print(f"  ipTM: {avg_iptm:.3f}")
        print(f"  pLDDT: {avg_plddt:.1f}")
        print(f"  Time: {avg_time:.1f}s")
        
        if run_scoring:
            accepted = sum(1 for r in successful if r.get('accepted'))
            print(f"\nScoring:")
            print(f"  Accepted: {accepted}/{len(successful)}")
            
            sc_values = [r.get('interface_sc') for r in successful if r.get('interface_sc') is not None]
            if sc_values:
                print(f"  Avg SC: {sum(sc_values)/len(sc_values):.3f}")
    
    print("\n" + "=" * 70)
    print(f"Results saved to: {output_path}")
    print("=" * 70 + "\n")


# =============================================================================
# UTILITY: COMPARE VALIDATION MODELS
# =============================================================================

@app.local_entrypoint()
def compare_of3_protenix(
    protenix_results: str,
    openfold3_results: str,
    output_file: str = "./validation_model_comparison.csv",
):
    """
    Compare Protenix vs OpenFold3 validation results.
    
    Useful for benchmarking the two open-source AF3 reproductions.
    
    Args:
        protenix_results: Path to Protenix validation_results.csv
        openfold3_results: Path to OpenFold3 validation_results.csv
        output_file: Path to save comparison CSV
    
    Example:
        modal run modal_boltz_ph/test_openfold3_validation.py::compare_of3_protenix \\
            --protenix-results ./protenix_test/validation_results.csv \\
            --openfold3-results ./openfold3_test/validation_results.csv
    """
    import pandas as pd
    
    print("\n" + "=" * 70)
    print("COMPARING PROTENIX vs OPENFOLD3")
    print("=" * 70)
    
    protenix_df = pd.read_csv(protenix_results)
    of3_df = pd.read_csv(openfold3_results)
    
    print(f"\nProtenix results: {len(protenix_df)} designs")
    print(f"OpenFold3 results: {len(of3_df)} designs")
    
    # Merge on design_id
    merged = protenix_df.merge(
        of3_df[['design_id', 'of3_iptm', 'of3_ptm', 'of3_plddt', 'of3_ipsae']],
        on='design_id',
        how='inner',
    )
    
    print(f"Matched designs: {len(merged)}")
    
    if len(merged) == 0:
        print("\nNo matching designs found!")
        return
    
    # Calculate differences
    if 'protenix_iptm' in merged.columns:
        merged['iptm_diff'] = merged['of3_iptm'] - merged['protenix_iptm']
        merged['plddt_diff'] = merged['of3_plddt'] - merged['protenix_plddt']
    
    # Save comparison
    merged.to_csv(output_file, index=False)
    print(f"\n✓ Saved comparison: {output_file}")
    
    if len(merged) > 2 and 'protenix_iptm' in merged.columns:
        # Calculate correlation
        iptm_corr = merged['protenix_iptm'].corr(merged['of3_iptm'])
        plddt_corr = merged['protenix_plddt'].corr(merged['of3_plddt'])
        print(f"\n--- Correlation ---")
        print(f"  ipTM correlation: {iptm_corr:.3f}")
        print(f"  pLDDT correlation: {plddt_corr:.3f}")
        
        print("\n--- Average Differences (OpenFold3 - Protenix) ---")
        print(f"  ipTM: {merged['iptm_diff'].mean():+.4f} (std: {merged['iptm_diff'].std():.4f})")
        print(f"  pLDDT: {merged['plddt_diff'].mean():+.2f} (std: {merged['plddt_diff'].std():.2f})")
    
    print("\n" + "=" * 70 + "\n")


@app.local_entrypoint()
def compare_of3_af3(
    openfold3_results: str,
    af3_results: str,
    output_file: str = "./of3_vs_af3_comparison.csv",
):
    """
    Compare OpenFold3 validation results with prior AF3 validation results.
    
    Args:
        openfold3_results: Path to OpenFold3 validation_results.csv
        af3_results: Path to AF3 af3_results.csv
        output_file: Path to save comparison CSV
    
    Example:
        modal run modal_boltz_ph/test_openfold3_validation.py::compare_of3_af3 \\
            --openfold3-results ./openfold3_test/validation_results.csv \\
            --af3-results ./modal_results/test_run/refolded/validation_results.csv
    """
    import pandas as pd
    
    print("\n" + "=" * 70)
    print("COMPARING OPENFOLD3 vs AF3 VALIDATION")
    print("=" * 70)
    
    # Load results
    of3_df = pd.read_csv(openfold3_results)
    af3_df = pd.read_csv(af3_results)
    
    print(f"\nOpenFold3 results: {len(of3_df)} designs")
    print(f"AF3 results: {len(af3_df)} designs")
    
    # Merge on design_id
    merged = of3_df.merge(
        af3_df[['design_id', 'af3_iptm', 'af3_ptm', 'af3_plddt', 'af3_ipsae']],
        on='design_id',
        how='inner',
        suffixes=('', '_af3_orig'),
    )
    
    print(f"Matched designs: {len(merged)}")
    
    if len(merged) == 0:
        print("\nNo matching designs found!")
        return
    
    # Calculate differences
    merged['iptm_diff'] = merged['of3_iptm'] - merged['af3_iptm']
    merged['plddt_diff'] = merged['of3_plddt'] - merged['af3_plddt']
    
    # Save comparison
    merged.to_csv(output_file, index=False)
    print(f"\n✓ Saved comparison: {output_file}")
    
    # Summary stats
    print("\n--- Correlation ---")
    if len(merged) > 2:
        iptm_corr = merged['of3_iptm'].corr(merged['af3_iptm'])
        plddt_corr = merged['of3_plddt'].corr(merged['af3_plddt'])
        print(f"  ipTM correlation: {iptm_corr:.3f}")
        print(f"  pLDDT correlation: {plddt_corr:.3f}")
    
    print("\n--- Average Differences (OpenFold3 - AF3) ---")
    print(f"  ipTM: {merged['iptm_diff'].mean():+.4f} (std: {merged['iptm_diff'].std():.4f})")
    print(f"  pLDDT: {merged['plddt_diff'].mean():+.2f} (std: {merged['plddt_diff'].std():.2f})")
    
    print("\n" + "=" * 70 + "\n")
