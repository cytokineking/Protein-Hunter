#!/usr/bin/env python
"""
Test harness for local open-source scoring.

This script allows isolated testing of the open-source scoring pipeline
(OpenMM relaxation + FreeSASA + sc-rs) using existing structures,
without running the full Boltz → Protenix pipeline.

Usage:
    # Quick synthetic test
    python -m boltz_ph.tests.test_scoring_local --synthetic
    
    # Test a single CIF/PDB file
    python -m boltz_ph.tests.test_scoring_local --structure-path ./refolded/design_001.cif
    
    # Test all structures in a folder
    python -m boltz_ph.tests.test_scoring_local --input-dir ./refolded --output-dir ./scoring_test

Input format:
    - CIF or PDB files with binder (chain A) and target (chain B)

Output:
    - scoring_results.csv: Interface metrics for each structure
    - relaxed/: Relaxed PDB structures from OpenMM
"""

import argparse
import csv
import json
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

# Ensure boltz_ph is importable
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


# =============================================================================
# TEST DATA
# =============================================================================

# A small synthetic PDB for testing (two short helices)
SYNTHETIC_PDB = """HEADER    TEST STRUCTURE
ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00  0.00           N
ATOM      2  CA  ALA A   1       1.458   0.000   0.000  1.00  0.00           C
ATOM      3  C   ALA A   1       2.009   1.420   0.000  1.00  0.00           C
ATOM      4  O   ALA A   1       1.251   2.400   0.000  1.00  0.00           O
ATOM      5  CB  ALA A   1       1.986  -0.751   1.223  1.00  0.00           C
ATOM      6  N   GLU A   2       3.320   1.550   0.000  1.00  0.00           N
ATOM      7  CA  GLU A   2       3.950   2.870   0.000  1.00  0.00           C
ATOM      8  C   GLU A   2       5.470   2.800   0.000  1.00  0.00           C
ATOM      9  O   GLU A   2       6.080   1.730   0.000  1.00  0.00           O
ATOM     10  CB  GLU A   2       3.520   3.730   1.190  1.00  0.00           C
ATOM     11  CG  GLU A   2       3.960   5.190   1.140  1.00  0.00           C
ATOM     12  CD  GLU A   2       3.530   6.000   2.350  1.00  0.00           C
ATOM     13  OE1 GLU A   2       2.310   6.300   2.450  1.00  0.00           O
ATOM     14  OE2 GLU A   2       4.370   6.340   3.220  1.00  0.00           O
ATOM     15  N   LYS A   3       6.070   3.980   0.000  1.00  0.00           N
ATOM     16  CA  LYS A   3       7.520   4.100   0.000  1.00  0.00           C
ATOM     17  C   LYS A   3       8.070   5.510   0.000  1.00  0.00           C
ATOM     18  O   LYS A   3       7.310   6.490   0.000  1.00  0.00           O
ATOM     19  CB  LYS A   3       8.050   3.350   1.220  1.00  0.00           C
ATOM     20  N   ALA A   4       9.380   5.640   0.000  1.00  0.00           N
ATOM     21  CA  ALA A   4      10.010   6.960   0.000  1.00  0.00           C
ATOM     22  C   ALA A   4      11.530   6.890   0.000  1.00  0.00           C
ATOM     23  O   ALA A   4      12.140   5.820   0.000  1.00  0.00           O
ATOM     24  CB  ALA A   4       9.580   7.820   1.190  1.00  0.00           C
TER
ATOM     25  N   MET B   1      15.000   0.000   0.000  1.00  0.00           N
ATOM     26  CA  MET B   1      16.458   0.000   0.000  1.00  0.00           C
ATOM     27  C   MET B   1      17.009   1.420   0.000  1.00  0.00           C
ATOM     28  O   MET B   1      16.251   2.400   0.000  1.00  0.00           O
ATOM     29  CB  MET B   1      16.986  -0.751   1.223  1.00  0.00           C
ATOM     30  N   LYS B   2      18.320   1.550   0.000  1.00  0.00           N
ATOM     31  CA  LYS B   2      18.950   2.870   0.000  1.00  0.00           C
ATOM     32  C   LYS B   2      20.470   2.800   0.000  1.00  0.00           C
ATOM     33  O   LYS B   2      21.080   1.730   0.000  1.00  0.00           O
ATOM     34  CB  LYS B   2      18.520   3.730   1.190  1.00  0.00           C
ATOM     35  N   GLU B   3      21.070   3.980   0.000  1.00  0.00           N
ATOM     36  CA  GLU B   3      22.520   4.100   0.000  1.00  0.00           C
ATOM     37  C   GLU B   3      23.070   5.510   0.000  1.00  0.00           C
ATOM     38  O   GLU B   3      22.310   6.490   0.000  1.00  0.00           O
ATOM     39  CB  GLU B   3      23.050   3.350   1.220  1.00  0.00           C
ATOM     40  N   ALA B   4      24.380   5.640   0.000  1.00  0.00           N
ATOM     41  CA  ALA B   4      25.010   6.960   0.000  1.00  0.00           C
ATOM     42  C   ALA B   4      26.530   6.890   0.000  1.00  0.00           C
ATOM     43  O   ALA B   4      27.140   5.820   0.000  1.00  0.00           O
ATOM     44  CB  ALA B   4      24.580   7.820   1.190  1.00  0.00           C
TER
END
"""


# =============================================================================
# TEST FUNCTIONS
# =============================================================================

def test_scoring_import():
    """Test that open-source scoring modules can be imported."""
    print("\n" + "=" * 70)
    print("TEST: Scoring Import")
    print("=" * 70)
    
    results = {}
    
    # Test OpenMM
    try:
        import openmm
        results["OpenMM"] = f"✓ v{openmm.__version__}"
    except ImportError as e:
        results["OpenMM"] = f"✗ {e}"
    
    # Test PDBFixer
    try:
        from pdbfixer import PDBFixer
        results["PDBFixer"] = "✓"
    except ImportError as e:
        results["PDBFixer"] = f"✗ {e}"
    
    # Test FreeSASA
    try:
        import freesasa
        results["FreeSASA"] = "✓"
    except ImportError as e:
        results["FreeSASA"] = f"✗ {e}"
    
    # Test Biopython
    try:
        from Bio.PDB import PDBParser
        results["Biopython"] = "✓"
    except ImportError as e:
        results["Biopython"] = f"✗ {e}"
    
    # Test our scoring module
    try:
        from boltz_ph.scoring.opensource_local import run_opensource_scoring_local
        results["opensource_local"] = "✓"
    except ImportError as e:
        results["opensource_local"] = f"✗ {e}"
    
    all_passed = True
    for name, status in results.items():
        print(f"  {name}: {status}")
        if status.startswith("✗"):
            all_passed = False
    
    return all_passed


def test_scoring_synthetic():
    """Test scoring with a synthetic structure."""
    print("\n" + "=" * 70)
    print("TEST: Synthetic Structure Scoring")
    print("=" * 70)
    
    try:
        from boltz_ph.scoring.opensource_local import run_opensource_scoring_local
    except ImportError as e:
        print(f"  ✗ Failed to import scoring module: {e}")
        return False
    
    print(f"  Synthetic PDB content")
    print(f"  Binder: chain A (4 residues)")
    print(f"  Target: chain B (4 residues)")
    
    try:
        print("\n  Running open-source scoring...")
        t0 = time.time()
        
        # The function expects structure content as string, not a file path
        result = run_opensource_scoring_local(
            design_id="synthetic_test",
            af3_structure=SYNTHETIC_PDB,
            af3_iptm=0.8,
            af3_ptm=0.8,
            af3_plddt=85.0,
            binder_chain="A",
            target_chain="B",
            target_type="protein",
            verbose=True,
        )
        
        elapsed = time.time() - t0
        print(f"\n  Completed in {elapsed:.1f}s")
        
        if "error" in result:
            print(f"  ✗ Error: {result['error']}")
            return False
        
        # Display results
        print(f"\n  --- Interface Metrics ---")
        print(f"  Shape Comp (SC): {result.get('interface_sc', 'N/A')}")
        print(f"  dSASA: {result.get('interface_dSASA', 'N/A')}")
        print(f"  dG: {result.get('interface_dG', 'N/A')}")
        print(f"  Interface residues: {result.get('interface_nres', 'N/A')}")
        print(f"  Surface hydrophobicity: {result.get('surface_hydrophobicity', 'N/A')}")
        print(f"  Binder score: {result.get('binder_score', 'N/A')}")
        
        status = "ACCEPTED" if result.get('accepted') else f"REJECTED ({result.get('rejection_reason', '?')})"
        print(f"\n  Status: {status}")
        
        print(f"  ✓ Scoring completed successfully")
        return True
        
    except Exception as e:
        import traceback
        print(f"  ✗ Exception: {e}")
        traceback.print_exc()
        return False


def test_single_structure(
    structure_path: str,
    output_dir: str = "./scoring_test",
    binder_chain: str = "A",
    target_chain: str = "B",
    verbose: bool = True
):
    """Test scoring on a single structure file."""
    print("\n" + "=" * 70)
    print("TEST: Single Structure Scoring")
    print("=" * 70)
    
    struct_file = Path(structure_path)
    if not struct_file.exists():
        print(f"  ✗ Structure file not found: {structure_path}")
        return False
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"  Structure: {structure_path}")
    print(f"  Binder chain: {binder_chain}")
    print(f"  Target chain: {target_chain}")
    
    try:
        from boltz_ph.scoring.opensource_local import run_opensource_scoring_local
    except ImportError as e:
        print(f"  ✗ Failed to import scoring module: {e}")
        return False
    
    design_id = struct_file.stem.replace("_af3", "").replace("_protenix", "")
    
    try:
        print(f"\n  Running open-source scoring for {design_id}...")
        t0 = time.time()
        
        # Determine if CIF or PDB
        is_cif = struct_file.suffix.lower() in ['.cif', '.mmcif']
        
        # Read structure content
        structure_content = struct_file.read_text()
        
        result = run_opensource_scoring_local(
            design_id=design_id,
            af3_structure=structure_content,
            af3_iptm=0.8,  # Placeholder
            af3_ptm=0.8,  # Placeholder
            af3_plddt=85.0,  # Placeholder
            binder_chain=binder_chain,
            target_chain=target_chain,
            target_type="protein",
            verbose=verbose,
        )
        
        elapsed = time.time() - t0
        
        print(f"\n  Completed in {elapsed:.1f}s")
        
        if "error" in result:
            print(f"  ✗ Error: {result['error']}")
            return False
        
        # Display results
        print(f"\n  --- Interface Metrics ---")
        print(f"  Shape Comp (SC): {result.get('interface_sc', 'N/A')}")
        print(f"  dSASA: {result.get('interface_dSASA', 'N/A')}")
        print(f"  dG: {result.get('interface_dG', 'N/A')}")
        print(f"  Interface residues: {result.get('interface_nres', 'N/A')}")
        print(f"  Surface hydrophobicity: {result.get('surface_hydrophobicity', 'N/A')}")
        
        status = "ACCEPTED" if result.get('accepted') else f"REJECTED ({result.get('rejection_reason', '?')})"
        print(f"\n  Status: {status}")
        
        # Save relaxed PDB if available
        if result.get('relaxed_pdb'):
            relaxed_path = output_path / f"{design_id}_relaxed.pdb"
            with open(relaxed_path, 'w') as f:
                f.write(result['relaxed_pdb'])
            print(f"  ✓ Saved relaxed structure: {relaxed_path}")
        
        # Save results JSON
        results_json = output_path / f"{design_id}_scoring.json"
        json_result = {k: v for k, v in result.items() if k != 'relaxed_pdb'}
        with open(results_json, 'w') as f:
            json.dump(json_result, f, indent=2)
        print(f"  ✓ Saved metrics: {results_json}")
        
        return True
        
    except Exception as e:
        import traceback
        print(f"  ✗ Exception: {e}")
        traceback.print_exc()
        return False


def test_folder(
    input_dir: str,
    output_dir: str = "./scoring_test_results",
    binder_chain: str = "A",
    target_chain: str = "B",
    max_designs: int = 0,
    verbose: bool = False,
):
    """Test scoring on all structures in a folder."""
    print("\n" + "=" * 70)
    print("TEST: Folder Scoring")
    print("=" * 70)
    print(f"  Input directory: {input_dir}")
    print(f"  Output directory: {output_dir}")
    print(f"  Max designs: {max_designs if max_designs > 0 else 'all'}")
    
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    relaxed_dir = output_path / "relaxed"
    relaxed_dir.mkdir(exist_ok=True)
    
    # Find structure files
    structures = list(input_path.glob("*.pdb")) + list(input_path.glob("*.cif"))
    structures = sorted(structures)
    
    if max_designs > 0:
        structures = structures[:max_designs]
    
    print(f"  Found {len(structures)} structure(s) to process")
    
    if not structures:
        print("  ✗ No structures found!")
        return False
    
    try:
        from boltz_ph.scoring.opensource_local import run_opensource_scoring_local
    except ImportError as e:
        print(f"  ✗ Failed to import scoring module: {e}")
        return False
    
    all_results = []
    
    for i, struct_file in enumerate(structures):
        design_id = struct_file.stem.replace("_af3", "").replace("_protenix", "")
        print(f"\n  [{i+1}/{len(structures)}] Processing {design_id}")
        
        t0 = time.time()
        try:
            # Read structure content
            structure_content = struct_file.read_text()
            
            result = run_opensource_scoring_local(
                design_id=design_id,
                af3_structure=structure_content,
                af3_iptm=0.8,
                af3_ptm=0.8,
                af3_plddt=85.0,
                binder_chain=binder_chain,
                target_chain=target_chain,
                target_type="protein",
                verbose=verbose,
            )
            
            elapsed = time.time() - t0
            
            if "error" in result:
                print(f"    ✗ Error ({elapsed:.1f}s): {result['error']}")
                result_entry = {"design_id": design_id, "error": result["error"]}
            else:
                sc = result.get('interface_sc', 0)
                dsasa = result.get('interface_dSASA', 0)
                status = "ACCEPTED" if result.get('accepted') else "REJECTED"
                
                print(f"    ✓ Success ({elapsed:.1f}s): SC={sc:.3f}, dSASA={dsasa:.1f} → {status}")
                
                if result.get('relaxed_pdb'):
                    relaxed_path = relaxed_dir / f"{design_id}_relaxed.pdb"
                    with open(relaxed_path, 'w') as f:
                        f.write(result['relaxed_pdb'])
                
                result_entry = {
                    "design_id": design_id,
                    "accepted": result.get('accepted'),
                    "rejection_reason": result.get('rejection_reason'),
                    "interface_sc": result.get('interface_sc'),
                    "interface_dG": result.get('interface_dG'),
                    "interface_dSASA": result.get('interface_dSASA'),
                    "interface_nres": result.get('interface_nres'),
                    "surface_hydrophobicity": result.get('surface_hydrophobicity'),
                    "binder_score": result.get('binder_score'),
                    "elapsed_time": elapsed,
                }
            
            all_results.append(result_entry)
            
        except Exception as e:
            elapsed = time.time() - t0
            print(f"    ✗ Exception ({elapsed:.1f}s): {e}")
            all_results.append({"design_id": design_id, "error": str(e)})
    
    # Save results
    if all_results:
        results_csv = output_path / "scoring_results.csv"
        fieldnames = list(all_results[0].keys())
        for r in all_results:
            for k in r.keys():
                if k not in fieldnames:
                    fieldnames.append(k)
        
        with open(results_csv, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_results)
        print(f"\n  ✓ Saved: {results_csv}")
    
    # Summary
    successful = [r for r in all_results if 'interface_sc' in r]
    accepted = sum(1 for r in successful if r.get('accepted'))
    
    print(f"\n  --- Summary ---")
    print(f"  Total: {len(all_results)}")
    print(f"  Successful: {len(successful)}")
    print(f"  Accepted: {accepted}/{len(successful)}")
    
    if successful:
        avg_sc = sum(r.get('interface_sc', 0) or 0 for r in successful) / len(successful)
        print(f"  Avg SC: {avg_sc:.3f}")
    
    return len(successful) > 0


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Test open-source scoring locally",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument("--structure-path", type=str, help="Path to a single structure file (PDB/CIF)")
    parser.add_argument("--input-dir", type=str, help="Directory containing structure files")
    parser.add_argument("--output-dir", type=str, default="./scoring_test", help="Output directory")
    parser.add_argument("--binder-chain", type=str, default="A", help="Binder chain ID")
    parser.add_argument("--target-chain", type=str, default="B", help="Target chain ID")
    parser.add_argument("--max-designs", type=int, default=0, help="Max designs to process (0=all)")
    parser.add_argument("--synthetic", action="store_true", help="Run synthetic test")
    parser.add_argument("--check-import", action="store_true", help="Only check if modules import")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    print("\n" + "=" * 70)
    print("OPEN-SOURCE SCORING LOCAL TEST")
    print("=" * 70)
    
    results = []
    
    # Always test imports first
    results.append(("Import", test_scoring_import()))
    
    if args.check_import:
        pass
    elif args.synthetic:
        results.append(("Synthetic", test_scoring_synthetic()))
    elif args.structure_path:
        results.append(("Single Structure", test_single_structure(
            args.structure_path, args.output_dir, args.binder_chain, args.target_chain, args.verbose
        )))
    elif args.input_dir:
        results.append(("Folder", test_folder(
            args.input_dir, args.output_dir, args.binder_chain, args.target_chain,
            args.max_designs, args.verbose
        )))
    else:
        # Default: run synthetic test
        results.append(("Synthetic", test_scoring_synthetic()))
    
    # Summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    
    all_passed = True
    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {name}: {status}")
        if not passed:
            all_passed = False
    
    print("=" * 70 + "\n")
    
    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
