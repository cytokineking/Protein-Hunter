#!/usr/bin/env python
"""
Test harness for local Protenix validation.

This script allows isolated testing of the Protenix validation pipeline using
existing Boltz design outputs or synthetic test cases, without running the
full design pipeline.

Supports both:
  - Subprocess mode (original): Each validation spawns a new process (~105s/design)
  - Persistent mode (new): Model stays loaded in GPU memory (~25s/design after first)

Usage:
    # Test a single PDB file (persistent mode - faster)
    python -m boltz_ph.tests.test_protenix_local --pdb-path ./best_designs/design_001.pdb --persistent
    
    # Test all designs in a folder (persistent mode recommended for batches)
    python -m boltz_ph.tests.test_protenix_local --input-dir ./best_designs --output-dir ./protenix_test --persistent
    
    # Quick synthetic test (no input files needed)
    python -m boltz_ph.tests.test_protenix_local --synthetic --persistent
    
    # Original subprocess mode (for comparison/debugging)
    python -m boltz_ph.tests.test_protenix_local --synthetic

Input format:
    - PDB files with designed binder (chain A) and target chains (B, C, D, ...)
    - MSAs will be automatically fetched from ColabFold if needed

Output:
    - validation_results.csv: Protenix validation metrics for each design
    - structures/: Predicted structures from Protenix (CIF format)
"""

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

# Ensure boltz_ph is importable
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


# =============================================================================
# PDB SEQUENCE EXTRACTION
# =============================================================================

AA_3TO1 = {
    'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C',
    'GLN': 'Q', 'GLU': 'E', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
    'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F', 'PRO': 'P',
    'SER': 'S', 'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V',
    'MSE': 'M', 'HSD': 'H', 'HSE': 'H', 'HSP': 'H',
}


def extract_sequences_from_pdb(pdb_path: str) -> Dict[str, str]:
    """Extract protein sequences from a PDB file."""
    chains = {}
    
    with open(pdb_path, 'r') as f:
        for line in f:
            if not line.startswith('ATOM'):
                continue
            
            atom_name = line[12:16].strip()
            if atom_name != 'CA':
                continue
            
            chain_id = line[21].strip()
            resnum = int(line[22:26].strip())
            resname = line[17:20].strip()
            
            if chain_id not in chains:
                chains[chain_id] = []
            chains[chain_id].append((resnum, resname))
    
    sequences = {}
    for chain_id, residues in chains.items():
        residues = sorted(set(residues), key=lambda x: x[0])
        seq = ''.join(AA_3TO1.get(resname, 'X') for _, resname in residues)
        sequences[chain_id] = seq
    
    return sequences


def parse_design_pdbs(input_dir: str) -> List[Dict[str, Any]]:
    """Parse all design PDB files in a directory."""
    input_path = Path(input_dir)
    designs = []
    
    pdb_files = sorted(input_path.glob("*.pdb"))
    
    for pdb_file in pdb_files:
        design_id = pdb_file.stem
        
        try:
            sequences = extract_sequences_from_pdb(str(pdb_file))
            
            if 'A' not in sequences:
                print(f"  Warning: No chain A (binder) in {design_id}, skipping")
                continue
            
            binder_seq = sequences['A']
            
            target_chains = []
            for chain_id in sorted(sequences.keys()):
                if chain_id != 'A':
                    target_chains.append((chain_id, sequences[chain_id]))
            
            if not target_chains:
                print(f"  Warning: No target chains in {design_id}, skipping")
                continue
            
            designs.append({
                "design_id": design_id,
                "pdb_path": str(pdb_file),
                "binder_seq": binder_seq,
                "target_chains": target_chains,
                "target_seq": ":".join(seq for _, seq in target_chains),
            })
            
        except Exception as e:
            print(f"  Warning: Failed to parse {design_id}: {e}")
    
    return designs


# =============================================================================
# TEST FUNCTIONS
# =============================================================================

def test_protenix_import():
    """Test that Protenix can be imported."""
    print("\n" + "=" * 70)
    print("TEST: Protenix Import")
    print("=" * 70)
    
    try:
        from boltz_ph.validation.protenix_local import (
            run_protenix_validation_local,
            run_protenix_validation_persistent,
            shutdown_persistent_runner,
            ensure_protenix_weights,
            DEFAULT_PROTENIX_MODEL,
        )
        print(f"  ✓ Protenix local module imported successfully")
        print(f"  ✓ Persistent runner available")
        print(f"  Default model: {DEFAULT_PROTENIX_MODEL}")
        return True
    except ImportError as e:
        print(f"  ✗ Failed to import protenix_local: {e}")
        return False


def test_protenix_weights():
    """Test that Protenix weights can be downloaded/verified."""
    print("\n" + "=" * 70)
    print("TEST: Protenix Weights")
    print("=" * 70)
    
    try:
        from boltz_ph.validation.protenix_local import (
            ensure_protenix_weights,
            DEFAULT_PROTENIX_MODEL,
        )
        
        print(f"  Checking weights for {DEFAULT_PROTENIX_MODEL}...")
        weights_dir = ensure_protenix_weights(DEFAULT_PROTENIX_MODEL)
        print(f"  ✓ Weights directory: {weights_dir}")
        
        weight_file = weights_dir / f"{DEFAULT_PROTENIX_MODEL}.pt"
        if weight_file.exists():
            size_mb = weight_file.stat().st_size / (1024**2)
            print(f"  ✓ Weight file exists: {weight_file} ({size_mb:.1f} MB)")
            return True
        else:
            print(f"  ✗ Weight file not found: {weight_file}")
            return False
            
    except Exception as e:
        print(f"  ✗ Failed to verify weights: {e}")
        return False


def test_protenix_synthetic(use_persistent: bool = False):
    """Test Protenix with a synthetic binder/target pair."""
    mode_str = "PERSISTENT" if use_persistent else "SUBPROCESS"
    print("\n" + "=" * 70)
    print(f"TEST: Protenix Synthetic Validation ({mode_str})")
    print("=" * 70)
    
    # Small synthetic sequences for quick testing
    binder_seq = "MKFLILLFNILCLFPVLAADNHGVGPLGITADAAQVKGATVFPKG"
    target_seq = "MHHHHHHENLYFQGSMRVLVLDNIEDHSILFQRVLAQQFDPNQYKVHPVTLFSAWTTLFEVLVDPQGM"
    
    print(f"  Binder: {len(binder_seq)} residues")
    print(f"  Target: {len(target_seq)} residues")
    print(f"  Mode: {mode_str}")
    
    try:
        if use_persistent:
            from boltz_ph.validation.protenix_local import (
                run_protenix_validation_persistent,
                shutdown_persistent_runner,
            )
            validation_func = run_protenix_validation_persistent
        else:
            from boltz_ph.validation.protenix_local import run_protenix_validation_local
            validation_func = run_protenix_validation_local
        
        print("\n  Running Protenix HOLO + APO validation...")
        t0 = time.time()
        
        result = validation_func(
            design_id="synthetic_test",
            binder_seq=binder_seq,
            target_seq=target_seq,
            target_msas=None,  # No MSA for quick test
            verbose=True,
        )
        
        elapsed = time.time() - t0
        print(f"\n  Completed in {elapsed:.1f}s")
        
        if "error" in result:
            print(f"  ✗ Error: {result['error']}")
            return False
        
        print(f"\n  --- Results ---")
        print(f"  ipTM: {result.get('af3_iptm', 'N/A')}")
        print(f"  pTM: {result.get('af3_ptm', 'N/A')}")
        print(f"  pLDDT: {result.get('af3_plddt', 'N/A')}")
        print(f"  ipSAE: {result.get('af3_ipsae', 'N/A')}")
        
        if use_persistent:
            holo_time = result.get('holo_time', 0)
            apo_time = result.get('apo_time', 0)
            if holo_time > 0:
                print(f"  HOLO time: {holo_time:.1f}s")
            if apo_time > 0:
                print(f"  APO time: {apo_time:.1f}s")
        
        if result.get('af3_iptm', 0) > 0:
            print(f"  ✓ Protenix validation succeeded")
            return True
        else:
            print(f"  ✗ Protenix returned zero metrics")
            return False
            
    except Exception as e:
        import traceback
        print(f"  ✗ Exception: {e}")
        traceback.print_exc()
        return False
    finally:
        if use_persistent:
            try:
                from boltz_ph.validation.protenix_local import shutdown_persistent_runner
                shutdown_persistent_runner()
            except Exception:
                pass


def test_single_pdb(
    pdb_path: str,
    output_dir: str = "./protenix_test",
    verbose: bool = True,
    use_persistent: bool = False,
):
    """Test Protenix validation on a single PDB file."""
    mode_str = "PERSISTENT" if use_persistent else "SUBPROCESS"
    print("\n" + "=" * 70)
    print(f"TEST: Protenix Single PDB Validation ({mode_str})")
    print("=" * 70)
    
    pdb_file = Path(pdb_path)
    if not pdb_file.exists():
        print(f"  ✗ PDB file not found: {pdb_path}")
        return False
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Extract sequences
    print(f"\n  PDB file: {pdb_path}")
    sequences = extract_sequences_from_pdb(pdb_path)
    
    if 'A' not in sequences:
        print("  ✗ No chain A (binder) found in PDB")
        return False
    
    binder_seq = sequences['A']
    target_chains = [(cid, seq) for cid, seq in sorted(sequences.items()) if cid != 'A']
    target_seq = ":".join(seq for _, seq in target_chains)
    
    print(f"\n  Chains detected:")
    print(f"    A (binder): {len(binder_seq)} residues")
    for chain_id, seq in target_chains:
        print(f"    {chain_id} (target): {len(seq)} residues")
    
    try:
        if use_persistent:
            from boltz_ph.validation.protenix_local import (
                run_protenix_validation_persistent,
                shutdown_persistent_runner,
            )
            validation_func = run_protenix_validation_persistent
        else:
            from boltz_ph.validation.protenix_local import run_protenix_validation_local
            validation_func = run_protenix_validation_local
        
        design_id = pdb_file.stem
        print(f"\n  Running Protenix validation for {design_id}...")
        
        t0 = time.time()
        result = validation_func(
            design_id=design_id,
            binder_seq=binder_seq,
            target_seq=target_seq,
            target_msas=None,  # Could add MSA fetching here
            verbose=verbose,
        )
        elapsed = time.time() - t0
        
        print(f"\n  Completed in {elapsed:.1f}s")
        
        if "error" in result:
            print(f"  ✗ Error: {result['error']}")
            return False
        
        # Display results
        print(f"\n  --- Protenix Confidence ---")
        print(f"  ipTM: {result.get('af3_iptm', 0):.4f}")
        print(f"  pTM: {result.get('af3_ptm', 0):.4f}")
        print(f"  pLDDT: {result.get('af3_plddt', 0):.2f}")
        print(f"  ipSAE: {result.get('af3_ipsae', 0):.4f}")
        
        # Save structure if available
        holo_struct = result.get('af3_structure') or result.get('holo_structure')
        if holo_struct:
            struct_path = output_path / f"{design_id}_protenix.cif"
            struct_path.write_text(holo_struct)
            print(f"\n  ✓ Saved structure: {struct_path}")
        
        # Save results JSON
        results_json = output_path / f"{design_id}_results.json"
        json_result = {k: v for k, v in result.items() 
                       if not k.endswith('_structure') and not k.endswith('_json')}
        with open(results_json, 'w') as f:
            json.dump(json_result, f, indent=2)
        print(f"  ✓ Saved metrics: {results_json}")
        
        return True
        
    except Exception as e:
        import traceback
        print(f"  ✗ Exception: {e}")
        traceback.print_exc()
        return False
    finally:
        if use_persistent:
            try:
                from boltz_ph.validation.protenix_local import shutdown_persistent_runner
                shutdown_persistent_runner()
            except Exception:
                pass


def test_folder(
    input_dir: str,
    output_dir: str = "./protenix_validation_test",
    max_designs: int = 0,
    verbose: bool = False,
    use_persistent: bool = False,
):
    """Test Protenix validation on all PDB files in a folder."""
    mode_str = "PERSISTENT" if use_persistent else "SUBPROCESS"
    print("\n" + "=" * 70)
    print(f"TEST: Protenix Folder Validation ({mode_str})")
    print("=" * 70)
    print(f"  Input directory: {input_dir}")
    print(f"  Output directory: {output_dir}")
    print(f"  Max designs: {max_designs if max_designs > 0 else 'all'}")
    print(f"  Mode: {mode_str}")
    
    if use_persistent:
        print("\n  💡 Using persistent runner - model loaded once, reused for all designs")
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    structures_dir = output_path / "structures"
    structures_dir.mkdir(exist_ok=True)
    
    # Parse designs
    print("\n  Parsing design PDB files...")
    designs = parse_design_pdbs(input_dir)
    
    if max_designs > 0:
        designs = designs[:max_designs]
    
    print(f"  Found {len(designs)} design(s) to process")
    
    if not designs:
        print("  ✗ No valid designs found!")
        return False
    
    try:
        if use_persistent:
            from boltz_ph.validation.protenix_local import (
                run_protenix_validation_persistent,
                shutdown_persistent_runner,
            )
            validation_func = run_protenix_validation_persistent
        else:
            from boltz_ph.validation.protenix_local import run_protenix_validation_local
            validation_func = run_protenix_validation_local
    except ImportError as e:
        print(f"  ✗ Failed to import protenix_local: {e}")
        return False
    
    all_results = []
    total_start = time.time()
    
    try:
        for i, design in enumerate(designs):
            design_id = design["design_id"]
            print(f"\n  [{i+1}/{len(designs)}] Processing {design_id}")
            print(f"    Binder: {len(design['binder_seq'])} residues")
            print(f"    Targets: {', '.join(f'{cid}={len(seq)}' for cid, seq in design['target_chains'])}")
            
            t0 = time.time()
            try:
                result = validation_func(
                    design_id=design_id,
                    binder_seq=design["binder_seq"],
                    target_seq=design["target_seq"],
                    target_msas=None,
                    verbose=verbose,
                )
                elapsed = time.time() - t0
                
                if "error" in result:
                    print(f"    ✗ Error ({elapsed:.1f}s): {result['error']}")
                    result_entry = {
                        "design_id": design_id,
                        "binder_length": len(design["binder_seq"]),
                        "error": result["error"],
                        "elapsed_time": elapsed,
                    }
                else:
                    iptm = result.get('af3_iptm', 0)
                    plddt = result.get('af3_plddt', 0)
                    
                    print(f"    ✓ Success ({elapsed:.1f}s): ipTM={iptm:.3f}, pLDDT={plddt:.1f}")
                    
                    holo_struct = result.get('af3_structure') or result.get('holo_structure')
                    if holo_struct:
                        struct_path = structures_dir / f"{design_id}_protenix.cif"
                        struct_path.write_text(holo_struct)
                    
                    result_entry = {
                        "design_id": design_id,
                        "binder_length": len(design["binder_seq"]),
                        "binder_seq": design["binder_seq"],
                        "protenix_iptm": result.get('af3_iptm'),
                        "protenix_ptm": result.get('af3_ptm'),
                        "protenix_plddt": result.get('af3_plddt'),
                        "protenix_ipsae": result.get('af3_ipsae'),
                        "elapsed_time": elapsed,
                    }
                    
                    # Add timing breakdown for persistent mode
                    if use_persistent:
                        result_entry["holo_time"] = result.get('holo_time', 0)
                        result_entry["apo_time"] = result.get('apo_time', 0)
                
                all_results.append(result_entry)
                
            except Exception as e:
                elapsed = time.time() - t0
                print(f"    ✗ Exception ({elapsed:.1f}s): {e}")
                all_results.append({
                    "design_id": design_id,
                    "binder_length": len(design["binder_seq"]),
                    "error": str(e),
                    "elapsed_time": elapsed,
                })
        
        total_elapsed = time.time() - total_start
        
    finally:
        # Always clean up persistent runner
        if use_persistent:
            try:
                print("\n  Shutting down persistent runner...")
                shutdown_persistent_runner()
            except Exception:
                pass
    
    # Save results
    if all_results:
        results_csv = output_path / "validation_results.csv"
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
    successful = [r for r in all_results if 'protenix_iptm' in r and r.get('protenix_iptm') is not None]
    failed = len(all_results) - len(successful)
    
    print(f"\n  " + "=" * 60)
    print(f"  SUMMARY")
    print(f"  " + "=" * 60)
    print(f"  Total designs: {len(all_results)}")
    print(f"  Successful: {len(successful)}")
    print(f"  Failed: {failed}")
    print(f"  Total time: {total_elapsed:.1f}s")
    
    if successful:
        avg_iptm = sum(r['protenix_iptm'] for r in successful) / len(successful)
        avg_plddt = sum(r['protenix_plddt'] for r in successful) / len(successful)
        avg_time = sum(r['elapsed_time'] for r in successful) / len(successful)
        
        print(f"\n  Average metrics:")
        print(f"    ipTM: {avg_iptm:.3f}")
        print(f"    pLDDT: {avg_plddt:.1f}")
        print(f"    Time/design: {avg_time:.1f}s")
        
        if len(successful) > 1:
            first_time = successful[0]['elapsed_time']
            subsequent_times = [r['elapsed_time'] for r in successful[1:]]
            avg_subsequent = sum(subsequent_times) / len(subsequent_times) if subsequent_times else 0
            
            if use_persistent and avg_subsequent > 0:
                print(f"\n  Persistent runner performance:")
                print(f"    First design (incl. model load): {first_time:.1f}s")
                print(f"    Avg subsequent designs: {avg_subsequent:.1f}s")
                estimated_subprocess = 105 * len(successful)
                speedup = estimated_subprocess / total_elapsed if total_elapsed > 0 else 0
                print(f"    Estimated speedup vs subprocess: {speedup:.1f}x")
    
    print(f"  " + "=" * 60)
    
    return len(successful) > 0


def test_persistent_runner():
    """Test the persistent runner specifically."""
    print("\n" + "=" * 70)
    print("TEST: Persistent Runner Load/Predict/Unload Cycle")
    print("=" * 70)
    
    try:
        from boltz_ph.validation.protenix_runner import PersistentProtenixRunner
        
        print("\n  Phase 1: Loading model...")
        runner = PersistentProtenixRunner.get_instance()
        load_time = runner.ensure_loaded()
        print(f"  ✓ Model loaded in {load_time:.1f}s")
        
        print("\n  Phase 2: Running predictions...")
        test_cases = [
            ("test_a", "MKFLILLFNILCLFPVLAADNHGVGPLGITADAAQVKGATVFPKG", 
             "MHHHHHHENLYFQGSMRVLVLDNIEDHSILFQRVLAQQFDPNQYKVHPVTLFSAWTTLFEVLVDPQGM"),
            ("test_b", "GAMGSEIEHIEEAIANAKTKADHERLVAHYEEEAKRLEKKSEEYQELAEKNREMGEKLLERLKTVEK",
             "MHHHHHHENLYFQGSMRVLVLDNIEDHSILFQRVLAQQFDPNQYKVHPVTLFSAWTTLFEVLVDPQGM"),
        ]
        
        times = []
        for design_id, binder_seq, target_seq in test_cases:
            t0 = time.time()
            result = runner.predict_holo(
                design_id=design_id,
                binder_seq=binder_seq,
                target_seq=target_seq,
                target_msas=None,
            )
            elapsed = time.time() - t0
            times.append(elapsed)
            
            if result.get("error"):
                print(f"    {design_id}: ✗ Error - {result['error']}")
            else:
                iptm = result.get('iptm', 0)
                print(f"    {design_id}: ✓ {elapsed:.1f}s, ipTM={iptm:.3f}")
        
        print("\n  Phase 3: Shutting down...")
        PersistentProtenixRunner.shutdown()
        print("  ✓ Runner shutdown complete")
        
        print("\n  --- Performance Summary ---")
        print(f"  Model load time: {load_time:.1f}s")
        if times:
            print(f"  First prediction: {times[0]:.1f}s")
            if len(times) > 1:
                print(f"  Second prediction: {times[1]:.1f}s")
        
        return True
        
    except Exception as e:
        import traceback
        print(f"  ✗ Exception: {e}")
        traceback.print_exc()
        return False


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Test Protenix validation locally",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument("--pdb-path", type=str, help="Path to a single PDB file to test")
    parser.add_argument("--input-dir", type=str, help="Directory containing PDB files to test")
    parser.add_argument("--output-dir", type=str, default="./protenix_test", help="Output directory")
    parser.add_argument("--max-designs", type=int, default=0, help="Max designs to process (0=all)")
    parser.add_argument("--synthetic", action="store_true", help="Run synthetic test (no input files)")
    parser.add_argument("--check-import", action="store_true", help="Only check if Protenix imports")
    parser.add_argument("--check-weights", action="store_true", help="Only check if weights are available")
    parser.add_argument("--test-runner", action="store_true", help="Test persistent runner lifecycle")
    parser.add_argument("--persistent", "-p", action="store_true", 
                        help="Use persistent runner (model stays loaded, ~4x faster for batches)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    print("\n" + "=" * 70)
    print("PROTENIX LOCAL VALIDATION TEST")
    if args.persistent:
        print("MODE: PERSISTENT (model stays loaded in GPU memory)")
    else:
        print("MODE: SUBPROCESS (original, each validation spawns new process)")
    print("=" * 70)
    
    results = []
    
    # Always test import first
    results.append(("Import", test_protenix_import()))
    
    if args.check_import:
        # Just check import
        pass
    elif args.check_weights:
        results.append(("Weights", test_protenix_weights()))
    elif args.test_runner:
        results.append(("Weights", test_protenix_weights()))
        results.append(("Persistent Runner", test_persistent_runner()))
    elif args.synthetic:
        results.append(("Weights", test_protenix_weights()))
        results.append(("Synthetic", test_protenix_synthetic(use_persistent=args.persistent)))
    elif args.pdb_path:
        results.append(("Weights", test_protenix_weights()))
        results.append(("Single PDB", test_single_pdb(
            args.pdb_path, args.output_dir, args.verbose, use_persistent=args.persistent
        )))
    elif args.input_dir:
        results.append(("Weights", test_protenix_weights()))
        results.append(("Folder", test_folder(
            args.input_dir, args.output_dir, args.max_designs, args.verbose, 
            use_persistent=args.persistent
        )))
    else:
        # Default: run basic tests
        results.append(("Weights", test_protenix_weights()))
    
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
