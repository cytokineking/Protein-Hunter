#!/usr/bin/env python3
"""
Local test script for ipSAE calculation.

This script tests the ipSAE calculation using saved PAE matrices,
without needing to spin up Modal containers.

Usage:
    python test_ipsae_local.py [--pae-file path/to/pae.npy]
    
To generate a PAE file, run a Protenix test with --save-pae flag.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

# Add the project root to path
sys.path.insert(0, str(Path(__file__).parent))

from modal_boltz_ph.validation.base import calculate_ipsae_from_pae


def test_ipsae_calculation(pae_matrix: np.ndarray, binder_length: int, target_length: int):
    """Test ipSAE calculation with debug output."""
    print(f"\n{'='*60}")
    print("ipSAE Calculation Test")
    print(f"{'='*60}")
    
    print(f"\nInput:")
    print(f"  PAE matrix shape: {pae_matrix.shape}")
    print(f"  Binder length: {binder_length}")
    print(f"  Target length: {target_length}")
    print(f"  Total: {binder_length + target_length}")
    
    # Basic PAE stats
    print(f"\nPAE Matrix Statistics:")
    print(f"  Full matrix min/max/mean: {pae_matrix.min():.2f}/{pae_matrix.max():.2f}/{pae_matrix.mean():.2f}")
    
    diagonal = np.diag(pae_matrix)
    print(f"  Diagonal (self-PAE) min/max/mean: {diagonal.min():.2f}/{diagonal.max():.2f}/{diagonal.mean():.2f}")
    
    # Interface PAE
    binder_idx = np.arange(binder_length)
    target_idx = np.arange(binder_length, binder_length + target_length)
    interface_pae = pae_matrix[np.ix_(binder_idx, target_idx)]
    
    print(f"\nInterface PAE (binder → target):")
    print(f"  Shape: {interface_pae.shape}")
    print(f"  Min/max/mean: {interface_pae.min():.2f}/{interface_pae.max():.2f}/{interface_pae.mean():.2f}")
    
    # Count values below different cutoffs
    for cutoff in [5.0, 10.0, 15.0, 20.0]:
        count = (interface_pae < cutoff).sum()
        pct = count / interface_pae.size * 100
        print(f"  Values < {cutoff:.1f}Å: {count} ({pct:.1f}%)")
    
    # Run the actual calculation
    print(f"\n{'='*60}")
    print("Running calculate_ipsae_from_pae()...")
    print(f"{'='*60}")
    
    result = calculate_ipsae_from_pae(
        pae_matrix,
        binder_length=binder_length,
        target_length=target_length,
        pae_cutoff=10.0,
    )
    
    print(f"\nResults:")
    print(f"  ipSAE: {result['ipsae']:.4f}")
    print(f"  ipSAE (binder→target): {result['ipsae_binder_to_target']:.4f}")
    print(f"  ipSAE (target→binder): {result['ipsae_target_to_binder']:.4f}")
    
    # Try with higher cutoff
    print(f"\n{'='*60}")
    print("Testing with higher cutoff (20.0Å)...")
    print(f"{'='*60}")
    
    result_20 = calculate_ipsae_from_pae(
        pae_matrix,
        binder_length=binder_length,
        target_length=target_length,
        pae_cutoff=20.0,
    )
    
    print(f"\nResults (cutoff=20.0Å):")
    print(f"  ipSAE: {result_20['ipsae']:.4f}")
    print(f"  ipSAE (binder→target): {result_20['ipsae_binder_to_target']:.4f}")
    print(f"  ipSAE (target→binder): {result_20['ipsae_target_to_binder']:.4f}")
    
    return result


def create_synthetic_pae(binder_length: int = 100, target_length: int = 150, 
                          interface_quality: str = "good"):
    """Create a synthetic PAE matrix for testing."""
    total = binder_length + target_length
    
    # Start with high PAE everywhere
    pae = np.full((total, total), 25.0)
    
    # Set diagonal to low (high confidence for self)
    np.fill_diagonal(pae, 0.3)
    
    # Set intra-chain PAE to medium-low
    pae[:binder_length, :binder_length] = np.random.uniform(1, 5, (binder_length, binder_length))
    pae[binder_length:, binder_length:] = np.random.uniform(1, 5, (target_length, target_length))
    np.fill_diagonal(pae, 0.3)
    
    # Set interface PAE based on quality
    if interface_quality == "good":
        # Good interface: low PAE (2-8 Å) for ~30% of interface
        interface = np.random.uniform(15, 28, (binder_length, target_length))
        # Add some low PAE hot spots
        hot_spots = np.random.choice(binder_length * target_length, size=int(0.3 * binder_length * target_length), replace=False)
        interface.flat[hot_spots] = np.random.uniform(2, 8, len(hot_spots))
    elif interface_quality == "medium":
        # Medium interface: mostly high PAE with few low spots
        interface = np.random.uniform(12, 25, (binder_length, target_length))
        hot_spots = np.random.choice(binder_length * target_length, size=int(0.1 * binder_length * target_length), replace=False)
        interface.flat[hot_spots] = np.random.uniform(5, 10, len(hot_spots))
    else:  # poor
        # Poor interface: all high PAE
        interface = np.random.uniform(18, 30, (binder_length, target_length))
    
    pae[:binder_length, binder_length:] = interface
    pae[binder_length:, :binder_length] = interface.T
    
    return pae


def main():
    parser = argparse.ArgumentParser(description="Test ipSAE calculation locally")
    parser.add_argument("--pae-file", type=str, help="Path to saved PAE matrix (.npy file)")
    parser.add_argument("--pae-json", type=str, help="Path to Protenix full_data JSON")
    parser.add_argument("--binder-length", type=int, default=143, help="Binder chain length")
    parser.add_argument("--target-length", type=int, default=190, help="Target chain(s) length")
    parser.add_argument("--synthetic", choices=["good", "medium", "poor"], 
                        help="Use synthetic PAE matrix with specified quality")
    args = parser.parse_args()
    
    if args.pae_file:
        print(f"Loading PAE matrix from: {args.pae_file}")
        pae_matrix = np.load(args.pae_file)
    elif args.pae_json:
        print(f"Loading PAE from Protenix full_data JSON: {args.pae_json}")
        with open(args.pae_json) as f:
            data = json.load(f)
        pae_matrix = np.array(data.get("token_pair_pae", []))
        if pae_matrix.size == 0:
            print("ERROR: No token_pair_pae found in JSON")
            sys.exit(1)
    elif args.synthetic:
        print(f"Creating synthetic PAE matrix with '{args.synthetic}' interface quality")
        pae_matrix = create_synthetic_pae(
            args.binder_length, args.target_length, args.synthetic
        )
    else:
        # Check for default saved PAE file
        default_pae = Path(__file__).parent / "test_data" / "protenix_pae_sample.npy"
        if default_pae.exists():
            print(f"Loading default PAE matrix from: {default_pae}")
            pae_matrix = np.load(default_pae)
        else:
            print("No PAE file specified. Running with synthetic 'medium' quality PAE.")
            print("To save real PAE data, run a Protenix test first.")
            pae_matrix = create_synthetic_pae(
                args.binder_length, args.target_length, "medium"
            )
    
    # Infer lengths from matrix if not specified
    total_len = pae_matrix.shape[0]
    if args.binder_length + args.target_length != total_len:
        print(f"Warning: binder_length ({args.binder_length}) + target_length ({args.target_length}) "
              f"!= matrix size ({total_len})")
        # Try to use the provided lengths anyway
    
    test_ipsae_calculation(pae_matrix, args.binder_length, args.target_length)


if __name__ == "__main__":
    main()
