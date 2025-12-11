"""
OpenFold3 validation functions.

This module provides open-source structure validation using OpenFold3,
an Apache 2.0 licensed reproduction of AlphaFold3 from the AlQuraishi Lab
at Columbia University.

OpenFold3 is fully permissive for commercial use and auto-downloads weights.
It's bundled with open-source scoring (OpenMM, FreeSASA, sc-rs) for efficiency.

Reference: https://github.com/aqlaboratory/openfold-3
"""

import json
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from modal_boltz_ph.app import app, cache_volume, openfold3_weights_volume
from modal_boltz_ph.images import openfold3_validation_image
from modal_boltz_ph.validation.base import (
    calculate_ipsae_from_pae,
    normalize_plddt_scale,
)

# =============================================================================
# CONSTANTS
# =============================================================================

OPENFOLD3_WEIGHTS_PATH = "/openfold3_weights"
OPENFOLD3_RUNNER_YAML = "/root/openfold3_config/runner.yaml"  # cuEquivariance + PAE enabled
DEFAULT_OF3_MODEL_SEED = 42
DEFAULT_OF3_DIFFUSION_SAMPLES = 1

# Global verbose flag
_OF3_VERBOSE = False


# =============================================================================
# WEIGHT MANAGEMENT
# =============================================================================

def ensure_openfold3_weights() -> Path:
    """
    Ensure OpenFold3 model weights are available on the volume.
    
    Downloads weights on first use via direct URL download from AWS S3.
    """
    from modal_boltz_ph.utils.weights import download_with_progress, verify_download
    
    weights_dir = Path(OPENFOLD3_WEIGHTS_PATH)
    weights_dir.mkdir(parents=True, exist_ok=True)
    
    # OpenFold3 checkpoint filename (from official download script)
    checkpoint_name = "of3_ft3_v1.pt"
    checkpoint_path = weights_dir / checkpoint_name
    
    # Check if weights already exist
    if verify_download(checkpoint_path, min_size_mb=100):
        size_mb = checkpoint_path.stat().st_size / (1024 * 1024)
        print(f"  ✓ OpenFold3 weights found at {checkpoint_path} ({size_mb:.1f} MB)")
        return weights_dir
    
    # Also check for any .pt files in the directory
    ckpt_files = list(weights_dir.glob("*.pt"))
    for f in ckpt_files:
        if verify_download(f, min_size_mb=100):
            size_mb = f.stat().st_size / (1024 * 1024)
            print(f"  ✓ OpenFold3 weights found at {f} ({size_mb:.1f} MB)")
            return weights_dir
    
    # Download weights directly from AWS S3 (bypassing setup_openfold which has conda/DeepSpeed issues)
    print("  Downloading OpenFold3 weights (one-time operation)...")
    
    # Official OpenFold3 weights URL (from download_openfold3_params.sh)
    weights_url = "https://openfold.s3.amazonaws.com/openfold3_params/of3_ft3_v1.pt"
    
    try:
        download_with_progress(
            url=weights_url,
            dest_path=checkpoint_path,
            description=f"  Downloading from {weights_url}...",
            print_every=500,  # Less frequent progress updates for large file
        )
        
        # Verify download
        if not verify_download(checkpoint_path, min_size_mb=100):
            size_mb = checkpoint_path.stat().st_size / (1024 * 1024) if checkpoint_path.exists() else 0
            raise RuntimeError(f"Downloaded file too small: {size_mb:.1f} MB")
        
        # Write ckpt_root file so OpenFold3 knows where to find weights
        ckpt_root_file = weights_dir / "ckpt_root"
        ckpt_root_file.write_text(str(weights_dir))
        
        # Commit volume changes
        openfold3_weights_volume.commit()
        size_mb = checkpoint_path.stat().st_size / (1024 * 1024)
        print(f"  ✓ OpenFold3 weights downloaded to {checkpoint_path} ({size_mb:.1f} MB)")
        
    except Exception as e:
        # Clean up partial download
        if checkpoint_path.exists():
            checkpoint_path.unlink()
        print(f"  ✗ Failed to download weights: {e}")
        raise
    
    return weights_dir


# =============================================================================
# INPUT/OUTPUT UTILITIES
# =============================================================================

def _build_openfold3_input(
    design_id: str,
    binder_seq: str,
    target_seq: str,
    msa_dir: Optional[Path] = None,
    work_dir: Optional[Path] = None,
) -> Path:
    """
    Build OpenFold3 JSON input file.
    
    OpenFold3 uses a query-based format:
    {
        "queries": {
            "<design_id>": {
                "chains": [
                    {"molecule_type": "protein", "chain_ids": ["A"], "sequence": "..."},
                    {"molecule_type": "protein", "chain_ids": ["B"], "sequence": "...",
                     "main_msa_file_paths": ["/path/to/msa.a3m"]}
                ]
            }
        }
    }
    
    Args:
        design_id: Unique identifier for this prediction
        binder_seq: Designed binder sequence (chain A, no MSA)
        target_seq: Target sequence(s), colon-separated for multi-chain
        msa_dir: Path to MSA directory for target chains
        work_dir: Working directory for output file
    
    Returns:
        Path to the generated JSON input file
    """
    if work_dir is None:
        work_dir = Path(tempfile.mkdtemp())
    
    chains = []
    
    # Binder chain (A) - no MSA for de novo designed sequence
    chains.append({
        "molecule_type": "protein",
        "chain_ids": ["A"],
        "sequence": binder_seq,
        # No MSA path = OpenFold3 will use single-sequence mode for this chain
    })
    
    # Target chain(s) (B, C, D, ...)
    target_chains = target_seq.split(":") if target_seq else []
    for i, seq in enumerate(target_chains):
        chain_id = chr(ord('B') + i)  # B, C, D, ...
        
        chain_entry = {
            "molecule_type": "protein",
            "chain_ids": [chain_id],
            "sequence": seq,
        }
        
        # Add MSA path if available (OpenFold3 expects 'colabfold_main.a3m')
        # If no MSA path provided, OpenFold3 will use single-sequence mode
        if msa_dir:
            msa_file = msa_dir / f"chain_{chain_id}" / "colabfold_main.a3m"
            if msa_file.exists():
                chain_entry["main_msa_file_paths"] = [str(msa_file)]
        
        chains.append(chain_entry)
    
    # Build OpenFold3 query format
    # Set use_paired_msas=false since our ColabFold MSAs don't have species info
    # for online pairing (which requires headers in format: str|str|str|species_id|str|str)
    of3_input = {
        "seeds": [DEFAULT_OF3_MODEL_SEED],  # Ensure consistent seeding for reproducibility
        "queries": {
            design_id: {
                "chains": chains,
                "use_paired_msas": False,  # Disable paired MSA creation for heteromers
            }
        }
    }
    
    json_path = work_dir / "query.json"
    json_content = json.dumps(of3_input, indent=2)
    json_path.write_text(json_content)
    
    return json_path


def _parse_openfold3_output(
    output_dir: Path,
    design_id: str,
    seed: int = DEFAULT_OF3_MODEL_SEED,
    sample: int = 1,
) -> Dict[str, Any]:
    """
    Parse OpenFold3 output files.
    
    Output structure:
        output_dir/<design_id>/seed_<seed>/<design_id>_seed_<seed>_sample_<n>_model.cif
        output_dir/<design_id>/seed_<seed>/<design_id>_seed_<seed>_sample_<n>_confidences_aggregated.json
        output_dir/<design_id>/seed_<seed>/<design_id>_seed_<seed>_sample_<n>_confidences.json
    
    Args:
        output_dir: Base output directory from OpenFold3
        design_id: Name used in the input JSON
        seed: Model seed used (default 42)
        sample: Sample index (default 1, OF3 uses 1-based)
    
    Returns:
        Dict with parsed metrics and structure
    """
    import glob
    
    # First, check if the design directory exists
    design_dir = output_dir / design_id
    if not design_dir.exists():
        # List all available directories for debugging
        available_dirs = list(output_dir.iterdir()) if output_dir.exists() else []
        raise FileNotFoundError(
            f"OpenFold3 output directory not found: {design_dir}\n"
            f"Available directories in {output_dir}: {[d.name for d in available_dirs]}"
        )
    
    # Discover the actual seed directory (may be seed_42, seed_<random>, etc.)
    result_dir = None
    
    # Try known patterns
    for pattern in [f"seed_{seed}", str(seed), "seed_*"]:
        candidates = list(design_dir.glob(pattern))
        if candidates:
            result_dir = candidates[0]
            break
    
    if result_dir is None or not result_dir.exists():
        # List available subdirs for debugging
        available_subdirs = list(design_dir.iterdir()) if design_dir.exists() else []
        raise FileNotFoundError(
            f"OpenFold3 seed directory not found in {design_dir}\n"
            f"Available subdirectories: {[d.name for d in available_subdirs]}"
        )
    
    # Extract actual seed from directory name for file matching
    actual_seed = result_dir.name.replace("seed_", "") if result_dir.name.startswith("seed_") else result_dir.name
    
    # File naming pattern - use actual_seed discovered from directory
    file_prefix = f"{design_id}_seed_{actual_seed}_sample_{sample}"
    
    # Aggregated confidence (main metrics) - discover the actual file
    agg_conf_file = None
    
    # Try expected patterns in order of preference
    agg_patterns = [
        f"{file_prefix}_confidences_aggregated.json",
        f"{design_id}_seed_*_sample_{sample}_confidences_aggregated.json",
        "*_confidences_aggregated.json",
    ]
    
    for pattern in agg_patterns:
        matches = list(result_dir.glob(pattern))
        if matches:
            agg_conf_file = matches[0]
            break
    
    if agg_conf_file is None or not agg_conf_file.exists():
        # List all files in result_dir for debugging
        available_files = list(result_dir.iterdir()) if result_dir.exists() else []
        raise FileNotFoundError(
            f"Confidence file not found in {result_dir}\n"
            f"Available files: {[f.name for f in available_files]}"
        )
    
    agg_confidence = json.loads(agg_conf_file.read_text())
    
    # Full confidence (pLDDT per atom, PDE, PAE if enabled)
    full_conf_file = None
    full_patterns = [
        f"{file_prefix}_confidences.json",
        f"{design_id}_seed_*_sample_{sample}_confidences.json",
        "*_confidences.json",
    ]
    
    for pattern in full_patterns:
        matches = [f for f in result_dir.glob(pattern) if "aggregated" not in f.name]
        if matches:
            full_conf_file = matches[0]
            break
    
    full_confidence_text = full_conf_file.read_text() if full_conf_file and full_conf_file.exists() else None
    
    # Structure file (CIF is default, PDB fallback)
    structure_file = None
    structure_patterns = [
        f"{file_prefix}_model.cif",  # Default format is CIF
        f"{file_prefix}_model.pdb",
        f"{design_id}_seed_*_sample_*_model.cif",
        f"{design_id}_seed_*_sample_*_model.pdb",
        "*_model.cif",
        "*_model.pdb",
    ]
    
    for pattern in structure_patterns:
        matches = list(result_dir.glob(pattern))
        if matches:
            structure_file = matches[0]
            break
    
    structure_content = structure_file.read_text() if structure_file and structure_file.exists() else None
    
    # Extract metrics from aggregated confidence
    return {
        "iptm": agg_confidence.get("iptm", 0.0),
        "ptm": agg_confidence.get("ptm", 0.0),
        "plddt": agg_confidence.get("avg_plddt", 0.0),  # Already 0-100 scale
        "gpde": agg_confidence.get("gpde", 0.0),
        "disorder": agg_confidence.get("disorder", 0.0),
        "has_clash": agg_confidence.get("has_clash", False),
        "ranking_score": agg_confidence.get("sample_ranking_score", 0.0),
        "chain_ptm": agg_confidence.get("chain_ptm", {}),
        "chain_pair_iptm": _convert_of3_chain_pair_iptm(agg_confidence.get("chain_pair_iptm", {})),
        "bespoke_iptm": _convert_of3_chain_pair_iptm(agg_confidence.get("bespoke_iptm", {})),
        "structure": structure_content,
        "confidence_json": full_confidence_text,
    }


def _convert_of3_chain_pair_iptm(chain_pair_dict: Dict[str, float]) -> Dict[str, float]:
    """
    Convert OpenFold3 chain pair ipTM format to standard format.
    
    OF3 uses: {"(A, B)": 0.82}
    Convert to: {"A_B": 0.82} for consistency with Protenix
    """
    converted = {}
    for key, value in chain_pair_dict.items():
        # Parse "(A, B)" format
        key_clean = key.strip("()")
        parts = [p.strip() for p in key_clean.split(",")]
        if len(parts) == 2:
            converted[f"{parts[0]}_{parts[1]}"] = value
    return converted


def convert_colabfold_msa_for_openfold3(msa_content: str, query_sequence: str) -> str:
    """
    Convert ColabFold API MSA format to OpenFold3-compatible format.
    
    ColabFold API returns MSAs with headers like:
        >101
        SEQUENCE...
        >UniRef100_A0A0A0A0A0/1-100
        SEQUENCE...
    
    OpenFold3 expects:
    1. Query sequence as the FIRST sequence in the MSA
    2. Proper species-style headers for pairing: <str>|<str>|<str>|<species>|<str>|<str>
    
    Args:
        msa_content: A3M content from ColabFold API
        query_sequence: The actual query sequence for this chain
        
    Returns:
        A3M content with OpenFold3-compatible format
    """
    lines = msa_content.strip().split('\n')
    converted_lines = []
    
    # Parse existing MSA entries
    entries = []
    current_header = None
    current_seq = []
    
    for line in lines:
        if line.startswith('>'):
            if current_header is not None:
                entries.append((current_header, ''.join(current_seq)))
            current_header = line
            current_seq = []
        else:
            current_seq.append(line)
    
    if current_header is not None:
        entries.append((current_header, ''.join(current_seq)))
    
    # Build output: query sequence first
    # Use header format that OpenFold3 expects: >query for first sequence
    converted_lines.append('>query')
    converted_lines.append(query_sequence)
    
    # Add other sequences, converting headers to species format if needed
    for header, seq in entries:
        header_text = header[1:].strip()  # Remove '>'
        
        # Skip the original query sequence (it was typically ">101" or just the sequence)
        seq_clean = seq.replace('-', '').replace('.', '').upper()
        if seq_clean == query_sequence.upper():
            continue
        
        # Skip if it's just a number header (original query)
        if header_text.isdigit():
            continue
        
        # Convert UniRef/UniProt headers to species format
        # e.g., >UniRef100_A0A0A0A0A0/1-100 -> >tr|A0A0A0A0A0|UNKNOWN|9606|1-100|FL=1
        if 'UniRef' in header_text or 'UniProt' in header_text or '|' in header_text:
            # Try to extract species info, default to UNKNOWN
            parts = header_text.replace('/', '|').split('|')
            if len(parts) >= 2:
                accession = parts[1] if len(parts) > 1 else parts[0]
                # Use format: >db|accession|name|species|range|desc
                new_header = f">tr|{accession}|{accession}|9606|1-{len(seq_clean)}|FL=1"
                converted_lines.append(new_header)
            else:
                converted_lines.append(header)
        else:
            converted_lines.append(header)
        
        converted_lines.append(seq)
    
    return '\n'.join(converted_lines)


def ensure_msa_files_for_of3(
    msa_dir: Path,
    target_msas: Dict[str, str],
    target_sequences: Dict[str, str],
) -> None:
    """
    Write MSA content to files for OpenFold3.
    
    OpenFold3 expects:
    - MSA files named 'colabfold_main.a3m' (matching the MSASettings default)
    - Query sequence as the FIRST entry in the MSA
    
    Args:
        msa_dir: Base MSA directory
        target_msas: Dict mapping chain_id -> A3M content
        target_sequences: Dict mapping chain_id -> sequence
    """
    for chain_id, msa_content in target_msas.items():
        chain_msa_dir = msa_dir / f"chain_{chain_id}"
        chain_msa_dir.mkdir(parents=True, exist_ok=True)
        
        # OpenFold3 expects files named 'colabfold_main.a3m'
        msa_file = chain_msa_dir / "colabfold_main.a3m"
        if not msa_file.exists():
            query_seq = target_sequences.get(chain_id, "")
            converted_msa = convert_colabfold_msa_for_openfold3(msa_content, query_seq)
            msa_file.write_text(converted_msa)


# =============================================================================
# PREDICTION IMPLEMENTATIONS
# =============================================================================

def _run_openfold3_prediction(
    input_json_path: Path,
    output_dir: Path,
    checkpoint_path: str = OPENFOLD3_WEIGHTS_PATH,
    num_diffusion_samples: int = 1,  # Single sample for validation (matches AF3/Protenix)
    num_model_seeds: int = 1,
    use_msa_server: bool = False,  # False = use precomputed MSAs
    timeout: int = 1800,
) -> subprocess.CompletedProcess:
    """
    Run OpenFold3 prediction via subprocess with optimized cuEquivariance kernels.
    
    Args:
        input_json_path: Path to query JSON file
        output_dir: Output directory for results
        checkpoint_path: Path to model checkpoint directory
        num_diffusion_samples: Number of diffusion samples per seed (default 5 for quality)
        num_model_seeds: Number of model seeds to run
        use_msa_server: Whether to use ColabFold MSA server
        timeout: Timeout in seconds
    
    Returns:
        subprocess.CompletedProcess result
    """
    import os
    
    # Path to runner config with cuEquivariance + PAE enabled
    runner_yaml = Path(OPENFOLD3_RUNNER_YAML)
    
    cmd = [
        "run_openfold", "predict",
        f"--query_json={input_json_path}",
        f"--output_dir={output_dir}",
        f"--num_diffusion_samples={num_diffusion_samples}",
        f"--num_model_seeds={num_model_seeds}",
        f"--use_msa_server={str(use_msa_server).lower()}",
    ]
    
    # Use runner.yaml for cuEquivariance kernel optimization + PAE enabled
    if runner_yaml.exists():
        cmd.append(f"--runner_yaml={runner_yaml}")
    
    # Point to the actual checkpoint file
    ckpt_file = Path(checkpoint_path or OPENFOLD3_WEIGHTS_PATH) / "of3_ft3_v1.pt"
    if ckpt_file.exists():
        cmd.append(f"--inference_ckpt_path={ckpt_file}")
    
    env = os.environ.copy()
    env["OPENFOLD_CACHE"] = checkpoint_path or OPENFOLD3_WEIGHTS_PATH
    # Ensure cuEquivariance fallback threshold is set
    env.setdefault("CUEQ_TRIATTN_FALLBACK_THRESHOLD", "256")
    # CUDA environment for DeepSpeed
    env["CUDA_HOME"] = "/usr/local/cuda"
    env["DS_BUILD_OPS"] = "0"
    # Create required directories
    os.makedirs("/root/.triton/autotune", exist_ok=True)
    
    proc = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=timeout,
        env=env,
    )
    
    return proc


def _run_openfold3_holo_impl(
    design_id: str,
    binder_seq: str,
    target_seq: str,
    msa_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Run OpenFold3 HOLO prediction (binder + target complex).
    """
    work_dir = Path(tempfile.mkdtemp())
    output_dir = work_dir / "output"
    output_dir.mkdir()
    
    holo_name = f"{design_id}_holo"
    
    # Build input JSON
    input_json = _build_openfold3_input(
        design_id=holo_name,
        binder_seq=binder_seq,
        target_seq=target_seq,
        msa_dir=msa_dir,
        work_dir=work_dir,
    )
    
    print(f"  [{design_id}] Running OpenFold3 HOLO prediction...")
    t0 = time.time()
    
    try:
        proc = _run_openfold3_prediction(
            input_json_path=input_json,
            output_dir=output_dir,
            num_diffusion_samples=1,  # Single sample for validation
            num_model_seeds=1,
            use_msa_server=False,  # Use precomputed MSAs
        )
    except subprocess.TimeoutExpired:
        return {"error": "OpenFold3 HOLO timed out after 30 minutes"}
    except Exception as e:
        return {"error": f"OpenFold3 HOLO failed: {e}"}
    
    elapsed = time.time() - t0
    
    # Log output on failure or verbose mode
    if proc.returncode != 0 or _OF3_VERBOSE:
        if proc.stdout:
            stdout_tail = proc.stdout[-1500:] if len(proc.stdout) > 1500 else proc.stdout
            print(f"  [{design_id}] OF3 stdout (last 1500 chars):\n{stdout_tail}")
        if proc.stderr:
            stderr_tail = proc.stderr[-1500:] if len(proc.stderr) > 1500 else proc.stderr
            print(f"  [{design_id}] OF3 stderr (last 1500 chars):\n{stderr_tail}")
    
    if proc.returncode != 0:
        return {
            "error": f"OpenFold3 HOLO failed with code {proc.returncode}",
            "stdout": proc.stdout,
            "stderr": proc.stderr,
        }
    
    # Parse output
    try:
        result = _parse_openfold3_output(output_dir, holo_name)
        result["design_id"] = design_id
        result["prediction_time"] = round(elapsed, 1)
        print(f"  [{design_id}] OF3 HOLO complete: ipTM={result['iptm']:.3f}, pLDDT={result['plddt']:.1f} ({elapsed:.1f}s)")
        return result
    except Exception as e:
        return {"error": f"Failed to parse OpenFold3 HOLO output: {e}"}


def _run_openfold3_apo_impl(
    design_id: str,
    binder_seq: str,
) -> Dict[str, Any]:
    """
    Run OpenFold3 APO prediction (binder alone).
    """
    work_dir = Path(tempfile.mkdtemp())
    output_dir = work_dir / "output"
    output_dir.mkdir()
    
    apo_name = f"{design_id}_apo"
    
    # Build input JSON (binder only)
    of3_input = {
        "queries": {
            apo_name: {
                "chains": [
                    {
                        "molecule_type": "protein",
                        "chain_ids": ["A"],
                        "sequence": binder_seq,
                    }
                ]
            }
        }
    }
    
    input_json = work_dir / "query.json"
    input_json.write_text(json.dumps(of3_input, indent=2))
    
    print(f"  [{design_id}] Running OpenFold3 APO prediction...")
    t0 = time.time()
    
    try:
        proc = _run_openfold3_prediction(
            input_json_path=input_json,
            output_dir=output_dir,
            num_diffusion_samples=1,  # Single sample for validation
            num_model_seeds=1,
            use_msa_server=False,
        )
    except subprocess.TimeoutExpired:
        return {"error": "OpenFold3 APO timed out"}
    except Exception as e:
        return {"error": f"OpenFold3 APO failed: {e}"}
    
    elapsed = time.time() - t0
    
    if proc.returncode != 0 or _OF3_VERBOSE:
        if proc.stdout:
            print(f"  [{design_id}] OF3 APO stdout:\n{proc.stdout[-1000:]}")
        if proc.stderr:
            print(f"  [{design_id}] OF3 APO stderr:\n{proc.stderr[-1000:]}")
    
    if proc.returncode != 0:
        return {"error": f"OpenFold3 APO failed with code {proc.returncode}"}
    
    try:
        result = _parse_openfold3_output(output_dir, apo_name)
        result["prediction_time"] = round(elapsed, 1)
        print(f"  [{design_id}] OF3 APO complete: pLDDT={result['plddt']:.1f} ({elapsed:.1f}s)")
        return result
    except Exception as e:
        return {"error": f"Failed to parse OpenFold3 APO output: {e}"}


def calculate_openfold3_ipsae(
    confidence_json: str,
    binder_length: int,
    target_length: int,
) -> Dict[str, float]:
    """
    Calculate ipSAE from OpenFold3 confidence JSON.
    
    Note: OF3 may not output PAE by default. The runner.yaml with pae_enabled
    preset should enable PAE output.
    """
    result = {'of3_ipsae': 0.0}
    
    if not confidence_json:
        return result
    
    try:
        confidence = json.loads(confidence_json)
        pae_data = confidence.get("pae", [])
        
        if not pae_data:
            # PAE not available - use PDE as fallback?
            return result
        
        pae_matrix = np.array(pae_data)
        
        ipsae_result = calculate_ipsae_from_pae(
            pae_matrix,
            binder_length=binder_length,
            target_length=target_length,
        )
        
        result['of3_ipsae'] = ipsae_result.get('ipsae', 0.0)
        
    except Exception as e:
        print(f"  Warning: OF3 ipSAE calculation failed: {e}")
    
    return result


# =============================================================================
# MODAL GPU FUNCTIONS
# =============================================================================

@app.function(
    image=openfold3_validation_image,
    gpu="A100",
    timeout=3600,
    volumes={"/cache": cache_volume, OPENFOLD3_WEIGHTS_PATH: openfold3_weights_volume},
    max_containers=20,
)
def run_openfold3_validation_A100(
    design_id: str,
    binder_seq: str,
    target_seq: str,
    target_msas: Optional[Dict[str, str]] = None,
    run_scoring: bool = True,
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    Run complete OpenFold3 validation: HOLO + APO + scoring (A100 GPU).
    """
    return _run_openfold3_validation_impl(
        design_id, binder_seq, target_seq, target_msas, run_scoring, verbose
    )


@app.function(
    image=openfold3_validation_image,
    gpu="A100-80GB",
    timeout=3600,
    volumes={"/cache": cache_volume, OPENFOLD3_WEIGHTS_PATH: openfold3_weights_volume},
    max_containers=20,
)
def run_openfold3_validation_A100_80GB(
    design_id: str,
    binder_seq: str,
    target_seq: str,
    target_msas: Optional[Dict[str, str]] = None,
    run_scoring: bool = True,
    verbose: bool = False,
) -> Dict[str, Any]:
    """Run OpenFold3 validation on A100-80GB GPU."""
    return _run_openfold3_validation_impl(
        design_id, binder_seq, target_seq, target_msas, run_scoring, verbose
    )


@app.function(
    image=openfold3_validation_image,
    gpu="H100",
    timeout=3600,
    volumes={"/cache": cache_volume, OPENFOLD3_WEIGHTS_PATH: openfold3_weights_volume},
    max_containers=20,
)
def run_openfold3_validation_H100(
    design_id: str,
    binder_seq: str,
    target_seq: str,
    target_msas: Optional[Dict[str, str]] = None,
    run_scoring: bool = True,
    verbose: bool = False,
) -> Dict[str, Any]:
    """Run OpenFold3 validation on H100 GPU."""
    return _run_openfold3_validation_impl(
        design_id, binder_seq, target_seq, target_msas, run_scoring, verbose
    )


@app.function(
    image=openfold3_validation_image,
    gpu="L40S",
    timeout=3600,
    volumes={"/cache": cache_volume, OPENFOLD3_WEIGHTS_PATH: openfold3_weights_volume},
    max_containers=20,
)
def run_openfold3_validation_L40S(
    design_id: str,
    binder_seq: str,
    target_seq: str,
    target_msas: Optional[Dict[str, str]] = None,
    run_scoring: bool = True,
    verbose: bool = False,
) -> Dict[str, Any]:
    """Run OpenFold3 validation on L40S GPU."""
    return _run_openfold3_validation_impl(
        design_id, binder_seq, target_seq, target_msas, run_scoring, verbose
    )


def _run_openfold3_validation_impl(
    design_id: str,
    binder_seq: str,
    target_seq: str,
    target_msas: Optional[Dict[str, str]] = None,
    run_scoring: bool = True,
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    Core implementation for OpenFold3 validation.
    
    Runs HOLO + APO predictions, calculates ipSAE, and optionally runs
    open-source interface scoring.
    """
    from modal_boltz_ph.scoring.opensource import (
        _run_opensource_scoring_impl,
        configure_verbose,
    )
    
    configure_verbose(verbose)
    
    global _OF3_VERBOSE
    _OF3_VERBOSE = verbose
    
    # Ensure OF3 weights are available
    ensure_openfold3_weights()
    
    result = {
        "design_id": design_id,
        "validation_model": "openfold3",
    }
    
    # Calculate total target length for ipSAE and build target_sequences map
    target_chains = target_seq.split(":") if target_seq else []
    total_target_length = sum(len(seq) for seq in target_chains)
    
    # Build chain_id -> sequence mapping for MSA conversion
    target_sequences = {}
    for i, seq in enumerate(target_chains):
        chain_id = chr(ord('B') + i)  # B, C, D, ...
        target_sequences[chain_id] = seq
    
    # Setup MSA directory if provided
    msa_dir = None
    if target_msas:
        msa_dir = Path(tempfile.mkdtemp()) / "msas"
        msa_dir.mkdir(parents=True)
        ensure_msa_files_for_of3(msa_dir, target_msas, target_sequences)
    
    # ========== 1. HOLO PREDICTION ==========
    holo_result = _run_openfold3_holo_impl(
        design_id, binder_seq, target_seq, msa_dir
    )
    
    if "error" in holo_result:
        result["error"] = holo_result["error"]
        return result
    
    # Calculate ipSAE
    ipsae_result = {}
    if holo_result.get("confidence_json"):
        ipsae_result = calculate_openfold3_ipsae(
            holo_result["confidence_json"],
            binder_length=len(binder_seq),
            target_length=total_target_length,
        )
    
    result.update({
        # OpenFold3-specific metrics
        "of3_iptm": holo_result["iptm"],
        "of3_ptm": holo_result["ptm"],
        "of3_plddt": holo_result["plddt"],
        "of3_ipsae": ipsae_result.get("of3_ipsae", 0.0),
        "of3_gpde": holo_result.get("gpde", 0.0),
        "of3_disorder": holo_result.get("disorder", 0.0),
        # Generic af3_* keys for pipeline compatibility
        "af3_iptm": holo_result["iptm"],
        "af3_ptm": holo_result["ptm"],
        "af3_plddt": holo_result["plddt"],
        "af3_ipsae": ipsae_result.get("of3_ipsae", 0.0),
        # Additional metrics
        "chain_pair_iptm": holo_result.get("chain_pair_iptm", {}),
        "ranking_score": holo_result.get("ranking_score", 0.0),
        "has_clash": holo_result.get("has_clash", False),
        # Structure outputs
        "holo_structure": holo_result.get("structure"),
        "af3_structure": holo_result.get("structure"),  # Alias for compatibility
        "holo_confidence_json": holo_result.get("confidence_json"),
        "af3_confidence_json": holo_result.get("confidence_json"),  # Alias
    })
    
    # ========== 2. APO PREDICTION ==========
    apo_result = _run_openfold3_apo_impl(design_id, binder_seq)
    
    if "error" not in apo_result:
        result["apo_structure"] = apo_result.get("structure")
        result["apo_plddt"] = apo_result.get("plddt", 0)
    
    # ========== 3. OPEN-SOURCE SCORING ==========
    if run_scoring and result.get("holo_structure"):
        print(f"  [{design_id}] Running open-source interface scoring...")
        
        try:
            scoring_result = _run_opensource_scoring_impl(
                design_id=design_id,
                af3_structure=result["holo_structure"],
                af3_iptm=result["af3_iptm"],
                af3_ptm=result["af3_ptm"],
                af3_plddt=result["af3_plddt"],
                binder_chain="A",
                target_chain="B",
                apo_structure=result.get("apo_structure"),
                af3_confidence_json=result.get("holo_confidence_json"),
                target_type="protein",
                verbose=verbose,
            )
            
            result.update(scoring_result)
            
            status_str = "ACCEPTED" if scoring_result.get("accepted") else f"REJECTED ({scoring_result.get('rejection_reason', 'unknown')})"
            print(f"  [{design_id}] Scoring: SC={scoring_result.get('interface_sc', 0):.2f}, "
                  f"dSASA={scoring_result.get('interface_dSASA', 0):.1f} → {status_str}")
            
        except Exception as e:
            print(f"  [{design_id}] Scoring failed: {e}")
            result["scoring_error"] = str(e)
            result["accepted"] = False
            result["rejection_reason"] = f"Scoring error: {e}"
    
    return result


# =============================================================================
# GPU FUNCTION MAPPING
# =============================================================================

OPENFOLD3_GPU_FUNCTIONS = {
    "A100": run_openfold3_validation_A100,
    "A100-40GB": run_openfold3_validation_A100,
    "A100-80GB": run_openfold3_validation_A100_80GB,
    "H100": run_openfold3_validation_H100,
    "L40S": run_openfold3_validation_L40S,
}

DEFAULT_OPENFOLD3_GPU = "A100"
