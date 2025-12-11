#!/usr/bin/env python3
"""
Modal Boltz Protein Hunter CLI - Design protein binders on Modal GPUs.

Quick Start:
    modal run modal_boltz_ph_cli.py::init_cache                    # Run once
    modal run modal_boltz_ph_cli.py::run_pipeline --help           # See all options
    modal run modal_boltz_ph_cli.py::run_pipeline --name "test" --protein-seqs "AFTVTVPK..."

See run_pipeline docstring for full documentation and examples.
"""

import datetime
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional


def str2bool(v):
    """Convert string to boolean, matching local pipeline behavior."""
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', '0'):
        return False
    else:
        raise ValueError(f'Boolean value expected, got: {v}')


# Import Modal app and shared resources
from modal_boltz_ph.app import app, GPU_TYPES, DEFAULT_GPU

# Import functions from modules
from modal_boltz_ph.design import GPU_FUNCTIONS
from modal_boltz_ph.validation_af3 import (
    AF3_GPU_FUNCTIONS,
    AF3_APO_GPU_FUNCTIONS,
    run_af3_single_A100_80GB,
    run_af3_apo_A100_80GB,
)
from modal_boltz_ph.validation_protenix import (
    PROTENIX_GPU_FUNCTIONS,
    DEFAULT_PROTENIX_GPU,
    run_protenix_validation_A100,
)
from modal_boltz_ph.validation_openfold3 import (
    OPENFOLD3_GPU_FUNCTIONS,
    DEFAULT_OPENFOLD3_GPU,
    run_openfold3_validation_A100,
)
from modal_boltz_ph.validation import (
    get_validation_function,
    get_default_validation_gpu,
    validate_model_gpu_combination,
)
from modal_boltz_ph.scoring_pyrosetta import (
    run_pyrosetta_single,
    configure_verbose as configure_pyrosetta_verbose,
)
from modal_boltz_ph.scoring_opensource import (
    OPENSOURCE_SCORING_GPU_FUNCTIONS,
    DEFAULT_OPENSOURCE_GPU,
    configure_verbose as configure_opensource_verbose,
)
from modal_boltz_ph.cache import initialize_cache, _upload_af3_weights_impl, precompute_msas
from modal_boltz_ph.sync import (
    _sync_worker,
    _stream_best_design,
    _stream_af3_result,
    _stream_final_result,
)
from modal_boltz_ph.tests import test_af3_image, _test_gpu
from modal_boltz_ph.helpers import (
    analyze_template_structure,
    validate_hotspots,
    convert_auth_to_canonical,
    print_target_analysis,
    print_gap_error,
    collapse_to_ranges,
)


# CSV column definitions (af3_* prefix used for both AF3 and Protenix validation)
CSV_EXCLUDE = {
    "relaxed_pdb", "_target_msas", "af3_confidence_json", "af3_structure",
    "apo_structure", "best_pdb", "cycles", "target_msas"
}

UNIFIED_DESIGN_COLUMNS = [
    "design_id", "design_num", "cycle",
    "binder_sequence", "binder_length", "cyclic", "alanine_count", "alanine_pct",
    "boltz_iptm", "boltz_ipsae", "boltz_plddt", "boltz_iplddt",
    "af3_iptm", "af3_ipsae", "af3_ptm", "af3_plddt",
    "interface_dG", "interface_sc", "interface_nres", "interface_dSASA",
    "interface_packstat", "interface_hbonds", "interface_delta_unsat_hbonds",
    "apo_holo_rmsd", "i_pae", "rg",
    "accepted", "rejection_reason",
    "contact_residues", "contact_residues_auth", "template_first_residue",
]


@app.local_entrypoint()
def upload_af3_weights(weights_path: str):
    """Upload AlphaFold3 weights to Modal volume."""
    import subprocess as sp
    
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
    """Initialize the cache (download model weights). Run once before using the pipeline."""
    print("Initializing Protein Hunter cache...")
    result = initialize_cache.remote()
    print(result)


@app.local_entrypoint()
def run_pipeline(
    # Job identity
    name: str = "protein_hunter_run",
    # Target specification
    protein_seqs: Optional[str] = None,
    ligand_ccd: Optional[str] = None,
    ligand_smiles: Optional[str] = None,
    nucleic_seq: Optional[str] = None,
    nucleic_type: str = "dna",
    # Template
    template_path: Optional[str] = None,
    template_cif_chain_id: Optional[str] = None,
    # Binder configuration
    seq: Optional[str] = None,
    min_protein_length: int = 100,
    max_protein_length: int = 150,
    percent_x: int = 90,
    cyclic: str = "false",
    exclude_p: str = "false",
    # Design parameters
    num_designs: int = 50,
    num_cycles: int = 5,
    contact_residues: Optional[str] = None,
    use_auth_numbering: str = "false",  # NEW: Interpret hotspots in auth/PDB numbering
    temperature: float = 0.1,
    omit_aa: str = "C",
    alanine_bias: str = "true",
    alanine_bias_start: float = -0.5,
    alanine_bias_end: float = -0.1,
    high_iptm_threshold: float = 0.8,
    high_plddt_threshold: float = 0.8,
    # Contact filtering
    no_contact_filter: str = "false",
    max_contact_filter_retries: int = 6,
    contact_cutoff: float = 15.0,
    # MSA options
    msa_mode: str = "mmseqs",  # "single" or "mmseqs"
    # Model parameters
    diffuse_steps: int = 200,
    recycling_steps: int = 3,
    randomly_kill_helix_feature: str = "false",
    negative_helix_constant: float = 0.2,
    grad_enabled: str = "false",
    logmd: str = "false",
    # Execution
    gpu: str = DEFAULT_GPU,
    max_concurrent: int = 0,  # 0 = unlimited
    output_dir: Optional[str] = None,
    no_stream: str = "false",
    sync_interval: float = 5.0,
    # Validation options (see docstring for details)
    validation_model: str = "none",
    scoring_method: str = "pyrosetta",
    validation_gpu: Optional[str] = None,
    use_msa_for_validation: str = "true",
    # Deprecated flags (still work, emit warnings)
    use_alphafold3_validation: str = "false",
    use_msa_for_af3: str = "true",
    af3_gpu: str = "A100-80GB",
    use_open_scoring: str = "false",
    open_scoring_gpu: str = DEFAULT_OPENSOURCE_GPU,
    # Verbosity
    verbose: str = "false",
):
    """
    Run the Protein Hunter design pipeline on Modal.
    
    EXAMPLES
    --------
    # Basic design (no validation)
    modal run modal_boltz_ph_cli.py::run_pipeline \\
        --name "PDL1_binder" --protein-seqs "AFTVTVPK..." --num-designs 5

    # With Protenix validation (RECOMMENDED - fully open-source)
    modal run modal_boltz_ph_cli.py::run_pipeline \\
        --name "PDL1_open" --protein-seqs "AFTVTVPK..." \\
        --validation-model protenix --scoring-method opensource

    # With OpenFold3 validation (Apache 2.0, fully open-source)
    modal run modal_boltz_ph_cli.py::run_pipeline \\
        --name "PDL1_of3" --protein-seqs "AFTVTVPK..." \\
        --validation-model openfold3 --scoring-method opensource

    # With AF3 validation (requires weights upload first)
    modal run modal_boltz_ph_cli.py::run_pipeline \\
        --name "PDL1_af3" --protein-seqs "AFTVTVPK..." \\
        --validation-model af3 --validation-gpu A100-80GB

    # With template and hotspots (PDB numbering)
    modal run modal_boltz_ph_cli.py::run_pipeline \\
        --name "CD33" --template-path "CD33.pdb" --template-cif-chain-id "B" \\
        --contact-residues "69,72" --use-auth-numbering

    # Small molecule binder
    modal run modal_boltz_ph_cli.py::run_pipeline \\
        --name "SAM_binder" --ligand-ccd "SAM" --num-designs 5
    
    VALIDATION & SCORING
    --------------------
    --validation-model {none,af3,protenix,openfold3}
        - none: Design only (default)
        - protenix: Open-source AF3 reproduction (recommended)
        - openfold3: AlQuraishi Lab's open-source AF3 reproduction
        - af3: AlphaFold3 (requires weights - see AF3 SETUP)

    --scoring-method {pyrosetta,opensource}
        - opensource: OpenMM + FreeSASA (no license needed)
        - pyrosetta: Full PyRosetta scoring

    --validation-gpu: A100 (protenix default), A100-80GB (af3 default)
    --use-msa-for-validation: Reuse MSAs from design phase (default: true)

    HOTSPOT NUMBERING
    -----------------
    --contact-residues "69,72,115"                        # Canonical (1-indexed)
    --contact-residues "69,72" --use-auth-numbering       # PDB/auth numbering
    --contact-residues "54,56|115"                        # Multi-chain (| separator)
    --contact-residues "||1,2,3"                          # Third chain only

    TEMPLATE AUTO-EXTRACTION
    ------------------------
    When --template-path is provided without --protein-seqs:
    - CIF: Extracts full sequence from _entity_poly_seq
    - PDB: Parses residues; errors if gaps detected
    Provide --protein-seqs with full sequence if gaps are detected.

    AF3 SETUP
    ---------
    Required only for --validation-model=af3:
        modal run modal_boltz_ph_cli.py::upload_af3_weights --weights-path ~/AF3/af3.bin.zst

    DEPRECATED FLAGS
    ----------------
    --use-alphafold3-validation → --validation-model=af3
    --use-open-scoring          → --scoring-method=opensource
    --use-msa-for-af3           → --use-msa-for-validation
    --af3-gpu / --open-scoring-gpu → --validation-gpu
    """
    import base64
    import warnings
    import pandas as pd
    
    # Convert string boolean parameters to actual booleans
    cyclic = str2bool(cyclic)
    exclude_p = str2bool(exclude_p)
    alanine_bias = str2bool(alanine_bias)
    no_contact_filter = str2bool(no_contact_filter)
    use_auth_numbering = str2bool(use_auth_numbering)
    randomly_kill_helix_feature = str2bool(randomly_kill_helix_feature)
    grad_enabled = str2bool(grad_enabled)
    logmd = str2bool(logmd)
    no_stream = str2bool(no_stream)
    use_alphafold3_validation = str2bool(use_alphafold3_validation)
    use_msa_for_af3 = str2bool(use_msa_for_af3)
    use_msa_for_validation = str2bool(use_msa_for_validation)
    use_open_scoring = str2bool(use_open_scoring)
    verbose = str2bool(verbose)
    
    # Configure verbose logging for scoring modules
    configure_opensource_verbose(verbose)
    configure_pyrosetta_verbose(verbose)
    
    # Handle deprecated flags
    # --use-alphafold3-validation (deprecated)
    if use_alphafold3_validation and validation_model == "none":
        warnings.warn(
            "--use-alphafold3-validation is deprecated. "
            "Use --validation-model=af3 instead.",
            DeprecationWarning,
            stacklevel=2
        )
        validation_model = "af3"
    
    # Handle --use-open-scoring (deprecated)
    if use_open_scoring and scoring_method == "pyrosetta":
        warnings.warn(
            "--use-open-scoring is deprecated. "
            "Use --scoring-method=opensource instead.",
            DeprecationWarning,
            stacklevel=2
        )
        scoring_method = "opensource"
    
    # Handle --af3-gpu (deprecated, but only if using af3)
    if validation_model == "af3" and validation_gpu is None:
        validation_gpu = af3_gpu
    
    # Handle --open-scoring-gpu (deprecated)
    if validation_model == "af3" and scoring_method == "opensource" and validation_gpu is None:
        validation_gpu = open_scoring_gpu
    
    # Handle --use-msa-for-af3 (deprecated)
    if not use_msa_for_af3:
        use_msa_for_validation = False
    
    # Set default validation GPU if not specified
    if validation_model != "none" and validation_gpu is None:
        validation_gpu = get_default_validation_gpu(validation_model)
    
    # Validate GPU selection
    if validation_model != "none":
        is_valid, error_msg = validate_model_gpu_combination(validation_model, validation_gpu)
        if not is_valid:
            print(f"Error: {error_msg}")
            return
    
    # Determine if validation is enabled
    validation_enabled = validation_model != "none"

    # Smart MSA mode default based on template and validation
    # - Template provided + no validation: skip MSAs (Boltz has template, no validation needs)
    # - Template provided + validation enabled: compute MSAs (AF3/Protenix need them)
    # - No template: compute MSAs (Boltz needs them)
    if template_path and msa_mode == "mmseqs" and not validation_enabled:
        print("Note: Template provided without validation - switching to --msa-mode single")
        print("      (Boltz uses template structure; MSAs not needed)\n")
        msa_mode = "single"
    
    stream = not no_stream
    run_id = f"{name}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # Template analysis and sequence auto-extraction
    template_content = ""
    template_analysis = None
    contact_residues_canonical = contact_residues  # Will be converted if using auth numbering
    contact_residues_auth = None  # Store original auth values for CSV output
    
    if template_path:
        template_file = Path(template_path)
        if not template_file.exists():
            print(f"Error: Template file not found: {template_path}")
            return
        
        # Parse template chain IDs - default to 'A' if not specified
        template_chains = [c.strip() for c in (template_cif_chain_id or "").split(",") if c.strip()]
        if not template_chains:
            print("⚠ Warning: No --template-cif-chain-id specified, defaulting to chain 'A'")
            template_chains = ["A"]
            template_cif_chain_id = "A"  # Update for downstream use
        
        # Analyze template structure
        template_bytes = template_file.read_bytes()
        template_analysis = analyze_template_structure(
            template_bytes,
            template_chains,
            filename=template_file.name,
        )
        
        if not template_analysis["success"]:
            print(f"\n{template_analysis['error']}")
            return
        
        # Check for gaps in each chain
        has_any_gaps = False
        for chain_id, chain_data in template_analysis["chains"].items():
            if chain_data["has_gaps"]:
                has_any_gaps = True
                if not protein_seqs:
                    # Cannot auto-extract - error with guidance
                    print(f"\n{print_gap_error(chain_id, chain_data)}")
                    return
        
        # Auto-extract sequences if not provided
        if not protein_seqs:
            if has_any_gaps:
                # Should not reach here (handled above), but be safe
                print("Error: Template has gaps. Please provide --protein-seqs with full sequence.")
                return
            
            # Build protein_seqs from extracted sequences
            extracted_seqs = []
            for chain_id in template_chains:
                chain_data = template_analysis["chains"].get(chain_id)
                if chain_data:
                    extracted_seqs.append(chain_data["sequence"])
                    print(f"Auto-extracted chain {chain_id}: {len(chain_data['sequence'])} residues")
                else:
                    print(f"Warning: No data for chain {chain_id}")
            
            if extracted_seqs:
                protein_seqs = ":".join(extracted_seqs)
                print(f"✓ Auto-extracted {len(extracted_seqs)} chain(s) from template")
        
        # Validate and convert hotspots
        if contact_residues:
            contact_lists = contact_residues.split("|")
            converted_lists = []
            
            for i, res_str in enumerate(contact_lists):
                if not res_str.strip():
                    converted_lists.append("")
                    continue
                
                # Parse positions
                positions = [int(x.strip()) for x in res_str.split(",") if x.strip()]
                if not positions:
                    converted_lists.append("")
                    continue
                
                # Get chain data
                chain_id = template_chains[i] if i < len(template_chains) else template_chains[0]
                chain_data = template_analysis["chains"].get(chain_id)
                
                if not chain_data:
                    print(f"Error: Chain {chain_id} not found in template analysis")
                    return
                
                # Validate hotspots
                if use_auth_numbering:
                    auth_set = set(chain_data["auth_residues"])
                    is_valid, error_msg = validate_hotspots(
                        positions,
                        len(chain_data["sequence"]),
                        chain_id,
                        use_auth_numbering=True,
                        auth_residue_set=auth_set,
                        auth_range=chain_data["auth_range"],
                    )
                    if not is_valid:
                        print(f"\n{error_msg}")
                        return
                    
                    # Convert to canonical
                    canonical_positions = convert_auth_to_canonical(
                        positions,
                        chain_data["auth_to_canonical"],
                    )
                    converted_lists.append(",".join(str(p) for p in canonical_positions))
                else:
                    # Canonical numbering - validate range
                    is_valid, error_msg = validate_hotspots(
                        positions,
                        len(chain_data["sequence"]),
                        chain_id,
                        use_auth_numbering=False,
                    )
                    if not is_valid:
                        print(f"\n{error_msg}")
                        return
                    
                    converted_lists.append(res_str)
            
            # Store original auth values and converted canonical values
            if use_auth_numbering:
                contact_residues_auth = contact_residues
                contact_residues_canonical = "|".join(converted_lists)
                print(f"✓ Converted hotspots: auth → canonical")
            else:
                contact_residues_canonical = contact_residues
        
        # Print target analysis visualization
        print_target_analysis(
            template_analysis["chains"],
            contact_residues,
            use_auth_numbering,
            template_filename=template_file.name,
        )
        
        # Encode template for upload
        template_content = base64.b64encode(template_bytes).decode('utf-8')
    
    # Validate inputs (protein_seqs may have been auto-extracted above)
    if not any([protein_seqs, ligand_ccd, ligand_smiles, nucleic_seq]):
        print("Error: Must provide at least one target (--protein-seqs, --ligand-ccd, --ligand-smiles, or --nucleic-seq)")
        return
    
    # Auto-derive target_type from inputs (replaces manual flag)
    if nucleic_seq:
        target_type = "nucleic"
    elif ligand_ccd or ligand_smiles:
        target_type = "small_molecule"
    else:
        target_type = "protein"
    
    # Setup output directory
    output_path = Path(output_dir) if output_dir else Path(f"./results_{name}")
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Print configuration
    print(f"\n{'='*70}")
    print("PROTEIN HUNTER (Modal)")
    print(f"{'='*70}")
    print(f"Run ID: {run_id}")
    print(f"Target: {protein_seqs[:50] + '...' if protein_seqs and len(protein_seqs) > 50 else protein_seqs or ligand_ccd or nucleic_seq}")
    print(f"Target type: {target_type}")
    print(f"Num designs: {num_designs}")
    print(f"Num cycles: {num_cycles}")
    print(f"GPU: {gpu}")
    print(f"Max concurrent: {max_concurrent if max_concurrent > 0 else 'unlimited'}")
    print(f"Output: {output_path}")
    if template_path:
        print(f"Template: {template_path}")
        print(f"Template chains: {template_cif_chain_id}")
    if contact_residues_canonical:
        if use_auth_numbering and contact_residues_auth:
            print(f"Hotspots (auth): {contact_residues_auth}")
            print(f"Hotspots (canonical): {contact_residues_canonical}")
        else:
            print(f"Hotspots: {contact_residues_canonical}")
    print(f"MSA mode: {msa_mode}")
    print(f"Alanine bias: {alanine_bias}")
    if validation_enabled:
        print(f"Validation: {validation_model} on {validation_gpu}")
        scoring_display = "open-source (OpenMM + FreeSASA)" if scoring_method == "opensource" else "PyRosetta"
        print(f"Scoring: {scoring_display}")
    print(f"{'='*70}\n")
    
    # Build template metadata for CSV output
    template_first_residues = {}
    if template_analysis:
        for chain_id, chain_data in template_analysis["chains"].items():
            template_first_residues[chain_id] = chain_data["first_auth_residue"]
    
    # Pre-compute MSAs (once, before dispatching parallel design tasks)
    precomputed_msas = {}
    if msa_mode == "mmseqs" and protein_seqs:
        # Get unique sequences to avoid redundant API calls
        protein_seqs_list = protein_seqs.split(":") if protein_seqs else []
        unique_seqs = list(set(seq for seq in protein_seqs_list if seq))
        
        if unique_seqs:
            print(f"Pre-computing MSAs for {len(unique_seqs)} unique target sequence(s)...")
            print("This avoids rate limiting when running parallel designs.\n")
            
            try:
                precomputed_msas = precompute_msas.remote(unique_seqs)
                print(f"✓ Pre-computed {len(precomputed_msas)} MSA(s)\n")
            except Exception as e:
                print(f"⚠ MSA pre-computation failed: {e}")
                print("  Falling back to per-design MSA fetching (may hit rate limits)\n")
                precomputed_msas = {}
    
    # Build tasks
    tasks = []
    for i in range(num_designs):
        task = {
            "run_id": run_id,
            "design_idx": i,
            "total_designs": num_designs,
            "stream_to_dict": stream,
            # Target
            "protein_seqs": protein_seqs or "",
            "ligand_ccd": ligand_ccd or "",
            "ligand_smiles": ligand_smiles or "",
            "nucleic_seq": nucleic_seq or "",
            "nucleic_type": nucleic_type,
            "template_content": template_content,
            "template_chain_ids": template_cif_chain_id or "",
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
            "contact_residues": contact_residues_canonical or "",  # Use canonical positions
            "contact_residues_auth": contact_residues_auth or "",  # Original auth values (for traceability)
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
            # Template metadata (for CSV output)
            "template_first_residues": template_first_residues,
            # Pre-computed MSAs (to avoid rate limiting)
            "precomputed_msas": precomputed_msas,
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
    
    # ==========================================================================
    # END-TO-END PER-DESIGN ORCHESTRATION
    # Each design completes: Boltz → Validation → Scoring before next slot freed
    # ==========================================================================
    
    # Select validation functions (if needed)
    # For AF3, we still use the individual function approach for fine-grained control
    # For Protenix with opensource scoring, we use the bundled approach
    af3_gpu_to_use = validation_gpu if validation_model == "af3" else (gpu if gpu in AF3_GPU_FUNCTIONS else "A100-80GB")
    af3_fn = AF3_GPU_FUNCTIONS.get(af3_gpu_to_use, run_af3_single_A100_80GB)
    af3_apo_fn = AF3_APO_GPU_FUNCTIONS.get(af3_gpu_to_use, run_af3_apo_A100_80GB)
    
    # For Protenix, select the appropriate GPU function
    protenix_gpu_to_use = validation_gpu if validation_model == "protenix" else DEFAULT_PROTENIX_GPU
    protenix_fn = PROTENIX_GPU_FUNCTIONS.get(protenix_gpu_to_use, run_protenix_validation_A100)
    
    # For OpenFold3, select the appropriate GPU function
    openfold3_gpu_to_use = validation_gpu if validation_model == "openfold3" else DEFAULT_OPENFOLD3_GPU
    openfold3_fn = OPENFOLD3_GPU_FUNCTIONS.get(openfold3_gpu_to_use, run_openfold3_validation_A100)
    
    def _run_full_pipeline_for_design(task_input: dict) -> dict:
        """
        Run complete pipeline for one design: Boltz → Validation → APO → Scoring.
        
        Validation can be AF3 or Protenix (set via --validation-model).
        Scoring can be PyRosetta or open-source (set via --scoring-method).
        All stages run sequentially for this design before the thread is freed.
        """
        design_idx = task_input.get("design_idx", 0)
        
        # ========== STAGE 1: BOLTZ DESIGN ==========
        try:
            design_result = gpu_fn.remote(task_input)
        except Exception as e:
            return {
                "status": "error",
                "error": f"Design failed: {e}",
                "design_idx": design_idx,
                "stage_failed": "design",
                "best_iptm": 0,
            }
        
        if design_result.get("status") != "success":
            return {**design_result, "stage_failed": "design"}
        
        # Stream best design result
        best_cycle = design_result.get("best_cycle", 0)
        design_id = f"{name}_d{design_idx}_c{best_cycle}"
        best_seq = design_result.get("best_seq", "")
        best_pdb = design_result.get("best_pdb", "")
        
        # Get metrics from best cycle
        best_cycle_data = None
        for cycle_data in design_result.get("cycles", []):
            if cycle_data.get("cycle") == best_cycle:
                best_cycle_data = cycle_data
                break
        
        if stream and best_seq:
            _stream_best_design(
                run_id=run_id,
                design_idx=design_idx,
                design_id=design_id,
                best_cycle=best_cycle,
                best_seq=best_seq,
                best_pdb=best_pdb,
                metrics={
                    "iptm": design_result.get("best_iptm", 0.0),
                    "ipsae": best_cycle_data.get("ipsae", 0.0) if best_cycle_data else 0.0,
                    "plddt": best_cycle_data.get("plddt", 0.0) if best_cycle_data else 0.0,
                    "iplddt": best_cycle_data.get("iplddt", 0.0) if best_cycle_data else 0.0,
                    "alanine_count": best_cycle_data.get("alanine_count", 0) if best_cycle_data else 0,
                    "cyclic": task_input.get("cyclic", False),
                },
            )
        
        # If no validation requested, return design result only
        if not validation_enabled:
            return design_result
        
        # Skip validation if no valid design was found (best_seq is None/empty)
        if not best_seq:
            print(f"  [{design_id}] Skipping validation: no valid design (all cycles had alanine >20%)")
            return {
                **design_result,
                "design_id": design_id,
                "accepted": False,
                "rejection_reason": "no cycle passed alanine threshold (≤20%)",
            }
        
        # Skip validation if Boltz metrics don't meet thresholds
        boltz_iptm = design_result.get("best_iptm", 0.0)
        boltz_plddt = design_result.get("best_plddt", 0.0)
        
        iptm_ok = boltz_iptm >= high_iptm_threshold
        plddt_ok = boltz_plddt >= high_plddt_threshold
        
        if not (iptm_ok and plddt_ok):
            failure_reasons = []
            if not iptm_ok:
                failure_reasons.append(f"ipTM {boltz_iptm:.3f} < {high_iptm_threshold}")
            if not plddt_ok:
                failure_reasons.append(f"pLDDT {boltz_plddt:.2f} < {high_plddt_threshold}")
            rejection_reason = "; ".join(failure_reasons)
            print(f"  [{design_id}] Skipping validation: {rejection_reason}")
            return {
                **design_result,
                "design_id": design_id,
                "accepted": False,
                "rejection_reason": rejection_reason,
            }
        
        # ========== STAGE 2: STRUCTURE VALIDATION ==========
        # Validates the designed complex using AF3 or Protenix (open-source)
        binder_seq = best_seq
        target_seq = task_input.get("protein_seqs", "")
        target_msas = design_result.get("target_msas", {}) if use_msa_for_validation else {}

        print(f"  [{design_id}] Starting {validation_model.upper()} validation...")
        
        # Get template info from task_input for validation
        task_template_content = task_input.get("template_content", "")
        task_template_chain_ids = task_input.get("template_chain_ids", "")
        
        validation_result = {}
        apo_structure = None
        
        if validation_model == "af3":
            # ===== AF3 VALIDATION =====
            try:
                af3_result = af3_fn.remote(
                    design_id, binder_seq, target_seq,
                    "A", "B",  # binder_chain, starting target_chain
                    target_msas,  # Full dict of chain_id -> MSA content
                    "reuse" if use_msa_for_validation else "none",
                    task_template_content if task_template_content else None,      # base64-encoded template
                    task_template_chain_ids if task_template_content else None     # chain IDs
                )
            except Exception as e:
                print(f"  [{design_id}] AF3 failed: {e}")
                return {**design_result, "design_id": design_id, "af3_error": str(e), "stage_failed": "validation"}
            
            if not af3_result.get("af3_structure"):
                return {**design_result, "design_id": design_id, **af3_result, "stage_failed": "validation"}
            
            validation_result = af3_result
            print(f"  [{design_id}] AF3 complete: ipTM={af3_result.get('af3_iptm', 0):.3f}, ipSAE={af3_result.get('af3_ipsae', 0):.3f}")
            
        elif validation_model == "protenix":
            # ===== PROTENIX VALIDATION =====
            # Protenix with opensource scoring runs in a single container
            run_bundled_scoring = (scoring_method == "opensource")
            
            try:
                protenix_result = protenix_fn.remote(
                    design_id, binder_seq, target_seq,
                    target_msas,
                    run_scoring=run_bundled_scoring,  # Bundled scoring if opensource
                    verbose=verbose,
                )
            except Exception as e:
                print(f"  [{design_id}] Protenix failed: {e}")
                return {**design_result, "design_id": design_id, "protenix_error": str(e), "stage_failed": "validation"}
            
            if not protenix_result.get("af3_structure"):
                return {**design_result, "design_id": design_id, **protenix_result, "stage_failed": "validation"}
            
            validation_result = protenix_result
            apo_structure = protenix_result.get("apo_structure")
            
            # Log results
            iptm = protenix_result.get("af3_iptm", protenix_result.get("protenix_iptm", 0))
            ipsae = protenix_result.get("af3_ipsae", protenix_result.get("protenix_ipsae", 0))
            print(f"  [{design_id}] Protenix complete: ipTM={iptm:.3f}, ipSAE={ipsae:.3f}")
            
            # If bundled scoring was used, we can skip stages 3 and 4
            if run_bundled_scoring:
                # Stream AF3 result (Protenix outputs are aliased to af3_* for compatibility)
                if stream:
                    _stream_af3_result(
                        run_id=run_id,
                        design_idx=design_idx,
                        design_id=design_id,
                        af3_iptm=validation_result.get("af3_iptm", 0.0),
                        af3_ipsae=validation_result.get("af3_ipsae", 0.0),
                        af3_ptm=validation_result.get("af3_ptm", 0.0),
                        af3_plddt=validation_result.get("af3_plddt", 0.0),
                        af3_structure=validation_result.get("af3_structure", ""),
                    )
                
                # Stream final result
                if stream and target_type == "protein":
                    _stream_final_result(
                        run_id=run_id,
                        design_idx=design_idx,
                        design_id=design_id,
                        accepted=validation_result.get("accepted", False),
                        rejection_reason=validation_result.get("rejection_reason", ""),
                        metrics={
                            "interface_dG": validation_result.get("interface_dG", 0.0),
                            "interface_sc": validation_result.get("interface_sc", 0.0),
                            "interface_nres": validation_result.get("interface_nres", 0),
                            "interface_dSASA": validation_result.get("interface_dSASA", 0.0),
                            "interface_packstat": validation_result.get("interface_packstat", 0.0),
                            "interface_dG_SASA_ratio": validation_result.get("interface_dG_SASA_ratio", 0.0),
                            "interface_interface_hbonds": validation_result.get("interface_interface_hbonds", 0),
                            "interface_delta_unsat_hbonds": validation_result.get("interface_delta_unsat_hbonds", 0),
                            "interface_hydrophobicity": validation_result.get("interface_hydrophobicity", 0.0),
                            "surface_hydrophobicity": validation_result.get("surface_hydrophobicity", 0.0),
                            "binder_sasa": validation_result.get("binder_sasa", 0.0),
                            "interface_fraction": validation_result.get("interface_fraction", 0.0),
                            "interface_hbond_percentage": validation_result.get("interface_hbond_percentage", 0.0),
                            "interface_delta_unsat_hbonds_percentage": validation_result.get("interface_delta_unsat_hbonds_percentage", 0.0),
                            "apo_holo_rmsd": validation_result.get("apo_holo_rmsd"),
                            "i_pae": validation_result.get("i_pae"),
                            "rg": validation_result.get("rg"),
                        },
                        relaxed_pdb=validation_result.get("relaxed_pdb", ""),
                    )
                
                # Return combined results (Protenix bundled includes scoring)
                return {
                    **design_result,
                    "design_id": design_id,
                    "design_num": design_idx,
                    "cycle": best_cycle,
                    **validation_result,
                    "binder_sequence": binder_seq,
                    "binder_length": len(binder_seq) if binder_seq else 0,
                    "ipsae": best_cycle_data.get("ipsae", 0.0) if best_cycle_data else 0.0,
                    "plddt": best_cycle_data.get("plddt", 0.0) if best_cycle_data else 0.0,
                    "iplddt": best_cycle_data.get("iplddt", 0.0) if best_cycle_data else 0.0,
                }
        
        elif validation_model == "openfold3":
            # ===== OPENFOLD3 VALIDATION =====
            # OpenFold3 with opensource scoring runs in a single container
            run_bundled_scoring = (scoring_method == "opensource")
            
            try:
                of3_result = openfold3_fn.remote(
                    design_id, binder_seq, target_seq,
                    target_msas,
                    run_scoring=run_bundled_scoring,  # Bundled scoring if opensource
                    verbose=verbose,
                )
            except Exception as e:
                print(f"  [{design_id}] OpenFold3 failed: {e}")
                return {**design_result, "design_id": design_id, "openfold3_error": str(e), "stage_failed": "validation"}
            
            if not of3_result.get("af3_structure"):
                return {**design_result, "design_id": design_id, **of3_result, "stage_failed": "validation"}
            
            validation_result = of3_result
            apo_structure = of3_result.get("apo_structure")
            
            # Log results
            iptm = of3_result.get("af3_iptm", of3_result.get("of3_iptm", 0))
            ipsae = of3_result.get("af3_ipsae", of3_result.get("of3_ipsae", 0))
            print(f"  [{design_id}] OpenFold3 complete: ipTM={iptm:.3f}, ipSAE={ipsae:.3f}")
            
            # If bundled scoring was used, we can skip stages 3 and 4
            if run_bundled_scoring:
                # Stream AF3 result (OpenFold3 outputs are aliased to af3_* for compatibility)
                if stream:
                    _stream_af3_result(
                        run_id=run_id,
                        design_idx=design_idx,
                        design_id=design_id,
                        af3_iptm=validation_result.get("af3_iptm", 0.0),
                        af3_ipsae=validation_result.get("af3_ipsae", 0.0),
                        af3_ptm=validation_result.get("af3_ptm", 0.0),
                        af3_plddt=validation_result.get("af3_plddt", 0.0),
                        af3_structure=validation_result.get("af3_structure", ""),
                    )
                
                # Stream final result
                if stream and target_type == "protein":
                    _stream_final_result(
                        run_id=run_id,
                        design_idx=design_idx,
                        design_id=design_id,
                        accepted=validation_result.get("accepted", False),
                        rejection_reason=validation_result.get("rejection_reason", ""),
                        metrics={
                            "interface_dG": validation_result.get("interface_dG", 0.0),
                            "interface_sc": validation_result.get("interface_sc", 0.0),
                            "interface_nres": validation_result.get("interface_nres", 0),
                            "interface_dSASA": validation_result.get("interface_dSASA", 0.0),
                            "interface_packstat": validation_result.get("interface_packstat", 0.0),
                            "interface_dG_SASA_ratio": validation_result.get("interface_dG_SASA_ratio", 0.0),
                            "interface_interface_hbonds": validation_result.get("interface_interface_hbonds", 0),
                            "interface_delta_unsat_hbonds": validation_result.get("interface_delta_unsat_hbonds", 0),
                            "interface_hydrophobicity": validation_result.get("interface_hydrophobicity", 0.0),
                            "surface_hydrophobicity": validation_result.get("surface_hydrophobicity", 0.0),
                            "binder_sasa": validation_result.get("binder_sasa", 0.0),
                            "interface_fraction": validation_result.get("interface_fraction", 0.0),
                            "interface_hbond_percentage": validation_result.get("interface_hbond_percentage", 0.0),
                            "interface_delta_unsat_hbonds_percentage": validation_result.get("interface_delta_unsat_hbonds_percentage", 0.0),
                            "apo_holo_rmsd": validation_result.get("apo_holo_rmsd"),
                            "i_pae": validation_result.get("i_pae"),
                            "rg": validation_result.get("rg"),
                        },
                        relaxed_pdb=validation_result.get("relaxed_pdb", ""),
                    )
                
                # Return combined results (OpenFold3 bundled includes scoring)
                return {
                    **design_result,
                    "design_id": design_id,
                    "design_num": design_idx,
                    "cycle": best_cycle,
                    **validation_result,
                    "binder_sequence": binder_seq,
                    "binder_length": len(binder_seq) if binder_seq else 0,
                    "ipsae": best_cycle_data.get("ipsae", 0.0) if best_cycle_data else 0.0,
                    "plddt": best_cycle_data.get("plddt", 0.0) if best_cycle_data else 0.0,
                    "iplddt": best_cycle_data.get("iplddt", 0.0) if best_cycle_data else 0.0,
                }
        
        # For AF3, we continue to the APO and scoring stages
        af3_result = validation_result
        
        # Stream AF3 result
        if stream:
            _stream_af3_result(
                run_id=run_id,
                design_idx=design_idx,
                design_id=design_id,
                af3_iptm=af3_result.get("af3_iptm", 0.0),
                af3_ipsae=af3_result.get("af3_ipsae", 0.0),
                af3_ptm=af3_result.get("af3_ptm", 0.0),
                af3_plddt=af3_result.get("af3_plddt", 0.0),
                af3_structure=af3_result.get("af3_structure", ""),
            )
        
        # ========== STAGE 3: APO PREDICTION (protein targets only) ==========
        # Skip if Protenix or OpenFold3 already generated APO structure
        if target_type == "protein" and apo_structure is None:
            try:
                apo_result = af3_apo_fn.remote(design_id, binder_seq, "A")
                apo_structure = apo_result.get("apo_structure")
                if apo_structure:
                    print(f"  [{design_id}] APO structure generated")
            except Exception as e:
                # APO failure is non-fatal, continue without RMSD
                print(f"  [{design_id}] APO prediction failed (non-fatal): {e}")
        
        # ========== STAGE 4: INTERFACE SCORING (protein targets only) ==========
        # Uses PyRosetta (default) or open-source scoring (OpenMM + FreeSASA)
        # based on --scoring-method flag
        scoring_result = {"accepted": True}  # Default for non-protein targets
        if target_type == "protein":
            if scoring_method == "opensource":
                # Open-source scoring (GPU-accelerated, PyRosetta-free)
                # Use a smaller GPU for scoring when AF3 uses A100-80GB
                scoring_gpu_to_use = open_scoring_gpu if validation_model == "af3" else DEFAULT_OPENSOURCE_GPU
                scoring_fn = OPENSOURCE_SCORING_GPU_FUNCTIONS.get(
                    scoring_gpu_to_use,
                    OPENSOURCE_SCORING_GPU_FUNCTIONS[DEFAULT_OPENSOURCE_GPU]
                )
                print(f"  [{design_id}] Running open-source scoring ({scoring_gpu_to_use})...")
                try:
                    scoring_result = scoring_fn.remote(
                        design_id,
                        af3_result.get("af3_structure"),
                        af3_result.get("af3_iptm", 0),
                        af3_result.get("af3_ptm", 0),
                        af3_result.get("af3_plddt", 0),
                        "A", "B",  # binder_chain, target_chain
                        apo_structure,
                        af3_result.get("af3_confidence_json"),
                        target_type,
                        verbose,  # Pass verbose flag to remote function
                    )
                    status_str = "ACCEPTED" if scoring_result.get("accepted") else f"REJECTED ({scoring_result.get('rejection_reason', 'unknown')})"
                    print(f"  [{design_id}] Open-source: SC={scoring_result.get('interface_sc', 0):.2f}, dSASA={scoring_result.get('interface_dSASA', 0):.1f} → {status_str}")
                except Exception as e:
                    print(f"  [{design_id}] Open-source scoring failed: {e}")
                    scoring_result = {"error": str(e), "accepted": False, "rejection_reason": f"Open-source scoring error: {e}"}
            else:
                # PyRosetta scoring (CPU-based)
                print(f"  [{design_id}] Running PyRosetta scoring...")
                try:
                    scoring_result = run_pyrosetta_single.remote(
                        design_id,
                        af3_result.get("af3_structure"),
                        af3_result.get("af3_iptm", 0),
                        af3_result.get("af3_ptm", 0),
                        af3_result.get("af3_plddt", 0),
                        "A", "B",  # binder_chain, target_chain
                        apo_structure,
                        af3_result.get("af3_confidence_json"),
                        target_type,
                    )
                    status_str = "ACCEPTED" if scoring_result.get("accepted") else f"REJECTED ({scoring_result.get('rejection_reason', 'unknown')})"
                    print(f"  [{design_id}] PyRosetta: dG={scoring_result.get('interface_dG', 0):.1f}, SC={scoring_result.get('interface_sc', 0):.2f} → {status_str}")
                except Exception as e:
                    print(f"  [{design_id}] PyRosetta failed: {e}")
                    scoring_result = {"error": str(e), "accepted": False, "rejection_reason": f"PyRosetta error: {e}"}
        
        # Stream final result (after interface scoring)
        if stream and target_type == "protein":
            _stream_final_result(
                run_id=run_id,
                design_idx=design_idx,
                design_id=design_id,
                accepted=scoring_result.get("accepted", False),
                rejection_reason=scoring_result.get("rejection_reason", ""),
                metrics={
                    "interface_dG": scoring_result.get("interface_dG", 0.0),
                    "interface_sc": scoring_result.get("interface_sc", 0.0),
                    "interface_nres": scoring_result.get("interface_nres", 0),
                    "interface_dSASA": scoring_result.get("interface_dSASA", 0.0),
                    "interface_packstat": scoring_result.get("interface_packstat", 0.0),
                    "interface_dG_SASA_ratio": scoring_result.get("interface_dG_SASA_ratio", 0.0),
                    "interface_interface_hbonds": scoring_result.get("interface_interface_hbonds", 0),
                    "interface_delta_unsat_hbonds": scoring_result.get("interface_delta_unsat_hbonds", 0),
                    "interface_hydrophobicity": scoring_result.get("interface_hydrophobicity", 0.0),
                    "surface_hydrophobicity": scoring_result.get("surface_hydrophobicity", 0.0),
                    "binder_sasa": scoring_result.get("binder_sasa", 0.0),
                    "interface_fraction": scoring_result.get("interface_fraction", 0.0),
                    "interface_hbond_percentage": scoring_result.get("interface_hbond_percentage", 0.0),
                    "interface_delta_unsat_hbonds_percentage": scoring_result.get("interface_delta_unsat_hbonds_percentage", 0.0),
                    "apo_holo_rmsd": scoring_result.get("apo_holo_rmsd"),
                    "i_pae": scoring_result.get("i_pae"),
                    "rg": scoring_result.get("rg"),
                },
                relaxed_pdb=scoring_result.get("relaxed_pdb", ""),
            )
        
        # ========== COMBINE ALL RESULTS ==========

        return {
            # Design results
            **design_result,
            # Identity
            "design_id": design_id,
            "design_num": design_idx,
            "cycle": best_cycle,
            # AF3 results
            "af3_iptm": af3_result.get("af3_iptm", 0),
            "af3_ipsae": af3_result.get("af3_ipsae", 0),
            "af3_ptm": af3_result.get("af3_ptm", 0),
            "af3_plddt": af3_result.get("af3_plddt", 0),
            "af3_structure": af3_result.get("af3_structure"),
            "af3_confidence_json": af3_result.get("af3_confidence_json"),
            # APO results
            "apo_structure": apo_structure,
            # Interface scoring results (merge all keys except design_id)
            **{k: v for k, v in scoring_result.items() if k != "design_id"},
            # Binder metadata
            "binder_sequence": binder_seq,
            "binder_length": len(binder_seq) if binder_seq else 0,
            "ipsae": best_cycle_data.get("ipsae", 0.0) if best_cycle_data else 0.0,
            "plddt": best_cycle_data.get("plddt", 0.0) if best_cycle_data else 0.0,
            "iplddt": best_cycle_data.get("iplddt", 0.0) if best_cycle_data else 0.0,
        }
    
    # Execute with concurrency limit
    all_results = []
    if validation_enabled:
        validation_name = validation_model.upper()
        scoring_backend = "OpenMM/FreeSASA" if scoring_method == "opensource" else "PyRosetta"
        pipeline_mode = f"full pipeline (Boltz→{validation_name}→{scoring_backend})"
    else:
        pipeline_mode = "design only"
    
    # Determine effective concurrency
    effective_concurrency = max_concurrent if max_concurrent > 0 else len(tasks)
    concurrency_str = f"{max_concurrent} GPUs" if max_concurrent > 0 else "unlimited"
    
    if validation_enabled:
        # Full pipeline mode: use ThreadPoolExecutor for sequential per-design orchestration
        print(f"Executing {len(tasks)} tasks [{pipeline_mode}] (concurrency: {concurrency_str})...\n")
        
        with ThreadPoolExecutor(max_workers=effective_concurrency) as executor:
            futures = {executor.submit(_run_full_pipeline_for_design, t): t for t in tasks}
            
            for i, future in enumerate(as_completed(futures)):
                result = future.result()
                all_results.append(result)
                
                # Print completion status
                design_idx = result.get("design_idx", "?")
                
                if result.get("af3_iptm") is not None:
                    # Full pipeline completed with AF3 validation
                    accepted = "✓ ACCEPTED" if result.get("accepted") else "✗ REJECTED"
                    print(f"\n[{i+1}/{len(tasks)}] Design {design_idx} COMPLETE: "
                          f"Boltz={result.get('best_iptm', 0):.3f}, AF3={result.get('af3_iptm', 0):.3f}, "
                          f"dG={result.get('interface_dG', 0):.1f} → {accepted}\n")
                elif result.get("rejection_reason"):
                    # AF3 was skipped due to Boltz metrics not meeting threshold
                    print(f"[{i+1}/{len(tasks)}] Design {design_idx}: ✗ REJECTED ({result.get('rejection_reason')})")
                else:
                    # Design completed but no AF3 result (shouldn't happen normally)
                    status = "✓" if result.get("status") == "success" else "✗"
                    print(f"[{i+1}/{len(tasks)}] {status} Design {design_idx}: ipTM={result.get('best_iptm', 0):.3f}")
    else:
        # Design-only mode: use Modal's native parallelism for maximum throughput
        print(f"Executing {len(tasks)} tasks [{pipeline_mode}] (concurrency: {concurrency_str})...")
        for i, result in enumerate(gpu_fn.map(tasks)):
            all_results.append(result)
            status = "✓" if result.get("status") == "success" else "✗"
            iptm = result.get("best_iptm", 0)
            print(f"[{i+1}/{len(tasks)}] {status} Design {result.get('design_idx')}: ipTM={iptm:.3f}")
    
    # Stop sync thread
    if sync_thread:
        print("\nStopping background sync...")
        stop_sync.set()
        sync_thread.join(timeout=30)
    
    # ==========================================================================
    # UNIFIED RESULT SAVING
    # Handles design-only, AF3 validation, and Protenix validation results
    # ==========================================================================
    print(f"\nSaving results to {output_path}...")
    
    # Create all directories
    designs_dir = output_path / "designs"
    best_dir = output_path / "best_designs"
    refolded_dir = output_path / "refolded"
    accepted_dir = output_path / "accepted_designs"
    rejected_dir = output_path / "rejected"
    
    designs_dir.mkdir(parents=True, exist_ok=True)
    best_dir.mkdir(exist_ok=True)
    
    if validation_enabled:
        refolded_dir.mkdir(exist_ok=True)
        accepted_dir.mkdir(exist_ok=True)
        rejected_dir.mkdir(exist_ok=True)
    
    # Process results
    best_rows = []
    accepted = []
    rejected = []
    
    for r in all_results:
        if r.get("status") != "success":
            continue
        
        design_idx = r.get("design_idx", 0)
        best_cycle = r.get("best_cycle", 0)
        design_id = r.get("design_id", f"{name}_d{design_idx}_c{best_cycle}")
        
        # Save best design PDB (from Boltz)
        if r.get("best_pdb"):
            (best_dir / f"{design_id}.pdb").write_text(r["best_pdb"])
        
        # Save refolded structure (if available)
        if r.get("af3_structure"):
            (refolded_dir / f"{design_id}_refolded.cif").write_text(r["af3_structure"])
        
        # Save relaxed PDB to accepted/rejected (if scoring ran)
        if validation_enabled and r.get("relaxed_pdb"):
            if r.get("accepted"):
                (accepted_dir / f"{design_id}_relaxed.pdb").write_text(r["relaxed_pdb"])
                accepted.append(r)
            else:
                (rejected_dir / f"{design_id}_relaxed.pdb").write_text(r["relaxed_pdb"])
                rejected.append(r)
        elif validation_enabled and r.get("af3_structure"):
            # No relaxed PDB but have validated structure - still classify
            if r.get("accepted"):
                accepted.append(r)
            else:
                rejected.append(r)
        
        # Get cycle data for metrics
        best_cycle_data = None
        for cycle_data in r.get("cycles", []):
            if cycle_data.get("cycle") == best_cycle:
                best_cycle_data = cycle_data
                break
        
        seq = r.get("best_seq", "")
        binder_length = len(seq) if seq else 0
        alanine_count = best_cycle_data.get("alanine_count", 0) if best_cycle_data else 0
        alanine_pct = (alanine_count / binder_length * 100) if binder_length > 0 else 0.0
        
        # Build template_first_residue string for CSV
        tfr_str = ""
        if template_first_residues:
            tfr_str = ",".join(f"{k}:{v}" for k, v in template_first_residues.items())
        
        best_rows.append({
            "design_id": design_id, "design_num": design_idx, "cycle": best_cycle,
            "binder_sequence": seq, "binder_length": binder_length, "cyclic": cyclic,
            "alanine_count": alanine_count, "alanine_pct": round(alanine_pct, 2),
            "boltz_iptm": r.get("best_iptm", 0.0),
            "boltz_ipsae": r.get("ipsae", best_cycle_data.get("ipsae", 0.0) if best_cycle_data else 0.0),
            "boltz_plddt": r.get("plddt", best_cycle_data.get("plddt", 0.0) if best_cycle_data else 0.0),
            "boltz_iplddt": r.get("iplddt", best_cycle_data.get("iplddt", 0.0) if best_cycle_data else 0.0),
            "af3_iptm": r.get("af3_iptm"), "af3_ipsae": r.get("af3_ipsae"),
            "af3_ptm": r.get("af3_ptm"), "af3_plddt": r.get("af3_plddt"),
            "interface_dG": r.get("interface_dG"), "interface_sc": r.get("interface_sc"),
            "interface_nres": r.get("interface_nres"), "interface_dSASA": r.get("interface_dSASA"),
            "interface_packstat": r.get("interface_packstat"),
            "interface_hbonds": r.get("interface_interface_hbonds"),
            "interface_delta_unsat_hbonds": r.get("interface_delta_unsat_hbonds"),
            "apo_holo_rmsd": r.get("apo_holo_rmsd"), "i_pae": r.get("i_pae"), "rg": r.get("rg"),
            "accepted": r.get("accepted"), "rejection_reason": r.get("rejection_reason"),
            "contact_residues": contact_residues_canonical or "",
            "contact_residues_auth": contact_residues_auth or "",
            "template_first_residue": tfr_str,
        })
    
    # Helper function to reorder columns using unified schema
    def reorder_columns(df: pd.DataFrame) -> pd.DataFrame:
        """Reorder DataFrame columns to match UNIFIED_DESIGN_COLUMNS."""
        ordered_cols = [c for c in UNIFIED_DESIGN_COLUMNS if c in df.columns]
        extra_cols = [c for c in df.columns if c not in UNIFIED_DESIGN_COLUMNS]
        return df[ordered_cols + extra_cols]
    
    # Save best_designs CSV with consistent column ordering
    if best_rows:
        best_df = pd.DataFrame(best_rows)
        best_df = reorder_columns(best_df)
        # Remove None columns for cleaner CSV
        best_df = best_df.dropna(axis=1, how='all')
        best_df.to_csv(best_dir / "best_designs.csv", index=False)
        print(f"  ✓ best_designs/ ({len(best_rows)} PDBs + best_designs.csv)")
    
    design_pdbs = list(designs_dir.glob("*.pdb"))
    print(f"  ✓ designs/ ({len(design_pdbs)} PDBs)")
    
    if validation_enabled:
        validation_rows = [{"design_id": r.get("design_id"), "af3_iptm": r.get("af3_iptm"),
                           "af3_ipsae": r.get("af3_ipsae"), "af3_ptm": r.get("af3_ptm"),
                           "af3_plddt": r.get("af3_plddt")}
                          for r in all_results if r.get("af3_iptm") is not None]
        if validation_rows:
            pd.DataFrame(validation_rows).to_csv(refolded_dir / "validation_results.csv", index=False)
            print(f"  ✓ refolded/ ({len(validation_rows)} structures)")
        
        accepted_rows = [row for row in best_rows if row.get("accepted") is True]
        rejected_rows = [row for row in best_rows if row.get("accepted") is False]
        
        if accepted_rows:
            accepted_df = pd.DataFrame(accepted_rows)
            accepted_df = reorder_columns(accepted_df)
            accepted_df = accepted_df.dropna(axis=1, how='all')
            accepted_df.to_csv(accepted_dir / "accepted_stats.csv", index=False)
            print(f"  ✓ accepted_designs/ ({len(accepted_rows)} designs)")
        
        if rejected_rows:
            rejected_df = pd.DataFrame(rejected_rows)
            rejected_df = reorder_columns(rejected_df)
            rejected_df = rejected_df.dropna(axis=1, how='all')
            rejected_df.to_csv(rejected_dir / "rejected_stats.csv", index=False)
            print(f"  ✓ rejected/ ({len(rejected_rows)} designs)")
    
    # ==========================================================================
    # PRINT SUMMARY
    # ==========================================================================
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    
    successful = [r for r in all_results if r.get("status") == "success"]
    print(f"Successful designs: {len(successful)}/{len(all_results)}")
    
    if successful:
        best_overall = max(successful, key=lambda r: r.get("best_iptm", 0))
        print(f"Best Boltz ipTM: {name}_d{best_overall['design_idx']} = {best_overall['best_iptm']:.3f}")
    
    if validation_enabled:
        validation_results = [r for r in successful if r.get("af3_iptm") is not None]
        if validation_results:
            best_val = max(validation_results, key=lambda r: r.get("af3_iptm", 0))
            val_label = validation_model.upper()
            print(f"Best {val_label} ipTM: {best_val.get('design_id')} = {best_val.get('af3_iptm', 0):.3f}")

        scoring_label = "Open-source Scoring" if scoring_method == "opensource" else "PyRosetta Filtering"
        print(f"\n{scoring_label}:")
        print(f"  Accepted: {len(accepted)}")
        print(f"  Rejected: {len(rejected)}")

        if accepted:
            print("\n  Accepted designs:")
            for r in accepted:
                print(f"    {r.get('design_id')}: dG={r.get('interface_dG', 0):.1f}, "
                      f"SC={r.get('interface_sc', 0):.2f}, "
                      f"RMSD={r.get('apo_holo_rmsd', 'N/A')}")
    
    print(f"\nOutput: {output_path}/")
    if validation_enabled:
        filter_label = "open-source" if scoring_method == "opensource" else "PyRosetta"
        val_label = validation_model.upper()
        print("  ├── designs/           # All cycles from Boltz")
        print("  ├── best_designs/      # Best cycle per design (Boltz PDBs)")
        print(f"  ├── refolded/          # {val_label} refolded structures")
        print(f"  ├── accepted_designs/  # Passed {filter_label} filters (relaxed PDBs)")
        print(f"  └── rejected/          # Failed {filter_label} filters")
    else:
        print("  ├── designs/           # All cycles")
        print("  └── best_designs/      # Best cycle per design")


@app.local_entrypoint()
def list_gpus():
    """List available GPU types."""
    print("\nAvailable GPU types:")
    print("-" * 40)
    for gpu, desc in GPU_TYPES.items():
        default = " (DEFAULT)" if gpu == DEFAULT_GPU else ""
        # Remove "(RECOMMENDED)" from description, keep just the specs
        desc_clean = desc.replace(" (RECOMMENDED)", "")
        print(f"  {gpu}: {desc_clean}{default}")
    print("\nUsage: --gpu H100")


@app.local_entrypoint()
def test_connection(gpu: str = DEFAULT_GPU):
    """Test Modal connection and GPU."""
    print(f"Testing Modal connection with GPU: {gpu}...")
    result = _test_gpu.remote()
    print(f"\n{result}")


@app.local_entrypoint()
def test_af3():
    """Test that the AF3 image is correctly configured."""
    print("Testing AF3 image configuration...")
    print("=" * 60)
    result = test_af3_image.remote()
    print(result)
    print("=" * 60)


if __name__ == "__main__":
    print("Use 'modal run modal_boltz_ph_cli.py::<entrypoint>' to execute")
    print("\nAvailable entrypoints:")
    print("  - init_cache          Initialize model weights cache")
    print("  - upload_af3_weights  Upload AlphaFold3 weights")
    print("  - run_pipeline        Run the design pipeline")
    print("  - list_gpus           List available GPU types")
    print("  - test_connection     Test Modal connection")
    print("  - test_af3            Test AF3 image configuration")

