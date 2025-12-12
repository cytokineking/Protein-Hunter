import argparse
import os
import sys
import warnings

# ============================================================================
# LOG SUPPRESSION - Must be set before any JAX/TensorFlow imports
# ============================================================================
# Suppress JAX backend probing spam (rocm, tpu warnings)
os.environ["JAX_PLATFORMS"] = "cuda,cpu"

# Suppress TensorFlow/JAX INFO logs (AF3 bucket size messages)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"  # 0=all, 1=no INFO, 2=no WARNING

# Suppress absl INFO logging (used by AF3)
import logging
logging.getLogger("absl").setLevel(logging.WARNING)

# Suppress PyTorch warning about numpy array conversion
warnings.filterwarnings("ignore", message=".*Creating a tensor from a list of numpy.ndarrays.*")

from pipeline import ProteinHunter_Boltz, MultiGPUOrchestrator

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

# Keep parse_args() here for CLI functionality
def parse_args():
    parser = argparse.ArgumentParser(
        description="Boltz protein design with cycle optimization (local Protein Hunter)"
    )
    # --- Existing Arguments (omitted for brevity, keep all original args) ---
    parser.add_argument("--gpu-id", "--gpu_id", dest="gpu_id", default=0, type=int,
        help="GPU device ID to use (default: 0)")
    parser.add_argument("--num-gpus", "--num_gpus", dest="num_gpus", default=1, type=int,
        help="Number of GPUs for parallel execution. When > 1, spawns multiple workers with automatic work distribution.")
    parser.add_argument("--grad-enabled", "--grad_enabled", dest="grad_enabled", action="store_true", default=False)
    parser.add_argument("--name", default="target_name_is_missing", type=str)
    parser.add_argument(
        "--mode", default="binder", choices=["binder", "unconditional"], type=str
    )
    parser.add_argument(
        "--num-designs", "--num_designs", dest="num_designs",
        default=None, 
        type=int,
        help="Number of designs to generate. At least one of --num-designs or --num-accepted required."
    )
    parser.add_argument(
        "--num-accepted", "--num_accepted", dest="num_accepted",
        default=None,
        type=int,
        help="Stop after this many designs pass all filters. Requires validation (--validation-model != none)."
    )
    parser.add_argument("--num-cycles", "--num_cycles", dest="num_cycles", default=5, type=int)
    parser.add_argument("--cyclic", action="store_true", default=False, help="Enable cyclic peptide design.")
    parser.add_argument("--min-protein-length", "--min_protein_length", dest="min_protein_length", default=100, type=int)
    parser.add_argument("--max-protein-length", "--max_protein_length", dest="max_protein_length", default=150, type=int)
    parser.add_argument("--seq", default="", type=str)
    parser.add_argument("--refiner-mode", "--refiner_mode", dest="refiner_mode", action="store_true", default=False)
    parser.add_argument(
        "--protein-seqs", "--protein_seqs", dest="protein_seqs",
        default="",
        type=str,
    )
    parser.add_argument("--msa-mode", "--msa_mode", dest="msa_mode", default="mmseqs", choices=["single", "mmseqs"], type=str)

    parser.add_argument("--ligand-smiles", "--ligand_smiles", dest="ligand_smiles", default="", type=str)
    parser.add_argument("--ligand-ccd", "--ligand_ccd", dest="ligand_ccd", default="", type=str)
    parser.add_argument(
        "--nucleic-type", "--nucleic_type", dest="nucleic_type", default="dna", choices=["dna", "rna"], type=str
    )
    parser.add_argument("--nucleic-seq", "--nucleic_seq", dest="nucleic_seq", default="", type=str)
    parser.add_argument(
        "--template-path", "--template_path", dest="template_path", default="", type=str
    )  #pdb code or path(s) to .cif/.pdb, multiple allowed separated by colon or comma
    parser.add_argument(
        "--template-cif-chain-id", "--template_cif_chain_id", dest="template_cif_chain_id", default="", type=str
    )  # for mmCIF files, the chain id to use for the template (for alignment)
    parser.add_argument("--diffuse-steps", "--diffuse_steps", dest="diffuse_steps", default=200, type=int)
    parser.add_argument("--recycling-steps", "--recycling_steps", dest="recycling_steps", default=3, type=int)
    parser.add_argument("--boltz-model-version", "--boltz_model_version", dest="boltz_model_version", default="boltz2", type=str)
    parser.add_argument(
        "--boltz-model-path", "--boltz_model_path", dest="boltz_model_path",
        default="~/.boltz/boltz2_conf.ckpt",
        type=str,
    )
    parser.add_argument("--ccd-path", "--ccd_path", dest="ccd_path", default="~/.boltz/mols", type=str)
    parser.add_argument("--randomly-kill-helix-feature", "--randomly_kill_helix_feature", dest="randomly_kill_helix_feature", action="store_true", default=False)
    parser.add_argument("--negative-helix-constant", "--negative_helix_constant", dest="negative_helix_constant", default=0.2, type=float)
    parser.add_argument("--logmd", action="store_true", default=False)
    parser.add_argument("--save-dir", "--save_dir", dest="save_dir", default="", type=str)
    parser.add_argument("--omit-aa", "--omit_AA", "--omit_aa", dest="omit_aa", default="C", type=str)
    parser.add_argument("--exclude-p", "--exclude_P", "--exclude_p", dest="exclude_p", action="store_true", default=False)
    parser.add_argument("--percent-x", "--percent_X", "--percent_x", dest="percent_x", default=90, type=int)
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Plot cycles figs per run (requires matplotlib)",
    )

    parser.add_argument(
        "--contact-residues", "--contact_residues", dest="contact_residues",
        default="", 
        type=str,
        help="Specify which target chain residues must contact the binder (currently only supports protein contacts). For more than two chains, separate by |, e.g., '1,2,5,10 | 3,5,10'."
    )
    parser.add_argument(
        "--use-auth-numbering", "--use_auth_numbering", dest="use_auth_numbering",
        action="store_true",
        default=False,
        help="Interpret --contact_residues in auth/PDB numbering instead of canonical 1-indexed positions. Residues will be automatically converted to canonical positions for Boltz."
    )

    parser.add_argument(
        "--no-contact-filter", "--no_contact_filter", dest="no_contact_filter",
        action="store_true",
        help="Do not filter or restart for unbound contact residues at cycle 0",
    )
    parser.add_argument("--max-contact-filter-retries", "--max_contact_filter_retries", dest="max_contact_filter_retries", default=6, type=int)
    parser.add_argument("--contact-cutoff", "--contact_cutoff", dest="contact_cutoff", default=15.0, type=float)

    parser.add_argument(
        "--alphafold-dir", "--alphafold_dir", dest="alphafold_dir", default=os.path.expanduser("~/alphafold3"), type=str
    )
    parser.add_argument("--af3-docker-name", "--af3_docker_name", dest="af3_docker_name", default="alphafold3", type=str,
        help="Docker image name for AlphaFold3 (default: alphafold3)")
    parser.add_argument(
        "--af3-database-settings", "--af3_database_settings", dest="af3_database_settings", default="~/alphafold3/alphafold3_data_save", type=str
    )
    parser.add_argument(
        "--hmmer-path", "--hmmer_path", dest="hmmer_path",
        default="~/.conda/envs/alphafold3_venv",
        type=str,
    )
    # Modal-style validation/scoring selectors
    parser.add_argument(
        "--validation-model",
        dest="validation_model",
        default="none",
        choices=["none", "af3", "protenix"],
        help="Validation model to use after Boltz design: none, af3 (AlphaFold3), or protenix (open-source AF3)."
    )
    parser.add_argument(
        "--scoring-method",
        dest="scoring_method",
        default="pyrosetta",
        choices=["pyrosetta", "opensource"],
        help="Interface scoring method for protein targets: pyrosetta (default) or opensource (OpenMM + FreeSASA)."
    )
    parser.add_argument(
        "--use-msa-for-validation",
        dest="use_msa_for_validation",
        default=True,
        type=str2bool,
        help="Reuse MSAs from design phase for validation (default: True)."
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=False,
        help="Enable verbose logging for validation/scoring subprocesses."
    )

    # Deprecated flags (kept for backward compatibility)
    parser.add_argument("--use-alphafold3-validation", "--use_alphafold3_validation",
        dest="use_alphafold3_validation", action="store_true", default=False,
        help=argparse.SUPPRESS)
    parser.add_argument("--use-msa-for-af3", "--use_msa_for_af3",
        dest="use_msa_for_af3", default=True, type=str2bool,
        help=argparse.SUPPRESS)
    parser.add_argument("--use-open-scoring", "--use_open_scoring",
        dest="use_open_scoring", default=False, type=str2bool,
        help=argparse.SUPPRESS)

    parser.add_argument("--work-dir", "--work_dir", dest="work_dir", default="", type=str)

    # temp and bias params
    parser.add_argument("--temperature", default=0.1, type=float)
    parser.add_argument("--alanine-bias-start", "--alanine_bias_start", dest="alanine_bias_start", default=-0.5, type=float)
    parser.add_argument("--alanine-bias-end", "--alanine_bias_end", dest="alanine_bias_end", default=-0.1, type=float)
    parser.add_argument("--alanine-bias", "--alanine_bias", dest="alanine_bias", default=True, type=str2bool,
        help="Apply alanine bias penalty during MPNN design (default: True)")
    parser.add_argument("--high-iptm-threshold", "--high_iptm_threshold", dest="high_iptm_threshold", default=0.8, type=float)
    parser.add_argument("--high-plddt-threshold", "--high_plddt_threshold", dest="high_plddt_threshold", default=0.8, type=float)
    # --- End Existing Arguments ---

    return parser.parse_args()

def print_args(args):
    # Deprecated flags to exclude from config print
    deprecated_flags = {"use_alphafold3_validation", "use_msa_for_af3", "use_open_scoring"}
    
    print("="*40)
    print("Design Configuration:")
    for k, v in vars(args).items():
        if k not in deprecated_flags:
            print(f"{k:30}: {v}")
    print("="*40)

def validate_args(args):
    """Validate command-line arguments for stopping conditions."""
    import sys
    
    # Validate stopping conditions
    if args.num_designs is None and args.num_accepted is None:
        print("ERROR: Must specify at least one: --num-designs or --num-accepted")
        sys.exit(1)
    
    if args.num_accepted is not None and getattr(args, "validation_model", "none") == "none":
        print("ERROR: --num-accepted requires validation. Set --validation-model af3 or protenix.")
        sys.exit(1)
    
    if args.num_accepted is not None and args.num_designs is None:
        print("⚠ WARNING: --num-accepted without --num-designs has no upper limit.")
        print("  Consider adding --num-designs as a safety limit.\n")


def validate_dependencies(args):
    """
    Validate that required dependencies are installed for the selected options.
    
    Checks:
    - PyRosetta availability if --scoring-method pyrosetta
    - OpenMM/pdbfixer availability if --scoring-method opensource
    - Protenix repo presence if --validation-model protenix
    - Docker + AF3 image if --validation-model af3
    """
    from pathlib import Path
    import subprocess
    
    validation_model = getattr(args, "validation_model", "none")
    scoring_method = getattr(args, "scoring_method", "pyrosetta")
    
    errors = []
    
    # -------------------------------------------------------------------------
    # Check PyRosetta availability (if selected)
    # -------------------------------------------------------------------------
    if scoring_method == "pyrosetta":
        try:
            import pyrosetta
        except ImportError:
            errors.append(
                "ERROR: --scoring-method pyrosetta requires PyRosetta, which is not installed.\n"
                "       Options:\n"
                "         1. Install PyRosetta: re-run setup.sh without --no-pyrosetta\n"
                "         2. Use open-source scoring: --scoring-method opensource"
            )
    
    # -------------------------------------------------------------------------
    # Check OpenMM/pdbfixer availability (if selected)
    # -------------------------------------------------------------------------
    if scoring_method == "opensource":
        missing_deps = []
        try:
            import openmm
        except ImportError:
            missing_deps.append("openmm")
        
        try:
            from pdbfixer import PDBFixer
        except ImportError:
            missing_deps.append("pdbfixer")
        
        if missing_deps:
            errors.append(
                f"ERROR: --scoring-method opensource requires OpenMM and pdbfixer.\n"
                f"       Missing: {', '.join(missing_deps)}\n"
                f"       Install with: conda install -c conda-forge openmm pdbfixer"
            )
    
    # -------------------------------------------------------------------------
    # Check Protenix availability (if selected)
    # -------------------------------------------------------------------------
    if validation_model == "protenix":
        # Check for Protenix repo in expected location
        protenix_path = Path(__file__).resolve().parents[1] / "Protenix"
        
        if not protenix_path.exists():
            errors.append(
                f"ERROR: --validation-model protenix requires the Protenix repository.\n"
                f"       Expected location: {protenix_path}\n"
                f"       Options:\n"
                f"         1. Re-run setup.sh without --no-protenix\n"
                f"         2. Manually clone: git clone https://github.com/bytedance/Protenix.git {protenix_path}"
            )
        else:
            # Verify Protenix is importable (basic check)
            try:
                # Just check the directory has expected structure
                runner_dir = protenix_path / "runner"
                if not runner_dir.exists():
                    errors.append(
                        f"ERROR: Protenix repository appears incomplete (missing runner/ directory).\n"
                        f"       Location: {protenix_path}\n"
                        f"       Try: cd {protenix_path} && git pull"
                    )
            except Exception as e:
                errors.append(
                    f"ERROR: Could not verify Protenix installation: {e}\n"
                    f"       Location: {protenix_path}"
                )
    
    # -------------------------------------------------------------------------
    # Check AF3 Docker availability (if selected)
    # -------------------------------------------------------------------------
    if validation_model == "af3":
        docker_name = getattr(args, "af3_docker_name", "alphafold3")
        
        # First check if Docker is available
        try:
            docker_check = subprocess.run(
                ["docker", "--version"],
                capture_output=True,
                text=True,
                timeout=10
            )
            if docker_check.returncode != 0:
                errors.append(
                    "ERROR: --validation-model af3 requires Docker, which is not working.\n"
                    "       Docker error: " + (docker_check.stderr or "unknown")
                )
            else:
                # Docker is available, check for AF3 image
                image_check = subprocess.run(
                    ["docker", "images", "-q", docker_name],
                    capture_output=True,
                    text=True,
                    timeout=10
                )
                if not image_check.stdout.strip():
                    errors.append(
                        f"ERROR: --validation-model af3 requires AlphaFold3 Docker image '{docker_name}'.\n"
                        f"       The image was not found on this system.\n"
                        f"       Setup instructions: See LOCAL_SETUP_GUIDE.md section 'Full Setup (With AF3 Validation)'\n"
                        f"       Quick steps:\n"
                        f"         1. git clone https://github.com/google-deepmind/alphafold3.git ~/alphafold3\n"
                        f"         2. cd ~/alphafold3 && docker build -t {docker_name} .\n"
                        f"         3. Download AF3 weights from Google DeepMind"
                    )
        except FileNotFoundError:
            errors.append(
                "ERROR: --validation-model af3 requires Docker, which is not installed.\n"
                "       Install Docker: https://docs.docker.com/get-docker/\n"
                "       Then install NVIDIA Container Toolkit for GPU support."
            )
        except subprocess.TimeoutExpired:
            # Don't fail on timeout, just warn
            print("⚠ WARNING: Docker availability check timed out. Proceeding anyway...")
        except Exception as e:
            # Don't fail on unexpected errors, just warn
            print(f"⚠ WARNING: Could not verify Docker/AF3 setup: {e}")
    
    # -------------------------------------------------------------------------
    # Report all errors and exit if any
    # -------------------------------------------------------------------------
    if errors:
        print("\n" + "="*70)
        print("DEPENDENCY CHECK FAILED")
        print("="*70 + "\n")
        for error in errors:
            print(error)
            print()
        print("="*70)
        print("Please install missing dependencies or adjust your command-line options.")
        print("="*70 + "\n")
        sys.exit(1)


def main():
    args = parse_args()

    # ------------------------------------------------------------------
    # Deprecation handling & flag migration
    # ------------------------------------------------------------------
    # Warn on underscore-style flags
    if any(arg.startswith("--") and "_" in arg for arg in sys.argv):
        warnings.warn(
            "Underscore CLI flags are deprecated. Please use kebab-case (e.g., --num-designs).",
            DeprecationWarning,
            stacklevel=2,
        )

    # Map deprecated AF3 validation flag to validation_model
    if getattr(args, "use_alphafold3_validation", False) and args.validation_model == "none":
        warnings.warn(
            "--use-alphafold3-validation is deprecated. Use --validation-model af3 instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        args.validation_model = "af3"

    # Map deprecated open scoring flag to scoring_method
    if getattr(args, "use_open_scoring", False) and args.scoring_method == "pyrosetta":
        warnings.warn(
            "--use-open-scoring is deprecated. Use --scoring-method opensource instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        args.scoring_method = "opensource"

    # Map deprecated MSA flag to use_msa_for_validation
    if any(a in sys.argv for a in ("--use-msa-for-af3", "--use_msa_for_af3")):
        warnings.warn(
            "--use-msa-for-af3 is deprecated. Use --use-msa-for-validation instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        args.use_msa_for_validation = getattr(args, "use_msa_for_af3", True)
    
    # Validate stopping condition arguments
    validate_args(args)
    
    # Validate that required dependencies are installed
    validate_dependencies(args)
    
    # Validate num_gpus
    if args.num_gpus < 1:
        print("ERROR: --num_gpus must be at least 1")
        sys.exit(1)
    
    # Smart MSA mode: skip MSAs when template provided without AF3 validation
    # - Template provided + no AF3 validation: skip MSAs (Boltz uses template, no AF3 needs)
    # - Template provided + AF3 validation: compute MSAs (AF3 needs them)
    # - No template: compute MSAs (Boltz needs them)
    validation_enabled = getattr(args, "validation_model", "none") != "none"
    if args.template_path and args.msa_mode == "mmseqs" and not validation_enabled:
        print("Note: Template provided without AF3 validation - switching to --msa_mode single")
        print("      (Boltz uses template structure; MSAs not needed)\n")
        args.msa_mode = "single"
    
    # Pretty print each argument in a row for better visualization
    print_args(args)
    
    # Choose execution mode based on num_gpus
    if args.num_gpus > 1:
        # Multi-GPU mode: use orchestrator for parallel execution
        orchestrator = MultiGPUOrchestrator(args)
        orchestrator.run()
    else:
        # Single-GPU mode: use standard pipeline
        protein_hunter = ProteinHunter_Boltz(args)
        protein_hunter.run_pipeline()

if __name__ == "__main__":
    main()
