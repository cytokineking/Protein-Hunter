"""
Modal image definitions for Boltz, AF3, and PyRosetta.

This module contains all Docker image definitions used by the Modal functions.
"""

import modal

# =============================================================================
# BOLTZ DESIGN IMAGE
# =============================================================================

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "wget", "build-essential")
    .pip_install(
        # Core dependencies (DON'T install boltz from PyPI - we use local fork)
        "torch>=2.2",
        "numpy>=1.24,<2.0",
        "pandas>=2.0",
        "pyyaml>=6.0",
        "biopython>=1.83",
        "gemmi>=0.6.3",
        "prody>=2.4",
        "matplotlib>=3.7",
        "rdkit>=2024.3.1",
        # ML dependencies
        "ml-collections>=0.1.1",
        "dm-tree>=0.1.8",
        "einops>=0.7",
        "scipy>=1.12",
        # Visualization (needed by model_utils.py imports)
        "py3Dmol",
        # Additional deps for custom boltz fork
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
    # Add protein-hunter code to image with copy=True so we can run commands after
    .add_local_dir("boltz_ph", "/root/protein_hunter/boltz_ph", copy=True)
    .add_local_dir("LigandMPNN", "/root/protein_hunter/LigandMPNN", copy=True)
    .add_local_dir("utils", "/root/protein_hunter/utils", copy=True)
    # Install the local boltz fork from boltz_ph
    .run_commands(
        "cd /root/protein_hunter/boltz_ph && pip install -e .",
        # Handle optional cuequivariance
        "pip install cuequivariance-torch || pip install cuequivariance_torch || echo 'cuequivariance not available'",
    )
)

# =============================================================================
# ALPHAFOLD3 IMAGE (for validation)
# =============================================================================
# Builds AF3 from source at image build time (no local AF3 repo needed)
# Replicates the official Dockerfile: https://github.com/google-deepmind/alphafold3

af3_image = (
    modal.Image.debian_slim(python_version="3.11")
    # Install system dependencies (same pattern as main Boltz image)
    .apt_install(
        "git", "wget", "gcc", "g++", "make", "zlib1g-dev", "zstd",
    )
    # Install HMMER from source with seq_limit patch
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
    # Install AF3 Python dependencies (versions from AF3 pyproject.toml)
    .pip_install(
        # JAX with CUDA - MUST match AF3 requirements exactly
        "jax==0.4.34",
        "jax[cuda12]==0.4.34",
        "jax-triton==0.2.0",
        "triton==3.1.0",
        # AF3 core dependencies from pyproject.toml
        "absl-py",
        "dm-haiku==0.0.13",
        "dm-tree",
        "jaxtyping==0.2.34",
        "typeguard==2.13.3",
        "numpy",
        "rdkit==2024.3.5",
        "tqdm",
        "zstandard",
        # Additional deps for full functionality
        "ml-collections",
        "pandas",
        "scipy",
        "biopython>=1.83",
        "gemmi>=0.6.3",
    )
    # Install AF3 package
    .run_commands(
        "pip install --no-deps /app/alphafold",
        # Build chemical components database
        "build_data || echo 'build_data may require GPU, skipping for now'",
    )
    # Set environment variables (from official Dockerfile)
    .env({
        "PATH": "/hmmer/bin:$PATH",
        "XLA_FLAGS": "--xla_gpu_enable_triton_gemm=false",
        "XLA_PYTHON_CLIENT_PREALLOCATE": "true",
        "XLA_CLIENT_MEM_FRACTION": "0.95",
    })
    # Add protein-hunter utils
    .add_local_dir("utils", "/root/protein_hunter/utils", copy=True)
)

# =============================================================================
# PYROSETTA IMAGE (for interface scoring)
# =============================================================================
# PyRosetta installed from conda-forge channel (same approach as BindCraft)
# Users are responsible for their own license compliance

pyrosetta_image = (
    modal.Image.from_registry("continuumio/miniconda3:latest")
    .apt_install("libgfortran5")  # Required by DAlphaBall for BUNS calculation
    .run_commands(
        # Install PyRosetta from rosettacommons conda channel (like BindCraft does)
        "conda install -y -c https://conda.rosettacommons.org pyrosetta",
        # Install other dependencies (matching local pipeline)
        # - biopython: CIF->PDB conversion (MMCIFParser + PDBIO)
        # - scipy: KD-tree for hotspot_residues
        "pip install pandas numpy biopython scipy",
    )
    .add_local_dir("utils", "/root/protein_hunter/utils", copy=True)
    .run_commands(
        # Make DAlphaBall executable (needed for BUNS calculation)
        "chmod +x /root/protein_hunter/utils/DAlphaBall.gcc",
    )
)

# =============================================================================
# OPEN-SOURCE SCORING IMAGE (GPU-enabled, PyRosetta-free)
# =============================================================================
# Uses OpenMM for GPU-accelerated relaxation, FreeSASA for SASA calculations,
# sc-rs for shape complementarity, and Biopython for interface residue detection.
# Modeled after FreeBindCraft's PyRosetta bypass approach.

opensource_scoring_image = (
    modal.Image.debian_slim(python_version="3.11")
    # Install system dependencies for OpenMM and compilation
    .apt_install(
        "git",
        "wget",
        "build-essential",
        "cmake",
        "libopenblas-dev",
        "libfftw3-dev",
        # OpenCL support (fallback if CUDA fails)
        "ocl-icd-opencl-dev",
        "opencl-headers",
    )
    # Install Miniforge (conda-forge only, no Anaconda TOS required)
    .run_commands(
        # Download and install Miniforge (uses conda-forge by default)
        "wget -q https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh -O /tmp/miniforge.sh",
        "bash /tmp/miniforge.sh -b -p /opt/conda",
        "rm /tmp/miniforge.sh",
        # Add conda to PATH
        "echo 'export PATH=/opt/conda/bin:$PATH' >> /root/.bashrc",
    )
    # Install OpenMM with CUDA support via conda-forge
    .run_commands(
        "/opt/conda/bin/conda install -y openmm cudatoolkit pdbfixer",
        # Verify OpenMM installation
        "/opt/conda/bin/python -c 'import openmm; print(f\"OpenMM {openmm.__version__} installed\")'",
    )
    # Install Python dependencies via pip (using conda's python)
    .run_commands(
        "/opt/conda/bin/pip install --no-cache-dir "
        "'numpy>=1.24,<2.0' "
        "'scipy>=1.11' "
        "'pandas>=2.0' "
        "'biopython>=1.83' "
        "'gemmi>=0.6.3' "
        "freesasa "
        "pyyaml "
        "tqdm "
        # Modal runtime dependencies (required when conda python is used)
        "typing_extensions "
        "protobuf "
        "grpclib "
        "synchronicity",
    )
    # Add protein-hunter utils (includes pre-compiled FASPR, sc-rs binaries)
    .add_local_dir("utils", "/root/protein_hunter/utils", copy=True)
    # Make binaries executable and verify
    .run_commands(
        # Make FASPR and sc-rs executable
        "chmod +x /root/protein_hunter/utils/opensource_scoring/FASPR",
        "chmod +x /root/protein_hunter/utils/opensource_scoring/sc",
        # Verify binaries exist
        "ls -la /root/protein_hunter/utils/opensource_scoring/",
        # Test that FASPR can at least show help (will fail without input, but proves binary works)
        "/root/protein_hunter/utils/opensource_scoring/FASPR -h || echo 'FASPR binary accessible'",
    )
    # Set environment variables
    .env({
        "PATH": "/opt/conda/bin:$PATH",
        # Binary locations
        "FASPR_BIN": "/root/protein_hunter/utils/opensource_scoring/FASPR",
        "SC_RS_BIN": "/root/protein_hunter/utils/opensource_scoring/sc",
        "FREESASA_CONFIG": "/root/protein_hunter/utils/opensource_scoring/freesasa_naccess.cfg",
        # OpenMM platform preference: try CUDA first, then OpenCL, then CPU
        "OPENMM_DEFAULT_PLATFORM": "CUDA",
    })
)

