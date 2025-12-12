#!/bin/bash
################## Protein Hunter installation script
################## Installs all components by default; use flags to opt-out of specific features
set -e

# =============================================================================
# Default configuration
# =============================================================================
pkg_manager='conda'
cuda=''
install_pyrosetta=true
install_protenix=true
install_chai=false
fix_channels=false
upgrade_mode=false
fresh_install=false

# =============================================================================
# Parse command-line options
# =============================================================================
OPTIONS=p:c:
LONGOPTIONS=pkg-manager:,cuda:,no-pyrosetta,no-protenix,install-chai,fix-channels,upgrade,fresh,help

print_usage() {
    echo "Usage: ./setup.sh [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  -p, --pkg-manager <conda|mamba>  Package manager to use (default: conda)"
    echo "  -c, --cuda <version>             CUDA version override (e.g., 12.1)"
    echo "      --no-pyrosetta               Skip PyRosetta installation (use OpenMM/FreeSASA instead)"
    echo "      --no-protenix                Skip Protenix installation"
    echo "      --install-chai               Install Chai-lab dependencies (opt-in)"
    echo "      --fix-channels               Apply conda channel configuration fixes"
    echo "      --upgrade                    Update existing environment (add new packages)"
    echo "      --fresh                      Remove existing environment and start fresh"
    echo "      --help                       Show this help message"
    echo ""
    echo "Examples:"
    echo "  ./setup.sh                           # Full installation (recommended)"
    echo "  ./setup.sh --no-pyrosetta            # Skip PyRosetta (no license required)"
    echo "  ./setup.sh --pkg-manager mamba       # Use mamba for faster installs"
    echo "  ./setup.sh --no-pyrosetta --no-protenix  # Minimal install (Boltz only)"
    echo "  ./setup.sh --upgrade                        # Update existing environment"
    echo "  ./setup.sh --fresh                          # Remove existing environment and start fresh"
}

PARSED=$(getopt --options=$OPTIONS --longoptions=$LONGOPTIONS --name "$0" -- "$@") || {
    print_usage
    exit 1
}
eval set -- "$PARSED"

while true; do
    case "$1" in
        -p|--pkg-manager)
            pkg_manager="$2"
            shift 2
            ;;
        -c|--cuda)
            cuda="$2"
            shift 2
            ;;
        --no-pyrosetta)
            install_pyrosetta=false
            shift
            ;;
        --no-protenix)
            install_protenix=false
            shift
            ;;
        --install-chai)
            install_chai=true
            shift
            ;;
        --fix-channels)
            fix_channels=true
            shift
            ;;
        --upgrade)
            upgrade_mode=true
            shift
            ;;
        --fresh)
            fresh_install=true
            shift
            ;;
        --help)
            print_usage
            exit 0
            ;;
        --)
            shift
            break
            ;;
        *)
            echo "Invalid option: $1" >&2
            print_usage
            exit 1
            ;;
    esac
done

# =============================================================================
# Display configuration
# =============================================================================
echo "🚀 Setting up Protein Hunter Environment..."
echo ""
echo "Configuration:"
echo "  Package manager:    $pkg_manager"
echo "  CUDA override:      ${cuda:-auto-detect}"
echo "  Install PyRosetta:  $install_pyrosetta"
echo "  Install Protenix:   $install_protenix"
echo "  Install Chai-lab:   $install_chai"
echo "  Fix channels:       $fix_channels"
echo "  Upgrade mode:       $upgrade_mode"
echo "  Fresh install:      $fresh_install"
echo ""

# =============================================================================
# Initialization
# =============================================================================
SECONDS=0
install_dir=$(pwd)

# Check for conda installation
CONDA_BASE=$(conda info --base 2>/dev/null) || {
    echo "❌ Conda not found. Please install Miniconda/Anaconda first."
    echo "   Download from: https://docs.conda.io/en/latest/miniconda.html"
    exit 1
}
echo "📍 Conda is installed at: $CONDA_BASE"

# Initialize conda for this shell
eval "$(conda shell.bash hook)"

# Apply channel configuration fix if requested
if [ "$fix_channels" = true ]; then
    echo ""
    echo "🔧 Applying channel configuration fixes..."
    conda config --remove channels defaults 2>/dev/null || true
    conda config --prepend channels conda-forge
    conda config --append channels nvidia
    if [ "$install_pyrosetta" = true ]; then
        conda config --append channels https://conda.rosettacommons.org
    fi
    conda config --set channel_priority strict
    # Install libmamba solver if available (much faster)
    conda install -n base conda-libmamba-solver -y 2>/dev/null || echo "  Note: Could not install libmamba solver"
    conda config --set solver libmamba 2>/dev/null || echo "  Note: Could not set libmamba solver"
    echo "  ✓ Channel configuration applied"
fi

# =============================================================================
# Create and activate environment
# =============================================================================
echo ""

# Check if environment already exists
env_exists=false
conda env list | grep -w 'proteinhunter' >/dev/null 2>&1 && env_exists=true

if [ "$env_exists" = true ]; then
    if [ "$fresh_install" = true ]; then
        echo "🗑️  Removing existing proteinhunter environment (--fresh)..."
        conda env remove --name proteinhunter -y || {
            echo "❌ Failed to remove existing environment"
            exit 1
        }
        env_exists=false
        echo "  ✓ Old environment removed"
    elif [ "$upgrade_mode" = true ]; then
        echo "📦 Upgrading existing proteinhunter environment (--upgrade)..."
        echo "  Existing packages will be preserved, new packages will be added."
    else
        echo ""
        echo "❌ Environment 'proteinhunter' already exists!"
        echo ""
        echo "Options:"
        echo "  1. Update existing environment:  ./setup.sh --upgrade [other flags]"
        echo "  2. Fresh install (removes old):  ./setup.sh --fresh [other flags]"
        echo "  3. Manually remove first:        conda env remove --name proteinhunter"
        echo ""
        exit 1
    fi
fi

# Create environment if it doesn't exist (or was just removed)
if [ "$env_exists" = false ]; then
    echo "📦 Creating conda environment 'proteinhunter'..."
    $pkg_manager create --name proteinhunter python=3.10 -y || {
        echo "❌ Failed to create proteinhunter conda environment"
        exit 1
    }
    
    # Verify environment was created
    conda env list | grep -w 'proteinhunter' >/dev/null 2>&1 || {
        echo "❌ Conda environment 'proteinhunter' does not exist after creation."
        exit 1
    }
    echo "  ✓ Environment created"
fi

# Activate environment
echo "📍 Activating proteinhunter environment..."
source ${CONDA_BASE}/bin/activate ${CONDA_BASE}/envs/proteinhunter || {
    echo "❌ Failed to activate the proteinhunter environment."
    exit 1
}

[ "$CONDA_DEFAULT_ENV" = "proteinhunter" ] || {
    echo "❌ The proteinhunter environment is not active."
    exit 1
}
echo "  ✓ Environment activated: $CONDA_DEFAULT_ENV"

# =============================================================================
# Install Boltz
# =============================================================================
echo ""
if [ -d "boltz_ph" ]; then
    echo "📂 Installing Boltz..."
    cd boltz_ph
    pip install -e .
    cd ..
    echo "  ✓ Boltz installed"
else
    echo "❌ boltz_ph directory not found. Please run this script from the project root."
    exit 1
fi

# =============================================================================
# Install base conda packages
# =============================================================================
echo ""
echo "🔧 Installing base conda packages..."

# Base packages needed regardless of optional components
BASE_PACKAGES="pip pandas matplotlib numpy biopython scipy seaborn tqdm jupyter ipykernel libgfortran5"

# OpenMM + pdbfixer for open-source scoring (always install - lightweight and useful)
OPENMM_PACKAGES="pdbfixer openmm"

# Build package list
CONDA_PACKAGES="$BASE_PACKAGES $OPENMM_PACKAGES"

# Install via conda/mamba
if [ -n "$cuda" ]; then
    echo "  Using CUDA override: $cuda"
    CONDA_OVERRIDE_CUDA="$cuda" $pkg_manager install $CONDA_PACKAGES -c conda-forge -c nvidia -y || {
        echo "❌ Failed to install base conda packages"
        exit 1
    }
else
    $pkg_manager install $CONDA_PACKAGES -c conda-forge -c nvidia -y || {
        echo "❌ Failed to install base conda packages"
        exit 1
    }
fi
echo "  ✓ Base conda packages installed"

# =============================================================================
# Install Python dependencies (pip)
# =============================================================================
echo ""
echo "🔧 Installing Python dependencies..."
pip install prody PyYAML requests pypdb py3Dmol py2Dmol logmd==0.1.45 ml_collections || {
    echo "❌ Failed to install Python dependencies"
    exit 1
}
echo "  ✓ Python dependencies installed"

# Install FreeSASA (optional but recommended for open-source scoring)
echo ""
echo "🔧 Installing FreeSASA..."
pip install freesasa || {
    echo "  ⚠ Warning: FreeSASA installation failed. SASA calculations will fall back to Biopython."
}
python -c "import freesasa" >/dev/null 2>&1 && echo "  ✓ FreeSASA installed" || echo "  ⚠ FreeSASA not available - using Biopython fallback"

# =============================================================================
# Install PyRosetta (optional)
# =============================================================================
echo ""
if [ "$install_pyrosetta" = true ]; then
    echo "⏳ Installing PyRosetta (this may take a while)..."
    pip install pyrosettacolabsetup pyrosetta-installer || {
        echo "❌ Failed to install PyRosetta installer"
        exit 1
    }
    python -c 'import pyrosetta_installer; pyrosetta_installer.install_pyrosetta()' || {
        echo "❌ Failed to install PyRosetta"
        echo "   If you don't have a PyRosetta license, re-run with --no-pyrosetta"
        exit 1
    }
    
    # Fix NumPy + Numba compatibility (PyRosetta may downgrade NumPy)
    echo "  🩹 Fixing NumPy/Numba version compatibility..."
    pip install --upgrade "numpy>=1.24,<2.0" numba || {
        echo "  ⚠ Warning: Could not fix NumPy version"
    }
    
    # Verify PyRosetta installation
    python -c "import pyrosetta" >/dev/null 2>&1 && echo "  ✓ PyRosetta installed" || {
        echo "❌ PyRosetta installation verification failed"
        exit 1
    }
else
    echo "⏭️  Skipping PyRosetta installation (--no-pyrosetta)"
    echo "   Open-source scoring (OpenMM + FreeSASA) will be used instead."
fi

# =============================================================================
# Download Boltz weights
# =============================================================================
echo ""
echo "⬇️  Downloading Boltz weights..."
python << 'PYCODE'
import sys
import os
sys.path.insert(0, os.path.join(os.getcwd(), 'boltz_ph'))

try:
    from boltz.main import download_boltz2
    from pathlib import Path
    cache = Path('~/.boltz').expanduser()
    cache.mkdir(parents=True, exist_ok=True)
    download_boltz2(cache)
    print("  ✓ Boltz weights downloaded successfully!")
except Exception as e:
    print(f"❌ Error downloading Boltz weights: {e}")
    sys.exit(1)
PYCODE

# =============================================================================
# Setup LigandMPNN
# =============================================================================
echo ""
if [ -d "LigandMPNN" ]; then
    echo "🧬 Setting up LigandMPNN..."
    cd LigandMPNN
    bash get_model_params.sh "./model_params" || {
        echo "  ⚠ Warning: LigandMPNN model params download failed"
    }
    cd ..
    echo "  ✓ LigandMPNN configured"
else
    echo "⚠ LigandMPNN directory not found - skipping"
fi

# =============================================================================
# Install Protenix (optional)
# =============================================================================
echo ""
if [ "$install_protenix" = true ]; then
    echo "🧠 Setting up Protenix (open-source AF3)..."
    
    # Clone Protenix if not already present
    if [ ! -d "Protenix" ]; then
        echo "  Cloning Protenix repository..."
        git clone https://github.com/bytedance/Protenix.git || {
            echo "❌ Failed to clone Protenix repository"
            exit 1
        }
    else
        echo "  Protenix directory already exists, updating..."
        cd Protenix
        git pull || echo "  ⚠ Warning: Could not update Protenix"
        cd ..
    fi
    
    # Install Protenix dependencies
    echo "  Installing Protenix dependencies..."
    cd Protenix
    pip install -e . || {
        echo "  ⚠ Warning: Protenix pip install failed, trying manual dependency install..."
        pip install torch einops || true
    }
    cd ..
    
    # Install additional Protenix runtime dependencies (not always in their setup.py)
    echo "  Installing additional Protenix dependencies..."
    pip install biotite optree ml-collections dm-tree modelcif gemmi einops hydra-core || {
        echo "  ⚠ Warning: Some Protenix dependencies failed to install"
    }
    
    # Install cuEquivariance (CUDA extension, may fail on CPU-only systems)
    pip install cuequivariance-torch cuequivariance-ops-torch-cu12 2>/dev/null || {
        echo "  ⚠ Note: cuEquivariance not available (will use fallback)"
    }
    
    # Download Protenix weights (~1.4GB)
    PROTENIX_WEIGHTS_DIR="$HOME/.protein-hunter/protenix_weights"
    PROTENIX_WEIGHTS_FILE="$PROTENIX_WEIGHTS_DIR/protenix_base_default_v0.5.0.pt"
    PROTENIX_WEIGHTS_URL="https://af3-dev.tos-cn-beijing.volces.com/release_model/protenix_base_default_v0.5.0.pt"
    
    mkdir -p "$PROTENIX_WEIGHTS_DIR"
    if [ ! -f "$PROTENIX_WEIGHTS_FILE" ]; then
        echo "  Downloading Protenix weights (~1.4GB)..."
        if command -v wget &> /dev/null; then
            wget -q --show-progress -O "$PROTENIX_WEIGHTS_FILE" "$PROTENIX_WEIGHTS_URL" || {
                echo "  ⚠ Warning: Failed to download Protenix weights"
                echo "    Weights will be downloaded on first use"
                rm -f "$PROTENIX_WEIGHTS_FILE"
            }
        elif command -v curl &> /dev/null; then
            curl -L --progress-bar -o "$PROTENIX_WEIGHTS_FILE" "$PROTENIX_WEIGHTS_URL" || {
                echo "  ⚠ Warning: Failed to download Protenix weights"
                echo "    Weights will be downloaded on first use"
                rm -f "$PROTENIX_WEIGHTS_FILE"
            }
        else
            echo "  ⚠ Warning: Neither wget nor curl found, skipping weight download"
            echo "    Weights will be downloaded on first use"
        fi
    else
        echo "  ✓ Protenix weights already present"
    fi
    
    echo "  ✓ Protenix configured"
else
    echo "⏭️  Skipping Protenix installation (--no-protenix)"
fi

# =============================================================================
# Install Chai-lab (optional, opt-in only)
# =============================================================================
echo ""
if [ "$install_chai" = true ]; then
    echo "🧠 Installing Chai-lab dependencies (opt-in)..."
    pip install --no-deps \
        git+https://github.com/sokrypton/chai-lab.git \
        'gemmi~=0.6.3' \
        'jaxtyping>=0.2.25' \
        'pandera>=0.24' \
        'antipickle==0.2.0' \
        'rdkit~=2024.9.5' \
        'modelcif>=1.0' \
        typing_inspect \
        beartype \
        typeguard \
        ihm \
        mypy_extensions \
        equinox \
        wadler_lindig || {
        echo "❌ Chai-lab installation failed"
        exit 1
    }
    
    # Verify chai-lab installation
    python -c "import chai_lab" >/dev/null 2>&1 && echo "  ✓ Chai-lab installed" || {
        echo "❌ Chai-lab installation verification failed"
        exit 1
    }
else
    echo "⏭️  Skipping Chai-lab installation (use --install-chai to enable)"
fi

# =============================================================================
# Make executables executable
# =============================================================================
echo ""
echo "🔧 Setting up executables..."

# DAlphaBall.gcc is only needed for PyRosetta
if [ "$install_pyrosetta" = true ]; then
    if [ -f "utils/DAlphaBall.gcc" ]; then
        chmod +x "utils/DAlphaBall.gcc" || echo "  ⚠ Warning: Failed to chmod DAlphaBall.gcc"
        echo "  ✓ DAlphaBall.gcc made executable"
    else
        echo "  ⚠ Warning: utils/DAlphaBall.gcc not found"
    fi
else
    echo "  ⏭️  Skipping DAlphaBall.gcc (not needed without PyRosetta)"
fi

# sc binary for shape complementarity (used by open-source scoring)
if [ -f "utils/opensource_scoring/sc" ]; then
    chmod +x "utils/opensource_scoring/sc" || echo "  ⚠ Warning: Failed to chmod sc"
    echo "  ✓ sc (shape complementarity) made executable"
fi

# FASPR binary for side-chain repacking
if [ -f "utils/opensource_scoring/FASPR" ]; then
    chmod +x "utils/opensource_scoring/FASPR" || echo "  ⚠ Warning: Failed to chmod FASPR"
    echo "  ✓ FASPR made executable"
fi

# =============================================================================
# Setup Jupyter kernel
# =============================================================================
echo ""
echo "📓 Setting up Jupyter kernel..."
python -m ipykernel install --user --name=proteinhunter --display-name="Protein Hunter" || {
    echo "  ⚠ Warning: Failed to install Jupyter kernel"
}
echo "  ✓ Jupyter kernel configured"

# =============================================================================
# Cleanup
# =============================================================================
echo ""
echo "🧹 Cleaning up temporary files..."
$pkg_manager clean -a -y >/dev/null 2>&1 || true

# =============================================================================
# Summary
# =============================================================================
t=$SECONDS
echo ""
echo "=============================================="
if [ "$upgrade_mode" = true ]; then
    echo "🎉 Protein Hunter upgrade complete!"
else
    echo "🎉 Protein Hunter installation complete!"
fi
echo "=============================================="
echo ""
echo "Completed in $(($t / 3600))h $((($t / 60) % 60))m $(($t % 60))s"
echo ""
echo "Installed components:"
echo "  ✓ Boltz (core design engine)"
echo "  ✓ LigandMPNN (sequence design)"
echo "  ✓ OpenMM + pdbfixer (structure relaxation)"
[ "$install_pyrosetta" = true ] && echo "  ✓ PyRosetta (interface scoring)" || echo "  ✗ PyRosetta (skipped)"
[ "$install_protenix" = true ] && echo "  ✓ Protenix (open-source validation)" || echo "  ✗ Protenix (skipped)"
[ "$install_chai" = true ] && echo "  ✓ Chai-lab (optional)" || echo "  ✗ Chai-lab (skipped)"
echo ""
echo "To activate the environment:"
echo "  conda activate proteinhunter"
echo ""
echo "Quick test:"
echo "  python boltz_ph/design.py --help"
echo ""

if [ "$install_pyrosetta" = false ] && [ "$install_protenix" = false ]; then
    echo "⚠ Note: You skipped both PyRosetta and Protenix."
    echo "   Validation features will be limited to AF3 (requires separate setup)."
    echo ""
fi

if [ "$install_pyrosetta" = false ]; then
    echo "💡 Tip: Without PyRosetta, use --scoring-method opensource for interface scoring."
    echo ""
fi

echo "For more information, see LOCAL_SETUP_GUIDE.md"
