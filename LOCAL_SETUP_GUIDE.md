# Local Setup Guide — Protein Hunter (Boltz Edition)

Complete instructions for setting up the Protein Hunter pipeline locally with multiple validation and scoring options.

---

## Table of Contents

- [Prerequisites](#prerequisites)
- [Installation Options](#installation-options)
- [Quick Setup](#quick-setup)
  - [Full Installation (Recommended)](#full-installation-recommended)
  - [Open-Source Only (No License Required)](#open-source-only-no-license-required)
  - [Minimal Installation](#minimal-installation)
- [Validation Setup](#validation-setup)
  - [Option A: Protenix (Open-Source)](#option-a-protenix-open-source)
  - [Option B: AlphaFold 3 (Requires Weights)](#option-b-alphafold-3-requires-weights)
- [Running the Pipeline](#running-the-pipeline)
  - [Basic Design (No Validation)](#basic-design-no-validation)
  - [With Protenix Validation (Open-Source)](#with-protenix-validation-open-source)
  - [With AF3 Validation](#with-af3-validation)
- [Output Structure](#output-structure)
- [Troubleshooting](#troubleshooting)
- [GPU Cloud Setup](#gpu-cloud-setup)

---

## Prerequisites

### Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| GPU | NVIDIA GPU with 24GB VRAM | A100/H100 (40GB+) |
| Storage | 50GB | 200GB+ (for all weights) |

### Software Requirements

- **Linux** (Ubuntu 20.04/22.04/24.04 recommended)
- **NVIDIA Driver** (525+ for CUDA 12)
- **Conda** (Miniconda or Anaconda)
- **Docker** with NVIDIA Container Toolkit (only if using AF3 validation)

---

## Installation Options

The setup script installs all components by default. Use flags to customize:

| Flag | Effect |
|------|--------|
| (default) | Install everything: Boltz, LigandMPNN, PyRosetta, Protenix, OpenMM |
| `--no-pyrosetta` | Skip PyRosetta (use OpenMM/FreeSASA for scoring instead) |
| `--no-protenix` | Skip Protenix (use AF3 for validation instead) |
| `--install-chai` | Install Chai-lab (opt-in, not needed for standard workflows) |
| `--pkg-manager mamba` | Use mamba instead of conda (faster) |
| `--fix-channels` | Fix conda channel priority issues |

### Component Overview

| Component | Purpose | License | Installed By Default |
|-----------|---------|---------|---------------------|
| **Boltz** | Structure prediction & design | Open-source | ✓ |
| **LigandMPNN** | Sequence optimization | Open-source | ✓ |
| **Protenix** | Validation (open-source AF3) | Open-source | ✓ |
| **OpenMM + FreeSASA** | Structure relaxation & scoring | Open-source | ✓ |
| **PyRosetta** | Interface scoring (optional) | Academic/Commercial | ✓ |
| **Chai-lab** | Alternative design engine | Open-source | ✗ (opt-in) |

---

## Quick Setup

### Full Installation (Recommended)

Installs all components including PyRosetta (requires license) and Protenix:

```bash
# Clone the repository
git clone https://github.com/cytokineking/Protein-Hunter.git
cd Protein-Hunter

# Run the setup script
chmod +x setup.sh
./setup.sh

# Activate the environment
conda activate proteinhunter
```

This installs:
- Boltz (core design engine)
- LigandMPNN (sequence design)
- PyRosetta (interface scoring)
- Protenix (open-source validation)
- OpenMM + FreeSASA (structure relaxation)

### Open-Source Only (No License Required)

Skip PyRosetta if you don't have a license. Uses OpenMM + FreeSASA for scoring:

```bash
./setup.sh --no-pyrosetta

# Activate
conda activate proteinhunter
```

When running designs, use `--scoring-method opensource`:

```bash
python boltz_ph/design.py \
    --name my_design \
    --protein-seqs "YOUR_SEQUENCE" \
    --validation-model protenix \
    --scoring-method opensource \
    --num-designs 5
```

### Minimal Installation

For basic design without validation (fastest setup):

```bash
./setup.sh --no-pyrosetta --no-protenix

# Activate
conda activate proteinhunter
```

---

## Validation Setup

Protein Hunter supports two validation backends:

| Backend | Requirements | Weights | License |
|---------|--------------|---------|---------|
| **Protenix** | Included in setup | Auto-download (~1.4GB) | Open-source |
| **AlphaFold 3** | Docker + manual setup | Request from Google (~5GB) | Research only |

### Option A: Protenix (Open-Source)

Protenix is installed by default. Weights are automatically downloaded on first use to `~/.protein-hunter/protenix_weights/`.

**No additional setup required!**

Test Protenix validation:

```bash
python boltz_ph/design.py \
    --name protenix_test \
    --protein-seqs "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVKALPDAQFEVVHSLAKWKRQQIAAALEHHHHHH" \
    --num-designs 1 \
    --num-cycles 3 \
    --validation-model protenix \
    --scoring-method opensource \
    --gpu-id 0
```

### Option B: AlphaFold 3 (Requires Weights)

AF3 requires manual Docker setup and weights from Google DeepMind.

#### Step 1: Install Docker with NVIDIA Support

```bash
# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

# Install NVIDIA Container Toolkit
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
    sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker

# Test GPU access in Docker
docker run --rm --gpus all nvidia/cuda:12.1-base nvidia-smi
```

#### Step 2: Clone and Build AlphaFold 3

```bash
git clone https://github.com/google-deepmind/alphafold3.git ~/alphafold3
cd ~/alphafold3
docker build -t alphafold3 .  # Takes 10-20 minutes
```

#### Step 3: Download AF3 Weights

Request access from Google DeepMind, then:

```bash
mkdir -p ~/alphafold3/models
# Copy your af3.bin.zst to ~/alphafold3/models/
cd ~/alphafold3/models
zstd -d af3.bin.zst  # Creates af3.bin (~5GB)
```

#### Verify AF3 Installation

```bash
docker run --rm --gpus all alphafold3 python -c "print('AF3 container works!')"
```

---

## Running the Pipeline

### Basic Design (No Validation)

Fast design without structure validation:

```bash
conda activate proteinhunter

python boltz_ph/design.py \
    --name my_binder \
    --protein-seqs "YOUR_TARGET_SEQUENCE" \
    --num-designs 5 \
    --num-cycles 7 \
    --min-protein-length 80 \
    --max-protein-length 120 \
    --alanine-bias \
    --gpu-id 0
```

### With Protenix Validation (Open-Source)

**Recommended for most users** - no external dependencies or licenses:

```bash
python boltz_ph/design.py \
    --name my_binder_validated \
    --protein-seqs "YOUR_TARGET_SEQUENCE" \
    --num-designs 10 \
    --num-cycles 7 \
    --min-protein-length 80 \
    --max-protein-length 120 \
    --alanine-bias \
    --validation-model protenix \
    --scoring-method opensource \
    --gpu-id 0
```

With PyRosetta scoring (if installed):

```bash
python boltz_ph/design.py \
    --name my_binder_validated \
    --protein-seqs "YOUR_TARGET_SEQUENCE" \
    --num-designs 10 \
    --validation-model protenix \
    --scoring-method pyrosetta \
    --gpu-id 0
```

### With AF3 Validation

For highest accuracy (requires AF3 setup):

```bash
python boltz_ph/design.py \
    --name my_binder_af3 \
    --protein-seqs "YOUR_TARGET_SEQUENCE" \
    --num-designs 10 \
    --num-cycles 7 \
    --alanine-bias \
    --validation-model af3 \
    --alphafold-dir ~/alphafold3 \
    --af3-docker-name alphafold3 \
    --gpu-id 0
```

### PDL1 Example

```bash
python boltz_ph/design.py \
    --name PDL1_design \
    --protein-seqs "AFTVTVPKDLYVVEYGSNMTIECKFPVEKQLDLAALIVYWEMEDKNIIQFVHGEEDLKVQHSSYRQRARLLKDQLSLGNAALQITDVKLQDAGVYRCMISYGGADYKRITVKVNAPYAAALE" \
    --num-designs 5 \
    --num-cycles 7 \
    --alanine-bias \
    --validation-model protenix \
    --scoring-method opensource \
    --min-protein-length 60 \
    --max-protein-length 90 \
    --gpu-id 0
```

### Validation & Scoring Options

| Option | Values | Description |
|--------|--------|-------------|
| `--validation-model` | `none`, `af3`, `protenix` | Structure validation method |
| `--scoring-method` | `pyrosetta`, `opensource` | Interface scoring method |

**Recommended combinations:**

| Use Case | Validation | Scoring |
|----------|------------|---------|
| Fully open-source | `protenix` | `opensource` |
| Best accuracy | `af3` | `pyrosetta` |
| No PyRosetta license | `protenix` or `af3` | `opensource` |
| Fast iteration | `none` | N/A |

---

## Output Structure

After a successful run with validation enabled:

```
results_{name}/
├── designs/                    # All design cycles
│   ├── design_stats.csv        # Metrics for all cycles
│   └── {name}_d{X}_c{Y}.pdb    # Structure files
├── best_designs/               # Best cycle per design
│   ├── best_designs.csv
│   └── {name}_d{X}_c{Y}.pdb
├── refolded/                   # Refolded structures (AF3 or Protenix)
│   ├── validation_results.csv  # ipTM, pTM, pLDDT scores
│   └── *_refolded.cif          # Predicted structures
├── accepted_designs/           # Passed all filters
│   ├── accepted_stats.csv      # Full metrics
│   └── *_relaxed.pdb           # Relaxed structures
└── rejected/                   # Failed filters
    ├── rejected_stats.csv      # Metrics + rejection reason
    └── *_relaxed.pdb           # Relaxed structures
```

---

## Troubleshooting

### Common Issues

#### 1. `PyRosetta not installed` error

You selected `--scoring-method pyrosetta` but PyRosetta isn't installed:

```bash
# Option 1: Re-run setup with PyRosetta
./setup.sh  # Full install includes PyRosetta

# Option 2: Use open-source scoring instead
python boltz_ph/design.py ... --scoring-method opensource
```

#### 2. `Protenix repository not found` error

You selected `--validation-model protenix` but Protenix isn't installed:

```bash
# Option 1: Re-run setup with Protenix
./setup.sh  # Full install includes Protenix

# Option 2: Manually clone Protenix
git clone https://github.com/bytedance/Protenix.git ~/Protein-Hunter/Protenix
```

#### 3. `libgfortran.so.5: cannot open shared object file`

PyRosetta's DAlphaBall requires libgfortran:

```bash
# Ubuntu/Debian
sudo apt-get install libgfortran5

# CentOS/RHEL
sudo yum install libgfortran
```

#### 4. `CUDA out of memory`

Reduce binder size or use a larger GPU:

```bash
--min-protein-length 50 --max-protein-length 80
```

#### 5. AF3 Docker Permission Denied

```bash
# Add user to docker group
sudo usermod -aG docker $USER
newgrp docker
```

#### 6. Conda dependency resolution slow/failing

Use mamba for faster resolution:

```bash
./setup.sh --pkg-manager mamba --fix-channels
```

#### 7. `conda: command not found`

Install Miniconda first:

```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
source ~/.bashrc
```

---

## GPU Cloud Setup

For cloud GPU instances (DataCrunch, Lambda, etc.):

### 1. SSH to Instance

```bash
ssh root@YOUR_INSTANCE_IP -i /path/to/ssh_key
```

### 2. Install System Dependencies

```bash
apt-get update && apt-get upgrade -y
apt-get install -y git wget curl zstd libgfortran5
```

### 3. Install Miniconda

```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b -p ~/miniconda
eval "$(~/miniconda/bin/conda shell.bash hook)"
conda init bash
source ~/.bashrc
```

### 4. Install Protein Hunter

```bash
git clone https://github.com/cytokineking/Protein-Hunter.git ~/Protein-Hunter
cd ~/Protein-Hunter

# Full installation (with PyRosetta)
./setup.sh

# OR open-source only (no license needed)
./setup.sh --no-pyrosetta
```

### 5. (Optional) Setup AF3 for Validation

If using AF3 instead of Protenix:

```bash
# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh && sh get-docker.sh

# Install NVIDIA Container Toolkit
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
    sudo tee /etc/apt/sources.list.d/nvidia-docker.list
apt-get update && apt-get install -y nvidia-container-toolkit
systemctl restart docker

# Build AF3
git clone https://github.com/google-deepmind/alphafold3.git ~/alphafold3
cd ~/alphafold3
docker build -t alphafold3 .

# Transfer weights from local machine
mkdir -p ~/alphafold3/models
# (transfer af3.bin.zst via rsync/scp)
cd ~/alphafold3/models && zstd -d af3.bin.zst
```

### 6. Run a Test Job

```bash
source ~/miniconda/etc/profile.d/conda.sh
conda activate proteinhunter
cd ~/Protein-Hunter

# Using Protenix (simplest)
python boltz_ph/design.py \
    --name cloud_test \
    --protein-seqs "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVKALPDAQFEVVHSLAKWKRQQIAAALEHHHHHH" \
    --num-designs 2 \
    --num-cycles 5 \
    --validation-model protenix \
    --scoring-method opensource \
    --gpu-id 0
```

### 7. Sync Results Back

```bash
# On LOCAL machine
rsync -avz --progress -e "ssh -i /path/to/ssh_key" \
    root@YOUR_INSTANCE_IP:~/Protein-Hunter/results_*/ \
    ./local_results/

# Or continuous sync (every 30 seconds)
while true; do
    rsync -avz -e "ssh -i /path/to/ssh_key" \
        root@YOUR_INSTANCE_IP:~/Protein-Hunter/results_*/ \
        ./local_results/
    sleep 30
done
```

---

*Last updated: December 2025*
