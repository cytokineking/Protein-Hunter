# Local Setup Guide — Protein Hunter (Boltz Edition)

Complete instructions for setting up the Protein Hunter pipeline locally with AlphaFold 3 validation and PyRosetta scoring.

---

## Table of Contents

- [Prerequisites](#prerequisites)
- [Quick Setup (Basic Pipeline)](#quick-setup-basic-pipeline)
- [Full Setup (With AF3 Validation)](#full-setup-with-af3-validation)
  - [Step 1: AlphaFold 3 Docker Setup](#step-1-alphafold-3-docker-setup)
  - [Step 2: AF3 Weights](#step-2-af3-weights)
- [Running the Pipeline](#running-the-pipeline)
  - [Basic Design (No Validation)](#basic-design-no-validation)
  - [Full Pipeline (With AF3 + PyRosetta)](#full-pipeline-with-af3--pyrosetta)
- [Output Structure](#output-structure)
- [Troubleshooting](#troubleshooting)
- [GPU Cloud Setup (DataCrunch/Lambda)](#gpu-cloud-setup-datacrunchlambda)

---

## Prerequisites

### Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| GPU | NVIDIA GPU with 40GB VRAM | A100/H100 (80GB+) |
| Storage | 100GB | 200GB+ (for AF3 weights + artifactrs) |

### Software Requirements

- **Linux** (Ubuntu 20.04/22.04/24.04 recommended)
- **NVIDIA Driver** (525+ for CUDA 12)
- **Docker** with NVIDIA Container Toolkit (for AF3)
- **Conda** (Miniconda or Anaconda)

---

## Quick Setup (Basic Pipeline)

This installs the Boltz design pipeline **without** AlphaFold 3 validation.

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

The setup script will:
1. Create a conda environment (`proteinhunter`)
2. Install Boltz and LigandMPNN
3. Install PyRosetta
4. Download Boltz model weights (~5GB to `~/.boltz/`)

### Test Basic Installation

```bash
conda activate proteinhunter

python boltz_ph/design.py \
    --name test_design \
    --protein_seqs "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVKALPDAQFEVVHSLAKWKRQQIAAALEHHHHHH" \
    --num_designs 1 \
    --num_cycles 3 \
    --min_protein_length 60 \
    --max_protein_length 90 \
    --gpu_id 0
```

---

## Full Setup (With AF3 Validation)

For production use, enable AlphaFold 3 validation and PyRosetta filtering to verify your designs.

### Step 1: AlphaFold 3 Docker Setup

#### Install Docker with NVIDIA Support

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

#### Clone and Build AlphaFold 3

```bash
# Clone AlphaFold 3
git clone https://github.com/google-deepmind/alphafold3.git ~/alphafold3
cd ~/alphafold3

# Build the Docker image (takes 10-20 minutes)
docker build -t alphafold3 .
```

### Step 2: AF3 Weights

AlphaFold 3 requires model weights (~5GB). You must request access from Google DeepMind.

#### Option A: Using Pre-downloaded Weights

If you have `af3.bin.zst`:

```bash
# Create models directory
mkdir -p ~/alphafold3/models

# Copy and decompress weights
cp /path/to/af3.bin.zst ~/alphafold3/models/
cd ~/alphafold3/models
zstd -d af3.bin.zst  # Creates af3.bin (~5GB)
```

#### Option B: Request from Google

1. Visit: https://github.com/google-deepmind/alphafold3
2. Follow their instructions to request weights
3. Download and place in `~/alphafold3/models/af3.bin`

### Verify AF3 Installation

```bash
# Test AF3 Docker container
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
    --protein_seqs "YOUR_TARGET_SEQUENCE" \
    --num_designs 5 \
    --num_cycles 7 \
    --min_protein_length 80 \
    --max_protein_length 120 \
    --alanine_bias \
    --gpu_id 0
```

### Full Pipeline (With AF3 + PyRosetta)

Enable validation to filter and score designs:

```bash
conda activate proteinhunter

python boltz_ph/design.py \
    --name my_binder_validated \
    --protein_seqs "YOUR_TARGET_SEQUENCE" \
    --num_designs 10 \
    --num_cycles 7 \
    --min_protein_length 80 \
    --max_protein_length 120 \
    --alanine_bias \
    --use_alphafold3_validation \
    --alphafold_dir ~/alphafold3 \
    --af3_docker_name alphafold3 \
    --gpu_id 0
```

### PDL1 Example (Tested)

```bash
python boltz_ph/design.py \
    --name PDL1_design \
    --protein_seqs "AFTVTVPKDLYVVEYGSNMTIECKFPVEKQLDLAALIVYWEMEDKNIIQFVHGEEDLKVQHSSYRQRARLLKDQLSLGNAALQITDVKLQDAGVYRCMISYGGADYKRITVKVNAPYAAALE" \
    --num_designs 5 \
    --num_cycles 7 \
    --alanine_bias \
    --use_alphafold3_validation \
    --alphafold_dir ~/alphafold3 \
    --af3_docker_name alphafold3 \
    --min_protein_length 60 \
    --max_protein_length 90 \
    --gpu_id 0
```

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

#### 1. `libgfortran.so.5: cannot open shared object file`

PyRosetta's DAlphaBall requires libgfortran:

```bash
# Ubuntu/Debian
sudo apt-get install libgfortran5

# CentOS/RHEL
sudo yum install libgfortran
```

#### 2. `CUDA out of memory`

Reduce batch size or use a larger GPU:

```bash
# Try smaller designs
--min_protein_length 50 --max_protein_length 80
```

#### 3. AF3 Docker Permission Denied

```bash
# Add user to docker group
sudo usermod -aG docker $USER
newgrp docker

# Or run with sudo (not recommended for production)
```

#### 4. `conda: command not found` in setup.sh

Install Miniconda first:

```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
source ~/.bashrc
```

---

## GPU Cloud Setup

For cloud GPU instances, follow this setup sequence:

### 1. SSH to Instance

```bash
ssh root@YOUR_INSTANCE_IP -i /path/to/ssh_key
```

### 2. Install Dependencies

```bash
# Update system
apt-get update && apt-get upgrade -y

# Install required packages
apt-get install -y git wget curl zstd libgfortran5

# Install Docker (if not pre-installed)
curl -fsSL https://get.docker.com -o get-docker.sh && sh get-docker.sh

# Install NVIDIA Container Toolkit
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
    sudo tee /etc/apt/sources.list.d/nvidia-docker.list
apt-get update && apt-get install -y nvidia-container-toolkit
systemctl restart docker
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

# Accept Conda TOS if needed
conda tos accept

# Run setup
chmod +x setup.sh
./setup.sh
```

### 5. Setup AlphaFold 3

```bash
# Clone AF3
git clone https://github.com/google-deepmind/alphafold3.git ~/alphafold3

# Build Docker image
cd ~/alphafold3
docker build -t alphafold3 .

# Create models directory
mkdir -p ~/alphafold3/models
```

### 6. Transfer AF3 Weights (from local)

```bash
# On your LOCAL machine
rsync -avz --progress -e "ssh -i /path/to/ssh_key" \
    /path/to/af3.bin.zst \
    root@YOUR_INSTANCE_IP:~/alphafold3/models/

# On REMOTE instance
cd ~/alphafold3/models
zstd -d af3.bin.zst
```

### 7. Run a Test Job

```bash
source ~/miniconda/etc/profile.d/conda.sh
conda activate proteinhunter
cd ~/Protein-Hunter

python boltz_ph/design.py \
    --name cloud_test \
    --protein_seqs "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVKALPDAQFEVVHSLAKWKRQQIAAALEHHHHHH" \
    --num_designs 2 \
    --num_cycles 5 \
    --alanine_bias \
    --use_alphafold3_validation \
    --alphafold_dir ~/alphafold3 \
    --af3_docker_name alphafold3 \
    --min_protein_length 60 \
    --max_protein_length 90 \
    --gpu_id 0
```

### 8. Sync Results Back (from local)

```bash
# On LOCAL machine - one-time sync
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

