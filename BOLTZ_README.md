# Protein Hunter — Boltz Edition ⚡

A protein binder design pipeline using Boltz structure prediction and LigandMPNN sequence design.

---

## Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Design Modes](#design-modes)
  - [1. De Novo Design](#1-de-novo-design)
  - [2. Optimization (Refinement)](#2-optimization-refinement)
- [Target Specification](#target-specification)
  - [Sequence-Only Mode](#sequence-only-mode)
  - [Template Structure Mode](#template-structure-mode)
- [Hotspot Configuration](#hotspot-configuration)
- [Command-Line Arguments Reference](#command-line-arguments-reference)
- [Output Files](#output-files)
- [Examples](#examples)
- [Modal Cloud Deployment](#modal-cloud-deployment)
  - [Modal Setup](#modal-setup)
  - [Modal Quick Start](#modal-quick-start)
  - [Modal CLI Arguments](#modal-cli-arguments)
  - [Parallelization & Concurrency](#parallelization--concurrency)
  - [Template Structures on Modal](#template-structures-on-modal)
  - [Modal Examples](#modal-examples)

---

## Overview

Protein Hunter uses an iterative **hallucination-based design** approach:

```
┌──────────────────────────────────────────────────────────────┐
│  1. Start with random/unknown sequence for binder (Chain A)  │
│                           ↓                                  │
│  2. Predict complex structure (Boltz)                        │
│                           ↓                                  │
│  3. Design sequence for predicted structure (LigandMPNN)     │
│                           ↓                                  │
│  4. Repeat steps 2-3 for N cycles                            │
│                           ↓                                  │
│  5. Output: Optimized binder sequence + structure            │
└──────────────────────────────────────────────────────────────┘
```

**Key terminology:**
- **Binder (Chain A)**: The protein being designed
- **Target (Chain B, C, ...)**: What you want to bind (provided by you)
- **Design run**: One complete optimization trajectory
- **Cycle**: One iteration of fold → design

---

## Installation

```bash
git clone https://github.com/your-fork/Protein-Hunter.git
cd Protein-Hunter
chmod +x setup.sh
./setup.sh
```

---

## Quick Start

Design a binder for a target protein:

```bash
python boltz_ph/design.py \
    --name my_first_design \
    --protein_seqs "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVKALPDAQFEVVHSLAKWKRQQIAAALEHHHHHH" \
    --num_designs 3 \
    --num_cycles 7 \
    --min_protein_length 80 \
    --max_protein_length 120 \
    --msa_mode mmseqs \
    --alanine_bias \
    --high_iptm_threshold 0.7 \
    --gpu_id 0
```

---

## Design Modes

### 1. De Novo Design

**Purpose**: Design a completely new binder from scratch.

The binder starts as a random sequence (with optional "X" unknown residues) and gets optimized through multiple cycles.

```bash
python boltz_ph/design.py \
    --name de_novo_binder \
    --protein_seqs "YOUR_TARGET_SEQUENCE" \
    --num_designs 5 \
    --num_cycles 7 \
    --min_protein_length 90 \
    --max_protein_length 150 \
    --percent_X 50 \
    --msa_mode mmseqs \
    --alanine_bias \
    --high_iptm_threshold 0.7 \
    --gpu_id 0
```

**Key parameters for de novo design:**

| Parameter | Description | Recommended |
|-----------|-------------|-------------|
| `--percent_X` | % of "unknown" residues in initial sequence | 50-100 |
| `--min_protein_length` | Minimum binder length | 80-100 |
| `--max_protein_length` | Maximum binder length | 120-150 |
| `--num_designs` | Independent design attempts | 3-10 |
| `--num_cycles` | Optimization iterations per design | 5-10 |

---

### 2. Optimization (Refinement)

**Purpose**: Improve an existing binder sequence.

Provide a starting sequence via `--seq` to refine it further.

```bash
python boltz_ph/design.py \
    --name refine_my_binder \
    --seq "GPDRERARELARILLKVIKLSDSPEARRQLLRNLEELAEKYKDPEVRRILEEAERYIK" \
    --protein_seqs "YOUR_TARGET_SEQUENCE" \
    --num_designs 3 \
    --num_cycles 5 \
    --msa_mode mmseqs \
    --high_iptm_threshold 0.8 \
    --gpu_id 0
```

**Key differences from de novo:**
- `--seq` provides the starting binder sequence
- `--min/max_protein_length` are ignored (length is fixed)
- `--percent_X` is ignored (sequence is provided)
- Typically use fewer cycles and higher quality thresholds

---

## Target Specification

You must specify what you want to design a binder FOR. There are two paradigms:

### Sequence-Only Mode

**Use when**: You don't have an experimental structure of your target.

Boltz will **predict the target structure** during design, using MSA information for accuracy.

```bash
python boltz_ph/design.py \
    --protein_seqs "YOUR_TARGET_SEQUENCE" \
    --msa_mode mmseqs \
    ...
```

| Parameter | Description |
|-----------|-------------|
| `--protein_seqs` | Target protein sequence (required) |
| `--msa_mode mmseqs` | Generate MSA for better structure prediction |

**For multi-chain targets** (e.g., dimers), separate sequences with `:`:

```bash
--protein_seqs "CHAIN_B_SEQUENCE:CHAIN_C_SEQUENCE"
```

---

### Template Structure Mode

**Use when**: You have an experimental structure (PDB/CIF) of your target.

The template provides **explicit coordinates** that anchor the target structure during design.

```bash
python boltz_ph/design.py \
    --protein_seqs "YOUR_TARGET_SEQUENCE" \
    --template_path "7KPL" \
    --template_cif_chain_id "A" \
    --msa_mode single \
    ...
```

| Parameter | Description |
|-----------|-------------|
| `--protein_seqs` | Target sequence (must match template) |
| `--template_path` | PDB code, file path, or AlphaFold ID |
| `--template_cif_chain_id` | Which chain in template to use |
| `--msa_mode single` | Often used with templates (MSA optional) |

**Template path options:**

| Format | Example | Description |
|--------|---------|-------------|
| PDB code | `"7KPL"` | Auto-downloads from RCSB |
| Local file | `"./structures/target.cif"` | Your own file (.pdb or .cif) |
| AlphaFold ID | `"P12345"` | Auto-downloads from AlphaFold DB |

**Multi-chain templates:**

```bash
--protein_seqs "SEQ_B:SEQ_C" \
--template_path "7KPL:7KPL" \
--template_cif_chain_id "A:B"
```

**Handling template gaps:**

If your template has missing residues (unresolved loops), provide the **full sequence** in `--protein_seqs`. Boltz will:
- Use template coordinates where available
- Predict missing regions

---

## Hotspot Configuration

**Hotspots** are target residues that the binder MUST contact.

### Basic Usage

```bash
--contact_residues "54,56,66,115,121"
```

This forces the binder to contact residues 54, 56, 66, 115, and 121 on the target.

### Multi-Chain Hotspots

Use `|` to separate hotspots for different target chains:

```bash
--protein_seqs "SEQ_B:SEQ_C" \
--contact_residues "10,20,30|5,15"
```

- Residues 10, 20, 30 on Chain B
- Residues 5, 15 on Chain C

### Hotspot Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--contact_residues` | `""` | Comma-separated residue positions |
| `--contact_cutoff` | `15.0` | Distance threshold (Å) for contact |
| `--max_contact_filter_retries` | `6` | Retries if contacts not satisfied |
| `--no_contact_filter` | `False` | Disable contact checking |

### How Contact Filtering Works

1. **Cycle 0**: After initial prediction, check if binder contacts all hotspots
2. **If failed**: Resample initial sequence, retry (up to `max_contact_filter_retries`)
3. **High-ipTM saving**: Designs only saved if contacts are satisfied

### Example with Hotspots

```bash
python boltz_ph/design.py \
    --name hotspot_design \
    --protein_seqs "TARGET_SEQUENCE" \
    --contact_residues "29,54,56,115,116,117" \
    --contact_cutoff 12.0 \
    --max_contact_filter_retries 10 \
    --num_designs 5 \
    --num_cycles 7 \
    --msa_mode mmseqs \
    --gpu_id 0
```

---

## Command-Line Arguments Reference

### Core Settings

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--name` | str | required | Job name (used for output folder) |
| `--gpu_id` | int | `0` | GPU device ID |
| `--num_designs` | int | `50` | Number of independent design runs |
| `--num_cycles` | int | `5` | Fold→design iterations per run |
| `--mode` | str | `"binder"` | `"binder"` or `"unconditional"` |

### Binder Sequence Settings

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--seq` | str | `""` | Starting binder sequence (empty = random) |
| `--min_protein_length` | int | `100` | Minimum binder length (if random start) |
| `--max_protein_length` | int | `150` | Maximum binder length (if random start) |
| `--percent_X` | int | `90` | % of "X" (unknown) in initial sequence |
| `--cyclic` | flag | `False` | Design cyclic peptide |

### Target Specification

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--protein_seqs` | str | `""` | Target protein sequence(s), `:` separated |
| `--ligand_smiles` | str | `""` | Target ligand as SMILES |
| `--ligand_ccd` | str | `""` | Target ligand as CCD code (e.g., `"SAM"`) |
| `--nucleic_seq` | str | `""` | Target DNA/RNA sequence |
| `--nucleic_type` | str | `"dna"` | `"dna"` or `"rna"` |

### Template Settings

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--template_path` | str | `""` | PDB code, file path, or AlphaFold ID |
| `--template_cif_chain_id` | str | `""` | Chain ID(s) in template for alignment |
| `--msa_mode` | str | `"mmseqs"` | `"mmseqs"` (generate MSA) or `"single"` (no MSA) |

### Hotspot/Contact Settings

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--contact_residues` | str | `""` | Target residues to contact (e.g., `"10,20,30"`) |
| `--contact_cutoff` | float | `15.0` | Contact distance threshold (Å) |
| `--max_contact_filter_retries` | int | `6` | Retries if contacts unsatisfied |
| `--no_contact_filter` | flag | `False` | Disable contact filtering |

### Sequence Design (MPNN) Settings

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--temperature` | float | `0.1` | MPNN sampling temperature |
| `--omit_AA` | str | `"C"` | Amino acids to exclude |
| `--alanine_bias` | flag | `False` | Penalize alanine (prevents poly-A) |
| `--alanine_bias_start` | float | `-0.5` | Initial alanine penalty |
| `--alanine_bias_end` | float | `-0.1` | Final alanine penalty |

### Quality Thresholds

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--high_iptm_threshold` | float | `0.8` | Min ipTM to save design |
| `--high_plddt_threshold` | float | `0.8` | Min pLDDT to save design |

### Model Settings

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--boltz_model_version` | str | `"boltz2"` | `"boltz1"` or `"boltz2"` |
| `--diffuse_steps` | int | `200` | Diffusion timesteps |
| `--recycling_steps` | int | `3` | Trunk recycles |

### Output Settings

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--save_dir` | str | `""` | Custom output directory |
| `--plot` | flag | `False` | Generate optimization plots |

### AlphaFold3 Validation (Optional)

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--use_alphafold3_validation` | flag | `False` | Enable AF3 cross-validation |
| `--alphafold_dir` | str | `"~/alphafold3"` | AF3 installation path |
| `--use_msa_for_af3` | flag | `False` | Use MSA in AF3 validation |

---

## Output Files

Results are saved to `./results_boltz/{name}/`:

```
results_boltz/my_design/
├── 0_protein_hunter_design/
│   ├── run_0/
│   │   ├── my_design_run_0_predicted_cycle_0.pdb
│   │   ├── my_design_run_0_predicted_cycle_1.pdb
│   │   ├── ...
│   │   └── my_design_run_0_best_structure.pdb
│   ├── run_1/
│   │   └── ...
│   └── B_msa.a3m                    # MSA file (if mmseqs used)
├── high_iptm_yaml/                  # YAML inputs for successful designs
├── high_iptm_pdb/                   # Structures for successful designs
├── summary_all_runs.csv             # All metrics for all runs/cycles
└── summary_high_iptm.csv            # Summary of high-quality designs
```

**Key files:**
- `summary_all_runs.csv` — Complete metrics for every run and cycle
- `high_iptm_yaml/` — Input files for designs exceeding quality thresholds
- `high_iptm_pdb/` — Structures ready for downstream analysis

---

## Examples

### Example 1: Basic Protein Binder Design

```bash
python boltz_ph/design.py \
    --name PDL1_binder \
    --protein_seqs "AFTVTVPKDLYVVEYGSNMTIECKFPVEKQLDLAALIVYWEMEDKNIIQFVHGEEDLKVQHSSYRQRARLLKDQLSLGNAALQITDVKLQDAGVYRCMISYGGADYKRITVKVNAPYAAALE" \
    --num_designs 5 \
    --num_cycles 7 \
    --min_protein_length 90 \
    --max_protein_length 150 \
    --percent_X 50 \
    --msa_mode mmseqs \
    --alanine_bias \
    --high_iptm_threshold 0.7 \
    --plot \
    --gpu_id 0
```

### Example 2: Design with Template Structure

```bash
python boltz_ph/design.py \
    --name PDL1_template_design \
    --protein_seqs "AFTVTVPKDLYVVEYGSNMTIECKFPVEKQLDLAALIVYWEMEDKNIIQFVHGEEDLKVQHSSYRQRARLLKDQLSLGNAALQITDVKLQDAGVYRCMISYGGADYKRITVKVNAPYAAALE" \
    --template_path "7KPL" \
    --template_cif_chain_id "B" \
    --msa_mode single \
    --num_designs 3 \
    --num_cycles 7 \
    --min_protein_length 90 \
    --max_protein_length 150 \
    --alanine_bias \
    --high_iptm_threshold 0.7 \
    --gpu_id 0
```

### Example 3: Hotspot-Directed Design

```bash
python boltz_ph/design.py \
    --name PDL1_hotspot_design \
    --protein_seqs "AFTVTVPKDLYVVEYGSNMTIECKFPVEKQLDLAALIVYWEMEDKNIIQFVHGEEDLKVQHSSYRQRARLLKDQLSLGNAALQITDVKLQDAGVYRCMISYGGADYKRITVKVNAPYAAALE" \
    --contact_residues "54,56,66,115,121" \
    --contact_cutoff 12.0 \
    --num_designs 5 \
    --num_cycles 7 \
    --msa_mode mmseqs \
    --alanine_bias \
    --high_iptm_threshold 0.7 \
    --gpu_id 0
```

### Example 4: Refine Existing Binder

```bash
python boltz_ph/design.py \
    --name refine_binder \
    --seq "MKAELRQRVQELAEQARQKLEEAEQKRVQELAEQARQKLEE" \
    --protein_seqs "TARGET_SEQUENCE" \
    --num_designs 3 \
    --num_cycles 5 \
    --msa_mode mmseqs \
    --high_iptm_threshold 0.85 \
    --gpu_id 0
```

### Example 5: Small Molecule Binder

```bash
python boltz_ph/design.py \
    --name SAM_binder \
    --ligand_ccd SAM \
    --num_designs 5 \
    --num_cycles 7 \
    --min_protein_length 130 \
    --max_protein_length 150 \
    --alanine_bias \
    --high_iptm_threshold 0.7 \
    --gpu_id 0
```

### Example 6: Cyclic Peptide Binder

```bash
python boltz_ph/design.py \
    --name cyclic_peptide \
    --protein_seqs "TARGET_SEQUENCE" \
    --cyclic \
    --num_designs 5 \
    --num_cycles 7 \
    --min_protein_length 10 \
    --max_protein_length 20 \
    --percent_X 100 \
    --msa_mode mmseqs \
    --high_iptm_threshold 0.8 \
    --gpu_id 0
```

### Example 7: Multimer Target (Homodimer)

```bash
python boltz_ph/design.py \
    --name dimer_binder \
    --protein_seqs "CHAIN_B_SEQ:CHAIN_B_SEQ" \
    --num_designs 3 \
    --num_cycles 7 \
    --min_protein_length 90 \
    --max_protein_length 150 \
    --msa_mode mmseqs \
    --alanine_bias \
    --high_iptm_threshold 0.7 \
    --gpu_id 0
```

---

## Tips & Best Practices

1. **Start with lower thresholds** (`--high_iptm_threshold 0.6-0.7`) for initial exploration, then increase for production runs.

2. **Use `--alanine_bias`** to avoid poly-alanine sequences that fold but aren't useful.

3. **More designs > more cycles**: Running 10 designs × 5 cycles often gives better diversity than 3 designs × 15 cycles.

4. **For difficult targets**, increase `--max_contact_filter_retries` and use hotspots to guide the design.

5. **Template mode** is recommended when you have high-confidence structures; sequence-only mode is better for exploring conformational flexibility.

---

## Modal Cloud Deployment

Run Protein Hunter on cloud GPUs using [Modal](https://modal.com). This enables:

- **Serverless execution** — No GPU setup, pay only for what you use
- **Parallel designs** — Run multiple designs simultaneously across GPUs
- **Template structures** — Full support for PDB template conditioning
- **Real-time streaming** — Results sync to local filesystem as they complete

### Modal Setup

1. **Install Modal CLI:**

```bash
pip install modal
modal setup  # Authenticate with Modal
```

2. **Initialize the cache** (run once, ~15 minutes):

```bash
modal run modal_protein_hunter.py::init_cache
```

This downloads Boltz2 weights, CCD data, and LigandMPNN models to a persistent Modal volume.

### Modal Quick Start

```bash
# Basic design run
modal run modal_protein_hunter.py::run_pipeline \
    --name "my_design" \
    --target-seq "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVKALPDAQFEVVHSLAKWKRQQIAAALEHHHHHH" \
    --num-designs 5 \
    --num-cycles 7 \
    --gpu H100

# List available GPUs
modal run modal_protein_hunter.py::list_gpus
```

### Modal CLI Arguments

#### Job Identity

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--name` | str | `"protein_hunter_run"` | Job name (used for output folder) |
| `--output-dir` | str | `"./results_modal/{name}"` | Local output directory |

#### Target Specification

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--target-seq` | str | None | Target protein sequence(s), `:` separated |
| `--ligand-ccd` | str | None | Target ligand CCD code (e.g., `"SAM"`) |
| `--ligand-smiles` | str | None | Target ligand SMILES string |
| `--nucleic-seq` | str | None | Target DNA/RNA sequence |
| `--nucleic-type` | str | `"dna"` | `"dna"` or `"rna"` |

#### Template Structure

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--template-path` | str | None | Local PDB/CIF file path |
| `--template-chain-id` | str | None | Comma-separated chain IDs from template |

#### Binder Configuration

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--seq` | str | None | Starting binder sequence (empty = random) |
| `--min-protein-length` | int | `100` | Minimum binder length |
| `--max-protein-length` | int | `150` | Maximum binder length |
| `--percent-x` | int | `50` | % of "X" (unknown) in initial sequence |
| `--cyclic` | flag | `False` | Design cyclic peptide |

#### Design Parameters

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--num-designs` | int | `5` | Number of independent design runs |
| `--num-cycles` | int | `7` | Fold→design iterations per run |
| `--contact-residues` | str | None | Hotspot residues (e.g., `"10,20,30"` or `"10,20\|5,15"` for multi-chain) |
| `--temperature` | float | `0.1` | MPNN sampling temperature |
| `--omit-aa` | str | `"C"` | Amino acids to exclude |
| `--alanine-bias` | flag | `True` | Penalize alanine (prevents poly-A) |
| `--high-iptm-threshold` | float | `0.7` | Min ipTM to save design |
| `--high-plddt-threshold` | float | `0.7` | Min pLDDT to save design |

#### Execution Control

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--gpu` | str | `"H100"` | GPU type (see below) |
| `--max-concurrent` | int | `0` | Max parallel GPUs (0 = unlimited) |
| `--no-stream` | flag | `False` | Disable real-time result streaming |
| `--sync-interval` | float | `5.0` | Sync polling interval (seconds) |

#### Available GPU Types

| GPU | VRAM | Cost/hour | Notes |
|-----|------|-----------|-------|
| `T4` | 16GB | $0.59 | Budget option |
| `L4` | 24GB | $0.80 | Good value |
| `A10G` | 24GB | $1.10 | |
| `L40S` | 48GB | $1.95 | |
| `A100-40GB` | 40GB | $2.10 | |
| `A100-80GB` | 80GB | $2.50 | |
| `H100` | 80GB | $3.95 | **Recommended** |

### Parallelization & Concurrency

Modal automatically parallelizes design runs across multiple GPUs:

```bash
# Run 16 designs in parallel (uses all available GPUs)
modal run modal_protein_hunter.py::run_pipeline \
    --num-designs 16 \
    --gpu H100

# Limit to 8 concurrent GPUs (runs in 2 batches)
modal run modal_protein_hunter.py::run_pipeline \
    --num-designs 16 \
    --max-concurrent 8 \
    --gpu H100
```

**How it works:**
- Each design run executes as an independent Modal container
- `--max-concurrent 0` (default): All designs run simultaneously
- `--max-concurrent N`: Designs run in batches of N

### Template Structures on Modal

Provide a local PDB file as a template structure. The file is automatically uploaded to Modal containers:

```bash
modal run modal_protein_hunter.py::run_pipeline \
    --name "pMHC_binder" \
    --target-seq "MHC_SEQ:BETA2M_SEQ:PEPTIDE_SEQ" \
    --template-path "/path/to/structure.pdb" \
    --template-chain-id "A,B,C" \
    --contact-residues "||1,2,3,4,5,6,7,8,9" \
    --num-designs 10 \
    --gpu H100
```

**Template chain mapping:**
- `--template-chain-id "A,B,C"` maps:
  - Template chain A → Target chain B (first sequence)
  - Template chain B → Target chain C (second sequence)
  - Template chain C → Target chain D (third sequence)

### Modal Examples

#### Example 1: Basic Protein Binder

```bash
modal run modal_protein_hunter.py::run_pipeline \
    --name "PDL1_binder" \
    --target-seq "AFTVTVPKDLYVVEYGSNMTIECKFPVEKQLDLAALIVYWEMEDKNIIQFVHGEEDLKVQHSSYRQRARLLKDQLSLGNAALQITDVKLQDAGVYRCMISYGGADYKRITVKVNAPYAAALE" \
    --num-designs 10 \
    --num-cycles 7 \
    --min-protein-length 90 \
    --max-protein-length 150 \
    --alanine-bias \
    --high-iptm-threshold 0.7 \
    --gpu H100 \
    --output-dir ./PDL1_results
```

#### Example 2: Hotspot-Directed Design

```bash
modal run modal_protein_hunter.py::run_pipeline \
    --name "PDL1_hotspot" \
    --target-seq "AFTVTVPKDLYVVEYGSNMTIECKFPVEKQLDLAALIVYWEMEDKNIIQFVHGEEDLKVQHSSYRQRARLLKDQLSLGNAALQITDVKLQDAGVYRCMISYGGADYKRITVKVNAPYAAALE" \
    --contact-residues "54,56,66,115,121" \
    --num-designs 10 \
    --num-cycles 7 \
    --high-iptm-threshold 0.7 \
    --gpu H100
```

#### Example 3: pMHC Binder with Template

```bash
modal run modal_protein_hunter.py::run_pipeline \
    --name "pMHC_TCRm" \
    --target-seq "MHC_ALPHA_SEQ:BETA2M_SEQ:PEPTIDE_SEQ" \
    --template-path "./my_pmhc_structure.pdb" \
    --template-chain-id "A,B,C" \
    --contact-residues "||1,2,3,4,5,6,7,8,9" \
    --num-designs 16 \
    --num-cycles 5 \
    --max-concurrent 8 \
    --min-protein-length 60 \
    --max-protein-length 120 \
    --high-iptm-threshold 0.8 \
    --gpu H100 \
    --output-dir ./pMHC_results
```

#### Example 4: Small Molecule Binder

```bash
modal run modal_protein_hunter.py::run_pipeline \
    --name "SAM_binder" \
    --ligand-ccd "SAM" \
    --num-designs 5 \
    --num-cycles 7 \
    --min-protein-length 130 \
    --max-protein-length 150 \
    --high-iptm-threshold 0.7 \
    --gpu H100
```

#### Example 5: Budget Run on L4 GPUs

```bash
modal run modal_protein_hunter.py::run_pipeline \
    --name "budget_design" \
    --target-seq "YOUR_TARGET" \
    --num-designs 5 \
    --num-cycles 5 \
    --gpu L4 \
    --max-concurrent 4
```

### Modal Output Files

Results are saved to your local filesystem:

```
output_dir/
├── summary_all_runs.csv          # All designs with per-cycle metrics
├── summary_high_iptm.csv         # High-quality designs only
├── high_iptm_yaml/               # YAML input files for high-quality designs
├── high_iptm_pdb/                # PDB structures for high-quality designs
├── best_structures/              # Best structure from each design run
└── design_N/                     # Per-design streaming results
    ├── cycle_0.pdb
    ├── cycle_0_metrics.json
    ├── cycle_1.pdb
    └── ...
```

### Modal Cost Estimation

| Scenario | GPUs | Time | Est. Cost |
|----------|------|------|-----------|
| 5 designs × 7 cycles (H100) | 5 | ~20 min | ~$6 |
| 16 designs × 5 cycles (H100, 8 concurrent) | 8 | ~40 min | ~$20 |
| 10 designs × 7 cycles (L4) | 10 | ~45 min | ~$6 |

---

## Citation

```bibtex
@article{cho2025protein,
  title={Protein Hunter: exploiting structure hallucination within diffusion for protein design},
  author={Cho, Yehlin and Rangel, Griffin and Bhardwaj, Gaurav and Ovchinnikov, Sergey},
  journal={bioRxiv},
  year={2025}
}
```

