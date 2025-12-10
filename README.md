# Protein Hunter (Cloud Edition)

> **Fork Notice**: This is a fork of [Protein Hunter](https://github.com/yehlincho/Protein-Hunter) 
> that concentrates on the Boltz pathway with enhanced Modal cloud support and improved output organization.
> For the original Chai+Boltz implementation, see [UPSTREAM_README.md](./UPSTREAM_README.md).

A protein binder design pipeline using Boltz structure prediction and LigandMPNN sequence design.

## Why This Fork?

1. **BindCraft-style Design Cycles** — Restructured workflow with resumable execution and early stopping
2. **Serverless Compute Ready** — Full Modal cloud compatibility with massive parallelization and real-time streaming
3. **Open-Source Future** — Working toward replacing AF3/PyRosetta with open-source alternatives

## What's Different in This Fork

| Feature | Upstream | This Fork |
|---------|----------|-----------|
| **Boltz Support** | ✅ | ✅ (primary focus) |
| **Chai Support** | ✅ | ❌ (legacy, not maintained in this fork) |
| **Modal Cloud Pipeline** | ❌ | ✅ Parallelized design runs |
| **ipSAE Scoring** | ❌ | ✅ Interface pSAE metric |
| **Output Organization** | Complex folder structure | Streamlined: `designs/`, `best_designs/`, `accepted_designs/`, etc. |
| **Resumable Execution** | ❌ | ✅ Resume interrupted jobs, stop at N accepted |

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
- [Resumable Execution](#resumable-execution)
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

Protein Hunter uses iterative **structure-sequence cycling** with **diffusion hallucination** to design protein binders. Based on the discovery that AF3-style diffusion models (Boltz, Chai, AF3) can hallucinate well-folded structures from out-of-distribution inputs like unknown "X" tokens, the method co-optimizes sequence and structure through repeated cycles.

```
┌──────────────────────────────────────────────────────────────┐
│  1. Initialize binder with all-X (unknown) tokens            │
│                           ↓                                  │
│  2. Diffusion hallucination: Boltz predicts a folded         │
│     structure despite the undefined sequence                 │
│                           ↓                                  │
│  3. Sequence design: LigandMPNN designs a sequence           │
│     compatible with the hallucinated backbone                │
│                           ↓                                  │
│  4. Structure refinement: Re-predict with designed sequence  │
│                           ↓                                  │
│  5. Repeat steps 3-4 for N cycles (structure improves,       │
│     alanine bias decreases, foldability increases)           │
│                           ↓                                  │
│  6. Output: Co-optimized binder sequence + structure         │
└──────────────────────────────────────────────────────────────┘
```

> **Key insight**: Unlike gradient-based methods (BindCraft, BoltzDesign) that can be slow and get stuck in local minima, or single-pass diffusion methods (RFdiffusion) where structure and sequence are optimized separately, Protein Hunter jointly refines both through zero-shot prediction cycles. This achieves high in silico success rates across diverse targets including proteins, cyclic peptides, small molecules, DNA, and RNA.

**Reference**: [Protein Hunter: exploiting structure hallucination within diffusion for protein design](https://www.biorxiv.org/content/10.1101/2025.10.10.681530v2) (Cho et al., 2025)

**Key terminology:**
- **Binder (Chain A)**: The protein being designed (initialized with X tokens)
- **Target (Chain B, C, ...)**: What you want to bind (protein, ligand, DNA/RNA)
- **Design run**: One complete optimization trajectory from initialization to final output
- **Cycle** (`--num_cycles`): One iteration of structure prediction → sequence design
- **Recycle** (`--recycling_steps`): Internal refinement passes within Boltz's structure prediction
- **Diffusion hallucination**: The model's ability to generate well-folded structures from undefined (X token) sequences

**Cycles vs Recycles:**
```
Design Run (1 of num_designs)
│
├── Cycle 0: Diffusion hallucination (X tokens → initial structure)
│   └── [recycle 1] → [recycle 2] → [recycle 3] → hallucinated structure
│
├── Cycle 1: Sequence design → structure refinement
│   └── [recycle 1] → [recycle 2] → [recycle 3] → improved structure
│
├── Cycle 2: Sequence design → structure refinement
│   └── ... (alanine content decreases, foldability improves)
│
└── ... (num_cycles iterations)
```

---

## Installation

```bash
git clone https://github.com/cytokineking/Protein-Hunter
cd Protein-Hunter
chmod +x setup.sh
./setup.sh
```

For detailed local setup including AF3 Docker configuration, see [LOCAL_SETUP_GUIDE.md](./LOCAL_SETUP_GUIDE.md).

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
    --high_iptm_threshold 0.7 \
    --gpu_id 0
```

> **Note**: `--alanine_bias` defaults to `True` in this fork. Use `--alanine_bias false` to disable.

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
    --high_iptm_threshold 0.7 \
    --gpu_id 0
```

**Key parameters for de novo design:**

| Parameter | Description | Recommended |
|-----------|-------------|-------------|
| `--percent_X` | % of "unknown" residues in initial sequence | 50-100 |
| `--min_protein_length` | Minimum binder length | 60-80 |
| `--max_protein_length` | Maximum binder length | 120-150 |
| `--num_designs` | Independent design attempts | 100+ for a full run (required) |
| `--num_cycles` | Optimization iterations per design | 7 |

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
    --template_path "7KPL" \
    --template_cif_chain_id "A" \
    --msa_mode single \
    ...
```

| Parameter | Description |
|-----------|-------------|
| `--protein_seqs` | Target sequence(s) — optional if template contains full sequence |
| `--template_path` | PDB code, file path, or AlphaFold ID |
| `--template_cif_chain_id` | Which chain in template to use |
| `--msa_mode single` | Often used with templates (MSA optional) |

**Sequence auto-extraction:** If you omit `--protein_seqs`, sequences are automatically extracted from the template structure. This works when the template contains the complete sequence you want to target.

**When to provide `--protein_seqs`:**
- Template has missing residues resulting in gaps (e.g., disordered termini not in PDB)
- You want to model additional regions beyond what's in the template
- Template sequence doesn't match the exact target you want

When provided, Boltz uses template coordinates where available and predicts any regions not covered by the template.

**Template path options:**

| Format | Example | Description |
|--------|---------|-------------|
| PDB code | `"7KPL"` | Auto-downloads from RCSB |
| Local file | `"./structures/target.cif"` | Your own file (.pdb or .cif) |
| AlphaFold ID | `"P12345"` | Auto-downloads from AlphaFold DB |

**Multi-chain templates:**

```bash
--template_path "7KPL:7KPL" \
--template_cif_chain_id "A:B"
# Sequences auto-extracted, or provide explicitly:
--protein_seqs "SEQ_B:SEQ_C"
```

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

### Author vs Canonical Numbering

By default, hotspot residue numbers use **canonical (1-indexed)** numbering. If your template has non-standard numbering (e.g., starting at residue 7), use `--use_auth_numbering`:

```bash
# Template has residues numbered 7-16 (author numbering)
--contact_residues "|7,8,9,10,11,12,13,14,15,16" \
--use_auth_numbering
```

The pipeline displays a **target sequence analysis** at startup showing the mapping:

```
═══════════════════════════════════════════════════════════════════════════════
🎯 TARGET SEQUENCE ANALYSIS
═══════════════════════════════════════════════════════════════════════════════
Chain B - 10 residues
├─ Source: template (/path/to/structure.pdb)
├─ Auth numbering: 7-16
└─ Hotspots: auth [7..16] → canonical [1..10]
    Hotspot residues (10 total): VVVGAVGVGK
    Range: canonical 1-10, auth 7-16

Sequence (hotspots in RED):
Canon Auth                                                Canon Auth
    1     7  VVVGAVGVGK     10    16
             **********
═══════════════════════════════════════════════════════════════════════════════
```

This visualization helps verify your hotspots map correctly to the intended residues.

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
| `--num_designs` | int | — | Total designs to generate (at least one of `--num_designs` or `--num_accepted` required) |
| `--num_accepted` | int | — | Stop after N designs pass filters (requires `--use_alphafold3_validation`) |
| `--num_cycles` | int | `5` | Fold→design iterations per run |
| `--num_gpus` | int | `1` | Number of GPUs for parallel design (multi-GPU mode) |
| `--mode` | str | `"binder"` | `"binder"` or `"unconditional"` |

**Stopping conditions:** You must specify at least one of `--num_designs` or `--num_accepted`. If both are provided, the pipeline stops when *either* target is reached (OR logic). See [Resumable Execution](#resumable-execution) for details.

### Multi-GPU Parallelization (Local)

Run designs in parallel across multiple GPUs on a single machine:

```bash
python boltz_ph/design.py \
    --name parallel_design \
    --protein_seqs "TARGET_SEQUENCE" \
    --num_designs 40 \
    --num_gpus 8 \
    --use_alphafold3_validation
```

**How it works:**
- Spawns one worker process per GPU
- Each worker handles complete designs (Boltz → LigandMPNN → AF3 validation)
- Centralized job queue with automatic load balancing
- Resume-aware: picks up from existing progress

**Output:**
- Per-worker logs: `results_{name}/worker_gpu*.log`
- Real-time progress in terminal (design completion, validation results)
- All outputs merged into standard folder structure

**When to use:**
- Multi-GPU servers (8×A100, 8×H100, etc.)
- Large design campaigns (100+ designs)
- When you want parallelization without Modal overhead

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
| `--use_auth_numbering` | flag | `False` | Use PDB "author" residue numbers for hotspots |
| `--contact_cutoff` | float | `15.0` | Contact distance threshold (Å) |
| `--max_contact_filter_retries` | int | `6` | Retries if contacts unsatisfied |
| `--no_contact_filter` | flag | `False` | Disable contact filtering |

### Sequence Design (MPNN) Settings

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--temperature` | float | `0.1` | MPNN sampling temperature |
| `--omit_AA` | str | `"C"` | Amino acids to exclude |
| `--alanine_bias` | str | **`True`** | Penalize alanine (use `false` to disable) |
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
| `--recycling_steps` | int | `3` | Model recycling passes per prediction |

**Note on `--recycling_steps`:**
This controls the number of internal refinement passes within each structure prediction. More recycles = more refined prediction per cycle, but slower. This is different from `--num_cycles` which controls the outer design loop (fold → redesign → fold → ...). Default of 3 is standard for AlphaFold-style models.

### Output Settings

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--save_dir` | str | `""` | Custom output directory |
| `--plot` | flag | `False` | Generate optimization plots |

### AlphaFold3 Validation (Optional)

AF3 validation provides orthogonal structure prediction to verify Boltz designs hold up when re-predicted by a different model. This is crucial because self-consistency (Boltz predicting its own designs) can be optimistically biased.

#### What It Does

When enabled, the pipeline:

1. **HOLO prediction**: Re-predicts the binder+target complex using AlphaFold3
2. **APO prediction** (protein targets): Predicts the binder *alone* to verify it folds independently
3. **PyRosetta scoring** (protein targets): Calculates interface properties and filters designs

#### Why Use It

- **Cross-validation**: Different model architecture reduces overfitting to Boltz's biases
- **Confidence calibration**: AF3's confidence metrics (ipTM, pLDDT) are well-validated experimentally
- **Physics-based filtering**: PyRosetta catches designs with suboptimal binding energetics, shape complementarity, hydrogen-bonding networks, and/or surface hydrophobicity 

#### CLI Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--use_alphafold3_validation` | flag | `False` | Enable AF3 cross-validation pipeline |
| `--alphafold_dir` | str | `"~/alphafold3"` | AF3 installation path (local only) |
| `--use_msa_for_af3` | bool | **`True`** | Reuse MSAs from design phase for AF3 |

#### APO Stability Check

For protein targets, the pipeline predicts the binder structure *without* the target present. This checks:

1. **Expression viability** — Unfolded binders are difficult to express and purify
2. **Binding thermodynamics** — Large conformational changes upon binding incur an **entropy penalty** (ΔS < 0), making binding less thermodynamically favorable. Designs with low `apo_holo_rmsd` bind via "lock-and-key" rather than "induced fit"

#### PyRosetta Metrics

PyRosetta runs automatically for protein targets when AF3 validation is enabled:

| Metric | Good Value | Description |
|--------|-----------|-------------|
| `surface_hydrophobicity` | < 0.30 | Surface hydrophobic fraction. Lower = better expressibility, less non-specific "stickiness" |
| `interface_sc` | > 0.65 | Shape complementarity (0-1). Higher = tighter geometric fit at interface |
| `interface_dG` | < -15 kcal/mol | Binding free energy. More negative = stronger binding |
| `interface_nres` | > 12 | Number of interface residues. More contacts = larger binding surface |
| `interface_delta_unsat_hbonds` | < 2 | Buried unsatisfied H-bonds (BUNS). Lower = better H-bond network |
| `apo_holo_rmsd` | < 2.0 Å | RMSD between bound and unbound binder. Lower = pre-organized for binding (no entropy penalty) |

#### Acceptance Criteria

Designs are accepted if they pass ALL of the following (protein targets):

**AF3 confidence:**
- `af3_iptm >= 0.7` (interface confidence)
- `af3_plddt >= 80` (structure confidence)

**PyRosetta filters:**
- `surface_hydrophobicity < 0.35` (expressibility, reduces non-specific binding)
- `interface_sc > 0.55` (shape complementarity — tight geometric fit)
- `interface_dG < 0` (favorable binding energy)
- `interface_nres > 7` (sufficient interface contacts)
- `interface_delta_unsat_hbonds < 4` (good H-bond satisfaction)
- `apo_holo_rmsd < 3.5` (maintains fold without target, minimal entropy penalty)

#### Setup Requirements

**Local pipeline**: Requires local AF3 and PyRosetta installations. See [LOCAL_SETUP_GUIDE.md](LOCAL_SETUP_GUIDE.md) for details.

**Modal pipeline**: Uses containerized AF3 (`docker.io/aaronr24/alphafold3-modal:latest`) and PyRosetta — no local installation needed.

#### Example: Full Validation Pipeline

```bash
# Local — run until 5 designs pass all filters (or 100 total attempts)
python boltz_ph/design.py \
    --name PDL1_validated \
    --protein_seqs "AFTVTVPK..." \
    --num_designs 100 \
    --num_accepted 5 \
    --use_alphafold3_validation \
    --alphafold_dir ~/alphafold3

# Modal (PyRosetta runs automatically for protein targets)
modal run modal_boltz_ph_cli.py::run_pipeline \
    --name PDL1_validated \
    --protein-seqs "AFTVTVPK..." \
    --num-designs 5 \
    --use-alphafold3-validation
```

---

## Output Files

Results are saved to `./results_{name}/`:

### Basic Output (Boltz Design Only)

```
results_my_design/
├── designs/                         # ALL cycles from all design runs
│   ├── my_design_d0_c0.pdb          # Design 0, Cycle 0
│   ├── my_design_d0_c1.pdb          # Design 0, Cycle 1
│   ├── my_design_d0_c2.pdb          # ...
│   ├── my_design_d1_c0.pdb          # Design 1, Cycle 0
│   ├── my_design_d1_c1.pdb          # ...
│   └── design_stats.csv             # Full metrics for ALL cycles
└── best_designs/                    # Best cycle per design run
    ├── my_design_d0_c2.pdb          # Best cycle from design 0
    ├── my_design_d1_c3.pdb          # Best cycle from design 1
    └── best_designs.csv             # Summary of best designs only
```

### Full Output (With AF3 Validation)

When using `--use_alphafold3_validation` (local) or `--use-alphafold3-validation` (Modal):

```
results_my_design/
├── designs/                         # All design cycles
│   ├── design_stats.csv
│   └── {name}_d{X}_c{Y}.pdb
├── best_designs/                    # Best cycle per design
│   ├── best_designs.csv
│   └── {name}_d{X}_c{Y}.pdb
├── af3_validation/                  # AlphaFold3 predictions
│   ├── af3_results.csv
│   └── *_af3.cif
├── accepted_designs/                # Passed all filters (protein targets)
│   ├── accepted_stats.csv
│   └── *_relaxed.pdb
└── rejected/                        # Failed filters
    ├── rejected_stats.csv
    └── *_relaxed.pdb
```

> **Note**: PyRosetta scoring runs automatically for protein targets when AF3 validation is enabled.

**File naming convention:** `{name}_d{design_num}_c{cycle}.pdb`

**Key files:**

| File | Description |
|------|-------------|
| `designs/design_stats.csv` | Complete metrics for every cycle of every design |
| `designs/*.pdb` | All PDB structures from all cycles |
| `best_designs/best_designs.csv` | Summary of best cycle per design (highest ipTM with ≤20% alanine) |
| `best_designs/*.pdb` | Best structure from each design run |

**CSV columns in `design_stats.csv`:**

| Column | Description |
|--------|-------------|
| `design_id` | Unique identifier (`{name}_d{N}_c{M}`) |
| `design_num` | Design run index |
| `cycle` | Cycle number within design run |
| `binder_sequence` | Designed binder sequence |
| `binder_length` | Length of binder |
| `cyclic` | Whether cyclic topology was used |
| `boltz_iptm` | Interface pTM score (from Boltz) |
| `boltz_ipsae` | Interface pSAE score (from Boltz, 0-1, higher is better) |
| `boltz_plddt` | Complex pLDDT (from Boltz) |
| `boltz_iplddt` | Interface pLDDT (from Boltz) |
| `alanine_count` | Number of alanines |
| `alanine_pct` | Percentage alanine |
| `target_seqs` | Target sequence(s) used |
| `contact_residues` | Hotspot residues (if specified) |
| `msa_mode` | MSA mode used |
| `timestamp` | When the cycle completed |

**Additional columns in `accepted_stats.csv` / `rejected_stats.csv`:**

| Column | Description |
|--------|-------------|
| `af3_iptm` | Interface pTM from AF3 validation |
| `af3_ipsae` | Interface pSAE from AF3 (calculated from PAE matrix) |
| `af3_ptm` | Global pTM from AF3 |
| `af3_plddt` | Average pLDDT from AF3 |
| `interface_dG` | Binding free energy (kcal/mol) |
| `interface_sc` | Shape complementarity (0-1) |
| `interface_hbonds` | Number of interface hydrogen bonds |
| `apo_holo_rmsd` | RMSD between bound and unbound binder |
| `accepted` | Whether design passed all filters |
| `rejection_reason` | Why design was rejected (if applicable) |

---

## Resumable Execution

The local pipeline supports **resumable jobs** and **dual stopping conditions**, enabling long-running design campaigns that can survive interruptions.

### How It Works

The pipeline saves results incrementally after each design completes. Progress is tracked by counting files on disk:

- **Completed designs**: Count of `*.pdb` files in `best_designs/`
- **Accepted designs**: Count of `*_relaxed.pdb` files in `accepted_designs/`

To resume an interrupted job, simply re-run the same command — the pipeline detects existing progress and continues from where it left off.

### Stopping Conditions

You must specify at least one stopping condition:

| Flag | Description |
|------|-------------|
| `--num_designs N` | Stop after N total designs generated |
| `--num_accepted N` | Stop after N designs pass all filters |

**Rules:**
1. At least one required
2. Both allowed — first condition met triggers exit (OR logic)
3. `--num_accepted` requires `--use_alphafold3_validation`
4. `--num_accepted` alone prints a warning (no upper limit on attempts)

### Examples

```bash
# Generate exactly 50 designs (classic mode, resumable)
python boltz_ph/design.py \
    --name PDL1 \
    --num_designs 50 \
    --protein_seqs "..."

# Generate until 10 pass filters (warning: no upper limit)
python boltz_ph/design.py \
    --name PDL1 \
    --num_accepted 10 \
    --use_alphafold3_validation \
    --protein_seqs "..."

# Recommended: Generate until 10 accepted OR 500 total (whichever comes first)
python boltz_ph/design.py \
    --name PDL1 \
    --num_designs 500 \
    --num_accepted 10 \
    --use_alphafold3_validation \
    --protein_seqs "..."

# Resume a crashed job (just re-run the same command)
python boltz_ph/design.py \
    --name PDL1 \
    --num_designs 50 \
    --protein_seqs "..."
# → "Found 32 existing designs. Resuming from design 32..."
```

### Progress Display

During execution, the pipeline shows real-time progress:

```
============================================================
Starting Design 47 | Progress: 47/500 designs, 7/10 accepted
============================================================
  Best cycle: 5 (boltz_iptm=0.82)
  AF3: iptm=0.71, ipsae=0.65
  PyRosetta: dG=-12.3, SC=0.72
  → ACCEPTED

...

✓ Target reached: 10/10 accepted designs
Pipeline complete. Generated 63 total designs to obtain 10 accepted.
```

### Design-by-Design Execution

Unlike batch processing, the local pipeline now processes each design through the **full validation pipeline** before starting the next:

```
For each design:
    Boltz cycles → Best → AF3 holo → AF3 apo → PyRosetta → Save → Next
```

This enables:
- **Early stopping**: Stop as soon as you have enough good designs
- **Crash recovery**: All completed designs are saved, even if the job dies mid-run
- **Real-time filtering**: Know immediately which designs pass/fail

---

## Metrics Reference

### ipTM (Interface pTM)

Interface pTM measures the predicted accuracy of interface residue positioning. Range: 0-1 (higher is better). This is the primary metric for design quality.

### ipSAE (Interface predicted Structural Alignment Error)

ipSAE measures the confidence of interface residue positioning between binder and target. Based on [Dunbrack et al. (2025)](https://www.biorxiv.org/content/10.1101/2025.02.10.637595v1).

**Key characteristics:**
- **Range**: 0-1 (higher is better)
- **Calculation**: Per-residue max of binder→target and target→binder PAE scores
- **Multi-chain handling**: Only scores binder↔target interfaces, excludes target-target contacts
- **Nucleic acid support**: Automatically adjusts d0 calculation for DNA/RNA targets

**Implementation**: `utils/ipsae_utils.py`

### pLDDT (predicted Local Distance Difference Test)

Per-residue confidence score from Boltz. Range: 0-1 (higher is better). Values >0.7 indicate confident predictions.

### PyRosetta Metrics (Validation Only)

When AF3 validation is enabled for protein targets, PyRosetta calculates interface energetics. See the [AlphaFold3 Validation](#alphafold3-validation-optional) section for full details.

| Metric | Good Value | Description |
|--------|-----------|-------------|
| `surface_hydrophobicity` | < 0.30 | Hydrophobic surface fraction — lower = better expressibility |
| `interface_sc` | > 0.65 | Shape complementarity — higher = tighter geometric fit |
| `interface_dG` | < -15 kcal/mol | Binding free energy — more negative = stronger binding |
| `interface_nres` | > 12 | Interface residue count — more = larger binding surface |
| `interface_delta_unsat_hbonds` | < 2 | Buried unsatisfied H-bonds — lower = better H-bond network |
| `apo_holo_rmsd` | < 2.0 Å | Bound vs unbound RMSD — lower = pre-organized (no entropy penalty) |

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
    --high_iptm_threshold 0.7 \
    --gpu_id 0
```

### Example 8: Multi-GPU Parallel Design with Template

```bash
python boltz_ph/design.py \
    --name parallel_pMHC \
    --template_path "./pmhc_structure.pdb" \
    --template_cif_chain_id "A,B" \
    --contact_residues "|7,8,9,10,11,12,13,14,15,16" \
    --use_auth_numbering \
    --num_designs 40 \
    --num_cycles 5 \
    --num_gpus 8 \
    --use_alphafold3_validation \
    --alphafold_dir ~/alphafold3
```

---

## Tips & Best Practices

1. **Start with lower thresholds** (`--high_iptm_threshold 0.6-0.7`) for initial exploration, then increase for production runs.

2. **Alanine bias is ON by default** to avoid poly-alanine sequences. Use `--alanine_bias false` only if you specifically want to allow high alanine content.

3. **More designs > more cycles**: Running 10 designs × 5 cycles often gives better diversity than 3 designs × 15 cycles.

4. **For difficult targets**, increase `--max_contact_filter_retries` and use hotspots to guide the design.

5. **Template mode** is recommended when you have high-confidence structures; sequence-only mode is better for exploring conformational flexibility.

6. **Use dual stopping conditions** for production runs: `--num_designs 500 --num_accepted 20` stops when you have 20 good designs OR after 500 attempts, whichever comes first.

7. **Jobs are resumable**: If a job crashes, just re-run the same command. The pipeline detects existing progress and continues from where it left off.

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
modal run modal_boltz_ph_cli.py::init_cache
```

This downloads Boltz2 weights, CCD data, and LigandMPNN models to a persistent Modal volume.

### Modal Quick Start

```bash
# Basic design run (alanine_bias is ON by default)
modal run modal_boltz_ph_cli.py::run_pipeline \
    --name "my_design" \
    --protein-seqs "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVKALPDAQFEVVHSLAKWKRQQIAAALEHHHHHH" \
    --gpu H100

# Quick test run (fewer designs)
modal run modal_boltz_ph_cli.py::run_pipeline \
    --name "quick_test" \
    --protein-seqs "YOUR_TARGET_SEQUENCE" \
    --num-designs 5 \
    --num-cycles 5 \
    --gpu H100

# Disable alanine bias (rare)
modal run modal_boltz_ph_cli.py::run_pipeline \
    --name "test" \
    --protein-seqs "YOUR_TARGET_SEQUENCE" \
    --alanine-bias=false \
    --gpu H100

# List available GPUs
modal run modal_boltz_ph_cli.py::list_gpus
```

### Modal CLI Arguments

The Modal CLI uses the same arguments as the local pipeline with these differences:

**Syntax:** Use kebab-case (`--protein-seqs`) instead of underscores (`--protein_seqs`)

**Boolean flags:** Use `--flag=true` or `--flag=false` syntax (e.g., `--alanine-bias=false`)

#### Modal-Specific Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--gpu` | str | `"H100"` | GPU type for Boltz design (see below) |
| `--max-concurrent` | int | `0` | Max parallel GPUs (0 = unlimited) |
| `--no-stream` | str | `"false"` | Disable real-time result streaming |
| `--sync-interval` | float | `5.0` | Sync polling interval (seconds) |
| `--output-dir` | str | `"./results_{name}"` | Local output directory |
| `--use-alphafold3-validation` | str | `"false"` | Enable AF3 + PyRosetta validation |
| `--af3-gpu` | str | `"A100-80GB"` | GPU type for AF3 validation |
| `--use-msa-for-af3` | str | `"true"` | Reuse MSAs from design phase for AF3 |

#### Available GPU Types

| GPU | VRAM | Cost/hour |
|-----|------|-----------|
| `T4` | 16GB | $0.59 |
| `L4` | 24GB | $0.80 |
| `A10G` | 24GB | $1.10 |
| `L40S` | 48GB | $1.95 |
| `A100-40GB` | 40GB | $2.10 |
| `A100-80GB` | 80GB | $2.50 |
| `H100` | 80GB | $3.95 |

### Parallelization & Concurrency

Modal automatically parallelizes design runs across multiple GPUs:

```bash
# Run 16 designs in parallel (uses all available GPUs)
modal run modal_boltz_ph_cli.py::run_pipeline \
    --num-designs 16 \
    --gpu H100

# Limit to 8 concurrent GPUs (runs in 2 batches)
modal run modal_boltz_ph_cli.py::run_pipeline \
    --num-designs 16 \
    --max-concurrent 8 \
    --gpu H100
```

**How it works:**
- Each design run executes as an independent Modal container
- `--max-concurrent 0` (default): All designs run simultaneously
- `--max-concurrent N`: Designs run in batches of N

### Template Structures on Modal

Provide a local PDB file as a template structure. The file is automatically uploaded to Modal containers.

**Sequence auto-extraction:** If you omit `--protein-seqs`, sequences are automatically extracted from the template chains. Provide `--protein-seqs` only if your template has missing residues resulting in gaps or you want to model additional residues beyond the template.

```bash
# With explicit sequences
modal run modal_boltz_ph_cli.py::run_pipeline \
    --name "pMHC_binder" \
    --protein-seqs "MHC_SEQ:PEPTIDE_SEQ" \
    --template-path "/path/to/structure.pdb" \
    --template-cif-chain-id "A,B" \
    --contact-residues "|7,8,9,10,11" \
    --use-auth-numbering=true \
    --num-designs 10 \
    --gpu H100

# Auto-extract sequences from template (no --protein-seqs needed)
modal run modal_boltz_ph_cli.py::run_pipeline \
    --name "pMHC_binder" \
    --template-path "/path/to/structure.pdb" \
    --template-cif-chain-id "A,B" \
    --contact-residues "|7,8,9,10,11" \
    --use-auth-numbering=true \
    --num-designs 10 \
    --gpu H100
```

**Template chain mapping:**
- `--template-cif-chain-id "A,B"` maps:
  - Template chain A → Target chain B (first sequence)
  - Template chain B → Target chain C (second sequence)

**Hotspot numbering:** Use `--use-auth-numbering=true` when your hotspot residue numbers match the PDB "author" numbering rather than 1-indexed canonical numbering.

### Modal Examples

#### Example 1: Basic Protein Binder

```bash
modal run modal_boltz_ph_cli.py::run_pipeline \
    --name "PDL1_binder" \
    --protein-seqs "AFTVTVPKDLYVVEYGSNMTIECKFPVEKQLDLAALIVYWEMEDKNIIQFVHGEEDLKVQHSSYRQRARLLKDQLSLGNAALQITDVKLQDAGVYRCMISYGGADYKRITVKVNAPYAAALE" \
    --num-designs 10 \
    --num-cycles 5 \
    --min-protein-length 90 \
    --max-protein-length 150 \
    --high-iptm-threshold 0.7 \
    --gpu H100 \
    --output-dir ./PDL1_results
```

#### Example 2: Hotspot-Directed Design

```bash
modal run modal_boltz_ph_cli.py::run_pipeline \
    --name "PDL1_hotspot" \
    --protein-seqs "AFTVTVPKDLYVVEYGSNMTIECKFPVEKQLDLAALIVYWEMEDKNIIQFVHGEEDLKVQHSSYRQRARLLKDQLSLGNAALQITDVKLQDAGVYRCMISYGGADYKRITVKVNAPYAAALE" \
    --contact-residues "54,56,66,115,121" \
    --num-designs 10 \
    --num-cycles 5 \
    --high-iptm-threshold 0.7 \
    --gpu H100
```

#### Example 3: pMHC Binder with Template

```bash
# Sequences auto-extracted from template; hotspots use PDB numbering
modal run modal_boltz_ph_cli.py::run_pipeline \
    --name "pMHC_TCRm" \
    --template-path "./my_pmhc_structure.pdb" \
    --template-cif-chain-id "A,B" \
    --contact-residues "|7,8,9,10,11,12,13,14,15,16" \
    --use-auth-numbering=true \
    --num-designs 16 \
    --num-cycles 5 \
    --max-concurrent 8 \
    --min-protein-length 60 \
    --max-protein-length 120 \
    --high-iptm-threshold 0.8 \
    --use-alphafold3-validation=true \
    --gpu H100 \
    --output-dir ./pMHC_results
```

#### Example 4: Small Molecule Binder

```bash
modal run modal_boltz_ph_cli.py::run_pipeline \
    --name "SAM_binder" \
    --ligand-ccd "SAM" \
    --num-designs 5 \
    --num-cycles 5 \
    --min-protein-length 130 \
    --max-protein-length 150 \
    --high-iptm-threshold 0.7 \
    --gpu H100
```


### Modal Output Files

Results are streamed to your local filesystem in real-time in the same structure as the local pipeline above. The folders are populated as each cycle completes across all parallel GPU workers. The .csv files are thread-safe and can be monitored during execution.



This runs `boltz_ph/pipeline.py` directly on a single Modal GPU, useful for:
- Testing local pipeline changes
- Debugging without the complexity of parallel execution
- Quick validation with minimal cycles
- Testing the full pipeline (Boltz → AF3 → PyRosetta) end-to-end

---

## Legacy Code

This fork contains code from the upstream repository that is **not actively maintained**:

### `legacy_notebooks/`

Colab notebooks from the upstream repository. These are preserved for reference but are **likely incompatible** with this fork due to CLI changes and new defaults. Use the examples in this README instead.

### `chai_ph/`

The Chai pathway from upstream. This fork focuses exclusively on Boltz with Modal cloud support. The Chai code has not been adapted for our changes. If you need Chai support, refer to the [upstream repository](https://github.com/yehlincho/Protein-Hunter).

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

