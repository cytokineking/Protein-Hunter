# Free Protein Hunter

> **Fork Notice**: This is a fork of [Protein Hunter](https://github.com/yehlincho/Protein-Hunter) 
> that concentrates on the Boltz pathway with enhanced Modal cloud support and improved output organization.
> For the original Chai+Boltz implementation, see [UPSTREAM_README.md](./UPSTREAM_README.md).

A protein binder design pipeline using Boltz structure prediction and LigandMPNN sequence design.

## Why This Fork?

1. **BindCraft-style Design Cycles** — Restructured workflow with resumable execution and early stopping
2. **Serverless Compute Ready** — Full Modal cloud compatibility with massive parallelization and real-time streaming
3. **Open-Source Scoring** — Optional PyRosetta-free scoring using OpenMM, FreeSASA, and sc-rs (adapted from [FreeBindCraft](https://github.com/cytokineking/FreeBindCraft))
4. **Open-Source Validation** — Protenix integration for fully open-source structure validation (no AF3 weights required)

## Fork Additions

This fork extends the upstream with the following capabilities:

| Addition | Description |
|----------|-------------|
| **Modal Cloud Pipeline** | Parallelized design runs with multi-GPU orchestration |
| **Open-Source Validation** | Protenix integration for structure validation (no AF3 weights required) |
| **Open-Source Scoring** | PyRosetta-free interface scoring (Modal only) |
| **ipSAE Scoring** | Interface pSAE metric for quality assessment |
| **Streamlined Output** | Organized folders: `designs/`, `best_designs/`, `accepted_designs/` |
| **Resumable Execution** | Resume interrupted jobs, stop at N accepted designs |

> **Note**: This fork focuses on the Boltz pathway. Chai support from upstream is not maintained here.

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
- [Open-Source Scoring (Modal)](#open-source-scoring-modal)

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
- **Cycle** (`--num-cycles`): One iteration of structure prediction → sequence design
- **Recycle** (`--recycling-steps`): Internal refinement passes within Boltz's structure prediction
- **Diffusion hallucination**: The model's ability to generate well-folded structures from undefined (X token) sequences

**Cycles vs Recycles:**
```
Design Run (1 of num-designs)
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
└── ... (num-cycles iterations)
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
    --protein-seqs "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVKALPDAQFEVVHSLAKWKRQQIAAALEHHHHHH" \
    --num-designs 3 \
    --num-cycles 7 \
    --min-protein-length 80 \
    --max-protein-length 120 \
    --msa-mode mmseqs \
    --high-iptm-threshold 0.7 \
    --gpu-id 0
```

> **Note**: `--alanine-bias` defaults to `True` in this fork. Use `--alanine-bias false` to disable.

---

## Design Modes

### 1. De Novo Design

**Purpose**: Design a completely new binder from scratch.

The binder starts as a random sequence (with optional "X" unknown residues) and gets optimized through multiple cycles.

```bash
python boltz_ph/design.py \
    --name de_novo_binder \
    --protein-seqs "YOUR_TARGET_SEQUENCE" \
    --num-designs 5 \
    --num-cycles 7 \
    --min-protein-length 90 \
    --max-protein-length 150 \
    --percent-x 50 \
    --msa-mode mmseqs \
    --high-iptm-threshold 0.7 \
    --gpu-id 0
```

**Key parameters for de novo design:**

| Parameter | Description | Recommended |
|-----------|-------------|-------------|
| `--percent-x` | % of "unknown" residues in initial sequence | 50-100 |
| `--min-protein-length` | Minimum binder length | 60-80 |
| `--max-protein-length` | Maximum binder length | 120-150 |
| `--num-designs` | Target valid designs (passed initial filters) | 100+ for a full run |
| `--num-cycles` | Optimization iterations per design | 7 |

---

### 2. Optimization (Refinement)

**Purpose**: Improve an existing binder sequence.

Provide a starting sequence via `--seq` to refine it further.

```bash
python boltz_ph/design.py \
    --name refine_my_binder \
    --seq "GPDRERARELARILLKVIKLSDSPEARRQLLRNLEELAEKYKDPEVRRILEEAERYIK" \
    --protein-seqs "YOUR_TARGET_SEQUENCE" \
    --num-designs 3 \
    --num-cycles 5 \
    --msa-mode mmseqs \
    --high-iptm-threshold 0.8 \
    --gpu-id 0
```

**Key differences from de novo:**
- `--seq` provides the starting binder sequence
- `--min-protein-length`/`--max-protein-length` are ignored (length is fixed)
- `--percent-x` is ignored (sequence is provided)
- Typically use fewer cycles and higher quality thresholds

---

## Target Specification

You must specify what you want to design a binder FOR. There are two paradigms:

### Sequence-Only Mode

**Use when**: You don't have an experimental structure of your target.

Boltz will **predict the target structure** during design, using MSA information for accuracy.

```bash
python boltz_ph/design.py \
    --protein-seqs "YOUR_TARGET_SEQUENCE" \
    --msa-mode mmseqs \
    ...
```

| Parameter | Description |
|-----------|-------------|
| `--protein-seqs` | Target protein sequence (required) |
| `--msa-mode mmseqs` | Generate MSA for better structure prediction |

**For multi-chain targets** (e.g., dimers), separate sequences with `:`:

```bash
--protein-seqs "CHAIN_B_SEQUENCE:CHAIN_C_SEQUENCE"
```

---

### Template Structure Mode

**Use when**: You have an experimental structure (PDB/CIF) of your target.

The template provides **explicit coordinates** that anchor the target structure during design.

```bash
python boltz_ph/design.py \
    --template-path "7KPL" \
    --template-cif-chain-id "A" \
    --msa-mode single \
    ...
```

| Parameter | Description |
|-----------|-------------|
| `--protein-seqs` | Target sequence(s) — optional if template contains full sequence |
| `--template-path` | PDB code, file path, or AlphaFold ID |
| `--template-cif-chain-id` | Which chain in template to use |
| `--msa-mode single` | Often used with templates (MSA optional) |

**Sequence auto-extraction:** If you omit `--protein-seqs`, sequences are automatically extracted from the template structure. This works when the template contains the complete sequence you want to target.

**When to provide `--protein-seqs`:**
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
--template-path "7KPL:7KPL" \
--template-cif-chain-id "A:B"
# Sequences auto-extracted, or provide explicitly:
--protein-seqs "SEQ_B:SEQ_C"
```

---

## Hotspot Configuration

**Hotspots** are target residues that the binder MUST contact.

### Basic Usage

```bash
--contact-residues "54,56,66,115,121"
```

This forces the binder to contact residues 54, 56, 66, 115, and 121 on the target.

### Multi-Chain Hotspots

Use `|` to separate hotspots for different target chains:

```bash
--protein-seqs "SEQ_B:SEQ_C" \
--contact-residues "10,20,30|5,15"
```

- Residues 10, 20, 30 on Chain B
- Residues 5, 15 on Chain C

### Author vs Canonical Numbering

By default, hotspot residue numbers use **canonical (1-indexed)** numbering. If your template has non-standard numbering (e.g., starting at residue 7), use `--use-auth-numbering`:

```bash
# Template has residues numbered 7-16 (author numbering)
--contact-residues "|7,8,9,10,11,12,13,14,15,16" \
--use-auth-numbering
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
| `--contact-residues` | `""` | Comma-separated residue positions |
| `--contact-cutoff` | `15.0` | Distance threshold (Å) for contact |
| `--max-contact-filter-retries` | `6` | Retries if contacts not satisfied |
| `--no-contact-filter` | `False` | Disable contact checking |

### How Contact Filtering Works

1. **Cycle 0**: After initial prediction, check if binder contacts all hotspots
2. **If failed**: Resample initial sequence, retry (up to `max_contact_filter_retries`)
3. **High-ipTM saving**: Designs only saved if contacts are satisfied

### Example with Hotspots

```bash
python boltz_ph/design.py \
    --name hotspot_design \
    --protein-seqs "TARGET_SEQUENCE" \
    --contact-residues "29,54,56,115,116,117" \
    --contact-cutoff 12.0 \
    --max-contact-filter-retries 10 \
    --num-designs 5 \
    --num-cycles 7 \
    --msa-mode mmseqs \
    --gpu-id 0
```

---

## Command-Line Arguments Reference

### Core Settings

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--name` | str | required | Job name (used for output folder) |
| `--gpu-id` | int | `0` | GPU device ID |
| `--num-designs` | int | — | Target number of **valid designs** (passed initial Boltz filters) |
| `--num-attempts` | int | — | Target number of design **attempts** (compute-budget mode) |
| `--num-accepted` | int | — | Target number of **accepted designs** (passed validation filters, requires `--validation-model` != none) |
| `--num-cycles` | int | `5` | Fold→design iterations per run |
| `--num-gpus` | int | `1` | Number of GPUs for parallel design (multi-GPU mode) |
| `--mode` | str | `"binder"` | `"binder"` or `"unconditional"` |

#### Stopping Conditions & Design Terminology

You must specify at least one stopping condition: `--num-designs`, `--num-attempts`, or `--num-accepted`. If multiple are provided, the pipeline stops when *any* target is reached (OR logic).

| Term | Definition | Counted By |
|------|------------|------------|
| **Attempt** | Every design cycle run, regardless of outcome | `design_stats.csv` row count |
| **Valid Design** | Attempt that passed initial Boltz filters (ipTM, pLDDT, %alanine) | PDBs in `best_designs/` |
| **Accepted** | Valid design that passed validation (AF3/Protenix) and scoring filters | PDBs in `accepted_designs/` |
| **Rejected** | Valid design that failed validation or scoring filters | PDBs in `rejected/` |

**Example progress output:**
```
Progress: 21/100 valid designs | (39/500 attempts) | 3/10 accepted | 8 rejected
```

**Use cases:**
- `--num-designs 100` — "I want 100 valid candidates to evaluate"
- `--num-attempts 500` — "I have compute budget for 500 tries, give me what you can"
- `--num-accepted 10` — "Stop when I have 10 winners"
- Combined: `--num-attempts 1000 --num-accepted 50` — "Run up to 1000 attempts, but stop early if we get 50 accepted"

### Multi-GPU Parallelization (Local)

Run designs in parallel across multiple GPUs on a single machine:

```bash
python boltz_ph/design.py \
    --name parallel_design \
    --protein-seqs "TARGET_SEQUENCE" \
    --num-designs 40 \
    --num-gpus 8 \
    --validation-model af3
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
| `--min-protein-length` | int | `100` | Minimum binder length (if random start) |
| `--max-protein-length` | int | `150` | Maximum binder length (if random start) |
| `--percent-x` | int | `90` | % of "X" (unknown) in initial sequence |
| `--cyclic` | flag | `False` | Design cyclic peptide |

### Target Specification

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--protein-seqs` | str | `""` | Target protein sequence(s), `:` separated |
| `--ligand-smiles` | str | `""` | Target ligand as SMILES |
| `--ligand-ccd` | str | `""` | Target ligand as CCD code (e.g., `"SAM"`) |
| `--nucleic-seq` | str | `""` | Target DNA/RNA sequence |
| `--nucleic-type` | str | `"dna"` | `"dna"` or `"rna"` |

### Template Settings

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--template-path` | str | `""` | PDB code, file path, or AlphaFold ID |
| `--template-cif-chain-id` | str | `""` | Chain ID(s) in template for alignment |
| `--msa-mode` | str | `"mmseqs"` | `"mmseqs"` (generate MSA) or `"single"` (no MSA) |

### Hotspot/Contact Settings

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--contact-residues` | str | `""` | Target residues to contact (e.g., `"10,20,30"`) |
| `--use-auth-numbering` | flag | `False` | Use PDB "author" residue numbers for hotspots |
| `--contact-cutoff` | float | `15.0` | Contact distance threshold (Å) |
| `--max-contact-filter-retries` | int | `6` | Retries if contacts unsatisfied |
| `--no-contact-filter` | flag | `False` | Disable contact filtering |

### Sequence Design (MPNN) Settings

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--temperature` | float | `0.1` | MPNN sampling temperature |
| `--omit-aa` | str | `"C"` | Amino acids to exclude |
| `--alanine-bias` | str | **`True`** | Penalize alanine (use `false` to disable) |
| `--alanine-bias-start` | float | `-0.5` | Initial alanine penalty |
| `--alanine-bias-end` | float | `-0.1` | Final alanine penalty |

### Quality Thresholds

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--high-iptm-threshold` | float | `0.8` | Min ipTM to save design |
| `--high-plddt-threshold` | float | `0.8` | Min pLDDT to save design |

### Model Settings

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--boltz-model-version` | str | `"boltz2"` | `"boltz1"` or `"boltz2"` |
| `--diffuse-steps` | int | `200` | Diffusion timesteps |
| `--recycling-steps` | int | `3` | Model recycling passes per prediction |

**Note on `--recycling-steps`:**
This controls the number of internal refinement passes within each structure prediction. More recycles = more refined prediction per cycle, but slower. This is different from `--num-cycles` which controls the outer design loop (fold → redesign → fold → ...). Default of 3 is standard for AlphaFold-style models.

### Output Settings

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--save-dir` | str | `""` | Custom output directory |
| `--plot` | flag | `False` | Generate optimization plots |

### Validation (Optional)

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
| `--validation-model` | choice | `"none"` | Validation backend: `none`, `af3` (AlphaFold3), or `protenix` (open-source AF3) |
| `--scoring-method` | choice | `"pyrosetta"` | Interface scoring: `pyrosetta` or `opensource` (OpenMM + FreeSASA) |
| `--alphafold-dir` | str | `"~/alphafold3"` | AF3 installation path (only for `--validation-model af3`) |
| `--use-msa-for-validation` | bool | **`True`** | Reuse MSAs from design phase for validation |
| `--verbose` | flag | `False` | Verbose logs for validation/scoring |

> Deprecated aliases still work locally: `--use-alphafold3-validation`, `--use-msa-for-af3`, `--use-open-scoring`.

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
    --protein-seqs "AFTVTVPK..." \
    --num-designs 100 \
    --num-accepted 5 \
    --validation-model af3 \
    --alphafold-dir ~/alphafold3

# Local with Protenix + open-source scoring (fully open-source)
python boltz_ph/design.py \
    --name PDL1_open \
    --protein-seqs "AFTVTVPK..." \
    --num-designs 100 \
    --num-accepted 5 \
    --validation-model protenix \
    --scoring-method opensource

# Modal with Protenix (fully open-source, recommended)
modal run modal_boltz_ph_cli.py::run_pipeline \
    --name PDL1_validated \
    --protein-seqs "AFTVTVPK..." \
    --num-designs 5 \
    --validation-model protenix \
    --scoring-method opensource

# Modal with AF3 (requires weights upload)
modal run modal_boltz_ph_cli.py::run_pipeline \
    --name PDL1_validated \
    --protein-seqs "AFTVTVPK..." \
    --num-designs 5 \
    --validation-model af3 \
    --validation-gpu A100-80GB
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

### Full Output (With Validation)

When using `--validation-model {af3,protenix}` (local or Modal):

```
results_my_design/
├── designs/                         # All cycles from all attempts
│   ├── design_stats.csv             # Metrics for every cycle of every attempt
│   └── {name}_d{X}_c{Y}.pdb
├── best_designs/                    # Valid designs (passed initial Boltz filters)
│   ├── best_designs.csv             # Summary of all valid designs
│   └── {name}_d{X}_c{Y}.pdb
├── refolded/                        # Refolded structures (AF3/Protenix)
│   ├── validation_results.csv
│   └── *_refolded.cif
├── accepted_designs/                # Accepted (passed validation + scoring)
│   ├── accepted_stats.csv
│   └── *_relaxed.pdb
└── rejected/                        # Rejected (failed validation or scoring)
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
| `--num-designs N` | Stop after N total designs generated |
| `--num-accepted N` | Stop after N designs pass all filters |

**Rules:**
1. At least one required
2. Both allowed — first condition met triggers exit (OR logic)
3. `--num-accepted` requires validation (`--validation-model` != none)
4. `--num-accepted` alone prints a warning (no upper limit on attempts)

### Examples

```bash
# Generate exactly 50 designs (classic mode, resumable)
python boltz_ph/design.py \
    --name PDL1 \
    --num-designs 50 \
    --protein-seqs "..."

# Generate until 10 pass filters (warning: no upper limit)
python boltz_ph/design.py \
    --name PDL1 \
    --num-accepted 10 \
    --validation-model af3 \
    --protein-seqs "..."

# Recommended: Generate until 10 accepted OR 500 total (whichever comes first)
python boltz_ph/design.py \
    --name PDL1 \
    --num-designs 500 \
    --num-accepted 10 \
    --validation-model af3 \
    --protein-seqs "..."

# Resume a crashed job (just re-run the same command)
python boltz_ph/design.py \
    --name PDL1 \
    --num-designs 50 \
    --protein-seqs "..."
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
    --protein-seqs "AFTVTVPKDLYVVEYGSNMTIECKFPVEKQLDLAALIVYWEMEDKNIIQFVHGEEDLKVQHSSYRQRARLLKDQLSLGNAALQITDVKLQDAGVYRCMISYGGADYKRITVKVNAPYAAALE" \
    --num-designs 5 \
    --num-cycles 7 \
    --min-protein-length 90 \
    --max-protein-length 150 \
    --percent-x 50 \
    --msa-mode mmseqs \
    --high-iptm-threshold 0.7 \
    --plot \
    --gpu-id 0
```

### Example 2: Design with Template Structure

```bash
python boltz_ph/design.py \
    --name PDL1_template_design \
    --protein-seqs "AFTVTVPKDLYVVEYGSNMTIECKFPVEKQLDLAALIVYWEMEDKNIIQFVHGEEDLKVQHSSYRQRARLLKDQLSLGNAALQITDVKLQDAGVYRCMISYGGADYKRITVKVNAPYAAALE" \
    --template-path "7KPL" \
    --template-cif-chain-id "B" \
    --msa-mode single \
    --num-designs 3 \
    --num-cycles 7 \
    --min-protein-length 90 \
    --max-protein-length 150 \
    --high-iptm-threshold 0.7 \
    --gpu-id 0
```

### Example 3: Hotspot-Directed Design

```bash
python boltz_ph/design.py \
    --name PDL1_hotspot_design \
    --protein-seqs "AFTVTVPKDLYVVEYGSNMTIECKFPVEKQLDLAALIVYWEMEDKNIIQFVHGEEDLKVQHSSYRQRARLLKDQLSLGNAALQITDVKLQDAGVYRCMISYGGADYKRITVKVNAPYAAALE" \
    --contact-residues "54,56,66,115,121" \
    --contact-cutoff 12.0 \
    --num-designs 5 \
    --num-cycles 7 \
    --msa-mode mmseqs \
    --high-iptm-threshold 0.7 \
    --gpu-id 0
```

### Example 4: Refine Existing Binder

```bash
python boltz_ph/design.py \
    --name refine_binder \
    --seq "MKAELRQRVQELAEQARQKLEEAEQKRVQELAEQARQKLEE" \
    --protein-seqs "TARGET_SEQUENCE" \
    --num-designs 3 \
    --num-cycles 5 \
    --msa-mode mmseqs \
    --high-iptm-threshold 0.85 \
    --gpu-id 0
```

### Example 5: Small Molecule Binder

```bash
python boltz_ph/design.py \
    --name SAM_binder \
    --ligand-ccd SAM \
    --num-designs 5 \
    --num-cycles 7 \
    --min-protein-length 130 \
    --max-protein-length 150 \
    --high-iptm-threshold 0.7 \
    --gpu-id 0
```

### Example 6: Cyclic Peptide Binder

```bash
python boltz_ph/design.py \
    --name cyclic_peptide \
    --protein-seqs "TARGET_SEQUENCE" \
    --cyclic \
    --num-designs 5 \
    --num-cycles 7 \
    --min-protein-length 10 \
    --max-protein-length 20 \
    --percent-x 100 \
    --msa-mode mmseqs \
    --high-iptm-threshold 0.8 \
    --gpu-id 0
```

### Example 7: Multimer Target (Homodimer)

```bash
python boltz_ph/design.py \
    --name dimer_binder \
    --protein-seqs "CHAIN_B_SEQ:CHAIN_B_SEQ" \
    --num-designs 3 \
    --num-cycles 7 \
    --min-protein-length 90 \
    --max-protein-length 150 \
    --msa-mode mmseqs \
    --high-iptm-threshold 0.7 \
    --gpu-id 0
```

### Example 8: Multi-GPU Parallel Design with Template

```bash
python boltz_ph/design.py \
    --name parallel_pMHC \
    --template-path "./pmhc_structure.pdb" \
    --template-cif-chain-id "A,B" \
    --contact-residues "|7,8,9,10,11,12,13,14,15,16" \
    --use-auth-numbering \
    --num-designs 40 \
    --num-cycles 5 \
    --num-gpus 8 \
    --validation-model af3 \
    --alphafold-dir ~/alphafold3
```

---

## Tips & Best Practices

1. **Start with lower thresholds** (`--high-iptm-threshold 0.6-0.7`) for initial exploration, then increase for production runs.

2. **Alanine bias is ON by default** to avoid poly-alanine sequences. Use `--alanine-bias false` only if you specifically want to allow high alanine content.

3. **More designs > more cycles**: Running 10 designs × 5 cycles often gives better diversity than 3 designs × 15 cycles.

4. **For difficult targets**, increase `--max-contact-filter-retries` and use hotspots to guide the design.

5. **Template mode** is recommended when you have high-confidence structures; sequence-only mode is better for exploring conformational flexibility.

6. **Use dual stopping conditions** for production runs: `--num-designs 500 --num-accepted 20` stops when you have 20 good designs OR after 500 attempts, whichever comes first.

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

# Quick test run (fewer attempts)
modal run modal_boltz_ph_cli.py::run_pipeline \
    --name "quick_test" \
    --protein-seqs "YOUR_TARGET_SEQUENCE" \
    --num-attempts 5 \
    --num-cycles 5 \
    --gpu H100

# With early stopping (stop when 3 designs are accepted)
modal run modal_boltz_ph_cli.py::run_pipeline \
    --name "my_design" \
    --protein-seqs "YOUR_TARGET_SEQUENCE" \
    --num-attempts 50 \
    --num-accepted 3 \
    --validation-model protenix \
    --scoring-method opensource \
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

**Syntax:** Use kebab-case (`--protein-seqs`). Underscore flags remain as deprecated aliases locally.

**Boolean flags:** Use `--flag=true` or `--flag=false` syntax (e.g., `--alanine-bias=false`)

#### Stopping Conditions (Modal)

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--num-attempts` | int | `50` | Number of design attempts to run |
| `--num-accepted` | int | — | Stop early when N designs are accepted (requires validation) |

> **Note**: Modal uses `--num-attempts` (local uses both `--num-attempts` and `--num-designs`). In Modal, every dispatched task is an attempt. Add `--num-accepted` to stop early when enough designs pass validation.

#### Modal-Specific Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--gpu` | str | `"H100"` | GPU type for Boltz design (see below) |
| `--max-concurrent` | int | `1` | Max parallel GPUs (0 = unlimited) |
| `--no-stream` | str | `"false"` | Disable real-time result streaming |
| `--sync-interval` | float | `5.0` | Sync polling interval (seconds) |
| `--output-dir` | str | `"./results_{name}"` | Local output directory |

#### Validation & Scoring Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--validation-model` | str | `"none"` | Structure validation: `none`, `af3`, or `protenix` |
| `--scoring-method` | str | `"pyrosetta"` | Scoring: `pyrosetta` or `opensource` |
| `--validation-gpu` | str | auto | GPU for validation (`A100` for protenix, `A100-80GB` for af3) |
| `--use-msa-for-validation` | str | `"true"` | Reuse MSAs from design phase for validation |

**Validation models:**
- `none` — Design only, no cross-validation (default)
- `protenix` — Open-source AF3 reproduction (Apache 2.0 license, recommended)
- `af3` — AlphaFold3 (requires proprietary weights, see [AF3 Setup](#alphafold3-validation-optional))

**Scoring methods:**
- `pyrosetta` — Full PyRosetta scoring with interface energetics
- `opensource` — OpenMM + FreeSASA + sc-rs (no license required)

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
# Run 16 attempts in parallel (uses all available GPUs)
modal run modal_boltz_ph_cli.py::run_pipeline \
    --num-attempts 16 \
    --gpu H100

# Limit to 8 concurrent GPUs (runs in 2 batches)
modal run modal_boltz_ph_cli.py::run_pipeline \
    --num-attempts 16 \
    --max-concurrent 8 \
    --gpu H100

# Stop early when 5 designs are accepted (validation + scoring)
modal run modal_boltz_ph_cli.py::run_pipeline \
    --num-attempts 100 \
    --num-accepted 5 \
    --validation-model protenix \
    --scoring-method opensource
```

**How it works:**
- Each design run executes as an independent Modal container
- `--max-concurrent 1` (default): Designs run sequentially (one at a time)
- `--max-concurrent N`: Designs run in batches of N
- `--max-concurrent 0`: Unlimited parallelism (all designs run simultaneously)

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
    --num-attempts 10 \
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
    --num-attempts 10 \
    --num-cycles 5 \
    --high-iptm-threshold 0.7 \
    --gpu H100
```

#### Example 3: pMHC Binder with Template + Validation

```bash
# Sequences auto-extracted from template; hotspots use PDB numbering
# Uses Protenix validation (fully open-source)
modal run modal_boltz_ph_cli.py::run_pipeline \
    --name "pMHC_TCRm" \
    --template-path "./my_pmhc_structure.pdb" \
    --template-cif-chain-id "A,B" \
    --contact-residues "|7,8,9,10,11,12,13,14,15,16" \
    --use-auth-numbering=true \
    --num-attempts 16 \
    --num-cycles 5 \
    --max-concurrent 8 \
    --min-protein-length 60 \
    --max-protein-length 120 \
    --high-iptm-threshold 0.8 \
    --validation-model protenix \
    --scoring-method opensource \
    --gpu H100 \
    --output-dir ./pMHC_results
```

#### Example 4: Small Molecule Binder

```bash
modal run modal_boltz_ph_cli.py::run_pipeline \
    --name "SAM_binder" \
    --ligand-ccd "SAM" \
    --num-attempts 5 \
    --num-cycles 5 \
    --min-protein-length 130 \
    --max-protein-length 150 \
    --high-iptm-threshold 0.7 \
    --gpu H100
```

#### Example 5: AF3 Validation with PyRosetta Scoring

```bash
# Requires AF3 weights upload first (see AF3 Setup section)
modal run modal_boltz_ph_cli.py::run_pipeline \
    --name "PDL1_af3" \
    --protein-seqs "AFTVTVPKDLYVVEYGSNMTIECKFPVEKQLDLAALIVYWEMEDKNIIQFVHGEEDLKVQHSSYRQRARLLKDQLSLGNAALQITDVKLQDAGVYRCMISYGGADYKRITVKVNAPYAAALE" \
    --num-attempts 10 \
    --num-cycles 5 \
    --validation-model af3 \
    --scoring-method pyrosetta \
    --validation-gpu A100-80GB \
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

## Open-Source Scoring (Modal)

The Modal pipeline supports an **open-source scoring pathway** as an alternative to PyRosetta. This allows running the full design-to-validation pipeline without a PyRosetta license.

The open-source scoring implementation is adapted from [FreeBindCraft](https://github.com/cytokineking/FreeBindCraft) by cytokineking.

### What's Included

| Component | Tool | Description |
|-----------|------|-------------|
| **Relaxation** | OpenMM + FASPR | GPU-accelerated structure relaxation (~3-4x faster than PyRosetta FastRelax) |
| **Shape Complementarity** | [sc-rs](https://github.com/cytokineking/sc-rs) | Rust implementation, nearly identical to PyRosetta SC |
| **SASA Calculations** | FreeSASA | Surface area metrics with NACCESS classifier |
| **Interface Detection** | Biopython | Contact-based interface residue identification |

### Usage

Use `--scoring-method opensource` to enable open-source scoring:

```bash
# Fully open-source pipeline (Protenix validation + opensource scoring)
modal run modal_boltz_ph_cli.py::run_pipeline \
    --name "my_design" \
    --protein-seqs "YOUR_TARGET_SEQUENCE" \
    --validation-model protenix \
    --scoring-method opensource \
    --gpu H100

# AF3 validation with open-source scoring
modal run modal_boltz_ph_cli.py::run_pipeline \
    --name "my_design" \
    --protein-seqs "YOUR_TARGET_SEQUENCE" \
    --validation-model af3 \
    --scoring-method opensource \
    --validation-gpu A100-80GB \
    --gpu H100
```

### Computed Metrics

The open-source pathway computes these interface metrics:

- `interface_sc` — Shape complementarity (sc-rs)
- `interface_dSASA` — Buried surface area at interface (FreeSASA)
- `interface_nres` — Number of interface residues (Biopython)
- `surface_hydrophobicity` — Fraction of hydrophobic surface (FreeSASA)
- `binder_sasa` — Total binder solvent-accessible surface area
- `rg` — Radius of gyration (compactness measure)
- `apo_holo_rmsd` — Structural change between bound and unbound states

### Placeholder Values

Some PyRosetta-specific metrics (e.g., `interface_dG`, `interface_packstat`, `interface_hbonds`) are set to placeholder values that pass default filters. Evaluate design quality using the computed metrics above.

### Example: Full Pipeline with Open-Source Validation + Scoring

```bash
# Recommended: Fully open-source pipeline
modal run modal_boltz_ph_cli.py::run_pipeline \
    --name "PDL1_opensource" \
    --protein-seqs "AFTVTVPKDLYVVEYGSNMTIECKFPVEKQLDLAALIVYWEMEDKNIIQFVHGEEDLKVQHSSYRQRARLLKDQLSLGNAALQITDVKLQDAGVYRCMISYGGADYKRITVKVNAPYAAALE" \
    --contact-residues "56,113,115,123" \
    --num-attempts 10 \
    --num-cycles 5 \
    --gpu A100 \
    --validation-model protenix \
    --scoring-method opensource \
    --max-concurrent 5 \
    --output-dir ./opensource_results
```

### Protenix: Open-Source Structure Validation

With `--validation-model protenix`, the pipeline uses [Protenix](https://github.com/bytedance/Protenix) — an open-source reproduction of AlphaFold3 released under the Apache 2.0 license. This enables a **fully open-source design pipeline** from structure prediction through scoring, with no proprietary dependencies.

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
