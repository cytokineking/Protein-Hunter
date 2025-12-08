import json
import logging
import os
import shutil
import sys
import subprocess
import time
import urllib
from pathlib import Path

import yaml
from tqdm import tqdm
import numpy as np
import pandas as pd

from Bio.Align import PairwiseAligner
from Bio.Data import IUPACData
from Bio.PDB import PDBParser, MMCIFParser

import requests, io

from boltz_ph.constants import RNA_CHAIN_POLY_TYPE
from utils.convert import convert_cif_files_to_pdb, download_with_progress, calculate_holo_apo_rmsd, RNA_DATABASE_INFO, AF3_SOURCE
from utils.msa_tools import Nhmmer, Msa, parse_fasta
from boltz_ph.model_utils import process_msa, get_cif # MMseqs2 wrapper and PDB fetcher
from typing import Optional, Dict, List
import math

# ============================================================================
# AF3 OUTPUT FILTERING
# ============================================================================
# Patterns to suppress from AF3 Docker output (log spam and disclaimers)
_AF3_SUPPRESSED_PATTERNS = [
    # Terms of use disclaimer (users already agreed to get the weights)
    "Running AlphaFold 3. Please note",
    "only available under terms of use",
    "If you do not agree to these terms",
    "cancel execution of AlphaFold 3",
    "use the model parameters.",
    # JAX backend probing (not relevant for CUDA users)
    "Unable to initialize backend 'rocm'",
    "Unable to initialize backend 'tpu'",
    # Verbose bucket size logging
    "Calculating bucket size for input",
    "Got bucket size",
    "resulting in",
    "padded tokens",
]

def _run_af3_subprocess(cmd: list, check: bool = True) -> subprocess.CompletedProcess:
    """
    Run an AF3 subprocess with output filtering to suppress log spam.
    
    Captures stdout/stderr, filters out noisy messages, and prints the rest.
    """
    proc = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
    )
    
    # Filter and print stdout
    for line in proc.stdout.splitlines():
        if not any(pattern in line for pattern in _AF3_SUPPRESSED_PATTERNS):
            print(line)
    
    # Filter and print stderr (AF3 logs go here)
    for line in proc.stderr.splitlines():
        if not any(pattern in line for pattern in _AF3_SUPPRESSED_PATTERNS):
            print(line, file=sys.stderr)
    
    if check and proc.returncode != 0:
        raise subprocess.CalledProcessError(proc.returncode, cmd)
    
    return proc


def _is_valid_value(val) -> bool:
    """Check if a value is valid (not None, not NaN, not empty string).
    
    This handles pandas NaN values which are truthy in Python.
    Follows the Modal pattern of strict validation before adding to JSON.
    """
    if val is None:
        return False
    if isinstance(val, float) and math.isnan(val):
        return False
    if isinstance(val, str) and val.strip() in ('', 'nan'):
        return False
    return True


def calculate_af3_ipsae(
    confidence_json: str,
    binder_length: int,
    target_length: int,
    pae_cutoff: float = 10.0,
) -> Dict[str, float]:
    """
    Calculate ipSAE from AF3 confidence JSON.
    
    Uses the same algorithm as Boltz ipSAE calculation, adapted for AF3's
    PAE matrix format. Aligns with Modal pipeline implementation.
    
    Args:
        confidence_json: Raw JSON string from AF3 confidence output
        binder_length: Number of residues in binder (chain A)
        target_length: Number of residues in target (chain B)
        pae_cutoff: PAE cutoff for considering residue pairs (default 10.0 Å)
    
    Returns:
        dict with:
            - 'af3_ipsae': max of binder and target direction scores
            - 'af3_ipsae_binder_to_target': per-residue max from binder → target
            - 'af3_ipsae_target_to_binder': per-residue max from target → binder
    """
    result = {
        'af3_ipsae': 0.0,
        'af3_ipsae_binder_to_target': 0.0,
        'af3_ipsae_target_to_binder': 0.0,
    }
    
    try:
        confidence = json.loads(confidence_json)
        pae_data = confidence.get("pae", [])
        
        if not pae_data:
            return result
        
        # Convert to numpy array
        pae_matrix = np.array(pae_data)
        
        # AF3 outputs PAE as a flat list or 2D array
        # Expected shape after conversion: (N, N) where N = binder_length + target_length
        total_length = binder_length + target_length
        
        if pae_matrix.ndim == 1:
            # Flatten format - reshape to square matrix
            expected_size = total_length * total_length
            if len(pae_matrix) != expected_size:
                return result
            pae_matrix = pae_matrix.reshape(total_length, total_length)
        
        if pae_matrix.shape != (total_length, total_length):
            return result
        
        # Define binder and target indices
        # AF3 orders chains alphabetically, so binder (A) comes first
        binder_indices = np.arange(binder_length)
        target_indices = np.arange(binder_length, total_length)
        
        # Extract interface PAE: binder rows → target columns
        interface_pae = pae_matrix[np.ix_(binder_indices, target_indices)]
        
        # Apply PAE cutoff mask
        valid_mask = interface_pae < pae_cutoff
        
        # PTM-style transformation for ipSAE
        def ptm_func(x: np.ndarray, d0: float) -> np.ndarray:
            return 1.0 / (1.0 + (x / d0) ** 2.0)
        
        def calc_d0(L: int) -> float:
            L = float(max(L, 27))
            d0 = 1.24 * (L - 15) ** (1.0 / 3.0) - 1.8
            return max(1.0, d0)
        
        # Calculate per-binder-residue ipSAE scores (binder → target direction)
        ipsae_byres_binder = []
        for i in range(binder_length):
            valid = valid_mask[i]
            if valid.any():
                n0res = valid.sum()
                d0res = calc_d0(n0res)
                ptm_vals = ptm_func(interface_pae[i][valid], d0res)
                ipsae_byres_binder.append(ptm_vals.mean())
            else:
                ipsae_byres_binder.append(0.0)
        
        ipsae_byres_binder = np.array(ipsae_byres_binder)
        ipsae_binder_max = float(ipsae_byres_binder.max()) if len(ipsae_byres_binder) > 0 else 0.0
        
        # Calculate reverse direction: target → binder
        interface_pae_rev = pae_matrix[np.ix_(target_indices, binder_indices)]
        valid_mask_rev = interface_pae_rev < pae_cutoff
        
        ipsae_byres_target = []
        for i in range(target_length):
            valid = valid_mask_rev[i]
            if valid.any():
                n0res = valid.sum()
                d0res = calc_d0(n0res)
                ptm_vals = ptm_func(interface_pae_rev[i][valid], d0res)
                ipsae_byres_target.append(ptm_vals.mean())
            else:
                ipsae_byres_target.append(0.0)
        
        ipsae_byres_target = np.array(ipsae_byres_target)
        ipsae_target_max = float(ipsae_byres_target.max()) if len(ipsae_byres_target) > 0 else 0.0
        
        # Take max of both directions
        ipsae = max(ipsae_binder_max, ipsae_target_max)
        
        result['af3_ipsae'] = round(ipsae, 4)
        result['af3_ipsae_binder_to_target'] = round(ipsae_binder_max, 4)
        result['af3_ipsae_target_to_binder'] = round(ipsae_target_max, 4)
        
    except Exception as e:
        print(f"  Warning: AF3 ipSAE calculation failed: {e}")
    
    return result

# Default settings for RNA MSA generation (from ColabNuFold/AlphaFold3 pipeline)
RNA_DEFAULT_SETTINGS = {
    "use_rfam_db": True,
    "use_rnacentral_db": True,
    "use_ntrna_db": False,
    "max_sequences_per_db": 10000,
    "e_value": 0.001,
    "time_limit_minutes": 120,
    "n_cpu": 2,
}

# Consolidated AF3 settings (using a temporary copy of default to ensure mutability)
AF3_DATABASE_SETTINGS = RNA_DEFAULT_SETTINGS.copy()


def download_selected_databases(database_settings: dict, afdb_dir: str):
    """Download only the databases selected in the settings."""
    afdb_dir = os.path.expanduser(afdb_dir)
    os.makedirs(afdb_dir, exist_ok=True)
    
    selected_db_files = [
        db_file for db_file, db_key in RNA_DATABASE_INFO.items()
        if database_settings.get(db_key, False)
    ]

    if not selected_db_files:
        print("⚠️ No RNA databases selected for download!")
        return

    print(
        f"🌐 Downloading {len(selected_db_files)} RNA databases: {', '.join([RNA_DATABASE_INFO[db] for db in selected_db_files])}"
    )

    missing_dbs = []
    for db in selected_db_files:
        db_path = os.path.join(afdb_dir, db)
        if not os.path.exists(db_path) or os.path.getsize(db_path) == 0:
            missing_dbs.append(db)

    if not missing_dbs:
        print("✅ All selected databases already downloaded.")
        return

    with tqdm(
        total=len(missing_dbs), desc="Overall progress", unit="db", position=0
    ) as main_pbar:
        for db in missing_dbs:
            dest_path = os.path.join(afdb_dir, f"{db}.zst")
            final_path = os.path.join(afdb_dir, db)

            print(f"📥 Downloading {db} ({RNA_DATABASE_INFO[db]})...")
            url = f"{AF3_SOURCE}/{db}.zst"
            if download_with_progress(url, dest_path):
                print(f"📦 Decompressing {db}...")
                try:
                    # Decompress with zstd (assumes zstd is in PATH)
                    subprocess.run(
                        ["zstd", "--decompress", "-f", dest_path, "-o", final_path],
                        check=True,
                    )
                    print(f"✅ Successfully processed {db}")
                    os.remove(dest_path)
                except Exception as e:
                    print(f"❌ Error decompressing {db}: {e}")

            main_pbar.update(1)


def af3_generate_rna_msa(rna_sequence: str, database_settings: dict, afdb_dir: str, hmmer_path: str) -> str:
    """Generate MSA for an RNA sequence using the AlphaFold3 HMMER pipeline."""
    
    rna_sequence = rna_sequence.upper().strip().replace('T', 'U') # RNA must be U, not T
    valid_bases = set("ACGU")
    if not all(base in valid_bases for base in rna_sequence):
        raise ValueError(
            f"Invalid RNA sequence. Must contain only A, C, G, U: {rna_sequence}"
        )

    # Setup paths to binaries and databases
    hmmer_path = os.path.expanduser(hmmer_path)
    afdb_dir = os.path.expanduser(afdb_dir)

    nhmmer_binary = os.path.join(hmmer_path, "bin/nhmmer")
    hmmalign_binary = os.path.join(hmmer_path, "bin/hmmalign")
    hmmbuild_binary = os.path.join(hmmer_path, "bin/hmmbuild")

    database_paths = {
        "Rfam": os.path.join(afdb_dir, RNA_DATABASE_INFO["rfam_14_9_clust_seq_id_90_cov_80_rep_seq.fasta"]),
        "RNAcentral": os.path.join(afdb_dir, RNA_DATABASE_INFO["rnacentral_active_seq_id_90_cov_80_linclust.fasta"]),
        "NT_RNA": os.path.join(afdb_dir, RNA_DATABASE_INFO["nt_rna_2023_02_23_clust_seq_id_90_cov_80_rep_seq.fasta"]),
    }

    # 1. Download missing databases
    selected_db_keys = [db_key for db_file, db_key in RNA_DATABASE_INFO.items() if database_settings.get(db_key, False)]
    download_selected_databases(database_settings, afdb_dir)
    
    # 2. Filter databases to only existing, selected ones
    filtered_db_paths = {}
    for db_key in selected_db_keys:
        db_path = database_paths.get(db_key)
        if db_path and os.path.exists(db_path) and os.path.getsize(db_path) > 0:
            filtered_db_paths[db_key] = db_path

    if not filtered_db_paths:
        print("❌ No selected RNA databases found or none selected. Returning query only.")
        return f">query\n{rna_sequence}\n"

    print(f"🔍 Will search {len(filtered_db_paths)} databases.")
    
    # 3. Run Nhmmer on each database
    msas = []
    rna_msa_start_time = time.time()
    
    with tqdm(total=len(filtered_db_paths), desc="Database searches", unit="db") as progress_bar:
        for db_name, db_path in filtered_db_paths.items():
            nhmmer_runner = Nhmmer(
                binary_path=nhmmer_binary,
                hmmalign_binary_path=hmmalign_binary,
                hmmbuild_binary_path=hmmbuild_binary,
                database_path=db_path,
                n_cpu=database_settings.get("n_cpu", 2),
                e_value=database_settings.get("e_value", 0.001),
                max_sequences=database_settings.get("max_sequences_per_db", 10000),
                alphabet="rna",
                time_limit_minutes=database_settings.get("time_limit_minutes"),
            )
            try:
                a3m_result = nhmmer_runner.query(rna_sequence)
                msa = Msa.from_a3m(
                    query_sequence=rna_sequence,
                    chain_poly_type=RNA_CHAIN_POLY_TYPE,
                    a3m=a3m_result,
                    deduplicate=False,
                )
                msas.append(msa)
                print(f"✅ Found {msa.depth} sequences in {db_name}")
            except Exception as e:
                print(f"❌ Error processing {db_name}: {e}")
            progress_bar.update(1)

    # 4. Merge and deduplicate MSAs
    if not msas:
        print("⚠️ No homologous sequences found. MSA contains only the query sequence.")
        a3m = f">query\n{rna_sequence}\n"
    else:
        rna_msa = Msa.from_multiple_msas(msas=msas, deduplicate=True)
        print(f"🎉 MSA construction complete! Found {rna_msa.depth} unique sequences.")
        a3m = rna_msa.to_a3m()

    elapsed_time = time.time() - rna_msa_start_time
    print(f"⏱️ Total RNA MSA generation time: {elapsed_time:.2f} seconds")

    return a3m


def extract_sequences_and_format(a3m_file_path: str, replace_query_seq: bool = False, query_seq: str = ""):
    """
    Reads an A3M file, optionally prepends the query sequence, and formats it for AF3 JSON input.
    """
    sequences = []
    
    # 1. Read the A3M file and parse sequences
    with open(a3m_file_path) as file:
        a3m_content = file.read()
    
    seqs, descs = parse_fasta(a3m_content)
    
    # 2. Format
    if replace_query_seq and query_seq:
        sequences.append(f">query\n{query_seq}")
        # Add the remaining sequences, starting from the second one in the original file
        # The original first sequence (index 0) is usually the query, replace it with wild-type or skip it.
        # Skip the original query and re-add the rest as wild-type
        for desc, seq in zip(descs[1:], seqs[1:]):
            sequences.append(f">{desc}\n{seq}")
    else:
        # If not replacing, just use the file content as is (but formatted)
         for desc, seq in zip(descs, seqs):
            sequences.append(f">{desc}\n{seq}")

    joined_string = "\n".join(sequences)
    return joined_string


def get_cif_alignment_json(query_seq, cif_or_id, chain_id=None):
    """
    Given a query sequence and a PDB id (str, e.g. '1G13') or a path to a .cif file,
    returns a dict with mmCIF text (filtered to selected chain), and alignment indices 
    (queryIndices, templateIndices) for the first chain (or optionally specified chain_id).
    """
    # Load mmCIF
    if cif_or_id.lower().endswith('.cif') and os.path.isfile(cif_or_id):
        with open(cif_or_id) as f:
            mmcif_text = f.read()
        pdb_id = os.path.splitext(os.path.basename(cif_or_id))[0]
    else:
        pdb_id = cif_or_id.upper()
        url = f"https://files.rcsb.org/download/{pdb_id}.cif"
        mmcif_text = requests.get(url).text

    # Parse structure and extract chain
    parser = MMCIFParser(QUIET=True)
    structure = parser.get_structure(pdb_id, io.StringIO(mmcif_text))
    if chain_id is not None:
        chain = next((ch for ch in structure.get_chains() if ch.id == chain_id), None)
        if chain is None:
            raise ValueError(f"Chain {chain_id} not found in structure {pdb_id}.")
    else:
        chain = next(structure.get_chains())  # default: first chain
    
    selected_chain_id = chain.id

    # Build 1-letter template sequence using 3-letter code mapping
    three_to_one = IUPACData.protein_letters_3to1
    template_seq = "".join(
        three_to_one.get(residue.resname.capitalize(), "X")
        for residue in chain.get_residues() if residue.id[0] == " "
    )

    # Align using PairwiseAligner (replaces deprecated pairwise2)
    aligner = PairwiseAligner()
    aligner.mode = 'global'
    aligner.match_score = 1
    aligner.mismatch_score = 0
    aligner.open_gap_score = 0
    aligner.extend_gap_score = 0
    alignments = list(aligner.align(query_seq, template_seq))
    if not alignments:
        raise ValueError(f"Could not align query to template sequence")
    alignment = alignments[0]
    
    # Extract index mappings using alignment.indices
    # indices[0] = query indices (-1 for gaps), indices[1] = target indices (-1 for gaps)
    indices = alignment.indices
    query_indices, template_indices = [], []
    for col in range(indices.shape[1]):
        q_idx = indices[0, col]
        t_idx = indices[1, col]
        if q_idx >= 0 and t_idx >= 0:
            query_indices.append(int(q_idx))
            template_indices.append(int(t_idx))

    # Filter mmCIF to only include the selected chain
    filtered_mmcif = filter_mmcif_by_chain(mmcif_text, selected_chain_id)

    return {
        "mmcif": filtered_mmcif,
        "queryIndices": query_indices,
        "templateIndices": template_indices
    }


def filter_mmcif_by_chain(mmcif_text, chain_id):
    """
    Filter mmCIF text to only include data for the specified chain.
    Keeps header information and filters atom_site and other relevant sections.
    """
    lines = mmcif_text.split('\n')
    filtered_lines = []
    in_atom_site = False
    atom_site_columns = {}
    chain_column_idx = None
    
    for line in lines:
        # Keep all header and metadata lines
        if line.startswith('data_') or line.startswith('_entry.') or \
           line.startswith('_audit') or line.startswith('_database') or \
           line.startswith('_entity') or line.startswith('_struct') or \
           line.startswith('_exptl') or line.startswith('_cell') or \
           line.startswith('_symmetry') or line.startswith('#'):
            filtered_lines.append(line)
            continue
        
        # Detect start of atom_site section
        if line.startswith('_atom_site.'):
            in_atom_site = True
            filtered_lines.append(line)
            # Parse column name
            col_name = line.split('.')[1].split()[0]
            atom_site_columns[col_name] = len(atom_site_columns)
            if col_name == 'label_asym_id' or col_name == 'auth_asym_id':
                chain_column_idx = len(atom_site_columns) - 1
            continue
        
        # Filter atom_site data lines
        if in_atom_site:
            if line.startswith('ATOM') or line.startswith('HETATM'):
                parts = line.split()
                if chain_column_idx is not None and len(parts) > chain_column_idx:
                    if parts[chain_column_idx] == chain_id:
                        filtered_lines.append(line)
            elif line.strip() == '' or line.startswith('_') or line.startswith('loop_'):
                in_atom_site = False
                filtered_lines.append(line)
            else:
                filtered_lines.append(line)
        else:
            filtered_lines.append(line)
    
    return '\n'.join(filtered_lines)


def run_af3_native(
    json_path: str,
    output_dir: str,
    model_dir: str = "/af3_weights",
    alphafold_script: str = "/app/alphafold/run_alphafold.py",
) -> Dict:
    """
    Run AF3 directly without Docker (for Modal containers or native installations).
    
    This function calls run_alphafold.py directly via subprocess, suitable for
    environments where AF3 is installed natively (e.g., Modal containers).
    
    Args:
        json_path: Path to AF3 input JSON file
        output_dir: Directory for AF3 output
        model_dir: Path to AF3 weights directory (default: /af3_weights for Modal)
        alphafold_script: Path to run_alphafold.py script
    
    Returns:
        Dict with af3_iptm, af3_ptm, af3_plddt, af3_structure, af3_confidence_json, error
    """
    name = Path(json_path).stem
    result = {
        "design_id": name,
        "af3_iptm": 0,
        "af3_ptm": 0,
        "af3_plddt": 0,
        "af3_structure": None,
        "af3_confidence_json": None,
        "error": None,
    }
    
    # Check weights exist
    weights_file = Path(model_dir) / "af3.bin"
    if not weights_file.exists():
        result["error"] = f"AF3 weights not found at {weights_file}"
        return result
    
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        proc = subprocess.run([
            "python", alphafold_script,
            f"--json_path={json_path}",
            f"--model_dir={model_dir}",
            f"--output_dir={output_dir}",
            "--run_data_pipeline=false",
            "--run_inference=true",
        ], capture_output=True, text=True, timeout=1800, cwd="/app/alphafold")
        
        if proc.returncode != 0:
            result["error"] = f"AF3 failed with code {proc.returncode}: {proc.stderr[:500]}"
            return result
            
    except subprocess.TimeoutExpired:
        result["error"] = "AF3 prediction timed out (30 min)"
        return result
    except Exception as e:
        result["error"] = f"AF3 subprocess error: {str(e)}"
        return result
    
    # Parse results - handle lowercase output dir name (AF3 convention)
    output_subdir = Path(output_dir) / name.lower()
    if not output_subdir.exists():
        output_subdir = Path(output_dir) / name  # Try original case
    
    if output_subdir.exists():
        # Try both naming conventions for confidence file
        confidence_file = output_subdir / f"{name}_confidences.json"
        if not confidence_file.exists():
            confidence_file = output_subdir / f"{name.lower()}_confidences.json"
        if not confidence_file.exists():
            confidence_file = output_subdir / "confidence.json"
        
        # Try both naming conventions for structure file
        structure_file = output_subdir / f"{name}_model.cif"
        if not structure_file.exists():
            structure_file = output_subdir / f"{name.lower()}_model.cif"
        if not structure_file.exists():
            structure_file = output_subdir / "model.cif"
        
        if confidence_file.exists():
            try:
                confidence_text = confidence_file.read_text()
                confidence = json.loads(confidence_text)
                result["af3_confidence_json"] = confidence_text
                
                # Get pLDDT - average of atom_plddts
                atom_plddts = confidence.get("atom_plddts", [])
                if atom_plddts:
                    result["af3_plddt"] = sum(atom_plddts) / len(atom_plddts)
                
                # Check for summary file with ipTM
                summary_file = output_subdir / f"{name}_summary_confidences.json"
                if not summary_file.exists():
                    summary_file = output_subdir / f"{name.lower()}_summary_confidences.json"
                
                if summary_file.exists():
                    summary = json.loads(summary_file.read_text())
                    result["af3_iptm"] = summary.get("iptm", summary.get("ptm", 0))
                    result["af3_ptm"] = summary.get("ptm", 0)
                else:
                    result["af3_iptm"] = confidence.get("iptm", confidence.get("ranking_score", 0))
                    result["af3_ptm"] = confidence.get("ptm", 0)
            except Exception as e:
                result["error"] = f"Error reading confidence: {e}"
        
        if structure_file.exists():
            result["af3_structure"] = structure_file.read_text()
    else:
        result["error"] = f"Output directory not found: {output_subdir}"
    
    return result


def build_af3_json_holo(
    design_id: str,
    binder_seq: str,
    target_seq: str,
    output_dir: str,
    binder_chain: str = "A",
    target_chain: str = "B",
) -> str:
    """
    Build AF3 JSON input for holo (complex) prediction.
    
    Args:
        design_id: Unique identifier for this design
        binder_seq: Binder protein sequence
        target_seq: Target protein sequence
        output_dir: Directory to write JSON file
        binder_chain: Chain ID for binder (default "A")
        target_chain: Chain ID for target (default "B")
    
    Returns:
        Path to the generated JSON file
    """
    os.makedirs(output_dir, exist_ok=True)
    
    af3_input = {
        "name": design_id,
        "modelSeeds": [1],
        "dialect": "alphafold3",
        "version": 1,
        "sequences": [
            {
                "protein": {
                    "id": binder_chain,
                    "sequence": binder_seq,
                    "unpairedMsa": f">query\n{binder_seq}\n",
                    "pairedMsa": f">query\n{binder_seq}\n",
                    "templates": [],
                }
            },
            {
                "protein": {
                    "id": target_chain,
                    "sequence": target_seq,
                    "unpairedMsa": f">query\n{target_seq}\n",
                    "pairedMsa": f">query\n{target_seq}\n",
                    "templates": [],
                }
            }
        ]
    }
    
    json_path = os.path.join(output_dir, f"{design_id}.json")
    with open(json_path, "w") as f:
        json.dump(af3_input, f, indent=2)
    
    return json_path


def build_af3_json_apo(
    design_id: str,
    binder_seq: str,
    output_dir: str,
    binder_chain: str = "A",
) -> str:
    """
    Build AF3 JSON input for apo (binder-only) prediction.
    
    Args:
        design_id: Unique identifier for this design
        binder_seq: Binder protein sequence
        output_dir: Directory to write JSON file
        binder_chain: Chain ID for binder (default "A")
    
    Returns:
        Path to the generated JSON file
    """
    os.makedirs(output_dir, exist_ok=True)
    
    apo_name = f"{design_id}_apo"
    af3_input = {
        "name": apo_name,
        "modelSeeds": [1],
        "dialect": "alphafold3",
        "version": 1,
        "sequences": [{
            "protein": {
                "id": binder_chain,
                "sequence": binder_seq,
                "unpairedMsa": f">query\n{binder_seq}\n",
                "pairedMsa": f">query\n{binder_seq}\n",
                "templates": [],
            }
        }]
    }
    
    json_path = os.path.join(output_dir, f"{apo_name}.json")
    with open(json_path, "w") as f:
        json.dump(af3_input, f, indent=2)
    
    return json_path


def run_alphafold_step(
    yaml_dir,
    alphafold_dir,
    af3_docker_name,
    af3_database_settings,
    hmmer_path,
    ligandmpnn_dir,
    work_dir,
    binder_id="A",
    gpu_id=0,
    high_iptm=False,
    use_msa_for_af3=False,
):
    """
    Orchestrates the AlphaFold validation step:
    1. Processes Boltz YAMLs into AF3 JSON inputs (holo and apo).
    2. Runs the AF3 Docker script for holo and apo states.
    3. Converts output CIFs to PDBs and calculates holo-apo RMSD.
    """
    print("Starting AlphaFold validation step...")

    alphafold_dir = os.path.expanduser(alphafold_dir)
    afdb_dir = os.path.expanduser(af3_database_settings)
    hmmer_path = os.path.expanduser(hmmer_path)

    # Create AlphaFold directories
    af_input_dir = f"{ligandmpnn_dir}/02_design_json_af3"
    af_output_dir = f"{ligandmpnn_dir}/02_design_final_af3"
    af_input_apo_dir = f"{ligandmpnn_dir}/02_design_json_af3_apo"
    af_output_apo_dir = f"{ligandmpnn_dir}/02_design_final_af3_apo"

    for dir_path in [af_input_dir, af_output_dir, af_input_apo_dir, af_output_apo_dir]:
        os.makedirs(dir_path, exist_ok=True)

    # 1. Process YAMLs into AF3 JSON
    process_yaml_files(
        yaml_dir,
        af_input_dir,
        af_input_apo_dir,
        binder_chain=binder_id,
        use_msa_for_af3=use_msa_for_af3,
        afdb_dir=afdb_dir,
        hmmer_path=hmmer_path
    )
    
    # Check if any JSON files were created
    if not list(Path(af_input_dir).glob("*.json")):
        print("No AF3 JSON input files were created. Skipping AlphaFold run.")
        return af_output_dir, af_output_apo_dir, None, None

    # 2. Run AlphaFold on holo state
    print("Running AlphaFold on HOLO state...")
    try:
        _run_af3_subprocess([
            f"{work_dir}/utils/alphafold.sh",
            af_input_dir,
            af_output_dir,
            str(gpu_id),
            alphafold_dir,
            af3_docker_name,
        ])
    except subprocess.CalledProcessError as e:
        print(f"ERROR: AlphaFold HOLO run failed: {e}")

    # 3. Run AlphaFold on apo state
    print("Running AlphaFold on APO state...")
    try:
        _run_af3_subprocess([
            f"{work_dir}/utils/alphafold.sh",
            af_input_apo_dir,
            af_output_apo_dir,
            str(gpu_id),
            alphafold_dir,
            af3_docker_name,
        ])
    except subprocess.CalledProcessError as e:
        print(f"ERROR: AlphaFold APO run failed: {e}")

    print("AlphaFold prediction step completed! Starting post-processing...")
    
    # 4. Convert CIFs to PDBs
    af_pdb_dir = f"{ligandmpnn_dir}/03_af_pdb_success"
    af_pdb_dir_apo = f"{ligandmpnn_dir}/03_af_pdb_apo"

    convert_cif_files_to_pdb(
        af_output_dir, af_pdb_dir, af_dir=True, high_iptm=high_iptm
    )
    
    # Note: We no longer exit early here. Let PyRosetta handle filtering and rejection tracking.
    # This aligns with the Modal pipeline behavior where ALL designs pass through to PyRosetta
    # and rejections are recorded with proper reasons (including low AF3 ipTM).
    if not any(f.endswith(".pdb") for f in os.listdir(af_pdb_dir)):
        print("Warning: No designs passed AF3 i-pTM threshold. Continuing to record rejections...")

    convert_cif_files_to_pdb(af_output_apo_dir, af_pdb_dir_apo, af_dir=True)
    print("Convert CIF files to PDB completed.")
    
    # 5. Calculate holo-apo RMSD
    calculate_holo_apo_rmsd(af_pdb_dir, af_pdb_dir_apo, binder_id)
    print("Calculate holo-apo RMSD completed.")

    return af_output_dir, af_output_apo_dir, af_pdb_dir, af_pdb_dir_apo


def process_yaml_files(
    yaml_dir: str,
    af_input_dir: str,
    af_input_apo_dir: str,
    binder_chain: str = "A",
    use_msa_for_af3: bool = False,
    afdb_dir: str = "",
    hmmer_path: str = ""
):
    """
    Process Boltz output YAML files into AlphaFold3 JSON input format.
    Handles multiple chain types (protein, RNA, DNA, ligand).
    NOTE: This is the consolidated version, removing the duplicate in the original file.
    """
    for yaml_path in Path(yaml_dir).glob("*.yaml"):
        print(f"Processing YAML: {yaml_path}")
        name = os.path.basename(yaml_path).split(".yaml")[0]
        with open(yaml_path) as file:
            yaml_data = yaml.safe_load(file)

        # 1. Parse all component types
        protein_entries = []
        rna_entries = []
        dna_entries = []
        ligand_entries = []
        metal_entries = []
        binder_protein_entry = None
        
        for i, seq in enumerate(yaml_data["sequences"]):
            if "protein" in seq:
                ids = seq["protein"].get("id", ["?"])
                if binder_chain in ids:
                    binder_protein_entry = seq["protein"]
                else:
                    protein_entries.append((i, seq["protein"]))
            elif "ligand" in seq:
                entry = seq["ligand"]
                if entry.get("smiles"):
                    ligand_entries.append(entry)
                elif entry.get("ccd"):
                    metal_entries.append(entry)
            elif "rna" in seq:
                rna_entries.append(seq["rna"])
            elif "dna" in seq:
                dna_entries.append(seq["dna"])

        # 2. Collect protein sequence and MSA information
        all_protein_ids, all_protein_seqs, all_protein_msas = [], [], []
        modification_ls, modification_chain, query_seqs = [], [], []

        # Target proteins first
        for _, p in protein_entries:
            all_protein_ids.append(p["id"][0])
            all_protein_seqs.append(p["sequence"])
            all_protein_msas.append(p.get("msa", "empty"))
            query_seqs.append(p["sequence"])
            modification_ls.append(p.get("modifications", []))
            modification_chain.append(p["id"][0])
            
        # Binder protein last
        if binder_protein_entry:
            all_protein_ids.append(binder_protein_entry["id"][0])
            all_protein_seqs.append(binder_protein_entry["sequence"])
            all_protein_msas.append(binder_protein_entry.get("msa", "empty"))
            query_seqs.append(binder_protein_entry["sequence"])
            modification_ls.append(binder_protein_entry.get("modifications", []))
            modification_chain.append(binder_protein_entry["id"][0])
        
        # 3. Process MSAs for AF3 and Handle RNA MSA Generation
        processed_msas = []
        for target_seq, msa_path, chain_id, query_seq in zip(
            all_protein_seqs, all_protein_msas, all_protein_ids, query_seqs
        ):
            if chain_id == binder_chain or (msa_path == "empty" and not use_msa_for_af3):
                # For binder or explicitly empty MSA, use query sequence only
                # Note: Trailing \n is required for proper FASTA format (matches Modal pipeline)
                processed_msas.append(f">query\n{query_seq}\n")
                continue

            msa_file = None
            if msa_path:
                msa_file = msa_path.replace(".npz", ".a3m")
                
                # If path exists, format it
                if os.path.exists(msa_file):
                    protein_msa = extract_sequences_and_format(
                        msa_file, replace_query_seq=True, query_seq=query_seq
                    )
                    processed_msas.append(protein_msa)
                elif use_msa_for_af3:
                     # Best effort: process MSA on-the-fly (assumes sequence is protein)
                    try:
                        msa_npz_path = process_msa(
                            chain_id=chain_id,
                            sequence=target_seq,
                            msa_dir=Path(yaml_dir),
                        )
                        msa_a3m_path = msa_npz_path.with_suffix('.a3m')
                        protein_msa = extract_sequences_and_format(
                            str(msa_a3m_path), replace_query_seq=True, query_seq=query_seq
                        )
                        processed_msas.append(protein_msa)
                    except Exception as e:
                        print(f"WARNING: Failed to get or make MSA for {chain_id}: {e}")
                        processed_msas.append(f">query\n{query_seq}\n")
                else:
                    processed_msas.append(f">query\n{query_seq}\n")
            else:
                processed_msas.append(f">query\n{query_seq}\n")
                

        # RNA MSA generation (needed only if there are RNA entries)
        rna_processed_msas = []
        if rna_entries:
            for rna_entry in rna_entries:
                rna_seq = rna_entry["sequence"]
                try:
                    rna_a3m = af3_generate_rna_msa(rna_seq, AF3_DATABASE_SETTINGS, afdb_dir, hmmer_path)
                    rna_processed_msas.append(rna_a3m)
                except Exception as e:
                    print(f"WARNING: Failed to generate RNA MSA for {rna_entry['id'][0]}: {e}")
                    rna_processed_msas.append(f">query\n{rna_seq}\n")
        
        # 4. Handle Templates
        all_protein_templates = []
        if "templates" in yaml_data:
            for template_entry in yaml_data["templates"]:
                template_path = template_entry.get("cif") or template_entry.get("pdb")
                template_chain_id = template_entry.get("chain_id")
                
                # Assuming single chain alignment for simplicity here
                if template_path and template_chain_id:
                    for i, (seq, chain) in enumerate(zip(all_protein_seqs, all_protein_ids)):
                        if chain == template_chain_id: # Template applies to this chain
                             try:
                                template_json = get_cif_alignment_json(seq, template_path, template_chain_id)
                                all_protein_templates.append({'chain_id': chain, 'template_json': template_json})
                             except Exception as e:
                                print(f"WARNING: Template processing failed for chain {chain} from {template_path}: {e}")

        # 5. Build JSON (HOLO)
        json_result = {
            "name": name, "sequences": [], "modelSeeds": [1], "dialect": "alphafold3", "version": 1,
        }
        # Proteins
        for i, (seq, chain) in enumerate(zip(all_protein_seqs, all_protein_ids)):
            matched_template_json = [tpl['template_json'] for tpl in all_protein_templates if tpl['chain_id'] == chain]
            
            protein_entry = {
                "protein": {
                    "id": chain,
                    "sequence": seq,
                    "pairedMsa": processed_msas[i],
                    "unpairedMsa": processed_msas[i],
                    "templates": matched_template_json,
                }
            }
            if modification_ls[i]:
                protein_entry["protein"]["modifications"] = [
                    {"ptmType": item.get("ccd") or item.get("ptmType"), "ptmPosition": item.get("position") or item.get("ptmPosition")}
                    for item in modification_ls[i]
                ]
            json_result["sequences"].append(protein_entry)
            
        # RNA/DNA - only add if sequence is valid (Modal pattern)
        for i, rna in enumerate(rna_entries):
            if _is_valid_value(rna.get("sequence")):
                json_result["sequences"].append({"rna": {"id": rna["id"][0], "sequence": rna["sequence"], "unpairedMsa": rna_processed_msas[i]}})
        for dna in dna_entries:
            if _is_valid_value(dna.get("sequence")):
                json_result["sequences"].append({"dna": {"id": dna["id"][0], "sequence": dna["sequence"]}})
             
        # Ligands - only add if smiles/ccd is valid (Modal pattern)
        for lig in ligand_entries:
            if _is_valid_value(lig.get("smiles")):
                json_result["sequences"].append({"ligand": {"id": lig["id"][0], "smiles": lig["smiles"]}})
        for metal in metal_entries:
            if _is_valid_value(metal.get("ccd")):
                json_result["sequences"].append({"ligand": {"id": metal["id"][0], "ccdCodes": [metal["ccd"]]}})


        # 6. Build JSON (APO) - only the binder
        json_result_apo = {
            "name": name, "sequences": [], "modelSeeds": [1], "dialect": "alphafold3", "version": 1,
        }
        if binder_protein_entry:
            i_binder = len(all_protein_ids) - 1
            seq = all_protein_seqs[i_binder]
            chain = all_protein_ids[i_binder]
            matched_template_json = [tpl['template_json'] for tpl in all_protein_templates if tpl['chain_id'] == chain]
            
            protein_entry = {
                "protein": {
                    "id": chain,
                    "sequence": seq,
                    "pairedMsa": processed_msas[i_binder],
                    "unpairedMsa": processed_msas[i_binder],
                    "templates": matched_template_json,
                }
            }
            if modification_ls[i_binder]:
                protein_entry["protein"]["modifications"] = [
                    {"ptmType": item.get("ccd") or item.get("ptmType"), "ptmPosition": item.get("position") or item.get("ptmPosition")}
                    for item in modification_ls[i_binder]
                ]
            json_result_apo["sequences"].append(protein_entry)

        # 7. Write files
        with open(os.path.join(af_input_dir, f"{name}.json"), "w") as f:
            json.dump(json_result, f, indent=4)

        if json_result_apo["sequences"]:
            with open(os.path.join(af_input_apo_dir, f"{name}.json"), "w") as f:
                json.dump(json_result_apo, f, indent=4)
        else:
             print(f"Skipping APO JSON generation for {name}: No binder protein found.")


def csv_row_to_yaml(row: Dict, binder_chain: str = "A") -> Dict:
    """
    Convert a row from design_stats.csv or best_designs.csv to YAML format
    compatible with process_yaml_files.
    
    This bridges the new CSV-based output structure with the existing AF3 pipeline.
    
    Args:
        row: Dictionary representing a CSV row (from pandas df.to_dict('records'))
        binder_chain: Chain ID for the binder (default "A")
    
    Returns:
        Dictionary in Boltz YAML format that process_yaml_files can handle
    """
    sequences = []
    
    # 1. Add binder sequence
    binder_entry = {
        "protein": {
            "id": [binder_chain],
            "sequence": row.get("binder_sequence", ""),
            "msa": row.get("msa_mode", "empty"),
        }
    }
    if row.get("cyclic"):
        binder_entry["protein"]["cyclic"] = True
    sequences.append(binder_entry)
    
    # 2. Add target sequences - handle both JSON-encoded and plain string formats
    target_seqs = row.get("target_seqs", "{}")
    if isinstance(target_seqs, str):
        # Try JSON first, fall back to plain sequence string
        try:
            target_seqs = json.loads(target_seqs)
        except json.JSONDecodeError:
            # Plain sequence string - assume it's chain B target
            if target_seqs:
                target_seqs = {"B": target_seqs}
            else:
                target_seqs = {}
    
    for chain_id, seq in target_seqs.items():
        target_entry = {
            "protein": {
                "id": [chain_id],
                "sequence": seq,
                "msa": row.get("msa_mode", "empty"),
            }
        }
        sequences.append(target_entry)
    
    # 3. Add ligand if present (with proper NaN validation - Modal pattern)
    ligand_smiles = row.get("ligand_smiles")
    ligand_ccd = row.get("ligand_ccd")
    if _is_valid_value(ligand_smiles):
        # Find next available chain letter
        used_chains = {binder_chain} | set(target_seqs.keys())
        ligand_chain = next(c for c in "DEFGHIJKLMNOPQRSTUVWXYZ" if c not in used_chains)
        sequences.append({
            "ligand": {
                "id": [ligand_chain],
                "smiles": ligand_smiles
            }
        })
    elif _is_valid_value(ligand_ccd):
        used_chains = {binder_chain} | set(target_seqs.keys())
        ligand_chain = next(c for c in "DEFGHIJKLMNOPQRSTUVWXYZ" if c not in used_chains)
        sequences.append({
            "ligand": {
                "id": [ligand_chain],
                "ccd": ligand_ccd
            }
        })
    
    # 4. Add nucleic acid if present (with proper NaN validation - Modal pattern)
    nucleic_seq = row.get("nucleic_seq")
    nucleic_type = row.get("nucleic_type")
    if _is_valid_value(nucleic_seq) and _is_valid_value(nucleic_type):
        used_chains = {binder_chain} | set(target_seqs.keys())
        if _is_valid_value(ligand_smiles) or _is_valid_value(ligand_ccd):
            used_chains.add(next(c for c in "DEFGHIJKLMNOPQRSTUVWXYZ" if c not in used_chains))
        nucleic_chain = next(c for c in "DEFGHIJKLMNOPQRSTUVWXYZ" if c not in used_chains)
        sequences.append({
            nucleic_type: {
                "id": [nucleic_chain],
                "sequence": nucleic_seq
            }
        })
    
    yaml_data = {"sequences": sequences}
    
    # 5. Add template info if present (for AF3)
    template_path = row.get("template_path")
    if template_path and str(template_path) not in ('nan', ''):
        template_mapping = row.get("template_mapping")
        chain_id = None
        if template_mapping and str(template_mapping) not in ('nan', ''):
            chain_id = str(template_mapping).split(":")[0]
        yaml_data["templates"] = [{
            "cif": template_path,
            "chain_id": chain_id
        }]
    
    return yaml_data


def generate_yamls_from_csv(
    csv_path: str,
    output_dir: str,
    binder_chain: str = "A"
) -> int:
    """
    Generate YAML files from a design stats CSV for AF3 processing.
    
    This function reads the CSV produced by the new output structure and
    creates temporary YAML files that the existing process_yaml_files 
    function can consume.
    
    Args:
        csv_path: Path to design_stats.csv or best_designs.csv
        output_dir: Directory to write YAML files to
        binder_chain: Chain ID for the binder (default "A")
    
    Returns:
        Number of YAML files generated
    """
    os.makedirs(output_dir, exist_ok=True)
    
    df = pd.read_csv(csv_path)
    count = 0
    
    for _, row in df.iterrows():
        design_id = row.get("design_id", f"design_{count}")
        yaml_data = csv_row_to_yaml(row.to_dict(), binder_chain)
        
        yaml_path = os.path.join(output_dir, f"{design_id}.yaml")
        with open(yaml_path, "w") as f:
            yaml.dump(yaml_data, f, default_flow_style=False)
        
        count += 1
    
    print(f"Generated {count} YAML files from {csv_path}")
    return count


def run_alphafold_step_from_csv(
    csv_path: str,
    alphafold_dir: str,
    af3_docker_name: str,
    af3_database_settings: str,
    hmmer_path: str,
    ligandmpnn_dir: str,
    work_dir: str,
    binder_id: str = "A",
    gpu_id: int = 0,
    high_iptm: bool = False,
    use_msa_for_af3: bool = False,
):
    """
    Run AlphaFold validation using a CSV file as input.
    
    This is a wrapper around run_alphafold_step that first converts
    CSV data to YAML format.
    
    Args:
        csv_path: Path to design_stats.csv or best_designs.csv
        ... (other args same as run_alphafold_step)
    
    Returns:
        Same as run_alphafold_step: (af_output_dir, af_output_apo_dir, af_pdb_dir, af_pdb_dir_apo)
    """
    # Create temporary YAML directory
    yaml_dir = os.path.join(ligandmpnn_dir, "tmp_yamls_for_af3")
    
    # Generate YAMLs from CSV
    count = generate_yamls_from_csv(csv_path, yaml_dir, binder_chain=binder_id)
    
    if count == 0:
        print("No designs found in CSV. Skipping AlphaFold step.")
        return None, None, None, None
    
    # Run the standard AlphaFold step with generated YAMLs
    result = run_alphafold_step(
        yaml_dir=yaml_dir,
        alphafold_dir=alphafold_dir,
        af3_docker_name=af3_docker_name,
        af3_database_settings=af3_database_settings,
        hmmer_path=hmmer_path,
        ligandmpnn_dir=ligandmpnn_dir,
        work_dir=work_dir,
        binder_id=binder_id,
        gpu_id=gpu_id,
        high_iptm=high_iptm,
        use_msa_for_af3=use_msa_for_af3,
    )
    
    # Clean up temporary YAMLs
    shutil.rmtree(yaml_dir, ignore_errors=True)
    
    return result