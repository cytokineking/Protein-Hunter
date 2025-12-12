import copy
import os
import random
import sys
import shutil
from collections import defaultdict
from multiprocessing import Process, Queue, Manager
from pathlib import Path
from typing import Dict
import numpy as np
import pandas as pd
import torch
import yaml

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from LigandMPNN.wrapper import LigandMPNNWrapper

from boltz_ph.constants import CHAIN_TO_NUMBER, UNIFIED_DESIGN_COLUMNS
from utils.metrics import get_CA_and_sequence # Used implicitly in design.py
from utils.convert import calculate_holo_apo_rmsd, convert_cif_files_to_pdb, convert_cif_to_pdb
from utils.ipsae_utils import calculate_ipsae_from_boltz_output
from utils.csv_utils import append_to_csv_safe, update_csv_row


from boltz_ph.model_utils import (
    analyze_template_structure,
    binder_binds_contacts,
    clean_memory,
    Colors,
    design_sequence,
    format_gap_error,
    get_boltz_model,
    get_cif,
    load_canonicals,
    plot_from_pdb,
    plot_run_metrics,
    print_target_analysis,
    process_msa,
    run_prediction,
    sample_seq,
    save_pdb,
    shallow_copy_tensor_dict,
    smart_split,
    validate_hotspots,
)


def _reorder_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Reorder DataFrame columns to match UNIFIED_DESIGN_COLUMNS (Modal alignment)."""
    ordered = [c for c in UNIFIED_DESIGN_COLUMNS if c in df.columns]
    extra = [c for c in df.columns if c not in UNIFIED_DESIGN_COLUMNS]
    return df[ordered + extra]

class InputDataBuilder:
    """Handles parsing command-line arguments and constructing the base Boltz input data dictionary."""

    def __init__(self, args):
        self.args = args
        # New output structure: results_{name}/
        self.save_dir = (
            args.save_dir if args.save_dir else f"./results_{args.name}"
        )
        # Flat designs directory (replaces 0_protein_hunter_design/run_X/)
        self.designs_dir = f"{self.save_dir}/designs"
        os.makedirs(self.designs_dir, exist_ok=True)
        # Keep for backward compatibility with MSA generation
        self.protein_hunter_save_dir = self.designs_dir
        
        # Template analysis results (populated by _analyze_templates)
        self._template_analysis = {}
        self._auto_extracted_seqs = []
        
        # Hotspot traceability (populated by _validate_and_convert_hotspots)
        self.contact_residues_canonical = ""
        self.contact_residues_auth = ""  # Original auth values if --use_auth_numbering
        self.template_first_residues = {}  # chain_id -> first auth residue
        
        # Target sequences actually used (populated by _process_sequence_inputs)
        # This captures auto-extracted sequences for CSV output/AF3 validation
        self.target_seqs_used = ""

    def _analyze_templates(self) -> dict:
        """
        Analyze templates for sequence extraction, gap detection, and auth→canonical mapping.
        
        Handles multi-chain templates correctly: when a single template file has multiple
        chain IDs specified (e.g., --template_cif_chain_id "A,B"), analyzes ALL chains.
        
        Returns:
            Dict mapping template_chain_id to analysis results from analyze_template_structure()
        
        Raises:
            SystemExit: If gaps are detected and --protein_seqs not provided
        """
        a = self.args
        
        if not a.template_path:
            return {}
        
        template_path_list = smart_split(a.template_path)
        template_cif_chain_id_list = (
            smart_split(a.template_cif_chain_id)
            if a.template_cif_chain_id
            else []
        )
        
        if not template_cif_chain_id_list:
            print("⚠ Warning: No --template_cif_chain_id specified, defaulting to chain 'A'")
            template_cif_chain_id_list = ["A"]
        
        all_analysis = {}
        auto_seqs = []
        
        # Get the first template file (handles PDB codes, downloads, etc.)
        # For multi-chain from single file, we analyze all chains from this file
        template_file = get_cif(template_path_list[0]) if template_path_list else None
        
        if not template_file:
            return {}
        
        # Store template chain IDs in order (for hotspot index mapping)
        self._template_chain_ids = template_cif_chain_id_list
        
        try:
            # Analyze ALL specified chains at once
            analysis = analyze_template_structure(template_file, template_cif_chain_id_list)
            
            for i, cif_chain in enumerate(template_cif_chain_id_list):
                chain_analysis = analysis.get(cif_chain, {})
                
                if not chain_analysis:
                    print(f"⚠ Warning: Chain '{cif_chain}' not found in template {template_file}")
                    auto_seqs.append("")
                    continue
                
                # Check for gaps
                if chain_analysis.get('has_gaps', False):
                    if not a.protein_seqs:
                        # Gaps detected and no --protein_seqs provided - error
                        error_msg = format_gap_error(cif_chain, chain_analysis)
                        print(error_msg)
                        sys.exit(1)
                    else:
                        # User provided sequences, just warn
                        print(f"⚠ Warning: Gaps detected in chain {cif_chain}, using provided --protein_seqs")
                
                # Store analysis keyed by TEMPLATE chain ID (A, B, etc.)
                # This matches the Modal pipeline approach
                all_analysis[cif_chain] = {
                    **chain_analysis,
                    'template_file': template_file,
                    'template_cif_chain': cif_chain,
                    'chain_index': i,  # Track order for hotspot mapping
                }
                
                # Auto-extract sequence
                auto_seqs.append(chain_analysis.get('sequence', ''))
                
        except Exception as e:
            print(f"⚠ Warning: Could not analyze template {template_file}: {e}")
            for _ in template_cif_chain_id_list:
                auto_seqs.append("")
        
        self._template_analysis = all_analysis
        self._auto_extracted_seqs = auto_seqs
        
        if auto_seqs and any(s for s in auto_seqs):
            print(f"✓ Analyzed {len([s for s in auto_seqs if s])} chain(s) from template")
        
        return all_analysis

    def _process_sequence_inputs(self):
        """
        Parses and groups protein sequences and MSAs from command line arguments.
        Ensures protein_seqs_list and protein_msas_list are aligned and padded.
        
        Now supports auto-extraction from templates when --protein_seqs is not provided.
        """
        a = self.args
        
        # First analyze templates to potentially auto-extract sequences
        if not self._template_analysis:
            self._analyze_templates()
        
        # Determine sequences: user-provided takes precedence over auto-extracted
        if a.protein_seqs:
            protein_seqs_list = smart_split(a.protein_seqs)
        elif self._auto_extracted_seqs:
            protein_seqs_list = self._auto_extracted_seqs
            if protein_seqs_list and any(s for s in protein_seqs_list):
                print(f"✓ Auto-extracted {len([s for s in protein_seqs_list if s])} sequence(s) from template(s)")
        else:
            protein_seqs_list = []

        # Store actual sequences used for CSV output and AF3 validation
        # This ensures auto-extracted sequences are available downstream
        self.target_seqs_used = ":".join(protein_seqs_list)

        # Handle special "empty" case: --protein_msas "empty" applies to all seqs
        if a.msa_mode == "single":
            protein_msas_list = ["empty"] * len(protein_seqs_list)
        elif a.msa_mode == "mmseqs":
            protein_msas_list = ["mmseqs"] * len(protein_seqs_list)
        else:
            raise ValueError(f"Invalid msa_mode: {a.msa_mode}")

        return protein_seqs_list, protein_msas_list

    def _validate_and_convert_hotspots(self, protein_chain_ids: list[str]) -> tuple[str, dict[str, list[int]], dict[str, list[int]]]:
        """
        Validate hotspots and convert from auth to canonical numbering if needed.
        
        Args:
            protein_chain_ids: List of internal protein chain IDs (e.g., ['B', 'C'])
        
        Returns:
            Tuple of:
            - contact_residues_canonical: Converted contact_residues string for Boltz
            - hotspots_per_chain: Dict mapping internal_chain_id to list of canonical hotspot positions
            - auth_hotspots_per_chain: Dict mapping internal_chain_id to list of auth hotspot positions
        
        Raises:
            SystemExit: If hotspot validation fails
        """
        a = self.args
        
        if not a.contact_residues or not a.contact_residues.strip():
            self.contact_residues_canonical = ""
            self.contact_residues_auth = ""
            return "", {}, {}
        
        residues_chains = a.contact_residues.split("|")
        converted_chains = []
        auth_chains = []  # Store original auth values
        hotspots_per_chain = {}
        auth_hotspots_per_chain = {}  # Store auth positions for display
        
        use_auth = getattr(a, 'use_auth_numbering', False)
        
        # Get template chain IDs (for mapping hotspot index → template chain)
        template_chain_ids = getattr(self, '_template_chain_ids', [])
        
        # Store template first residues for CSV output (keyed by template chain ID)
        for chain_id, analysis in self._template_analysis.items():
            auth_range = analysis.get('auth_range', (1, 1))
            self.template_first_residues[chain_id] = auth_range[0]
        
        for i, residues_str in enumerate(residues_chains):
            residues_str = residues_str.strip()
            if not residues_str:
                converted_chains.append("")
                auth_chains.append("")
                continue
            
            # Internal chain ID for Boltz (B, C, D, etc.)
            internal_chain_id = protein_chain_ids[i] if i < len(protein_chain_ids) else chr(ord('B') + i)
            
            # Template chain ID for analysis lookup (A, B, etc. from --template_cif_chain_id)
            template_chain_id = template_chain_ids[i] if i < len(template_chain_ids) else None
            
            positions = [int(r.strip()) for r in residues_str.split(",") if r.strip()]
            
            if not positions:
                converted_chains.append("")
                auth_chains.append("")
                continue
            
            # Get chain analysis using TEMPLATE chain ID (not internal chain ID)
            chain_analysis = self._template_analysis.get(template_chain_id, {}) if template_chain_id else {}
            seq_length = len(chain_analysis.get('sequence', ''))
            auth_residue_set = chain_analysis.get('auth_residues', set())
            auth_range = chain_analysis.get('auth_range', (1, seq_length))
            auth_to_can = chain_analysis.get('auth_to_canonical', {})
            can_to_auth = chain_analysis.get('canonical_to_auth', {})
            
            if use_auth and chain_analysis:
                # Validate in auth numbering
                is_valid, error_msg = validate_hotspots(
                    positions,
                    seq_length,
                    template_chain_id,  # Use template chain ID for error messages
                    use_auth_numbering=True,
                    auth_residue_set=auth_residue_set,
                    auth_range=auth_range,
                )
                
                if not is_valid:
                    print(error_msg)
                    sys.exit(1)
                
                # Convert auth to canonical
                canonical_positions = [auth_to_can.get(p, p) for p in positions]
                hotspots_per_chain[internal_chain_id] = canonical_positions
                auth_hotspots_per_chain[internal_chain_id] = positions  # Store original auth
                converted_chains.append(",".join(str(p) for p in canonical_positions))
                auth_chains.append(",".join(str(p) for p in positions))
            else:
                # Validate in canonical numbering
                if seq_length > 0:
                    is_valid, error_msg = validate_hotspots(
                        positions,
                        seq_length,
                        template_chain_id or internal_chain_id,
                        use_auth_numbering=False,
                    )
                    
                    if not is_valid:
                        print(error_msg)
                        sys.exit(1)
                
                hotspots_per_chain[internal_chain_id] = positions
                # If not using auth numbering, compute auth positions if we have the mapping
                if can_to_auth:
                    auth_positions = [can_to_auth.get(p, p) for p in positions]
                    auth_hotspots_per_chain[internal_chain_id] = auth_positions
                    auth_chains.append(",".join(str(p) for p in auth_positions))
                else:
                    auth_hotspots_per_chain[internal_chain_id] = positions  # Use canonical as fallback
                    auth_chains.append("")
                converted_chains.append(",".join(str(p) for p in positions))
        
        self.contact_residues_canonical = "|".join(converted_chains)
        self.contact_residues_auth = "|".join(auth_chains) if use_auth else ""
        
        return self.contact_residues_canonical, hotspots_per_chain, auth_hotspots_per_chain


    def build(self):
        """
        Constructs the base Boltz input data dictionary (sequences, templates, constraints).
        """
        a = self.args

        if a.mode == "unconditional":
            data = self._build_unconditional_data()
            pocket_conditioning = False
        else:
            data, pocket_conditioning = self._build_conditional_data()

        # Sort sequences by chain ID for consistent processing
        data["sequences"] = sorted(
            data["sequences"], key=lambda entry: list(entry.values())[0]["id"][0]
        )

        print("Data dictionary:\n", data)
        return data, pocket_conditioning


    def _build_unconditional_data(self):
        """Constructs data for unconditional binder design."""
        data = {
            "sequences": [
                {
                    "protein": {
                        "id": ["A"],
                        "sequence": "X",
                        "msa": "empty",
                    }
                }
            ]
        }
        return data


    def _build_conditional_data(self):
        """Constructs data for conditioned design (binder + target + optional non-protein)."""
        a = self.args
        protein_seqs_list, protein_msas_list = (
            self._process_sequence_inputs()
        )
        sequences = []
        
        # Assign chain IDs to proteins first
        protein_chain_ids = [chr(ord('B') + i) for i in range(len(protein_seqs_list))]
        
        # Find next available chain letter
        next_chain_idx = len(protein_chain_ids)
        
        ligand_chain_id = None
        if a.ligand_smiles or a.ligand_ccd:
            ligand_chain_id = chr(ord('B') + next_chain_idx)
            next_chain_idx += 1
            
        nucleic_chain_id = None
        if a.nucleic_seq:
            nucleic_chain_id = chr(ord('B') + next_chain_idx)
            next_chain_idx += 1

        # Validate and convert hotspots (auth → canonical if needed)
        contact_residues_canonical, hotspots_per_chain, auth_hotspots_per_chain = self._validate_and_convert_hotspots(protein_chain_ids)

        # Step 1: Determine canonical MSA for each unique target sequence
        seq_to_indices = defaultdict(list)
        for idx, seq in enumerate(protein_seqs_list):
            if seq:
                seq_to_indices[seq].append(idx)
        
        seq_to_final_msa = {}
        for seq, idx_list in seq_to_indices.items():
            chosen_msa = next(
                (
                    protein_msas_list[i]
                    for i in idx_list
                ),
                None
            )
            chosen_msa = chosen_msa if chosen_msa is not None else ""

            if chosen_msa == "mmseqs":
                pid = protein_chain_ids[idx_list[0]]
                msa_value = process_msa(pid, seq, Path(self.protein_hunter_save_dir))
                seq_to_final_msa[seq] = str(msa_value)
            elif chosen_msa == "empty":
                seq_to_final_msa[seq] = "empty"
            else:
                raise ValueError(f"Invalid msa_mode: {a.msa_mode}")

        # Step 2: Build sequences list for target proteins
        for i, (seq, msa) in enumerate(zip(protein_seqs_list, protein_msas_list)):
            if not seq:
                continue
            pid = protein_chain_ids[i]
            final_msa = seq_to_final_msa.get(seq, "empty")
            sequences.append(
                {
                    "protein": {
                        "id": [pid],
                        "sequence": seq,
                        "msa": final_msa,
                    }
                }
            )

        # Step 3: Add binder chain
        sequences.append(
            {
                "protein": {
                    "id": ["A"], # Hardcoded 'A'
                    "sequence": "X",
                    "msa": "empty",
                    "cyclic": a.cyclic,
                }
            }
        )

        # Step 4: Add ligands/nucleic acids
        if a.ligand_smiles:
            sequences.append(
                {"ligand": {"id": [ligand_chain_id], "smiles": a.ligand_smiles}}
            )
        elif a.ligand_ccd:
            sequences.append({"ligand": {"id": [ligand_chain_id], "ccd": a.ligand_ccd}})
        if a.nucleic_seq:
            sequences.append(
                {a.nucleic_type: {"id": [nucleic_chain_id], "sequence": a.nucleic_seq}}
            )

        # Step 5: Add templates
        templates = self._build_templates(protein_chain_ids)

        data = {"sequences": sequences}
        if templates:
            data["templates"] = templates

        # Step 6: Add constraints (pocket conditioning) - use converted canonical positions
        pocket_conditioning = bool(contact_residues_canonical and contact_residues_canonical.strip())
        if pocket_conditioning:
            contacts = []
            residues_chains = contact_residues_canonical.split("|")
            for i, residues_chain in enumerate(residues_chains):
                residues = residues_chain.split(",")
                contacts.extend([
                    [protein_chain_ids[i], int(res)]
                    for res in residues
                    if res.strip() != ""
                ])
            constraints = [{"pocket": {"binder": "A", "contacts": contacts}}]
            data["constraints"] = constraints

        # Step 7: Print target sequence visualization (if we have template analysis)
        if self._template_analysis and protein_seqs_list:
            template_source = f"template ({a.template_path})" if a.template_path else ""
            
            # Remap hotspots from internal chain ID (B, C) to template chain ID (A, B)
            # for display purposes
            template_chain_ids = getattr(self, '_template_chain_ids', [])
            canonical_hotspots_for_display = {}
            auth_hotspots_for_display = {}
            for i, template_chain_id in enumerate(template_chain_ids):
                internal_chain_id = protein_chain_ids[i] if i < len(protein_chain_ids) else chr(ord('B') + i)
                if internal_chain_id in hotspots_per_chain:
                    canonical_hotspots_for_display[template_chain_id] = hotspots_per_chain[internal_chain_id]
                if internal_chain_id in auth_hotspots_per_chain:
                    auth_hotspots_for_display[template_chain_id] = auth_hotspots_per_chain[internal_chain_id]
            
            print_target_analysis(
                self._template_analysis,
                canonical_hotspots_for_display,
                auth_hotspots_for_display,
                template_source,
            )

        return data, pocket_conditioning

    def _build_templates(self, protein_chain_ids):
        """
        Constructs the list of template dictionaries.
        
        Handles multi-chain templates correctly: when a single template file has
        multiple chain IDs specified (e.g., --template_cif_chain_id "A,B"), creates
        a template entry for each protein chain using the same file.
        """
        a = self.args
        templates = []
        if a.template_path:
            template_path_list = smart_split(a.template_path)
            template_cif_chain_id_list = (
                smart_split(a.template_cif_chain_id)
                if a.template_cif_chain_id
                else []
            )
            
            num_proteins = len(protein_chain_ids)
            num_template_chains = len(template_cif_chain_id_list)
            
            # For multi-chain from single file: reuse the same template file for each chain
            # This is the key fix: iterate over template_cif_chain_id_list, not template_path_list
            for i in range(num_proteins):
                # Get the template file: if there's only one file but multiple chains,
                # reuse the same file for all chains
                if i < len(template_path_list):
                    template_file_path = template_path_list[i]
                elif len(template_path_list) == 1:
                    # Single file with multiple chains - reuse it
                    template_file_path = template_path_list[0]
                else:
                    template_file_path = ""
                
                if not template_file_path:
                    continue  # Skip if no template path
                
                # Get the cif_chain_id for this protein
                cif_chain = template_cif_chain_id_list[i] if i < num_template_chains else ""
                if not cif_chain:
                    continue  # Skip if no cif chain specified
                    
                template_file = get_cif(template_file_path)
                
                t_block = (
                    {"cif": template_file}
                    if template_file.endswith(".cif")
                    else {"pdb": template_file}
                )
                
                t_block["chain_id"] = protein_chain_ids[i]  # Internal chain ID (e.g., 'B')
                t_block["cif_chain_id"] = cif_chain  # Template chain ID (e.g., 'A')
                
                templates.append(t_block)
                
        return templates

class ProteinHunter_Boltz:
    """
    Core class to manage the protein design pipeline, including Boltz structure
    prediction, LigandMPNN design, cycle optimization, and downstream validation.
    """

    def __init__(self, args):
        self.args = args
        self.device = f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu"
        print("Using device:", self.device)

        # 1. Initialize shared resources (persists across all runs)
        self.ccd_path = Path(args.ccd_path).expanduser()
        self.ccd_lib = load_canonicals(str(self.ccd_path))
        
        # Initialize the Input Data Builder
        self.data_builder = InputDataBuilder(args)
        
        # 2. Initialize Models
        self.boltz_model = self._load_boltz_model()
        self.designer = LigandMPNNWrapper("./LigandMPNN/run.py")

        # 3. Setup Directories
        self.save_dir = self.data_builder.save_dir
        self.designs_dir = self.data_builder.designs_dir
        self.protein_hunter_save_dir = self.data_builder.protein_hunter_save_dir  # Keep for backward compat
        self.binder_chain = "A"

        print("✅ ProteinHunter_Boltz initialized.")

    def _load_boltz_model(self):
        """Loads and configures the Boltz model."""
        predict_args = {
            "recycling_steps": self.args.recycling_steps,
            "sampling_steps": self.args.diffuse_steps,
            "diffusion_samples": 1,
            "write_confidence_summary": True,
            "write_full_pae": True,  # Enable for ipSAE calculation
            "write_full_pde": False,
            "max_parallel_samples": 1,
        }
        return get_boltz_model(
            checkpoint=self.args.boltz_model_path,
            predict_args=predict_args,
            device=self.device,
            model_version=self.args.boltz_model_version,
            no_potentials=False if self.args.contact_residues else True,
            grad_enabled=self.args.grad_enabled,
        )

    def _run_design_cycle(self, data_cp, run_id, pocket_conditioning):
        """
        Executes a single design run, including Cycle 0 contact filtering and
        the subsequent design/prediction cycles.

        Note: The cycle logic remains highly coupled due to the nature of the
        sequential design process, but relies on modular imports.
        """
        import datetime as dt

        a = self.args
        design_idx = int(run_id)  # run_id is already the design index

        # Initialize run metrics and tracking variables
        best_iptm = float("-inf")
        best_seq = None
        best_structure = None
        best_output = None
        best_pdb_filename = None
        best_cycle_idx = -1
        best_alanine_percentage = None
        best_ipsae = None
        best_plddt = None
        best_iplddt = None
        best_alanine_count = None
        run_metrics = {"run_id": run_id}


        if a.seq =="":
            binder_length = random.randint(
                a.min_protein_length, a.max_protein_length
            )
        else:
            binder_length = len(a.seq)

        if "constraints" in data_cp:
            protein_chain_ids = sorted({chain for chain, _ in data_cp["constraints"][0]["pocket"]["contacts"]})

        # Helper function to update sequence in the data dictionary
        def update_binder_sequence(new_seq):
            for seq_entry in data_cp["sequences"]:
                if (
                    "protein" in seq_entry
                    and self.binder_chain in seq_entry["protein"]["id"]
                ):
                    seq_entry["protein"]["sequence"] = new_seq
                    return
            # Should not happen if data_cp is built correctly
            raise ValueError("Binder chain not found in data dictionary.")

        # Set initial binder sequence
        if a.seq =="":
            initial_seq = sample_seq(
                binder_length, exclude_P=a.exclude_p, frac_X=a.percent_x/100
            )
        else:
            initial_seq = a.seq
        update_binder_sequence(initial_seq)
        print(f"Binder initial sequence length: {binder_length}")

        # --- Cycle 0 structure prediction, with contact filtering check ---
        contact_filter_attempt = 0
        pdb_filename = ""
        structure = None
        output = None

        batch_feats = None  # Will be set in the loop
        while True:
            output, structure, batch_feats = run_prediction(
                data_cp,
                self.binder_chain,
                randomly_kill_helix_feature=a.randomly_kill_helix_feature,
                negative_helix_constant=a.negative_helix_constant,
                boltz_model=self.boltz_model,
                ccd_lib=self.ccd_lib,
                ccd_path=self.ccd_path,
                logmd=a.logmd,
                device=self.device,
                boltz_model_version=a.boltz_model_version,
                pocket_conditioning=pocket_conditioning,
                return_feats=True,
            )
            # New naming: {name}_d{N}_c{M}.pdb in flat designs/ folder
            design_id = f"{a.name}_d{design_idx}_c0"
            pdb_filename = f"{self.designs_dir}/{design_id}.pdb"
            plddts = output["plddt"].detach().cpu().numpy()[0]
            save_pdb(structure, output["coords"], plddts, pdb_filename)

            contact_check_okay = True
            # Use constraints directly - the single source of truth for contact residues
            # This avoids index alignment issues between a.contact_residues and protein_chain_ids
            contacts = data_cp.get("constraints", [{}])[0].get("pocket", {}).get("contacts", [])
            if contacts and not a.no_contact_filter:
                try:
                    # Build chain → residues mapping from constraints (already canonical numbering)
                    chain_to_res = defaultdict(list)
                    for chain_id, res in contacts:
                        chain_to_res[chain_id].append(int(res))
                    
                    # Check each chain's contacts
                    binds = True
                    for chain_id, residues in chain_to_res.items():
                        unique_res = sorted(set(residues))
                        residues_str = ",".join(str(r) for r in unique_res)
                        if not binder_binds_contacts(
                            pdb_filename,
                            self.binder_chain,
                            chain_id,
                            residues_str,
                            cutoff=a.contact_cutoff,
                        ):
                            binds = False
                            break
                    
                    if not binds:
                        print(
                            "❌ Binder does NOT contact required residues after cycle 0. Retrying..."
                        )
                        contact_check_okay = False
                except Exception as e:
                    print(f"WARNING: Could not perform binder-contact check: {e}")
                    contact_check_okay = True  # Fail open

            if contact_check_okay:
                break
            contact_filter_attempt += 1
            if contact_filter_attempt >= a.max_contact_filter_retries:
                print("WARNING: Max retries for contact filtering reached. Proceeding.")
                break

            # Resample initial sequence
            new_seq = sample_seq(binder_length, exclude_P=a.exclude_p, frac_X=a.percent_x/100)
            update_binder_sequence(new_seq)
            clean_memory()

        clean_memory()
        
        # Check for valid output (Boltz may OOM and return incomplete results)
        if "pair_chains_iptm" not in output or "plddt" not in output:
            print(f"ERROR: Boltz returned incomplete output at cycle 0 (likely OOM). Skipping this design.")
            return run_metrics  # Return early with no valid design
        
        # Capture Cycle 0 metrics
        binder_chain_idx = CHAIN_TO_NUMBER[self.binder_chain]
        pair_chains = output["pair_chains_iptm"]
        
        # Calculate i-pTM
        if len(pair_chains) > 1:
            values = [
                (
                    pair_chains[binder_chain_idx][i].detach().cpu().numpy()
                    + pair_chains[i][binder_chain_idx].detach().cpu().numpy()
                )
                / 2.0
                for i in range(len(pair_chains))
                if i != binder_chain_idx
            ]
            cycle_0_iptm = float(np.mean(values) if values else 0.0)
        else:
            cycle_0_iptm = 0.0

        # Calculate ipSAE for cycle 0
        ipsae_result = calculate_ipsae_from_boltz_output(
            output, batch_feats, binder_chain_idx=binder_chain_idx
        )
        cycle_0_ipsae = ipsae_result['ipSAE']

        cycle_0_plddt = float(
            output.get("complex_plddt", torch.tensor([0.0])).detach().cpu().numpy()[0]
        )
        cycle_0_iplddt = float(
            output.get("complex_iplddt", torch.tensor([0.0])).detach().cpu().numpy()[0]
        )

        run_metrics["cycle_0_iptm"] = cycle_0_iptm
        run_metrics["cycle_0_ipsae"] = cycle_0_ipsae
        run_metrics["cycle_0_plddt"] = cycle_0_plddt
        run_metrics["cycle_0_iplddt"] = cycle_0_iplddt
        run_metrics["cycle_0_alanine"] = 0

        print(f"Cycle 0: ipTM={cycle_0_iptm:.3f}, ipSAE={cycle_0_ipsae:.3f}")

        # Write cycle 0 to design_stats.csv
        csv_row = {
            "design_id": design_id,
            "design_num": design_idx,
            "cycle": 0,
            "binder_sequence": initial_seq,
            "binder_length": binder_length,
            "cyclic": a.cyclic,
            "iptm": cycle_0_iptm,
            "ipsae": cycle_0_ipsae,
            "plddt": cycle_0_plddt,
            "iplddt": cycle_0_iplddt,
            "alanine_count": 0,
            "alanine_pct": 0.0,
            "target_seqs": self.data_builder.target_seqs_used,
            "contact_residues": self.data_builder.contact_residues_canonical or a.contact_residues or "",
            "contact_residues_auth": self.data_builder.contact_residues_auth or "",
            "template_first_residue": ",".join(f"{k}:{v}" for k, v in self.data_builder.template_first_residues.items()) if self.data_builder.template_first_residues else "",
            "msa_mode": a.msa_mode,
            # Fields for AF3 reconstruction
            "ligand_smiles": a.ligand_smiles or "",
            "ligand_ccd": a.ligand_ccd or "",
            "nucleic_seq": a.nucleic_seq or "",
            "nucleic_type": a.nucleic_type or "",
            "template_path": a.template_path or "",
            "template_mapping": a.template_cif_chain_id or "",
            "timestamp": dt.datetime.now().isoformat(),
        }
        append_to_csv_safe(Path(self.designs_dir) / "design_stats.csv", csv_row)

        # --- Optimization Cycles ---
        for cycle in range(a.num_cycles):
            print(f"\n--- Run {run_id}, Cycle {cycle + 1} ---")

            # Calculate temperature and bias for the cycle
            cycle_norm = (cycle / (a.num_cycles - 1)) if a.num_cycles > 1 else 0.0
            alpha = a.alanine_bias_start - cycle_norm * (
                a.alanine_bias_start - a.alanine_bias_end
            )

            # 1. Design Sequence
            model_type = (
                "ligand_mpnn"
                if (a.ligand_smiles or a.ligand_ccd or a.nucleic_seq)
                else "soluble_mpnn"
            )
            design_kwargs = {
                "pdb_file": pdb_filename,
                "temperature": a.temperature,
                "chains_to_design": self.binder_chain,
                "omit_AA": f"{a.omit_aa},P" if cycle == 0 else a.omit_aa,
                "bias_AA": f"A:{alpha}" if a.alanine_bias else "",
            }

            seq_str, logits = design_sequence(
                self.designer, model_type, **design_kwargs
            )
            # The output seq_str is a dictionary-like string, we extract the binder chain sequence
            seq = seq_str.split(":")[CHAIN_TO_NUMBER[self.binder_chain]] 

            # Update data_cp with new sequence
            alanine_count = seq.count("A")
            alanine_percentage = (
                alanine_count / binder_length if binder_length != 0 else 0.0
            )
            update_binder_sequence(seq) # Use the helper function

            # 2. Structure Prediction
            output, structure, batch_feats = run_prediction(
                data_cp,
                self.binder_chain,
                seq=seq,
                randomly_kill_helix_feature=False,
                negative_helix_constant=0.0,
                boltz_model=self.boltz_model,
                ccd_lib=self.ccd_lib,
                ccd_path=self.ccd_path,
                logmd=False,
                device=self.device,
                return_feats=True,
            )

            # Check for valid output (Boltz may OOM and return incomplete results)
            if "pair_chains_iptm" not in output or "plddt" not in output:
                print(f"ERROR: Boltz returned incomplete output (likely OOM). Skipping this design.")
                return run_metrics  # Return early with no valid design

            # Calculate ipTM
            current_chain_idx = CHAIN_TO_NUMBER[self.binder_chain]
            pair_chains = output["pair_chains_iptm"]
            if len(pair_chains) > 1:
                values = [
                    (
                        pair_chains[current_chain_idx][i].detach().cpu().numpy()
                        + pair_chains[i][current_chain_idx].detach().cpu().numpy()
                    )
                    / 2.0
                    for i in range(len(pair_chains))
                    if i != current_chain_idx
                ]
                current_iptm = float(np.mean(values) if values else 0.0)
            else:
                current_iptm = 0.0

            # Calculate ipSAE
            ipsae_result = calculate_ipsae_from_boltz_output(
                output, batch_feats, binder_chain_idx=current_chain_idx
            )
            current_ipsae = ipsae_result['ipSAE']

            # 3. Log Metrics and Save PDB (compute pLDDT before best selection)
            curr_plddt = float(
                output.get("complex_plddt", torch.tensor([0.0]))
                .detach()
                .cpu()
                .numpy()[0]
            )
            curr_iplddt = float(
                output.get("complex_iplddt", torch.tensor([0.0]))
                .detach()
                .cpu()
                .numpy()[0]
            )

            # Update best structure - must pass BOTH thresholds:
            # 1. Alanine content <= 20%
            # 2. ipTM >= threshold (default 0.8)
            # 3. pLDDT >= threshold (default 0.8)
            # Tiebreaker: highest ipTM
            passes_alanine = alanine_percentage <= 0.20
            passes_iptm = current_iptm >= a.high_iptm_threshold
            passes_plddt = curr_plddt >= a.high_plddt_threshold
            
            if passes_alanine and passes_iptm and passes_plddt and current_iptm > best_iptm:
                best_iptm = current_iptm
                best_seq = seq
                best_structure = copy.deepcopy(structure)
                best_output = shallow_copy_tensor_dict(output)
                best_cycle_idx = cycle + 1
                best_alanine_percentage = alanine_percentage
                best_ipsae = current_ipsae
                best_alanine_count = alanine_count
                best_plddt = curr_plddt
                best_iplddt = curr_iplddt

            run_metrics[f"cycle_{cycle + 1}_iptm"] = current_iptm
            run_metrics[f"cycle_{cycle + 1}_ipsae"] = current_ipsae
            run_metrics[f"cycle_{cycle + 1}_plddt"] = curr_plddt
            run_metrics[f"cycle_{cycle + 1}_iplddt"] = curr_iplddt
            run_metrics[f"cycle_{cycle + 1}_alanine"] = alanine_count
            run_metrics[f"cycle_{cycle + 1}_seq"] = seq

            # New naming: {name}_d{N}_c{M}.pdb in flat designs/ folder
            design_id = f"{a.name}_d{design_idx}_c{cycle + 1}"
            pdb_filename = f"{self.designs_dir}/{design_id}.pdb"
            plddts = output["plddt"].detach().cpu().numpy()[0]
            save_pdb(structure, output["coords"], plddts, pdb_filename)

            # Write cycle to design_stats.csv
            csv_row = {
                "design_id": design_id,
                "design_num": design_idx,
                "cycle": cycle + 1,
                "binder_sequence": seq,
                "binder_length": binder_length,
                "cyclic": a.cyclic,
                "iptm": current_iptm,
                "ipsae": current_ipsae,
                "plddt": curr_plddt,
                "iplddt": curr_iplddt,
                "alanine_count": alanine_count,
                "alanine_pct": round(alanine_percentage * 100, 2),
                "target_seqs": self.data_builder.target_seqs_used,
                "contact_residues": self.data_builder.contact_residues_canonical or a.contact_residues or "",
                "contact_residues_auth": self.data_builder.contact_residues_auth or "",
                "template_first_residue": ",".join(f"{k}:{v}" for k, v in self.data_builder.template_first_residues.items()) if self.data_builder.template_first_residues else "",
                "msa_mode": a.msa_mode,
                # Fields for AF3 reconstruction
                "ligand_smiles": a.ligand_smiles or "",
                "ligand_ccd": a.ligand_ccd or "",
                "nucleic_seq": a.nucleic_seq or "",
                "nucleic_type": a.nucleic_type or "",
                "template_path": a.template_path or "",
                "template_mapping": a.template_cif_chain_id or "",
                "timestamp": dt.datetime.now().isoformat(),
            }
            append_to_csv_safe(Path(self.designs_dir) / "design_stats.csv", csv_row)

            clean_memory()

            print(
                f"ipTM: {current_iptm:.3f} ipSAE: {current_ipsae:.3f} pLDDT: {curr_plddt:.2f} iPLDDT: {curr_iplddt:.2f} Ala: {alanine_count}"
            )

            # Note: high_iptm_yaml and high_iptm_pdb folders are eliminated.
            # All designs stream to designs/ and downstream tools query the CSV.
        # Build best_pdb_filename from best cycle info
        if best_cycle_idx >= 0:
            best_design_id = f"{a.name}_d{design_idx}_c{best_cycle_idx}"
            best_pdb_filename = f"{self.designs_dir}/{best_design_id}.pdb"
        else:
            best_pdb_filename = None

        # End of cycle visualization
        if best_structure is not None and best_pdb_filename and a.plot:
            plot_from_pdb(best_pdb_filename)
        elif best_structure is None:
            print(
                f"\nNo structure was generated for run {run_id} (no cycle passed all criteria: "
                f"alanine <= 20%, ipTM >= {a.high_iptm_threshold}, pLDDT >= {a.high_plddt_threshold})."
            )

        # Finalize best metrics for CSV
        # best_plddt and best_iplddt are now stored during selection
        if best_structure is not None and best_plddt is not None:
            run_metrics["best_iptm"] = float(best_iptm)
            run_metrics["best_cycle"] = best_cycle_idx
            run_metrics["best_plddt"] = float(best_plddt)
            run_metrics["best_iplddt"] = float(best_iplddt) if best_iplddt is not None else float("nan")
            run_metrics["best_seq"] = best_seq
            run_metrics["best_ipsae"] = best_ipsae
            run_metrics["best_alanine_count"] = best_alanine_count
            run_metrics["best_pdb_filename"] = best_pdb_filename
        else:
            run_metrics["best_iptm"] = float("nan")
            run_metrics["best_cycle"] = None
            run_metrics["best_plddt"] = float("nan")
            run_metrics["best_iplddt"] = float("nan")
            run_metrics["best_seq"] = None
            run_metrics["best_ipsae"] = float("nan")
            run_metrics["best_alanine_count"] = None
            run_metrics["best_pdb_filename"] = None

        if a.plot:
            plot_run_metrics(self.designs_dir, a.name, run_id, a.num_cycles, run_metrics)

        return run_metrics

    def _run_single_design_validation(self, design_metrics: dict) -> dict:
        """
        Run validation + interface scoring for a SINGLE design using the selected backends.

        Supports:
          - AF3 validation via local docker (`validation_model=af3`)
          - Protenix validation via local subprocess (`validation_model=protenix`)

        Scoring:
          - PyRosetta (AF3 only locally)
          - Open-source scoring (OpenMM + FreeSASA)
        """
        import json
        import glob

        from utils.alphafold_utils import run_alphafold_step_from_csv, calculate_af3_ipsae
        # Note: pyrosetta_utils import is deferred to avoid init when not needed

        a = self.args
        validation_model = getattr(a, "validation_model", "none")
        scoring_method = getattr(a, "scoring_method", "pyrosetta")
        verbose = getattr(a, "verbose", False)

        design_idx = int(design_metrics.get("run_id", 0))
        best_cycle = design_metrics.get("best_cycle", 0)
        design_id = f"{a.name}_d{design_idx}_c{best_cycle}"

        binder_seq = design_metrics.get("best_seq", "")
        binder_length = len(binder_seq)

        # Initialize result with design metrics
        result = {
            "design_id": design_id,
            "design_num": design_idx,
            "cycle": best_cycle,
            "binder_sequence": binder_seq,
            "binder_length": binder_length,
            "cyclic": a.cyclic,
            "alanine_count": design_metrics.get("best_alanine_count", 0),
            "alanine_pct": 0.0,
            "boltz_iptm": design_metrics.get("best_iptm", 0.0),
            "boltz_ipsae": design_metrics.get("best_ipsae", 0.0),
            "boltz_plddt": design_metrics.get("best_plddt", 0.0),
            "boltz_iplddt": design_metrics.get("best_iplddt", 0.0),
            # Validation metrics (aliased to af3_* for unified schema)
            "af3_iptm": 0.0,
            "af3_ipsae": 0.0,
            "af3_ptm": 0.0,
            "af3_plddt": 0.0,
            # Acceptance status
            "accepted": False,
            "rejection_reason": "validation_not_run",
        }

        if binder_length > 0:
            result["alanine_pct"] = round(result["alanine_count"] / binder_length * 100, 2)

        best_pdb = design_metrics.get("best_pdb_filename")
        if not best_pdb or not os.path.exists(best_pdb):
            result["rejection_reason"] = "no_valid_design"
            return result

        # Determine target type
        any_ligand_or_nucleic = a.ligand_smiles or a.ligand_ccd or a.nucleic_seq
        if a.nucleic_type.strip() and a.nucleic_seq.strip():
            target_type = "nucleic"
        elif any_ligand_or_nucleic:
            target_type = "small_molecule"
        else:
            target_type = "protein"

        work_dir_validation = os.path.join(self.save_dir, "_validation_work", design_id)
        os.makedirs(work_dir_validation, exist_ok=True)

        try:
            if validation_model == "af3":
                print(f"  Running AF3 validation for {design_id}...")

                # STEP 1: create single-design CSV for AF3
                temp_csv_dir = os.path.join(work_dir_validation, "temp_csv")
                os.makedirs(temp_csv_dir, exist_ok=True)
                temp_csv_path = os.path.join(temp_csv_dir, "single_design.csv")

                csv_row = {
                    "design_id": design_id,
                    "design_num": design_idx,
                    "cycle": best_cycle,
                    "binder_sequence": binder_seq,
                    "binder_length": binder_length,
                    "cyclic": a.cyclic,
                    "boltz_iptm": result["boltz_iptm"],
                    "boltz_ipsae": result["boltz_ipsae"],
                    "boltz_plddt": result["boltz_plddt"],
                    "boltz_iplddt": result["boltz_iplddt"],
                    "target_seqs": self.data_builder.target_seqs_used,
                    "contact_residues": a.contact_residues or "",
                    "msa_mode": a.msa_mode,
                    "ligand_smiles": a.ligand_smiles or "",
                    "ligand_ccd": a.ligand_ccd or "",
                    "nucleic_seq": a.nucleic_seq or "",
                    "nucleic_type": a.nucleic_type or "",
                    "template_path": a.template_path or "",
                    "template_mapping": a.template_cif_chain_id or "",
                }
                pd.DataFrame([csv_row]).to_csv(temp_csv_path, index=False)

                # STEP 2: run AF3 for this design
                af_output_dir, af_output_apo_dir, af_pdb_dir, af_pdb_dir_apo = (
                    run_alphafold_step_from_csv(
                        csv_path=temp_csv_path,
                        alphafold_dir=os.path.expanduser(a.alphafold_dir),
                        af3_docker_name=a.af3_docker_name,
                        af3_database_settings=os.path.expanduser(a.af3_database_settings),
                        hmmer_path=os.path.expanduser(a.hmmer_path),
                        ligandmpnn_dir=work_dir_validation,
                        work_dir=os.path.expanduser(a.work_dir) or os.getcwd(),
                        binder_id=self.binder_chain,
                        gpu_id=getattr(a, "physical_gpu_id", a.gpu_id),
                        high_iptm=False,
                        use_msa_for_af3=getattr(a, "use_msa_for_validation", True),
                        msa_cache_dir=self.designs_dir,
                    )
                )

                # STEP 3: parse AF3 output
                af3_cif_path = None
                conf_json_text = None
                if af_output_dir and os.path.exists(af_output_dir):
                    for design_subdir in os.listdir(af_output_dir):
                        design_path = os.path.join(af_output_dir, design_subdir)
                        if not os.path.isdir(design_path):
                            continue
                        cif_files = [
                            f for f in glob.glob(os.path.join(design_path, "*_model.cif"))
                            if "_seed-" not in f and "_sample-" not in f
                        ]
                        summary_conf_files = [
                            f for f in glob.glob(os.path.join(design_path, "*_summary_confidences.json"))
                            if "_seed-" not in f and "_sample-" not in f
                        ]
                        conf_files = [
                            f for f in glob.glob(os.path.join(design_path, "*_confidences.json"))
                            if "_seed-" not in f and "_sample-" not in f
                        ]
                        if cif_files:
                            af3_cif_path = cif_files[0]
                            if summary_conf_files:
                                try:
                                    summary = json.loads(Path(summary_conf_files[0]).read_text())
                                    result["af3_iptm"] = round(summary.get("iptm", 0.0), 4)
                                    result["af3_ptm"] = round(summary.get("ptm", 0.0), 4)
                                except Exception as e:
                                    print(f"    Warning: Could not read summary confidence: {e}")
                            if conf_files:
                                try:
                                    conf_json_text = Path(conf_files[0]).read_text()
                                    conf = json.loads(conf_json_text)
                                    atom_plddts = conf.get("atom_plddts", [])
                                    if atom_plddts:
                                        result["af3_plddt"] = round(sum(atom_plddts) / len(atom_plddts), 2)

                                    target_seqs = a.protein_seqs or ""
                                    target_length = len(target_seqs.split(":")[0]) if target_seqs else 0
                                    if binder_length > 0 and target_length > 0:
                                        ipsae_result = calculate_af3_ipsae(
                                            conf_json_text, binder_length, target_length
                                        )
                                        result["af3_ipsae"] = ipsae_result.get("af3_ipsae", 0.0)
                                except Exception as e:
                                    print(f"    Warning: Could not read confidence: {e}")
                            break

                # STEP 4: copy AF3 CIF to refolded/
                refolded_dir = os.path.join(self.save_dir, "refolded")
                os.makedirs(refolded_dir, exist_ok=True)
                af3_structure_text = None
                if af3_cif_path and os.path.exists(af3_cif_path):
                    dest_cif = os.path.join(refolded_dir, f"{design_id}_refolded.cif")
                    shutil.copy(af3_cif_path, dest_cif)
                    if scoring_method == "opensource":
                        try:
                            af3_structure_text = Path(af3_cif_path).read_text()
                        except Exception:
                            af3_structure_text = None

                # STEP 5: scoring
                if target_type != "protein":
                    result["accepted"] = True
                    result["rejection_reason"] = ""
                    return result

                if scoring_method == "opensource":
                    from boltz_ph.scoring.opensource_local import run_opensource_scoring_local

                    scoring_result = run_opensource_scoring_local(
                        design_id=design_id,
                        af3_structure=af3_structure_text,
                        af3_iptm=result["af3_iptm"],
                        af3_ptm=result["af3_ptm"],
                        af3_plddt=result["af3_plddt"],
                        binder_chain="A",
                        target_chain="B",
                        apo_structure=None,
                        af3_confidence_json=conf_json_text,
                        target_type=target_type,
                        verbose=verbose,
                    )

                    result["accepted"] = scoring_result.get("accepted", False)
                    result["rejection_reason"] = scoring_result.get("rejection_reason", "")
                    for k, v in scoring_result.items():
                        if k in ("design_id", "relaxed_pdb"):
                            continue
                        result[k] = v
                    if scoring_result.get("relaxed_pdb"):
                        dest_dir = "accepted_designs" if result["accepted"] else "rejected"
                        out_dir = Path(self.save_dir) / dest_dir
                        out_dir.mkdir(parents=True, exist_ok=True)
                        (out_dir / f"{design_id}_relaxed.pdb").write_text(scoring_result["relaxed_pdb"])
                else:
                    pyrosetta_result = None
                    if af_pdb_dir and os.path.exists(af_pdb_dir):
                        pdb_files = [f for f in os.listdir(af_pdb_dir) if f.endswith(".pdb")]
                        if pdb_files:
                            run_rosetta_step(
                                work_dir_validation,
                                af_pdb_dir,
                                af_pdb_dir_apo,
                                binder_id=self.binder_chain,
                                target_type=target_type,
                            )

                            rosetta_success_dir = os.path.join(work_dir_validation, "af_pdb_rosetta_success")
                            success_csv = os.path.join(rosetta_success_dir, "success_designs.csv")
                            failed_csv = os.path.join(rosetta_success_dir, "failed_designs.csv")

                            if os.path.exists(success_csv) and os.path.getsize(success_csv) > 0:
                                try:
                                    success_df = pd.read_csv(success_csv)
                                    if len(success_df) > 0:
                                        row = success_df.iloc[0]
                                        pyrosetta_result = {
                                            "accepted": True,
                                            "rejection_reason": "",
                                            "pdb_path": row.get("PDB", ""),
                                            "model_name": row.get("Model", ""),
                                            **{k: row.get(k) for k in row.index},
                                        }
                                except pd.errors.EmptyDataError:
                                    pass

                            if pyrosetta_result is None and os.path.exists(failed_csv) and os.path.getsize(failed_csv) > 0:
                                try:
                                    failed_df = pd.read_csv(failed_csv)
                                    if len(failed_df) > 0:
                                        row = failed_df.iloc[0]
                                        pyrosetta_result = {
                                            "accepted": False,
                                            "rejection_reason": row.get("failure_reason", "pyrosetta_filter_failed"),
                                            "pdb_path": row.get("PDB", ""),
                                            "model_name": row.get("Model", ""),
                                            **{k: row.get(k) for k in row.index},
                                        }
                                except pd.errors.EmptyDataError:
                                    pass

                    if pyrosetta_result is None:
                        pyrosetta_result = {"accepted": False, "rejection_reason": "pyrosetta_failed"}

                    result["accepted"] = pyrosetta_result.get("accepted", False)
                    result["rejection_reason"] = pyrosetta_result.get("rejection_reason", "")
                    for key in [
                        "binder_score", "total_score", "interface_sc", "interface_packstat",
                        "interface_dG", "interface_dSASA", "interface_dG_SASA_ratio",
                        "interface_nres", "interface_hbonds", "interface_hbond_percentage",
                        "interface_delta_unsat_hbonds", "interface_delta_unsat_hbonds_percentage",
                        "interface_hydrophobicity", "surface_hydrophobicity", "binder_sasa",
                        "interface_fraction", "apo_holo_rmsd", "i_pae", "rg",
                    ]:
                        if key in pyrosetta_result:
                            result[key] = pyrosetta_result[key]

                    if pyrosetta_result.get("pdb_path") and pyrosetta_result.get("model_name"):
                        src_pdb = os.path.join(pyrosetta_result["pdb_path"], pyrosetta_result["model_name"])
                        if os.path.exists(src_pdb):
                            dest_dir = "accepted_designs" if result["accepted"] else "rejected"
                            out_dir = Path(self.save_dir) / dest_dir
                            out_dir.mkdir(parents=True, exist_ok=True)
                            shutil.copy(src_pdb, out_dir / f"{design_id}_relaxed.pdb")

            elif validation_model == "protenix":
                print(f"  Running Protenix validation for {design_id}...")
                # Use persistent runner (~4x faster - model stays loaded in GPU memory)
                from boltz_ph.validation.protenix_local import run_protenix_validation_persistent

                target_seq = self.data_builder.target_seqs_used
                target_msas = self._load_target_msas_for_validation()
                protenix_result = run_protenix_validation_persistent(
                    design_id=design_id,
                    binder_seq=binder_seq,
                    target_seq=target_seq,
                    target_msas=target_msas if target_msas else None,
                    verbose=verbose,
                )

                if protenix_result.get("error"):
                    result["rejection_reason"] = protenix_result["error"]
                    return result

                for k in ("af3_iptm", "af3_ptm", "af3_plddt", "af3_ipsae"):
                    result[k] = protenix_result.get(k, 0.0)

                holo_cif = protenix_result.get("af3_structure")
                conf_json_text = protenix_result.get("af3_confidence_json")
                apo_cif = protenix_result.get("apo_structure")

                refolded_dir = Path(self.save_dir) / "refolded"
                refolded_dir.mkdir(parents=True, exist_ok=True)
                if holo_cif:
                    (refolded_dir / f"{design_id}_refolded.cif").write_text(holo_cif)

                if target_type != "protein":
                    result["accepted"] = True
                    result["rejection_reason"] = ""
                    return result

                if scoring_method == "opensource":
                    from boltz_ph.scoring.opensource_local import run_opensource_scoring_local
                    scoring_result = run_opensource_scoring_local(
                        design_id=design_id,
                        af3_structure=holo_cif,
                        af3_iptm=result["af3_iptm"],
                        af3_ptm=result["af3_ptm"],
                        af3_plddt=result["af3_plddt"],
                        binder_chain="A",
                        target_chain="B",
                        apo_structure=apo_cif,
                        af3_confidence_json=conf_json_text,
                        target_type=target_type,
                        verbose=verbose,
                    )
                    result["accepted"] = scoring_result.get("accepted", False)
                    result["rejection_reason"] = scoring_result.get("rejection_reason", "")
                    for k, v in scoring_result.items():
                        if k in ("design_id", "relaxed_pdb"):
                            continue
                        result[k] = v
                    if scoring_result.get("relaxed_pdb"):
                        dest_dir = "accepted_designs" if result["accepted"] else "rejected"
                        out_dir = Path(self.save_dir) / dest_dir
                        out_dir.mkdir(parents=True, exist_ok=True)
                        (out_dir / f"{design_id}_relaxed.pdb").write_text(scoring_result["relaxed_pdb"])
                else:
                    # PyRosetta scoring for Protenix - convert CIF to PDB and run
                    from utils.pyrosetta_utils import run_rosetta_step

                    # Create temp PDB directories for PyRosetta
                    protenix_pdb_dir = os.path.join(work_dir_validation, "protenix_pdb_holo")
                    protenix_pdb_dir_apo = os.path.join(work_dir_validation, "protenix_pdb_apo")
                    os.makedirs(protenix_pdb_dir, exist_ok=True)
                    os.makedirs(protenix_pdb_dir_apo, exist_ok=True)

                    # Convert HOLO CIF to PDB
                    holo_cif_path = refolded_dir / f"{design_id}_refolded.cif"
                    holo_pdb_path = os.path.join(protenix_pdb_dir, f"{design_id}.pdb")
                    if holo_cif_path.exists():
                        convert_cif_to_pdb(str(holo_cif_path), holo_pdb_path)

                    # Convert APO CIF to PDB if available
                    if apo_cif:
                        apo_cif_path = os.path.join(work_dir_validation, f"{design_id}_apo.cif")
                        Path(apo_cif_path).write_text(apo_cif)
                        apo_pdb_path = os.path.join(protenix_pdb_dir_apo, f"{design_id}_apo.pdb")
                        convert_cif_to_pdb(apo_cif_path, apo_pdb_path)

                    # Run PyRosetta scoring
                    pyrosetta_result = None
                    if os.path.exists(holo_pdb_path):
                        run_rosetta_step(
                            work_dir_validation,
                            protenix_pdb_dir,
                            protenix_pdb_dir_apo,
                            binder_id=self.binder_chain,
                            target_type=target_type,
                        )

                        # Parse PyRosetta results (same pattern as AF3 path)
                        rosetta_success_dir = os.path.join(work_dir_validation, "af_pdb_rosetta_success")
                        success_csv = os.path.join(rosetta_success_dir, "success_designs.csv")
                        failed_csv = os.path.join(rosetta_success_dir, "failed_designs.csv")

                        if os.path.exists(success_csv) and os.path.getsize(success_csv) > 0:
                            try:
                                success_df = pd.read_csv(success_csv)
                                if len(success_df) > 0:
                                    row = success_df.iloc[0]
                                    pyrosetta_result = {
                                        "accepted": True,
                                        "rejection_reason": "",
                                        "pdb_path": row.get("PDB", ""),
                                        "model_name": row.get("Model", ""),
                                        **{k: row.get(k) for k in row.index},
                                    }
                            except pd.errors.EmptyDataError:
                                pass

                        if pyrosetta_result is None and os.path.exists(failed_csv) and os.path.getsize(failed_csv) > 0:
                            try:
                                failed_df = pd.read_csv(failed_csv)
                                if len(failed_df) > 0:
                                    row = failed_df.iloc[0]
                                    pyrosetta_result = {
                                        "accepted": False,
                                        "rejection_reason": row.get("failure_reason", "pyrosetta_filter_failed"),
                                        "pdb_path": row.get("PDB", ""),
                                        "model_name": row.get("Model", ""),
                                        **{k: row.get(k) for k in row.index},
                                    }
                            except pd.errors.EmptyDataError:
                                pass

                    if pyrosetta_result is None:
                        pyrosetta_result = {"accepted": False, "rejection_reason": "pyrosetta_failed"}

                    result["accepted"] = pyrosetta_result.get("accepted", False)
                    result["rejection_reason"] = pyrosetta_result.get("rejection_reason", "")
                    for key in [
                        "binder_score", "total_score", "interface_sc", "interface_packstat",
                        "interface_dG", "interface_dSASA", "interface_dG_SASA_ratio",
                        "interface_nres", "interface_hbonds", "interface_hbond_percentage",
                        "interface_delta_unsat_hbonds", "interface_delta_unsat_hbonds_percentage",
                        "interface_hydrophobicity", "surface_hydrophobicity", "binder_sasa",
                        "interface_fraction", "apo_holo_rmsd", "i_pae", "rg",
                    ]:
                        if key in pyrosetta_result:
                            result[key] = pyrosetta_result[key]

                    if pyrosetta_result.get("pdb_path") and pyrosetta_result.get("model_name"):
                        src_pdb = os.path.join(pyrosetta_result["pdb_path"], pyrosetta_result["model_name"])
                        if os.path.exists(src_pdb):
                            dest_dir = "accepted_designs" if result["accepted"] else "rejected"
                            out_dir = Path(self.save_dir) / dest_dir
                            out_dir.mkdir(parents=True, exist_ok=True)
                            shutil.copy(src_pdb, out_dir / f"{design_id}_relaxed.pdb")

            else:
                result["rejection_reason"] = "validation_not_run"

        except Exception as e:
            import traceback
            print(f"    Error during validation: {e}")
            traceback.print_exc()
            result["rejection_reason"] = f"validation_error: {str(e)}"

        finally:
            try:
                shutil.rmtree(work_dir_validation, ignore_errors=True)
            except Exception:
                pass

        return result

    def _save_design_result(self, result: dict) -> None:
        """
        Save a single design's validation results to all relevant CSVs.
        
        This enables incremental saving for the design-by-design execution model.
        Results are appended to:
        - refolded/validation_results.csv
        - accepted_designs/accepted_stats.csv (if accepted)
        - rejected/rejected_stats.csv (if rejected)
        
        Args:
            result: Full result dict from _run_single_design_validation()
        """
        # Save validation results
        refolded_dir = os.path.join(self.save_dir, "refolded")
        os.makedirs(refolded_dir, exist_ok=True)
        
        validation_row = {
            "design_id": result.get("design_id"),
            "af3_iptm": result.get("af3_iptm", 0.0),
            "af3_ipsae": result.get("af3_ipsae", 0.0),
            "af3_ptm": result.get("af3_ptm", 0.0),
            "af3_plddt": result.get("af3_plddt", 0.0),
        }
        append_to_csv_safe(Path(refolded_dir) / "validation_results.csv", validation_row)
        
        # Build full stats row
        stats_row = _reorder_columns(pd.DataFrame([result])).iloc[0].to_dict()
        
        # Save to accepted or rejected
        if result.get("accepted", False):
            accepted_dir = os.path.join(self.save_dir, "accepted_designs")
            os.makedirs(accepted_dir, exist_ok=True)
            append_to_csv_safe(Path(accepted_dir) / "accepted_stats.csv", stats_row)
        else:
            rejected_dir = os.path.join(self.save_dir, "rejected")
            os.makedirs(rejected_dir, exist_ok=True)
            append_to_csv_safe(Path(rejected_dir) / "rejected_stats.csv", stats_row)

    def _count_completed_designs(self) -> int:
        """Count completed designs by checking best_designs/ folder."""
        best_dir = Path(self.save_dir) / "best_designs"
        if not best_dir.exists():
            return 0
        return len(list(best_dir.glob("*.pdb")))

    def _count_accepted_designs(self) -> int:
        """Count accepted designs by checking accepted_designs/ folder."""
        accepted_dir = Path(self.save_dir) / "accepted_designs"
        if not accepted_dir.exists():
            return 0
        return len(list(accepted_dir.glob("*_relaxed.pdb")))

    def _get_next_design_index(self) -> int:
        """Determine next design index from design_stats.csv (all attempts).
        
        Uses design_stats.csv as the source of truth since it records all design
        attempts, including those that fail (e.g., all cycles >20% alanine).
        This prevents filename collisions and CSV duplicates.
        """
        csv_path = Path(self.designs_dir) / "design_stats.csv"
        if not csv_path.exists():
            return 0
        
        try:
            df = pd.read_csv(csv_path)
            if "design_num" in df.columns and len(df) > 0:
                return int(df["design_num"].max()) + 1
        except Exception:
            pass
        
        return 0

    def _load_target_msas_for_validation(self) -> Dict[str, str]:
        """
        Load cached ColabFold MSAs for target chains for validation backends.

        Local Boltz design caches MSAs under:
            designs_dir/{chain_id}_env/msa.a3m

        Returns:
            Dict mapping chain letter -> A3M content.
        """
        a = self.args
        if not getattr(a, "use_msa_for_validation", True):
            return {}

        target_seqs = getattr(self.data_builder, "target_seqs_used", "") or (a.protein_seqs or "")
        if not target_seqs:
            return {}

        msas: Dict[str, str] = {}
        target_chains = target_seqs.split(":")
        for i, _seq in enumerate(target_chains):
            chain_id = chr(ord("B") + i)
            msa_a3m = Path(self.designs_dir) / f"{chain_id}_env" / "msa.a3m"
            if msa_a3m.exists():
                try:
                    msas[chain_id] = msa_a3m.read_text()
                except Exception:
                    continue
        return msas

    def _should_continue(self) -> bool:
        """
        Check if pipeline should continue based on stopping conditions.
        
        Stopping conditions (OR logic):
        - num_designs: Stop after N total designs generated
        - num_accepted: Stop after N designs pass all filters
        
        Returns:
            True if should continue, False if target reached
        """
        total = self._count_completed_designs()
        accepted = self._count_accepted_designs()
        
        if self.args.num_designs is not None and total >= self.args.num_designs:
            print(f"✓ Target reached: {total}/{self.args.num_designs} designs generated")
            return False
        
        if hasattr(self.args, 'num_accepted') and self.args.num_accepted is not None and accepted >= self.args.num_accepted:
            print(f"✓ Target reached: {accepted}/{self.args.num_accepted} accepted designs")
            return False
        
        return True

    def run_single_design(self, design_idx: int, base_data: dict, pocket_conditioning: bool) -> dict:
        """
        Execute a single design run and return results.
        
        This method is designed to be called by the multi-GPU orchestrator,
        allowing parallel execution across multiple GPUs.
        
        Args:
            design_idx: The unique index for this design
            base_data: Pre-built base data dictionary from data_builder.build()
            pocket_conditioning: Whether pocket conditioning is enabled
            
        Returns:
            dict containing:
            - design_idx: The design index
            - run_metrics: Output from _run_design_cycle()
            - design_id: The unique design identifier
            - has_valid_pdb: Whether a valid PDB was generated
            - validation_result: If AF3 validation ran, the result dict (or None)
            - accepted: Final acceptance status
        """
        data_cp = copy.deepcopy(base_data)
        run_id = str(design_idx)
        
        # Run the design cycle
        run_metrics = self._run_design_cycle(data_cp, run_id, pocket_conditioning)
        
        # Save best design
        design_id, has_valid_pdb = self._save_single_best_design(run_metrics)
        
        result = {
            "design_idx": design_idx,
            "run_metrics": run_metrics,
            "design_id": design_id,
            "has_valid_pdb": has_valid_pdb,
            "validation_result": None,
            "accepted": None,
            "rejection_reason": "",
        }
        
        if not has_valid_pdb:
            result["accepted"] = False
            result["rejection_reason"] = f"no cycle passed all criteria (alanine <= 20%, ipTM >= {self.args.high_iptm_threshold}, pLDDT >= {self.args.high_plddt_threshold})"
            return result
        
        # Check if validation is enabled
        if getattr(self.args, "validation_model", "none") == "none":
            return result
        
        # Check Boltz thresholds before expensive AF3 validation
        boltz_iptm = run_metrics.get("best_iptm", 0.0)
        boltz_plddt = run_metrics.get("best_plddt", 0.0)
        
        iptm_ok = boltz_iptm >= self.args.high_iptm_threshold
        plddt_ok = boltz_plddt >= self.args.high_plddt_threshold
        
        if not (iptm_ok and plddt_ok):
            failure_reasons = []
            if not iptm_ok:
                failure_reasons.append(f"boltz_iptm < {self.args.high_iptm_threshold}")
            if not plddt_ok:
                failure_reasons.append(f"boltz_plddt < {self.args.high_plddt_threshold}")
            result["accepted"] = False
            result["rejection_reason"] = "; ".join(failure_reasons)
            self._update_design_status(design_id, False, result["rejection_reason"])
            return result
        
        # Run full AF3 + PyRosetta validation
        validation_result = self._run_single_design_validation(run_metrics)
        self._save_design_result(validation_result)
        
        result["validation_result"] = validation_result
        result["accepted"] = validation_result.get("accepted", False)
        result["rejection_reason"] = validation_result.get("rejection_reason", "")
        
        # Update best_designs.csv with final validation result
        self._update_design_status(
            design_id,
            result["accepted"],
            result["rejection_reason"]
        )
        
        return result

    def run_pipeline(self):
        """Orchestrates the entire protein design and validation pipeline."""
        # 1. Prepare Base Data (using the new InputDataBuilder)
        base_data, pocket_conditioning = self.data_builder.build()

        # 2. Check for existing progress (resumable execution)
        start_design_idx = self._get_next_design_index()
        if start_design_idx > 0:
            print(f"\n{'='*60}")
            print(f"📁 Found {start_design_idx} existing designs. Resuming from design {start_design_idx}...")
            print(f"{'='*60}")

        # 3. Run Design Cycles with resumable while loop
        all_run_metrics = []
        while self._should_continue():
            design_idx = self._get_next_design_index()
            
            # Build progress display
            total = self._count_completed_designs()
            accepted = self._count_accepted_designs()
            progress_parts = []
            if self.args.num_designs is not None:
                progress_parts.append(f"{total}/{self.args.num_designs} designs")
            else:
                progress_parts.append(f"{total} designs")
            if hasattr(self.args, 'num_accepted') and self.args.num_accepted is not None:
                progress_parts.append(f"{accepted}/{self.args.num_accepted} accepted")
            
            print(f"\n{'='*60}")
            print(f"Starting Design {design_idx} | Progress: {', '.join(progress_parts)}")
            print(f"{'='*60}")

            data_cp = copy.deepcopy(base_data)
            run_id = str(design_idx)

            run_metrics = self._run_design_cycle(data_cp, run_id, pocket_conditioning)
            all_run_metrics.append(run_metrics)
            
            # 4. Save best design incrementally (always saves to best_designs.csv for audit trail)
            design_id, has_valid_pdb = self._save_single_best_design(run_metrics)
            if has_valid_pdb:
                print(f"  ✓ Saved best design {design_idx} to best_designs/")
            else:
                print(f"  ✗ Design {design_idx} failed: no cycle passed all criteria (alanine/ipTM/pLDDT)")
            
            # 5. Run validation for THIS design (design-by-design execution)
            if getattr(self.args, "validation_model", "none") != "none" and has_valid_pdb:
                # Check if design meets minimum Boltz thresholds before expensive AF3 validation
                boltz_iptm = run_metrics.get("best_iptm", 0.0)
                boltz_plddt = run_metrics.get("best_plddt", 0.0)
                
                iptm_ok = boltz_iptm >= self.args.high_iptm_threshold
                plddt_ok = boltz_plddt >= self.args.high_plddt_threshold
                meets_threshold = iptm_ok and plddt_ok
                
                if not meets_threshold:
                    # Skip validation - update best_designs.csv with rejection reason
                    failure_reasons = []
                    if not iptm_ok:
                        failure_reasons.append(f"boltz_iptm < {self.args.high_iptm_threshold}")
                    if not plddt_ok:
                        failure_reasons.append(f"boltz_plddt < {self.args.high_plddt_threshold}")
                    rejection_reason = "; ".join(failure_reasons)
                    
                    # Update the design status in best_designs.csv
                    self._update_design_status(design_id, False, rejection_reason)
                    print(f"  ⏭ Skipping validation: {rejection_reason}")
                    continue  # Move to next design
                
                # Run full AF3 + PyRosetta validation
                validation_result = self._run_single_design_validation(run_metrics)
                self._save_design_result(validation_result)
                
                # Update best_designs.csv with final validation result
                self._update_design_status(
                    design_id,
                    validation_result.get("accepted", False),
                    validation_result.get("rejection_reason", "")
                )
                
                # Print validation summary for this design
                if validation_result.get("accepted"):
                    print(f"  ✓ ACCEPTED: af3_iptm={validation_result.get('af3_iptm', 0):.3f}, "
                          f"af3_ipsae={validation_result.get('af3_ipsae', 0):.3f}, "
                          f"dG={validation_result.get('interface_dG', 0):.1f}")
                else:
                    reason = validation_result.get("rejection_reason", "unknown")
                    print(f"  ✗ REJECTED: {reason}")
                
                # Check if we should stop early (num_accepted target reached)
                if not self._should_continue():
                    break

        # 6. Print summary
        self._print_summary(all_run_metrics)
        
        # 7. Print final validation summary
        if getattr(self.args, "validation_model", "none") != "none":
            total = self._count_completed_designs()
            accepted = self._count_accepted_designs()
            rejected = total - accepted if total > accepted else 0
            
            print(f"\n{'='*60}")
            print("PIPELINE COMPLETE")
            print(f"{'='*60}")
            print(f"Total designs generated: {total}")
            print(f"Accepted: {accepted}")
            print(f"Rejected: {rejected}")
            if self.args.num_accepted:
                print(f"Target accepted: {self.args.num_accepted}")
            print(f"\nOutput structure:")
            print(f"  {self.save_dir}/")
            print(f"  ├── designs/              # All cycle PDBs + design_stats.csv")
            print(f"  ├── best_designs/         # Best per design + best_designs.csv")
            print(f"  ├── refolded/             # Refolded structures + metrics")
            print(f"  │   ├── validation_results.csv")
            print(f"  │   └── *_af3.cif")
            print(f"  ├── accepted_designs/     # Passed filters")
            print(f"  │   ├── accepted_stats.csv")
            print(f"  │   └── *_relaxed.pdb")
            print(f"  └── rejected/             # Failed filters")
            print(f"      ├── rejected_stats.csv")
            print(f"      └── *_relaxed.pdb")

    def _save_single_best_design(self, metrics: dict) -> tuple[str, bool]:
        """
        Save a single design to best_designs/ folder with acceptance tracking.
        
        Always saves to best_designs.csv (even failed designs) for complete audit trail.
        This enables resumable execution and tracks rejection reasons.
        
        Args:
            metrics: Output from _run_design_cycle() for one design
        
        Returns:
            Tuple of (design_id, has_valid_pdb):
            - design_id: The unique identifier for this design
            - has_valid_pdb: True if a valid PDB was saved, False if design failed
        """
        a = self.args
        best_dir = os.path.join(self.save_dir, "best_designs")
        os.makedirs(best_dir, exist_ok=True)
        
        design_idx = int(metrics.get("run_id", 0))
        best_pdb = metrics.get("best_pdb_filename")
        has_valid_pdb = best_pdb and os.path.exists(best_pdb)
        
        if has_valid_pdb:
            # Valid design - copy PDB and mark as pending validation
            best_cycle = metrics.get("best_cycle", 0)
            design_id = f"{a.name}_d{design_idx}_c{best_cycle}"
            dest_pdb = os.path.join(best_dir, f"{design_id}.pdb")
            shutil.copy(best_pdb, dest_pdb)
            
            seq = metrics.get("best_seq", "")
            binder_length = len(seq) if seq else 0
            alanine_count = metrics.get("best_alanine_count", 0)
            alanine_pct = (alanine_count / binder_length * 100) if binder_length > 0 else 0.0
            
            # Pending validation - accepted will be updated later
            accepted = None
            rejection_reason = ""
        else:
            # All cycles failed criteria: alanine <= 20%, ipTM >= threshold, pLDDT >= threshold
            design_id = f"{a.name}_d{design_idx}_failed"
            seq = ""
            binder_length = 0
            alanine_count = 0
            alanine_pct = 0.0
            best_cycle = None

            accepted = False
            rejection_reason = f"no cycle passed all criteria (alanine <= 20%, ipTM >= {a.high_iptm_threshold}, pLDDT >= {a.high_plddt_threshold})"
        
        # Build row for best_designs.csv
        row = {
            "design_id": design_id,
            "design_num": design_idx,
            "cycle": best_cycle,
            "binder_sequence": seq,
            "binder_length": binder_length,
            "cyclic": a.cyclic,
            "boltz_iptm": metrics.get("best_iptm", 0.0) if has_valid_pdb else float("nan"),
            "boltz_ipsae": metrics.get("best_ipsae", 0.0) if has_valid_pdb else float("nan"),
            "boltz_plddt": metrics.get("best_plddt", 0.0) if has_valid_pdb else float("nan"),
            "boltz_iplddt": metrics.get("best_iplddt", 0.0) if has_valid_pdb else float("nan"),
            "alanine_count": alanine_count,
            "alanine_pct": round(alanine_pct, 2),
            "target_seqs": self.data_builder.target_seqs_used,
            "contact_residues": self.data_builder.contact_residues_canonical or a.contact_residues or "",
            "contact_residues_auth": self.data_builder.contact_residues_auth or "",
            "template_first_residue": ",".join(f"{k}:{v}" for k, v in self.data_builder.template_first_residues.items()) if self.data_builder.template_first_residues else "",
            "msa_mode": a.msa_mode,
            # Fields for AF3 reconstruction
            "ligand_smiles": a.ligand_smiles or "",
            "ligand_ccd": a.ligand_ccd or "",
            "nucleic_seq": a.nucleic_seq or "",
            "nucleic_type": a.nucleic_type or "",
            "template_path": a.template_path or "",
            "template_mapping": a.template_cif_chain_id or "",
            # Acceptance tracking
            "accepted": accepted,
            "rejection_reason": rejection_reason,
        }
        
        # Append to CSV incrementally
        csv_path = Path(best_dir) / "best_designs.csv"
        append_to_csv_safe(csv_path, row)
        
        return design_id, has_valid_pdb

    def _update_design_status(self, design_id: str, accepted: bool, rejection_reason: str) -> None:
        """
        Update acceptance status for a design in best_designs.csv.
        
        Called after validation completes or when pre-validation thresholds fail.
        Uses update_csv_row for thread-safe in-place update.
        
        Args:
            design_id: The unique identifier for the design (e.g., "mydesign_d0_c2")
            accepted: True if design passed all filters, False if rejected
            rejection_reason: Human-readable reason for rejection (empty string if accepted)
        """
        csv_path = Path(self.save_dir) / "best_designs" / "best_designs.csv"
        update_csv_row(
            csv_path,
            key_col="design_id",
            key_val=design_id,
            update_data={"accepted": accepted, "rejection_reason": rejection_reason}
        )

    def _print_summary(self, all_run_metrics):
        """Print summary of all runs."""
        a = self.args
        designs_dir = self.designs_dir

        # Count designs
        design_pdbs = list(Path(designs_dir).glob("*.pdb"))
        successful = [m for m in all_run_metrics if m.get("best_cycle") is not None]

        print(f"\n{'='*70}")
        print("SUMMARY")
        print(f"{'='*70}")
        print(f"Successful: {len(successful)}/{len(all_run_metrics)}")

        if successful:
            best_overall = max(successful, key=lambda m: m.get("best_iptm", 0))
            print(f"Best overall: {a.name}_d{best_overall['run_id']} with ipTM={best_overall['best_iptm']:.3f}")

        print(f"Total cycles saved: {len(design_pdbs)}")
        print(f"Best designs: {len(successful)}")
        print(f"\nOutput structure:")
        print(f"  {self.save_dir}/")
        print(f"  ├── designs/           # ALL cycles (PDBs + design_stats.csv)")
        print(f"  └── best_designs/      # Best cycle per design run")
        print(f"\nResults saved to: {self.save_dir}")


# =============================================================================
# Multi-GPU Orchestrator
# =============================================================================

def _gpu_worker(gpu_id: int, args, task_queue: Queue, result_queue: Queue, shutdown_event):
    """
    Worker process that runs on a specific GPU.

    Each worker:
    1. Initializes its own ProteinHunter_Boltz instance on the assigned GPU
    2. Pulls design indices from the shared task queue
    3. Executes designs and puts results in the result queue
    4. Exits gracefully when receiving None (poison pill) or shutdown signal

    Args:
        gpu_id: CUDA device ID for this worker
        args: Command-line arguments (will be modified to set gpu_id)
        task_queue: Queue to pull design indices from
        result_queue: Queue to put results into
        shutdown_event: Multiprocessing event to signal shutdown
    """
    import copy
    import sys
    
    # CRITICAL: Set CUDA_VISIBLE_DEVICES for this worker process
    # This ensures all child processes (LigandMPNN, etc.) use the correct GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    # Set GPU for this worker
    worker_args = copy.deepcopy(args)
    worker_args.gpu_id = 0  # Now always 0 since CUDA_VISIBLE_DEVICES restricts to one GPU
    worker_args.physical_gpu_id = gpu_id  # Keep track of actual GPU for Docker (AF3)
    
    # Create per-worker log file for detailed output
    save_dir = args.save_dir if args.save_dir else f"./results_{args.name}"
    os.makedirs(save_dir, exist_ok=True)
    log_path = os.path.join(save_dir, f"worker_gpu{gpu_id}.log")
    log_file = open(log_path, "w", buffering=1)  # Line-buffered for real-time updates
    
    # Redirect stdout/stderr to log file (captures all output including subprocesses)
    sys.stdout = log_file
    sys.stderr = log_file

    try:
        # Initialize model on this GPU
        print(f"[GPU {gpu_id}] Initializing worker...")
        print(f"[GPU {gpu_id}] Log file: {log_path}")
        hunter = ProteinHunter_Boltz(worker_args)
        
        # Build base data (done once per worker)
        base_data, pocket_conditioning = hunter.data_builder.build()
        print(f"[GPU {gpu_id}] Ready for designs")
        
        while not shutdown_event.is_set():
            try:
                # Non-blocking get with timeout to check shutdown
                design_idx = task_queue.get(timeout=1.0)
            except:
                continue  # Timeout - check shutdown event and retry
            
            if design_idx is None:
                # Poison pill - exit gracefully
                print(f"[GPU {gpu_id}] Received shutdown signal, finishing...")
                break
            
            print(f"\n{'='*60}")
            print(f"[GPU {gpu_id}] Starting design {design_idx}")
            print(f"{'='*60}")
            
            try:
                result = hunter.run_single_design(design_idx, base_data, pocket_conditioning)
                result["gpu_id"] = gpu_id
                result_queue.put(result)
                
                # Log completion to worker log
                print(f"\n[GPU {gpu_id}] Design {design_idx} complete")
                
            except Exception as e:
                print(f"[GPU {gpu_id}] Error on design {design_idx}: {e}")
                import traceback
                traceback.print_exc()
                result_queue.put({
                    "design_idx": design_idx,
                    "gpu_id": gpu_id,
                    "error": str(e),
                    "has_valid_pdb": False,
                    "accepted": False,
                })
                
    except Exception as e:
        print(f"[GPU {gpu_id}] Worker initialization failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        log_file.close()


class MultiGPUOrchestrator:
    """
    Orchestrates parallel protein design across multiple GPUs.
    
    Uses a producer-consumer pattern:
    - Orchestrator manages a task queue of design indices
    - Worker processes pull from queue and execute designs
    - Results are collected and progress is tracked centrally
    
    Design IDs are assigned atomically from the queue, preventing collisions.
    Workers finish their current design before shutting down (graceful exit).
    """
    
    def __init__(self, args):
        """
        Initialize the orchestrator.
        
        Args:
            args: Command-line arguments with num_gpus set
        """
        self.args = args
        self.num_gpus = args.num_gpus
        
        # Create save directory structure
        self.save_dir = args.save_dir if args.save_dir else f"./results_{args.name}"
        os.makedirs(self.save_dir, exist_ok=True)
        os.makedirs(f"{self.save_dir}/designs", exist_ok=True)
        os.makedirs(f"{self.save_dir}/best_designs", exist_ok=True)
        
    def _count_completed_designs(self) -> int:
        """Count completed designs by checking best_designs/ folder."""
        best_dir = Path(self.save_dir) / "best_designs"
        if not best_dir.exists():
            return 0
        return len(list(best_dir.glob("*.pdb")))
    
    def _count_accepted_designs(self) -> int:
        """Count accepted designs by checking accepted_designs/ folder."""
        accepted_dir = Path(self.save_dir) / "accepted_designs"
        if not accepted_dir.exists():
            return 0
        return len(list(accepted_dir.glob("*_relaxed.pdb")))
    
    def _get_next_design_index(self) -> int:
        """Get next design index from design_stats.csv."""
        csv_path = Path(self.save_dir) / "designs" / "design_stats.csv"
        if not csv_path.exists():
            return 0
        try:
            df = pd.read_csv(csv_path)
            if "design_num" in df.columns and len(df) > 0:
                return int(df["design_num"].max()) + 1
        except Exception:
            pass
        return 0
    
    def _print_design_result(self, result: dict) -> None:
        """Print detailed design result summary to terminal."""
        gpu_id = result.get("gpu_id", "?")
        design_idx = result.get("design_idx", "?")
        
        print(f"\n[GPU {gpu_id}] Design {design_idx} complete:", flush=True)
        
        # Check for errors first
        if result.get("error"):
            print(f"  └─ ✗ ERROR: {result['error']}", flush=True)
            return
        
        # Get metrics from run_metrics
        metrics = result.get("run_metrics", {}) or {}
        best_cycle = metrics.get("best_cycle")
        best_iptm = metrics.get("best_iptm") or 0
        best_plddt = metrics.get("best_plddt") or 0
        best_ipsae = metrics.get("best_ipsae") or 0
        alanine_count = metrics.get("best_alanine_count") or 0
        best_seq = metrics.get("best_seq") or ""
        binder_len = len(best_seq) if best_seq else 1
        alanine_pct = (alanine_count / binder_len * 100) if binder_len > 0 else 0
        
        has_valid_pdb = result.get("has_valid_pdb", False)
        
        if not has_valid_pdb:
            print(f"  └─ ✗ FAILED: no cycle passed criteria (Ala≤20%, ipTM≥{self.args.high_iptm_threshold}, pLDDT≥{self.args.high_plddt_threshold})", flush=True)
            return
        
        # Print best design stats
        print(f"  ├─ Best: cycle {best_cycle}, ipTM={best_iptm:.3f}, pLDDT={best_plddt:.2f}, ipSAE={best_ipsae:.3f}, Ala={alanine_pct:.0f}%", flush=True)
        
        # Check disposition
        accepted = result.get("accepted")
        rejection_reason = result.get("rejection_reason", "")
        validation_result = result.get("validation_result")
        
        if accepted is None:
            # No validation was run (AF3 disabled)
            print(f"  └─ ✓ Saved (no validation)", flush=True)
        elif rejection_reason and not validation_result:
            # Rejected before validation (thresholds not met)
            print(f"  └─ ✗ SKIPPED validation: {rejection_reason}", flush=True)
        elif validation_result:
            # AF3 validation was run
            af3_iptm = validation_result.get("af3_iptm", 0)
            af3_ipsae = validation_result.get("af3_ipsae", 0)
            interface_dG = validation_result.get("interface_dG", 0)
            interface_hbonds = validation_result.get("interface_hbonds", 0)
            
            print(f"  ├─ AF3: iptm={af3_iptm:.3f}, ipsae={af3_ipsae:.3f}, dG={interface_dG:.1f}, hbonds={interface_hbonds}", flush=True)
            
            if accepted:
                print(f"  └─ ✓✓ ACCEPTED", flush=True)
            else:
                print(f"  └─ ✗ REJECTED: {rejection_reason}", flush=True)
        elif accepted is False:
            print(f"  └─ ✗ REJECTED: {rejection_reason}", flush=True)
        else:
            print(f"  └─ ? Unknown status", flush=True)
    
    def run(self):
        """
        Run the multi-GPU design pipeline.
        
        Spawns worker processes, distributes work, and collects results.
        """
        print(f"\n{'='*70}")
        print(f"🚀 MULTI-GPU ORCHESTRATOR")
        print(f"{'='*70}")
        print(f"GPUs: {self.num_gpus}")
        print(f"Target designs: {self.args.num_designs}")
        if self.args.num_accepted:
            print(f"Target accepted: {self.args.num_accepted}")
        print(f"Output: {self.save_dir}")
        print(f"Worker logs: {self.save_dir}/worker_gpu*.log")
        print(f"{'='*70}\n")
        print("Starting workers... (detailed logs in worker_gpu*.log files)\n")
        
        # Create shared queues and shutdown event
        manager = Manager()
        task_queue = manager.Queue()
        result_queue = manager.Queue()
        shutdown_event = manager.Event()
        
        # Check for existing progress (resume support)
        existing_count = self._get_next_design_index()
        completed_count = self._count_completed_designs()
        accepted_count = self._count_accepted_designs()
        
        if existing_count > 0:
            print(f"📁 Found {existing_count} existing design attempts, {completed_count} completed")
            print(f"   Resuming from design index {existing_count}...")
        
        next_design_idx = existing_count
        
        # Spawn worker processes
        workers = []
        for gpu_id in range(self.num_gpus):
            p = Process(
                target=_gpu_worker,
                args=(gpu_id, self.args, task_queue, result_queue, shutdown_event)
            )
            p.start()
            workers.append(p)
        
        # Initial work distribution - one design per worker
        in_flight = 0
        target_designs = self.args.num_designs or float('inf')
        target_accepted = self.args.num_accepted or float('inf')
        
        for _ in range(min(self.num_gpus, int(target_designs - completed_count))):
            task_queue.put(next_design_idx)
            next_design_idx += 1
            in_flight += 1
        
        # Main loop: monitor results and queue more work
        all_results = []
        rejected_count = 0
        try:
            while completed_count < target_designs and accepted_count < target_accepted:
                if in_flight == 0:
                    break  # No more work in flight and targets not met
                
                try:
                    result = result_queue.get(timeout=5.0)
                except:
                    continue  # Timeout - check conditions and retry
                
                in_flight -= 1
                all_results.append(result)
                
                # Print detailed result summary
                self._print_design_result(result)
                
                if result.get("has_valid_pdb"):
                    completed_count += 1
                
                if result.get("accepted"):
                    accepted_count += 1
                elif result.get("has_valid_pdb") or result.get("error"):
                    rejected_count += 1
                
                # Progress update
                progress_parts = [f"{completed_count}/{int(target_designs)} designs"]
                if self.args.num_accepted:
                    progress_parts.append(f"{accepted_count}/{int(target_accepted)} accepted")
                progress_parts.append(f"{rejected_count} rejected")
                print(f"{'─'*70}")
                print(f"Progress: {' | '.join(progress_parts)}")
                print(f"{'─'*70}\n")
                
                # Queue more work if needed
                if completed_count < target_designs and accepted_count < target_accepted:
                    task_queue.put(next_design_idx)
                    next_design_idx += 1
                    in_flight += 1
            
            print(f"\n✓ Target reached!")
            
        except KeyboardInterrupt:
            print("\n⚠ Interrupted! Waiting for workers to finish current designs...")
        
        finally:
            # Signal shutdown
            shutdown_event.set()
            
            # Send poison pills to stop workers
            for _ in range(self.num_gpus):
                task_queue.put(None)
            
            # Wait for workers to finish (they complete current design first)
            print("Waiting for workers to complete current designs...")
            for p in workers:
                p.join(timeout=600)  # 10 minute timeout per worker
                if p.is_alive():
                    print(f"Worker {p.pid} didn't finish in time, terminating...")
                    p.terminate()
        
        # Final summary
        self._print_final_summary(all_results)
    
    def _print_final_summary(self, all_results: list):
        """Print final summary of the multi-GPU run."""
        completed = self._count_completed_designs()
        accepted = self._count_accepted_designs()
        
        print(f"\n{'='*70}")
        print("MULTI-GPU PIPELINE COMPLETE")
        print(f"{'='*70}")
        print(f"Total designs generated: {completed}")
        print(f"Accepted: {accepted}")
        print(f"Rejected: {completed - accepted}")
        
        # Per-GPU stats
        gpu_counts = {}
        for r in all_results:
            gpu_id = r.get("gpu_id", -1)
            gpu_counts[gpu_id] = gpu_counts.get(gpu_id, 0) + 1
        
        if gpu_counts:
            print(f"\nPer-GPU distribution:")
            for gpu_id in sorted(gpu_counts.keys()):
                print(f"  GPU {gpu_id}: {gpu_counts[gpu_id]} designs")
        
        print(f"\nOutput structure:")
        print(f"  {self.save_dir}/")
        print(f"  ├── worker_gpu*.log       # Detailed per-worker logs")
        print(f"  ├── designs/              # All cycle PDBs + design_stats.csv")
        print(f"  ├── best_designs/         # Best per design + best_designs.csv")
        if getattr(self.args, "validation_model", "none") != "none":
            print(f"  ├── refolded/             # Refolded structures + metrics")
            print(f"  ├── accepted_designs/     # Passed filters")
            print(f"  └── rejected/             # Failed filters")
        print(f"\nTo view detailed logs: tail -f {self.save_dir}/worker_gpu*.log")
