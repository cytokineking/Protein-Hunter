import copy
import os
import random
import sys
import shutil
from collections import defaultdict
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import yaml

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from LigandMPNN.wrapper import LigandMPNNWrapper

from boltz_ph.constants import CHAIN_TO_NUMBER, UNIFIED_DESIGN_COLUMNS
from utils.metrics import get_CA_and_sequence # Used implicitly in design.py
from utils.convert import calculate_holo_apo_rmsd, convert_cif_files_to_pdb
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

        # Handle special "empty" case: --protein_msas "empty" applies to all seqs
        if a.msa_mode == "single":
            protein_msas_list = ["empty"] * len(protein_seqs_list)
        elif a.msa_mode == "mmseqs":
            protein_msas_list = ["mmseqs"] * len(protein_seqs_list)
        else:
            raise ValueError(f"Invalid msa_mode: {a.msa_mode}")

        return protein_seqs_list, protein_msas_list

    def _validate_and_convert_hotspots(self, protein_chain_ids: list[str]) -> tuple[str, dict[str, list[int]]]:
        """
        Validate hotspots and convert from auth to canonical numbering if needed.
        
        Args:
            protein_chain_ids: List of internal protein chain IDs (e.g., ['B', 'C'])
        
        Returns:
            Tuple of:
            - contact_residues_canonical: Converted contact_residues string for Boltz
            - hotspots_per_chain: Dict mapping internal_chain_id to list of canonical hotspot positions
        
        Raises:
            SystemExit: If hotspot validation fails
        """
        a = self.args
        
        if not a.contact_residues or not a.contact_residues.strip():
            self.contact_residues_canonical = ""
            self.contact_residues_auth = ""
            return "", {}
        
        residues_chains = a.contact_residues.split("|")
        converted_chains = []
        auth_chains = []  # Store original auth values
        hotspots_per_chain = {}
        
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
                converted_chains.append(",".join(str(p) for p in canonical_positions))
                auth_chains.append(",".join(str(p) for p in positions))  # Store original auth
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
                converted_chains.append(",".join(str(p) for p in positions))
                # If not using auth numbering, compute auth positions if we have the mapping
                if can_to_auth:
                    auth_positions = [can_to_auth.get(p, p) for p in positions]
                    auth_chains.append(",".join(str(p) for p in auth_positions))
                else:
                    auth_chains.append("")
        
        self.contact_residues_canonical = "|".join(converted_chains)
        self.contact_residues_auth = "|".join(auth_chains) if use_auth else ""
        
        return self.contact_residues_canonical, hotspots_per_chain


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
        contact_residues_canonical, hotspots_per_chain = self._validate_and_convert_hotspots(protein_chain_ids)

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
            use_auth = getattr(a, 'use_auth_numbering', False)
            template_source = f"template ({a.template_path})" if a.template_path else ""
            
            # Remap hotspots from internal chain ID (B, C) to template chain ID (A, B)
            # for display purposes
            template_chain_ids = getattr(self, '_template_chain_ids', [])
            hotspots_for_display = {}
            for i, template_chain_id in enumerate(template_chain_ids):
                internal_chain_id = protein_chain_ids[i] if i < len(protein_chain_ids) else chr(ord('B') + i)
                if internal_chain_id in hotspots_per_chain:
                    hotspots_for_display[template_chain_id] = hotspots_per_chain[internal_chain_id]
            
            print_target_analysis(
                self._template_analysis,
                hotspots_for_display,
                use_auth,
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
                binder_length, exclude_P=a.exclude_P, frac_X=a.percent_X/100
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
            if a.contact_residues.strip() and not a.no_contact_filter:
                try:
                    binds = all([binder_binds_contacts(
                        pdb_filename,
                        self.binder_chain,
                        protein_chain_ids[i],
                        contact_res,
                        cutoff=a.contact_cutoff,
                    )
                    for i, contact_res in enumerate(a.contact_residues.split("|"))
                ])
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
            new_seq = sample_seq(binder_length, exclude_P=a.exclude_P, frac_X=a.percent_X/100)
            update_binder_sequence(new_seq)
            clean_memory()

        clean_memory()
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
            "target_seqs": a.protein_seqs or "",
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
                "omit_AA": f"{a.omit_AA},P" if cycle == 0 else a.omit_AA,
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

            # Update best structure (only if alanine content is acceptable)
            if alanine_percentage <= 0.20 and current_iptm > best_iptm:
                best_iptm = current_iptm
                best_seq = seq
                best_structure = copy.deepcopy(structure)
                best_output = shallow_copy_tensor_dict(output)
                best_cycle_idx = cycle + 1
                best_alanine_percentage = alanine_percentage
                best_ipsae = current_ipsae
                best_alanine_count = alanine_count

            # 3. Log Metrics and Save PDB
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
                "target_seqs": a.protein_seqs or "",
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
                f"\nNo structure was generated for run {run_id} (no eligible best design with <= 20% alanine)."
            )

        # Finalize best metrics for CSV
        if best_alanine_percentage is not None and best_alanine_percentage <= 0.20:
            run_metrics["best_iptm"] = float(best_iptm)
            run_metrics["best_cycle"] = best_cycle_idx
            run_metrics["best_plddt"] = float(
                best_output.get("complex_plddt", torch.tensor([0.0]))
                .detach()
                .cpu()
                .numpy()[0]
            )
            run_metrics["best_iplddt"] = float(
                best_output.get("complex_iplddt", torch.tensor([0.0]))
                .detach()
                .cpu()
                .numpy()[0]
            )
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

    def _run_batch_validation_on_existing_designs(self):
        """
        Batch-validate ALL designs in best_designs/ folder using AF3 + PyRosetta.
        
        NOTE: This method is NOT called automatically during normal pipeline execution.
        The pipeline now uses design-by-design validation via _run_single_design_validation().
        
        This method is preserved for standalone/post-hoc validation use cases:
        
        USE CASES:
        ----------
        1. Re-validate designs generated without --use_alphafold3_validation
        2. Re-run validation with updated filter thresholds
        3. Validate designs from external sources (e.g., other design tools)
        4. Batch processing when design-by-design overhead is not desired
        
        FUTURE CLI INTEGRATION:
        -----------------------
        This could be exposed via a CLI flag like:
        
            python boltz_ph/design.py --validate-existing results_my_design/
        
        Or as a standalone script:
        
            python boltz_ph/validate_designs.py --results-dir results_my_design/
        
        REQUIREMENTS:
        -------------
        - best_designs/best_designs.csv must exist with design metadata
        - best_designs/*.pdb files should exist (used for AF3 input reconstruction)
        - AF3 Docker image must be available
        - PyRosetta must be installed (for protein targets)
        
        OUTPUT STRUCTURE:
        -----------------
        af3_validation/
        ├── af3_results.csv
        └── *_af3.cif
        accepted_designs/
        ├── accepted_stats.csv
        └── *_relaxed.pdb
        rejected/
        ├── rejected_stats.csv
        └── *_relaxed.pdb
        
        EXAMPLE PROGRAMMATIC USAGE:
        ---------------------------
        >>> # Create a minimal args object with required fields
        >>> args = argparse.Namespace(
        ...     alphafold_dir="~/alphafold3",
        ...     af3_docker_name="alphafold3",
        ...     af3_database_settings="~/alphafold3/alphafold3_data_save",
        ...     hmmer_path="~/.conda/envs/alphafold3_venv",
        ...     work_dir="",
        ...     gpu_id=0,
        ...     use_msa_for_af3=True,
        ...     ligand_smiles="", ligand_ccd="", nucleic_seq="", nucleic_type="dna",
        ... )
        >>> hunter = ProteinHunter_Boltz(args)
        >>> hunter.save_dir = "results_my_prior_run"
        >>> hunter._run_batch_validation_on_existing_designs()
        """
        import json
        import glob
        
        sys.path.append(os.path.join(os.path.dirname(__file__), "utils"))
        from utils.alphafold_utils import run_alphafold_step_from_csv, calculate_af3_ipsae
        from utils.pyrosetta_utils import run_rosetta_step

        a = self.args

        # Determine target type for Rosetta validation
        any_ligand_or_nucleic = a.ligand_smiles or a.ligand_ccd or a.nucleic_seq
        if a.nucleic_type.strip() and a.nucleic_seq.strip():
            target_type = "nucleic"
        elif any_ligand_or_nucleic:
            target_type = "small_molecule"
        else:
            target_type = "protein"  # Default for unconditional mode

        # Temporary working directory for AF3/Rosetta (will be cleaned up)
        work_dir_validation = f"{self.save_dir}/_validation_work"

        # Use best_designs.csv for downstream validation
        best_designs_csv = os.path.join(self.save_dir, "best_designs", "best_designs.csv")

        if not os.path.exists(best_designs_csv):
            print(f"No best_designs.csv found at {best_designs_csv}. Skipping downstream validation.")
            return

        print("Starting downstream validation (AlphaFold3 and Rosetta)...")
        print(f"Using designs from: {best_designs_csv}")

        # Load design metadata for later joining
        best_designs_df = pd.read_csv(best_designs_csv)
        design_metadata = {row["design_id"]: row.to_dict() for _, row in best_designs_df.iterrows()}

        # --- AlphaFold Step ---
        # Note: high_iptm=False to convert ALL CIFs without filtering.
        # This aligns with the Modal pipeline where PyRosetta handles all filtering
        # and records proper rejection reasons (including low AF3 ipTM).
        af_output_dir, af_output_apo_dir, af_pdb_dir, af_pdb_dir_apo = (
            run_alphafold_step_from_csv(
                csv_path=best_designs_csv,
                alphafold_dir=os.path.expanduser(a.alphafold_dir),
                af3_docker_name=a.af3_docker_name,
                af3_database_settings=os.path.expanduser(a.af3_database_settings),
                hmmer_path=os.path.expanduser(a.hmmer_path),
                ligandmpnn_dir=work_dir_validation,
                work_dir=os.path.expanduser(a.work_dir) or os.getcwd(),
                binder_id=self.binder_chain,
                gpu_id=a.gpu_id,
                high_iptm=False,  # Don't filter at AF3 stage - let PyRosetta handle all filtering
                use_msa_for_af3=a.use_msa_for_af3,
            )
        )

        # ===================================================================
        # STEP 1: CREATE AF3_VALIDATION FOLDER (BEFORE PyRosetta)
        # This ensures af3_validation/ is created even if PyRosetta fails
        # ===================================================================
        print("\nCreating af3_validation folder...")
        
        af3_dir = os.path.join(self.save_dir, "af3_validation")
        accepted_dir = os.path.join(self.save_dir, "accepted_designs")
        rejected_dir = os.path.join(self.save_dir, "rejected")
        os.makedirs(af3_dir, exist_ok=True)
        os.makedirs(accepted_dir, exist_ok=True)
        os.makedirs(rejected_dir, exist_ok=True)

        af3_results_rows = []
        
        # Find AF3 output CIFs and copy to af3_validation/
        if af_output_dir and os.path.exists(af_output_dir):
            for design_subdir in os.listdir(af_output_dir):
                design_path = os.path.join(af_output_dir, design_subdir)
                if not os.path.isdir(design_path):
                    continue
                
                # Find only the ranked best CIF (not samples) and confidence files
                # AF3 generates: design_id_model.cif (ranked best) + design_id_seed-X_sample-Y_model.cif
                cif_files = [f for f in glob.glob(os.path.join(design_path, "*_model.cif"))
                            if "_seed-" not in f and "_sample-" not in f]
                summary_conf_files = [f for f in glob.glob(os.path.join(design_path, "*_summary_confidences.json"))
                                     if "_seed-" not in f and "_sample-" not in f]
                conf_files = [f for f in glob.glob(os.path.join(design_path, "*_confidences.json"))
                             if "_seed-" not in f and "_sample-" not in f]
                
                if cif_files:
                    design_id = design_subdir
                    
                    # Copy CIF with new naming
                    src_cif = cif_files[0]
                    dest_cif = os.path.join(af3_dir, f"{design_id}_af3.cif")
                    shutil.copy(src_cif, dest_cif)
                    
                    # Extract AF3 metrics
                    af3_iptm = 0.0
                    af3_ptm = 0.0
                    af3_plddt = 0.0
                    af3_ipsae = 0.0
                    
                    if summary_conf_files:
                        try:
                            with open(summary_conf_files[0], 'r') as f:
                                summary = json.load(f)
                                af3_iptm = summary.get('iptm', 0.0)
                                af3_ptm = summary.get('ptm', 0.0)
                        except Exception as e:
                            print(f"  Warning: Could not read summary confidence for {design_id}: {e}")
                    
                    if conf_files:
                        try:
                            with open(conf_files[0], 'r') as f:
                                conf_json_text = f.read()
                            conf = json.loads(conf_json_text)
                            atom_plddts = conf.get('atom_plddts', [])
                            if atom_plddts:
                                af3_plddt = sum(atom_plddts) / len(atom_plddts)
                            
                            # Calculate AF3 ipSAE from PAE matrix
                            metadata = design_metadata.get(design_id, {})
                            binder_seq = metadata.get("binder_sequence", "")
                            target_seqs = metadata.get("target_seqs", "")
                            binder_length = len(binder_seq) if binder_seq else 0
                            # Handle target_seqs: could be colon-separated for multiple chains
                            target_length = len(target_seqs.split(":")[0]) if target_seqs else 0
                            
                            if binder_length > 0 and target_length > 0:
                                ipsae_result = calculate_af3_ipsae(
                                    conf_json_text, binder_length, target_length
                                )
                                af3_ipsae = ipsae_result.get("af3_ipsae", 0.0)
                        except Exception as e:
                            print(f"  Warning: Could not read confidence for {design_id}: {e}")
                    
                    af3_results_rows.append({
                        "design_id": design_id,
                        "af3_iptm": round(af3_iptm, 4),
                        "af3_ipsae": round(af3_ipsae, 4),
                        "af3_ptm": round(af3_ptm, 4),
                        "af3_plddt": round(af3_plddt, 2),
                    })

        # Save af3_results.csv
        if af3_results_rows:
            af3_results_df = pd.DataFrame(af3_results_rows)
            af3_results_df.to_csv(os.path.join(af3_dir, "af3_results.csv"), index=False)
            print(f"  ✓ Saved af3_validation/af3_results.csv ({len(af3_results_rows)} designs)")
        else:
            print("  Warning: No AF3 results found to save")

        # Check if we have any PDB files to process
        if not af_pdb_dir or not os.path.exists(af_pdb_dir) or not any(f.endswith(".pdb") for f in os.listdir(af_pdb_dir)):
            print("No AF3 PDB conversions available. Skipping PyRosetta step.")
            print(f"\nResults saved to: {self.save_dir}")
            print(f"  - af3_validation/af3_results.csv: {len(af3_results_rows)} designs")
            return

        # ===================================================================
        # STEP 2: PYROSETTA FILTERING
        # ===================================================================
        if target_type == "protein":
            print("\nRunning PyRosetta filtering...")
            run_rosetta_step(
                work_dir_validation,
                af_pdb_dir,
                af_pdb_dir_apo,
                binder_id=self.binder_chain,
                target_type=target_type,
            )

        # ===================================================================
        # STEP 3: REORGANIZE ACCEPTED/REJECTED FROM ROSETTA RESULTS
        # ===================================================================
        print("\nReorganizing outputs to standardized structure...")

        # --- Reorganize accepted/rejected from Rosetta results ---
        rosetta_success_dir = os.path.join(work_dir_validation, "af_pdb_rosetta_success")
        success_csv = os.path.join(rosetta_success_dir, "success_designs.csv")
        failed_csv = os.path.join(rosetta_success_dir, "failed_designs.csv")

        accepted_rows = []
        rejected_rows = []

        # Create af3_results lookup for joining
        af3_lookup = {r["design_id"]: r for r in af3_results_rows}

        # Process success designs (check file exists AND has data)
        if os.path.exists(success_csv) and os.path.getsize(success_csv) > 0:
            try:
                success_df = pd.read_csv(success_csv)
            except pd.errors.EmptyDataError:
                success_df = pd.DataFrame()  # Empty dataframe if file has no data
            for _, row in success_df.iterrows():
                # Extract design_id from Model name (e.g., "relax_design_id_model.pdb")
                model_name = row.get("Model", "")
                design_id = model_name.replace("relax_", "").replace("_model.pdb", "")
                
                # Get design metadata
                metadata = design_metadata.get(design_id, {})
                af3_data = af3_lookup.get(design_id, {})
                
                # Build full row
                full_row = {
                    "design_id": design_id,
                    "design_num": metadata.get("design_num", 0),
                    "cycle": metadata.get("cycle", 0),
                    "binder_sequence": metadata.get("binder_sequence", row.get("aa_seq", "")),
                    "binder_length": metadata.get("binder_length", len(row.get("aa_seq", ""))),
                    "cyclic": metadata.get("cyclic", False),
                    "alanine_count": metadata.get("alanine_count", 0),
                    "alanine_pct": metadata.get("alanine_pct", 0.0),
                    # Boltz design metrics (from best_designs.csv)
                    "boltz_iptm": metadata.get("boltz_iptm", 0.0),
                    "boltz_ipsae": metadata.get("boltz_ipsae", 0.0),
                    "boltz_plddt": metadata.get("boltz_plddt", 0.0),
                    "boltz_iplddt": metadata.get("boltz_iplddt", 0.0),
                    # AF3 validation metrics
                    "af3_iptm": af3_data.get("af3_iptm", row.get("iptm", 0)),
                    "af3_ipsae": af3_data.get("af3_ipsae", 0.0),
                    "af3_ptm": af3_data.get("af3_ptm", 0),
                    "af3_plddt": af3_data.get("af3_plddt", row.get("plddt", 0)),
                    "accepted": True,
                    "rejection_reason": "",
                    # PyRosetta metrics
                    "binder_score": row.get("binder_score", 0),
                    "total_score": row.get("total_score", 0),
                    "interface_sc": row.get("interface_sc", 0),
                    "interface_packstat": row.get("interface_packstat", 0),
                    "interface_dG": row.get("interface_dG", 0),
                    "interface_dSASA": row.get("interface_dSASA", 0),
                    "interface_dG_SASA_ratio": row.get("interface_dG_SASA_ratio", 0),
                    "interface_nres": row.get("interface_nres", 0),
                    "interface_hbonds": row.get("interface_interface_hbonds", 0),  # Renamed from interface_interface_hbonds
                    "interface_hbond_percentage": row.get("interface_hbond_percentage", 0),
                    "interface_delta_unsat_hbonds": row.get("interface_delta_unsat_hbonds", 0),
                    "interface_delta_unsat_hbonds_percentage": row.get("interface_delta_unsat_hbonds_percentage", 0),
                    "interface_hydrophobicity": row.get("interface_hydrophobicity", 0),
                    "surface_hydrophobicity": row.get("surface_hydrophobicity", 0),
                    "binder_sasa": row.get("binder_sasa", 0),
                    "interface_fraction": row.get("interface_fraction", 0),
                    # Secondary metrics
                    "apo_holo_rmsd": row.get("apo_holo_rmsd"),
                    "i_pae": row.get("i_pae"),
                    "rg": row.get("rg"),
                }
                accepted_rows.append(full_row)
                
                # Copy relaxed PDB
                src_pdb = os.path.join(row.get("PDB", ""), model_name) if row.get("PDB") else None
                if src_pdb and os.path.exists(src_pdb):
                    dest_pdb = os.path.join(accepted_dir, f"{design_id}_relaxed.pdb")
                    shutil.copy(src_pdb, dest_pdb)

        # Process failed designs (check file exists AND has data)
        if os.path.exists(failed_csv) and os.path.getsize(failed_csv) > 0:
            try:
                failed_df = pd.read_csv(failed_csv)
            except pd.errors.EmptyDataError:
                failed_df = pd.DataFrame()  # Empty dataframe if file has no data
            for _, row in failed_df.iterrows():
                model_name = row.get("Model", "")
                design_id = model_name.replace("relax_", "").replace("_model.pdb", "")
                
                metadata = design_metadata.get(design_id, {})
                af3_data = af3_lookup.get(design_id, {})
                
                full_row = {
                    "design_id": design_id,
                    "design_num": metadata.get("design_num", 0),
                    "cycle": metadata.get("cycle", 0),
                    "binder_sequence": metadata.get("binder_sequence", row.get("aa_seq", "")),
                    "binder_length": metadata.get("binder_length", len(row.get("aa_seq", "")) if row.get("aa_seq") else 0),
                    "cyclic": metadata.get("cyclic", False),
                    "alanine_count": metadata.get("alanine_count", 0),
                    "alanine_pct": metadata.get("alanine_pct", 0.0),
                    # Boltz design metrics (from best_designs.csv)
                    "boltz_iptm": metadata.get("boltz_iptm", 0.0),
                    "boltz_ipsae": metadata.get("boltz_ipsae", 0.0),
                    "boltz_plddt": metadata.get("boltz_plddt", 0.0),
                    "boltz_iplddt": metadata.get("boltz_iplddt", 0.0),
                    # AF3 validation metrics
                    "af3_iptm": af3_data.get("af3_iptm", row.get("iptm", 0)),
                    "af3_ipsae": af3_data.get("af3_ipsae", 0.0),
                    "af3_ptm": af3_data.get("af3_ptm", 0),
                    "af3_plddt": af3_data.get("af3_plddt", row.get("plddt", 0)),
                    "accepted": False,
                    "rejection_reason": row.get("failure_reason", ""),
                    # PyRosetta metrics
                    "binder_score": row.get("binder_score", 0),
                    "total_score": row.get("total_score", 0),
                    "interface_sc": row.get("interface_sc", 0),
                    "interface_packstat": row.get("interface_packstat", 0),
                    "interface_dG": row.get("interface_dG", 0),
                    "interface_dSASA": row.get("interface_dSASA", 0),
                    "interface_dG_SASA_ratio": row.get("interface_dG_SASA_ratio", 0),
                    "interface_nres": row.get("interface_nres", 0),
                    "interface_hbonds": row.get("interface_interface_hbonds", 0),  # Renamed from interface_interface_hbonds
                    "interface_hbond_percentage": row.get("interface_hbond_percentage", 0),
                    "interface_delta_unsat_hbonds": row.get("interface_delta_unsat_hbonds", 0),
                    "interface_delta_unsat_hbonds_percentage": row.get("interface_delta_unsat_hbonds_percentage", 0),
                    "interface_hydrophobicity": row.get("interface_hydrophobicity", 0),
                    "surface_hydrophobicity": row.get("surface_hydrophobicity", 0),
                    "binder_sasa": row.get("binder_sasa", 0),
                    "interface_fraction": row.get("interface_fraction", 0),
                    # Secondary metrics
                    "apo_holo_rmsd": row.get("apo_holo_rmsd"),
                    "i_pae": row.get("i_pae"),
                    "rg": row.get("rg"),
                }
                rejected_rows.append(full_row)
                
                # Copy relaxed PDB to rejected folder
                src_pdb = os.path.join(row.get("PDB", ""), model_name) if row.get("PDB") else None
                if src_pdb and os.path.exists(src_pdb):
                    dest_pdb = os.path.join(rejected_dir, f"{design_id}_relaxed.pdb")
                    shutil.copy(src_pdb, dest_pdb)

        # Save accepted/rejected CSVs
        if accepted_rows:
            accepted_df = pd.DataFrame(accepted_rows)
            accepted_df = _reorder_columns(accepted_df)
            accepted_df.to_csv(os.path.join(accepted_dir, "accepted_stats.csv"), index=False)
            print(f"  ✓ Saved accepted_designs/accepted_stats.csv ({len(accepted_rows)} designs)")

        if rejected_rows:
            rejected_df = pd.DataFrame(rejected_rows)
            rejected_df = _reorder_columns(rejected_df)
            rejected_df.to_csv(os.path.join(rejected_dir, "rejected_stats.csv"), index=False)
            print(f"  ✓ Saved rejected/rejected_stats.csv ({len(rejected_rows)} designs)")

        # --- Step 3: Clean up old validation work directory ---
        try:
            shutil.rmtree(work_dir_validation, ignore_errors=True)
            print(f"  ✓ Cleaned up temporary validation directory")
        except Exception as e:
            print(f"  Warning: Could not clean up {work_dir_validation}: {e}")

        # Print summary
        print(f"\n{'='*60}")
        print("DOWNSTREAM VALIDATION COMPLETE")
        print(f"{'='*60}")
        print(f"AF3 validated: {len(af3_results_rows)} designs")
        print(f"Accepted: {len(accepted_rows)}")
        print(f"Rejected: {len(rejected_rows)}")
        print(f"\nOutput structure:")
        print(f"  {self.save_dir}/")
        print(f"  ├── af3_validation/       # AF3 structures + metrics")
        print(f"  │   ├── af3_results.csv")
        print(f"  │   └── *_af3.cif")
        print(f"  ├── accepted_designs/     # Passed filters")
        print(f"  │   ├── accepted_stats.csv")
        print(f"  │   └── *_relaxed.pdb")
        print(f"  └── rejected/             # Failed filters")
        print(f"      ├── rejected_stats.csv")
        print(f"      └── *_relaxed.pdb")

    def _run_single_design_validation(self, design_metrics: dict) -> dict:
        """
        Run AF3 + PyRosetta validation for a SINGLE design.
        
        This implements the design-by-design execution model where each design
        is fully validated before proceeding to the next. This enables:
        - Resumable execution at the validation level
        - Early stopping when num_accepted target is reached
        - Real-time progress tracking
        
        Args:
            design_metrics: Output from _run_design_cycle() for one design
        
        Returns:
            dict with design_metrics + af3 results + pyrosetta results + acceptance status
        """
        import json
        import glob
        
        from utils.alphafold_utils import run_alphafold_step_from_csv, calculate_af3_ipsae
        from utils.pyrosetta_utils import run_rosetta_step
        
        a = self.args
        design_idx = int(design_metrics.get("run_id", 0))
        best_cycle = design_metrics.get("best_cycle", 0)
        design_id = f"{a.name}_d{design_idx}_c{best_cycle}"
        
        # Initialize result with design metrics
        result = {
            "design_id": design_id,
            "design_num": design_idx,
            "cycle": best_cycle,
            "binder_sequence": design_metrics.get("best_seq", ""),
            "binder_length": len(design_metrics.get("best_seq", "")),
            "cyclic": a.cyclic,
            "alanine_count": design_metrics.get("best_alanine_count", 0),
            "alanine_pct": 0.0,
            "boltz_iptm": design_metrics.get("best_iptm", 0.0),
            "boltz_ipsae": design_metrics.get("best_ipsae", 0.0),
            "boltz_plddt": design_metrics.get("best_plddt", 0.0),
            "boltz_iplddt": design_metrics.get("best_iplddt", 0.0),
            # AF3 metrics (initialized)
            "af3_iptm": 0.0,
            "af3_ipsae": 0.0,
            "af3_ptm": 0.0,
            "af3_plddt": 0.0,
            # Acceptance status
            "accepted": False,
            "rejection_reason": "validation_not_run",
        }
        
        # Calculate alanine percentage
        binder_length = result["binder_length"]
        if binder_length > 0:
            result["alanine_pct"] = round(result["alanine_count"] / binder_length * 100, 2)
        
        # Check if we have a valid design to validate
        best_pdb = design_metrics.get("best_pdb_filename")
        if not best_pdb or not os.path.exists(best_pdb):
            result["rejection_reason"] = "no_valid_design"
            return result
        
        print(f"  Running AF3 + PyRosetta validation for {design_id}...")
        
        # Create temporary working directory for this design's validation
        work_dir_validation = os.path.join(self.save_dir, "_validation_work", design_id)
        os.makedirs(work_dir_validation, exist_ok=True)
        
        try:
            # ===================================================================
            # STEP 1: CREATE SINGLE-DESIGN CSV FOR AF3
            # ===================================================================
            temp_csv_dir = os.path.join(work_dir_validation, "temp_csv")
            os.makedirs(temp_csv_dir, exist_ok=True)
            temp_csv_path = os.path.join(temp_csv_dir, "single_design.csv")
            
            # Build row matching best_designs.csv format
            csv_row = {
                "design_id": design_id,
                "design_num": design_idx,
                "cycle": best_cycle,
                "binder_sequence": design_metrics.get("best_seq", ""),
                "binder_length": binder_length,
                "cyclic": a.cyclic,
                "boltz_iptm": design_metrics.get("best_iptm", 0.0),
                "boltz_ipsae": design_metrics.get("best_ipsae", 0.0),
                "boltz_plddt": design_metrics.get("best_plddt", 0.0),
                "boltz_iplddt": design_metrics.get("best_iplddt", 0.0),
                "target_seqs": a.protein_seqs or "",
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
            
            # ===================================================================
            # STEP 2: RUN AF3 FOR THIS SINGLE DESIGN
            # ===================================================================
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
                    gpu_id=a.gpu_id,
                    high_iptm=False,
                    use_msa_for_af3=a.use_msa_for_af3,
                )
            )
            
            # ===================================================================
            # STEP 3: PARSE AF3 OUTPUT
            # ===================================================================
            af3_cif_path = None
            
            if af_output_dir and os.path.exists(af_output_dir):
                for design_subdir in os.listdir(af_output_dir):
                    design_path = os.path.join(af_output_dir, design_subdir)
                    if not os.path.isdir(design_path):
                        continue
                    
                    # Find ranked best CIF
                    cif_files = [f for f in glob.glob(os.path.join(design_path, "*_model.cif"))
                                if "_seed-" not in f and "_sample-" not in f]
                    summary_conf_files = [f for f in glob.glob(os.path.join(design_path, "*_summary_confidences.json"))
                                         if "_seed-" not in f and "_sample-" not in f]
                    conf_files = [f for f in glob.glob(os.path.join(design_path, "*_confidences.json"))
                                 if "_seed-" not in f and "_sample-" not in f]
                    
                    if cif_files:
                        af3_cif_path = cif_files[0]
                        
                        # Parse summary confidences
                        if summary_conf_files:
                            try:
                                with open(summary_conf_files[0], 'r') as f:
                                    summary = json.load(f)
                                    result["af3_iptm"] = round(summary.get('iptm', 0.0), 4)
                                    result["af3_ptm"] = round(summary.get('ptm', 0.0), 4)
                            except Exception as e:
                                print(f"    Warning: Could not read summary confidence: {e}")
                        
                        # Parse full confidences for pLDDT and ipSAE
                        if conf_files:
                            try:
                                with open(conf_files[0], 'r') as f:
                                    conf_json_text = f.read()
                                conf = json.loads(conf_json_text)
                                
                                # Calculate pLDDT
                                atom_plddts = conf.get('atom_plddts', [])
                                if atom_plddts:
                                    result["af3_plddt"] = round(sum(atom_plddts) / len(atom_plddts), 2)
                                
                                # Calculate AF3 ipSAE
                                binder_seq = design_metrics.get("best_seq", "")
                                target_seqs = a.protein_seqs or ""
                                target_length = len(target_seqs.split(":")[0]) if target_seqs else 0
                                
                                if binder_length > 0 and target_length > 0:
                                    ipsae_result = calculate_af3_ipsae(
                                        conf_json_text, binder_length, target_length
                                    )
                                    result["af3_ipsae"] = ipsae_result.get("af3_ipsae", 0.0)
                            except Exception as e:
                                print(f"    Warning: Could not read confidence: {e}")
                        
                        break  # Found our design
            
            # ===================================================================
            # STEP 4: COPY AF3 CIF TO af3_validation/
            # ===================================================================
            af3_dir = os.path.join(self.save_dir, "af3_validation")
            os.makedirs(af3_dir, exist_ok=True)
            
            if af3_cif_path and os.path.exists(af3_cif_path):
                dest_cif = os.path.join(af3_dir, f"{design_id}_af3.cif")
                shutil.copy(af3_cif_path, dest_cif)
            
            # ===================================================================
            # STEP 5: RUN PYROSETTA (if we have PDBs and target is protein)
            # ===================================================================
            any_ligand_or_nucleic = a.ligand_smiles or a.ligand_ccd or a.nucleic_seq
            if a.nucleic_type.strip() and a.nucleic_seq.strip():
                target_type = "nucleic"
            elif any_ligand_or_nucleic:
                target_type = "small_molecule"
            else:
                target_type = "protein"
            
            pyrosetta_result = None
            
            if af_pdb_dir and os.path.exists(af_pdb_dir) and target_type == "protein":
                pdb_files = [f for f in os.listdir(af_pdb_dir) if f.endswith(".pdb")]
                if pdb_files:
                    # Run PyRosetta
                    run_rosetta_step(
                        work_dir_validation,
                        af_pdb_dir,
                        af_pdb_dir_apo,
                        binder_id=self.binder_chain,
                        target_type=target_type,
                    )
                    
                    # Parse PyRosetta results
                    rosetta_success_dir = os.path.join(work_dir_validation, "af_pdb_rosetta_success")
                    success_csv = os.path.join(rosetta_success_dir, "success_designs.csv")
                    failed_csv = os.path.join(rosetta_success_dir, "failed_designs.csv")
                    
                    # Check success
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
                                    "binder_score": row.get("binder_score", 0),
                                    "total_score": row.get("total_score", 0),
                                    "interface_sc": row.get("interface_sc", 0),
                                    "interface_packstat": row.get("interface_packstat", 0),
                                    "interface_dG": row.get("interface_dG", 0),
                                    "interface_dSASA": row.get("interface_dSASA", 0),
                                    "interface_dG_SASA_ratio": row.get("interface_dG_SASA_ratio", 0),
                                    "interface_nres": row.get("interface_nres", 0),
                                    "interface_hbonds": row.get("interface_interface_hbonds", 0),
                                    "interface_hbond_percentage": row.get("interface_hbond_percentage", 0),
                                    "interface_delta_unsat_hbonds": row.get("interface_delta_unsat_hbonds", 0),
                                    "interface_delta_unsat_hbonds_percentage": row.get("interface_delta_unsat_hbonds_percentage", 0),
                                    "interface_hydrophobicity": row.get("interface_hydrophobicity", 0),
                                    "surface_hydrophobicity": row.get("surface_hydrophobicity", 0),
                                    "binder_sasa": row.get("binder_sasa", 0),
                                    "interface_fraction": row.get("interface_fraction", 0),
                                    "apo_holo_rmsd": row.get("apo_holo_rmsd"),
                                    "i_pae": row.get("i_pae"),
                                    "rg": row.get("rg"),
                                }
                        except pd.errors.EmptyDataError:
                            pass
                    
                    # Check failed
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
                                    "binder_score": row.get("binder_score", 0),
                                    "total_score": row.get("total_score", 0),
                                    "interface_sc": row.get("interface_sc", 0),
                                    "interface_packstat": row.get("interface_packstat", 0),
                                    "interface_dG": row.get("interface_dG", 0),
                                    "interface_dSASA": row.get("interface_dSASA", 0),
                                    "interface_dG_SASA_ratio": row.get("interface_dG_SASA_ratio", 0),
                                    "interface_nres": row.get("interface_nres", 0),
                                    "interface_hbonds": row.get("interface_interface_hbonds", 0),
                                    "interface_hbond_percentage": row.get("interface_hbond_percentage", 0),
                                    "interface_delta_unsat_hbonds": row.get("interface_delta_unsat_hbonds", 0),
                                    "interface_delta_unsat_hbonds_percentage": row.get("interface_delta_unsat_hbonds_percentage", 0),
                                    "interface_hydrophobicity": row.get("interface_hydrophobicity", 0),
                                    "surface_hydrophobicity": row.get("surface_hydrophobicity", 0),
                                    "binder_sasa": row.get("binder_sasa", 0),
                                    "interface_fraction": row.get("interface_fraction", 0),
                                    "apo_holo_rmsd": row.get("apo_holo_rmsd"),
                                    "i_pae": row.get("i_pae"),
                                    "rg": row.get("rg"),
                                }
                        except pd.errors.EmptyDataError:
                            pass
            elif target_type != "protein":
                # Non-protein targets: accept without PyRosetta filtering
                pyrosetta_result = {
                    "accepted": True,
                    "rejection_reason": "",
                }
            
            # ===================================================================
            # STEP 6: MERGE PYROSETTA RESULTS INTO RESULT
            # ===================================================================
            if pyrosetta_result:
                result["accepted"] = pyrosetta_result.get("accepted", False)
                result["rejection_reason"] = pyrosetta_result.get("rejection_reason", "")
                
                # Copy all PyRosetta metrics
                for key in ["binder_score", "total_score", "interface_sc", "interface_packstat",
                           "interface_dG", "interface_dSASA", "interface_dG_SASA_ratio",
                           "interface_nres", "interface_hbonds", "interface_hbond_percentage",
                           "interface_delta_unsat_hbonds", "interface_delta_unsat_hbonds_percentage",
                           "interface_hydrophobicity", "surface_hydrophobicity", "binder_sasa",
                           "interface_fraction", "apo_holo_rmsd", "i_pae", "rg"]:
                    if key in pyrosetta_result:
                        result[key] = pyrosetta_result[key]
                
                # Copy relaxed PDB to appropriate folder
                if pyrosetta_result.get("pdb_path") and pyrosetta_result.get("model_name"):
                    src_pdb = os.path.join(pyrosetta_result["pdb_path"], pyrosetta_result["model_name"])
                    if os.path.exists(src_pdb):
                        if result["accepted"]:
                            dest_dir = os.path.join(self.save_dir, "accepted_designs")
                        else:
                            dest_dir = os.path.join(self.save_dir, "rejected")
                        os.makedirs(dest_dir, exist_ok=True)
                        dest_pdb = os.path.join(dest_dir, f"{design_id}_relaxed.pdb")
                        shutil.copy(src_pdb, dest_pdb)
            else:
                result["rejection_reason"] = "validation_failed"
            
        except Exception as e:
            import traceback
            print(f"    Error during validation: {e}")
            traceback.print_exc()
            result["rejection_reason"] = f"validation_error: {str(e)}"
        
        finally:
            # Clean up temporary validation directory
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
        - af3_validation/af3_results.csv
        - accepted_designs/accepted_stats.csv (if accepted)
        - rejected/rejected_stats.csv (if rejected)
        
        Args:
            result: Full result dict from _run_single_design_validation()
        """
        # Save AF3 results
        af3_dir = os.path.join(self.save_dir, "af3_validation")
        os.makedirs(af3_dir, exist_ok=True)
        
        af3_row = {
            "design_id": result.get("design_id"),
            "af3_iptm": result.get("af3_iptm", 0.0),
            "af3_ipsae": result.get("af3_ipsae", 0.0),
            "af3_ptm": result.get("af3_ptm", 0.0),
            "af3_plddt": result.get("af3_plddt", 0.0),
        }
        append_to_csv_safe(Path(af3_dir) / "af3_results.csv", af3_row)
        
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
                print(f"  ✗ Design {design_idx} failed: all cycles had >20% alanine")
            
            # 5. Run validation for THIS design (design-by-design execution)
            if self.args.use_alphafold3_validation and has_valid_pdb:
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
        if self.args.use_alphafold3_validation:
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
            print(f"  ├── af3_validation/       # AF3 structures + metrics")
            print(f"  │   ├── af3_results.csv")
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
            # All cycles failed (e.g., all >20% alanine) - record as rejected
            design_id = f"{a.name}_d{design_idx}_failed"
            seq = ""
            binder_length = 0
            alanine_count = 0
            alanine_pct = 0.0
            best_cycle = None
            
            accepted = False
            rejection_reason = "alanine_pct > 20% (all cycles)"
        
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
            "target_seqs": a.protein_seqs or "",
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