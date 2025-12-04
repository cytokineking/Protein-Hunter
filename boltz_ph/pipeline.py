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

from boltz_ph.constants import CHAIN_TO_NUMBER
from utils.metrics import get_CA_and_sequence # Used implicitly in design.py
from utils.convert import calculate_holo_apo_rmsd, convert_cif_files_to_pdb
from utils.ipsae_utils import calculate_ipsae_from_boltz_output
from utils.csv_utils import append_to_csv_safe


from boltz_ph.model_utils import (
    binder_binds_contacts,
    clean_memory,
    design_sequence,
    get_boltz_model,
    get_cif,
    load_canonicals,
    plot_from_pdb,
    plot_run_metrics,
    process_msa,
    run_prediction,
    sample_seq,
    save_pdb,
    shallow_copy_tensor_dict,
    smart_split,
)

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


    def _process_sequence_inputs(self):
        """
        Parses and groups protein sequences and MSAs from command line arguments.
        Ensures protein_seqs_list and protein_msas_list are aligned and padded.
        """
        a = self.args
        
        protein_seqs_list = smart_split(a.protein_seqs) if a.protein_seqs else []

        # Handle special "empty" case: --protein_msas "empty" applies to all seqs
        if a.msa_mode == "single":
            protein_msas_list = ["empty"] * len(protein_seqs_list)
        elif a.msa_mode == "mmseqs":
            protein_msas_list = ["mmseqs"] * len(protein_seqs_list)
        else:
            raise ValueError(f"Invalid msa_mode: {a.msa_mode}")

        return protein_seqs_list, protein_msas_list


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
        # --- END NEW ---

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

        # Step 6: Add constraints (pocket conditioning)
        pocket_conditioning = bool(a.contact_residues and a.contact_residues.strip())
        if pocket_conditioning:
            contacts = []
            residues_chains = a.contact_residues.split("|")
            for i, residues_chain in enumerate(residues_chains):
                residues = residues_chain.split(",")
                contacts.extend([
                    [protein_chain_ids[i], int(res)]
                    for res in residues
                    if res.strip() != ""
                ])
            constraints = [{"pocket": {"binder": "A", "contacts": contacts}}]
            data["constraints"] = constraints

        return data, pocket_conditioning

    def _build_templates(self, protein_chain_ids):
        """
        Constructs the list of template dictionaries.
        """
        a = self.args
        templates = []
        if a.template_path:
            template_path_list = smart_split(a.template_path)
            # We use the internal protein_chain_ids list
            template_cif_chain_id_list = (
                smart_split(a.template_cif_chain_id)
                if a.template_cif_chain_id
                else []
            )
            
            # Use protein_chain_ids to determine the number of expected templates
            num_proteins = len(protein_chain_ids)
            
            # Pad template paths list to match number of proteins
            if len(template_path_list) != num_proteins:
                print(f"Warning: Mismatch between number of proteins ({num_proteins}) and template paths ({len(template_path_list)}). Padding with empty entries.")
                while len(template_path_list) < num_proteins:
                    template_path_list.append("")
            
            # Pad cif chains list to match number of proteins
            if len(template_cif_chain_id_list) != num_proteins:
                print(f"Warning: Mismatch between number of proteins ({num_proteins}) and template CIF chains ({len(template_cif_chain_id_list)}). Padding with empty entries.")
                while len(template_cif_chain_id_list) < num_proteins:
                    template_cif_chain_id_list.append("")
            
            # Now, iterate up to num_proteins, linking them
            for i in range(num_proteins):
                template_file_path = template_path_list[i]
                if not template_file_path:
                    continue # Skip if no template path for this protein
                    
                template_file = get_cif(template_file_path)
                
                t_block = (
                    {"cif": template_file}
                    if template_file.endswith(".cif")
                    else {"pdb": template_file}
                )
                
                t_block["chain_id"] = protein_chain_ids[i] # e.g., 'B'
                
                # Only add cif_chain_id if provided for this template
                cif_chain = template_cif_chain_id_list[i]
                if cif_chain:
                    t_block["cif_chain_id"] = cif_chain # e.g., 'P'
                
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
            "contact_residues": a.contact_residues or "",
            "msa_mode": a.msa_mode,
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
                "contact_residues": a.contact_residues or "",
                "msa_mode": a.msa_mode,
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

    def _save_summary_metrics(self, all_run_metrics):
        """Saves all run metrics to a single CSV file."""
        a = self.args
        columns = ["run_id"]
        # Columns for cycle 0 through num_cycles
        for i in range(a.num_cycles + 1):
            columns.extend(
                [
                    f"cycle_{i}_iptm",
                    f"cycle_{i}_ipsae",
                    f"cycle_{i}_plddt",
                    f"cycle_{i}_iplddt",
                    f"cycle_{i}_alanine",
                    f"cycle_{i}_seq",
                ]
            )
        # Best metric columns
        columns.extend(["best_iptm", "best_cycle", "best_plddt", "best_seq"])

        summary_csv = os.path.join(self.save_dir, "summary_all_runs.csv")
        df = pd.DataFrame(all_run_metrics)
        
        # Ensure all expected columns are present (filling missing with NaN)
        for col in columns:
            if col not in df.columns:
                df[col] = float("nan")
        
        # Filter to columns in the correct order (and existing ones)
        df = df[[c for c in columns if c in df.columns]]
        df.to_csv(summary_csv, index=False)
        print(f"\n✅ All run/cycle metrics saved to {summary_csv}")

    def _run_downstream_validation(self):
        # This ensures they are in scope when called later in this function
        sys.path.append(os.path.join(os.path.dirname(__file__), "utils"))
        from utils.alphafold_utils import run_alphafold_step
        from utils.pyrosetta_utils import run_rosetta_step

        """Executes AlphaFold and Rosetta validation steps."""
        a = self.args

        # Determine target type for Rosetta validation
        any_ligand_or_nucleic = a.ligand_smiles or a.ligand_ccd or a.nucleic_seq
        if a.nucleic_type.strip() and a.nucleic_seq.strip():
            target_type = "nucleic"
        elif any_ligand_or_nucleic:
            target_type = "small_molecule"
        else:
            target_type = "protein"  # Default for unconditional mode

        success_dir = f"{self.save_dir}/1_af3_rosetta_validation"

        # Use best_designs/ folder for downstream validation (replaces high_iptm_yaml/)
        best_designs_dir = os.path.join(self.save_dir, "best_designs")

        if os.path.exists(best_designs_dir) and list(Path(best_designs_dir).glob("*.pdb")):
            print("Starting downstream validation (AlphaFold3 and Rosetta)...")
            print(f"Using designs from: {best_designs_dir}")

            # Note: alphafold_utils may need updates to work with PDBs instead of YAML
            # For now, we pass the best_designs_dir which contains PDBs
            # TODO: Update alphafold_utils.run_alphafold_step to work with new structure

            # --- AlphaFold Step ---
            af_output_dir, af_output_apo_dir, af_pdb_dir, af_pdb_dir_apo = (
                run_alphafold_step(
                    best_designs_dir,  # Now passes best_designs/ with PDBs
                    os.path.expanduser(a.alphafold_dir),
                    a.af3_docker_name,
                    os.path.expanduser(a.af3_database_settings),
                    os.path.expanduser(a.hmmer_path),
                    success_dir,
                    os.path.expanduser(a.work_dir) or os.getcwd(),
                    binder_id=self.binder_chain,
                    gpu_id=a.gpu_id,
                    high_iptm=True,
                    use_msa_for_af3=a.use_msa_for_af3,
                )
            )
            if target_type == "protein":
                # --- Rosetta Step ---
                run_rosetta_step(
                    success_dir,
                    af_pdb_dir,
                    af_pdb_dir_apo,
                    binder_id=self.binder_chain,
                    target_type=target_type,
                )
        else:
            print("No best designs found for downstream validation.")


    def run_pipeline(self):
        """Orchestrates the entire protein design and validation pipeline."""
        # 1. Prepare Base Data (using the new InputDataBuilder)
        base_data, pocket_conditioning = self.data_builder.build()

        # 2. Run Design Cycles
        all_run_metrics = []
        for design_id in range(self.args.num_designs):
            run_id = str(design_id)
            print("\n=======================================================")
            print(f"=== Starting Design Run {run_id}/{self.args.num_designs - 1} ===")
            print("=======================================================")

            data_cp = copy.deepcopy(base_data)

            run_metrics = self._run_design_cycle(data_cp, run_id, pocket_conditioning)
            all_run_metrics.append(run_metrics)

        # 3. Save best_designs/ folder (copy best cycle per design)
        self._save_best_designs(all_run_metrics)

        # 4. Print summary
        self._print_summary(all_run_metrics)

        # 5. Run Downstream Validation
        if self.args.use_alphafold3_validation:
            self._run_downstream_validation()

    def _save_best_designs(self, all_run_metrics):
        """Copy best cycle PDBs to best_designs/ folder and create best_designs.csv."""
        a = self.args
        best_dir = os.path.join(self.save_dir, "best_designs")
        os.makedirs(best_dir, exist_ok=True)

        best_rows = []
        for metrics in all_run_metrics:
            best_pdb = metrics.get("best_pdb_filename")
            if best_pdb and os.path.exists(best_pdb):
                # Copy PDB to best_designs/
                design_idx = int(metrics.get("run_id", 0))
                best_cycle = metrics.get("best_cycle", 0)
                design_id = f"{a.name}_d{design_idx}_c{best_cycle}"
                dest_pdb = os.path.join(best_dir, f"{design_id}.pdb")
                shutil.copy(best_pdb, dest_pdb)

                # Build row for best_designs.csv
                seq = metrics.get("best_seq", "")
                binder_length = len(seq) if seq else 0
                alanine_count = metrics.get("best_alanine_count", 0)
                alanine_pct = (alanine_count / binder_length * 100) if binder_length > 0 else 0.0

                best_rows.append({
                    "design_id": design_id,
                    "design_num": design_idx,
                    "cycle": best_cycle,
                    "binder_sequence": seq,
                    "binder_length": binder_length,
                    "cyclic": a.cyclic,
                    "iptm": metrics.get("best_iptm", 0.0),
                    "ipsae": metrics.get("best_ipsae", 0.0),
                    "plddt": metrics.get("best_plddt", 0.0),
                    "iplddt": metrics.get("best_iplddt", 0.0),
                    "alanine_count": alanine_count,
                    "alanine_pct": round(alanine_pct, 2),
                    "target_seqs": a.protein_seqs or "",
                    "contact_residues": a.contact_residues or "",
                    "msa_mode": a.msa_mode,
                })

        if best_rows:
            best_df = pd.DataFrame(best_rows)
            best_df.to_csv(os.path.join(best_dir, "best_designs.csv"), index=False)
            print(f"\n✅ Saved {len(best_rows)} best designs to {best_dir}/")

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