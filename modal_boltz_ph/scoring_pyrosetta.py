"""
PyRosetta interface scoring and analysis.

This module provides comprehensive interface analysis using PyRosetta,
including FastRelax, InterfaceAnalyzerMover, BUNS calculation, and
various quality metrics for binder evaluation.
"""

import json
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional

from modal_boltz_ph.app import app
from modal_boltz_ph.images import pyrosetta_image


@app.function(image=pyrosetta_image, cpu=4, timeout=1800, max_containers=20)
def run_pyrosetta_single(
    design_id: str,
    af3_structure: str,
    af3_iptm: float,
    af3_plddt: float,
    binder_chain: str = "A",
    target_chain: str = "B",
    apo_structure: Optional[str] = None,
    af3_confidence_json: Optional[str] = None,
    target_type: str = "protein",
) -> Dict[str, Any]:
    """
    Run full PyRosetta analysis on a SINGLE design.
    
    This function performs comprehensive interface analysis matching
    the local pipeline, including:
    - CIF→PDB conversion using BioPython
    - Multi-chain collapse for >2 chain complexes
    - FastRelax with proper settings
    - InterfaceAnalyzerMover for detailed interface metrics
    - hotspot_residues for interface_nres calculation
    - BUNS calculation with DAlphaBall
    - APO-HOLO RMSD calculation (if apo_structure provided)
    - i_pae calculation (if af3_confidence_json provided)
    - Radius of gyration calculation
    - Returns relaxed PDB content
    
    Args:
        design_id: Unique identifier for this design
        af3_structure: CIF text of AF3 holo structure
        af3_iptm: ipTM score from AF3
        af3_plddt: pLDDT score from AF3
        binder_chain: Chain ID of the binder (default "A")
        target_chain: Chain ID of the target (default "B")
        apo_structure: Optional CIF text of AF3 apo (binder-only) structure
        af3_confidence_json: Optional JSON text of AF3 confidence data (for i_pae)
        target_type: Type of target - "protein", "peptide", "small_molecule", or "nucleic"
                     Affects filtering thresholds
    
    Returns:
        Dict with acceptance status, interface scores, and relaxed_pdb content
    """
    import numpy as np
    
    # Add utils to path for imports
    sys.path.insert(0, "/root/protein_hunter")
    
    result = {
        "design_id": design_id,
        "af3_iptm": float(af3_iptm),
        "af3_plddt": float(af3_plddt),
        "accepted": False,
        "rejection_reason": None,
        "relaxed_pdb": None,
        # Interface metrics
        "binder_score": 0.0,
        "total_score": 0.0,
        "interface_sc": 0.0,
        "interface_packstat": 0.0,
        "interface_dG": 0.0,
        "interface_dSASA": 0.0,
        "interface_dG_SASA_ratio": 0.0,
        "interface_nres": 0,
        "interface_interface_hbonds": 0,
        "interface_delta_unsat_hbonds": 0,
        "interface_hydrophobicity": 0.0,
        "surface_hydrophobicity": 0.0,
        # Additional metrics
        "binder_sasa": 0.0,
        "interface_fraction": 0.0,
        "interface_hbond_percentage": 0.0,
        "interface_delta_unsat_hbonds_percentage": 0.0,
        # Secondary quality metrics
        "apo_holo_rmsd": None,
        "i_pae": None,
        "rg": None,
    }
    
    if not af3_structure:
        result["rejection_reason"] = "No AF3 structure"
        return result
    
    work_dir = Path(tempfile.mkdtemp())
    
    # ========================================
    # HELPER: Multi-chain collapse
    # ========================================
    def collapse_multiple_chains(pdb_in: str, pdb_out: str, binder_chain: str = "A", collapse_target: str = "B"):
        """Collapse all non-binder chains into a single target chain."""
        with open(pdb_in, "r") as f:
            lines = f.readlines()

        atom_indices = []
        chain_list = []
        for i, line in enumerate(lines):
            if line.startswith(("ATOM  ", "HETATM")):
                atom_indices.append(i)
                chain_list.append(line[21])

        all_chains = sorted(set(chain_list))
        collapse_chains = [c for c in all_chains if c != binder_chain]

        # Detect transitions
        transitions = []
        for (idx1, c1), (idx2, c2) in zip(
                zip(atom_indices, chain_list),
                zip(atom_indices[1:], chain_list[1:])):
            if c1 != c2:
                transitions.append((idx1, c1, c2))

        ter_after = set()
        seen_binder_to_collapsed = False
        for idx, c1, c2 in transitions:
            if c1 == binder_chain and c2 in collapse_chains:
                if not seen_binder_to_collapsed:
                    ter_after.add(idx)
                    seen_binder_to_collapsed = True

        # Find last collapsed atom
        last_collapsed_idx = None
        for i, line in enumerate(lines):
            if line.startswith(("ATOM  ", "HETATM")) and line[21] in collapse_chains:
                last_collapsed_idx = i
        if last_collapsed_idx is not None:
            ter_after.add(last_collapsed_idx)

        # Write output without existing TERs
        temp_out = []
        for i, line in enumerate(lines):
            if line.startswith(("ATOM  ", "HETATM")):
                temp_out.append(line)
                if i in ter_after:
                    temp_out.append("TER\n")
            elif not line.startswith("TER"):
                temp_out.append(line)

        # Collapse chain IDs
        final_out = []
        for line in temp_out:
            if line.startswith(("ATOM  ", "HETATM")) and line[21] in collapse_chains:
                line = line[:21] + collapse_target + line[22:]
            final_out.append(line)

        with open(pdb_out, "w") as f:
            f.writelines(final_out)
    
    try:
        # ========================================
        # CIF to PDB CONVERSION
        # ========================================
        from Bio.PDB import MMCIFParser, PDBIO, PDBParser, Selection
        
        # Write CIF structure to temp file
        cif_file = work_dir / f"{design_id}.cif"
        cif_file.write_text(af3_structure)
        
        # Convert CIF to PDB using BioPython
        pdb_file = work_dir / f"{design_id}.pdb"
        try:
            parser = MMCIFParser(QUIET=True)
            structure = parser.get_structure(design_id, str(cif_file))
            
            # Rename long chain IDs to single letters if needed
            next_chain_idx = 0
            def int_to_chain(i):
                if i < 26:
                    return chr(ord("A") + i)
                elif i < 52:
                    return chr(ord("a") + i - 26)
                else:
                    return chr(ord("0") + i - 52)
            
            chainmap = {}
            for chain in structure.get_chains():
                if len(chain.id) != 1:
                    while True:
                        c = int_to_chain(next_chain_idx)
                        if c not in chainmap:
                            chainmap[c] = chain.id
                            chain.id = c
                            break
                        next_chain_idx += 1
            
            # Truncate long residue names
            for model in structure:
                for chain in model:
                    for residue in chain:
                        if len(residue.resname) > 3:
                            residue.resname = residue.resname[:3]
            
            io = PDBIO()
            io.set_structure(structure)
            io.save(str(pdb_file))
            print(f"  Converted CIF to PDB for {design_id}")
        except Exception as e:
            result["rejection_reason"] = f"CIF to PDB conversion failed: {e}"
            return result
        
        # ========================================
        # MULTI-CHAIN COLLAPSE
        # ========================================
        pdb_parser = PDBParser(QUIET=True)
        temp_structure = pdb_parser.get_structure("check", str(pdb_file))
        total_chains = [chain.id for model in temp_structure for chain in model]
        
        if len(total_chains) > 2:
            collapsed_pdb = work_dir / f"{design_id}_collapsed.pdb"
            collapse_multiple_chains(str(pdb_file), str(collapsed_pdb), binder_chain, "B")
            pdb_file = collapsed_pdb
            print(f"  Collapsed {len(total_chains)} chains to 2 for interface analysis")
        
        # ========================================
        # PYROSETTA INITIALIZATION
        # ========================================
        import pyrosetta as pr
        from pyrosetta.rosetta.core.kinematics import MoveMap
        from pyrosetta.rosetta.protocols.relax import FastRelax
        from pyrosetta.rosetta.protocols.analysis import InterfaceAnalyzerMover
        from pyrosetta.rosetta.core.select.residue_selector import ChainSelector
        from pyrosetta.rosetta.protocols.simple_moves import AlignChainMover
        from pyrosetta.rosetta.protocols.rosetta_scripts import XmlObjects
        
        # DAlphaBall path for BUNS calculation
        dalphaball_path = "/root/protein_hunter/utils/DAlphaBall.gcc"
        pr.init(f"-ignore_unrecognized_res -ignore_zero_occupancy -mute all -holes:dalphaball {dalphaball_path} -corrections::beta_nov16 true -relax:default_repeats 1")
        
        # Load pose from PDB
        pose = pr.pose_from_file(str(pdb_file))
        start_pose = pose.clone()
        
        # Get score function
        sfxn = pr.get_fa_scorefxn()
        
        # ========================================
        # FAST RELAX
        # ========================================
        mmf = MoveMap()
        mmf.set_chi(True)
        mmf.set_bb(True)
        mmf.set_jump(False)
        
        fastrelax = FastRelax()
        fastrelax.set_scorefxn(sfxn)
        fastrelax.set_movemap(mmf)
        fastrelax.max_iter(200)
        fastrelax.min_type("lbfgs_armijo_nonmonotone")
        fastrelax.constrain_relax_to_start_coords(True)
        fastrelax.apply(pose)
        
        # Align relaxed structure to original
        align = AlignChainMover()
        align.source_chain(0)
        align.target_chain(0)
        align.pose(start_pose)
        align.apply(pose)
        
        # Copy B factors from start_pose for visualization
        for resid in range(1, pose.total_residue() + 1):
            if pose.residue(resid).is_protein():
                try:
                    bfactor = start_pose.pdb_info().bfactor(resid, 1)
                    for atom_id in range(1, pose.residue(resid).natoms() + 1):
                        pose.pdb_info().bfactor(resid, atom_id, bfactor)
                except Exception:
                    pass
        
        # Save relaxed PDB
        relaxed_pdb_path = work_dir / f"{design_id}_relaxed.pdb"
        pose.dump_pdb(str(relaxed_pdb_path))
        
        # Clean PDB
        with open(relaxed_pdb_path) as f_in:
            relevant_lines = [
                line for line in f_in
                if line.startswith(("ATOM", "HETATM", "MODEL", "TER", "END"))
            ]
        with open(relaxed_pdb_path, "w") as f_out:
            f_out.writelines(relevant_lines)
        
        result["relaxed_pdb"] = relaxed_pdb_path.read_text()
        result["total_score"] = float(sfxn(pose))
        
        # ========================================
        # HOTSPOT RESIDUES
        # ========================================
        from scipy.spatial import cKDTree
        
        def _hotspot_residues(pdb_path, binder_chain, target_chain, atom_distance_cutoff=4.0):
            """Identify interface residues."""
            aa3to1_map = {
                "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
                "GLU": "E", "GLN": "Q", "GLY": "G", "HIS": "H", "ILE": "I",
                "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
                "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V"
            }
            parser = PDBParser(QUIET=True)
            try:
                structure = parser.get_structure("complex", pdb_path)
            except Exception as e:
                print(f"[ERROR] Could not parse PDB: {e}")
                return {}
            
            model = structure[0]
            if binder_chain not in model:
                print(f"[WARNING] Binder chain '{binder_chain}' not found.")
                return {}
            
            binder_atoms = Selection.unfold_entities(model[binder_chain], "A")
            if len(binder_atoms) == 0:
                return {}
            
            # Target = all non-binder chains
            target_atoms = []
            for chain in model:
                if chain.id != binder_chain:
                    target_atoms.extend(Selection.unfold_entities(chain, "A"))
            
            if len(target_atoms) == 0:
                return {}
            
            # KD-tree for fast contact search
            binder_coords = np.array([a.coord for a in binder_atoms])
            target_coords = np.array([a.coord for a in target_atoms])
            
            binder_tree = cKDTree(binder_coords)
            target_tree = cKDTree(target_coords)
            pairs = binder_tree.query_ball_tree(target_tree, atom_distance_cutoff)
            
            interacting = {}
            for binder_idx, close_list in enumerate(pairs):
                if not close_list:
                    continue
                atom = binder_atoms[binder_idx]
                residue = atom.get_parent()
                resnum = residue.id[1]
                res3 = residue.get_resname().upper()
                aa1 = aa3to1_map.get(res3, "X")
                interacting[resnum] = aa1
            
            return interacting
        
        # Use relaxed PDB for interface analysis
        interface_residues_set = _hotspot_residues(
            str(relaxed_pdb_path), 
            binder_chain=binder_chain, 
            target_chain=target_chain,
            atom_distance_cutoff=4.0
        )
        interface_nres = len(interface_residues_set)
        result["interface_nres"] = int(interface_nres)
        
        # Calculate interface hydrophobicity
        HYDROPHOBIC_AA = set("AILMFVWY")
        hydrophobic_count = sum(1 for aa in interface_residues_set.values() if aa in HYDROPHOBIC_AA)
        result["interface_hydrophobicity"] = round((hydrophobic_count / interface_nres) * 100, 2) if interface_nres > 0 else 0.0
        
        # ========================================
        # INTERFACE ANALYSIS
        # ========================================
        interface_string = f"{binder_chain}_{target_chain}"
        
        iam = InterfaceAnalyzerMover()
        iam.set_interface(interface_string)
        iam.set_scorefunction(sfxn)
        iam.set_compute_packstat(True)
        iam.set_compute_interface_energy(True)
        iam.set_calc_dSASA(True)
        iam.set_calc_hbond_sasaE(True)
        iam.set_compute_interface_sc(True)
        iam.set_pack_separated(True)
        iam.apply(pose)
        
        # Get interface scores
        interface_data = iam.get_all_data()
        result["interface_sc"] = round(float(interface_data.sc_value), 3)
        result["interface_interface_hbonds"] = int(interface_data.interface_hbonds)
        result["interface_dG"] = round(float(iam.get_interface_dG()), 2)
        result["interface_dSASA"] = round(float(iam.get_interface_delta_sasa()), 2)
        result["interface_packstat"] = round(float(iam.get_interface_packstat()), 3)
        result["interface_dG_SASA_ratio"] = round(float(interface_data.dG_dSASA_ratio) * 100, 2)
        
        # ========================================
        # BUNS (Buried Unsatisfied Hbonds)
        # ========================================
        try:
            buns_filter = XmlObjects.static_get_filter(
                '<BuriedUnsatHbonds report_all_heavy_atom_unsats="true" scorefxn="scorefxn" ignore_surface_res="false" use_ddG_style="true" dalphaball_sasa="1" probe_radius="1.1" burial_cutoff_apo="0.2" confidence="0" />'
            )
            result["interface_delta_unsat_hbonds"] = int(buns_filter.report_sm(pose))
        except Exception as e:
            print(f"  BUNS calculation failed: {e}")
            result["interface_delta_unsat_hbonds"] = 0
        
        # ========================================
        # BINDER ENERGY AND SASA
        # ========================================
        chain_selector = ChainSelector(binder_chain)
        tem = pr.rosetta.core.simple_metrics.metrics.TotalEnergyMetric()
        tem.set_scorefunction(sfxn)
        tem.set_residue_selector(chain_selector)
        result["binder_score"] = round(float(tem.calculate(pose)), 2)
        
        # Binder SASA
        bsasa = pr.rosetta.core.simple_metrics.metrics.SasaMetric()
        bsasa.set_residue_selector(chain_selector)
        binder_sasa = float(bsasa.calculate(pose))
        result["binder_sasa"] = round(binder_sasa, 2)
        
        # Interface fraction
        result["interface_fraction"] = round(
            (result["interface_dSASA"] / binder_sasa) * 100 if binder_sasa > 0 else 0, 2
        )
        
        # Interface hbond percentage
        result["interface_hbond_percentage"] = round(
            (result["interface_interface_hbonds"] / interface_nres) * 100 if interface_nres > 0 else 0, 2
        )
        
        # BUNS percentage
        result["interface_delta_unsat_hbonds_percentage"] = round(
            (result["interface_delta_unsat_hbonds"] / interface_nres) * 100 if interface_nres > 0 else 0, 2
        )
        
        # ========================================
        # SURFACE HYDROPHOBICITY
        # ========================================
        binder_pose = None
        try:
            for i in range(1, pose.num_chains() + 1):
                chain_begin = pose.conformation().chain_begin(i)
                chain_id = pose.pdb_info().chain(chain_begin)
                if chain_id == binder_chain:
                    binder_pose = pose.split_by_chain()[i]
                    break
            
            if binder_pose:
                layer_sel = pr.rosetta.core.select.residue_selector.LayerSelector()
                layer_sel.set_layers(pick_core=False, pick_boundary=False, pick_surface=True)
                surface_res = layer_sel.apply(binder_pose)
                
                exp_apol_count = 0
                total_count = 0
                for i in range(1, binder_pose.total_residue() + 1):
                    if surface_res[i]:
                        res = binder_pose.residue(i)
                        if res.is_apolar() or res.name() in ["PHE", "TRP", "TYR"]:
                            exp_apol_count += 1
                        total_count += 1
                
                result["surface_hydrophobicity"] = round(exp_apol_count / total_count, 3) if total_count > 0 else 0.0
        except Exception as e:
            print(f"  Surface hydrophobicity calculation failed: {e}")
            result["surface_hydrophobicity"] = 0.0
        
        # ========================================
        # RADIUS OF GYRATION
        # ========================================
        try:
            if binder_pose:
                ca_coords = []
                for i in range(1, binder_pose.total_residue() + 1):
                    if binder_pose.residue(i).has("CA"):
                        ca_atom = binder_pose.residue(i).xyz("CA")
                        ca_coords.append([ca_atom.x, ca_atom.y, ca_atom.z])
                
                if ca_coords:
                    coords = np.array(ca_coords)
                    centroid = coords.mean(axis=0)
                    rg = np.sqrt(np.mean(np.sum((coords - centroid)**2, axis=1)))
                    result["rg"] = round(float(rg), 2)
        except Exception as e:
            print(f"  Radius of gyration calculation failed: {e}")
            result["rg"] = None
        
        # ========================================
        # i_pae CALCULATION
        # ========================================
        if af3_confidence_json:
            try:
                confidence = json.loads(af3_confidence_json)
                pae_matrix = np.array(confidence.get('pae', []))
                
                if len(pae_matrix) > 0:
                    # Get binder length
                    binder_len = 0
                    for resid in range(1, pose.total_residue() + 1):
                        if pose.pdb_info().chain(resid) == binder_chain:
                            binder_len += 1
                    
                    if binder_len > 0 and pae_matrix.shape[0] > binder_len:
                        # i_pae = mean of off-diagonal blocks (binder ↔ target)
                        interface_pae1 = np.mean(pae_matrix[:binder_len, binder_len:])
                        interface_pae2 = np.mean(pae_matrix[binder_len:, :binder_len])
                        result["i_pae"] = round((interface_pae1 + interface_pae2) / 2, 2)
            except Exception as e:
                print(f"  i_pae calculation failed: {e}")
                result["i_pae"] = None
        
        # ========================================
        # APO-HOLO RMSD
        # ========================================
        if apo_structure:
            try:
                from scipy.spatial.transform import Rotation
                
                # Write and convert APO structure
                apo_cif_file = work_dir / f"{design_id}_apo.cif"
                apo_cif_file.write_text(apo_structure)
                
                apo_pdb_file = work_dir / f"{design_id}_apo.pdb"
                apo_parser = MMCIFParser(QUIET=True)
                apo_struct = apo_parser.get_structure(f"{design_id}_apo", str(apo_cif_file))
                
                # Rename chains if needed
                for chain in apo_struct.get_chains():
                    if len(chain.id) != 1:
                        chain.id = "A"  # APO is always single chain
                
                apo_io = PDBIO()
                apo_io.set_structure(apo_struct)
                apo_io.save(str(apo_pdb_file))
                
                # Get CA coordinates from holo (binder chain) and apo
                def get_ca_coords_from_pdb(pdb_path, chain_id):
                    pdb_parser = PDBParser(QUIET=True)
                    struct = pdb_parser.get_structure("s", str(pdb_path))
                    coords = []
                    for model in struct:
                        for chain in model:
                            if chain.id == chain_id:
                                for residue in chain:
                                    if 'CA' in residue:
                                        coords.append(residue['CA'].coord)
                    return np.array(coords)
                
                def np_rmsd(xyz1, xyz2):
                    """Kabsch-aligned RMSD."""
                    if len(xyz1) != len(xyz2) or len(xyz1) == 0:
                        return None
                    centroid1 = xyz1.mean(axis=0)
                    centroid2 = xyz2.mean(axis=0)
                    xyz1_centered = xyz1 - centroid1
                    xyz2_centered = xyz2 - centroid2
                    rotation, _ = Rotation.align_vectors(xyz1_centered, xyz2_centered)
                    xyz2_rotated = rotation.apply(xyz2_centered)
                    return float(np.sqrt(np.mean(np.sum((xyz1_centered - xyz2_rotated)**2, axis=1))))
                
                holo_coords = get_ca_coords_from_pdb(relaxed_pdb_path, binder_chain)
                apo_coords = get_ca_coords_from_pdb(apo_pdb_file, "A")  # APO is always chain A
                
                if len(holo_coords) == len(apo_coords) and len(holo_coords) > 0:
                    result["apo_holo_rmsd"] = round(np_rmsd(holo_coords, apo_coords), 2)
                else:
                    print(f"  Warning: Chain length mismatch for RMSD: holo={len(holo_coords)}, apo={len(apo_coords)}")
                    result["apo_holo_rmsd"] = None
            except Exception as e:
                print(f"  APO-HOLO RMSD calculation failed: {e}")
                result["apo_holo_rmsd"] = None
        
        # ========================================
        # ACCEPTANCE CRITERIA
        # ========================================
        rejection_reasons = []
        
        # Target-specific thresholds
        if target_type == "peptide":
            nres_threshold = 4
            buns_threshold = 2
        else:  # protein, small_molecule, nucleic
            nres_threshold = 7
            buns_threshold = 4
        
        # Primary interface quality filters
        if af3_iptm < 0.7:
            rejection_reasons.append(f"Low AF3 ipTM: {af3_iptm:.3f}")
        
        if af3_plddt < 80:
            rejection_reasons.append(f"Low AF3 pLDDT: {af3_plddt:.1f}")
        
        if result["binder_score"] >= 0:
            rejection_reasons.append(f"binder_score >= 0: {result['binder_score']}")
        
        if result["surface_hydrophobicity"] >= 0.35:
            rejection_reasons.append(f"surface_hydrophobicity >= 0.35: {result['surface_hydrophobicity']}")
        
        if result["interface_sc"] <= 0.55:
            rejection_reasons.append(f"interface_sc <= 0.55: {result['interface_sc']}")
        
        if result["interface_packstat"] <= 0:
            rejection_reasons.append(f"interface_packstat <= 0: {result['interface_packstat']}")
        
        if result["interface_dG"] >= 0:
            rejection_reasons.append(f"interface_dG >= 0: {result['interface_dG']}")
        
        if result["interface_dSASA"] <= 1:
            rejection_reasons.append(f"interface_dSASA <= 1: {result['interface_dSASA']}")
        
        if result["interface_dG_SASA_ratio"] >= 0:
            rejection_reasons.append(f"interface_dG_SASA_ratio >= 0: {result['interface_dG_SASA_ratio']}")
        
        if result["interface_nres"] <= nres_threshold:
            rejection_reasons.append(f"interface_nres <= {nres_threshold}: {result['interface_nres']}")
        
        if result["interface_interface_hbonds"] <= 3:
            rejection_reasons.append(f"interface_interface_hbonds <= 3: {result['interface_interface_hbonds']}")
        
        if result["interface_hbond_percentage"] <= 0:
            rejection_reasons.append(f"interface_hbond_percentage <= 0: {result['interface_hbond_percentage']}")
        
        if result["interface_delta_unsat_hbonds"] >= buns_threshold:
            rejection_reasons.append(f"interface_delta_unsat_hbonds >= {buns_threshold}: {result['interface_delta_unsat_hbonds']}")
        
        # Secondary quality filters
        if result.get("i_pae") is not None and result["i_pae"] >= 15:
            rejection_reasons.append(f"i_pae >= 15: {result['i_pae']}")
        
        if result.get("rg") is not None and result["rg"] >= 17:
            rejection_reasons.append(f"rg >= 17: {result['rg']}")
        
        if result.get("apo_holo_rmsd") is not None and result["apo_holo_rmsd"] >= 3.5:
            rejection_reasons.append(f"apo_holo_rmsd >= 3.5: {result['apo_holo_rmsd']}")
        
        if rejection_reasons:
            result["accepted"] = False
            result["rejection_reason"] = "; ".join(rejection_reasons)
        else:
            result["accepted"] = True
            
    except Exception as e:
        import traceback
        result["rejection_reason"] = f"PyRosetta error: {str(e)[:200]}"
        print(f"PyRosetta error for {design_id}: {traceback.format_exc()}")
    
    return result

