"""
Shared helper functions for Modal Protein Hunter.

This module contains utility functions used across the design and validation pipeline.
"""

import random
from typing import Any, Dict, Optional


def sample_seq(length: int, exclude_P: bool = False, frac_X: float = 0.5) -> str:
    """
    Generate a random sequence with specified fraction of X residues.
    
    Args:
        length: Length of the sequence to generate
        exclude_P: If True, exclude proline from the amino acid alphabet
        frac_X: Fraction of residues to be X (unknown/design positions)
    
    Returns:
        Random amino acid sequence string
    """
    aa_list = list("ARNDCEQGHILKMFPSTWYV")
    if exclude_P:
        aa_list.remove("P")
    
    seq = []
    for _ in range(length):
        if random.random() < frac_X:
            seq.append("X")
        else:
            seq.append(random.choice(aa_list))
    return "".join(seq)


def shallow_copy_tensor_dict(d: Dict) -> Dict:
    """
    Create a shallow copy of a dict with tensors.
    
    Recursively handles nested dicts and lists containing tensors.
    
    Args:
        d: Dictionary potentially containing PyTorch tensors
    
    Returns:
        New dictionary with cloned tensors
    """
    import torch
    result = {}
    for k, v in d.items():
        if isinstance(v, torch.Tensor):
            result[k] = v.detach().clone()
        elif isinstance(v, dict):
            result[k] = shallow_copy_tensor_dict(v)
        elif isinstance(v, list):
            result[k] = [x.detach().clone() if isinstance(x, torch.Tensor) else x for x in v]
        else:
            result[k] = v
    return result


def get_cif_alignment_json(
    query_seq: str,
    cif_path: str,
    chain_id: Optional[str] = None
) -> Dict[str, Any]:
    """
    Build AF3 template JSON with sequence alignment indices.
    
    This function is used for template integration in AF3 validation,
    matching the local pipeline's approach.
    
    Args:
        query_seq: The query protein sequence to align against
        cif_path: Path to the template CIF/PDB file
        chain_id: Optional chain ID to extract from template (default: first chain)
    
    Returns:
        Dict with 'mmcif', 'queryIndices', and 'templateIndices' for AF3 JSON
    
    Raises:
        ValueError: If chain not found or alignment fails
    """
    from Bio import pairwise2
    from Bio.Data import IUPACData
    from Bio.PDB import MMCIFParser, PDBParser
    
    # Determine parser based on file extension
    if cif_path.lower().endswith('.cif'):
        parser = MMCIFParser(QUIET=True)
    else:
        parser = PDBParser(QUIET=True)
    
    structure = parser.get_structure("template", cif_path)
    
    # Get the specified chain or first chain
    if chain_id:
        chain = next((ch for ch in structure.get_chains() if ch.id == chain_id), None)
        if chain is None:
            raise ValueError(f"Chain {chain_id} not found in {cif_path}")
    else:
        chain = next(structure.get_chains())
    
    # Extract template sequence using 3-letter to 1-letter mapping
    three_to_one = IUPACData.protein_letters_3to1
    template_seq = "".join(
        three_to_one.get(residue.resname.capitalize(), "X")
        for residue in chain.get_residues() if residue.id[0] == " "
    )
    
    # Perform global sequence alignment
    alignments = pairwise2.align.globalxx(query_seq, template_seq)
    if not alignments:
        raise ValueError(f"Could not align query sequence to template from {cif_path}")
    
    align = alignments[0]
    q_aln, t_aln = align.seqA, align.seqB
    
    # Extract aligned index mappings
    query_indices = []
    template_indices = []
    q_pos = t_pos = 0
    for q_char, t_char in zip(q_aln, t_aln):
        if q_char != "-" and t_char != "-":
            query_indices.append(q_pos)
            template_indices.append(t_pos)
        if q_char != "-":
            q_pos += 1
        if t_char != "-":
            t_pos += 1
    
    # Read mmCIF/PDB text
    with open(cif_path) as f:
        structure_text = f.read()
    
    return {
        "mmcif": structure_text,
        "queryIndices": query_indices,
        "templateIndices": template_indices,
    }

