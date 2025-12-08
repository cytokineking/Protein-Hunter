"""
Central place for global constants, chain mappings, and amino acid conversions.
"""

# Map PDB chain ID (A, B, C...) to Boltz-style internal chain index (0, 1, 2...)
# Max 10 chains supported here.
CHAIN_TO_NUMBER = {
    "A": 0,
    "B": 1,
    "C": 2,
    "D": 3,
    "E": 4,
    "F": 5,
    "G": 6,
    "H": 7,
    "I": 8,
    "J": 9,
}

# Amino acid 3-letter to 1-letter code conversion dict
RESTYPE_3TO1 = {
    "ALA": "A",
    "CYS": "C",
    "ASP": "D",
    "GLU": "E",
    "PHE": "F",
    "GLY": "G",
    "HIS": "H",
    "ILE": "I",
    "LYS": "K",
    "LEU": "L",
    "MET": "M",
    "ASN": "N",
    "PRO": "P",
    "GLN": "Q",
    "ARG": "R",
    "SER": "S",
    "THR": "T",
    "VAL": "V",
    "TRP": "W",
    "TYR": "Y",
    "MSE": "M",  # Selenomethionine maps to Methionine
}

# RNA sequence type constant
RNA_CHAIN_POLY_TYPE = "polyribonucleotide"

# Cutoff for short RNA sequences in Nhmmer
SHORT_SEQUENCE_CUTOFF = 50

# Hydrophobic amino acids set for scoring
HYDROPHOBIC_AA = set("ACFILMPVWY")

# Unified column schema for best_designs, accepted_stats, and rejected_stats CSVs
# Matches Modal pipeline output format for consistency
UNIFIED_DESIGN_COLUMNS = [
    # Identity
    "design_id", "design_num", "cycle",
    # Binder info
    "binder_sequence", "binder_length", "cyclic", "alanine_count", "alanine_pct",
    # Boltz design metrics (prefixed for clarity)
    "boltz_iptm", "boltz_ipsae", "boltz_plddt", "boltz_iplddt",
    # AF3 validation metrics
    "af3_iptm", "af3_ipsae", "af3_ptm", "af3_plddt",
    # PyRosetta interface metrics
    "interface_dG", "interface_sc", "interface_nres", "interface_dSASA",
    "interface_packstat", "interface_hbonds", "interface_delta_unsat_hbonds",
    # Secondary quality metrics
    "apo_holo_rmsd", "i_pae", "rg",
    # Acceptance status
    "accepted", "rejection_reason",
]