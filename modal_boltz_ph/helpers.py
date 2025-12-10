"""
Shared helper functions for Modal Protein Hunter.

This module contains utility functions used across the design and validation pipeline.
"""

import random
import sys
from typing import Any, Dict, List, Optional, Set, Tuple


# =============================================================================
# COLORS AND TERMINAL FORMATTING
# =============================================================================

class Colors:
    """ANSI escape codes with TTY detection fallback."""
    _enabled = hasattr(sys.stdout, 'isatty') and sys.stdout.isatty()
    
    RED = '\033[91m' if _enabled else ''
    GREEN = '\033[92m' if _enabled else ''
    YELLOW = '\033[93m' if _enabled else ''
    BLUE = '\033[94m' if _enabled else ''
    MAGENTA = '\033[95m' if _enabled else ''
    CYAN = '\033[96m' if _enabled else ''
    BOLD = '\033[1m' if _enabled else ''
    RESET = '\033[0m' if _enabled else ''
    
    @classmethod
    def disable(cls):
        """Disable all color codes."""
        cls.RED = cls.GREEN = cls.YELLOW = cls.BLUE = ''
        cls.MAGENTA = cls.CYAN = cls.BOLD = cls.RESET = ''
    
    @classmethod
    def enable(cls):
        """Enable color codes (if TTY supports it)."""
        if hasattr(sys.stdout, 'isatty') and sys.stdout.isatty():
            cls.RED = '\033[91m'
            cls.GREEN = '\033[92m'
            cls.YELLOW = '\033[93m'
            cls.BLUE = '\033[94m'
            cls.MAGENTA = '\033[95m'
            cls.CYAN = '\033[96m'
            cls.BOLD = '\033[1m'
            cls.RESET = '\033[0m'


# =============================================================================
# THREE-LETTER TO ONE-LETTER AMINO ACID MAPPING
# =============================================================================

THREE_TO_ONE = {
    'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C',
    'GLN': 'Q', 'GLU': 'E', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
    'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F', 'PRO': 'P',
    'SER': 'S', 'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V',
    # Common modified residues
    'MSE': 'M',  # Selenomethionine
    'CSE': 'C',  # Selenocysteine
    'SEC': 'U',  # Selenocysteine (standard)
    'PYL': 'O',  # Pyrrolysine
}


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
    
    AF3 requires mmCIF format, so PDB files are converted automatically.
    The mmCIF is also filtered to only include the relevant chain.
    
    Args:
        query_seq: The query protein sequence to align against
        cif_path: Path to the template CIF/PDB file
        chain_id: Optional chain ID to extract from template (default: first chain)
    
    Returns:
        Dict with 'mmcif', 'queryIndices', and 'templateIndices' for AF3 JSON
    
    Raises:
        ValueError: If chain not found or alignment fails
    """
    import io
    from Bio.Align import PairwiseAligner
    from Bio.Data import IUPACData
    from Bio.PDB import MMCIFParser, PDBParser, MMCIFIO, Select
    
    # Determine parser based on file extension
    is_cif = cif_path.lower().endswith('.cif')
    if is_cif:
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
    
    selected_chain_id = chain.id
    
    # Extract template sequence using 3-letter to 1-letter mapping
    three_to_one = IUPACData.protein_letters_3to1
    template_seq = "".join(
        three_to_one.get(residue.resname.capitalize(), "X")
        for residue in chain.get_residues() if residue.id[0] == " "
    )
    
    # Perform global sequence alignment using PairwiseAligner (replaces deprecated pairwise2)
    aligner = PairwiseAligner()
    aligner.mode = 'global'
    aligner.match_score = 1
    aligner.mismatch_score = 0
    aligner.open_gap_score = 0
    aligner.extend_gap_score = 0
    alignments = list(aligner.align(query_seq, template_seq))
    if not alignments:
        raise ValueError(f"Could not align query sequence to template from {cif_path}")
    
    alignment = alignments[0]
    
    # Extract index mappings using alignment.indices
    # indices[0] = query indices (-1 for gaps), indices[1] = target indices (-1 for gaps)
    indices = alignment.indices
    query_indices = []
    template_indices = []
    for col in range(indices.shape[1]):
        q_idx = indices[0, col]
        t_idx = indices[1, col]
        if q_idx >= 0 and t_idx >= 0:
            query_indices.append(int(q_idx))
            template_indices.append(int(t_idx))
    
    # Generate mmCIF format (convert PDB to mmCIF if needed)
    # AF3 requires mmCIF format for templates
    if is_cif:
        # For CIF files, read and filter by chain
        with open(cif_path) as f:
            mmcif_text = f.read()
        # Filter to selected chain only
        mmcif_text = _filter_mmcif_by_chain(mmcif_text, selected_chain_id)
    else:
        # For PDB files, convert to mmCIF using Biopython
        # Create a chain selector
        class ChainSelect(Select):
            def accept_chain(self, chain):
                return chain.id == selected_chain_id
        
        mmcif_io = MMCIFIO()
        mmcif_io.set_structure(structure)
        
        # Write to string buffer
        output = io.StringIO()
        mmcif_io.save(output, ChainSelect())
        mmcif_text = output.getvalue()
    
    # AF3 requires templates to have a release date - add if missing
    mmcif_text = _ensure_mmcif_release_date(mmcif_text)
    
    return {
        "mmcif": mmcif_text,
        "queryIndices": query_indices,
        "templateIndices": template_indices,
    }


def _ensure_mmcif_release_date(mmcif_text: str) -> str:
    """
    Ensure mmCIF has a release date required by AF3 templates.
    
    AF3 requires templates to have _pdbx_audit_revision_history.revision_date
    in ISO-8601 format. If missing, adds a placeholder date.
    """
    # Check if release date already exists
    if "_pdbx_audit_revision_history.revision_date" in mmcif_text:
        return mmcif_text
    
    # Add the required audit revision history section after data_ line
    # Use a generic historical date (before any designs would be created)
    audit_block = """
#
loop_
_pdbx_audit_revision_history.ordinal
_pdbx_audit_revision_history.data_content_type
_pdbx_audit_revision_history.major_revision
_pdbx_audit_revision_history.minor_revision
_pdbx_audit_revision_history.revision_date
1 'Structure model' 1 0 2020-01-01
#
"""
    
    # Insert after the data_ line
    lines = mmcif_text.split('\n')
    result_lines = []
    inserted = False
    
    for line in lines:
        result_lines.append(line)
        # Insert after data_ line
        if line.startswith('data_') and not inserted:
            result_lines.append(audit_block)
            inserted = True
    
    # If no data_ line found, prepend with a generic data block
    if not inserted:
        return f"data_template\n{audit_block}\n{mmcif_text}"
    
    return '\n'.join(result_lines)


def _filter_mmcif_by_chain(mmcif_text: str, chain_id: str) -> str:
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


# =============================================================================
# TEMPLATE ANALYSIS AND HOTSPOT VALIDATION
# =============================================================================

def analyze_template_structure(
    template_bytes: bytes,
    chain_ids: List[str],
    filename: str = "template",
) -> Dict[str, Any]:
    """
    Analyze PDB/CIF template for sequence extraction, gap detection,
    and auth→canonical residue mapping.
    
    Args:
        template_bytes: Raw bytes of the PDB/CIF file
        chain_ids: List of chain IDs to extract (e.g., ['A', 'B'])
        filename: Original filename (used to detect format)
    
    Returns:
        Dictionary with:
            - success: bool
            - error: Optional error message
            - chains: Dict[chain_id, {
                sequence: str,
                auth_residues: List[int],  # Auth residue numbers in order
                auth_to_canonical: Dict[int, int],  # auth_resnum → canonical (1-indexed)
                canonical_to_auth: Dict[int, int],  # canonical (1-indexed) → auth_resnum
                auth_range: Tuple[int, int],  # (min, max) auth residue numbers
                has_gaps: bool,
                gaps: List[Dict],  # [{start, end, count}]
                first_auth_residue: int,
              }]
    """
    import io
    from Bio.PDB import MMCIFParser, PDBParser
    
    result = {
        "success": False,
        "error": None,
        "chains": {},
    }
    
    # Detect format and parse
    is_cif = filename.lower().endswith('.cif') or filename.lower().endswith('.mmcif')
    
    try:
        if is_cif:
            parser = MMCIFParser(QUIET=True)
        else:
            parser = PDBParser(QUIET=True)
        
        # Parse from bytes
        structure = parser.get_structure("template", io.StringIO(template_bytes.decode('utf-8')))
    except Exception as e:
        result["error"] = f"Failed to parse template: {e}"
        return result
    
    # Process each requested chain
    for chain_id in chain_ids:
        chain = None
        for model in structure:
            for ch in model:
                if ch.id == chain_id:
                    chain = ch
                    break
            if chain:
                break
        
        if chain is None:
            result["error"] = f"Chain '{chain_id}' not found in template"
            return result
        
        # Extract residues (only standard amino acids)
        residues = []
        for residue in chain.get_residues():
            het_flag = residue.id[0]
            if het_flag != " ":
                continue  # Skip non-standard residues (HETATM)
            
            resname = residue.resname.upper()
            resnum = residue.id[1]
            
            # Convert 3-letter to 1-letter
            one_letter = THREE_TO_ONE.get(resname, 'X')
            residues.append((resnum, one_letter))
        
        if not residues:
            result["error"] = f"No standard amino acids found in chain '{chain_id}'"
            return result
        
        # Sort by auth residue number
        residues.sort(key=lambda x: x[0])
        
        # Extract sequence and build mappings
        auth_residues = [r[0] for r in residues]
        sequence = "".join(r[1] for r in residues)
        
        # Build auth↔canonical mappings (canonical is 1-indexed)
        auth_to_canonical = {auth: i + 1 for i, auth in enumerate(auth_residues)}
        canonical_to_auth = {i + 1: auth for i, auth in enumerate(auth_residues)}
        
        # Detect gaps (non-consecutive auth residue numbers)
        gaps = []
        for i in range(1, len(auth_residues)):
            prev_auth = auth_residues[i - 1]
            curr_auth = auth_residues[i]
            if curr_auth != prev_auth + 1:
                # Gap detected
                gap_start = prev_auth + 1
                gap_end = curr_auth - 1
                gap_count = gap_end - gap_start + 1
                gaps.append({
                    "start": gap_start,
                    "end": gap_end,
                    "count": gap_count,
                })
        
        has_gaps = len(gaps) > 0
        auth_range = (min(auth_residues), max(auth_residues)) if auth_residues else (0, 0)
        
        result["chains"][chain_id] = {
            "sequence": sequence,
            "auth_residues": auth_residues,
            "auth_to_canonical": auth_to_canonical,
            "canonical_to_auth": canonical_to_auth,
            "auth_range": auth_range,
            "has_gaps": has_gaps,
            "gaps": gaps,
            "first_auth_residue": auth_residues[0] if auth_residues else 0,
        }
    
    result["success"] = True
    return result


def validate_hotspots(
    hotspot_positions: List[int],
    sequence_length: int,
    chain_id: str,
    use_auth_numbering: bool = False,
    auth_residue_set: Optional[Set[int]] = None,
    auth_range: Optional[Tuple[int, int]] = None,
) -> Tuple[bool, Optional[str]]:
    """
    Validate all hotspot positions exist in target chain.
    
    Args:
        hotspot_positions: List of residue positions to validate
        sequence_length: Length of the target sequence
        chain_id: Chain ID (for error messages)
        use_auth_numbering: If True, validate against auth residue numbers
        auth_residue_set: Set of valid auth residue numbers (required if use_auth_numbering)
        auth_range: (min, max) auth residue range (for error messages)
    
    Returns:
        Tuple of (is_valid, error_message)
    """
    if not hotspot_positions:
        return True, None
    
    invalid = []
    
    if use_auth_numbering:
        if auth_residue_set is None:
            return False, "auth_residue_set required when use_auth_numbering=True"
        
        for pos in hotspot_positions:
            if pos not in auth_residue_set:
                invalid.append(pos)
        
        if invalid:
            range_str = f"{auth_range[0]}-{auth_range[1]}" if auth_range else "unknown"
            error_lines = [
                "═" * 65,
                "❌ ERROR: Hotspot residue not found in chain",
                "═" * 65,
                "",
                f"Chain {chain_id}:",
                f"  Auth residue range: {range_str}",
                "",
                "Invalid hotspot(s):",
            ]
            for pos in invalid:
                error_lines.append(f"  • Auth residue {pos} not found (valid range: {range_str})")
            error_lines.extend([
                "",
                "Hint: Check that your hotspot residue numbers match the",
                "      numbering in your PDB/CIF file.",
                "═" * 65,
            ])
            return False, "\n".join(error_lines)
    else:
        # Canonical numbering (1-indexed)
        for pos in hotspot_positions:
            if pos < 1 or pos > sequence_length:
                invalid.append(pos)
        
        if invalid:
            error_lines = [
                "═" * 65,
                "❌ ERROR: Hotspot position out of range",
                "═" * 65,
                "",
                f"Chain {chain_id}:",
                f"  Sequence length: {sequence_length} residues",
                f"  Valid position range: 1-{sequence_length}",
                "",
                "Invalid hotspot(s):",
            ]
            for pos in invalid:
                error_lines.append(f"  • Position {pos} out of range (max: {sequence_length})")
            error_lines.extend([
                "",
                "Hint: Canonical positions are 1-indexed.",
                "      If using PDB residue numbers, add --use-auth-numbering.",
                "═" * 65,
            ])
            return False, "\n".join(error_lines)
    
    return True, None


def collapse_to_ranges(positions: List[int]) -> str:
    """
    Collapse consecutive positions into ranges for display.
    
    Example: [1, 2, 3, 4, 5, 72, 99, 100, 101] → "1..5, 72, 99..101"
    
    Args:
        positions: List of integer positions (will be sorted)
    
    Returns:
        Formatted string with collapsed ranges
    """
    if not positions:
        return ""
    
    sorted_pos = sorted(set(positions))
    ranges = []
    start = end = sorted_pos[0]
    
    for pos in sorted_pos[1:]:
        if pos == end + 1:
            end = pos
        else:
            if start == end:
                ranges.append(str(start))
            else:
                ranges.append(f"{start}..{end}")
            start = end = pos
    
    # Add final range
    if start == end:
        ranges.append(str(start))
    else:
        ranges.append(f"{start}..{end}")
    
    return ", ".join(ranges)


def convert_auth_to_canonical(
    auth_positions: List[int],
    auth_to_canonical: Dict[int, int],
) -> List[int]:
    """
    Convert auth residue numbers to canonical (1-indexed) positions.
    
    Args:
        auth_positions: List of auth residue numbers
        auth_to_canonical: Mapping from auth → canonical
    
    Returns:
        List of canonical positions
    """
    return [auth_to_canonical.get(pos, -1) for pos in auth_positions]


def print_target_analysis(
    chains_info: Dict[str, Dict[str, Any]],
    contact_residues_str: str,
    use_auth_numbering: bool,
    template_filename: str = "",
) -> None:
    """
    Print formatted target sequence analysis with hotspot visualization.
    
    Args:
        chains_info: Dictionary from analyze_template_structure()['chains']
        contact_residues_str: Contact residues string (e.g., "69,72|1,2,3")
        use_auth_numbering: Whether hotspots are in auth numbering
        template_filename: Original template filename for display
    """
    c = Colors
    
    # Parse contact residues per chain
    chain_ids = list(chains_info.keys())
    contact_lists = contact_residues_str.split("|") if contact_residues_str else []
    
    print(f"\n{c.BOLD}{'═' * 79}{c.RESET}")
    print(f"{c.BOLD}🎯 TARGET SEQUENCE ANALYSIS{c.RESET}")
    print(f"{c.BOLD}{'═' * 79}{c.RESET}\n")
    
    for i, chain_id in enumerate(chain_ids):
        chain_data = chains_info[chain_id]
        sequence = chain_data["sequence"]
        auth_range = chain_data["auth_range"]
        auth_to_canonical = chain_data["auth_to_canonical"]
        canonical_to_auth = chain_data["canonical_to_auth"]
        
        # Get hotspots for this chain
        hotspots_str = contact_lists[i] if i < len(contact_lists) else ""
        hotspot_positions = [int(x.strip()) for x in hotspots_str.split(",") if x.strip()]
        
        # Convert hotspots to canonical if using auth numbering
        if use_auth_numbering and hotspot_positions:
            canonical_hotspots = convert_auth_to_canonical(hotspot_positions, auth_to_canonical)
            auth_hotspots = hotspot_positions
        else:
            canonical_hotspots = hotspot_positions
            auth_hotspots = [canonical_to_auth.get(pos, pos) for pos in hotspot_positions]
        
        # Chain header
        print(f"{c.BOLD}Chain {chain_id}{c.RESET} - {len(sequence)} residues")
        if template_filename:
            print(f"├─ Source: template ({template_filename}, chain {chain_id})")
        print(f"├─ Auth numbering: {auth_range[0]}-{auth_range[1]}")
        
        if hotspot_positions:
            auth_display = collapse_to_ranges(auth_hotspots)
            canonical_display = collapse_to_ranges(canonical_hotspots)
            if use_auth_numbering:
                print(f"└─ Hotspots: auth [{auth_display}] → canonical [{canonical_display}]")
            else:
                print(f"└─ Hotspots: canonical [{canonical_display}] (auth [{auth_display}])")
        else:
            print(f"└─ Hotspots: none specified")
        
        # Hotspot detail table (if hotspots specified)
        if hotspot_positions:
            print()
            print(f"┌──────────┬──────────┬─────────┬─────────────────────────────┐")
            print(f"│ Canonical│ Auth     │ Residue │ Context                     │")
            print(f"├──────────┼──────────┼─────────┼─────────────────────────────┤")
            
            for canon_pos, auth_pos in zip(canonical_hotspots, auth_hotspots):
                if 1 <= canon_pos <= len(sequence):
                    residue = sequence[canon_pos - 1]
                    # Build context (5 residues on each side)
                    ctx_start = max(0, canon_pos - 6)
                    ctx_end = min(len(sequence), canon_pos + 5)
                    context = sequence[ctx_start:ctx_end]
                    
                    # Highlight the hotspot residue in context
                    local_pos = canon_pos - 1 - ctx_start
                    context_display = (
                        "..." +
                        context[:local_pos] +
                        f"{c.RED}{c.BOLD}{context[local_pos]}{c.RESET}" +
                        context[local_pos + 1:] +
                        "..."
                    )
                    
                    print(f"│ {canon_pos:>8} │ {auth_pos:>8} │    {residue}    │ {context_display:<28}│")
                else:
                    print(f"│ {canon_pos:>8} │ {auth_pos:>8} │    ?    │ (out of range)              │")
            
            print(f"└──────────┴──────────┴─────────┴─────────────────────────────┘")
        
        # Sequence display with hotspots marked
        print()
        print(f"Sequence (hotspots in {c.RED}RED{c.RESET}):")
        
        # Create set of canonical hotspot positions for quick lookup
        hotspot_set = set(canonical_hotspots)
        
        # Print sequence in chunks of 60 with numbering
        chunk_size = 60
        first_auth = chain_data["first_auth_residue"]
        
        for chunk_start in range(0, len(sequence), chunk_size):
            chunk_end = min(chunk_start + chunk_size, len(sequence))
            chunk = sequence[chunk_start:chunk_end]
            
            # Calculate auth numbers for this chunk
            canon_start = chunk_start + 1
            canon_end = chunk_end
            auth_start = canonical_to_auth.get(canon_start, first_auth + chunk_start)
            auth_end = canonical_to_auth.get(canon_end, first_auth + chunk_end - 1)
            
            # Header line with numbering
            print(f"Canon Auth{' ' * (chunk_size - 6)}Canon Auth")
            
            # Build colored sequence
            colored_seq = ""
            marker_line = ""
            for j, aa in enumerate(chunk):
                canon_pos = chunk_start + j + 1
                if canon_pos in hotspot_set:
                    colored_seq += f"{c.RED}{c.BOLD}{aa}{c.RESET}"
                    marker_line += "*"
                else:
                    colored_seq += aa
                    marker_line += " "
            
            # Print sequence line with numbers
            print(f"{canon_start:>5} {auth_start:>4}  {colored_seq}  {canon_end:>5} {auth_end:>4}")
            
            # Print marker line if there are hotspots in this chunk
            # Prefix must match sequence line: 5 (canon) + 1 (space) + 4 (auth) + 2 (spaces) = 12 chars
            if "*" in marker_line:
                print(f"            {marker_line}")
        
        print()
    
    print(f"{c.BOLD}{'═' * 79}{c.RESET}")
    print(f"{c.GREEN}✓ Ready to proceed with design{c.RESET}")
    print(f"{c.BOLD}{'═' * 79}{c.RESET}\n")


def print_gap_error(chain_id: str, chain_data: Dict[str, Any]) -> str:
    """
    Generate error message for gap detection.
    
    Args:
        chain_id: Chain identifier
        chain_data: Chain data from analyze_template_structure()
    
    Returns:
        Formatted error message string
    """
    auth_range = chain_data["auth_range"]
    gaps = chain_data["gaps"]
    
    lines = [
        "═" * 65,
        "❌ ERROR: Structure has gaps - cannot auto-extract sequences",
        "═" * 65,
        "",
        f"Chain {chain_id}:",
        f"  Resolved residues: {len(chain_data['auth_residues'])}",
        f"  Range: {auth_range[0]} - {auth_range[1]}",
        "  Gaps detected:",
    ]
    
    for gap in gaps:
        if gap["start"] == gap["end"]:
            lines.append(f"    • Missing residue {gap['start']}")
        else:
            lines.append(f"    • Missing residues {gap['start']}-{gap['end']} ({gap['count']} residues)")
    
    lines.extend([
        "",
        "⚠️  Please provide --protein-seqs with full sequence.",
        "═" * 65,
    ])
    
    return "\n".join(lines)

