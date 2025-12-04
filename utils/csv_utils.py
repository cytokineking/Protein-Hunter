"""
Cross-platform thread-safe CSV utilities for Protein Hunter.

Provides safe CSV append operations with file locking to handle concurrent
writes from multiple Modal containers or local processes.
"""

import os
import time
from pathlib import Path
from typing import Dict, Any


def append_to_csv_safe(csv_path: Path, row: Dict[str, Any], timeout: float = 30.0) -> None:
    """
    Cross-platform thread-safe CSV append with file locking.

    Uses a lock file pattern that works on Unix, macOS, and Windows.

    Args:
        csv_path: Path to the CSV file
        row: Dictionary of column -> value for the new row
        timeout: Maximum seconds to wait for lock (default 30s)

    Raises:
        RuntimeError: If lock cannot be acquired within timeout
    """
    import pandas as pd

    csv_path = Path(csv_path)
    lock_path = csv_path.with_suffix('.csv.lock')
    df_row = pd.DataFrame([row])

    start_time = time.time()
    while True:
        try:
            # Try to create lock file exclusively (atomic operation)
            fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            try:
                # We have the lock - safe to write
                needs_header = not csv_path.exists() or csv_path.stat().st_size == 0
                df_row.to_csv(csv_path, mode='a', header=needs_header, index=False)
            finally:
                os.close(fd)
                try:
                    os.unlink(lock_path)  # Remove lock file
                except OSError:
                    pass  # Ignore if already removed
            return  # Success

        except FileExistsError:
            # Lock held by another process
            if time.time() - start_time > timeout:
                raise RuntimeError(
                    f"Could not acquire lock for {csv_path} after {timeout}s. "
                    f"Delete {lock_path} if stale."
                )
            time.sleep(0.05)  # Wait 50ms and retry


def format_contact_residues(contact_residues: str) -> str:
    """
    Convert contact residues to compact notation.

    Input formats:
        - "1,2,3,4,5" -> "1-5" (if consecutive)
        - "1,2,3,10,11,12" -> "1-3,10-12"
        - "1,3,5,7" -> "1,3,5,7" (non-consecutive)

    Multi-chain format with pipe separator is preserved:
        - "1,2,3|10,11,12" -> "1-3|10-12"

    Args:
        contact_residues: Comma-separated residue numbers, optionally with pipe separators

    Returns:
        Compact notation string
    """
    if not contact_residues or not contact_residues.strip():
        return ""

    def compress_chain(residue_str: str) -> str:
        """Compress a single chain's residue list."""
        if not residue_str.strip():
            return ""

        try:
            residues = sorted([int(r.strip()) for r in residue_str.split(",") if r.strip()])
        except ValueError:
            return residue_str  # Return original if can't parse

        if not residues:
            return ""

        # Build ranges
        ranges = []
        start = residues[0]
        end = residues[0]

        for r in residues[1:]:
            if r == end + 1:
                end = r
            else:
                if start == end:
                    ranges.append(str(start))
                else:
                    ranges.append(f"{start}-{end}")
                start = end = r

        # Add final range
        if start == end:
            ranges.append(str(start))
        else:
            ranges.append(f"{start}-{end}")

        return ",".join(ranges)

    # Handle multi-chain format
    chains = contact_residues.split("|")
    compressed = [compress_chain(chain) for chain in chains]
    return "|".join(compressed)
