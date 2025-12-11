"""
Shared utilities for downloading model weights.

This module provides common functionality for downloading large model weight
files with progress indication, used by both Protenix and OpenFold3.
"""

import urllib.request
from pathlib import Path
from typing import Optional


def download_with_progress(
    url: str,
    dest_path: Path,
    description: Optional[str] = None,
    print_every: int = 100,
) -> None:
    """
    Download a file with progress indicator.
    
    Shows download progress as: "Progress: X.X% (Y.Y/Z.Z MB)"
    
    Args:
        url: URL to download from
        dest_path: Destination file path
        description: Optional description to print before downloading
        print_every: Print progress every N blocks (default 100)
    
    Raises:
        urllib.error.URLError: If download fails
    """
    if description:
        print(description)
    
    def progress_callback(block_num: int, block_size: int, total_size: int) -> None:
        downloaded = block_num * block_size
        if total_size > 0:
            percent = min(100, downloaded * 100 / total_size)
            mb_downloaded = downloaded / (1024 * 1024)
            mb_total = total_size / (1024 * 1024)
            if block_num % print_every == 0:
                print(f"\r    Progress: {percent:.1f}% ({mb_downloaded:.1f}/{mb_total:.1f} MB)", end="", flush=True)
    
    urllib.request.urlretrieve(url, str(dest_path), reporthook=progress_callback)
    print()  # Newline after progress


def verify_download(path: Path, min_size_mb: float = 100) -> bool:
    """
    Verify that a downloaded file exists and meets minimum size requirements.
    
    Args:
        path: Path to the downloaded file
        min_size_mb: Minimum expected file size in MB
    
    Returns:
        True if file exists and is large enough, False otherwise
    """
    if not path.exists():
        return False
    size_mb = path.stat().st_size / (1024 * 1024)
    return size_mb >= min_size_mb
