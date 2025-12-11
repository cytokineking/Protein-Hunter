"""
Shared utilities package.

This package provides common utilities used across the Modal Protein Hunter
pipeline, including logging, weight downloading, and helper functions.

Modules:
    logging: Verbose logging utilities
    weights: Model weight download helpers
"""

from modal_boltz_ph.utils.logging import (
    configure_verbose,
    is_verbose,
    vprint,
)

from modal_boltz_ph.utils.weights import (
    download_with_progress,
    verify_download,
)

__all__ = [
    # Logging
    "configure_verbose",
    "is_verbose",
    "vprint",
    # Weights
    "download_with_progress",
    "verify_download",
]
