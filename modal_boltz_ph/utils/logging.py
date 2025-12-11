"""
Shared verbose logging utilities.

This module provides a centralized verbose logging mechanism that works
across Modal remote functions. Use configure_verbose() at the start of
your entrypoint to control verbosity globally.
"""

import time


# Module-level verbose flag (simple approach that works across Modal remote functions)
_VERBOSE = False


def configure_verbose(verbose: bool = False) -> None:
    """
    Configure verbose logging globally.
    
    Call this at the start of your entrypoint to control verbosity.
    When verbose=False (default), only warnings/errors are shown.
    When verbose=True, detailed timing and progress messages are shown.
    
    Args:
        verbose: Enable verbose logging
    """
    global _VERBOSE
    _VERBOSE = verbose


def is_verbose() -> bool:
    """Check if verbose mode is enabled."""
    return _VERBOSE


def vprint(msg: str) -> None:
    """
    Verbose print - only shown when verbose mode is enabled.
    
    Uses a simple flag-based approach that works across Modal remote functions.
    Messages are prefixed with a timestamp for debugging.
    
    Args:
        msg: Message to print
    """
    if _VERBOSE:
        timestamp = time.strftime("%H:%M:%S")
        print(f"[{timestamp}] {msg}", flush=True)
