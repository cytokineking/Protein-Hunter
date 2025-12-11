"""
Scoring package.

This package provides interface scoring functions using either PyRosetta
(proprietary) or open-source alternatives (OpenMM, FreeSASA, sc-rs).

Modules:
    opensource: Open-source scoring with OpenMM, FreeSASA, and sc-rs
    pyrosetta: PyRosetta-based scoring (requires license)
"""

# Re-export from opensource module
from modal_boltz_ph.scoring.opensource import (
    OPENSOURCE_SCORING_GPU_FUNCTIONS,
    DEFAULT_OPENSOURCE_GPU,
    configure_verbose as configure_opensource_verbose,
)

# Re-export from pyrosetta module
from modal_boltz_ph.scoring.pyrosetta import (
    run_pyrosetta_single,
    configure_verbose as configure_pyrosetta_verbose,
)

__all__ = [
    # Open-source scoring
    "OPENSOURCE_SCORING_GPU_FUNCTIONS",
    "DEFAULT_OPENSOURCE_GPU",
    "configure_opensource_verbose",
    # PyRosetta scoring
    "run_pyrosetta_single",
    "configure_pyrosetta_verbose",
]
