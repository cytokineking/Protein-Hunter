"""
Local scoring backends for Protein Hunter (Boltz edition).

Supports:
  - Open-source scoring (OpenMM + FreeSASA + sc-rs)
  - PyRosetta scoring remains in utils.pyrosetta_utils (legacy)
"""

from boltz_ph.scoring.opensource_local import run_opensource_scoring_local

__all__ = [
    "run_opensource_scoring_local",
]

