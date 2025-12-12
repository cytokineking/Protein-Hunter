"""
Local validation backends for Protein Hunter (Boltz edition).

Currently supports:
  - Protenix (open-source AF3 reproduction)

OpenFold3 is intentionally excluded from the local pipeline for now.
"""

from boltz_ph.validation.protenix_local import run_protenix_validation_local

__all__ = [
    "run_protenix_validation_local",
]

