"""
Local validation backends for Protein Hunter (Boltz edition).

Currently supports:
  - Protenix (open-source AF3 reproduction)
    - run_protenix_validation_local: Subprocess-based (each call loads model)
    - run_protenix_validation_persistent: Persistent runner (~6x faster after first call)

OpenFold3 is intentionally excluded from the local pipeline for now.
"""

from boltz_ph.validation.protenix_local import (
    run_protenix_validation_local,
    run_protenix_validation_persistent,
    shutdown_persistent_runner,
)
from boltz_ph.validation.protenix_runner import PersistentProtenixRunner

__all__ = [
    "run_protenix_validation_local",
    "run_protenix_validation_persistent",
    "shutdown_persistent_runner",
    "PersistentProtenixRunner",
]

