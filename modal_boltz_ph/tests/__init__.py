"""
Test harnesses for Modal Protein Hunter.

This package contains test scripts for validating infrastructure and running
isolated tests of specific pipeline components without the full design loop.

Test scripts:
    test_infra.py: GPU and image configuration tests
    test_protenix.py: Standalone Protenix validation testing
    test_openfold3.py: Standalone OpenFold3 validation testing
    test_scoring.py: Scoring method comparison tests

Usage:
    # Test GPU and AF3 image
    modal run modal_boltz_ph/tests/test_infra.py::test_connection
    modal run modal_boltz_ph/tests/test_infra.py::test_af3

    # Test Protenix validation on existing designs
    modal run modal_boltz_ph/tests/test_protenix.py::test_single --pdb-path ./design.pdb

    # Test scoring methods
    modal run modal_boltz_ph/tests/test_scoring.py::test_scoring --input-dir ./refolded
"""

# Re-export test functions for backwards compatibility
from modal_boltz_ph.tests.test_infra import test_af3_image, _test_gpu

__all__ = ["test_af3_image", "_test_gpu"]
