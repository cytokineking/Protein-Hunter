"""
Local Protenix validation utilities.

This is a local-port of `modal_boltz_ph/validation/protenix.py` with Modal-specific
decorators/volumes removed. It runs Protenix via subprocess, manages local
weight caching, and returns results using the unified `af3_*` schema.
"""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
import time
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from utils.convert import download_with_progress


# =============================================================================
# CONSTANTS
# =============================================================================

DEFAULT_PROTENIX_MODEL = "protenix_base_default_v0.5.0"

# Protenix model weight URLs (from Protenix official dependency_url.py)
PROTENIX_MODEL_URLS = {
    "protenix_base_default_v0.5.0": "https://af3-dev.tos-cn-beijing.volces.com/release_model/protenix_base_default_v0.5.0.pt",
    "protenix_base_constraint_v0.5.0": "https://af3-dev.tos-cn-beijing.volces.com/release_model/protenix_base_constraint_v0.5.0.pt",
    "protenix_mini_default_v0.5.0": "https://af3-dev.tos-cn-beijing.volces.com/release_model/protenix_mini_default_v0.5.0.pt",
    "protenix_tiny_default_v0.5.0": "https://af3-dev.tos-cn-beijing.volces.com/release_model/protenix_tiny_default_v0.5.0.pt",
}

# Approximate expected sizes for verification
PROTENIX_MODEL_SIZES = {
    "protenix_base_default_v0.5.0": 1.4 * 1024**3,
    "protenix_mini_default_v0.5.0": 0.5 * 1024**3,
    "protenix_tiny_default_v0.5.0": 0.2 * 1024**3,
}

# Local repo root for Protenix
PROTENIX_REPO_ROOT = Path(__file__).resolve().parents[2] / "Protenix"

# Global verbose toggle set by run_protenix_validation_local
_PROTENIX_VERBOSE = False


# =============================================================================
# SHARED VALIDATION UTILITIES
# =============================================================================

def calculate_ipsae_from_pae(
    pae_matrix: np.ndarray,
    binder_length: int,
    target_length: int,
    pae_cutoff: float = 10.0,
) -> Dict[str, float]:
    """
    Calculate ipSAE from PAE matrix. Ported from Modal shared validation base.
    """
    result = {
        "ipsae": 0.0,
        "ipsae_binder_to_target": 0.0,
        "ipsae_target_to_binder": 0.0,
    }
    if pae_matrix is None or len(pae_matrix) == 0:
        return result

    try:
        total_length = binder_length + target_length
        if pae_matrix.ndim == 1:
            expected_size = total_length * total_length
            if len(pae_matrix) != expected_size:
                return result
            pae_matrix = pae_matrix.reshape(total_length, total_length)

        if pae_matrix.shape != (total_length, total_length):
            return result

        binder_indices = np.arange(binder_length)
        target_indices = np.arange(binder_length, total_length)

        def ptm_func(x: np.ndarray, d0: float) -> np.ndarray:
            return 1.0 / (1.0 + (x / d0) ** 2.0)

        def calc_d0(L: int) -> float:
            L = float(max(L, 27))
            d0 = 1.24 * (L - 15) ** (1.0 / 3.0) - 1.8
            return max(1.0, d0)

        interface_pae = pae_matrix[np.ix_(binder_indices, target_indices)]
        valid_mask = interface_pae < pae_cutoff

        ipsae_byres_binder = []
        for i in range(binder_length):
            valid = valid_mask[i]
            if valid.any():
                n0res = valid.sum()
                d0res = calc_d0(n0res)
                ptm_vals = ptm_func(interface_pae[i][valid], d0res)
                ipsae_byres_binder.append(ptm_vals.mean())
            else:
                ipsae_byres_binder.append(0.0)

        ipsae_byres_binder = np.array(ipsae_byres_binder)
        ipsae_binder_max = float(ipsae_byres_binder.max()) if len(ipsae_byres_binder) > 0 else 0.0

        interface_pae_rev = pae_matrix[np.ix_(target_indices, binder_indices)]
        valid_mask_rev = interface_pae_rev < pae_cutoff

        ipsae_byres_target = []
        for i in range(target_length):
            valid = valid_mask_rev[i]
            if valid.any():
                n0res = valid.sum()
                d0res = calc_d0(n0res)
                ptm_vals = ptm_func(interface_pae_rev[i][valid], d0res)
                ipsae_byres_target.append(ptm_vals.mean())
            else:
                ipsae_byres_target.append(0.0)

        ipsae_byres_target = np.array(ipsae_byres_target)
        ipsae_target_max = float(ipsae_byres_target.max()) if len(ipsae_byres_target) > 0 else 0.0

        ipsae = max(ipsae_binder_max, ipsae_target_max)
        result["ipsae"] = round(ipsae, 4)
        result["ipsae_binder_to_target"] = round(ipsae_binder_max, 4)
        result["ipsae_target_to_binder"] = round(ipsae_target_max, 4)
    except Exception:
        pass

    return result


# =============================================================================
# WEIGHTS MANAGEMENT
# =============================================================================

def _get_weights_dir() -> Path:
    env_dir = os.environ.get("PROTENIX_WEIGHTS_DIR")
    if env_dir:
        return Path(env_dir).expanduser()
    return Path("~/.protein-hunter/protenix_weights").expanduser()


def ensure_protenix_weights(model_name: str = DEFAULT_PROTENIX_MODEL) -> Path:
    """
    Ensure Protenix weights are present locally. Downloads on first use.
    """
    import torch

    weights_dir = _get_weights_dir()
    weights_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = weights_dir / f"{model_name}.pt"
    if checkpoint_path.exists():
        size_bytes = checkpoint_path.stat().st_size
        expected_size = PROTENIX_MODEL_SIZES.get(model_name, 1.0 * 1024**3)
        if size_bytes >= expected_size * 0.9:
            try:
                ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
                if isinstance(ckpt, dict) and "model" in ckpt:
                    del ckpt
                    print(f"  ✓ Protenix weights verified: {checkpoint_path}")
                    return weights_dir
            except Exception:
                pass
        print("  ⚠ Incomplete/corrupt Protenix weights found; re-downloading.")
        try:
            checkpoint_path.unlink()
        except Exception:
            pass

    if model_name not in PROTENIX_MODEL_URLS:
        raise ValueError(f"Unknown Protenix model '{model_name}'. Available: {list(PROTENIX_MODEL_URLS)}")

    url = PROTENIX_MODEL_URLS[model_name]
    print(f"  Downloading Protenix weights ({model_name}) to {checkpoint_path}...")
    ok = download_with_progress(url, str(checkpoint_path))
    if not ok:
        raise RuntimeError(f"Failed to download Protenix weights from {url}")

    # Basic verification
    try:
        _ = torch.load(checkpoint_path, map_location="cpu")
    except Exception as e:
        raise RuntimeError(f"Downloaded weights appear invalid: {e}")

    return weights_dir


# =============================================================================
# MSA HANDLING
# =============================================================================

def convert_colabfold_msa_to_protenix_format(msa_content: str) -> str:
    """Ported from Modal. Adds pseudo taxonomy IDs to ColabFold MSAs."""
    lines = msa_content.strip().split("\n")
    converted_lines: list[str] = []
    pseudo_tax_id = 0

    for line in lines:
        if line.startswith(">"):
            header = line[1:].strip()
            if header.isdigit() or header == "query":
                converted_lines.append(">query")
            elif header.startswith("UniRef100_") and "/" in header:
                first_part = header.split("/")[0]
                if first_part.count("_") >= 2:
                    converted_lines.append(line)
                else:
                    pseudo_tax_id += 1
                    base, range_part = header.rsplit("/", 1)
                    converted_lines.append(f">{base}_{pseudo_tax_id}/{range_part}")
            elif (
                "UniRef" in header
                or "UniProt" in header
                or header.startswith("tr|")
                or header.startswith("sp|")
            ):
                pseudo_tax_id += 1
                parts = header.split()
                identifier = parts[0].replace("|", "_").replace("/", "_")[:30]
                converted_lines.append(f">UniRef100_{identifier}_{pseudo_tax_id}/1-1000")
            else:
                pseudo_tax_id += 1
                clean_id = "".join(c if c.isalnum() else "_" for c in header[:20])
                converted_lines.append(f">UniRef100_{clean_id}_{pseudo_tax_id}/1-1000")
        else:
            converted_lines.append(line)

    return "\n".join(converted_lines)


def ensure_msa_files(msa_dir: Path, target_msas: Dict[str, str]) -> None:
    """
    Persist target MSAs to disk in Protenix expected layout.
    """
    for chain_id, msa_content in target_msas.items():
        chain_msa_dir = msa_dir / f"chain_{chain_id}"
        msa_file = chain_msa_dir / "non_pairing.a3m"
        if msa_file.exists():
            continue
        protenix_msa = convert_colabfold_msa_to_protenix_format(msa_content)
        chain_msa_dir.mkdir(parents=True, exist_ok=True)
        msa_file.write_text(protenix_msa)


# =============================================================================
# PROTENIX INPUT/OUTPUT UTILITIES
# =============================================================================

def _build_protenix_input(
    design_id: str,
    binder_seq: str,
    target_seq: str,
    msa_dir: Optional[Path] = None,
    work_dir: Optional[Path] = None,
) -> Path:
    """
    Build Protenix JSON input file for HOLO predictions.

    Binder is entity 0 (chain A), target chains follow (B/C/...).
    """
    if work_dir is None:
        work_dir = Path(tempfile.mkdtemp())

    sequences = []

    binder_entry: Dict[str, Any] = {
        "proteinChain": {"sequence": binder_seq, "count": 1}
    }

    if msa_dir:
        binder_msa_dir = msa_dir / "chain_A"
        binder_msa_dir.mkdir(parents=True, exist_ok=True)
        binder_msa_file = binder_msa_dir / "non_pairing.a3m"
        if not binder_msa_file.exists():
            binder_msa_file.write_text(f">binder\n{binder_seq}\n")
        binder_entry["proteinChain"]["msa"] = {
            "precomputed_msa_dir": str(binder_msa_dir),
            "pairing_db": "uniref100",
        }

    sequences.append(binder_entry)

    target_chains = target_seq.split(":") if target_seq else []
    for i, seq in enumerate(target_chains):
        chain_id = chr(ord("B") + i)
        target_entry: Dict[str, Any] = {
            "proteinChain": {"sequence": seq, "count": 1}
        }
        if msa_dir:
            chain_msa_dir = msa_dir / f"chain_{chain_id}"
            if chain_msa_dir.exists():
                target_entry["proteinChain"]["msa"] = {
                    "precomputed_msa_dir": str(chain_msa_dir),
                    "pairing_db": "uniref100",
                }
        sequences.append(target_entry)

    protenix_input = [{
        "name": design_id,
        "sequences": sequences,
        "use_msa": True,
    }]

    json_path = work_dir / "query.json"
    json_path.write_text(json.dumps(protenix_input, indent=2))
    return json_path


def _parse_protenix_output(
    output_dir: Path,
    design_id: str,
    seed: int = 101,
    sample: int = 0,
) -> Dict[str, Any]:
    """
    Parse Protenix output for a single prediction name.
    """
    result_dir = output_dir / design_id / f"seed_{seed}" / "predictions"
    if not result_dir.exists():
        result_dir = output_dir / design_id / str(seed)

    conf_file = result_dir / f"{design_id}_summary_confidence_sample_{sample}.json"
    if not conf_file.exists():
        conf_file = result_dir / f"{design_id}_{seed}_summary_confidence_sample_{sample}.json"
    if not conf_file.exists():
        for pattern in [
            f"{design_id}_{seed}_summary_confidence.json",
            f"summary_confidence_sample_{sample}.json",
            "summary_confidence.json",
        ]:
            alt_file = result_dir / pattern
            if alt_file.exists():
                conf_file = alt_file
                break

    cif_file = result_dir / f"{design_id}_{seed}_sample_{sample}.cif"
    if not cif_file.exists():
        cif_file = result_dir / f"{design_id}_seed_{seed}_sample_{sample}.cif"
    if not cif_file.exists():
        for pattern in [
            f"{design_id}_sample_{sample}.cif",
            f"{design_id}.cif",
            "*.cif",
        ]:
            matches = list(result_dir.glob(pattern))
            if matches:
                cif_file = matches[0]
                break

    if not conf_file.exists() or not cif_file.exists():
        raise FileNotFoundError(f"Missing Protenix outputs in {result_dir}")

    confidence = json.loads(conf_file.read_text())
    structure_cif = cif_file.read_text()

    plddt_raw = float(confidence.get("plddt", 0.0))

    # Optional chain-level metrics
    chain_plddt = confidence.get("chain_plddt", {})
    chain_ptm = confidence.get("chain_ptm", {})
    chain_pair_iptm = confidence.get("chain_pair_iptm", {})

    full_data_json = ""
    for pattern in [
        f"full_data_sample_{sample}.json",
        f"{design_id}_full_data_sample_{sample}.json",
        "full_data.json",
    ]:
        full_data_file = result_dir / pattern
        if full_data_file.exists():
            full_data_json = full_data_file.read_text()
            break

    return {
        "iptm": confidence.get("iptm", 0.0),
        "ptm": confidence.get("ptm", 0.0),
        "plddt": plddt_raw,
        "chain_plddt": chain_plddt,
        "chain_ptm": chain_ptm,
        "chain_pair_iptm": chain_pair_iptm,
        "ranking_score": confidence.get("ranking_score", 0.0),
        "has_clash": confidence.get("has_clash", False),
        "structure_cif": structure_cif,
        "confidence_json": conf_file.read_text(),
        "full_data_json": full_data_json,
    }


def _run_protenix_prediction(
    input_json_path: Path,
    output_dir: Path,
    seed: int = 101,
    model_name: str = DEFAULT_PROTENIX_MODEL,
    n_sample: int = 1,
    n_cycle: int = 10,
    n_step: int = 200,
    use_msa: bool = True,
    timeout: int = 1800,
    checkpoint_dir: Optional[str] = None,
) -> subprocess.CompletedProcess:
    """Run Protenix `runner.inference` via subprocess."""
    if checkpoint_dir is None:
        checkpoint_dir = str(_get_weights_dir())

    python_bin = os.environ.get("PROTENIX_PYTHON", sys.executable)

    cmd = [
        python_bin,
        "-m", "runner.inference",
        "--input_json_path", str(input_json_path),
        "--dump_dir", str(output_dir),
        "--seeds", str(seed),
        "--model_name", model_name,
        "--load_checkpoint_dir", checkpoint_dir,
        "--sample_diffusion.N_sample", str(n_sample),
        "--model.N_cycle", str(n_cycle),
        "--sample_diffusion.N_step", str(n_step),
        "--enable_tf32", "true",
        "--enable_efficient_fusion", "true",
        "--enable_diffusion_shared_vars_cache", "true",
        "--need_atom_confidence", "true",
        "--use_msa", str(use_msa).lower(),
    ]

    env = os.environ.copy()
    env["LAYERNORM_TYPE"] = "torch"
    env["PYTHONPATH"] = str(PROTENIX_REPO_ROOT) + os.pathsep + env.get("PYTHONPATH", "")

    return subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=timeout,
        env=env,
        cwd=str(PROTENIX_REPO_ROOT),
    )


def _run_protenix_holo_impl(
    design_id: str,
    binder_seq: str,
    target_seq: str,
    msa_dir: Optional[Path] = None,
    model_name: str = DEFAULT_PROTENIX_MODEL,
) -> Dict[str, Any]:
    work_dir = Path(tempfile.mkdtemp())
    output_dir = work_dir / "output"
    output_dir.mkdir()

    holo_name = f"{design_id}_holo"
    input_json = _build_protenix_input(
        design_id=holo_name,
        binder_seq=binder_seq,
        target_seq=target_seq,
        msa_dir=msa_dir,
        work_dir=work_dir,
    )

    print(f"  [{design_id}] Running Protenix HOLO prediction...")
    t0 = time.time()
    try:
        proc = _run_protenix_prediction(
            input_json_path=input_json,
            output_dir=output_dir,
            model_name=model_name,
            n_sample=1,
            use_msa=bool(msa_dir),
        )
    except subprocess.TimeoutExpired:
        return {"error": "Protenix HOLO timed out after 30 minutes"}
    except Exception as e:
        return {"error": f"Protenix HOLO failed: {e}"}

    elapsed = time.time() - t0
    show_output = proc.returncode != 0 or _PROTENIX_VERBOSE
    if show_output:
        if proc.stdout:
            print(f"  [{design_id}] Protenix stdout (tail):\n{proc.stdout[-1500:]}")
        if proc.stderr:
            print(f"  [{design_id}] Protenix stderr (tail):\n{proc.stderr[-1500:]}")

    if proc.returncode != 0:
        return {"error": f"Protenix HOLO failed (rc={proc.returncode})"}

    try:
        parsed = _parse_protenix_output(output_dir, holo_name)
        parsed["elapsed"] = elapsed
        return parsed
    except Exception as e:
        return {"error": f"Failed to parse Protenix HOLO output: {e}"}


def _run_protenix_apo_impl(
    design_id: str,
    binder_seq: str,
    model_name: str = DEFAULT_PROTENIX_MODEL,
) -> Dict[str, Any]:
    work_dir = Path(tempfile.mkdtemp())
    output_dir = work_dir / "output"
    output_dir.mkdir()

    apo_name = f"{design_id}_apo"
    protenix_input = [{
        "name": apo_name,
        "sequences": [{"proteinChain": {"sequence": binder_seq, "count": 1}}],
        "use_msa": False,
    }]
    input_json = work_dir / "query.json"
    input_json.write_text(json.dumps(protenix_input, indent=2))

    print(f"  [{design_id}] Running Protenix APO prediction...")
    try:
        proc = _run_protenix_prediction(
            input_json_path=input_json,
            output_dir=output_dir,
            model_name=model_name,
            n_sample=1,
            use_msa=False,
            timeout=1200,
        )
    except subprocess.TimeoutExpired:
        return {"error": "Protenix APO timed out"}
    except Exception as e:
        return {"error": f"Protenix APO failed: {e}"}

    show_output = proc.returncode != 0 or _PROTENIX_VERBOSE
    if show_output:
        if proc.stdout:
            print(f"  [{design_id}] Protenix APO stdout (tail):\n{proc.stdout[-1500:]}")
        if proc.stderr:
            print(f"  [{design_id}] Protenix APO stderr (tail):\n{proc.stderr[-1500:]}")

    if proc.returncode != 0:
        return {"error": f"Protenix APO failed (rc={proc.returncode})"}

    try:
        return _parse_protenix_output(output_dir, apo_name)
    except Exception as e:
        return {"error": f"Failed to parse Protenix APO output: {e}"}


def calculate_protenix_ipsae(
    full_data_json: str,
    binder_length: int,
    target_length: int,
) -> Dict[str, float]:
    """Calculate ipSAE from Protenix full_data JSON."""
    result: Dict[str, float] = {"protenix_ipsae": 0.0}
    if not full_data_json:
        return result
    try:
        full_data = json.loads(full_data_json)
        pae_data = full_data.get("token_pair_pae") or full_data.get("pae") or []
        if not pae_data:
            return result
        pae_matrix = np.array(pae_data)
        ipsae_result = calculate_ipsae_from_pae(
            pae_matrix,
            binder_length=binder_length,
            target_length=target_length,
        )
        result["protenix_ipsae"] = ipsae_result.get("ipsae", 0.0)
        result["protenix_ipsae_binder_to_target"] = ipsae_result.get("ipsae_binder_to_target", 0.0)
        result["protenix_ipsae_target_to_binder"] = ipsae_result.get("ipsae_target_to_binder", 0.0)
    except Exception as e:
        print(f"  Warning: Protenix ipSAE calculation failed: {e}")
    return result


# =============================================================================
# PUBLIC ENTRY POINT
# =============================================================================

def run_protenix_validation_local(
    design_id: str,
    binder_seq: str,
    target_seq: str,
    target_msas: Optional[Dict[str, str]] = None,
    verbose: bool = False,
    model_name: str = DEFAULT_PROTENIX_MODEL,
) -> Dict[str, Any]:
    """
    Run Protenix HOLO (+ optional APO) validation locally.

    Returns a dict compatible with the unified `af3_*` schema.
    """
    global _PROTENIX_VERBOSE
    _PROTENIX_VERBOSE = verbose

    result: Dict[str, Any] = {
        "af3_iptm": 0.0,
        "af3_ipsae": 0.0,
        "af3_ptm": 0.0,
        "af3_plddt": 0.0,
        "af3_structure": None,
        "af3_confidence_json": None,
        "apo_structure": None,
    }

    # Ensure weights
    try:
        ensure_protenix_weights(model_name)
    except Exception as e:
        return {**result, "error": f"Protenix weights unavailable: {e}"}

    msa_dir: Optional[Path] = None
    if target_msas:
        msa_dir = Path(tempfile.mkdtemp()) / "msas"
        msa_dir.mkdir(parents=True)
        ensure_msa_files(msa_dir, target_msas)

    target_chains = target_seq.split(":") if target_seq else []
    total_target_length = sum(len(seq) for seq in target_chains)

    holo_result = _run_protenix_holo_impl(
        design_id, binder_seq, target_seq, msa_dir, model_name=model_name
    )
    if "error" in holo_result:
        return {**result, "error": holo_result["error"]}

    ipsae_result: Dict[str, float] = {}
    if holo_result.get("full_data_json"):
        ipsae_result = calculate_protenix_ipsae(
            holo_result["full_data_json"],
            binder_length=len(binder_seq),
            target_length=total_target_length,
        )

    result.update({
        "protenix_iptm": holo_result.get("iptm", 0.0),
        "protenix_ptm": holo_result.get("ptm", 0.0),
        "protenix_plddt": holo_result.get("plddt", 0.0),
        "protenix_ipsae": ipsae_result.get("protenix_ipsae", 0.0),
        "af3_iptm": holo_result.get("iptm", 0.0),
        "af3_ptm": holo_result.get("ptm", 0.0),
        "af3_plddt": holo_result.get("plddt", 0.0),
        "af3_ipsae": ipsae_result.get("protenix_ipsae", 0.0),
        "af3_structure": holo_result.get("structure_cif"),
        "af3_confidence_json": holo_result.get("confidence_json"),
        "chain_pair_iptm": holo_result.get("chain_pair_iptm", {}),
        "ranking_score": holo_result.get("ranking_score", 0.0),
        "has_clash": holo_result.get("has_clash", False),
    })

    # APO prediction (binder only)
    apo_result = _run_protenix_apo_impl(design_id, binder_seq, model_name=model_name)
    if "error" not in apo_result:
        result["apo_structure"] = apo_result.get("structure_cif")
        result["apo_plddt"] = apo_result.get("plddt", 0.0)

    return result
