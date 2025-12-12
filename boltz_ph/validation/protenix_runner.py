"""
Persistent Protenix runner that keeps model loaded in GPU memory.

This eliminates the ~70s model loading overhead per prediction, reducing
per-design validation time from ~105s to ~15s after the first load.

Usage:
    runner = PersistentProtenixRunner.get_instance()
    runner.ensure_loaded()  # First call loads model (~70s)
    result = runner.predict_holo(...)  # Each call ~15s

The runner uses a singleton pattern per process, so multi-GPU workflows
automatically get one runner per GPU worker.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch

# Protenix repo path
PROTENIX_REPO_ROOT = Path(__file__).resolve().parents[2] / "Protenix"
DEFAULT_MODEL = "protenix_base_default_v0.5.0"

# Inference settings (match AF3 defaults)
N_CYCLE = 10
N_STEP = 200
N_SAMPLE = 1
SEED = 101


def _get_weights_dir() -> Path:
    """Get Protenix weights directory."""
    env_dir = os.environ.get("PROTENIX_WEIGHTS_DIR")
    if env_dir:
        return Path(env_dir).expanduser()
    return Path("~/.protein-hunter/protenix_weights").expanduser()


class PersistentProtenixRunner:
    """
    Singleton-style persistent Protenix runner.
    
    Keeps the model loaded in GPU memory across multiple predictions,
    eliminating the ~70s initialization overhead per prediction.
    """
    
    _instance: Optional["PersistentProtenixRunner"] = None
    
    @classmethod
    def get_instance(cls, device: int = 0) -> "PersistentProtenixRunner":
        """
        Get or create the singleton runner instance.
        
        Args:
            device: GPU device ID (only used on first call)
        
        Returns:
            The singleton PersistentProtenixRunner instance
        """
        if cls._instance is None:
            cls._instance = cls(device=device)
        return cls._instance
    
    @classmethod
    def shutdown(cls) -> None:
        """Explicitly unload model and release GPU memory."""
        if cls._instance is not None:
            cls._instance.unload()
            cls._instance = None
    
    @classmethod
    def is_loaded(cls) -> bool:
        """Check if the runner is currently loaded."""
        return cls._instance is not None and cls._instance._loaded
    
    def __init__(self, device: int = 0):
        """
        Initialize the runner (does NOT load model yet).
        
        Args:
            device: GPU device ID to use
        """
        self.device = device
        self.weights_dir = _get_weights_dir()
        self.runner = None  # Protenix InferenceRunner
        self.configs = None
        self._loaded = False
        self._protenix_imported = False
        self._load_time = 0.0
    
    def _setup_imports(self) -> None:
        """Setup Protenix imports (done once)."""
        if self._protenix_imported:
            return
        
        # Add Protenix to path
        sys.path.insert(0, str(PROTENIX_REPO_ROOT))
        os.chdir(str(PROTENIX_REPO_ROOT))
        
        # Set environment
        os.environ["LAYERNORM_TYPE"] = "torch"
        os.environ["CUDA_VISIBLE_DEVICES"] = str(self.device)
        
        self._protenix_imported = True
    
    def ensure_loaded(self) -> float:
        """
        Load model if not already loaded.
        
        Returns:
            Time taken to load (0.0 if already loaded)
        """
        if self._loaded:
            return 0.0
        return self._load_model()
    
    def _load_model(self) -> float:
        """
        Internal: load Protenix model into GPU memory.
        
        Returns:
            Time taken to load the model
        """
        self._setup_imports()
        
        print(f"  Loading Protenix model from {self.weights_dir}...")
        start_time = time.time()
        
        # Import after path setup
        from configs.configs_base import configs as configs_base
        from configs.configs_data import data_configs
        from configs.configs_inference import inference_configs
        from configs.configs_model_type import model_configs
        from ml_collections.config_dict import ConfigDict
        from protenix.config import parse_configs
        from runner.inference import InferenceRunner, update_gpu_compatible_configs
        
        # Build arg string to mimic CLI invocation
        dump_dir = tempfile.mkdtemp()
        arg_str = " ".join([
            f"--model_name={DEFAULT_MODEL}",
            f"--load_checkpoint_dir={self.weights_dir}",
            f"--dump_dir={dump_dir}",
            f"--input_json_path=/tmp/placeholder.json",
            f"--seeds={SEED}",
            f"--sample_diffusion.N_sample={N_SAMPLE}",
            f"--model.N_cycle={N_CYCLE}",
            f"--sample_diffusion.N_step={N_STEP}",
            "--enable_tf32=true",
            "--enable_efficient_fusion=true",
            "--enable_diffusion_shared_vars_cache=true",
            "--need_atom_confidence=true",
            "--use_msa=true",
            "--num_workers=0",
        ])
        
        # Parse configs exactly like Protenix does
        base_configs = {**configs_base, **{"data": data_configs}, **inference_configs}
        self.configs = parse_configs(
            configs=base_configs,
            arg_str=arg_str,
            fill_required_with_null=True,
        )
        
        # Add model-specific configs
        model_specific = ConfigDict(model_configs[DEFAULT_MODEL])
        self.configs.update(model_specific)
        
        # GPU compatibility
        self.configs = update_gpu_compatible_configs(self.configs)
        
        # Create runner (loads model!)
        self.runner = InferenceRunner(self.configs)
        
        self._load_time = time.time() - start_time
        self._loaded = True
        
        print(f"  ✓ Protenix model loaded in {self._load_time:.1f}s")
        return self._load_time
    
    def predict_holo(
        self,
        design_id: str,
        binder_seq: str,
        target_seq: str,
        target_msas: Optional[Dict[str, str]] = None,
        output_dir: Optional[Path] = None,
    ) -> Dict[str, Any]:
        """
        Run HOLO prediction (binder + target) using loaded model.
        
        Args:
            design_id: Unique identifier for this design
            binder_seq: Binder sequence (chain A)
            target_seq: Target sequence(s), colon-separated for multi-chain
            target_msas: Optional dict mapping chain_id -> MSA content
            output_dir: Optional output directory (temp if not provided)
        
        Returns:
            Dict with prediction results including iptm, ptm, plddt, structure_cif, etc.
        """
        if not self._loaded:
            raise RuntimeError("Model not loaded. Call ensure_loaded() first.")
        
        self._setup_imports()
        
        from protenix.data.infer_data_pipeline import InferenceDataset
        from protenix.utils.seed import seed_everything
        from runner.inference import update_inference_configs
        
        # Create work directory
        work_dir = Path(output_dir) if output_dir else Path(tempfile.mkdtemp())
        work_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup MSA directory structure
        msa_dir = self._setup_msa_directory(work_dir, binder_seq, target_seq, target_msas)
        
        # Build input JSON
        holo_name = f"{design_id}_holo"
        input_dict = self._build_input_dict(holo_name, binder_seq, target_seq, msa_dir)
        
        json_path = work_dir / "query.json"
        json_path.write_text(json.dumps([input_dict], indent=2))
        
        # Run prediction
        total_start = time.time()
        
        # Update configs for this prediction
        self.configs.input_json_path = str(json_path)
        self.configs.dump_dir = str(work_dir / "output")
        self.runner.dump_dir = str(work_dir / "output")
        self.runner.dumper.base_dir = str(work_dir / "output")
        os.makedirs(self.configs.dump_dir, exist_ok=True)
        
        # Load data
        data_start = time.time()
        dataset = InferenceDataset(
            input_json_path=str(json_path),
            dump_dir=self.configs.dump_dir,
            use_msa=self.configs.use_msa,
            configs=self.configs,
        )
        data, atom_array, error_msg = dataset[0]
        data_time = time.time() - data_start
        
        if error_msg:
            return {"error": f"Data featurization failed: {error_msg}"}
        
        # Run prediction
        seed_everything(seed=SEED, deterministic=False)
        
        new_configs = update_inference_configs(self.configs, data["N_token"].item())
        self.runner.update_model_configs(new_configs)
        
        forward_start = time.time()
        prediction = self.runner.predict(data)
        forward_time = time.time() - forward_start
        
        # Save results
        self.runner.dumper.dump(
            dataset_name="",
            pdb_id=holo_name,
            seed=SEED,
            pred_dict=prediction,
            atom_array=atom_array,
            entity_poly_type=data["entity_poly_type"],
        )
        
        total_time = time.time() - total_start
        
        # Parse output
        result = self._parse_output(work_dir / "output", holo_name)
        result.update({
            "total_time": total_time,
            "data_time": data_time,
            "forward_time": forward_time,
        })
        
        # Clear CUDA cache
        torch.cuda.empty_cache()
        
        return result
    
    def predict_apo(
        self,
        design_id: str,
        binder_seq: str,
        output_dir: Optional[Path] = None,
    ) -> Dict[str, Any]:
        """
        Run APO prediction (binder only) using loaded model.
        
        Args:
            design_id: Unique identifier for this design
            binder_seq: Binder sequence
            output_dir: Optional output directory
        
        Returns:
            Dict with prediction results
        """
        if not self._loaded:
            raise RuntimeError("Model not loaded. Call ensure_loaded() first.")
        
        self._setup_imports()
        
        from protenix.data.infer_data_pipeline import InferenceDataset
        from protenix.utils.seed import seed_everything
        from runner.inference import update_inference_configs
        
        # Create work directory
        work_dir = Path(output_dir) if output_dir else Path(tempfile.mkdtemp())
        work_dir.mkdir(parents=True, exist_ok=True)
        
        # Build APO input (binder only, no MSA)
        apo_name = f"{design_id}_apo"
        apo_input = [{
            "name": apo_name,
            "sequences": [{"proteinChain": {"sequence": binder_seq, "count": 1}}],
            "use_msa": False,
        }]
        
        json_path = work_dir / "apo_query.json"
        json_path.write_text(json.dumps(apo_input, indent=2))
        
        # Run prediction
        total_start = time.time()
        
        # Update configs
        self.configs.input_json_path = str(json_path)
        self.configs.dump_dir = str(work_dir / "apo_output")
        self.runner.dump_dir = str(work_dir / "apo_output")
        self.runner.dumper.base_dir = str(work_dir / "apo_output")
        os.makedirs(self.configs.dump_dir, exist_ok=True)
        
        # Temporarily disable MSA for APO
        original_use_msa = self.configs.use_msa
        self.configs.use_msa = False
        
        try:
            # Load data
            dataset = InferenceDataset(
                input_json_path=str(json_path),
                dump_dir=self.configs.dump_dir,
                use_msa=False,
                configs=self.configs,
            )
            data, atom_array, error_msg = dataset[0]
            
            if error_msg:
                return {"error": f"APO data featurization failed: {error_msg}"}
            
            # Run prediction
            seed_everything(seed=SEED, deterministic=False)
            
            new_configs = update_inference_configs(self.configs, data["N_token"].item())
            self.runner.update_model_configs(new_configs)
            
            prediction = self.runner.predict(data)
            
            # Save results
            self.runner.dumper.dump(
                dataset_name="",
                pdb_id=apo_name,
                seed=SEED,
                pred_dict=prediction,
                atom_array=atom_array,
                entity_poly_type=data["entity_poly_type"],
            )
            
            total_time = time.time() - total_start
            
            # Parse output
            result = self._parse_output(work_dir / "apo_output", apo_name)
            result["total_time"] = total_time
            
        finally:
            # Restore MSA setting
            self.configs.use_msa = original_use_msa
        
        torch.cuda.empty_cache()
        
        return result
    
    def _setup_msa_directory(
        self,
        work_dir: Path,
        binder_seq: str,
        target_seq: str,
        target_msas: Optional[Dict[str, str]] = None,
    ) -> Path:
        """Setup MSA directory structure for Protenix."""
        msa_dir = work_dir / "msas"
        msa_dir.mkdir(parents=True, exist_ok=True)
        
        # Binder MSA (chain A) - single sequence
        chain_a_dir = msa_dir / "chain_A"
        chain_a_dir.mkdir(parents=True, exist_ok=True)
        (chain_a_dir / "non_pairing.a3m").write_text(f">binder\n{binder_seq}\n")
        
        # Target MSAs (chain B, C, ...)
        target_chains = target_seq.split(":") if target_seq else []
        for i, seq in enumerate(target_chains):
            chain_id = chr(ord("B") + i)
            chain_dir = msa_dir / f"chain_{chain_id}"
            chain_dir.mkdir(parents=True, exist_ok=True)
            
            if target_msas and chain_id in target_msas:
                # Use provided MSA
                msa_content = self._convert_msa_format(target_msas[chain_id])
                (chain_dir / "non_pairing.a3m").write_text(msa_content)
            else:
                # Single sequence fallback
                (chain_dir / "non_pairing.a3m").write_text(f">target_{chain_id}\n{seq}\n")
        
        return msa_dir
    
    def _convert_msa_format(self, msa_content: str) -> str:
        """Convert ColabFold MSA format to Protenix format."""
        lines = msa_content.strip().split("\n")
        converted = []
        pseudo_tax = 0
        
        for line in lines:
            if line.startswith(">"):
                header = line[1:].strip()
                if header.isdigit() or header == "query":
                    converted.append(">query")
                elif "UniRef" in header or "UniProt" in header:
                    pseudo_tax += 1
                    parts = header.split()
                    identifier = parts[0].replace("|", "_").replace("/", "_")[:30]
                    converted.append(f">UniRef100_{identifier}_{pseudo_tax}/1-1000")
                else:
                    pseudo_tax += 1
                    clean_id = "".join(c if c.isalnum() else "_" for c in header[:20])
                    converted.append(f">UniRef100_{clean_id}_{pseudo_tax}/1-1000")
            else:
                converted.append(line)
        
        return "\n".join(converted)
    
    def _build_input_dict(
        self,
        name: str,
        binder_seq: str,
        target_seq: str,
        msa_dir: Path,
    ) -> Dict[str, Any]:
        """Build Protenix input dictionary."""
        sequences = []
        
        # Binder (chain A)
        binder_entry = {"proteinChain": {"sequence": binder_seq, "count": 1}}
        binder_msa_dir = msa_dir / "chain_A"
        if binder_msa_dir.exists():
            binder_entry["proteinChain"]["msa"] = {
                "precomputed_msa_dir": str(binder_msa_dir),
                "pairing_db": "uniref100",
            }
        sequences.append(binder_entry)
        
        # Target chains (B, C, ...)
        target_chains = target_seq.split(":") if target_seq else []
        for i, seq in enumerate(target_chains):
            chain_id = chr(ord("B") + i)
            target_entry = {"proteinChain": {"sequence": seq, "count": 1}}
            chain_msa_dir = msa_dir / f"chain_{chain_id}"
            if chain_msa_dir.exists():
                target_entry["proteinChain"]["msa"] = {
                    "precomputed_msa_dir": str(chain_msa_dir),
                    "pairing_db": "uniref100",
                }
            sequences.append(target_entry)
        
        return {"name": name, "sequences": sequences, "use_msa": True}
    
    def _parse_output(self, output_dir: Path, name: str) -> Dict[str, Any]:
        """Parse Protenix output files."""
        result = {
            "iptm": 0.0,
            "ptm": 0.0,
            "plddt": 0.0,
            "ranking_score": 0.0,
            "has_clash": False,
            "structure_cif": None,
            "confidence_json": None,
            "full_data_json": None,
            "chain_plddt": {},
            "chain_ptm": {},
            "chain_pair_iptm": {},
        }
        
        # Find result directory
        for subdir in [output_dir / name, output_dir]:
            for seed_dir in [f"seed_{SEED}", str(SEED), ""]:
                check_dir = subdir / seed_dir / "predictions" if seed_dir else subdir / "predictions"
                if not check_dir.exists():
                    check_dir = subdir / seed_dir if seed_dir else subdir
                if not check_dir.exists():
                    continue
                
                # Find confidence JSON
                for pattern in [
                    f"{name}_summary_confidence_sample_0.json",
                    f"{name}_{SEED}_summary_confidence_sample_0.json",
                    "summary_confidence*.json",
                ]:
                    matches = list(check_dir.glob(pattern))
                    for conf_file in matches:
                        try:
                            conf = json.loads(conf_file.read_text())
                            result["iptm"] = conf.get("iptm", 0.0)
                            result["ptm"] = conf.get("ptm", 0.0)
                            result["plddt"] = conf.get("plddt", 0.0)
                            result["ranking_score"] = conf.get("ranking_score", 0.0)
                            result["has_clash"] = conf.get("has_clash", False)
                            result["chain_plddt"] = conf.get("chain_plddt", {})
                            result["chain_ptm"] = conf.get("chain_ptm", {})
                            result["chain_pair_iptm"] = conf.get("chain_pair_iptm", {})
                            result["confidence_json"] = conf_file.read_text()
                            break
                        except Exception:
                            pass
                
                # Find structure CIF
                for pattern in [
                    f"{name}_{SEED}_sample_0.cif",
                    f"{name}_seed_{SEED}_sample_0.cif",
                    f"{name}_sample_0.cif",
                    "*.cif",
                ]:
                    matches = list(check_dir.glob(pattern))
                    if matches:
                        result["structure_cif"] = matches[0].read_text()
                        break
                
                # Find full data JSON (for PAE/ipSAE)
                for pattern in [
                    "full_data_sample_0.json",
                    f"{name}_full_data_sample_0.json",
                    "full_data.json",
                ]:
                    full_data_file = check_dir / pattern
                    if full_data_file.exists():
                        result["full_data_json"] = full_data_file.read_text()
                        break
                
                if result["confidence_json"]:
                    return result
        
        return result
    
    def unload(self) -> None:
        """Free GPU memory by unloading the model."""
        if self.runner is not None:
            try:
                del self.runner.model
            except AttributeError:
                pass
            del self.runner
            self.runner = None
        
        self.configs = None
        self._loaded = False
        
        torch.cuda.empty_cache()
        print("  ✓ Protenix model unloaded")
    
    @property
    def load_time(self) -> float:
        """Return the time taken to load the model."""
        return self._load_time
