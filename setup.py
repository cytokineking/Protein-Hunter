from setuptools import setup, find_packages
import os
import subprocess
from pathlib import Path

def post_install():
    """Replicates environment setup after install."""
    try:
        print("🚀 Setting up Boltz Design Environment...")

        # Download Boltz weights
        from boltz.main import download_boltz2
        cache = Path("~/.boltz").expanduser()
        cache.mkdir(parents=True, exist_ok=True)
        download_boltz2(cache)
        print("✅ Boltz weights downloaded successfully!")

        # Make DAlphaBall.gcc executable if present
        dalpha = Path("boltz/utils/DAlphaBall.gcc")
        if dalpha.exists():
            dalpha.chmod(0o755)
            print("🔧 DAlphaBall.gcc made executable")

        # Optional: Setup LigandMPNN
        ligand_dir = Path("LigandMPNN")
        if ligand_dir.exists():
            print("🧬 Setting up LigandMPNN...")
            subprocess.run(
                ["bash", "get_model_params.sh", "./model_params"],
                cwd=str(ligand_dir),
                check=True,
            )

    except Exception as e:
        print(f"⚠️ Post-install setup failed: {e}")

setup(
    name="boltz",
    version="0.1.0",
    description="Boltz Protein Hunter – setup automation for Boltz design environment",
    author="Your Name",
    author_email="you@example.com",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "matplotlib",
        "seaborn",
        "prody",
        "tqdm",
        "PyYAML",
        "requests",
        "pypdb",
        "py3Dmol",
        "logmd==0.1.45",
        "ml_collections",
        "pyrosettacolabsetup",
        "pyrosetta-installer",
        "numpy>=1.24,<1.27",
        "numba",
        "ipykernel"
    ],
    python_requires=">=3.10",
    entry_points={
        "console_scripts": [
            "boltz-setup=boltz.__main__:main"
        ],
    },
)

# Run post-install setup
if __name__ == "__main__":
    post_install()
