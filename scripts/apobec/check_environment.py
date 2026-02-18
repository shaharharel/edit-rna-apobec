"""Check that all required tools and data are available.

Run this script to verify the environment is set up correctly
before running experiments.

Usage:
    python scripts/apobec/check_environment.py
"""

import importlib
import os
import shutil
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def check_python_packages():
    """Check required Python packages."""
    print("=== Python Packages ===")
    packages = [
        ("pandas", "pandas"),
        ("numpy", "numpy"),
        ("torch", "torch"),
        ("sklearn", "sklearn"),
        ("openpyxl", "openpyxl"),
        ("pyfaidx", "pyfaidx"),
        ("yaml", "pyyaml"),
    ]
    all_ok = True
    for import_name, display_name in packages:
        try:
            mod = importlib.import_module(import_name)
            version = getattr(mod, "__version__", "unknown")
            print(f"  [OK] {display_name}: {version}")
        except ImportError:
            print(f"  [MISSING] {display_name}")
            all_ok = False
    return all_ok


def check_vienna_rna():
    """Check ViennaRNA installation."""
    print("\n=== ViennaRNA ===")
    rnafold = Path("/opt/miniconda3/envs/vienna/bin/RNAfold")
    if not rnafold.exists():
        print(f"  [MISSING] RNAfold not found at {rnafold}")
        return False

    try:
        result = subprocess.run(
            [str(rnafold), "--version"],
            capture_output=True, text=True, timeout=5
        )
        version = result.stdout.strip() or result.stderr.strip()
        print(f"  [OK] RNAfold: {version}")

        # Quick functional test
        result = subprocess.run(
            [str(rnafold), "--noPS"],
            input="GCGCUUAUUUGCGC",
            capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0:
            print("  [OK] RNAfold functional test passed")
            return True
        else:
            print("  [FAIL] RNAfold functional test failed")
            return False
    except Exception as e:
        print(f"  [FAIL] RNAfold error: {e}")
        return False


def check_hg19_reference():
    """Check for hg19 reference genome."""
    print("\n=== hg19 Reference Genome ===")
    # Common locations to check
    common_paths = [
        Path.home() / "reference" / "hg19.fa",
        Path.home() / "genomes" / "hg19" / "hg19.fa",
        Path("/data/reference/hg19/hg19.fa"),
        Path("/opt/genomes/hg19/hg19.fa"),
        Path("/Users/shaharharel/Documents/github/edit-rna-apobec/data/raw/hg19.fa"),
    ]

    for p in common_paths:
        if p.exists():
            size_gb = p.stat().st_size / (1024 ** 3)
            print(f"  [OK] Found at {p} ({size_gb:.1f} GB)")
            # Check for index
            fai = p.with_suffix(".fa.fai")
            if fai.exists():
                print(f"  [OK] Index found: {fai}")
            else:
                print(f"  [WARN] No .fai index. Run: samtools faidx {p}")
            return True

    print("  [MISSING] hg19 reference genome not found")
    print("  To download:")
    print("    wget https://hgdownload.soe.ucsc.edu/goldenPath/hg19/bigZips/hg19.fa.gz")
    print("    gunzip hg19.fa.gz")
    print("    samtools faidx hg19.fa")
    print("  Or set hg19_fasta path in configs/base.yaml")
    return False


def check_data_files():
    """Check processed data files."""
    print("\n=== Data Files ===")
    advisor_dir = PROJECT_ROOT / "data" / "processed" / "advisor"
    unified = advisor_dir / "unified_editing_sites.csv"

    if unified.exists():
        import pandas as pd
        df = pd.read_csv(unified)
        print(f"  [OK] Unified site table: {len(df)} sites, {len(df.columns)} columns")
    else:
        print("  [MISSING] Unified site table. Run: python scripts/apobec/parse_advisor_excel.py")

    # Check published data
    pub_dir = PROJECT_ROOT / "data" / "raw" / "published"
    for paper in ["pecori_2019", "sharma_2015", "alqassim_2021"]:
        paper_dir = pub_dir / paper
        if paper_dir.exists():
            files = list(paper_dir.glob("*.xlsx"))
            if files:
                print(f"  [OK] {paper}: {len(files)} file(s)")
            else:
                readme = paper_dir / "DOWNLOAD_INSTRUCTIONS.txt"
                if readme.exists():
                    print(f"  [WARN] {paper}: manual download needed (see {readme})")
                else:
                    print(f"  [WARN] {paper}: no data files found")
        else:
            print(f"  [MISSING] {paper}: directory not found")


def check_project_structure():
    """Check project directory structure."""
    print("\n=== Project Structure ===")
    required_dirs = [
        "src", "src/data", "src/embedding", "src/models", "src/utils",
        "experiments", "experiments/apobec",
        "scripts", "scripts/apobec",
        "tests",
        "data", "data/raw", "data/processed",
        "configs",
    ]
    all_ok = True
    for d in required_dirs:
        path = PROJECT_ROOT / d
        if path.exists():
            print(f"  [OK] {d}/")
        else:
            print(f"  [MISSING] {d}/")
            all_ok = False
    return all_ok


def main():
    print(f"Environment Check for APOBEC RNA Editing Project")
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Python: {sys.executable} ({sys.version.split()[0]})")
    print()

    check_project_structure()
    check_python_packages()
    check_vienna_rna()
    check_hg19_reference()
    check_data_files()

    print("\n=== Summary ===")
    print("Run experiments with: /opt/miniconda3/envs/quris/bin/python")
    print("ViennaRNA available at: /opt/miniconda3/envs/vienna/bin/")


if __name__ == "__main__":
    main()
