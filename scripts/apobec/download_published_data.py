"""Download supplementary data from published APOBEC3A editing studies.

Three key papers:
1. Pecori et al. 2019 (IJMS): "Functions and Consequences of AID/APOBEC-Mediated DNA and RNA Deamination"
   - Table S1: Known APOBEC editing sites
   - DOI: 10.3390/ijms20225621

2. Sharma et al. 2015 (Nat Commun): "APOBEC3A cytidine deaminase induces RNA editing in monocytes and macrophages"
   - Supplementary Data 1-3: Editing sites in monocytes/macrophages
   - DOI: 10.1038/ncomms7881

3. Alqassim et al. 2021 (Commun Biol): "RNA editing enzyme APOBEC3A promotes pro-inflammatory M1 macrophage polarization"
   - Supplementary Data: Editing sites linked to macrophage polarization
   - DOI: 10.1038/s42003-020-01620-x

Usage:
    python scripts/apobec/download_published_data.py [--output-dir PATH]
"""

import argparse
import logging
import urllib.request
import urllib.error
from pathlib import Path

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "raw" / "published"

# Supplementary data URLs
# Note: Nature and MDPI supplementary files can sometimes change URLs.
# These are best-effort URLs; manual download may be needed.
DATASETS = {
    "pecori_2019": {
        "description": "Pecori et al. 2019 - APOBEC editing sites review (IJMS)",
        "doi": "10.3390/ijms20225621",
        "urls": [
            # MDPI supplementary files for article 5621
            ("https://www.mdpi.com/1422-0067/20/22/5621/s1", "pecori_2019_table_s1.xlsx"),
        ],
    },
    "sharma_2015": {
        "description": "Sharma et al. 2015 - APOBEC3A RNA editing in monocytes (Nat Commun)",
        "doi": "10.1038/ncomms7881",
        "urls": [
            # Nature supplementary data files
            ("https://static-content.springer.com/esm/art%3A10.1038%2Fncomms7881/MediaObjects/41467_2015_BFncomms7881_MOESM1249_ESM.xlsx",
             "sharma_2015_supp_data1.xlsx"),
            ("https://static-content.springer.com/esm/art%3A10.1038%2Fncomms7881/MediaObjects/41467_2015_BFncomms7881_MOESM1250_ESM.xlsx",
             "sharma_2015_supp_data2.xlsx"),
            ("https://static-content.springer.com/esm/art%3A10.1038%2Fncomms7881/MediaObjects/41467_2015_BFncomms7881_MOESM1251_ESM.xlsx",
             "sharma_2015_supp_data3.xlsx"),
        ],
    },
    "alqassim_2021": {
        "description": "Alqassim et al. 2021 - APOBEC3A in M1 macrophage polarization (Commun Biol)",
        "doi": "10.1038/s42003-020-01620-x",
        "urls": [
            ("https://static-content.springer.com/esm/art%3A10.1038%2Fs42003-020-01620-x/MediaObjects/42003_2020_1620_MOESM4_ESM.xlsx",
             "alqassim_2021_supp_data1.xlsx"),
            ("https://static-content.springer.com/esm/art%3A10.1038%2Fs42003-020-01620-x/MediaObjects/42003_2020_1620_MOESM5_ESM.xlsx",
             "alqassim_2021_supp_data2.xlsx"),
        ],
    },
}


def download_file(url: str, output_path: Path, timeout: int = 30) -> bool:
    """Download a file from URL. Returns True on success."""
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=timeout) as response:
            data = response.read()
            output_path.write_bytes(data)
            logger.info("  Downloaded: %s (%d bytes)", output_path.name, len(data))
            return True
    except (urllib.error.URLError, urllib.error.HTTPError, OSError) as e:
        logger.warning("  Failed to download %s: %s", url, e)
        return False


def download_all(output_dir: Path) -> dict[str, list[Path]]:
    """Download all published datasets. Returns dict of paper -> list of downloaded files."""
    output_dir.mkdir(parents=True, exist_ok=True)
    results = {}

    for paper_id, info in DATASETS.items():
        logger.info("Downloading: %s", info["description"])
        logger.info("  DOI: %s", info["doi"])
        paper_dir = output_dir / paper_id
        paper_dir.mkdir(exist_ok=True)

        downloaded = []
        for url, filename in info["urls"]:
            out_path = paper_dir / filename
            if out_path.exists():
                logger.info("  Already exists: %s", filename)
                downloaded.append(out_path)
                continue
            if download_file(url, out_path):
                downloaded.append(out_path)

        results[paper_id] = downloaded

        if not downloaded:
            logger.warning("  No files downloaded for %s. Manual download may be needed.", paper_id)
            logger.warning("  Visit: https://doi.org/%s", info["doi"])
            # Write a README with instructions
            readme = paper_dir / "DOWNLOAD_INSTRUCTIONS.txt"
            readme.write_text(
                f"Manual download needed for: {info['description']}\n"
                f"DOI: https://doi.org/{info['doi']}\n\n"
                f"Please download supplementary data files and place them in this directory.\n"
                f"Expected files:\n" +
                "\n".join(f"  - {fn}" for _, fn in info["urls"]) + "\n"
            )

    return results


def main():
    parser = argparse.ArgumentParser(description="Download published APOBEC editing datasets")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT,
                        help="Output directory for downloaded data")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    results = download_all(args.output_dir)

    # Summary
    total = sum(len(v) for v in results.values())
    logger.info("Download complete: %d files across %d papers", total, len(results))
    for paper_id, files in results.items():
        status = f"{len(files)} files" if files else "MANUAL DOWNLOAD NEEDED"
        logger.info("  %s: %s", paper_id, status)


if __name__ == "__main__":
    main()
