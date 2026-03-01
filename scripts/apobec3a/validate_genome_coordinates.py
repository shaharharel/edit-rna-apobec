"""Validate which genome assembly (hg19 vs hg38) each dataset's coordinates match.

For C-to-U editing sites:
- On + strand: the reference base should be C
- On - strand: the reference base should be G (complement of C)

We check each site against both hg19 and hg38 to determine which genome
the coordinates belong to.

Usage:
    python scripts/apobec3a/validate_genome_coordinates.py
"""

import sys
from pathlib import Path
from collections import defaultdict

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

HG19_PATH = PROJECT_ROOT / "data" / "raw" / "genomes" / "hg19.fa"
HG38_PATH = PROJECT_ROOT / "data" / "raw" / "genomes" / "hg38.fa"
COMBINED_CSV = PROJECT_ROOT / "data" / "processed" / "all_datasets_combined.csv"


def get_base(genome, chrom, pos):
    """Get the base at a 0-based position."""
    try:
        if chrom not in genome:
            return None
        chrom_len = len(genome[chrom])
        if pos < 0 or pos >= chrom_len:
            return None
        return str(genome[chrom][pos]).upper()
    except Exception:
        return None


def validate_site(genome, chrom, start, strand):
    """Check if the site has the expected base for C-to-U editing."""
    base = get_base(genome, chrom, start)
    if base is None:
        return "missing", base

    if strand == "+":
        expected = "C"
    else:
        expected = "G"  # Complement of C on - strand

    if base == expected:
        return "match", base
    else:
        return "mismatch", base


def check_flanking_context(genome, chrom, start, strand, flank=5):
    """Get flanking sequence for manual inspection."""
    try:
        seq = str(genome[chrom][start - flank:start + flank + 1]).upper()
        if strand == "-":
            comp = {"A": "T", "T": "A", "C": "G", "G": "C", "N": "N"}
            seq = "".join(comp.get(b, "N") for b in reversed(seq))
        return seq
    except Exception:
        return "N/A"


def main():
    from pyfaidx import Fasta

    print("=" * 80)
    print("GENOME COORDINATE VALIDATION")
    print("=" * 80)

    # Load datasets
    df = pd.read_csv(COMBINED_CSV)
    print(f"\nLoaded {len(df)} sites from {COMBINED_CSV.name}")
    print(f"Datasets: {df['dataset_source'].value_counts().to_dict()}")

    # Check genome availability
    have_hg19 = HG19_PATH.exists()
    have_hg38 = HG38_PATH.exists()
    print(f"\nhg19 genome: {'FOUND' if have_hg19 else 'MISSING'} ({HG19_PATH})")
    print(f"hg38 genome: {'FOUND' if have_hg38 else 'MISSING'} ({HG38_PATH})")

    genomes = {}
    if have_hg19:
        print("\nLoading hg19 genome (mmap, low RAM)...")
        genomes["hg19"] = Fasta(str(HG19_PATH))
    if have_hg38:
        print("Loading hg38 genome (mmap, low RAM)...")
        genomes["hg38"] = Fasta(str(HG38_PATH))

    if not genomes:
        print("ERROR: No genomes available!")
        return

    # Validate each dataset against each genome
    datasets = df["dataset_source"].unique()

    results = {}
    example_mismatches = defaultdict(list)

    for ds in sorted(datasets):
        ds_df = df[df["dataset_source"] == ds]
        results[ds] = {}

        for genome_name, genome in genomes.items():
            match_count = 0
            mismatch_count = 0
            missing_count = 0
            mismatch_bases = defaultdict(int)

            for _, row in ds_df.iterrows():
                status, base = validate_site(
                    genome, row["chr"], row["start"], row["strand"]
                )
                if status == "match":
                    match_count += 1
                elif status == "mismatch":
                    mismatch_count += 1
                    mismatch_bases[base] += 1
                    if len(example_mismatches[(ds, genome_name)]) < 3:
                        flanking = check_flanking_context(
                            genome, row["chr"], row["start"], row["strand"]
                        )
                        example_mismatches[(ds, genome_name)].append({
                            "site_id": row["site_id"],
                            "gene": row["gene"],
                            "chr": row["chr"],
                            "start": row["start"],
                            "strand": row["strand"],
                            "found_base": base,
                            "expected": "C" if row["strand"] == "+" else "G",
                            "flanking": flanking,
                        })
                else:
                    missing_count += 1

            total = len(ds_df)
            match_pct = 100 * match_count / total if total > 0 else 0
            results[ds][genome_name] = {
                "total": total,
                "match": match_count,
                "mismatch": mismatch_count,
                "missing": missing_count,
                "match_pct": match_pct,
                "mismatch_bases": dict(mismatch_bases),
            }

    # Print results
    print("\n" + "=" * 80)
    print("RESULTS: Base Match Rate Per Dataset Per Genome")
    print("=" * 80)

    print(f"\n{'Dataset':<20} {'N':>6}", end="")
    for gn in genomes:
        print(f"  {gn + ' match':>12} {gn + ' miss':>10} {gn + ' N/A':>8}", end="")
    print()
    print("-" * 80)

    for ds in sorted(datasets):
        n = results[ds][list(genomes.keys())[0]]["total"]
        print(f"{ds:<20} {n:>6}", end="")
        for gn in genomes:
            r = results[ds][gn]
            print(f"  {r['match_pct']:>10.1f}% {r['mismatch']:>9} {r['missing']:>7}", end="")
        print()

    # Print verdict per dataset
    print("\n" + "=" * 80)
    print("VERDICT: Which genome do coordinates belong to?")
    print("=" * 80)

    for ds in sorted(datasets):
        best_genome = None
        best_pct = 0
        for gn in genomes:
            pct = results[ds][gn]["match_pct"]
            if pct > best_pct:
                best_pct = pct
                best_genome = gn

        other_pcts = {gn: results[ds][gn]["match_pct"] for gn in genomes if gn != best_genome}
        other_str = ", ".join(f"{gn}={pct:.1f}%" for gn, pct in other_pcts.items())

        print(f"  {ds:<20} -> {best_genome} ({best_pct:.1f}% match)  [{other_str}]")

    # Print mismatch examples
    for (ds, gn), examples in example_mismatches.items():
        if examples and results[ds][gn]["match_pct"] < 95:
            print(f"\n  Example mismatches for {ds} on {gn}:")
            for ex in examples[:3]:
                print(f"    {ex['site_id']} {ex['gene']} {ex['chr']}:{ex['start']} "
                      f"({ex['strand']}) expected={ex['expected']} found={ex['found_base']} "
                      f"flanking={ex['flanking']}")

    # Final recommendation
    print("\n" + "=" * 80)
    print("RECOMMENDATION")
    print("=" * 80)

    all_hg38 = all(
        results[ds].get("hg38", {}).get("match_pct", 0) > 95
        for ds in datasets
    )
    all_hg19 = all(
        results[ds].get("hg19", {}).get("match_pct", 0) > 95
        for ds in datasets
    )

    if all_hg38:
        print("  ALL datasets have hg38 coordinates. Use hg38.fa for sequence extraction.")
    elif all_hg19:
        print("  ALL datasets have hg19 coordinates. Use hg19.fa for sequence extraction.")
    else:
        print("  MIXED coordinate systems detected!")
        for ds in sorted(datasets):
            for gn in genomes:
                pct = results[ds][gn]["match_pct"]
                if pct > 95:
                    print(f"    {ds}: {gn}")
        print("  NEED LiftOver for datasets on wrong assembly before extraction.")

    # Also validate the existing sequences_and_structures.csv if it exists
    seq_csv = PROJECT_ROOT / "data" / "processed" / "sequences_and_structures.csv"
    if seq_csv.exists():
        print("\n" + "=" * 80)
        print("VALIDATING EXISTING sequences_and_structures.csv")
        print("=" * 80)
        seq_df = pd.read_csv(seq_csv)
        print(f"  Rows: {len(seq_df)}")
        if "sequence" in seq_df.columns:
            # Check if center base is C (for RNA, U is represented as T in DNA)
            center = 100  # center of 201-nt window
            center_matches = 0
            center_mismatches = 0
            for _, row in seq_df.iterrows():
                seq = row.get("sequence", "")
                if len(seq) >= 201:
                    center_base = seq[center].upper()
                    # After reverse complement (if - strand), center should be C
                    # The sequence is already in RNA orientation, so center should be C
                    if center_base == "C":
                        center_matches += 1
                    else:
                        center_mismatches += 1
                        if center_mismatches <= 3:
                            print(f"    Mismatch: {row.get('site_id', '?')} center={center_base} "
                                  f"(expected C)")
            print(f"  Center base = C: {center_matches}/{len(seq_df)} "
                  f"({100*center_matches/len(seq_df):.1f}%)")
            if center_mismatches > 0:
                print(f"  WARNING: {center_mismatches} sites have wrong center base!")
            else:
                print(f"  All center bases are C — sequences look correct for the genome used.")

    # Check site_sequences.json too
    import json
    seq_json = PROJECT_ROOT / "data" / "processed" / "site_sequences.json"
    if seq_json.exists():
        print("\n" + "=" * 80)
        print("VALIDATING EXISTING site_sequences.json")
        print("=" * 80)
        with open(seq_json) as f:
            site_seqs = json.load(f)
        print(f"  Total sequences: {len(site_seqs)}")

        # Sample check: center base should be C
        center = 100
        ok = 0
        bad = 0
        bad_examples = []
        for sid, seq in site_seqs.items():
            if len(seq) >= 201:
                if seq[center].upper() == "C":
                    ok += 1
                else:
                    bad += 1
                    if bad <= 5:
                        bad_examples.append((sid, seq[center], seq[center-2:center+3]))
        print(f"  Center base = C: {ok}/{ok+bad} ({100*ok/(ok+bad):.1f}%)")
        if bad > 0:
            print(f"  WARNING: {bad} sites have wrong center base!")
            for sid, base, ctx in bad_examples:
                print(f"    {sid}: center={base} context={ctx}")
        else:
            print(f"  All center bases are C — sequences extracted correctly.")


if __name__ == "__main__":
    main()
