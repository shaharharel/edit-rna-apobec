#!/usr/bin/env python
"""Cross-species GB scoring: A4 analysis + chimp ortholog feature scoring.

Part 1: A4-specific cross-species analysis
  - Parse A4-correlated (181) and A4-exclusive (21) sites from Levanon T3 sheet
  - Cross-reference with the existing cross-species liftover data
  - Compare conservation, motif preservation, tissue breadth for A4 vs other enzymes

Part 2: GB model scoring of chimp orthologs
  - Extract 201-nt chimp sequences from panTro6
  - Fold with ViennaRNA to get structure features
  - Compute 40-dim hand features (motif 24 + struct delta 7 + loop 9)
  - Train XGBoost on multi-enzyme v3 data
  - Score both human and chimp feature vectors
  - Compare: do chimp orthologs score as editable as human versions?

Usage:
    conda run -n quris python scripts/multi_enzyme/cross_species_scoring.py
"""

import gc
import json
import logging
import sys
import time
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.apobec_feature_extraction import (
    extract_motif_features,
    extract_motif_from_seq,
    extract_loop_features,
    extract_structure_delta_features,
    build_hand_features,
    compute_vienna_features,
    LOOP_FEATURE_COLS,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
EXCEL_PATH = PROJECT_ROOT / "data" / "raw" / "C2TFinalSites.DB.xlsx"
SEQ_JSON_MAIN = PROJECT_ROOT / "data" / "processed" / "site_sequences.json"
SEQ_JSON_ME = PROJECT_ROOT / "data" / "processed" / "multi_enzyme" / "multi_enzyme_sequences_v3_with_negatives.json"
SPLITS_CSV = PROJECT_ROOT / "data" / "processed" / "multi_enzyme" / "splits_multi_enzyme_v3_with_negatives.csv"
LEVANON_CSV = PROJECT_ROOT / "data" / "processed" / "multi_enzyme" / "levanon_all_categories.csv"
LOOP_CSV = PROJECT_ROOT / "data" / "processed" / "multi_enzyme" / "loop_position_per_site_v3.csv"
STRUCT_CACHE = PROJECT_ROOT / "data" / "processed" / "embeddings" / "vienna_structure_cache.npz"
STRUCT_CACHE_ME = PROJECT_ROOT / "data" / "processed" / "multi_enzyme" / "structure_cache_multi_enzyme_v3.npz"
CROSS_SPECIES_CSV = PROJECT_ROOT / "experiments" / "multi_enzyme" / "outputs" / "cross_species" / "editing_sites_cross_species.csv"
UNIFIED_CSV = PROJECT_ROOT / "data" / "processed" / "advisor" / "unified_editing_sites.csv"

HG38_FA = PROJECT_ROOT / "data" / "raw" / "genomes" / "hg38.fa"
PANTRO6_FA = PROJECT_ROOT / "data" / "raw" / "genomes" / "panTro6.fa"

OUTPUT_DIR = PROJECT_ROOT / "experiments" / "multi_enzyme" / "outputs" / "cross_species"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
CENTER = 100
ENZYME_COLS = ["A1", "A3A", "A3B", "A3G", "A3H", "A4"]


def reverse_complement(seq):
    comp = str.maketrans("ACGTNacgtn", "TGCANtgcan")
    return seq.translate(comp)[::-1]


def extract_window(genome, chrom, pos, strand, window=100):
    """Extract 2*window+1 nt centered on pos (0-based)."""
    try:
        start = max(0, pos - window)
        end = pos + window + 1
        seq = str(genome[chrom][start:end]).upper()
        if len(seq) < 2 * window + 1:
            return None
        if strand == "-":
            seq = reverse_complement(seq)
        return seq
    except (KeyError, ValueError):
        return None


def extract_window_recentered(genome, chrom, pos, strand, window=100,
                              search_radius=5):
    """Extract 2*window+1 nt from genome, re-centering on the nearest C.

    LiftOver coordinates often have an off-by-one (or small offset) such that
    the target-species C is not exactly at *pos* but within a few bases of it.
    This helper extracts a slightly larger window, finds the C closest to the
    nominal center, and returns a 201-nt sequence with C at position *window*.

    Returns:
        (sequence, offset) or (None, None) if no C found within search_radius.
    """
    try:
        ext_start = max(0, pos - window - search_radius)
        ext_end = pos + window + search_radius + 1
        raw = str(genome[chrom][ext_start:ext_end]).upper()
        if strand == "-":
            raw = reverse_complement(raw)

        # Nominal center in this extended window
        nominal = (pos - ext_start) if strand == "+" else (ext_end - 1 - pos)

        # Find nearest C
        best_offset, best_dist = None, search_radius + 1
        for off in range(-search_radius, search_radius + 1):
            p = nominal + off
            if 0 <= p < len(raw) and raw[p] == "C" and abs(off) < best_dist:
                best_dist = abs(off)
                best_offset = off

        if best_offset is None:
            return None, None

        actual_center = nominal + best_offset
        s = actual_center - window
        e = actual_center + window + 1
        if s < 0 or e > len(raw):
            return None, None
        seq = raw[s:e]
        if len(seq) != 2 * window + 1:
            return None, None
        return seq, best_offset
    except (KeyError, ValueError):
        return None, None


# ============================================================================
# Part 1: A4-specific cross-species analysis
# ============================================================================

def parse_a4_from_t3():
    """Parse A4-correlated and A4-exclusive sites from Levanon T3 sheet.
    Returns (a4_site_ids, a4_exclusive_site_ids, t3_mapped_df).
    """
    logger.info("=== Parsing A4 sites from T3 sheet ===")
    t3 = pd.read_excel(EXCEL_PATH, sheet_name="T3-APOBECs Correlations", header=2)

    cols_needed = ["Chr", "Start", "End", "Genomic Category", "Gene Name",
                   "Exonic Function", "EditedInTissuesN", "EditedInTissues",
                   "AffectingAPOBEC"] + ENZYME_COLS
    t3 = t3[cols_needed].dropna(subset=["Chr"]).copy()
    t3.columns = ["chr", "start", "end", "genomic_category", "gene_name",
                   "exonic_function", "edited_in_tissues_n", "edited_in_tissues",
                   "affecting_apobec"] + ENZYME_COLS

    for col in ENZYME_COLS:
        t3[col] = t3[col].fillna(False).astype(bool)
    t3["start"] = t3["start"].astype(int)
    t3["end"] = t3["end"].astype(int)
    t3["edited_in_tissues_n"] = pd.to_numeric(t3["edited_in_tissues_n"], errors="coerce")

    # Merge with unified to get site_ids
    unified = pd.read_csv(UNIFIED_CSV)
    unified_key = unified[["site_id", "chr", "start"]].copy()
    unified_key["start"] = unified_key["start"].astype(int)
    t3_mapped = t3.merge(unified_key, on=["chr", "start"], how="left")

    n_matched = t3_mapped["site_id"].notna().sum()
    logger.info("  Matched %d / %d T3 sites to C2U site_ids", n_matched, len(t3))

    # A4-correlated (all 181)
    a4_mask = t3_mapped["A4"]
    a4_df = t3_mapped[a4_mask].copy()

    # A4-exclusive: A4=True and all others False
    others_false = (t3_mapped[["A1", "A3A", "A3B", "A3G", "A3H"]] == False).all(axis=1)
    a4_exclusive = t3_mapped[a4_mask & others_false].copy()

    logger.info("  A4-correlated: %d sites", len(a4_df))
    logger.info("  A4-exclusive: %d sites", len(a4_exclusive))

    a4_site_ids = set(a4_df["site_id"].dropna().tolist())
    a4_excl_site_ids = set(a4_exclusive["site_id"].dropna().tolist())

    return a4_site_ids, a4_excl_site_ids, t3_mapped


def run_a4_cross_species(a4_site_ids, a4_excl_site_ids):
    """Analyze A4 sites in the existing cross-species data."""
    logger.info("=== A4 cross-species analysis ===")

    # Load existing cross-species results
    cs = pd.read_csv(CROSS_SPECIES_CSV)
    logger.info("  Cross-species CSV: %d rows", len(cs))

    # Load Levanon data for enzyme categories
    lev = pd.read_csv(LEVANON_CSV)
    lev_ids = set(lev["site_id"].tolist())

    # Cross-species data uses site_ids from multi-enzyme splits
    # The Levanon sites have C2U_ prefixes. Check overlap
    cs_levanon = cs[cs["site_id"].isin(lev_ids)]
    logger.info("  Cross-species sites matching Levanon: %d", len(cs_levanon))

    # Tag cross-species sites with A4 status
    cs = cs.copy()
    cs["is_a4_correlated"] = cs["site_id"].isin(a4_site_ids)
    cs["is_a4_exclusive"] = cs["site_id"].isin(a4_excl_site_ids)

    # Also tag with Levanon enzyme_category
    lev_cat_map = dict(zip(lev["site_id"], lev["enzyme_category"]))
    cs["levanon_category"] = cs["site_id"].map(lev_cat_map)

    # How many A4 sites have cross-species orthologs?
    n_a4_in_cs = cs["is_a4_correlated"].sum()
    n_a4_excl_in_cs = cs["is_a4_exclusive"].sum()
    logger.info("  A4-correlated sites with chimp orthologs: %d / %d", n_a4_in_cs, len(a4_site_ids))
    logger.info("  A4-exclusive sites with chimp orthologs: %d / %d", n_a4_excl_in_cs, len(a4_excl_site_ids))

    # Conservation analysis by enzyme + A4 status
    results = {"a4_analysis": {}}

    # Levanon sites in cross-species
    cs_lev = cs[cs["levanon_category"].notna()].copy()
    logger.info("  Levanon sites in cross-species: %d", len(cs_lev))

    # Conservation by enzyme category (Levanon sites only)
    conservation_by_cat = {}
    for cat in ["A3A", "A3G", "A3A_A3G", "Neither", "Unknown"]:
        mask = cs_lev["levanon_category"] == cat
        subset = cs_lev[mask]
        if len(subset) > 0:
            conservation_by_cat[cat] = {
                "n_sites": int(len(subset)),
                "conserved_frac": float(subset["conserved"].mean()),
                "motif_preserved_frac": float(subset["motif_preserved"].mean()),
                "mean_sub_rate": float(subset["sub_rate"].mean()),
            }

    # A4 cross-enzyme breakdown: which enzyme categories do A4 sites belong to?
    a4_in_cs = cs[cs["is_a4_correlated"]]
    a4_cats = a4_in_cs["levanon_category"].value_counts().to_dict()
    a4_enzyme_breakdown = {}
    for cat in a4_in_cs["enzyme"].unique():
        sub = a4_in_cs[a4_in_cs["enzyme"] == cat]
        a4_enzyme_breakdown[cat] = {
            "n_sites": int(len(sub)),
            "conserved_frac": float(sub["conserved"].mean()) if len(sub) > 0 else None,
            "motif_preserved_frac": float(sub["motif_preserved"].mean()) if len(sub) > 0 else None,
        }

    # A4 conservation vs non-A4 (among Levanon sites)
    a4_lev = cs_lev[cs_lev["site_id"].isin(a4_site_ids)]
    non_a4_lev = cs_lev[~cs_lev["site_id"].isin(a4_site_ids)]

    a4_vs_non_a4 = {}
    if len(a4_lev) > 0 and len(non_a4_lev) > 0:
        # Conservation test
        a4_cons = a4_lev["conserved"].mean()
        non_a4_cons = non_a4_lev["conserved"].mean()
        tab = np.array([
            [a4_lev["conserved"].sum(), (~a4_lev["conserved"]).sum()],
            [non_a4_lev["conserved"].sum(), (~non_a4_lev["conserved"]).sum()],
        ])
        fisher_or, fisher_p = stats.fisher_exact(tab) if tab.min() >= 0 else (None, None)

        # Sub rate test
        sr_a4 = a4_lev["sub_rate"].dropna()
        sr_non = non_a4_lev["sub_rate"].dropna()
        mw_stat, mw_p = stats.mannwhitneyu(sr_a4, sr_non, alternative="two-sided") if len(sr_a4) > 0 and len(sr_non) > 0 else (None, None)

        a4_vs_non_a4 = {
            "a4_n": int(len(a4_lev)),
            "non_a4_n": int(len(non_a4_lev)),
            "a4_conserved_frac": float(a4_cons),
            "non_a4_conserved_frac": float(non_a4_cons),
            "fisher_or": float(fisher_or) if fisher_or else None,
            "fisher_p": float(fisher_p) if fisher_p else None,
            "a4_mean_sub_rate": float(sr_a4.mean()),
            "non_a4_mean_sub_rate": float(sr_non.mean()),
            "mannwhitney_p": float(mw_p) if mw_p else None,
        }

    # A4-exclusive analysis
    a4_excl_lev = cs_lev[cs_lev["site_id"].isin(a4_excl_site_ids)]
    a4_excl_info = {}
    if len(a4_excl_lev) > 0:
        a4_excl_info = {
            "n_with_orthologs": int(len(a4_excl_lev)),
            "conserved_frac": float(a4_excl_lev["conserved"].mean()),
            "motif_preserved_frac": float(a4_excl_lev["motif_preserved"].mean()),
            "mean_sub_rate": float(a4_excl_lev["sub_rate"].mean()),
            "dinucleotide_counts": a4_excl_lev["human_dinuc"].value_counts().to_dict(),
        }

    results["a4_analysis"] = {
        "a4_correlated_total": len(a4_site_ids),
        "a4_exclusive_total": len(a4_excl_site_ids),
        "a4_in_cross_species": n_a4_in_cs,
        "a4_exclusive_in_cross_species": n_a4_excl_in_cs,
        "a4_levanon_category_breakdown": {k: int(v) for k, v in a4_cats.items()},
        "a4_enzyme_breakdown": a4_enzyme_breakdown,
        "a4_vs_non_a4": a4_vs_non_a4,
        "a4_exclusive": a4_excl_info,
        "conservation_by_levanon_category": conservation_by_cat,
    }

    logger.info("  A4 conservation: %.1f%% vs non-A4: %.1f%%",
                100 * a4_vs_non_a4.get("a4_conserved_frac", 0),
                100 * a4_vs_non_a4.get("non_a4_conserved_frac", 0))

    return results, cs


# ============================================================================
# Part 2: GB model scoring of chimp orthologs
# ============================================================================

def load_structure_delta(cache_paths):
    """Load structure delta features from npz caches."""
    structure_delta = {}
    for path in cache_paths:
        if not path.exists():
            continue
        data = np.load(str(path), allow_pickle=True)
        if "site_ids" in data and "delta_features" in data:
            sids = list(data["site_ids"])
            deltas = data["delta_features"]
            for i, sid in enumerate(sids):
                structure_delta[str(sid)] = deltas[i]
            logger.info("  Loaded %d structure deltas from %s", len(sids), path.name)
    return structure_delta


def compute_chimp_features(chimp_seqs):
    """Compute 40-dim hand features for chimp sequences using ViennaRNA.

    Args:
        chimp_seqs: dict of site_id -> 201-nt chimp sequence

    Returns:
        (features_40d, struct_delta_dict, loop_rows)
    """
    logger.info("Computing ViennaRNA features for %d chimp sequences...", len(chimp_seqs))

    struct_delta = {}
    loop_rows = []
    motif_features = []
    site_ids_ordered = []

    t0 = time.time()
    for i, (sid, seq) in enumerate(chimp_seqs.items()):
        site_ids_ordered.append(sid)
        seq_rna = seq.upper().replace("T", "U")

        # Motif features (fast, no ViennaRNA needed)
        motif_feat = extract_motif_from_seq(seq_rna)
        motif_features.append(motif_feat)

        # ViennaRNA structure features
        try:
            sd, lf, _ = compute_vienna_features(seq_rna)
            struct_delta[sid] = sd
            row = {"site_id": sid}
            for j, col in enumerate(LOOP_FEATURE_COLS):
                row[col] = float(lf[j])
            loop_rows.append(row)
        except Exception as e:
            logger.warning("  Failed for %s: %s", sid, e)
            struct_delta[sid] = np.zeros(7, dtype=np.float32)
            row = {"site_id": sid}
            for col in LOOP_FEATURE_COLS:
                row[col] = 0.0
            loop_rows.append(row)

        if (i + 1) % 200 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            remaining = (len(chimp_seqs) - i - 1) / rate
            logger.info("  Computed %d/%d (%.0f seq/s, ~%.0f min remaining)",
                       i + 1, len(chimp_seqs), rate, remaining / 60)

    elapsed = time.time() - t0
    logger.info("  Finished %d chimp sequences in %.1f min (%.1f seq/s)",
               len(chimp_seqs), elapsed / 60, len(chimp_seqs) / max(elapsed, 1))

    # Build loop DataFrame
    loop_df = pd.DataFrame(loop_rows).set_index("site_id")

    # Assemble 40-dim features
    motif_arr = np.array(motif_features, dtype=np.float32)  # [N, 24]
    struct_arr = np.array([struct_delta[sid] for sid in site_ids_ordered], dtype=np.float32)  # [N, 7]
    loop_arr = np.array([
        loop_df.loc[sid, LOOP_FEATURE_COLS].values.astype(np.float32)
        if sid in loop_df.index else np.zeros(len(LOOP_FEATURE_COLS), dtype=np.float32)
        for sid in site_ids_ordered
    ], dtype=np.float32)  # [N, 9]

    features_40d = np.concatenate([motif_arr, struct_arr, loop_arr], axis=1)
    features_40d = np.nan_to_num(features_40d, nan=0.0)

    return features_40d, struct_delta, loop_df, site_ids_ordered


def train_gb_model(splits_csv, seq_json, struct_delta, loop_df):
    """Train XGBoost classifier on multi-enzyme v3 data."""
    from xgboost import XGBClassifier

    logger.info("=== Training GB model on multi-enzyme v3 data ===")

    df = pd.read_csv(splits_csv)
    with open(seq_json) as f:
        sequences = json.load(f)

    site_ids = df["site_id"].astype(str).tolist()
    labels = df["is_edited"].values

    logger.info("  Training set: %d sites (%d pos, %d neg)",
               len(site_ids), (labels == 1).sum(), (labels == 0).sum())

    # Build 40-dim features
    X = build_hand_features(site_ids, sequences, struct_delta, loop_df)
    logger.info("  Feature matrix: %s", X.shape)

    # Train XGBoost
    model = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=SEED,
        eval_metric="logloss",
        use_label_encoder=False,
    )
    model.fit(X, labels)

    # Quick training AUROC
    from sklearn.metrics import roc_auc_score
    train_probs = model.predict_proba(X)[:, 1]
    train_auroc = roc_auc_score(labels, train_probs)
    logger.info("  Training AUROC: %.4f", train_auroc)

    return model, X, site_ids, labels


def run_gb_scoring(cs_df, a4_site_ids):
    """Score chimp orthologs with GB model."""
    from pyfaidx import Fasta

    logger.info("=== GB scoring of chimp orthologs ===")

    # Load genomes
    logger.info("Loading genomes...")
    hg38 = Fasta(str(HG38_FA))
    pantro6 = Fasta(str(PANTRO6_FA))

    # Load sequences and structure data for human sites
    logger.info("Loading human feature data...")
    with open(SEQ_JSON_MAIN) as f:
        seqs_main = json.load(f)
    with open(SEQ_JSON_ME) as f:
        seqs_me = json.load(f)
    # Merge
    all_seqs = {**seqs_me, **seqs_main}

    # Load structure delta
    struct_delta = load_structure_delta([STRUCT_CACHE, STRUCT_CACHE_ME])

    # Load loop features
    loop_df = pd.DataFrame()
    if LOOP_CSV.exists():
        loop_df = pd.read_csv(LOOP_CSV)
        loop_df["site_id"] = loop_df["site_id"].astype(str)
        loop_df = loop_df.set_index("site_id")
        logger.info("  Loop features: %d sites", len(loop_df))

    # Train GB model on full multi-enzyme data
    model, _, _, _ = train_gb_model(SPLITS_CSV, SEQ_JSON_ME, struct_delta, loop_df)

    # Get cross-species positive sites (conserved C at center)
    cs_conserved = cs_df[cs_df["conserved"] == True].copy()
    logger.info("  Conserved sites for scoring: %d", len(cs_conserved))

    # Extract chimp 201-nt sequences
    # NOTE: LiftOver coordinates often have a small offset (typically -1 bp)
    # so the C is not exactly at chimp_pos.  We use extract_window_recentered
    # to search within +-5 bp for the nearest C and re-center the window.
    logger.info("Extracting chimp sequences from panTro6 (with re-centering)...")
    chimp_seqs = {}
    human_seqs_for_scoring = {}
    site_meta = []
    offset_counts = Counter()
    skip_reasons = Counter()

    for _, row in cs_conserved.iterrows():
        sid = row["site_id"]

        # Human sequence - from pre-computed or genome
        h_seq = all_seqs.get(sid)
        if h_seq is None:
            h_seq = extract_window(hg38, row["human_chr"], int(row["human_pos"]),
                                   row["human_strand"])
        if h_seq is None:
            skip_reasons["no_human_seq"] += 1
            continue

        if h_seq[CENTER] not in ("C", "c"):
            skip_reasons["human_center_not_C"] += 1
            continue

        # Chimp sequence from panTro6 with re-centering
        c_seq, offset = extract_window_recentered(
            pantro6, row["chimp_chr"], int(row["chimp_pos"]),
            row["chimp_strand"], window=CENTER, search_radius=5)

        if c_seq is None or c_seq[CENTER] not in ("C", "c"):
            skip_reasons["chimp_no_C_nearby"] += 1
            continue

        offset_counts[offset] += 1
        chimp_id = f"chimp_{sid}"
        chimp_seqs[chimp_id] = c_seq
        human_seqs_for_scoring[sid] = h_seq
        site_meta.append({
            "site_id": sid,
            "chimp_id": chimp_id,
            "enzyme": row["enzyme"],
            "motif_preserved": row["motif_preserved"],
            "is_a4": sid in a4_site_ids,
            "chimp_recenter_offset": offset,
        })

    logger.info("  Sites with both human+chimp C-center seqs: %d", len(site_meta))
    logger.info("  Re-centering offsets: %s", dict(sorted(offset_counts.items())))
    if skip_reasons:
        logger.info("  Skip reasons: %s", dict(skip_reasons))

    if len(site_meta) == 0:
        logger.warning("No valid sites for GB scoring!")
        return {}

    meta_df = pd.DataFrame(site_meta)
    human_ids = meta_df["site_id"].tolist()
    chimp_ids = meta_df["chimp_id"].tolist()

    # Compute human features (from caches)
    logger.info("Computing human features...")
    X_human = build_hand_features(human_ids, all_seqs, struct_delta, loop_df)
    logger.info("  Human feature matrix: %s, NaN count: %d",
               X_human.shape, np.isnan(X_human).sum())

    # Compute chimp features (ViennaRNA folding)
    logger.info("Computing chimp ViennaRNA features (this may take 10-20 min)...")
    X_chimp, chimp_sd, chimp_loop_df, chimp_ordered = compute_chimp_features(chimp_seqs)
    logger.info("  Chimp feature matrix: %s", X_chimp.shape)

    # Score with GB model
    logger.info("Scoring with GB model...")
    human_probs = model.predict_proba(X_human)[:, 1]
    chimp_probs = model.predict_proba(X_chimp)[:, 1]

    meta_df["human_gb_score"] = human_probs
    meta_df["chimp_gb_score"] = chimp_probs
    meta_df["score_diff"] = chimp_probs - human_probs
    meta_df["score_ratio"] = chimp_probs / np.clip(human_probs, 1e-6, None)

    # Summary statistics
    logger.info("=== GB Scoring Results ===")
    logger.info("  Human mean score:  %.4f (median %.4f)", human_probs.mean(), np.median(human_probs))
    logger.info("  Chimp mean score:  %.4f (median %.4f)", chimp_probs.mean(), np.median(chimp_probs))
    logger.info("  Mean score diff:   %.4f", meta_df["score_diff"].mean())

    # Paired test
    wil_stat, wil_p = stats.wilcoxon(human_probs, chimp_probs)
    logger.info("  Wilcoxon signed-rank p: %.2e", wil_p)

    # Correlation
    spearman_r, spearman_p = stats.spearmanr(human_probs, chimp_probs)
    pearson_r, pearson_p = stats.pearsonr(human_probs, chimp_probs)
    logger.info("  Spearman(human, chimp): r=%.4f, p=%.2e", spearman_r, spearman_p)
    logger.info("  Pearson(human, chimp):  r=%.4f, p=%.2e", pearson_r, pearson_p)

    # Per-enzyme breakdown
    per_enzyme = {}
    for enz in sorted(meta_df["enzyme"].unique()):
        mask = meta_df["enzyme"] == enz
        sub = meta_df[mask]
        h = sub["human_gb_score"].values
        c = sub["chimp_gb_score"].values
        wil_p_enz = stats.wilcoxon(h, c).pvalue if len(h) > 10 else None
        per_enzyme[enz] = {
            "n_sites": int(len(sub)),
            "human_mean_score": float(h.mean()),
            "chimp_mean_score": float(c.mean()),
            "mean_diff": float((c - h).mean()),
            "wilcoxon_p": float(wil_p_enz) if wil_p_enz else None,
            "pct_chimp_higher": float((c > h).sum() / len(h) * 100),
        }
        logger.info("  %s (n=%d): human=%.3f, chimp=%.3f, diff=%.4f, pct_chimp_higher=%.1f%%",
                    enz, len(sub), h.mean(), c.mean(), (c - h).mean(),
                    (c > h).sum() / len(h) * 100)

    # Score distribution by threshold
    thresholds = [0.3, 0.5, 0.7]
    threshold_analysis = {}
    for t in thresholds:
        h_above = (human_probs >= t).sum()
        c_above = (chimp_probs >= t).sum()
        threshold_analysis[f"t{t}"] = {
            "human_above": int(h_above),
            "chimp_above": int(c_above),
            "human_frac": float(h_above / len(human_probs)),
            "chimp_frac": float(c_above / len(chimp_probs)),
        }

    # Feature difference analysis: which features change most?
    feat_names = (
        [f"motif_{i}" for i in range(24)] +
        [f"struct_delta_{i}" for i in range(7)] +
        LOOP_FEATURE_COLS
    )
    feat_diffs = X_chimp - X_human
    feat_diff_summary = {}
    for j, name in enumerate(feat_names):
        diffs = feat_diffs[:, j]
        feat_diff_summary[name] = {
            "mean_diff": float(diffs.mean()),
            "std_diff": float(diffs.std()),
            "pct_changed": float((np.abs(diffs) > 1e-6).sum() / len(diffs) * 100),
        }

    # Sort by absolute mean diff
    sorted_feats = sorted(feat_diff_summary.items(), key=lambda x: abs(x[1]["mean_diff"]), reverse=True)
    logger.info("  Top 10 feature differences (human->chimp):")
    for name, info in sorted_feats[:10]:
        logger.info("    %s: mean_diff=%.4f, pct_changed=%.1f%%",
                    name, info["mean_diff"], info["pct_changed"])

    # Save detailed results
    meta_df.to_csv(OUTPUT_DIR / "gb_scoring_human_vs_chimp.csv", index=False)
    logger.info("  Saved scoring CSV: %s", OUTPUT_DIR / "gb_scoring_human_vs_chimp.csv")

    # Generate plots
    _plot_scoring_results(meta_df, human_probs, chimp_probs, per_enzyme)

    scoring_results = {
        "n_sites_scored": int(len(meta_df)),
        "human_mean_score": float(human_probs.mean()),
        "human_median_score": float(np.median(human_probs)),
        "chimp_mean_score": float(chimp_probs.mean()),
        "chimp_median_score": float(np.median(chimp_probs)),
        "mean_score_diff": float(meta_df["score_diff"].mean()),
        "wilcoxon_p": float(wil_p),
        "spearman_r": float(spearman_r),
        "spearman_p": float(spearman_p),
        "pearson_r": float(pearson_r),
        "pearson_p": float(pearson_p),
        "per_enzyme": per_enzyme,
        "threshold_analysis": threshold_analysis,
        "top_feature_diffs": {name: info for name, info in sorted_feats[:15]},
    }

    return scoring_results


def _plot_scoring_results(meta_df, human_probs, chimp_probs, per_enzyme):
    """Generate plots for GB scoring comparison."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # Plot 1: Scatter plot human vs chimp scores
    ax = axes[0, 0]
    enzymes = sorted(meta_df["enzyme"].unique())
    colors = {"A3A": "#e41a1c", "A3B": "#377eb8", "A3G": "#4daf4a",
              "A3A_A3G": "#ff7f00", "Neither": "#984ea3", "Unknown": "#999999"}
    for enz in enzymes:
        mask = meta_df["enzyme"] == enz
        ax.scatter(meta_df.loc[mask, "human_gb_score"],
                  meta_df.loc[mask, "chimp_gb_score"],
                  alpha=0.3, s=10, label=enz, color=colors.get(enz, "#333333"))
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5, linewidth=1)
    ax.set_xlabel("Human GB Score")
    ax.set_ylabel("Chimp GB Score")
    ax.set_title("GB Editability Score: Human vs Chimp")
    ax.legend(fontsize=8, markerscale=2)
    r = stats.spearmanr(human_probs, chimp_probs).statistic
    ax.text(0.05, 0.95, f"Spearman r = {r:.3f}", transform=ax.transAxes,
            fontsize=10, verticalalignment="top")

    # Plot 2: Score difference distribution
    ax = axes[0, 1]
    diffs = chimp_probs - human_probs
    ax.hist(diffs, bins=50, color="#377eb8", alpha=0.7, edgecolor="white")
    ax.axvline(0, color="red", linestyle="--", linewidth=1.5)
    ax.axvline(diffs.mean(), color="orange", linestyle="-", linewidth=1.5,
               label=f"Mean = {diffs.mean():.4f}")
    ax.set_xlabel("Chimp Score - Human Score")
    ax.set_ylabel("Count")
    ax.set_title("Score Difference Distribution")
    ax.legend()

    # Plot 3: Per-enzyme comparison
    ax = axes[1, 0]
    enz_names = sorted(per_enzyme.keys())
    x_pos = range(len(enz_names))
    h_scores = [per_enzyme[e]["human_mean_score"] for e in enz_names]
    c_scores = [per_enzyme[e]["chimp_mean_score"] for e in enz_names]
    width = 0.35
    bars1 = ax.bar([x - width/2 for x in x_pos], h_scores, width, label="Human", color="#e41a1c", alpha=0.7)
    bars2 = ax.bar([x + width/2 for x in x_pos], c_scores, width, label="Chimp", color="#377eb8", alpha=0.7)
    ax.set_xlabel("Enzyme")
    ax.set_ylabel("Mean GB Score")
    ax.set_title("Mean GB Score by Enzyme: Human vs Chimp")
    ax.set_xticks(list(x_pos))
    ax.set_xticklabels(enz_names, rotation=30, ha="right")
    ax.legend()
    # Add n labels
    for i, enz in enumerate(enz_names):
        ax.text(i, max(h_scores[i], c_scores[i]) + 0.01,
                f"n={per_enzyme[enz]['n_sites']}", ha="center", fontsize=8)

    # Plot 4: CDF comparison
    ax = axes[1, 1]
    h_sorted = np.sort(human_probs)
    c_sorted = np.sort(chimp_probs)
    ax.plot(h_sorted, np.linspace(0, 1, len(h_sorted)), label="Human", color="#e41a1c", linewidth=2)
    ax.plot(c_sorted, np.linspace(0, 1, len(c_sorted)), label="Chimp", color="#377eb8", linewidth=2)
    ax.set_xlabel("GB Score")
    ax.set_ylabel("Cumulative Fraction")
    ax.set_title("CDF of GB Editability Scores")
    ax.legend()
    ks_stat, ks_p = stats.ks_2samp(human_probs, chimp_probs)
    ax.text(0.05, 0.95, f"KS stat = {ks_stat:.3f}, p = {ks_p:.2e}",
            transform=ax.transAxes, fontsize=10, verticalalignment="top")

    plt.tight_layout()
    plot_path = OUTPUT_DIR / "gb_scoring_human_vs_chimp.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("  Saved plot: %s", plot_path)

    # Additional plot: A4 sites highlighted
    a4_mask = meta_df["is_a4"]
    if a4_mask.sum() > 0:
        fig, ax = plt.subplots(figsize=(8, 8))
        non_a4 = meta_df[~a4_mask]
        a4 = meta_df[a4_mask]
        ax.scatter(non_a4["human_gb_score"], non_a4["chimp_gb_score"],
                  alpha=0.2, s=8, color="#999999", label=f"Non-A4 (n={len(non_a4)})")
        ax.scatter(a4["human_gb_score"], a4["chimp_gb_score"],
                  alpha=0.7, s=30, color="#ff8c00", edgecolors="black", linewidth=0.5,
                  label=f"A4-correlated (n={len(a4)})")
        ax.plot([0, 1], [0, 1], "k--", alpha=0.5)
        ax.set_xlabel("Human GB Score")
        ax.set_ylabel("Chimp GB Score")
        ax.set_title("A4-correlated Sites: Human vs Chimp GB Scores")
        ax.legend()
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "gb_scoring_a4_highlighted.png", dpi=150, bbox_inches="tight")
        plt.close()
        logger.info("  Saved A4 highlight plot")


# ============================================================================
# Main
# ============================================================================

def main():
    t_start = time.time()

    # ----- Part 1: A4 cross-species analysis -----
    a4_site_ids, a4_excl_site_ids, t3_mapped = parse_a4_from_t3()
    a4_results, cs_df = run_a4_cross_species(a4_site_ids, a4_excl_site_ids)

    # ----- Part 2: GB scoring of chimp orthologs -----
    scoring_results = run_gb_scoring(cs_df, a4_site_ids)

    # ----- Combine and save results -----
    all_results = {
        **a4_results,
        "gb_scoring": scoring_results,
        "runtime_minutes": (time.time() - t_start) / 60,
    }

    results_path = OUTPUT_DIR / "cross_species_scoring_results.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    logger.info("Saved results: %s", results_path)

    elapsed = time.time() - t_start
    logger.info("=== COMPLETE in %.1f min ===", elapsed / 60)

    # Print summary
    print("\n" + "=" * 70)
    print("CROSS-SPECIES SCORING SUMMARY")
    print("=" * 70)

    a4 = all_results.get("a4_analysis", {})
    print(f"\n--- A4 Analysis ---")
    print(f"A4-correlated: {a4.get('a4_correlated_total', '?')} sites")
    print(f"A4-exclusive:  {a4.get('a4_exclusive_total', '?')} sites")
    print(f"A4 in cross-species: {a4.get('a4_in_cross_species', '?')}")
    a4_vs = a4.get("a4_vs_non_a4", {})
    if a4_vs:
        print(f"A4 conservation: {a4_vs.get('a4_conserved_frac', 0)*100:.1f}% vs non-A4: {a4_vs.get('non_a4_conserved_frac', 0)*100:.1f}%")
        print(f"A4 sub rate: {a4_vs.get('a4_mean_sub_rate', 0):.4f} vs non-A4: {a4_vs.get('non_a4_mean_sub_rate', 0):.4f}")
        print(f"Fisher OR: {a4_vs.get('fisher_or', '?')}, p={a4_vs.get('fisher_p', '?')}")

    print(f"\nConservation by Levanon category:")
    for cat, info in a4.get("conservation_by_levanon_category", {}).items():
        print(f"  {cat}: {info['conserved_frac']*100:.1f}% conserved (n={info['n_sites']})")

    gb = all_results.get("gb_scoring", {})
    if gb:
        print(f"\n--- GB Scoring ---")
        print(f"Sites scored: {gb.get('n_sites_scored', '?')}")
        print(f"Human mean score:  {gb.get('human_mean_score', 0):.4f}")
        print(f"Chimp mean score:  {gb.get('chimp_mean_score', 0):.4f}")
        print(f"Score diff:        {gb.get('mean_score_diff', 0):.4f}")
        print(f"Wilcoxon p:        {gb.get('wilcoxon_p', '?'):.2e}")
        print(f"Spearman r:        {gb.get('spearman_r', 0):.4f}")

        print(f"\nPer-enzyme GB scores:")
        for enz, info in sorted(gb.get("per_enzyme", {}).items()):
            print(f"  {enz}: human={info['human_mean_score']:.3f}, chimp={info['chimp_mean_score']:.3f}, "
                  f"diff={info['mean_diff']:.4f}, chimp_higher={info['pct_chimp_higher']:.1f}%")

    print(f"\nRuntime: {all_results.get('runtime_minutes', 0):.1f} min")
    print("=" * 70)


if __name__ == "__main__":
    main()
