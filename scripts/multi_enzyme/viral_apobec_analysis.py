#!/usr/bin/env python3
"""B7: HIV/HBV APOBEC Mutation Analysis.

Analyzes APOBEC-driven mutations in viral genomes (HIV and HBV) and compares
with our model's predictions. Since direct download of viral mutation catalogs
requires manual interaction with the Los Alamos HIV database, this script:

1. Documents available data sources and published findings
2. Constructs test sequences based on published APOBEC viral editing patterns
3. Scores them with our trained GB model to test cross-domain predictions
4. Compares model scores for known viral APOBEC target contexts vs non-targets

Background:
-----------
HIV: APOBEC3G restricts HIV-1 by deaminating C->U on the minus-strand DNA
     during reverse transcription. This appears as G->A on the plus strand.
     - A3G prefers CC dinucleotide on minus strand = GG->AG on plus strand
     - A3F prefers TC dinucleotide on minus strand = GA->AA on plus strand
     - Vif protein normally blocks APOBEC3G; hypermutation occurs when Vif is defective

HBV: APOBEC3A/3B edit hepatitis B viral DNA/RNA
     - A3B has highest editing efficiency (~65% of cytidines converted in vitro)
     - A3G is the dominant deaminase in vivo (~35% of HBV genomes edited)
     - Editing occurs during RNA->(-)-DNA reverse transcription in viral capsid

Key references:
- Sadler et al. (2010) J Virol: In-depth NGS analysis of G->A in HIV env
- Suspene et al. (2004, 2006): Twin gradients in APOBEC3 edited HIV DNA
- Vartanian et al. (2010) PLoS Pathog: Massive A3 editing of HBV in cirrhosis
- Suspene et al. (2005) PNAS: Extensive editing of HBV DNA strands by APOBEC3

Usage:
    conda run -n quris python scripts/multi_enzyme/viral_apobec_analysis.py
"""

import gc
import json
import logging
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.apobec_feature_extraction import extract_motif_from_seq, LOOP_FEATURE_COLS

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Paths
DATA_DIR = PROJECT_ROOT / "data"
MULTI_SPLITS = DATA_DIR / "processed/multi_enzyme/splits_multi_enzyme_v3_with_negatives.csv"
MULTI_SEQS = DATA_DIR / "processed/multi_enzyme/multi_enzyme_sequences_v3_with_negatives.json"
LOOP_CSV = DATA_DIR / "processed/multi_enzyme/loop_position_per_site_v3.csv"
STRUCT_CACHE = DATA_DIR / "processed/multi_enzyme/structure_cache_multi_enzyme_v3.npz"
OUTPUT_DIR = PROJECT_ROOT / "experiments/multi_enzyme/outputs/viral_apobec"

SEED = 42


# ============================================================================
# Published viral APOBEC motif preferences (from literature)
# ============================================================================

VIRAL_APOBEC_DATA = {
    "HIV_A3G": {
        "enzyme": "APOBEC3G",
        "virus": "HIV-1",
        "mechanism": "Deaminates C->U on minus-strand DNA during reverse transcription",
        "plus_strand_signature": "GG->AG (G-to-A at second G of GG dinucleotide)",
        "minus_strand_target": "CC dinucleotide (C at 3' end of CC)",
        "preferred_context_minus": "CC",  # C is target, preceded by C
        "hot_trinucleotides_minus": ["CCC", "TCC", "ACC"],  # most edited
        "cold_trinucleotides_minus": ["GCA", "GCG", "GCT"],  # least edited
        "references": [
            "Sadler et al. (2010) J Virol 84:7923-7931",
            "Chelico et al. (2006) Nat Struct Mol Biol 13:392-399",
            "Suspene et al. (2004) Nucleic Acids Res 32:2421-2429",
        ],
        "key_findings": [
            "66-73% of G->A mutations in GG dinucleotide context",
            "Processivity: 3' to 5' on minus strand (5' to 3' on plus strand)",
            "Gradient: more editing near 5' end of plus strand (3' of minus strand)",
            "All-or-nothing: sequences are either heavily mutated or not at all",
        ],
        "editing_rate_range": "1.3-6.5% of guanines mutated (endogenous levels)",
    },
    "HIV_A3F": {
        "enzyme": "APOBEC3F",
        "virus": "HIV-1",
        "mechanism": "Deaminates C->U on minus-strand DNA (same as A3G but different context)",
        "plus_strand_signature": "GA->AA (G-to-A at G of GA dinucleotide)",
        "minus_strand_target": "TC dinucleotide (C at 3' end of TC)",
        "preferred_context_minus": "TC",
        "hot_trinucleotides_minus": ["TCA", "TCT", "TCC"],
        "cold_trinucleotides_minus": ["GCA", "GCG"],
        "references": [
            "Liddament et al. (2004) Curr Biol 14:1385-1391",
            "Wiegand et al. (2004) Mol Cell 14:795-803",
        ],
        "key_findings": [
            "72-76% of G->A mutations in GA dinucleotide context",
            "A3F and A3G rarely co-mutate the same HIV genome",
            "A3F contributes less to in vivo restriction than A3G",
        ],
    },
    "HBV_A3B": {
        "enzyme": "APOBEC3B",
        "virus": "HBV",
        "mechanism": "Edits HBV minus-strand DNA during reverse transcription",
        "target_context": "TC (same as human RNA editing by A3A/A3B)",
        "preferred_context": "TC",
        "hot_trinucleotides": ["TCA", "TCT", "TCC"],
        "references": [
            "Suspene et al. (2005) PNAS 102:8321-8326",
            "Vartanian et al. (2010) PLoS Pathog 6:e1000928",
        ],
        "key_findings": [
            "A3B has highest editing efficiency (up to 65% of cytidines)",
            "3-fold higher mutation rate than A3H-II or A3C",
            "TC context preference same as for human RNA editing",
        ],
    },
    "HBV_A3G": {
        "enzyme": "APOBEC3G",
        "virus": "HBV",
        "mechanism": "Edits HBV minus-strand DNA (dominant in vivo)",
        "target_context": "CC (same as human RNA editing by A3G)",
        "preferred_context": "CC",
        "hot_trinucleotides": ["CCA", "CCT", "CCC"],
        "references": [
            "Suspene et al. (2005) PNAS 102:8321-8326",
            "Vartanian et al. (2010) PLoS Pathog 6:e1000928",
        ],
        "key_findings": [
            "Dominant deaminase in vivo (up to 35% of genomes edited)",
            "CC context preference consistent with human A3G RNA editing",
            "In vivo editing confirmed in cirrhosis patients",
        ],
    },
}


# ============================================================================
# HIV-1 reference sequences (HXB2 env region, partial)
# ============================================================================

# HXB2 env V3 loop and flanking region (~500 nt) — well-studied for A3G editing
# This is a representative segment of the HIV-1 HXB2 reference (GenBank K03455)
# The env gene is the primary target for A3G analysis in published studies
HIV1_ENV_SEGMENT = (
    "CTTTTGCTATTGTAAAAAGTGTTGCTTTCATTGCCAAGTTTGTTTCATAACAAAA"
    "GGCTTAGGCATCTCCTATGGCAGGAAGAAGCGGAGACAGCGACGAAGAGCTCCTCAAGGCAGTCAGACTCATCAAGTTTCTCTATCAAAGCAGTAAGTAGTACATGTAATG"
    "CAACCTATAATAGTAGCAATAGTAGCATTAGTAGTAGCAATAATAATAGCAATAGTTGTGTGGTCCATAGTAATCATAGAATATAGGAAAATATTAAGACAAAGAAAAATAG"
    "ACAGGTTAATTGATAGACTAATAGAAAGAGCAGAAGACAGTGGCAATGAGAGTGAAGGAGAAATATCAGCACTTGTGGAGATGGGGGTGGAAATGGGGCACCATGCTCCTTG"
    "GGATGTTGATGATCTGTAGTGCTACAGAAAAATTGTGGGTCACAGTCTATTATGGGGTACCTGTGTGGAAGGAAGCAACCACCACTCTATTTTGTGCATCAGATGCTAAAGC"
)

# HBV genome segment (from GenBank V00866, HBV ayw subtype, preC/C region)
# Well-studied target for A3G editing in hepatocytes
HBV_PREC_SEGMENT = (
    "ATGGCTTTGGGGCATGGACATTGACCCGTATAAAGAATTTGGAGCTTCTGTGGAGTTACTCTCTTTTTTGCCTTCTGACTTCTTTCCTTCCGTCAGAGATCTCCTAGACACCGC"
    "CTCAGCTCTGTATCGGGAAGCCTTAGAGTCTCCTGAGCATTGTTCACCTCACCATACTGCACTCAGGCAAGCCATTCTCTGCTGGGGGGAATTGATGACTCTAGCTACCTGG"
    "GTGGGTAATAATTTGGAAGATCCAGCATCTAGGGATCTTGTAGTAAATTATGTTAATACTAACATGGGCCTAAAAATCAGACAACTATTGTGGTTTCATATATCTTGCCTTAC"
)


def generate_test_sequences_from_viral(viral_seq, edit_positions=None, window=100):
    """Generate 201-nt test sequences centered on each C position in a viral sequence.

    For each C in the sequence (within valid window), creates a 201-nt context window.
    These represent potential APOBEC editing targets in the viral genome.

    Returns list of dicts with: seq_201nt, position, trinucleotide, dinucleotide_5p
    """
    seq = viral_seq.upper()
    test_seqs = []

    for i in range(window, len(seq) - window):
        if seq[i] != "C":
            continue

        context_201 = seq[i - window:i + window + 1]
        if len(context_201) != 201:
            continue

        trinuc = seq[max(0, i-1):i+2]
        dinuc_5p = seq[max(0, i-1):i+1] if i > 0 else "NC"

        test_seqs.append({
            "position": i,
            "seq_201nt": context_201,
            "trinucleotide": trinuc,
            "dinucleotide_5p": dinuc_5p,
            "is_tc": dinuc_5p in ["TC", "tC"],
            "is_cc": dinuc_5p in ["CC", "cC"],
        })

    return test_seqs


def generate_synthetic_apobec_targets(n_per_context=200):
    """Generate synthetic 201-nt sequences with known APOBEC target/non-target contexts.

    Creates sequences with controlled dinucleotide context at center to test
    whether our model correctly identifies APOBEC-preferred contexts.
    """
    rng = np.random.RandomState(SEED)
    bases = list("ACGT")
    seqs = []

    contexts = {
        "TC": ("T", "C"),  # A3A/A3B preferred (human + HBV)
        "CC": ("C", "C"),  # A3G preferred (HIV + HBV)
        "AC": ("A", "C"),  # Non-preferred
        "GC": ("G", "C"),  # Non-preferred
    }

    for ctx_name, (pre_base, _) in contexts.items():
        for i in range(n_per_context):
            # Random flanking sequence
            left = "".join(rng.choice(bases, 99))
            right = "".join(rng.choice(bases, 100))
            seq = left + pre_base + "C" + right
            assert len(seq) == 201 and seq[100] == "C"

            seqs.append({
                "context": ctx_name,
                "seq_201nt": seq,
                "is_tc": ctx_name == "TC",
                "is_cc": ctx_name == "CC",
                "is_apobec_target": ctx_name in ["TC", "CC"],
            })

    return seqs


def train_gb_model():
    """Train GB_HandFeatures model on multi-enzyme v3 data (motif-only 24-dim for speed)."""
    from xgboost import XGBClassifier

    logger.info("Training GB model...")
    splits = pd.read_csv(MULTI_SPLITS)
    with open(MULTI_SEQS) as f:
        sequences = json.load(f)

    site_ids = splits["site_id"].astype(str).tolist()
    labels = splits["is_edited"].values

    # Motif features only (24-dim) — sufficient for context analysis
    motif_feats = np.array([extract_motif_from_seq(sequences.get(sid, "N" * 201))
                            for sid in site_ids], dtype=np.float32)

    # Also train full 40-dim if structure cache available
    loop_df = pd.read_csv(LOOP_CSV)
    loop_df["site_id"] = loop_df["site_id"].astype(str)
    loop_df = loop_df.set_index("site_id")
    loop_feats = np.zeros((len(site_ids), 9), dtype=np.float32)
    for i, sid in enumerate(site_ids):
        if sid in loop_df.index:
            for j, col in enumerate(LOOP_FEATURE_COLS):
                if col in loop_df.columns:
                    val = loop_df.loc[sid, col]
                    if hasattr(val, 'iloc'):
                        val = val.iloc[0]
                    loop_feats[i, j] = float(val) if pd.notna(val) else 0.0

    struct_feats = np.zeros((len(site_ids), 7), dtype=np.float32)
    if STRUCT_CACHE.exists():
        data = np.load(str(STRUCT_CACHE), allow_pickle=True)
        sid_map = {str(s): i for i, s in enumerate(data["site_ids"])}
        for i, sid in enumerate(site_ids):
            if sid in sid_map:
                struct_feats[i] = data["delta_features"][sid_map[sid]]
        del data
        gc.collect()

    X_motif = motif_feats
    X_full = np.nan_to_num(np.concatenate([motif_feats, struct_feats, loop_feats], axis=1), nan=0.0)

    xgb_params = dict(n_estimators=500, max_depth=6, learning_rate=0.1,
                      subsample=0.8, colsample_bytree=0.8, random_state=SEED,
                      eval_metric="logloss", use_label_encoder=False)

    motif_model = XGBClassifier(**xgb_params)
    motif_model.fit(X_motif, labels)

    full_model = XGBClassifier(**xgb_params)
    full_model.fit(X_full, labels)

    from sklearn.metrics import roc_auc_score
    logger.info(f"  MotifOnly AUC: {roc_auc_score(labels, motif_model.predict_proba(X_motif)[:, 1]):.4f}")
    logger.info(f"  Full 40-dim AUC: {roc_auc_score(labels, full_model.predict_proba(X_full)[:, 1]):.4f}")

    return motif_model, full_model


def score_sequences(seqs_201, motif_model):
    """Score a list of 201-nt sequences with the motif-only model."""
    feats = np.array([extract_motif_from_seq(s) for s in seqs_201], dtype=np.float32)
    scores = motif_model.predict_proba(feats)[:, 1]
    return scores


def main():
    t_start = time.time()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ================================================================
    # Part 1: Document available data sources
    # ================================================================
    logger.info("=" * 60)
    logger.info("Part 1: Documenting viral APOBEC data sources")

    data_sources = {
        "hiv_databases": {
            "name": "HIV Sequence Database (Los Alamos)",
            "url": "https://www.hiv.lanl.gov/",
            "tools": {
                "Hypermut": "https://www.hiv.lanl.gov/content/sequence/HYPERMUT/hypermut.html",
                "Hypermut2": "Identifies APOBEC3G (GG) and APOBEC3F (GA) contexts",
                "Hypermut3": "Multi-state character analysis for mixed editing",
            },
            "available_data": [
                "2829 nominally normal and 88 nominally hypermutated full-length HIV-1 genomes",
                "Env gene sequences with G->A annotation",
                "Subtype-specific alignments with reference sequences",
            ],
            "access_notes": "Interactive web interface; bulk download requires alignment search. "
                           "Hypermut tool analyzes user-uploaded sequences against a reference.",
        },
        "published_hiv_studies": [
            {
                "study": "Sadler et al. 2010",
                "journal": "J Virol 84:7923-7931",
                "data": "454 pyrosequencing of HIV-1 env, >140,000 sequences",
                "finding": "1.3-6.5% of G mutated to A, predominantly GG context (A3G)",
            },
            {
                "study": "Suspene et al. 2004",
                "journal": "Nucleic Acids Res 32:2421-2429",
                "data": "Twin gradients in A3G-edited HIV DNA",
                "finding": "3'->5' processivity on minus strand, gradient of editing",
            },
            {
                "study": "Hache et al. 2008",
                "journal": "PLoS Genet 4:e1000028",
                "data": "2917 full-length HIV-1 genomes from LANL database",
                "finding": "88 hypermutated genomes; all-or-nothing editing phenomenon",
            },
        ],
        "published_hbv_studies": [
            {
                "study": "Vartanian et al. 2010",
                "journal": "PLoS Pathog 6:e1000928",
                "data": "Massive A3 editing of HBV DNA in cirrhosis patients",
                "finding": "A3G dominant in vivo, up to 35% of HBV genomes edited",
            },
            {
                "study": "Suspene et al. 2005",
                "journal": "PNAS 102:8321-8326",
                "data": "Extensive editing of HBV DNA by A3B, A3C, A3F, A3G",
                "finding": "A3B highest efficiency (65% of cytidines), TC context",
            },
            {
                "study": "Liang et al. 2021",
                "journal": "J Biol Chem 296:100878",
                "data": "APOBEC3-induced HBV mutations during RT",
                "finding": "Editing occurs during RNA->(-)-DNA reverse transcription",
            },
        ],
    }

    with open(OUTPUT_DIR / "data_sources.json", "w") as f:
        json.dump(data_sources, f, indent=2)

    # ================================================================
    # Part 2: Document published motif preferences
    # ================================================================
    logger.info("=" * 60)
    logger.info("Part 2: Documenting published motif preferences")

    with open(OUTPUT_DIR / "viral_apobec_motif_data.json", "w") as f:
        json.dump(VIRAL_APOBEC_DATA, f, indent=2)

    # ================================================================
    # Part 3: Train model and score viral sequences
    # ================================================================
    logger.info("=" * 60)
    logger.info("Part 3: Training model and scoring viral sequences")

    motif_model, full_model = train_gb_model()

    # --- Score HIV env segment ---
    logger.info("\nScoring HIV-1 env segment...")
    hiv_targets = generate_test_sequences_from_viral(HIV1_ENV_SEGMENT)
    logger.info(f"  {len(hiv_targets)} C positions in HIV env segment")

    hiv_scores = score_sequences([t["seq_201nt"] for t in hiv_targets], motif_model)
    for i, t in enumerate(hiv_targets):
        t["score"] = float(hiv_scores[i])

    hiv_df = pd.DataFrame(hiv_targets)
    hiv_df.to_csv(OUTPUT_DIR / "hiv_env_scores.csv", index=False)

    # --- Score HBV preC segment ---
    logger.info("\nScoring HBV preC/C segment...")
    hbv_targets = generate_test_sequences_from_viral(HBV_PREC_SEGMENT)
    logger.info(f"  {len(hbv_targets)} C positions in HBV preC segment")

    hbv_scores = score_sequences([t["seq_201nt"] for t in hbv_targets], motif_model)
    for i, t in enumerate(hbv_targets):
        t["score"] = float(hbv_scores[i])

    hbv_df = pd.DataFrame(hbv_targets)
    hbv_df.to_csv(OUTPUT_DIR / "hbv_prec_scores.csv", index=False)

    # --- Score synthetic sequences ---
    logger.info("\nScoring synthetic APOBEC target sequences...")
    synthetic = generate_synthetic_apobec_targets(n_per_context=500)
    syn_scores = score_sequences([s["seq_201nt"] for s in synthetic], motif_model)
    for i, s in enumerate(synthetic):
        s["score"] = float(syn_scores[i])

    syn_df = pd.DataFrame(synthetic)
    syn_df.to_csv(OUTPUT_DIR / "synthetic_context_scores.csv", index=False)

    # ================================================================
    # Part 4: Analysis and comparison
    # ================================================================
    logger.info("=" * 60)
    logger.info("Part 4: Analysis and comparison")

    results = {}

    # --- HIV analysis ---
    if len(hiv_df) > 0:
        hiv_tc = hiv_df[hiv_df["is_tc"]]
        hiv_cc = hiv_df[hiv_df["is_cc"]]
        hiv_other = hiv_df[~hiv_df["is_tc"] & ~hiv_df["is_cc"]]

        results["hiv_env"] = {
            "n_total_c": len(hiv_df),
            "n_tc": len(hiv_tc),
            "n_cc": len(hiv_cc),
            "n_other": len(hiv_other),
            "mean_score_all": round(float(hiv_df["score"].mean()), 4),
            "mean_score_tc": round(float(hiv_tc["score"].mean()), 4) if len(hiv_tc) > 0 else None,
            "mean_score_cc": round(float(hiv_cc["score"].mean()), 4) if len(hiv_cc) > 0 else None,
            "mean_score_other": round(float(hiv_other["score"].mean()), 4) if len(hiv_other) > 0 else None,
            "interpretation": (
                "HIV A3G edits CC context on minus strand. Our model (trained on human RNA editing) "
                "should give higher scores to TC (A3A/A3B target) and CC (A3G target) contexts. "
                "If CC positions score higher than other non-TC contexts, it suggests the model "
                "captures some A3G-relevant features beyond just TC motif."
            ),
        }

        if len(hiv_tc) > 0 and len(hiv_other) > 0:
            _, p = stats.mannwhitneyu(hiv_tc["score"], hiv_other["score"], alternative="greater")
            results["hiv_env"]["tc_vs_other_p"] = float(p)
        if len(hiv_cc) > 0 and len(hiv_other) > 0:
            _, p = stats.mannwhitneyu(hiv_cc["score"], hiv_other["score"], alternative="greater")
            results["hiv_env"]["cc_vs_other_p"] = float(p)

        logger.info(f"  HIV env: TC={results['hiv_env']['mean_score_tc']}, "
                     f"CC={results['hiv_env']['mean_score_cc']}, "
                     f"Other={results['hiv_env']['mean_score_other']}")

    # --- HBV analysis ---
    if len(hbv_df) > 0:
        hbv_tc = hbv_df[hbv_df["is_tc"]]
        hbv_cc = hbv_df[hbv_df["is_cc"]]
        hbv_other = hbv_df[~hbv_df["is_tc"] & ~hbv_df["is_cc"]]

        results["hbv_prec"] = {
            "n_total_c": len(hbv_df),
            "n_tc": len(hbv_tc),
            "n_cc": len(hbv_cc),
            "n_other": len(hbv_other),
            "mean_score_all": round(float(hbv_df["score"].mean()), 4),
            "mean_score_tc": round(float(hbv_tc["score"].mean()), 4) if len(hbv_tc) > 0 else None,
            "mean_score_cc": round(float(hbv_cc["score"].mean()), 4) if len(hbv_cc) > 0 else None,
            "mean_score_other": round(float(hbv_other["score"].mean()), 4) if len(hbv_other) > 0 else None,
            "interpretation": (
                "HBV is edited by both A3B (TC context, dominant in vitro) and A3G (CC context, "
                "dominant in vivo). Our model should rank TC > CC > AC/GC for editability."
            ),
        }

        if len(hbv_tc) > 0 and len(hbv_other) > 0:
            _, p = stats.mannwhitneyu(hbv_tc["score"], hbv_other["score"], alternative="greater")
            results["hbv_prec"]["tc_vs_other_p"] = float(p)

        logger.info(f"  HBV preC: TC={results['hbv_prec']['mean_score_tc']}, "
                     f"CC={results['hbv_prec']['mean_score_cc']}, "
                     f"Other={results['hbv_prec']['mean_score_other']}")

    # --- Synthetic context analysis ---
    results["synthetic"] = {}
    for ctx in ["TC", "CC", "AC", "GC"]:
        ctx_df = syn_df[syn_df["context"] == ctx]
        results["synthetic"][ctx] = {
            "n": len(ctx_df),
            "mean_score": round(float(ctx_df["score"].mean()), 4),
            "std_score": round(float(ctx_df["score"].std()), 4),
            "median_score": round(float(ctx_df["score"].median()), 4),
        }

    # Pairwise comparisons
    tc_scores = syn_df[syn_df["context"] == "TC"]["score"].values
    cc_scores = syn_df[syn_df["context"] == "CC"]["score"].values
    ac_scores = syn_df[syn_df["context"] == "AC"]["score"].values
    gc_scores = syn_df[syn_df["context"] == "GC"]["score"].values

    _, p_tc_ac = stats.mannwhitneyu(tc_scores, ac_scores, alternative="greater")
    _, p_cc_ac = stats.mannwhitneyu(cc_scores, ac_scores, alternative="greater")
    _, p_tc_cc = stats.mannwhitneyu(tc_scores, cc_scores, alternative="greater")
    _, p_tc_gc = stats.mannwhitneyu(tc_scores, gc_scores, alternative="greater")
    _, p_cc_gc = stats.mannwhitneyu(cc_scores, gc_scores, alternative="greater")

    results["synthetic"]["comparisons"] = {
        "TC_vs_AC": {"p": float(p_tc_ac), "direction": "TC > AC"},
        "CC_vs_AC": {"p": float(p_cc_ac), "direction": "CC > AC"},
        "TC_vs_CC": {"p": float(p_tc_cc), "direction": "TC > CC"},
        "TC_vs_GC": {"p": float(p_tc_gc), "direction": "TC > GC"},
        "CC_vs_GC": {"p": float(p_cc_gc), "direction": "CC > GC"},
    }

    results["synthetic"]["interpretation"] = (
        "Expected ranking: TC > CC >> AC ~ GC. "
        "TC is the primary A3A/A3B target (human RNA + HBV). "
        "CC is the A3G target (HIV + HBV in vivo). "
        "If TC >> CC, the model primarily captures A3A/A3B-like editing. "
        "If CC is also elevated, the model generalizes to A3G-like contexts."
    )

    logger.info(f"\n  Synthetic context scores:")
    for ctx in ["TC", "CC", "AC", "GC"]:
        r = results["synthetic"][ctx]
        logger.info(f"    {ctx}: mean={r['mean_score']:.4f} +/- {r['std_score']:.4f}")

    # --- Cross-validation with published findings ---
    results["cross_validation"] = {
        "hiv_a3g_prediction": {
            "published_target": "CC dinucleotide on minus strand (GG on plus strand)",
            "model_score_for_cc": results["synthetic"]["CC"]["mean_score"],
            "model_score_for_nontarget": (results["synthetic"]["AC"]["mean_score"] +
                                           results["synthetic"]["GC"]["mean_score"]) / 2,
            "model_correctly_identifies_a3g_target": (
                results["synthetic"]["CC"]["mean_score"] >
                (results["synthetic"]["AC"]["mean_score"] +
                 results["synthetic"]["GC"]["mean_score"]) / 2
            ),
        },
        "hiv_a3f_prediction": {
            "published_target": "TC dinucleotide (GA on plus strand)",
            "model_score_for_tc": results["synthetic"]["TC"]["mean_score"],
            "note": "A3F target (TC) matches A3A/A3B target — our model should score these highest",
        },
        "hbv_prediction": {
            "published_targets": "A3B: TC (in vitro), A3G: CC (in vivo)",
            "model_ranking_matches": (
                results["synthetic"]["TC"]["mean_score"] >
                results["synthetic"]["CC"]["mean_score"] >
                results["synthetic"]["AC"]["mean_score"]
            ),
            "note": "Model trained on human A3A sites should rank TC > CC > AC/GC",
        },
    }

    # Save results
    with open(OUTPUT_DIR / "viral_apobec_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    # ================================================================
    # Part 5: Figures
    # ================================================================
    logger.info("\n" + "=" * 60)
    logger.info("Part 5: Generating figures")

    # Figure 1: Synthetic context scores — violin plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Panel 1: Synthetic dinucleotide context scores
    ax = axes[0]
    ctx_data = []
    ctx_labels = []
    ctx_colors = []
    for ctx, color in [("TC", "#dc2626"), ("CC", "#f59e0b"), ("AC", "#94a3b8"), ("GC", "#6b7280")]:
        ctx_data.append(syn_df[syn_df["context"] == ctx]["score"].values)
        ctx_labels.append(ctx)
        ctx_colors.append(color)

    parts = ax.violinplot(ctx_data, positions=[1, 2, 3, 4], showmeans=True, showmedians=True)
    for i, (body, color) in enumerate(zip(parts["bodies"], ctx_colors)):
        body.set_facecolor(color)
        body.set_alpha(0.7)

    ax.set_xticks([1, 2, 3, 4])
    ax.set_xticklabels(ctx_labels)
    ax.set_ylabel("Editability Score")
    ax.set_title("Model Scores by Dinucleotide Context\n(Synthetic sequences)")
    ax.axhline(0.5, color="gray", linestyle=":", alpha=0.5)

    # Annotate with means
    for i, ctx in enumerate(ctx_labels):
        mean_val = results["synthetic"][ctx]["mean_score"]
        ax.text(i + 1, ax.get_ylim()[1] * 0.95, f"mean={mean_val:.3f}",
                ha="center", fontsize=8, style="italic")

    # Panel 2: HIV env scores by context
    ax = axes[1]
    if len(hiv_df) > 0:
        for ctx_label, mask, color in [
            ("TC", hiv_df["is_tc"], "#dc2626"),
            ("CC", hiv_df["is_cc"], "#f59e0b"),
            ("Other", ~hiv_df["is_tc"] & ~hiv_df["is_cc"], "#94a3b8"),
        ]:
            subset = hiv_df[mask]["score"]
            if len(subset) > 0:
                ax.hist(subset, bins=20, alpha=0.5, label=f"{ctx_label} (n={len(subset)})",
                        color=color, density=True)
        ax.set_xlabel("Editability Score")
        ax.set_ylabel("Density")
        ax.set_title("HIV-1 env: Score Distribution by Context")
        ax.legend(fontsize=8)

    # Panel 3: HBV scores by context
    ax = axes[2]
    if len(hbv_df) > 0:
        for ctx_label, mask, color in [
            ("TC", hbv_df["is_tc"], "#dc2626"),
            ("CC", hbv_df["is_cc"], "#f59e0b"),
            ("Other", ~hbv_df["is_tc"] & ~hbv_df["is_cc"], "#94a3b8"),
        ]:
            subset = hbv_df[mask]["score"]
            if len(subset) > 0:
                ax.hist(subset, bins=20, alpha=0.5, label=f"{ctx_label} (n={len(subset)})",
                        color=color, density=True)
        ax.set_xlabel("Editability Score")
        ax.set_ylabel("Density")
        ax.set_title("HBV preC: Score Distribution by Context")
        ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "viral_apobec_scores.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Figure 2: Summary — comparison of viral vs human APOBEC contexts
    fig, ax = plt.subplots(figsize=(10, 6))

    categories = []
    means = []
    colors = []
    yerr = []

    # Synthetic baselines
    for ctx, color in [("TC", "#dc2626"), ("CC", "#f59e0b"), ("AC", "#94a3b8"), ("GC", "#6b7280")]:
        categories.append(f"Synthetic\n{ctx}")
        means.append(results["synthetic"][ctx]["mean_score"])
        yerr.append(results["synthetic"][ctx]["std_score"])
        colors.append(color)

    # HIV scores
    if "hiv_env" in results:
        for label, key, color in [
            ("HIV env\nTC", "mean_score_tc", "#dc2626"),
            ("HIV env\nCC", "mean_score_cc", "#f59e0b"),
            ("HIV env\nOther", "mean_score_other", "#94a3b8"),
        ]:
            val = results["hiv_env"].get(key)
            if val is not None:
                categories.append(label)
                means.append(val)
                yerr.append(0)
                colors.append(color)

    # HBV scores
    if "hbv_prec" in results:
        for label, key, color in [
            ("HBV preC\nTC", "mean_score_tc", "#dc2626"),
            ("HBV preC\nCC", "mean_score_cc", "#f59e0b"),
            ("HBV preC\nOther", "mean_score_other", "#94a3b8"),
        ]:
            val = results["hbv_prec"].get(key)
            if val is not None:
                categories.append(label)
                means.append(val)
                yerr.append(0)
                colors.append(color)

    x = np.arange(len(categories))
    bars = ax.bar(x, means, color=colors, alpha=0.8, edgecolor="black", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=8, rotation=45, ha="right")
    ax.set_ylabel("Mean Editability Score")
    ax.set_title("APOBEC Editability: Human Model Applied to Viral Sequences")
    ax.axhline(0.5, color="gray", linestyle=":", alpha=0.5, label="Neutral threshold")

    # Add value labels
    for bar, val in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{val:.3f}", ha="center", fontsize=7)

    # Add vertical separators
    ax.axvline(3.5, color="black", linestyle="--", alpha=0.3)
    if "hiv_env" in results:
        hiv_end = 3 + sum(1 for k in ["mean_score_tc", "mean_score_cc", "mean_score_other"]
                          if results["hiv_env"].get(k) is not None)
        ax.axvline(hiv_end + 0.5, color="black", linestyle="--", alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "viral_vs_human_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()

    # ================================================================
    # Summary
    # ================================================================
    elapsed = time.time() - t_start
    logger.info(f"\n{'='*60}")
    logger.info(f"COMPLETE ({elapsed / 60:.1f} min)")
    logger.info(f"Results: {OUTPUT_DIR}")

    logger.info(f"\n{'='*60}")
    logger.info("KEY FINDINGS:")
    logger.info(f"  Synthetic TC (A3A/A3B/A3F target): mean score = "
                f"{results['synthetic']['TC']['mean_score']:.4f}")
    logger.info(f"  Synthetic CC (A3G target):          mean score = "
                f"{results['synthetic']['CC']['mean_score']:.4f}")
    logger.info(f"  Synthetic AC (non-target):           mean score = "
                f"{results['synthetic']['AC']['mean_score']:.4f}")
    logger.info(f"  Synthetic GC (non-target):           mean score = "
                f"{results['synthetic']['GC']['mean_score']:.4f}")

    logger.info(f"\n  TC vs AC: p = {results['synthetic']['comparisons']['TC_vs_AC']['p']:.2e}")
    logger.info(f"  CC vs AC: p = {results['synthetic']['comparisons']['CC_vs_AC']['p']:.2e}")
    logger.info(f"  TC vs CC: p = {results['synthetic']['comparisons']['TC_vs_CC']['p']:.2e}")

    if "hiv_env" in results:
        logger.info(f"\n  HIV env segment: {results['hiv_env']['n_total_c']} C positions scored")
    if "hbv_prec" in results:
        logger.info(f"  HBV preC segment: {results['hbv_prec']['n_total_c']} C positions scored")

    logger.info(f"\n  Cross-validation with published findings:")
    cv = results.get("cross_validation", {})
    if "hiv_a3g_prediction" in cv:
        logger.info(f"    Model correctly identifies A3G target (CC): "
                     f"{cv['hiv_a3g_prediction']['model_correctly_identifies_a3g_target']}")
    if "hbv_prediction" in cv:
        logger.info(f"    Model ranking matches HBV biology (TC > CC > AC): "
                     f"{cv['hbv_prediction']['model_ranking_matches']}")


if __name__ == "__main__":
    main()
