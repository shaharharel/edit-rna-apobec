#!/usr/bin/env python3
"""Three follow-up tests on 'Neither = APOBEC1':

  1. Intestine-subset test. The 206 Neither sites split by tissue:
       Intestine Specific (63)
       Ubiquitous (46)
       Testis Specific (44)
       Non-Specific (37)
       Blood Specific (16)
     Score both heads (APOBEC1 mouse-trained, Neither adapter) on each subset.
     If APOBEC1 biology is tissue-localized, APOBEC1 head should rank
     intestine-specific Neither sites above the rest.

  2. Human APOBEC1 known-target check. Literature targets:
       APOB, GLUD2, NF1, GLI1, CYFIP2, BLCAP, COG3, SERPINA1, HDAC9, HDGF
     Identify which appear in our Neither set. Score them + rank among the 206
     Neither sites under each head.

  3. Mouse-Neither overlap. Map mouse APOBEC1 training positives to human via
     mm9 -> hg38 liftover AND gene-symbol overlap (mouse sentence-case ->
     human upper-case). Count positional + gene-level overlap with 206 Neither.

Inputs:
  data/processed/multi_enzyme/levanon_all_categories.csv   (636 Levanon sites)
  data/processed/multi_enzyme/splits_multi_enzyme_v3_with_negatives.csv
  data/processed/multi_enzyme/multi_enzyme_sequences_v3_with_negatives.json
  data/processed/multi_enzyme/loop_position_per_site_v3.csv
  data/processed/multi_enzyme/structure_cache_multi_enzyme_v3.npz
  data/processed/multi_enzyme/embeddings/rnafm_pooled_v3.pt
  data/processed/multi_enzyme/embeddings/rnafm_pooled_edited_v3.pt
  data/processed/apobec1/apobec1_positives_raw.csv         (484 mouse positives)
  experiments/multi_enzyme/outputs/phase3_neural_true_validation/phase3_neural_full.pt
  experiments/multi_enzyme/outputs/apobec1_head/apobec1_head.pt

Output: experiments/multi_enzyme/outputs/apobec1_head/subset_tests.json
        experiments/multi_enzyme/outputs/apobec1_head/subset_tests.log
"""
from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy import stats
from sklearn.metrics import roc_auc_score

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
from src.data.apobec_feature_extraction import build_hand_features  # noqa

OUT_DIR = PROJECT_ROOT / "experiments" / "multi_enzyme" / "outputs" / "apobec1_head"
OUT_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = OUT_DIR / "subset_tests.log"
RESULTS_JSON = OUT_DIR / "subset_tests.json"

sys.stdout.reconfigure(line_buffering=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(LOG_FILE, mode="w")],
)
logger = logging.getLogger(__name__)

DATA_DIR = PROJECT_ROOT / "data"
ME_DIR = DATA_DIR / "processed" / "multi_enzyme"
LEVANON_CAT = ME_DIR / "levanon_all_categories.csv"
V3_SPLITS = ME_DIR / "splits_multi_enzyme_v3_with_negatives.csv"
V3_SEQS = ME_DIR / "multi_enzyme_sequences_v3_with_negatives.json"
V3_LOOP = ME_DIR / "loop_position_per_site_v3.csv"
V3_STRUCT = ME_DIR / "structure_cache_multi_enzyme_v3.npz"
V3_RNAFM_ORIG = ME_DIR / "embeddings" / "rnafm_pooled_v3.pt"
V3_RNAFM_EDITED = ME_DIR / "embeddings" / "rnafm_pooled_edited_v3.pt"
MOUSE_RAW = DATA_DIR / "processed" / "apobec1" / "apobec1_positives_raw.csv"

PHASE3_CKPT = PROJECT_ROOT / "experiments" / "multi_enzyme" / "outputs" / "phase3_neural_true_validation" / "phase3_neural_full.pt"
APOBEC1_CKPT = OUT_DIR / "apobec1_head.pt"

KNOWN_APOBEC1_TARGETS = [
    "APOB", "GLUD2", "NF1", "GLI1", "CYFIP2", "BLCAP", "COG3",
    "SERPINA1", "HDAC9", "HDGF", "A1CF", "APOBEC1",
    # Additional hits from Rosenberg 2011 / Blanc 2014 that have human orthologs
    "NEAT1", "MALAT1", "TMEM65", "CDK5R2", "MGP",
]

D_INPUT = 1320
D_SHARED = 128
ENZYMES = ["A3A", "A3B", "A3G", "A3A_A3G", "Neither"]
N_ENZYMES_CLS = 6
DEVICE = (
    torch.device("cuda") if torch.cuda.is_available()
    else torch.device("mps") if hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    else torch.device("cpu")
)
logger.info("Device: %s", DEVICE)


class Phase3Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.shared_encoder = nn.Sequential(
            nn.Linear(D_INPUT, 256), nn.GELU(), nn.Dropout(0.3), nn.LayerNorm(256),
            nn.Linear(256, D_SHARED), nn.GELU(), nn.Dropout(0.2),
        )
        self.binary_head = nn.Linear(D_SHARED, 1)
        self.enzyme_adapters = nn.ModuleDict({
            enz: nn.Sequential(nn.Linear(D_SHARED, 32), nn.GELU(), nn.Linear(32, 1))
            for enz in ENZYMES
        })
        self.enzyme_classifier = nn.Sequential(
            nn.Linear(D_SHARED, 64), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(64, N_ENZYMES_CLS),
        )


class APOBEC1Head(nn.Module):
    def __init__(self, d_shared: int = 128):
        super().__init__()
        self.species_proj = nn.Sequential(
            nn.Linear(1, 16), nn.GELU(), nn.Linear(16, d_shared),
        )
        self.head = nn.Sequential(
            nn.Linear(d_shared, 32), nn.GELU(), nn.Linear(32, 1),
        )

    def forward(self, shared, species):
        bias = self.species_proj(species)
        return self.head(shared + bias).squeeze(-1)


def load_neither_features(include_negatives: bool = True):
    """Load Neither positives (+ motif-matched negatives) with aligned features."""
    logger.info("Loading Neither features from v3 dataset...")
    splits = pd.read_csv(V3_SPLITS)
    nei = splits[splits["enzyme"] == "Neither"].copy()
    if not include_negatives:
        nei = nei[nei["is_edited"] == 1]
    site_ids = nei["site_id"].tolist()

    with open(V3_SEQS) as f:
        seqs_all = json.load(f)
    loop_df = pd.read_csv(V3_LOOP).drop_duplicates(subset=["site_id"]).set_index("site_id")
    sc = np.load(V3_STRUCT, allow_pickle=True)
    struct_map = {sid: sc["delta_features"][i] for i, sid in enumerate(sc["site_ids"])}

    have_seq = [sid for sid in site_ids if sid in seqs_all]
    if len(have_seq) < len(site_ids):
        logger.warning("  %d/%d Neither sites missing sequences, dropping",
                       len(site_ids) - len(have_seq), len(site_ids))
    nei = nei[nei["site_id"].isin(set(have_seq))].reset_index(drop=True)
    site_ids = nei["site_id"].tolist()
    n = len(site_ids)

    hand = build_hand_features(site_ids, seqs_all, struct_map, loop_df)
    rnafm_o = torch.load(V3_RNAFM_ORIG, weights_only=False, map_location="cpu")
    rnafm_e = torch.load(V3_RNAFM_EDITED, weights_only=False, map_location="cpu")
    emb_o = np.zeros((n, 640), dtype=np.float32)
    emb_d = np.zeros((n, 640), dtype=np.float32)
    for i, sid in enumerate(site_ids):
        o = rnafm_o.get(sid); e = rnafm_e.get(sid)
        if o is None or e is None:
            continue
        if isinstance(o, torch.Tensor): o = o.numpy()
        if isinstance(e, torch.Tensor): e = e.numpy()
        emb_o[i] = o
        emb_d[i] = e - o

    X = np.concatenate([emb_o, emb_d, hand], axis=1).astype(np.float32)
    return {
        "X": X,
        "site_ids": site_ids,
        "is_edited": nei["is_edited"].values.astype(int),
        "splits": nei,
    }


def score_both_heads(X: np.ndarray, model: Phase3Model, head: APOBEC1Head, species_val: float = 0.0):
    model.eval(); head.eval()
    apobec1, neither = [], []
    BATCH = 256
    with torch.no_grad():
        for b in range(0, len(X), BATCH):
            xb = torch.from_numpy(X[b:b + BATCH]).float().to(DEVICE)
            shared = model.shared_encoder(xb)
            sb = torch.full((xb.shape[0], 1), species_val, dtype=torch.float32, device=DEVICE)
            apobec1.append(torch.sigmoid(head(shared, sb)).cpu().numpy())
            neither.append(torch.sigmoid(model.enzyme_adapters["Neither"](shared).squeeze(-1)).cpu().numpy())
    return np.concatenate(apobec1), np.concatenate(neither)


# =============================================================================
# Experiment 1 — intestine-subset test
# =============================================================================
def exp1_intestine_subset(scored_pos: pd.DataFrame, scored_neg: pd.DataFrame | None):
    """scored_pos columns: site_id, is_edited, apobec1, neither, tissue_classification (+ others).
    We compute per-tissue mean score under each head, and test whether
    APOBEC1 head is significantly higher for intestine-specific positives
    than for the rest.
    """
    logger.info("\n" + "=" * 70)
    logger.info("Experiment 1 — intestine-subset test (Neither positives)")
    logger.info("=" * 70)

    tissues = sorted(scored_pos["tissue_classification"].dropna().unique())
    per_tissue = []
    for t in tissues:
        sub = scored_pos[scored_pos["tissue_classification"] == t]
        per_tissue.append({
            "tissue": str(t),
            "n": int(len(sub)),
            "apobec1_mean": float(sub["apobec1"].mean()),
            "apobec1_median": float(sub["apobec1"].median()),
            "neither_mean": float(sub["neither"].mean()),
            "neither_median": float(sub["neither"].median()),
        })

    # Intestine vs rest for APOBEC1 head + Neither head
    intestine = scored_pos[scored_pos["tissue_classification"] == "Intestine Specific"]
    other = scored_pos[scored_pos["tissue_classification"] != "Intestine Specific"]
    tests = {}
    for head_label in ("apobec1", "neither"):
        u, p = stats.mannwhitneyu(intestine[head_label], other[head_label], alternative="greater")
        tests[head_label] = {
            "intestine_n": int(len(intestine)),
            "intestine_mean": float(intestine[head_label].mean()),
            "other_n": int(len(other)),
            "other_mean": float(other[head_label].mean()),
            "mann_whitney_u": float(u),
            "p_one_sided_intestine_gt_other": float(p),
            "diff_mean": float(intestine[head_label].mean() - other[head_label].mean()),
        }
        logger.info(
            "  %s:  intestine(n=%d) mean=%.4f vs other(n=%d) mean=%.4f   Δ=%+.4f   p=%.3e",
            head_label, len(intestine), intestine[head_label].mean(),
            len(other), other[head_label].mean(),
            intestine[head_label].mean() - other[head_label].mean(), p,
        )

    # Also AUROC when positives/negatives available
    within_subset_auroc = {}
    if scored_neg is not None:
        logger.info("\n  AUROC by tissue subset (positives of that tissue vs ALL negatives):")
        for t in tissues:
            sub_pos = scored_pos[scored_pos["tissue_classification"] == t]
            if len(sub_pos) < 10:
                continue
            y = np.concatenate([np.ones(len(sub_pos)), np.zeros(len(scored_neg))])
            row = {"tissue": str(t), "n_pos": int(len(sub_pos)), "n_neg": int(len(scored_neg))}
            for head_label in ("apobec1", "neither"):
                scores = np.concatenate([sub_pos[head_label].values, scored_neg[head_label].values])
                try:
                    row[f"auroc_{head_label}"] = float(roc_auc_score(y, scores))
                except Exception:
                    row[f"auroc_{head_label}"] = float("nan")
            within_subset_auroc[t] = row
            logger.info(
                "    %-20s n_pos=%3d  APOBEC1 AUROC=%.3f   Neither AUROC=%.3f",
                t, len(sub_pos), row["auroc_apobec1"], row["auroc_neither"],
            )

    return {
        "per_tissue_summary": per_tissue,
        "intestine_vs_other_tests": tests,
        "within_subset_auroc": within_subset_auroc,
    }


# =============================================================================
# Experiment 2 — known human APOBEC1 target check
# =============================================================================
def exp2_known_targets(scored_pos: pd.DataFrame):
    logger.info("\n" + "=" * 70)
    logger.info("Experiment 2 — known human APOBEC1 targets")
    logger.info("=" * 70)
    logger.info("  Looking for: %s", ", ".join(KNOWN_APOBEC1_TARGETS))

    # Rank the 206 positives under each head
    scored = scored_pos.copy()
    scored["rank_apobec1"] = scored["apobec1"].rank(ascending=False, method="min")
    scored["rank_neither"] = scored["neither"].rank(ascending=False, method="min")
    n_total = len(scored)

    target_rows = []
    for g in KNOWN_APOBEC1_TARGETS:
        hits = scored[scored["gene_refseq"] == g]
        for _, r in hits.iterrows():
            target_rows.append({
                "gene": g,
                "site_id": r["site_id"],
                "apobec1_score": float(r["apobec1"]),
                "apobec1_rank": int(r["rank_apobec1"]),
                "apobec1_percentile_top": float(r["rank_apobec1"] / n_total),
                "neither_score": float(r["neither"]),
                "neither_rank": int(r["rank_neither"]),
                "neither_percentile_top": float(r["rank_neither"] / n_total),
                "tissue": str(r.get("tissue_classification", "?")),
                "exonic_function": str(r.get("exonic_function", "?")),
                "mean_rate": float(r.get("mean_gtex_editing_rate", float("nan"))),
            })
    for row in target_rows:
        logger.info(
            "  %-12s %-18s APOBEC1 rank %4d/%d (top %.1f%%, score %.3f)   "
            "Neither rank %4d/%d (top %.1f%%, score %.3f)",
            row["gene"], row["site_id"],
            row["apobec1_rank"], n_total, row["apobec1_percentile_top"] * 100, row["apobec1_score"],
            row["neither_rank"], n_total, row["neither_percentile_top"] * 100, row["neither_score"],
        )

    # Aggregate stat: does APOBEC1 head rank known targets better than random?
    if target_rows:
        target_ap_ranks = np.array([r["apobec1_rank"] for r in target_rows])
        target_ne_ranks = np.array([r["neither_rank"] for r in target_rows])
        mean_ap_pct = float(target_ap_ranks.mean() / n_total)
        mean_ne_pct = float(target_ne_ranks.mean() / n_total)
        # Expected mean percentile under uniform = 0.5
        # Wilcoxon against 0.5 * n_total for sanity
        logger.info(
            "\n  %d known-target sites found:  APOBEC1 mean percentile %.2f  "
            "Neither mean percentile %.2f  (random=0.50)",
            len(target_rows), mean_ap_pct, mean_ne_pct,
        )
    else:
        mean_ap_pct = mean_ne_pct = float("nan")

    return {
        "n_target_sites_in_neither": len(target_rows),
        "target_sites": target_rows,
        "mean_apobec1_percentile": mean_ap_pct,
        "mean_neither_percentile": mean_ne_pct,
        "n_total_neither_positives": int(n_total),
    }


# =============================================================================
# Experiment 3 — mouse training set vs human Neither overlap
# =============================================================================
def exp3_mouse_neither_overlap():
    logger.info("\n" + "=" * 70)
    logger.info("Experiment 3 — mouse APOBEC1 training set vs human Neither overlap")
    logger.info("=" * 70)

    mouse = pd.read_csv(MOUSE_RAW)
    mouse_pos = mouse[mouse["is_edited"] == 1].copy()
    logger.info("  Mouse positives: %d", len(mouse_pos))

    lev = pd.read_csv(LEVANON_CAT)
    nei = lev[lev["enzyme_category"] == "Neither"].copy()
    logger.info("  Human Neither: %d", len(nei))

    # 3a. Gene-symbol overlap (mouse Dpyd -> human DPYD)
    mouse_genes_up = set(g.upper() for g in mouse_pos["gene"].dropna().astype(str))
    nei_genes_up = set(g.upper() for g in nei["gene_refseq"].dropna().astype(str))
    overlap_genes = mouse_genes_up & nei_genes_up
    logger.info("  Gene-symbol overlap (uppercased): %d mouse genes  %d human genes  |intersection|=%d",
                len(mouse_genes_up), len(nei_genes_up), len(overlap_genes))
    if overlap_genes:
        logger.info("  Overlapping genes: %s", ", ".join(sorted(overlap_genes)))

    # 3b. Position-level overlap via mm9 -> hg38 liftover
    position_overlap = {"n_mouse_lifted": 0, "n_mouse_failed": 0, "position_hits": []}
    try:
        from pyliftover import LiftOver
        logger.info("  Loading mm9 -> hg38 chain (first use auto-downloads from UCSC)...")
        lo = LiftOver("mm9", "hg38")
        # Build human Neither position index for fast lookup
        nei_positions = set()
        for _, r in nei.iterrows():
            nei_positions.add((str(r["chr"]), int(r["start"])))

        for _, m in mouse_pos.iterrows():
            chrom = str(m["chr"])
            pos = int(m["start"])
            converted = lo.convert_coordinate(chrom, pos)
            if converted is None or len(converted) == 0:
                position_overlap["n_mouse_failed"] += 1
                continue
            position_overlap["n_mouse_lifted"] += 1
            for (new_chr, new_pos, _, _) in converted:
                # Check within ±5 nt window (liftover can shift slightly)
                for offset in range(-5, 6):
                    if (new_chr, new_pos + offset) in nei_positions:
                        position_overlap["position_hits"].append({
                            "mouse_site": f"{chrom}:{pos}",
                            "mouse_gene": str(m.get("gene", "?")),
                            "human_hit_pos": f"{new_chr}:{new_pos + offset}",
                            "offset": int(offset),
                        })
                        break

        logger.info("  Lifted %d / %d mouse sites; failed %d",
                    position_overlap["n_mouse_lifted"],
                    len(mouse_pos),
                    position_overlap["n_mouse_failed"])
        logger.info("  Position-level overlaps with Neither (±5 nt window): %d",
                    len(position_overlap["position_hits"]))
        for hit in position_overlap["position_hits"][:15]:
            logger.info("    mouse %-20s gene=%-12s  -> human %s  Δ=%+d",
                        hit["mouse_site"], hit["mouse_gene"], hit["human_hit_pos"], hit["offset"])
    except Exception as e:
        logger.warning("  Liftover failed: %s", e)
        position_overlap["error"] = str(e)

    return {
        "n_mouse_pos": int(len(mouse_pos)),
        "n_human_neither": int(len(nei)),
        "gene_overlap": {
            "n_mouse_genes": len(mouse_genes_up),
            "n_neither_genes": len(nei_genes_up),
            "n_intersection": len(overlap_genes),
            "intersection_genes": sorted(overlap_genes),
        },
        "position_overlap": position_overlap,
    }


# =============================================================================
# Main
# =============================================================================
def main():
    t0 = time.time()
    logger.info("Loading models...")
    ckpt = torch.load(PHASE3_CKPT, weights_only=False, map_location="cpu")
    state = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
    model = Phase3Model(); model.load_state_dict(state, strict=False); model.to(DEVICE)
    head = APOBEC1Head(); head.load_state_dict(torch.load(APOBEC1_CKPT, weights_only=False, map_location="cpu"))
    head.to(DEVICE)

    # Score both Neither positives and negatives with both heads
    data = load_neither_features(include_negatives=True)
    ap, ne = score_both_heads(data["X"], model, head, species_val=0.0)

    # Merge with tissue info (positives only have tissue info from Levanon)
    lev = pd.read_csv(LEVANON_CAT)[["site_id", "tissue_classification", "gene_refseq",
                                     "exonic_function", "mean_gtex_editing_rate",
                                     "max_gtex_editing_rate", "mrna_location_refseq"]]
    merged = data["splits"].copy()
    merged["apobec1"] = ap
    merged["neither"] = ne
    merged = merged.merge(lev, on="site_id", how="left")

    scored_pos = merged[merged["is_edited"] == 1].copy()
    scored_neg = merged[merged["is_edited"] == 0].copy()

    logger.info("Scored %d Neither positives + %d negatives", len(scored_pos), len(scored_neg))

    results = {}
    results["experiment_1_intestine"] = exp1_intestine_subset(scored_pos, scored_neg)
    results["experiment_2_known_targets"] = exp2_known_targets(scored_pos)
    results["experiment_3_mouse_overlap"] = exp3_mouse_neither_overlap()

    with open(RESULTS_JSON, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info("\nSaved: %s", RESULTS_JSON)
    logger.info("Total: %.1f min", (time.time() - t0) / 60)


if __name__ == "__main__":
    main()
