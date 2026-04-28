#!/usr/bin/env python3
"""PCAWG Section-1 replication end-to-end.

For each PCAWG cancer bucket:
  1. Load C>T SNPs from data/raw/pcawg/by_cancer/<cancer>_pcawg_mutations.txt
  2. Deduplicate positions; generate N_CONTROLS matched same-gene controls (seed=42)
  3. Extract 201-nt hg19 sequences (centered on each C)
  4. ViennaRNA fold (wild-type + edited) -> 16-dim struct+loop features
  5. Build 40-dim hand features (24 motif + 16 struct/loop)
  6. Compute RNA-FM pooled embeddings (original + edited) -> 2*640-d
  7. Score Phase3Model (binary + enzyme adapters) + APOBEC1 head
  8. Cache scores per cancer

After all cancers:
  9. OR@{p50,p75,p90,p95} per cancer (Fisher), per head
 10. Pooled GI vs non-GI OR (global threshold, Fisher)
 11. Permutation reversal test (exhaustive)

Defaults: GI cancers + SKCM.  Pass --cancers to change.

Output:
  experiments/multi_enzyme/outputs/pcawg/<cancer>_scores.csv
  experiments/multi_enzyme/outputs/pcawg/<cancer>_cache.npz (seqs, folds, rnafm, hand)
  experiments/multi_enzyme/outputs/pcawg/pcawg_section1.json
  experiments/multi_enzyme/outputs/pcawg/run.log

Usage:
  conda run -n quris python experiments/multi_enzyme/exp_pcawg_end2end.py
  conda run -n quris python experiments/multi_enzyme/exp_pcawg_end2end.py --cancers lihc,coadread --n-controls 2
"""
from __future__ import annotations

import argparse
import gc
import gzip
import json
import logging
import multiprocessing as mp
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pyfaidx import Fasta
from scipy import stats
from scipy.stats import fisher_exact

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
from src.data.apobec_feature_extraction import extract_motif_from_seq  # noqa

sys.stdout.reconfigure(line_buffering=True)

# ----------------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------------
DATA_DIR = PROJECT_ROOT / "data"
HG19_FA = DATA_DIR / "raw" / "genomes" / "hg19.fa"
REFGENE_HG19 = DATA_DIR / "raw" / "genomes" / "refGene_hg19.txt"
PCAWG_BY_CANCER = DATA_DIR / "raw" / "pcawg" / "by_cancer"
OUT_DIR = PROJECT_ROOT / "experiments" / "multi_enzyme" / "outputs" / "pcawg"
OUT_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = OUT_DIR / "run.log"

PHASE3_CKPT = (
    PROJECT_ROOT / "experiments" / "multi_enzyme" / "outputs"
    / "phase3_neural_true_validation" / "phase3_neural_full.pt"
)
APOBEC1_CKPT = (
    PROJECT_ROOT / "experiments" / "multi_enzyme" / "outputs"
    / "apobec1_head" / "apobec1_head.pt"
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(LOG_FILE, mode="w")],
)
logger = logging.getLogger(__name__)

CENTER = 100
SEED = 42
DEFAULT_N_CONTROLS = 2
GI_CANCERS = {"coadread", "stad", "esca", "lihc"}
NON_GI_CANCERS = {"blca", "brca", "cesc", "lusc", "hnsc", "skcm"}

D_INPUT = 1320
D_SHARED = 128
ENZYMES = ["A3A", "A3B", "A3G", "A3A_A3G", "Neither"]
N_ENZYMES_CLS = 6

DEVICE = (
    torch.device("cuda") if torch.cuda.is_available()
    else torch.device("mps") if hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    else torch.device("cpu")
)

# ----------------------------------------------------------------------------
# Models (exact replica of Phase3Model + APOBEC1Head)
# ----------------------------------------------------------------------------


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


# ----------------------------------------------------------------------------
# Data handling (mirrors cache_tcga_hand_features.py)
# ----------------------------------------------------------------------------
def parse_exons(refgene_path):
    exons = defaultdict(list)
    with open(refgene_path) as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 11:
                continue
            chrom = parts[2]
            strand = parts[3]
            gene = parts[12] if len(parts) > 12 else parts[1]
            starts = [int(x) for x in parts[9].split(",") if x]
            ends = [int(x) for x in parts[10].split(",") if x]
            for s, e in zip(starts, ends):
                exons[gene].append((chrom, s, e, strand))
    return exons


def precompute_gene_c_positions(exons_by_gene, genome):
    gene_chrom_positions = {}
    for gene, gene_exons in exons_by_gene.items():
        by_chrom = defaultdict(list)
        for chrom, start, end, strand in gene_exons:
            by_chrom[chrom].append((start, end))
        for chrom, intervals in by_chrom.items():
            c_positions = []
            for start, end in intervals:
                try:
                    exon_seq = str(genome[chrom][start:end]).upper()
                    for i, base in enumerate(exon_seq):
                        pos = start + i
                        if base == "C":
                            c_positions.append((chrom, pos, "+"))
                        elif base == "G":
                            c_positions.append((chrom, pos, "-"))
                except (KeyError, ValueError):
                    continue
            if c_positions:
                gene_chrom_positions[(gene, chrom)] = c_positions
    return gene_chrom_positions


def parse_ct_mutations(maf_path):
    rows = []
    for chunk in pd.read_csv(maf_path, sep="\t", comment="#", low_memory=False, chunksize=50000):
        ct = (chunk["Reference_Allele"] == "C") & (chunk["Tumor_Seq_Allele2"] == "T")
        ga = (chunk["Reference_Allele"] == "G") & (chunk["Tumor_Seq_Allele2"] == "A")
        sub = chunk[ct | ga].copy()
        if len(sub) > 0:
            chrom = sub["Chromosome"].astype(str)
            if not chrom.str.startswith("chr").any():
                chrom = "chr" + chrom
            sub["chrom"] = chrom
            sub["pos"] = sub["Start_Position"].astype(int) - 1
            sub["strand_inf"] = np.where(sub["Reference_Allele"] == "C", "+", "-")
            sub["gene"] = sub.get("Hugo_Symbol", "unknown")
            rows.append(sub[["chrom", "pos", "strand_inf", "gene"]].copy())
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows).drop_duplicates(subset=["chrom", "pos"])


def get_matched_controls(mutations_df, gene_chrom_positions, n_controls=5):
    rng = np.random.RandomState(SEED)
    controls = []
    for _, row in mutations_df.iterrows():
        gene, chrom, mut_pos = row["gene"], row["chrom"], row["pos"]
        c_positions = gene_chrom_positions.get((gene, chrom), [])
        if not c_positions:
            continue
        filtered = [(ch, p, s) for ch, p, s in c_positions if p != mut_pos]
        if len(filtered) >= n_controls:
            chosen = rng.choice(len(filtered), n_controls, replace=False)
            for idx in chosen:
                controls.append({"chrom": filtered[idx][0], "pos": filtered[idx][1],
                                 "strand_inf": filtered[idx][2], "gene": gene})
        elif filtered:
            for ch, p, s in filtered[:n_controls]:
                controls.append({"chrom": ch, "pos": p, "strand_inf": s, "gene": gene})
    return pd.DataFrame(controls) if controls else pd.DataFrame()


def extract_sequences(positions_df, genome):
    seqs = []
    for _, row in positions_df.iterrows():
        chrom, pos, strand = row["chrom"], int(row["pos"]), row["strand_inf"]
        try:
            start, end = pos - CENTER, pos + CENTER + 1
            if start < 0 or end > len(genome[chrom]):
                seqs.append(None)
                continue
            seq = str(genome[chrom][start:end]).upper()
            if strand == "-":
                seq = seq.translate(str.maketrans("ACGT", "TGCA"))[::-1]
            seqs.append(seq)
        except (KeyError, ValueError):
            seqs.append(None)
    return seqs


# ----------------------------------------------------------------------------
# ViennaRNA folding
# ----------------------------------------------------------------------------
def fold_one(seq_and_idx):
    import RNA
    idx, seq = seq_and_idx
    if seq is None or len(seq) != 201 or seq[CENTER] != "C":
        return idx, None
    rna_wt = seq.replace("T", "U")
    rna_ed = rna_wt[:CENTER] + "U" + rna_wt[CENTER + 1:]
    try:
        fc_w = RNA.fold_compound(rna_wt)
        mfe_w_str, mfe_w = fc_w.mfe()
        fc_w.pf()
        bpp_w = np.array(fc_w.bpp())  # upper triangular
        fc_e = RNA.fold_compound(rna_ed)
        mfe_e_str, mfe_e = fc_e.mfe()
        fc_e.pf()
        bpp_e = np.array(fc_e.bpp())

        # Pairing probability at center: sum of bpp over center column
        def _center_p(bpp):
            n = bpp.shape[0] - 1
            col = bpp[:, CENTER + 1].sum() + bpp[CENTER + 1, :].sum()
            return float(col)

        def _window_p(bpp, half=10):
            n = bpp.shape[0] - 1
            lo, hi = max(1, CENTER + 1 - half), min(n, CENTER + 1 + half)
            probs = []
            for i in range(lo, hi + 1):
                pi = bpp[:, i].sum() + bpp[i, :].sum()
                probs.append(float(pi))
            return probs

        return idx, {
            "bpp_wt_center": _center_p(bpp_w),
            "bpp_ed_center": _center_p(bpp_e),
            "bpp_wt_window": _window_p(bpp_w),
            "bpp_ed_window": _window_p(bpp_e),
            "mfe_wt": float(mfe_w),
            "mfe_ed": float(mfe_e),
            "struct_wt": mfe_w_str,
        }
    except Exception:
        return idx, None


def derive_features_from_fold(fold_result):
    if fold_result is None:
        return None
    pw, pe = fold_result["bpp_wt_center"], fold_result["bpp_ed_center"]
    dp = pe - pw
    dm = fold_result["mfe_ed"] - fold_result["mfe_wt"]
    dw = np.array(fold_result["bpp_ed_window"]) - np.array(fold_result["bpp_wt_window"])

    def ent(p):
        if p <= 0 or p >= 1:
            return 0.0
        return -(p * np.log2(p + 1e-10) + (1 - p) * np.log2(1 - p + 1e-10))

    de = ent(pe) - ent(pw)
    md = float(np.mean(dw)) if len(dw) > 0 else 0.0
    sd = float(np.std(dw)) if len(dw) > 0 else 0.0
    struct_delta = np.array([dp, -dp, de, dm, md, sd, -md], dtype=np.float32)

    s = fold_result["struct_wt"]
    up = 1.0 if s[CENTER] == "." else 0.0
    ls = dj = da = 0.0
    rlp = 0.5
    lst = rst = mas = lu = 0.0
    if up:
        l = CENTER
        while l > 0 and s[l] == ".":
            l -= 1
        r = CENTER
        while r < len(s) - 1 and s[r] == ".":
            r += 1
        ls = float(r - l - 1)
        if ls > 0:
            p = CENTER - l - 1
            rlp = p / max(ls - 1, 1)
            da = abs(p - (ls - 1) / 2)
        i = l
        c = 0
        while i >= 0 and s[i] in "()":
            c += 1
            i -= 1
        lst = float(c)
        i = r
        c = 0
        while i < len(s) and s[i] in "()":
            c += 1
            i += 1
        rst = float(c)
        mas = max(lst, rst)
    reg = s[max(0, CENTER - 10):min(len(s), CENTER + 11)]
    lu = sum(1 for ch in reg if ch == ".") / max(len(reg), 1)
    loop_features = np.array([up, ls, dj, da, rlp, lst, rst, mas, lu], dtype=np.float32)
    return struct_delta, loop_features


# ----------------------------------------------------------------------------
# RNA-FM inference
# ----------------------------------------------------------------------------
def load_rnafm_model():
    try:
        import fm
    except ImportError:
        logger.error("RNA-FM package 'fm' not installed.")
        raise
    logger.info("Loading RNA-FM checkpoint...")
    model, alphabet = fm.pretrained.rna_fm_t12()
    batch_converter = alphabet.get_batch_converter()
    model = model.eval().to(DEVICE)
    return model, batch_converter


def rnafm_pooled_batch(seqs, model, batch_converter, batch_size=32):
    """Compute pooled RNA-FM embeddings (mean over sequence dimension)."""
    n = len(seqs)
    out = np.zeros((n, 640), dtype=np.float32)
    with torch.no_grad():
        for b in range(0, n, batch_size):
            e = min(b + batch_size, n)
            batch = [(f"s{b+i}", (s or "N" * 201).replace("T", "U")) for i, s in enumerate(seqs[b:e])]
            _, _, tokens = batch_converter(batch)
            tokens = tokens.to(DEVICE)
            try:
                result = model(tokens, repr_layers=[12])
                # mean pool over non-pad positions (RNA-FM uses attention mask implicitly)
                emb = result["representations"][12][:, 1:-1, :].mean(dim=1).cpu().numpy()
                out[b:e] = emb.astype(np.float32)
            except Exception as ex:
                logger.warning("  RNA-FM batch %d failed: %s", b, ex)
    return out


# ----------------------------------------------------------------------------
# Scoring
# ----------------------------------------------------------------------------
def score_phase3(X, model, apobec1_head):
    """Returns dict of per-head sigmoid scores."""
    model.eval(); apobec1_head.eval()
    n = X.shape[0]
    scores = {k: np.zeros(n, dtype=np.float32) for k in ("binary", "A3A", "A3B", "A3G", "A3A_A3G", "Neither", "apobec1")}
    BATCH = 256
    with torch.no_grad():
        for b in range(0, n, BATCH):
            e = min(b + BATCH, n)
            xb = torch.from_numpy(X[b:e]).float().to(DEVICE)
            shared = model.shared_encoder(xb)
            scores["binary"][b:e] = torch.sigmoid(model.binary_head(shared).squeeze(-1)).cpu().numpy()
            for enz in ENZYMES:
                scores[enz][b:e] = torch.sigmoid(model.enzyme_adapters[enz](shared).squeeze(-1)).cpu().numpy()
            sb = torch.zeros((e - b, 1), dtype=torch.float32, device=DEVICE)
            scores["apobec1"][b:e] = torch.sigmoid(apobec1_head(shared, sb)).cpu().numpy()
    return scores


# ----------------------------------------------------------------------------
# Per-cancer pipeline
# ----------------------------------------------------------------------------
def process_cancer(cancer, genome, gene_c_pos, phase3, apobec1_head,
                   rnafm_model, rnafm_conv, n_controls=DEFAULT_N_CONTROLS,
                   n_fold_workers=6, force=False):
    t0 = time.time()
    out_scores = OUT_DIR / f"{cancer}_scores.csv"
    if out_scores.exists() and not force:
        logger.info("  SKIP %s (already cached at %s)", cancer, out_scores)
        return pd.read_csv(out_scores)

    maf_path = PCAWG_BY_CANCER / f"{cancer}_pcawg_mutations.txt"
    if not maf_path.exists():
        logger.warning("  %s: no MAF at %s", cancer, maf_path)
        return None

    logger.info("\n=== %s ===", cancer.upper())

    # 1. Parse mutations (dedup on chrom,pos)
    mut_df = parse_ct_mutations(str(maf_path))
    if len(mut_df) == 0:
        logger.warning("  %s: no C>T after parse", cancer)
        return None
    logger.info("  unique mutation positions: %d", len(mut_df))

    # 2. Generate matched controls
    ctrl_df = get_matched_controls(mut_df, gene_c_pos, n_controls)
    logger.info("  controls: %d (%.1f per mut)", len(ctrl_df),
                len(ctrl_df) / max(1, len(mut_df)))

    # 3. Extract sequences
    mut_df = mut_df.reset_index(drop=True)
    ctrl_df = ctrl_df.reset_index(drop=True)
    mut_seqs = extract_sequences(mut_df, genome)
    ctrl_seqs = extract_sequences(ctrl_df, genome)

    # Keep only valid (length 201, center=C)
    def _filter(df, seqs):
        keep = [(s is not None and len(s) == 201 and s[CENTER] == "C") for s in seqs]
        return df[keep].reset_index(drop=True), [s for s, k in zip(seqs, keep) if k]

    mut_df, mut_seqs = _filter(mut_df, mut_seqs)
    ctrl_df, ctrl_seqs = _filter(ctrl_df, ctrl_seqs)
    logger.info("  valid (201nt, center=C): mut=%d ctrl=%d", len(mut_df), len(ctrl_df))

    all_seqs = mut_seqs + ctrl_seqs
    edited_seqs = [s[:CENTER] + "T" + s[CENTER + 1:] for s in all_seqs]
    n_total = len(all_seqs)

    # 4. ViennaRNA fold (parallel)
    logger.info("  ViennaRNA folding %d positions (%d workers)...", n_total, n_fold_workers)
    t_fold = time.time()
    with mp.Pool(n_fold_workers) as pool:
        fold_results = [None] * n_total
        for idx, res in pool.imap_unordered(fold_one, enumerate(all_seqs), chunksize=100):
            fold_results[idx] = res
    logger.info("    done in %.1f min (%.2f ms/pos)",
                (time.time() - t_fold) / 60, 1000 * (time.time() - t_fold) / max(1, n_total))

    # 5. Hand features (40-d)
    hand = np.zeros((n_total, 40), dtype=np.float32)
    for i in range(n_total):
        hand[i, :24] = extract_motif_from_seq(all_seqs[i])
        res = derive_features_from_fold(fold_results[i])
        if res is not None:
            hand[i, 24:31] = res[0]
            hand[i, 31:40] = res[1]
    hand = np.nan_to_num(hand, nan=0.0)

    # 6. RNA-FM embeddings (orig + edited)
    logger.info("  RNA-FM pooled embedding (orig %d + edited %d) on %s ...",
                n_total, n_total, DEVICE)
    t_fm = time.time()
    emb_o = rnafm_pooled_batch(all_seqs, rnafm_model, rnafm_conv, batch_size=16)
    emb_e = rnafm_pooled_batch(edited_seqs, rnafm_model, rnafm_conv, batch_size=16)
    emb_d = (emb_e - emb_o).astype(np.float32)
    logger.info("    done in %.1f min", (time.time() - t_fm) / 60)

    # 7. Build X and score
    X = np.concatenate([emb_o, emb_d, hand], axis=1).astype(np.float32)
    scores = score_phase3(X, phase3, apobec1_head)

    # 8. Assemble per-site score table
    rec = {
        "chrom": list(mut_df["chrom"]) + list(ctrl_df["chrom"]),
        "pos":   list(mut_df["pos"])   + list(ctrl_df["pos"]),
        "gene":  list(mut_df["gene"])  + list(ctrl_df["gene"]),
        "type":  ["mutation"] * len(mut_df) + ["control"] * len(ctrl_df),
    }
    for key, vals in scores.items():
        rec[f"score_{key}"] = vals
    df_out = pd.DataFrame(rec)
    df_out.to_csv(out_scores, index=False)

    # Cache intermediate (hand + embeddings) for re-scoring
    cache = OUT_DIR / f"{cancer}_cache.npz"
    np.savez_compressed(
        cache,
        chrom=np.array(rec["chrom"], dtype=object),
        pos=np.array(rec["pos"], dtype=np.int64),
        gene=np.array(rec["gene"], dtype=object),
        type=np.array(rec["type"], dtype=object),
        hand=hand, emb_o=emb_o, emb_d=emb_d,
    )
    logger.info("  wrote %s and %s   (%.1f min total)", out_scores, cache, (time.time() - t0) / 60)
    return df_out


# ----------------------------------------------------------------------------
# OR computation (re-uses logic from exp_apobec1_vs_neither_gi.py)
# ----------------------------------------------------------------------------
def or_at_percentile(scores, mut_mask, ctrl_mask, pct, threshold=None):
    pool = scores[mut_mask | ctrl_mask]
    if len(pool) < 50:
        return {"OR": float("nan"), "p": 1.0, "threshold": float("nan")}
    thresh = threshold if threshold is not None else float(np.percentile(pool, pct))
    ma = int((scores[mut_mask] >= thresh).sum())
    mb = int((scores[mut_mask] < thresh).sum())
    ca = int((scores[ctrl_mask] >= thresh).sum())
    cb = int((scores[ctrl_mask] < thresh).sum())
    if all(x > 0 for x in (ma, mb, ca, cb)):
        OR, p = fisher_exact([[ma, mb], [ca, cb]])
    else:
        OR, p = float("nan"), 1.0
    return {"OR": float(OR), "p": float(p), "threshold": thresh,
            "mut_above": ma, "mut_below": mb, "ctrl_above": ca, "ctrl_below": cb}


def pooled_OR_ct(scores_list, mut_list, ctrl_list, pct):
    """Pool across cancers using a global percentile threshold."""
    s = np.concatenate(scores_list)
    m = np.concatenate(mut_list)
    c = np.concatenate(ctrl_list)
    return or_at_percentile(s, m, c, pct)


def compute_all_ORs(per_cancer_df, cancers_to_use):
    """per_cancer_df: dict cancer -> DataFrame with score_* cols + type col."""
    HEADS = ["binary", "A3A", "A3B", "A3G", "A3A_A3G", "Neither", "apobec1"]
    result = {"per_cancer": [], "pooled_GI": {}, "pooled_nonGI": {}, "heads": HEADS}

    # Per-cancer
    for c in cancers_to_use:
        df = per_cancer_df.get(c)
        if df is None:
            continue
        mut = (df["type"].values == "mutation")
        ctl = (df["type"].values == "control")
        row = {"cancer": c, "n_mut": int(mut.sum()), "n_ctrl": int(ctl.sum()),
               "group": "GI" if c in GI_CANCERS else "non-GI"}
        for h in HEADS:
            col = f"score_{h}"
            if col not in df.columns:
                continue
            for pct in (50, 75, 90, 95):
                stat = or_at_percentile(df[col].values, mut, ctl, pct)
                row[f"{h}_OR_p{pct}"] = stat["OR"]
                row[f"{h}_p_p{pct}"] = stat["p"]
        result["per_cancer"].append(row)

    # Pooled (global threshold over all cancers used)
    def _build_pool(gs):
        scores_by_head = {h: [] for h in HEADS}
        mut_list, ctl_list = [], []
        for c in gs:
            df = per_cancer_df.get(c)
            if df is None:
                continue
            for h in HEADS:
                scores_by_head[h].append(df[f"score_{h}"].values)
            mut_list.append((df["type"].values == "mutation"))
            ctl_list.append((df["type"].values == "control"))
        return scores_by_head, mut_list, ctl_list

    for group_name, group_set in [("GI", [c for c in cancers_to_use if c in GI_CANCERS]),
                                   ("nonGI", [c for c in cancers_to_use if c in NON_GI_CANCERS])]:
        if not group_set:
            continue
        sbh, mlist, clist = _build_pool(group_set)
        key = f"pooled_{group_name}"
        result[key] = {"cancers": group_set}
        for h in HEADS:
            if not sbh[h]:
                continue
            for pct in (50, 75, 90, 95):
                stat = pooled_OR_ct(sbh[h], mlist, clist, pct)
                result[key].setdefault(h, {})[f"p{pct}"] = stat
    return result


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cancers", default="lihc,coadread,stad,esca,skcm",
                    help="comma-separated cancer bucket names")
    ap.add_argument("--n-controls", type=int, default=DEFAULT_N_CONTROLS)
    ap.add_argument("--n-fold-workers", type=int, default=6)
    ap.add_argument("--force", action="store_true", help="recompute even if cached")
    args = ap.parse_args()

    cancers = [c.strip() for c in args.cancers.split(",") if c.strip()]
    logger.info("Device: %s", DEVICE)
    logger.info("Cancers: %s", cancers)
    logger.info("N controls per mut: %d", args.n_controls)

    # Load genome and exons (shared across cancers)
    logger.info("Loading hg19 genome...")
    genome = Fasta(str(HG19_FA))
    logger.info("Parsing refGene exons...")
    exons = parse_exons(str(REFGENE_HG19))
    logger.info("Precomputing gene C/G positions...")
    gene_c_pos = precompute_gene_c_positions(exons, genome)
    logger.info("  %d (gene,chrom) pairs", len(gene_c_pos))

    # Load models
    logger.info("Loading Phase3Model + APOBEC1 head...")
    ckpt = torch.load(PHASE3_CKPT, weights_only=False, map_location="cpu")
    state = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
    phase3 = Phase3Model()
    phase3.load_state_dict(state, strict=False)
    phase3.to(DEVICE)
    apobec1_head = APOBEC1Head()
    apobec1_head.load_state_dict(torch.load(APOBEC1_CKPT, weights_only=False, map_location="cpu"))
    apobec1_head.to(DEVICE)

    logger.info("Loading RNA-FM (this takes ~30s)...")
    rnafm_model, rnafm_conv = load_rnafm_model()

    # Process each cancer
    per_cancer_df = {}
    for cancer in cancers:
        try:
            df = process_cancer(cancer, genome, gene_c_pos, phase3, apobec1_head,
                                rnafm_model, rnafm_conv,
                                n_controls=args.n_controls,
                                n_fold_workers=args.n_fold_workers,
                                force=args.force)
            if df is not None:
                per_cancer_df[cancer] = df
        except Exception as e:
            logger.exception("  %s FAILED: %s", cancer, e)
            continue
        gc.collect()
        if DEVICE.type == "mps":
            torch.mps.empty_cache()

    # OR analysis
    logger.info("\nComputing OR tables...")
    result = compute_all_ORs(per_cancer_df, cancers)

    # Print summary
    logger.info("\n" + "=" * 70)
    logger.info("PCAWG Section-1 replication — per-cancer OR@p90 (binary, Neither, apobec1)")
    logger.info("=" * 70)
    for row in result["per_cancer"]:
        c = row["cancer"]
        logger.info(
            "  %-9s [%-6s] n_mut=%7d n_ctrl=%7d | bin=%.3f  nei=%.3f  apo=%.3f",
            c, row["group"], row["n_mut"], row["n_ctrl"],
            row.get("binary_OR_p90", float("nan")),
            row.get("Neither_OR_p90", float("nan")),
            row.get("apobec1_OR_p90", float("nan")),
        )

    logger.info("\nPooled GI vs non-GI OR@p90:")
    for key in ("pooled_GI", "pooled_nonGI"):
        if key not in result:
            continue
        info = result[key]
        logger.info("  %s (cancers=%s):", key, info.get("cancers"))
        for h in result["heads"]:
            if h not in info:
                continue
            r = info[h].get("p90", {})
            logger.info("    %-10s OR=%.3f  p=%.2e  thresh=%.4f",
                        h, r.get("OR", float("nan")), r.get("p", 1.0), r.get("threshold", float("nan")))

    with (OUT_DIR / "pcawg_section1.json").open("w") as f:
        json.dump(result, f, indent=2, default=str)
    logger.info("\nSaved: %s", OUT_DIR / "pcawg_section1.json")


if __name__ == "__main__":
    main()
