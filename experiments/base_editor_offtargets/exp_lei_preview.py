#!/usr/bin/env python3
"""Lei BE4max gRNA-independent off-target enrichment preview.

Pre-registered: experiments/base_editor_offtargets/PRE_REGISTRATION_lei_preview.md

Loads ~565 Lei Detect-seq off-target sites + 5x motif-matched controls,
scores with Phase3 binary + 5 enzyme adapters + APOBEC1 head, computes
top-percentile enrichment ORs with bootstrap CIs.

Run on Mac MPS, ~15min wall clock. Output: outputs/lei_preview/
"""
from __future__ import annotations
import gzip
import json
import logging
import multiprocessing as mp
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pyfaidx import Fasta
from scipy.stats import fisher_exact
from statsmodels.stats.multitest import multipletests

ROOT = Path("/Users/shaharharel/Documents/github/edit-rna-apobec")
LEI_DIR = ROOT / "data/raw/be_off_target/lei_2021"
HG38 = ROOT / "data/raw/genomes/hg38.fa"
PHASE3_CKPT = ROOT / "experiments/multi_enzyme/outputs/phase3_mfe_only/phase3_mfe_only.pt"
APOBEC1_CKPT = ROOT / "experiments/multi_enzyme/outputs/apobec1_head/apobec1_head_mfe_only.pt"
OUT = ROOT / "experiments/base_editor_offtargets/outputs/lei_preview"
OUT.mkdir(parents=True, exist_ok=True)

CENTER = 100
N_CONTROLS = 5
HALF_KB = 5000
MIN_GAP = 10
SEED = 42
N_BOOTSTRAP = 1000
PERCENTILES = [90, 95, 99]

ENZYMES = ["A3A", "A3B", "A3G", "A3A_A3G", "Neither"]
D_INPUT = 1320
D_SHARED = 128
EMB_DIM = 640
N_ENZYMES_CLS = 6

LEI_FILES = [
    ("VEGFA",     "GSE151265_293T-VEGFA-Detect-seq_pRBS.bed.gz", "293T"),
    ("HEK4_293T", "GSE151265_293T-HEK4-Detect-seq_pRBS.bed.gz",  "293T"),
    ("HEK4_MCF7", "GSE151265_MCF7-HEK4-Detect-seq_pRBS.bed.gz",  "MCF7"),
    ("EMX1",      "GSE151265_293T-EMX1-Detect-seq_pRBS.bed.gz",  "293T"),
]

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("lei_preview")

# ---------------------------------------------------------------------------
# Models (mirrored from scripts/gcp_panel/score_panel.py)
# ---------------------------------------------------------------------------
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
        self.species_proj = nn.Sequential(nn.Linear(1, 16), nn.GELU(), nn.Linear(16, d_shared))
        self.head = nn.Sequential(nn.Linear(d_shared, 32), nn.GELU(), nn.Linear(32, 1))

    def forward(self, shared, species):
        bias = self.species_proj(species)
        return self.head(shared + bias).squeeze(-1)


# ---------------------------------------------------------------------------
# Step 1: Load Lei positives
# ---------------------------------------------------------------------------
def load_lei_positives() -> pd.DataFrame:
    rows = []
    for sgRNA, fn, cell in LEI_FILES:
        path = LEI_DIR / fn
        with gzip.open(path, "rt") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or line.startswith("track") or line.startswith("browser"):
                    continue
                parts = line.split("\t")
                chrom = parts[0]
                try:
                    start = int(parts[1])
                    end = int(parts[2])
                except (ValueError, IndexError):
                    continue  # header row or malformed
                strand = parts[5] if len(parts) >= 6 else "+"
                pos = start
                rows.append({
                    "chrom": chrom, "pos": pos, "end": end, "strand": strand,
                    "sgRNA": sgRNA, "cell": cell, "label": "positive",
                })
    df = pd.DataFrame(rows)
    log.info("Loaded %d Lei positives across %d sgRNAs", len(df), len(LEI_FILES))
    return df


# ---------------------------------------------------------------------------
# Step 2: Sequence extraction + control generation
# ---------------------------------------------------------------------------
COMP = str.maketrans("ACGTN", "TGCAN")

def get_seq(genome: Fasta, chrom: str, pos: int, strand: str, flank: int = 100):
    try:
        chrom_len = len(genome[chrom])
    except KeyError:
        return None
    s, e = pos - flank, pos + flank + 1
    if s < 0 or e > chrom_len:
        return None
    seq = str(genome[chrom][s:e]).upper()
    if strand == "-":
        seq = seq.translate(COMP)[::-1]
    if len(seq) != 201:
        return None
    return seq


def attach_seq_and_filter(df: pd.DataFrame, genome: Fasta) -> pd.DataFrame:
    """Attach 201-nt window, drop sites where center is not C."""
    seqs = []
    keep = []
    for _, r in df.iterrows():
        seq = get_seq(genome, r.chrom, r.pos, r.strand)
        if seq is None:
            seqs.append(None); keep.append(False); continue
        # If center is not C, try shifting by 1 (BED can be 0-based start of feature)
        if seq[CENTER] != "C":
            # Try the other strand if the center base is the complement
            if seq[CENTER] == "G":
                # Off-target was reported on the other strand; revcomp
                rc_strand = "-" if r.strand == "+" else "+"
                seq_alt = get_seq(genome, r.chrom, r.pos, rc_strand)
                if seq_alt is not None and seq_alt[CENTER] == "C":
                    seqs.append(seq_alt); keep.append(True); continue
            seqs.append(None); keep.append(False); continue
        seqs.append(seq); keep.append(True)
    out = df.copy()
    out["seq"] = seqs
    out = out[pd.Series(keep, index=out.index)].reset_index(drop=True)
    return out


def gen_controls(positives: pd.DataFrame, genome: Fasta) -> pd.DataFrame:
    """Sample 5 motif-matched same-region Cs per positive, ±5kb."""
    rng = random.Random(SEED)
    pos_set = set(zip(positives["chrom"], positives["pos"]))
    controls = []
    skipped = 0
    for _, r in positives.iterrows():
        trinuc = r.seq[CENTER - 1] + r.seq[CENTER] + r.seq[CENTER + 1]  # XCY
        try:
            chrom_len = len(genome[r.chrom])
        except KeyError:
            skipped += 1; continue
        ws = max(0, r.pos - HALF_KB)
        we = min(chrom_len, r.pos + HALF_KB)
        chunk = str(genome[r.chrom][ws:we]).upper()
        candidates = []
        for i in range(1, len(chunk) - 1):
            if chunk[i] == "C" and chunk[i - 1] == trinuc[0] and chunk[i + 1] == trinuc[2]:
                cpos = ws + i
                if abs(cpos - r.pos) < MIN_GAP:
                    continue
                if (r.chrom, cpos) in pos_set:
                    continue
                candidates.append(cpos)
        if len(candidates) < N_CONTROLS:
            skipped += 1
            continue
        sampled = rng.sample(candidates, N_CONTROLS)
        for cpos in sampled:
            controls.append({
                "chrom": r.chrom, "pos": cpos, "end": cpos + 1, "strand": r.strand,
                "sgRNA": r.sgRNA, "cell": r.cell, "label": "control",
                "matched_to": r.pos,
            })
    df = pd.DataFrame(controls)
    log.info("Generated %d controls for %d positives (skipped %d positives)",
             len(df), len(positives) - skipped, skipped)
    return df


# ---------------------------------------------------------------------------
# Step 3: ViennaRNA hand features (motif 24 + struct delta 7 zeroed + loop 9)
# ---------------------------------------------------------------------------
def _compute_hand_one(seq: str):
    """Compute 40-d hand features for one sequence. Struct delta is zeroed
    (matches MFE-only training regime of phase3_mfe_only.pt)."""
    import RNA
    motif = np.zeros(24, dtype=np.float32)
    seq_u = seq.replace("T", "U")
    bases = ["A", "C", "G", "U"]
    if CENTER > 0:
        up = seq_u[CENTER - 1]
        for j, m in enumerate(["UC", "CC", "AC", "GC"]):
            if up + "C" == m: motif[j] = 1.0
    if CENTER < len(seq_u) - 1:
        down = seq_u[CENTER + 1]
        for j, m in enumerate(["CA", "CG", "CU", "CC"]):
            if "C" + down == m: motif[4 + j] = 1.0
    for offset, bo in [(-2, 8), (-1, 12)]:
        p = CENTER + offset
        if 0 <= p < len(seq_u):
            for bi, b in enumerate(bases):
                if seq_u[p] == b: motif[bo + bi] = 1.0
    for offset, bo in [(1, 16), (2, 20)]:
        p = CENTER + offset
        if 0 <= p < len(seq_u):
            for bi, b in enumerate(bases):
                if seq_u[p] == b: motif[bo + bi] = 1.0

    struct_delta = np.zeros(7, dtype=np.float32)  # MFE-only regime

    loop = np.zeros(9, dtype=np.float32)
    valid = True
    try:
        s, _ = RNA.fold(seq_u)
        up = 1.0 if s[CENTER] == "." else 0.0
        loop[0] = up
        if up:
            l = CENTER
            while l > 0 and s[l] == ".":
                l -= 1
            r = CENTER
            while r < len(s) - 1 and s[r] == ".":
                r += 1
            ls = float(r - l - 1)
            loop[1] = ls
            if ls > 0:
                p = CENTER - l - 1
                loop[4] = p / max(ls - 1, 1)
                loop[3] = abs(p - (ls - 1) / 2)
            i = l; c = 0
            while i >= 0 and s[i] in "()":
                c += 1; i -= 1
            loop[5] = float(c)
            i = r; c = 0
            while i < len(s) and s[i] in "()":
                c += 1; i += 1
            loop[6] = float(c)
            loop[7] = max(loop[5], loop[6])
        reg = s[max(0, CENTER - 10):min(len(s), CENTER + 11)]
        loop[8] = sum(1 for ch in reg if ch == ".") / max(len(reg), 1)
    except Exception:
        valid = False

    hand = np.concatenate([motif, struct_delta, loop])
    return hand.astype(np.float32), valid


def _hand_worker(args):
    idx, seq = args
    h, v = _compute_hand_one(seq)
    return idx, h, v


def compute_hand40(seqs: list[str], n_workers: int = 8):
    n = len(seqs)
    out = np.zeros((n, 40), dtype=np.float32)
    valid = np.zeros(n, dtype=bool)
    t0 = time.time()
    with mp.Pool(n_workers) as pool:
        for idx, h, v in pool.imap_unordered(_hand_worker, list(enumerate(seqs)), chunksize=32):
            out[idx] = h
            valid[idx] = v
    log.info("hand40: %d sequences in %.1fs", n, time.time() - t0)
    return out, valid


# ---------------------------------------------------------------------------
# Step 4: RNA-FM embeddings (orig + edited delta)
# ---------------------------------------------------------------------------
def compute_rnafm(seqs: list[str], device: torch.device, batch: int = 32):
    import fm
    log.info("Loading RNA-FM (rna_fm_t12)...")
    model, alphabet = fm.pretrained.rna_fm_t12()
    bc = alphabet.get_batch_converter()
    model = model.eval().to(device)

    n = len(seqs)
    orig = np.zeros((n, EMB_DIM), dtype=np.float16)
    delta = np.zeros((n, EMB_DIM), dtype=np.float16)

    def edited(s):
        return s[:CENTER] + "U" + s[CENTER + 1:]

    t0 = time.time()
    with torch.no_grad():
        for i in range(0, n, batch):
            j = min(i + batch, n)
            data_o = [(f"o{k}", seqs[k].replace("T", "U")) for k in range(i, j)]
            data_e = [(f"e{k}", edited(seqs[k].replace("T", "U"))) for k in range(i, j)]
            _, _, t_o = bc(data_o)
            _, _, t_e = bc(data_e)
            t_o = t_o.to(device)
            t_e = t_e.to(device)
            r_o = model(t_o, repr_layers=[12])["representations"][12]
            r_e = model(t_e, repr_layers=[12])["representations"][12]
            # mean pool over tokens (excluding cls)
            emb_o = r_o[:, 1:-1, :].mean(dim=1).cpu().float().numpy()
            emb_e = r_e[:, 1:-1, :].mean(dim=1).cpu().float().numpy()
            orig[i:j] = emb_o.astype(np.float16)
            delta[i:j] = (emb_e - emb_o).astype(np.float16)
            if i % (batch * 10) == 0:
                log.info("  RNA-FM %d/%d (%.1fs)", j, n, time.time() - t0)
    log.info("RNA-FM done: %d sequences in %.1fs", n, time.time() - t0)
    return orig, delta


# ---------------------------------------------------------------------------
# Step 5: Score
# ---------------------------------------------------------------------------
def score_all(orig, delta, hand40, valid, device):
    log.info("Loading Phase3...")
    p3 = Phase3Model()
    state = torch.load(PHASE3_CKPT, weights_only=False, map_location="cpu")
    state = state["model_state_dict"] if isinstance(state, dict) and "model_state_dict" in state else state
    missing, unexpected = p3.load_state_dict(state, strict=False)
    if missing: log.warning("Phase3 missing keys: %s", missing[:5])
    if unexpected: log.warning("Phase3 unexpected keys: %s", unexpected[:5])
    p3 = p3.eval().to(device)

    log.info("Loading APOBEC1 head...")
    a1 = APOBEC1Head()
    a1.load_state_dict(torch.load(APOBEC1_CKPT, weights_only=False, map_location="cpu"))
    a1 = a1.eval().to(device)

    n = len(orig)
    x = np.empty((n, D_INPUT), dtype=np.float32)
    x[:, :EMB_DIM] = orig.astype(np.float32)
    x[:, EMB_DIM:2 * EMB_DIM] = delta.astype(np.float32)
    x[:, 2 * EMB_DIM:] = hand40
    x[~valid] = 0.0

    HEADS = ["phase3_binary"] + [f"phase3_{e}" for e in ENZYMES] + ["apobec1"]
    scores = {h: np.zeros(n, dtype=np.float32) for h in HEADS}

    B = 512
    with torch.no_grad():
        for i in range(0, n, B):
            j = min(i + B, n)
            xt = torch.from_numpy(x[i:j]).to(device)
            shared = p3.shared_encoder(xt)
            scores["phase3_binary"][i:j] = torch.sigmoid(p3.binary_head(shared).squeeze(-1)).cpu().numpy()
            for enz in ENZYMES:
                scores[f"phase3_{enz}"][i:j] = torch.sigmoid(
                    p3.enzyme_adapters[enz](shared).squeeze(-1)).cpu().numpy()
            sp = torch.zeros((j - i, 1), dtype=torch.float32, device=device)
            scores["apobec1"][i:j] = torch.sigmoid(a1(shared, sp)).cpu().numpy()
    return scores


# ---------------------------------------------------------------------------
# Step 6: Enrichment + bootstrap
# ---------------------------------------------------------------------------
def bootstrap_or(scores, is_pos, thr, n_boot=N_BOOTSTRAP, rng=None):
    if rng is None: rng = np.random.default_rng(SEED)
    n_pos = is_pos.sum()
    n_ctrl = (~is_pos).sum()
    pos_idx = np.where(is_pos)[0]
    ctrl_idx = np.where(~is_pos)[0]
    ors = []
    for _ in range(n_boot):
        pi = rng.choice(pos_idx, n_pos, replace=True)
        ci = rng.choice(ctrl_idx, n_ctrl, replace=True)
        pa = (scores[pi] >= thr).sum()
        pb = n_pos - pa
        ca = (scores[ci] >= thr).sum()
        cb = n_ctrl - ca
        if pb > 0 and ca > 0:
            ors.append((pa * cb) / (pb * ca))
    if not ors:
        return np.nan, np.nan
    lo, hi = np.percentile(ors, [2.5, 97.5])
    return float(lo), float(hi)


def compute_enrichment(scores_dict, labels):
    is_pos = (labels == "positive").values
    n_pos = is_pos.sum()
    n_ctrl = (~is_pos).sum()
    log.info("Enrichment: %d pos vs %d ctrl", n_pos, n_ctrl)
    rows = []
    for head, scores in scores_dict.items():
        for pct in PERCENTILES:
            thr = np.percentile(scores, pct)
            above = scores >= thr
            pa = int((is_pos & above).sum())
            pb = int((is_pos & ~above).sum())
            ca = int((~is_pos & above).sum())
            cb = int((~is_pos & ~above).sum())
            try:
                or_val = (pa * cb) / (pb * ca) if pb > 0 and ca > 0 else (
                    float("inf") if pb == 0 else 0.0)
            except ZeroDivisionError:
                or_val = float("nan")
            _, p = fisher_exact([[pa, pb], [ca, cb]])
            ci_lo, ci_hi = bootstrap_or(scores, is_pos, thr)
            rows.append({
                "head": head, "percentile": pct, "threshold": float(thr),
                "n_pos_above": pa, "n_pos_below": pb,
                "n_ctrl_above": ca, "n_ctrl_below": cb,
                "or": or_val, "p_value": float(p),
                "or_ci_lo": ci_lo, "or_ci_hi": ci_hi,
            })
    df = pd.DataFrame(rows)
    # BH-FDR correction across all (head, percentile) tests
    df["q_value"] = multipletests(df["p_value"].fillna(1.0), method="fdr_bh")[1]
    return df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    t_global = time.time()
    log.info("=" * 60)
    log.info("Lei BE4max preview — pre-registered run")
    log.info("=" * 60)

    log.info("Opening hg38...")
    genome = Fasta(str(HG38), as_raw=True)

    log.info("Loading Lei positives...")
    pos = load_lei_positives()
    pos = attach_seq_and_filter(pos, genome)
    log.info("Positives with valid 201-nt C-centered window: %d", len(pos))

    log.info("Generating controls...")
    ctrl = gen_controls(pos, genome)
    ctrl = attach_seq_and_filter(ctrl, genome)
    log.info("Controls with valid window: %d", len(ctrl))

    all_df = pd.concat([pos, ctrl], ignore_index=True)
    log.info("Total sites: %d (%d pos + %d ctrl)", len(all_df), len(pos), len(ctrl))

    log.info("Computing hand40 features (parallel ViennaRNA)...")
    hand40, hand_valid = compute_hand40(all_df["seq"].tolist(), n_workers=8)

    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    log.info("Device: %s", device)

    log.info("Computing RNA-FM embeddings...")
    orig, delta = compute_rnafm(all_df["seq"].tolist(), device, batch=16)

    valid = hand_valid & (orig.sum(axis=1) != 0)
    log.info("Valid sites: %d / %d", valid.sum(), len(all_df))

    log.info("Scoring with all heads...")
    scores = score_all(orig, delta, hand40, valid, device)

    # Save raw scores
    scored_df = all_df[["chrom", "pos", "strand", "sgRNA", "cell", "label"]].copy()
    for h, s in scores.items():
        scored_df[f"score_{h}"] = s
    scored_df["valid"] = valid
    scored_df.to_csv(OUT / "scored_sites.csv", index=False)
    log.info("Wrote %s", OUT / "scored_sites.csv")

    # Filter to valid for enrichment
    valid_mask = scored_df["valid"].values
    valid_df = scored_df[valid_mask].reset_index(drop=True)
    valid_scores = {h: valid_df[f"score_{h}"].values for h in scores}

    log.info("Computing enrichment...")
    enrich = compute_enrichment(valid_scores, valid_df["label"])
    enrich_path = OUT / "enrichment.csv"
    enrich.to_csv(enrich_path, index=False)
    log.info("Wrote %s", enrich_path)

    # Summary JSON
    summary = {
        "n_positives_input": int(len(pos)),
        "n_controls_input": int(len(ctrl)),
        "n_valid_total": int(valid.sum()),
        "n_pos_valid": int(((valid_df["label"] == "positive") & valid_df["valid"]).sum()),
        "n_ctrl_valid": int(((valid_df["label"] == "control") & valid_df["valid"]).sum()),
        "wall_clock_min": (time.time() - t_global) / 60,
        "results_at_p90": enrich[enrich["percentile"] == 90].to_dict(orient="records"),
    }
    with open(OUT / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=float)
    log.info("Wrote %s", OUT / "summary.json")

    print("\n" + "=" * 60)
    print("Top-line p90 results (sorted by OR descending):")
    print("=" * 60)
    p90 = enrich[enrich["percentile"] == 90].sort_values("or", ascending=False)
    print(p90[["head", "or", "or_ci_lo", "or_ci_hi", "p_value", "q_value",
               "n_pos_above", "n_ctrl_above"]].to_string(index=False))

    log.info("Total time: %.1f min", (time.time() - t_global) / 60)


if __name__ == "__main__":
    main()
