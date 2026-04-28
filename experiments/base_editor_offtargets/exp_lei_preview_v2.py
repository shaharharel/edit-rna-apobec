#!/usr/bin/env python3
"""Lei BE4max gRNA-independent off-target enrichment — v2 (post-QA fixes).

Fixes from v1 (per QA agent + user direction):
1. BED parsed as 22-nt windows (not single C positions). Per Option A:
   ALL Cs within each window (forward + reverse strand) are treated as
   candidate deamination sites = positives. Loses no input rows.
2. dist_to_junction (loop slot 2) properly computed via canonical
   src/data/apobec_feature_extraction.py logic.
3. APOBEC1 head scored with BOTH species=1 (mouse APOBEC1 channel,
   matched to BE4max=rAPOBEC1) AND species=0 (human Neither channel).
4. Two control sets: (a) same-region motif-matched ±5kb,
   (b) global motif-matched random Cs anywhere on hg38.
5. Sanity check: report TCN fraction in positives + each control set.
6. Motif-only baseline: trinucleotide-frequency-derived score per C.

Pre-registered: PRE_REGISTRATION_lei_preview.md (v1 hypotheses still apply).
"""
from __future__ import annotations
import gzip
import json
import logging
import multiprocessing as mp
import random
import time
from collections import Counter
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
OUT = ROOT / "experiments/base_editor_offtargets/outputs/lei_preview_v2"
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
log = logging.getLogger("lei_v2")

COMP = str.maketrans("ACGTN", "TGCAN")


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
# Step 1: Parse Lei BEDs as 22-nt windows; expand to per-C positives
# ---------------------------------------------------------------------------
def _revcomp(s: str) -> str:
    return s.translate(COMP)[::-1]


def parse_pRBS_windows() -> pd.DataFrame:
    """Read all 4 _pRBS BEDs, return DataFrame: chrom, w_start, w_end, sgRNA, cell, off_target_index."""
    rows = []
    for sgRNA, fn, cell in LEI_FILES:
        path = LEI_DIR / fn
        with gzip.open(path, "rt") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or line.startswith("track"):
                    continue
                parts = line.split("\t")
                try:
                    start = int(parts[1])
                    end = int(parts[2])
                except (ValueError, IndexError):
                    continue
                ot_index = parts[3] if len(parts) > 3 else ""
                rows.append({
                    "chrom": parts[0], "w_start": start, "w_end": end,
                    "sgRNA": sgRNA, "cell": cell, "ot_index": ot_index,
                })
    df = pd.DataFrame(rows)
    log.info("Parsed %d Lei pRBS windows", len(df))
    return df


def expand_windows_to_positives(windows: pd.DataFrame, genome: Fasta) -> pd.DataFrame:
    """For each window, find all Cs (forward + reverse strand within window).
    Each C → one positive site with strand recorded.
    """
    out = []
    seen = set()  # (chrom, pos, strand)
    skipped_bounds = 0
    for _, w in windows.iterrows():
        try:
            chrom_len = len(genome[w.chrom])
        except KeyError:
            continue
        if w.w_start < 0 or w.w_end > chrom_len:
            skipped_bounds += 1
            continue
        win_seq = str(genome[w.chrom][w.w_start:w.w_end]).upper()
        for offset, base in enumerate(win_seq):
            cpos = w.w_start + offset
            if base == "C":
                key = (w.chrom, cpos, "+")
                if key not in seen:
                    seen.add(key)
                    out.append({
                        "chrom": w.chrom, "pos": cpos, "strand": "+",
                        "sgRNA": w.sgRNA, "cell": w.cell, "ot_index": w.ot_index,
                        "label": "positive",
                    })
            elif base == "G":  # C on reverse strand
                key = (w.chrom, cpos, "-")
                if key not in seen:
                    seen.add(key)
                    out.append({
                        "chrom": w.chrom, "pos": cpos, "strand": "-",
                        "sgRNA": w.sgRNA, "cell": w.cell, "ot_index": w.ot_index,
                        "label": "positive",
                    })
    df = pd.DataFrame(out)
    log.info("Expanded to %d unique candidate Cs (skipped %d windows out-of-bounds)",
             len(df), skipped_bounds)
    return df


# ---------------------------------------------------------------------------
# Step 2: Sequence + window extraction
# ---------------------------------------------------------------------------
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
        seq = _revcomp(seq)
    if len(seq) != 201 or seq[CENTER] != "C":
        return None
    return seq


def attach_seq(df: pd.DataFrame, genome: Fasta) -> pd.DataFrame:
    seqs = [get_seq(genome, r.chrom, r.pos, r.strand) for r in df.itertuples()]
    df = df.copy()
    df["seq"] = seqs
    df = df[df["seq"].notna()].reset_index(drop=True)
    return df


def trinuc_at_center(seq: str) -> str:
    return seq[CENTER - 1] + seq[CENTER] + seq[CENTER + 1]


# ---------------------------------------------------------------------------
# Step 3: Generate controls
# ---------------------------------------------------------------------------
def gen_same_region_controls(positives: pd.DataFrame, genome: Fasta) -> pd.DataFrame:
    """5 motif-matched Cs within ±5kb of each positive. Strand inherited."""
    rng = random.Random(SEED)
    pos_set = set(zip(positives["chrom"], positives["pos"], positives["strand"]))
    out = []
    skipped = 0
    for _, r in positives.iterrows():
        trinuc = trinuc_at_center(r.seq)  # XCY on the positive's strand
        try:
            chrom_len = len(genome[r.chrom])
        except KeyError:
            skipped += 1; continue
        ws = max(0, r.pos - HALF_KB)
        we = min(chrom_len, r.pos + HALF_KB)
        chunk_fwd = str(genome[r.chrom][ws:we]).upper()
        chunk = chunk_fwd if r.strand == "+" else _revcomp(chunk_fwd)
        cands = []
        for i in range(1, len(chunk) - 1):
            if chunk[i] == "C" and chunk[i - 1] == trinuc[0] and chunk[i + 1] == trinuc[2]:
                if r.strand == "+":
                    cpos = ws + i
                else:
                    cpos = we - 1 - i  # map back to fwd strand coordinate
                if abs(cpos - r.pos) < MIN_GAP:
                    continue
                if (r.chrom, cpos, r.strand) in pos_set:
                    continue
                cands.append(cpos)
        if len(cands) < N_CONTROLS:
            skipped += 1
            continue
        for cpos in rng.sample(cands, N_CONTROLS):
            out.append({
                "chrom": r.chrom, "pos": cpos, "strand": r.strand,
                "sgRNA": r.sgRNA, "cell": r.cell, "ot_index": r.ot_index,
                "label": "control_local", "matched_to": r.pos,
            })
    df = pd.DataFrame(out)
    log.info("Same-region controls: %d (skipped %d positives lacking matches)",
             len(df), skipped)
    return df


def gen_global_controls(positives: pd.DataFrame, genome: Fasta,
                         chroms: list[str], n_per_pos: int = N_CONTROLS) -> pd.DataFrame:
    """Random Cs on hg38, motif-matched per positive's trinuc context."""
    rng = random.Random(SEED + 1)
    pos_set = set(zip(positives["chrom"], positives["pos"], positives["strand"]))
    # Group positives by trinuc to amortize random sampling
    by_tri = {}
    for r in positives.itertuples():
        by_tri.setdefault(trinuc_at_center(r.seq), []).append(r)
    out = []
    chrom_lens = {c: len(genome[c]) for c in chroms}
    chrom_choices = list(chrom_lens.keys())
    chrom_weights = [chrom_lens[c] for c in chrom_choices]
    total_w = sum(chrom_weights)
    chrom_probs = [w / total_w for w in chrom_weights]
    rng_np = np.random.default_rng(SEED + 1)
    for trinuc, group in by_tri.items():
        n_needed = len(group) * n_per_pos
        # Oversample then filter
        n_try = n_needed * 20
        attempts = 0
        chosen = []
        while len(chosen) < n_needed and attempts < n_try:
            ci = rng_np.choice(len(chrom_choices), size=200, p=chrom_probs)
            for k in ci:
                chrom = chrom_choices[k]
                pos = rng_np.integers(110, chrom_lens[chrom] - 110)
                # Sample strand uniformly
                strand = "+" if rng_np.random() < 0.5 else "-"
                # Get just trinuc
                if strand == "+":
                    s = str(genome[chrom][pos - 1:pos + 2]).upper()
                else:
                    s = _revcomp(str(genome[chrom][pos - 1:pos + 2]).upper())
                if len(s) != 3 or s[1] != "C":
                    continue
                if s != trinuc:
                    continue
                if (chrom, pos, strand) in pos_set:
                    continue
                chosen.append((chrom, pos, strand))
                if len(chosen) >= n_needed:
                    break
            attempts += 1
        log.info("Global trinuc %s: collected %d/%d after %d batches",
                 trinuc, len(chosen), n_needed, attempts)
        # Distribute across the positives in this group
        for i, p in enumerate(group):
            for j in range(n_per_pos):
                idx = i * n_per_pos + j
                if idx >= len(chosen):
                    break
                chrom, cpos, strand = chosen[idx]
                out.append({
                    "chrom": chrom, "pos": cpos, "strand": strand,
                    "sgRNA": p.sgRNA, "cell": p.cell, "ot_index": p.ot_index,
                    "label": "control_global", "matched_to": p.pos,
                })
    df = pd.DataFrame(out)
    log.info("Global controls: %d (target %d)", len(df), len(positives) * n_per_pos)
    return df


# ---------------------------------------------------------------------------
# Step 4: Hand40 features (motif 24 + struct delta zeros 7 + loop 9 with FIXED dist_to_junction)
# ---------------------------------------------------------------------------
def _stem_length(s: str, idx: int, side: str) -> int:
    # walk from idx in the given direction while in '(' or ')'
    n = len(s)
    cnt = 0
    if side == "left":
        i = idx
        while i >= 0 and s[i] in "()":
            cnt += 1; i -= 1
    else:
        i = idx
        while i < n and s[i] in "()":
            cnt += 1; i += 1
    return cnt


def _compute_hand_one(seq: str):
    import RNA
    seq_u = seq.replace("T", "U")

    # --- motif (24-d) ---
    motif = np.zeros(24, dtype=np.float32)
    bases = ["A", "C", "G", "U"]
    if CENTER > 0:
        for j, m in enumerate(["UC", "CC", "AC", "GC"]):
            if seq_u[CENTER - 1] + "C" == m: motif[j] = 1.0
    if CENTER < len(seq_u) - 1:
        for j, m in enumerate(["CA", "CG", "CU", "CC"]):
            if "C" + seq_u[CENTER + 1] == m: motif[4 + j] = 1.0
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

    # --- struct delta zeroed (MFE-only regime) ---
    struct_delta = np.zeros(7, dtype=np.float32)

    # --- loop (9-d) with FIXED dist_to_junction (slot 2) ---
    loop = np.zeros(9, dtype=np.float32)
    valid = True
    try:
        s, _ = RNA.fold(seq_u)
        n = len(s)
        is_unpaired = s[CENTER] == "."
        loop[0] = float(is_unpaired)
        # local unpaired fraction (slot 8)
        w_start = max(0, CENTER - 10)
        w_end = min(n, CENTER + 11)
        local_region = s[w_start:w_end]
        loop[8] = sum(1 for c in local_region if c == ".") / len(local_region)
        if is_unpaired:
            left = CENTER - 1
            while left >= 0 and s[left] == ".":
                left -= 1
            right = CENTER + 1
            while right < n and s[right] == ".":
                right += 1
            loop_start = (left + 1) if left >= 0 else 0
            loop_end = (right - 1) if right < n else n - 1
            loop_size = loop_end - loop_start + 1
            dist_left = CENTER - loop_start
            dist_right = loop_end - CENTER
            relative_pos = dist_left / (loop_size - 1) if loop_size > 1 else 0.5
            apex = (loop_start + loop_end) / 2.0
            dist_to_apex = abs(CENTER - apex)
            left_stem = _stem_length(s, left, "left")
            right_stem = _stem_length(s, right, "right")
            loop[1] = loop_size
            loop[2] = float(min(dist_left, dist_right))  # FIXED: dist_to_junction
            loop[3] = dist_to_apex
            loop[4] = relative_pos
            loop[5] = left_stem
            loop[6] = right_stem
            loop[7] = max(left_stem, right_stem)
        else:
            dist_left = 0
            i = CENTER - 1
            while i >= 0 and s[i] in "()":
                dist_left += 1; i -= 1
            dist_right = 0
            j = CENTER + 1
            while j < n and s[j] in "()":
                dist_right += 1; j += 1
            loop[2] = float(min(dist_left, dist_right))  # FIXED: dist_to_junction (paired branch)
            stem_left = _stem_length(s, CENTER, "left")
            stem_right = _stem_length(s, CENTER, "right")
            loop[5] = stem_left
            loop[6] = stem_right
            loop[7] = max(stem_left, stem_right)
    except Exception:
        valid = False

    return np.concatenate([motif, struct_delta, loop]).astype(np.float32), valid


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
# Step 5: RNA-FM embeddings
# ---------------------------------------------------------------------------
def compute_rnafm(seqs: list[str], device: torch.device, batch: int = 16):
    import fm
    log.info("Loading RNA-FM (rna_fm_t12)...")
    model, alphabet = fm.pretrained.rna_fm_t12()
    bc = alphabet.get_batch_converter()
    model = model.eval().to(device)
    n = len(seqs)
    orig = np.zeros((n, EMB_DIM), dtype=np.float16)
    delta = np.zeros((n, EMB_DIM), dtype=np.float16)
    t0 = time.time()
    with torch.no_grad():
        for i in range(0, n, batch):
            j = min(i + batch, n)
            data_o = [(f"o{k}", seqs[k].replace("T", "U")) for k in range(i, j)]
            data_e = [(f"e{k}", seqs[k][:CENTER].replace("T", "U") + "U" + seqs[k][CENTER + 1:].replace("T", "U")) for k in range(i, j)]
            _, _, t_o = bc(data_o); _, _, t_e = bc(data_e)
            t_o = t_o.to(device); t_e = t_e.to(device)
            r_o = model(t_o, repr_layers=[12])["representations"][12]
            r_e = model(t_e, repr_layers=[12])["representations"][12]
            emb_o = r_o[:, 1:-1, :].mean(dim=1).cpu().float().numpy()
            emb_e = r_e[:, 1:-1, :].mean(dim=1).cpu().float().numpy()
            orig[i:j] = emb_o.astype(np.float16)
            delta[i:j] = (emb_e - emb_o).astype(np.float16)
            if i % (batch * 20) == 0:
                log.info("  RNA-FM %d/%d (%.1fs)", j, n, time.time() - t0)
    log.info("RNA-FM done in %.1fs", time.time() - t0)
    return orig, delta


# ---------------------------------------------------------------------------
# Step 6: Score with all heads (BOTH species=0 and species=1 for APOBEC1)
# ---------------------------------------------------------------------------
def score_all(orig, delta, hand40, valid, device):
    log.info("Loading Phase3...")
    p3 = Phase3Model()
    state = torch.load(PHASE3_CKPT, weights_only=False, map_location="cpu")
    state = state["model_state_dict"] if isinstance(state, dict) and "model_state_dict" in state else state
    p3.load_state_dict(state, strict=False)
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

    HEADS = ["phase3_binary"] + [f"phase3_{e}" for e in ENZYMES] + ["apobec1_sp1", "apobec1_sp0"]
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
            sp1 = torch.ones((j - i, 1), dtype=torch.float32, device=device)
            sp0 = torch.zeros((j - i, 1), dtype=torch.float32, device=device)
            scores["apobec1_sp1"][i:j] = torch.sigmoid(a1(shared, sp1)).cpu().numpy()
            scores["apobec1_sp0"][i:j] = torch.sigmoid(a1(shared, sp0)).cpu().numpy()
    return scores


# ---------------------------------------------------------------------------
# Step 7: Motif-only baseline (per-trinuc empirical preference)
# ---------------------------------------------------------------------------
def motif_only_score(seqs: list[str], pos_trinuc_freq: dict[str, float]) -> np.ndarray:
    """Score = empirical fraction of Lei positives at this trinuc context.
    Higher when trinuc is favored by BE4max (TC contexts).
    """
    return np.array([pos_trinuc_freq.get(trinuc_at_center(s), 0.0) for s in seqs],
                    dtype=np.float32)


# ---------------------------------------------------------------------------
# Step 8: Enrichment
# ---------------------------------------------------------------------------
def bootstrap_or(scores, is_pos, thr, n_boot=N_BOOTSTRAP, rng=None):
    if rng is None: rng = np.random.default_rng(SEED)
    n_pos = is_pos.sum(); n_ctrl = (~is_pos).sum()
    pos_idx = np.where(is_pos)[0]; ctrl_idx = np.where(~is_pos)[0]
    ors = []
    for _ in range(n_boot):
        pi = rng.choice(pos_idx, n_pos, replace=True)
        ci = rng.choice(ctrl_idx, n_ctrl, replace=True)
        pa = (scores[pi] >= thr).sum(); pb = n_pos - pa
        ca = (scores[ci] >= thr).sum(); cb = n_ctrl - ca
        if pb > 0 and ca > 0:
            ors.append((pa * cb) / (pb * ca))
    if not ors: return np.nan, np.nan
    return tuple(np.percentile(ors, [2.5, 97.5]))


def enrichment_table(scores_dict, labels, control_label, name):
    is_pos = (labels == "positive").values
    is_ctrl = (labels == control_label).values
    keep = is_pos | is_ctrl
    is_pos_k = is_pos[keep]
    rows = []
    for head, scores in scores_dict.items():
        sc = scores[keep]
        n_pos = is_pos_k.sum()
        n_ctrl = (~is_pos_k).sum()
        for pct in PERCENTILES:
            thr = float(np.percentile(sc, pct))
            above = sc >= thr
            pa = int((is_pos_k & above).sum())
            pb = int((is_pos_k & ~above).sum())
            ca = int((~is_pos_k & above).sum())
            cb = int((~is_pos_k & ~above).sum())
            try:
                or_val = (pa * cb) / (pb * ca) if pb > 0 and ca > 0 else (
                    float("inf") if pb == 0 else 0.0)
            except ZeroDivisionError:
                or_val = float("nan")
            _, p = fisher_exact([[pa, pb], [ca, cb]])
            ci_lo, ci_hi = bootstrap_or(sc, is_pos_k, thr)
            rows.append({
                "control_set": name, "head": head, "percentile": pct,
                "threshold": thr,
                "n_pos_above": pa, "n_pos_below": pb,
                "n_ctrl_above": ca, "n_ctrl_below": cb,
                "or": or_val, "p_value": float(p),
                "or_ci_lo": float(ci_lo), "or_ci_hi": float(ci_hi),
            })
    df = pd.DataFrame(rows)
    df["q_value"] = multipletests(df["p_value"].fillna(1.0), method="fdr_bh")[1]
    return df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    t0 = time.time()
    log.info("=" * 70)
    log.info("Lei BE4max preview v2 — post-QA fixes")
    log.info("=" * 70)

    log.info("Opening hg38...")
    genome = Fasta(str(HG38), as_raw=True)

    # Step 1: parse pRBS windows
    windows = parse_pRBS_windows()
    pos_df = expand_windows_to_positives(windows, genome)
    pos_df = attach_seq(pos_df, genome)
    log.info("Positives (per-C, with valid 201-nt window): %d", len(pos_df))

    # Sanity: TCN fraction
    pos_trinucs = [trinuc_at_center(s) for s in pos_df["seq"]]
    pos_tri_counts = Counter(pos_trinucs)
    n_pos = len(pos_df)
    pos_tri_freq = {t: c / n_pos for t, c in pos_tri_counts.items()}
    pos_tcn_frac = sum(c for t, c in pos_tri_counts.items() if t.startswith("U") or t.startswith("T"))
    pos_tcn_frac = (sum(c for t, c in pos_tri_counts.items() if t[0] in "UT")) / n_pos
    log.info("SANITY: positives TCN fraction = %.1f%% (BE4max expected ~70-80%%)",
             100 * pos_tcn_frac)

    # Step 2: same-region controls
    log.info("Generating same-region controls...")
    ctrl_local = gen_same_region_controls(pos_df, genome)
    ctrl_local = attach_seq(ctrl_local, genome)

    # Step 3: global controls
    log.info("Generating global controls...")
    chroms = [f"chr{i}" for i in list(range(1, 23)) + ["X", "Y"]]
    chroms = [c for c in chroms if c in genome.keys()]
    ctrl_global = gen_global_controls(pos_df, genome, chroms)
    ctrl_global = attach_seq(ctrl_global, genome)

    log.info("Final counts: pos=%d local_ctrl=%d global_ctrl=%d",
             len(pos_df), len(ctrl_local), len(ctrl_global))

    local_tcn = sum(1 for s in ctrl_local["seq"] if trinuc_at_center(s)[0] in "UT") / max(len(ctrl_local), 1)
    global_tcn = sum(1 for s in ctrl_global["seq"] if trinuc_at_center(s)[0] in "UT") / max(len(ctrl_global), 1)
    log.info("SANITY: local control TCN%% = %.1f%%, global control TCN%% = %.1f%%",
             100 * local_tcn, 100 * global_tcn)

    all_df = pd.concat([pos_df, ctrl_local, ctrl_global], ignore_index=True)
    log.info("Total sites: %d", len(all_df))

    # Step 4: hand40 features
    log.info("Computing hand40 features...")
    hand40, hand_valid = compute_hand40(all_df["seq"].tolist(), n_workers=8)

    # Step 5: RNA-FM
    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    log.info("Device: %s", device)
    log.info("Computing RNA-FM embeddings...")
    orig, delta = compute_rnafm(all_df["seq"].tolist(), device, batch=16)

    valid = hand_valid & (orig.sum(axis=1) != 0)
    log.info("Valid sites: %d / %d", valid.sum(), len(all_df))

    # Step 6: score
    log.info("Scoring with all heads (incl. APOBEC1 species=0 AND species=1)...")
    scores = score_all(orig, delta, hand40, valid, device)

    # Step 7: motif-only baseline (using positive empirical trinuc frequency)
    motif_scores = motif_only_score(all_df["seq"].tolist(), pos_tri_freq)
    scores["motif_only"] = motif_scores

    # Save scored sites
    scored_df = all_df[["chrom", "pos", "strand", "sgRNA", "cell", "label",
                         "ot_index", "matched_to"]].copy() if "matched_to" in all_df.columns \
                 else all_df[["chrom", "pos", "strand", "sgRNA", "cell", "label"]].copy()
    for h, s in scores.items():
        scored_df[f"score_{h}"] = s
    scored_df["valid"] = valid
    scored_df["trinuc"] = [trinuc_at_center(s) for s in all_df["seq"]]
    scored_df.to_csv(OUT / "scored_sites.csv", index=False)
    log.info("Wrote %s", OUT / "scored_sites.csv")

    # Step 8: enrichment - both control sets
    valid_df = scored_df[valid].reset_index(drop=True)
    valid_scores = {h: valid_df[f"score_{h}"].values for h in scores}

    log.info("Enrichment vs same-region controls...")
    enrich_local = enrichment_table(valid_scores, valid_df["label"], "control_local", "same_region")
    log.info("Enrichment vs global controls...")
    enrich_global = enrichment_table(valid_scores, valid_df["label"], "control_global", "global")
    enrich_all = pd.concat([enrich_local, enrich_global], ignore_index=True)
    enrich_all["q_value"] = multipletests(enrich_all["p_value"].fillna(1.0), method="fdr_bh")[1]
    enrich_all.to_csv(OUT / "enrichment.csv", index=False)
    log.info("Wrote %s", OUT / "enrichment.csv")

    # Summary
    summary = {
        "n_windows": int(len(windows)),
        "n_positives_per_c": int(len(pos_df)),
        "n_local_controls": int(len(ctrl_local)),
        "n_global_controls": int(len(ctrl_global)),
        "n_valid": int(valid.sum()),
        "pos_tcn_pct": 100 * pos_tcn_frac,
        "local_ctrl_tcn_pct": 100 * local_tcn,
        "global_ctrl_tcn_pct": 100 * global_tcn,
        "wall_clock_min": (time.time() - t0) / 60,
        "top10_motifs_in_positives": pos_tri_counts.most_common(10),
    }
    with open(OUT / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    log.info("Wrote %s", OUT / "summary.json")

    print("\n" + "=" * 70)
    print("SANITY CHECKS")
    print("=" * 70)
    print(f"Positives per-C: {len(pos_df)} (from {len(windows)} windows)")
    print(f"TCN fraction: positives={100*pos_tcn_frac:.1f}%, local={100*local_tcn:.1f}%, global={100*global_tcn:.1f}%")
    print(f"Top-5 trinuc in positives: {pos_tri_counts.most_common(5)}")
    print()

    print("=" * 70)
    print("p90 enrichment vs SAME-REGION controls (motif-matched, ±5kb)")
    print("=" * 70)
    p90l = enrich_local[enrich_local["percentile"] == 90].sort_values("or", ascending=False)
    print(p90l[["head", "or", "or_ci_lo", "or_ci_hi", "p_value", "q_value"]].to_string(index=False))

    print()
    print("=" * 70)
    print("p90 enrichment vs GLOBAL controls (motif-matched, anywhere)")
    print("=" * 70)
    p90g = enrich_global[enrich_global["percentile"] == 90].sort_values("or", ascending=False)
    print(p90g[["head", "or", "or_ci_lo", "or_ci_hi", "p_value", "q_value"]].to_string(index=False))

    log.info("Total wall clock: %.1f min", (time.time() - t0) / 60)


if __name__ == "__main__":
    main()
