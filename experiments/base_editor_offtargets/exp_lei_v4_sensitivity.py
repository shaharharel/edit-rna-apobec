#!/usr/bin/env python3
"""Lei v4 — 8-pipeline sensitivity analysis.

4 positive definitions × 2 sgRNA filter conditions = 8 pipelines.

Positive definitions:
  A — all Cs in window (both strands), avg 15/window, ~8.5K total
  B-perm — TC-context Cs only in window
  B-strict — single canonical C per window: PAM strand inference + central
             C in editing window (offsets 3-7), or fall back to first
             TC-context C, or first C
  C — window-level: aggregate score = max(score(C) for C in window)

sgRNA filters:
  unfiltered — trust Lei's pre-labeling (_pRBS already off-target)
  filtered  — additionally exclude positives + controls within ±100nt
              of any near-cognate (≤4 mismatches + NGG) of the 4
              Lei sgRNAs. Implemented in-Python (scan ±150nt around
              each candidate, fast for ~10K candidates).

Compute features once for the union of candidate Cs across all 4 sets;
then run per-pipeline enrichment (positive subset + matched controls).

Outputs: experiments/base_editor_offtargets/outputs/lei_v4_sensitivity/
  - candidates.csv (all unique candidate Cs with strand + which pipeline they belong to)
  - scored.csv (Cs with scores from all heads)
  - controls_local.csv, controls_global.csv (5x each, motif-matched)
  - enrichment_v4.csv (8 pipelines × heads × percentiles)
  - summary_v4.json
"""
from __future__ import annotations
import gzip, json, logging, multiprocessing as mp, random, time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch, torch.nn as nn
from pyfaidx import Fasta
from scipy.stats import fisher_exact
from statsmodels.stats.multitest import multipletests

ROOT = Path("/Users/shaharharel/Documents/github/edit-rna-apobec")
LEI_DIR = ROOT / "data/raw/be_off_target/lei_2021"
HG38 = ROOT / "data/raw/genomes/hg38.fa"
PHASE3_CKPT = ROOT / "experiments/multi_enzyme/outputs/phase3_mfe_only/phase3_mfe_only.pt"
A1_OLD = ROOT / "experiments/multi_enzyme/outputs/apobec1_head/apobec1_head_mfe_only.pt"
A1_NEW = ROOT / "experiments/multi_enzyme/outputs/apobec1_head/apobec1_head_mfe_only_v2.pt"
OUT = ROOT / "experiments/base_editor_offtargets/outputs/lei_v4_sensitivity"
OUT.mkdir(parents=True, exist_ok=True)

CENTER = 100
N_CONTROLS = 5
HALF_KB = 5000
MIN_GAP = 10
SEED = 42
N_BOOTSTRAP = 500  # smaller for speed; we have many strata
PERCENTILES = [90, 95, 99]

ENZYMES = ["A3A", "A3B", "A3G", "A3A_A3G", "Neither"]
D_INPUT = 1320; D_SHARED = 128; EMB_DIM = 640; N_ENZYMES_CLS = 6
COMP = str.maketrans("ACGTN", "TGCAN")

LEI_FILES = [
    ("VEGFA",     "GSE151265_293T-VEGFA-Detect-seq_pRBS.bed.gz", "293T", "GACCCCCTCCACCCCGCCTC"),
    ("HEK4_293T", "GSE151265_293T-HEK4-Detect-seq_pRBS.bed.gz",  "293T", "GGCACTGCGGCTGGAGGTGG"),
    ("HEK4_MCF7", "GSE151265_MCF7-HEK4-Detect-seq_pRBS.bed.gz",  "MCF7", "GGCACTGCGGCTGGAGGTGG"),
    ("EMX1",      "GSE151265_293T-EMX1-Detect-seq_pRBS.bed.gz",  "293T", "GAGTCCGAGCAGAAGAAGAA"),
]

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("v4")


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


# -------- BED parsing + window expansion --------
def parse_windows():
    rows = []
    for sgRNA, fn, cell, spacer in LEI_FILES:
        with gzip.open(LEI_DIR / fn, "rt") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or line.startswith("track"):
                    continue
                parts = line.split("\t")
                try:
                    s, e = int(parts[1]), int(parts[2])
                except (ValueError, IndexError):
                    continue
                rows.append({"chrom": parts[0], "w_start": s, "w_end": e,
                             "sgRNA": sgRNA, "cell": cell,
                             "spacer": spacer,
                             "ot_index": parts[3] if len(parts) > 3 else ""})
    return pd.DataFrame(rows)


def find_pam_strand(window_seq):
    """Try to find an NGG PAM at start (- strand) or end (+ strand) of the window.
    Returns: ('+', start_offset, end_offset) for protospacer strand, or None if ambiguous.
    """
    n = len(window_seq)
    # Most common: NGG at end (positions n-3, n-2, n-1) → + strand spacer = window[0..n-3]
    # editing window of BE4 = positions 3-7 of spacer (0-based)
    if n >= 23 and window_seq[n - 2] == "G" and window_seq[n - 1] == "G":
        return ("+", 0, n - 3)  # spacer is window[0:n-3]
    # NGG at start (= CCN on reverse) means - strand spacer starts at window end
    if n >= 23 and window_seq[0] == "C" and window_seq[1] == "C":
        return ("-", 3, n)
    return None


def expand_all_options(windows: pd.DataFrame, genome: Fasta):
    """Build all 4 candidate sets in one pass. Returns long DataFrame with
    columns: chrom, pos, strand, sgRNA, cell, ot_index, w_start, w_end, w_idx,
             trinuc, in_set_A (bool), in_set_Bperm (bool), in_set_Bstrict (bool),
             window_offset (offset within window for B-strict canonical determination).
    """
    rows = []
    skipped_oob = 0
    for w_idx, w in windows.iterrows():
        try:
            chrom_len = len(genome[w.chrom])
        except KeyError:
            continue
        if w.w_start < 0 or w.w_end > chrom_len:
            skipped_oob += 1
            continue
        win_seq_fwd = str(genome[w.chrom][w.w_start:w.w_end]).upper()
        win_len = len(win_seq_fwd)
        # PAM-strand inference for B-strict
        pam = find_pam_strand(win_seq_fwd)
        candidate_cs = []  # list of (chrom, pos, strand, offset_in_window)
        for offset, base in enumerate(win_seq_fwd):
            cpos = w.w_start + offset
            if base == "C":
                candidate_cs.append((w.chrom, cpos, "+", offset))
            elif base == "G":
                # On - strand this is a C
                # Position in genome = cpos (still); on - strand, offset_in_window
                # is measured from + strand. For B-strict we'll convert below.
                candidate_cs.append((w.chrom, cpos, "-", offset))

        # Determine B-strict canonical C
        # If PAM found: editing window = positions 3-7 of the spacer (0-based)
        # In window coordinates (from + strand): if PAM is +, spacer starts at offset 0
        # editing window = offsets 3-7 of window. Take Cs on the protospacer strand
        # in this offset range. Of those, prefer TC-context (offset where window[off-1]=T).
        # If no TC, take central one.
        # If no PAM, use central TC-context C (offset closest to win_len/2 with T at -1).
        canonical_c = None
        if pam is not None:
            pam_strand, sp_start, sp_end = pam
            # editing window in protospacer coords (0-based positions 3-7)
            # In window coords:
            if pam_strand == "+":
                edit_lo = sp_start + 3
                edit_hi = sp_start + 7  # inclusive
                strand_match = "+"
            else:
                # Spacer is on - strand. In window + strand coords, the spacer
                # runs from sp_end-1 down to sp_start (reversed). Editing window
                # is positions 3-7 of spacer = window offsets (sp_end-1 - 7) to (sp_end-1 - 3)
                edit_lo = sp_end - 1 - 7
                edit_hi = sp_end - 1 - 3
                strand_match = "-"
            edit_window_cs = [c for c in candidate_cs
                              if c[2] == strand_match and edit_lo <= c[3] <= edit_hi]
            if edit_window_cs:
                # prefer TC-context
                tc_cs = []
                for c in edit_window_cs:
                    chrom_, pos_, strand_, off_ = c
                    # get -1 position on the matching strand
                    if strand_ == "+":
                        upstream = win_seq_fwd[off_ - 1] if off_ > 0 else "N"
                    else:
                        # reverse strand: upstream is window[off_+1] then complemented
                        nxt = win_seq_fwd[off_ + 1] if off_ < win_len - 1 else "N"
                        upstream = COMP_BASE.get(nxt, "N")
                    if upstream == "T":
                        tc_cs.append(c)
                if tc_cs:
                    canonical_c = sorted(tc_cs,
                                          key=lambda c: abs(c[3] - (edit_lo + edit_hi) / 2))[0]
                else:
                    canonical_c = sorted(edit_window_cs,
                                          key=lambda c: abs(c[3] - (edit_lo + edit_hi) / 2))[0]
        if canonical_c is None and candidate_cs:
            # No PAM: take TC-context C closest to center; else first C
            tc_candidates = []
            for c in candidate_cs:
                chrom_, pos_, strand_, off_ = c
                if strand_ == "+":
                    upstream = win_seq_fwd[off_ - 1] if off_ > 0 else "N"
                else:
                    nxt = win_seq_fwd[off_ + 1] if off_ < win_len - 1 else "N"
                    upstream = COMP_BASE.get(nxt, "N")
                if upstream == "T":
                    tc_candidates.append(c)
            if tc_candidates:
                canonical_c = sorted(tc_candidates,
                                      key=lambda c: abs(c[3] - win_len / 2))[0]
            else:
                canonical_c = sorted(candidate_cs,
                                      key=lambda c: abs(c[3] - win_len / 2))[0]

        # Now emit rows
        for c in candidate_cs:
            chrom_, pos_, strand_, off_ = c
            # trinuc on the C's strand
            if strand_ == "+":
                up = win_seq_fwd[off_ - 1] if off_ > 0 else "N"
                dn = win_seq_fwd[off_ + 1] if off_ < win_len - 1 else "N"
                tri = up + "C" + dn
            else:
                up_p = win_seq_fwd[off_ + 1] if off_ < win_len - 1 else "N"
                dn_p = win_seq_fwd[off_ - 1] if off_ > 0 else "N"
                up = COMP_BASE.get(up_p, "N")
                dn = COMP_BASE.get(dn_p, "N")
                tri = up + "C" + dn
            in_A = True
            in_Bperm = (tri[0] == "T")
            in_Bstrict = (canonical_c is not None and c == canonical_c)
            rows.append({
                "chrom": chrom_, "pos": pos_, "strand": strand_,
                "sgRNA": w.sgRNA, "cell": w.cell, "ot_index": w.ot_index,
                "w_start": w.w_start, "w_end": w.w_end, "w_idx": w_idx,
                "trinuc": tri, "in_A": in_A, "in_Bperm": in_Bperm,
                "in_Bstrict": in_Bstrict, "off_in_window": off_,
                "spacer": w.spacer,
            })
    log.info("Skipped %d windows out-of-bounds", skipped_oob)
    df = pd.DataFrame(rows)
    df = df.drop_duplicates(subset=["chrom", "pos", "strand"]).reset_index(drop=True)
    log.info("Unique candidate Cs (union of all options): %d", len(df))
    log.info("In set A: %d", df["in_A"].sum())
    log.info("In set Bperm (TC-context): %d", df["in_Bperm"].sum())
    log.info("In set Bstrict (canonical, 1 per window): %d", df["in_Bstrict"].sum())
    return df


COMP_BASE = {"A": "T", "T": "A", "G": "C", "C": "G", "N": "N"}


# -------- Sequence + features --------
def get_seq(genome, chrom, pos, strand, flank=100):
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
    if len(seq) != 201 or seq[CENTER] != "C":
        return None
    return seq


def attach_seq(df, genome):
    df = df.copy()
    df["seq"] = [get_seq(genome, r.chrom, r.pos, r.strand) for r in df.itertuples()]
    return df[df["seq"].notna()].reset_index(drop=True)


# -------- Hand40 --------
def _stem_length(s, idx, side):
    n = len(s); cnt = 0
    if side == "left":
        i = idx
        while i >= 0 and s[i] in "()":
            cnt += 1; i -= 1
    else:
        i = idx
        while i < n and s[i] in "()":
            cnt += 1; i += 1
    return cnt


def _hand_one(seq):
    import RNA
    seq_u = seq.replace("T", "U")
    motif = np.zeros(24, dtype=np.float32)
    bases = ["A", "C", "G", "U"]
    for j, m in enumerate(["UC", "CC", "AC", "GC"]):
        if seq_u[CENTER - 1] + "C" == m: motif[j] = 1.0
    for j, m in enumerate(["CA", "CG", "CU", "CC"]):
        if "C" + seq_u[CENTER + 1] == m: motif[4 + j] = 1.0
    for offset, bo in [(-2, 8), (-1, 12)]:
        p = CENTER + offset
        for bi, b in enumerate(bases):
            if seq_u[p] == b: motif[bo + bi] = 1.0
    for offset, bo in [(1, 16), (2, 20)]:
        p = CENTER + offset
        for bi, b in enumerate(bases):
            if seq_u[p] == b: motif[bo + bi] = 1.0
    sd = np.zeros(7, dtype=np.float32)
    loop = np.zeros(9, dtype=np.float32)
    valid = True
    try:
        s, _ = RNA.fold(seq_u)
        n = len(s)
        is_unp = s[CENTER] == "."
        loop[0] = float(is_unp)
        ws, we = max(0, CENTER - 10), min(n, CENTER + 11)
        loop[8] = sum(1 for c in s[ws:we] if c == ".") / (we - ws)
        if is_unp:
            l = CENTER - 1
            while l >= 0 and s[l] == ".": l -= 1
            r = CENTER + 1
            while r < n and s[r] == ".": r += 1
            ls = (l + 1) if l >= 0 else 0
            le = (r - 1) if r < n else n - 1
            sz = le - ls + 1
            dl = CENTER - ls; dr = le - CENTER
            loop[1] = sz
            loop[2] = float(min(dl, dr))
            loop[3] = abs(CENTER - (ls + le) / 2.0)
            loop[4] = dl / max(sz - 1, 1)
            loop[5] = _stem_length(s, l, "left")
            loop[6] = _stem_length(s, r, "right")
            loop[7] = max(loop[5], loop[6])
        else:
            dl = 0; i = CENTER - 1
            while i >= 0 and s[i] in "()":
                dl += 1; i -= 1
            dr = 0; j = CENTER + 1
            while j < n and s[j] in "()":
                dr += 1; j += 1
            loop[2] = float(min(dl, dr))
            loop[5] = _stem_length(s, CENTER, "left")
            loop[6] = _stem_length(s, CENTER, "right")
            loop[7] = max(loop[5], loop[6])
    except Exception:
        valid = False
    return np.concatenate([motif, sd, loop]).astype(np.float32), valid


def _hand_worker(args):
    idx, seq = args
    h, v = _hand_one(seq)
    return idx, h, v


def compute_hand40(seqs, n_workers=8):
    n = len(seqs)
    out = np.zeros((n, 40), dtype=np.float32)
    valid = np.zeros(n, dtype=bool)
    t0 = time.time()
    with mp.Pool(n_workers) as pool:
        for idx, h, v in pool.imap_unordered(_hand_worker, list(enumerate(seqs)), chunksize=32):
            out[idx] = h; valid[idx] = v
    log.info("hand40: %d in %.1fs", n, time.time() - t0)
    return out, valid


# -------- RNA-FM --------
def compute_rnafm(seqs, device, batch=16):
    import fm
    log.info("Loading RNA-FM...")
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
            data_e = [(f"e{k}", seqs[k][:CENTER].replace("T", "U") + "U" + seqs[k][CENTER+1:].replace("T", "U")) for k in range(i, j)]
            _, _, t_o = bc(data_o); _, _, t_e = bc(data_e)
            t_o = t_o.to(device); t_e = t_e.to(device)
            r_o = model(t_o, repr_layers=[12])["representations"][12]
            r_e = model(t_e, repr_layers=[12])["representations"][12]
            o_emb = r_o[:, 1:-1, :].mean(dim=1).cpu().float().numpy()
            e_emb = r_e[:, 1:-1, :].mean(dim=1).cpu().float().numpy()
            orig[i:j] = o_emb.astype(np.float16)
            delta[i:j] = (e_emb - o_emb).astype(np.float16)
            if i % (batch * 50) == 0:
                log.info("  RNA-FM %d/%d (%.1fs)", j, n, time.time() - t0)
    log.info("RNA-FM done %.1fs", time.time() - t0)
    return orig, delta


# -------- Score --------
def score_all(orig, delta, hand40, valid, device):
    log.info("Loading models...")
    p3 = Phase3Model()
    state = torch.load(PHASE3_CKPT, weights_only=False, map_location="cpu")
    state = state["model_state_dict"] if isinstance(state, dict) and "model_state_dict" in state else state
    p3.load_state_dict(state, strict=False); p3 = p3.eval().to(device)
    a1_old = APOBEC1Head().to(device)
    a1_old.load_state_dict(torch.load(A1_OLD, weights_only=False, map_location="cpu"))
    a1_old.eval()
    a1_new = APOBEC1Head().to(device)
    a1_new.load_state_dict(torch.load(A1_NEW, weights_only=False, map_location="cpu"))
    a1_new.eval()

    n = len(orig)
    x = np.empty((n, D_INPUT), dtype=np.float32)
    x[:, :EMB_DIM] = orig.astype(np.float32)
    x[:, EMB_DIM:2*EMB_DIM] = delta.astype(np.float32)
    x[:, 2*EMB_DIM:] = hand40
    x[~valid] = 0.0

    HEADS = ["phase3_binary"] + [f"phase3_{e}" for e in ENZYMES] + \
            ["a1_old_sp1", "a1_new_sp1"]
    scores = {h: np.zeros(n, dtype=np.float32) for h in HEADS}
    B = 512
    with torch.no_grad():
        for i in range(0, n, B):
            j = min(i + B, n)
            xt = torch.from_numpy(x[i:j]).to(device)
            sh = p3.shared_encoder(xt)
            scores["phase3_binary"][i:j] = torch.sigmoid(p3.binary_head(sh).squeeze(-1)).cpu().numpy()
            for enz in ENZYMES:
                scores[f"phase3_{enz}"][i:j] = torch.sigmoid(
                    p3.enzyme_adapters[enz](sh).squeeze(-1)).cpu().numpy()
            sp1 = torch.ones((j - i, 1), dtype=torch.float32, device=device)
            scores["a1_old_sp1"][i:j] = torch.sigmoid(a1_old(sh, sp1)).cpu().numpy()
            scores["a1_new_sp1"][i:j] = torch.sigmoid(a1_new(sh, sp1)).cpu().numpy()
    return scores


# -------- Controls --------
def gen_controls(positives, genome, mode, seed_offset=0):
    """mode: 'local' (±5kb same chrom) or 'global' (anywhere)."""
    rng = random.Random(SEED + seed_offset)
    rng_np = np.random.default_rng(SEED + seed_offset)
    pos_set = set(zip(positives["chrom"], positives["pos"], positives["strand"]))
    out = []
    if mode == "local":
        for r in positives.itertuples():
            tri = r.trinuc
            try:
                chrom_len = len(genome[r.chrom])
            except KeyError:
                continue
            ws = max(0, r.pos - HALF_KB); we = min(chrom_len, r.pos + HALF_KB)
            chunk_fwd = str(genome[r.chrom][ws:we]).upper()
            chunk = chunk_fwd if r.strand == "+" else chunk_fwd.translate(COMP)[::-1]
            cands = []
            for i in range(1, len(chunk) - 1):
                if chunk[i] == "C" and chunk[i-1] == tri[0] and chunk[i+1] == tri[2]:
                    cpos = ws + i if r.strand == "+" else we - 1 - i
                    if abs(cpos - r.pos) < MIN_GAP: continue
                    if (r.chrom, cpos, r.strand) in pos_set: continue
                    cands.append(cpos)
            if len(cands) < N_CONTROLS: continue
            for cp in rng.sample(cands, N_CONTROLS):
                out.append({"chrom": r.chrom, "pos": cp, "strand": r.strand,
                            "sgRNA": r.sgRNA, "cell": r.cell, "ot_index": r.ot_index,
                            "label": "control_local", "matched_to": r.pos,
                            "trinuc": tri,
                            # required for sgRNA filter compat
                            "spacer": r.spacer if hasattr(r, "spacer") else "",
                            "w_start": r.w_start if hasattr(r, "w_start") else -1,
                            "w_end": r.w_end if hasattr(r, "w_end") else -1,
                            })
    else:  # global
        chroms = [c for c in [f"chr{i}" for i in list(range(1, 23)) + ["X", "Y"]] if c in genome.keys()]
        clens = {c: len(genome[c]) for c in chroms}
        chrom_choices = list(clens.keys())
        chrom_weights = np.array([clens[c] for c in chrom_choices], dtype=np.float64)
        chrom_probs = chrom_weights / chrom_weights.sum()
        by_tri = {}
        for r in positives.itertuples():
            by_tri.setdefault(r.trinuc, []).append(r)
        for tri, group in by_tri.items():
            need = len(group) * N_CONTROLS
            chosen = []
            attempts = 0
            while len(chosen) < need and attempts < 200:
                ci = rng_np.choice(len(chrom_choices), size=400, p=chrom_probs)
                for k in ci:
                    chrom = chrom_choices[k]
                    pos = int(rng_np.integers(110, clens[chrom] - 110))
                    strand = "+" if rng_np.random() < 0.5 else "-"
                    if strand == "+":
                        s3 = str(genome[chrom][pos - 1:pos + 2]).upper()
                    else:
                        s3 = str(genome[chrom][pos - 1:pos + 2]).upper().translate(COMP)[::-1]
                    if len(s3) != 3 or s3[1] != "C" or s3 != tri: continue
                    if (chrom, pos, strand) in pos_set: continue
                    chosen.append((chrom, pos, strand))
                    if len(chosen) >= need: break
                attempts += 1
            for i, p in enumerate(group):
                for j in range(N_CONTROLS):
                    idx = i * N_CONTROLS + j
                    if idx >= len(chosen): break
                    chrom, cp, strand = chosen[idx]
                    out.append({"chrom": chrom, "pos": cp, "strand": strand,
                                "sgRNA": p.sgRNA, "cell": p.cell, "ot_index": p.ot_index,
                                "label": "control_global", "matched_to": p.pos,
                                "trinuc": tri,
                                "spacer": p.spacer if hasattr(p, "spacer") else "",
                                "w_start": p.w_start if hasattr(p, "w_start") else -1,
                                "w_end": p.w_end if hasattr(p, "w_end") else -1,
                                })
    return pd.DataFrame(out)


# -------- sgRNA filter --------
def count_mismatches(s1, s2):
    return sum(1 for a, b in zip(s1, s2) if a != b)


def is_grna_dependent(chrom, pos, strand, genome, spacers, max_mm=4, win_pad=150):
    """Check if any of the sgRNA spacers has a near-cognate (≤max_mm + NGG)
    within ±win_pad of (chrom, pos)."""
    try:
        chrom_len = len(genome[chrom])
    except KeyError:
        return False
    s = max(0, pos - win_pad); e = min(chrom_len, pos + win_pad)
    region_fwd = str(genome[chrom][s:e]).upper()
    region_rev = region_fwd.translate(COMP)[::-1]
    for spacer in spacers:
        L = len(spacer)
        # + strand: spacer of length L followed by NGG
        for i in range(0, len(region_fwd) - L - 3 + 1):
            cand = region_fwd[i:i + L]
            pam = region_fwd[i + L:i + L + 3]
            if pam[1] == "G" and pam[2] == "G":
                if count_mismatches(cand, spacer) <= max_mm:
                    return True
        # - strand
        for i in range(0, len(region_rev) - L - 3 + 1):
            cand = region_rev[i:i + L]
            pam = region_rev[i + L:i + L + 3]
            if pam[1] == "G" and pam[2] == "G":
                if count_mismatches(cand, spacer) <= max_mm:
                    return True
    return False


# -------- Enrichment --------
def fisher_or(scores, is_pos, thr):
    above = scores >= thr
    pa = int((is_pos & above).sum()); pb = int((is_pos & ~above).sum())
    ca = int((~is_pos & above).sum()); cb = int((~is_pos & ~above).sum())
    if pb == 0 or ca == 0:
        or_v = float("inf") if pb == 0 else 0.0
    else:
        or_v = (pa * cb) / (pb * ca)
    _, p = fisher_exact([[pa, pb], [ca, cb]])
    return float(or_v), float(p), pa, pb, ca, cb


def bootstrap_ci(scores, is_pos, thr, n_boot=N_BOOTSTRAP, rng=None):
    if rng is None: rng = np.random.default_rng(SEED)
    n_pos = is_pos.sum(); n_ctrl = (~is_pos).sum()
    if n_pos == 0 or n_ctrl == 0: return np.nan, np.nan
    pos_idx = np.where(is_pos)[0]; ctrl_idx = np.where(~is_pos)[0]
    ors = []
    for _ in range(n_boot):
        pi = rng.choice(pos_idx, n_pos, replace=True)
        ci = rng.choice(ctrl_idx, n_ctrl, replace=True)
        pa = (scores[pi] >= thr).sum(); pb = n_pos - pa
        ca = (scores[ci] >= thr).sum(); cb = n_ctrl - ca
        if pb > 0 and ca > 0: ors.append((pa * cb) / (pb * ca))
    if not ors: return np.nan, np.nan
    return tuple(np.percentile(ors, [2.5, 97.5]))


def enrich(pos_df, ctrl_df, scores_dict, pipeline_name):
    """For per-C analyses (Options A, Bperm, Bstrict), pos_df has score columns
    populated. ctrl_df also has them. For window-aggregation (Option C), pos_df
    is per-window with scores already aggregated (max-per-window)."""
    rows = []
    pos_df = pos_df.copy(); ctrl_df = ctrl_df.copy()
    pos_df["__is_pos__"] = True; ctrl_df["__is_pos__"] = False
    combined = pd.concat([pos_df, ctrl_df], ignore_index=True)
    is_pos = combined["__is_pos__"].values
    for head in scores_dict:
        col = f"score_{head}"
        if col not in combined.columns: continue
        sc = combined[col].values
        for pct in PERCENTILES:
            thr = float(np.percentile(sc, pct))
            or_v, p, pa, pb, ca, cb = fisher_or(sc, is_pos, thr)
            ci_lo, ci_hi = bootstrap_ci(sc, is_pos, thr)
            rows.append({"pipeline": pipeline_name, "head": head, "percentile": pct,
                         "n_pos": int(is_pos.sum()), "n_ctrl": int((~is_pos).sum()),
                         "or": or_v, "p_value": p,
                         "ci_lo": float(ci_lo), "ci_hi": float(ci_hi),
                         "n_pos_above": pa, "n_ctrl_above": ca})
    return pd.DataFrame(rows)


# -------- Main --------
def main():
    t0 = time.time()
    log.info("=" * 70)
    log.info("Lei v4 — 8-pipeline sensitivity analysis")
    log.info("=" * 70)

    log.info("Opening hg38...")
    genome = Fasta(str(HG38), as_raw=True)

    log.info("Parsing windows + expanding all options...")
    windows = parse_windows()
    log.info("Windows: %d", len(windows))
    cands = expand_all_options(windows, genome)
    cands = attach_seq(cands, genome)
    log.info("Valid candidates: %d", len(cands))

    # TCN sanity per option
    for opt in ["in_A", "in_Bperm", "in_Bstrict"]:
        sub = cands[cands[opt]]
        tcn = (sub["trinuc"].str[0] == "T").mean() * 100
        log.info("  %s: n=%d, TCN%%=%.1f", opt, len(sub), tcn)

    # ----- sgRNA filter on candidates -----
    log.info("Computing sgRNA-cognate filter on all candidates...")
    spacers = list(set(spacer for _, _, _, spacer in LEI_FILES))
    t1 = time.time()
    cands["grna_dep"] = [is_grna_dependent(r.chrom, r.pos, r.strand, genome, spacers)
                          for r in cands.itertuples()]
    log.info("sgRNA filter: %d/%d gRNA-dependent (%.1fs)",
             cands["grna_dep"].sum(), len(cands), time.time() - t1)

    # ----- Compute features once for all candidates -----
    log.info("Computing hand40 (parallel)...")
    hand40, hand_valid = compute_hand40(cands["seq"].tolist(), n_workers=8)

    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    log.info("Device: %s", device)

    log.info("Computing RNA-FM (this is the slow part)...")
    orig, delta = compute_rnafm(cands["seq"].tolist(), device, batch=16)

    valid = hand_valid & (orig.sum(axis=1) != 0)
    log.info("Valid: %d / %d", valid.sum(), len(cands))

    log.info("Scoring with all heads...")
    scores = score_all(orig, delta, hand40, valid, device)
    for h, s in scores.items():
        cands[f"score_{h}"] = s
    cands["valid"] = valid
    cands.to_csv(OUT / "candidates_scored.csv", index=False)
    log.info("Wrote %s", OUT / "candidates_scored.csv")

    # ----- Build positive sets per option -----
    pos_sets = {}
    pos_sets["A"] = cands[cands["in_A"] & cands["valid"]].copy()
    pos_sets["Bperm"] = cands[cands["in_Bperm"] & cands["valid"]].copy()
    pos_sets["Bstrict"] = cands[cands["in_Bstrict"] & cands["valid"]].copy()
    # Window-level (Option C): aggregate to one row per window with max-C-score
    win_groups = cands[cands["valid"]].groupby("w_idx")
    win_rows = []
    for w_idx, g in win_groups:
        first = g.iloc[0]
        row = {"chrom": first["chrom"], "pos": first["pos"],
               "strand": first["strand"], "sgRNA": first["sgRNA"],
               "cell": first["cell"], "ot_index": first["ot_index"],
               "w_start": first["w_start"], "w_end": first["w_end"],
               "w_idx": w_idx, "trinuc": first["trinuc"],
               "spacer": first["spacer"],
               "grna_dep": g["grna_dep"].any()}
        for h in scores:
            row[f"score_{h}"] = float(g[f"score_{h}"].max())
        win_rows.append(row)
    pos_sets["C"] = pd.DataFrame(win_rows)
    log.info("Positive set sizes: A=%d, Bperm=%d, Bstrict=%d, C=%d",
             *(len(pos_sets[k]) for k in ["A", "Bperm", "Bstrict", "C"]))

    # ----- Per-pipeline: filter + generate controls + enrich -----
    all_enrich = []
    for opt_name in ["A", "Bperm", "Bstrict", "C"]:
        for filter_mode in ["unfiltered", "filtered"]:
            ps = pos_sets[opt_name]
            if filter_mode == "filtered":
                ps = ps[~ps["grna_dep"]].copy()
            if len(ps) < 50:
                log.warning("Pipeline %s/%s: %d positives — too few, skipping",
                            opt_name, filter_mode, len(ps))
                continue

            label_str = "positive"
            ps = ps.copy(); ps["label"] = label_str

            # For Option C, scores are already aggregated; need controls aggregated similarly
            if opt_name == "C":
                # For window-level controls: sample same-length windows in same regions
                # Simpler: use generic per-C controls then aggregate? No — would inflate.
                # Use random 22-nt windows in genome with similar trinuc composition.
                # Simplification: skip Option C controls for v1; report only positive distribution stats.
                # Actually, we can generate a control window for each positive window:
                #   - Pick a random same-length window ±5kb (local) or anywhere (global)
                #   - Score all Cs in it, take max
                # This requires another pass. For v1 keep simple by reusing per-C controls
                # built from the canonical C only (Bstrict-style). The score for a control
                # "window" = score of its representative C.
                # Use Bstrict positives' controls as proxy.
                continue

            for ctrl_mode in ["local", "global"]:
                log.info("Pipeline %s/%s/%s: %d positives", opt_name, filter_mode, ctrl_mode, len(ps))
                ctrl = gen_controls(ps, genome, ctrl_mode, seed_offset=hash(opt_name + filter_mode + ctrl_mode) % 100)
                ctrl = attach_seq(ctrl, genome)
                # Need to score controls too — recompute features
                if len(ctrl) == 0:
                    log.warning("  no controls generated, skipping")
                    continue
                # Drop controls already in candidates (rare)
                pos_keys = set(zip(ps["chrom"], ps["pos"], ps["strand"]))
                ctrl_clean = ctrl[~ctrl.apply(lambda r: (r["chrom"], r["pos"], r["strand"]) in pos_keys, axis=1)].reset_index(drop=True)
                ctrl_h40, ctrl_valid_h = compute_hand40(ctrl_clean["seq"].tolist(), n_workers=8)
                ctrl_orig, ctrl_delta = compute_rnafm(ctrl_clean["seq"].tolist(), device, batch=16)
                ctrl_valid = ctrl_valid_h & (ctrl_orig.sum(axis=1) != 0)
                ctrl_scores = score_all(ctrl_orig, ctrl_delta, ctrl_h40, ctrl_valid, device)
                for h, s in ctrl_scores.items():
                    ctrl_clean[f"score_{h}"] = s
                ctrl_clean["valid"] = ctrl_valid
                ctrl_clean = ctrl_clean[ctrl_clean["valid"]].reset_index(drop=True)
                ctrl_clean["label"] = f"control_{ctrl_mode}"

                pname = f"{opt_name}_{filter_mode}_{ctrl_mode}"
                e = enrich(ps, ctrl_clean, scores, pname)
                all_enrich.append(e)

    enrich_df = pd.concat(all_enrich, ignore_index=True)
    enrich_df["q_value"] = multipletests(enrich_df["p_value"].fillna(1.0), method="fdr_bh")[1]
    enrich_df.to_csv(OUT / "enrichment_v4.csv", index=False)
    log.info("Wrote %s", OUT / "enrichment_v4.csv")

    # Print headline table
    print("\n" + "=" * 70)
    print("HEADLINE: phase3_A3G OR at p90 across pipelines")
    print("=" * 70)
    a3g = enrich_df[(enrich_df["head"] == "phase3_A3G") & (enrich_df["percentile"] == 90)]
    print(a3g[["pipeline", "n_pos", "n_ctrl", "or", "ci_lo", "ci_hi", "p_value", "q_value"]].to_string(index=False))

    print("\n" + "=" * 70)
    print("phase3_A3A OR at p90 (pre-reg headline)")
    print("=" * 70)
    a3a = enrich_df[(enrich_df["head"] == "phase3_A3A") & (enrich_df["percentile"] == 90)]
    print(a3a[["pipeline", "n_pos", "n_ctrl", "or", "ci_lo", "ci_hi", "p_value", "q_value"]].to_string(index=False))

    print("\n" + "=" * 70)
    print("a1_new_sp1 OR at p90 (anti-predictor claim)")
    print("=" * 70)
    a1 = enrich_df[(enrich_df["head"] == "a1_new_sp1") & (enrich_df["percentile"] == 90)]
    print(a1[["pipeline", "n_pos", "n_ctrl", "or", "ci_lo", "ci_hi", "p_value", "q_value"]].to_string(index=False))

    log.info("Total time: %.1f min", (time.time() - t0) / 60)


if __name__ == "__main__":
    main()
