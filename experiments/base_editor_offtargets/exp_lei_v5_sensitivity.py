#!/usr/bin/env python3
"""Lei v5 — single-pass 8-pipeline sensitivity analysis.

Architecture (efficiency fix from v4):
1. Parse all 561 Lei windows. Expand to ALL unique candidate Cs (~10K).
   For each, mark which option(s) it belongs to: A / Bperm / Bstrict.
2. Generate 5 local + 5 global motif-matched controls per candidate C
   (ONCE — each candidate has its 10 controls regardless of pipeline).
3. Compute hand40 + RNA-FM ONCE for the union (candidates + all controls),
   ~110K rows total.
4. Score all heads ONCE.
5. Compute sgRNA-cognate flag (grna_dep) for every candidate + every control.
6. For each of 8 pipelines (3 per-C options × 2 filter × per-C; plus
   2 window-level pipelines for Option C) build subset and run enrichment.

Pipelines:
  A_unfiltered, A_filtered (per-C, all Cs)
  Bperm_unfiltered, Bperm_filtered (per-C, TC-only)
  Bstrict_unfiltered, Bstrict_filtered (per-C, canonical 1 per window)
  Cmax_unfiltered, Cmax_filtered (window-level, max-C-score)

Controls:
  local (±5kb same-region, motif-matched)
  global (anywhere on hg38, motif-matched)
  Both reported for every pipeline.

Compute estimate: ~50 min total on Mac MPS (110K RNA-FM passes).
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
OUT = ROOT / "experiments/base_editor_offtargets/outputs/lei_v5_sensitivity"
OUT.mkdir(parents=True, exist_ok=True)

CENTER = 100
N_CONTROLS = 5
HALF_KB = 5000
MIN_GAP = 10
SEED = 42
N_BOOTSTRAP = 500
PERCENTILES = [90, 95, 99]
GRNA_FILTER_PAD = 100  # ±nt around near-cognate to exclude
GRNA_MAX_MM = 4
GRNA_SCAN_PAD = 150  # scan ±nt around each candidate for spacers

ENZYMES = ["A3A", "A3B", "A3G", "A3A_A3G", "Neither"]
D_INPUT = 1320; D_SHARED = 128; EMB_DIM = 640; N_ENZYMES_CLS = 6
COMP_TBL = str.maketrans("ACGTN", "TGCAN")
COMP_BASE = {"A": "T", "T": "A", "G": "C", "C": "G", "N": "N"}

LEI_FILES = [
    ("VEGFA",     "GSE151265_293T-VEGFA-Detect-seq_pRBS.bed.gz", "293T", "GACCCCCTCCACCCCGCCTC"),
    ("HEK4_293T", "GSE151265_293T-HEK4-Detect-seq_pRBS.bed.gz",  "293T", "GGCACTGCGGCTGGAGGTGG"),
    ("HEK4_MCF7", "GSE151265_MCF7-HEK4-Detect-seq_pRBS.bed.gz",  "MCF7", "GGCACTGCGGCTGGAGGTGG"),
    ("EMX1",      "GSE151265_293T-EMX1-Detect-seq_pRBS.bed.gz",  "293T", "GAGTCCGAGCAGAAGAAGAA"),
]
SPACERS = list({sp for _, _, _, sp in LEI_FILES})

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("v5")


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
                             "sgRNA": sgRNA, "cell": cell, "spacer": spacer,
                             "ot_index": parts[3] if len(parts) > 3 else ""})
    return pd.DataFrame(rows)


def find_pam(window_seq):
    n = len(window_seq)
    if n >= 23 and window_seq[n - 2] == "G" and window_seq[n - 1] == "G":
        return ("+", 0, n - 3)
    if n >= 23 and window_seq[0] == "C" and window_seq[1] == "C":
        return ("-", 3, n)
    return None


def expand_candidates(windows, genome):
    rows = []
    for w_idx, w in windows.iterrows():
        try:
            chrom_len = len(genome[w.chrom])
        except KeyError:
            continue
        if w.w_start < 0 or w.w_end > chrom_len:
            continue
        win_seq = str(genome[w.chrom][w.w_start:w.w_end]).upper()
        n = len(win_seq)
        pam = find_pam(win_seq)
        cs = []  # (chrom, pos, strand, off, trinuc)
        for off, base in enumerate(win_seq):
            cpos = w.w_start + off
            if base == "C":
                up = win_seq[off - 1] if off > 0 else "N"
                dn = win_seq[off + 1] if off < n - 1 else "N"
                cs.append((w.chrom, cpos, "+", off, up + "C" + dn))
            elif base == "G":
                # C on - strand: trinuc on - strand
                up_p = win_seq[off + 1] if off < n - 1 else "N"
                dn_p = win_seq[off - 1] if off > 0 else "N"
                tri = COMP_BASE[up_p] + "C" + COMP_BASE[dn_p]
                cs.append((w.chrom, cpos, "-", off, tri))

        # Bstrict: canonical C
        canonical = None
        if pam is not None:
            pam_s, sp_s, sp_e = pam
            if pam_s == "+":
                el, eh = sp_s + 3, sp_s + 7
                want_strand = "+"
            else:
                el, eh = sp_e - 1 - 7, sp_e - 1 - 3
                want_strand = "-"
            edit_cs = [c for c in cs if c[2] == want_strand and el <= c[3] <= eh]
            if edit_cs:
                tc = [c for c in edit_cs if c[4][0] == "T"]
                pool = tc if tc else edit_cs
                canonical = min(pool, key=lambda c: abs(c[3] - (el + eh) / 2))
        if canonical is None and cs:
            tc = [c for c in cs if c[4][0] == "T"]
            pool = tc if tc else cs
            canonical = min(pool, key=lambda c: abs(c[3] - n / 2))

        for c in cs:
            chrom_, pos_, strand_, off_, tri = c
            rows.append({
                "chrom": chrom_, "pos": pos_, "strand": strand_, "trinuc": tri,
                "sgRNA": w.sgRNA, "cell": w.cell, "ot_index": w.ot_index,
                "spacer": w.spacer, "w_start": w.w_start, "w_end": w.w_end,
                "w_idx": int(w_idx), "off_in_window": off_,
                "in_A": True, "in_Bperm": tri[0] == "T",
                "in_Bstrict": (canonical is not None and c == canonical),
            })
    df = pd.DataFrame(rows)
    return df.drop_duplicates(subset=["chrom", "pos", "strand"]).reset_index(drop=True)


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
        seq = seq.translate(COMP_TBL)[::-1]
    if len(seq) != 201 or seq[CENTER] != "C":
        return None
    return seq


def trinuc(seq):
    return seq[CENTER - 1] + seq[CENTER] + seq[CENTER + 1]


# -------- Controls (single pass, generates 5 local + 5 global per candidate) --------
def gen_all_controls(candidates, genome):
    rng = random.Random(SEED)
    rng_np = np.random.default_rng(SEED)
    pos_set = set(zip(candidates["chrom"], candidates["pos"], candidates["strand"]))
    chroms = [c for c in [f"chr{i}" for i in list(range(1, 23)) + ["X", "Y"]] if c in genome.keys()]
    clens = {c: len(genome[c]) for c in chroms}
    chrom_choices = list(clens.keys())
    chrom_w = np.array([clens[c] for c in chrom_choices], dtype=np.float64)
    chrom_p = chrom_w / chrom_w.sum()

    out = []
    skipped_local = 0
    for r in candidates.itertuples():
        tri = r.trinuc
        # Local controls
        try:
            chrom_len = len(genome[r.chrom])
        except KeyError:
            continue
        ws = max(0, r.pos - HALF_KB); we = min(chrom_len, r.pos + HALF_KB)
        chunk_fwd = str(genome[r.chrom][ws:we]).upper()
        chunk = chunk_fwd if r.strand == "+" else chunk_fwd.translate(COMP_TBL)[::-1]
        cands = []
        for i in range(1, len(chunk) - 1):
            if chunk[i] == "C" and chunk[i-1] == tri[0] and chunk[i+1] == tri[2]:
                cp = ws + i if r.strand == "+" else we - 1 - i
                if abs(cp - r.pos) < MIN_GAP: continue
                if (r.chrom, cp, r.strand) in pos_set: continue
                cands.append(cp)
        if len(cands) < N_CONTROLS:
            skipped_local += 1
        else:
            for cp in rng.sample(cands, N_CONTROLS):
                out.append({
                    "chrom": r.chrom, "pos": cp, "strand": r.strand,
                    "trinuc": tri, "sgRNA": r.sgRNA, "cell": r.cell,
                    "ot_index": r.ot_index, "spacer": r.spacer,
                    "w_idx": r.w_idx, "matched_to": r.pos, "label": "control_local",
                })

    # Global: collect groups by trinuc and sample in batch
    by_tri = candidates.groupby("trinuc")
    log.info("Generating global controls (rejection sampling per trinuc)...")
    global_rows = []
    for tri, group in by_tri:
        need = len(group) * N_CONTROLS
        chosen = []
        attempts = 0
        max_attempts = 500
        while len(chosen) < need and attempts < max_attempts:
            ci = rng_np.choice(len(chrom_choices), size=400, p=chrom_p)
            for k in ci:
                chrom = chrom_choices[k]
                pos = int(rng_np.integers(110, clens[chrom] - 110))
                strand = "+" if rng_np.random() < 0.5 else "-"
                if strand == "+":
                    s3 = str(genome[chrom][pos - 1:pos + 2]).upper()
                else:
                    s3 = str(genome[chrom][pos - 1:pos + 2]).upper().translate(COMP_TBL)[::-1]
                if len(s3) != 3 or s3 != tri: continue
                if (chrom, pos, strand) in pos_set: continue
                chosen.append((chrom, pos, strand))
                if len(chosen) >= need: break
            attempts += 1
        log.info("  trinuc %s: collected %d/%d (%d attempts)", tri, len(chosen), need, attempts)
        for i, (_, p) in enumerate(group.iterrows()):
            for j in range(N_CONTROLS):
                idx = i * N_CONTROLS + j
                if idx >= len(chosen): break
                ch, cp, st = chosen[idx]
                global_rows.append({
                    "chrom": ch, "pos": cp, "strand": st,
                    "trinuc": tri, "sgRNA": p.sgRNA, "cell": p.cell,
                    "ot_index": p.ot_index, "spacer": p.spacer,
                    "w_idx": p.w_idx, "matched_to": p.pos, "label": "control_global",
                })
    out_df = pd.DataFrame(out)
    glob_df = pd.DataFrame(global_rows)
    log.info("Local controls: %d (skipped %d candidates lacking matches)", len(out_df), skipped_local)
    log.info("Global controls: %d", len(glob_df))
    return out_df, glob_df


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


# -------- RNA-FM (compute once) --------
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
            if i % (batch * 100) == 0:
                log.info("  RNA-FM %d/%d (%.1fs)", j, n, time.time() - t0)
    log.info("RNA-FM done %.1fs", time.time() - t0)
    return orig, delta


def score_all(orig, delta, hand40, valid, device):
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
    HEADS = ["phase3_binary"] + [f"phase3_{e}" for e in ENZYMES] + ["a1_old_sp1", "a1_new_sp1"]
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


# -------- sgRNA cognate detection --------
def count_mm(s1, s2):
    return sum(1 for a, b in zip(s1, s2) if a != b)


def grna_dep_check(chrom, pos, genome, spacers, max_mm=GRNA_MAX_MM, pad=GRNA_SCAN_PAD):
    try:
        cl = len(genome[chrom])
    except KeyError:
        return False
    s = max(0, pos - pad); e = min(cl, pos + pad)
    region_fwd = str(genome[chrom][s:e]).upper()
    region_rev = region_fwd.translate(COMP_TBL)[::-1]
    for spacer in spacers:
        L = len(spacer)
        for region in (region_fwd, region_rev):
            for i in range(0, len(region) - L - 3 + 1):
                cand = region[i:i + L]
                pam = region[i + L:i + L + 3]
                if pam[1:3] == "GG":
                    if count_mm(cand, spacer) <= max_mm:
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


def main():
    t0 = time.time()
    log.info("=" * 70)
    log.info("Lei v5 — single-pass 8-pipeline sensitivity")
    log.info("=" * 70)

    log.info("Opening hg38...")
    genome = Fasta(str(HG38), as_raw=True)

    log.info("Parsing windows + expanding all candidates...")
    windows = parse_windows()
    log.info("Windows: %d", len(windows))
    cands = expand_candidates(windows, genome)

    # Attach sequences
    cands["seq"] = [get_seq(genome, r.chrom, r.pos, r.strand) for r in cands.itertuples()]
    cands = cands[cands["seq"].notna()].reset_index(drop=True)
    log.info("Valid candidates: %d", len(cands))

    log.info("Sanity TCN%% per option:")
    for opt in ["in_A", "in_Bperm", "in_Bstrict"]:
        sub = cands[cands[opt]]
        tcn = (sub["trinuc"].str[0] == "T").mean() * 100
        log.info("  %s: n=%d, TCN%%=%.1f", opt, len(sub), tcn)

    # Generate all controls
    log.info("Generating local + global controls (5x each per candidate)...")
    ctrl_local, ctrl_global = gen_all_controls(cands, genome)
    ctrl_local["seq"] = [get_seq(genome, r.chrom, r.pos, r.strand) for r in ctrl_local.itertuples()]
    ctrl_global["seq"] = [get_seq(genome, r.chrom, r.pos, r.strand) for r in ctrl_global.itertuples()]
    ctrl_local = ctrl_local[ctrl_local["seq"].notna()].reset_index(drop=True)
    ctrl_global = ctrl_global[ctrl_global["seq"].notna()].reset_index(drop=True)
    log.info("Controls: local=%d, global=%d", len(ctrl_local), len(ctrl_global))

    # Mark labels on candidates
    cands["label"] = "positive"

    # sgRNA cognate flag for everyone
    log.info("Computing sgRNA-cognate flags for all rows (positives + controls)...")
    t1 = time.time()
    cands["grna_dep"] = [grna_dep_check(r.chrom, r.pos, genome, SPACERS) for r in cands.itertuples()]
    log.info("Candidates: %d/%d gRNA-dependent (%.1fs)", cands["grna_dep"].sum(), len(cands), time.time() - t1)
    t1 = time.time()
    ctrl_local["grna_dep"] = [grna_dep_check(r.chrom, r.pos, genome, SPACERS) for r in ctrl_local.itertuples()]
    ctrl_global["grna_dep"] = [grna_dep_check(r.chrom, r.pos, genome, SPACERS) for r in ctrl_global.itertuples()]
    log.info("Controls grna_dep: local=%d/%d, global=%d/%d (%.1fs)",
             ctrl_local["grna_dep"].sum(), len(ctrl_local),
             ctrl_global["grna_dep"].sum(), len(ctrl_global), time.time() - t1)

    # Combine all into one DataFrame with a tag
    cands["src"] = "candidate"
    ctrl_local["src"] = "ctrl_local"
    ctrl_global["src"] = "ctrl_global"
    # Standardize columns
    common_cols = ["chrom", "pos", "strand", "trinuc", "sgRNA", "cell",
                   "ot_index", "spacer", "w_idx", "label", "grna_dep", "src", "seq"]
    extra_cands = ["in_A", "in_Bperm", "in_Bstrict", "off_in_window"]
    extra_ctrl = ["matched_to"]
    for c in extra_cands + extra_ctrl:
        if c not in cands.columns: cands[c] = pd.NA
        if c not in ctrl_local.columns: ctrl_local[c] = pd.NA
        if c not in ctrl_global.columns: ctrl_global[c] = pd.NA
    full_cols = common_cols + extra_cands + extra_ctrl
    all_df = pd.concat([cands[full_cols], ctrl_local[full_cols], ctrl_global[full_cols]],
                       ignore_index=True)
    log.info("Total rows for feature compute: %d", len(all_df))

    # Compute features
    log.info("Computing hand40...")
    hand40, hand_valid = compute_hand40(all_df["seq"].tolist(), n_workers=8)

    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    log.info("Device: %s", device)

    log.info("Computing RNA-FM (single pass for %d sequences)...", len(all_df))
    orig, delta = compute_rnafm(all_df["seq"].tolist(), device, batch=16)
    valid = hand_valid & (orig.sum(axis=1) != 0)
    log.info("Valid: %d/%d", valid.sum(), len(all_df))

    log.info("Scoring...")
    scores = score_all(orig, delta, hand40, valid, device)
    for h, s in scores.items():
        all_df[f"score_{h}"] = s
    all_df["valid"] = valid
    all_df.to_csv(OUT / "all_scored.csv", index=False)
    log.info("Wrote %s", OUT / "all_scored.csv")

    # Build pipelines
    score_cols = [f"score_{h}" for h in scores]
    valid_df = all_df[valid].reset_index(drop=True)

    def get_subset(option_filter, only_grna_indep):
        if option_filter == "A":
            sub = valid_df[(valid_df["src"] == "candidate") & (valid_df["in_A"] == True)]
        elif option_filter == "Bperm":
            sub = valid_df[(valid_df["src"] == "candidate") & (valid_df["in_Bperm"] == True)]
        elif option_filter == "Bstrict":
            sub = valid_df[(valid_df["src"] == "candidate") & (valid_df["in_Bstrict"] == True)]
        else:
            return None
        if only_grna_indep:
            sub = sub[~sub["grna_dep"]]
        return sub.copy()

    def get_controls(option_filter, ctrl_mode, only_grna_indep):
        # Controls are from ctrl_local or ctrl_global (matched per-candidate)
        # Filter to controls whose matched_to candidate is in the option's positive set
        if option_filter == "A":
            allowed_pos = set(zip(valid_df[(valid_df["src"] == "candidate") & valid_df["in_A"]]["chrom"],
                                  valid_df[(valid_df["src"] == "candidate") & valid_df["in_A"]]["pos"]))
        elif option_filter == "Bperm":
            allowed_pos = set(zip(valid_df[(valid_df["src"] == "candidate") & valid_df["in_Bperm"]]["chrom"],
                                  valid_df[(valid_df["src"] == "candidate") & valid_df["in_Bperm"]]["pos"]))
        elif option_filter == "Bstrict":
            allowed_pos = set(zip(valid_df[(valid_df["src"] == "candidate") & valid_df["in_Bstrict"]]["chrom"],
                                  valid_df[(valid_df["src"] == "candidate") & valid_df["in_Bstrict"]]["pos"]))
        ctrl = valid_df[valid_df["src"] == f"ctrl_{ctrl_mode}"]
        ctrl = ctrl[ctrl.apply(lambda r: (r["chrom"], r["matched_to"]) in allowed_pos, axis=1)]
        if only_grna_indep:
            ctrl = ctrl[~ctrl["grna_dep"]]
        return ctrl.copy()

    enrich_rows = []
    for opt in ["A", "Bperm", "Bstrict"]:
        for filt_mode in ["unfiltered", "filtered"]:
            only_indep = (filt_mode == "filtered")
            ps = get_subset(opt, only_indep)
            if len(ps) < 50:
                log.warning("Skipping %s/%s: only %d positives", opt, filt_mode, len(ps))
                continue
            for ctrl_mode in ["local", "global"]:
                ctrl = get_controls(opt, ctrl_mode, only_indep)
                if len(ctrl) < 50:
                    log.warning("Skipping %s/%s/%s: only %d controls", opt, filt_mode, ctrl_mode, len(ctrl))
                    continue
                pname = f"{opt}_{filt_mode}_{ctrl_mode}"
                log.info("Pipeline %s: pos=%d ctrl=%d", pname, len(ps), len(ctrl))
                # Compute enrichment
                combined = pd.concat([ps.assign(__pos__=True), ctrl.assign(__pos__=False)],
                                      ignore_index=True)
                is_pos = combined["__pos__"].values
                for h in scores:
                    sc = combined[f"score_{h}"].values
                    for pct in PERCENTILES:
                        thr = float(np.percentile(sc, pct))
                        or_v, p, pa, pb, ca, cb = fisher_or(sc, is_pos, thr)
                        ci_lo, ci_hi = bootstrap_ci(sc, is_pos, thr)
                        enrich_rows.append({
                            "pipeline": pname, "option": opt, "filter": filt_mode,
                            "ctrl_set": ctrl_mode, "head": h, "percentile": pct,
                            "n_pos": int(is_pos.sum()), "n_ctrl": int((~is_pos).sum()),
                            "or": or_v, "p_value": p,
                            "ci_lo": float(ci_lo), "ci_hi": float(ci_hi),
                            "n_pos_above": pa, "n_ctrl_above": ca,
                        })

    # Add Cmax pipelines
    log.info("Computing window-level (Cmax) aggregation...")
    win_scores = {}  # w_idx → max score per head over all positive Cs in that window
    for h in scores:
        win_scores[h] = valid_df[valid_df["src"] == "candidate"].groupby("w_idx")[f"score_{h}"].max()
    cands_w = valid_df[(valid_df["src"] == "candidate")].groupby("w_idx").first().reset_index()
    for h in scores:
        cands_w[f"score_{h}"] = cands_w["w_idx"].map(win_scores[h])
    # For window-level controls: aggregate per (matched_to → window) too. Each window has 5 control Cs.
    # Use the max over those 5 as the control-window score.
    log.info("  Aggregating controls per matched_to window...")
    for ctrl_mode in ["local", "global"]:
        ctrl_pool = valid_df[valid_df["src"] == f"ctrl_{ctrl_mode}"]
        # Group by matched_to (each pos C had 5 controls)
        # We aggregate to one "matched window control" per candidate window
        # But controls are per-C, not per-window. Each window has many Cs (~15), each has 5 controls.
        # → collapse all controls of any C in that window to a single window-control max-of-max
        ctrl_w = ctrl_pool.copy()
        # find which window each control is matched to
        ctrl_w["w_idx_match"] = ctrl_w["w_idx"].astype("Int64")
        for h in scores:
            agg = ctrl_w.groupby("w_idx_match")[f"score_{h}"].max()
            cands_w[f"ctrl_{ctrl_mode}_score_{h}"] = cands_w["w_idx"].map(agg)
        cands_w[f"ctrl_{ctrl_mode}_grna_dep"] = cands_w["w_idx"].map(
            ctrl_w.groupby("w_idx_match")["grna_dep"].any())
    # For Cmax pipelines: the "controls" are the per-window aggregated control scores
    for filt_mode in ["unfiltered", "filtered"]:
        only_indep = (filt_mode == "filtered")
        cw = cands_w.copy()
        if only_indep:
            cw = cw[~cw["grna_dep"]]
        if len(cw) < 50: continue
        for ctrl_mode in ["local", "global"]:
            ctrl_score_cols = [f"ctrl_{ctrl_mode}_score_{h}" for h in scores]
            cw_present = cw.dropna(subset=ctrl_score_cols)
            if only_indep:
                cw_present = cw_present[~cw_present[f"ctrl_{ctrl_mode}_grna_dep"]]
            if len(cw_present) < 50: continue
            pname = f"Cmax_{filt_mode}_{ctrl_mode}"
            log.info("Pipeline %s: pos_windows=%d", pname, len(cw_present))
            for h in scores:
                pos_scores = cw_present[f"score_{h}"].values
                ctrl_scores = cw_present[f"ctrl_{ctrl_mode}_score_{h}"].values
                # paired analysis: both arrays length N
                combined = np.concatenate([pos_scores, ctrl_scores])
                is_pos = np.concatenate([np.ones(len(pos_scores), bool),
                                          np.zeros(len(ctrl_scores), bool)])
                for pct in PERCENTILES:
                    thr = float(np.percentile(combined, pct))
                    or_v, p, pa, pb, ca, cb = fisher_or(combined, is_pos, thr)
                    ci_lo, ci_hi = bootstrap_ci(combined, is_pos, thr)
                    enrich_rows.append({
                        "pipeline": pname, "option": "Cmax", "filter": filt_mode,
                        "ctrl_set": ctrl_mode, "head": h, "percentile": pct,
                        "n_pos": int(is_pos.sum()), "n_ctrl": int((~is_pos).sum()),
                        "or": or_v, "p_value": p,
                        "ci_lo": float(ci_lo), "ci_hi": float(ci_hi),
                        "n_pos_above": pa, "n_ctrl_above": ca,
                    })

    enrich_df = pd.DataFrame(enrich_rows)
    enrich_df["q_value"] = multipletests(enrich_df["p_value"].fillna(1.0), method="fdr_bh")[1]
    enrich_df.to_csv(OUT / "enrichment_v5.csv", index=False)
    log.info("Wrote %s (%d rows)", OUT / "enrichment_v5.csv", len(enrich_df))

    # Print headlines
    print("\n" + "=" * 78)
    print("HEADLINE: phase3_A3G OR at p90 across all 8 pipelines")
    print("=" * 78)
    a3g = enrich_df[(enrich_df["head"] == "phase3_A3G") & (enrich_df["percentile"] == 90)]
    print(a3g[["pipeline", "n_pos", "n_ctrl", "or", "ci_lo", "ci_hi", "p_value", "q_value"]].to_string(index=False))
    print()
    print("phase3_A3A OR at p90:")
    a3a = enrich_df[(enrich_df["head"] == "phase3_A3A") & (enrich_df["percentile"] == 90)]
    print(a3a[["pipeline", "n_pos", "n_ctrl", "or", "ci_lo", "ci_hi", "p_value", "q_value"]].to_string(index=False))
    print()
    print("a1_new_sp1 OR at p90:")
    a1 = enrich_df[(enrich_df["head"] == "a1_new_sp1") & (enrich_df["percentile"] == 90)]
    print(a1[["pipeline", "n_pos", "n_ctrl", "or", "ci_lo", "ci_hi", "p_value", "q_value"]].to_string(index=False))

    log.info("Total: %.1f min", (time.time() - t0) / 60)


if __name__ == "__main__":
    main()
