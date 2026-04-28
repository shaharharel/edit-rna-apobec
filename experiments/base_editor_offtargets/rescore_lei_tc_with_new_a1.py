#!/usr/bin/env python3
"""Rescore Lei v2 TC-subset with new A1 head v2.

Loads Lei v2 scored_sites.csv, filters to TC-context positives + their
matched controls (same-region + global). Recomputes features. Scores with
old A1 head (sp=0, sp=1) and new A1 head v2 (sp=0, sp=1). Recomputes
TC-stratified enrichment.

~10 min wall clock on Mac MPS.
"""
from __future__ import annotations
import gzip, json, logging, multiprocessing as mp, time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pyfaidx import Fasta
from scipy.stats import fisher_exact
from statsmodels.stats.multitest import multipletests

ROOT = Path("/Users/shaharharel/Documents/github/edit-rna-apobec")
HG38 = ROOT / "data/raw/genomes/hg38.fa"
PHASE3_CKPT = ROOT / "experiments/multi_enzyme/outputs/phase3_mfe_only/phase3_mfe_only.pt"
A1_OLD = ROOT / "experiments/multi_enzyme/outputs/apobec1_head/apobec1_head_mfe_only.pt"
A1_NEW = ROOT / "experiments/multi_enzyme/outputs/apobec1_head/apobec1_head_mfe_only_v2.pt"
LEI_V2 = ROOT / "experiments/base_editor_offtargets/outputs/lei_preview_v2/scored_sites.csv"
OUT = ROOT / "experiments/base_editor_offtargets/outputs/lei_v2_with_new_a1"
OUT.mkdir(parents=True, exist_ok=True)

CENTER = 100
D_INPUT = 1320; D_SHARED = 128; EMB_DIM = 640
ENZYMES = ["A3A", "A3B", "A3G", "A3A_A3G", "Neither"]
N_ENZYMES_CLS = 6
SEED = 42
PERCENTILES = [90, 95, 99]
N_BOOTSTRAP = 1000

COMP = str.maketrans("ACGTN", "TGCAN")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("rescore")


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
    struct_delta = np.zeros(7, dtype=np.float32)
    loop = np.zeros(9, dtype=np.float32)
    valid = True
    try:
        s, _ = RNA.fold(seq_u)
        n = len(s)
        is_unpaired = s[CENTER] == "."
        loop[0] = float(is_unpaired)
        wstart = max(0, CENTER - 10); wend = min(n, CENTER + 11)
        loop[8] = sum(1 for c in s[wstart:wend] if c == ".") / (wend - wstart)
        if is_unpaired:
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
    return np.concatenate([motif, struct_delta, loop]).astype(np.float32), valid


def _hand_worker(args):
    idx, seq = args
    return idx, *_hand_one(seq)


def main():
    t0 = time.time()
    log.info("Loading Lei v2 scored sites + filtering to TC subset...")
    df = pd.read_csv(LEI_V2)
    df = df[df["valid"]].reset_index(drop=True)
    if "trinuc" not in df.columns:
        log.error("trinuc column missing"); return
    tc_mask = df["trinuc"].str[0].isin(["T", "U"])
    df = df[tc_mask].reset_index(drop=True)
    log.info("TC-subset: %d sites", len(df))

    # Re-extract sequences
    log.info("Opening hg38, re-extracting sequences...")
    genome = Fasta(str(HG38), as_raw=True)
    seqs = []
    keep = []
    for r in df.itertuples():
        s = get_seq(genome, r.chrom, r.pos, r.strand)
        seqs.append(s)
        keep.append(s is not None)
    df = df[pd.Series(keep, index=df.index)].reset_index(drop=True)
    seqs = [s for s in seqs if s is not None]
    log.info("Sequences extracted: %d", len(seqs))

    # Hand40
    log.info("Computing hand40 features (parallel)...")
    n = len(seqs)
    hand40 = np.zeros((n, 40), dtype=np.float32)
    valid_h = np.zeros(n, dtype=bool)
    with mp.Pool(8) as pool:
        for idx, h, v in pool.imap_unordered(_hand_worker, list(enumerate(seqs)), chunksize=32):
            hand40[idx] = h; valid_h[idx] = v
    log.info("hand40 done: valid=%d/%d", valid_h.sum(), n)

    # RNA-FM
    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    log.info("Loading RNA-FM (rna_fm_t12)...")
    import fm
    model, alphabet = fm.pretrained.rna_fm_t12()
    bc = alphabet.get_batch_converter()
    model = model.eval().to(device)
    log.info("Computing RNA-FM embeddings on %s...", device)
    orig = np.zeros((n, EMB_DIM), dtype=np.float16)
    delta = np.zeros((n, EMB_DIM), dtype=np.float16)
    B = 16
    t1 = time.time()
    with torch.no_grad():
        for i in range(0, n, B):
            j = min(i + B, n)
            data_o = [(f"o{k}", seqs[k].replace("T", "U")) for k in range(i, j)]
            data_e = [(f"e{k}", seqs[k][:CENTER].replace("T", "U") + "U" + seqs[k][CENTER+1:].replace("T", "U")) for k in range(i, j)]
            _, _, t_o = bc(data_o); _, _, t_e = bc(data_e)
            t_o = t_o.to(device); t_e = t_e.to(device)
            r_o = model(t_o, repr_layers=[12])["representations"][12]
            r_e = model(t_e, repr_layers=[12])["representations"][12]
            orig[i:j] = r_o[:, 1:-1, :].mean(dim=1).cpu().float().numpy().astype(np.float16)
            delta[i:j] = (r_e[:, 1:-1, :].mean(dim=1).cpu().float() -
                          r_o[:, 1:-1, :].mean(dim=1).cpu().float()).numpy().astype(np.float16)
            if i % (B * 50) == 0:
                log.info("  RNA-FM %d/%d (%.1fs)", j, n, time.time() - t1)
    log.info("RNA-FM done in %.1fs", time.time() - t1)

    # Phase3 shared encoder
    log.info("Loading Phase3 + computing shared embeddings...")
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

    valid = valid_h & (orig.sum(axis=1) != 0)
    x = np.empty((n, D_INPUT), dtype=np.float32)
    x[:, :EMB_DIM] = orig.astype(np.float32)
    x[:, EMB_DIM:2*EMB_DIM] = delta.astype(np.float32)
    x[:, 2*EMB_DIM:] = hand40
    x[~valid] = 0.0

    new_scores = {}
    log.info("Scoring with old + new A1 heads (sp=0 and sp=1)...")
    with torch.no_grad():
        n_b = 512
        for nm, head in [("old", a1_old), ("new", a1_new)]:
            for sp_val in [0.0, 1.0]:
                key = f"a1_{nm}_sp{int(sp_val)}"
                arr = np.zeros(n, dtype=np.float32)
                for i in range(0, n, n_b):
                    j = min(i + n_b, n)
                    xt = torch.from_numpy(x[i:j]).to(device)
                    sh = p3.shared_encoder(xt)
                    sp = torch.full((j - i, 1), sp_val, dtype=torch.float32, device=device)
                    arr[i:j] = torch.sigmoid(head(sh, sp)).cpu().numpy()
                new_scores[key] = arr
                log.info("  %s done", key)

    # Add to df
    for k, v in new_scores.items():
        df[f"score_{k}"] = v
    df["valid_recomp"] = valid
    df.to_csv(OUT / "scored_tc_with_new_a1.csv", index=False)
    log.info("Wrote %s", OUT / "scored_tc_with_new_a1.csv")

    # Enrichment recompute on TC subset
    log.info("Computing enrichment vs same-region and global controls...")
    rng = np.random.default_rng(SEED)

    def fisher_or(scores, is_pos, thr):
        above = scores >= thr
        pa = (is_pos & above).sum(); pb = (is_pos & ~above).sum()
        ca = ((~is_pos) & above).sum(); cb = ((~is_pos) & ~above).sum()
        if pb == 0 or ca == 0:
            or_v = float("inf") if pb == 0 else 0.0
        else:
            or_v = (pa * cb) / (pb * ca)
        _, p = fisher_exact([[pa, pb], [ca, cb]])
        return float(or_v), float(p), int(pa), int(ca)

    rows = []
    for ctrl_label, name in [("control_local", "TC_local"), ("control_global", "TC_global")]:
        is_pos = (df["label"] == "positive").values & valid
        is_ctrl = (df["label"] == ctrl_label).values & valid
        keep = is_pos | is_ctrl
        for head_col in df.columns:
            if not head_col.startswith("score_"):
                continue
            sc = df[head_col].values[keep]
            ipk = (df["label"].values[keep] == "positive")
            for pct in PERCENTILES:
                thr = float(np.percentile(sc, pct))
                or_v, p, pa, ca = fisher_or(sc, ipk, thr)
                rows.append({
                    "control_set": name, "head": head_col.replace("score_", ""),
                    "percentile": pct, "or": or_v, "p_value": p,
                    "n_pos": int(ipk.sum()), "n_ctrl": int((~ipk).sum()),
                    "n_pos_above": pa, "n_ctrl_above": ca,
                })
    enrich_df = pd.DataFrame(rows)
    enrich_df["q_value"] = multipletests(enrich_df["p_value"].fillna(1.0), method="fdr_bh")[1]
    enrich_df.to_csv(OUT / "enrichment_tc_new_a1.csv", index=False)

    # Print summary at p90
    print("\n" + "=" * 70)
    print("TC subset, p90, vs same-region controls (old vs new A1)")
    print("=" * 70)
    sub = enrich_df[(enrich_df["control_set"] == "TC_local") & (enrich_df["percentile"] == 90)].sort_values("or", ascending=False)
    print(sub[["head", "or", "n_pos", "n_ctrl", "p_value", "q_value"]].to_string(index=False))
    print()
    print("=" * 70)
    print("TC subset, p90, vs global controls")
    print("=" * 70)
    sub = enrich_df[(enrich_df["control_set"] == "TC_global") & (enrich_df["percentile"] == 90)].sort_values("or", ascending=False)
    print(sub[["head", "or", "n_pos", "n_ctrl", "p_value", "q_value"]].to_string(index=False))

    log.info("Total time: %.1f min", (time.time() - t0) / 60)


if __name__ == "__main__":
    main()
