#!/usr/bin/env python3
"""Score Lei BE4max Bperm subset with v4 model and compare to v3 + GC baseline.

v4 architecture (per the v4 brief):
- D_INPUT=1320 (640 RNA-FM orig + 640 RNA-FM delta + 40 hand)
- D_SHARED=128
- ENZYMES = ["A3A","A3B","A3G","A3A_A3G"] (4 adapters, no "Neither")
- ENZYME_CLASSES = 5-way (last = "Unknown")
- Separate APOBEC1 head v4

Key fix: trinucleotide-matched negatives → no anti-TCW polarity bug.

Compares:
- v3 binary, A3A, A3G, A1 (existing scores in scored_with_combined.csv)
- v4 binary, A3A, A3B, A3G, A3A_A3G, A1_v4 (compute fresh)
- GC content baseline (already computed)
"""
from __future__ import annotations
import json, logging, multiprocessing as mp, time
from pathlib import Path
import numpy as np
import pandas as pd
import torch, torch.nn as nn
from scipy.stats import fisher_exact
from statsmodels.stats.multitest import multipletests

ROOT = Path("/Users/shaharharel/Documents/github/edit-rna-apobec")
IN_CSV = ROOT / "experiments/base_editor_offtargets/outputs/lei_simple_baselines/scored_with_baselines.csv"
V4_PHASE3 = ROOT / "experiments/multi_enzyme/outputs/v4_cds_unbiased/phase3_v4_cds.pt"
V4_A1 = ROOT / "experiments/multi_enzyme/outputs/apobec1_head_v4_cds/apobec1_head_v4_cds.pt"
OUT = ROOT / "experiments/base_editor_offtargets/outputs/lei_v4_test"
OUT.mkdir(parents=True, exist_ok=True)

CENTER = 100
D_INPUT = 1320; D_SHARED = 128; EMB_DIM = 640
ENZYMES_V4 = ["A3A", "A3B", "A3G", "A3A_A3G"]   # NOTE: only 4, no "Neither"
N_ENZYME_CLASSES_V4 = 5
PERCENTILES = [60, 70, 75, 80, 85, 90, 95]

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("v4_test")


class Phase3ModelV4(nn.Module):
    """v4 architecture — 4 enzyme adapters, 5-way classifier."""
    def __init__(self):
        super().__init__()
        self.shared_encoder = nn.Sequential(
            nn.Linear(D_INPUT, 256), nn.GELU(), nn.Dropout(0.3), nn.LayerNorm(256),
            nn.Linear(256, D_SHARED), nn.GELU(), nn.Dropout(0.2),
        )
        self.binary_head = nn.Linear(D_SHARED, 1)
        self.enzyme_adapters = nn.ModuleDict({
            e: nn.Sequential(nn.Linear(D_SHARED, 32), nn.GELU(), nn.Linear(32, 1))
            for e in ENZYMES_V4
        })
        self.enzyme_classifier = nn.Sequential(
            nn.Linear(D_SHARED, 64), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(64, N_ENZYME_CLASSES_V4),
        )


class APOBEC1HeadV4(nn.Module):
    def __init__(self, d_shared=128):
        super().__init__()
        self.species_proj = nn.Sequential(nn.Linear(1, 16), nn.GELU(), nn.Linear(16, d_shared))
        self.head = nn.Sequential(nn.Linear(d_shared, 32), nn.GELU(), nn.Linear(32, 1))


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
    sd = np.zeros(7, dtype=np.float32)  # zeroed in mfe-only / v4 regime
    loop = np.zeros(9, dtype=np.float32)
    valid = True
    try:
        s, _ = RNA.fold(seq_u); n = len(s)
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
            loop[1] = sz; loop[2] = float(min(dl, dr))
            loop[3] = abs(CENTER - (ls + le) / 2.0)
            loop[4] = dl / max(sz - 1, 1)
            loop[5] = _stem_length(s, l, "left"); loop[6] = _stem_length(s, r, "right")
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
                log.info("  RNA-FM %d/%d (%.0fs)", j, n, time.time() - t0)
    log.info("RNA-FM done %.0fs", time.time() - t0)
    return orig, delta


def fisher_or(scores, is_pos, thr):
    above = scores >= thr
    pa = int((is_pos & above).sum()); pb = int((is_pos & ~above).sum())
    ca = int((~is_pos & above).sum()); cb = int((~is_pos & ~above).sum())
    if pb == 0 or ca == 0:
        or_v = float("inf") if pb == 0 else 0.0
    else:
        or_v = (pa * cb) / (pb * ca)
    _, p = fisher_exact([[pa, pb], [ca, cb]])
    return float(or_v), float(p), pa, ca


def main():
    t0 = time.time()
    log.info("Loading Lei subset (Bperm) with sequences + existing v3 scores + GC baselines...")
    df = pd.read_csv(IN_CSV, low_memory=False)
    if "valid_recomp" in df.columns:
        df = df[df["valid_recomp"]].reset_index(drop=True)
    log.info("Rows: %d", len(df))

    # Compute features fresh
    log.info("Computing hand40 (parallel CPU)...")
    hand40, hand_valid = compute_hand40(df["seq"].tolist(), n_workers=8)
    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    log.info("Device: %s", device)
    log.info("Computing RNA-FM (~%d seqs)...", len(df))
    orig, delta = compute_rnafm(df["seq"].tolist(), device)

    valid = hand_valid & (orig.sum(axis=1) != 0)
    log.info("Valid: %d/%d", valid.sum(), len(df))

    # Build input
    n = len(df)
    X = np.empty((n, D_INPUT), dtype=np.float32)
    X[:, :EMB_DIM] = orig.astype(np.float32)
    X[:, EMB_DIM:2*EMB_DIM] = delta.astype(np.float32)
    X[:, 2*EMB_DIM:] = hand40
    X[~valid] = 0.0

    # Load v4 phase3
    log.info("Loading v4 Phase3...")
    v4 = Phase3ModelV4().to(device)
    state = torch.load(V4_PHASE3, map_location=device, weights_only=False)
    state = state["model_state_dict"] if isinstance(state, dict) and "model_state_dict" in state else state
    missing, unexpected = v4.load_state_dict(state, strict=False)
    if missing: log.warning("v4 missing keys: %s", missing[:5])
    if unexpected: log.warning("v4 unexpected: %s", unexpected[:5])
    v4 = v4.eval()

    log.info("Loading v4 APOBEC1...")
    v4_a1 = APOBEC1HeadV4().to(device)
    a1_state = torch.load(V4_A1, map_location=device, weights_only=False)
    a1_state = a1_state["model_state_dict"] if isinstance(a1_state, dict) and "model_state_dict" in a1_state else a1_state
    v4_a1.load_state_dict(a1_state, strict=False)
    v4_a1.eval()

    # Score
    log.info("Scoring with v4 (binary + 4 enzyme adapters + APOBEC1_v4)...")
    score_cols = {h: np.zeros(n, dtype=np.float32) for h in
                    ["v4_binary"] + [f"v4_{e}" for e in ENZYMES_V4] + ["v4_apobec1"]}
    B = 512
    with torch.no_grad():
        for i in range(0, n, B):
            j = min(i + B, n)
            xt = torch.from_numpy(X[i:j]).to(device)
            sh = v4.shared_encoder(xt)
            score_cols["v4_binary"][i:j] = torch.sigmoid(v4.binary_head(sh).squeeze(-1)).cpu().numpy()
            for enz in ENZYMES_V4:
                score_cols[f"v4_{enz}"][i:j] = torch.sigmoid(
                    v4.enzyme_adapters[enz](sh).squeeze(-1)).cpu().numpy()
            sp = torch.zeros((j - i, 1), dtype=torch.float32, device=device)  # 0 = human
            score_cols["v4_apobec1"][i:j] = torch.sigmoid(
                v4_a1.head(sh + v4_a1.species_proj(sp)).squeeze(-1)).cpu().numpy()

    for c, s in score_cols.items():
        df[f"score_{c}"] = s
    df["valid_v4"] = valid
    df.to_csv(OUT / "lei_with_v4.csv", index=False)
    log.info("Wrote %s", OUT / "lei_with_v4.csv")

    # Enrichment computation: v4 heads vs v3 heads vs GC baselines
    HEADS = [
        # v4 heads
        "score_v4_binary", "score_v4_A3A", "score_v4_A3B", "score_v4_A3G", "score_v4_A3A_A3G",
        "score_v4_apobec1",
        # v3 heads (existing for comparison)
        "score_phase3_A3A", "score_phase3_A3G", "score_a1_old_sp1",
        # Composite + baselines
        "score_combined_positive",
        "base_gc_local", "base_gc_proto", "base_gc_201",
        "base_max_g_run", "base_random",
    ]
    rows = []
    for ctrl_lbl, ctrl_name in [("control_local", "local"), ("control_global", "global")]:
        is_pos = (df["label"] == "positive").values & valid
        is_ctrl = (df["label"] == ctrl_lbl).values & valid
        keep = is_pos | is_ctrl
        ip = is_pos[keep]
        n_pos = int(ip.sum()); n_ctrl = int((~ip).sum())
        log.info("Control %s: pos=%d ctrl=%d", ctrl_name, n_pos, n_ctrl)
        for head in HEADS:
            if head not in df.columns: continue
            sc = df.loc[keep, head].values
            for pct in PERCENTILES:
                thr = float(np.percentile(sc, pct))
                or_v, p, pa, ca = fisher_or(sc, ip, thr)
                rows.append({
                    "ctrl_set": ctrl_name, "head": head.replace("score_", ""),
                    "percentile": pct,
                    "n_pos": n_pos, "n_ctrl": n_ctrl,
                    "pos_above": pa, "ctrl_above": ca,
                    "or": or_v, "p_value": p,
                })
    enrich = pd.DataFrame(rows)
    enrich["q_value"] = multipletests(enrich["p_value"].fillna(1.0), method="fdr_bh")[1]
    enrich.to_csv(OUT / "enrichment_v4_vs_v3_vs_gc.csv", index=False)

    # Print summary
    print("\n" + "=" * 100)
    print("v4 vs v3 vs GC content baseline — OR by percentile (Bperm/global)")
    print("=" * 100)
    sub = enrich[enrich["ctrl_set"] == "global"]
    piv = sub.pivot_table(index="head", columns="percentile", values="or", aggfunc="first")
    piv_q = sub.pivot_table(index="head", columns="percentile", values="q_value", aggfunc="first")
    out_str = piv.copy().astype(object)
    for h in piv.index:
        for pct in piv.columns:
            or_v = piv.loc[h, pct]; q = piv_q.loc[h, pct]
            out_str.loc[h, pct] = f"{or_v:.2f}(q={q:.0e})" if not pd.isna(or_v) else "—"
    # Order: v4 first, v3 next, baselines last
    order = [h.replace("score_", "") for h in HEADS if h.replace("score_", "") in out_str.index]
    print(out_str.reindex(order).to_string())

    print("\n" + "=" * 100)
    print("vs LOCAL (same-region) controls")
    print("=" * 100)
    sub = enrich[enrich["ctrl_set"] == "local"]
    piv = sub.pivot_table(index="head", columns="percentile", values="or", aggfunc="first")
    piv_q = sub.pivot_table(index="head", columns="percentile", values="q_value", aggfunc="first")
    out_str = piv.copy().astype(object)
    for h in piv.index:
        for pct in piv.columns:
            or_v = piv.loc[h, pct]; q = piv_q.loc[h, pct]
            out_str.loc[h, pct] = f"{or_v:.2f}(q={q:.0e})" if not pd.isna(or_v) else "—"
    print(out_str.reindex(order).to_string())

    log.info("Total: %.1f min", (time.time() - t0) / 60)


if __name__ == "__main__":
    main()
