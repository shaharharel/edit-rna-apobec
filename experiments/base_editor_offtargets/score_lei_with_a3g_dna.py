#!/usr/bin/env python3
"""Score Lei TC-subset (Bperm + Bstrict) with new A3G-DNA head.

Recomputes features for the subset (Bperm = ~7300 sites with their controls).
Compares enrichment at p60-p99 against the RNA-trained A3G adapter.
"""
from __future__ import annotations
import json, logging, multiprocessing as mp, time
from pathlib import Path
import numpy as np
import pandas as pd
import torch, torch.nn as nn
from pyfaidx import Fasta
from scipy.stats import fisher_exact
from statsmodels.stats.multitest import multipletests

ROOT = Path("/Users/shaharharel/Documents/github/edit-rna-apobec")
HG38 = ROOT / "data/raw/genomes/hg38.fa"
PHASE3_CKPT = ROOT / "experiments/multi_enzyme/outputs/phase3_mfe_only/phase3_mfe_only.pt"
A3G_DNA_CKPT = ROOT / "experiments/multi_enzyme/outputs/a3g_dna_head/a3g_dna_head_v1.pt"
V5_DIR = ROOT / "experiments/base_editor_offtargets/outputs/lei_v5_sensitivity"
OUT = ROOT / "experiments/base_editor_offtargets/outputs/lei_a3g_dna_test"
OUT.mkdir(parents=True, exist_ok=True)

CENTER = 100; D_INPUT = 1320; D_SHARED = 128; EMB_DIM = 640
N_ENZYMES_CLS = 6
ENZYMES = ["A3A", "A3B", "A3G", "A3A_A3G", "Neither"]
COMP = str.maketrans("ACGTN", "TGCAN")
PERCENTILES = [60, 70, 75, 80, 85, 90, 95, 99]
N_BOOTSTRAP = 500; SEED = 42

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("a3g_dna_test")


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


class A3GDNAHead(nn.Module):
    def __init__(self, d_shared=128):
        super().__init__()
        self.species_proj = nn.Sequential(nn.Linear(1, 16), nn.GELU(), nn.Linear(16, d_shared))
        self.head = nn.Sequential(nn.Linear(d_shared, 32), nn.GELU(), nn.Linear(32, 1))

    def forward(self, shared, species):
        bias = self.species_proj(species)
        return self.head(shared + bias).squeeze(-1)


def get_seq(genome, chrom, pos, strand, flank=100):
    try: cl = len(genome[chrom])
    except KeyError: return None
    s, e = pos - flank, pos + flank + 1
    if s < 0 or e > cl: return None
    seq = str(genome[chrom][s:e]).upper()
    if strand == "-": seq = seq.translate(COMP)[::-1]
    if len(seq) != 201 or seq[CENTER] != "C": return None
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
    sd = np.zeros(7, dtype=np.float32)
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
    log.info("RNA-FM done %.1fs", time.time() - t0)
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
    return float(or_v), float(p), pa, pb, ca, cb


def bootstrap_ci(scores, is_pos, thr, n_boot=N_BOOTSTRAP):
    rng = np.random.default_rng(SEED)
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
    log.info("Loading Lei v5 all_scored...")
    df = pd.read_csv(V5_DIR / "all_scored.csv", low_memory=False)
    df = df[df["valid"]].reset_index(drop=True)

    # Subset to Bperm + global controls (where the existing A3G signal lives at p70-p85)
    log.info("Filtering to Bperm subset...")
    is_cand = df["src"] == "candidate"
    bperm_pos = df[is_cand & (df["in_Bperm"] == True)].copy()
    pos_set = set(zip(bperm_pos["chrom"], bperm_pos["pos"]))
    ctrl_global = df[(df["src"] == "ctrl_global") & df.apply(lambda r: (r["chrom"], r["matched_to"]) in pos_set, axis=1)].copy()
    ctrl_local = df[(df["src"] == "ctrl_local") & df.apply(lambda r: (r["chrom"], r["matched_to"]) in pos_set, axis=1)].copy()
    log.info("Bperm: pos=%d, ctrl_local=%d, ctrl_global=%d", len(bperm_pos), len(ctrl_local), len(ctrl_global))

    bperm_pos["label"] = "positive"
    ctrl_local["label"] = "control_local"
    ctrl_global["label"] = "control_global"
    all_subset = pd.concat([bperm_pos, ctrl_local, ctrl_global], ignore_index=True)

    # Re-extract sequences
    log.info("Opening hg38, re-extracting %d sequences...", len(all_subset))
    genome = Fasta(str(HG38), as_raw=True)
    seqs = [get_seq(genome, r["chrom"], r["pos"], r["strand"]) for _, r in all_subset.iterrows()]
    valid_seq = [s is not None for s in seqs]
    all_subset = all_subset[pd.Series(valid_seq, index=all_subset.index)].reset_index(drop=True)
    seqs = [s for s in seqs if s is not None]
    log.info("Valid sequences: %d", len(seqs))

    # Compute features
    log.info("Computing hand40...")
    hand40, hand_valid = compute_hand40(seqs, n_workers=8)
    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    log.info("Device: %s", device)
    log.info("Computing RNA-FM (~%d sites)...", len(seqs))
    orig, delta = compute_rnafm(seqs, device)
    valid = hand_valid & (orig.sum(axis=1) != 0)

    # Phase3 shared
    log.info("Loading Phase3 + computing shared embeddings...")
    p3 = Phase3Model()
    state = torch.load(PHASE3_CKPT, weights_only=False, map_location="cpu")
    state = state["model_state_dict"] if isinstance(state, dict) and "model_state_dict" in state else state
    p3.load_state_dict(state, strict=False); p3 = p3.eval().to(device)

    n = len(seqs)
    x = np.empty((n, D_INPUT), dtype=np.float32)
    x[:, :EMB_DIM] = orig.astype(np.float32)
    x[:, EMB_DIM:2*EMB_DIM] = delta.astype(np.float32)
    x[:, 2*EMB_DIM:] = hand40
    x[~valid] = 0.0
    shared = np.zeros((n, D_SHARED), dtype=np.float32)
    a3g_dna_score = np.zeros(n, dtype=np.float32)

    log.info("Loading A3G-DNA head v1 + scoring...")
    a3g_dna = A3GDNAHead().to(device)
    a3g_dna.load_state_dict(torch.load(A3G_DNA_CKPT, weights_only=False, map_location="cpu"))
    a3g_dna.eval()

    B = 512
    with torch.no_grad():
        for i in range(0, n, B):
            j = min(i + B, n)
            xt = torch.from_numpy(x[i:j]).to(device)
            sh = p3.shared_encoder(xt)
            shared[i:j] = sh.cpu().numpy()
            sp = torch.full((j - i, 1), 0.5, dtype=torch.float32, device=device)
            a3g_dna_score[i:j] = torch.sigmoid(a3g_dna(sh, sp)).cpu().numpy()

    all_subset["score_a3g_dna_v1"] = a3g_dna_score
    all_subset["valid_recomp"] = valid
    all_subset.to_csv(OUT / "lei_bperm_with_a3g_dna.csv", index=False)
    log.info("Wrote %s", OUT / "lei_bperm_with_a3g_dna.csv")

    # Enrichment: A3G-DNA vs both control sets, vs old A3G (for comparison)
    log.info("Computing enrichment...")
    rows = []
    for ctrl_lbl, ctrl_name in [("control_local", "local"), ("control_global", "global")]:
        is_pos = (all_subset["label"] == "positive").values & valid
        is_ctrl = (all_subset["label"] == ctrl_lbl).values & valid
        keep = is_pos | is_ctrl
        ip = is_pos[keep]
        # Score with both heads for direct comparison
        for head_col in ["score_a3g_dna_v1", "score_phase3_A3G", "score_a1_new_sp1", "score_a1_old_sp1"]:
            sc = all_subset.loc[keep, head_col].values
            for pct in PERCENTILES:
                thr = float(np.percentile(sc, pct))
                or_v, p, pa, pb, ca, cb = fisher_or(sc, ip, thr)
                ci_lo, ci_hi = bootstrap_ci(sc, ip, thr)
                rows.append({
                    "control_set": ctrl_name, "head": head_col.replace("score_", ""),
                    "percentile": pct,
                    "n_pos": int(ip.sum()), "n_ctrl": int((~ip).sum()),
                    "n_pos_above": pa, "n_ctrl_above": ca,
                    "or": or_v, "p_value": p,
                    "ci_lo": float(ci_lo) if ci_lo == ci_lo else np.nan,
                    "ci_hi": float(ci_hi) if ci_hi == ci_hi else np.nan,
                })
    enrich_df = pd.DataFrame(rows)
    enrich_df["q_value"] = multipletests(enrich_df["p_value"].fillna(1.0), method="fdr_bh")[1]
    enrich_df.to_csv(OUT / "enrichment_a3g_dna_vs_others.csv", index=False)
    log.info("Wrote %s", OUT / "enrichment_a3g_dna_vs_others.csv")

    # Print compact comparison
    print("\n" + "=" * 90)
    print("A3G-DNA v1 vs RNA-trained A3G (Bperm subset, both control sets)")
    print("=" * 90)
    for cs in ["local", "global"]:
        print(f"\n--- vs {cs} controls ---")
        sub = enrich_df[enrich_df["control_set"] == cs]
        piv = sub.pivot_table(index="head", columns="percentile", values="or", aggfunc="first")
        piv_q = sub.pivot_table(index="head", columns="percentile", values="q_value", aggfunc="first")
        # Format combined
        out_str = piv.copy().astype(object)
        for h in piv.index:
            for pct in piv.columns:
                or_v = piv.loc[h, pct]; q = piv_q.loc[h, pct]
                if pd.isna(or_v): out_str.loc[h, pct] = "—"
                else: out_str.loc[h, pct] = f"{or_v:.2f}(q={q:.1e})"
        print(out_str.to_string())

    log.info("Total: %.1f min", (time.time() - t0) / 60)


if __name__ == "__main__":
    main()
