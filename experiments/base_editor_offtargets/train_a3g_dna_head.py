#!/usr/bin/env python3
"""Train new A3G-DNA head on HIV cDNA hypermutation data.

Input: CSV with columns chrom, pos, strand, source (where pos is position
in HXB2 of an A3G-deaminated C, typically reported as G→A on the plus
strand of HXB2 since the deamination happens on cDNA = minus strand).

Pipeline:
1. Load positives (HIV hypermutation sites)
2. Generate motif-matched negatives (random Cs in HXB2 with same XCY context)
3. Extract 201nt windows from HXB2 around each C
4. Compute hand40 features (motif + ViennaRNA fold + loop geometry)
5. Compute RNA-FM embeddings (orig + delta)
6. Compute Phase3 shared encoder embeddings (128-d)
7. Train new APOBEC1-style head MLP on shared embeddings
8. 5-fold CV
9. Save head as a3g_dna_head.pt
10. (Separate script) score Lei BE off-targets with new head

Usage:
  python train_a3g_dna_head.py --positives <path_to_csv> [--n_neg_per_pos 1]

Argparse options to allow flexibility once we know data format.
"""
from __future__ import annotations
import argparse
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
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold

ROOT = Path("/Users/shaharharel/Documents/github/edit-rna-apobec")
HXB2 = ROOT / "data/raw/a3g_dna/HXB2_K03455.fasta"
PHASE3_CKPT = ROOT / "experiments/multi_enzyme/outputs/phase3_mfe_only/phase3_mfe_only.pt"
OUT_MODEL = ROOT / "experiments/multi_enzyme/outputs/a3g_dna_head/a3g_dna_head_v1.pt"
OUT_DIR = ROOT / "experiments/multi_enzyme/outputs/a3g_dna_head"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CENTER = 100
SEED = 42
D_INPUT = 1320; D_SHARED = 128; EMB_DIM = 640; N_ENZYMES_CLS = 6
ENZYMES = ["A3A", "A3B", "A3G", "A3A_A3G", "Neither"]
COMP = str.maketrans("ACGTN", "TGCAN")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("a3g_dna")


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
    """Same architecture as APOBEC1Head — small MLP on Phase3 shared 128-d output.
    Species channel kept for compatibility (set to constant during training)."""
    def __init__(self, d_shared: int = 128):
        super().__init__()
        self.species_proj = nn.Sequential(nn.Linear(1, 16), nn.GELU(), nn.Linear(16, d_shared))
        self.head = nn.Sequential(nn.Linear(d_shared, 32), nn.GELU(), nn.Linear(32, 1))

    def forward(self, shared, species):
        bias = self.species_proj(species)
        return self.head(shared + bias).squeeze(-1)


def get_seq_hxb2(genome, pos, strand, flank=100):
    """Extract 201nt window with C at position 100. For HIV hypermutation:
    typically the data reports the position of a G→A on plus strand of HXB2,
    meaning the deaminated C is on the minus strand. Strand handling:
      - If strand="+": C is at pos on plus strand (rare for A3G HIV data)
      - If strand="-": C is at pos on minus strand → revcomp window so C lands at center
    """
    chrom = list(genome.keys())[0]  # K03455.1
    chrom_len = len(genome[chrom])
    s, e = pos - flank, pos + flank + 1
    if s < 0 or e > chrom_len:
        return None
    seq = str(genome[chrom][s:e]).upper()
    if strand == "-":
        seq = seq.translate(COMP)[::-1]
    if len(seq) != 201 or seq[CENTER] != "C":
        return None
    return seq


def gen_negatives(positives, genome, n_per_pos=1, seed=SEED):
    """Sample n motif-matched random Cs from HXB2 NOT in positives."""
    rng = random.Random(seed)
    chrom = list(genome.keys())[0]
    full_seq = str(genome[chrom]).upper()
    full_seq_rc = full_seq.translate(COMP)[::-1]
    n = len(full_seq)

    pos_set = set(zip(positives["chrom"], positives["pos"], positives["strand"]))

    by_tri = {}
    # Index all Cs in HXB2 by trinuc on each strand
    for i in range(1, n - 1):
        tri_p = full_seq[i - 1] + full_seq[i] + full_seq[i + 1]
        if full_seq[i] == "C":
            by_tri.setdefault(("+", tri_p), []).append(i)
        # Reverse strand: C at fwd pos i corresponds to G at fwd[i] = ?
        # On reverse strand at pos i (fwd-coord), the C is when fwd[i]=G
        if full_seq[i] == "G":
            # On - strand, the trinuc context is (revcomp of fwd[i+1])(C)(revcomp of fwd[i-1])
            tri_n = COMP_BASE[full_seq[i + 1]] + "C" + COMP_BASE[full_seq[i - 1]]
            by_tri.setdefault(("-", tri_n), []).append(i)

    negatives = []
    skipped = 0
    pos_trinuc_counts = Counter(zip(positives["strand"], positives["trinuc"])) if "trinuc" in positives.columns else None
    for _, p in positives.iterrows():
        # Compute trinuc if not in df
        seq = get_seq_hxb2(genome, p["pos"], p["strand"])
        if seq is None:
            skipped += 1; continue
        tri = seq[CENTER - 1] + "C" + seq[CENTER + 1]
        key = (p["strand"], tri)
        cands = [i for i in by_tri.get(key, []) if (p["chrom"], i, p["strand"]) not in pos_set]
        if len(cands) < n_per_pos:
            skipped += 1; continue
        sampled = rng.sample(cands, n_per_pos)
        for cp in sampled:
            negatives.append({
                "chrom": p["chrom"], "pos": cp, "strand": p["strand"],
                "trinuc": tri, "source": "negative",
            })
    log.info("Generated %d negatives (skipped %d positives lacking matches)",
             len(negatives), skipped)
    return pd.DataFrame(negatives)


COMP_BASE = {"A": "T", "T": "A", "G": "C", "C": "G", "N": "N"}


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


def get_shared_embeddings(orig, delta, hand40, valid, p3, device, batch=512):
    n = len(orig)
    out = np.zeros((n, D_SHARED), dtype=np.float32)
    x = np.empty((n, D_INPUT), dtype=np.float32)
    x[:, :EMB_DIM] = orig.astype(np.float32)
    x[:, EMB_DIM:2*EMB_DIM] = delta.astype(np.float32)
    x[:, 2*EMB_DIM:] = hand40
    x[~valid] = 0.0
    p3.eval()
    with torch.no_grad():
        for i in range(0, n, batch):
            j = min(i + batch, n)
            xt = torch.from_numpy(x[i:j]).to(device)
            sh = p3.shared_encoder(xt)
            out[i:j] = sh.cpu().numpy()
    return out


def train_head(shared, labels, species_val, n_folds=5, epochs=50, lr=1e-3, device="cpu"):
    rng = np.random.default_rng(SEED)
    n = len(shared)
    idx = np.arange(n)
    rng.shuffle(idx)
    kf = KFold(n_splits=n_folds, shuffle=False)
    aucs = []
    fold_preds = np.zeros(n, dtype=np.float32)
    species = np.full((n, 1), species_val, dtype=np.float32)
    for fold, (tr, va) in enumerate(kf.split(idx)):
        tr_i = idx[tr]; va_i = idx[va]
        head = A3GDNAHead().to(device)
        opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=1e-4)
        bce = nn.BCEWithLogitsLoss()
        sh_tr = torch.from_numpy(shared[tr_i]).to(device)
        sp_tr = torch.from_numpy(species[tr_i]).to(device)
        y_tr = torch.from_numpy(labels[tr_i].astype(np.float32)).to(device)
        sh_va = torch.from_numpy(shared[va_i]).to(device)
        sp_va = torch.from_numpy(species[va_i]).to(device)
        y_va = labels[va_i]
        head.train()
        for ep in range(epochs):
            perm = torch.randperm(len(tr_i), device=device)
            for k in range(0, len(perm), 64):
                b = perm[k:k + 64]
                logit = head(sh_tr[b], sp_tr[b])
                loss = bce(logit, y_tr[b])
                opt.zero_grad(); loss.backward(); opt.step()
        head.eval()
        with torch.no_grad():
            prob = torch.sigmoid(head(sh_va, sp_va)).cpu().numpy()
        try:
            auc = roc_auc_score(y_va, prob)
        except ValueError:
            auc = float("nan")
        log.info("  fold %d AUC=%.4f", fold, auc)
        aucs.append(auc); fold_preds[va_i] = prob

    log.info("Training final head on all data...")
    head = A3GDNAHead().to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=1e-4)
    bce = nn.BCEWithLogitsLoss()
    sh_all = torch.from_numpy(shared).to(device)
    sp_all = torch.from_numpy(species).to(device)
    y_all = torch.from_numpy(labels.astype(np.float32)).to(device)
    head.train()
    for ep in range(epochs):
        perm = torch.randperm(n, device=device)
        for k in range(0, len(perm), 64):
            b = perm[k:k + 64]
            logit = head(sh_all[b], sp_all[b])
            loss = bce(logit, y_all[b])
            opt.zero_grad(); loss.backward(); opt.step()
    head.eval()
    return head, aucs, fold_preds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--positives", required=True, help="CSV: chrom,pos,strand,source")
    ap.add_argument("--n_neg_per_pos", type=int, default=1)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--reference", default=str(HXB2))
    ap.add_argument("--species_val", type=float, default=0.5,
                     help="species channel (HIV is human host; pick 0.5 as neutral)")
    args = ap.parse_args()

    t0 = time.time()
    log.info("=" * 60)
    log.info("A3G-DNA head training v1")
    log.info("=" * 60)

    log.info("Loading positives from %s", args.positives)
    pos = pd.read_csv(args.positives)
    log.info("Positives: %d rows", len(pos))
    log.info("  columns: %s", list(pos.columns))
    if "strand" not in pos.columns: pos["strand"] = "-"  # HIV cDNA convention default
    if "source" not in pos.columns: pos["source"] = "positive"

    log.info("Opening HIV reference: %s", args.reference)
    genome = Fasta(args.reference, as_raw=True)
    log.info("  reference: %s, length: %d", list(genome.keys())[0], len(genome[list(genome.keys())[0]]))

    log.info("Generating negatives (motif-matched, %d per positive)...", args.n_neg_per_pos)
    neg = gen_negatives(pos, genome, n_per_pos=args.n_neg_per_pos)

    pos["label"] = 1; neg["label"] = 0
    pos["source"] = pos.get("source", "positive")
    if "trinuc" not in pos.columns:
        seqs_p = [get_seq_hxb2(genome, p["pos"], p["strand"]) for _, p in pos.iterrows()]
        pos["trinuc"] = [s[CENTER-1]+"C"+s[CENTER+1] if s else None for s in seqs_p]
    if "trinuc" not in neg.columns:
        seqs_n = [get_seq_hxb2(genome, p["pos"], p["strand"]) for _, p in neg.iterrows()]
        neg["trinuc"] = [s[CENTER-1]+"C"+s[CENTER+1] if s else None for s in seqs_n]

    all_df = pd.concat([pos, neg], ignore_index=True)
    all_df["seq"] = [get_seq_hxb2(genome, r["pos"], r["strand"]) for _, r in all_df.iterrows()]
    all_df = all_df[all_df["seq"].notna()].reset_index(drop=True)
    log.info("Total valid (pos+neg): %d (%d pos, %d neg)",
             len(all_df), int((all_df["label"] == 1).sum()), int((all_df["label"] == 0).sum()))

    # Sanity checks
    pos_tcn = (all_df[all_df["label"] == 1]["trinuc"].str[0] == "T").mean() * 100
    pos_ccn = (all_df[all_df["label"] == 1]["trinuc"].str[0] == "C").mean() * 100
    log.info("SANITY: A3G positives — TCN=%.1f%%, CCN=%.1f%% (A3G prefers CC)",
             pos_tcn, pos_ccn)

    # Features
    log.info("Computing hand40...")
    hand40, hand_valid = compute_hand40(all_df["seq"].tolist(), n_workers=8)

    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    log.info("Device: %s", device)

    log.info("Computing RNA-FM...")
    orig, delta = compute_rnafm(all_df["seq"].tolist(), device)
    valid = hand_valid & (orig.sum(axis=1) != 0)
    log.info("Valid: %d/%d", valid.sum(), len(all_df))

    log.info("Loading Phase3 + computing shared embeddings...")
    p3 = Phase3Model()
    state = torch.load(PHASE3_CKPT, weights_only=False, map_location="cpu")
    state = state["model_state_dict"] if isinstance(state, dict) and "model_state_dict" in state else state
    p3.load_state_dict(state, strict=False); p3 = p3.eval().to(device)
    shared = get_shared_embeddings(orig, delta, hand40, valid, p3, device)
    log.info("  shared shape: %s", shared.shape)

    # Filter to valid only for training
    valid_mask = valid
    sh_tr = shared[valid_mask]
    labels = all_df["label"].values[valid_mask].astype(np.int64)
    log.info("Training set: %d (%d pos, %d neg)", len(sh_tr), (labels == 1).sum(), (labels == 0).sum())

    log.info("Training new A3G-DNA head with 5-fold CV...")
    head, aucs, _ = train_head(sh_tr, labels, args.species_val,
                                 n_folds=5, epochs=args.epochs, lr=1e-3, device=device)
    log.info("Mean CV AUC: %.4f (std %.4f)", np.mean(aucs), np.std(aucs))
    log.info("Saving head to %s", OUT_MODEL)
    torch.save(head.state_dict(), OUT_MODEL)

    summary = {
        "n_positives": int(len(pos)),
        "n_negatives": int(len(neg)),
        "n_train_valid": int(len(sh_tr)),
        "pos_tcn_pct": float(pos_tcn),
        "pos_ccn_pct": float(pos_ccn),
        "cv_auc_mean": float(np.mean(aucs)),
        "cv_auc_std": float(np.std(aucs)),
        "cv_aucs": [float(a) for a in aucs],
        "saved_to": str(OUT_MODEL),
        "wall_clock_min": (time.time() - t0) / 60,
    }
    with open(OUT_DIR / "training_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    log.info("Total time: %.1f min", (time.time() - t0) / 60)


if __name__ == "__main__":
    main()
