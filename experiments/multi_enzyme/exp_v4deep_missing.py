#!/usr/bin/env python
"""V4Deep missing experiments: CpG stratification, ClinVar nonsense, TSG enrichment.

Uses trained models + existing TCGA/ClinVar data.

Output: experiments/multi_enzyme/outputs/v4deep/v4deep_missing_results.json
"""

import gc
import gzip
import json
import math
import sys
import time
import traceback
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import stats
from scipy.stats import binomtest
from torch.utils.data import DataLoader, Dataset

sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

OUTPUT_DIR = PROJECT_ROOT / "experiments" / "multi_enzyme" / "outputs" / "v4deep"
RESULTS_PATH = OUTPUT_DIR / "v4deep_missing_results.json"

D_RNAFM = 640
D_HAND = 40
D_SHARED = 128
CENTER = 100
PER_ENZYME_HEADS = ["A3A", "A3B", "A3G", "A3A_A3G", "Neither"]
N_ENZYMES = 6

DEVICE = (
    torch.device("mps")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    else torch.device("cpu")
)
print(f"Using device: {DEVICE}")


def save_results(results):
    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2, default=str)


# ── Model Architecture (matching exp_v4deep_replication.py exactly) ──


class H4Mixin:
    def _init_heads(self):
        self.binary_head = nn.Linear(D_SHARED, 1)
        self.enzyme_classifier = nn.Sequential(
            nn.Linear(D_SHARED, 64), nn.GELU(), nn.Dropout(0.2), nn.Linear(64, N_ENZYMES),
        )
        self.private_encoders = nn.ModuleDict({
            enz: nn.Sequential(nn.Linear(D_SHARED, 32), nn.GELU(), nn.Dropout(0.1))
            for enz in PER_ENZYME_HEADS
        })
        self.enzyme_adapters = nn.ModuleDict({
            enz: nn.Linear(D_SHARED + 32, 1) for enz in PER_ENZYME_HEADS
        })

    def _apply_heads(self, shared):
        binary_logit = self.binary_head(shared).squeeze(-1)
        per_enzyme_logits = []
        for enz in PER_ENZYME_HEADS:
            private = self.private_encoders[enz](shared)
            combined = torch.cat([shared, private], dim=-1)
            per_enzyme_logits.append(self.enzyme_adapters[enz](combined).squeeze(-1))
        enzyme_cls_logits = self.enzyme_classifier(shared)
        return binary_logit, per_enzyme_logits, enzyme_cls_logits


class H1Mixin:
    def _init_heads(self):
        self.binary_head = nn.Linear(D_SHARED, 1)
        self.enzyme_adapters = nn.ModuleDict({
            enz: nn.Sequential(nn.Linear(D_SHARED, 32), nn.GELU(), nn.Linear(32, 1))
            for enz in PER_ENZYME_HEADS
        })
        self.enzyme_classifier = nn.Sequential(
            nn.Linear(D_SHARED, 64), nn.GELU(), nn.Dropout(0.2), nn.Linear(64, N_ENZYMES),
        )

    def _apply_heads(self, shared):
        binary_logit = self.binary_head(shared).squeeze(-1)
        per_enzyme_logits = [
            self.enzyme_adapters[enz](shared).squeeze(-1) for enz in PER_ENZYME_HEADS
        ]
        enzyme_cls_logits = self.enzyme_classifier(shared)
        return binary_logit, per_enzyme_logits, enzyme_cls_logits


def _make_a8_encoder():
    d_local = 41
    local_proj = nn.Linear(d_local, 64)
    local_pos_enc = nn.Parameter(torch.randn(41, 64) * 0.02)
    encoder_layer = nn.TransformerEncoderLayer(
        d_model=64, nhead=4, dim_feedforward=128,
        dropout=0.1, activation="gelu", batch_first=True,
    )
    local_transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
    local_pool_attn = nn.Linear(64, 1)
    cross_q = nn.Linear(64, 64)
    cross_k = nn.Linear(D_RNAFM, 64)
    cross_v = nn.Linear(D_RNAFM, 64)
    d_fused = 64 + 64 + D_RNAFM + D_HAND
    encoder = nn.Sequential(
        nn.Linear(d_fused, 256), nn.GELU(), nn.Dropout(0.3), nn.LayerNorm(256),
        nn.Linear(256, D_SHARED), nn.GELU(), nn.Dropout(0.2),
    )
    return local_proj, local_pos_enc, local_transformer, local_pool_attn, cross_q, cross_k, cross_v, encoder


def _a8_encode(model, batch):
    bp = batch["bp_submatrix"].squeeze(1)  # [B, 41, 41]
    rnafm = batch["rnafm"]
    hand = batch["hand_feat"]
    local_in = model.local_proj(bp) + model.local_pos_enc.unsqueeze(0)
    local_out = model.local_transformer(local_in)
    attn_w = torch.softmax(model.local_pool_attn(local_out), dim=1)
    local_repr = (local_out * attn_w).sum(dim=1)
    q = model.cross_q(local_repr).unsqueeze(1)
    k = model.cross_k(rnafm).unsqueeze(1)
    v = model.cross_v(rnafm).unsqueeze(1)
    attn_scores = (q * k).sum(-1) / math.sqrt(64)
    attn_weights = torch.softmax(attn_scores, dim=-1)
    cross_repr = (attn_weights.unsqueeze(-1) * v).squeeze(1)
    fused = torch.cat([local_repr, cross_repr, rnafm, hand], dim=-1)
    return model.encoder(fused)


class HierarchicalAttention_H4(nn.Module, H4Mixin):
    name = "A8_H4"
    def __init__(self):
        super().__init__()
        (self.local_proj, self.local_pos_enc, self.local_transformer,
         self.local_pool_attn, self.cross_q, self.cross_k, self.cross_v,
         self.encoder) = _make_a8_encoder()
        self._init_heads()

    def forward(self, batch):
        shared = _a8_encode(self, batch)
        return self._apply_heads(shared)


class HierarchicalAttention_H1(nn.Module, H1Mixin):
    name = "A8_H1"
    def __init__(self):
        super().__init__()
        (self.local_proj, self.local_pos_enc, self.local_transformer,
         self.local_pool_attn, self.cross_q, self.cross_k, self.cross_v,
         self.encoder) = _make_a8_encoder()
        self._init_heads()

    def forward(self, batch):
        shared = _a8_encode(self, batch)
        return self._apply_heads(shared)


class ScoringDataset(Dataset):
    def __init__(self, rnafm, hand_feat, bp_sub=None):
        self.rnafm = rnafm
        self.hand_feat = hand_feat
        self.bp_sub = bp_sub
        self.n = rnafm.shape[0]

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        bp = self.bp_sub[idx] if self.bp_sub is not None else np.zeros((41, 41), dtype=np.float32)
        return {
            "rnafm": torch.from_numpy(self.rnafm[idx]),
            "bp_submatrix": torch.from_numpy(bp).unsqueeze(0),
            "hand_feat": torch.from_numpy(self.hand_feat[idx]),
        }


def standard_collate(batch_list):
    result = {}
    for key in batch_list[0]:
        vals = [b[key] for b in batch_list]
        if isinstance(vals[0], torch.Tensor):
            result[key] = torch.stack(vals)
        else:
            result[key] = vals
    return result


def score_dataset(model, dataset, device, batch_size=2048):
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        collate_fn=standard_collate, num_workers=0)
    all_binary = []
    all_per_enzyme = {enz: [] for enz in PER_ENZYME_HEADS}

    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            binary_logit, per_enzyme_logits, _ = model(batch)
            all_binary.append(torch.sigmoid(binary_logit).cpu().numpy())
            for h_idx, enz in enumerate(PER_ENZYME_HEADS):
                all_per_enzyme[enz].append(torch.sigmoid(per_enzyme_logits[h_idx]).cpu().numpy())

    return {
        "binary": np.concatenate(all_binary),
        **{enz: np.concatenate(all_per_enzyme[enz]) for enz in PER_ENZYME_HEADS},
    }


def compute_enrichment_at_percentile(scores_mut, scores_ctrl, percentile):
    all_scores = np.concatenate([scores_mut, scores_ctrl])
    threshold = np.percentile(all_scores, percentile)
    am = int((scores_mut >= threshold).sum())
    bm = int((scores_mut < threshold).sum())
    ac = int((scores_ctrl >= threshold).sum())
    bc = int((scores_ctrl < threshold).sum())
    if all(x > 0 for x in [am, bm, ac, bc]):
        OR, p = stats.fisher_exact([[am, bm], [ac, bc]])
    else:
        OR, p = float("nan"), 1.0
    return {
        "OR": float(OR), "p": float(p), "threshold": float(threshold),
        "mut_above": am, "mut_below": bm, "ctrl_above": ac, "ctrl_below": bc,
    }


# ── Load models ──

model_configs = {
    "A8_T1_H4": ("model_A8_T1_H4.pt", HierarchicalAttention_H4),
    "A8_T6_H4": ("model_A8_T6_H4.pt", HierarchicalAttention_H4),
    "A8_T4_H1": ("model_A8_T4_H1.pt", HierarchicalAttention_H1),
}

models = {}
for name, (fname, cls) in model_configs.items():
    print(f"Loading {name}...")
    model = cls()
    state = torch.load(OUTPUT_DIR / fname, map_location="cpu", weights_only=False)
    model.load_state_dict(state)
    model.to(DEVICE)
    model.eval()
    models[name] = model
    print(f"  Loaded {name}: {sum(p.numel() for p in model.parameters())} params")

results = {}

# ══════════════════════════════════════════════════════════════════════════
# 1c. CpG Stratification (NN)
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("1c. CpG Stratification")
print("=" * 70)

hand_dir = PROJECT_ROOT / "data" / "processed" / "multi_enzyme" / "tcga_hand_features"
emb_dir = PROJECT_ROOT / "data" / "processed" / "multi_enzyme" / "embeddings"

cpg_results = {}
target_cancers = ["blca", "brca", "cesc", "lusc"]

for cancer in target_cancers:
    print(f"\n--- {cancer.upper()} ---")
    t0 = time.time()
    try:
        emb_path = emb_dir / f"rnafm_tcga_{cancer}.pt"
        emb_data = torch.load(emb_path, map_location="cpu", weights_only=False)
        n_mut = emb_data["n_mut"]
        n_total = emb_data["n_total"]
        pooled_orig = emb_data["pooled_orig"].numpy().astype(np.float32)

        aligned_path = hand_dir / f"{cancer}_hand40_aligned.npy"
        hand_feat = np.load(str(aligned_path)).astype(np.float32)
        if hand_feat.shape[0] > n_total:
            hand_feat = hand_feat[:n_total]
        elif hand_feat.shape[0] < n_total:
            pad = np.zeros((n_total - hand_feat.shape[0], D_HAND), dtype=np.float32)
            hand_feat = np.concatenate([hand_feat, pad])

        mut_mask = np.zeros(n_total, dtype=bool)
        mut_mask[:n_mut] = True
        ctrl_mask = ~mut_mask

        # Hand features motif layout:
        # [0:4]  5' dinuc: UC, CC, AC, GC
        # [4:8]  3' dinuc: CA, CG, CU, CC
        # Index 0 = UC = TC context
        # Index 5 = CG = CpG context
        tc_mask = hand_feat[:, 0] == 1.0
        cpg_mask = hand_feat[:, 5] == 1.0
        tc_cpg = tc_mask & cpg_mask
        tc_noncpg = tc_mask & ~cpg_mask

        print(f"  n_total={n_total}, TC={tc_mask.sum()}, TC+CpG={tc_cpg.sum()}, TC+nonCpG={tc_noncpg.sum()}")

        dataset = ScoringDataset(pooled_orig, hand_feat)
        cancer_result = {"n_total": int(n_total), "n_mut": int(n_mut)}

        for model_name, model in models.items():
            print(f"  Scoring {model_name}...")
            scores = score_dataset(model, dataset, DEVICE)
            binary_scores = scores["binary"]

            model_result = {}
            for label, mask in [("tc_cpg", tc_cpg), ("tc_noncpg", tc_noncpg)]:
                m_mut = mut_mask & mask
                m_ctrl = ctrl_mask & mask
                if m_mut.sum() > 50 and m_ctrl.sum() > 50:
                    enrichment = compute_enrichment_at_percentile(
                        binary_scores[m_mut], binary_scores[m_ctrl], 90
                    )
                    enrichment["n_mut"] = int(m_mut.sum())
                    enrichment["n_ctrl"] = int(m_ctrl.sum())
                    model_result[f"{label}_p90"] = enrichment
                else:
                    model_result[f"{label}_p90"] = {"error": "too few sites",
                                                     "n_mut": int(m_mut.sum()),
                                                     "n_ctrl": int(m_ctrl.sum())}

            cancer_result[model_name] = model_result

        cpg_results[cancer] = cancer_result
        print(f"  Done in {time.time()-t0:.0f}s")
        del emb_data, pooled_orig, hand_feat
        gc.collect()

    except Exception as e:
        print(f"  ERROR: {e}")
        traceback.print_exc()
        cpg_results[cancer] = {"error": str(e)}

results["cpg_stratification"] = cpg_results
save_results(results)

# ══════════════════════════════════════════════════════════════════════════
# 1d & 1f. Expression and Per-Sample (documentation only)
# ══════════════════════════════════════════════════════════════════════════

results["expression_stratification"] = {
    "status": "NN expression stratification not feasible: TCGA embeddings are at site level without gene identity. XGB results (V2) remain the reference.",
}

results["per_sample_brca"] = {
    "status": "NN per-sample BRCA not feasible: TCGA embeddings pool all samples. XGB per-sample results (V2) remain the reference: APOBEC-high OR=1.546, APOBEC-low OR=1.454.",
}
save_results(results)

# ══════════════════════════════════════════════════════════════════════════
# 4b. ClinVar Nonsense Rate (NN)
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("4b. ClinVar Nonsense Rate")
print("=" * 70)

try:
    clinvar_df = pd.read_csv(
        PROJECT_ROOT / "experiments" / "apobec3a" / "outputs" / "clinvar_prediction" / "clinvar_all_scores.csv"
    )
    print(f"  ClinVar: {len(clinvar_df)} variants")

    # Parse VCF for molecular consequences
    vcf_path = PROJECT_ROOT / "data" / "raw" / "clinvar" / "clinvar_grch38.vcf.gz"
    print(f"  Parsing ClinVar VCF...")
    consequence_map = {}
    n_parsed = 0
    with gzip.open(vcf_path, "rt") as f:
        for line in f:
            if line.startswith("#"):
                continue
            fields = line.strip().split("\t")
            if len(fields) < 8:
                continue
            chrom = fields[0] if fields[0].startswith("chr") else "chr" + fields[0]
            pos = int(fields[1])
            ref = fields[3]
            alt = fields[4]
            if not ((ref == "C" and alt == "T") or (ref == "G" and alt == "A")):
                continue
            info = fields[7]
            mc = ""
            for part in info.split(";"):
                if part.startswith("MC="):
                    mc = part[3:]
                    break
            key = f"{chrom}:{pos}"
            if "nonsense" in mc.lower() or "stop_gained" in mc.lower():
                consequence_map[key] = "nonsense"
            elif "missense" in mc.lower():
                consequence_map[key] = "missense"
            elif "synonymous" in mc.lower():
                consequence_map[key] = "synonymous"
            else:
                consequence_map[key] = "other"
            n_parsed += 1
            if n_parsed % 500000 == 0:
                print(f"    {n_parsed} C>T variants parsed...")

    print(f"  VCF: {n_parsed} C>T variants, {len(consequence_map)} in map")

    # Match
    clinvar_df["vcf_key"] = "chr" + clinvar_df["chr"].astype(str).str.replace("chr", "") + ":" + clinvar_df["start"].astype(str)
    clinvar_df["consequence"] = clinvar_df["vcf_key"].map(consequence_map).fillna("unknown")

    path_mask_df = (clinvar_df["significance_simple"] == "Pathogenic") | (clinvar_df["significance_simple"] == "Likely_pathogenic")
    path_df = clinvar_df[path_mask_df]
    print(f"  Pathogenic: {len(path_df)}")
    for c, n in path_df["consequence"].value_counts().items():
        print(f"    {c}: {n} ({100*n/len(path_df):.1f}%)")

    # Load ClinVar embeddings
    emb_path = PROJECT_ROOT / "data" / "processed" / "multi_enzyme" / "embeddings" / "rnafm_clinvar.pt"
    emb_data = torch.load(emb_path, map_location="cpu", weights_only=False)
    clinvar_rnafm = emb_data["pooled_orig"].numpy().astype(np.float32)
    clinvar_site_ids = list(emb_data["site_ids"])
    n_clinvar = clinvar_rnafm.shape[0]

    feat_data = np.load(PROJECT_ROOT / "data" / "processed" / "clinvar_features_cache.npz", allow_pickle=True)
    clinvar_hand = feat_data["hand_46"][:, :40].astype(np.float32)
    feat_site_ids = list(feat_data["site_ids"])

    if len(feat_site_ids) != n_clinvar or feat_site_ids[:5] != clinvar_site_ids[:5]:
        feat_idx = {sid: i for i, sid in enumerate(feat_site_ids)}
        aligned = np.zeros((n_clinvar, 40), dtype=np.float32)
        for i, sid in enumerate(clinvar_site_ids):
            if sid in feat_idx:
                aligned[i] = clinvar_hand[feat_idx[sid]]
        clinvar_hand = aligned

    label_idx = {sid: i for i, sid in enumerate(clinvar_df["site_id"].tolist())}
    dataset = ScoringDataset(clinvar_rnafm, clinvar_hand)

    nonsense_results = {}
    for model_name, model in models.items():
        print(f"\n  {model_name}: scoring ClinVar...")
        scores = score_dataset(model, dataset, DEVICE, batch_size=4096)

        mapped_scores = []
        mapped_sig = []
        mapped_cons = []
        mapped_gene = []
        for i, sid in enumerate(clinvar_site_ids):
            if sid in label_idx:
                row_idx = label_idx[sid]
                mapped_scores.append(scores["binary"][i])
                mapped_sig.append(clinvar_df.iloc[row_idx]["significance_simple"])
                mapped_cons.append(clinvar_df.iloc[row_idx]["consequence"])
                mapped_gene.append(clinvar_df.iloc[row_idx]["gene"])

        mapped_scores = np.array(mapped_scores)
        mapped_sig = np.array(mapped_sig)
        mapped_cons = np.array(mapped_cons)
        mapped_gene = np.array(mapped_gene)

        path_mask = (mapped_sig == "Pathogenic") | (mapped_sig == "Likely_pathogenic")
        path_scores = mapped_scores[path_mask]
        path_cons = mapped_cons[path_mask]

        top_idx = np.argsort(path_scores)[-1000:]
        top_cons = path_cons[top_idx]
        nonsense_top = int((top_cons == "nonsense").sum())
        baseline_nonsense = (path_cons == "nonsense").sum() / len(path_cons) if len(path_cons) > 0 else 0

        model_result = {
            "top_1000_nonsense_count": nonsense_top,
            "top_1000_nonsense_rate": round(nonsense_top / 1000, 4),
            "baseline_nonsense_rate": round(float(baseline_nonsense), 4),
            "total_pathogenic": int(path_mask.sum()),
        }
        print(f"    Top-1000: {nonsense_top}/1000 nonsense ({nonsense_top/10:.1f}%), baseline={baseline_nonsense:.1%}")

        # Also store per-enzyme adapter top-1000
        for enz in PER_ENZYME_HEADS:
            enz_mapped = []
            for i, sid in enumerate(clinvar_site_ids):
                if sid in label_idx:
                    enz_mapped.append(scores[enz][i])
            enz_mapped = np.array(enz_mapped)
            enz_path = enz_mapped[path_mask]
            enz_top_idx = np.argsort(enz_path)[-1000:]
            enz_top_cons = path_cons[enz_top_idx]
            enz_nonsense = int((enz_top_cons == "nonsense").sum())
            model_result[f"{enz}_top1000_nonsense"] = enz_nonsense
            model_result[f"{enz}_top1000_nonsense_rate"] = round(enz_nonsense / 1000, 4)

        nonsense_results[model_name] = model_result

    results["clinvar_nonsense"] = nonsense_results
    del clinvar_rnafm, clinvar_hand, emb_data
    gc.collect()

except Exception as e:
    print(f"  ERROR: {e}")
    traceback.print_exc()
    results["clinvar_nonsense"] = {"error": str(e)}

save_results(results)

# ══════════════════════════════════════════════════════════════════════════
# 4c. TSG Enrichment
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("4c. TSG Enrichment")
print("=" * 70)

try:
    known_tsgs = {
        "TP53", "RB1", "APC", "BRCA1", "BRCA2", "PTEN", "VHL", "WT1", "NF1", "NF2",
        "TSC1", "TSC2", "SMAD4", "CDKN2A", "CDH1", "MEN1", "MLH1", "MSH2", "MSH6",
        "PMS2", "STK11", "PTCH1", "SDHB", "SDHD", "SDHA", "SDHC", "SDHAF2",
        "BAP1", "SMARCB1", "ARID1A", "ARID1B", "KDM6A", "FBXW7", "ATM", "ATR",
        "CHEK2", "PALB2", "MUTYH", "BMPR1A", "DICER1", "FLCN", "FH", "MAX",
        "TMEM127", "SUFU", "EPCAM", "NBN", "WRN",
    }

    clinvar_df_tsg = pd.read_csv(
        PROJECT_ROOT / "experiments" / "apobec3a" / "outputs" / "clinvar_prediction" / "clinvar_all_scores.csv"
    )
    path_mask_gb = (clinvar_df_tsg["significance_simple"] == "Pathogenic") | (clinvar_df_tsg["significance_simple"] == "Likely_pathogenic")
    ben_mask_gb = (clinvar_df_tsg["significance_simple"] == "Benign") | (clinvar_df_tsg["significance_simple"] == "Likely_benign")
    gb_scores = clinvar_df_tsg["p_edited_gb"].values
    genes = clinvar_df_tsg["gene"].values

    # GB TSG analysis
    tsg_gb = {}
    for gene in known_tsgs:
        gm = genes == gene
        gp = gm & path_mask_gb.values
        gb_ = gm & ben_mask_gb.values
        if gp.sum() >= 3 and gb_.sum() >= 3:
            ps = gb_scores[gp]
            bs = gb_scores[gb_]
            diff = ps.mean() - bs.mean()
            _, p = stats.mannwhitneyu(ps, bs, alternative="two-sided")
            tsg_gb[gene] = {"n_path": int(gp.sum()), "n_ben": int(gb_.sum()),
                            "mean_path": round(float(ps.mean()), 4), "mean_ben": round(float(bs.mean()), 4),
                            "path_gt_ben": bool(diff > 0), "p": float(p)}

    n_gb = len(tsg_gb)
    n_gb_pos = sum(1 for v in tsg_gb.values() if v["path_gt_ben"])
    sign_p_gb = binomtest(n_gb_pos, n_gb, 0.5, alternative="greater").pvalue
    print(f"  GB: {n_gb_pos}/{n_gb} TSGs path>ben, sign test p={sign_p_gb:.2e}")

    # NN TSG analysis
    emb_path = PROJECT_ROOT / "data" / "processed" / "multi_enzyme" / "embeddings" / "rnafm_clinvar.pt"
    emb_data = torch.load(emb_path, map_location="cpu", weights_only=False)
    clinvar_rnafm = emb_data["pooled_orig"].numpy().astype(np.float32)
    clinvar_site_ids = list(emb_data["site_ids"])
    n_clinvar = clinvar_rnafm.shape[0]

    feat_data = np.load(PROJECT_ROOT / "data" / "processed" / "clinvar_features_cache.npz", allow_pickle=True)
    clinvar_hand = feat_data["hand_46"][:, :40].astype(np.float32)
    feat_site_ids = list(feat_data["site_ids"])
    if len(feat_site_ids) != n_clinvar or feat_site_ids[:5] != clinvar_site_ids[:5]:
        feat_idx = {sid: i for i, sid in enumerate(feat_site_ids)}
        aligned = np.zeros((n_clinvar, 40), dtype=np.float32)
        for i, sid in enumerate(clinvar_site_ids):
            if sid in feat_idx:
                aligned[i] = clinvar_hand[feat_idx[sid]]
        clinvar_hand = aligned

    label_idx = {sid: i for i, sid in enumerate(clinvar_df_tsg["site_id"].tolist())}
    dataset = ScoringDataset(clinvar_rnafm, clinvar_hand)

    tsg_nn = {}
    for model_name, model in models.items():
        print(f"\n  TSG with {model_name}...")
        scores = score_dataset(model, dataset, DEVICE, batch_size=4096)

        mapped_scores_nn = []
        mapped_sig_nn = []
        mapped_gene_nn = []
        for i, sid in enumerate(clinvar_site_ids):
            if sid in label_idx:
                row_idx = label_idx[sid]
                mapped_scores_nn.append(scores["binary"][i])
                mapped_sig_nn.append(clinvar_df_tsg.iloc[row_idx]["significance_simple"])
                mapped_gene_nn.append(clinvar_df_tsg.iloc[row_idx]["gene"])

        mapped_scores_nn = np.array(mapped_scores_nn)
        mapped_sig_nn = np.array(mapped_sig_nn)
        mapped_gene_nn = np.array(mapped_gene_nn)

        path_mn = (mapped_sig_nn == "Pathogenic") | (mapped_sig_nn == "Likely_pathogenic")
        ben_mn = (mapped_sig_nn == "Benign") | (mapped_sig_nn == "Likely_benign")

        model_tsg = {}
        for gene in known_tsgs:
            gm = mapped_gene_nn == gene
            gp = gm & path_mn
            gb_ = gm & ben_mn
            if gp.sum() >= 3 and gb_.sum() >= 3:
                ps = mapped_scores_nn[gp]
                bs = mapped_scores_nn[gb_]
                diff = ps.mean() - bs.mean()
                _, p = stats.mannwhitneyu(ps, bs, alternative="two-sided")
                model_tsg[gene] = {"n_path": int(gp.sum()), "n_ben": int(gb_.sum()),
                                   "mean_path": round(float(ps.mean()), 4), "mean_ben": round(float(bs.mean()), 4),
                                   "path_gt_ben": bool(diff > 0), "p": float(p)}

        n_nn = len(model_tsg)
        n_nn_pos = sum(1 for v in model_tsg.values() if v["path_gt_ben"])
        sp = binomtest(n_nn_pos, n_nn, 0.5, alternative="greater").pvalue if n_nn > 0 else 1.0
        print(f"    {n_nn_pos}/{n_nn} TSGs path>ben, sign test p={sp:.2e}")

        tsg_nn[model_name] = {
            "n_genes_tested": n_nn, "n_path_gt_ben": n_nn_pos, "sign_test_p": float(sp),
            "per_gene": model_tsg,
        }

    results["tsg_enrichment"] = {
        "gb": {"n_genes_tested": n_gb, "n_path_gt_ben": n_gb_pos, "sign_test_p": float(sign_p_gb), "per_gene": tsg_gb},
        "nn": tsg_nn,
    }

    del clinvar_rnafm, clinvar_hand, emb_data
    gc.collect()

except Exception as e:
    print(f"  ERROR: {e}")
    traceback.print_exc()
    results["tsg_enrichment"] = {"error": str(e)}

save_results(results)

# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("DONE")
print("=" * 70)
print(f"Results: {RESULTS_PATH}")
print(f"Keys: {list(results.keys())}")
