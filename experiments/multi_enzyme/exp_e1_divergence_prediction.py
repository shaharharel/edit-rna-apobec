#!/usr/bin/env python
"""
E1: Editability as Predictor of Human-Chimp Divergence

For every scored exonic C in the human exome editability map:
  1. Lift over hg19 → panTro6 (chimp)
  2. Check if the C is conserved or diverged
  3. Test: do high-editability positions show LESS divergence? (purifying selection)
  4. Logistic regression: P(diverged) ~ editability + trinuc + CpG
  5. Percentile-threshold OR analysis (p50, p75, p90, p95)

Input:  experiments/multi_enzyme/outputs/exome_map/exome_editability_chr*.csv.gz
Output: experiments/multi_enzyme/outputs/evolutionary/e1_divergence_prediction/
"""

import sys
import os
import glob
import json
import logging
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from pyliftover import LiftOver
from pyfaidx import Fasta

warnings.filterwarnings('ignore')

# ── Paths ────────────────────────────────────────────────────────────────────
PROJECT = Path("/Users/shaharharel/Documents/github/edit-rna-apobec")
EXOME_DIR = PROJECT / "experiments/multi_enzyme/outputs/exome_map"
OUT_DIR = PROJECT / "experiments/multi_enzyme/outputs/evolutionary/e1_divergence_prediction"
OUT_DIR.mkdir(parents=True, exist_ok=True)

PANTRO6_FA = PROJECT / "data/raw/genomes/panTro6.fa"
HG19_FA = PROJECT / "data/raw/genomes/hg19.fa"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(OUT_DIR / "e1_log.txt", mode='w'),
    ],
)
log = logging.getLogger(__name__)

COMP = {'A': 'T', 'T': 'A', 'C': 'G', 'G': 'C', 'N': 'N'}


def complement(base):
    return COMP.get(base.upper(), 'N')


def load_exome_map():
    """Load all per-chromosome exome editability files."""
    files = sorted(glob.glob(str(EXOME_DIR / "exome_editability_chr*.csv.gz")))
    # Exclude the full_model aggregate file
    files = [f for f in files if "full_model" not in f]
    dfs = []
    for f in files:
        df = pd.read_csv(f)
        dfs.append(df)
        log.info(f"  Loaded {Path(f).name}: {len(df):,} sites")
    combined = pd.concat(dfs, ignore_index=True)
    log.info(f"Total exome map: {len(combined):,} positions")
    return combined


def liftover_and_check(df, lo, chimp_fa):
    """
    For each position, lift over to panTro6 and check conservation.

    Returns arrays: chimp_base (str), lifted (bool), conserved (bool)
    """
    n = len(df)
    lifted = np.zeros(n, dtype=bool)
    conserved = np.zeros(n, dtype=bool)
    chimp_bases = np.empty(n, dtype='U1')
    chimp_bases[:] = ''

    chrs = df['chr'].values
    positions = df['pos'].values
    strands = df['strand'].values

    # Get available chimp chromosomes
    chimp_chroms = set(chimp_fa.keys())

    batch_size = 500_000
    n_batches = (n + batch_size - 1) // batch_size

    for batch_idx in range(n_batches):
        start = batch_idx * batch_size
        end = min(start + batch_size, n)
        if batch_idx % 2 == 0:
            log.info(f"  Liftover batch {batch_idx+1}/{n_batches} "
                     f"(positions {start:,}-{end:,})")

        for i in range(start, end):
            chrom = chrs[i]
            pos = int(positions[i])

            # Lift over
            try:
                result = lo.convert_coordinate(chrom, pos)
            except Exception:
                continue

            if not result or len(result) == 0:
                continue

            chimp_chrom, chimp_pos, chimp_strand, _ = result[0]

            # Check chimp chromosome exists
            if chimp_chrom not in chimp_chroms:
                continue

            # Get chimp base
            try:
                chimp_base_raw = str(chimp_fa[chimp_chrom][int(chimp_pos)]).upper()
            except (IndexError, KeyError):
                continue

            if chimp_base_raw not in 'ACGT':
                continue

            # If strand flipped during liftover, complement the chimp base
            # to compare on the same strand as the human reference
            if chimp_strand == '-':
                chimp_base = complement(chimp_base_raw)
            else:
                chimp_base = chimp_base_raw

            lifted[i] = True
            chimp_bases[i] = chimp_base

            # The human base is always C (these are all exonic C positions)
            # For minus-strand entries in the exome map, the 'C' is already
            # on the coding strand. The liftover works on genomic coords.
            # We need to check: is the human genomic base the same as chimp?
            # Human genomic base at this position:
            #   strand '+' → genomic = C
            #   strand '-' → genomic = G (complement of C on coding strand)
            human_strand = strands[i]
            if human_strand == '+':
                human_genomic = 'C'
            else:
                human_genomic = 'G'

            conserved[i] = (chimp_base == human_genomic)

    return lifted, conserved, chimp_bases


def compute_cpg_context(df, hg19_fa):
    """Check if each C position is in a CpG context."""
    n = len(df)
    is_cpg = np.zeros(n, dtype=bool)
    chrs = df['chr'].values
    positions = df['pos'].values
    strands = df['strand'].values

    for i in range(n):
        chrom = chrs[i]
        pos = int(positions[i])
        strand = strands[i]
        try:
            if strand == '+':
                # C at pos, check if next base is G → CpG
                next_base = str(hg19_fa[chrom][pos + 1]).upper()
                is_cpg[i] = (next_base == 'G')
            else:
                # Genomic base is G, check if prev base is C → CpG
                prev_base = str(hg19_fa[chrom][pos - 1]).upper()
                is_cpg[i] = (prev_base == 'C')
        except (IndexError, KeyError):
            pass

    return is_cpg


def percentile_or_analysis(df):
    """Compute OR for divergence at editability percentile thresholds."""
    thresholds = [50, 75, 90, 95]
    results = []

    for pct in thresholds:
        cutoff = np.percentile(df['score_full'], pct)
        high = df['score_full'] >= cutoff
        low = ~high

        n_high = high.sum()
        n_low = low.sum()
        div_high = df.loc[high, 'diverged'].sum()
        div_low = df.loc[low, 'diverged'].sum()
        cons_high = n_high - div_high
        cons_low = n_low - div_low

        rate_high = div_high / n_high if n_high > 0 else 0
        rate_low = div_low / n_low if n_low > 0 else 0

        # OR for divergence: high editability vs low
        table = np.array([[div_high, cons_high], [div_low, cons_low]])
        or_val, p_val = stats.fisher_exact(table) if min(table.flatten()) < 5 else (None, None)
        if or_val is None:
            # Use log OR from contingency table
            or_val = (div_high * cons_low) / (div_low * cons_high) if (div_low * cons_high) > 0 else np.nan
            # Chi-square test for large samples
            chi2, p_val, _, _ = stats.chi2_contingency(table)

        results.append({
            'percentile': pct,
            'cutoff': cutoff,
            'n_high': n_high,
            'n_low': n_low,
            'div_rate_high': rate_high,
            'div_rate_low': rate_low,
            'OR': or_val,
            'p_value': p_val,
            'log2_OR': np.log2(or_val) if or_val > 0 else np.nan,
        })

    return pd.DataFrame(results)


def logistic_regression_analysis(df):
    """
    Logistic regression: P(diverged) ~ editability + trinuc + CpG

    Returns model summary dict.
    """
    # Prepare features
    features = pd.DataFrame()
    features['score_full'] = df['score_full'].values
    features['score_motif'] = df['score_motif'].values
    features['is_cpg'] = df['is_cpg'].astype(int).values

    # Trinucleotide dummies (top trinucs only to avoid too many features)
    trinuc_dummies = pd.get_dummies(df['trinuc'], prefix='trinuc', drop_first=True)
    features = pd.concat([features, trinuc_dummies], axis=1)

    y = df['diverged'].astype(int).values

    # Standardize continuous features
    scaler = StandardScaler()
    features_scaled = features.copy()
    for col in ['score_full', 'score_motif']:
        features_scaled[col] = scaler.fit_transform(features[[col]])

    X = features_scaled.values

    log.info(f"  Logistic regression: {X.shape[0]:,} samples, {X.shape[1]} features")

    model = LogisticRegression(max_iter=1000, solver='lbfgs', C=1.0)
    model.fit(X, y)

    # Extract editability coefficient
    coef_names = list(features.columns)
    coefs = dict(zip(coef_names, model.coef_[0]))

    # Compute pseudo-R² (McFadden)
    from sklearn.metrics import log_loss
    ll_model = -log_loss(y, model.predict_proba(X), normalize=False)
    ll_null = -log_loss(y, np.full_like(y, y.mean(), dtype=float), normalize=False)
    pseudo_r2 = 1 - (ll_model / ll_null) if ll_null != 0 else 0

    # Also run with just editability (no confounds) for comparison
    X_edit_only = features_scaled[['score_full']].values
    model_edit = LogisticRegression(max_iter=1000, solver='lbfgs', C=1.0)
    model_edit.fit(X_edit_only, y)
    coef_edit_only = model_edit.coef_[0][0]

    # Wald test for score_full coefficient significance
    # Using bootstrap approximation for SE
    from sklearn.utils import resample
    n_boot = 200
    boot_coefs = []
    rng = np.random.RandomState(42)
    n_samples = min(len(y), 500_000)  # subsample for speed
    if len(y) > n_samples:
        idx = rng.choice(len(y), n_samples, replace=False)
        X_sub, y_sub = X[idx], y[idx]
    else:
        X_sub, y_sub = X, y

    for _ in range(n_boot):
        idx_b = rng.choice(len(y_sub), len(y_sub), replace=True)
        m = LogisticRegression(max_iter=500, solver='lbfgs', C=1.0)
        m.fit(X_sub[idx_b], y_sub[idx_b])
        boot_coefs.append(m.coef_[0][0])  # score_full coef

    se_edit = np.std(boot_coefs)
    z_edit = coefs['score_full'] / se_edit if se_edit > 0 else 0
    p_edit = 2 * stats.norm.sf(abs(z_edit))

    return {
        'coef_score_full': coefs['score_full'],
        'coef_score_full_edit_only': coef_edit_only,
        'coef_score_full_se': se_edit,
        'coef_score_full_z': z_edit,
        'coef_score_full_p': p_edit,
        'coef_is_cpg': coefs.get('is_cpg', np.nan),
        'coef_score_motif': coefs.get('score_motif', np.nan),
        'pseudo_r2': pseudo_r2,
        'n_samples': len(y),
        'n_diverged': y.sum(),
        'divergence_rate': y.mean(),
        'all_coefs': coefs,
    }


def trinuc_stratified_analysis(df):
    """Stratified analysis by trinucleotide context."""
    # Group into TC, CC, other
    df = df.copy()
    trinuc = df['trinuc'].str.upper()

    # TC context: middle base is C, previous is T → trinuc starts with T
    # CC context: previous is C → trinuc starts with C (but middle is C)
    # Actually trinuc is 3-char centered on C: [prev][C][next]
    df['motif_group'] = 'Other'
    df.loc[trinuc.str[0] == 'T', 'motif_group'] = 'TC'
    df.loc[trinuc.str[0] == 'C', 'motif_group'] = 'CC'

    results = []
    for group in ['TC', 'CC', 'Other']:
        sub = df[df['motif_group'] == group]
        if len(sub) < 100:
            continue

        div_rate = sub['diverged'].mean()
        n = len(sub)
        n_div = sub['diverged'].sum()

        # OR at p75 threshold
        cutoff_75 = np.percentile(sub['score_full'], 75)
        high = sub['score_full'] >= cutoff_75
        low = ~high
        div_high = sub.loc[high, 'diverged'].sum()
        div_low = sub.loc[low, 'diverged'].sum()
        cons_high = high.sum() - div_high
        cons_low = low.sum() - div_low

        if cons_high > 0 and div_low > 0:
            or_75 = (div_high * cons_low) / (div_low * cons_high)
        else:
            or_75 = np.nan

        # Spearman correlation
        sp_r, sp_p = stats.spearmanr(sub['score_full'], sub['diverged'])

        results.append({
            'motif_group': group,
            'n': n,
            'n_diverged': n_div,
            'divergence_rate': div_rate,
            'OR_p75': or_75,
            'spearman_r': sp_r,
            'spearman_p': sp_p,
            'mean_score_full': sub['score_full'].mean(),
        })

    return pd.DataFrame(results)


def score_bin_analysis(df, n_bins=20):
    """Compute divergence rate in score bins for plotting."""
    df = df.copy()
    df['score_bin'] = pd.qcut(df['score_full'], n_bins, duplicates='drop')
    grouped = df.groupby('score_bin', observed=True).agg(
        n=('diverged', 'count'),
        n_diverged=('diverged', 'sum'),
        mean_score=('score_full', 'mean'),
    ).reset_index()
    grouped['div_rate'] = grouped['n_diverged'] / grouped['n']
    # 95% CI via normal approximation (Wald interval)
    p = grouped['div_rate']
    n = grouped['n']
    z = 1.96
    se = np.sqrt(p * (1 - p) / n)
    grouped['ci_lo'] = (p - z * se).clip(lower=0)
    grouped['ci_hi'] = (p + z * se).clip(upper=1)
    return grouped


def plot_divergence_vs_editability(bin_df, out_path):
    """Plot divergence rate vs editability score bins."""
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.errorbar(
        bin_df['mean_score'], bin_df['div_rate'],
        yerr=[bin_df['div_rate'] - bin_df['ci_lo'],
              bin_df['ci_hi'] - bin_df['div_rate']],
        fmt='o-', color='#2c7bb6', markersize=6, capsize=3, linewidth=1.5,
        label='Divergence rate'
    )

    ax.set_xlabel('Editability Score (full model)', fontsize=13)
    ax.set_ylabel('Human-Chimp Divergence Rate', fontsize=13)
    ax.set_title('Editability vs. Human-Chimp Divergence at Exonic C Positions', fontsize=14)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0, decimals=2))

    # Add trend line
    z = np.polyfit(bin_df['mean_score'], bin_df['div_rate'], 1)
    x_line = np.linspace(bin_df['mean_score'].min(), bin_df['mean_score'].max(), 100)
    ax.plot(x_line, np.polyval(z, x_line), '--', color='red', alpha=0.7,
            label=f'Linear fit (slope={z[0]:.4f})')

    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    log.info(f"  Saved: {out_path}")


def plot_or_thresholds(or_df, out_path):
    """Plot OR at percentile thresholds."""
    fig, ax = plt.subplots(figsize=(8, 5))

    colors = ['#4575b4', '#74add1', '#f46d43', '#d73027']
    bars = ax.bar(
        range(len(or_df)), or_df['OR'].values,
        color=colors[:len(or_df)], edgecolor='black', linewidth=0.5
    )

    ax.axhline(1.0, color='gray', linestyle='--', linewidth=1, label='OR=1 (no effect)')

    ax.set_xticks(range(len(or_df)))
    ax.set_xticklabels([f'p{int(r["percentile"])}\n(>{r["cutoff"]:.3f})'
                         for _, r in or_df.iterrows()], fontsize=11)
    ax.set_ylabel('Odds Ratio for Divergence', fontsize=13)
    ax.set_title('Divergence OR: High vs Low Editability', fontsize=14)

    # Annotate bars with OR and p-value
    for i, (_, row) in enumerate(or_df.iterrows()):
        p_str = f'p={row["p_value"]:.1e}' if row['p_value'] < 0.001 else f'p={row["p_value"]:.3f}'
        ax.text(i, row['OR'] + 0.002, f'OR={row["OR"]:.4f}\n{p_str}',
                ha='center', va='bottom', fontsize=9)

    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    log.info(f"  Saved: {out_path}")


def plot_trinuc_stratified(trinuc_df, out_path):
    """Plot divergence rate and OR by trinucleotide context."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: divergence rate by group
    ax = axes[0]
    groups = trinuc_df['motif_group'].values
    rates = trinuc_df['divergence_rate'].values
    colors = {'TC': '#2c7bb6', 'CC': '#d7191c', 'Other': '#999999'}
    bar_colors = [colors.get(g, '#999999') for g in groups]
    ax.bar(groups, rates, color=bar_colors, edgecolor='black', linewidth=0.5)
    ax.set_ylabel('Divergence Rate', fontsize=12)
    ax.set_title('Divergence by Motif Context', fontsize=13)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0, decimals=2))

    for i, (g, r) in enumerate(zip(groups, rates)):
        n = trinuc_df.loc[trinuc_df['motif_group'] == g, 'n'].values[0]
        ax.text(i, r + 0.0002, f'{r:.4f}\nn={n:,}', ha='center', va='bottom', fontsize=9)

    # Right: OR at p75 by group
    ax = axes[1]
    ors = trinuc_df['OR_p75'].values
    ax.bar(groups, ors, color=bar_colors, edgecolor='black', linewidth=0.5)
    ax.axhline(1.0, color='gray', linestyle='--', linewidth=1)
    ax.set_ylabel('OR (p75 threshold)', fontsize=12)
    ax.set_title('Divergence OR by Motif Context', fontsize=13)

    for i, (g, o) in enumerate(zip(groups, ors)):
        ax.text(i, o + 0.002, f'{o:.4f}', ha='center', va='bottom', fontsize=9)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    log.info(f"  Saved: {out_path}")


def plot_divergence_by_decile_and_motif(df, out_path):
    """Plot divergence rate by score decile, stratified by motif group."""
    df = df.copy()
    trinuc = df['trinuc'].str.upper()
    df['motif_group'] = 'Other'
    df.loc[trinuc.str[0] == 'T', 'motif_group'] = 'TC'
    df.loc[trinuc.str[0] == 'C', 'motif_group'] = 'CC'

    df['score_decile'] = pd.qcut(df['score_full'], 10, labels=False, duplicates='drop')

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = {'TC': '#2c7bb6', 'CC': '#d7191c', 'Other': '#999999'}

    for group in ['TC', 'CC', 'Other']:
        sub = df[df['motif_group'] == group]
        grouped = sub.groupby('score_decile').agg(
            div_rate=('diverged', 'mean'),
            mean_score=('score_full', 'mean'),
            n=('diverged', 'count'),
        ).reset_index()
        ax.plot(grouped['mean_score'], grouped['div_rate'], 'o-',
                color=colors[group], label=f'{group} (n={len(sub):,})',
                markersize=5, linewidth=1.5)

    ax.set_xlabel('Editability Score (full model)', fontsize=13)
    ax.set_ylabel('Divergence Rate', fontsize=13)
    ax.set_title('Divergence vs Editability by Motif Context', fontsize=14)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0, decimals=2))
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    log.info(f"  Saved: {out_path}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    log.info("=" * 70)
    log.info("E1: Editability as Predictor of Human-Chimp Divergence")
    log.info("=" * 70)

    # ── 1-3. Load data / liftover / CpG (or use cache) ────────────────
    cache_path = OUT_DIR / "lifted_positions.csv.gz"
    if cache_path.exists():
        log.info("[1-3/6] Loading cached lifted_positions.csv.gz...")
        df_lifted = pd.read_csv(cache_path)
        log.info(f"  Loaded {len(df_lifted):,} lifted positions from cache")
        log.info(f"  Conserved: {df_lifted['conserved'].sum():,} "
                 f"({100*df_lifted['conserved'].mean():.2f}%)")
        log.info(f"  Diverged: {df_lifted['diverged'].sum():,} "
                 f"({100*df_lifted['diverged'].mean():.2f}%)")
        log.info(f"  CpG positions: {df_lifted['is_cpg'].sum():,} "
                 f"({100*df_lifted['is_cpg'].mean():.1f}%)")
    else:
        log.info("\n[1/6] Loading exome editability map...")
        df = load_exome_map()

        log.info("\n[2/6] Liftover hg19 → panTro6 and checking conservation...")
        lo = LiftOver('hg19', 'panTro6')
        chimp_fa = Fasta(str(PANTRO6_FA))
        hg19_fa = Fasta(str(HG19_FA))

        lifted, conserved, chimp_bases = liftover_and_check(df, lo, chimp_fa)

        df['lifted'] = lifted
        df['conserved'] = conserved
        df['chimp_base'] = chimp_bases
        df['diverged'] = lifted & ~conserved

        n_lifted = lifted.sum()
        n_cons = (lifted & conserved).sum()
        n_div = (lifted & ~conserved).sum()
        log.info(f"  Lifted: {n_lifted:,} / {len(df):,} ({100*n_lifted/len(df):.1f}%)")
        log.info(f"  Conserved: {n_cons:,} ({100*n_cons/n_lifted:.2f}%)")
        log.info(f"  Diverged: {n_div:,} ({100*n_div/n_lifted:.2f}%)")

        df_lifted = df[df['lifted']].copy().reset_index(drop=True)
        log.info(f"  Working set: {len(df_lifted):,} positions")

        log.info("\n[3/6] Computing CpG context...")
        df_lifted['is_cpg'] = compute_cpg_context(df_lifted, hg19_fa)
        cpg_n = df_lifted['is_cpg'].sum()
        log.info(f"  CpG positions: {cpg_n:,} ({100*cpg_n/len(df_lifted):.1f}%)")

        save_cols = ['chr', 'pos', 'strand', 'score_full', 'score_motif', 'trinuc',
                     'gene', 'conserved', 'diverged', 'chimp_base', 'is_cpg']
        df_lifted[save_cols].to_csv(cache_path, index=False, compression='gzip')
        log.info(f"  Saved lifted_positions.csv.gz")

    # ── 4. Percentile OR analysis ────────────────────────────────────────
    log.info("\n[4/6] Percentile threshold OR analysis...")
    or_df = percentile_or_analysis(df_lifted)
    or_df.to_csv(OUT_DIR / "or_thresholds.csv", index=False)
    log.info(f"\n  OR at percentile thresholds:")
    for _, row in or_df.iterrows():
        direction = "LESS div" if row['OR'] < 1 else "MORE div"
        log.info(f"    p{int(row['percentile']):2d} (>{row['cutoff']:.3f}): "
                 f"OR={row['OR']:.4f} ({direction}), "
                 f"div_high={row['div_rate_high']:.4f}, "
                 f"div_low={row['div_rate_low']:.4f}, "
                 f"p={row['p_value']:.2e}")

    # ── 5. Logistic regression ───────────────────────────────────────────
    log.info("\n[5/6] Logistic regression: P(diverged) ~ editability + trinuc + CpG...")
    lr_results = logistic_regression_analysis(df_lifted)

    log.info(f"\n  Logistic Regression Results:")
    log.info(f"    N = {lr_results['n_samples']:,}, "
             f"diverged = {lr_results['n_diverged']:,} "
             f"({100*lr_results['divergence_rate']:.2f}%)")
    log.info(f"    score_full coef (edit-only model): {lr_results['coef_score_full_edit_only']:.4f}")
    log.info(f"    score_full coef (full model):      {lr_results['coef_score_full']:.4f} "
             f"(SE={lr_results['coef_score_full_se']:.4f}, "
             f"z={lr_results['coef_score_full_z']:.2f}, "
             f"p={lr_results['coef_score_full_p']:.2e})")
    log.info(f"    CpG coef:  {lr_results['coef_is_cpg']:.4f}")
    log.info(f"    Motif coef: {lr_results['coef_score_motif']:.4f}")
    log.info(f"    Pseudo-R²: {lr_results['pseudo_r2']:.6f}")

    direction = "protective (less divergence)" if lr_results['coef_score_full'] < 0 \
        else "risk (more divergence)"
    log.info(f"    → Editability is {direction} after controlling for confounds")

    # Save LR results
    lr_save = {k: v for k, v in lr_results.items() if k != 'all_coefs'}
    lr_save['top_trinuc_coefs'] = {k: v for k, v in
                                    sorted(lr_results['all_coefs'].items(),
                                           key=lambda x: abs(x[1]), reverse=True)[:10]}
    with open(OUT_DIR / "logistic_regression.json", 'w') as f:
        json.dump(lr_save, f, indent=2, default=str)

    # ── 6. Stratified analyses and plots ─────────────────────────────────
    log.info("\n[6/6] Stratified analyses and plots...")

    # Trinucleotide stratification
    trinuc_df = trinuc_stratified_analysis(df_lifted)
    trinuc_df.to_csv(OUT_DIR / "trinuc_stratified.csv", index=False)
    log.info(f"\n  Trinucleotide-stratified results:")
    for _, row in trinuc_df.iterrows():
        log.info(f"    {row['motif_group']}: n={row['n']:,}, "
                 f"div={row['divergence_rate']:.4f}, "
                 f"OR_p75={row['OR_p75']:.4f}, "
                 f"spearman_r={row['spearman_r']:.4f} (p={row['spearman_p']:.2e})")

    # Score bin analysis
    bin_df = score_bin_analysis(df_lifted, n_bins=20)
    bin_df.to_csv(OUT_DIR / "score_bins.csv", index=False)

    # ── Plots ────────────────────────────────────────────────────────────
    log.info("\n  Generating plots...")
    plot_divergence_vs_editability(bin_df, OUT_DIR / "divergence_vs_editability.png")
    plot_or_thresholds(or_df, OUT_DIR / "or_thresholds.png")
    plot_trinuc_stratified(trinuc_df, OUT_DIR / "trinuc_stratified.png")
    plot_divergence_by_decile_and_motif(df_lifted, OUT_DIR / "divergence_by_decile_motif.png")

    # ── Summary ──────────────────────────────────────────────────────────
    log.info("\n" + "=" * 70)
    log.info("SUMMARY")
    log.info("=" * 70)
    log.info(f"Total lifted positions: {len(df_lifted):,}")
    log.info(f"Overall divergence rate: {lr_results['divergence_rate']:.4f} "
             f"({lr_results['n_diverged']:,} / {lr_results['n_samples']:,})")
    log.info(f"Editability coefficient (full model): {lr_results['coef_score_full']:.4f} "
             f"(p={lr_results['coef_score_full_p']:.2e})")

    key_or = or_df[or_df['percentile'] == 90].iloc[0]
    log.info(f"OR at p90: {key_or['OR']:.4f} (p={key_or['p_value']:.2e})")

    if lr_results['coef_score_full'] < 0:
        log.info("\nCONCLUSION: High editability predicts LOWER divergence → "
                 "consistent with purifying selection on editable positions")
    else:
        log.info("\nCONCLUSION: High editability predicts HIGHER divergence → "
                 "editable positions are NOT under additional purifying selection")

    log.info(f"\nOutputs saved to: {OUT_DIR}")
    log.info("Done.")


if __name__ == "__main__":
    main()
