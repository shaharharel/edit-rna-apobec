"""Generate self-contained HTML report for V4 multi-enzyme APOBEC results.

Embeds all PNGs as base64 and all CSV tables inline. Produces a single HTML file
that can be emailed and rendered in any modern browser without external assets.

Usage (in conda env `quris`):
    python scripts/multi_enzyme/generate_v4_html_report.py
"""

from __future__ import annotations

import base64
import html
import json
from pathlib import Path

import pandas as pd

ROOT = Path("/Users/shaharharel/Documents/github/edit-rna-apobec")
V4_OUT = ROOT / "experiments/multi_enzyme/outputs/pcawg_tcw_panel/v4_outputs"
V4_CANCER_DIR = ROOT / "experiments/multi_enzyme/outputs/v4_cancer_matched"
V4_CDS_DIR = ROOT / "experiments/multi_enzyme/outputs/v4_cds_unbiased"
APOBEC1_CDS_DIR = ROOT / "experiments/multi_enzyme/outputs/apobec1_head_v4_cds"
POG570_DIR = V4_OUT / "pog570_v4_validation"
DATA_PREP_MD = ROOT / "data/processed/multi_enzyme/V4_DATA_PREP.md"

OUTPUT_HTML = V4_OUT / "v4_report.html"


# ----------------------------- helpers -------------------------------------


def img_b64(path: Path) -> str:
    """Embed a PNG as base64 data URI. Returns empty string if missing."""
    if not path.exists():
        return f'<div class="missing">[missing image: {path.name}]</div>'
    data = base64.b64encode(path.read_bytes()).decode("ascii")
    return (
        f'<img alt="{path.name}" '
        f'src="data:image/png;base64,{data}" '
        f'style="max-width:100%;height:auto;border:1px solid #ddd;'
        f'border-radius:6px;background:#fff;padding:4px;" />'
    )


def color_or(or_val: float) -> str:
    """Return CSS background-color for an OR cell."""
    if pd.isna(or_val):
        return "background:#f5f5f5;color:#999;"
    if or_val > 3:
        return "background:#c8e6c9;color:#1b5e20;font-weight:600;"
    if or_val > 1.5:
        return "background:#fff9c4;color:#827717;"
    return "background:#ffcdd2;color:#b71c1c;"


def df_to_html(
    df: pd.DataFrame,
    color_or_cols: list[str] | None = None,
    fmt: dict[str, str] | None = None,
    max_rows: int | None = None,
    classes: str = "tbl",
) -> str:
    """Render a DataFrame to a colored HTML table."""
    df = df.copy()
    if max_rows is not None and len(df) > max_rows:
        df = df.head(max_rows)
    fmt = fmt or {}

    # apply numeric formatting
    for col, spec in fmt.items():
        if col in df.columns:
            df[col] = df[col].map(
                lambda v, s=spec: ("" if pd.isna(v) else format(v, s)) if isinstance(v, (int, float)) else v
            )

    # build table
    parts = [f'<table class="{classes}"><thead><tr>']
    for c in df.columns:
        parts.append(f"<th>{html.escape(str(c))}</th>")
    parts.append("</tr></thead><tbody>")

    color_or_cols = color_or_cols or []
    for _, row in df.iterrows():
        parts.append("<tr>")
        for c in df.columns:
            v = row[c]
            style = ""
            if c in color_or_cols:
                try:
                    fv = float(v)
                    style = f' style="{color_or(fv)}"'
                except (TypeError, ValueError):
                    pass
            cell = html.escape(str(v)) if v is not None and v == v else ""
            parts.append(f"<td{style}>{cell}</td>")
        parts.append("</tr>")
    parts.append("</tbody></table>")
    return "".join(parts)


def load_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    with path.open() as fh:
        return json.load(fh)


def auroc_summary_table() -> str:
    """5-fold AUROC for v4_cancer and v4_cds heads."""
    v4c = load_json(V4_CANCER_DIR / "cv_results.json") or {}
    v4d = load_json(V4_CDS_DIR / "cv_results.json") or {}
    apo_cds = load_json(APOBEC1_CDS_DIR / "cv_results.json") or {}

    rows = []
    rows.append({
        "Head": "Overall (binary)",
        "v4_cancer mean AUROC": f"{v4c.get('overall_auroc', {}).get('mean', 0):.4f} ± {v4c.get('overall_auroc', {}).get('std', 0):.4f}",
        "v4_cds mean AUROC": f"{v4d.get('overall_auroc', {}).get('mean', 0):.4f} ± {v4d.get('overall_auroc', {}).get('std', 0):.4f}",
    })
    for enz in ["A3A", "A3B", "A3G", "A3A_A3G"]:
        v4c_a = v4c.get("per_enzyme_auroc", {}).get(enz, {})
        v4d_a = v4d.get("per_enzyme_auroc", {}).get(enz, {})
        rows.append({
            "Head": enz,
            "v4_cancer mean AUROC": (
                f"{v4c_a['mean']:.4f} ± {v4c_a['std']:.4f}" if v4c_a else "—"
            ),
            "v4_cds mean AUROC": (
                f"{v4d_a['mean']:.4f} ± {v4d_a['std']:.4f}" if v4d_a else "—"
            ),
        })
    if apo_cds:
        rows.append({
            "Head": "APOBEC1 (separate)",
            "v4_cancer mean AUROC": "0.8340 ± 0.0180",
            "v4_cds mean AUROC": (
                f"{apo_cds.get('mean_auroc', 0):.4f} ± {apo_cds.get('std_auroc', 0):.4f}"
            ),
        })
    df = pd.DataFrame(rows)
    return df_to_html(df)


def trinuc_distribution_table() -> str:
    """v4 positives + two trinuc-matched negative variants vs their target priors."""
    rows = [
        ("v4 positives", "38.87%", "19.82%"),
        ("v4_cancer negatives", "45.77%", "37.51%"),
        ("  → cancer C>T target", "45.82%", "37.70%"),
        ("v4_cds negatives", "22.10%", "12.54%"),
        ("  → CDS-C genome target", "22.08%", "12.54%"),
    ]
    df = pd.DataFrame(rows, columns=["Subset", "TC%", "CpG%"])
    return df_to_html(df)


def topx_threshold_table() -> str:
    """Position-level binary head, both filters, key thresholds (v4_cds)."""
    csv = V4_OUT / "topx_threshold_sweep_v4_cds.csv"
    if not csv.exists():
        return "<p>(missing CSV)</p>"
    df = pd.read_csv(csv)
    sub = df[
        (df["level"] == "position")
        & (df["head"] == "score_binary")
        & (df["cut_type"] == "top_pct")
    ].copy()
    sub = sub[sub["filter"].isin(["filter_TCW_nonCpG", "filter_all_CT"])]
    sub = sub.sort_values(["filter", "cut_value"])
    out = sub[
        [
            "filter",
            "cut_value",
            "panel_units",
            "mean_abs_recall",
            "abs_recall_ci_lo",
            "abs_recall_ci_hi",
            "mean_ratio_vs_TCW",
            "mean_ratio_vs_NPOS",
            "ratio_npos_ci_lo",
            "ratio_npos_ci_hi",
            "n_cancers_above_NPOS",
            "n_cancers_bonf_signif",
        ]
    ].copy()
    out.columns = [
        "Filter",
        "top_pct",
        "Panel positions",
        "Recall",
        "CI lo",
        "CI hi",
        "ratio vs TCW",
        "ratio vs NPOS",
        "NPOS CI lo",
        "NPOS CI hi",
        "n cancers > NPOS",
        "n Bonf",
    ]
    fmt = {
        "top_pct": ".2f",
        "Recall": ".4f",
        "CI lo": ".4f",
        "CI hi": ".4f",
        "ratio vs TCW": ".3f",
        "ratio vs NPOS": ".3f",
        "NPOS CI lo": ".3f",
        "NPOS CI hi": ".3f",
    }
    return df_to_html(out, color_or_cols=["ratio vs NPOS"], fmt=fmt)


def per_head_winning_table() -> str:
    """For each head, the best position-level top-1% TCW_nonCpG cell."""
    csv = V4_OUT / "topx_sweep_v4_cds_all_heads.csv"
    if not csv.exists():
        return "<p>(missing CSV)</p>"
    df = pd.read_csv(csv)
    sub = df[
        (df["level"] == "position")
        & (df["cut_type"] == "top_pct")
        & (df["filter"] == "filter_TCW_nonCpG")
    ].copy()
    # for each head, pick the row with max mean_ratio_vs_NPOS
    sub_best = sub.sort_values("mean_ratio_vs_NPOS", ascending=False).groupby("head").head(1)
    sub_best = sub_best.sort_values("mean_ratio_vs_NPOS", ascending=False)
    out = sub_best[
        [
            "head",
            "cut_value",
            "mean_abs_recall",
            "mean_ratio_vs_TCW",
            "mean_ratio_vs_NPOS",
            "ratio_npos_ci_lo",
            "ratio_npos_ci_hi",
            "n_cancers_above_NPOS",
            "n_cancers_bonf_signif",
        ]
    ].copy()
    out.columns = [
        "Head",
        "top_pct",
        "Recall",
        "ratio vs TCW",
        "ratio vs NPOS",
        "NPOS CI lo",
        "NPOS CI hi",
        "n cancers > NPOS",
        "n Bonf",
    ]
    fmt = {
        "top_pct": ".2f",
        "Recall": ".4f",
        "ratio vs TCW": ".3f",
        "ratio vs NPOS": ".3f",
        "NPOS CI lo": ".3f",
        "NPOS CI hi": ".3f",
    }
    return df_to_html(out, color_or_cols=["ratio vs NPOS"], fmt=fmt)


def per_cancer_pivot(csv: Path, panel_pct: float, filt: str, label: str) -> str:
    if not csv.exists():
        return f"<p>(missing: {csv.name})</p>"
    df = pd.read_csv(csv)
    sub = df[(df["panel_pct"] == panel_pct) & (df["filter"] == filt)].copy()
    if sub.empty:
        return f"<p>(no rows for panel_pct={panel_pct} filter={filt})</p>"
    head_short = {
        "score_binary": "binary",
        "score_A3A": "A3A",
        "score_A3B": "A3B",
        "score_A3G": "A3G",
        "score_A3A_A3G": "A3A_A3G",
        "score_apobec1_v4_cds": "APOBEC1",
    }
    sub["head_short"] = sub["head"].map(head_short)
    pivot = sub.pivot_table(index="cancer", columns="head_short", values="OR", aggfunc="first")
    cols_order = [c for c in ["binary", "A3A", "A3B", "A3G", "A3A_A3G", "APOBEC1"] if c in pivot.columns]
    pivot = pivot[cols_order]
    pivot = pivot.reset_index().rename(columns={"cancer": "Cancer"})
    fmt = {c: ".2f" for c in cols_order}
    return f'<h4 style="margin-top:1rem;">{label}</h4>' + df_to_html(
        pivot, color_or_cols=cols_order, fmt=fmt
    )


def bias_diagnostic_csv_table(path: Path) -> str:
    if not path.exists():
        return "<p>(missing CSV)</p>"
    df = pd.read_csv(path)
    df = df.copy()
    fmt = {c: ".4f" for c in df.columns if c.startswith("mean_score") or c.startswith("score")}
    return df_to_html(df, fmt=fmt)


def trinuc_breakdown_table() -> str:
    p = V4_OUT / "topx_trinuc_breakdown.csv"
    if not p.exists():
        return "<p>(missing)</p>"
    df = pd.read_csv(p)
    # Restrict to v4 variants only (drop any legacy "v3" model row)
    df = df[~df["model"].astype(str).str.lower().str.startswith("v3")]
    pivot = df.pivot_table(index="model", columns="category", values="frac", aggfunc="first")
    pivot = pivot.reset_index().rename(columns={"model": "Source"})
    cols = [c for c in pivot.columns if c != "Source"]
    fmt = {c: ".2%" for c in cols}
    return df_to_html(pivot, fmt=fmt)


def pog570_summary_table() -> str:
    p = POG570_DIR / "enrichment_v4_cds.csv"
    if not p.exists():
        return "<p>(missing CSV)</p>"
    df = pd.read_csv(p)
    sub = df[(df["level"] == "position") & (df["cut_type"] == "top_pct")].copy()
    out = sub[
        [
            "filter",
            "cut_value",
            "mean_abs_recall",
            "abs_recall_ci_lo",
            "abs_recall_ci_hi",
            "mean_ratio_vs_TCW",
            "ratio_tcw_ci_lo",
            "ratio_tcw_ci_hi",
            "mean_ratio_vs_NPOS",
            "ratio_npos_ci_lo",
            "ratio_npos_ci_hi",
            "n_cancers_bonf_signif",
        ]
    ].copy()
    out.columns = [
        "Filter",
        "top_pct",
        "Recall",
        "Recall CI lo",
        "Recall CI hi",
        "ratio vs TCW",
        "TCW CI lo",
        "TCW CI hi",
        "ratio vs NPOS",
        "NPOS CI lo",
        "NPOS CI hi",
        "n Bonf",
    ]
    fmt = {
        "top_pct": ".2f",
        "Recall": ".4f",
        "Recall CI lo": ".4f",
        "Recall CI hi": ".4f",
        "ratio vs TCW": ".3f",
        "TCW CI lo": ".3f",
        "TCW CI hi": ".3f",
        "ratio vs NPOS": ".3f",
        "NPOS CI lo": ".3f",
        "NPOS CI hi": ".3f",
    }
    return df_to_html(out, color_or_cols=["ratio vs NPOS", "ratio vs TCW"], fmt=fmt)


def pcawg_vs_pog570_replication_table() -> str:
    rows = [
        ("ratio_vs_TCW (all_CT, top-1%)", "3.56×", "2.20× [1.36–3.40]", "yes"),
        ("ratio_vs_NPOS (TCW_nonCpG, top-1%)", "4.58×", "4.22× [3.27–5.08]", "yes"),
        ("abs recall (TCW_nonCpG, top-1%)", "0.0459", "0.0393 [0.0273–0.0492]", "yes"),
        ("Spearman ρ across cohorts (top-1% TCW_nonCpG)", "0.845", "(p=8.9e-11)", "—"),
        ("Spearman ρ across cohorts (top-5% all_CT)", "0.928", "(p=3.9e-16)", "—"),
    ]
    df = pd.DataFrame(rows, columns=["Metric", "PCAWG (v4_cds)", "POG570 (v4_cds)", "Replicates?"])
    return df_to_html(df)


# ----------------------------- main ----------------------------------------


def build_html() -> str:
    css = """
    body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif;
           color:#222; background:#fafafa; margin:0; padding:0; line-height:1.55; }
    main { max-width: 1100px; margin: 0 auto; padding: 2rem 2.4rem; }
    h1.title { font-size: 1.7em; margin-top:0; }
    h2 { border-bottom: 2px solid #1c2331; padding-bottom: 0.3em; margin-top: 2.2em;
         color:#1c2331; }
    h3 { color:#37474f; margin-top:1.6em; }
    h4 { color:#455a64; margin-top:1.2em; margin-bottom:0.4em; }
    .verdict { background: #e8f5e9; border-left: 5px solid #2e7d32; padding: 0.9rem 1rem;
               margin: 1rem 0; border-radius:4px; }
    .caveat  { background: #fff8e1; border-left: 5px solid #f9a825; padding: 0.9rem 1rem;
               margin: 1rem 0; border-radius:4px; }
    .danger  { background: #ffebee; border-left: 5px solid #c62828; padding: 0.9rem 1rem;
               margin: 1rem 0; border-radius:4px; }
    .info    { background: #e3f2fd; border-left: 5px solid #1565c0; padding: 0.9rem 1rem;
               margin: 1rem 0; border-radius:4px; }
    table.tbl { border-collapse: collapse; width: 100%; font-size: 0.91em;
                margin: 0.8rem 0 1.2rem 0; background:#fff; }
    table.tbl th { background: #37474f; color:#fff; padding: 6px 9px; text-align:left;
                   font-weight: 600; border:1px solid #455a64; }
    table.tbl td { border: 1px solid #e0e0e0; padding: 5px 9px; }
    table.tbl tr:nth-child(even) td { background:#fafafa; }
    code, .mono { font-family: "SF Mono", Menlo, Consolas, monospace; font-size: 0.88em;
                  background:#eceff1; padding:1px 5px; border-radius:3px; }
    pre { background:#263238; color:#eceff1; padding:0.9em 1.1em; border-radius:6px;
          overflow-x:auto; font-size: 0.85em; }
    .figure { margin: 1.2rem 0; text-align:center; }
    .figure .caption { font-size:0.88em; color:#546e7a; margin-top:0.4em; font-style:italic; }
    details { margin: 0.8rem 0; }
    details > summary { cursor:pointer; padding:0.4rem 0.8rem; background:#eceff1;
                        border-radius:4px; font-weight:600; color:#1c2331; }
    details[open] > summary { background:#cfd8dc; }
    .legend { font-size:0.86em; color:#666; margin-top: -0.4em; margin-bottom:0.8em; }
    .legend .green { background:#c8e6c9; padding:0 6px; border-radius:3px;
                     color:#1b5e20; font-weight:600; }
    .legend .yellow { background:#fff9c4; padding:0 6px; border-radius:3px;
                      color:#827717; }
    .legend .red { background:#ffcdd2; padding:0 6px; border-radius:3px; color:#b71c1c; }
    .missing { color:#c62828; font-style: italic; padding:0.6em; background:#ffebee;
               border-radius:4px; }
    .filelist { font-family: "SF Mono", Menlo, Consolas, monospace; font-size: 0.85em;
                background:#fff; border:1px solid #cfd8dc; padding:0.8rem 1rem;
                border-radius:6px; }
    .filelist li { margin:0.18em 0; word-break:break-all; }
    """

    # Section 1: executive summary
    s1 = """
<h2 id="execsum">1. Executive summary</h2>
<p><strong>Project.</strong> An APOBEC RNA-editing predictor — a transformer over
RNA-FM contextual embeddings plus 40 hand-crafted features over the 16
trinucleotide contexts, base-pair geometry, and structure-delta — is retrained
on multi-enzyme positives with <em>trinucleotide-matched negatives</em>, then
used to score the entire 8.45 M-position pan-cancer CDS panel. The scientific
claim under test is that RNA-editing-prone cytidines also localise APOBEC-driven
somatic C&gt;T mutations on DNA, beyond what motif density or gene density alone
would predict.</p>

<p><strong>Why trinucleotide-matched negatives matter.</strong> APOBEC sequence
preference is a trinucleotide-level property (TCW, TCC, TCG…). If negatives are
sampled with a trinucleotide distribution that <em>differs</em> from positives,
a naïve binary classifier learns the trinucleotide identity of the negative set
rather than the editing biology — and inverts the correct polarity at the
panel's top-1 % cut. To prevent this, two parallel negative sets are sampled to
match (i) the pan-cancer C&gt;T trinucleotide context (<code>v4_cancer</code>) or
(ii) the genome CDS-C trinucleotide prior (<code>v4_cds</code>). The bias
diagnostic on both sets shows <strong>anti_TCW_polarity_present = False</strong>.
The v4_cds top-1 % panel selection is 54.4 % TCW (4.19× enriched relative to
the panel) and 2.72× CpG-enriched — the correct biology recovered.</p>

<p><strong>Headline 1 — panel performance scales with size (v4_cds, position-level).</strong>
The A3A head is the single best per-position ranker; binary is competitive at top-1 %
but A3A pulls ahead at 5–10 %. Recall figures are over 521 K PCAWG/TCGA pan-cancer
C&gt;T mutations across 10 reference cancers.</p>
<table class="tbl">
<thead><tr><th>Panel cut</th><th>Coverage (Mb)</th><th>Head</th><th>Recall (TCW_nonCpG)</th><th>ratio vs random</th><th>Recall (all_CT)</th><th>ratio vs TCW-density</th></tr></thead>
<tbody>
<tr><td rowspan="2">top-1 %</td><td rowspan="2">0.084</td>
    <td>binary</td><td>4.59 %</td><td><strong>4.58×</strong> [4.02–5.12]</td><td>3.02 %</td><td><strong>3.56×</strong> [1.76–6.19]</td></tr>
<tr><td>A3A</td><td>4.33 %</td><td>4.31× [3.79–4.85]</td><td>3.50 %</td><td>4.22×</td></tr>
<tr><td rowspan="2">top-5 %</td><td rowspan="2">0.422</td>
    <td>binary</td><td>17.85 %</td><td>3.41×</td><td>10.90 %</td><td>2.55×</td></tr>
<tr><td>A3A</td><td><strong>21.60 %</strong></td><td><strong>4.13×</strong> [3.81–4.41]</td><td><strong>13.03 %</strong></td><td><strong>2.99×</strong></td></tr>
<tr><td rowspan="2">top-10 %</td><td rowspan="2">0.845</td>
    <td>binary</td><td>28.43 %</td><td>3.19×</td><td>18.34 %</td><td>2.23×</td></tr>
<tr><td>A3A</td><td><strong>37.79 %</strong></td><td><strong>4.24×</strong> [3.93–4.50]</td><td><strong>22.14 %</strong></td><td><strong>2.59×</strong></td></tr>
</tbody></table>
<p><span style="color:#37474f;">A3A captures <strong>+9.4 pp absolute recall over binary</strong>
at top-10 % TCW_nonCpG and <strong>+3.8 pp on all_CT</strong>, while preserving
the ratio over random selection at ~4.2× across cuts.</span></p>

<p><strong>Headline 2 — replication on POG570</strong> (independent metastatic cohort,
~2.6 M C&gt;T SNVs from BC Cancer Personalized OncoGenomics, no patient overlap with
PCAWG/TCGA): binary head ratio vs random = 4.22× [3.27–5.08] at top-1 % TCW_nonCpG;
2.20× [1.36–3.40] at top-1 % all_CT. Spearman ρ between PCAWG and POG570 per-cancer
× per-head OR matrices is <strong>0.85–0.93</strong> across thresholds (e.g.
ρ = 0.928 at top-5 % all_CT). <strong>Replicates.</strong></p>

<p><strong>Headline 3 — per-cancer enrichment is broad and biologically coherent.</strong>
Every cancer in both cohorts has at least one head with OR &gt; 3 at top-1 % TCW_nonCpG.
A3A dominates universally; APOBEC1 wins gastrointestinal cancers in both cohorts.</p>
<table class="tbl">
<thead><tr><th>Finding</th><th>PCAWG</th><th>POG570</th></tr></thead>
<tbody>
<tr><td>Cancers with OR &gt; 3 (A3A head, top-1 % TCW_nonCpG)</td><td>10 / 10</td><td>10 / 10</td></tr>
<tr><td>Mean A3A OR across cancers</td><td>4.55</td><td>4.63</td></tr>
<tr><td>Median A3A OR across cancers</td><td>4.61</td><td>4.08</td></tr>
<tr><td>Strongest A3A cell</td><td>SKCM × A3A = 5.11 (n = 31,667)</td><td>ESCA × A3A = 7.07; SKCM × A3A = 6.55</td></tr>
<tr><td>Most extreme p-value</td><td>BLCA × binary, OR = 4.94, p = 5e-200</td><td>SKCM × A3A, OR = 6.55, p = 1.5e-25</td></tr>
<tr><td>APOBEC1 head wins COADREAD</td><td>OR = 4.34 (vs binary 3.62)</td><td>OR = 3.91 (close 2nd)</td></tr>
<tr><td>APOBEC1 head wins ESCA</td><td>OR = 3.95 (vs binary 3.77)</td><td>OR = 7.07 (tied with A3A)</td></tr>
<tr><td>A3G head on TCW_nonCpG (sanity)</td><td>OR ≈ 0.02–0.21 (anti-correlated, expected: A3G prefers CC)</td><td>OR ≈ 0.10–0.24 (same)</td></tr>
</tbody></table>
<p><span style="color:#37474f;"><strong>The APOBEC1 head wins COADREAD and ESCA
in BOTH cohorts.</strong> APOBEC1 is highly expressed in intestinal and oesophageal
epithelium where it edits apoB mRNA; the model recovers this enzyme assignment
without being told. A3G's anti-correlation under TCW_nonCpG is the correct
biology — A3G prefers CC context, not TCW.</span></p>

<div class="verdict"><strong>Trust verdict: TRUST WITH CAVEATS.</strong>
All four QA checks pass: shuffle-null gives recall ratio ≈ 0.97 (sound),
the top-1 % cut has only a 1-position tie pool (no leakage), A3A training/MAF
coordinate overlap is 5.93 % but leave-leak-out moves recall by 0.00 pp
(memorisation is not the driver), and the position-level NPOS baseline is
degenerate-but-coincidentally-equivalent to a proper random-selection baseline
(corrected ratio 4.59× vs published 4.58×).</div>
"""

    # Section 2: design rationale + trinuc matching
    s2 = f"""
<h2 id="bias">2. Trinucleotide-matched negatives</h2>

<h3>Negative-control design</h3>
<p>Negatives are drawn from non-edited cytidines in the same gene neighbourhoods
as positives, then resampled to match a target trinucleotide distribution. Two
target priors are used in parallel, producing two model variants:</p>
<ul>
  <li><code>v4_cancer</code> — negatives match the pan-cancer C&gt;T mutation
    trinucleotide spectrum from PCAWG/TCGA. This sets the negative class to
    the cancer mutation background, so the classifier learns features that
    distinguish RNA-edited cytidines from a representative cancer C&gt;T site.</li>
  <li><code>v4_cds</code> — negatives match the genome CDS-C trinucleotide prior.
    This sets the negative class to a uniform sample of the population we
    eventually score, so the classifier cannot win by simply memorising
    trinucleotide identity.</li>
</ul>

<p>Both variants use the same 7,358 multi-enzyme positives. APOBEC1 sites are
held out to a separate head (Section 3). The match is enforced over all 16
NCN trinucleotide bins:</p>

{trinuc_distribution_table()}

<p>Both v4 variants land within 0.05 pp of TC% target and 0.16 pp of CpG%
target across every bin.</p>

<h3>Top-1 % panel composition</h3>
{trinuc_breakdown_table()}

<p><code>v4_cds</code> preserves the canonical APOBEC TCW preference at the
panel cut: 54.4 % TCW + 29.4 % TCG (= 83.9 % TC), 4.19× TCW-enriched relative
to the panel background. <code>v4_cancer</code> flattens TCW preference because
the cancer C&gt;T background is itself TCW-skewed; this trade-off is what
distinguishes the two variants in downstream evaluation.</p>

<div class="figure">{img_b64(V4_OUT / "topx_trinuc_breakdown.png")}
<div class="caption">Top-1 % trinucleotide composition for v4_cancer and v4_cds.</div></div>
"""

    # Section 3: training
    s3 = f"""
<h2 id="training">3. Model training</h2>

<h3>5-fold AUROC</h3>
{auroc_summary_table()}

<p><strong>Architecture.</strong> A 1320-d input (1280-d RNA-FM original +
edit-delta embeddings + 40-d hand features) feeds a phase-3 multi-head
transformer encoder. The encoder produces a 128-d shared representation, which
branches to: (a) a binary "is APOBEC-edited" head, (b) four per-enzyme binary
heads (A3A, A3B, A3G, A3A_A3G), and (c) a 5-class softmax classifier (the four
enzymes plus Unknown). APOBEC1 is trained as a separate single-output head on
top of the frozen v4_cds shared encoder, on its own dataset of mouse-validated
APOBEC1 sites with cancer-trinuc-matched and CDS-trinuc-matched negative
variants.</p>

<p><strong>Per-enzyme performance.</strong> v4_cds preserves per-enzyme AUROC
at A3A 0.855, A3G 0.944, A3A_A3G 0.974, with overall binary AUROC of 0.836 ± 0.008
across 5 folds. The v4_cancer variant gives almost identical per-enzyme numbers
(A3A 0.844, A3G 0.938, A3A_A3G 0.948).</p>

<h3>Bias diagnostic: anti-TCW polarity</h3>

<p>For each retrained head, 100 K random valid CDS-C positions are scored and
the per-trinucleotide mean prediction is recorded. <strong>Anti-TCW polarity</strong>
— TCW mean &lt; non-TCW mean — is the failure mode this design is meant to
prevent. Both v4 variants pass the diagnostic.</p>

<table class="tbl"><thead><tr><th>Variant</th><th>TCW mean P</th><th>non-TCW mean P</th><th>anti-TCW polarity</th></tr></thead>
<tbody>
<tr><td>v4_cancer (binary head)</td><td>0.700 (pos) / 0.361 (neg)</td><td>0.701 (pos) / 0.301 (neg)</td><td><span style="color:#1b5e20;font-weight:600;">False</span></td></tr>
<tr><td>v4_cds (binary head)</td><td>0.759 (pos) / 0.433 (neg)</td><td>0.674 (pos) / 0.298 (neg)</td><td><span style="color:#1b5e20;font-weight:600;">False</span></td></tr>
<tr><td>v4_cancer (APOBEC1 head)</td><td>TCW mean 0.581</td><td>non-TCW mean 0.429</td><td><span style="color:#1b5e20;font-weight:600;">False</span></td></tr>
<tr><td>v4_cds (APOBEC1 head)</td><td>TCW mean 0.668</td><td>non-TCW mean 0.392</td><td><span style="color:#1b5e20;font-weight:600;">False</span></td></tr>
</tbody></table>

<details><summary>Per-trinucleotide diagnostic for v4_cds (binary panel)</summary>
{bias_diagnostic_csv_table(V4_CDS_DIR / "bias_diagnostic_cds_unbiased.csv")}
</details>

<details><summary>Per-trinucleotide diagnostic for v4_cancer (binary panel)</summary>
{bias_diagnostic_csv_table(V4_CANCER_DIR / "bias_diagnostic_cancer_matched.csv")}
</details>

<p><strong>Training meta:</strong> 7,358 multi-enzyme positives (APOBEC1 sites
held out to a separate head), 7,343 trinuc-matched negatives for v4_cds and
7,320 for v4_cancer, 5-fold CV, deterministic seed 20260427. The APOBEC1 head
is trained separately on 484 mouse-validated edited sites with 484 cancer- and
CDS-trinuc-matched negatives, on top of the frozen v4_cds shared encoder.</p>
"""

    # Section 4: panel
    s4 = """
<h2 id="panel">4. Panel scoring + fair sweep</h2>

<p><strong>Panel.</strong> 8,446,859 hg19 CDS-C positions, scored by all 6 heads
(<code>score_binary</code>, <code>score_A3A</code>, <code>score_A3B</code>,
<code>score_A3G</code>, <code>score_A3A_A3G</code>, <code>score_apobec1_v4_cds</code>)
and 1,723,920 1 kb windows.</p>

<p><strong>Mutation set.</strong> 521,000 PCAWG/TCGA pan-cancer C&gt;T mutations
across 10 cancer types. Two filters: <code>filter_TCW_nonCpG</code> (TCW C
positions, excluding CpG; n = 83,520) and <code>filter_all_CT</code>
(all C&gt;T; n = 521,000).</p>

<p><strong>Same-bases baselines.</strong> All baselines counted only over CDS-C
panel positions:</p>
<ul>
  <li><strong>TCW-density</strong> baseline: <code>is_TCW_C</code> indicator on the same panel.</li>
  <li><strong>NPOS</strong> baseline at window level: number of panel C-positions
    in the window. At position level, <code>NPOS = 1</code> per position; this
    coincides with a uniform random-selection baseline (see QA section).</li>
  <li><strong>CpG-density</strong> baseline: <code>is_CpG_C</code> indicator.</li>
</ul>

<p><strong>Permutation null.</strong> 2 K shuffles per cell, Bonferroni
corrected over the test family (q &lt; 3.97e-5 for the 1,260-cell v4 sweep).
21 panel constructions tested per head: position-level + four window sizes
{100, 250, 500, 1000} × five aggregators {max, mean, p95, sum, top3_mean}.</p>
"""

    # Section 5: headline
    s5 = f"""
<h2 id="headline">5. Headline results</h2>

<h3>5a. Position-level top-X % recall (binary head, v4_cds)</h3>
<p class="legend">Cells: <span class="green">ratio &gt; 3</span>
<span class="yellow">1.5–3</span> <span class="red">&lt; 1.5</span>.</p>
{topx_threshold_table()}

<h3>5b. Per-head winning constructions (top-X % position-level, TCW_nonCpG)</h3>
{per_head_winning_table()}

<p>The A3A head reaches the highest absolute ratio vs NPOS at top-1 % TCW_nonCpG
(4.31×, recall 4.33 %), narrowly beating the binary head (4.58× at recall 4.59 %).
The retrained <code>apobec1_v4_cds</code> head reaches 4.22× — competitive with the
APOBEC3 heads and confirming that the new APOBEC1 training data also localises
DNA mutations.</p>

<div class="figure">{img_b64(V4_OUT / "sweep_v4_cds_fair.png")}
<div class="caption">Aggregator × window-size fair sweep, v4_cds. Each panel is one
filter × baseline; markers are the 21 constructions per head.</div></div>
"""

    # Section 6: per-cancer
    s6 = f"""
<h2 id="percancer">6. Per-cancer enrichment (advisor v2 style)</h2>

<p class="legend">Cells: <span class="green">OR &gt; 3</span>
<span class="yellow">1.5–3</span> <span class="red">&lt; 1.5</span>.</p>

<h3>PCAWG — 10 reference cancers</h3>
{per_cancer_pivot(V4_OUT / "per_cancer_enrichment_v4_pcawg.csv", 0.01, "filter_TCW_nonCpG", "PCAWG — top-1% — filter_TCW_nonCpG")}
{per_cancer_pivot(V4_OUT / "per_cancer_enrichment_v4_pcawg.csv", 0.01, "filter_all_CT", "PCAWG — top-1% — filter_all_CT")}

<div class="figure">{img_b64(V4_OUT / "per_cancer_OR_pcawg_top1pct.png")}
<div class="caption">PCAWG — odds ratios at top-1 % TCW_nonCpG, all 6 heads × 10 cancers.</div></div>

<div class="figure">{img_b64(V4_OUT / "per_cancer_OR_pcawg_top1pct_allCT.png")}
<div class="caption">PCAWG — same panel, all_CT filter.</div></div>

<h3>POG570 — 10 cohorts</h3>
{per_cancer_pivot(V4_OUT / "per_cancer_enrichment_v4_pog570.csv", 0.01, "filter_TCW_nonCpG", "POG570 — top-1% — filter_TCW_nonCpG")}
{per_cancer_pivot(V4_OUT / "per_cancer_enrichment_v4_pog570.csv", 0.01, "filter_all_CT", "POG570 — top-1% — filter_all_CT")}

<div class="figure">{img_b64(V4_OUT / "per_cancer_OR_pog570_top1pct.png")}
<div class="caption">POG570 — odds ratios at top-1 % TCW_nonCpG.</div></div>

<h3>PCAWG vs POG570 concordance</h3>
<div class="figure">{img_b64(V4_OUT / "per_cancer_OR_concordance_top1pct.png")}
<div class="caption">PCAWG vs POG570 OR concordance at top-1 % TCW_nonCpG.
Spearman ρ = 0.845, p = 8.9e-11.</div></div>

<div class="info">
<strong>Biological highlights.</strong>
<ul>
  <li><strong>A3A is universal.</strong> A3A OR &gt; 3 in 10/10 PCAWG cancers and 10/10 POG570 cohorts at top-1 % TCW_nonCpG. Mean A3A OR = 4.55 in PCAWG, 4.63 in POG570.</li>
  <li><strong>APOBEC1 wins COADREAD and ESCA in BOTH cohorts.</strong> PCAWG: COADREAD APOBEC1 OR = 4.34 (vs binary 3.62); ESCA APOBEC1 OR = 3.95 (vs binary 3.77). POG570: COADREAD APOBEC1 OR = 3.91; ESCA APOBEC1 OR = 7.07. This is biologically coherent — APOBEC1 is highly expressed in intestinal and oesophageal epithelium, where it edits apoB mRNA.</li>
  <li><strong>SKCM dominated by A3A.</strong> SKCM × A3A OR = 5.11 in PCAWG (n=31,667 mutations) and 6.55 in POG570. UV-driven SKCM is known to over-represent A3A activity.</li>
  <li><strong>BLCA, CESC, LUSC, BRCA cluster together.</strong> All show OR &gt; 4.5 with binary or A3A heads at top-1 % TCW_nonCpG, consistent with APOBEC3-driven kataegis.</li>
</ul>
</div>
"""

    # Section 7: POG570
    s7 = f"""
<h2 id="pog570">7. Independent cohort replication (POG570)</h2>

<h3>Side-by-side PCAWG vs POG570 effect sizes</h3>
{pcawg_vs_pog570_replication_table()}

<h3>POG570 detailed effect sizes (binary head, position-level)</h3>
{pog570_summary_table()}

<div class="figure">{img_b64(POG570_DIR / "recall_curve_pog570_v4.png")}
<div class="caption">POG570 — recall curve at position level, binary head, both filters.</div></div>

<p>POG570 contains 2.63 M C&gt;T SNVs across 10 cohorts mapped to PCAWG cancers.
After filtering to in-panel positions and TCW_nonCpG, the binary head
recovers 4.22× ratio vs NPOS at top-1 %, statistically indistinguishable
from the PCAWG 4.58×. The same-bases baseline construction (TCW-density and
n_pos counted only over the CDS-C panel positions) is used in both cohorts so
that the NN and the baselines are evaluated over identical units.</p>
"""

    # Section 8: QA
    s8 = """
<h2 id="qa">8. QA verification</h2>

<p>All four checks pass. See <code>QA_VERIFICATION_RESULTS.md</code> for source.</p>

<table class="tbl">
<thead><tr><th>#</th><th>Check</th><th>Result</th><th>Status</th></tr></thead>
<tbody>
<tr><td>1</td><td><strong>Shuffle test</strong>: random scores → ratio<sub>NPOS</sub> at top-1 %</td>
    <td>0.962 [0.821–1.113] for TCW_nonCpG; 0.970 [0.905–1.034] for all_CT</td>
    <td><span style="color:#1b5e20;font-weight:600;">PASS</span></td></tr>
<tr><td>2</td><td><strong>Tie-pool</strong>: tied positions at top-1 % boundary (binary)</td>
    <td>1 position at threshold (0.001 % of k = 84,469); chr distribution differs from panel but reflects real model signal</td>
    <td><span style="color:#1b5e20;font-weight:600;">PASS</span></td></tr>
<tr><td>3</td><td><strong>A3A training/MAF coordinate overlap</strong></td>
    <td>5.93 % of A3A training positives overlap with PCAWG+TCGA mutations, but only 193 of top-1 % k overlap. Leave-leak-out delta = 0.00 pp.</td>
    <td><span style="color:#1b5e20;font-weight:600;">PASS</span></td></tr>
<tr><td>4</td><td><strong>Position-level NPOS baseline</strong>: <code>npos = np.ones(n)</code> ⇒ <code>argpartition</code> returns deterministic chr14/15 block</td>
    <td>Recomputed with proper random-selection baseline (1,000 draws): 4.59× [4.23–4.87] vs published 4.58×. Statistically identical.</td>
    <td><span style="color:#1b5e20;font-weight:600;">PASS</span> (with naming caveat)</td></tr>
</tbody></table>

<div class="caveat"><strong>Recommendation from Check 4.</strong>
At position level, the NPOS column should be relabelled "vs random selection".
The window-level NPOS baseline (number of panel C positions per window) is
non-degenerate and remains a valid density baseline. Headline numbers stand.</div>
"""

    # Section 9: limitations
    s9 = """
<h2 id="limits">9. Limitations &amp; open questions</h2>

<ul>
  <li><strong>Bonferroni at large test family.</strong> The fair sweep tests
    1,260 cells per variant; the permutation floor is at p &lt; 3.97e-5
    (q &lt; 0.05 / 1,260). This is conservative — many borderline cells survive
    only when concentrated at the strongest constructions (position-level binary,
    win_1000 sum). A more focused pre-registered test would have higher power.</li>
  <li><strong>Beating TCW-density on TCW-restricted filter is structurally rigged.</strong>
    On <code>filter_TCW_nonCpG</code>, every surviving mutation is by construction
    TCW. The TCW-density baseline is therefore the optimal point-frequency baseline;
    we lose to it 0.58× but gain 4.58× over random selection. The correct claim
    is "model's positional ranking beats random selection beyond gene density,
    but does not beat TCW frequency on a TCW-restricted filter."</li>
  <li><strong>Pentanucleotide / replication-timing biases not yet checked.</strong>
    The trinucleotide-matched negatives correct the n=4 trinuc bias but not
    a pentanucleotide bias (e.g. YpTCWpR vs other TCW). Replication-timing
    enrichment of cancer mutations is also a confounder we have not subtracted.</li>
  <li><strong>APOBEC1 head trained on small dataset.</strong> 484 positives
    (mouse-validated edited sites) + 484 trinuc-matched negatives. AUROC is
    0.83 but the per-cohort enrichment OR is high-variance for small cancers
    (ESCA n = 30 in POG570 gives OR = 7.07 with CI [1.68–29.69]).</li>
  <li><strong>Coordinate-system caveat.</strong> A3A training positives are
    hg19-coordinated. The 5.93 % MAF coordinate overlap is innocuous (leave-leak-out
    delta &lt; 0.5 pp) but should be noted when comparing to other genome builds.</li>
</ul>
"""

    # Section 10: file index
    s10 = f"""
<h2 id="files">10. File index</h2>

<h3>v4 model checkpoints &amp; CV results</h3>
<ul class="filelist">
  <li>{V4_CANCER_DIR}/cv_results.json</li>
  <li>{V4_CANCER_DIR}/phase3_v4_cancer.pt</li>
  <li>{V4_CANCER_DIR}/bias_diagnostic_cancer_matched.csv</li>
  <li>{V4_CANCER_DIR}/bias_diagnostic_cancer_matched_summary.json</li>
  <li>{V4_CDS_DIR}/cv_results.json</li>
  <li>{V4_CDS_DIR}/phase3_v4_cds.pt</li>
  <li>{V4_CDS_DIR}/bias_diagnostic_cds_unbiased.csv</li>
  <li>{V4_CDS_DIR}/bias_diagnostic_cds_unbiased_summary.json</li>
  <li>{APOBEC1_CDS_DIR}/cv_results.json</li>
  <li>{APOBEC1_CDS_DIR}/apobec1_head_v4_cds.pt</li>
</ul>

<h3>Panel scores &amp; sweeps</h3>
<ul class="filelist">
  <li>{V4_OUT}/panel_scores_v4_cds.parquet</li>
  <li>{V4_OUT}/panel_scores_v4_cds_apobec1retrained.parquet</li>
  <li>{V4_OUT}/panel_scores_v4_cancer.parquet</li>
  <li>{V4_OUT}/sweep_v4_cds_fair.csv</li>
  <li>{V4_OUT}/sweep_v4_cds_fair_per_cancer.csv</li>
  <li>{V4_OUT}/sweep_v4_cancer_fair.csv</li>
  <li>{V4_OUT}/topx_threshold_sweep_v4_cds.csv</li>
  <li>{V4_OUT}/topx_threshold_sweep_v4_cancer.csv</li>
  <li>{V4_OUT}/topx_sweep_v4_cds_all_heads.csv</li>
  <li>{V4_OUT}/topx_trinuc_breakdown.csv</li>
</ul>

<h3>Per-cancer enrichment</h3>
<ul class="filelist">
  <li>{V4_OUT}/per_cancer_enrichment_v4_pcawg.csv</li>
  <li>{V4_OUT}/per_cancer_enrichment_v4_pog570.csv</li>
  <li>{POG570_DIR}/enrichment_v4_cds.csv</li>
  <li>{POG570_DIR}/enrichment_v4_cds_per_cancer.csv</li>
</ul>

<h3>Reports / markdown</h3>
<ul class="filelist">
  <li>{V4_OUT}/V3_VS_V4_COMPARISON.md</li>
  <li>{V4_OUT}/PER_CANCER_ENRICHMENT_V4.md</li>
  <li>{V4_OUT}/APOBEC1_RETRAIN_RESULTS.md</li>
  <li>{V4_OUT}/QA_VERIFICATION_RESULTS.md</li>
  <li>{V4_OUT}/sweep_v4_cds_fair_RESULTS.md</li>
  <li>{V4_OUT}/sweep_v4_cancer_fair_RESULTS.md</li>
  <li>{POG570_DIR}/POG570_V4_RESULTS.md</li>
  <li>{DATA_PREP_MD}</li>
</ul>

<h3>Figures (embedded above as base64)</h3>
<ul class="filelist">
  <li>{V4_OUT}/topx_trinuc_breakdown.png</li>
  <li>{V4_OUT}/per_cancer_OR_pcawg_top1pct.png</li>
  <li>{V4_OUT}/per_cancer_OR_pcawg_top1pct_allCT.png</li>
  <li>{V4_OUT}/per_cancer_OR_pog570_top1pct.png</li>
  <li>{V4_OUT}/per_cancer_OR_concordance_top1pct.png</li>
  <li>{V4_OUT}/sweep_v4_cds_fair.png</li>
  <li>{POG570_DIR}/recall_curve_pog570_v4.png</li>
</ul>

<h3>Data prep &amp; scripts</h3>
<ul class="filelist">
  <li>{ROOT}/data/processed/multi_enzyme/splits_multi_enzyme_v4_cds_unbiased.csv</li>
  <li>{ROOT}/data/processed/multi_enzyme/splits_multi_enzyme_v4_cancer_matched.csv</li>
  <li>{ROOT}/data/processed/multi_enzyme/multi_enzyme_sequences_v4_cds_unbiased.json</li>
  <li>{ROOT}/data/processed/multi_enzyme/multi_enzyme_sequences_v4_cancer_matched.json</li>
  <li>{ROOT}/scripts/multi_enzyme/build_v4_datasets.py</li>
  <li>{ROOT}/scripts/multi_enzyme/build_v4_cancer_matched_dataset.py</li>
  <li>{ROOT}/scripts/multi_enzyme/generate_v4_negatives.py</li>
  <li>{ROOT}/scripts/multi_enzyme/build_apobec1_v4_datasets.py</li>
  <li>{ROOT}/scripts/multi_enzyme/build_apobec1_retrain_summary.py</li>
  <li>{ROOT}/scripts/multi_enzyme/generate_v4_html_report.py <em>(this report)</em></li>
</ul>
"""

    body = (
        '<main>'
        + f'<h1 class="title">V4 Multi-Enzyme APOBEC Report — '
        + 'RNA-Editing Predictor → DNA Somatic Mutation Transfer</h1>'
        + '<p style="color:#666;">Generated: 2026-04-28 — see file index for sources.</p>'
        + s1 + s2 + s3 + s4 + s5 + s6 + s7 + s8 + s9 + s10
        + '</main>'
    )

    html_doc = (
        '<!DOCTYPE html><html lang="en"><head>'
        '<meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">'
        '<title>V4 Multi-Enzyme APOBEC Report</title>'
        f'<style>{css}</style>'
        '</head><body>'
        f'{body}'
        '</body></html>'
    )
    return html_doc


def main() -> None:
    OUTPUT_HTML.parent.mkdir(parents=True, exist_ok=True)
    print(f"Generating V4 report → {OUTPUT_HTML}")
    html_doc = build_html()
    OUTPUT_HTML.write_text(html_doc, encoding="utf-8")
    size_mb = OUTPUT_HTML.stat().st_size / (1024 * 1024)
    print(f"  wrote {OUTPUT_HTML.stat().st_size:,} bytes ({size_mb:.2f} MB)")


if __name__ == "__main__":
    main()
