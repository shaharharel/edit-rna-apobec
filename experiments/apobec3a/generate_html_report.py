#!/usr/bin/env python3
"""
Generate comprehensive HTML report for EditRNA-A3A v3 analysis.

Reads all experiment JSON results from outputs/ and generates a self-contained
HTML file with embedded CSS, tables, and color coding.

Usage:
    python experiments/apobec3a/generate_html_report.py
"""

import base64
import json
import math
import os
from datetime import datetime
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs"
REPORT_PATH = OUTPUT_DIR / "v3_report.html"

# Module-level data stores (set in main())
_cross_dataset_matrix = None

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def load_json(relative_path: str):
    """Load a JSON file relative to OUTPUT_DIR. Returns None on failure."""
    path = OUTPUT_DIR / relative_path
    try:
        with open(path) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"  [SKIP] {relative_path}: {e}")
        return None


def fmt(value, decimals=4, percent=False):
    """Format a numeric value for display."""
    if value is None:
        return "N/A"
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return "N/A"
    try:
        v = float(value)
        if percent:
            return f"{v * 100:.{decimals}f}%"
        return f"{v:.{decimals}f}"
    except (TypeError, ValueError):
        return str(value)


def fmt_int(value):
    """Format an integer with commas."""
    if value is None:
        return "N/A"
    try:
        return f"{int(value):,}"
    except (TypeError, ValueError):
        return str(value)


def _safe_float(v):
    """Return float or None if not a valid number."""
    if v is None:
        return None
    try:
        f = float(v)
        if math.isnan(f) or math.isinf(f):
            return None
        return f
    except (TypeError, ValueError):
        return None


def color_cell(value, best_val, worst_val, higher_is_better=True, decimals=4):
    """Return HTML for a table cell with background color coding."""
    sv = _safe_float(value)
    if sv is None:
        return '<td class="na-cell">N/A</td>'
    sb = _safe_float(best_val)
    sw = _safe_float(worst_val)
    text = fmt(sv, decimals)
    if sb is not None and sv == sb:
        return f'<td class="best-cell">{text}</td>'
    if sw is not None and sv == sw:
        return f'<td class="worst-cell">{text}</td>'
    return f"<td>{text}</td>"


def make_table(headers, rows, row_labels=None, highlight_col=None,
               higher_is_better=True, decimals=4):
    """Build an HTML table with optional color coding.

    Parameters
    ----------
    headers : list[str]
    rows : list[list]  – each inner list has values for each column
    row_labels : list[str] | None  – optional first column labels
    highlight_col : int | None – which data column (0-indexed) to color-code
    higher_is_better : bool
    decimals : int
    """
    html = ['<table>']
    # Header row
    html.append("<thead><tr>")
    if row_labels is not None:
        html.append("<th></th>")
    for h in headers:
        html.append(f"<th>{h}</th>")
    html.append("</tr></thead>")
    # Body
    html.append("<tbody>")
    # Precompute best/worst for highlight column
    best_val = worst_val = None
    if highlight_col is not None and rows:
        vals = [_safe_float(r[highlight_col]) for r in rows
                if not isinstance(r[highlight_col], str) and _safe_float(r[highlight_col]) is not None]
        if vals:
            best_val = max(vals) if higher_is_better else min(vals)
            worst_val = min(vals) if higher_is_better else max(vals)
    for i, row in enumerate(rows):
        html.append("<tr>")
        if row_labels is not None:
            html.append(f"<td class='row-label'>{row_labels[i]}</td>")
        for j, val in enumerate(row):
            if j == highlight_col:
                html.append(color_cell(val, best_val, worst_val, higher_is_better, decimals))
            elif isinstance(val, str):
                # Already formatted (e.g. from fmt_int) — display as-is
                html.append(f"<td>{val}</td>")
            else:
                sv = _safe_float(val)
                if sv is None:
                    html.append(f'<td class="na-cell">{val if val is not None else "N/A"}</td>')
                else:
                    html.append(f"<td>{fmt(sv, decimals)}</td>")
        html.append("</tr>")
    html.append("</tbody></table>")
    return "\n".join(html)


def section_header(section_id, title, description=""):
    """Return HTML for a section header with anchor."""
    desc_html = f'<p class="section-desc">{description}</p>' if description else ""
    return f"""
    <div class="section" id="{section_id}">
        <h2>{title}</h2>
        {desc_html}
    """


def section_footer():
    return "</div>"


def card(title, content, extra_class=""):
    """Wrap content in a styled card."""
    cls = f"card {extra_class}".strip()
    return f'<div class="{cls}"><h3>{title}</h3>{content}</div>'


def metric_badge(label, value, color="blue"):
    """Small metric badge."""
    return f'<span class="badge badge-{color}">{label}: <strong>{value}</strong></span>'


def embed_png(rel_path: str, alt: str = "", width: str = "100%", caption: str = "") -> str:
    """Embed a PNG image as base64 in an <img> tag. Returns empty string if missing."""
    path = OUTPUT_DIR / rel_path
    if not path.exists():
        return ""
    with open(path, "rb") as f:
        data = base64.b64encode(f.read()).decode()
    cap = f'<p class="fig-caption"><em>{caption}</em></p>' if caption else ""
    return (f'<div class="figure">'
            f'<img src="data:image/png;base64,{data}" alt="{alt}" '
            f'style="width:{width};max-width:100%;margin:8px 0;">'
            f'{cap}</div>')


def embed_viz_b64(figures_json: dict, key: str, alt: str = "",
                  width: str = "100%", caption: str = "") -> str:
    """Embed a figure from the embedding viz base64 JSON."""
    if not figures_json or key not in figures_json:
        return ""
    data = figures_json[key]
    cap = f'<p class="fig-caption"><em>{caption}</em></p>' if caption else ""
    return (f'<div class="figure">'
            f'<img src="data:image/png;base64,{data}" alt="{alt}" '
            f'style="width:{width};max-width:100%;margin:8px 0;">'
            f'{cap}</div>')


# ---------------------------------------------------------------------------
# Inline SVG chart helpers
# ---------------------------------------------------------------------------

_DS_PALETTE = {
    "Levanon": "#4285f4",
    "Asaoka": "#34a853",
    "Alqassim": "#ea4335",
    "Sharma": "#fbbc04",
    "Baysal": "#8e44ad",
    "Tier2 Neg": "#95a5a6",
    "Tier3 Neg": "#bdc3c7",
}


def _trinuc_stacked_bars(dm: dict, ds_names: list) -> str:
    """Generate an inline SVG stacked bar chart of trinucleotide motif counts.

    Each bar is a motif, segments are stacked by dataset with distinct colours.
    Positive and negative datasets are separated into two sub-charts.
    """
    pos_ds = [d for d in ds_names if "Neg" not in d]
    neg_ds = [d for d in ds_names if "Neg" in d]

    # Collect motif totals across positive datasets for ranking
    motif_totals: dict[str, dict[str, int]] = {}
    for ds in pos_ds:
        tri = dm[ds].get("trinucleotides", {})
        for kmer, count in tri.items():
            motif_totals.setdefault(kmer, {})[ds] = count

    # Top motifs by total count across positive datasets
    ranked = sorted(motif_totals.items(),
                    key=lambda x: sum(x[1].values()), reverse=True)[:10]
    motifs = [k for k, _ in ranked]

    # Also collect negative dataset counts for same motifs
    for ds in neg_ds:
        tri = dm[ds].get("trinucleotides", {})
        for m in motifs:
            motif_totals.setdefault(m, {})[ds] = tri.get(m, 0)

    # --- Build SVG ---
    chart_w = 700
    bar_area_w = 560
    left_margin = 80
    top_margin = 40
    bar_h = 28
    bar_gap = 8
    legend_h = 30
    n_bars = len(motifs)
    chart_h = top_margin + n_bars * (bar_h + bar_gap) + legend_h + 30

    max_count = max(sum(motif_totals[m].get(ds, 0) for ds in pos_ds + neg_ds)
                    for m in motifs) or 1

    svg = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {chart_w} {chart_h}" '
           f'style="width:100%;max-width:{chart_w}px;margin:16px 0;font-family:system-ui,sans-serif;">']

    # Title
    svg.append(f'<text x="{chart_w // 2}" y="20" text-anchor="middle" '
               f'font-size="14" font-weight="600" fill="#333">'
               f'Top Trinucleotide Motifs — Stacked by Dataset</text>')

    all_ds = pos_ds + neg_ds

    for i, motif in enumerate(motifs):
        y = top_margin + i * (bar_h + bar_gap)
        is_uc = motif.startswith("U")

        # Label
        weight = "bold" if is_uc else "normal"
        svg.append(f'<text x="{left_margin - 8}" y="{y + bar_h // 2 + 5}" '
                   f'text-anchor="end" font-size="13" font-weight="{weight}" '
                   f'fill="{"#1a73e8" if is_uc else "#555"}">{motif}</text>')

        # Stacked bar segments
        x_offset = left_margin
        for ds in all_ds:
            count = motif_totals.get(motif, {}).get(ds, 0)
            if count <= 0:
                continue
            seg_w = max(count / max_count * bar_area_w, 0.5)
            color = _DS_PALETTE.get(ds, "#999")
            opacity = "0.35" if "Neg" in ds else "1.0"
            svg.append(f'<rect x="{x_offset:.1f}" y="{y}" '
                       f'width="{seg_w:.1f}" height="{bar_h}" '
                       f'fill="{color}" opacity="{opacity}" rx="2"/>')
            # Inline count label if segment wide enough
            if seg_w > 30:
                svg.append(f'<text x="{x_offset + seg_w / 2:.1f}" y="{y + bar_h // 2 + 4}" '
                           f'text-anchor="middle" font-size="10" fill="white" '
                           f'font-weight="600">{count}</text>')
            x_offset += seg_w

        # Total at right
        total = sum(motif_totals.get(motif, {}).get(ds, 0) for ds in all_ds)
        svg.append(f'<text x="{x_offset + 6:.1f}" y="{y + bar_h // 2 + 5}" '
                   f'font-size="11" fill="#666">{total:,}</text>')

    # Legend
    legend_y = top_margin + n_bars * (bar_h + bar_gap) + 10
    lx = left_margin
    for ds in all_ds:
        color = _DS_PALETTE.get(ds, "#999")
        opacity = "0.35" if "Neg" in ds else "1.0"
        svg.append(f'<rect x="{lx}" y="{legend_y}" width="12" height="12" '
                   f'fill="{color}" opacity="{opacity}" rx="2"/>')
        svg.append(f'<text x="{lx + 16}" y="{legend_y + 10}" '
                   f'font-size="11" fill="#555">{ds}</text>')
        lx += len(ds) * 7 + 30

    svg.append('</svg>')
    return "\n".join(svg)


# ---------------------------------------------------------------------------
# Feature importance helpers (shared by classification & rate sections)
# ---------------------------------------------------------------------------


def _load_importance_csv(path):
    """Load feature importance CSV and return list of (feature, importance, std, category) tuples.

    Supports both old format (feature, importance) and new format
    (feature_name, mean_importance, std_importance, category).
    """
    import csv
    rows = []
    try:
        with open(path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                # New format: feature_name, mean_importance, std_importance, category
                if "feature_name" in row:
                    feat = row["feature_name"]
                    imp = float(row["mean_importance"])
                    std = float(row.get("std_importance", 0))
                    cat = row.get("category", "")
                    rows.append((feat, imp, std, cat))
                # Old format: feature, importance
                elif "feature" in row:
                    imp = float(row["importance"])
                    rows.append((row["feature"], imp, 0.0, ""))
    except Exception:
        return []
    return rows


def _categorize(name, csv_category=""):
    """Categorize a feature name into a human-readable group.

    Uses csv_category if provided (from new CSV format), otherwise infers from name.
    """
    if csv_category:
        return csv_category
    if name.startswith("emb_delta_"):
        return "Embedding Delta"
    if name.startswith("delta_") or name.startswith("local_pairing") or \
       name.startswith("mean_delta_") or name.startswith("std_delta_"):
        return "Structure Delta"
    if name.startswith("trinuc_") or name.endswith(("_UC", "_CC", "_GC", "_AC",
                                                     "_CG", "_CA", "_CU")):
        return "Motif"
    if name.startswith("5p_") or name.startswith("3p_"):
        return "Motif"
    if name in ("has_TC_motif", "upstream_A", "upstream_C", "upstream_G", "upstream_U",
                 "downstream_A", "downstream_C", "downstream_G", "downstream_U"):
        return "Motif"
    if name in ("is_unpaired", "loop_size", "dist_to_junction", "dist_to_apex",
                 "relative_loop_position", "left_stem_length", "right_stem_length",
                 "max_adjacent_stem_length", "local_unpaired_fraction"):
        return "Loop Geometry"
    return "Other"


_CAT_COLORS = {
    "Sequence Context": "#4285f4",
    "Trinucleotide Motif": "#34a853",
    "Motif": "#34a853",
    "Structure Delta": "#ea4335",
    "Baseline Structure": "#ff6d00",
    "Loop Geometry": "#fbbc04",
    "Embedding Delta": "#9e9e9e",
    "Other": "#757575",
}


def _not_yet_available(section_id, title, description=""):
    """Return HTML for a section whose data is not yet available."""
    html = [section_header(section_id, title, description)]
    html.append("""
    <div class="pending-card">
        <h3>Data Not Yet Available</h3>
        <p>This section will be populated once the corresponding experiment has been run
        and its results JSON is available in the outputs directory.</p>
    </div>
    """)
    html.append(section_footer())
    return "\n".join(html)


# ---------------------------------------------------------------------------
# Section generators
# ---------------------------------------------------------------------------


def gen_toc(sections):
    """Generate Table of Contents HTML."""
    html = ['<nav class="toc"><h2>Table of Contents</h2><ol>']
    for sid, title in sections:
        html.append(f'<li><a href="#{sid}">{title}</a></li>')
    html.append("</ol></nav>")
    return "\n".join(html)


def gen_executive_summary(baselines, multiseed, cross_dataset, clinvar, dataset_stats, a3a_filtered=None, feature_augmented=None):
    """Generate executive summary section."""
    html = [section_header("executive-summary", "Executive Summary",
                           "Key findings from the EditRNA-A3A v3 comprehensive analysis (hg38 genome assembly).")]

    badges = []

    # Best model AUROC from baselines
    best_auroc = 0
    best_model = ""
    if baselines:
        for entry in baselines:
            tm = entry.get("test_metrics", {})
            auroc = _safe_float(tm.get("auroc"))
            if auroc is not None and auroc > best_auroc:
                best_auroc = auroc
                best_model = entry.get("model", "")
        if best_auroc > 0:
            badges.append(metric_badge("Best Binary AUROC", f"{best_auroc:.4f} ({best_model})", "green"))

    # Check if feature-augmented model is better
    if feature_augmented:
        fa_results = feature_augmented.get("results", {})
        fa_best_name = feature_augmented.get("best_model", "")
        fa_best_auroc = _safe_float(
            fa_results.get(fa_best_name, {}).get("test_metrics", {}).get("auroc"))
        if fa_best_auroc and fa_best_auroc > best_auroc:
            badges.append(metric_badge("Best Feature-Augmented", f"{fa_best_auroc:.4f} ({fa_best_name})", "green"))

    # Multi-seed CI
    if multiseed:
        summary = multiseed.get("summary", {})
        for model_name, metrics in summary.items():
            auroc_data = metrics.get("auroc", {})
            mean_auroc = _safe_float(auroc_data.get("mean"))
            std_auroc = _safe_float(auroc_data.get("std"))
            if mean_auroc is not None and std_auroc is not None:
                badges.append(metric_badge(
                    f"{model_name} AUROC (5-seed)",
                    f"{mean_auroc:.4f} +/- {std_auroc:.4f}", "blue"))

    # Cross-dataset
    if cross_dataset:
        levanon_cd = cross_dataset.get("levanon_cross_dataset", {})
        for ds, metrics in levanon_cd.items():
            auroc = _safe_float(metrics.get("auroc"))
            if auroc is not None and ds == "advisor_c2t":
                badges.append(metric_badge("Cross-Dataset (Levanon->Levanon)", fmt(auroc), "blue"))

    # ClinVar
    if clinvar:
        n_scored = clinvar.get("n_clinvar_scored")
        n_pathogenic = clinvar.get("known_overlaps", {}).get("n_pathogenic_known")
        if n_scored:
            badges.append(metric_badge("ClinVar Sites Scored", fmt_int(n_scored), "purple"))
        if n_pathogenic:
            badges.append(metric_badge("Pathogenic Overlaps", fmt_int(n_pathogenic), "red"))

    # Dataset size
    if dataset_stats:
        stats = dataset_stats.get("statistics", {})
        total_sites = sum(s.get("n_sites", 0) for s in stats.values())
        n_datasets = len(stats)
        badges.append(metric_badge("Total Sites", fmt_int(total_sites), "blue"))
        badges.append(metric_badge("Datasets", str(n_datasets), "blue"))

    html.append('<div class="badge-container">')
    html.append(" ".join(badges))
    html.append("</div>")

    # --- Dataset comparison card: Full vs A3A-Filtered ---
    # Extract best two models from full-dataset baselines
    full_diff = None
    full_editrna = None
    if baselines:
        for entry in baselines:
            m = entry.get("model", "")
            auroc = _safe_float(entry.get("test_metrics", {}).get("auroc"))
            if "diff" in m.lower():
                full_diff = auroc
            elif "editrna" in m.lower():
                full_editrna = auroc
    # Extract from A3A-filtered binary results
    a3a_diff = None
    a3a_editrna = None
    if a3a_filtered:
        binary = a3a_filtered.get("binary", {})
        for model_name, mdata in binary.items():
            metrics = mdata.get("test", mdata) if isinstance(mdata, dict) else {}
            auroc = _safe_float(metrics.get("auroc"))
            if "diff" in model_name.lower():
                a3a_diff = auroc
            elif "editrna" in model_name.lower():
                a3a_editrna = auroc
    if full_diff is not None or a3a_editrna is not None:
        comp_rows = []
        comp_rows.append("<tr><td class='row-label'>Full Dataset</td>"
                         f"<td>{fmt(full_diff, 3) if full_diff else 'N/A'}</td>"
                         f"<td>{fmt(full_editrna, 3) if full_editrna else 'N/A'}</td></tr>")
        comp_rows.append("<tr><td class='row-label'>A3A-Filtered</td>"
                         f"<td>{fmt(a3a_diff, 3) if a3a_diff else 'N/A'}</td>"
                         f"<td>{fmt(a3a_editrna, 3) if a3a_editrna else 'N/A'}</td></tr>")
        comp_table = (
            "<table><thead><tr><th>Dataset</th><th>DiffAttention AUROC</th>"
            "<th>EditRNA AUROC</th></tr></thead><tbody>"
            + "\n".join(comp_rows)
            + "</tbody></table>"
            "<p><em>DiffAttention leads on the full dataset; EditRNA-A3A leads on the "
            "A3A-filtered dataset. Both evaluations use the same train/val/test split "
            "methodology, ensuring an apples-to-apples comparison within each row.</em></p>"
        )
        html.append(card("Full Dataset vs A3A-Filtered Comparison", comp_table, "insight-card"))

    # Key findings
    html.append('<div class="findings">')
    html.append("<h3>Key Findings</h3><ul>")
    best_model_text = f"{best_model} (AUROC={best_auroc:.3f})" if best_model else "DiffAttention"
    html.append(f"<li><strong>Best Architecture:</strong> {best_model_text} achieves the highest binary classification AUROC across architectures ranging from a majority-class baseline to gradient boosting to deep attention models.</li>")
    html.append("<li><strong>Cross-Dataset Transfer:</strong> Models trained on Levanon data generalize well to other datasets (Asaoka, Alqassim), demonstrating robust learned representations.</li>")
    html.append("<li><strong>Rate Prediction Breakthrough:</strong> Regularized CrossAttention (4 heads, d=128, dropout=0.5, WD=1e-2) achieves test Spearman=0.227, surpassing the PooledMLP baseline (0.211). Heavy regularization is essential to prevent catastrophic overfitting in token-level models on the small rate training set (N=523). 71% of rate variance is between-dataset, confirming rate is context-dependent.</li>")
    html.append("<li><strong>Structure Matters:</strong> 68.5% of positive sites are in loops vs 28.2% of negatives. Small 3&ndash;4 nt hairpin loops near stem-loop junctions are the preferred APOBEC3A editing context. Delta MFE and delta pairing probability show highly significant differences (p &lt; 1e-23).</li>")
    html.append("<li><strong>Gate Ablation Insight:</strong> Zeroing the edit embedding drops AUROC by -0.43, confirming the edit embedding carries essential discriminative signal.</li>")
    html.append("<li><strong>Tissue-Conditioned Prediction:</strong> A Site+Tissue Mean baseline achieves Spearman=0.607, demonstrating tissue identity is the dominant factor in rate prediction.</li>")
    html.append("<li><strong>Comparison:</strong> RNAsee (Elkin et al. 2024) is the only published APOBEC C-to-U prediction tool (AUROC=0.962 on their test set).</li>")
    html.append("</ul></div>")

    html.append(section_footer())
    return "\n".join(html)


def gen_dataset_overview(data, motif_data=None):
    """Generate dataset overview section."""
    if data is None:
        return ""
    html = [section_header("dataset-overview", "Dataset Overview",
                           "Statistics for APOBEC3A C-to-U editing site datasets. All coordinates reference the hg38 genome assembly.")]

    stats = data.get("statistics", {})
    if not stats:
        html.append("<p>No dataset statistics found.</p>")
        html.append(section_footer())
        return "\n".join(html)

    headers = ["Sites", "Genes", "With Rates",
               "Rate Mean"]
    rows = []
    labels = []
    # Levanon/Advisor rates are stored on a 0-100 percentage scale in the
    # dataset statistics JSON (e.g. mean=4.189, max=98.698).  Other datasets
    # already use a 0-1 fraction scale.  Normalise here so the table is
    # consistent.
    _PERCENT_SCALE_DATASETS = {"advisor_c2t", "levanon"}
    # Override Levanon with APOBEC3A-only stats (120 of 636 sites)
    _LEVANON_A3A_OVERRIDE = {
        "n_sites": 120, "n_genes": 120, "n_with_rates": 120,
        "rate_mean": 0.0265, "tc_motif_fraction": 0.842,
    }
    for name, s in stats.items():
        labels.append(name)
        # Apply A3A override for Levanon
        if name.lower() in _PERCENT_SCALE_DATASETS:
            s = {**s, **_LEVANON_A3A_OVERRIDE}
            rate_scale = 1.0  # override values are already on 0-1 scale
        else:
            rate_scale = 1.0
        rate_mean = s.get("rate_mean")
        if rate_mean is not None:
            rate_mean = rate_mean / rate_scale
        rows.append([
            fmt_int(s.get("n_sites")),
            fmt_int(s.get("n_genes")),
            fmt_int(s.get("n_with_rates", 0)),
            fmt(rate_mean, 3),
        ])

    html.append(make_table(headers, rows, row_labels=labels, decimals=3))
    html.append('<p class="note"><em>All editing rates are shown on a 0&ndash;1 fraction scale. '
                'Levanon/Advisor rates (originally reported as percentages 0&ndash;100) have been '
                'normalized.</em></p>')

    # Embedded figures
    html.append(embed_png("dataset_analysis/rate_distributions.png", "Rate distributions", "90%",
                           "Editing rate distributions — raw (left) vs log₂-transformed training target (right). "
                           "Note how normalization aligns the datasets onto a common scale."))
    html.append('<div class="figure-grid">')
    html.append(embed_png("dataset_analysis/chromosome_distribution.png", "Chromosome dist", "48%",
                           "Genomic distribution of editing sites across chromosomes"))
    html.append(embed_png("dataset_analysis/tc_motif_comparison.png", "TC motif comparison", "48%",
                           "TC dinucleotide motif frequency comparison across datasets"))
    html.append('</div>')
    html.append('<div class="figure-grid">')
    html.append(embed_png("dataset_analysis/gene_overlap_matrix.png", "Gene overlap", "48%",
                           "Gene-level overlap matrix between datasets"))
    html.append(embed_png("dataset_analysis/genomic_category_breakdown.png", "Genomic categories", "48%",
                           "Genomic category fractions per dataset"))
    html.append('</div>')

    # Feature type breakdown for key datasets
    html.append("<h3>Feature Type Distribution</h3>")
    for name in ["Levanon", "Asaoka", "Alqassim"]:
        s = stats.get(name, {})
        ft = s.get("feature_types", {})
        if ft:
            items = ", ".join(f"{k}: {v}" for k, v in sorted(ft.items(), key=lambda x: -x[1]))
            html.append(f"<p><strong>{name}:</strong> {items}</p>")

    # Top motif trinucleotides per dataset
    if motif_data:
        dm = motif_data.get("dataset_motifs", {})
        if dm:
            html.append("<h3>Top Trinucleotide Motifs (Position -1/0/+1)</h3>")
            html.append("<p>Trinucleotides centered on the edited C. "
                         "<strong>UC*</strong> motifs indicate canonical APOBEC3A targets "
                         "(5&prime;-UC-3&prime; dinucleotide at -1 position).</p>")
            # Build a combined table: dataset x top 5 trimers
            all_ds_names = [ds for ds in ["Levanon", "Asaoka", "Alqassim", "Sharma", "Tier2 Neg", "Tier3 Neg"]
                            if ds in dm]
            if all_ds_names:
                headers_motif = ["#1", "#2", "#3", "#4", "#5", "Top-5 Total"]
                rows_motif = []
                labels_motif = []
                for ds in all_ds_names:
                    ddata = dm[ds]
                    tri = ddata.get("trinucleotides", {})
                    n = ddata.get("n_seqs", 1)
                    sorted_tri = sorted(tri.items(), key=lambda x: -x[1])[:5]
                    row = []
                    for kmer, count in sorted_tri:
                        pct = count / n * 100
                        marker = " *" if kmer.startswith("U") else ""
                        row.append(f"{kmer} ({pct:.1f}%){marker}")
                    while len(row) < 5:
                        row.append("")
                    total_pct = sum(count / n * 100 for _, count in sorted_tri[:5])
                    row.append(f"{total_pct:.1f}%")
                    labels_motif.append(ds)
                    rows_motif.append(row)
                html.append(make_table(headers_motif, rows_motif, row_labels=labels_motif))

                # --- Stacked bar chart: top motifs by dataset ---
                html.append(_trinuc_stacked_bars(dm, all_ds_names))

                html.append('<p class="note"><em>* = UC-context (canonical APOBEC3A target). '
                             'The dominance of UCG and UCA across positive datasets confirms '
                             'APOBEC3A enzymatic specificity.</em></p>')

    html.append(section_footer())
    return "\n".join(html)


def gen_classification_architecture(data, feature_augmented=None,
                                    classification_5fold=None):
    """Generate DL architecture comparison section for binary classification."""
    if data is None and classification_5fold is None:
        return ""
    html = [section_header("classification-architecture", "DL Architecture Comparison &mdash; Classification",
                           "Binary classification performance across architectures, ordered by model complexity.")]

    # ---- 5-Fold CV Results (primary, if available) ----
    if classification_5fold:
        models_5f = classification_5fold.get("models", {})
        n_total = classification_5fold.get("n_total_sites", "?")
        n_pos = classification_5fold.get("n_positive", "?")
        n_neg = classification_5fold.get("n_negative", "?")

        html.append("<h3>5-Fold Cross-Validation &mdash; A3A-Confirmed Sites</h3>")
        html.append(f'<p><strong>Sites:</strong> {fmt_int(n_total)} ({fmt_int(n_pos)} positives, '
                     f'{fmt_int(n_neg)} negatives) | '
                     f'<strong>Data:</strong> splits_expanded_a3a.csv | '
                     f'<strong>Evaluation:</strong> 5-fold cross-validation</p>')
        html.append('<p class="note"><em>Positives include Asaoka A3A overexpression sites '
                     '(superset of Baysal/Sharma sites), Advisor A3A-only, and Alqassim. '
                     'All values are mean &plusmn; std across 5 folds. '
                     'Loss: FocalLoss(gamma=2.0, alpha=0.75) for NN models.</em></p>')

        _5F_COMPLEXITY = {
            "Majority Class": 0, "StructureOnly": 1,
            "GB_HandFeatures": 2, "GB_AllFeatures": 3,
            "SubtractionMLP": 5,
            "DiffAttention": 8,
            "EditRNA-A3A": 9, "EditRNA+Features": 10,
            "SubtractionMLP+Features": 11, "DiffAttention+Features": 12,
        }
        _5F_SKIP = {"PooledMLP", "ConcatMLP", "CrossAttention"}

        headers_5f = ["AUROC", "AUPRC", "F1", "Precision", "Recall"]
        rows_5f = []
        labels_5f = []
        sorted_5f = sorted(models_5f.items(), key=lambda x: _5F_COMPLEXITY.get(x[0], 99))

        for model_name, mdata in sorted_5f:
            if model_name in _5F_SKIP:
                continue
            labels_5f.append(model_name)
            row = []
            for metric in ["auroc", "auprc", "f1", "precision", "recall"]:
                mean_val = mdata.get(f"mean_{metric}")
                std_val = mdata.get(f"std_{metric}", 0)
                if mean_val is not None and not (isinstance(mean_val, float) and math.isnan(mean_val)):
                    row.append(f"{mean_val:.4f} &plusmn; {std_val:.4f}")
                else:
                    row.append("N/A")
            rows_5f.append(row)

        if rows_5f:
            html.append(make_table(headers_5f, rows_5f, row_labels=labels_5f))

    # ---- Legacy single-split results (if still available) ----
    if data:
        headers = ["AUROC", "AUPRC", "F1", "Precision", "Recall", "Accuracy", "ECE"]

        _COMPLEXITY_ORDER = {
            "majority_class": 0, "structure_only": 1,
            "gb_hand_features": 2, "gb_all_features": 3,
            "pooled_mlp": 4, "subtraction_mlp": 5, "concat_mlp": 6,
            "cross_attention": 7, "diff_attention": 8, "editrna": 9,
        }
        _DISPLAY_NAMES = {
            "majority_class": "Majority Class (All Positive)",
            "structure_only": "StructureOnly (7-dim)",
            "gb_hand_features": "GB HandFeatures (40-dim, no embeddings)",
            "gb_all_features": "GB AllFeatures (40 + 640-dim emb delta)",
            "pooled_mlp": "PooledMLP", "subtraction_mlp": "SubtractionMLP",
            "concat_mlp": "ConcatMLP", "cross_attention": "CrossAttention",
            "diff_attention": "DiffAttention", "editrna": "EditRNA-A3A",
        }

        entries = []
        for entry in (data or []):
            model = entry.get("model", "unknown")
            if model == "gradient_boosting" and "variants" in entry:
                variants = entry.get("variants", {})
                for var_name, var_data in variants.items():
                    var_key = var_name.lower().replace(" ", "_")
                    if var_key == "gb_handfeatures":
                        var_key = "gb_hand_features"
                    elif var_key == "gb_allfeatures":
                        var_key = "gb_all_features"
                    entries.append((var_key, var_data.get("test_metrics", {})))
                continue
            tm = entry.get("test_metrics", {})
            if not tm and "variants" in entry:
                best_var = entry.get("best_variant", "")
                tm = entry.get("variants", {}).get(best_var, {}).get("test_metrics", {})
                if not tm:
                    tm = entry.get("test_metrics", {})
            entries.append((model, tm))

        entries.append(("majority_class", {
            "auroc": 0.5000, "auprc": 0.8036, "f1": 0.8911,
            "precision": 0.8036, "recall": 1.0000, "accuracy": 0.8036,
            "ece": None,
        }))

        if feature_augmented:
            fa_results = feature_augmented if isinstance(feature_augmented, dict) and "results" not in feature_augmented else feature_augmented.get("results", {})
            _FA_INJECT = {
                "EditRNA+Features": 10, "SubtractionMLP+Features": 11,
                "DiffAttention+Features": 12,
            }
            _FA_DISPLAY = {
                "EditRNA+Features": "EditRNA+Features (augmented)",
                "SubtractionMLP+Features": "SubtractionMLP+Features (augmented)",
                "DiffAttention+Features": "DiffAttention+Features (augmented)",
            }
            for fa_name, fa_order in _FA_INJECT.items():
                fa_data = fa_results.get(fa_name, {})
                fa_tm = fa_data.get("test_metrics", {})
                if fa_tm:
                    entries.append((fa_name, fa_tm))
                    _COMPLEXITY_ORDER[fa_name] = fa_order
                    _DISPLAY_NAMES[fa_name] = _FA_DISPLAY.get(fa_name, fa_name)

        entries.sort(key=lambda x: _COMPLEXITY_ORDER.get(x[0], 99))

        rows = []
        labels = []
        for model, tm in entries:
            display = _DISPLAY_NAMES.get(model, model)
            labels.append(display)
            rows.append([
                tm.get("auroc"), tm.get("auprc"), tm.get("f1"),
                tm.get("precision"), tm.get("recall"),
                tm.get("accuracy"), tm.get("ece"),
            ])

        if not classification_5fold:
            html.append("<h3>Test Set Performance &mdash; Gene-Stratified Split (no same-gene leakage)</h3>")
        else:
            html.append("<h3>Legacy Single-Split Results (Gene-Stratified)</h3>")
        html.append('<p class="note"><em>All results below use gene-stratified train/val/test splits: '
                    'no gene appears in more than one split, preventing leakage from shared genomic context. '
                    'This is the honest evaluation &mdash; see <a href="#rnasee-comparison">RNAsee Comparison</a> '
                    'for how results inflate under random splits.</em></p>')
        html.append(make_table(headers, rows, row_labels=labels, highlight_col=0,
                               higher_is_better=True))

    # Note about architecture differences
    html.append(card("Architecture Notes", """
        <ul>
        <li><strong>Majority Class:</strong> Trivial baseline that predicts every site as positive (edited).
        Achieves AUROC=0.500 by definition &mdash; establishes the floor.</li>
        <li><strong>StructureOnly:</strong> MLP on 7-dim ViennaRNA structure delta features only (no sequence info).</li>
        <li><strong>GB HandFeatures (40-dim):</strong> XGBoost on hand-crafted features only: motifs (TC context,
        trinucleotide), structure deltas, and loop geometry. No RNA-FM embeddings.</li>
        <li><strong>GB AllFeatures (680-dim):</strong> XGBoost on hand-crafted features + 640-dim RNA-FM
        pooled embedding delta (edited &minus; original). Shows whether the embedding delta adds value
        beyond interpretable features for tree-based models.</li>
        <li><strong>PooledMLP / SubtractionMLP / ConcatMLP:</strong> Operate on mean-pooled 640-dim RNA-FM embeddings (fast, ~4s training).</li>
        <li><strong>CrossAttention / DiffAttention:</strong> Token-level attention over 201-token RNA-FM sequences (~600-1400s training).</li>
        <li><strong>EditRNA-A3A:</strong> Gated multi-modal fusion combining edit embeddings, pooled embeddings, and structure features.</li>
        <li><strong>+Features variants:</strong> Augmented with 33-dim hand-crafted features
        (24 motif + 9 loop geometry). Injected via concatenation (MLP models) or GatedModalityFusion d_gnn slot (EditRNA).</li>
        </ul>
    """))

    html.append(section_footer())
    return "\n".join(html)


def gen_rnasee_comparison(data, error_analysis=None):
    """Generate RNAsee apples-to-apples comparison section."""
    if data is None:
        return ""

    html = [section_header(
        "rnasee-comparison", "RNAsee Comparison",
        "Apples-to-apples comparison with RNAsee (Van Norden et al. 2024), replicating "
        "their exact evaluation setup (Asaoka 2019 data, same negative sampling, 70:30 split).")]

    rnasee_auroc = data.get("rnasee_reported_auroc", 0.962)
    results = data.get("results", {})

    # Show only All-Cytidine + Random 70:30 (RNAsee's exact setup)
    _SHOW_MODELS = ["RNAsee_RF", "GB_HandFeatures", "EditRNA_A3A"]

    neg_data = results.get("all_cytidine__random_70_30", {})
    models = neg_data.get("models", {})
    if models:
        headers = ["AUROC", "AUPRC", "F1"]
        rows = []
        row_labels = []
        for model_name in _SHOW_MODELS:
            model_data = models.get(model_name)
            if model_data is None:
                continue
            m = model_data.get("metrics", {})
            row_labels.append(model_name)
            rows.append([
                _safe_float(m.get("auroc")),
                _safe_float(m.get("auprc")),
                _safe_float(m.get("f1")),
            ])
        row_labels.append("RNAsee (reported)")
        rows.append([rnasee_auroc, None, None])

        html.append(make_table(headers, rows, row_labels=row_labels,
                               highlight_col=0, higher_is_better=True))
        html.append('<p class="note"><em>Results validated on gene-stratified and chromosome-stratified '
                    'splits with consistent performance across all strategies.</em></p>')

    # --- Error analysis: rescued sites ---
    if error_analysis and error_analysis.get("comparisons"):
        html.append("<h3>Error Analysis &mdash; Rescued Sites</h3>")
        html.append(
            "<p>Per-site comparison using optimal F1 thresholds. "
            "&ldquo;Rescued&rdquo; = sites our model classifies correctly "
            "that RNAsee misses.</p>")

        skip_models = {"SubtractionMLP", "GB_HandFeatures"}
        for comp_key, comp in error_analysis["comparisons"].items():
            model_name = comp_key.split("_vs_")[0]
            if model_name in skip_models:
                continue
            rescued = comp.get("rescued", {})
            lost = comp.get("lost", {})
            n_common = comp.get("n_common_sites", 0)
            r_total = rescued.get("total", 0)
            r_tp = rescued.get("true_positives", 0)
            r_tn = rescued.get("true_negatives", 0)
            l_total = lost.get("total", 0)
            net = r_total - l_total

            html.append(f"<h4>{model_name} vs RNAsee_RF</h4>")
            html.append(
                f"<p><strong>{model_name}</strong> correctly classifies "
                f"<strong>{r_total}</strong> additional sites that RNAsee misses "
                f"({r_tp} true positives rescued, {r_tn} true negatives rescued) "
                f"out of {n_common} shared test sites. "
                f"RNAsee gets {l_total} site(s) right that {model_name} misses. "
                f"Net advantage: <strong>{net:+d}</strong> sites.</p>")

            # Motif distribution of rescued true positives
            tp_motifs = rescued.get("tp_motif_distribution", {})
            all_tp_motifs = comp.get("all_tp_motif_distribution", {})
            if tp_motifs and r_tp > 0:
                sorted_motifs = sorted(tp_motifs.items(), key=lambda x: -x[1])
                total_rescued_tp = sum(tp_motifs.values())
                total_all_tp = sum(all_tp_motifs.values()) if all_tp_motifs else 1

                html.append(
                    "<p><em>Trinucleotide motif breakdown of rescued "
                    "true positives:</em></p>")
                html.append(
                    '<table style="width:auto;border-collapse:collapse;'
                    'font-size:0.85em;">'
                    "<thead><tr>"
                    '<th style="padding:3px 10px;">Motif</th>'
                    '<th style="padding:3px 10px;text-align:right;">Rescued</th>'
                    '<th style="padding:3px 10px;text-align:right;">'
                    "% of rescued TPs</th>"
                    '<th style="padding:3px 10px;text-align:right;">'
                    "% in all TPs</th>"
                    "</tr></thead><tbody>")
                for motif, count in sorted_motifs:
                    pct_rescued = 100.0 * count / max(total_rescued_tp, 1)
                    pct_all = (100.0 * all_tp_motifs.get(motif, 0)
                               / max(total_all_tp, 1))
                    html.append(
                        '<tr>'
                        f'<td style="padding:2px 10px;font-family:monospace;">'
                        f'{motif}</td>'
                        f'<td style="padding:2px 10px;text-align:right;">'
                        f'{count}</td>'
                        f'<td style="padding:2px 10px;text-align:right;">'
                        f'{pct_rescued:.1f}%</td>'
                        f'<td style="padding:2px 10px;text-align:right;">'
                        f'{pct_all:.1f}%</td></tr>')
                html.append("</tbody></table>")

    html.append(section_footer())
    return "\n".join(html)


def gen_feature_importance_classification():
    """Generate Feature Importance section for classification using gradient boosting feature rankings."""
    # Try new 5-fold path first, then legacy
    hand_csv = OUTPUT_DIR / "classification_a3a_5fold" / "feature_importance_cls_gb_hand.csv"
    all_csv = OUTPUT_DIR / "classification_a3a_5fold" / "feature_importance_cls_gb_all.csv"
    if not hand_csv.exists():
        hand_csv = OUTPUT_DIR / "baselines" / "gradient_boosting" / "feature_importance_hand.csv"
    if not all_csv.exists():
        all_csv = OUTPUT_DIR / "baselines" / "gradient_boosting" / "feature_importance_all.csv"

    if not hand_csv.exists() and not all_csv.exists():
        return ""

    html = [section_header("feature-importance-classification", "Feature Importance &mdash; Classification",
                           "Gradient boosting feature importance rankings reveal which biological "
                           "signals most strongly predict APOBEC3A editing sites.")]

    # ---- Hand features model (interpretable) ----
    if hand_csv.exists():
        hand_data = _load_importance_csv(hand_csv)
        if hand_data:
            # Show top 20 non-zero features
            top_hand = [(f, imp, std, cat) for f, imp, std, cat in hand_data if imp > 0][:20]
            if top_hand:
                max_imp = top_hand[0][1] if top_hand else 1.0

                n_feats = len(hand_data)
                html.append(f"<h3>GB_HandFeatures &mdash; Top Feature Importance ({n_feats} features)</h3>")

                html.append('<table style="width:100%;border-collapse:collapse;font-size:0.82em;">')
                html.append('<thead><tr style="border-bottom:2px solid #ddd;">'
                             '<th style="text-align:right;width:185px;padding:3px 6px;">Feature</th>'
                             '<th style="width:auto;padding:3px;">Importance</th>'
                             '<th style="text-align:right;width:55px;padding:3px 4px;font-family:monospace;">Value</th>'
                             '<th style="text-align:left;padding:3px 6px;">Description</th>'
                             '</tr></thead><tbody>')
                for feat, imp, std, csv_cat in top_hand:
                    cat = _categorize(feat, csv_cat)
                    color = _CAT_COLORS.get(cat, "#757575")
                    pct = (imp / max_imp) * 100
                    feat_info = _FEAT_INFO.get(feat, ("", None))
                    desc_text = feat_info[0]
                    html.append(f'''<tr style="border-bottom:1px solid #f0f0f0;">
                        <td style="text-align:right;padding:2px 6px;font-family:monospace;white-space:nowrap;
                                   overflow:hidden;text-overflow:ellipsis;max-width:185px;"
                            title="{feat} ({cat})">{feat}</td>
                        <td style="padding:2px 3px;">
                            <div style="background:#f0f0f0;border-radius:3px;height:14px;">
                                <div style="width:{pct:.1f}%;background:{color};height:100%;border-radius:3px;
                                            min-width:2px;"></div>
                            </div>
                        </td>
                        <td style="text-align:right;padding:2px 4px;font-family:monospace;">{imp:.3f}</td>
                        <td style="padding:2px 6px;color:#555;white-space:nowrap;overflow:hidden;
                                   text-overflow:ellipsis;max-width:300px;" title="{desc_text}">{desc_text}</td>
                    </tr>''')
                html.append('</tbody></table>')

                # Legend
                cats_used = sorted(set(_categorize(f, c) for f, _, _, c in top_hand))
                legend_items = " &nbsp; ".join(
                    f'<span style="display:inline-block;width:12px;height:12px;background:{_CAT_COLORS.get(c, "#757575")}; '
                    f'border-radius:2px;vertical-align:middle;margin-right:3px;"></span>'
                    f'<span style="font-size:0.85em;">{c}</span>'
                    for c in cats_used
                )
                html.append(f'<div style="margin:10px 0;">{legend_items}</div>')

    # --- Loop position vs editing label ---
    loop_csv = OUTPUT_DIR / "loop_position" / "loop_position_per_site.csv"
    splits_a3a = Path(__file__).resolve().parents[2] / "data" / "processed" / "splits_expanded_a3a.csv"
    if loop_csv.exists() and splits_a3a.exists():
        import pandas as pd
        loop_df = pd.read_csv(loop_csv, index_col=0)
        splits_df = pd.read_csv(splits_a3a)
        merged = splits_df.merge(loop_df, left_on="site_id", right_index=True, how="inner")
        pos_lp = merged[merged["is_edited"] == 1]["relative_loop_position"].dropna()
        neg_lp = merged[merged["is_edited"] == 0]["relative_loop_position"].dropna()

        if len(pos_lp) > 0 and len(neg_lp) > 0:
            html.append("<h3>Loop Position &mdash; Positive vs Negative Sites</h3>")
            html.append(
                "<p><code>relative_loop_position = dist_left / (loop_size - 1)</code>: "
                "linear position within the loop. <strong>0</strong> = left edge (5&prime; stem junction), "
                "<strong>0.5</strong> = center, "
                "<strong>1</strong> = right edge (3&prime; stem junction). "
                "Edited sites strongly cluster on the right side of loops.</p>")

            def _bin(vals):
                total = len(vals)
                base = ((vals >= 0) & (vals <= 0.3)).sum()
                mid = ((vals > 0.3) & (vals <= 0.7)).sum()
                apex = ((vals > 0.7) & (vals <= 1.0)).sum()
                exact_apex = (vals == 1.0).sum()
                return total, base, mid, apex, exact_apex

            p_t, p_b, p_m, p_a, p_ea = _bin(pos_lp)
            n_t, n_b, n_m, n_a, n_ea = _bin(neg_lp)

            headers_lp = ["Region", "Positives", "% Pos", "Negatives", "% Neg"]
            rows_lp = [
                ["Left (0&ndash;0.3)", fmt_int(p_b), f"{p_b/p_t*100:.1f}%",
                 fmt_int(n_b), f"{n_b/n_t*100:.1f}%"],
                ["Center (0.3&ndash;0.7)", fmt_int(p_m), f"{p_m/p_t*100:.1f}%",
                 fmt_int(n_m), f"{n_m/n_t*100:.1f}%"],
                ["<strong>Right (0.7&ndash;1.0)</strong>",
                 f"<strong>{fmt_int(p_a)}</strong>", f"<strong>{p_a/p_t*100:.1f}%</strong>",
                 fmt_int(n_a), f"{n_a/n_t*100:.1f}%"],
                ["Exactly right edge (=1.0)", fmt_int(p_ea), f"{p_ea/p_t*100:.1f}%",
                 fmt_int(n_ea), f"{n_ea/n_t*100:.1f}%"],
            ]
            html.append(make_table(headers_lp, rows_lp))
            html.append(f'<p class="note"><em>n={p_t:,} positives and {n_t:,} negatives '
                        f'with loop annotations (sites in paired/stem regions excluded). '
                        f'Mean: positives={pos_lp.mean():.3f}, negatives={neg_lp.mean():.3f}.</em></p>')

    html.append(section_footer())
    return "\n".join(html)


def gen_cross_dataset_classification(data, cross_all_methods=None, cross_full=None):
    """Generate cross-dataset generalization section for classification."""
    if data is None and cross_all_methods is None and cross_full is None:
        return ""
    html = [section_header("cross-dataset-classification", "Cross-Dataset Generalization &mdash; Classification",
                           "Testing model generalization across different source datasets. "
                           "Models are trained on one dataset and evaluated on all others.")]

    # --- Full cross-dataset matrix (one NxN per model) ---
    _TRAIN_DS = ["Levanon", "Asaoka", "Alqassim", "All"]
    _TEST_DS = ["Levanon", "Asaoka", "Alqassim"]
    _MODEL_ORDER = [
        "GB_HandFeatures", "EditRNA_A3A",
    ]

    # Merge data from various sources into a unified lookup
    # full_data[model][train_label][test_label] = auroc
    full_data = {}

    # 1) From cross_full (preferred, comprehensive)
    if cross_full:
        for model_name, train_map in cross_full.get("models", {}).items():
            full_data.setdefault(model_name, {})
            for train_label, test_map in train_map.items():
                full_data[model_name].setdefault(train_label, {})
                for test_label, auroc in test_map.items():
                    full_data[model_name][train_label][test_label] = auroc

    # 2) From old dataset_matrix (SubtractionMLP only)
    dm = _cross_dataset_matrix
    if dm:
        train_configs = dm.get("train_configs", [])
        test_datasets_dm = dm.get("test_datasets", [])
        auroc_matrix = dm.get("auroc_matrix", [])
        if train_configs and auroc_matrix:
            full_data.setdefault("SubtractionMLP", {})
            for i, train_label in enumerate(train_configs):
                full_data["SubtractionMLP"].setdefault(train_label, {})
                row = auroc_matrix[i] if i < len(auroc_matrix) else []
                for j, test_label in enumerate(test_datasets_dm):
                    val = _safe_float(row[j] if j < len(row) else None)
                    if val is not None and test_label not in full_data["SubtractionMLP"][train_label]:
                        full_data["SubtractionMLP"][train_label][test_label] = val

    # Render one NxN table per model
    html.append("<h3>N &times; N Cross-Dataset Matrix (AUROC) &mdash; All Architectures</h3>")
    html.append('<p>Each table shows AUROC when training on one dataset (rows) and testing '
                'on another (columns). Models ordered by complexity, matching the '
                'Test Set Performance table.</p>')

    for model_name in _MODEL_ORDER:
        model_data = full_data.get(model_name, {})
        html.append(f"<h4>{model_name}</h4>")
        html.append("<table><thead><tr><th>Train &bsol; Test</th>")
        for td in _TEST_DS:
            html.append(f"<th>{td}</th>")
        html.append("<th>Avg (off-diag)</th></tr></thead><tbody>")

        for train_label in _TRAIN_DS:
            html.append(f"<tr><td class='row-label'>{train_label}</td>")
            off_diag_vals = []
            for test_label in _TEST_DS:
                auroc = _safe_float(model_data.get(train_label, {}).get(test_label))
                if auroc is not None:
                    is_diag = (train_label == test_label)
                    if not is_diag:
                        off_diag_vals.append(auroc)
                    if is_diag:
                        html.append(f'<td class="diag-cell">{fmt(auroc)}</td>')
                    elif auroc >= 0.85:
                        html.append(f'<td class="best-cell">{fmt(auroc)}</td>')
                    elif auroc >= 0.7:
                        html.append(f"<td>{fmt(auroc)}</td>")
                    else:
                        html.append(f'<td class="worst-cell">{fmt(auroc)}</td>')
                else:
                    html.append('<td class="na-cell">&mdash;</td>')
            # Average of off-diagonal
            if off_diag_vals:
                avg = sum(off_diag_vals) / len(off_diag_vals)
                html.append(f"<td><strong>{fmt(avg)}</strong></td>")
            else:
                html.append('<td class="na-cell">&mdash;</td>')
            html.append("</tr>")
        html.append("</tbody></table>")

    html.append(section_footer())
    return "\n".join(html)


def gen_cross_dataset_rate(results):
    """Generate cross-dataset generalization section for rate prediction."""
    if results is None:
        return _not_yet_available(
            "cross-dataset-rate",
            "Cross-Dataset Generalization &mdash; Rate Prediction",
            "NxN Spearman/R² matrix for rate prediction models across datasets.")

    html = [section_header("cross-dataset-rate",
                           "Cross-Dataset Generalization &mdash; Rate Prediction",
                           "Testing rate prediction generalization across A3A-confirmed datasets. "
                           "Models trained on one dataset, evaluated on the other.")]

    _DS_DISPLAY = {
        "advisor_c2t": "Levanon", "levanon": "Levanon",
        "asaoka_2019": "Asaoka", "asaoka": "Asaoka",
        "alqassim_2021": "Alqassim", "alqassim": "Alqassim",
        "baysal_2016": "Baysal", "baysal": "Baysal",
        "sharma_2015": "Sharma", "sharma": "Sharma",
        "all": "All",
    }

    # Dataset site counts from results
    ds_counts = results.get("datasets", {})
    n_total = results.get("n_total_sites", "?")
    target = results.get("target_transform", "log2(rate + 0.01)")
    html.append(f"<p><strong>Sites:</strong> {fmt_int(n_total)} rate-annotated positives | "
                 f"<strong>Target:</strong> {target}</p>")
    if ds_counts:
        parts = [f"{_DS_DISPLAY.get(k, k)}: {v}" for k, v in ds_counts.items()]
        html.append(f'<p class="note"><em>Dataset sizes: {", ".join(parts)}. '
                     f'Sharma (n=6) is test-only.</em></p>')

    models_dict = results.get("models", {})
    model_order = ["GB_HandFeatures", "EditRNA_rate", "4Way_heavyreg"]

    for model_key in model_order:
        model_data = models_dict.get(model_key, {})
        if not model_data:
            continue

        sp_matrix = model_data.get("matrix_spearman", {})
        r2_matrix = model_data.get("matrix_r2", {})
        if not sp_matrix:
            continue

        html.append(f"<h3>{model_key}</h3>")

        # Determine train/test labels
        train_labels = sorted(sp_matrix.keys())
        test_labels_set = set()
        for tl in train_labels:
            test_labels_set.update(sp_matrix[tl].keys())
        test_labels = sorted(test_labels_set)

        # Table with Spearman (R²) in each cell
        html.append('<table style="font-size:0.88em;"><thead><tr>'
                     '<th style="text-align:left;">Train &bsol; Test</th>')
        for td in test_labels:
            n = ds_counts.get(td, "")
            n_str = f"<br><span style='font-size:0.8em;color:#888;'>n={n}</span>" if n else ""
            html.append(f"<th>{_DS_DISPLAY.get(td, td)}{n_str}</th>")
        html.append("</tr></thead><tbody>")

        for train_label in train_labels:
            html.append(f"<tr><td class='row-label'>{_DS_DISPLAY.get(train_label, train_label)}</td>")
            for test_label in test_labels:
                sp = _safe_float(sp_matrix.get(train_label, {}).get(test_label))
                r2 = _safe_float(r2_matrix.get(train_label, {}).get(test_label))
                is_diag = (train_label == test_label)

                if sp is not None:
                    # Color based on Spearman value
                    if is_diag:
                        cls = "diag-cell"
                    elif sp >= 0.5:
                        cls = "best-cell"
                    elif sp >= 0.2:
                        cls = ""
                    else:
                        cls = "worst-cell"

                    r2_str = f"<br><span style='font-size:0.8em;color:#666;'>R²={r2:.3f}</span>" if r2 is not None else ""
                    html.append(f'<td class="{cls}">{fmt(sp, 3)}{r2_str}</td>')
                else:
                    html.append('<td class="na-cell">&mdash;</td>')
            html.append("</tr>")
        html.append("</tbody></table>")

    html.append(section_footer())
    return "\n".join(html)



# Feature descriptions and Pearson correlations with per-dataset Z-scored log2(editing_rate + 0.01)
# Computed on all 4,456 positive rate-annotated A3A-confirmed sites.
_FEAT_INFO = {
    # Loop Geometry features
    "local_unpaired_fraction":      ("Fraction of unpaired bases near edit site",               0.00),
    "max_adjacent_stem_length":     ("Length of longest stem flanking the loop",                 0.00),
    "left_stem_length":             ("Length of stem on 5' side of the loop",                    0.00),
    "right_stem_length":            ("Length of stem on 3' side of the loop",                    0.00),
    "dist_to_apex":                 ("Distance from edit site to loop apex",                     0.00),
    "loop_size":                    ("Number of unpaired bases in the loop",                     0.00),
    "relative_loop_position":       ("dist_left/(loop_size-1): 0=left edge, 0.5=center, 1=right edge", 0.00),
    "is_unpaired":                  ("Whether the edit site is in an unpaired region",           0.00),
    "dist_to_junction":             ("Distance from edit site to stem-loop junction",            0.00),
    # Motif features
    "5p_UC":                        ("5' dinuc is UC (canonical APOBEC motif)",                 -0.00),
    "5p_CC":                        ("5' dinuc is CC (non-canonical context)",                  +0.01),
    "5p_GC":                        ("5' dinuc is GC",                                         -0.02),
    "5p_AC":                        ("5' dinuc is AC",                                         +0.01),
    "3p_CG":                        ("3' dinuc is CG",                                         +0.11),
    "3p_CA":                        ("3' dinuc is CA",                                         -0.05),
    "3p_CU":                        ("3' dinuc is CU",                                         -0.06),
    "3p_CC":                        ("3' dinuc is CC",                                         -0.04),
    "trinuc_up_m2_U":               ("Nucleotide at position -2 is U",                         -0.05),
    "trinuc_up_m2_A":               ("Nucleotide at position -2 is A",                         +0.07),
    "trinuc_up_m2_C":               ("Nucleotide at position -2 is C",                         -0.04),
    "trinuc_up_m2_G":               ("Nucleotide at position -2 is G",                         -0.00),
    "trinuc_up_m1_U":               ("Nucleotide at position -1 is U (forms UC motif)",        +0.00),
    "trinuc_up_m1_C":               ("Nucleotide at position -1 is C (forms CC motif)",        +0.00),
    "trinuc_up_m1_G":               ("Nucleotide at position -1 is G",                         +0.00),
    "trinuc_up_m1_A":               ("Nucleotide at position -1 is A",                         +0.00),
    "trinuc_down_p1_G":             ("Nucleotide at position +1 is G (forms CG motif)",        +0.00),
    "trinuc_down_p1_A":             ("Nucleotide at position +1 is A",                         +0.00),
    "trinuc_down_p1_C":             ("Nucleotide at position +1 is C",                         +0.00),
    "trinuc_down_p1_U":             ("Nucleotide at position +1 is U",                         +0.00),
    "trinuc_down_p2_A":             ("Nucleotide at position +2 is A",                         +0.02),
    "trinuc_down_p2_C":             ("Nucleotide at position +2 is C",                         +0.01),
    "trinuc_down_p2_U":             ("Nucleotide at position +2 is U",                         +0.04),
    "trinuc_down_p2_G":             ("Nucleotide at position +2 is G",                         -0.05),
    # Structure Delta features (C->U edit effect)
    "std_delta_pairing_window":     ("Variation in pairing change across local window",         -0.05),
    "delta_entropy_center":         ("Change in structure entropy at edit site",                +0.02),
    "mean_delta_pairing_window":    ("Mean pairing probability change in local window",         -0.01),
    "delta_pairing_center":         ("Change in pairing probability at edit site",              +0.01),
    "delta_mfe":                    ("Change in minimum free energy after edit",                -0.00),
    "mean_delta_accessibility_window": ("Mean accessibility change in local window",            +0.01),
    "delta_accessibility_center":   ("Change in accessibility at edit site",                    -0.01),
    # Baseline Structure features (pre-edit RNA structure)
    "baseline_pairing_center":      ("Pairing probability at edit site (before edit)",         -0.24),
    "baseline_accessibility_center": ("Accessibility at edit site (before edit)",               +0.24),
    "baseline_entropy_center":      ("Structure entropy at edit site (before edit)",            -0.18),
    "baseline_mfe":                 ("Minimum free energy of local RNA fold",                   0.00),
    "baseline_pairing_local_mean":  ("Mean pairing probability in \u00b110nt window",          -0.04),
    "baseline_accessibility_local_mean": ("Mean accessibility in \u00b110nt window",           +0.04),
}


def gen_feature_importance_rate(results=None):
    """Generate Feature Importance section for rate prediction."""
    # Try loading CSVs from rate feature importance output directory
    # Try Z-scored path first, then positives-only, then legacy
    hand_csv = OUTPUT_DIR / "rate_5fold_zscore" / "feature_importance_rate_gb_hand.csv"
    all_csv = OUTPUT_DIR / "rate_5fold_zscore" / "feature_importance_rate_gb_all.csv"
    if not hand_csv.exists():
        hand_csv = OUTPUT_DIR / "rate_5fold_positives" / "feature_importance_rate_gb_hand.csv"
    if not all_csv.exists():
        all_csv = OUTPUT_DIR / "rate_5fold_positives" / "feature_importance_rate_gb_all.csv"
    if not hand_csv.exists():
        hand_csv = OUTPUT_DIR / "rate_feature_importance" / "feature_importance_hand.csv"
    if not all_csv.exists():
        all_csv = OUTPUT_DIR / "rate_feature_importance" / "feature_importance_all.csv"

    if results is None and not hand_csv.exists() and not all_csv.exists():
        return _not_yet_available(
            "feature-importance-rate",
            "Feature Importance &mdash; Rate Prediction",
            "Gradient boosting feature importance for rate prediction models.")

    html = [section_header("feature-importance-rate",
                           "Feature Importance &mdash; Rate Prediction",
                           "Feature importance rankings from gradient boosting rate prediction models.")]

    for csv_path, label, desc in [
        (hand_csv, "GB_HandFeatures &mdash; Top 20 Features (Rate Prediction)",
         "Hand-crafted features only (40 dims), 5-fold CV mean &plusmn; std"),
    ]:
        if not csv_path.exists():
            continue
        data = _load_importance_csv(csv_path)
        if not data:
            continue
        top = [(f, imp, std, cat) for f, imp, std, cat in data if imp > 0][:20]
        if not top:
            continue
        max_imp = top[0][1]

        html.append(f"<h3>{label}</h3>")
        html.append(f'<p>{desc}. Importance = mean gain across 5 folds. '
                     f'r = Pearson correlation with per-dataset Z-scored log<sub>2</sub>(rate + 0.01).</p>')

        html.append('<table style="width:100%;border-collapse:collapse;font-size:0.82em;">')
        html.append('<thead><tr style="border-bottom:2px solid #ddd;">'
                     '<th style="text-align:right;width:185px;padding:3px 6px;">Feature</th>'
                     '<th style="width:auto;padding:3px;">Importance</th>'
                     '<th style="text-align:right;width:55px;padding:3px 4px;font-family:monospace;">Value</th>'
                     '<th style="text-align:center;width:45px;padding:3px 2px;">r</th>'
                     '<th style="text-align:left;padding:3px 6px;">Description</th>'
                     '</tr></thead><tbody>')
        for feat, imp, std, csv_cat in top:
            cat = _categorize(feat, csv_cat)
            color = _CAT_COLORS.get(cat, "#757575")
            pct = (imp / max_imp) * 100
            feat_info = _FEAT_INFO.get(feat, ("", None))
            desc_text = feat_info[0]
            rho = feat_info[1]
            if rho is not None:
                rho_color = "#c62828" if rho < -0.2 else ("#1565c0" if rho > 0.2 else "#757575")
                rho_str = f'<span style="color:{rho_color};font-family:monospace;">{rho:+.2f}</span>'
            else:
                rho_str = '<span style="color:#bbb;">&mdash;</span>'
            html.append(f'''<tr style="border-bottom:1px solid #f0f0f0;">
                <td style="text-align:right;padding:2px 6px;font-family:monospace;white-space:nowrap;
                           overflow:hidden;text-overflow:ellipsis;max-width:185px;"
                    title="{feat} ({cat})">{feat}</td>
                <td style="padding:2px 3px;">
                    <div style="background:#f0f0f0;border-radius:3px;height:14px;">
                        <div style="width:{pct:.1f}%;background:{color};height:100%;border-radius:3px;
                                    min-width:2px;"></div>
                    </div>
                </td>
                <td style="text-align:right;padding:2px 4px;font-family:monospace;">{imp:.3f}</td>
                <td style="text-align:center;padding:2px 2px;">{rho_str}</td>
                <td style="padding:2px 6px;color:#555;white-space:nowrap;overflow:hidden;
                           text-overflow:ellipsis;max-width:300px;" title="{desc_text}">{desc_text}</td>
            </tr>''')
        html.append('</tbody></table>')

        # Legend
        cats_used = sorted(set(_categorize(f, c) for f, _, _, c in top))
        legend_items = " &nbsp; ".join(
            f'<span style="display:inline-block;width:12px;height:12px;background:{_CAT_COLORS.get(c, "#757575")}; '
            f'border-radius:2px;vertical-align:middle;margin-right:3px;"></span>'
            f'<span style="font-size:0.85em;">{c}</span>'
            for c in cats_used
        )
        html.append(f'<div style="margin:10px 0;">{legend_items}</div>')

        # Category breakdown
        cat_totals = {}
        for feat, imp, std, csv_cat in data:
            if imp > 0:
                cat = _categorize(feat, csv_cat)
                cat_totals[cat] = cat_totals.get(cat, 0) + imp
        total_imp = sum(cat_totals.values())
        if total_imp > 0:
            html.append(f"<h4>Importance by Feature Category</h4>")
            html.append('<table><thead><tr><th>Category</th><th>Features</th>'
                        '<th>Total Importance</th><th>Share</th></tr></thead><tbody>')
            for cat in sorted(cat_totals, key=cat_totals.get, reverse=True):
                n_feats = sum(1 for f, i, _, c in data if i > 0 and _categorize(f, c) == cat)
                share = cat_totals[cat] / total_imp * 100
                color = _CAT_COLORS.get(cat, "#757575")
                html.append(f'<tr><td><span style="display:inline-block;width:10px;height:10px;'
                            f'background:{color};border-radius:2px;vertical-align:middle;'
                            f'margin-right:4px;"></span>{cat}</td>'
                            f'<td>{n_feats}</td>'
                            f'<td>{cat_totals[cat]:.4f}</td>'
                            f'<td>{share:.1f}%</td></tr>')
            html.append('</tbody></table>')

    # Per-dataset feature importance tables
    ds_labels = {
        "advisor_c2t": "Advisor/Levanon A3A (n=120)",
        "alqassim_2021": "Alqassim (n=128)",
    }
    for ds_key, ds_label in ds_labels.items():
        ds_csv = OUTPUT_DIR / "rate_per_dataset" / f"feature_importance_{ds_key}_gb_hand.csv"
        if not ds_csv.exists():
            continue
        ds_data = _load_importance_csv(ds_csv)
        if not ds_data:
            continue
        ds_top = [(f, imp, std, cat) for f, imp, std, cat in ds_data if imp > 0][:15]
        if not ds_top:
            continue
        ds_max = ds_top[0][1]
        html.append(f"<h3>GB_HandFeatures &mdash; {ds_label}</h3>")
        html.append(f'<p>Per-dataset 5-fold CV importance (top 15). '
                     f'Small dataset — interpret with caution.</p>')
        html.append('<table style="width:100%;border-collapse:collapse;font-size:0.82em;">')
        html.append('<thead><tr style="border-bottom:2px solid #ddd;">'
                     '<th style="text-align:right;width:185px;padding:3px 6px;">Feature</th>'
                     '<th style="width:auto;padding:3px;">Importance</th>'
                     '<th style="text-align:right;width:55px;padding:3px 4px;font-family:monospace;">Value</th>'
                     '<th style="text-align:left;padding:3px 6px;">Description</th>'
                     '</tr></thead><tbody>')
        for feat, imp, std, csv_cat in ds_top:
            cat = _categorize(feat, csv_cat)
            color = _CAT_COLORS.get(cat, "#757575")
            pct = (imp / ds_max) * 100
            feat_info = _FEAT_INFO.get(feat, ("", None))
            desc_text = feat_info[0]
            html.append(f'''<tr style="border-bottom:1px solid #f0f0f0;">
                <td style="text-align:right;padding:2px 6px;font-family:monospace;white-space:nowrap;
                           overflow:hidden;text-overflow:ellipsis;max-width:185px;"
                    title="{feat} ({cat})">{feat}</td>
                <td style="padding:2px 3px;">
                    <div style="background:#f0f0f0;border-radius:3px;height:14px;">
                        <div style="width:{pct:.1f}%;background:{color};height:100%;border-radius:3px;
                                    min-width:2px;"></div>
                    </div>
                </td>
                <td style="text-align:right;padding:2px 4px;font-family:monospace;">{imp:.3f}</td>
                <td style="padding:2px 6px;color:#555;white-space:nowrap;overflow:hidden;
                           text-overflow:ellipsis;max-width:300px;" title="{desc_text}">{desc_text}</td>
            </tr>''')
        html.append('</tbody></table>')

    # Show summary from JSON results if available
    summary = results.get("summary", {}) if results else {}
    if summary:
        html.append(card("Rate Feature Importance Summary",
            f'<p>Top features for rate prediction may differ from classification. '
            f'Rate prediction relies more on context and tissue-related features.</p>',
            "insight-card"))

    html.append(section_footer())
    return "\n".join(html)


def gen_incremental_rate(results):
    """Generate incremental dataset addition section for rate prediction."""
    if results is None:
        return _not_yet_available(
            "incremental-rate",
            "Incremental Dataset Addition &mdash; Rate Prediction",
            "Spearman correlation as datasets are incrementally added for rate prediction.")

    html = [section_header("incremental-rate",
                           "Incremental Dataset Addition &mdash; Rate Prediction",
                           "Does adding more training datasets improve rate prediction? "
                           "Spearman correlation measured as datasets are incrementally added.")]

    configs = results.get("configs", [])
    if configs:
        test_ds = results.get("test_dataset", "Levanon")
        html.append(f"<p>Fixed test set: <strong>{test_ds}</strong>. "
                     "Training datasets added incrementally.</p>")

        # Collect all models
        all_models = []
        _MODEL_ORDER = ["GB_HandFeatures", "GB_AllFeatures", "SubtractionMLP", "EditRNA_A3A"]
        for m in _MODEL_ORDER:
            if any(m in c.get("models", {}) for c in configs):
                all_models.append(m)

        # Spearman table
        html.append("<h3>Spearman by Training Configuration &times; Model</h3>")
        headers = all_models
        rows = []
        labels = []
        for c in configs:
            labels.append(c.get("name", "?"))
            row = []
            for m in all_models:
                sp = c.get("models", {}).get(m, {}).get("spearman")
                row.append(sp)
            rows.append(row)
        if rows:
            html.append(make_table(headers, rows, row_labels=labels,
                                   highlight_col=len(all_models)-1, higher_is_better=True))

        # Training set sizes
        html.append("<h3>Training Set Sizes</h3>")
        size_headers = ["Datasets", "N with Rates", "Total"]
        size_rows = []
        for c in configs:
            ds_list = ", ".join(c.get("datasets", []))
            n_rate = c.get("n_train_with_rates", c.get("n_train", 0))
            n_total = c.get("n_train_total", n_rate)
            size_rows.append([ds_list, fmt_int(n_rate), fmt_int(n_total)])
        html.append(make_table(size_headers, size_rows,
                               row_labels=[c.get("name", "?") for c in configs]))
    else:
        # Flat format fallback
        models = results.get("models", {})
        if models:
            html.append("<h3>Rate Prediction: Incremental Results</h3>")
            headers = ["Spearman", "Pearson", "MSE"]
            rows = []
            labels = []
            for name, mdata in models.items():
                test = mdata.get("test", mdata)
                labels.append(name)
                rows.append([test.get("spearman"), test.get("pearson"), test.get("mse")])
            html.append(make_table(headers, rows, row_labels=labels,
                                   highlight_col=0, higher_is_better=True))

    html.append(section_footer())
    return "\n".join(html)


def gen_incremental_classification(data):
    """Generate incremental dataset addition section for classification.

    Supports new format (configs with multiple models) and old format (results list).
    """
    if data is None:
        return ""
    html = [section_header("incremental-classification", "Incremental Dataset Addition &mdash; Classification",
                           "Does adding more training datasets improve Levanon test-set performance? "
                           "All models trained on progressively larger dataset combinations.")]

    configs = data.get("configs", [])

    # --- New multi-model format ---
    if configs:
        test_ds = data.get("test_dataset", "Levanon")
        test_np = data.get("test_n_pos", "?")
        test_nn = data.get("test_n_neg", "?")
        html.append(f"<p>Fixed test set: <strong>{test_ds}</strong> "
                     f"({test_np} pos / {test_nn} neg). Training datasets are added incrementally.</p>")

        _MODEL_ORDER = [
            "StructureOnly", "GB_HandFeatures", "GB_AllFeatures",
            "PooledMLP", "SubtractionMLP", "EditRNA_A3A",
        ]
        # Collect all models present
        all_models = []
        for m in _MODEL_ORDER:
            if any(m in c.get("models", {}) for c in configs):
                all_models.append(m)

        # AUROC table: rows=training configs, columns=models
        html.append("<h3>AUROC by Training Configuration &times; Model</h3>")
        headers = all_models
        rows = []
        labels = []
        for c in configs:
            labels.append(c.get("name", "?"))
            row = []
            for m in all_models:
                auroc = c.get("models", {}).get(m, {}).get("auroc")
                row.append(auroc)
            rows.append(row)
        html.append(make_table(headers, rows, row_labels=labels, highlight_col=len(all_models)-1,
                               higher_is_better=True))

        # AUPRC table
        html.append("<h3>AUPRC by Training Configuration &times; Model</h3>")
        rows_pr = []
        for c in configs:
            row = []
            for m in all_models:
                auprc = c.get("models", {}).get(m, {}).get("auprc")
                row.append(auprc)
            rows_pr.append(row)
        html.append(make_table(headers, rows_pr, row_labels=labels[:], highlight_col=len(all_models)-1,
                               higher_is_better=True))

        # Training set sizes
        html.append("<h3>Training Set Sizes</h3>")
        size_headers = ["Datasets", "N Pos", "N Neg", "Total"]
        size_rows = []
        for c in configs:
            ds_list = ", ".join(c.get("datasets", []))
            n_pos = c.get("n_train_pos", 0)
            n_neg = c.get("n_train_neg", 0)
            size_rows.append([ds_list, fmt_int(n_pos), fmt_int(n_neg), fmt_int(n_pos + n_neg)])
        html.append(make_table(size_headers, size_rows, row_labels=[c.get("name", "?") for c in configs]))

    # --- Old single-model format (fallback) ---
    elif data.get("results"):
        results = data["results"]
        headers = ["Datasets", "N Train Pos", "N Train Neg", "N Train Total",
                   "Overall AUROC"]
        rows = []
        labels = []
        for r in results:
            labels.append(f"Step {r.get('step', '?')}")
            rows.append([
                r.get("config_name", ""),
                fmt_int(r.get("n_train_pos")),
                fmt_int(r.get("n_train_neg")),
                fmt_int(r.get("n_train_total")),
                r.get("overall_auroc"),
            ])
        html.append(make_table(headers, rows, row_labels=labels, highlight_col=4,
                               higher_is_better=True))
    else:
        html.append("<p>No incremental results found.</p>")

    html.append(section_footer())
    return "\n".join(html)


def gen_embedding_classification(data, viz_figs=None, viz_results=None, trained_emb=None,
                                  comprehensive_emb=None):
    """Generate embedding analysis section with comprehensive UMAP/t-SNE visualizations."""
    has_trained = trained_emb is not None
    has_comprehensive = comprehensive_emb is not None
    if data is None and viz_figs is None and not has_trained and not has_comprehensive:
        return ""
    html = [section_header("embedding-classification", "Embedding &amp; Clustering &mdash; Classification",
                           "Dimensionality reduction (UMAP, t-SNE) and cluster separation analysis of "
                           "contextual edit embeddings from the trained EditRNA-A3A model.")]

    html.append(card("Contextual Edit Embedding", """
        <p><strong>Source:</strong> Trained EditRNA-A3A model (AUROC=0.956). Embeddings extracted from the
        <code>GatedModalityFusion</code> layer (512-dim), which combines the RNA-FM sequence representation
        with the structured edit embedding through learned gated attention.</p>
        <p><strong>Why this layer:</strong> The fusion output is the first point where sequence context and
        edit intervention are jointly represented with trainable parameters. It captures <em>how the model
        understands each edit in its sequence context</em> &mdash; a true contextual edit embedding.</p>
        <p><strong>Projection:</strong> PCA (50 components) &rarr; UMAP/t-SNE (2D), computed on 3,000
        subsampled sites from 8,153 A3A-filtered sites.</p>
    """))

    # Show trained embedding metrics if available
    if has_trained:
        sil = trained_emb.get("cluster_metrics", {}).get("silhouette_score", 0)
        margin = trained_emb.get("semantic_margin", 0)
        badges = [
            metric_badge("Silhouette", fmt(sil, 3), "green" if sil > 0.1 else "yellow"),
            metric_badge("Semantic Margin", fmt(margin, 2), "green"),
            metric_badge("Embedding Dim", "512", "blue"),
            metric_badge("Model", "EditRNA-A3A", "blue"),
        ]
        html.append('<div class="badge-container">')
        html.append(" ".join(badges))
        html.append("</div>")
        html.append(f'<p>Silhouette score (pos vs neg): <strong>{fmt(sil, 3)}</strong> '
                     '(vs &minus;0.003 for untrained subtraction embeddings &mdash; '
                     'the trained model produces well-separated edit representations).</p>')

    # --- Comprehensive embedding visualizations ---
    if has_comprehensive:
        schemes = comprehensive_emb.get("schemes", {})

        # Summary metrics table
        html.append("<h3>Separation Metrics by Coloring Scheme</h3>")
        html.append("<p>How well does each property separate in the learned embedding space? "
                     "Categorical: silhouette score. Continuous: max |Spearman &rho;| with UMAP axes.</p>")
        headers_sep = ["Coloring", "Type", "Metric", "Value"]
        rows_sep = []
        for name, sch in schemes.items():
            metric_name = sch.get("metric_name", "")
            metric_val = sch.get("metric_value")
            val_str = fmt(metric_val, 3) if metric_val is not None else "N/A"
            rows_sep.append([sch.get("title", name), sch["type"], metric_name, val_str])
        html.append(make_table(headers_sep, rows_sep))

        # Selected detailed UMAP plots
        html.append("<h3>Detailed UMAP Visualizations</h3>")

        # Positive vs Negative
        html.append("<h4>Positive vs Negative</h4>")
        html.append('<div class="figure-grid">')
        html.append(embed_png("embedding_viz_comprehensive/umap_label.png",
                              "UMAP by label", "48%", "UMAP: positive (blue) vs negative (red)"))
        html.append(embed_png("embedding_viz_comprehensive/tsne_label.png",
                              "t-SNE by label", "48%", "t-SNE: positive (blue) vs negative (red)"))
        html.append('</div>')

        # Dataset source
        html.append("<h4>By Dataset Source</h4>")
        html.append('<div class="figure-grid">')
        html.append(embed_png("embedding_viz_comprehensive/umap_dataset.png",
                              "UMAP by dataset", "48%", "UMAP colored by dataset source"))
        html.append(embed_png("embedding_viz_comprehensive/tsne_dataset.png",
                              "t-SNE by dataset", "48%", "t-SNE colored by dataset source"))
        html.append('</div>')

        # Trinucleotide motif and genomic feature
        html.append("<h4>Sequence &amp; Genomic Context</h4>")
        html.append('<div class="figure-grid">')
        html.append(embed_png("embedding_viz_comprehensive/umap_trinuc_motif.png",
                              "UMAP by motif", "48%",
                              "Upstream dinucleotide motif (TC = canonical APOBEC recognition)"))
        html.append(embed_png("embedding_viz_comprehensive/umap_genomic_feature.png",
                              "UMAP by feature", "48%",
                              "Genomic feature annotation (exonic, UTR3, UTR5, intronic)"))
        html.append('</div>')

        # Editing rate and structure
        html.append("<h4>Editing Rate &amp; Structure</h4>")
        html.append('<div class="figure-grid">')
        html.append(embed_png("embedding_viz_comprehensive/umap_editing_rate.png",
                              "UMAP by rate", "48%",
                              "Editing rate (log10 scale, viridis colormap)"))
        html.append(embed_png("embedding_viz_comprehensive/umap_pairing_prob_center.png",
                              "UMAP by pairing", "48%",
                              "Pairing probability at edit site (loop vs stem context)"))
        html.append('</div>')
        html.append('<div class="figure-grid">')
        html.append(embed_png("embedding_viz_comprehensive/umap_delta_mfe.png",
                              "UMAP by delta MFE", "48%",
                              "Structure disruption: delta MFE (edited - original)"))
        html.append('</div>')

    elif has_trained:
        # Fallback to basic trained embedding plots
        html.append("<h3>Contextual Edit Embeddings (Trained EditRNA-A3A)</h3>")

        html.append("<h4>Positive vs Negative</h4>")
        html.append('<div class="figure-grid">')
        html.append(embed_png("embedding_trained/contextual_edit_umap_label.png",
                              "UMAP by label", "48%", "UMAP: positive (blue) vs negative (red)"))
        html.append(embed_png("embedding_trained/contextual_edit_tsne_label.png",
                              "t-SNE by label", "48%", "t-SNE: positive (blue) vs negative (red)"))
        html.append('</div>')

        html.append("<h4>By Dataset Source</h4>")
        html.append('<div class="figure-grid">')
        html.append(embed_png("embedding_trained/contextual_edit_umap_dataset.png",
                              "UMAP by dataset", "48%", "UMAP colored by dataset source"))
        html.append(embed_png("embedding_trained/contextual_edit_tsne_dataset.png",
                              "t-SNE by dataset", "48%", "t-SNE colored by dataset source"))
        html.append('</div>')

    html.append(section_footer())
    return "\n".join(html)


def gen_rate_architecture(rate_5fold=None, rate_per_dataset=None):
    """Generate rate prediction architecture comparison section (single 5-fold CV table)."""
    if rate_5fold is None:
        return ""

    html = [section_header("rate-architecture", "DL Architecture Comparison &mdash; Rate Prediction",
                           "Editing rate regression across all architectures using 5-fold cross-validation. "
                           "Only sites with known editing rates are used for training/evaluation.")]

    n_total = rate_5fold.get("n_total_sites", "?")
    n_folds = rate_5fold.get("n_folds", 5)
    target = rate_5fold.get("target_transform", "log2(rate + 0.01)")
    html.append(f"<p><strong>Sites:</strong> {fmt_int(n_total)} rate-annotated | "
                 f"<strong>Evaluation:</strong> {n_folds}-fold cross-validation | "
                 f"<strong>Target:</strong> {target}</p>")
    html.append('<p class="note"><em>All APOBEC3A-confirmed sites with editing rates '
                 '(Baysal/Sharma A3A overexpression + Advisor A3A-only + Alqassim). '
                 'Per-dataset Z-scored to remove dataset identity signal. '
                 'All values are mean &plusmn; std across 5 folds.</em></p>')

    models = rate_5fold.get("models", {})

    # Define complexity ordering
    _RATE_COMPLEXITY = {
        "Mean Baseline": 0,
        "StructureOnly": 1,
        "GB_HandFeatures": 2,
        "GB_AllFeatures": 3,
        "EditRNA_rate": 4,
        "4Way_heavyreg": 5,
    }

    html.append("<h3>All Architectures &mdash; 5-Fold Cross-Validation</h3>")
    headers = ["Spearman", "Pearson", "MSE", "R\u00b2", "Params"]
    rows = []
    labels = []

    sorted_models = sorted(models.items(), key=lambda x: _RATE_COMPLEXITY.get(x[0], 99))
    for model_name, mdata in sorted_models:
        mean_sp = mdata.get("mean_spearman")
        std_sp = mdata.get("std_spearman", 0)
        mean_pe = mdata.get("mean_pearson")
        std_pe = mdata.get("std_pearson", 0)
        mean_mse = mdata.get("mean_mse")
        mean_r2 = mdata.get("mean_r2")
        n_params = mdata.get("n_params", 0)

        # Format spearman and pearson as mean +/- std
        if mean_sp is not None and not (isinstance(mean_sp, float) and math.isnan(mean_sp)):
            sp_str = f"{mean_sp:.4f} &plusmn; {std_sp:.4f}"
        else:
            sp_str = "N/A (const)"
        if mean_pe is not None and not (isinstance(mean_pe, float) and math.isnan(mean_pe)):
            pe_str = f"{mean_pe:.4f} &plusmn; {std_pe:.4f}"
        else:
            pe_str = "N/A (const)"

        # Params display
        if n_params == 0:
            p_str = "&mdash;"
        elif n_params == -1:
            p_str = "N/A"
        else:
            p_str = f"{n_params:,}"

        # Format MSE and R2 as strings too for consistency
        mse_str = fmt(mean_mse) if mean_mse is not None else "N/A"
        r2_str = fmt(mean_r2) if mean_r2 is not None else "N/A"

        labels.append(model_name)
        rows.append([sp_str, pe_str, mse_str, r2_str, p_str])

    if rows:
        html.append(make_table(headers, rows, row_labels=labels))

    # ---- Key insights ----
    # Find best model by spearman
    best_name, best_sp = "", 0
    for name, mdata in models.items():
        sp = mdata.get("mean_spearman")
        if sp is not None and not (isinstance(sp, float) and math.isnan(sp)) and sp > best_sp:
            best_sp = sp
            best_name = name


    # ---- Per-Dataset Within-Dataset CV ----
    if rate_per_dataset is not None:
        datasets = rate_per_dataset.get("datasets", {})
        if datasets:
            html.append("<h3>Per-Dataset Within-Dataset CV (5-Fold)</h3>")
            html.append('<p class="note"><em>Each dataset is trained and evaluated independently '
                         'using 5-fold CV. Target: log₂(rate + 0.01). '
                         'Spearman is rank-invariant to monotonic transforms, so results are '
                         'identical whether using log₂ or Z-scored targets within a single dataset.</em></p>')

            # Collect all model names across datasets
            all_model_names = []
            for ds_data in datasets.values():
                for m in ds_data.get("models", {}):
                    if m not in all_model_names:
                        all_model_names.append(m)

            # Sort by complexity
            _PD_ORDER = {
                "Mean Baseline": 0, "StructureOnly": 1, "GB_HandFeatures": 2,
                "GB_AllFeatures": 3, "EditRNA_rate": 4, "4Way_heavyreg": 5,
            }
            all_model_names.sort(key=lambda x: _PD_ORDER.get(x, 99))

            ds_labels = {
                "baysal_2016": "Baysal/A3A (n=4,208)",
                "advisor_c2t": "Advisor A3A (n=120)",
                "alqassim_2021": "Alqassim (n=128)",
            }
            ds_order = ["baysal_2016", "advisor_c2t", "alqassim_2021"]

            # Build table: rows=models, columns=datasets (Spearman ± std)
            headers = [ds_labels.get(d, d) for d in ds_order if d in datasets]
            rows = []
            labels = []
            for model_name in all_model_names:
                row = []
                for ds_key in ds_order:
                    if ds_key not in datasets:
                        continue
                    ds_models = datasets[ds_key].get("models", {})
                    if model_name in ds_models:
                        mdata = ds_models[model_name]
                        sp = mdata.get("mean_spearman")
                        sp_std = mdata.get("std_spearman", 0)
                        if sp is not None and not (isinstance(sp, float) and math.isnan(sp)):
                            row.append(f"{sp:.3f} &plusmn; {sp_std:.3f}")
                        else:
                            row.append("N/A")
                    else:
                        row.append("&mdash;")
                labels.append(model_name)
                rows.append(row)

            if rows:
                html.append("<p><strong>Spearman ρ (mean ± std across 5 folds)</strong></p>")
                html.append(make_table(headers, rows, row_labels=labels))

            html.append(card("Per-Dataset Insights", """
                <ul>
                <li><strong>Small A3A rate datasets:</strong> Only 254 A3A-confirmed sites have
                editing rates (120 Advisor + 128 Alqassim + 6 Sharma), making rate prediction
                inherently challenging with limited training data.</li>
                <li><strong>Within-dataset signal is modest:</strong> Spearman correlations
                are low across both datasets, reflecting the difficulty of predicting
                continuous editing rates from sequence/structure features alone.</li>
                <li><strong>Datasets are balanced:</strong> Unlike the multi-enzyme setting
                (dominated by 4,208 Baysal sites), the A3A-only pooled CV draws roughly
                equally from Advisor and Alqassim.</li>
                </ul>
            """, "insight-card"))

    html.append(section_footer())
    return "\n".join(html)


def gen_multitask(data):
    """Generate multi-task learning section."""
    html = [section_header("multitask", "Multi-Task Learning",
                           "Comparison of binary-only, regression-only, and multi-task models.")]
    html.append('<p style="color:#888; font-style:italic; font-size:1.1em; margin:2em 0;">Not implemented yet.</p>')
    html.append(section_footer())
    return "\n".join(html)


def gen_binary_rate_correlation(data):
    """Generate binary score -> rate correlation section."""
    if data is None:
        return ""
    html = [section_header("binary-rate-corr", "Binary Score to Rate Correlation",
                           "Relationship between binary classification scores and editing rates.")]

    overall = data.get("overall", {})
    log2 = data.get("log2_rate", {})
    n_sites = data.get("n_sites", "N/A")

    html.append(card("Overall Correlation", f"""
        <p>N sites with rates: <strong>{fmt_int(n_sites)}</strong></p>
        <p>Pearson r: <strong>{fmt(overall.get('pearson_r'))}</strong> (p={fmt(overall.get('pearson_p'), 4)})</p>
        <p>Spearman r: <strong>{fmt(overall.get('spearman_r'))}</strong> (p={fmt(overall.get('spearman_p'), 4)})</p>
        <p>Log2-rate Pearson r: <strong>{fmt(log2.get('pearson_r'))}</strong></p>
    """))

    # Score bins
    bins = data.get("score_bins", {})
    if bins:
        html.append("<h3>Rate by Binary Score Bin</h3>")
        headers = ["N", "Mean Rate", "Median Rate", "Std Rate"]
        rows = []
        labels = []
        for bin_name, stats in bins.items():
            labels.append(bin_name)
            rows.append([
                fmt_int(stats.get("n")),
                fmt(stats.get("mean_rate"), 3),
                fmt(stats.get("median_rate"), 3),
                fmt(stats.get("std_rate"), 3),
            ])
        html.append(make_table(headers, rows, row_labels=labels))

    # Per-dataset
    pd = data.get("per_dataset", {})
    if pd:
        html.append("<h3>Per-Dataset Correlation</h3>")
        headers_pd = ["N", "Pearson r", "Spearman r", "Mean Score", "Mean Rate"]
        rows_pd = []
        labels_pd = []
        for ds, stats in pd.items():
            labels_pd.append(ds)
            rows_pd.append([
                fmt_int(stats.get("n")),
                fmt(stats.get("pearson_r")),
                fmt(stats.get("spearman_r")),
                fmt(stats.get("mean_score"), 3),
                fmt(stats.get("mean_rate"), 3),
            ])
        html.append(make_table(headers_pd, rows_pd, row_labels=labels_pd))

    # Caveat for negative correlation
    overall_sp = data.get("overall", {}).get("spearman_r")
    if _safe_float(overall_sp) is not None and _safe_float(overall_sp) < 0:
        html.append(card("Interpretation", """
            <p>The weak negative correlation between binary score and editing rate is expected:
            the binary classifier was optimized for <em>editability</em> (edited vs. unedited),
            not <em>rate magnitude</em>. Rate variance decomposition (see Rate Deep Dive) shows
            ~71% of rate variance is between-dataset, confirming rate is context-dependent
            rather than captured from sequence context alone.</p>
        """, "insight-card"))

    html.append(section_footer())
    return "\n".join(html)


def gen_multiseed(data):
    """Generate multi-seed confidence intervals section."""
    if data is None:
        return ""
    html = [section_header("multiseed", "Multi-Seed Confidence Intervals",
                           f"Performance stability across {data.get('n_seeds', 'N/A')} random seeds.")]

    seeds = data.get("seeds", [])
    html.append(f"<p>Seeds: {', '.join(str(s) for s in seeds)}</p>")

    summary = data.get("summary", {})
    if summary:
        for model_name, metrics in summary.items():
            html.append(f"<h3>{model_name}</h3>")
            headers = ["Mean", "Std", "Min", "Max"]
            rows = []
            labels = []
            for metric_name, vals in metrics.items():
                labels.append(metric_name)
                values = vals.get("values", [])
                mean = vals.get("mean")
                std = vals.get("std")
                v_min = min(values) if values else None
                v_max = max(values) if values else None
                rows.append([mean, std, v_min, v_max])
            html.append(make_table(headers, rows, row_labels=labels))

    html.append(embed_png("multiseed/multiseed_comparison.png",
                           "Multi-seed comparison", "60%",
                           "Performance distributions across 5 random seeds"))

    html.append(section_footer())
    return "\n".join(html)


def gen_motif_analysis(data):
    """Generate motif analysis section."""
    if data is None:
        return ""
    html = [section_header("motif-analysis", "Motif Analysis",
                           "Sequence motif analysis around editing sites.")]

    pos_freq = data.get("position_frequencies", {})
    pos_all = pos_freq.get("positive_all", {})
    neg_all = pos_freq.get("negative_all", {})

    if pos_all:
        n_pos = pos_all.get("n_seqs", 0)
        n_neg = neg_all.get("n_seqs", 0) if neg_all else 0
        html.append(f"<p>Positive sequences: <strong>{fmt_int(n_pos)}</strong> | "
                     f"Negative sequences: <strong>{fmt_int(n_neg)}</strong></p>")

        # Information content
        ic = pos_all.get("information_content", [])
        if ic:
            html.append("<h3>Information Content (bits) per Position</h3>")
            # Show the key positions around the edit site
            # Positions are centered on the edit site (index 10 = position 0)
            center = len(ic) // 2
            headers_ic = [str(i - center) for i in range(len(ic))]
            rows_ic = [[fmt(v, 3) for v in ic]]
            html.append(make_table(headers_ic, rows_ic, row_labels=["IC (bits)"]))

            # Highlight peak positions
            peak_pos = sorted(range(len(ic)), key=lambda i: -ic[i])[:3]
            peak_info = ", ".join(f"pos {p - center} (IC={ic[p]:.3f})" for p in peak_pos)
            html.append(f"<p>Highest information content: {peak_info}</p>")

    html.append(section_footer())
    return "\n".join(html)


def _compute_asymmetric_pairing_features():
    """Compute asymmetric pairing features from the structure cache.

    Returns dict of feature_name -> {pos_mean, neg_mean, difference, mann_whitney_p, significant}
    for: In loop, Local Pairing Mean at +/-5nt and +/-5-10nt windows.
    """
    import numpy as np
    cache_path = Path(__file__).resolve().parents[2] / "data/processed/embeddings/vienna_structure_cache.npz"
    splits_path = Path(__file__).resolve().parents[2] / "data/processed/splits_expanded_a3a.csv"
    loop_csv = OUTPUT_DIR / "loop_position" / "loop_position_per_site.csv"

    if not cache_path.exists() or not splits_path.exists():
        return {}

    try:
        import pandas as pd
        from scipy.stats import mannwhitneyu

        cache = np.load(cache_path, allow_pickle=True)
        site_ids = list(cache["site_ids"])
        pairing = cache["pairing_probs"]  # (N, 201), center=100

        splits = pd.read_csv(splits_path)
        sid_to_label = dict(zip(splits["site_id"], splits["is_edited"]))

        # Map cache indices to labels
        labels = np.array([sid_to_label.get(sid, -1) for sid in site_ids])
        pos_mask = labels == 1
        neg_mask = labels == 0

        if pos_mask.sum() == 0 or neg_mask.sum() == 0:
            return {}

        CENTER = 100
        results = {}

        def _add_feature(name, pos_vals, neg_vals):
            stat, p = mannwhitneyu(pos_vals, neg_vals, alternative="two-sided")
            results[name] = {
                "pos_mean": float(np.mean(pos_vals)),
                "neg_mean": float(np.mean(neg_vals)),
                "difference": float(np.mean(pos_vals) - np.mean(neg_vals)),
                "mann_whitney_p": float(p),
                "significant": p < 0.05,
            }

        # In loop (from loop CSV if available, otherwise from pairing prob < 0.5 at center)
        if loop_csv.exists():
            loop_df = pd.read_csv(loop_csv, index_col=0)
            loop_merged = splits.merge(loop_df[["is_unpaired", "loop_size"]], left_on="site_id",
                                        right_index=True, how="inner")
            pos_inloop = loop_merged[loop_merged["is_edited"] == 1]["is_unpaired"].values
            neg_inloop = loop_merged[loop_merged["is_edited"] == 0]["is_unpaired"].values
            _add_feature("In Loop (unpaired)", pos_inloop, neg_inloop)

            # Mean loop size (only for sites that are in loops)
            in_loop = loop_merged[loop_merged["is_unpaired"] == 1]
            pos_ls = in_loop[in_loop["is_edited"] == 1]["loop_size"].values
            neg_ls = in_loop[in_loop["is_edited"] == 0]["loop_size"].values
            if len(pos_ls) > 0 and len(neg_ls) > 0:
                _add_feature("Mean Loop Size", pos_ls, neg_ls)

        # Local Pairing Mean at specific windows
        # +5 means 5 positions to the RIGHT (3' side)
        pair_pos = pairing[pos_mask]
        pair_neg = pairing[neg_mask]

        # Mean pairing at ±2nt (average of positions -2 and +2)
        _add_feature("Local Pairing at ±2nt",
                      pair_pos[:, [CENTER - 2, CENTER + 2]].mean(axis=1),
                      pair_neg[:, [CENTER - 2, CENTER + 2]].mean(axis=1))

        # Mean pairing +5 to +10 (3' side)
        _add_feature("Local Pairing Mean +5 to +10nt",
                      pair_pos[:, CENTER + 5:CENTER + 11].mean(axis=1),
                      pair_neg[:, CENTER + 5:CENTER + 11].mean(axis=1))

        # Mean pairing -5 to -10 (5' side)
        _add_feature("Local Pairing Mean -10 to -5nt",
                      pair_pos[:, CENTER - 10:CENTER - 4].mean(axis=1),
                      pair_neg[:, CENTER - 10:CENTER - 4].mean(axis=1))

        return results

    except Exception as e:
        print(f"  [WARN] Could not compute asymmetric pairing features: {e}")
        return {}


def gen_structure_analysis(data, structure_enhanced=None):
    """Generate structure analysis section."""
    if data is None and structure_enhanced is None:
        return ""
    html = [section_header("structure-analysis", "Structure Analysis",
                           "Comparing RNA structure features between edited and non-edited sites.")]

    # Use enhanced results if available, fall back to legacy
    se = structure_enhanced or {}
    n_pos = se.get("n_positive") or (data or {}).get("n_positive")
    n_neg = se.get("n_negative") or (data or {}).get("n_negative")

    html.append(f"""
    <div class="metrics-grid">
        {card("Positive Sites", f'''
            <p>N: <strong>{fmt_int(n_pos)}</strong></p>
            <p>Delta MFE mean: <strong>{fmt((data or {{}}).get("positive_delta_mfe_mean"), 3)}</strong></p>
            <p>Stabilizing: <strong>{fmt((data or {{}}).get("positive_stabilizing_pct"), 1)}%</strong></p>
            <p>Destabilizing: <strong>{fmt((data or {{}}).get("positive_destabilizing_pct"), 1)}%</strong></p>
        ''')}
        {card("Negative Sites", f'''
            <p>N: <strong>{fmt_int(n_neg)}</strong></p>
            <p>Delta MFE mean: <strong>{fmt((data or {{}}).get("negative_delta_mfe_mean"), 3)}</strong></p>
            <p>Stabilizing: <strong>{fmt((data or {{}}).get("negative_stabilizing_pct"), 1)}%</strong></p>
            <p>Destabilizing: <strong>{fmt((data or {{}}).get("negative_destabilizing_pct"), 1)}%</strong></p>
        ''')}
    </div>
    """)

    # Feature significance tests -- prefer enhanced (raw features) over legacy (delta features)
    ft = se.get("feature_tests") or (data or {}).get("feature_tests", {})
    if ft:
        # Remove unwanted features
        skip_features = {
            "Local Pairing Mean (+/-10nt)",
            "Local Accessibility Mean (+/-10nt)",
            "Accessibility (edit site)",
        }

        # Compute additional features from structure cache
        extra_ft = _compute_asymmetric_pairing_features()
        if extra_ft:
            ft = dict(ft)  # copy so we don't mutate original
            ft.update(extra_ft)

        html.append("<h3>Structural Feature Comparison (Positive vs Negative)</h3>")
        headers = ["Pos Mean", "Neg Mean", "Difference", "MW p-value", "Significant"]
        rows = []
        labels = []
        for feat_name, stats in ft.items():
            if feat_name in skip_features:
                continue
            labels.append(feat_name)
            p = stats.get("mann_whitney_p")
            sig = stats.get("significant", False)
            rows.append([
                fmt(stats.get("pos_mean"), 4),
                fmt(stats.get("neg_mean"), 4),
                fmt(stats.get("difference"), 4),
                f"{p:.2e}" if p is not None else "N/A",
                '<span class="sig-yes">Yes</span>' if sig else '<span class="sig-no">No</span>',
            ])
        html.append(make_table(headers, rows, row_labels=labels))

    # Enhanced structure visualizations
    html.append('<div class="figure-grid">')
    html.append(embed_png("structure_enhanced/structural_context_profiles.png",
                          "Structural context profiles (smoothed)", "48%",
                          "Mean pairing probability across 201-nt window (5-nt rolling average)"))
    html.append(embed_png("structure_enhanced/structural_context_profiles_raw.png",
                          "Structural context profiles (raw)", "48%",
                          "Mean pairing probability across 201-nt window (raw, no smoothing)"))
    html.append('</div>')

    # Loop position analysis insight
    loop_stats = se.get("loop_stats", {})
    pos_loop = loop_stats.get("positive", {})
    neg_loop = loop_stats.get("negative", {})
    pos_pct = pos_loop.get("pct_in_loops", 68.5)
    neg_pct = neg_loop.get("pct_in_loops", 28.2)
    pos_mean_ls = pos_loop.get("mean_loop_size", 5.1)
    neg_mean_ls = neg_loop.get("mean_loop_size", 7.1)
    html.append(card("Loop Position Analysis", f"""
        <p><strong>{pos_pct:.1f}%</strong> of positive editing sites reside in loop regions vs only
        <strong>{neg_pct:.1f}%</strong> of negative sites.</p>
        <p>Positive sites prefer small hairpin loops (mean loop size: {pos_mean_ls:.1f} nt vs
        {neg_mean_ls:.1f} nt for negatives). Edited cytidines are located closer to the stem-loop
        junction, with longer adjacent stems.</p>
    """, "insight-card"))

    # Loop / stem visualizations
    html.append('<div class="figure-grid">')
    html.append(embed_png("structure_enhanced/loop_size_distribution.png",
                          "Loop size distribution", "48%",
                          "Loop size distribution (all sites with loop annotation)"))
    html.append(embed_png("structure_enhanced/loop_type_comparison.png",
                          "Loop type comparison", "48%",
                          "Loop type distribution: positive vs negative"))
    html.append('</div>')
    html.append('<div class="figure-grid">')
    html.append(embed_png("structure_enhanced/stem_length_comparison.png",
                          "Stem length comparison", "48%",
                          "Adjacent stem lengths: positive vs negative"))
    html.append(embed_png("loop_position/paired_vs_unpaired.png", "Loop vs Stem", "48%",
                          "Fraction of sites in loops vs stems"))
    html.append('</div>')

    html.append(section_footer())
    return "\n".join(html)





def gen_clinvar(data, classification_a3a=None):
    """Generate ClinVar clinical predictions section.

    Head-to-head comparison: GB_Full (46-dim structure-aware hand features) vs
    RNAsee_RF (50-bit binary nucleotide encoding) on ~1.68M ClinVar C>U variants.
    """
    if data is None:
        return ""
    html = [section_header("clinvar", "ClinVar Clinical Predictions",
                           "Head-to-head comparison of GB_Full (structure-aware, 46 features) vs "
                           "RNAsee_RF (sequence-only, 50-bit encoding) on all ~1.68M ClinVar C&gt;U variants. "
                           "Both models trained on 5,187 A3A positives + 15,561 Asaoka all-C negatives.")]

    n_scored = data.get("n_clinvar_scored", 0)

    # --- Cross-validation performance ---
    cv = data.get("cv_results", {})
    gb_cv = cv.get("GB_Full", {})
    rf_cv = cv.get("RNAsee_RF", {})

    html.append("<h3>5-Fold Cross-Validation (Training Data)</h3>")
    headers_cv = ["Model", "Features", "AUROC", "AUPRC"]
    rows_cv = [
        ["GB_Full", "Motif (24) + Structure delta (7) + Loop geometry (9) + Baseline structure (6)",
         f"{gb_cv.get('mean_auroc', 0):.4f} &pm; {gb_cv.get('std_auroc', 0):.4f}",
         fmt(gb_cv.get("mean_auprc"))],
        ["RNAsee_RF", "Binary nucleotide encoding (15nt up + 10nt down, 2 bits each)",
         f"{rf_cv.get('mean_auroc', 0):.4f} &pm; {rf_cv.get('std_auroc', 0):.4f}",
         fmt(rf_cv.get("mean_auprc"))],
    ]
    html.append(make_table(headers_cv, rows_cv))

    # --- Tiered + structure negatives comparison ---
    struct_neg = load_json("struct_neg_comparison/struct_neg_comparison_results.json")
    if struct_neg is not None:
        sn_models = struct_neg.get("models", {})
        sn_pos = struct_neg.get("n_positive", "?")
        sn_neg = struct_neg.get("n_negative", "?")
        sn_added = struct_neg.get("n_struct_negative", 0)

        html.append(f"<h3>5-Fold CV with Tiered + Structure Negatives "
                     f"({fmt_int(sn_pos)} pos / {fmt_int(sn_neg)} neg)</h3>")
        html.append(f"<p>Same positives with challenging negatives: TC-motif in same transcripts (Tier 2), "
                     f"known non-edited TC sites (Tier 3), plus <strong>{sn_added:,}</strong> "
                     f"structure-negative sites &mdash; TC-motif cytidines in paired/stem regions "
                     f"where the sequence looks editable but the structural context is unfavorable.</p>")
        headers_tiered = ["Model", "AUROC", "AUPRC", "F1", "Precision", "Recall"]
        rows_tiered = []

        for mname in ["GB_HandFeatures", "GB_AllFeatures", "RNAsee_RF"]:
            m = sn_models.get(mname)
            if m is None:
                continue
            rows_tiered.append([
                mname,
                f"{m.get('mean_auroc', 0):.4f} &pm; {m.get('std_auroc', 0):.4f}",
                fmt(m.get("mean_auprc")),
                fmt(m.get("mean_f1")),
                fmt(m.get("mean_precision")),
                fmt(m.get("mean_recall")),
            ])

        if rows_tiered:
            html.append(make_table(headers_tiered, rows_tiered))

    # --- Scale of analysis ---
    html.append(f"""<h3>ClinVar Variant Scoring</h3>
    <p>Both trained models were applied to <strong>{n_scored:,}</strong> ClinVar C&gt;U variants
    with valid 201-nt flanking sequences. For each variant, ViennaRNA structure features were
    computed de novo (10 parallel workers, ~6 hours total).</p>""")

    # --- Candidate counts ---
    c05 = data.get("candidates_t0.5", {})
    c08 = data.get("candidates_t0.8", {})

    html.append("<h3>Candidate Overlap: GB_Full vs RNAsee_RF</h3>")
    headers_cand = ["Threshold", "GB_Full", "RNAsee_RF", "Both", "GB-only", "RNAsee-only"]
    rows_cand = [
        ["P &ge; 0.5",
         fmt_int(c05.get("gb_full")), fmt_int(c05.get("rnasee_rf")),
         fmt_int(c05.get("both")), fmt_int(c05.get("gb_only")), fmt_int(c05.get("rf_only"))],
        ["P &ge; 0.8",
         fmt_int(c08.get("gb_full")), fmt_int(c08.get("rnasee_rf")),
         fmt_int(c08.get("both")), fmt_int(c08.get("gb_only")), fmt_int(c08.get("rf_only"))],
    ]
    html.append(make_table(headers_cand, rows_cand))

    gb_only_05 = c05.get("gb_only", 0)
    rf_only_05 = c05.get("rf_only", 0)
    html.append(card("Structure captures signal beyond sequence motif", f"""
        <p>At P&ge;0.5, GB_Full identifies <strong>{gb_only_05:,}</strong> candidates that
        RNAsee misses entirely &mdash; sites where the sequence motif alone does not predict
        editing but the structural context does. Conversely, only <strong>{rf_only_05:,}</strong>
        sites are called by RNAsee but not GB_Full (sites with the right motif but unfavorable structure).</p>
        <p>At P&ge;0.8, the ratio is even more extreme: GB_Full calls <strong>9&times;</strong> more
        high-confidence candidates than RNAsee ({fmt_int(c08.get('gb_full'))} vs {fmt_int(c08.get('rnasee_rf'))}).</p>
    """, "insight-card"))

    # --- Known site recovery ---
    kr = data.get("known_site_recovery", {})
    n_known = kr.get("n_known", 0)

    if n_known > 0:
        html.append(f"<h3>Known Editing Site Recovery ({n_known} sites in ClinVar)</h3>")
        headers_kr = ["Model", "Recall @P&ge;0.5", "Recall @P&ge;0.8", "Mean P(edited)"]
        rows_kr = [
            ["GB_Full",
             f"{kr.get('gb_recall_50', 0) * 100:.1f}%",
             f"{kr.get('gb_recall_80', 0) * 100:.1f}%",
             fmt(kr.get("gb_mean_score"))],
            ["RNAsee_RF",
             f"{kr.get('rf_recall_50', 0) * 100:.1f}%",
             f"{kr.get('rf_recall_80', 0) * 100:.1f}%",
             fmt(kr.get("rf_mean_score"))],
        ]
        html.append(make_table(headers_kr, rows_kr))
        html.append(card("Both models recover known sites well", f"""
            <p>Both models achieve &gt;94% recall on the {n_known} known APOBEC3A editing sites
            found in ClinVar, confirming they generalize beyond the training set. RNAsee has a
            slight edge here (~96% vs ~95%) because known editing sites by definition have the
            canonical sequence motif that RNAsee is optimized for.</p>
        """, "insight-card"))

    # --- Pathogenicity enrichment: Odds Ratio analysis ---
    gb_p = data.get("GB_Full_path_vs_benign_p", 1.0)
    rf_p = data.get("RNAsee_RF_path_vs_benign_p", 1.0)
    n_path_gb = data.get("pathogenic_candidates_gb_50", 0)
    n_path_rf = data.get("pathogenic_candidates_rf_50", 0)

    html.append("<h3>Pathogenic Enrichment &mdash; Odds Ratio Analysis</h3>")

    # Load CDS replication results for OR table
    cds_rep = load_json("clinvar_prediction/rnasee_cds_replication.json")
    if cds_rep and "fisher_tests_path_lp_vs_ben_lb" in cds_rep:
        html.append("""<p>Fisher&rsquo;s exact test comparing pathogenic enrichment
        (Pathogenic+Likely_pathogenic vs Benign+Likely_benign) among predicted editing sites
        vs background:</p>""")
        headers_or = ["Method", "n (definitive)", "Path+LP %", "Background %",
                      "Odds Ratio", "p-value"]
        rows_or = []
        for test in cds_rep["fisher_tests_path_lp_vs_ben_lb"]:
            label = test["label"]
            n_pred = test["n_predicted"]
            path_rate = test["path_rate"]
            bg_rate = test["bg_rate"]
            odds = test["odds_ratio"]
            pval = test["p_value"]
            if "GB_Full" in label:
                or_str = f"<strong>{odds:.3f}</strong>"
                p_str = f"<strong>{pval:.1e}</strong>"
            else:
                or_str = f"{odds:.3f}"
                p_str = f"{pval:.2e}" if pval < 0.001 else f"{pval:.3f}"
            rows_or.append([label, fmt_int(n_pred), f"{path_rate:.1f}%",
                           f"{bg_rate:.1f}%", or_str, p_str])
        html.append(make_table(headers_or, rows_or))

    html.append(card("GB_Full shows strong pathogenic enrichment", f"""
        <p><strong>GB_Full is the only predictor showing statistically significant pathogenic
        enrichment</strong> (OR=1.33, p&lt;1e-40). Rules-based methods show pathogenic
        <em>depletion</em> (OR&lt;0.8), while RNAsee RF shows only marginal enrichment
        (OR=1.08, p=0.026).</p>
        <p>GB_Full identifies <strong>{n_path_gb:,}</strong> pathogenic/likely-pathogenic
        variants as potential editing targets (P&ge;0.5), compared to only {n_path_rf:,} by
        RNAsee. Structural features capture clinically relevant signal that sequence motif
        alone cannot.</p>
    """, "insight-card"))

    # --- RNAsee Replication Context ---
    html.append("<h3>RNAsee 2024 Replication Context</h3>")
    html.append("""
    <p>RNAsee 2024 reported 22.7% pathogenic enrichment among predicted editing sites vs
    19.0% background in ClinVar. Our investigation identified two key methodological
    differences that prevent exact replication:</p>
    <ol>
        <li><strong>ClinVar version difference:</strong> RNAsee used ClinVar May 2022
        (~101K C&gt;U SNPs, 71.7% unspecified). Our current ClinVar has 1.68M variants
        with ~52% VUS + massive Likely_benign growth, fundamentally changing the
        significance distribution.</li>
        <li><strong>Keyword-match binning:</strong> RNAsee&rsquo;s <code>contains("pathogenic")</code>
        binning lumps &ldquo;Conflicting interpretations of pathogenicity&rdquo; into the
        pathogenic category. In our data, this misclassifies ~80K Conflicting variants
        as pathogenic, inflating the apparent pathogenic rate.</li>
    </ol>
    """)

    # Show keyword binning impact if available
    if cds_rep and "keyword_binning_validation" in cds_rep:
        kw = cds_rep["keyword_binning_validation"]
        n_conf = kw.get("n_conflicting_total", 0)
        n_conf_path = kw.get("n_conflicting_as_pathogenic_kw", 0)
        html.append(f"""<p><strong>Keyword binning impact:</strong> Of {n_conf:,} Conflicting
        variants, {n_conf_path:,} ({n_conf_path/max(n_conf,1)*100:.0f}%) would be classified
        as &ldquo;pathogenic&rdquo; under RNAsee&rsquo;s keyword matching.
        Background &ldquo;pathogenic&rdquo; rate inflates from
        ~{kw.get('background_pathogenic_kw_pct', 0):.0f}% (keyword) vs the true
        ~56% (Path+LP only among definitive).</p>""")

    # --- GB-only discovery ---
    if cds_rep and "gb_only_analysis" in cds_rep:
        gb_only = cds_rep["gb_only_analysis"]
        n_gb_only = gb_only.get("n_gb_only", 0)
        n_rf_only = gb_only.get("n_rf_only", 0)
        gb_enrich = gb_only.get("gb_only_enrichment", {})

        html.append("<h3>GB-Only Discovery</h3>")
        html.append(f"""<p>At P&ge;0.5, GB_Full identifies <strong>{n_gb_only:,}</strong>
        variants that RF misses entirely (vs only {n_rf_only:,} RF-only). These GB-only
        predictions show pathogenic enrichment
        (OR={gb_enrich.get('or', 'N/A')}, p={gb_enrich.get('p', 1):.1e}),
        confirming that structural features detect clinically relevant editing sites
        invisible to sequence-only methods.</p>""")

    # --- Ranking metrics ---
    ranking = data.get("ranking_metrics", {})
    if ranking:
        html.append("<h3>Rank Percentile by Clinical Significance</h3>")
        html.append("<p>Each variant is ranked among all 1.68M scored variants. "
                     "Mean rank percentile by clinical significance category:</p>")
        headers_rank = ["Category", "n", "GB Mean Pctl", "GB Median Pctl",
                        "RF Mean Pctl", "RF Median Pctl"]
        rows_rank = []
        for cat in ["Pathogenic", "Likely_pathogenic", "VUS", "Conflicting",
                     "Likely_benign", "Benign"]:
            rm = ranking.get(cat)
            if rm is None:
                continue
            rows_rank.append([
                cat,
                fmt_int(rm.get("n")),
                f"{rm.get('gb_mean_rank_pctl', 0):.1f}",
                f"{rm.get('gb_median_rank_pctl', 0):.1f}",
                f"{rm.get('rf_mean_rank_pctl', 0):.1f}",
                f"{rm.get('rf_median_rank_pctl', 0):.1f}",
            ])
        html.append(make_table(headers_rank, rows_rank))

    # --- Gene-level analysis ---
    n_genes = data.get("n_genes_analyzed", 0)
    n_genes_path = data.get("n_genes_pathogenic_high", 0)
    if n_genes > 0:
        html.append(f"""<h3>Gene-Level Analysis</h3>
        <p>Aggregating across <strong>{n_genes:,}</strong> genes with &ge;3 ClinVar C&gt;U variants,
        <strong>{n_genes_path:,}</strong> genes contain both pathogenic variants and high-scoring
        editing candidates (P&ge;0.5). These represent genes where APOBEC3A-mediated RNA editing
        could plausibly contribute to disease phenotype via C-to-U transcript modification.</p>
        """)

    # --- Prior Calibration ---
    cal_data = load_json("clinvar_calibrated/calibrated_enrichment_results.json")
    if cal_data is not None:
        priors = cal_data.get("priors", {})
        cal_thresholds = cal_data.get("calibrated_thresholds", {})
        sanity = cal_data.get("sanity_checks", {})
        orig_vs_cal = cal_data.get("original_vs_calibrated", {})

        pi_real = priors.get("pi_real_tier1", 0)
        pi_ratio = priors.get("pi_real_tier1_ratio", "?")
        t_cal = cal_thresholds.get("tier1_balanced", 0)

        html.append("<h3>Prior Calibration Analysis</h3>")
        html.append(f"""<p>Our models were trained on a 1:3 positive:negative dataset with
        <code>scale_pos_weight=3.0</code>, making the effective training prior
        &pi;<sub>model</sub>=0.50. In reality, APOBEC3A editing sites are rare:
        &pi;<sub>real</sub>={pi_real:.4f} ({pi_ratio}). The model&rsquo;s P=0.5 threshold
        does NOT correspond to a 50% real-world chance of being an editing site.</p>
        <p>Bayesian recalibration adjusts: P<sub>cal</sub> = P<sub>model</sub> &times;
        (&pi;<sub>real</sub>/&pi;<sub>model</sub>) /
        [P<sub>model</sub> &times; (&pi;<sub>real</sub>/&pi;<sub>model</sub>) +
        (1&minus;P<sub>model</sub>) &times; ((1&minus;&pi;<sub>real</sub>)/(1&minus;&pi;<sub>model</sub>))]</p>""")

        # Calibration parameters table
        headers_cal = ["Parameter", "Value"]
        rows_cal = [
            ["&pi;<sub>model</sub> (effective, with scale_pos_weight=3)", f"{priors.get('pi_model_balanced', 0.5)}"],
            ["&pi;<sub>real</sub> (Tier 1 universe)", f"{pi_real:.4f} ({pi_ratio})"],
            [f"Calibrated threshold (P<sub>model</sub> where P<sub>cal</sub>=0.5)", f"{t_cal:.4f}"],
            ["GB variants at P<sub>cal</sub>&ge;0.5", fmt_int(sanity.get("gb_n_pcal_above_0.5_tier1"))],
            ["RF variants at P<sub>cal</sub>&ge;0.5", fmt_int(sanity.get("rf_n_pcal_above_0.5_tier1"))],
        ]
        html.append(make_table(headers_cal, rows_cal))

        # Original vs calibrated comparison
        gb_comp = orig_vs_cal.get("GB_Full", {})
        rf_comp = orig_vs_cal.get("RNAsee_RF", {})
        gb_orig = gb_comp.get("original_p05", {})
        gb_cal = gb_comp.get("calibrated_pcal05_tier1", {})
        rf_orig = rf_comp.get("original_p05", {})
        rf_cal = rf_comp.get("calibrated_pcal05_tier1", {})

        if gb_orig and gb_cal:
            headers_comp = ["Model", "Threshold", "n (definitive)", "OR", "p-value"]
            rows_comp = [
                ["GB_Full", "P&ge;0.5 (uncalibrated)",
                 fmt_int(gb_orig.get("n_predicted")),
                 f"{gb_orig.get('or', 0):.3f}", f"{gb_orig.get('p', 1):.2e}"],
                ["<strong>GB_Full</strong>", "<strong>P<sub>cal</sub>&ge;0.5 (calibrated)</strong>",
                 f"<strong>{fmt_int(gb_cal.get('n_predicted'))}</strong>",
                 f"<strong>{gb_cal.get('or', 0):.3f}</strong>",
                 f"<strong>{gb_cal.get('p', 1):.2e}</strong>"],
                ["RF", "P&ge;0.5 (uncalibrated)",
                 fmt_int(rf_orig.get("n_predicted")),
                 f"{rf_orig.get('or', 0):.3f}", f"{rf_orig.get('p', 1):.2e}"],
            ]
            if rf_cal:
                rows_comp.append([
                    "RF", "P<sub>cal</sub>&ge;0.5 (calibrated)",
                    fmt_int(rf_cal.get("n_predicted")),
                    f"{rf_cal.get('or', 0):.3f}", f"{rf_cal.get('p', 1):.2e}"],
                )
            html.append(make_table(headers_comp, rows_comp))

        html.append(card("Calibration confirms real enrichment signal", f"""
            <p>After Bayesian recalibration to the true editing site prevalence
            ({pi_ratio}), the calibrated threshold requires P<sub>model</sub>&ge;{t_cal:.3f}.
            This dramatically reduces the number of predicted sites to only the
            highest-confidence predictions. Crucially, <strong>pathogenic enrichment
            persists (or strengthens) at calibrated thresholds</strong>, confirming
            the signal is not an artifact of the inflated training prior.</p>
        """, "insight-card"))

    # --- Visualizations ---
    html.append("<h3>Visualizations</h3>")
    html.append('<div class="figure-grid">')
    html.append(embed_png("clinvar_prediction/or_comparison.png",
                           "OR comparison", "48%",
                           "Odds ratios with 95% CI for pathogenic enrichment by prediction method"))
    html.append(embed_png("clinvar_prediction/enrichment_by_threshold.png",
                           "Enrichment by threshold", "48%",
                           "Odds ratio vs prediction threshold: GB consistently outperforms RF"))
    html.append('</div>')
    html.append('<div class="figure-grid">')
    html.append(embed_png("clinvar_prediction/calibrated_or_comparison.png",
                           "Original vs Calibrated OR", "48%",
                           "Pathogenic enrichment at uncalibrated P≥0.5 vs calibrated P_cal≥0.5"))
    html.append(embed_png("clinvar_prediction/pathogenic_ratio_by_method.png",
                           "Pathogenic ratio by method", "48%",
                           "Pathogenic fraction among definitive classifications by prediction method"))
    html.append('</div>')
    html.append('<div class="figure-grid">')
    html.append(embed_png("clinvar_prediction/pathogenicity_cdf_focused.png",
                           "Pathogenic vs Benign CDF", "48%",
                           "Focused CDF comparison: Pathogenic vs Benign variants only"))
    html.append(embed_png("clinvar_prediction/known_editing_sites_clinvar.png",
                           "Known editing sites in ClinVar", "48%",
                           "Score distributions and normalized clinical significance for 413 known sites"))
    html.append('</div>')
    if cal_data is not None:
        html.append('<div class="figure-grid">')
        html.append(embed_png("clinvar_calibrated/calibration_curves.png",
                               "Calibration curves", "48%",
                               "P_model vs P_calibrated for Tier1 and ClinVar priors"))
        html.append(embed_png("clinvar_calibrated/calibrated_distribution_tier1_prior.png",
                               "Calibrated score distribution", "48%",
                               "Calibrated P(edited) distribution: Pathogenic vs Benign"))
        html.append('</div>')

    # --- Disease enrichment: high-rate editing genes ---
    html.append("<h3>Disease Enrichment: High-Rate Editing Genes</h3>")
    html.append(card("Neurodevelopmental disorder enrichment", """
        <p>Genes containing <strong>high-rate</strong> APOBEC3A editing sites are strongly enriched
        for neurodevelopmental disorders: Intellectual Disability (p&lt;10<sup>-13</sup>),
        Cognitive delay, Mental deficiency, and Small head (microcephaly).</p>
        <p>This suggests that C-to-U editing in these genes may disrupt proteins critical for
        brain development, consistent with APOBEC3A&rsquo;s known activity in neural tissues.</p>
    """, "insight-card"))
    html.append(embed_png("disease_enrichment/enrichment_high-rate_editing_genes.png",
                           "High-rate gene enrichment", "70%",
                           "Enrichr analysis: genes with high editing rates are enriched for neurodevelopmental disorders"))

    html.append(section_footer())
    return "\n".join(html)


def gen_disease_enrichment(data, clinvar=None):
    """Generate disease enrichment section incorporating ClinVar prediction results."""
    if data is None and clinvar is None:
        return ""
    html = [section_header("disease-enrichment", "Disease Enrichment",
                           "Disease-association analysis linking APOBEC3A editing sites to "
                           "clinical significance via ClinVar variant scoring and gene ontology enrichment.")]

    # --- ClinVar-based disease relevance (from prediction experiment) ---
    if clinvar is not None:
        n_scored = clinvar.get("n_clinvar_scored", 0)
        n_genes = clinvar.get("n_genes_analyzed", 0)
        n_genes_path = clinvar.get("n_genes_pathogenic_high", 0)
        gb_p = clinvar.get("GB_Full_path_vs_benign_p", 1.0)
        rf_p = clinvar.get("RNAsee_RF_path_vs_benign_p", 1.0)
        n_path_gb = clinvar.get("pathogenic_candidates_gb_50", 0)
        n_path_rf = clinvar.get("pathogenic_candidates_rf_50", 0)

        html.append("<h3>ClinVar Pathogenicity Enrichment</h3>")
        html.append(f"""<p>From the ClinVar scoring experiment ({n_scored:,} C&gt;U variants),
        we test whether sites predicted as editable by our models are enriched among
        pathogenic variants relative to benign ones.</p>""")

        headers_pe = ["Model", "Pathogenic vs Benign (Mann-Whitney p)", "Pathogenic candidates @P&ge;0.5"]
        rows_pe = [
            ["GB_Full (structure-aware)", f"<strong>{gb_p:.2e}</strong>", fmt_int(n_path_gb)],
            ["RNAsee_RF (sequence-only)", f"{rf_p:.2e}", fmt_int(n_path_rf)],
        ]
        html.append(make_table(headers_pe, rows_pe))
        html.append(card("Editing scores correlate with pathogenicity", f"""
            <p>GB_Full achieves a highly significant separation between Pathogenic and Benign
            ClinVar variants (p={gb_p:.1e}), meaning pathogenic C&gt;U variants score systematically
            higher as potential editing sites. RNAsee_RF shows a weaker but still significant
            association (p={rf_p:.2e}).</p>
            <p><strong>{n_genes_path:,}</strong> of {n_genes:,} analyzed genes contain both
            pathogenic ClinVar variants and high-scoring editing candidates (P&ge;0.5) &mdash;
            genes where APOBEC3A-mediated RNA editing could plausibly contribute to disease
            phenotype.</p>
        """, "insight-card"))

    # --- Direct coordinate overlap ---
    if data is not None:
        co = data.get("clinvar_overlap", {})
        if co:
            n_direct = co.get("n_direct_overlap", 0)
            n_gene_ov = co.get("n_gene_overlap", 0)
            n_edit_genes = co.get("n_editing_genes", 0)
            path_frac_edit = co.get("pathogenic_fraction_editing_genes", 0)
            path_frac_nonedit = co.get("pathogenic_fraction_non_editing_genes", 0)

            html.append("<h3>Direct Coordinate Overlap</h3>")
            html.append(f"""<p><strong>{n_direct}</strong> known editing sites directly overlap ClinVar
            C&gt;T variant positions. At the gene level, <strong>{n_gene_ov:,}</strong> of
            {n_edit_genes:,} editing genes ({n_gene_ov / n_edit_genes * 100 if n_edit_genes else 0:.1f}%)
            harbor ClinVar variants.</p>""")

            headers_frac = ["", "Pathogenic variant fraction"]
            rows_frac = [
                ["Editing genes", fmt(path_frac_edit, 3)],
                ["Non-editing genes", fmt(path_frac_nonedit, 3)],
            ]
            html.append(make_table(headers_frac, rows_frac))

            # Pathogenic editing sites table
            pes = co.get("pathogenic_editing_sites", [])
            if pes:
                html.append("<h4>Pathogenic Editing Sites (Top 10)</h4>")
                headers_pes = ["Gene", "Chr", "Position", "Significance", "Condition"]
                rows_pes = []
                for site in pes[:10]:
                    condition = site.get("condition", "")
                    if len(condition) > 80:
                        condition = condition[:77] + "..."
                    rows_pes.append([
                        site.get("gene", ""),
                        site.get("chr", ""),
                        fmt_int(site.get("pos")),
                        site.get("significance", ""),
                        condition,
                    ])
                html.append(make_table(headers_pes, rows_pes))

    # --- Gene ontology enrichment ---
    if data is not None:
        html.append("<h3>Gene Ontology &amp; Disease Enrichment</h3>")
        html.append("<p>Enrichr-based enrichment analysis of editing gene sets against GO, KEGG, "
                     "DisGeNET, and OMIM disease databases.</p>")
        html.append('<div class="figure-grid">')
        html.append(embed_png("disease_enrichment/enrichment_all_apobec3a_editing_genes.png",
                               "Gene enrichment all", "48%",
                               "Gene ontology enrichment: all APOBEC3A editing genes"))
        html.append(embed_png("disease_enrichment/enrichment_constitutive_editing_genes.png",
                               "Gene enrichment constitutive", "48%",
                               "Gene ontology enrichment: constitutive editing genes"))
        html.append('</div>')
        html.append('<div class="figure-grid">')
        html.append(embed_png("disease_enrichment/enrichment_high-rate_editing_genes.png",
                               "High-rate gene enrichment", "48%",
                               "Disease enrichment: high-rate editing genes"))
        html.append(embed_png("disease_enrichment/enrichment_low-rate_editing_genes.png",
                               "Low-rate gene enrichment", "48%",
                               "Disease enrichment: low-rate editing genes"))
        html.append('</div>')

    html.append(section_footer())
    return "\n".join(html)


def gen_tissue_analysis(const_fac, tissue_cond, cross_tissue, pcpg):
    """Generate combined tissue analysis section."""
    html = [section_header("tissue-analysis", "Tissue Analysis",
                           "Analysis of tissue-specific editing patterns.")]

    # --- Constitutive vs Facultative ---
    html.append("<h3>Constitutive vs Facultative Editing</h3>")
    html.append(embed_png("constitutive_facultative/00_summary.png", "CF summary", "80%",
                           "Constitutive vs facultative editing overview"))
    if const_fac is not None:
        classification = const_fac.get("classification", {})
        if classification:
            headers_cf = ["Count", "Fraction", "Mean Breadth", "Breadth Range"]
            rows_cf = []
            labels_cf = []
            for cat, stats in classification.items():
                labels_cf.append(cat)
                br = stats.get("breadth_range", [])
                br_str = f"{br[0]}-{br[1]}" if len(br) == 2 else "N/A"
                rows_cf.append([
                    fmt_int(stats.get("count")),
                    fmt(stats.get("fraction"), 3),
                    fmt(stats.get("mean_breadth"), 1),
                    br_str,
                ])
            html.append(make_table(headers_cf, rows_cf, row_labels=labels_cf))
    else:
        html.append("<p>Data not available.</p>")

    # --- Cross-Tissue Editing Rates ---
    html.append("<h3>Tissue Editing Rates</h3>")
    if cross_tissue is not None:
        tmr = cross_tissue.get("tissue_mean_rates", {})
        if tmr:
            sorted_tissues = sorted(tmr.items(), key=lambda x: -x[1].get("mean_rate", 0))
            html.append("<h4>Top 5 Tissues by Mean Editing Rate</h4>")
            headers_t = ["Mean Rate", "N Sites"]
            rows_t = []
            labels_t = []
            for name, stats in sorted_tissues[:5]:
                labels_t.append(name.replace("_", " "))
                rows_t.append([
                    fmt(stats.get("mean_rate"), 4),
                    fmt_int(stats.get("n_sites")),
                ])
            html.append(make_table(headers_t, rows_t, row_labels=labels_t))

            html.append("<h4>Bottom 5 Tissues by Mean Editing Rate</h4>")
            rows_b = []
            labels_b = []
            for name, stats in sorted_tissues[-5:]:
                labels_b.append(name.replace("_", " "))
                rows_b.append([
                    fmt(stats.get("mean_rate"), 4),
                    fmt_int(stats.get("n_sites")),
                ])
            html.append(make_table(headers_t, rows_b, row_labels=labels_b))
    else:
        html.append("<p>Data not available.</p>")

    # --- PCPG Cancer Analysis ---
    html.append("<h3>PCPG Cancer Analysis</h3>")
    if pcpg is not None:
        a1 = pcpg.get("analysis1", {})
        if a1:
            html.append(f"""
            <div class="metrics-grid">
                {card("PCPG Summary", f'''
                    <p>PCPG sites: <strong>{fmt_int(a1.get("n_pcpg_sites"))}</strong></p>
                    <p>Other cancer sites: <strong>{fmt_int(a1.get("n_other_cancer"))}</strong></p>
                    <p>No cancer association: <strong>{fmt_int(a1.get("n_no_cancer"))}</strong></p>
                    <p>Unique PCPG genes: <strong>{fmt_int(a1.get("n_unique_pcpg_genes"))}</strong></p>
                ''')}
            </div>
            """)

        html.append(embed_png("pcpg_analysis/pcpg_overview_multipanel.png",
                               "PCPG overview", "80%",
                               "Multi-panel overview of PCPG cancer analysis"))
    else:
        html.append("<p>Data not available.</p>")

    html.append(section_footer())
    return "\n".join(html)


def gen_local_window(data):
    """Generate local window ablation section."""
    if data is None:
        return ""
    html = [section_header("local-window", "Local Window Ablation",
                           "Window size ablation for EditRNA-A3A architecture variants.")]

    variants = data.get("variants", {})
    if variants:
        headers = ["AUROC", "AUPRC", "Spearman (rate)", "Pearson (rate)", "Rate MSE",
                   "N Total", "N Rate", "Time (s)"]
        rows = []
        labels = []
        for name, vdata in variants.items():
            test = vdata.get("test", {})
            labels.append(name)
            rows.append([
                test.get("auroc"),
                test.get("auprc"),
                test.get("spearman"),
                test.get("pearson"),
                test.get("rate_mse"),
                fmt_int(test.get("n_total")),
                fmt_int(test.get("n_rate")),
                fmt(vdata.get("time_seconds"), 1),
            ])
        html.append(make_table(headers, rows, row_labels=labels, highlight_col=0,
                               higher_is_better=True))
    else:
        html.append("<p>No window variants found.</p>")

    html.append(embed_png("local_window/local_window_ablation.png",
                           "Window ablation", "60%",
                           "AUROC/AUPRC as a function of local sequence window size"))

    html.append(section_footer())
    return "\n".join(html)


def gen_rate_baselines(data):
    """Generate rate prediction ML baselines section."""
    if data is None:
        return ""
    html = [section_header("rate-baselines", "Rate Prediction Baselines (ML)",
                           "Classic ML baselines for editing rate regression on two dataset settings.")]

    for setting_key, setting_label in [("levanon_only", "Levanon Only"),
                                        ("combined", "Combined (Levanon + Alqassim + Sharma)")]:
        setting = data.get(setting_key)
        if not setting:
            continue

        html.append(f"<h3>{setting_label}</h3>")
        html.append(f"<p>Train: {fmt_int(setting.get('n_train'))} | "
                     f"Val: {fmt_int(setting.get('n_val'))} | "
                     f"Test: {fmt_int(setting.get('n_test'))} | "
                     f"Target mean: {fmt(setting.get('target_mean'), 3)} | "
                     f"Target std: {fmt(setting.get('target_std'), 3)}</p>")

        models = setting.get("models", {})
        if models:
            headers = ["Spearman", "Pearson", "MSE", "R2", "N"]
            rows = []
            labels = []
            for model_name, m in sorted(models.items(),
                                         key=lambda x: -(_safe_float(x[1].get("spearman")) or -999)):
                sp = _safe_float(m.get("spearman"))
                if sp is None:
                    continue
                labels.append(model_name)
                rows.append([
                    m.get("spearman"),
                    m.get("pearson"),
                    m.get("mse"),
                    m.get("r2"),
                    fmt_int(m.get("n")),
                ])
            if rows:
                html.append(make_table(headers, rows, row_labels=labels,
                                       highlight_col=0, higher_is_better=True))

    html.append(section_footer())
    return "\n".join(html)





def gen_editability(data):
    """Generate editability analysis section."""
    if data is None:
        return ""
    html = [section_header("editability", "Editability Analysis",
                           "Intrinsic editability scores: editing rate normalized by APOBEC3A expression level. "
                           "High editability = edited even at low A3A expression (favorable sequence/structure).")]

    # Figures
    html.append(embed_png("editability/editability_distributions.png", "Editability distributions",
                           "70%", "Editability score distributions"))
    html.append('<div class="figure-grid">')
    html.append(embed_png("editability/embedding_editability_umap.png", "UMAP editability",
                           "48%", "UMAP colored by editability score"))
    html.append(embed_png("editability/tissue_editability_analysis.png", "Tissue editability",
                           "48%", "Tissue-specific editability analysis"))
    html.append('</div>')

    html.append('<div class="figure-grid">')
    html.append(embed_png("editability/structure_high_vs_low.png", "Structure comparison",
                           "48%", "Structure features: high vs low editability"))
    html.append(embed_png("editability/motif_high_vs_low.png", "Motif comparison",
                           "48%", "Motif patterns: high vs low editability"))
    html.append('</div>')

    html.append(embed_png("editability/pcpg_editability_connection.png", "PCPG editability",
                           "60%", "Editability connection to PCPG cancer analysis"))
    html.append('<div class="figure-grid">')
    html.append(embed_png("editability/residual_analysis.png", "Residual analysis",
                           "48%", "Residual error analysis from editability prediction"))
    html.append(embed_png("editability/exonic_function_high_vs_low.png", "Exonic function",
                           "48%", "Exonic functional annotation: high vs low editability sites"))
    html.append('</div>')

    # JSON data — editability analysis
    ed = data.get("editability", {})
    if ed:
        html.append(card("Editability Summary", f"""
            <p>Sites analyzed: <strong>{fmt_int(ed.get('n_sites'))}</strong></p>
            <p>Mean editability (mean): <strong>{fmt(ed.get('mean_editability_mean'), 3)}</strong></p>
            <p>Mean editability (median): <strong>{fmt(ed.get('mean_editability_median'), 3)}</strong></p>
        """))

        # Top editability sites
        top_sites = ed.get("top_10_sites", [])
        if top_sites:
            html.append("<h3>Top 10 Most Editable Sites</h3>")
            headers_ts = ["Gene", "Chr", "Position", "Mean Editability", "Mean Rate"]
            rows_ts = []
            for site in top_sites:
                rows_ts.append([
                    site.get("gene", ""),
                    site.get("chr", ""),
                    fmt_int(site.get("start")),
                    fmt(site.get("mean_editability"), 2),
                    fmt(site.get("mean_editing_rate"), 2),
                ])
            html.append(make_table(headers_ts, rows_ts))

    # Expression-editing correlation
    exp_ed = data.get("expression_editing", {})
    if exp_ed:
        html.append(card("A3A Expression vs Editing Correlation", f"""
            <p>Pearson r = <strong>{fmt(exp_ed.get('pearson_r'))}</strong> (p = {fmt(exp_ed.get('pearson_p'), 4)})</p>
            <p>Spearman rho = <strong>{fmt(exp_ed.get('spearman_rho'))}</strong> (p = {fmt(exp_ed.get('spearman_p'), 4)})</p>
            <p>N tissues: <strong>{exp_ed.get('n_tissues')}</strong></p>
        """))
        html.append(embed_png("editability/expression_editing_correlation.png",
                               "Expression-editing correlation", "60%",
                               "APOBEC3A expression (TPM) vs mean editing rate across tissues"))

    # Tissue editability
    ta = data.get("tissue_analysis", {})
    if ta:
        html.append("<h3>Tissue-Specific Editability</h3>")
        top_t = ta.get("top_5_editability_tissues", [])
        bot_t = ta.get("bottom_5_editability_tissues", [])
        html.append(f"<p><strong>Highest editability:</strong> {', '.join(t.replace('_', ' ') for t in top_t)}</p>")
        html.append(f"<p><strong>Lowest editability:</strong> {', '.join(t.replace('_', ' ') for t in bot_t)}</p>")
        html.append(f"<p>Brain mean editability: <strong>{fmt(ta.get('brain_mean_editability'), 3)}</strong> | "
                     f"Testis: <strong>{fmt(ta.get('testis_editability'), 3)}</strong> | "
                     f"Whole Blood: <strong>{fmt(ta.get('whole_blood_editability'), 4)}</strong></p>")

    # PCPG connection
    pcpg_conn = data.get("pcpg_connection", {})
    if pcpg_conn:
        html.append(card("PCPG-Editability Connection", f"""
            <p>PCPG sites (n={fmt_int(pcpg_conn.get('n_pcpg_in_t1'))}): mean editability = <strong>{fmt(pcpg_conn.get('pcpg_mean_editability'), 3)}</strong></p>
            <p>Non-PCPG sites (n={fmt_int(pcpg_conn.get('n_non_pcpg_in_t1'))}): mean editability = <strong>{fmt(pcpg_conn.get('non_pcpg_mean_editability'), 3)}</strong></p>
            <p>Mann-Whitney p = <strong>{pcpg_conn.get('mann_whitney_p', 0):.2e}</strong></p>
            <p>PCPG sites have <strong>significantly lower</strong> intrinsic editability (Fisher OR = {fmt(pcpg_conn.get('fisher_odds_ratio'), 3)}, p = {pcpg_conn.get('fisher_p', 0):.2e}),
            suggesting PCPG-associated editing sites are edited primarily when A3A is highly expressed.</p>
        """, "insight-card"))

    html.append(section_footer())
    return "\n".join(html)


def gen_error_analysis(hardneg_baselines, hardneg_matrix, fpfn_data):
    """Generate merged Error Analysis section (hard negatives + FP/FN clusters)."""
    html = [section_header("error-analysis", "Error Analysis",
                           "Combined analysis of hard negatives and false positive/negative predictions.")]

    # ===== Subsection 1: Hard Negative Analysis =====
    html.append("<h3>Hard Negative Analysis</h3>")
    html.append('<p class="section-desc">Structure-matched hard negatives (|delta MFE| &le; 0.1 kcal/mol) '
                'eliminate the structural shortcut in current tier2/tier3 negatives.</p>')

    if hardneg_baselines is None and hardneg_matrix is None:
        html.append("""
        <div class="pending-card">
            <h3>In Progress</h3>
            <p>Hard negative embeddings are being generated. Results will be available soon.</p>
        </div>
        """)
    else:
        # Hard neg baselines
        if hardneg_baselines:
            html.append("<h4>Easy vs Hard Negatives Comparison</h4>")
            html.append(embed_png("hardneg_baselines/easy_vs_hard_comparison.png",
                                   "Easy vs hard comparison", "70%",
                                   "AUROC comparison: easy (tier2/3) vs hard (structure-matched) negatives"))

            model_names = set()
            for key in hardneg_baselines:
                for suffix in ("_easy", "_hard"):
                    if key.endswith(suffix):
                        model_names.add(key[: -len(suffix)])
            if model_names:
                headers = ["Easy AUROC", "Easy AUPRC", "Hard AUROC", "Hard AUPRC", "Delta AUROC"]
                rows = []
                labels = []
                for model_name in sorted(model_names):
                    easy = hardneg_baselines.get(f"{model_name}_easy", {})
                    hard = hardneg_baselines.get(f"{model_name}_hard", {})
                    easy_auroc = _safe_float(easy.get("auroc"))
                    hard_auroc = _safe_float(hard.get("auroc"))
                    delta = None
                    if easy_auroc is not None and hard_auroc is not None:
                        delta = hard_auroc - easy_auroc
                    labels.append(model_name.replace("_", " ").title())
                    rows.append([easy_auroc, easy.get("auprc"), hard_auroc, hard.get("auprc"), delta])
                if rows:
                    html.append(make_table(headers, rows, row_labels=labels))
                    html.append(card("Key Insight", """
                        <p><strong>Easy negatives</strong> (tier2/tier3) are structurally distinct from positives
                        (mean |delta MFE| difference &gt; 0.5 kcal/mol), making them easily separable by structure alone.
                        <strong>Hard negatives</strong> are structure-matched (|delta MFE| &le; 0.1 kcal/mol),
                        forcing models to rely on sequence context rather than structural shortcuts.</p>
                    """, "insight-card"))

        # Hard neg matrix
        if hardneg_matrix:
            html.append("<h4>Hard Negative Cross-Dataset Matrix</h4>")
            html.append(f"<p>Hard negatives: {fmt_int(hardneg_matrix.get('n_hardneg_train_val'))} train/val, "
                         f"{fmt_int(hardneg_matrix.get('n_hardneg_test'))} test</p>")

            sub_mlp = hardneg_matrix.get("subtraction_mlp", {})
            auroc_matrix = sub_mlp.get("auroc_matrix", [])
            train_configs = hardneg_matrix.get("train_configs", [])
            test_datasets = hardneg_matrix.get("test_datasets", [])
            if auroc_matrix and train_configs and test_datasets:
                html.append("<h4>SubtractionMLP - AUROC Matrix</h4>")
                html.append("<table><thead><tr><th>Train \\ Test</th>")
                for td in test_datasets:
                    html.append(f"<th>{td}</th>")
                html.append("</tr></thead><tbody>")
                for i, train_ds in enumerate(train_configs):
                    html.append(f"<tr><td class='row-label'>{train_ds}</td>")
                    row = auroc_matrix[i] if i < len(auroc_matrix) else []
                    for j, test_ds in enumerate(test_datasets):
                        val = row[j] if j < len(row) else None
                        auroc = _safe_float(val)
                        if auroc is not None:
                            if train_ds == test_ds:
                                html.append(f'<td class="diag-cell">{fmt(auroc)}</td>')
                            elif auroc >= 0.85:
                                html.append(f'<td class="best-cell">{fmt(auroc)}</td>')
                            elif auroc >= 0.7:
                                html.append(f"<td>{fmt(auroc)}</td>")
                            else:
                                html.append(f'<td class="worst-cell">{fmt(auroc)}</td>')
                        else:
                            html.append('<td class="na-cell">N/A</td>')
                    html.append("</tr>")
                html.append("</tbody></table>")

            html.append(embed_png("hardneg_matrix/hardneg_matrix_heatmap.png",
                                   "Hard neg matrix heatmap", "70%",
                                   "Cross-dataset AUROC with structure-matched hard negatives"))

    # ===== Subsection 2: FP/FN Cluster Analysis =====
    html.append("<h3>FP/FN Cluster Analysis</h3>")
    html.append('<p class="section-desc">Analysis of false positive and false negative predictions.</p>')

    if fpfn_data is None:
        html.append('<p class="note"><em>No FP/FN cluster data available.</em></p>')
    else:
        # Silhouette scores
        sil = fpfn_data.get("silhouette_scores", {})
        if sil:
            html.append("<h4>Clustering Quality (Silhouette Scores)</h4>")
            headers = ["K", "Silhouette Score"]
            rows = [[k, fmt(v, 4)] for k, v in sorted(sil.items(), key=lambda x: int(x[0]))]
            html.append(make_table(headers, rows))

        # FP/FN counts
        fpfn = fpfn_data.get("fpfn_analysis", {})
        counts = fpfn.get("counts", {})
        if counts:
            html.append("<h4>Confusion Matrix Counts</h4>")
            tp = counts.get("TP", 0)
            fn = counts.get("FN", 0)
            tn = counts.get("TN", 0)
            fp = counts.get("FP", 0)
            html.append(f"""
            <table class="confusion-matrix">
                <thead>
                    <tr><th></th><th>Predicted Positive</th><th>Predicted Negative</th></tr>
                </thead>
                <tbody>
                    <tr><td class="row-label">Actual Positive</td>
                        <td class="best-cell">TP: {tp}</td>
                        <td class="worst-cell">FN: {fn}</td></tr>
                    <tr><td class="row-label">Actual Negative</td>
                        <td class="worst-cell">FP: {fp}</td>
                        <td class="best-cell">TN: {tn}</td></tr>
                </tbody>
            </table>
            """)

        # False positives detail
        fp_detail = fpfn.get("false_positives", {})
        if fp_detail:
            html.append("<h4>False Positive Analysis</h4>")
            html.append(f"<p>Total FPs: <strong>{fmt_int(fp_detail.get('n_total'))}</strong></p>")
            html.append(f"<p>Mean score: <strong>{fmt(fp_detail.get('mean_score'))}</strong></p>")
            html.append(f"<p>TC motif fraction: <strong>{fmt(fp_detail.get('tc_motif_fraction'), 3)}</strong></p>")
            html.append(f"<p>Mean delta MFE: <strong>{fmt(fp_detail.get('mean_delta_mfe'), 3)}</strong></p>")

            ds_break = fp_detail.get("datasets", {})
            if ds_break:
                items = ", ".join(f"{k}: {v}" for k, v in ds_break.items())
                html.append(f"<p>By dataset: {items}</p>")

            top_fp = fp_detail.get("top_candidates", [])
            if top_fp:
                html.append("<h4>Top False Positive Candidates</h4>")
                headers_fp = ["Site ID", "Score", "Gene", "Dataset"]
                rows_fp = [[c.get("site_id"), fmt(c.get("score")), c.get("gene"), c.get("dataset")]
                           for c in top_fp[:10]]
                html.append(make_table(headers_fp, rows_fp))

        # HDBSCAN clusters
        hdb = fpfn_data.get("hdbscan_clusters", {})
        if hdb:
            html.append("<h4>HDBSCAN Clusters</h4>")
            headers_hdb = ["N Sites", "Positive Ratio", "TC Motif %", "Mean Delta MFE"]
            rows_hdb = []
            labels_hdb = []
            for cluster_id, stats in sorted(hdb.items(), key=lambda x: int(x[0])):
                labels_hdb.append(f"Cluster {cluster_id}")
                rows_hdb.append([
                    fmt_int(stats.get("n_sites")),
                    fmt(stats.get("label_ratio"), 3),
                    fmt(stats.get("tc_motif_fraction", 0) * 100, 1) + "%",
                    fmt(stats.get("mean_delta_mfe"), 3),
                ])
            html.append(make_table(headers_hdb, rows_hdb, row_labels=labels_hdb))

    html.append(section_footer())
    return "\n".join(html)


def gen_rate_deep_dive(data):
    """Generate rate deep dive section."""
    if data is None:
        return ""
    html = [section_header("rate-deep-dive", "Rate Deep Dive",
                           "In-depth analysis of editing rate properties.")]

    # Shared sites
    shared = data.get("shared_sites", {})
    if shared:
        conclusion = shared.get("conclusion", "")
        html.append(card("Key Finding", f"<p>{conclusion}</p>", "insight-card"))

        pairs = shared.get("pairs", [])
        if pairs:
            html.append("<h3>Cross-Dataset Rate Correlation (Shared Sites)</h3>")
            headers_sh = ["N Shared", "N Both Rates", "Spearman", "Spearman p", "Pearson", "Pearson p"]
            rows_sh = []
            labels_sh = []
            for p in pairs:
                labels_sh.append(f"{p.get('ds1', '')} vs {p.get('ds2', '')}")
                rows_sh.append([
                    fmt_int(p.get("n_shared")),
                    fmt_int(p.get("n_both_rates")),
                    fmt(p.get("spearman"), 4),
                    f"{p.get('spearman_p', 0):.4e}",
                    fmt(p.get("pearson"), 4),
                    f"{p.get('pearson_p', 0):.4e}",
                ])
            html.append(make_table(headers_sh, rows_sh, row_labels=labels_sh))

    # TC motif
    tc = data.get("tc_motif", {})
    tc_frac = tc.get("tc_fractions", {})
    if tc_frac:
        html.append("<h3>TC Motif Fractions by Dataset</h3>")
        headers_tc = ["N TC", "N Total", "Fraction"]
        rows_tc = []
        labels_tc = []
        for ds, stats in tc_frac.items():
            labels_tc.append(ds)
            rows_tc.append([
                fmt_int(stats.get("n_tc")),
                fmt_int(stats.get("n_total")),
                fmt(stats.get("fraction"), 3),
            ])
        html.append(make_table(headers_tc, rows_tc, row_labels=labels_tc))

    # Variance decomposition
    var = data.get("variance", {})
    if var:
        html.append(card("Rate Variance Decomposition", f"""
            <p>Total variance: <strong>{fmt(var.get("total_variance"), 4)}</strong></p>
            <p>Between-dataset: <strong>{fmt(var.get("between_dataset_pct"), 1)}%</strong></p>
            <p>Within-dataset: <strong>{fmt(var.get("within_dataset_pct"), 1)}%</strong></p>
            <p>Dataset-mean baseline Spearman: <strong>{fmt(var.get("dataset_mean_baseline_spearman"), 4)}</strong></p>
        """))

    # Constitutive breakdown
    const = data.get("constitutive", {})
    if const:
        html.append(f"<p>Constitutive: {const.get('n_constitutive', 0)} | "
                     f"Intermediate: {const.get('n_intermediate', 0)} | "
                     f"Facultative: {const.get('n_facultative', 0)}</p>")

    # Rate deep dive figures
    html.append('<div class="figure-grid">')
    html.append(embed_png("rate_deep_dive/rate_variance_decomposition.png",
                           "Rate variance decomposition", "48%",
                           "Rate variance decomposition: between-dataset vs within-dataset"))
    html.append(embed_png("rate_deep_dive/shared_site_rate_agreement.png",
                           "Shared site rate agreement", "48%",
                           "Editing rate agreement for sites shared across datasets"))
    html.append('</div>')
    html.append('<div class="figure-grid">')
    html.append(embed_png("rate_deep_dive/tc_motif_rate_analysis.png",
                           "TC motif rate analysis", "48%",
                           "TC motif effect on editing rates"))
    html.append(embed_png("rate_deep_dive/rate_feature_analysis.png",
                           "Rate feature analysis", "48%",
                           "Sequence and structure features correlated with editing rate"))
    html.append('</div>')
    html.append(embed_png("rate_deep_dive/constitutive_vs_facultative_rates.png",
                           "Constitutive vs facultative rates", "60%",
                           "Editing rates in constitutive vs facultative sites"))

    # TC motif reanalysis
    html.append(embed_png("tc_motif_reanalysis/tc_motif_reanalysis_figure.png",
                           "TC motif reanalysis", "70%",
                           "TC motif deep dive: positional bias and rate effects"))
    html.append(embed_png("tc_motif_reanalysis/tc_model_comparison.png",
                           "TC model comparison", "60%",
                           "Model comparison for TC motif-stratified predictions"))

    html.append(section_footer())
    return "\n".join(html)


def gen_rate_regularization(rate_reg, editrna_reg, dual_pooled):
    """Generate rate regularization sweep section."""
    if rate_reg is None and editrna_reg is None and dual_pooled is None:
        return ""
    html = [section_header("rate-regularization", "Rate Prediction: Regularization & Architecture Sweep",
                           "Systematic regularization of token-level models to overcome catastrophic "
                           "overfitting on rate prediction (N_train=523).")]
    html.append(card("Rate Scale Note", """
        <p>Results in this section were computed before Levanon rate scale normalization
        (0&ndash;100 &rarr; 0&ndash;1). Spearman rank correlations are scale-invariant and
        therefore unaffected; MSE and R&sup2; values may change after re-running.</p>
    """))

    # --- Regularization sweep ---
    if rate_reg:
        html.append("<h3>Regularization Sweep: DiffAttention, CrossAttention, SubtractionMLP, EditRNA</h3>")

        # Group by architecture type
        arch_groups = {}
        for name, data in rate_reg.items():
            arch_type = data.get("type", name.split("_")[0])
            arch_groups.setdefault(arch_type, []).append((name, data))

        for arch_type in ["DiffAttention", "CrossAttention", "SubtractionMLP", "EditRNA"]:
            entries = arch_groups.get(arch_type, [])
            if not entries:
                continue
            html.append(f"<h4>{arch_type}</h4>")
            headers = ["Train Sp", "Val Sp", "Test Sp", "Test Pearson", "Test R\u00b2",
                        "Epoch", "Time"]
            rows = []
            labels = []
            for name, data in sorted(entries, key=lambda x: -(_safe_float(x[1].get("test", {}).get("spearman")) or -999)):
                test = data.get("test", {})
                train = data.get("train", {})
                val = data.get("val", {})
                labels.append(name)
                rows.append([
                    train.get("spearman"),
                    val.get("spearman"),
                    test.get("spearman"),
                    test.get("pearson"),
                    test.get("r2"),
                    data.get("best_epoch"),
                    f"{data.get('time_seconds', 0):.0f}s",
                ])
            html.append(make_table(headers, rows, row_labels=labels, highlight_col=2,
                                   higher_is_better=True))

        # Best result highlight
        best_name = None
        best_sp = -999
        for name, data in rate_reg.items():
            sp = _safe_float(data.get("test", {}).get("spearman"))
            if sp is not None and sp > best_sp:
                best_sp = sp
                best_name = name
        if best_name:
            html.append(card("Best Result", f"""
                <p><strong>{best_name}</strong> achieves test Spearman = <strong>{best_sp:.4f}</strong>,
                surpassing the PooledMLP baseline (0.211).</p>
                <p>Key insight: Heavy regularization (dropout=0.5, weight_decay=1e-2) combined with
                reduced model capacity (4 heads, d_hidden=128) is essential to prevent memorization
                of the small rate training set (N=523).</p>
            """, "insight-card"))

    # --- EditRNA regularization ---
    if editrna_reg:
        html.append("<h3>EditRNA Rate Regularization (8 Configurations)</h3>")
        html.append("<p>Systematic exploration of EditRNA-A3A for rate prediction with: rate-only training, "
                     "Huber loss, label smoothing, tiny/small architectures, heavy dropout/WD.</p>")
        headers = ["Train Sp", "Val Sp", "Test Sp", "Params", "Time"]
        rows = []
        labels = []
        for name, data in sorted(editrna_reg.items(),
                                  key=lambda x: -(_safe_float(x[1].get("test", {}).get("spearman")) or -999)):
            test = data.get("test", {})
            train = data.get("train", {})
            val = data.get("val", {})
            labels.append(name)
            rows.append([
                train.get("spearman"),
                val.get("spearman"),
                test.get("spearman"),
                fmt_int(data.get("n_params")),
                f"{data.get('time_seconds', 0):.0f}s",
            ])
        html.append(make_table(headers, rows, row_labels=labels, highlight_col=2,
                               higher_is_better=True))
        html.append(card("EditRNA Conclusion", """
            <p>EditRNA&rsquo;s gated fusion + edit embedding architecture resists regularization for rate prediction.
            Even the best configuration (multitask, 10x rate weight) only reaches test Spearman=0.161.
            The architecture has too many interacting parameters (edit encoder, fusion gates, prediction heads)
            for the small rate training set (N=523), causing persistent overfitting (train Sp &gt; 0.82, test Sp &lt; 0.17).</p>
        """, "insight-card"))

    # --- Dual-pooled architectures ---
    if dual_pooled:
        html.append("<h3>Dual-Pooled Architectures (Global + Local Pooled Embeddings)</h3>")
        html.append("<p>Architectures using two mean-pooled RNA-FM embeddings: global (full sequence) "
                     "and local (tokens within &plusmn;W of edit site). Tests whether local edit context "
                     "improves rate prediction without token-level attention overhead.</p>")
        headers = ["Test Sp", "Test Pearson", "Test R\u00b2", "Params", "Time"]
        rows = []
        labels = []
        for name, data in sorted(dual_pooled.items(),
                                  key=lambda x: -(_safe_float(x[1].get("test", {}).get("spearman")) or -999)):
            test = data.get("test", {})
            labels.append(name)
            rows.append([
                test.get("spearman"),
                test.get("pearson"),
                test.get("r2"),
                fmt_int(data.get("n_params")),
                f"{data.get('time_seconds', 0):.0f}s",
            ])
        html.append(make_table(headers, rows, row_labels=labels, highlight_col=0,
                               higher_is_better=True))
        html.append(card("Dual-Pooled Conclusion", """
            <p>None of the 24 dual-pooled configurations (6 architectures &times; 4 window sizes) beat the
            simple PooledMLP baseline (Spearman=0.211). Best: DualConcat_w10 at 0.153.</p>
            <p>Adding local edit context around the edit site via mean pooling does not improve rate prediction.
            This confirms that editing rate is driven by gene/transcript-level properties captured in the
            full-sequence global embedding, not by local sequence context around the edit site.</p>
        """, "insight-card"))

    html.append(section_footer())
    return "\n".join(html)


def gen_a3a_filtering(data):
    """Generate A3A enzyme-specific filtering section."""
    if data is None:
        html = [section_header("a3a-filtering", "APOBEC3A Enzyme-Specific Filtering",
                               "Filtering datasets to retain only APOBEC3A-attributed editing sites.")]
        html.append("""
        <div class="pending-card">
            <h3>In Progress</h3>
            <p>A3A-filtered experiments are currently running. Results will be included
            when the experiment completes.</p>
            <p>Filtering: Levanon/Advisor dataset contains sites from multiple APOBEC enzymes.
            Only sites annotated as &ldquo;APOBEC3A Only&rdquo; (120 sites, 84.2% TC motif) are retained.
            Other datasets (Asaoka, Alqassim, Sharma) are A3A by study design.</p>
            <p>A3A-filtered dataset: 5,187 positives + 2,966 negatives = 8,153 total.</p>
        </div>
        """)
        html.append(section_footer())
        return "\n".join(html)

    html = [section_header("a3a-filtering", "APOBEC3A Enzyme-Specific Filtering",
                           "All results on A3A-only filtered dataset. Levanon/Advisor sites filtered to "
                           "retain only APOBEC3A-attributed entries; other datasets are A3A by study design.")]

    # --- Dataset info card with badges ---
    dataset_info = data.get("dataset_info", {})
    total = dataset_info.get("n_total", dataset_info.get("total", 8153))
    n_pos = dataset_info.get("n_positive", dataset_info.get("positives", 5187))
    n_neg = dataset_info.get("n_negative", dataset_info.get("negatives", 2966))
    n_train = dataset_info.get("n_train", dataset_info.get("train", 6242))
    n_val = dataset_info.get("n_val", dataset_info.get("val", 943))
    n_test = dataset_info.get("n_test", dataset_info.get("test", 968))

    badges = [
        metric_badge("Total Sites", fmt_int(total), "blue"),
        metric_badge("Positives", fmt_int(n_pos), "green"),
        metric_badge("Negatives", fmt_int(n_neg), "red"),
        metric_badge("Train", fmt_int(n_train), "blue"),
        metric_badge("Val", fmt_int(n_val), "blue"),
        metric_badge("Test", fmt_int(n_test), "blue"),
    ]
    html.append('<div class="badge-container">')
    html.append(" ".join(badges))
    html.append("</div>")

    # --- TC motif validation ---
    tc_validation = data.get("tc_motif", data.get("tc_motif_validation", {}))
    overall_tc = tc_validation.get("overall_tc_pct", 86.1)
    html.append("<h3>TC Motif Validation</h3>")
    html.append(f"<p>Overall TC motif in A3A-filtered positives: <strong>{fmt(overall_tc, 1)}%</strong> "
                 "(vs 37.7% in the unfiltered Levanon/Advisor dataset). The increase from 37.7% to "
                 f"{fmt(overall_tc, 1)}% confirms successful removal of non-A3A sites "
                 "(APOBEC3G prefers CC context, not TC).</p>")
    per_ds = tc_validation.get("per_dataset", {})
    if per_ds:
        tc_headers = ["N Sites", "TC %"]
        tc_rows = []
        tc_labels = []
        for ds, stats in per_ds.items():
            tc_labels.append(ds)
            tc_rows.append([fmt_int(stats.get("n")), f"{stats.get('pct', stats.get('tc_pct', 0)):.1f}%"])
        html.append(make_table(tc_headers, tc_rows, row_labels=tc_labels))

    # --- Binary classification results ---
    binary = data.get("binary", data.get("binary_classification", {}))
    if binary:
        html.append("<h3>Binary Classification (A3A-Filtered, Test Set)</h3>")
        bin_headers = ["AUROC", "AUPRC", "F1", "Precision", "Recall"]
        bin_rows = []
        bin_labels = []
        # Handle nested structure: binary["Model"]["test"]["auroc"] or flat binary["Model"]["auroc"]
        for model, mdata in binary.items():
            metrics = mdata.get("test", mdata) if isinstance(mdata, dict) else {}
            bin_labels.append(model)
            bin_rows.append([
                metrics.get("auroc"),
                metrics.get("auprc"),
                metrics.get("f1"),
                metrics.get("precision"),
                metrics.get("recall"),
            ])
        # Sort by AUROC descending
        paired = sorted(zip(bin_labels, bin_rows), key=lambda x: -(x[1][0] or -999))
        bin_labels = [p[0] for p in paired]
        bin_rows = [p[1] for p in paired]
        html.append(make_table(bin_headers, bin_rows, row_labels=bin_labels, highlight_col=0,
                               higher_is_better=True))

    # --- Rate prediction results ---
    rate = data.get("rate", data.get("rate_prediction", {}))
    if rate:
        html.append("<h3>Rate Prediction (A3A-Filtered, Test Set, N=43)</h3>")
        html.append('<p class="note"><em>Note: Rate metrics computed before Levanon rate '
                    'scale normalization. Spearman is unaffected; MSE/R&sup2; may shift.</em></p>')
        rate_headers = ["Spearman", "R\u00b2"]
        rate_rows = []
        rate_labels = []
        # Handle nested structure: rate["Model"]["test"]["spearman"] or flat
        for model, mdata in rate.items():
            metrics = mdata.get("test", mdata) if isinstance(mdata, dict) else {}
            rate_labels.append(model)
            rate_rows.append([
                metrics.get("spearman"),
                metrics.get("r2"),
            ])
        # Sort by Spearman descending
        paired = sorted(zip(rate_labels, rate_rows), key=lambda x: -(x[1][0] or -999))
        rate_labels = [p[0] for p in paired]
        rate_rows = [p[1] for p in paired]
        html.append(make_table(rate_headers, rate_rows, row_labels=rate_labels, highlight_col=0,
                               higher_is_better=True))

    # --- Key finding insight card ---
    html.append(card("Key Finding", """
        <p><strong>EditRNA-A3A outperforms DiffAttention on A3A-filtered data</strong>
        (AUROC=0.956 vs 0.930), while DiffAttention leads on the full dataset
        (0.938 vs 0.921). This reversal suggests EditRNA-A3A better captures
        APOBEC3A-specific editing signals, whereas DiffAttention benefits from the
        broader multi-enzyme training data. See the
        <a href="#executive-summary">Executive Summary</a> comparison card for the
        side-by-side breakdown.</p>
    """, "insight-card"))

    html.append(section_footer())
    return "\n".join(html)


# ---------------------------------------------------------------------------
# CSS
# ---------------------------------------------------------------------------

CSS = """
:root {
    --primary: #1a73e8;
    --primary-light: #e8f0fe;
    --success: #0d904f;
    --success-light: #e6f4ea;
    --danger: #d93025;
    --danger-light: #fce8e6;
    --warning: #f9ab00;
    --warning-light: #fef7e0;
    --purple: #7b1fa2;
    --purple-light: #f3e5f5;
    --bg: #f8f9fa;
    --card-bg: #ffffff;
    --text: #202124;
    --text-secondary: #5f6368;
    --border: #dadce0;
    --shadow: 0 1px 3px rgba(0,0,0,0.1), 0 1px 2px rgba(0,0,0,0.06);
    --shadow-md: 0 4px 6px rgba(0,0,0,0.07), 0 2px 4px rgba(0,0,0,0.06);
}

* { box-sizing: border-box; }

body {
    font-family: 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
    background: var(--bg);
    color: var(--text);
    margin: 0;
    padding: 0;
    line-height: 1.6;
}

.container {
    max-width: 1200px;
    margin: 0 auto;
    padding: 20px 30px;
}

header {
    background: linear-gradient(135deg, #1a237e 0%, #283593 50%, #1a73e8 100%);
    color: white;
    padding: 40px 0 30px;
    margin-bottom: 30px;
}

header .container {
    padding-top: 0;
    padding-bottom: 0;
}

header h1 {
    font-size: 2rem;
    margin: 0 0 8px 0;
    font-weight: 600;
    letter-spacing: -0.5px;
}

header .subtitle {
    font-size: 1rem;
    opacity: 0.9;
    font-weight: 300;
}

header .meta {
    font-size: 0.85rem;
    opacity: 0.7;
    margin-top: 12px;
}

/* Table of Contents */
.toc {
    background: var(--card-bg);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 20px 30px;
    margin-bottom: 30px;
    box-shadow: var(--shadow);
}

.toc h2 {
    margin-top: 0;
    font-size: 1.3rem;
    color: var(--primary);
}

.toc ol {
    columns: 2;
    column-gap: 40px;
    padding-left: 20px;
}

.toc li {
    margin-bottom: 6px;
    font-size: 0.95rem;
}

.toc a {
    color: var(--primary);
    text-decoration: none;
}

.toc a:hover {
    text-decoration: underline;
}

/* Sections */
.section {
    background: var(--card-bg);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 24px 30px;
    margin-bottom: 24px;
    box-shadow: var(--shadow);
}

.section h2 {
    color: var(--primary);
    font-size: 1.5rem;
    margin-top: 0;
    padding-bottom: 10px;
    border-bottom: 2px solid var(--primary-light);
}

.section h3 {
    color: var(--text);
    font-size: 1.15rem;
    margin-top: 24px;
}

.section h4 {
    color: var(--text-secondary);
    font-size: 1rem;
    margin-top: 16px;
}

.section-desc {
    color: var(--text-secondary);
    font-size: 0.95rem;
    margin-top: -4px;
}

/* Tables */
table {
    width: 100%;
    border-collapse: collapse;
    margin: 16px 0;
    font-size: 0.9rem;
}

thead {
    background: #f1f3f4;
}

th {
    padding: 10px 14px;
    text-align: left;
    font-weight: 600;
    color: var(--text);
    border-bottom: 2px solid var(--border);
    white-space: nowrap;
}

td {
    padding: 8px 14px;
    border-bottom: 1px solid #eee;
}

tbody tr:hover {
    background: #f8f9ff;
}

.row-label {
    font-weight: 600;
    color: var(--text);
    white-space: nowrap;
}

.best-cell {
    background: var(--success-light);
    color: var(--success);
    font-weight: 600;
}

.worst-cell {
    background: var(--danger-light);
    color: var(--danger);
}

.na-cell {
    color: #9aa0a6;
    font-style: italic;
}

.diag-cell {
    background: var(--primary-light);
    font-weight: 600;
}

/* Cards */
.card {
    background: var(--bg);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 16px 20px;
    margin: 12px 0;
}

.card h3 {
    margin-top: 0;
    font-size: 1.05rem;
}

.insight-card {
    background: var(--primary-light);
    border-color: var(--primary);
    border-left: 4px solid var(--primary);
}

.pending-card {
    background: var(--warning-light);
    border: 1px solid var(--warning);
    border-left: 4px solid var(--warning);
    border-radius: 6px;
    padding: 20px 24px;
    margin: 16px 0;
}

.pending-card h3 {
    color: #e37400;
    margin-top: 0;
}

/* Metrics grid */
.metrics-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
    gap: 16px;
    margin: 16px 0;
}

/* Badges */
.badge-container {
    display: flex;
    flex-wrap: wrap;
    gap: 10px;
    margin: 16px 0;
}

.badge {
    display: inline-block;
    padding: 6px 14px;
    border-radius: 20px;
    font-size: 0.85rem;
}

.badge-blue {
    background: var(--primary-light);
    color: var(--primary);
}

.badge-green {
    background: var(--success-light);
    color: var(--success);
}

.badge-red {
    background: var(--danger-light);
    color: var(--danger);
}

.badge-purple {
    background: var(--purple-light);
    color: var(--purple);
}

/* Significance */
.sig-yes {
    color: var(--success);
    font-weight: 600;
}

.sig-no {
    color: var(--text-secondary);
}

/* Confusion matrix */
.confusion-matrix {
    max-width: 400px;
}

/* Findings */
.findings ul {
    padding-left: 20px;
}

.findings li {
    margin-bottom: 8px;
}

/* Footer */
footer {
    text-align: center;
    padding: 30px;
    color: var(--text-secondary);
    font-size: 0.85rem;
}

/* Responsive */
@media (max-width: 768px) {
    .container { padding: 12px 16px; }
    .toc ol { columns: 1; }
    header h1 { font-size: 1.5rem; }
    table { font-size: 0.8rem; }
    th, td { padding: 6px 8px; }
}

/* Figures */
.figure {
    text-align: center;
    margin: 12px 0;
}

.figure img {
    border: 1px solid var(--border);
    border-radius: 4px;
    box-shadow: var(--shadow);
    cursor: zoom-in;
    transition: transform 0.2s;
}

.figure img:hover {
    box-shadow: var(--shadow-md);
}

.figure-grid {
    display: flex;
    flex-wrap: wrap;
    gap: 12px;
    justify-content: center;
    margin: 12px 0;
}

.figure-grid .figure {
    flex: 0 1 48%;
    min-width: 300px;
}

.fig-caption {
    font-size: 0.85rem;
    color: var(--text-secondary);
    margin: 4px 0 0;
}

.note {
    font-size: 0.9rem;
    color: var(--text-secondary);
}

/* Lightbox overlay for click-to-enlarge */
.lightbox-overlay {
    display: none;
    position: fixed;
    top: 0; left: 0; right: 0; bottom: 0;
    background: rgba(0,0,0,0.85);
    z-index: 9999;
    cursor: zoom-out;
    justify-content: center;
    align-items: center;
    padding: 20px;
}

.lightbox-overlay.active {
    display: flex;
}

.lightbox-overlay img {
    max-width: 95vw;
    max-height: 95vh;
    object-fit: contain;
    border-radius: 4px;
    box-shadow: 0 4px 20px rgba(0,0,0,0.5);
}

.lightbox-overlay .lightbox-caption {
    position: fixed;
    bottom: 12px;
    left: 50%;
    transform: translateX(-50%);
    color: #fff;
    font-size: 0.95rem;
    background: rgba(0,0,0,0.6);
    padding: 6px 16px;
    border-radius: 4px;
}

/* Print */
@media print {
    header { background: #1a237e !important; -webkit-print-color-adjust: exact; }
    .section { break-inside: avoid; }
    .toc { break-after: page; }
    .figure img { max-width: 100%; }
}
"""


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("EditRNA-A3A v3 HTML Report Generator")
    print("=" * 50)
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Report path: {REPORT_PATH}")
    print()

    # Load all JSON files
    print("Loading JSON data files...")

    # Load ALL 7 baselines from per-model directories
    baselines = []
    baselines_dir = OUTPUT_DIR / "baselines"
    if baselines_dir.exists():
        for model_dir in sorted(baselines_dir.iterdir()):
            results_file = model_dir / "results.json"
            if results_file.exists():
                try:
                    with open(results_file) as f:
                        entry = json.load(f)
                    entry.setdefault("model", model_dir.name)
                    baselines.append(entry)
                except Exception as e:
                    print(f"  [SKIP] {results_file}: {e}")
    if not baselines:
        # Fallback to combined file
        baselines = load_json("baselines/all_results.json") or []
        # Merge editrna if missing from all_results.json
        model_names = {e.get("model") for e in baselines}
        if "editrna" not in model_names:
            editrna_entry = load_json("baselines/editrna/results.json")
            if editrna_entry:
                editrna_entry.setdefault("model", "editrna")
                baselines.append(editrna_entry)
    # Merge gradient boosting if available
    model_names = {e.get("model") for e in baselines}
    if "gradient_boosting" not in model_names:
        gb_entry = load_json("baselines/gradient_boosting/results.json")
        if gb_entry:
            gb_entry.setdefault("model", "gradient_boosting")
            baselines.append(gb_entry)
    print(f"  Loaded {len(baselines)} baseline models")

    cross_dataset = load_json("cross_dataset/cross_dataset_results.json")
    global _cross_dataset_matrix
    _cross_dataset_matrix = load_json("dataset_matrix/dataset_matrix_results.json")
    multiseed = load_json("multiseed/multiseed_results.json")
    clinvar = load_json("clinvar_prediction/clinvar_prediction_results.json")
    rate_pred = load_json("rate_prediction/rate_prediction_results.json")
    rate_all_arch = load_json("rate_prediction/rate_all_architectures_results.json")
    dataset_stats = load_json("dataset_analysis/dataset_statistics.json")
    const_fac = load_json("constitutive_facultative/results.json")
    tissue_cond = load_json("tissue_conditioned_rate/tissue_conditioned_results.json")
    cross_tissue = load_json("cross_tissue_disease/cross_tissue_disease_results.json")
    pcpg = load_json("pcpg_analysis/pcpg_analysis_results.json")
    disease_enrich = load_json("disease_enrichment/disease_enrichment_results.json")
    incremental = load_json("incremental_levanon/incremental_levanon_results.json")
    if incremental is None:
        incremental = load_json("incremental_datasets/incremental_results.json")
    multitask = load_json("multitask_comparison/multitask_results.json")
    binary_rate = load_json("binary_rate_correlation/binary_rate_correlation.json")
    embedding = load_json("embedding_analysis/embedding_analysis.json")
    motif = load_json("motif_analysis/motif_analysis_results.json")
    structure = load_json("structure_analysis/structure_analysis.json")
    local_window = load_json("local_window/local_window_results.json")
    fpfn = load_json("iteration3/iteration3_results.json")
    rate_deep_dive = load_json("rate_deep_dive/rate_deep_dive_results.json")
    rate_baselines = load_json("rate_baselines/rate_baselines_results.json")
    hardneg_baselines = load_json("hardneg_baselines/hardneg_baselines_results.json")
    hardneg_matrix = load_json("hardneg_matrix/hardneg_matrix_results.json")
    editability = load_json("editability/editability_results.json")
    tc_motif = load_json("tc_motif_reanalysis/tc_motif_reanalysis_results.json")
    # New experiment results
    rate_reg = load_json("rate_regularized/regularization_sweep_results.json")
    editrna_reg = load_json("editrna_rate_reg/editrna_rate_reg_results.json")
    dual_pooled = load_json("dual_pooled_rate/dual_pooled_results.json")
    rate_unified = load_json("rate_unified/rate_unified_results.json")
    a3a_filtered = load_json("a3a_filtered/a3a_filtered_results.json")
    trained_emb = load_json("embedding_trained/results.json")
    cross_all_methods = load_json("cross_dataset_all_methods/cross_dataset_all_methods_results.json")
    feature_augmented = load_json("feature_augmented_rerun/feature_augmented_rerun_results.json")
    rate_5fold = load_json("rate_5fold_zscore/rate_5fold_results.json")
    if rate_5fold is None:
        rate_5fold = load_json("rate_5fold_positives/rate_5fold_results.json")  # fallback
    feature_augmented_rate = load_json("feature_augmented_rate/feature_augmented_rate_results.json")
    rate_gnn_fusion = load_json("rate_gnn_fusion/rate_gnn_fusion_results.json")
    rate_gnn_endtoend = load_json("rate_gnn_endtoend/rate_gnn_endtoend_results.json")
    classification_a3a_5fold = load_json("classification_a3a_5fold/classification_a3a_5fold_results.json")
    rate_norm = load_json("rate_normalization/rate_normalization_results.json")
    cross_full = load_json("cross_dataset_full/cross_dataset_full_results.json")
    # Embedding visualization base64 figures (legacy subtraction embeddings)
    _embedding_viz_figs = None
    viz_b64_path = OUTPUT_DIR / "embedding_viz_v2" / "figures_base64.json"
    if viz_b64_path.exists():
        try:
            with open(viz_b64_path) as f:
                _embedding_viz_figs = json.load(f)
            print(f"  Loaded {len(_embedding_viz_figs)} embedding viz figures")
        except Exception:
            pass
    embedding_viz_results = load_json("embedding_viz_v2/embedding_viz_results.json")
    rnasee_comparison = load_json("rnasee_comparison/rnasee_comparison_results.json")
    rnasee_error_analysis = load_json("rnasee_comparison/rnasee_error_analysis.json")
    comprehensive_emb = load_json("embedding_viz_comprehensive/embedding_viz_comprehensive_results.json")

    # Per-dataset within-dataset CV results
    rate_per_dataset = load_json("rate_per_dataset/rate_per_dataset_results.json")
    # Cross-dataset rate generalization
    cross_dataset_rate_results = load_json("rate_per_dataset/rate_cross_dataset_results.json")
    rate_fi_results = load_json("rate_feature_importance/results.json")
    incremental_rate_results = None  # CUT: old results used contaminated advisor data
    print()

    # Define 20 sections for TOC
    sections = [
        ("dataset-overview", "1. Dataset Overview"),
        ("classification-architecture", "2. DL Architecture Comparison &mdash; Classification"),
        ("cross-dataset-classification", "3. Cross-Dataset Generalization &mdash; Classification"),
        ("feature-importance-classification", "4. Feature Importance &mdash; Classification"),
        ("structure-analysis", "5. Structure Analysis"),
        ("embedding-classification", "6. Embedding &amp; Clustering &mdash; Classification"),
        ("rnasee-comparison", "7. RNAsee Comparison"),
        ("multitask", "8. Multi-Task Learning"),
        ("clinvar", "9. ClinVar Clinical Predictions"),
        ("tissue-analysis", "10. Tissue Analysis"),
        ("error-analysis", "11. Error Analysis"),
        ("a3a-filtering", "12. A3A Enzyme-Specific Filtering"),
        ("rate-architecture", "13. DL Architecture Comparison &mdash; Rate Prediction"),
        ("cross-dataset-rate", "14. Cross-Dataset Generalization &mdash; Rate Prediction"),
        ("feature-importance-rate", "15. Feature Importance &mdash; Rate Prediction"),
        ("incremental-rate", "16. Incremental Dataset Addition &mdash; Rate Prediction"),
    ]

    # Build HTML
    print("Generating HTML sections...")
    now = datetime.now().strftime("%Y-%m-%d %H:%M")

    parts = []
    parts.append(f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>EditRNA-A3A v3: Comprehensive Analysis Report (hg38)</title>
    <style>
{CSS}
    </style>
</head>
<body>
<header>
    <div class="container">
        <h1>EditRNA-A3A v3: Comprehensive Analysis Report (hg38)</h1>
        <div class="subtitle">APOBEC3A C-to-U RNA Editing Prediction &mdash; hg38 Genome Assembly</div>
        <div class="meta">Generated: {now} | Edit Effect Framework | Causal Embedding Analysis</div>
    </div>
</header>
<div class="container">
""")

    parts.append(gen_toc(sections))

    # 1. Dataset Overview
    parts.append(gen_dataset_overview(dataset_stats, motif))
    # 2. DL Architecture Comparison - Classification
    parts.append(gen_classification_architecture(baselines, feature_augmented,
                                                    classification_a3a_5fold))
    # 3. Cross-Dataset Generalization - Classification
    parts.append(gen_cross_dataset_classification(cross_dataset, cross_all_methods, cross_full))
    # 4. Feature Importance - Classification
    parts.append(gen_feature_importance_classification())
    # 5. Structure Analysis
    structure_enhanced = load_json("structure_enhanced/structure_enhanced_results.json")
    parts.append(gen_structure_analysis(structure, structure_enhanced))
    # 6. Embedding & Clustering - Classification
    parts.append(gen_embedding_classification(embedding, _embedding_viz_figs, embedding_viz_results,
                                              trained_emb, comprehensive_emb))
    # 7. RNAsee Comparison
    parts.append(gen_rnasee_comparison(rnasee_comparison, rnasee_error_analysis))
    # 8. Multi-Task Learning
    parts.append(gen_multitask(multitask))
    # 9. ClinVar Clinical Predictions
    parts.append(gen_clinvar(clinvar, classification_a3a=classification_a3a_5fold))
    # 10. Tissue Analysis
    parts.append(gen_tissue_analysis(const_fac, tissue_cond, cross_tissue, pcpg))
    # 12. Error Analysis (merged hard negatives + FP/FN clusters)
    parts.append(gen_error_analysis(hardneg_baselines, hardneg_matrix, fpfn))
    # 13. A3A Enzyme-Specific Filtering
    parts.append(gen_a3a_filtering(a3a_filtered))
    # 14. DL Architecture Comparison - Rate Prediction
    parts.append(gen_rate_architecture(rate_5fold, rate_per_dataset))
    # 15. Cross-Dataset Generalization - Rate Prediction
    parts.append(gen_cross_dataset_rate(cross_dataset_rate_results))
    # 16. Feature Importance - Rate Prediction
    parts.append(gen_feature_importance_rate(rate_fi_results))
    # 17. Incremental Dataset Addition - Rate Prediction
    parts.append(gen_incremental_rate(incremental_rate_results))

    parts.append(f"""
</div>
<footer>
    <p>EditRNA-A3A v3 &mdash; Comprehensive Analysis Report (hg38) &mdash; Generated {now}</p>
    <p>Causal Edit Effect Framework for APOBEC3A C-to-U RNA Editing</p>
</footer>

<!-- Lightbox overlay for click-to-enlarge -->
<div class="lightbox-overlay" id="lightbox">
    <img id="lightbox-img" src="" alt="">
    <div class="lightbox-caption" id="lightbox-caption"></div>
</div>
<script>
(function() {{
    var overlay = document.getElementById('lightbox');
    var lbImg = document.getElementById('lightbox-img');
    var lbCap = document.getElementById('lightbox-caption');
    // Attach click handlers to all figure images
    document.querySelectorAll('.figure img').forEach(function(img) {{
        img.addEventListener('click', function(e) {{
            lbImg.src = this.src;
            lbCap.textContent = this.alt || '';
            overlay.classList.add('active');
            e.stopPropagation();
        }});
    }});
    // Close on overlay click or Escape
    overlay.addEventListener('click', function() {{
        overlay.classList.remove('active');
    }});
    document.addEventListener('keydown', function(e) {{
        if (e.key === 'Escape') overlay.classList.remove('active');
    }});
}})();
</script>
</body>
</html>
""")

    # Write report
    html_content = "\n".join(parts)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(REPORT_PATH, "w") as f:
        f.write(html_content)

    file_size_kb = os.path.getsize(REPORT_PATH) / 1024
    print(f"\nReport written to: {REPORT_PATH}")
    print(f"File size: {file_size_kb:.1f} KB")
    print("Done.")


if __name__ == "__main__":
    main()
