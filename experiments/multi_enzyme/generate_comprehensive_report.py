#!/usr/bin/env python
"""
Generate comprehensive, self-contained HTML report for multi-enzyme APOBEC analysis.

Produces a tabbed dashboard with one tab per enzyme category plus a multi-enzyme
comparison tab. Uses CSS-only tabs (radio buttons + :checked).

Usage:
    conda run -n quris python experiments/multi_enzyme/generate_comprehensive_report.py
"""
import base64
import csv
import json
import sys

import numpy as np
from pathlib import Path
from collections import defaultdict

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUTS_DIR = Path(__file__).parent / "outputs"
REPORT_FILE = OUTPUTS_DIR / "comprehensive_multi_enzyme_report.html"

# ---------------------------------------------------------------------------
# Data paths
# ---------------------------------------------------------------------------
PATHS = {
    "a3a_cls": PROJECT_ROOT / "experiments/apobec3a/outputs/classification_a3a_5fold/classification_a3a_5fold_results.json",
    "a3b_cls": PROJECT_ROOT / "experiments/apobec3b/outputs/classification/classification_results.json",
    "a3g_cls": PROJECT_ROOT / "experiments/apobec3g/outputs/classification/classification_results.json",
    "both_cls": PROJECT_ROOT / "experiments/apobec_both/outputs/classification/classification_results.json",
    "neither_cls": PROJECT_ROOT / "experiments/apobec_neither/outputs/classification/classification_results.json",
    "unknown_cls": PROJECT_ROOT / "experiments/apobec_unknown/outputs/classification/classification_results.json",
    "a4_cls": PROJECT_ROOT / "experiments/apobec4/outputs/classification/classification_results.json",
    "a4_fi": PROJECT_ROOT / "experiments/apobec4/outputs/classification/feature_importance.csv",
    "pairing_A4": PROJECT_ROOT / "experiments/multi_enzyme/outputs/structure_analysis_v3/A4_unpaired_mfe.png",
    "a3a_fi": PROJECT_ROOT / "experiments/apobec3a/outputs/classification_a3a_5fold/feature_importance_cls_gb_hand.csv",
    "a3b_fi": PROJECT_ROOT / "experiments/apobec3b/outputs/classification/feature_importance.csv",
    "a3g_fi": PROJECT_ROOT / "experiments/apobec3g/outputs/classification/feature_importance.csv",
    "both_fi": PROJECT_ROOT / "experiments/apobec_both/outputs/classification/feature_importance.csv",
    "neither_fi": PROJECT_ROOT / "experiments/apobec_neither/outputs/classification/feature_importance.csv",
    "unknown_fi": PROJECT_ROOT / "experiments/apobec_unknown/outputs/classification/feature_importance.csv",
    "unified_v2": PROJECT_ROOT / "experiments/multi_enzyme/outputs/unified_v2/unified_v2_results.json",
    "catboost_fi": PROJECT_ROOT / "experiments/multi_enzyme/outputs/unified_v2/catboost_feature_importance.csv",
    "ucc": PROJECT_ROOT / "experiments/common/outputs/ucc_trinucleotide/ucc_trinucleotide_results.json",
    "apobec1": PROJECT_ROOT / "experiments/common/outputs/apobec1_validation/apobec1_validation_results.json",
    "tissue_clustering": PROJECT_ROOT / "experiments/common/outputs/tissue_clustering/tissue_clustering_results.json",
    "logistic_regression": PROJECT_ROOT / "experiments/common/outputs/logistic_regression/logistic_regression_results.json",
    "a3b_clinvar": PROJECT_ROOT / "experiments/apobec3b/outputs/clinvar/a3b_clinvar_results.json",
    "a3g_clinvar": PROJECT_ROOT / "experiments/apobec3g/outputs/clinvar/a3g_clinvar_results.json",
    "loop_pos": PROJECT_ROOT / "data/processed/multi_enzyme/loop_position_per_site_v3.csv",
    "a3g_rate": PROJECT_ROOT / "experiments/apobec3g/outputs/rate_analysis/rate_analysis_results.json",
    "both_rate": PROJECT_ROOT / "experiments/apobec_both/outputs/rate_analysis/rate_analysis_results.json",
    "neither_rate": PROJECT_ROOT / "experiments/apobec_neither/outputs/rate_analysis/rate_analysis_results.json",
    # New results (March 21)
    "editrna_per_enzyme": PROJECT_ROOT / "experiments/multi_enzyme/outputs/editrna_per_enzyme/editrna_per_enzyme_results.json",
    "catboost_per_enzyme": PROJECT_ROOT / "experiments/multi_enzyme/outputs/catboost_per_enzyme.json",
    "unified_v1": PROJECT_ROOT / "experiments/multi_enzyme/outputs/unified_network_v1/unified_network_results.json",
    "neural_baselines": PROJECT_ROOT / "experiments/multi_enzyme/outputs/neural_baselines/neural_baseline_results.json",
    "struct_analysis_v3": PROJECT_ROOT / "experiments/multi_enzyme/outputs/structure_analysis_v3/structure_analysis_all_enzymes.json",
    # Pairing profile PNGs
    "pairing_multi_smooth5": PROJECT_ROOT / "experiments/multi_enzyme/outputs/structure_analysis_v3/multi_enzyme_pairing_profile_smooth5.png",
    "pairing_multi_smooth11": PROJECT_ROOT / "experiments/multi_enzyme/outputs/structure_analysis_v3/multi_enzyme_pairing_profile_smooth11.png",
    "pairing_A3A": PROJECT_ROOT / "experiments/multi_enzyme/outputs/structure_analysis_v3/A3A_pairing_single.png",
    "pairing_A3B": PROJECT_ROOT / "experiments/multi_enzyme/outputs/structure_analysis_v3/A3B_pairing_single.png",
    "pairing_A3G": PROJECT_ROOT / "experiments/multi_enzyme/outputs/structure_analysis_v3/A3G_pairing_single.png",
    "pairing_A3A_A3G": PROJECT_ROOT / "experiments/multi_enzyme/outputs/structure_analysis_v3/A3A_A3G_pairing_single.png",
    "pairing_Neither": PROJECT_ROOT / "experiments/multi_enzyme/outputs/structure_analysis_v3/Neither_pairing_single.png",
}

# Feature index to name mapping (40-dim hand features)
FEATURE_NAMES = [
    "5p_UC", "5p_CC", "5p_AC", "5p_GC",
    "3p_CA", "3p_CG", "3p_CU", "3p_CC",
    "m2_A", "m2_C", "m2_G", "m2_U",
    "m1_A", "m1_C", "m1_G", "m1_U",
    "p1_A", "p1_C", "p1_G", "p1_U",
    "p2_A", "p2_C", "p2_G", "p2_U",
    "is_unpaired", "loop_size", "dist_to_junction",
    "dist_to_apex", "relative_loop_position",
    "left_stem_length", "right_stem_length",
    "max_adjacent_stem_length", "local_unpaired_fraction",
    "delta_pairing_center", "delta_accessibility_center",
    "delta_entropy_center", "delta_mfe",
    "mean_delta_pairing_window", "mean_delta_accessibility_window",
    "std_delta_pairing_window",
]

FEATURE_CATEGORY = {}
for i, name in enumerate(FEATURE_NAMES):
    if i < 24:
        FEATURE_CATEGORY[name] = "Motif"
    elif i < 33:
        FEATURE_CATEGORY[name] = "Loop"
    else:
        FEATURE_CATEGORY[name] = "StructDelta"

# Also map indexed names
for i, name in enumerate(FEATURE_NAMES):
    if i < 24:
        FEATURE_CATEGORY[f"motif_{i}"] = "Motif"
    elif i < 33:
        FEATURE_CATEGORY[f"motif_{i}"] = "Loop"
    else:
        FEATURE_CATEGORY[f"struct_delta_{i - 33}"] = "StructDelta"

INDEX_TO_NAME = {}
for i, name in enumerate(FEATURE_NAMES):
    INDEX_TO_NAME[f"motif_{i}"] = name
    if i >= 33:
        INDEX_TO_NAME[f"struct_delta_{i - 33}"] = name
# Named features map to themselves
for name in FEATURE_NAMES:
    INDEX_TO_NAME[name] = name


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_json(path: Path) -> dict:
    if not path.exists():
        print(f"  [WARN] Missing: {path}", file=sys.stderr)
        return {}
    with open(path) as f:
        return json.load(f)


def load_csv_rows(path: Path) -> list:
    if not path.exists():
        print(f"  [WARN] Missing: {path}", file=sys.stderr)
        return []
    with open(path) as f:
        return list(csv.DictReader(f))


def fmt(v, digits=3):
    if v is None or v == "N/A":
        return "N/A"
    try:
        v = float(v)
    except (ValueError, TypeError):
        return str(v)
    if v != v:  # NaN
        return "N/A"
    return f"{v:.{digits}f}"


def fmt_p(p):
    if p is None:
        return "N/A"
    try:
        p = float(p)
    except (ValueError, TypeError):
        return "N/A"
    if p != p:
        return "N/A"
    if p < 1e-100:
        return "<1e-100"
    if p < 0.001:
        return f"{p:.2e}"
    return f"{p:.4f}"


def resolve_feature_name(raw_name):
    return INDEX_TO_NAME.get(raw_name, raw_name)


def get_feature_category(raw_name):
    cat = FEATURE_CATEGORY.get(raw_name)
    if cat:
        return cat
    resolved = resolve_feature_name(raw_name)
    return FEATURE_CATEGORY.get(resolved, "Other")


def category_color(cat):
    return {"Motif": "#1a73e8", "Loop": "#0d904f", "StructDelta": "#e8710a"}.get(cat, "#888")


def category_bg(cat):
    return {"Motif": "#e8f0fe", "Loop": "#e6f4ea", "StructDelta": "#fef7e0"}.get(cat, "#f0f0f0")


def _load_clinical_insight(enzyme):
    """Load clinical interpretation text for an enzyme from paper/clinical_insights.md."""
    import re as _re
    ci_path = PROJECT_ROOT / "paper/clinical_insights.md"
    if not ci_path.exists():
        return ""
    content = ci_path.read_text()

    # Map enzyme names to section headers in the file
    header_map = {
        "A3A": "APOBEC3A (A3A)",
        "A3B": "APOBEC3B (A3B)",
        "A3G": "APOBEC3G (A3G)",
        "A3A_A3G": "A3A_A3G (Dual-Enzyme Sites)",
        "Neither": "Neither (Putative APOBEC1)",
        "Unknown": "Unknown (NaN Enzyme Assignment)",
        "A4": "APOBEC4",
    }
    header = header_map.get(enzyme, enzyme)
    # Extract the section between this header and the next ---
    pattern = rf"## {_re.escape(header)}\s*\n(.*?)(?=\n---|\Z)"
    match = _re.search(pattern, content, _re.DOTALL)
    if not match:
        return ""
    section = match.group(1).strip()

    # Convert markdown to HTML
    # Bold
    section = _re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', section)
    # Split into subsections by **header:** pattern
    parts = _re.split(r'\n\n+', section)
    html = ""
    for part in parts:
        part = part.strip()
        if not part:
            continue
        html += f"<p>{part}</p>\n"
    return html


# ---------------------------------------------------------------------------
# Load all data
# ---------------------------------------------------------------------------

def load_all_data():
    data = {}

    # Classification results per enzyme
    cls_keys = {
        "A3A": "a3a_cls", "A3B": "a3b_cls", "A3G": "a3g_cls",
        "A3A_A3G": "both_cls", "Neither": "neither_cls", "Unknown": "unknown_cls", "A4": "a4_cls",
    }
    data["cls"] = {}
    for enz, key in cls_keys.items():
        data["cls"][enz] = load_json(PATHS[key])

    # Feature importance per enzyme
    fi_keys = {
        "A3A": "a3a_fi", "A3B": "a3b_fi", "A3G": "a3g_fi",
        "A3A_A3G": "both_fi", "Neither": "neither_fi", "Unknown": "unknown_fi", "A4": "a4_fi",
    }
    data["fi"] = {}
    for enz, key in fi_keys.items():
        data["fi"][enz] = load_csv_rows(PATHS[key])

    # Logistic regression
    lr_raw = load_json(PATHS["logistic_regression"])
    if isinstance(lr_raw, list):
        data["lr"] = {x["enzyme"]: x for x in lr_raw}
    else:
        data["lr"] = lr_raw

    # UCC trinucleotide
    data["ucc"] = load_json(PATHS["ucc"])

    # APOBEC1 validation
    data["apobec1"] = load_json(PATHS["apobec1"])

    # Tissue clustering
    data["tissue"] = load_json(PATHS["tissue_clustering"])

    # ClinVar
    data["a3b_clinvar"] = load_json(PATHS["a3b_clinvar"])
    data["a3g_clinvar"] = load_json(PATHS["a3g_clinvar"])

    # Rate analysis
    data["a3g_rate"] = load_json(PATHS["a3g_rate"])
    data["both_rate"] = load_json(PATHS["both_rate"])
    data["neither_rate"] = load_json(PATHS["neither_rate"])

    # Loop position per site (v3 multi-enzyme)
    data["loop_pos"] = load_csv_rows(PATHS["loop_pos"])

    # Also load A3A v1 pipeline loop positions (correct data for A3A tab)
    a3a_v1_loop_path = PROJECT_ROOT / "experiments/apobec3a/outputs/loop_position/loop_position_per_site.csv"
    a3a_v1_splits_path = PROJECT_ROOT / "data/processed/splits_expanded_a3a.csv"
    if a3a_v1_loop_path.exists() and a3a_v1_splits_path.exists():
        a3a_v1_splits = {}
        for row in load_csv_rows(a3a_v1_splits_path):
            a3a_v1_splits[row.get("site_id", "")] = row.get("is_edited", "0")
        a3a_v1_rows = []
        for row in load_csv_rows(a3a_v1_loop_path):
            sid = row.get("site_id", "")
            row["enzyme"] = "A3A"
            row["label"] = a3a_v1_splits.get(sid, "0")
            a3a_v1_rows.append(row)
        data["a3a_v1_loop_pos"] = a3a_v1_rows
    else:
        data["a3a_v1_loop_pos"] = None

    # Unified v2
    data["unified"] = load_json(PATHS["unified_v2"])

    return data


# ---------------------------------------------------------------------------
# Compute structure stats from loop_position_per_site_v3.csv
# ---------------------------------------------------------------------------

def compute_structure_stats(loop_rows):
    """Compute per-enzyme structure statistics from loop position data."""
    stats = {}
    by_enzyme = defaultdict(list)
    for row in loop_rows:
        enz = row.get("enzyme", "")
        by_enzyme[enz].append(row)

    for enz, rows in by_enzyme.items():
        positives = [r for r in rows if r.get("label") == "1"]
        negatives = [r for r in rows if r.get("label") == "0"]

        def compute_group(group):
            n = len(group)
            if n == 0:
                return {"n": 0, "unpaired_pct": 0, "mean_rlp": 0, "mean_loop_size": 0}
            unpaired = [r for r in group if r.get("is_unpaired", "").lower() == "true"]
            unpaired_pct = len(unpaired) / n * 100
            rlps = []
            loop_sizes = []
            for r in unpaired:
                try:
                    rlp = float(r.get("relative_loop_position", 0))
                    if rlp > 0:
                        rlps.append(rlp)
                except (ValueError, TypeError):
                    pass
                try:
                    ls = float(r.get("loop_size", 0))
                    if ls > 0:
                        loop_sizes.append(ls)
                except (ValueError, TypeError):
                    pass
            mean_rlp = sum(rlps) / len(rlps) if rlps else 0
            mean_ls = sum(loop_sizes) / len(loop_sizes) if loop_sizes else 0
            return {"n": n, "unpaired_pct": unpaired_pct, "mean_rlp": mean_rlp, "mean_loop_size": mean_ls}

        stats[enz] = {"pos": compute_group(positives), "neg": compute_group(negatives)}

    return stats


# ---------------------------------------------------------------------------
# Extract classification metrics
# ---------------------------------------------------------------------------

def get_cls_models(cls_data):
    """Extract model metrics from classification results JSON.
    Returns list of (model_name, auroc, auprc, f1, precision, recall, std_auroc)."""
    models = cls_data.get("models", {})
    results = []
    for name, info in models.items():
        mm = info.get("mean_metrics", info)
        auroc = mm.get("auroc", mm.get("mean_auroc"))
        auprc = mm.get("auprc", mm.get("mean_auprc"))
        f1 = mm.get("f1", mm.get("mean_f1"))
        prec = mm.get("precision", mm.get("mean_precision"))
        rec = mm.get("recall", mm.get("mean_recall"))
        sm = info.get("std_metrics", info)
        std_auroc = sm.get("std_auroc", sm.get("auroc", 0))
        if isinstance(std_auroc, float) and std_auroc > 0.5:
            std_auroc = 0  # This was the mean, not std
        results.append((name, auroc, auprc, f1, prec, rec, std_auroc))
    return results


def get_fi_top15(fi_rows):
    """Get top 15 features from feature importance CSV rows."""
    result = []
    for row in fi_rows[:15]:
        # A3A format: feature_name, mean_importance, std_importance, category
        # Others: feature, importance
        name = row.get("feature_name", row.get("feature", ""))
        imp = float(row.get("mean_importance", row.get("importance", 0)))
        result.append((name, imp))
    return result


# ---------------------------------------------------------------------------
# HTML generation
# ---------------------------------------------------------------------------

ENZYME_DISPLAY = {
    "A3A": "APOBEC3A",
    "A3B": "APOBEC3B",
    "A3G": "APOBEC3G",
    "A3A_A3G": "Both (A3A+A3G)",
    "Neither": "Neither",
    "Unknown": "Unknown",
    "A4": "APOBEC4",
}

ENZYME_COLORS = {
    "A3A": "#1a73e8",
    "A3B": "#e8710a",
    "A3G": "#0d904f",
    "A3A_A3G": "#7b1fa2",
    "Neither": "#d93025",
    "Unknown": "#5f6368",
    "A4": "#795548",
}

ENZYME_SOURCES = {
    "A3A": "Levanon/Advisor (120 A3A-only), Asaoka 2019, Alqassim 2021, Sharma 2015",
    "A3B": "Kockler 2026 (BT-474 cell line), Dang 2021",
    "A3G": "Levanon/Advisor (60 A3G-only), Kockler 2026",
    "A3A_A3G": "Levanon/Advisor (178 dual-editor sites)",
    "Neither": "Levanon/Advisor (206 sites, unknown editor)",
    "Unknown": "Levanon/Advisor (72 sites, no enzyme assignment)",
    "A4": "Levanon T3 (181 expression-correlated sites, 21 exclusive)",
}


def html_css():
    return """
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
    --orange: #e8710a;
    --orange-light: #fef0e0;
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
    padding: 40px 0 20px;
    margin-bottom: 0;
}

header .container { padding-top: 0; padding-bottom: 0; }

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

/* Tabs */
.tabs {
    background: var(--card-bg);
    border-bottom: 1px solid var(--border);
    box-shadow: var(--shadow);
    position: sticky;
    top: 0;
    z-index: 100;
}

.tabs .container {
    padding-top: 0;
    padding-bottom: 0;
    display: flex;
    gap: 0;
    overflow-x: auto;
}

.tabs input[type="radio"] { display: none; }

.tabs label {
    padding: 14px 20px;
    cursor: pointer;
    font-size: 0.95rem;
    font-weight: 500;
    color: var(--text-secondary);
    border-bottom: 3px solid transparent;
    transition: all 0.2s;
    white-space: nowrap;
    user-select: none;
}

.tabs label:hover {
    color: var(--primary);
    background: var(--primary-light);
}

.tab-content { display: none; }

#tab-a3a:checked ~ .tabs label[for="tab-a3a"],
#tab-a3b:checked ~ .tabs label[for="tab-a3b"],
#tab-a3g:checked ~ .tabs label[for="tab-a3g"],
#tab-both:checked ~ .tabs label[for="tab-both"],
#tab-neither:checked ~ .tabs label[for="tab-neither"],
#tab-unknown:checked ~ .tabs label[for="tab-unknown"],
#tab-multi:checked ~ .tabs label[for="tab-multi"] {
    color: var(--primary);
    border-bottom-color: var(--primary);
    font-weight: 600;
}

#tab-a3a:checked ~ .content .content-a3a,
#tab-a3b:checked ~ .content .content-a3b,
#tab-a3g:checked ~ .content .content-a3g,
#tab-both:checked ~ .content .content-both,
#tab-neither:checked ~ .content .content-neither,
#tab-unknown:checked ~ .content .content-unknown,
#tab-multi:checked ~ .content .content-multi {
    display: block;
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
    font-size: 1.4rem;
    margin-top: 0;
    padding-bottom: 10px;
    border-bottom: 2px solid var(--primary-light);
}

.section h3 {
    color: var(--text);
    font-size: 1.1rem;
    margin-top: 20px;
}

.section-desc {
    color: var(--text-secondary);
    font-size: 0.92rem;
    margin-top: -4px;
}

/* Tables */
table {
    width: 100%;
    border-collapse: collapse;
    font-size: 0.9rem;
    margin-top: 12px;
}

th {
    background: #f1f3f4;
    font-weight: 600;
    text-align: left;
    padding: 10px 12px;
    border-bottom: 2px solid var(--border);
    font-size: 0.85rem;
    text-transform: uppercase;
    letter-spacing: 0.3px;
    color: var(--text-secondary);
}

td {
    padding: 8px 12px;
    border-bottom: 1px solid #eee;
}

tr:hover td { background: #f8f9ff; }

.best-cell { background: var(--success-light) !important; font-weight: 600; }
.worst-cell { background: var(--danger-light) !important; }

/* KPI cards */
.kpi-row {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
    gap: 16px;
    margin: 16px 0;
}

.kpi {
    background: var(--primary-light);
    border-radius: 8px;
    padding: 16px;
    text-align: center;
}

.kpi .value {
    font-size: 1.8rem;
    font-weight: 700;
    color: var(--primary);
}

.kpi .label {
    font-size: 0.8rem;
    color: var(--text-secondary);
    text-transform: uppercase;
    letter-spacing: 0.5px;
}

.kpi.green { background: var(--success-light); }
.kpi.green .value { color: var(--success); }
.kpi.orange { background: var(--orange-light); }
.kpi.orange .value { color: var(--orange); }
.kpi.purple { background: var(--purple-light); }
.kpi.purple .value { color: var(--purple); }
.kpi.red { background: var(--danger-light); }
.kpi.red .value { color: var(--danger); }

/* Feature importance bars */
.fi-bar-container {
    display: flex;
    align-items: center;
    gap: 8px;
}
.fi-bar {
    height: 16px;
    border-radius: 3px;
    min-width: 2px;
}
.fi-name {
    font-size: 0.85rem;
    min-width: 180px;
    font-family: 'Consolas', 'Monaco', monospace;
}
.fi-val {
    font-size: 0.8rem;
    color: var(--text-secondary);
    min-width: 55px;
    text-align: right;
}

/* Badge */
.badge {
    display: inline-block;
    padding: 2px 8px;
    border-radius: 12px;
    font-size: 0.75rem;
    font-weight: 600;
}
.badge-motif { background: #e8f0fe; color: #1a73e8; }
.badge-loop { background: #e6f4ea; color: #0d904f; }
.badge-struct { background: #fef7e0; color: #e8710a; }

/* Callout */
.callout {
    padding: 16px 20px;
    border-radius: 8px;
    margin: 16px 0;
    border-left: 4px solid;
}
.callout-info { background: var(--primary-light); border-color: var(--primary); }
.callout-success { background: var(--success-light); border-color: var(--success); }
.callout-warning { background: var(--warning-light); border-color: var(--warning); }
.callout-danger { background: var(--danger-light); border-color: var(--danger); }

/* Placeholder */
.placeholder {
    background: #fffbe6;
    border: 1px dashed #f9ab00;
    border-radius: 8px;
    padding: 16px;
    color: #856404;
    font-style: italic;
}

/* Responsive */
@media (max-width: 800px) {
    .container { padding: 12px 16px; }
    .tabs label { padding: 10px 12px; font-size: 0.85rem; }
    .kpi-row { grid-template-columns: repeat(2, 1fr); }
}
"""


def html_header():
    from datetime import datetime
    now = datetime.now().strftime("%B %d, %Y at %H:%M")
    return f"""
<header>
    <div class="container">
        <h1>Multi-Enzyme APOBEC RNA Editing Analysis</h1>
        <div class="subtitle">Comprehensive dashboard: classification, structure, motif, ClinVar, and tissue analysis across 6 enzyme categories</div>
        <div class="meta">Last updated: {now} | Levanon/Advisor 636 + Kockler/Dang/Asaoka/Alqassim | hg38/hg19 | Full 201nt sequences</div>
    </div>
</header>
"""


def build_classification_table(cls_data, lr_data=None, extra_rows=None):
    """Build HTML table for classification results."""
    models = get_cls_models(cls_data)
    if not models:
        return '<p class="placeholder">Classification data not available.</p>'

    # Filter out models with NaN AUROC
    valid = [(n, a, ap, f, p, r, s) for n, a, ap, f, p, r, s in models
             if a is not None and (not isinstance(a, float) or a == a)]
    if not valid:
        return '<p class="placeholder">No valid model results.</p>'

    # Find best AUROC
    aurocs = [a for _, a, *_ in valid if isinstance(a, (int, float))]
    best_auroc = max(aurocs) if aurocs else None
    worst_auroc = min(aurocs) if aurocs else None

    # Add LR if available
    if lr_data:
        lr_auroc = lr_data.get("mean_auroc")
        lr_auprc = lr_data.get("mean_auprc")
        lr_std = lr_data.get("std_auroc", 0)
        valid.append(("LogisticRegression", lr_auroc, lr_auprc, None, None, None, lr_std))
        if lr_auroc and isinstance(lr_auroc, (int, float)):
            aurocs.append(lr_auroc)

    # Add extra rows (EditRNA, CatBoost, CNN, etc.)
    if extra_rows:
        for row in extra_rows:
            valid.append(row)
            a = row[1]
            if a and isinstance(a, (int, float)) and a == a:
                aurocs.append(a)

    if aurocs:
        best_auroc = max(aurocs)
        worst_auroc = min(aurocs)

    rows = ""
    for name, auroc, auprc, f1, prec, rec, std_a in valid:
        a_class = ""
        if auroc == best_auroc and len(aurocs) > 1:
            a_class = ' class="best-cell"'
        elif auroc == worst_auroc and auroc != best_auroc:
            a_class = ' class="worst-cell"'
        std_str = f" +/-{fmt(std_a)}" if std_a and isinstance(std_a, float) and std_a > 0 else ""
        rows += f"""<tr>
            <td><strong>{name}</strong></td>
            <td{a_class}>{fmt(auroc)}{std_str}</td>
            <td>{fmt(auprc)}</td>
            <td>{fmt(f1)}</td>
            <td>{fmt(prec)}</td>
            <td>{fmt(rec)}</td>
        </tr>"""

    return f"""
    <table>
        <tr><th>Model</th><th>AUROC</th><th>AUPRC</th><th>F1</th><th>Precision</th><th>Recall</th></tr>
        {rows}
    </table>"""


def build_top_motifs(ucc_data, enzyme):
    """Build compact top-5 trinucleotide motif display."""
    enz_data = ucc_data.get(enzyme, {})
    if not enz_data:
        return ""
    top_tri = enz_data.get("top_trinucleotides", [])[:5]
    n = enz_data.get("n_sites", 1)
    if not top_tri:
        return ""
    items = " &nbsp;".join(
        f'<code style="background:#f1f3f4;padding:2px 6px;border-radius:3px">{t}</code> <span style="color:var(--text-secondary)">{c/n*100:.1f}%</span>'
        for t, c in top_tri
    )
    return f'<p style="margin-top:8px"><strong>Top motifs (trinucleotide):</strong> {items}</p>'


FEATURE_DESCRIPTIONS = {
    "5p_UC": "U before C — A3A motif", "5p_CC": "C before C — A3G motif",
    "5p_AC": "A before C — APOBEC1", "5p_GC": "G before C",
    "3p_CA": "A after C", "3p_CG": "G after C — CpG", "3p_CU": "U after C", "3p_CC": "C after C",
    "m2_A": "A at pos -2", "m2_C": "C at pos -2", "m2_G": "G at pos -2", "m2_U": "U at pos -2",
    "m1_A": "A at pos -1", "m1_C": "C at pos -1 (CC)", "m1_G": "G at pos -1", "m1_U": "U at pos -1 (TC)",
    "p1_A": "A at pos +1", "p1_C": "C at pos +1", "p1_G": "G at pos +1 (CG)", "p1_U": "U at pos +1",
    "p2_A": "A at pos +2", "p2_C": "C at pos +2", "p2_G": "G at pos +2", "p2_U": "U at pos +2",
    "delta_pairing_center": "Pairing change at edit (C→U)",
    "delta_accessibility_center": "Accessibility change at edit",
    "delta_entropy_center": "Entropy change at edit",
    "delta_mfe": "MFE change (kcal/mol, C→U)",
    "mean_delta_pairing_window": "Mean pairing change ±10nt",
    "mean_delta_accessibility_window": "Mean accessibility change ±10nt",
    "std_delta_pairing_window": "Pairing variability ±10nt",
    "is_unpaired": "Edit site in loop (unpaired)",
    "loop_size": "Loop size (nt)",
    "dist_to_junction": "Distance to stem-loop junction",
    "dist_to_apex": "Distance to loop center",
    "relative_loop_position": "Position in loop (0=5', 1=3')",
    "left_stem_length": "5' stem length",
    "right_stem_length": "3' stem length",
    "max_adjacent_stem_length": "Max stem length",
    "local_unpaired_fraction": "Unpaired fraction ±10nt",
}


def build_fi_table_with_desc(fi_rows, enzyme, data):
    """Build feature importance table with description + pos/neg values per enzyme."""
    items = get_fi_top15(fi_rows)
    if not items:
        return '<p class="placeholder">Feature importance data not available.</p>'

    # Compute pos/neg feature values for this enzyme
    pos_neg = compute_feature_pos_neg(enzyme, data)

    max_val = max(v for _, v in items) if items else 1
    rows = ""
    for rank, (raw_name, val) in enumerate(items[:15], 1):
        name = resolve_feature_name(raw_name)
        cat = get_feature_category(raw_name)
        color = category_color(cat)
        pct = val / max_val * 100 if max_val > 0 else 0
        desc = FEATURE_DESCRIPTIONS.get(name, "")
        pn = pos_neg.get(name, {})
        pos_v = pn.get("pos", "—")
        neg_v = pn.get("neg", "—")
        pos_str = f"{pos_v:.3f}" if isinstance(pos_v, float) else str(pos_v)
        neg_str = f"{neg_v:.3f}" if isinstance(neg_v, float) else str(neg_v)

        rows += f"""<tr>
            <td style="text-align:center">{rank}</td>
            <td><code>{name}</code></td>
            <td><span style="color:{color};font-weight:600;font-size:0.8em">{cat}</span></td>
            <td style="font-size:0.82em;color:var(--text-secondary)">{desc}</td>
            <td>
                <div style="background:#f1f3f4;border-radius:3px;height:14px;width:120px;display:inline-block;vertical-align:middle">
                    <div style="width:{pct:.0f}%;background:{color};height:100%;border-radius:3px"></div>
                </div>
                <span style="font-size:0.82em;margin-left:4px">{fmt(val, 3)}</span>
            </td>
            <td style="text-align:center;font-size:0.85em"><strong>{pos_str}</strong></td>
            <td style="text-align:center;font-size:0.85em">{neg_str}</td>
        </tr>"""

    return f"""
    <table style="font-size:0.88em">
        <thead><tr>
            <th>#</th><th>Feature</th><th>Type</th><th>Description</th>
            <th>Importance</th><th>Pos</th><th>Neg</th>
        </tr></thead>
        <tbody>{rows}</tbody>
    </table>"""


def compute_feature_pos_neg(enzyme, data):
    """Compute mean feature values for positives and negatives of an enzyme."""
    import gc as _gc

    # For A3A, use v1 pipeline data
    if enzyme == "A3A":
        splits_path = PROJECT_ROOT / "data/processed/splits_expanded_a3a.csv"
        seqs_path = PROJECT_ROOT / "data/processed/site_sequences.json"
        struct_path = PROJECT_ROOT / "data/processed/embeddings/vienna_structure_cache.npz"
        loop_path = PROJECT_ROOT / "experiments/apobec3a/outputs/loop_position/loop_position_per_site.csv"
    else:
        splits_path = PROJECT_ROOT / "data/processed/multi_enzyme/splits_multi_enzyme_v3_with_negatives.csv"
        seqs_path = PROJECT_ROOT / "data/processed/multi_enzyme/multi_enzyme_sequences_v3_with_negatives.json"
        struct_path = PROJECT_ROOT / "data/processed/multi_enzyme/structure_cache_multi_enzyme_v3.npz"
        loop_path = PROJECT_ROOT / "data/processed/multi_enzyme/loop_position_per_site_v3.csv"

    try:
        import pandas as pd
        sys.path.insert(0, str(PROJECT_ROOT))
        splits = pd.read_csv(splits_path)
        if enzyme != "A3A":
            splits = splits[splits["enzyme"] == enzyme]

        with open(seqs_path) as f:
            seqs = json.load(f)

        sd = {}
        if Path(struct_path).exists():
            d = np.load(str(struct_path), allow_pickle=True)
            for i, sid in enumerate(d["site_ids"].astype(str)):
                sd[sid] = d["delta_features"][i]
            del d; _gc.collect()

        loop_df = pd.read_csv(loop_path)
        loop_df["site_id"] = loop_df["site_id"].astype(str)
        loop_df = loop_df.set_index("site_id")

        from src.data.apobec_feature_extraction import (
            extract_motif_features, extract_loop_features, extract_structure_delta_features,
        )

        site_ids = splits["site_id"].values
        y = splits["is_edited"].values.astype(int)

        motif = extract_motif_features(seqs, list(site_ids))
        struct = extract_structure_delta_features(sd, list(site_ids))
        loop = extract_loop_features(loop_df, list(site_ids))
        X = np.nan_to_num(np.concatenate([motif, struct, loop], axis=1), nan=0.0)

        NAMES = [
            '5p_UC', '5p_CC', '5p_AC', '5p_GC', '3p_CA', '3p_CG', '3p_CU', '3p_CC',
            'm2_A', 'm2_C', 'm2_G', 'm2_U', 'm1_A', 'm1_C', 'm1_G', 'm1_U',
            'p1_A', 'p1_C', 'p1_G', 'p1_U', 'p2_A', 'p2_C', 'p2_G', 'p2_U',
            'delta_pairing_center', 'delta_accessibility_center', 'delta_entropy_center',
            'delta_mfe', 'mean_delta_pairing_window', 'mean_delta_accessibility_window',
            'std_delta_pairing_window',
            'is_unpaired', 'loop_size', 'dist_to_junction', 'dist_to_apex',
            'relative_loop_position', 'left_stem_length', 'right_stem_length',
            'max_adjacent_stem_length', 'local_unpaired_fraction',
        ]

        result = {}
        for i, name in enumerate(NAMES):
            result[name] = {"pos": float(X[y == 1, i].mean()), "neg": float(X[y == 0, i].mean())}
        return result
    except Exception as e:
        print(f"  Warning: could not compute pos/neg for {enzyme}: {e}")
        return {}


def build_fi_bars(fi_rows, max_bars=15):
    """Build feature importance horizontal bar chart."""
    items = get_fi_top15(fi_rows)
    if not items:
        return '<p class="placeholder">Feature importance data not available.</p>'

    max_val = max(v for _, v in items) if items else 1
    html = '<div style="margin-top:12px">'
    for raw_name, val in items[:max_bars]:
        name = resolve_feature_name(raw_name)
        cat = get_feature_category(raw_name)
        color = category_color(cat)
        bg = category_bg(cat)
        pct = val / max_val * 100 if max_val > 0 else 0
        badge_cls = {"Motif": "badge-motif", "Loop": "badge-loop", "StructDelta": "badge-struct"}.get(cat, "")
        html += f"""
        <div class="fi-bar-container" style="margin-bottom:4px">
            <span class="fi-name">{name}</span>
            <span class="badge {badge_cls}">{cat}</span>
            <div style="flex:1;background:#f1f3f4;border-radius:3px;height:16px">
                <div class="fi-bar" style="width:{pct:.1f}%;background:{color}"></div>
            </div>
            <span class="fi-val">{fmt(val, 4)}</span>
        </div>"""
    html += '</div>'
    return html


def embed_png(path):
    """Encode PNG as base64 data URI for inline embedding."""
    import base64
    if not Path(path).exists():
        return ""
    with open(path, "rb") as f:
        data = base64.b64encode(f.read()).decode()
    return f'<img src="data:image/png;base64,{data}" style="width:100%;max-width:900px;border-radius:4px;border:1px solid #e5e7eb;margin:12px 0" />'


def build_structure_section(struct_stats, enzyme):
    """Build structure analysis section with comparison table + pairing profile."""
    s = struct_stats.get(enzyme, {})
    pos = s.get("pos", {})
    neg = s.get("neg", {})

    if pos.get("n", 0) == 0:
        return '<p class="placeholder">Structure data not available for this enzyme.</p>'

    html = f"""
    <h3>Structural Feature Comparison (Positive vs Negative)</h3>
    <table>
        <thead><tr><th>Property</th><th>Positives (n={pos['n']})</th><th>Negatives (n={neg['n']})</th><th>Delta</th></tr></thead>
        <tbody>
        <tr>
            <td>Unpaired %</td>
            <td><strong>{pos['unpaired_pct']:.1f}%</strong></td>
            <td>{neg['unpaired_pct']:.1f}%</td>
            <td>{pos['unpaired_pct'] - neg['unpaired_pct']:+.1f}%</td>
        </tr>
        <tr>
            <td>Mean Relative Loop Position (unpaired only)</td>
            <td><strong>{pos['mean_rlp']:.3f}</strong></td>
            <td>{neg['mean_rlp']:.3f}</td>
            <td>{pos['mean_rlp'] - neg['mean_rlp']:+.3f}</td>
        </tr>
        <tr>
            <td>Mean Loop Size (unpaired only)</td>
            <td><strong>{pos['mean_loop_size']:.1f}</strong></td>
            <td>{neg['mean_loop_size']:.1f}</td>
            <td>{pos['mean_loop_size'] - neg['mean_loop_size']:+.1f}</td>
        </tr>
        </tbody>
    </table>"""

    # Add full structural comparison from v3 analysis if available
    FEATURE_ORDER = [
        "Is Unpaired", "Loop Size", "Relative Loop Position", "Dist to Apex",
        "Dist to Junction", "Local Unpaired Fraction", "Left Stem Length",
        "Right Stem Length", "Delta Pairing (edit site)", "Delta MFE (kcal/mol)",
    ]
    struct_v3 = load_json(PATHS.get("struct_analysis_v3", ""))
    if struct_v3 and enzyme in struct_v3:
        comp = struct_v3[enzyme].get("comparison", [])
        if comp:
            # Build lookup by feature name
            comp_dict = {r["feature"]: r for r in comp}
            # Filter and order
            ordered = [comp_dict[f] for f in FEATURE_ORDER if f in comp_dict]
            if ordered:
                html += """<h3>Structural Feature Comparison (Mann-Whitney)</h3>
                <table><thead><tr><th>Feature</th><th>Pos Mean</th><th>Neg Mean</th><th>Diff</th><th>p-value</th><th>Sig</th></tr></thead><tbody>"""
                for row in ordered:
                    sig_cls = 'class="best-cell"' if row.get("significant") == "Yes" else ""
                    p_str = f"{row['p_value']:.2e}" if row.get("p_value", 1) < 0.01 else f"{row.get('p_value', 1):.4f}"
                    # For Loop Size: show mean/median
                    pos_val = row['pos_mean']
                    med = row.get("pos_median")
                    if row['feature'] == 'Loop Size' and isinstance(med, (int, float)):
                        pos_str = f"{pos_val:.1f} / {med:.0f}"
                    else:
                        pos_str = f"{pos_val:.4f}"
                    html += f"""<tr><td>{row['feature']}</td>
                        <td>{pos_str}</td><td>{row['neg_mean']:.4f}</td>
                        <td>{row['diff']:+.4f}</td><td>{p_str}</td>
                        <td {sig_cls}>{row.get('significant', 'N/A')}</td></tr>"""
                html += "</tbody></table>"

    # Add MFE unpaired fraction profile (sharper signal)
    # For A3A, prefer the v1 pipeline version (correct data)
    if enzyme == "A3A":
        unpaired_path = PROJECT_ROOT / "experiments/multi_enzyme/outputs/structure_analysis_v3/A3A_v1_unpaired_mfe.png"
    else:
        unpaired_path = PROJECT_ROOT / f"experiments/multi_enzyme/outputs/structure_analysis_v3/{enzyme}_unpaired_mfe.png"
    if unpaired_path.exists():
        html += f"""<h3>Unpaired Probability Profile — MFE Structure (Edited vs Unedited)</h3>
        <p class="section-desc">Fraction of positions unpaired in the MFE structure, averaged across sites (window=5).
        The <strong>peak at position 0</strong> shows edited sites sit in unpaired loops, flanked by paired stems.</p>
        {embed_png(unpaired_path)}"""

    # Add zoomed ±10nt MFE profile
    if enzyme == "A3A":
        zoom_path = PROJECT_ROOT / "experiments/multi_enzyme/outputs/structure_analysis_v3/A3A_v1_unpaired_mfe_zoom10.png"
    else:
        zoom_path = PROJECT_ROOT / f"experiments/multi_enzyme/outputs/structure_analysis_v3/{enzyme}_unpaired_mfe_zoom10.png"
    if zoom_path.exists():
        html += f"""<h3>Unpaired Profile — Zoomed ±10nt (Per-Position, No Smoothing)</h3>
        <p class="section-desc">Raw per-position unpaired fraction within ±10nt of the edit site. Shows the stem-loop structure at single-nucleotide resolution: the peak at position 0 is the loop, flanking valleys are the stems.</p>
        {embed_png(zoom_path)}"""

    return html


def build_motif_section(ucc_data, enzyme):
    """Build motif analysis section from UCC trinucleotide data."""
    enz_data = ucc_data.get(enzyme, {})
    if not enz_data:
        return '<p class="placeholder">Motif data not available.</p>'

    di = enz_data.get("dinucleotide", {})
    tri_top = enz_data.get("top_trinucleotides", [])[:6]
    penta_top = enz_data.get("top_pentanucleotides", [])[:5]
    n = enz_data.get("n_sites", 0)

    di_html = ""
    for dinuc in ["UC", "CC", "AC", "GC"]:
        count = di.get(dinuc, 0)
        pct = di.get(f"{dinuc}_pct", 0)
        di_html += f"<tr><td>{dinuc}</td><td>{count}</td><td>{pct:.1f}%</td></tr>"

    tri_html = ""
    for tri, count in tri_top:
        tri_html += f"<tr><td><code>{tri}</code></td><td>{count}</td><td>{count/n*100:.1f}%</td></tr>"

    penta_html = ""
    for penta, count in penta_top:
        penta_html += f"<tr><td><code>{penta}</code></td><td>{count}</td></tr>"

    return f"""
    <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:16px">
        <div>
            <h3 style="margin-top:0">Dinucleotide (n={n})</h3>
            <table><tr><th>Context</th><th>Count</th><th>%</th></tr>{di_html}</table>
        </div>
        <div>
            <h3 style="margin-top:0">Top Trinucleotides</h3>
            <table><tr><th>Motif</th><th>Count</th><th>%</th></tr>{tri_html}</table>
        </div>
        <div>
            <h3 style="margin-top:0">Top Pentanucleotides</h3>
            <table><tr><th>Motif</th><th>Count</th></tr>{penta_html}</table>
        </div>
    </div>"""


def build_enzyme_tab(enzyme, data, struct_stats):
    """Build complete content for one enzyme tab."""
    cls = data["cls"].get(enzyme, {})
    fi = data["fi"].get(enzyme, [])
    lr = data["lr"].get(enzyme) if data["lr"] else None
    ucc = data["ucc"]
    apobec1 = data["apobec1"]

    display_name = ENZYME_DISPLAY.get(enzyme, enzyme)
    color = ENZYME_COLORS.get(enzyme, "#1a73e8")
    sources = ENZYME_SOURCES.get(enzyme, "N/A")

    # Dataset overview
    n_pos = cls.get("n_positive", cls.get("n_positives", "?"))
    n_neg = cls.get("n_negative", cls.get("n_negatives", "?"))

    # Motif %
    enz_ucc = ucc.get(enzyme, {})
    di = enz_ucc.get("dinucleotide", {})
    tc_pct = di.get("UC_pct", "?")
    cc_pct = di.get("CC_pct", "?")
    if isinstance(tc_pct, float):
        tc_pct = f"{tc_pct:.1f}"
    if isinstance(cc_pct, float):
        cc_pct = f"{cc_pct:.1f}"

    # Structure from apobec1 validation (Levanon-only enzymes)
    struct_key = f"{enzyme}_structure" if enzyme not in ("A3A",) else None
    a1_struct = apobec1.get(struct_key, {}) if struct_key else {}
    unpaired_pct_a1 = a1_struct.get("is_unpaired_frac", None)
    if unpaired_pct_a1 is not None:
        unpaired_pct_a1 = f"{unpaired_pct_a1*100:.1f}%"
    else:
        # Fall back to loop position stats
        ps = struct_stats.get(enzyme, {}).get("pos", {})
        unpaired_pct_a1 = f"{ps.get('unpaired_pct', 0):.1f}%" if ps.get("n", 0) > 0 else "?"

    # Tissue class
    tc_key = f"{enzyme}_tissue_class"
    tissue_class = apobec1.get(tc_key, {})
    top_tissue = max(tissue_class, key=tissue_class.get) if tissue_class else "?"
    top_tissue_pct = ""
    if tissue_class:
        total_tc = sum(tissue_class.values())
        top_tissue_pct = f" ({tissue_class[top_tissue]/total_tc*100:.0f}%)"

    # Genomic location
    gen_key = f"{enzyme}_genomic"
    genomic = apobec1.get(gen_key, {})
    cds_n = genomic.get("CDS", 0)
    nc_n = genomic.get("Non Coding mRNA", 0)
    cds_pct = cds_n / (cds_n + nc_n) * 100 if (cds_n + nc_n) > 0 else 0

    html = f"""
    <!-- {display_name} Dataset Overview -->
    <div class="section">
        <h2 style="border-bottom-color:{color}">{display_name} -- Dataset Overview</h2>
        <div class="kpi-row">
            <div class="kpi"><div class="value">{n_pos}</div><div class="label">Positives</div></div>
            <div class="kpi"><div class="value">{n_neg}</div><div class="label">Negatives</div></div>
            <div class="kpi green"><div class="value">{tc_pct}%</div><div class="label">TC Context</div></div>
            <div class="kpi orange"><div class="value">{cc_pct}%</div><div class="label">CC Context</div></div>
            <div class="kpi purple"><div class="value">{unpaired_pct_a1}</div><div class="label">Unpaired</div></div>
        </div>
        <table>
            <tr><th>Property</th><th>Value</th></tr>
            <tr><td>Sources</td><td>{sources}</td></tr>
            <tr><td>Dominant tissue class</td><td>{top_tissue}{top_tissue_pct}</td></tr>
            <tr><td>CDS fraction</td><td>{cds_pct:.1f}% ({cds_n} CDS / {nc_n} non-coding mRNA)</td></tr>
        </table>
        {build_top_motifs(ucc, enzyme)}
    </div>
    """

    # Classification — add EditRNA and CatBoost results
    editrna = load_json(PATHS.get("editrna_per_enzyme", ""))
    catboost = None  # CatBoost removed — no improvement over XGBoost
    neural = load_json(PATHS.get("neural_baselines", ""))

    extra_models_html = ""
    extra_rows = []

    # Get existing model names from the base classification to avoid duplicates
    existing_models = set(n for n, *_ in get_cls_models(cls))

    if editrna and enzyme in editrna:
        for mname in ["SubtractionMLP", "PooledMLP", "EditRNA", "EditRNA+Hand"]:
            if mname in existing_models:
                continue  # skip duplicates
            mr = editrna[enzyme].get(mname, {})
            if mr:
                extra_rows.append((mname, mr.get("auroc"), mr.get("auprc"), None, None, None, mr.get("auroc_std", 0)))

    if catboost and enzyme in catboost:
        cb = catboost[enzyme]
        extra_rows.append(("CatBoost", cb.get("auroc"), None, None, None, None, cb.get("std", 0)))

    if neural and enzyme in neural:
        for mname in ["HandMLP", "SeqCNN+Hand"]:
            mr = neural[enzyme].get(mname, {})
            if mr:
                extra_rows.append((mname, mr.get("auroc"), mr.get("auprc"), None, None, None, mr.get("auroc_std", 0)))

    html += f"""
    <div class="section">
        <h2 style="border-bottom-color:{color}">{display_name} -- Classification (5-Fold CV)</h2>
        <p class="section-desc">Binary classification: edited vs. motif-matched negatives. Includes GB (XGBoost), EditRNA (RNA-FM), CNN, and neural baselines.</p>
        {build_classification_table(cls, lr, extra_rows)}
    </div>
    """

    # Feature Importance — with descriptions and pos/neg values
    html += f"""
    <div class="section">
        <h2 style="border-bottom-color:{color}">{display_name} -- Feature Importance (GB)</h2>
        <p class="section-desc">Top features from Gradient Boosting, with description and positive vs negative values.
            <span class="badge badge-motif">Motif</span>
            <span class="badge badge-loop">Loop Geometry</span>
            <span class="badge badge-struct">Structure Delta</span>
        </p>
        {build_fi_table_with_desc(fi, enzyme, data)}
    </div>
    """

    # Structure Analysis
    html += f"""
    <div class="section">
        <h2 style="border-bottom-color:{color}">{display_name} -- Structure Analysis</h2>
        <p class="section-desc">Structural properties from ViennaRNA, comparing positives vs. negatives.</p>
        {build_structure_section(struct_stats, enzyme)}
    </div>
    """

    # Motif Analysis
    if enz_ucc:
        html += f"""
        <div class="section">
            <h2 style="border-bottom-color:{color}">{display_name} -- Motif Analysis</h2>
            <p class="section-desc">Sequence context distribution around edited cytidines (Levanon sites only).</p>
            {build_motif_section(ucc, enzyme)}
        </div>
        """

    # Enzyme-specific sections
    if enzyme == "A3A":
        html += build_a3a_extra(data)
    elif enzyme == "A3B":
        html += build_a3b_extra(data)
    elif enzyme == "A3G":
        html += build_a3g_extra(data)
    elif enzyme == "Neither":
        html += build_neither_extra(data)

    # Clinical Interpretation -- use deep analysis content if available
    clinical_sections = build_clinical_deep_section()
    if enzyme in clinical_sections:
        html += clinical_sections[enzyme]
    else:
        # Fall back to clinical_insights.md content
        clinical_html = _load_clinical_insight(enzyme)
        if clinical_html:
            html += f"""
            <div class="section">
                <h2 style="border-bottom-color:{color}">{display_name} -- Clinical Interpretation</h2>
                {clinical_html}
            </div>
            """
        else:
            html += f"""
            <div class="section">
                <h2 style="border-bottom-color:{color}">{display_name} -- Clinical Interpretation</h2>
                <p class="section-desc">Clinical interpretation pending additional analysis.</p>
            </div>
            """

    return html


def build_a3a_extra(data):
    """A3A-specific: RNAsee comparison, ClinVar."""
    return """
    <div class="section">
        <h2>APOBEC3A -- ClinVar Pathogenic Enrichment</h2>
        <p class="section-desc">GB_Full model scoring of 1.69M ClinVar C-to-U variants.</p>
        <table>
            <tr><th>Model</th><th>Threshold</th><th>Odds Ratio</th><th>p-value</th></tr>
            <tr><td><strong>GB_Full</strong></td><td>P &ge; 0.5</td><td class="best-cell">1.279</td><td>&lt;1e-138</td></tr>
            <tr><td>RNAsee Rules-based</td><td>score &gt; 0</td><td class="worst-cell">0.76</td><td>&lt;1e-50</td></tr>
            <tr><td>RNAsee RF</td><td>P &ge; 0.5</td><td>1.08</td><td>marginal</td></tr>
        </table>
        <div class="callout callout-success">
            <strong>Key finding:</strong> GB_Full is the only model with robust ClinVar pathogenic enrichment.
            RNAsee's rules-based approach shows pathogenic <em>depletion</em> (OR=0.76), meaning it systematically
            avoids pathogenic variants. Prior calibration (pi=0.019) confirms the signal is real.
        </div>
    </div>
    """


def build_a3b_extra(data):
    """A3B-specific: dual role section + ClinVar."""
    html = """
    <div class="section">
        <h2 style="border-bottom-color:#e8710a">APOBEC3B -- Dual Role: DNA Mutator &amp; RNA Editor</h2>
        <p class="section-desc">A3B is primarily known as a somatic DNA mutator in cancer, but recent work reveals it also performs RNA editing with distinct preferences.</p>

        <table>
            <thead><tr><th>Property</th><th>DNA Mutation (COSMIC SBS2/SBS13)</th><th>RNA Editing (This Study)</th></tr></thead>
            <tbody>
            <tr><td><strong>Substrate</strong></td><td>Single-stranded DNA (replication fork)</td><td>RNA hairpin loops</td></tr>
            <tr><td><strong>Motif</strong></td><td><strong>TCW</strong> (W=A/T) — strong TC preference</td><td><strong>Mixed</strong> — TC only 32%, near-random</td></tr>
            <tr><td><strong>Top trinucleotide</strong></td><td>TCA, TCT (dominant)</td><td>UCC (12%), UCA (11%), CCA (8%) — no dominant</td></tr>
            <tr><td><strong>Structural context</strong></td><td>ssDNA at replication fork, lagging strand</td><td>Large RNA loops (mean 7.8nt), no positional bias</td></tr>
            <tr><td><strong>Context dependence</strong></td><td>Replication timing, fork stalling</td><td>Extended sequence composition (±10-20nt)</td></tr>
            <tr><td><strong>Cancer relevance</strong></td><td>Major mutational source in breast, bladder, cervical</td><td>Transcriptome modulation in tumors (newly appreciated)</td></tr>
            </tbody>
        </table>

        <div class="callout callout-warning">
            <strong>Key insight:</strong> A3B's RNA and DNA substrates have fundamentally different recognition rules.
            DNA mutations follow the canonical TCW motif; RNA editing shows <strong>no motif preference</strong> and instead
            depends on extended sequence context and loose loop architecture. This suggests A3B uses different
            binding modes or cofactors for DNA vs RNA substrates — a finding with implications for understanding
            APOBEC mutagenesis in cancer.
        </div>

        <h3>Why A3B RNA Editing is Hard to Predict</h3>
        <p>Standard 40-dim hand features (motif &plusmn;2nt + local structure) achieve only <strong>AUROC=0.575</strong> for A3B --
        near random. The critical discovery: after fixing a data leakage bug in v3 loop position features (NaN coverage
        differed 40% vs 87% between positives and negatives), <strong>structure is genuinely uninformative</strong>
        for A3B (all structure features achieve 0.50-0.55 AUROC). The signal is in extended sequence composition.</p>

        <h4>Feature Engineering Progression</h4>
        <table>
            <thead><tr><th>Feature Set</th><th>Dim</th><th>AUROC</th><th>Key Insight</th></tr></thead>
            <tbody>
            <tr><td>Hand40 (motif+loop+struct)</td><td>40</td><td>0.575</td><td>Too local for A3B</td></tr>
            <tr><td>Trinuc freq &plusmn;20</td><td>64</td><td>0.736</td><td>Wider context helps</td></tr>
            <tr><td>4-mer spectrum &plusmn;50</td><td>256</td><td><strong>0.804</strong></td><td>Matches EditRNA (0.810)</td></tr>
            <tr class="best-row"><td><strong>MEGA seq (depth=8)</strong></td><td>1553</td><td><strong>0.828</strong></td><td><strong>BEATS neural model</strong></td></tr>
            <tr><td>RNA-FM 640-dim</td><td>640</td><td>0.822</td><td>Hand features match this</td></tr>
            <tr><td>RNA-FM + MEGA seq</td><td>2193</td><td>0.855</td><td>Complementary information</td></tr>
            </tbody>
        </table>

        <div class="callout callout-success">
            <strong>Gap closed:</strong> Best hand-engineered features (MEGA seq, AUROC=0.828) <strong>exceed</strong>
            the EditRNA+Hand neural model target (0.810) by 0.018. The signal is extended sequence composition over
            &plusmn;50-100nt, acting as a proxy for gene type, genomic neighborhood, or chromatin context. This is what
            RNA-FM captures in its 640-dim embeddings, explaining why hand features can match the neural model.
        </div>
    </div>
    """

    cv = data.get("a3b_clinvar", {})
    raw_rows = ""
    for item in cv.get("enrichment_raw", [])[:4]:
        t = item.get("threshold", "?")
        or_val = item.get("odds_ratio", 0)
        p = item.get("p_value", 1)
        raw_rows += f"<tr><td>P &ge; {t}</td><td>{or_val:.3f}</td><td>{fmt_p(p)}</td></tr>"

    cal_rows = ""
    for item in cv.get("enrichment_calibrated", [])[:3]:
        orig_t = item.get("original_threshold", "?")
        cal_t = item.get("calibrated_threshold", 0)
        or_val = item.get("odds_ratio", 0)
        p = item.get("p_value", 1)
        cal_rows += f"<tr><td>P &ge; {orig_t} &rarr; {cal_t:.4f}</td><td>{or_val:.3f}</td><td>{fmt_p(p)}</td></tr>"

    if not raw_rows:
        return html

    html += f"""
    <div class="section">
        <h2 style="border-bottom-color:#e8710a">APOBEC3B -- ClinVar Pathogenic Enrichment</h2>
        <h3>Raw Enrichment</h3>
        <table>
            <tr><th>Threshold</th><th>Odds Ratio</th><th>p-value</th></tr>
            {raw_rows}
        </table>
        <h3>Calibrated Enrichment (pi_real from prevalence)</h3>
        <table>
            <tr><th>Threshold</th><th>Odds Ratio</th><th>p-value</th></tr>
            {cal_rows}
        </table>
        <div class="callout callout-info">
            <strong>A3B ClinVar:</strong> Moderate raw enrichment (OR=1.08 at t=0.3) improves substantially
            after Bayesian calibration (OR=1.55 at calibrated t=0.008), confirming the signal is real but
            requires proper prior adjustment due to the low editing prevalence.
        </div>
    </div>
    """
    return html


def build_a3g_extra(data):
    """A3G-specific: ClinVar CC-context, rate analysis."""
    cv = data.get("a3g_clinvar", {})

    # CC-context enrichment
    cc_enr = cv.get("enrichment_cc_context_only", {})
    cc_rows = ""
    for k in sorted(cc_enr.keys()):
        item = cc_enr[k]
        t = item.get("threshold", k)
        or_val = item.get("odds_ratio", 0)
        p = item.get("p_value", 1)
        cc_rows += f"<tr><td>{t}</td><td>{or_val:.3f}</td><td>{fmt_p(p)}</td></tr>"

    # Rate analysis
    rate = data.get("a3g_rate", {})
    top_tissues = rate.get("top_tissues", [])[:5]
    tissue_rows = ""
    for t in top_tissues:
        tissue_rows += f"<tr><td>{t['tissue']}</td><td>{t['mean_rate_pct']:.3f}%</td></tr>"

    html = ""
    if cc_rows:
        html += f"""
        <div class="section">
            <h2 style="border-bottom-color:#0d904f">APOBEC3G -- ClinVar CC-Context Enrichment</h2>
            <p class="section-desc">Enrichment restricted to CC-context ClinVar variants (n=515k), matching A3G's motif preference.</p>
            <table>
                <tr><th>Threshold</th><th>Odds Ratio</th><th>p-value</th></tr>
                {cc_rows}
            </table>
            <div class="callout callout-success">
                <strong>Key finding:</strong> A3G shows strong CC-context enrichment (OR=1.76 at t=0.4, p&lt;1e-300),
                confirming that A3G preferentially targets CC motifs at pathogenic sites. The signal is specific to
                CC-context variants and disappears for all C-to-U variants.
            </div>
        </div>
        """

    if tissue_rows:
        html += f"""
        <div class="section">
            <h2 style="border-bottom-color:#0d904f">APOBEC3G -- Tissue Rate Profile</h2>
            <p class="section-desc">Top tissues by mean editing rate across A3G sites (Levanon 54-tissue GTEx data).</p>
            <table>
                <tr><th>Tissue</th><th>Mean Rate</th></tr>
                {tissue_rows}
            </table>
        </div>
        """

    return html


def build_neither_extra(data):
    """Neither-specific: APOBEC1 evidence."""
    a1 = data.get("apobec1", {})
    evidence = a1.get("apobec1_evidence", {})
    tests = evidence.get("tests", [])
    score = evidence.get("score", "?")
    total = evidence.get("total", "?")

    if not tests:
        return ""

    test_rows = ""
    for t in tests:
        passed = str(t.get("pass", "")).lower() == "true"
        icon = "&#10003;" if passed else "&#10007;"
        color = "var(--success)" if passed else "var(--danger)"
        test_rows += f"""<tr>
            <td style="color:{color};font-weight:bold">{icon}</td>
            <td>{t.get('name', '')}</td>
            <td>{t.get('detail', '')}</td>
        </tr>"""

    # Tissue enrichment
    top_tissues = a1.get("tissue_enrichment", {}).get("neither_top10", [])[:5]
    tissue_rows = ""
    for name, rate in top_tissues:
        tissue_rows += f"<tr><td>{name}</td><td>{rate:.3f}%</td></tr>"

    mooring = a1.get("Neither_mooring", {})
    mooring_au = mooring.get("mooring_au_fraction", 0)

    return f"""
    <div class="section">
        <h2 style="border-bottom-color:#d93025">Neither -- APOBEC1 Hypothesis</h2>
        <p class="section-desc">Evidence assessment for whether "Neither" sites are edited by APOBEC1 (the canonical C-to-U editor).</p>
        <div class="callout callout-warning">
            <strong>APOBEC1 Evidence Score: {score}/{total}</strong> -- Moderate support for APOBEC1 as the editor of "Neither" sites.
        </div>
        <table>
            <tr><th></th><th>Test</th><th>Detail</th></tr>
            {test_rows}
        </table>
        <h3>Top Tissues (Neither sites)</h3>
        <table>
            <tr><th>Tissue</th><th>Mean Rate</th></tr>
            {tissue_rows}
        </table>
        <p><strong>Mooring AU fraction:</strong> {mooring_au:.3f} (significantly higher than A3A: p=1.4e-6)</p>
    </div>
    """


def build_mutation_coupling_section():
    """Build Mutation Coupling section for multi-enzyme tab."""
    v3_path = OUTPUTS_DIR / "mutation_coupling/mutation_coupling_v3_results.json"
    v4_path = OUTPUTS_DIR / "mutation_coupling/mutation_coupling_v4_results.json"
    v3 = load_json(v3_path)
    v4 = load_json(v4_path)

    if not v3 and not v4:
        return ""

    # Window-level enrichment table from v3
    window_rows = ""
    main_enr = v3.get("main_enrichment", {})
    for win in ["25", "50", "100", "250", "500", "1000"]:
        e = main_enr.get(win, {})
        if not e:
            continue
        window_rows += f"""<tr>
            <td>&plusmn;{win}bp</td>
            <td>{e.get('edit_mean', 0):.2f}</td>
            <td>{e.get('ctrl_mean', 0):.2f}</td>
            <td><strong>{e.get('ratio', 0):.3f}</strong></td>
            <td>{fmt_p(e.get('p_value'))}</td>
        </tr>"""

    # CpG decomposition from v4
    cpg_rows = ""
    cpg_strat = v4.get("cpg_stratified", {})
    for ctx in ["CpG", "non-CpG"]:
        key = f"{ctx}_100"
        e = cpg_strat.get(key, {})
        if not e:
            continue
        cpg_rows += f"""<tr>
            <td>{ctx}</td>
            <td>{e.get('n_edit', 0):,}</td>
            <td><strong>{e.get('ratio', 0):.3f}</strong></td>
            <td>{fmt_p(e.get('p_value'))}</td>
        </tr>"""

    # Per-enzyme from v3
    per_enz = v3.get("per_enzyme", {})
    enz_rows = ""
    for enz_name in ["A3A", "A3A_A3G", "A3G", "Neither", "Unknown"]:
        e = per_enz.get(enz_name, {})
        if not e:
            continue
        enz_rows += f"""<tr>
            <td><strong>{ENZYME_DISPLAY.get(enz_name, enz_name)}</strong></td>
            <td>{e.get('n', 0):,}</td>
            <td>{e.get('ratio', 0):.3f}</td>
            <td>{fmt_p(e.get('p_value'))}</td>
        </tr>"""

    html = f"""
    <div class="section">
        <h2>DNA Mutation Coupling at RNA Editing Sites</h2>
        <p class="section-desc">Are APOBEC RNA editing sites enriched for nearby DNA C&gt;T mutations?
        Analysis of 5,703 editing sites against 1.69M ClinVar C&gt;T variants, with same-exon trinucleotide-matched controls.
        Four iterations of increasingly stringent controls (v1-v4) refined the estimate from 3.4x (confounded) to ~25-35% (real).</p>

        <h3>Window-Level Enrichment (Same-Exon Controls)</h3>
        <table>
            <thead><tr><th>Window</th><th>Edit Mean</th><th>Ctrl Mean</th><th>Ratio</th><th>p-value</th></tr></thead>
            <tbody>{window_rows}</tbody>
        </table>

        <div class="callout callout-info">
            <strong>Distance-dependent signal:</strong>
            Enrichment peaks at close range (&plusmn;25bp: 1.53x) and decays monotonically to baseline by ~2kb,
            consistent with a local mutational or genomic context effect rather than a genome-wide artifact.
        </div>

        <h3>CpG vs Non-CpG Decomposition (&plusmn;100bp)</h3>
        <table>
            <thead><tr><th>Context</th><th>n Sites</th><th>Ratio</th><th>p-value</th></tr></thead>
            <tbody>{cpg_rows}</tbody>
        </table>

        <div class="callout callout-warning">
            <strong>CpG drives the paired-test signal:</strong>
            CpG editing sites overlap ClinVar positions at 5.1x the rate of non-CpG sites (15.9% vs 3.1%).
            Paired Wilcoxon test is significant for CpG (p=2.1e-7) but not non-CpG (p=0.52), indicating
            the enrichment primarily reflects CpG mutation hotspots co-occurring with editing sites in the same genomic regions.
        </div>

        <h3>Per-Enzyme Enrichment (&plusmn;250bp)</h3>
        <table>
            <thead><tr><th>Enzyme</th><th>n Sites</th><th>Ratio</th><th>p-value</th></tr></thead>
            <tbody>{enz_rows}</tbody>
        </table>

        <div class="callout callout-success">
            <strong>Enzyme-specific:</strong>
            Only A3A and A3A+A3G show significant enrichment. A3G, Neither, and Unknown do not.
            The enrichment is best explained by co-occurrence of CpG mutation hotspots and APOBEC editing sites
            in gene-dense, exonic, accessible chromatin regions, rather than direct APOBEC-mediated DNA mutagenesis.
        </div>
    </div>
    """
    return html


def build_germline_mutation_section():
    """Build Germline Mutation Analysis section for multi-enzyme tab."""
    html = """
    <div class="section">
        <h2>Germline Mutation Analysis: A3G-Testis Hypothesis</h2>
        <p class="section-desc">Testing whether APOBEC RNA editing in germline tissues (testis, ovary) causes heritable
        C&gt;T mutations. Uses 636 Levanon/Advisor editing sites with per-tissue rates and ClinVar variant density.</p>

        <h3>Hypothesis: REJECTED</h3>
        <p>If A3G edits RNA in testis, it might also mutate DNA in germline cells, causing heritable C&gt;T mutations.
        This predicts that testis-specific A3G editing sites should show elevated germline variant density.</p>

        <table>
            <thead><tr><th>Comparison</th><th>Result</th><th>p-value</th></tr></thead>
            <tbody>
            <tr><td>Testis-edited sites vs non-testis</td><td>LOWER ClinVar density (6.56 vs 8.84)</td><td>0.003</td></tr>
            <tr><td>A3G testis-specific vs other A3G</td><td>LOWER density (3.06 vs 5.69)</td><td>0.33 (ns)</td></tr>
            <tr><td>Testis editing rate vs variant density</td><td>Spearman r = -0.065</td><td>0.198 (ns)</td></tr>
            <tr><td>CC&gt;CT enrichment at A3G sites</td><td>OR = 0.655 (trend toward DEPLETION)</td><td>0.13 (ns)</td></tr>
            <tr><td>TC&gt;TT enrichment at A3A sites</td><td>OR = 0.805 (trend toward DEPLETION)</td><td>0.054 (ns)</td></tr>
            </tbody>
        </table>

        <div class="callout callout-warning">
            <strong>Informative negative result:</strong>
            Testis-specific editing sites show <em>lower</em>, not higher, germline variant density.
            APOBEC-motif variants (CC&gt;CT, TC&gt;TT) trend toward depletion (OR=0.655-0.805),
            suggesting <strong>purifying selection</strong> against APOBEC-like germline mutations at editing sites.
            This is consistent with the sites being under selective constraint, not subject to mutagenesis.
        </div>

        <h3>Why the Hypothesis Fails</h3>
        <table>
            <thead><tr><th>Explanation</th><th>Detail</th></tr></thead>
            <tbody>
            <tr><td>RNA vs DNA substrate</td><td>A3G is primarily an RNA editor in this context; its DNA-editing activity (HIV restriction) may not operate on host genomic DNA</td></tr>
            <tr><td>Post-transcriptional timing</td><td>RNA editing occurs on mature mRNA, not genomic DNA, even if the enzyme is present in germline cells</td></tr>
            <tr><td>Selection against damage</td><td>Sites we observe today are precisely those where APOBEC acts on RNA without damaging DNA</td></tr>
            </tbody>
        </table>

        <div class="callout callout-info">
            <strong>What ClinVar enrichment DOES mean:</strong>
            All editing sites show 3-7x ClinVar enrichment vs matched controls, but this reflects their location
            in functionally important, well-characterized coding genes under strong selection -- not germline mutagenesis.
        </div>
    </div>
    """
    return html


def build_aid_apobec_comparison_section():
    """Build AID/APOBEC DNA mutation signature comparison section."""
    summary_path = OUTPUTS_DIR / "cosmic_overlap/analysis_summary.json"
    summary = load_json(summary_path)

    if not summary:
        return ""

    sig_dist = summary.get("signature_distribution", {})
    path_enr = summary.get("pathogenic_enrichment_by_class", {})

    # Signature distribution table
    sig_rows = ""
    sig_labels = {
        "APOBEC_DNA_hotspot": ("APOBEC DNA hotspot (SBS2/SBS13)", "TCA, TCT"),
        "APOBEC_TC_other": ("APOBEC TC (other)", "TCG, TCC"),
        "AID_hotspot": ("AID hotspot (SBS84/85)", "WRC (upstream purine)"),
        "A3G_CC": ("A3G-like (CC context)", "CC"),
        "Other": ("Other", "AC, GC"),
    }
    for key, (label, motif) in sig_labels.items():
        n = sig_dist.get(key, 0)
        pe = path_enr.get(key, {})
        or_val = pe.get("odds_ratio", 0)
        pr_hi = pe.get("path_rate_high", 0)
        pr_lo = pe.get("path_rate_low", 0)
        or_cls = ""
        if or_val > 1.3:
            or_cls = ' class="best-cell"'
        elif or_val < 0.9:
            or_cls = ' class="worst-cell"'
        sig_rows += f"""<tr>
            <td>{label}</td>
            <td>{motif}</td>
            <td>{n:,}</td>
            <td>{pr_hi*100:.2f}%</td>
            <td>{pr_lo*100:.2f}%</td>
            <td{or_cls}><strong>{or_val:.2f}</strong></td>
        </tr>"""

    html = f"""
    <div class="section">
        <h2>AID/APOBEC DNA Mutation Signature Comparison</h2>
        <p class="section-desc">Comparing COSMIC DNA mutation signatures (SBS2, SBS13, SBS84/85) with APOBEC RNA editing
        predictions from the GB model, using 1.69M ClinVar C&gt;U variants scored for A3A editing probability.</p>

        <h3>Mutually Exclusive Motifs</h3>
        <p>AID and APOBEC target <strong>completely non-overlapping</strong> sequence contexts: AID requires upstream purine
        (A/G), APOBEC requires upstream pyrimidine (T/U). Zero sites share both motifs.</p>

        <h3>Pathogenic Enrichment by Mutation Signature</h3>
        <table>
            <thead><tr><th>Signature</th><th>Motif</th><th>n ClinVar</th><th>Path% (high-score)</th><th>Path% (low-score)</th><th>Odds Ratio</th></tr></thead>
            <tbody>{sig_rows}</tbody>
        </table>

        <div class="callout callout-success">
            <strong>APOBEC DNA hotspots show the strongest pathogenic enrichment (OR=1.61):</strong>
            Among the ~18% of APOBEC TCA/TCT sites predicted as structurally favorable for editing,
            pathogenic variants are 61% more likely than among those predicted as unfavorable.
            Conversely, AID hotspots show pathogenic <em>depletion</em> (OR=0.79) -- AID mutations cause disease
            through somatic hypermutation in B cells, a mechanism unrelated to RNA editing structural features.
        </div>

        <h3>RNA vs DNA Substrate Preferences</h3>
        <table>
            <thead><tr><th>Enzyme</th><th>DNA Motif</th><th>RNA Motif</th><th>Same?</th></tr></thead>
            <tbody>
            <tr><td><strong>APOBEC3A</strong></td><td>TC (SBS2/13)</td><td>TC (strong, 84%)</td><td style="color:var(--success)">Yes</td></tr>
            <tr><td><strong>APOBEC3B</strong></td><td>TC (SBS2/13)</td><td>Mixed, no strong bias</td><td style="color:var(--danger)">No</td></tr>
            <tr><td><strong>APOBEC3G</strong></td><td>CC</td><td>CC (strong, 93%)</td><td style="color:var(--success)">Yes</td></tr>
            <tr><td><strong>AID</strong></td><td>WRC</td><td>N/A (no known RNA editing)</td><td>N/A</td></tr>
            </tbody>
        </table>
    </div>
    """
    return html


def build_unified_interpretability_section():
    """Build updated Unified Network interpretability section."""
    html = """
    <div class="section">
        <h2>Unified Network Interpretability: Cross-Enzyme Transfer Learning</h2>
        <p class="section-desc">Analysis of what the unified (jointly-trained) model learns differently from per-enzyme models,
        based on rescued/lost sites and feature attribution.</p>

        <h3>Rescued Sites: Non-Canonical Motifs</h3>
        <p>Sites correctly classified by the unified model but missed by per-enzyme models reveal the mechanism
        of cross-enzyme knowledge transfer:</p>

        <table>
            <thead><tr><th>Enzyme</th><th>Rescued</th><th>Dominant Rescue Pattern</th><th>Key Insight</th></tr></thead>
            <tbody>
            <tr><td><strong>A3A</strong></td><td>488</td><td>CC (22.8%), AC (8%), GC (10.3%) contexts</td><td>Per-enzyme overfits to TC; unified learns non-TC can be edited</td></tr>
            <tr><td><strong>A3B</strong></td><td>463</td><td>70.2% TC rescued (vs 32.3% baseline)</td><td>Unified imposes A3A's TC preference on A3B</td></tr>
            <tr><td><strong>A3G</strong></td><td>55</td><td>Structurally atypical CC sites (lower RLP)</td><td>Unified relaxes overly strict structural profile</td></tr>
            <tr><td><strong>Unknown</strong></td><td>57</td><td>Diverse motifs, 80% unpaired</td><td>Strong structural signal from large-data enzymes</td></tr>
            </tbody>
        </table>

        <div class="callout callout-warning">
            <strong>Why A3B doesn't benefit (+0.002 AUROC):</strong>
            A3B's "floppy structure" preference (edits in diverse structural contexts, including paired/stem regions)
            conflicts with the A3A/A3G structural priors in the shared backbone. The unified model rescues A3B-TC-in-loop
            sites but loses A3B-TC-in-stem sites (54 lost positives: 63% TC, only 26% unpaired). Gains and losses
            nearly cancel.
        </div>

        <h3>Disease-Relevant Rescued Genes</h3>
        <table>
            <thead><tr><th>Gene</th><th>Enzyme</th><th>Unified Score</th><th>Per-Enzyme Score</th><th>Relevance</th></tr></thead>
            <tbody>
            <tr><td><strong>FUS</strong></td><td>A3B</td><td>0.913</td><td>0.003</td><td>ALS / frontotemporal dementia. RNA-binding protein; editing could alter autoregulatory binding.</td></tr>
            <tr><td><strong>DDX31</strong></td><td>A3B</td><td>0.981</td><td>0.004</td><td>DEAD-box RNA helicase; ribosome biogenesis, implicated in cancers. Editing could alter substrate landscape.</td></tr>
            <tr><td><strong>RDM1</strong></td><td>A3A</td><td>0.964</td><td>0.113</td><td>RAD52-like DNA repair motif; CC context at loop apex. Editing in DNA repair gene could modulate damage response.</td></tr>
            </tbody>
        </table>

        <div class="callout callout-info">
            <strong>Biological insight:</strong>
            The unified model separates structural editability from motif preference. Structure determines <em>editability</em>;
            motif determines <em>which enzyme</em> performs the edit. Per-enzyme models conflate these two levels,
            overfitting to the dominant motif and rejecting structurally valid sites with non-canonical flanking sequences.
            Rescued sites include genes in neurodegeneration (FUS), DNA repair (RDM1), and cancer signaling (DDX31).
        </div>
    </div>
    """
    return html


def build_clinical_deep_section():
    """Build clinical deep analysis sections for per-enzyme tabs."""
    # Returns a dict of enzyme -> HTML for each enzyme's clinical section
    sections = {}

    # A3A clinical
    sections["A3A"] = """
    <div class="section">
        <h2 style="border-bottom-color:#1a73e8">APOBEC3A -- Clinical Deep Analysis</h2>

        <h3>Pathogenic Editing Sites: 89% Create Stop Codons</h3>
        <p>36 experimentally validated editing sites overlap with pathogenic/likely-pathogenic ClinVar variants.
        Of these, <strong>32 of 36 (89%) create premature stop codons</strong> (nonsense mutations).
        Only 4 are missense (CHD2, COQ8A, DNM2, MMUT). This reflects codon biochemistry:
        C-to-U editing at CAG, CGA, or CAA codons produces TAG, TGA, or TAA stop codons.</p>

        <div class="callout callout-warning">
            <strong>Critical: 35 of 36 are from overexpression experiments.</strong>
            Only SDHB has confirmed endogenous editing in normal human tissues (Levanon/Advisor GTEx data).
            The other 35 demonstrate that A3A <em>can</em> edit these positions, but physiological relevance requires
            demonstration of endogenous editing.
        </div>

        <h3>SDHB: The Definitive Case Study</h3>
        <table>
            <thead><tr><th>Property</th><th>Value</th></tr></thead>
            <tbody>
            <tr><td>Gene</td><td>SDHB (Succinate Dehydrogenase Complex Iron Sulfur Subunit B)</td></tr>
            <tr><td>Disease</td><td>Pheochromocytoma/paraganglioma syndrome (OMIM 115310)</td></tr>
            <tr><td>Consequence</td><td>Stopgain (premature stop codon in tumor suppressor)</td></tr>
            <tr><td>Endogenous editing rate</td><td>Whole blood: 1.22% (10x higher than other tissues)</td></tr>
            <tr><td>Tissue specificity</td><td>Blood-specific (matches A3A expression in monocytes/macrophages)</td></tr>
            <tr><td>GB model score</td><td>0.977 (top 13th percentile of all SDHB C&gt;T variants)</td></tr>
            <tr><td>ClinVar pathogenic variants</td><td>26 pathogenic + 30 likely pathogenic in SDHB</td></tr>
            </tbody>
        </table>

        <div class="callout callout-success">
            <strong>Misannotation risk:</strong>
            SDHB is the single clearest case for clinical misannotation. RNA-based SDHB mutation screening on blood
            samples could detect the ~1.2% C&gt;U editing signal and report it as a pathogenic germline mutation.
            Variant calling pipelines should flag known RNA editing sites, especially for RNA-based diagnostic assays.
        </div>

        <h3>Three Hypotheses for Pathogenic Enrichment</h3>
        <table>
            <thead><tr><th>Hypothesis</th><th>Evidence</th><th>Assessment</th></tr></thead>
            <tbody>
            <tr>
                <td><strong>1. Structural vulnerability</strong></td>
                <td>Within-gene analysis: 62.3% of genes show path &gt; benign scores (p=1.5e-30). RLP is #1 feature.</td>
                <td style="color:var(--success)"><strong>MOST LIKELY</strong></td>
            </tr>
            <tr>
                <td><strong>2. Codon context selection</strong></td>
                <td>TC-rich codons may cluster with functionally constrained positions</td>
                <td>Partially controlled for by within-gene analysis</td>
            </tr>
            <tr>
                <td><strong>3. Active mutagenic contribution</strong></td>
                <td>23 known editing sites are pathogenic. APOBEC SBS2/SBS13 signatures in cancer</td>
                <td>Possible but modest effect size (OR~1.16) argues against strong contribution</td>
            </tr>
            </tbody>
        </table>

        <h3>Cancer Gene Depletion: GoF vs LoF Explanation</h3>
        <p>Cancer-related pathogenic variants are paradoxically <em>less</em> likely to be predicted editing sites (OR=0.804, p=2.7e-5).</p>
        <table>
            <thead><tr><th>Gene Category</th><th>Mean GB Score</th><th>Editing Fraction (&ge;0.5)</th><th>Pattern</th></tr></thead>
            <tbody>
            <tr><td>LoF tumor suppressors (pathogenic)</td><td>0.744</td><td>79.1%</td><td>Path &gt; Benign (+0.028)</td></tr>
            <tr><td>GoF oncogenes (pathogenic)</td><td>0.726</td><td>75.6%</td><td>Path &lt; Benign (-0.044)</td></tr>
            </tbody>
        </table>
        <p><strong>Explanation:</strong> GoF mutations occur at catalytic residues in structured protein regions (base-paired mRNA),
        while LoF mutations (especially nonsense) can occur in APOBEC-accessible loop regions. The model correctly
        identifies structural accessibility, which overlaps with LoF-vulnerable but not GoF hotspot positions.</p>

        <h3>Ciliopathy Pathway Cluster</h3>
        <p>Among the 36 pathogenic editing sites, 4 genes form a striking ciliopathy cluster:</p>
        <table>
            <thead><tr><th>Gene</th><th>GB Score</th><th>Disease</th></tr></thead>
            <tbody>
            <tr><td>CEP290</td><td>0.999</td><td>Joubert syndrome / polycystic kidney</td></tr>
            <tr><td>OFD1</td><td>0.996</td><td>Orofaciodigital syndrome</td></tr>
            <tr><td>DYNC2H1</td><td>0.995</td><td>Jeune thoracic dystrophy</td></tr>
            <tr><td>IFT74</td><td>0.799</td><td>IFT74-related disorder</td></tr>
            </tbody>
        </table>
    </div>
    """

    # A3B gets the feature challenge update integrated into its extra section
    sections["A3B"] = """
    <div class="section">
        <h2 style="border-bottom-color:#e8710a">APOBEC3B -- Clinical Interpretation</h2>

        <h3>Clinical Context</h3>
        <p>A3B is the primary APOBEC enzyme implicated in cancer mutagenesis (breast, bladder, cervical cancers).
        Its ClinVar enrichment (calibrated OR=1.552) supports the hypothesis that A3B-mediated mutations occur
        preferentially at functionally important sites. The absence of a dominant motif means A3B editing sites
        cannot be identified by sequence scanning alone, making structure-aware prediction essential.</p>

        <h3>Tissue Context</h3>
        <p>A3B is constitutively expressed across many solid tissues, unlike the interferon-induced A3A.
        This explains its relevance to epithelial cancers and means A3B RNA editing may be a persistent
        modifier of tumor transcriptomes beyond its well-characterized DNA mutagenesis role.</p>
    </div>
    """

    # A3G clinical
    sections["A3G"] = """
    <div class="section">
        <h2 style="border-bottom-color:#0d904f">APOBEC3G -- Clinical Interpretation</h2>

        <h3>Clinical Context</h3>
        <p>A3G shows the strongest raw ClinVar enrichment (CC-context OR=1.759 at t=0.4). Its extreme CC specificity
        and tetraloop structural requirement create a small but high-confidence target set. The very small training set
        (n=119 sites) means clinical predictions should be interpreted with caution.</p>

        <h3>Testis-Specific Editing</h3>
        <p>A3G editing is concentrated in testis (31/60 = 52% of sites), suggesting relevance for male germline
        transcriptome regulation. However, the germline mutation hypothesis is rejected (see Germline Analysis section):
        testis-specific A3G sites show <em>lower</em> variant density, consistent with purifying selection.</p>
    </div>
    """

    return sections


def build_multi_enzyme_tab(data, struct_stats):
    """Build the multi-enzyme comparison tab."""
    html = ""

    # Cross-enzyme classification comparison — include deep models + unified
    enzymes = ["A3A", "A3B", "A3G", "A3A_A3G", "Neither"]
    editrna_data = load_json(PATHS.get("editrna_per_enzyme", ""))
    neural_data = load_json(PATHS.get("neural_baselines", ""))
    unified_v1_data = load_json(PATHS.get("unified_v1", ""))

    # Extract unified V1 per-enzyme AUROCs
    unified_v1_aurocs = {}
    if unified_v1_data and "v1" in unified_v1_data:
        for enz in enzymes:
            vals = [r.get("per_enzyme_auroc", {}).get(enz, 0) for r in unified_v1_data["v1"]]
            if any(v > 0 for v in vals):
                unified_v1_aurocs[enz] = float(np.mean(vals))

    comp_rows = ""
    for enz in enzymes:
        cls = data["cls"].get(enz, {})
        models = get_cls_models(cls)
        gb = next(((a, s) for n, a, ap, f, p, r, s in models if "GB" in n and "Hand" in n), (None, None))
        motif = next(((a, s) for n, a, ap, f, p, r, s in models if "Motif" in n), (None, None))
        struct = next(((a, s) for n, a, ap, f, p, r, s in models if "Struct" in n and "Only" in n), (None, None))
        n_pos = cls.get("n_positive", cls.get("n_positives", "?"))

        # EditRNA results
        er = editrna_data.get(enz, {}) if editrna_data else {}
        sub_mlp = er.get("SubtractionMLP", {}).get("auroc")
        editrna = er.get("EditRNA", {}).get("auroc")
        editrna_h = er.get("EditRNA+Hand", {}).get("auroc")

        # Neural baselines
        nn = neural_data.get(enz, {}) if neural_data else {}
        cnn_hand = nn.get("SeqCNN+Hand", {}).get("auroc")

        # Unified V1
        unified_v1 = unified_v1_aurocs.get(enz)

        # Find best across all models
        all_vals = [v for v in [gb[0] if gb else None, editrna_h, cnn_hand, unified_v1]
                    if v and isinstance(v, float)]
        best = max(all_vals) if all_vals else None

        def cell(v, is_best=False):
            if v is None: return '<td class="na-cell">—</td>'
            css = ' class="best-cell"' if is_best and v == best else ""
            return f"<td{css}>{v:.3f}</td>"

        comp_rows += f"""<tr>
            <td><strong>{ENZYME_DISPLAY.get(enz, enz)}</strong></td>
            <td>{n_pos}</td>
            {cell(gb[0] if gb else None, True)}
            {cell(motif[0] if motif else None)}
            {cell(struct[0] if struct else None)}
            {cell(editrna_h, True)}
            {cell(cnn_hand, True)}
            {cell(unified_v1, True)}
        </tr>"""

    html += f"""
    <div class="section">
        <h2>Cross-Enzyme Classification Comparison</h2>
        <p class="section-desc">Per-enzyme models (trained separately) vs Unified V1 (trained jointly on all enzymes with shared backbone).
        EditRNA+H uses pooled RNA-FM embeddings + hand features. Unified V1 uses the same features with a shared backbone + binary/enzyme heads.
        A3A's full EditRNA+Features with token-level cross-attention achieves <strong>0.935</strong> (see A3A tab).</p>
        <table>
            <thead><tr><th>Enzyme</th><th>n_pos</th><th>GB</th><th>Motif</th><th>Struct</th><th>EditRNA+H</th><th>CNN+Hand</th><th>Unified V1</th></tr></thead>
            <tbody>{comp_rows}</tbody>
        </table>
        <div class="callout callout-info">
            <strong>Key observations:</strong>
            <strong>Unified V1 (joint training)</strong> dramatically improves data-scarce enzymes: Neither +0.10, A3G +0.06.
            Per-enzyme models win for data-rich A3B (0.971 EditRNA+H vs 0.916 unified).
            For A3A and A3B, structure is the primary signal (StructOnly &gt; MotifOnly).
            For Both, Neither, and Unknown, motif is more discriminative than structure.
        </div>
    </div>
    """

    # 6-class Enzyme Classification (positives only)
    enz_cls = load_json(PROJECT_ROOT / "experiments/multi_enzyme/outputs/enzyme_classification/gb_6class_results.json")
    if enz_cls:
        per_class = enz_cls.get("per_class", {})
        enz_cls_rows = ""
        for cls_name in enz_cls.get("classes", []):
            pc = per_class.get(cls_name, {})
            enz_cls_rows += f"""<tr>
                <td><strong>{ENZYME_DISPLAY.get(cls_name, cls_name)}</strong></td>
                <td>{pc.get('precision', 0):.3f}</td>
                <td>{pc.get('recall', 0):.3f}</td>
                <td>{pc.get('f1', 0):.3f}</td>
                <td>{int(pc.get('support', 0))}</td>
            </tr>"""

        # Feature importance for enzyme discrimination
        fi = enz_cls.get("feature_importance", {})
        fi_sorted = sorted(fi.items(), key=lambda x: -x[1])[:10]
        fi_rows = ""
        for rank, (fname, imp) in enumerate(fi_sorted, 1):
            desc = FEATURE_DESCRIPTIONS.get(fname, "")
            cat = get_feature_category(fname)
            color = category_color(cat)
            fi_rows += f"""<tr>
                <td>{rank}</td>
                <td><code>{fname}</code></td>
                <td><span style="color:{color};font-weight:600;font-size:0.8em">{cat}</span></td>
                <td style="font-size:0.82em">{desc}</td>
                <td>{imp:.3f}</td>
            </tr>"""

        html += f"""
        <div class="section">
            <h2>6-Class Enzyme Classification (Positives Only)</h2>
            <p class="section-desc">Single XGBoost model predicting which enzyme edits each site. Trained on all positives (10,002 sites, 6 classes).</p>
            <div class="kpi-row">
                <div class="kpi"><div class="value">{enz_cls['accuracy']:.1%}</div><div class="label">Accuracy</div></div>
                <div class="kpi"><div class="value">{enz_cls['macro_f1']:.3f}</div><div class="label">Macro F1</div></div>
                <div class="kpi"><div class="value">{enz_cls['weighted_f1']:.3f}</div><div class="label">Weighted F1</div></div>
            </div>
            <table>
                <thead><tr><th>Enzyme</th><th>Precision</th><th>Recall</th><th>F1</th><th>n</th></tr></thead>
                <tbody>{enz_cls_rows}</tbody>
            </table>
            <div class="callout callout-warning">
                <strong>Class imbalance challenge:</strong> A3A (5,187) and A3B (4,180) dominate, achieving F1&gt;0.95.
                Small classes (A3A_A3G=178, Unknown=72) collapse. A3G (179 sites) achieves F1=0.424 — partially separable by CC motif + tetraloop.
            </div>

            <h3>Top Features for Enzyme Discrimination</h3>
            <p class="section-desc">What separates the enzymes — primarily motif context and structure delta, not loop geometry.</p>
            <table style="font-size:0.88em">
                <thead><tr><th>#</th><th>Feature</th><th>Type</th><th>Description</th><th>Importance</th></tr></thead>
                <tbody>{fi_rows}</tbody>
            </table>
        </div>
        """

    # Structure comparison across enzymes
    struct_rows = ""
    for enz in enzymes:
        s = struct_stats.get(enz, {}).get("pos", {})
        if s.get("n", 0) == 0:
            continue
        struct_rows += f"""<tr>
            <td><strong>{ENZYME_DISPLAY.get(enz, enz)}</strong></td>
            <td>{s['n']}</td>
            <td>{s['unpaired_pct']:.1f}%</td>
            <td>{s['mean_rlp']:.3f}</td>
            <td>{s['mean_loop_size']:.1f}</td>
        </tr>"""

    if struct_rows:
        # Add MFE unpaired overlay (primary — sharper signal)
        pairing_img = ""
        mfe_path = PROJECT_ROOT / "experiments/multi_enzyme/outputs/structure_analysis_v3/multi_enzyme_unpaired_mfe_smooth11.png"
        if mfe_path.exists():
            pairing_img += f"""
            <h3>Unpaired Probability — All Enzymes (MFE Structure)</h3>
            <p class="section-desc">Fraction unpaired in MFE structure, positives only (window=11). The <strong>peak at position 0</strong> reveals each enzyme's loop signature.</p>
            {embed_png(mfe_path)}
            <div class="callout callout-info">
                <strong>Key insight:</strong> A3G (green) and Both (purple) show the sharpest peak — strongest stem-loop requirement with asymmetric flanking stems.
                A3A (blue) is moderate. A3B (orange) is shallow/broad. Neither (red) has the weakest peak — consistent with structure-independent APOBEC1 recognition.
            </div>"""

        # Also add zoomed ±10nt overlay
        mfe_zoom = PROJECT_ROOT / "experiments/multi_enzyme/outputs/structure_analysis_v3/multi_enzyme_unpaired_mfe_zoom10.png"
        if mfe_zoom.exists():
            pairing_img += f"""
            <h3>Unpaired Probability — All Enzymes (±10nt, Per-Position)</h3>
            <p class="section-desc">Raw per-position unpaired fraction within ±10nt. Shows stem-loop structure at single-nucleotide resolution per enzyme.</p>
            {embed_png(mfe_zoom)}"""

        html += f"""
        <div class="section">
            <h2>Cross-Enzyme Structure Comparison</h2>
            <p class="section-desc">Structural properties of edited sites across enzyme categories.</p>
            <table>
                <thead><tr><th>Enzyme</th><th>n</th><th>Unpaired %</th><th>Mean RLP</th><th>Mean Loop Size</th></tr></thead>
                <tbody>{struct_rows}</tbody>
            </table>
            {pairing_img}
        </div>
        """

    # Unified Network Results
    unified_v1 = load_json(PATHS.get("unified_v1", ""))
    if unified_v1 and "v1" in unified_v1:
        v1_folds = unified_v1["v1"]
        unified_rows = ""
        per_enz_ref = {"A3A": 0.880, "A3B": 0.971, "A3G": 0.893, "A3A_A3G": 0.935, "Neither": 0.829}
        for enz in ["A3A", "A3B", "A3G", "A3A_A3G", "Neither", "Unknown"]:
            vals = [r.get("per_enzyme_auroc", {}).get(enz, 0) for r in v1_folds]
            if not any(v > 0 for v in vals):
                continue
            unified_auroc = np.mean(vals)
            ref = per_enz_ref.get(enz, 0)
            delta = unified_auroc - ref if ref > 0 else 0
            delta_cls = 'class="best-cell"' if delta > 0.02 else ('class="worst-cell"' if delta < -0.02 else "")
            unified_rows += f"""<tr>
                <td><strong>{ENZYME_DISPLAY.get(enz, enz)}</strong></td>
                <td>{ref:.3f}</td>
                <td>{unified_auroc:.3f}</td>
                <td {delta_cls}>{delta:+.3f}</td>
            </tr>"""

        overall = np.mean([r["overall_auroc"] for r in v1_folds])
        enz_acc = np.mean([r["enzyme_accuracy"] for r in v1_folds])

        html += f"""
        <div class="section">
            <h2>Unified Multi-Enzyme Network (Shared Learning)</h2>
            <p class="section-desc">Single network trained on ALL enzymes jointly. Shared backbone learns common editing signals, enzyme-specific heads learn differences.</p>
            <div class="kpi-grid">
                <div class="kpi"><div class="kpi-value">{overall:.3f}</div><div class="kpi-label">Overall Binary AUROC</div></div>
                <div class="kpi"><div class="kpi-value">{enz_acc:.1%}</div><div class="kpi-label">Enzyme Classification Accuracy</div></div>
            </div>
            <h3>Per-Enzyme: Unified vs Per-Enzyme Training</h3>
            <table>
                <thead><tr><th>Enzyme</th><th>Per-Enzyme (EditRNA+H)</th><th>Unified V1</th><th>Delta</th></tr></thead>
                <tbody>{unified_rows}</tbody>
            </table>
            <div class="callout callout-success">
                <strong>Shared learning helps data-scarce enzymes:</strong>
                Neither gains +0.10 AUROC (0.829 → 0.931), A3G gains +0.06 (0.893 → 0.953).
                The shared backbone transfers structural editing patterns from data-rich enzymes (A3A, A3B)
                to categories with few samples. A3B loses 0.05 — its unique patterns get diluted in the shared representation.
            </div>
        </div>
        """

    # Motif comparison
    motif_rows = ""
    for enz in enzymes:
        enz_ucc = data["ucc"].get(enz, {})
        di = enz_ucc.get("dinucleotide", {})
        if not di:
            continue
        n = enz_ucc.get("n_sites", 0)
        tc = di.get("UC_pct", 0)
        cc = di.get("CC_pct", 0)
        ac = di.get("AC_pct", di.get("AC", 0))
        if isinstance(ac, int) and n > 0:
            ac = ac / n * 100
        gc_val = di.get("GC", 0)
        if isinstance(gc_val, int) and n > 0:
            gc_pct = gc_val / n * 100
        else:
            gc_pct = 0
        top_tri = enz_ucc.get("top_trinucleotides", [])
        top_tri_str = ", ".join(f"{t[0]}({t[1]})" for t in top_tri[:3])
        motif_rows += f"""<tr>
            <td><strong>{ENZYME_DISPLAY.get(enz, enz)}</strong></td>
            <td>{n}</td>
            <td>{tc:.1f}%</td>
            <td>{cc:.1f}%</td>
            <td>{top_tri_str}</td>
        </tr>"""

    if motif_rows:
        html += f"""
        <div class="section">
            <h2>Cross-Enzyme Motif Comparison</h2>
            <p class="section-desc">Dinucleotide and trinucleotide preferences across Levanon enzyme categories.</p>
            <table>
                <tr><th>Enzyme</th><th>n</th><th>TC %</th><th>CC %</th><th>Top 3 Trinucleotides</th></tr>
                {motif_rows}
            </table>
            <div class="callout callout-info">
                <strong>Three distinct editing programs:</strong>
                A3A = TC-dominant (84%), A3G = CC-dominant (93%), A3A_A3G = mixed TC/CC (33%/65%).
                "Neither" shows no strong preference (TC=24%, CC=35%), consistent with a non-APOBEC3 editor.
            </div>
        </div>
        """

    # Tissue class comparison
    tissue_rows = ""
    for enz in enzymes:
        tc_key = f"{enz}_tissue_class"
        tc = data["apobec1"].get(tc_key, {})
        if not tc:
            continue
        total = sum(tc.values())
        top = max(tc, key=tc.get)
        top_pct = tc[top] / total * 100
        dist = ", ".join(f"{k}={v}" for k, v in sorted(tc.items(), key=lambda x: -x[1])[:3])
        tissue_rows += f"""<tr>
            <td><strong>{ENZYME_DISPLAY.get(enz, enz)}</strong></td>
            <td>{top} ({top_pct:.0f}%)</td>
            <td style="font-size:0.85rem">{dist}</td>
        </tr>"""

    if tissue_rows:
        html += f"""
        <div class="section">
            <h2>Cross-Enzyme Tissue Distribution</h2>
            <p class="section-desc">Dominant tissue class per enzyme, based on GTEx 54-tissue editing rate clustering.</p>
            <table>
                <tr><th>Enzyme</th><th>Dominant Tissue Class</th><th>Top 3 Classes</th></tr>
                {tissue_rows}
            </table>
            <div class="callout callout-success">
                <strong>Key biological insight:</strong>
                A3A sites are predominantly blood-specific (68%), A3G sites are testis-specific (52%),
                while "Neither" sites are intestine-specific (31%) -- consistent with APOBEC1, which is
                highly expressed in the small intestine.
            </div>
        </div>
        """

    # Tissue clustering
    tc_data = data.get("tissue", {})
    clusters = tc_data.get("tissue_clusters", {})
    if clusters:
        by_cluster = defaultdict(list)
        for tissue, cl in clusters.items():
            by_cluster[cl].append(tissue.replace("_", " ").title())

        cluster_names = {
            "1": "Brain / Neural",
            "2": "Immune / Reproductive",
            "3": "GI Tract",
            "4": "Broad / Mesenchymal",
        }

        cluster_rows = ""
        for cl in sorted(by_cluster.keys()):
            tissues = by_cluster[cl]
            name = cluster_names.get(cl, f"Cluster {cl}")
            tissue_str = ", ".join(sorted(tissues)[:8])
            if len(tissues) > 8:
                tissue_str += f", ... (+{len(tissues)-8} more)"
            cluster_rows += f"<tr><td><strong>{name}</strong></td><td>{len(tissues)}</td><td style='font-size:0.85rem'>{tissue_str}</td></tr>"

        html += f"""
        <div class="section">
            <h2>Tissue Clustering (54 GTEx Tissues)</h2>
            <p class="section-desc">Hierarchical clustering of tissues based on editing rate profiles across all 636 Levanon sites.</p>
            <table>
                <tr><th>Cluster</th><th>n_tissues</th><th>Representative Tissues</th></tr>
                {cluster_rows}
            </table>
        </div>
        """

    # ClinVar summary
    html += """
    <div class="section">
        <h2>Cross-Enzyme ClinVar Summary</h2>
        <p class="section-desc">Pathogenic enrichment at predicted editing sites across 1.69M ClinVar C-to-U variants.</p>
        <table>
            <tr><th>Enzyme</th><th>Model</th><th>Best OR</th><th>Threshold</th><th>p-value</th><th>Context</th></tr>
            <tr>
                <td><strong>A3A</strong></td><td>GB_Full</td>
                <td class="best-cell">1.279</td><td>P &ge; 0.5</td><td>&lt;1e-138</td><td>All C-to-U</td>
            </tr>
            <tr>
                <td><strong>A3B</strong></td><td>GB_Hand</td>
                <td>1.082</td><td>P &ge; 0.3</td><td>1.5e-24</td><td>All C-to-U</td>
            </tr>
            <tr>
                <td><strong>A3B</strong></td><td>GB_Hand (calibrated)</td>
                <td>1.552</td><td>cal. 0.008</td><td>3.0e-31</td><td>All C-to-U</td>
            </tr>
            <tr>
                <td><strong>A3G</strong></td><td>GB_MotifOnly</td>
                <td class="best-cell">1.759</td><td>t=0.4</td><td>&lt;1e-300</td><td>CC-context only</td>
            </tr>
        </table>
        <div class="callout callout-success">
            <strong>All three APOBEC enzymes show ClinVar pathogenic enrichment</strong> at predicted editing sites.
            A3G has the strongest signal in CC-context (OR=1.76), A3A has the most robust all-context signal (OR=1.28),
            and A3B shows meaningful enrichment after Bayesian calibration (OR=1.55).
        </div>
    </div>
    """

    # Deaminase Family Comparison
    family_md = PROJECT_ROOT / "paper/deaminase_family_comparison.md"
    if family_md.exists():
        with open(family_md) as f:
            family_content = f.read()
        # Extract the HTML table
        import re as _re
        table_match = _re.search(r'(<table>.*?</table>)', family_content, _re.DOTALL)
        family_table = table_match.group(1) if table_match else ""
        # Extract narrative
        narr_match = _re.search(r'## Narrative.*?\n\n(.*?)(?=\n---|\n## Key References)', family_content, _re.DOTALL)
        narrative = narr_match.group(1).strip() if narr_match else ""
        # Convert markdown bold to HTML
        narrative = _re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', narrative)
        narrative_html = "".join(f"<p>{p.strip()}</p>" for p in narrative.split("\n\n") if p.strip())
        # Connection section
        conn_match = _re.search(r'## Connection to This Study\n\n(.*?)$', family_content, _re.DOTALL)
        connection = conn_match.group(1).strip() if conn_match else ""
        connection = _re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', connection)
        connection_html = "".join(f"<p>{p.strip()}</p>" for p in connection.split("\n\n") if p.strip())

        if family_table:
            html += f"""
            <div class="section">
                <h2>C-to-U Deaminase Family Comparison</h2>
                <p class="section-desc">Comprehensive comparison of all 11 human AID/APOBEC family members — substrate preferences, motifs, structural requirements, and disease relevance.</p>
                <div style="overflow-x:auto;font-size:0.82em">{family_table}</div>
                <h3>Evolutionary Context</h3>
                {narrative_html}
                <h3>Connection to This Study</h3>
                {connection_html}
            </div>
            """

    # Embedding Visualization
    emb_umap = PROJECT_ROOT / "experiments/multi_enzyme/outputs/embedding_viz/umap_grid.png"
    emb_pca = PROJECT_ROOT / "experiments/multi_enzyme/outputs/embedding_viz/pca_grid.png"
    if emb_umap.exists():
        html += f"""
        <div class="section">
            <h2>Embedding Space Visualization (Unified Network)</h2>
            <p class="section-desc">256-dim shared backbone representations projected to 2D. Each column shows a different semantic coloring: enzyme identity, edited/not, motif context, and structural state.</p>
            <h3>UMAP</h3>
            {embed_png(emb_umap)}
            <h3>PCA (58.4% variance in first 2 components)</h3>
            {embed_png(emb_pca)}
            <div class="callout callout-info">
                <strong>What the embeddings learned:</strong>
                Enzyme identity emerges naturally — A3A (blue), A3B (orange), and A3G (green) form distinct clusters without explicit enzyme supervision in the shared backbone.
                The motif context (TC/CC) aligns with enzyme clusters, confirming motif is a key discriminator.
                Unpaired sites concentrate in positive regions, consistent with structure being the primary editing determinant.
            </div>
        </div>
        """

    # New analytical sections (March 2026)
    html += build_mutation_coupling_section()
    html += build_germline_mutation_section()
    html += build_aid_apobec_comparison_section()
    html += build_unified_interpretability_section()

    # Key findings summary
    html += """
    <div class="section">
        <h2>Key Findings Summary</h2>
        <div style="display:grid;grid-template-columns:1fr 1fr;gap:16px">
            <div class="callout callout-info" style="margin:0">
                <strong>1. Three distinct editing programs</strong><br>
                A3A = TC motif + moderate 3' preference<br>
                A3B = structure-only (no motif preference)<br>
                A3G = CC motif + extreme 3' tetraloop bias
            </div>
            <div class="callout callout-success" style="margin:0">
                <strong>2. Structure is the primary predictor for APOBEC3 enzymes</strong><br>
                A3A and A3B: StructOnly AUROC &gt; 0.93, structure carries most signal.<br>
                A3G: structure leads (0.812 vs motif 0.706).<br>
                Both/Neither/Unknown: motif is more discriminative than structure.
            </div>
            <div class="callout callout-warning" style="margin:0">
                <strong>3. "Neither" sites are consistent with APOBEC1</strong><br>
                ACA trinucleotide (APOBEC1 signature), AU-rich mooring (p=1.4e-6), intestine top tissue.
                However, GI vs immune tissue enrichment is not significant (p=0.82).
            </div>
            <div class="callout callout-info" style="margin:0">
                <strong>4. Universal ClinVar enrichment</strong><br>
                All three APOBEC3 enzymes show pathogenic enrichment at predicted sites.
                This supports C-to-U editing as a contributor to disease across multiple enzymes.
            </div>
        </div>
    </div>
    """

    # Multi-enzyme clinical interpretation
    html += """
    <div class="section">
        <h2>Multi-Enzyme Clinical Interpretation</h2>

        <h3>Cross-Enzyme Disease Implications</h3>
        <p>All three APOBEC3 enzymes show statistically significant ClinVar pathogenic enrichment at predicted editing sites,
        but through distinct biological mechanisms:</p>

        <table>
            <thead><tr><th>Enzyme</th><th>Primary Tissue</th><th>Best OR</th><th>Mechanism</th><th>Clinical Domain</th></tr></thead>
            <tbody>
            <tr><td><strong>A3A</strong></td><td>Blood (68%)</td><td>1.279</td><td>Structural vulnerability in hairpin loops</td><td>Hematologic/inflammatory</td></tr>
            <tr><td><strong>A3B</strong></td><td>Broad/epithelial</td><td>1.552 (cal.)</td><td>Extended sequence context recognition</td><td>Solid tumors</td></tr>
            <tr><td><strong>A3G</strong></td><td>Testis (52%)</td><td>1.759 (CC)</td><td>CC tetraloop structural constraint</td><td>Germline/reproductive</td></tr>
            </tbody>
        </table>

        <h3>Key Clinical Findings</h3>
        <div style="display:grid;grid-template-columns:1fr 1fr;gap:12px">
            <div class="callout callout-success" style="margin:0">
                <strong>89% of pathogenic editing sites create stop codons</strong><br>
                32 of 36 pathogenic editing sites produce premature termination. The C-to-U change at CAG/CGA/CAA codons
                creates TAG/TGA/TAA stops.
            </div>
            <div class="callout callout-warning" style="margin:0">
                <strong>SDHB: definitive clinical case</strong><br>
                Endogenous editing (1.2% in blood), stopgain in tumor suppressor, pathogenic for paraganglioma.
                The only confirmed case requiring clinical attention for misannotation risk.
            </div>
            <div class="callout callout-info" style="margin:0">
                <strong>Cancer gene depletion validates the model</strong><br>
                GoF oncogene variants score LOWER (OR=0.804) because catalytic residues are in structured RNA regions
                inaccessible to APOBEC. LoF tumor suppressors show the expected enrichment.
            </div>
            <div class="callout callout-info" style="margin:0">
                <strong>Germline mutagenesis hypothesis: rejected</strong><br>
                Testis-specific editing sites show LOWER variant density. APOBEC-motif germline variants trend toward
                depletion (OR=0.80), consistent with purifying selection.
            </div>
        </div>

        <h3>Clinical Recommendations</h3>
        <table>
            <thead><tr><th>#</th><th>Recommendation</th><th>Rationale</th></tr></thead>
            <tbody>
            <tr><td>1</td><td>Flag C&gt;T VUS at predicted editing sites</td><td>148 VUS overlap known editing sites; may represent normal editing, not germline mutations</td></tr>
            <tr><td>2</td><td>Confirm RNA-detected variants with DNA sequencing</td><td>SDHB editing (1.2% in blood) could be misreported as a pathogenic mutation</td></tr>
            <tr><td>3</td><td>Consider tissue context in variant interpretation</td><td>A3A blood-specific editing has different implications than A3G testis-specific editing</td></tr>
            <tr><td>4</td><td>Prioritize stopgain editing sites for clinical review</td><td>19 stopgain editing events (SDHB, APOB, RHEB) represent the highest-impact cases</td></tr>
            </tbody>
        </table>
    </div>
    """

    return html


def generate_report():
    print("Loading data...")
    data = load_all_data()

    print("Computing structure statistics...")
    struct_stats = compute_structure_stats(data["loop_pos"])

    # Override A3A with v1 pipeline data (correct dataset)
    if data.get("a3a_v1_loop_pos"):
        a3a_v1_stats = compute_structure_stats(data["a3a_v1_loop_pos"])
        if "A3A" in a3a_v1_stats:
            struct_stats["A3A"] = a3a_v1_stats["A3A"]
            print(f"  A3A overridden with v1 pipeline: pos unpaired={a3a_v1_stats['A3A']['pos']['unpaired_pct']:.1f}%")

    print("Generating HTML...")

    enzymes_order = ["A3A", "A3B", "A3G", "A3A_A3G", "Neither", "Unknown", "A4"]
    tab_ids = ["a3a", "a3b", "a3g", "both", "neither", "unknown", "a4", "multi"]
    tab_labels = ["A3A", "A3B", "A3G", "Both (A3A+A3G)", "Neither", "Unknown", "A4", "Multi-Enzyme"]

    # Radio inputs (must be before .tabs and .content in DOM)
    radios = ""
    for i, tid in enumerate(tab_ids):
        checked = " checked" if i == 0 else ""
        radios += f'<input type="radio" name="tabs" id="tab-{tid}"{checked}>\n'

    # Tab labels
    tab_labels_html = ""
    for tid, label in zip(tab_ids, tab_labels):
        tab_labels_html += f'<label for="tab-{tid}">{label}</label>\n'

    # Tab contents
    enzyme_contents = ""
    for enz, tid in zip(enzymes_order, tab_ids[:7]):
        enzyme_contents += f'<div class="tab-content content-{tid}">\n'
        enzyme_contents += build_enzyme_tab(enz, data, struct_stats)
        enzyme_contents += '</div>\n'

    # Multi-enzyme tab
    multi_content = f'<div class="tab-content content-multi">\n'
    multi_content += build_multi_enzyme_tab(data, struct_stats)
    multi_content += '</div>\n'

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Multi-Enzyme APOBEC RNA Editing Report</title>
    <style>
{html_css()}
    </style>
</head>
<body>

{html_header()}

{radios}

<div class="tabs">
    <div class="container">
        {tab_labels_html}
    </div>
</div>

<div class="content">
    <div class="container">
        {enzyme_contents}
        {multi_content}
    </div>
</div>

</body>
</html>"""

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_FILE.write_text(html)
    size_kb = REPORT_FILE.stat().st_size / 1024
    print(f"Report written to {REPORT_FILE} ({size_kb:.0f} KB)")


if __name__ == "__main__":
    generate_report()
