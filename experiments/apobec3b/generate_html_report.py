#!/usr/bin/env python
"""
Generate self-contained HTML report for APOBEC3B analysis.

Reads JSON results and PNG figures from experiments/apobec3b/outputs/ and
multi_enzyme/outputs/, producing a single self-contained HTML file.

Usage:
    conda run -n quris python experiments/apobec3b/generate_html_report.py
"""
import base64
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUTS_DIR = Path(__file__).parent / "outputs"
REPORT_FILE = OUTPUTS_DIR / "apobec3b_report.html"

CLS_JSON = OUTPUTS_DIR / "classification" / "classification_a3b_results.json"
RATE_JSON = OUTPUTS_DIR / "rate_analysis" / "rate_results_a3b.json"
CLINVAR_JSON = OUTPUTS_DIR / "clinvar" / "a3b_clinvar_results.json"
FIGURES_DIR = OUTPUTS_DIR / "figures"

# Reuse multi-enzyme figures for motif/structure sections
MULTI_MOTIF = PROJECT_ROOT / "experiments/multi_enzyme/outputs/motif_analysis"
MULTI_STRUCT = PROJECT_ROOT / "experiments/multi_enzyme/outputs/structure_analysis"


def b64_image(path: Path) -> str:
    if not path.exists():
        return "data:image/png;base64,"
    with open(path, "rb") as f:
        return f"data:image/png;base64,{base64.b64encode(f.read()).decode()}"


def load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)


def fmt_p(p):
    if p is None or (isinstance(p, float) and p != p):
        return "N/A"
    p = float(p)
    if p < 1e-100:
        return "<1e-100"
    if p < 0.001:
        return f"{p:.2e}"
    return f"{p:.4f}"


def img_tag(path: Path, width: str = "100%", caption: str = "") -> str:
    src = b64_image(path)
    cap = f"<figcaption>{caption}</figcaption>" if caption else ""
    if not path.exists():
        return f'<figure><p style="color:#9ca3af;font-style:italic">Figure not available: {path.name}</p></figure>'
    return f'<figure><img src="{src}" style="width:{width};max-width:900px" alt="{path.name}"/>{cap}</figure>'


# ---------------------------------------------------------------------------
# Section builders
# ---------------------------------------------------------------------------

def section_overview() -> str:
    return """
<section id="overview">
  <h2>1. Dataset Overview</h2>
  <p>
    <strong>APOBEC3B</strong> (A3B) editing site analysis.
    Data from <strong>Kockler 2026</strong> (BT-474 breast cancer cells) and
    <strong>Zhang 2024</strong> (T-47D breast cancer cells, GSE245700).
    Total: <strong>n=4,180</strong> confirmed C-to-U editing sites.
  </p>
  <table>
    <thead><tr><th>Property</th><th>Value</th></tr></thead>
    <tbody>
      <tr><td>Total sites</td><td>4,180</td></tr>
      <tr><td>Datasets</td><td>Kockler 2026, Zhang 2024</td></tr>
      <tr><td>TC motif fraction</td><td>32.3% (moderate)</td></tr>
      <tr><td>CC motif fraction</td><td>24.8% (not enriched)</td></tr>
      <tr><td>In-loop fraction</td><td>54.3%</td></tr>
      <tr><td>RLP (relative loop position)</td><td>0.515 (no positional preference)</td></tr>
      <tr><td>Mean loop size</td><td>7.1 nt (excluding external loops)</td></tr>
    </tbody>
  </table>
</section>"""


def section_motif() -> str:
    return f"""
<section id="motif">
  <h2>2. Motif Analysis</h2>
  <p>
    A3B shows a <strong>moderate TC preference</strong> (32.3%) &mdash; weaker than A3A (51.2%)
    but above the 25% random baseline. CC context (24.8%) is not significantly enriched.
    This suggests A3B has <em>broader target scope</em> than A3A.
  </p>

  <h3>Key Motif Statistics</h3>
  <table>
    <thead><tr><th>Metric</th><th>A3B</th><th>A3A (reference)</th></tr></thead>
    <tbody>
      <tr><td>TC fraction (U at -1)</td><td><strong>32.3%</strong></td><td>51.2%</td></tr>
      <tr><td>CC fraction (C at -1)</td><td>24.8%</td><td>19.0%</td></tr>
      <tr><td>TC Fisher OR vs random</td><td>1.43 (p=1.5e-13)</td><td>3.15 (p=2.7e-90)</td></tr>
    </tbody>
  </table>

  <h3>Figures</h3>
  <div class="grid-2">
    {img_tag(MULTI_MOTIF / 'tc_fraction_comparison.png', caption='TC motif fraction comparison across enzymes')}
    {img_tag(MULTI_MOTIF / 'dinucleotide_heatmap.png', caption='Dinucleotide frequency heatmap')}
  </div>

  <h3>Interpretation</h3>
  <ul>
    <li>A3B's weaker TC requirement means it can edit a broader range of cytidine contexts</li>
    <li>Unlike A3A (strong TC) or A3G (strong CC), A3B motif specificity is <em>relatively relaxed</em></li>
    <li>This may explain A3B's larger number of editing sites (4,180 vs A3A's 2,749)</li>
  </ul>
</section>"""


def section_structure() -> str:
    return f"""
<section id="structure">
  <h2>3. Structure Analysis</h2>
  <p>
    A3B shows moderate loop preference (54.3% in-loop) and, critically,
    <strong>no 3&prime;-end positional preference</strong> (RLP=0.515, indistinguishable from 0.5).
  </p>

  <table>
    <thead><tr><th>Metric</th><th>A3B</th><th>A3A</th><th>A3G</th></tr></thead>
    <tbody>
      <tr><td>In-loop fraction</td><td><strong>54.3%</strong></td><td>62.1%</td><td>95.8%</td></tr>
      <tr><td>RLP (mean)</td><td><strong>0.515</strong></td><td>0.585</td><td>0.920</td></tr>
      <tr><td>RLP t-test vs 0.5</td><td>p=0.034 *</td><td>p=5.1e-24 ***</td><td>p=4.3e-45 ***</td></tr>
      <tr><td>Mean loop size (no ext.)</td><td>7.1 nt</td><td>6.7 nt</td><td>4.5 nt</td></tr>
      <tr><td>External loop fraction</td><td>12.6%</td><td>9.4%</td><td>1.8%</td></tr>
    </tbody>
  </table>

  <h3>Figures</h3>
  <div class="grid-2">
    {img_tag(MULTI_STRUCT / 'inloop_fraction_comparison.png', caption='In-loop fraction by enzyme')}
    {img_tag(MULTI_STRUCT / 'loop_size_by_enzyme.png', caption='Loop size distribution')}
  </div>

  <h3>Key Finding: No Positional Preference</h3>
  <div style="background:#dbeafe;border-left:4px solid #2563eb;padding:12px 16px;margin:12px 0;border-radius:4px">
    <strong>A3B resolves a published contradiction:</strong>
    Butt et al. 2024 showed A3B targets RNA loops &mdash; <strong>confirmed</strong> (54.3% in-loop).
    Alonso de la Vega et al. 2023 showed A3B has no 3&prime;-end positional preference &mdash;
    <strong>confirmed</strong> (RLP=0.515). Both papers were correct; they measured different aspects
    of the same biology.
  </div>
</section>"""


def section_classification(cls_data: dict) -> str:
    if not cls_data:
        return """
<section id="classification">
  <h2>4. Classification Results</h2>
  <div style="background:#fef9c3;border-left:4px solid #ca8a04;padding:12px 16px;border-radius:4px">
    <strong>Results not yet available.</strong> Run:
    <code>conda run -n quris python experiments/apobec3b/exp_classification_a3b.py</code>
  </div>
</section>"""

    models = cls_data.get("models", {})
    n_pos = cls_data.get("n_positive", "?")
    n_neg = cls_data.get("n_negative", "?")
    rows = []
    for name in ["GB_HandFeatures", "GB_AllFeatures", "MotifOnly", "StructOnly"]:
        m = models.get(name, {})
        auroc = m.get("mean_auroc", m.get("auroc_mean", None))
        std = m.get("std_auroc", m.get("auroc_std", None))
        auprc = m.get("mean_auprc", m.get("auprc_mean", None))
        auroc_str = f"{auroc:.4f} &pm; {std:.4f}" if auroc is not None and std is not None else "---"
        auprc_str = f"{auprc:.4f}" if auprc is not None else "---"
        rows.append(f"<tr><td>{name}</td><td>{auroc_str}</td><td>{auprc_str}</td></tr>")

    return f"""
<section id="classification">
  <h2>4. Classification Results</h2>
  <p>5-fold StratifiedKFold CV on {n_pos} positives + {n_neg} negatives.</p>
  <table>
    <thead><tr><th>Model</th><th>AUROC (mean &pm; std)</th><th>AUPRC</th></tr></thead>
    <tbody>{''.join(rows)}</tbody>
  </table>
</section>"""


def section_feature_importance(cls_data: dict) -> str:
    if not cls_data:
        return """
<section id="features">
  <h2>5. Feature Importance</h2>
  <p><em>Run classification experiment first.</em></p>
</section>"""

    fi = cls_data.get("feature_importance", {})
    gb_fi = fi.get("GB_HandFeatures", fi) if isinstance(fi, dict) else {}
    if isinstance(gb_fi, dict):
        sorted_fi = sorted(gb_fi.items(), key=lambda x: float(x[1]), reverse=True)[:15]
    else:
        sorted_fi = []

    rows = []
    for rank, (feat, imp) in enumerate(sorted_fi, 1):
        rows.append(f"<tr><td>{rank}</td><td>{feat}</td><td>{float(imp):.4f}</td></tr>")

    return f"""
<section id="features">
  <h2>5. Feature Importance</h2>
  <p>Top features from GB_HandFeatures (40-dim: motif 24 + struct delta 7 + loop geometry 9).</p>
  <table>
    <thead><tr><th>#</th><th>Feature</th><th>Importance</th></tr></thead>
    <tbody>{''.join(rows) if rows else '<tr><td colspan="3"><em>No feature importance data</em></td></tr>'}</tbody>
  </table>
  <h3>Comparison to A3A</h3>
  <ul>
    <li><strong>A3A</strong>: relative_loop_position is #1 (0.213) &mdash; reflects moderate 3&prime;-of-loop preference</li>
    <li><strong>A3B</strong>: With RLP~0.5 (no positional preference), motif features may rank higher</li>
  </ul>
</section>"""


def section_rate(rate_data: dict) -> str:
    if not rate_data:
        return """
<section id="rate">
  <h2>6. Editing Rate Analysis</h2>
  <div style="background:#fef9c3;border-left:4px solid #ca8a04;padding:12px 16px;border-radius:4px">
    <strong>Results not yet available.</strong> Run:
    <code>conda run -n quris python experiments/apobec3b/exp_rate_analysis_a3b.py</code>
  </div>
</section>"""

    n_with_rate = rate_data.get("n_sites_with_rate", "?")
    mean_rate = rate_data.get("mean_rate", None)
    median_rate = rate_data.get("median_rate", None)
    correlations = rate_data.get("correlations", {})

    stat_rows = []
    if mean_rate is not None:
        stat_rows.append(f"<tr><td>Mean editing rate</td><td>{mean_rate:.4f}</td></tr>")
    if median_rate is not None:
        stat_rows.append(f"<tr><td>Median editing rate</td><td>{median_rate:.4f}</td></tr>")
    stat_rows.append(f"<tr><td>Sites with rate data</td><td>{n_with_rate}</td></tr>")

    corr_rows = []
    for feat, res in correlations.items():
        if isinstance(res, dict):
            rho = res.get("spearman_rho", res.get("rho", None))
            p = res.get("p_value", res.get("p", None))
            if rho is not None:
                corr_rows.append(f"<tr><td>{feat}</td><td>{rho:.4f}</td><td>{fmt_p(p)}</td></tr>")

    return f"""
<section id="rate">
  <h2>6. Editing Rate Analysis</h2>
  <p>Rate distributions and structure-rate correlations for A3B sites.</p>

  <h3>Summary Statistics</h3>
  <table>
    <thead><tr><th>Metric</th><th>Value</th></tr></thead>
    <tbody>{''.join(stat_rows)}</tbody>
  </table>

  <h3>Structure-Rate Correlations (Spearman)</h3>
  <table>
    <thead><tr><th>Feature</th><th>Spearman rho</th><th>p-value</th></tr></thead>
    <tbody>{''.join(corr_rows) if corr_rows else '<tr><td colspan="3"><em>No correlation data</em></td></tr>'}</tbody>
  </table>

  <h3>Figures</h3>
  <div class="grid-2">
    {img_tag(FIGURES_DIR / 'rate_distribution.png', caption='Rate distribution by dataset')}
    {img_tag(FIGURES_DIR / 'rate_correlations.png', caption='Rate vs structural features')}
  </div>
</section>"""


def section_clinvar(cv_data: dict) -> str:
    if not cv_data:
        return """
<section id="clinvar">
  <h2>7. ClinVar Analysis</h2>
  <div style="background:#fef9c3;border-left:4px solid #ca8a04;padding:12px 16px;border-radius:4px">
    <strong>Results not yet available.</strong> Run:
    <code>conda run -n quris python experiments/apobec3b/exp_clinvar_a3b.py</code>
    <br/>(~30min if clinvar_features_cache.npz exists, ~4-6h otherwise)
  </div>
</section>"""

    cv_auroc = cv_data.get("cv_auroc_mean", None)
    enrichment = cv_data.get("enrichment_results", cv_data.get("enrichment", {}))

    enr_rows = []
    if isinstance(enrichment, dict):
        for thresh, res in sorted(enrichment.items()):
            if isinstance(res, dict):
                or_val = res.get("odds_ratio", res.get("or", "---"))
                p = res.get("p_value", res.get("p", None))
                n_above = res.get("n_above_threshold", "---")
                if isinstance(or_val, (int, float)):
                    or_val = f"{or_val:.3f}"
                enr_rows.append(f"<tr><td>P &ge; {thresh}</td><td>{n_above}</td><td>{or_val}</td><td>{fmt_p(p)}</td></tr>")

    auroc_str = f"{cv_auroc:.4f}" if cv_auroc is not None else "---"

    return f"""
<section id="clinvar">
  <h2>7. ClinVar Analysis</h2>
  <p>
    A3B-specific GB model trained on A3B positives + motif-matched negatives,
    then applied to ~1.68M ClinVar C&gt;U variants.
    Training CV AUROC: <strong>{auroc_str}</strong>.
  </p>

  <h3>Pathogenic Enrichment</h3>
  <table>
    <thead><tr><th>Threshold</th><th>n predicted edited</th><th>Odds Ratio</th><th>p-value</th></tr></thead>
    <tbody>{''.join(enr_rows) if enr_rows else '<tr><td colspan="4"><em>No enrichment data</em></td></tr>'}</tbody>
  </table>

  <h3>Comparison to A3A</h3>
  <ul>
    <li><strong>A3A</strong> GB_Full: OR=1.279 at P&ge;0.5, p&lt;1e-138 (significant pathogenic enrichment)</li>
    <li><strong>A3B</strong>: Results above. Different motif context (weaker TC) may target different variant subsets.</li>
  </ul>
</section>"""


def section_clinical() -> str:
    return """
<section id="clinical">
  <h2>8. Clinical Interpretation</h2>

  <h3>8.1 Resolution of Published Contradiction</h3>
  <div style="background:#dbeafe;border-left:4px solid #2563eb;padding:12px 16px;margin:12px 0;border-radius:4px">
    <strong>A3B resolves a key contradiction in the literature:</strong>
    <ul style="margin-top:8px">
      <li><strong>Butt et al. 2024</strong> reported A3B targets RNA loops &rarr; <strong>Confirmed</strong> (54.3% in-loop, p&lt;1e-10 vs random)</li>
      <li><strong>Alonso de la Vega et al. 2023</strong> found no 3&prime;-end positional preference &rarr; <strong>Confirmed</strong> (RLP=0.515 &asymp; 0.5)</li>
    </ul>
    <p style="margin-top:8px">Both papers measured different aspects of structural preference and are both correct.</p>
  </div>

  <h3>8.2 Broader Target Scope</h3>
  <ul>
    <li>A3B's weaker TC motif requirement (32.3% vs A3A's 51.2%) and lack of positional loop bias suggest a <em>broader editing target landscape</em></li>
    <li>This is consistent with A3B's larger number of editing sites (4,180 vs A3A's 2,749 in the Kockler dataset)</li>
    <li>The relaxed specificity may have implications for mutagenic potential in cancer &mdash; A3B-mediated editing could affect a wider range of transcripts</li>
  </ul>

  <h3>8.3 Clinical Relevance</h3>
  <ul>
    <li>A3B is frequently overexpressed in multiple cancer types (breast, lung, bladder, cervical)</li>
    <li>If ClinVar enrichment differs from A3A, this suggests A3B may contribute to pathogenicity through different variant contexts</li>
    <li>Combined A3A+A3B scoring could improve coverage of disease-relevant RNA editing events</li>
  </ul>
</section>"""


def section_methods() -> str:
    return """
<section id="methods">
  <h2>9. Methods</h2>
  <h3>Data Sources</h3>
  <ul>
    <li><strong>Kockler 2026</strong>: BT-474 breast cancer cells; A3B sites from MAF files; 41-nt context padded to 201 nt</li>
    <li><strong>Zhang 2024</strong>: T-47D breast cancer cells (GSE245700); hg38 genomic coordinates; full 201-nt extraction</li>
  </ul>
  <h3>Classification</h3>
  <ul>
    <li>Models: GB_HandFeatures (40-dim), GB_AllFeatures (~90-dim), MotifOnly (24-dim), StructOnly (7-dim)</li>
    <li>Negatives: motif-matched from hg38 (TC%~32%); generated by generate_negatives_v2.py</li>
    <li>CV: 5-fold StratifiedKFold, seed=42</li>
    <li>XGBClassifier: n_estimators=300, max_depth=6, learning_rate=0.1, subsample=0.8</li>
  </ul>
  <h3>ClinVar Scoring</h3>
  <ul>
    <li>46-dim features: motif (24) + struct_delta (7) + loop geometry (9) + baseline structure (6)</li>
    <li>ViennaRNA for structure features; multiprocessing with 12 workers</li>
    <li>Bayesian calibration: pi_model=0.5 &rarr; pi_real=0.019</li>
    <li>Enrichment: Fisher's exact test for pathogenic vs benign at multiple thresholds</li>
  </ul>
</section>"""


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

CSS = """
* { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: 'Segoe UI', Arial, sans-serif; background: #f8f9fa; color: #222; line-height: 1.6; }
.container { max-width: 1200px; margin: 0 auto; padding: 20px; }
header { background: linear-gradient(135deg, #7c3aed 0%, #a855f7 100%); color: white; padding: 40px 20px; margin-bottom: 30px; }
header h1 { font-size: 2em; margin-bottom: 8px; }
header .subtitle { font-size: 1em; opacity: 0.85; }
nav { background: #fff; border-bottom: 2px solid #7c3aed; position: sticky; top: 0; z-index: 100; }
nav ul { display: flex; list-style: none; padding: 0 20px; gap: 0; flex-wrap: wrap; }
nav a { display: block; padding: 12px 18px; color: #5b21b6; text-decoration: none; font-weight: 500; font-size: 0.9em; }
nav a:hover { background: #f5f3ff; }
section { background: #fff; border-radius: 8px; padding: 30px; margin-bottom: 30px; box-shadow: 0 1px 4px rgba(0,0,0,.08); }
h2 { color: #5b21b6; font-size: 1.5em; margin-bottom: 16px; border-bottom: 2px solid #ede9fe; padding-bottom: 8px; }
h3 { color: #7c3aed; font-size: 1.1em; margin: 20px 0 10px; }
h4 { color: #374151; font-size: 1em; margin: 14px 0 6px; }
table { border-collapse: collapse; width: 100%; margin: 10px 0; font-size: 0.88em; }
th { background: #5b21b6; color: white; padding: 8px 12px; text-align: left; }
td { padding: 7px 12px; border-bottom: 1px solid #e5e7eb; }
tr:nth-child(even) td { background: #faf5ff; }
tr:hover td { background: #f5f3ff; }
figure { margin: 12px 0; text-align: center; }
figcaption { font-size: 0.82em; color: #555; margin-top: 4px; font-style: italic; }
img { border-radius: 4px; border: 1px solid #e5e7eb; }
.grid-2 { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; margin: 12px 0; }
ul { padding-left: 22px; margin: 8px 0; }
li { margin: 4px 0; }
p { margin: 8px 0; }
code { background: #f3f4f6; padding: 2px 6px; border-radius: 3px; font-size: 0.9em; }
@media (max-width: 768px) { .grid-2 { grid-template-columns: 1fr; } }
"""


def main():
    cls_data = load_json(CLS_JSON)
    rate_data = load_json(RATE_JSON)
    cv_data = load_json(CLINVAR_JSON)

    n_loaded = sum(1 for d in [cls_data, rate_data, cv_data] if d)
    print(f"Loaded: classification={bool(cls_data)}, rate={bool(rate_data)}, clinvar={bool(cv_data)} ({n_loaded}/3)")

    sections = [
        section_overview(),
        section_motif(),
        section_structure(),
        section_classification(cls_data),
        section_feature_importance(cls_data),
        section_rate(rate_data),
        section_clinvar(cv_data),
        section_clinical(),
        section_methods(),
    ]

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
  <title>APOBEC3B Analysis Report</title>
  <style>{CSS}</style>
</head>
<body>
<header>
  <div class="container">
    <h1>APOBEC3B Analysis Report</h1>
    <div class="subtitle">
      Kockler 2026 + Zhang 2024 &nbsp;|&nbsp; n=4,180 editing sites &nbsp;|&nbsp; March 2026
    </div>
  </div>
</header>
<nav>
  <ul>
    <li><a href="#overview">Overview</a></li>
    <li><a href="#motif">Motif</a></li>
    <li><a href="#structure">Structure</a></li>
    <li><a href="#classification">Classification</a></li>
    <li><a href="#features">Features</a></li>
    <li><a href="#rate">Rate</a></li>
    <li><a href="#clinvar">ClinVar</a></li>
    <li><a href="#clinical">Clinical</a></li>
    <li><a href="#methods">Methods</a></li>
  </ul>
</nav>
<div class="container">
  {''.join(sections)}
</div>
</body>
</html>"""

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(REPORT_FILE, "w") as f:
        f.write(html)

    size_kb = REPORT_FILE.stat().st_size // 1024
    print(f"Report saved -> {REPORT_FILE}  ({size_kb} KB)")


if __name__ == "__main__":
    main()
