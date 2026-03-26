#!/usr/bin/env python
"""
Generate self-contained HTML report for APOBEC3G analysis.

Reads JSON results and PNG figures from experiments/apobec3g/outputs/ and
multi_enzyme/outputs/, producing a single self-contained HTML file.

Usage:
    conda run -n quris python experiments/apobec3g/generate_html_report.py
"""
import base64
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUTS_DIR = Path(__file__).parent / "outputs"
REPORT_FILE = OUTPUTS_DIR / "apobec3g_report.html"

CLS_JSON = OUTPUTS_DIR / "classification" / "classification_a3g_results.json"
RATE_JSON = OUTPUTS_DIR / "rate_analysis" / "rate_results_a3g.json"
CLINVAR_JSON = OUTPUTS_DIR / "clinvar" / "a3g_clinvar_results.json"
FIGURES_DIR = OUTPUTS_DIR / "figures"

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

  <div style="background:#fef9c3;border-left:4px solid #ca8a04;padding:12px 16px;margin:0 0 16px 0;border-radius:4px">
    <strong>Small dataset caveat:</strong> This analysis is based on only <strong>n=119</strong>
    editing sites from a single study (Dang 2019, NK cells under hypoxia). Results should be
    interpreted as proof-of-concept. Statistical power is limited, especially for classification
    and ClinVar enrichment analysis.
  </div>

  <p>
    <strong>APOBEC3G</strong> (A3G) editing site analysis.
    Data from <strong>Dang 2019</strong> (GSE114519, NK cells, hypoxia condition).
    Total: <strong>n=119</strong> confirmed C-to-U editing sites.
  </p>
  <table>
    <thead><tr><th>Property</th><th>Value</th></tr></thead>
    <tbody>
      <tr><td>Total sites</td><td>119</td></tr>
      <tr><td>Dataset</td><td>Dang 2019 (NK_Hyp)</td></tr>
      <tr><td>CC motif fraction</td><td><strong>90.8%</strong> (dominant)</td></tr>
      <tr><td>TC motif fraction</td><td>9.2% (depleted)</td></tr>
      <tr><td>In-loop fraction</td><td><strong>95.8%</strong></td></tr>
      <tr><td>RLP (relative loop position)</td><td><strong>0.920</strong> (extreme 3&prime;-end)</td></tr>
      <tr><td>Mean loop size</td><td>4.5 nt (tetraloop preference)</td></tr>
    </tbody>
  </table>

  <div style="background:#fef9c3;border-left:4px solid #ca8a04;padding:12px 16px;margin:16px 0 0 0;border-radius:4px;font-size:0.9em">
    <strong>N-padding note:</strong> Dang 2019 provides 31-nt flanking sequences, padded with N
    to 201 nt for uniform processing. ViennaRNA treats N as fixed-unpaired, so
    <strong>loop geometry features are valid</strong> but <strong>absolute MFE values are not
    comparable</strong> to full-length 201-nt sequences.
  </div>
</section>"""


def section_motif() -> str:
    return f"""
<section id="motif">
  <h2>2. Motif Analysis</h2>
  <p>
    A3G shows a <strong>dominant CC motif</strong> (90.8% have C at position -1) &mdash; a stark
    contrast to A3A (TC=51.2%) and A3B (TC=32.3%). TC context is actively <em>depleted</em>
    in A3G targets (9.2%, below the 25% random baseline).
  </p>

  <h3>Key Motif Statistics</h3>
  <table>
    <thead><tr><th>Metric</th><th>A3G</th><th>A3A (reference)</th><th>A3B (reference)</th></tr></thead>
    <tbody>
      <tr><td>CC fraction (C at -1)</td><td><strong>90.8%</strong></td><td>19.0%</td><td>24.8%</td></tr>
      <tr><td>TC fraction (U at -1)</td><td>9.2%</td><td>51.2%</td><td>32.3%</td></tr>
      <tr><td>CC Fisher OR vs random</td><td><strong>30.5</strong> (p=6.2e-27)</td><td>0.70</td><td>0.99</td></tr>
      <tr><td>TC Fisher OR vs random</td><td>0.32 (p=0.003)</td><td>3.15</td><td>1.43</td></tr>
    </tbody>
  </table>

  <h3>Figures</h3>
  <div class="grid-2">
    {img_tag(MULTI_MOTIF / 'tc_fraction_comparison.png', caption='TC motif fraction comparison')}
    {img_tag(MULTI_MOTIF / 'tc_vs_cc_comparison.png', caption='TC vs CC fraction scatter')}
  </div>

  <h3>Interpretation</h3>
  <ul>
    <li>A3G has the most extreme motif specificity of the three enzymes &mdash; 90.8% CC</li>
    <li>This is consistent with A3G's known HHLA &rarr; HXLA deamination preference (HIV cDNA context)</li>
    <li>The CC requirement means A3G targets a fundamentally different subset of cytidines than A3A/A3B</li>
  </ul>
</section>"""


def section_structure() -> str:
    return f"""
<section id="structure">
  <h2>3. Structure Analysis</h2>
  <p>
    A3G shows the <strong>most extreme structural preferences</strong> of all three enzymes:
    95.8% in-loop with an extreme 3&prime;-end position (RLP=0.920).
  </p>

  <table>
    <thead><tr><th>Metric</th><th>A3G</th><th>A3A</th><th>A3B</th></tr></thead>
    <tbody>
      <tr><td>In-loop fraction</td><td><strong>95.8%</strong></td><td>62.1%</td><td>54.3%</td></tr>
      <tr><td>RLP (mean)</td><td><strong>0.920</strong></td><td>0.585</td><td>0.515</td></tr>
      <tr><td>RLP (median)</td><td><strong>1.000</strong></td><td>0.600</td><td>0.500</td></tr>
      <tr><td>RLP t-test vs 0.5</td><td>p=4.3e-45 ***</td><td>p=5.1e-24 ***</td><td>p=0.034 *</td></tr>
      <tr><td>Mean loop size (no ext.)</td><td><strong>4.5 nt</strong></td><td>6.7 nt</td><td>7.1 nt</td></tr>
      <tr><td>Median loop size</td><td><strong>4</strong></td><td>6</td><td>6</td></tr>
    </tbody>
  </table>

  <h3>Figures</h3>
  <div class="grid-2">
    {img_tag(MULTI_STRUCT / 'inloop_fraction_comparison.png', caption='In-loop fraction by enzyme')}
    {img_tag(MULTI_STRUCT / 'loop_size_by_enzyme.png', caption='Loop size distribution')}
  </div>

  <h3>Tetraloop Hypothesis</h3>
  <div style="background:#d1fae5;border-left:4px solid #059669;padding:12px 16px;margin:12px 0;border-radius:4px">
    <strong>A3G prefers tetraloops.</strong> The combination of small loop size (median=4 nt)
    and extreme 3&prime;-end position (RLP=0.920, median=1.0) is consistent with Sharma 2017's
    tetraloop requirement. A3G appears to edit the penultimate C in CC-context at the very
    tip of small hairpin loops.
  </div>
</section>"""


def section_classification(cls_data: dict) -> str:
    if not cls_data:
        return """
<section id="classification">
  <h2>4. Classification Results</h2>
  <div style="background:#fef9c3;border-left:4px solid #ca8a04;padding:12px 16px;border-radius:4px">
    <strong>Results not yet available.</strong> Run:
    <code>conda run -n quris python experiments/apobec3g/exp_classification_a3g.py</code>
  </div>
</section>"""

    models = cls_data.get("models", {})
    n_pos = cls_data.get("n_positive", "?")
    n_neg = cls_data.get("n_negative", "?")
    warning = cls_data.get("small_dataset_warning", "")

    rows = []
    for name in ["GB_HandFeatures", "GB_AllFeatures", "MotifOnly", "StructOnly"]:
        m = models.get(name, {})
        auroc = m.get("mean_auroc", m.get("auroc_mean", None))
        std = m.get("std_auroc", m.get("auroc_std", None))
        bs_mean = m.get("bootstrap_auroc_mean", None)
        bs_lo = m.get("bootstrap_auroc_ci95_lo", None)
        bs_hi = m.get("bootstrap_auroc_ci95_hi", None)

        auroc_str = f"{auroc:.4f} &pm; {std:.4f}" if auroc is not None and std is not None else "---"
        bs_str = f"{bs_mean:.4f} [{bs_lo:.4f}, {bs_hi:.4f}]" if bs_mean is not None else "---"
        rows.append(f"<tr><td>{name}</td><td>{auroc_str}</td><td>{bs_str}</td></tr>")

    return f"""
<section id="classification">
  <h2>4. Classification Results</h2>

  <div style="background:#fef9c3;border-left:4px solid #ca8a04;padding:12px 16px;margin:0 0 16px 0;border-radius:4px">
    <strong>Small dataset warning:</strong> {warning or f'n={n_pos} positives limits statistical reliability. Bootstrap CIs are provided for robustness.'}
  </div>

  <p>5-fold StratifiedKFold CV + bootstrap CI (n=1000) on {n_pos} positives + {n_neg} negatives.</p>
  <table>
    <thead><tr><th>Model</th><th>CV AUROC (mean &pm; std)</th><th>Bootstrap AUROC [95% CI]</th></tr></thead>
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
  <h3>Expected Pattern</h3>
  <ul>
    <li>For A3G, the <strong>CC motif features</strong> should dominate (90.8% CC specificity)</li>
    <li><strong>Relative loop position</strong> should also be highly informative (RLP=0.920 vs ~0.5 for negatives)</li>
    <li>Contrast with A3A where relative_loop_position is #1 (0.213) but motif features are secondary</li>
  </ul>
</section>"""


def section_rate(rate_data: dict) -> str:
    if not rate_data:
        return """
<section id="rate">
  <h2>6. Editing Rate Analysis</h2>
  <div style="background:#fef9c3;border-left:4px solid #ca8a04;padding:12px 16px;border-radius:4px">
    <strong>Results not yet available.</strong> Run:
    <code>conda run -n quris python experiments/apobec3g/exp_rate_analysis_a3g.py</code>
  </div>
</section>"""

    n_with_rate = rate_data.get("n_sites_with_rate", "?")
    insufficient = rate_data.get("insufficient_data", False)

    if insufficient or (isinstance(n_with_rate, int) and n_with_rate < 10):
        return f"""
<section id="rate">
  <h2>6. Editing Rate Analysis</h2>
  <p>Insufficient rate data available (n={n_with_rate} sites with rates). Minimum 10 required for meaningful analysis.</p>
</section>"""

    mean_rate = rate_data.get("mean_rate", None)
    correlations = rate_data.get("correlations", {})
    condition_comparison = rate_data.get("condition_comparison", {})

    stat_str = f"Mean rate: {mean_rate:.4f}" if mean_rate is not None else ""

    corr_rows = []
    for feat, res in correlations.items():
        if isinstance(res, dict):
            rho = res.get("spearman_rho", res.get("rho", None))
            p = res.get("p_value", res.get("p", None))
            if rho is not None:
                corr_rows.append(f"<tr><td>{feat}</td><td>{rho:.4f}</td><td>{fmt_p(p)}</td></tr>")

    cond_html = ""
    if condition_comparison:
        hyp_mean = condition_comparison.get("NK_Hyp_mean", None)
        norm_mean = condition_comparison.get("NK_Norm_mean", None)
        mw_p = condition_comparison.get("mannwhitney_p", None)
        if hyp_mean is not None:
            cond_html = f"""
  <h3>Condition Comparison: NK_Hyp vs NK_Norm</h3>
  <table>
    <thead><tr><th>Condition</th><th>Mean rate</th><th>Mann-Whitney p</th></tr></thead>
    <tbody>
      <tr><td>NK_Hyp (hypoxia)</td><td>{hyp_mean:.4f}</td><td rowspan="2">{fmt_p(mw_p)}</td></tr>
      <tr><td>NK_Norm (normoxia)</td><td>{norm_mean:.4f if norm_mean is not None else '---'}</td></tr>
    </tbody>
  </table>"""

    return f"""
<section id="rate">
  <h2>6. Editing Rate Analysis</h2>
  <p>Rate analysis for A3G sites ({n_with_rate} sites with rate data). {stat_str}</p>

  {cond_html}

  <h3>Structure-Rate Correlations (Spearman)</h3>
  <table>
    <thead><tr><th>Feature</th><th>Spearman rho</th><th>p-value</th></tr></thead>
    <tbody>{''.join(corr_rows) if corr_rows else '<tr><td colspan="3"><em>No correlation data</em></td></tr>'}</tbody>
  </table>

  <h3>Figures</h3>
  {img_tag(FIGURES_DIR / 'rate_vs_rlp_a3g.png', caption='Editing rate vs relative loop position')}
</section>"""


def section_clinvar(cv_data: dict) -> str:
    if not cv_data:
        return """
<section id="clinvar">
  <h2>7. ClinVar Analysis</h2>
  <div style="background:#fef9c3;border-left:4px solid #ca8a04;padding:12px 16px;border-radius:4px">
    <strong>Results not yet available.</strong> Run:
    <code>conda run -n quris python experiments/apobec3g/exp_clinvar_a3g.py</code>
  </div>
</section>"""

    caveat = cv_data.get("confidence_caveat", "Small training set limits reliability.")
    cv_auroc = cv_data.get("cv_auroc_mean", None)
    n_cc = cv_data.get("n_cc_context_variants", "?")
    enrichment_all = cv_data.get("enrichment_all_c2u", {})
    enrichment_cc = cv_data.get("enrichment_cc_context_only", {})

    def _enr_rows(enrichment):
        rows = []
        if isinstance(enrichment, dict):
            for thresh, res in sorted(enrichment.items()):
                if isinstance(res, dict):
                    or_val = res.get("odds_ratio", res.get("or", "---"))
                    p = res.get("p_value", res.get("p", None))
                    if isinstance(or_val, (int, float)):
                        or_val = f"{or_val:.3f}"
                    rows.append(f"<tr><td>P &ge; {thresh}</td><td>{or_val}</td><td>{fmt_p(p)}</td></tr>")
        return rows

    all_rows = _enr_rows(enrichment_all)
    cc_rows = _enr_rows(enrichment_cc)
    auroc_str = f"{cv_auroc:.4f}" if cv_auroc is not None else "---"

    return f"""
<section id="clinvar">
  <h2>7. ClinVar Analysis</h2>

  <div style="background:#fee2e2;border-left:4px solid #dc2626;padding:12px 16px;margin:0 0 16px 0;border-radius:4px">
    <strong>Low confidence warning:</strong> {caveat}
  </div>

  <p>
    A3G-specific GB model trained on {cv_data.get('training_n_positive', 119)} positives +
    {cv_data.get('training_n_negative', 119)} negatives. Training CV AUROC: <strong>{auroc_str}</strong>.
    Focus on CC-context ClinVar variants ({n_cc} variants).
  </p>

  <h3>Enrichment: All C&gt;U Variants</h3>
  <table>
    <thead><tr><th>Threshold</th><th>Odds Ratio</th><th>p-value</th></tr></thead>
    <tbody>{''.join(all_rows) if all_rows else '<tr><td colspan="3"><em>No data</em></td></tr>'}</tbody>
  </table>

  <h3>Enrichment: CC-Context Variants Only</h3>
  <p>Restricted to ClinVar variants with C at position -1 (matching A3G's CC motif preference).</p>
  <table>
    <thead><tr><th>Threshold</th><th>Odds Ratio</th><th>p-value</th></tr></thead>
    <tbody>{''.join(cc_rows) if cc_rows else '<tr><td colspan="3"><em>No data</em></td></tr>'}</tbody>
  </table>
</section>"""


def section_clinical() -> str:
    return """
<section id="clinical">
  <h2>8. Clinical Interpretation</h2>

  <h3>8.1 A3G's Distinct Biology</h3>
  <ul>
    <li><strong>A3G is known primarily for HIV/SIV restriction</strong> via cDNA deamination
        (HHLA &rarr; HXLA dinucleotide preference). RNA editing by A3G in CC-context represents
        a distinct biological activity from its antiviral function.</li>
    <li>The <strong>extreme 3&prime;-loop preference</strong> (RLP=0.920) suggests highly specific
        structural recognition &mdash; A3G appears to access only the very tip of small hairpin loops.</li>
    <li>The <strong>tetraloop preference</strong> (median loop size = 4 nt) is consistent with
        Sharma 2017's structural requirements and distinguishes A3G from A3A/A3B.</li>
  </ul>

  <h3>8.2 Comparison to A3A and A3B</h3>
  <table>
    <thead><tr><th>Property</th><th>A3A</th><th>A3B</th><th>A3G</th></tr></thead>
    <tbody>
      <tr><td><strong>Dominant motif</strong></td><td>TC (51.2%)</td><td>Weak TC (32.3%)</td><td><strong>CC (90.8%)</strong></td></tr>
      <tr><td><strong>Loop preference</strong></td><td>62.1%</td><td>54.3%</td><td><strong>95.8%</strong></td></tr>
      <tr><td><strong>Positional bias</strong></td><td>Moderate 3&prime;</td><td>None</td><td><strong>Extreme 3&prime;</strong></td></tr>
      <tr><td><strong>Target scope</strong></td><td>TC cytidines in moderate loops</td><td>Broad (relaxed motif)</td><td><strong>CC cytidines at tetraloop tips</strong></td></tr>
    </tbody>
  </table>

  <h3>8.3 Limitations</h3>
  <div style="background:#fee2e2;border-left:4px solid #dc2626;padding:12px 16px;margin:12px 0;border-radius:4px">
    <strong>Small dataset limits clinical conclusions.</strong>
    <ul style="margin-top:8px">
      <li>n=119 provides proof-of-concept only &mdash; larger A3G editing datasets are needed</li>
      <li>Single study (Dang 2019) from one cell type (NK cells) under one condition (hypoxia)</li>
      <li>ClinVar enrichment analysis has limited statistical power</li>
      <li>N-padded sequences (31-nt real context) limit MFE-based analyses</li>
    </ul>
  </div>
</section>"""


def section_methods() -> str:
    return """
<section id="methods">
  <h2>9. Methods</h2>
  <h3>Data Source</h3>
  <ul>
    <li><strong>Dang 2019</strong> (GSE114519): NK cells, NK_Hyp (hypoxia) condition; A3G editing sites;
        31-nt flanking sequences from supplementary MOESM4 (n=119); padded to 201-nt with N</li>
  </ul>
  <h3>Classification</h3>
  <ul>
    <li>Models: GB_HandFeatures (40-dim), GB_AllFeatures (~90-dim), MotifOnly (24-dim), StructOnly (7-dim)</li>
    <li>Negatives: CC-context matched from hg38 (CC%~91%); generated by generate_negatives_v2.py</li>
    <li>CV: 5-fold StratifiedKFold + bootstrap (n=1000) for 95% CI</li>
    <li>XGBClassifier: n_estimators=200, max_depth=4 (reduced complexity for small n)</li>
  </ul>
  <h3>ClinVar Scoring</h3>
  <ul>
    <li>46-dim features: motif (24) + struct_delta (7) + loop geometry (9) + baseline structure (6)</li>
    <li>CC-context variant filtering: ClinVar variants with C at position -1</li>
    <li>Bayesian calibration: pi_model=0.5 &rarr; pi_real=0.019</li>
    <li>Enrichment computed separately for all C&gt;U variants and CC-context only</li>
  </ul>
  <h3>Caveats</h3>
  <ul>
    <li>Small training set (n=119) limits model complexity and generalization</li>
    <li>N-padded sequences: loop geometry valid, absolute MFE not comparable cross-dataset</li>
    <li>Single-study, single-condition data: findings require external validation</li>
  </ul>
</section>"""


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

CSS = """
* { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: 'Segoe UI', Arial, sans-serif; background: #f8f9fa; color: #222; line-height: 1.6; }
.container { max-width: 1200px; margin: 0 auto; padding: 20px; }
header { background: linear-gradient(135deg, #0f766e 0%, #14b8a6 100%); color: white; padding: 40px 20px; margin-bottom: 30px; }
header h1 { font-size: 2em; margin-bottom: 8px; }
header .subtitle { font-size: 1em; opacity: 0.85; }
nav { background: #fff; border-bottom: 2px solid #0f766e; position: sticky; top: 0; z-index: 100; }
nav ul { display: flex; list-style: none; padding: 0 20px; gap: 0; flex-wrap: wrap; }
nav a { display: block; padding: 12px 18px; color: #0f766e; text-decoration: none; font-weight: 500; font-size: 0.9em; }
nav a:hover { background: #f0fdfa; }
section { background: #fff; border-radius: 8px; padding: 30px; margin-bottom: 30px; box-shadow: 0 1px 4px rgba(0,0,0,.08); }
h2 { color: #0f766e; font-size: 1.5em; margin-bottom: 16px; border-bottom: 2px solid #ccfbf1; padding-bottom: 8px; }
h3 { color: #14b8a6; font-size: 1.1em; margin: 20px 0 10px; }
h4 { color: #374151; font-size: 1em; margin: 14px 0 6px; }
table { border-collapse: collapse; width: 100%; margin: 10px 0; font-size: 0.88em; }
th { background: #0f766e; color: white; padding: 8px 12px; text-align: left; }
td { padding: 7px 12px; border-bottom: 1px solid #e5e7eb; }
tr:nth-child(even) td { background: #f0fdfa; }
tr:hover td { background: #ccfbf1; }
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
  <title>APOBEC3G Analysis Report</title>
  <style>{CSS}</style>
</head>
<body>
<header>
  <div class="container">
    <h1>APOBEC3G Analysis Report</h1>
    <div class="subtitle">
      Dang 2019 (NK_Hyp) &nbsp;|&nbsp; n=119 editing sites &nbsp;|&nbsp; March 2026
      &nbsp;|&nbsp; <span style="background:rgba(255,255,255,0.2);padding:2px 8px;border-radius:3px">Small dataset</span>
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
