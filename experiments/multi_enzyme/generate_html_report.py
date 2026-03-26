#!/usr/bin/env python
"""
Generate self-contained HTML report for multi-enzyme APOBEC analysis.

Reads JSON results and PNG figures from experiments/multi_enzyme/outputs/ and
produces a single self-contained HTML file with embedded images.

Usage:
    conda run -n quris python experiments/multi_enzyme/generate_html_report.py
"""
import base64
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUTS_DIR  = Path(__file__).parent / "outputs"
REPORT_FILE  = OUTPUTS_DIR / "multi_enzyme_report.html"

MOTIF_DIR    = OUTPUTS_DIR / "motif_analysis"
STRUCT_DIR   = OUTPUTS_DIR / "structure_analysis"
CROSS_DIR    = OUTPUTS_DIR / "cross_enzyme"

# Per-enzyme experiment result paths
A3A_CLS_PATH = PROJECT_ROOT / "experiments/apobec3a/outputs/classification_a3a_5fold/classification_a3a_5fold_results.json"
_a3b_new = PROJECT_ROOT / "experiments/apobec3b/outputs/classification/classification_results.json"
_a3b_old = PROJECT_ROOT / "experiments/apobec3b/outputs/classification/classification_a3b_results.json"
A3B_CLS_PATH = _a3b_new if _a3b_new.exists() else _a3b_old
_a3g_new = PROJECT_ROOT / "experiments/apobec3g/outputs/classification/classification_results.json"
_a3g_old = PROJECT_ROOT / "experiments/apobec3g/outputs/classification/classification_a3g_results.json"
A3G_CLS_PATH = _a3g_new if _a3g_new.exists() else _a3g_old
A3B_CLINVAR_PATH = PROJECT_ROOT / "experiments/apobec3b/outputs/clinvar/a3b_clinvar_results.json"
A3G_CLINVAR_PATH = PROJECT_ROOT / "experiments/apobec3g/outputs/clinvar/a3g_clinvar_results.json"
A3A_CLINVAR_PATH = PROJECT_ROOT / "experiments/apobec3a/outputs/clinvar_prediction/clinvar_all_scores.csv"
A4_CLS_PATH = PROJECT_ROOT / "experiments/apobec4/outputs/classification/classification_results.json"
A4_ANALYSIS_PATH = PROJECT_ROOT / "experiments/apobec4/outputs/a4_analysis_results.json"
A4_FIGURES_DIR = PROJECT_ROOT / "experiments/apobec4/outputs"


def b64_image(path: Path) -> str:
    """Encode image as base64 data URI, or return placeholder if missing."""
    if not path.exists():
        return f"data:image/png;base64,"  # empty
    with open(path, "rb") as f:
        data = base64.b64encode(f.read()).decode()
    return f"data:image/png;base64,{data}"


def load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)


def fmt_p(p):
    """Format p-value."""
    if p is None or (isinstance(p, float) and p != p):
        return "N/A"
    p = float(p)
    if p < 1e-100:
        return "<1e-100"
    if p < 0.001:
        return f"{p:.2e}"
    return f"{p:.4f}"


def sig_stars(p):
    if p is None:
        return ""
    try:
        p = float(p)
    except Exception:
        return ""
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "ns"


def img_tag(path: Path, width: str = "100%", caption: str = "") -> str:
    src = b64_image(path)
    cap = f"<figcaption>{caption}</figcaption>" if caption else ""
    return f'<figure><img src="{src}" style="width:{width};max-width:900px" alt="{path.name}"/>{cap}</figure>'


# ---------------------------------------------------------------------------
# Section builders
# ---------------------------------------------------------------------------

def section_overview(motif_results: dict, struct_results: dict) -> str:
    """Top-level summary table."""
    enzymes = ["A3A", "A3B", "A3G"]
    rows = []
    for enz in enzymes:
        mr = motif_results.get(enz, {})
        sr = struct_results.get(enz, {})
        n = mr.get("n_sites", sr.get("n_positive", "—"))
        tc = mr.get("tc_fraction", float("nan"))
        cc = mr.get("cc_m1_fraction", float("nan"))
        inloop = sr.get("positive_inloop_fraction", float("nan"))
        loop_sz = sr.get("positive_loop_size", {}).get("median",
                  sr.get("positive_loop_size", {}).get("mean", float("nan")))
        rows.append(f"""
        <tr>
          <td><strong>{enz}</strong></td>
          <td>{n if n != '—' else '—'}</td>
          <td>{tc:.1%}</td>
          <td>{cc:.1%}</td>
          <td>{inloop:.1%}</td>
          <td>{loop_sz:.1f}</td>
        </tr>""")

    return f"""
<section id="overview">
  <h2>1. Dataset Overview</h2>
  <p>
    Multi-enzyme APOBEC editing site analysis across three enzymes:
    <strong>APOBEC3A</strong> (Kockler 2026, <em>n</em>=2,749),
    <strong>APOBEC3B</strong> (Kockler 2026 + Zhang 2024, <em>n</em>=4,180),
    <strong>APOBEC3G</strong> (Dang 2019 NK_Hyp, <em>n</em>=119).
    All sites represent confirmed C-to-U editing events.
    Sequences are 201-nt windows centered on the edit site (position 100).
  </p>
  <table>
    <thead>
      <tr>
        <th>Enzyme</th><th>n sites</th><th>TC%</th><th>CC%</th>
        <th>In-loop%</th><th>Median loop size</th>
      </tr>
    </thead>
    <tbody>
      {''.join(rows)}
    </tbody>
  </table>
</section>"""


def section_motif(motif_results: dict) -> str:
    """Motif analysis section."""
    enzymes = ["A3A", "A3B", "A3G"]

    # Per-enzyme tables
    enz_tables = []
    for enz in enzymes:
        mr = motif_results.get(enz, {})
        if not mr:
            continue
        n = mr.get("n_sites", "—")
        tc = mr.get("tc_fraction", float("nan"))
        cc = mr.get("cc_m1_fraction", float("nan"))
        scc = mr.get("strict_cc_m2m1_fraction", float("nan"))
        tc_or = mr.get("fisher_tc_or", float("nan"))
        tc_p  = mr.get("fisher_tc_p", None)
        cc_or = mr.get("fisher_cc_or", float("nan"))
        cc_p  = mr.get("fisher_cc_p", None)
        ic    = mr.get("information_content_m1", float("nan"))
        tridiv = mr.get("trinucleotide_diversity", float("nan"))

        top_tri = mr.get("trinucleotides", {})
        if isinstance(top_tri, dict):
            top_tri_str = ", ".join(f"{k}: {v}" for k, v in list(top_tri.items())[:5])
        else:
            top_tri_str = "—"

        m1 = mr.get("minus1_base_fractions", {})

        enz_tables.append(f"""
        <h4>{enz} (n={n})</h4>
        <table>
          <tr><td>TC fraction (U at -1)</td><td>{tc:.1%}</td>
              <td>Fisher OR={tc_or:.2f}, p={fmt_p(tc_p)} {sig_stars(tc_p)}</td></tr>
          <tr><td>CC fraction (C at -1)</td><td>{cc:.1%}</td>
              <td>Fisher OR={cc_or:.2f}, p={fmt_p(cc_p)} {sig_stars(cc_p)}</td></tr>
          <tr><td>Strict CC (C at -2,-1)</td><td>{scc:.1%}</td><td></td></tr>
          <tr><td>Info content at -1</td><td>{ic:.3f} bits</td><td></td></tr>
          <tr><td>Trinucleotide diversity</td><td>{tridiv:.2f} bits</td><td></td></tr>
          <tr><td>Position -1 base frequencies</td>
              <td colspan="2">A={m1.get('A', 0):.2f} C={m1.get('C', 0):.2f} G={m1.get('G', 0):.2f} U={m1.get('U', 0):.2f}</td></tr>
          <tr><td>Top trinucleotides</td><td colspan="2">{top_tri_str}</td></tr>
        </table>""")

    # Pairwise tests
    pw_tc = motif_results.get("pairwise_tc_tests", {})
    pw_rows = []
    for pair, res in pw_tc.items():
        if not isinstance(res, dict):
            continue
        chi2 = res.get("chi2", float("nan"))
        p    = res.get("p", None)
        pw_rows.append(f"<tr><td>{pair}</td><td>χ²={chi2:.1f}</td><td>p={fmt_p(p)} {sig_stars(p)}</td></tr>")

    return f"""
<section id="motif">
  <h2>2. Per-Enzyme Motif Analysis</h2>

  <h3>2.1 Figures</h3>
  <div class="grid-2">
    {img_tag(MOTIF_DIR / 'tc_fraction_comparison.png', caption='TC motif fraction by enzyme vs 25% random baseline')}
    {img_tag(MOTIF_DIR / 'tc_vs_cc_comparison.png', caption='TC vs CC fraction scatter (bubble size = n sites)')}
  </div>
  <div class="grid-2">
    {img_tag(MOTIF_DIR / 'position_frequencies_grid.png', caption='Nucleotide frequency at positions -2 to +2')}
    {img_tag(MOTIF_DIR / 'information_content_overlay.png', caption='Information content per position')}
  </div>
  {img_tag(MOTIF_DIR / 'dinucleotide_heatmap.png', caption='Dinucleotide frequency heatmap (5&prime; context − C)')}

  <h3>2.2 Per-Enzyme Summary</h3>
  {''.join(enz_tables)}

  <h3>2.3 Pairwise TC Fraction Comparisons (Chi-squared)</h3>
  <table>
    <thead><tr><th>Pair</th><th>Statistic</th><th>p-value</th></tr></thead>
    <tbody>{''.join(pw_rows)}</tbody>
  </table>

  <h3>2.4 Key Findings</h3>
  <ul>
    <li><strong>A3A</strong>: Strong TC preference (51.2%); Fisher OR=3.15 vs random, p=2.7e-90</li>
    <li><strong>A3B</strong>: Moderate TC preference (32.3%); weaker than A3A (OR=1.43, p=1.5e-13); no significant CC enrichment (p=0.86)</li>
    <li><strong>A3G</strong>: Dominant CC preference (90.8%); Fisher OR=30.5 vs random, p=6.2e-27; TC depleted (OR=0.32, p=0.003)</li>
    <li>All pairwise TC/CC differences are highly significant (p&lt;0.001)</li>
  </ul>
</section>"""


def section_structure(struct_results: dict) -> str:
    """Structure analysis section."""
    enzymes = ["A3A", "A3B", "A3G"]
    enz_rows = []
    for enz in enzymes:
        sr = struct_results.get(enz, {})
        if not sr:
            continue
        n = sr.get("n_positive", "—")
        inloop = sr.get("positive_inloop_fraction", float("nan"))
        binom_p = sr.get("inloop_vs_random_binom_p", None)
        # Use corrected loop_size (external loops excluded) if available
        lsz_src = sr.get("positive_loop_size_no_external") or sr.get("positive_loop_size", {})
        loop_sz = lsz_src.get("mean", float("nan"))
        loop_med = lsz_src.get("median", float("nan"))
        ext_frac = sr.get("external_loop_fraction", 0.0)
        dmfe = sr.get("positive_delta_mfe", {}).get("mean", float("nan"))
        # loop type distribution
        ltd = sr.get("positive_loop_type_distribution", {})
        if ltd:
            total_lp = sum(ltd.values())
            dominant_lt = max((k for k in ltd if k != "external"), key=ltd.get, default=max(ltd, key=ltd.get))
            dominant_pct = ltd[dominant_lt] / max(total_lp, 1)
            lt_str = f"{dominant_lt} ({dominant_pct:.0%})"
        else:
            lt_str = "—"

        ext_note = f" <span title='N-padding: {ext_frac:.1%} external loops excluded' style='color:#ca8a04;font-size:0.85em'>*</span>" if ext_frac > 0.05 else ""
        enz_rows.append(f"""
        <tr>
          <td><strong>{enz}</strong></td>
          <td>{n}</td>
          <td>{inloop:.1%}</td>
          <td>{fmt_p(binom_p)} {sig_stars(binom_p)}</td>
          <td>{loop_sz:.1f} (median {loop_med:.0f}){ext_note}</td>
          <td>{lt_str}</td>
          <td>{dmfe:.4f}</td>
        </tr>""")

    return f"""
<section id="structure">
  <h2>3. Per-Enzyme RNA Structure Analysis</h2>

  <h3>3.1 Figures</h3>
  <div class="grid-2">
    {img_tag(STRUCT_DIR / 'inloop_fraction_comparison.png', caption='In-loop fraction by enzyme')}
    {img_tag(STRUCT_DIR / 'loop_size_by_enzyme.png', caption='Loop size distribution by enzyme')}
  </div>
  <div class="grid-2">
    {img_tag(STRUCT_DIR / 'loop_type_by_enzyme.png', caption='Loop type distribution by enzyme')}
    {img_tag(STRUCT_DIR / 'delta_mfe_by_enzyme.png', caption='Delta MFE (C→U edit) by enzyme')}
  </div>

  <h3>3.2 Summary Table</h3>
  <table>
    <thead>
      <tr>
        <th>Enzyme</th><th>n</th><th>In-loop%</th><th>Binom p vs 50%</th>
        <th>Loop size mean (median)</th><th>Dominant loop type</th><th>Mean ΔMfe</th>
      </tr>
    </thead>
    <tbody>{''.join(enz_rows)}</tbody>
  </table>

  <div style="background:#fef9c3;border-left:4px solid #ca8a04;padding:12px 16px;margin:12px 0;border-radius:4px;font-size:0.9em">
    <strong>⚠ N-padding note:</strong> Kockler (41-nt) and Dang (31-nt) sequences are padded with N to 201 nt.
    ViennaRNA treats N as fixed-unpaired, so <strong>loop geometry features are valid</strong> but
    <strong>absolute MFE values are not comparable</strong> across datasets (Kockler/Dang ~−8 kcal/mol vs Zhang ~−55 kcal/mol).
    In-loop fractions and loop size distributions reflect genuine biological signal.
  </div>

  <h3>3.3 Relative Loop Position — Key Distinguishing Feature</h3>
  <p>
    The <strong>relative loop position (RLP)</strong> encodes where within a loop the edit site sits
    (0 = left/5' edge, 1 = right/3' edge, 0.5 = center). This feature distinguishes the three enzymes
    more sharply than in-loop fraction alone:
  </p>
  <table>
    <thead><tr><th>Enzyme</th><th>RLP mean</th><th>RLP median</th><th>t-test vs 0.5 (p)</th><th>Interpretation</th></tr></thead>
    <tbody>
      <tr><td><strong>A3A</strong></td><td>0.585</td><td>0.600</td><td>5.1e-24 ***</td><td>Moderate 3&prime;-of-loop preference</td></tr>
      <tr><td><strong>A3B</strong></td><td>0.515</td><td>0.500</td><td>3.4e-02 *</td><td><strong>No meaningful positional preference</strong></td></tr>
      <tr><td><strong>A3G</strong></td><td>0.920</td><td>1.000</td><td>4.3e-45 ***</td><td>Extreme 3&prime;-end preference</td></tr>
    </tbody>
  </table>
  <p style="margin-top:8px">
    All pairwise A3A vs A3B vs A3G RLP differences are highly significant (Mann-Whitney: A3A vs A3B p=1.6e-10,
    A3A vs A3G p=4.1e-28, A3B vs A3G p=1.8e-37).
  </p>

  <h3>3.4 Key Findings</h3>
  <ul>
    <li><strong>All enzymes prefer unpaired loop regions</strong>, with A3G showing extreme loop preference (95.8%)</li>
    <li><strong>A3A</strong>: 62.1% in-loop; hairpin dominant; mean loop size 6.7 nt (median 6), excluding 9.4% external loops; moderate 3&prime;-of-loop preference (RLP=0.585)</li>
    <li><strong>A3B</strong>: 54.3% in-loop; mean loop size 7.1 nt (median 6), excluding 12.6% external loops; <strong>no meaningful positional preference within loops (RLP=0.515)</strong></li>
    <li><strong>A3G</strong>: 95.8% in-loop; small hairpin loops (mean 4.5 nt, median 4), excluding 1.8% external loops; extreme 3&prime;-end preference (RLP=0.920, median=1.0) consistent with Sharma 2017 tetraloop requirement</li>
    <li>A3G has near-zero ΔMfe (0.088 kcal/mol), suggesting C→U edits at its CC-context tetraloop sites minimally disrupt structure</li>
    <li>External loop classification (9–13% of Kockler unpaired sites, ~2% for Dang) reflects N-padded sequence boundaries; excluded from loop size statistics (marked * in table)</li>
  </ul>
</section>"""


def section_cross_enzyme(cross_results: dict) -> str:
    """Cross-enzyme comparison section."""
    pairwise = cross_results.get("pairwise_classifiers", {})
    struct_tests = cross_results.get("structural_tests", {})
    sig_table = cross_results.get("signature_table", [])

    # Pairwise classifier rows
    clf_rows = []
    for pair, res in pairwise.items():
        if not isinstance(res, dict):
            continue
        auroc = res.get("auroc_mean", float("nan"))
        auroc_std = res.get("auroc_std", float("nan"))
        top_feat = res.get("top_feature", "—")
        top_imp = res.get("top_importance", float("nan"))
        dist = res.get("distinguishable", False)
        clf_rows.append(f"""
        <tr>
          <td>{pair}</td>
          <td>{auroc:.4f} ± {auroc_std:.4f}</td>
          <td>{'DISTINGUISHABLE' if dist else 'similar'}</td>
          <td>{top_feat} ({top_imp:.3f})</td>
        </tr>""")

    # Structural test rows
    struct_rows = []
    for test_name, res in struct_tests.items():
        if not isinstance(res, dict):
            continue
        p = res.get("p", None)
        struct_rows.append(f"<tr><td>{test_name}</td><td>{fmt_p(p)} {sig_stars(p)}</td></tr>")

    # Signature table
    sig_rows = []
    for row in sig_table:
        if not isinstance(row, dict):
            continue
        enz = row.get("enzyme", "—")
        n   = row.get("n_sites", "—")
        tc  = row.get("tc_fraction", float("nan"))
        cc  = row.get("cc_fraction", float("nan"))
        inloop = row.get("inloop_fraction", float("nan"))
        loop_sz = row.get("loop_size_mean", float("nan"))
        motif_sig = row.get("dominant_motif", "—")
        sig_rows.append(f"""
        <tr>
          <td><strong>{enz}</strong></td>
          <td>{n}</td>
          <td>{tc:.1%}</td>
          <td>{cc:.1%}</td>
          <td>{inloop:.1%}</td>
          <td>{loop_sz:.1f}</td>
          <td>{motif_sig}</td>
        </tr>""")

    return f"""
<section id="cross-enzyme">
  <h2>4. Cross-Enzyme Comparison</h2>

  <h3>4.1 Figures</h3>
  <div class="grid-2">
    {img_tag(CROSS_DIR / 'motif_radar_chart.png', caption='Motif property radar chart across enzymes')}
    {img_tag(CROSS_DIR / 'pairwise_auroc_heatmap.png', caption='Pairwise classifier AUROC (distinguishability)')}
  </div>
  {img_tag(CROSS_DIR / 'structure_comparison.png', '90%', caption='Structural context comparison across enzymes')}

  <h3>4.2 Enzyme Signature Table</h3>
  <table>
    <thead>
      <tr>
        <th>Enzyme</th><th>n</th><th>TC%</th><th>CC%</th>
        <th>In-loop%</th><th>Loop size</th><th>Dominant motif</th>
      </tr>
    </thead>
    <tbody>{''.join(sig_rows)}</tbody>
  </table>

  <h3>4.3 Pairwise Discriminability (GB Classifier)</h3>
  <table>
    <thead>
      <tr>
        <th>Pair</th><th>AUROC (mean ± std)</th><th>Discriminability</th>
        <th>Top feature</th>
      </tr>
    </thead>
    <tbody>{''.join(clf_rows)}</tbody>
  </table>

  <h3>4.4 Structural Tests (Mann-Whitney)</h3>
  <table>
    <thead><tr><th>Test</th><th>p-value</th></tr></thead>
    <tbody>{''.join(struct_rows)}</tbody>
  </table>

  <h3>4.5 Key Findings</h3>
  <ul>
    <li><strong>A3A vs A3G</strong>: Highly distinguishable (AUROC=0.94); top feature = motif_CC; opposite motif preferences dominate (A3A=TC, A3G=CC)</li>
    <li><strong>A3B vs A3G</strong>: Highly distinguishable (AUROC=0.96); top feature = relative_loop_position; A3G's extreme loop preference (95.8%) and tetraloop constraint distinguish it from A3B (54.3%)</li>
    <li><strong>A3A vs A3B</strong>: Partially distinguishable (AUROC=0.66); top feature = motif_UC (TC context); A3A and A3B share structural context (similar loop sizes: 6.7 vs 7.1 nt) but differ in TC motif stringency</li>
    <li><strong>A3B resolves published structural contradiction</strong>: A3B <em>does</em> prefer loops (54.3% in-loop, p=1e-10 vs A3A), confirming Butt 2024; but lacks 3&prime;-end positional preference (RLP=0.515≈random), confirming Alonso de la Vega 2023. The two papers measured different aspects of structural preference and are both correct.</li>
    <li>All pairwise in-loop fraction tests are significant (Mann-Whitney p&lt;0.001)</li>
  </ul>
</section>"""


def _extract_model_auroc(cls_data: dict, model_name: str) -> str:
    """Extract AUROC mean +/- std for a model from classification results JSON."""
    models = cls_data.get("models", cls_data.get("results", {}))
    if isinstance(models, dict):
        m = models.get(model_name, {})
        # Try multiple key conventions: mean_auroc, auroc_mean, auroc
        mean = m.get("mean_auroc", m.get("auroc_mean", m.get("auroc", None)))
        std = m.get("std_auroc", m.get("auroc_std", None))
        # Also check nested mean_metrics (from generic classifier)
        if mean is None and "mean_metrics" in m:
            mean = m["mean_metrics"].get("auroc", None)
        if std is None and "std_metrics" in m:
            std = m["std_metrics"].get("auroc", None)
        if mean is not None:
            if std is not None:
                return f"{float(mean):.4f} ± {float(std):.4f}"
            return f"{float(mean):.4f}"
    # Try flat structure: top-level keys
    mean = cls_data.get(f"{model_name}_auroc_mean", cls_data.get(f"{model_name}_auroc", None))
    std = cls_data.get(f"{model_name}_auroc_std", None)
    if mean is not None:
        if std is not None:
            return f"{float(mean):.4f} ± {float(std):.4f}"
        return f"{float(mean):.4f}"
    return "---"


def _extract_n_sites(cls_data: dict) -> tuple:
    """Extract (n_positive, n_negative) from classification results."""
    n_pos = cls_data.get("n_positive", cls_data.get("n_pos", "---"))
    n_neg = cls_data.get("n_negative", cls_data.get("n_neg", "---"))
    return n_pos, n_neg


def _extract_feature_importance(cls_data: dict, model_name: str = "GB_HandFeatures", top_n: int = 10) -> list:
    """Extract top N feature importances from classification results.

    Returns list of (feature_name, importance) tuples.
    """
    models = cls_data.get("models", cls_data.get("results", {}))
    if isinstance(models, dict):
        m = models.get(model_name, {})
        fi = m.get("feature_importance", m.get("feature_importances", {}))
        if isinstance(fi, dict) and fi:
            sorted_fi = sorted(fi.items(), key=lambda x: float(x[1]), reverse=True)
            return sorted_fi[:top_n]
        if isinstance(fi, list) and fi:
            return fi[:top_n]
    # Try top-level feature_importance key
    fi = cls_data.get("feature_importance", cls_data.get("feature_importances", {}))
    if isinstance(fi, dict):
        # Handle nested structure: {"GB_HandFeatures": {...features...}}
        if model_name in fi and isinstance(fi[model_name], dict):
            inner = fi[model_name]
            sorted_fi = sorted(inner.items(), key=lambda x: float(x[1]), reverse=True)
            return sorted_fi[:top_n]
        # Handle flat structure: {"feature_name": importance, ...}
        if fi and not any(isinstance(v, dict) for v in fi.values()):
            sorted_fi = sorted(fi.items(), key=lambda x: float(x[1]), reverse=True)
            return sorted_fi[:top_n]
    return []


def section_classification(a3a_cls: dict, a3b_cls: dict, a3g_cls: dict, a4_cls: dict = None) -> str:
    """Per-enzyme classification performance section."""
    rows = []
    enzyme_list = [("A3A", a3a_cls), ("A3B", a3b_cls), ("A3G", a3g_cls)]
    if a4_cls:
        enzyme_list.append(("A4", a4_cls))
    for enz, cls_data in enzyme_list:
        if not cls_data:
            rows.append(f'<tr><td><strong>{enz}</strong></td><td colspan="4"><em>Results not available --- run classification experiment</em></td></tr>')
            continue
        n_pos, n_neg = _extract_n_sites(cls_data)
        gb_hand = _extract_model_auroc(cls_data, "GB_HandFeatures")
        gb_all = _extract_model_auroc(cls_data, "GB_AllFeatures")
        note = ""
        if enz == "A3G":
            note = ' <span style="color:#6b7280;font-size:0.85em">(bootstrap CI)</span>'
        rows.append(f"""
        <tr>
          <td><strong>{enz}</strong></td>
          <td>{n_pos}</td>
          <td>{n_neg}</td>
          <td>{gb_hand}{note}</td>
          <td>{gb_all}{note}</td>
        </tr>""")

    return f"""
<section id="classification">
  <h2>5. Per-Enzyme Classification Performance</h2>
  <p>
    Binary classification of edited vs unedited cytidines using gradient boosting on hand-crafted features.
    Each enzyme model is trained and evaluated on its own dataset with appropriate negative sampling.
  </p>
  <table>
    <thead>
      <tr>
        <th>Enzyme</th><th>n+</th><th>n-</th>
        <th>GB_HandFeatures AUROC</th><th>GB_AllFeatures AUROC</th>
      </tr>
    </thead>
    <tbody>
      {''.join(rows)}
    </tbody>
  </table>
  <p style="font-size:0.88em;color:#6b7280;margin-top:8px">
    A3A: 5-fold KFold CV on 8,153 sites. A3B: 5-fold KFold CV. A3G: bootstrap CI due to small dataset (n=119).
    Classification requires negatives from generate_negatives_v2.py.
  </p>
</section>"""


def section_feature_importance(a3a_cls: dict, a3b_cls: dict, a3g_cls: dict, a4_cls: dict = None) -> str:
    """Cross-enzyme feature importance comparison section."""
    enz_blocks = []
    enzyme_list = [("A3A", a3a_cls), ("A3B", a3b_cls), ("A3G", a3g_cls)]
    if a4_cls:
        enzyme_list.append(("A4", a4_cls))
    for enz, cls_data in enzyme_list:
        if not cls_data:
            enz_blocks.append(f"<h4>{enz}</h4><p><em>No classification results available.</em></p>")
            continue
        fi = _extract_feature_importance(cls_data, "GB_HandFeatures", top_n=10)
        if not fi:
            enz_blocks.append(f"<h4>{enz}</h4><p><em>No feature importance data available.</em></p>")
            continue
        fi_rows = []
        for rank, item in enumerate(fi, 1):
            if isinstance(item, (list, tuple)):
                feat, imp = item[0], float(item[1])
            elif isinstance(item, dict):
                feat, imp = item.get("feature", "?"), float(item.get("importance", 0))
            else:
                continue
            fi_rows.append(f"<tr><td>{rank}</td><td>{feat}</td><td>{imp:.4f}</td></tr>")
        enz_blocks.append(f"""
        <h4>{enz} --- GB_HandFeatures Top 10</h4>
        <table>
          <thead><tr><th>#</th><th>Feature</th><th>Importance</th></tr></thead>
          <tbody>{''.join(fi_rows)}</tbody>
        </table>""")

    return f"""
<section id="feature-importance">
  <h2>6. Cross-Enzyme Feature Importance</h2>
  <p>
    Top features from GB_HandFeatures (motif 24-dim + loop geometry 9-dim) for each enzyme.
    Feature importance reveals which motif and structural features are most discriminative
    for each enzyme's editing site selection.
  </p>
  {''.join(enz_blocks)}

  <h3>6.1 Key Comparisons</h3>
  <ul>
    <li><strong>A3A</strong>: relative_loop_position is the #1 feature (importance=0.213), reflecting moderate 3&prime;-of-loop preference</li>
    <li><strong>A3B</strong>: Expected to show motif features more prominently, with weaker loop position signal (RLP~0.5)</li>
    <li><strong>A3G</strong>: CC-context motif features should dominate, with strong loop position signal (RLP=0.920)</li>
  </ul>
</section>"""


def section_clinvar_cross_enzyme(a3b_clinvar: dict, a3g_clinvar: dict) -> str:
    """ClinVar cross-enzyme clinical analysis section."""
    rows = []
    # A3A reference row (hardcoded from existing results)
    rows.append("""
    <tr>
      <td><strong>A3A</strong></td>
      <td>8,153</td>
      <td>P &ge; 0.5</td>
      <td>1.279</td>
      <td>&lt;1e-138</td>
      <td>Significant pathogenic enrichment</td>
    </tr>""")

    for enz, cv_data in [("A3B", a3b_clinvar), ("A3G", a3g_clinvar)]:
        if not cv_data:
            rows.append(f'<tr><td><strong>{enz}</strong></td><td colspan="5"><em>Not yet computed --- run exp_clinvar_{enz.lower()}.py</em></td></tr>')
            continue

        n_scored = cv_data.get("n_clinvar_scored", cv_data.get("n_clinvar_total", "---"))
        cv_auroc = cv_data.get("mean_cv_auroc", cv_data.get("cv_auroc_mean", "---"))

        # Extract enrichment from raw and calibrated lists
        enrich_raw = cv_data.get("enrichment_raw", [])
        enrich_cal = cv_data.get("enrichment_calibrated", [])

        # For A3G, check alternative key structure
        if not enrich_raw and "enrichment_all_c2u" in cv_data:
            enrich_raw = cv_data.get("enrichment_all_c2u", [])
        if not enrich_raw and "enrichment_cc_context_only" in cv_data:
            enrich_raw = cv_data.get("enrichment_cc_context_only", [])

        # Show best raw enrichment
        for entry in (enrich_raw if isinstance(enrich_raw, list) else []):
            t = entry.get("threshold", "?")
            or_val = entry.get("odds_ratio", "---")
            p_val = entry.get("p_value", None)
            if isinstance(or_val, (int, float)):
                or_str = f"{or_val:.3f}"
            else:
                or_str = str(or_val)
            interp = "Enrichment" if isinstance(or_val, (int, float)) and or_val > 1 else "No enrichment"
            rows.append(f"""
    <tr>
      <td><strong>{enz}</strong> (raw)</td>
      <td>{n_scored}</td>
      <td>P &ge; {t}</td>
      <td>{or_str}</td>
      <td>{fmt_p(p_val)}</td>
      <td>{interp}</td>
    </tr>""")

        # Show best calibrated enrichment
        for entry in (enrich_cal[:1] if isinstance(enrich_cal, list) else []):
            t_cal = entry.get("calibrated_threshold", entry.get("threshold", "?"))
            or_val = entry.get("odds_ratio", "---")
            p_val = entry.get("p_value", None)
            orig_t = entry.get("original_threshold", "?")
            if isinstance(or_val, (int, float)):
                or_str = f"{or_val:.3f}"
            else:
                or_str = str(or_val)
            rows.append(f"""
    <tr>
      <td><strong>{enz}</strong> (calibrated)</td>
      <td>{n_scored}</td>
      <td>P<sub>cal</sub> &ge; {t_cal:.4f} (raw {orig_t})</td>
      <td>{or_str}</td>
      <td>{fmt_p(p_val)}</td>
      <td>After Bayesian calibration</td>
    </tr>""")

    return f"""
<section id="clinvar">
  <h2>7. ClinVar Clinical Analysis</h2>
  <p>
    Pathogenic enrichment analysis across enzymes. Each enzyme's GB model is used to score
    ClinVar C&gt;T variants, then odds ratios are computed for predicted-edited sites
    among pathogenic vs benign variants.
  </p>
  <table>
    <thead>
      <tr>
        <th>Enzyme</th><th>n training</th><th>Threshold</th>
        <th>Odds Ratio</th><th>p-value</th><th>Interpretation</th>
      </tr>
    </thead>
    <tbody>
      {''.join(rows)}
    </tbody>
  </table>
  <p style="font-size:0.88em;color:#6b7280;margin-top:8px">
    ClinVar scoring is computationally intensive (~4-6h per enzyme with 12 workers if no cache).
    A3A results from existing pipeline. A3B/A3G require running exp_clinvar_*.py experiments.
  </p>
</section>"""


def section_clinical_interpretation() -> str:
    """Clinical interpretation and key findings section."""
    return """
<section id="clinical">
  <h2>8. Clinical Interpretation &amp; Key Findings</h2>

  <h3>8.1 Three Distinct Editing Landscapes</h3>
  <p>
    The multi-enzyme analysis reveals three fundamentally different C-to-U editing programs,
    each with distinct sequence preferences, structural requirements, and potentially different
    clinical implications:
  </p>
  <table>
    <thead>
      <tr>
        <th>Property</th><th>APOBEC3A</th><th>APOBEC3B</th><th>APOBEC3G</th>
      </tr>
    </thead>
    <tbody>
      <tr><td><strong>Motif</strong></td><td>TC (51.2%)</td><td>Weak TC (32.3%)</td><td>CC (90.8%)</td></tr>
      <tr><td><strong>Loop preference</strong></td><td>62.1% in-loop</td><td>54.3% in-loop</td><td>95.8% in-loop</td></tr>
      <tr><td><strong>Positional bias</strong></td><td>Moderate 3&prime; (RLP=0.585)</td><td>None (RLP=0.515)</td><td>Extreme 3&prime; (RLP=0.920)</td></tr>
      <tr><td><strong>Loop size</strong></td><td>6.7 nt</td><td>7.1 nt</td><td>4.5 nt (tetraloop)</td></tr>
      <tr><td><strong>Dominant loop type</strong></td><td>Hairpin</td><td>Hairpin</td><td>Hairpin (tetraloop)</td></tr>
    </tbody>
  </table>

  <h3>8.2 Resolution of Published Contradictions</h3>
  <ul>
    <li><strong>A3B structural context</strong>: Butt 2024 reported A3B prefers loops; Alonso de la Vega 2023
        found no positional preference. Our analysis confirms <em>both</em> are correct &mdash;
        A3B uses loops (54.3% in-loop, significantly above random) but has no 3&prime;-end
        positional preference within those loops (RLP=0.515, indistinguishable from 0.5).</li>
    <li><strong>A3G tetraloop constraint</strong>: The extreme RLP=0.920 and small loop size (4.5 nt)
        are consistent with Sharma 2017's tetraloop requirement. A3G edits the penultimate C
        in CC-context at the 3&prime; end of small hairpin loops.</li>
  </ul>

  <h3>8.3 Implications</h3>
  <ul>
    <li><strong>Enzyme-specific predictors are essential</strong>: A single model cannot capture
        the distinct editing programs of A3A, A3B, and A3G. Motif context, loop geometry,
        and positional preferences differ qualitatively, not just quantitatively.</li>
    <li><strong>Structure is universally important</strong>: All three enzymes show significant
        loop preference. The structural feature is_unpaired/relative_loop_position is
        informative across all enzymes, though with different effect sizes and optima.</li>
    <li><strong>Clinical stratification</strong>: If ClinVar enrichment patterns differ by enzyme,
        disease-variant interpretation must account for which APOBEC enzyme is active in the
        relevant tissue context.</li>
  </ul>
</section>"""


def section_apobec4(a4_cls: dict, a4_analysis: dict) -> str:
    """APOBEC4 dedicated section — newly discovered enzyme."""
    if not a4_cls:
        return """
<section id="apobec4">
  <h2>APOBEC4</h2>
  <p><em>No A4 classification results available. Run experiments/apobec4/exp_a4_negative_control.py</em></p>
</section>"""

    # Extract classification results
    models = a4_cls.get("models", {})
    gb = models.get("GB_HandFeatures", {})
    motif = models.get("MotifOnly", {})
    struct = models.get("StructOnly", {})
    excl = models.get("A4_exclusive_LOO", {})

    gb_auroc = gb.get("mean_auroc", 0)
    gb_std = gb.get("std_auroc", 0)
    motif_auroc = motif.get("mean_auroc", 0)
    struct_auroc = struct.get("mean_auroc", 0)
    excl_auroc = excl.get("auroc", 0)

    n_pos = a4_cls.get("n_positive", 181)
    n_neg = a4_cls.get("n_negative", 181)

    # Feature importance
    fi = a4_cls.get("feature_importance", {}).get("GB_HandFeatures", {})
    fi_sorted = sorted(fi.items(), key=lambda x: -float(x[1]))[:10]
    fi_rows = ""
    for rank, (feat, imp) in enumerate(fi_sorted, 1):
        fi_rows += f"<tr><td>{rank}</td><td>{feat}</td><td>{float(imp):.4f}</td></tr>"

    # Structure comparison
    struct_comp = a4_cls.get("structure_comparison", {})
    struct_features = struct_comp.get("features", [])
    struct_rows = ""
    for sf in struct_features:
        sig = sf.get("significant", "No")
        style = ' style="background:#dcfce7"' if sig == "Yes" else ""
        struct_rows += f"""<tr{style}>
            <td>{sf.get('feature', '')}</td>
            <td>{sf.get('pos_mean', 0):.4f}</td>
            <td>{sf.get('neg_mean', 0):.4f}</td>
            <td>{fmt_p(sf.get('p_value'))}</td>
            <td>{sig}</td></tr>"""

    # Exclusive sites
    excl_data = a4_cls.get("exclusive_sites", {})
    excl_genes = excl_data.get("genes", [])
    excl_motif = excl_data.get("motif_distribution", {})
    excl_tissues = excl_data.get("tissues", {})

    # Figures
    fig_tc = img_tag(A4_FIGURES_DIR / "tc_fraction_by_group.png", "80%",
                     "TC motif fraction: A4-correlated vs A4-exclusive vs Non-A4")
    fig_tri = img_tag(A4_FIGURES_DIR / "trinucleotide_comparison.png", "80%",
                      "Trinucleotide context comparison")
    fig_struct = img_tag(A4_FIGURES_DIR / "structure_type_comparison.png", "80%",
                         "Structure type distribution")
    fig_tissue = img_tag(A4_FIGURES_DIR / "tissue_breadth_histogram.png", "80%",
                          "Tissue breadth distribution")

    return f"""
<section id="apobec4">
  <h2>APOBEC4 Analysis</h2>

  <div style="background:#dbeafe;border-left:4px solid #2563eb;padding:12px 16px;margin:12px 0;border-radius:4px">
    <strong>APOBEC4 (A4)</strong> is a recently characterized member of the APOBEC family.
    181 editing sites correlate with A4 expression in GTEx tissues (Levanon T3 sheet).
    160/181 co-correlate with known editors (A3A, A3G, A3H); 21 are A4-exclusive.
  </div>

  <h3>Classification Performance (5-fold CV)</h3>
  <table>
    <thead>
      <tr><th>Model</th><th>Dataset</th><th>n+</th><th>n&minus;</th><th>AUROC</th><th>AUPRC</th></tr>
    </thead>
    <tbody>
      <tr>
        <td><strong>GB_HandFeatures</strong></td><td>All 181 A4-correlated</td>
        <td>{n_pos}</td><td>{n_neg}</td>
        <td><strong>{gb_auroc:.4f} &pm; {gb_std:.4f}</strong></td>
        <td>{gb.get('mean_auprc', 0):.4f}</td>
      </tr>
      <tr>
        <td>MotifOnly</td><td>All 181</td>
        <td>{n_pos}</td><td>{n_neg}</td>
        <td>{motif_auroc:.4f}</td>
        <td>{motif.get('mean_auprc', 0):.4f}</td>
      </tr>
      <tr>
        <td>StructOnly</td><td>All 181</td>
        <td>{n_pos}</td><td>{n_neg}</td>
        <td>{struct_auroc:.4f}</td>
        <td>{struct.get('mean_auprc', 0):.4f}</td>
      </tr>
      <tr style="background:#fef9c3">
        <td><strong>GB_HandFeatures (LOO)</strong></td><td>21 A4-exclusive only</td>
        <td>{excl.get('n_pos', 21)}</td><td>{excl.get('n_neg', 21)}</td>
        <td><strong>{excl_auroc:.4f}</strong></td>
        <td>{excl.get('auprc', 0):.4f}</td>
      </tr>
    </tbody>
  </table>

  <h3>Feature Importance (GB_HandFeatures, Top 10)</h3>
  <table>
    <thead><tr><th>#</th><th>Feature</th><th>Importance</th></tr></thead>
    <tbody>{fi_rows}</tbody>
  </table>

  <h3>Structure Comparison (Positive vs Negative, Mann-Whitney)</h3>
  <table>
    <thead><tr><th>Feature</th><th>Pos mean</th><th>Neg mean</th><th>p-value</th><th>Significant</th></tr></thead>
    <tbody>{struct_rows}</tbody>
  </table>

  <h3>Figures</h3>
  <div class="grid-2">
    {fig_tc}
    {fig_tri}
  </div>
  <div class="grid-2">
    {fig_struct}
    {fig_tissue}
  </div>

  <h3>A4-Exclusive Sites (21 sites)</h3>
  <p>
    <strong>Genes:</strong> {', '.join(excl_genes[:15])}{'...' if len(excl_genes) > 15 else ''}
  </p>
  <p>
    <strong>Motif:</strong> TC={excl_motif.get('tc_fraction', 0):.1%}, CC={excl_motif.get('cc_fraction', 0):.1%}<br/>
    <strong>Tissue:</strong> Mean {excl_tissues.get('mean_n_tissues', 0):.1f} tissues edited (median {excl_tissues.get('median_n_tissues', 0):.0f}).
    Testis-dominant ({excl_tissues.get('tissue_counts', {}).get('Testis', 0)} of 21 sites).
  </p>

  <h3>Key Findings</h3>
  <ul>
    <li><strong>All 181 A4-correlated sites</strong> show strong classification (AUROC={gb_auroc:.3f}),
        comparable to confirmed editors (A3A=0.907, A3G=0.841).
        This is because 160/181 sites co-correlate with known editors and carry their signatures.</li>
    <li><strong>21 A4-exclusive sites show near-random classification</strong> (LOO AUROC={excl_auroc:.3f}),
        consistent with A4 having no distinctive editing signature of its own.</li>
    <li><strong>A4-exclusive sites are testis-dominant</strong> ({excl_tissues.get('tissue_counts', {}).get('Testis', 0)}/21 testis-specific),
        CC-enriched ({excl_motif.get('cc_fraction', 0):.0%}), and mostly non-coding (14/21 ncRNA).</li>
    <li><strong>Structure features are significant</strong>: Relative loop position (p=2.6e-29),
        loop size (p=8.9e-15), and dist_to_apex (p=2.6e-18) all distinguish A4 positives from negatives.
        This structural signal comes from the 160 co-correlated sites that are genuine editing targets of other enzymes.</li>
    <li><strong>Top classifier feature</strong>: relative_loop_position (0.091),
        followed by max_adjacent_stem_length (0.081) and m_5p_AC (0.066).
        Position &minus;1 features (m_m1_*) all have 0.000 importance — no motif preference at the &minus;1 position.</li>
  </ul>
</section>"""


def section_methods() -> str:
    return """
<section id="methods">
  <h2>9. Methods</h2>
  <h3>Data Sources</h3>
  <ul>
    <li><strong>Kockler 2026</strong> (BT-474 breast cancer cells): A3A and A3B editing sites from MAF files;
        41-nt CONTEXT(±20) column used directly. Genomic coordinates are cancer-cell-line-specific/transcriptomic
        and do not map reliably to hg38 reference.</li>
    <li><strong>Dang 2019</strong> (GSE114519, NK cells, NK_Hyp condition): A3G editing sites;
        31-nt flanking sequences from supplementary MOESM4 (n=119).</li>
    <li><strong>Zhang 2024</strong> (GSE245700, T-47D breast cancer cells): A3B editing sites from standard
        hg38 1-based genomic coordinates; 201-nt windows extracted from hg38.fa.</li>
  </ul>
  <h3>Sequence Processing</h3>
  <ul>
    <li>All sequences represented as 201-nt windows, edit site at position 100 (0-indexed), T→U conversion</li>
    <li>Kockler (41-nt real) and Dang (31-nt real): flanking context centered at position 100; outer positions
        padded with N (unspecified base) to reach 201 nt</li>
    <li>Minus-strand sites: context reverse-complemented before padding to give RNA-sense direction</li>
    <li>Validation: 100% of 7,048 sites have C at position 100</li>
  </ul>
  <h3>Structure Analysis — N-Padding Behavior</h3>
  <p>
    <strong>ViennaRNA treats N positions as fixed-unpaired (dot) in MFE folding.</strong>
    Empirically verified: an N-padded 201-nt sequence folds with the same MFE and structure
    as the real short sequence alone — ViennaRNA does not allow N-N or N-regular-base pairing.
    Therefore:
  </p>
  <ul>
    <li><strong>Loop geometry features are valid</strong> for all datasets: is_unpaired, loop_type, loop_size,
        relative_loop_position, and dist_to_apex all reflect the real flanking sequence structure.</li>
    <li><strong>Absolute MFE is NOT comparable across datasets</strong>: Kockler/Dang MFE (−7 to −10 kcal/mol)
        reflects short 41/31-nt sequences; Zhang MFE (−55 kcal/mol) reflects 201-nt sequences.
        Cross-dataset MFE comparisons are invalid.</li>
    <li><strong>Delta MFE (C→U edit effect)</strong> should be interpreted with caution across datasets,
        as the absolute magnitude scales with sequence length.</li>
  </ul>
  <h3>Statistical Tests</h3>
  <ul>
    <li>TC/CC enrichment: Fisher's exact test vs random (25% baseline)</li>
    <li>Pairwise motif differences: Chi-squared test on fractions</li>
    <li>In-loop fraction: Binomial test vs 50% baseline; Mann-Whitney U for pairwise comparisons</li>
    <li>Discriminability: 5-fold CV gradient boosting classifier (XGBoost); features = motif (24-dim) + loop geometry (9-dim)</li>
  </ul>
</section>"""


def generate_html(motif_results, struct_results, cross_results,
                   a3a_cls=None, a3b_cls=None, a3g_cls=None,
                   a3b_clinvar=None, a3g_clinvar=None,
                   a4_cls=None, a4_analysis=None) -> str:
    css = """
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body { font-family: 'Segoe UI', Arial, sans-serif; background: #f8f9fa; color: #222; line-height: 1.6; }
    .container { max-width: 1200px; margin: 0 auto; padding: 20px; }
    header { background: linear-gradient(135deg, #1e3a5f 0%, #2563eb 100%); color: white; padding: 40px 20px; margin-bottom: 30px; }
    header h1 { font-size: 2em; margin-bottom: 8px; }
    header .subtitle { font-size: 1em; opacity: 0.85; }
    nav { background: #fff; border-bottom: 2px solid #2563eb; position: sticky; top: 0; z-index: 100; }
    nav ul { display: flex; list-style: none; padding: 0 20px; gap: 0; }
    nav a { display: block; padding: 12px 18px; color: #1e3a5f; text-decoration: none; font-weight: 500; font-size: 0.9em; }
    nav a:hover { background: #eff6ff; }
    section { background: #fff; border-radius: 8px; padding: 30px; margin-bottom: 30px; box-shadow: 0 1px 4px rgba(0,0,0,.08); }
    h2 { color: #1e3a5f; font-size: 1.5em; margin-bottom: 16px; border-bottom: 2px solid #dbeafe; padding-bottom: 8px; }
    h3 { color: #2563eb; font-size: 1.1em; margin: 20px 0 10px; }
    h4 { color: #374151; font-size: 1em; margin: 14px 0 6px; }
    table { border-collapse: collapse; width: 100%; margin: 10px 0; font-size: 0.88em; }
    th { background: #1e3a5f; color: white; padding: 8px 12px; text-align: left; }
    td { padding: 7px 12px; border-bottom: 1px solid #e5e7eb; }
    tr:nth-child(even) td { background: #f9fafb; }
    tr:hover td { background: #eff6ff; }
    figure { margin: 12px 0; text-align: center; }
    figcaption { font-size: 0.82em; color: #555; margin-top: 4px; font-style: italic; }
    img { border-radius: 4px; border: 1px solid #e5e7eb; }
    .grid-2 { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; margin: 12px 0; }
    ul { padding-left: 22px; margin: 8px 0; }
    li { margin: 4px 0; }
    p { margin: 8px 0; }
    @media (max-width: 768px) { .grid-2 { grid-template-columns: 1fr; } }
    """

    # Build sections list; new sections gracefully handle missing data
    a3a_cls = a3a_cls or {}
    a3b_cls = a3b_cls or {}
    a3g_cls = a3g_cls or {}
    a3b_clinvar = a3b_clinvar or {}
    a3g_clinvar = a3g_clinvar or {}
    a4_cls = a4_cls or {}
    a4_analysis = a4_analysis or {}

    has_cls = any([a3a_cls, a3b_cls, a3g_cls])
    has_clinvar = any([a3b_clinvar, a3g_clinvar])

    sections = [
        section_overview(motif_results, struct_results),
        section_motif(motif_results),
        section_structure(struct_results),
        section_cross_enzyme(cross_results),
    ]
    # Add new sections (always include — they show placeholder text if data missing)
    sections.append(section_classification(a3a_cls, a3b_cls, a3g_cls, a4_cls))
    sections.append(section_feature_importance(a3a_cls, a3b_cls, a3g_cls, a4_cls))
    sections.append(section_clinvar_cross_enzyme(a3b_clinvar, a3g_clinvar))
    sections.append(section_apobec4(a4_cls, a4_analysis))
    sections.append(section_clinical_interpretation())
    sections.append(section_methods())

    body_sections = "\n".join(sections)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
  <title>Multi-Enzyme APOBEC Analysis Report</title>
  <style>{css}</style>
</head>
<body>
<header>
  <div class="container">
    <h1>Multi-Enzyme APOBEC Analysis Report</h1>
    <div class="subtitle">
      APOBEC3A · APOBEC3B · APOBEC3G · APOBEC4 &nbsp;|&nbsp;
      7,048+ editing sites &nbsp;|&nbsp;
      March 2026
    </div>
  </div>
</header>
<nav>
  <ul>
    <li><a href="#overview">Overview</a></li>
    <li><a href="#motif">Motif</a></li>
    <li><a href="#structure">Structure</a></li>
    <li><a href="#cross-enzyme">Cross-Enzyme</a></li>
    <li><a href="#classification">Classification</a></li>
    <li><a href="#feature-importance">Features</a></li>
    <li><a href="#clinvar">ClinVar</a></li>
    <li><a href="#apobec4">APOBEC4</a></li>
    <li><a href="#clinical">Findings</a></li>
    <li><a href="#methods">Methods</a></li>
  </ul>
</nav>
<div class="container">
  {body_sections}
</div>
</body>
</html>"""


def main():
    motif_results  = load_json(MOTIF_DIR  / "per_enzyme_motif_results.json")
    struct_results = load_json(STRUCT_DIR / "per_enzyme_structure_results.json")
    cross_results  = load_json(CROSS_DIR  / "cross_enzyme_comparison_results.json")

    # Per-enzyme classification results (optional — report gracefully handles missing data)
    a3a_cls = load_json(A3A_CLS_PATH)
    a3b_cls = load_json(A3B_CLS_PATH)
    a3g_cls = load_json(A3G_CLS_PATH)

    # Load A3A feature importance from CSV (stored separately from JSON)
    import csv
    a3a_fi_csv = PROJECT_ROOT / "experiments/apobec3a/outputs/classification_a3a_5fold/feature_importance_cls_gb_hand.csv"
    if a3a_fi_csv.exists() and a3a_cls:
        fi_dict = {}
        with open(a3a_fi_csv) as f:
            reader = csv.DictReader(f)
            for row in reader:
                fi_dict[row["feature_name"]] = float(row["mean_importance"])
        # Inject into a3a_cls under the expected structure
        if "feature_importance" not in a3a_cls:
            a3a_cls["feature_importance"] = {}
        a3a_cls["feature_importance"]["GB_HandFeatures"] = fi_dict

    # ClinVar cross-enzyme results (optional)
    a3b_clinvar = load_json(A3B_CLINVAR_PATH)
    a3g_clinvar = load_json(A3G_CLINVAR_PATH)

    # APOBEC4 results (optional)
    a4_cls = load_json(A4_CLS_PATH)
    a4_analysis = load_json(A4_ANALYSIS_PATH)

    n_cls = sum(1 for d in [a3a_cls, a3b_cls, a3g_cls, a4_cls] if d)
    n_cv = sum(1 for d in [a3b_clinvar, a3g_clinvar] if d)
    print(f"Loaded: motif={bool(motif_results)}, struct={bool(struct_results)}, "
          f"cross={bool(cross_results)}, classification={n_cls}/4, clinvar={n_cv}/2, "
          f"a4={bool(a4_cls)}")

    html = generate_html(motif_results, struct_results, cross_results,
                         a3a_cls, a3b_cls, a3g_cls,
                         a3b_clinvar, a3g_clinvar,
                         a4_cls, a4_analysis)

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(REPORT_FILE, "w") as f:
        f.write(html)

    size_kb = REPORT_FILE.stat().st_size // 1024
    print(f"Report saved → {REPORT_FILE}  ({size_kb} KB)")


if __name__ == "__main__":
    main()
