#!/usr/bin/env python3
"""Plot Phase 1.6 sweep: mean recall ratio vs window size for A and B.

Reads sweep JSONs:
  experiments/multi_enzyme/outputs/pcawg_tcw_panel/analysis_A_pcawg_wgs/
    enrichment_primary_phase1_6_A_w{100,250,500,1000,2000}.json
  same for analysis_B_coding_panel with kind=B.

Output: phase1_6_sweep.png (two panels: A and B).
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PANEL_DIR = PROJECT_ROOT / "experiments/multi_enzyme/outputs/pcawg_tcw_panel"
WINDOWS = [100, 250, 500, 1000, 2000]


def load_sweep(kind: str):
    sub = "analysis_A_pcawg_wgs" if kind == "A" else "analysis_B_coding_panel"
    out = {}
    for w in WINDOWS:
        path = PANEL_DIR / sub / f"enrichment_primary_phase1_6_{kind}_w{w}.json"
        if not path.exists():
            print(f"  missing {path}")
            continue
        try:
            j = json.load(open(path))
        except Exception as ex:
            print(f"  load error {path}: {ex}")
            continue
        per_cancer = []
        for cancer, det in j.get("per_cancer", {}).items():
            r = det.get("primary", {}).get("ratio", float("nan"))
            if r == r:
                per_cancer.append(r)
        out[w] = {
            "mean": float(np.mean(per_cancer)) if per_cancer else float("nan"),
            "median": float(np.median(per_cancer)) if per_cancer else float("nan"),
            "ratios": per_cancer,
            "pc_lo": float(np.percentile(per_cancer, 25)) if per_cancer else float("nan"),
            "pc_hi": float(np.percentile(per_cancer, 75)) if per_cancer else float("nan"),
            "n_signif": j.get("pooled", {}).get("n_cancers_signif_q05", 0),
            "PASS": j.get("pass_criteria", {}).get("PASS", False),
        }
    return out


def main():
    a = load_sweep("A")
    b = load_sweep("B")
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), sharey=False)

    for ax, sweep, title in zip(axes, (a, b), ("Analysis A - PCAWG WGS",
                                                "Analysis B - TCGA + PCAWG-coding")):
        if not sweep:
            ax.text(0.5, 0.5, "(no data)", ha="center", va="center")
            ax.set_title(title)
            continue
        ws = sorted(sweep.keys())
        means = [sweep[w]["mean"] for w in ws]
        los = [sweep[w]["pc_lo"] for w in ws]
        his = [sweep[w]["pc_hi"] for w in ws]
        for w in ws:
            for r in sweep[w]["ratios"]:
                ax.scatter([w], [r], color="lightgray", s=15, alpha=0.5, zorder=1)
        ax.fill_between(ws, los, his, alpha=0.18, color="C0", label="IQR (25-75%)", zorder=2)
        ax.plot(ws, means, "o-", color="C0", lw=2.4, markersize=8,
                label="mean recall ratio", zorder=3)
        for w, m in zip(ws, means):
            ax.annotate(f"  q<.025: {sweep[w]['n_signif']}/{len(sweep[w]['ratios'])}",
                        xy=(w, m), fontsize=8, color="darkblue", zorder=4)
        ax.axhline(1.5, color="red", linestyle="--", alpha=0.7, label="threshold 1.5x")
        ax.axhline(1.0, color="gray", linestyle=":", alpha=0.6, label="null 1.0x")
        ax.set_xscale("log")
        ax.set_xticks(ws)
        ax.set_xticklabels([f"{w}" for w in ws])
        ax.set_xlabel("Window size (bp)")
        ax.set_ylabel("Recall ratio (model / cpg-density)")
        ax.set_title(title)
        ax.legend(loc="upper right", fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.suptitle("Phase 1.6 sweep - recall ratio vs window size (max-pool, 10K perms)",
                 fontsize=12, y=1.00)
    plt.tight_layout()
    out = PANEL_DIR / "phase1_6_sweep.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
