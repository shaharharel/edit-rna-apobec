#!/usr/bin/env python3
"""Full-rewrite PHASED_STATUS.md every 10 min based on the live state of:
- ai-gpu2 RNA-FM CDS run + score watcher
- local analysis A and B status

Run this in the background on local Mac:
    python3 scripts/gcp_panel/update_phased_status.py --interval 600 &

It reads the gcloud SSH outputs every cycle and rewrites PHASED_STATUS.md as a
single-file tail-able status. NEVER appends — always full rewrite.
"""
from __future__ import annotations
import argparse
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
STATUS = ROOT / "experiments/multi_enzyme/outputs/pcawg_tcw_panel/PHASED_STATUS.md"
RETRY_LOG = ROOT / "experiments/multi_enzyme/outputs/pcawg_tcw_panel/.retry_log.txt"


def ssh(cmd: str, timeout: int = 60) -> str:
    out = subprocess.run(
        ["gcloud", "compute", "ssh", "ai-gpu2",
         "--zone=us-central1-a", "--tunnel-through-iap",
         "--command", cmd],
        capture_output=True, text=True, timeout=timeout,
    )
    return (out.stdout or "") + (out.stderr or "")


def get_rnafm_status() -> dict:
    out = ssh(
        "tmux ls 2>/dev/null | grep rnafm_cds; "
        "tail -3 ~/logs/rnafm_cds.log; "
        "ls ~/data/panel/rnafm/ 2>/dev/null | wc -l; "
        "df -h / | tail -1; "
        "nvidia-smi --query-gpu=utilization.gpu,memory.used,temperature.gpu --format=csv,noheader 2>/dev/null"
    )
    lines = out.splitlines()
    tmux_alive = any("rnafm_cds" in l for l in lines)
    log_lines = [l for l in lines if "ETA" in l or "INFO" in l][-3:]
    # Parse rate
    rate = None; eta = None; pct = None; chrom = None
    for l in log_lines:
        m = re.search(r"(chr\w+) (\d+)/(\d+) \(([\d.]+)/s, ETA ([\d.]+)h\)", l)
        if m:
            chrom = m.group(1)
            done, total = int(m.group(2)), int(m.group(3))
            rate = float(m.group(4))
            eta = float(m.group(5))
            pct = 100 * done / max(total, 1)
            break
    n_npz = 0
    for l in lines:
        if l.strip().isdigit():
            n_npz = int(l.strip()); break
    disk = next((l for l in lines if "/dev/root" in l or "%" in l and "G" in l), "")
    gpu = next((l for l in lines if "%" in l and "MiB" in l), "")
    return {
        "tmux_alive": tmux_alive,
        "log_tail": "\n".join(log_lines),
        "rate_seqs_per_s": rate,
        "eta_hours": eta,
        "current_chrom": chrom,
        "current_pct": pct,
        "n_npz_files": n_npz,
        "disk": disk.strip(),
        "gpu": gpu.strip(),
    }


def get_watcher_status() -> dict:
    out = ssh(
        "tmux ls 2>/dev/null | grep score_watcher_cds; "
        "tail -5 ~/logs/score_watcher_cds_stdout.log; "
        "ls ~/data/panel/scored_chroms_cds/ 2>/dev/null | wc -l; "
        "ls ~/data/panel/scored_chroms_cds/ 2>/dev/null"
    )
    lines = out.splitlines()
    tmux_alive = any("score_watcher_cds" in l for l in lines)
    n_scored = 0
    for l in lines:
        if l.strip().isdigit():
            n_scored = int(l.strip()); break
    scored = sorted(l for l in lines if l.endswith(".parquet"))
    log_tail = "\n".join(l for l in lines if "[INFO]" in l or "[ERROR]" in l)[-1500:]
    return {
        "tmux_alive": tmux_alive,
        "n_scored": n_scored,
        "scored_files": scored,
        "log_tail": log_tail,
    }


def get_local_status() -> dict:
    panel_path = ROOT / "experiments/multi_enzyme/outputs/pcawg_tcw_panel/panel_scores_cds.parquet"
    a_done = (ROOT / "experiments/multi_enzyme/outputs/pcawg_tcw_panel/analysis_A_pcawg_wgs/REPORT.md").exists()
    b_done = (ROOT / "experiments/multi_enzyme/outputs/pcawg_tcw_panel/analysis_B_coding_panel/REPORT.md").exists()
    cmp_done = (ROOT / "experiments/multi_enzyme/outputs/pcawg_tcw_panel/COMPARISON_PHASE1.md").exists()
    final_done = (ROOT / "experiments/multi_enzyme/outputs/pcawg_tcw_panel/FINAL_REPORT_PHASE1.md").exists()
    flag_done = (ROOT / "experiments/multi_enzyme/outputs/pcawg_tcw_panel/PHASE_1_DONE.flag").exists()
    return {
        "panel_local": panel_path.exists(),
        "panel_size_mb": panel_path.stat().st_size / 1e6 if panel_path.exists() else 0,
        "analysis_A_done": a_done,
        "analysis_B_done": b_done,
        "comparison_done": cmp_done,
        "final_report_done": final_done,
        "phase_1_done_flag": flag_done,
    }


def render(rfm, watcher, local) -> str:
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines = [
        f"# Phased Pipeline — pcawg_tcw_panel\n",
        f"**Phase**: 1 (CDS panel, 8.45 M positions)",
        f"**Last update**: {now}",
        "",
        "> Status file is full-rewritten by `update_phased_status.py` every ~10 min. NEVER appended.",
        "",
        "## Phase 1 streams\n",
        "| ID | Stream | State | Detail |",
        "|----|--------|-------|--------|",
    ]
    # P1.1
    if rfm.get("tmux_alive"):
        s = (f"RUNNING — chrom={rfm.get('current_chrom')} "
             f"({rfm.get('current_pct') or 0:.1f}%), "
             f"{rfm.get('rate_seqs_per_s') or 0:.0f}/s, "
             f"ETA {rfm.get('eta_hours') or 0:.1f}h")
    elif local.get("panel_local"):
        s = "DONE"
    else:
        s = "NOT RUNNING"
    lines.append(f"| P1.1 | RNA-FM CDS on ai-gpu2 V100 | {s} | npz files: {rfm.get('n_npz_files', 0)}/24 |")
    # P1.2
    if watcher.get("tmux_alive"):
        ws = f"RUNNING — {watcher.get('n_scored')}/24 scored"
    elif watcher.get("n_scored", 0) >= 24:
        ws = "DONE — 24/24 scored"
    else:
        ws = f"NOT RUNNING — {watcher.get('n_scored', 0)}/24 scored"
    lines.append(f"| P1.2 | Score watcher on ai-gpu2 | {ws} | {', '.join(watcher.get('scored_files', [])) or '(none yet)'} |")
    # P1.3
    lines.append(f"| P1.3 | Pre-registration | DONE (PRE_REGISTRATION.md present) | — |")
    # P1.4
    lines.append(f"| P1.4 | Analysis A (PCAWG WGS) | {'DONE' if local.get('analysis_A_done') else 'WAITING for panel'} | — |")
    # P1.5
    lines.append(f"| P1.5 | Analysis B (TCGA + PCAWG coding) | {'DONE' if local.get('analysis_B_done') else 'WAITING for panel'} | — |")
    # P1.6
    lines.append(f"| P1.6 | Comparison + final report | {'DONE' if local.get('final_report_done') else 'WAITING'} | — |")

    lines += [
        "",
        "## Phase 2 streams (idle until user authorization)\n",
        "| ID | Stream | State |",
        "|----|--------|-------|",
        "| P2.* | Vienna MFE-only on ai-chem (regulatory) | IDLE |",
        "| P2.* | RNA-FM regulatory on ai-gpu2 | IDLE |",
        "| P2.* | Score regulatory + re-run analyses at 28.6 M scope | IDLE |",
        "",
        "## VM state",
        "",
        f"### ai-gpu2 (us-central1-a, V100)",
        f"- Disk: `{rfm.get('disk', 'unknown')}`",
        f"- GPU: `{rfm.get('gpu', 'unknown')}`",
        f"- npz files in ~/data/panel/rnafm/: {rfm.get('n_npz_files', 0)}/24",
        f"- scored chroms in ~/data/panel/scored_chroms_cds/: {watcher.get('n_scored', 0)}/24",
        "",
        "### Local Mac",
        f"- panel_scores_cds.parquet present: {local.get('panel_local')}",
        f"- panel_size_mb: {local.get('panel_size_mb', 0):.1f}",
        f"- Analysis A REPORT.md: {local.get('analysis_A_done')}",
        f"- Analysis B REPORT.md: {local.get('analysis_B_done')}",
        f"- COMPARISON_PHASE1.md: {local.get('comparison_done')}",
        f"- FINAL_REPORT_PHASE1.md: {local.get('final_report_done')}",
        f"- PHASE_1_DONE.flag: {local.get('phase_1_done_flag')}",
        "",
        "## Recent log lines",
        "",
        "### P1.1 RNA-FM CDS (last 3 ETA lines)",
        "```",
        rfm.get("log_tail", "(none)"),
        "```",
        "",
        "### P1.2 Score watcher (last 5 INFO lines)",
        "```",
        watcher.get("log_tail", "(none)"),
        "```",
        "",
        "## QA fixes applied (pre-analysis)",
        "",
        "Pre-registration committed at git hash `a350c26` (timestamp proof).",
        "Fix log: `FIXES_APPLIED.md`. Summary:",
        "- B1 SBS join broken => cancer-level SBS aggregation",
        "- B2 PRIMARY_FILTER mismatch => set to apobec_signature",
        "- B3 Fisher 2x2 malformed => permutation null on score labels",
        "- M1 v3 site_id parser => use chr/start columns",
        "- M2 hg38 in mask => filter coordinate_system='hg19'",
        "- M3 fake timestamp => git-committed pre-reg",
        "- M4 TCW minus-strand bug => corrected rev-comp (unit tested)",
        "- M5 spurious CGC list => Bailey-style minus length confounders",
        "",
        "## Retry log",
        "",
    ]
    if RETRY_LOG.exists():
        lines.append("```")
        lines.append(RETRY_LOG.read_text()[-2000:])
        lines.append("```")
    else:
        lines.append("(none)")
    lines += [
        "",
        "## ETA",
        "",
    ]
    if rfm.get("eta_hours"):
        lines.append(f"- P1.1 RNA-FM finish ≈ {rfm['eta_hours']:.1f} h from now")
    if watcher.get("n_scored", 0) > 0:
        lines.append(f"- P1.2 trails P1.1; per-chrom scoring ~5-15 min")
    lines.append(f"- Phase 1 done ≈ when P1.1 finishes + ~2 h analyses")
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--interval", type=int, default=600, help="Sleep seconds between updates")
    ap.add_argument("--once", action="store_true")
    args = ap.parse_args()

    while True:
        try:
            rfm = get_rnafm_status()
        except Exception as ex:
            rfm = {"error": str(ex)}
        try:
            watcher = get_watcher_status()
        except Exception as ex:
            watcher = {"error": str(ex)}
        local = get_local_status()
        try:
            STATUS.write_text(render(rfm, watcher, local))
        except Exception as ex:
            print("write error:", ex, file=sys.stderr)
        print(f"[{datetime.now().strftime('%H:%M:%S')}] updated PHASED_STATUS.md "
              f"(rfm chrom={rfm.get('current_chrom')} {rfm.get('current_pct', 0):.1f}%, "
              f"scored {watcher.get('n_scored', 0)}/24)",
              flush=True)
        if args.once:
            return
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
