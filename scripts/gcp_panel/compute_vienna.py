#!/usr/bin/env python3
"""Run ViennaRNA folding (WT + edited) for every CDS candidate position.

Chunked, resumable. Writes one npz per chromosome so that (a) the job survives
VM restarts, (b) future work can extend the cache to other regions
(promoter, 3'UTR) without recomputing CDS, and (c) any downstream consumer
can load only the chroms it needs.

Per-chromosome outputs (on this VM):
  ~/data/panel/vienna/vienna_chr{chrom}.npz
    keys:
      chrom   : bytes  (e.g. b'chr1')
      pos     : (N,) int64   — 0-based genomic position
      strand  : (N,) bytes   — b'+' or b'-'
      struct  : (N, 16) float32   — 7-d delta + 9-d loop geom
      motif   : (N, 24) float32   — 24-d trinuc/dinuc motif
      done    : (N,) bool   — True for positions that have been folded
      valid   : (N,) bool   — False for positions where seq extraction failed
Global outputs:
  ~/data/panel/vienna_features_CDS.npz  (assembled after all chunks done)
    keys: chrom (N,), pos (N,), strand (N,), struct (N,16), motif (N,24),
          valid (N,), cand_idx (N,) int64 — order matches candidates_CDS.parquet
Progress log: ~/logs/vienna_progress.log (one line per ~50k positions).

Usage on ai-chem:
  source ~/miniconda3/etc/profile.d/conda.sh && conda activate apobec
  mkdir -p ~/data/panel/vienna ~/logs
  python compute_vienna.py \
      --candidates ~/data/panel/candidates_CDS.parquet \
      --hg19 ~/data/ref/hg19.fa \
      --out ~/data/panel/vienna \
      --progress-log ~/logs/vienna_progress.log \
      --workers 14
  # Smoke test:
  python compute_vienna.py ... --limit 50000
  # Resume: just re-run; already-done chroms are detected via done mask.
"""
from __future__ import annotations

import argparse
import logging
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from pyfaidx import Fasta

CENTER = 100
STRUCT_N = 16  # 7 struct delta + 9 loop
MOTIF_N = 24


def _setup_logging(log_path: Path, progress_log: Path | None):
    handlers = [logging.StreamHandler(sys.stdout), logging.FileHandler(log_path, mode="a")]
    if progress_log is not None:
        progress_log.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(progress_log, mode="a"))
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=handlers,
    )


COMP = str.maketrans("ACGTN", "TGCAN")


def extract_seq(genome, chrom, pos, strand, flank=100):
    try:
        chrom_len = len(genome[chrom])
        s, e = pos - flank, pos + flank + 1
        if s < 0 or e > chrom_len:
            return None
        seq = str(genome[chrom][s:e]).upper()
        if strand == "-":
            seq = seq.translate(COMP)[::-1]
        if len(seq) != 201 or seq[CENTER] != "C":
            return None
        return seq
    except Exception:
        return None


def extract_motif_from_seq(seq: str) -> np.ndarray:
    seq = seq.upper().replace("T", "U")
    ep = CENTER
    bases = ["A", "C", "G", "U"]
    up = seq[ep - 1] if ep > 0 else "N"
    down = seq[ep + 1] if ep < len(seq) - 1 else "N"
    feat_5p = [1.0 if up + "C" == m else 0.0 for m in ["UC", "CC", "AC", "GC"]]
    feat_3p = [1.0 if "C" + down == m else 0.0 for m in ["CA", "CG", "CU", "CC"]]
    trinuc_up = [0.0] * 8
    for offset, bo in [(-2, 0), (-1, 4)]:
        pos = ep + offset
        if 0 <= pos < len(seq):
            for bi, b in enumerate(bases):
                if seq[pos] == b:
                    trinuc_up[bo + bi] = 1.0
    trinuc_down = [0.0] * 8
    for offset, bo in [(1, 0), (2, 4)]:
        pos = ep + offset
        if 0 <= pos < len(seq):
            for bi, b in enumerate(bases):
                if seq[pos] == b:
                    trinuc_down[bo + bi] = 1.0
    return np.array(feat_5p + feat_3p + trinuc_up + trinuc_down, dtype=np.float32)


def fold_wrap(args):
    """Module-level wrapper so multiprocessing can pickle it."""
    idx, seq = args
    if seq is None:
        return idx, np.zeros(STRUCT_N, dtype=np.float32), np.zeros(MOTIF_N, dtype=np.float32), False
    return fold_one((idx, seq))


def fold_one(args):
    import RNA
    idx, seq = args
    motif = extract_motif_from_seq(seq)
    struct = np.zeros(STRUCT_N, dtype=np.float32)
    valid = True
    try:
        rna_wt = seq.replace("T", "U")
        rna_ed = rna_wt[:CENTER] + "U" + rna_wt[CENTER + 1:]

        fc_w = RNA.fold_compound(rna_wt)
        mfe_w_str, mfe_w = fc_w.mfe()
        fc_w.pf()
        bpp_w = np.array(fc_w.bpp())

        fc_e = RNA.fold_compound(rna_ed)
        mfe_e_str, mfe_e = fc_e.mfe()
        fc_e.pf()
        bpp_e = np.array(fc_e.bpp())

        def _center_p(bpp):
            return float(bpp[:, CENTER + 1].sum() + bpp[CENTER + 1, :].sum())

        def _window_p(bpp, half=10):
            n = bpp.shape[0] - 1
            lo, hi = max(1, CENTER + 1 - half), min(n, CENTER + 1 + half)
            return [float(bpp[:, i].sum() + bpp[i, :].sum()) for i in range(lo, hi + 1)]

        pw = _center_p(bpp_w); pe = _center_p(bpp_e)
        dp = pe - pw
        dm = float(mfe_e) - float(mfe_w)
        dw = np.array(_window_p(bpp_e)) - np.array(_window_p(bpp_w))

        def ent(p):
            if p <= 0 or p >= 1:
                return 0.0
            return -(p * np.log2(p + 1e-10) + (1 - p) * np.log2(1 - p + 1e-10))

        de = ent(pe) - ent(pw)
        md = float(np.mean(dw)) if len(dw) else 0.0
        sd = float(np.std(dw)) if len(dw) else 0.0
        # BUGFIX (2026-04-24): canonical layout is
        # [dp_c, da_c, d_entropy, d_mfe, mean_dp_win, mean_da_win, std_dp_win].
        # Prior order swapped slots 5 and 6 (std vs mean_da). All prior ai-chem-
        # produced vienna_chr{N}.npz used the buggy order — if any downstream
        # consumer compares against this cache, they saw the swapped columns.
        # In the current mfe_only regime these 7 slots are zeroed at inference
        # anyway, so this mostly matters for future runs that DO use struct_delta.
        struct_delta = np.array([dp, -dp, de, dm, md, -md, sd], dtype=np.float32)

        s = mfe_w_str
        up = 1.0 if s[CENTER] == "." else 0.0
        ls = dj = da = 0.0
        rlp = 0.5
        lst = rst = mas = lu = 0.0
        if up:
            l = CENTER
            while l > 0 and s[l] == ".":
                l -= 1
            r = CENTER
            while r < len(s) - 1 and s[r] == ".":
                r += 1
            ls = float(r - l - 1)
            if ls > 0:
                p = CENTER - l - 1
                rlp = p / max(ls - 1, 1)
                da = abs(p - (ls - 1) / 2)
            i = l; c = 0
            while i >= 0 and s[i] in "()":
                c += 1; i -= 1
            lst = float(c)
            i = r; c = 0
            while i < len(s) and s[i] in "()":
                c += 1; i += 1
            rst = float(c)
            mas = max(lst, rst)
        reg = s[max(0, CENTER - 10):min(len(s), CENTER + 11)]
        lu = sum(1 for ch in reg if ch == ".") / max(len(reg), 1)
        loop_features = np.array([up, ls, dj, da, rlp, lst, rst, mas, lu], dtype=np.float32)

        struct[:7] = struct_delta
        struct[7:16] = loop_features
    except Exception:
        valid = False
    return idx, struct, motif, valid


def process_chrom(chrom, sub, genome, out_dir, workers, chunksize, logger,
                   total_done_so_far, total_n, t0_global):
    """Fold all positions for one chromosome, checkpointing every ~50k rows.
    Returns (n_processed_this_chrom, final_done_count)."""
    chrom_safe = chrom.replace("chr", "")
    npz_path = out_dir / f"vienna_{chrom}.npz"
    n = len(sub)
    pos_arr = sub["pos"].values.astype(np.int64)
    strand_arr = np.array([s.encode() for s in sub["strand"].values], dtype="S1")
    struct_arr = np.zeros((n, STRUCT_N), dtype=np.float32)
    motif_arr = np.zeros((n, MOTIF_N), dtype=np.float32)
    done_mask = np.zeros(n, dtype=bool)
    valid_mask = np.zeros(n, dtype=bool)

    # Resume from existing npz if present
    if npz_path.exists():
        try:
            z = np.load(npz_path, allow_pickle=False)
            if (z["pos"].shape == pos_arr.shape
                and np.array_equal(z["pos"], pos_arr)
                and np.array_equal(z["strand"], strand_arr)):
                struct_arr = z["struct"].astype(np.float32)
                motif_arr = z["motif"].astype(np.float32)
                done_mask = z["done"].astype(bool)
                valid_mask = z["valid"].astype(bool)
                logger.info("  resume %s: %d/%d done", chrom, int(done_mask.sum()), n)
            else:
                logger.warning("  %s: stale npz (shape/pos mismatch) — restarting chrom", chrom)
        except Exception as ex:
            logger.warning("  %s: resume failed (%s) — restarting chrom", chrom, ex)

    remaining = int((~done_mask).sum())
    if remaining == 0:
        logger.info("  %s: already complete (%d rows)", chrom, n)
        return 0, int(done_mask.sum())

    # Build seq iterator only for remaining indices (pull seqs via pyfaidx)
    def seq_gen():
        for i in range(n):
            if done_mask[i]:
                continue
            seq = extract_seq(genome, chrom, int(pos_arr[i]),
                              strand_arr[i].decode() if isinstance(strand_arr[i], bytes) else strand_arr[i])
            if seq is None:
                # emit a placeholder — fold_one will succeed on N*201 but valid stays False
                yield (i, None)
            else:
                yield (i, seq)

    t0 = time.time()
    processed = 0
    since_save = 0
    save_every = 50_000
    log_every = 50_000  # aligns with progress-log cadence

    with mp.Pool(workers) as pool:
        for idx, struct, motif, valid in pool.imap_unordered(fold_wrap, seq_gen(), chunksize=chunksize):
            struct_arr[idx] = struct
            motif_arr[idx] = motif
            done_mask[idx] = True
            valid_mask[idx] = bool(valid)
            processed += 1
            since_save += 1
            total_done_so_far_local = total_done_so_far + processed
            if processed % log_every == 0 or processed == remaining:
                elapsed = time.time() - t0
                rate = processed / max(elapsed, 1e-9)
                elapsed_global = time.time() - t0_global
                # ETA is based on the chrom-average rate but scaled by remaining global rows.
                global_rate = (processed) / max(elapsed, 1e-9)
                remain_global = max(0, total_n - total_done_so_far_local) / max(global_rate, 1e-9)
                logger.info(
                    "[%s] %s %d/%d (%.1f/s, ETA %.1fh)",
                    time.strftime("%Y-%m-%dT%H:%M:%S"),
                    chrom, processed, remaining, rate, remain_global / 3600.0,
                )
            if since_save >= save_every:
                np.savez(npz_path,
                         chrom=np.array([chrom], dtype="S16"),
                         pos=pos_arr, strand=strand_arr,
                         struct=struct_arr, motif=motif_arr,
                         done=done_mask, valid=valid_mask)
                since_save = 0

    np.savez(npz_path,
             chrom=np.array([chrom], dtype="S16"),
             pos=pos_arr, strand=strand_arr,
             struct=struct_arr, motif=motif_arr,
             done=done_mask, valid=valid_mask)
    logger.info("  %s: done — %d/%d, %d valid", chrom, int(done_mask.sum()), n, int(valid_mask.sum()))
    return processed, int(done_mask.sum())


def assemble_global(cand: pd.DataFrame, out_dir: Path, logger):
    """Concatenate per-chrom npz into one global npz aligned with `cand` row order."""
    n = len(cand)
    struct = np.zeros((n, STRUCT_N), dtype=np.float32)
    motif = np.zeros((n, MOTIF_N), dtype=np.float32)
    valid = np.zeros(n, dtype=bool)
    # Build a (chrom, pos, strand) → row index for cand
    key = (cand["chrom"].astype(str).values.astype("U16"),
           cand["pos"].values.astype(np.int64),
           cand["strand"].astype(str).values.astype("U1"))
    cand_idx_df = pd.DataFrame({"chrom": key[0], "pos": key[1], "strand": key[2],
                                 "row": np.arange(n, dtype=np.int64)})
    cand_idx_df = cand_idx_df.set_index(["chrom", "pos", "strand"])

    missing = 0
    for chrom in sorted(cand["chrom"].unique()):
        npz_path = out_dir / f"vienna_{chrom}.npz"
        if not npz_path.exists():
            logger.warning("  missing chrom npz: %s", npz_path)
            missing += 1
            continue
        z = np.load(npz_path, allow_pickle=False)
        pos = z["pos"]
        strand = np.array([s.decode() if isinstance(s, bytes) else s for s in z["strand"]], dtype="U1")
        chrom_col = np.full(pos.shape, chrom, dtype="U16")
        sub_idx = pd.MultiIndex.from_arrays([chrom_col, pos, strand])
        rows = cand_idx_df.reindex(sub_idx)["row"].values
        mask = ~pd.isna(rows)
        rows = rows[mask].astype(np.int64)
        struct[rows] = z["struct"][mask]
        motif[rows] = z["motif"][mask]
        valid[rows] = z["valid"][mask]
    chrom_col = cand["chrom"].astype(str).values.astype("U16")
    pos_col = cand["pos"].values.astype(np.int64)
    strand_col = cand["strand"].astype(str).values.astype("U1")
    hand40 = np.concatenate([motif, struct], axis=1).astype(np.float32)

    out_npz = out_dir.parent / "vienna_features_CDS.npz"
    np.savez(out_npz,
             chrom=chrom_col, pos=pos_col, strand=strand_col,
             struct=struct, motif=motif, hand40=hand40, valid=valid,
             cand_idx=np.arange(n, dtype=np.int64))
    # Also write hand40 as a plain npy for score_panel.py
    hand_path = out_dir.parent / "hand40.npy"
    np.save(hand_path, hand40)
    valid_path = out_dir.parent / "valid_vienna.npy"
    np.save(valid_path, valid)
    logger.info("Assembled %s (%.1f MB), hand40=%s, missing_chroms=%d",
                out_npz, out_npz.stat().st_size / 1e6, hand_path, missing)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", required=True, type=Path)
    ap.add_argument("--hg19", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path, help="Per-chrom npz directory")
    ap.add_argument("--progress-log", type=Path, default=None)
    ap.add_argument("--workers", type=int, default=14)
    ap.add_argument("--limit", type=int, default=0, help="Smoke test: process only first N positions overall")
    ap.add_argument("--chunksize", type=int, default=200)
    ap.add_argument("--skip-assemble", action="store_true",
                    help="Skip the final global-npz assembly (useful for smoke tests)")
    ap.add_argument("--chrom", type=str, default="", help="Only process this chromosome (debug)")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    _setup_logging(args.out / "run.log", args.progress_log)
    logger = logging.getLogger(__name__)

    logger.info("=" * 60)
    logger.info("compute_vienna.py starting (chunked, resumable)")
    logger.info("  candidates: %s", args.candidates)
    logger.info("  hg19:       %s", args.hg19)
    logger.info("  out:        %s", args.out)
    logger.info("  workers:    %d", args.workers)

    cand = pd.read_parquet(args.candidates)
    # Always sort so chunk boundaries are stable.
    cand = cand.sort_values(["chrom", "pos", "strand"]).reset_index(drop=True)
    n_total = len(cand)
    logger.info("  %d total candidates", n_total)
    if args.limit > 0:
        cand = cand.iloc[:args.limit].reset_index(drop=True)
        logger.info("  SMOKE: limited to %d rows", len(cand))

    logger.info("Loading hg19 genome ...")
    genome = Fasta(str(args.hg19))

    total_n = len(cand)
    t0_global = time.time()
    total_done = 0

    chrom_list = sorted(cand["chrom"].unique())
    if args.chrom:
        chrom_list = [args.chrom]
        cand = cand[cand["chrom"] == args.chrom].reset_index(drop=True)

    for chrom in chrom_list:
        sub = cand[cand["chrom"] == chrom].reset_index(drop=True)
        if len(sub) == 0:
            continue
        proc, done = process_chrom(chrom, sub, genome, args.out, args.workers,
                                    args.chunksize, logger,
                                    total_done_so_far=total_done,
                                    total_n=total_n, t0_global=t0_global)
        total_done += done

    logger.info("All chroms processed. Total elapsed: %.1f min", (time.time() - t0_global) / 60)
    if not args.skip_assemble and not args.limit:
        logger.info("Assembling global npz ...")
        assemble_global(cand, args.out, logger)
    # Write completion flag
    flag = Path.home() / "phase_2a_done.flag"
    try:
        flag.write_text(f"vienna_CDS done at {time.strftime('%Y-%m-%dT%H:%M:%S')} — "
                        f"{total_n} rows, {(time.time()-t0_global)/60:.1f} min\n")
    except Exception:
        pass


if __name__ == "__main__":
    main()
