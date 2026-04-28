#!/usr/bin/env python3
"""Compute RNA-FM pooled embeddings (orig + edit delta) for every candidate.

Chunked, resumable. Writes one npz per chromosome so that: (a) the job survives
VM restarts, (b) future work can rescore only new positions without
recomputing the whole cache, and (c) downstream consumers can load per-chrom.

Per-chromosome outputs:
  ~/data/panel/rnafm/rnafm_chr{chrom}.npz
    keys:
      chrom  : (1,) bytes
      pos    : (N,) int64
      strand : (N,) bytes
      orig   : (N, 640) fp16
      delta  : (N, 640) fp16
      done   : (N,) bool
      valid  : (N,) bool

Global outputs (assembled after all chroms done):
  ~/data/panel/rnafm_orig.npy    (N, 640) fp16
  ~/data/panel/rnafm_delta.npy   (N, 640) fp16
  ~/data/panel/valid_rnafm.npy   (N,) bool

Progress log: ~/logs/rnafm_progress.log (one line per ~0.5% of progress).

Usage (ai-gpu2):
  python3 compute_rnafm.py \
      --candidates ~/data/panel/candidates_all.parquet \
      --hg19 ~/data/ref/hg19.fa \
      --out ~/data/panel/rnafm \
      --progress-log ~/logs/rnafm_progress.log \
      --batch 64 --fp16
  # Smoke test:
  python3 compute_rnafm.py ... --limit 50000
"""
from __future__ import annotations

import argparse
import gc
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from pyfaidx import Fasta

CENTER = 100
EMB_DIM = 640


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


def load_rnafm(device):
    import fm
    logger = logging.getLogger(__name__)
    logger.info("Loading RNA-FM checkpoint ...")
    model, alphabet = fm.pretrained.rna_fm_t12()
    batch_converter = alphabet.get_batch_converter()
    model = model.eval().to(device)
    return model, batch_converter


def embed_batch(seqs, model, batch_converter, device, use_fp16):
    data = [(f"s{i}", s.replace("T", "U")) for i, s in enumerate(seqs)]
    _, _, tokens = batch_converter(data)
    tokens = tokens.to(device)
    with torch.no_grad():
        if use_fp16:
            with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                r = model(tokens, repr_layers=[12])
        else:
            r = model(tokens, repr_layers=[12])
        emb = r["representations"][12][:, 1:-1, :].mean(dim=1)
    return emb.float().cpu().numpy()


def process_chrom(chrom, sub, genome, model, batch_converter, device,
                   out_dir, batch_size, use_fp16, logger,
                   total_done_so_far, total_n, t0_global):
    npz_path = out_dir / f"rnafm_{chrom}.npz"
    n = len(sub)
    pos_arr = sub["pos"].values.astype(np.int64)
    strand_arr = np.array([s.encode() for s in sub["strand"].values], dtype="S1")
    dtype = np.float16 if use_fp16 else np.float32
    orig_arr = np.zeros((n, EMB_DIM), dtype=dtype)
    delta_arr = np.zeros((n, EMB_DIM), dtype=dtype)
    done_mask = np.zeros(n, dtype=bool)
    valid_mask = np.zeros(n, dtype=bool)

    if npz_path.exists():
        try:
            z = np.load(npz_path, allow_pickle=False)
            if (z["pos"].shape == pos_arr.shape
                and np.array_equal(z["pos"], pos_arr)
                and np.array_equal(z["strand"], strand_arr)):
                orig_arr = z["orig"].astype(dtype)
                delta_arr = z["delta"].astype(dtype)
                done_mask = z["done"].astype(bool)
                valid_mask = z["valid"].astype(bool)
                logger.info("  resume %s: %d/%d done", chrom, int(done_mask.sum()), n)
            else:
                logger.warning("  %s: stale npz — restarting chrom", chrom)
        except Exception as ex:
            logger.warning("  %s: resume failed (%s) — restarting chrom", chrom, ex)

    remaining = int((~done_mask).sum())
    if remaining == 0:
        logger.info("  %s: already complete (%d rows)", chrom, n)
        return 0, int(done_mask.sum())

    t0 = time.time()
    processed = 0
    since_ckpt = 0
    ckpt_every = 200_000
    log_every = max(batch_size * 10, remaining // 200 if remaining > 200 * batch_size else batch_size * 10)

    i = 0
    while i < n:
        batch_idx = []
        batch_seqs_orig = []
        batch_seqs_edit = []
        while i < n and len(batch_idx) < batch_size:
            if done_mask[i]:
                i += 1
                continue
            seq = extract_seq(genome, chrom, int(pos_arr[i]),
                              strand_arr[i].decode() if isinstance(strand_arr[i], bytes) else strand_arr[i])
            if seq is None:
                done_mask[i] = True
                valid_mask[i] = False
                processed += 1
                since_ckpt += 1
                i += 1
                continue
            batch_idx.append(i)
            batch_seqs_orig.append(seq)
            batch_seqs_edit.append(seq[:CENTER] + "U" + seq[CENTER + 1:])
            i += 1
        if not batch_idx:
            continue

        try:
            emb_o = embed_batch(batch_seqs_orig, model, batch_converter, device, use_fp16)
            emb_e = embed_batch(batch_seqs_edit, model, batch_converter, device, use_fp16)
        except torch.cuda.OutOfMemoryError:
            logger.warning("OOM at batch starting %d, fallback to size-1 ...", batch_idx[0])
            emb_o = np.zeros((len(batch_seqs_orig), EMB_DIM), dtype=np.float32)
            emb_e = np.zeros_like(emb_o)
            torch.cuda.empty_cache()
            for k in range(len(batch_seqs_orig)):
                emb_o[k:k+1] = embed_batch(batch_seqs_orig[k:k+1], model, batch_converter, device, use_fp16)
                emb_e[k:k+1] = embed_batch(batch_seqs_edit[k:k+1], model, batch_converter, device, use_fp16)
        except Exception as ex:
            # Log the batch and continue — do not halt the pipeline.
            logger.warning("Batch exception at idx %d: %s — marking invalid", batch_idx[0], ex)
            for idx in batch_idx:
                done_mask[idx] = True
                valid_mask[idx] = False
            processed += len(batch_idx)
            since_ckpt += len(batch_idx)
            continue

        delta = (emb_e - emb_o).astype(dtype)
        emb_o_d = emb_o.astype(dtype)
        for k, idx in enumerate(batch_idx):
            orig_arr[idx] = emb_o_d[k]
            delta_arr[idx] = delta[k]
            valid_mask[idx] = True
            done_mask[idx] = True
        processed += len(batch_idx)
        since_ckpt += len(batch_idx)

        if processed % log_every < batch_size or processed == remaining:
            elapsed = time.time() - t0
            rate = processed / max(elapsed, 1e-9)
            total_so_far = total_done_so_far + processed
            remain_global = max(0, total_n - total_so_far) / max(rate, 1e-9)
            logger.info(
                "[%s] %s %d/%d (%.1f/s, ETA %.1fh)",
                time.strftime("%Y-%m-%dT%H:%M:%S"),
                chrom, processed, remaining, rate, remain_global / 3600.0,
            )

        if since_ckpt >= ckpt_every:
            np.savez(npz_path,
                     chrom=np.array([chrom], dtype="S16"),
                     pos=pos_arr, strand=strand_arr,
                     orig=orig_arr, delta=delta_arr,
                     done=done_mask, valid=valid_mask)
            since_ckpt = 0

    np.savez(npz_path,
             chrom=np.array([chrom], dtype="S16"),
             pos=pos_arr, strand=strand_arr,
             orig=orig_arr, delta=delta_arr,
             done=done_mask, valid=valid_mask)
    logger.info("  %s: done — %d/%d, %d valid", chrom, int(done_mask.sum()), n, int(valid_mask.sum()))
    return processed, int(done_mask.sum())


def assemble_global(cand: pd.DataFrame, out_dir: Path, logger, use_fp16=True):
    n = len(cand)
    dtype = np.float16 if use_fp16 else np.float32
    # Use memmap to keep RAM use bounded at ~80 GB.
    orig_path = out_dir.parent / "rnafm_orig.npy"
    delta_path = out_dir.parent / "rnafm_delta.npy"
    valid_path = out_dir.parent / "valid_rnafm.npy"
    orig_arr = np.lib.format.open_memmap(orig_path, mode="w+", dtype=dtype, shape=(n, EMB_DIM))
    delta_arr = np.lib.format.open_memmap(delta_path, mode="w+", dtype=dtype, shape=(n, EMB_DIM))
    valid = np.zeros(n, dtype=bool)

    chrom_col = cand["chrom"].astype(str).values.astype("U16")
    pos_col = cand["pos"].values.astype(np.int64)
    strand_col = cand["strand"].astype(str).values.astype("U1")
    cand_idx_df = pd.DataFrame({"chrom": chrom_col, "pos": pos_col, "strand": strand_col,
                                 "row": np.arange(n, dtype=np.int64)}
                               ).set_index(["chrom", "pos", "strand"])

    missing = 0
    for chrom in sorted(cand["chrom"].unique()):
        npz_path = out_dir / f"rnafm_{chrom}.npz"
        if not npz_path.exists():
            logger.warning("  missing chrom npz: %s", npz_path)
            missing += 1
            continue
        z = np.load(npz_path, allow_pickle=False)
        pos = z["pos"]
        strand = np.array([s.decode() if isinstance(s, bytes) else s for s in z["strand"]], dtype="U1")
        chrom_arr = np.full(pos.shape, chrom, dtype="U16")
        sub_idx = pd.MultiIndex.from_arrays([chrom_arr, pos, strand])
        rows = cand_idx_df.reindex(sub_idx)["row"].values
        mask = ~pd.isna(rows)
        rows = rows[mask].astype(np.int64)
        orig_arr[rows] = z["orig"][mask]
        delta_arr[rows] = z["delta"][mask]
        valid[rows] = z["valid"][mask]
    orig_arr.flush(); delta_arr.flush()
    np.save(valid_path, valid)
    del orig_arr, delta_arr
    logger.info("Assembled %s / %s (missing %d chroms)", orig_path, delta_path, missing)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", required=True, type=Path)
    ap.add_argument("--hg19", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path, help="Per-chrom npz dir")
    ap.add_argument("--progress-log", type=Path, default=None)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--chrom", type=str, default="")
    ap.add_argument("--skip-assemble", action="store_true")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    _setup_logging(args.out / "run.log", args.progress_log)
    logger = logging.getLogger(__name__)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    logger.info("=" * 60)
    logger.info("compute_rnafm.py starting — device=%s, fp16=%s, batch=%d",
                device, args.fp16, args.batch)

    cand = pd.read_parquet(args.candidates)
    cand = cand.sort_values(["chrom", "pos", "strand"]).reset_index(drop=True)
    n_total = len(cand)
    logger.info("  %d total candidates", n_total)
    if args.limit > 0:
        cand = cand.iloc[:args.limit].reset_index(drop=True)
        logger.info("  SMOKE: limited to %d rows", len(cand))

    logger.info("Loading hg19 genome ...")
    genome = Fasta(str(args.hg19))
    model, batch_converter = load_rnafm(device)

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
        proc, done = process_chrom(chrom, sub, genome, model, batch_converter, device,
                                    args.out, args.batch, args.fp16, logger,
                                    total_done_so_far=total_done,
                                    total_n=total_n, t0_global=t0_global)
        total_done += done

    logger.info("All chroms processed. Total: %.1f min", (time.time() - t0_global) / 60)
    if not args.skip_assemble and not args.limit:
        logger.info("Assembling global arrays ...")
        assemble_global(cand, args.out, logger, use_fp16=args.fp16)

    flag = Path.home() / "phase_2a_done.flag"
    try:
        flag.write_text(f"rnafm done at {time.strftime('%Y-%m-%dT%H:%M:%S')} — "
                        f"{total_n} rows, {(time.time()-t0_global)/60:.1f} min\n")
    except Exception:
        pass


if __name__ == "__main__":
    main()
