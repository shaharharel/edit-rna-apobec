#!/bin/bash
# Process one Doman clone end-to-end: download → align → keep BAM.
# Usage: run_one_clone.sh <SRR_ACCESSION> <SAMPLE_LABEL>
# Example: run_one_clone.sh SRR10413104 BE4_clone1
#
# Per-clone disk usage: ~30 GB FASTQ (deleted after align) + ~80-120 GB BAM (kept).
# Time: ~30 min download + ~30-45 min align = ~60-75 min wall clock on 16 cores.

set -euo pipefail
SRR=$1
LABEL=$2
DATA=/mnt/data
THREADS=16

source $DATA/tools/miniconda/etc/profile.d/conda.sh
conda activate doman

mkdir -p $DATA/{fastq,bam,logs}/$LABEL
cd $DATA/fastq/$LABEL

LOG=$DATA/logs/${LABEL}_pipeline.log
echo "[$(date)] === Starting $LABEL ($SRR) ===" | tee -a $LOG

# 1. Download FASTQ
if [ ! -f ${SRR}_1.fastq ]; then
  echo "[$(date)] Downloading $SRR..." | tee -a $LOG
  prefetch --max-size 100g $SRR 2>&1 | tail -5 | tee -a $LOG
  fasterq-dump --threads $THREADS --split-files $SRR 2>&1 | tail -5 | tee -a $LOG
  rm -rf $SRR
fi
ls -lh ${SRR}_1.fastq ${SRR}_2.fastq | tee -a $LOG

# 2. Align with BWA-MEM2
BAM=$DATA/bam/$LABEL/${LABEL}.sorted.bam
if [ ! -f $BAM ]; then
  echo "[$(date)] Aligning $LABEL with BWA-MEM2 ($THREADS threads)..." | tee -a $LOG
  RG="@RG\tID:${LABEL}\tSM:${LABEL}\tLB:${LABEL}\tPL:ILLUMINA"
  bwa mem -t $THREADS -R "$RG" $DATA/ref/hg19.fa \
    ${SRR}_1.fastq ${SRR}_2.fastq 2> >(tail -3 >> $LOG) | \
    samtools sort -@ $THREADS -o $BAM - 2>&1 | tee -a $LOG
  samtools index -@ $THREADS $BAM
  echo "[$(date)] Aligned: $(ls -lh $BAM)" | tee -a $LOG
fi

# 3. Mark duplicates
DEDUP_BAM=$DATA/bam/$LABEL/${LABEL}.dedup.bam
if [ ! -f $DEDUP_BAM ]; then
  echo "[$(date)] MarkDuplicates..." | tee -a $LOG
  gatk MarkDuplicates -I $BAM -O $DEDUP_BAM \
    -M $DATA/bam/$LABEL/${LABEL}.metrics.txt 2>&1 | tail -10 | tee -a $LOG
  samtools index -@ $THREADS $DEDUP_BAM
fi

# 4. Free FASTQ disk
echo "[$(date)] Removing FASTQ to free disk..." | tee -a $LOG
rm -f $DATA/fastq/$LABEL/${SRR}_1.fastq $DATA/fastq/$LABEL/${SRR}_2.fastq

echo "[$(date)] === $LABEL DONE: $(ls -lh $DEDUP_BAM | awk '{print $5}') ===" | tee -a $LOG
df -h $DATA | tee -a $LOG
