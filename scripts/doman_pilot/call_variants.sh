#!/bin/bash
# Call somatic variants: treated clone vs nCas9 normal.
# Usage: call_variants.sh <TREATED_LABEL> <NORMAL_LABEL>
# Example: call_variants.sh BE4_clone1 nCas9_clone3
#
# Outputs per-clone novel C>T VCF in /mnt/data/vcf/<TREATED_LABEL>/
# Time: ~2-4 hr per pair on 16 cores.

set -euo pipefail
TREATED=$1
NORMAL=$2
DATA=/mnt/data
THREADS=16

source $DATA/tools/miniconda/etc/profile.d/conda.sh
conda activate doman

mkdir -p $DATA/vcf/$TREATED
cd $DATA/vcf/$TREATED
LOG=$DATA/logs/${TREATED}_vs_${NORMAL}.log

echo "[$(date)] === Mutect2: $TREATED vs $NORMAL ===" | tee -a $LOG

T_BAM=$DATA/bam/$TREATED/${TREATED}.dedup.bam
N_BAM=$DATA/bam/$NORMAL/${NORMAL}.dedup.bam

# Mutect2 in tumor-vs-normal mode (treats nCas9 clone as the matched normal)
RAW_VCF=$DATA/vcf/$TREATED/${TREATED}.raw.vcf.gz
if [ ! -f $RAW_VCF ]; then
  echo "[$(date)] Running Mutect2..." | tee -a $LOG
  gatk Mutect2 \
    -R $DATA/ref/hg19.fa \
    -I $T_BAM -tumor $TREATED \
    -I $N_BAM -normal $NORMAL \
    --native-pair-hmm-threads $THREADS \
    -O $RAW_VCF 2>&1 | tail -20 | tee -a $LOG
fi

# Filter Mutect calls
FILTERED_VCF=$DATA/vcf/$TREATED/${TREATED}.filtered.vcf.gz
if [ ! -f $FILTERED_VCF ]; then
  echo "[$(date)] FilterMutectCalls..." | tee -a $LOG
  gatk FilterMutectCalls \
    -R $DATA/ref/hg19.fa \
    -V $RAW_VCF \
    -O $FILTERED_VCF 2>&1 | tail -10 | tee -a $LOG
fi

# Filter to PASS C>T (and G>A for opposite strand) only
NOVEL_CT=$DATA/vcf/$TREATED/${TREATED}.novel_ct.vcf.gz
echo "[$(date)] Filtering to PASS C>T variants..." | tee -a $LOG
bcftools view -f PASS $FILTERED_VCF | \
  bcftools view -e 'GT[0]=="ref"' | \
  awk 'BEGIN{OFS="\t"} /^#/ {print; next} \
       ($4=="C" && $5=="T") || ($4=="G" && $5=="A")' | \
  bcftools view -Oz -o $NOVEL_CT
tabix -p vcf $NOVEL_CT

N_VARIANTS=$(bcftools view -H $NOVEL_CT | wc -l)
echo "[$(date)] $TREATED novel C>T variants: $N_VARIANTS" | tee -a $LOG

# Output simple BED for downstream scoring
BED=$DATA/vcf/$TREATED/${TREATED}.novel_ct.bed
bcftools view -H $NOVEL_CT | awk 'BEGIN{OFS="\t"} {print $1, $2-1, $2, $4">"$5, ".", "+"}' > $BED
echo "[$(date)] BED rows: $(wc -l < $BED)" | tee -a $LOG
echo "[$(date)] === DONE $TREATED ===" | tee -a $LOG
