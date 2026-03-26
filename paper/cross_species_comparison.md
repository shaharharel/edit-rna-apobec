# Cross-Species Comparison: Human vs Chimpanzee APOBEC RNA Editing Sites

## Executive Summary

We compared 3,640 APOBEC RNA editing sites (true orthologs) between human and chimpanzee (panTro6) to test whether the "editable transcriptome" is under evolutionary constraint. Key findings:

1. **Editing sites are in significantly more conserved regions** than matched controls (0.81% vs 1.07% divergence, p=5.94e-37)
2. **The edited cytidine is almost universally conserved** (99.3% vs 98.7% for controls, Fisher OR=1.79, p=0.017)
3. **APOBEC motifs are preserved** at 98.4% of editing sites (vs 97.9% controls)
4. **97.4% of editing sites have identical motif features** in human and chimp — the editable transcriptome is nearly static between species
5. **TC and CC motif densities are unchanged** between species at editing site windows (no evidence of motif gain/loss)

## Methods

### Data
- **Editing sites**: 7,564 C-to-U editing positions from multi-enzyme dataset v3 (A3A, A3B, A3G, A3A_A3G, Neither, Unknown)
- **Controls**: 7,788 motif-matched negative sites from the same dataset
- **Genome builds**: hg19 (6,428 sites) and hg38 (1,136 sites) → panTro6 (chimp)

### Pipeline
1. LiftOver coordinates to panTro6 using pyliftover (hg19→panTro6 or hg38→panTro6)
2. Extract 201-nt windows (±100 bp) from both human and chimp genomes
3. **Ortholog quality filter**: Exclude sites with >5% sequence divergence in the 201-nt window (removes cancer cell line coordinate artifacts and paralogous mappings)
4. Compare: center base conservation, dinucleotide motif preservation, substitution rates, motif density changes

### Ortholog Filter
Many sites (especially Kockler 2026 cancer cell line coordinates) produced liftOver results to non-orthologous chimp regions. We applied a strict filter: only sites with <5% divergence in the 201-nt window are retained.

| Category | Before filter | After filter | Retained |
|----------|:---:|:---:|:---:|
| Editing sites | 7,534 | 3,640 | 48.3% |
| Controls | 7,729 | 3,340 | 43.2% |

The ~48% retention rate reflects that ~half the dataset uses cancer cell line transcriptomic coordinates (Kockler 2026) that don't map reliably to reference genomes.

## Results

### 1. Editing Sites Are in Conserved Regions

| Metric | Editing sites | Controls | p-value |
|--------|:---:|:---:|:---:|
| **Substitution rate** | **0.81% ± 0.84%** | **1.07% ± 0.98%** | **5.94e-37** |
| Center C conserved | 99.3% (3,613/3,640) | 98.7% (3,296/3,340) | 0.017 |
| Dinucleotide motif preserved | 98.4% (3,581/3,640) | 97.9% (3,270/3,340) | — |
| Identical motif features | 97.4% | — | — |

**The substitution rate at editing sites (0.81%) is 24% lower than at matched controls (1.07%).**
This is highly significant (Mann-Whitney p=5.94e-37) and indicates that APOBEC editing sites reside in functionally constrained genomic regions that are under stronger purifying selection.

### 2. Per-Enzyme Conservation

| Enzyme | n orthologs | C conserved | Motif preserved | Sub rate |
|--------|:---:|:---:|:---:|:---:|
| A3A | 1,259 | 99.5% | 98.7% | 0.76% |
| A3B | 2,121 | 99.2% | 98.3% | 0.83% |
| A3G | 75 | 98.7% | 96.0% | 0.78% |
| A3A_A3G | 59 | 96.6% | 96.6% | 0.67% |
| Neither | 99 | 100.0% | 97.0% | 0.99% |
| Unknown | 27 | 100.0% | 100.0% | 1.23% |

All enzymes show >96% center C conservation. "Neither" (likely APOBEC1) and "Unknown" sites show 100% conservation of the target cytidine.

A3A_A3G ("Both") sites have the lowest substitution rate (0.67%), suggesting these dual-enzyme targets are in the most conserved regions.

### 3. Non-Conserved Sites (27/3,640 = 0.7%)

Only 27 editing sites have a non-C base at the orthologous chimp position:
- A3A: 6 sites (0.5%)
- A3B: 18 sites (0.8%)
- A3A_A3G: 2 sites (3.4%)
- A3G: 1 site (1.3%)

These represent recent C→non-C substitutions in the chimp lineage at established editing sites, or (less likely) recent C gains in the human lineage that created new editing substrates.

### 4. Motif Density Is Unchanged Between Species

| Motif | Human (editing) | Chimp (editing) | Human (control) | Chimp (control) |
|-------|:---:|:---:|:---:|:---:|
| TC per kb | 61.4 | 61.3 | 64.2 | 64.2 |
| CC per kb | 87.5 | 87.3 | 86.1 | 86.1 |

TC and CC motif densities in the ±100bp window are virtually identical between human and chimp (delta < 0.2/kb). There is no evidence that APOBEC-prone motifs are being gained or lost at editing sites relative to the rest of the genome.

### 5. The Editable Transcriptome Is Static

97.4% of editing sites have identical 24-dimensional motif feature vectors in human and chimp. This means:
- The trinucleotide context (±2 positions) is perfectly conserved
- The dinucleotide context (5' and 3') is preserved
- If the chimp transcript adopts similar RNA secondary structure (which we expect given the low divergence), **virtually all human APOBEC editing sites would also be targetable in chimpanzee**

## Interpretation

### What These Results Mean

1. **APOBEC editing sites are under evolutionary constraint**: The 24% lower divergence rate at editing sites vs controls (p=5.94e-37) demonstrates that these regions are under stronger purifying selection. This is consistent with editing sites residing in functionally important RNA structures.

2. **The editable transcriptome is conserved between human and chimp**: Nearly all editing sites (99.3%) retain the target C, and 98.4% retain the upstream motif. This suggests that the repertoire of APOBEC-targetable cytidines is largely shared between human and chimp. Any differences in actual editing would come from:
   - Enzyme expression differences (e.g., tissue-specific A3A levels)
   - RNA secondary structure changes (not measurable from sequence alone)
   - Regulatory differences (editing rate modulation)

3. **Editing sites are NOT enriched for APOBEC-driven divergence**: If APOBEC mutagenesis were driving sequence evolution at editing sites, we would expect HIGHER divergence and motif turnover. Instead, we see the opposite — lower divergence. This argues against the hypothesis that APOBEC editing contributes significantly to germline mutation at these sites.

### Connection to ClinVar Findings

This result strengthens the "structural vulnerability" hypothesis for ClinVar pathogenic enrichment (OR=1.33):
- Editing sites are in conserved, functionally constrained regions
- These same regions are enriched for pathogenic variants
- The shared driver is RNA structural accessibility + functional importance
- APOBEC editing and disease-causing mutations target the same positions because both require structurally accessible cytidines in important regulatory contexts

## Output Files

- `experiments/multi_enzyme/outputs/cross_species/cross_species_results.json` — full analysis results
- `experiments/multi_enzyme/outputs/cross_species/editing_sites_cross_species.csv` — per-site conservation data
- `experiments/multi_enzyme/outputs/cross_species/conservation_overview.png` — 3-panel conservation figure
- `experiments/multi_enzyme/outputs/cross_species/per_enzyme_conservation.png` — per-enzyme breakdown
- `experiments/multi_enzyme/outputs/cross_species/motif_density_comparison.png` — TC/CC density human vs chimp
- `experiments/multi_enzyme/outputs/cross_species/trinucleotide_conservation.png` — trinucleotide transition heatmap
- `scripts/multi_enzyme/cross_species_comparison.py` — analysis script
