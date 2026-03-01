# Genome Build Discovery in APOBEC3A C-to-U Editing Datasets

## Summary

During the development of our EditRNA-A3A model for predicting APOBEC3A-mediated C-to-U RNA editing, we discovered that the genomic coordinates in the C2TFinalSites.DB.xlsx database (636 editing sites from Table T1) are in the GRCh38 (hg38) assembly, not GRCh37 (hg19). This is not explicitly stated in the Excel file.

## How We Discovered This

We initially used the coordinates from T1 with an hg19 reference genome for all downstream analyses (sequence extraction, structure prediction, motif analysis). During a systematic validation of TC-motif frequencies, we noticed that only ~22% of Levanon sites had a cytosine at the expected editing position when looking up the hg19 reference — the base distribution was essentially uniform (C: 22%, A: 27%, G: 22%, T: 27%), suggesting the coordinates were pointing to effectively random positions in hg19.

We confirmed the genome build by cross-referencing known gene boundaries. For example, GLUD2 on chrX at position 121,048,265 falls within the hg38 gene range (121,044,805–121,076,327) but not the hg19 range (118,882,434–118,914,956). A systematic check of 30 sites showed 100% have C at the editing position in hg38, compared to only 27% in hg19.

## Scope of the Issue

Further investigation revealed that 3 out of 5 datasets in our unified catalog use hg38 coordinates:

| Dataset | Native Build | Sites | C at editing position (hg19) | C at editing position (hg38) |
|---------|-------------|-------|------------------------------|------------------------------|
| Levanon (C2TFinalSites.DB) | **hg38** | 636 | 22% | 100% |
| Alqassim et al. 2021 | **hg38** | 209 | 23% | 100% |
| Asaoka et al. 2019 | **hg38** | 5,208 | 23% | 100% |
| Sharma et al. 2015 | hg19 | 278 | 100% | — |
| Baysal et al. 2016 | hg38 (already lifted) | 4,370 | 100% | 100% |

The Baysal 2016 dataset was already correctly handled because its parsing script explicitly performs LiftOver from hg38 to hg19. The Sharma 2015 dataset is natively in hg19.

## Impact on Previous Analyses

With the incorrect coordinates, all sequence-based analyses for the affected datasets (Levanon, Alqassim, Asaoka) were extracting sequences from wrong genomic positions. This affected:

1. **RNA-FM embeddings** — computed from wrong sequences
2. **Secondary structure predictions** — ViennaRNA folding of wrong sequences
3. **TC-motif analysis** — incorrectly reported only 7.3% TC-motif for Levanon sites (the true value with correct coordinates is ~38%)
4. **Cross-dataset overlap analysis** — failed to detect the substantial overlap between datasets (0 overlaps with wrong coordinates vs. 4,570 with correct hg38 coordinates)

## Resolution

We standardized the entire pipeline to GRCh38 (hg38), using native coordinates for the three hg38 datasets and applying LiftOver (hg19 → hg38) only for Sharma 2015. After this correction:

- All 636 Levanon sites correctly show cytosine at the editing position (99.7% after accounting for 2 unmappable sites)
- Cross-dataset overlaps now match biological expectations (e.g., 4,116 overlapping sites between Asaoka and Baysal)
- TC-motif frequency for the Levanon dataset is ~38%, compared to ~90% for Sharma and ~98% for Baysal

## Recommendation

We suggest adding a clear statement of genome build (GRCh38/hg38) to the C2TFinalSites.DB.xlsx metadata or documentation to prevent similar issues for other users of this resource.

---

*Prepared February 22, 2026*
*EditRNA-A3A project, Quris*
