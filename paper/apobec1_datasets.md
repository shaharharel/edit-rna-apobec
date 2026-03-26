# APOBEC1 External Datasets — Available for Integration

## Top Priority Datasets

### 1. Mouse Genome-Wide Catalog (Davidson 2014, Genome Biology)
- **GEO GSE57910** — publicly available
- 56 intestinal + 22 liver editing sites from WT vs Apobec1-/- mice
- RNA-seq with 76.7M reads
- Definitive: sites disappear in knockout → confirmed APOBEC1 targets

### 2. Human Transcriptome-Wide Screen (Rosenberg 2011, Nat Struct Mol Biol)
- **GEO GSE24958** — publicly available
- 32 human APOBEC1 targets in 3'UTRs
- Small intestine specific
- All in AU-rich 3'UTR regions

### 3. 177 Murine Editing Events (Blanc 2021, RNA Journal)
- 177 events across 119 transcripts, 103 Sanger-confirmed
- Prediction model explains 84% of editing efficiency variance
- Features: cis-sequence, structure, cofactor dependencies (A1CF, RBM47)

### 4. Multi-Cell-Type Profiling (2017, PNAS)
- **GEO GSE58798** — publicly available
- Cell-type-specific editing frequencies

## Key APOBEC1 Biology (for validation)

- **Mooring sequence**: WCWN₂₋₄WRAUYANUAU, 4nt downstream of edited C
- **3'UTR preference**: nearly all APOBEC1 sites are in 3'UTRs
- **AU-rich context**: flanking A/U on both sides (ACA/UCA)
- **Cofactors**: A1CF and RBM47 required for editing
- **Tissues**: intestine (primary), liver, kidney

## Integration Plan
1. Download GSE57910 (mouse) + Rosenberg human sites
2. Map mouse sites to human orthologs
3. Check overlap with our 206 "Neither" sites
4. Extract features, train APOBEC1-specific model
5. Validate: mooring sequence enrichment, 3'UTR bias, intestine specificity
