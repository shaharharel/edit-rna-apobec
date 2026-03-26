# Clinical & Interpretability Iteration Plan

## Iteration 1 (Current — agents running)
- Clinical: ClinVar deep dive, gene-level analysis, tissue-disease connections
- Interpretability: Rescued sites analysis, embedding comparison, feature attribution

## Iteration 2 (After first results)
- Clinical: Follow up on top genes — what pathways? KEGG/GO enrichment?
- Clinical: Are editing sites at protein domains? Active sites? Splice junctions?
- Clinical: Do high-score ClinVar variants cluster in specific disease categories (ICD-10)?
- Interpretability: Analyze the specific rescued cases — what biological pattern?
- Interpretability: Can we find a "shared structural motif" learned by the backbone?

## Iteration 3 (Deep dive on interesting findings)
- Clinical: Pick the 3-5 most interesting gene/disease findings and write detailed case studies
- Clinical: Compare our predictions to known APOBEC mutation signatures in cancer (COSMIC)
- Interpretability: Visualize attention/gradient maps on specific rescued cases
- Interpretability: Show the decision boundary shift between unified and per-enzyme

## Iteration 4 (Polish & narrative)
- Clinical: Write the final publication-quality clinical sections
- Interpretability: Create the "showcase" figure with the best examples
- Both: Review by biology agent for accuracy

## Key Questions to Answer
1. Are APOBEC editing sites over-represented at protein functional domains?
2. Do high-confidence editing predictions overlap with known somatic mutation hotspots?
3. What specific genes show the strongest editing-pathogenicity connection?
4. Why does unified training specifically help A3G and Neither? What do they share?
5. What makes A3B lose from unified training? What's unique about A3B?
