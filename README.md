# NSUN2-tRNA NuXL figure data

This repository contains the data and scripts used for the paper figures based on the experimental NSUN2-tRNA structure.

## Inputs

- `20260529_Model_pXL-EM_AC_EL 1.pdb`: experimental protein-tRNA structure.
- `Peak2_NSUN2_tRNA.tsv`: NuXL result table used for residue localization and scoring.

## ECDF figure

`calculate_plot_nuxl_trna_ecdf.py` calculates the minimum distance from any atom of each protein residue to any tRNA nucleobase heavy atom. Sugar, phosphate, and hydrogen atoms are excluded from the tRNA atom set. It writes the ECDF as PNG and PDF, the plotted long-format CSV, and the KS-test results to `experimental_structure_analysis/`.

