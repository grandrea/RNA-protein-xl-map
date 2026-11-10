# NSUN2-tRNA NuXL figure data

This repository contains the data and scripts used for the paper figures based on the experimental NSUN2-tRNA structure.

## Inputs

- `20260529_Model_pXL-EM_AC_EL 1.pdb`: experimental protein-tRNA structure.
- `Peak2_NSUN2_tRNA.tsv`: NuXL result table used for residue localization and scoring.

## ECDF figure

`calculate_plot_nuxl_trna_ecdf.py` calculates the minimum distance from any atom of each protein residue to any tRNA nucleobase heavy atom. Sugar, phosphate, and hydrogen atoms are excluded from the tRNA atom set. It writes the ECDF as PNG and PDF, the plotted long-format CSV, and the KS-test results to `experimental_structure_analysis/`.

```powershell
& 'C:\Users\andrea.graziadei\software\anaconda3\envs\bio\python.exe' .\calculate_plot_nuxl_trna_ecdf.py
```

The Excel figure-data table documents the plotted series and column definitions.

## ChimeraX confidence views

`generate_xlinked_ca_chimerax.py` maps localized NuXL sites to protein chain A and colors them by score. The deposited command files are:

- `nuxl_xlinked_residues_score60.cxc`
- `nuxl_xlinked_residues_score80.cxc`

```powershell
& 'C:\Users\andrea.graziadei\software\anaconda3\envs\bio\python.exe' .\generate_xlinked_ca_chimerax.py --score-threshold 60 --out nuxl_xlinked_residues_score60.cxc --selection-name nuxl_xlinked_protein_residues_score60
& 'C:\Users\andrea.graziadei\software\anaconda3\envs\bio\python.exe' .\generate_xlinked_ca_chimerax.py --score-threshold 80 --out nuxl_xlinked_residues_score80.cxc --selection-name nuxl_xlinked_protein_residues_score80
```

For each threshold, the lower score boundary is split at the midpoint to 85; scores at or above 85 are high confidence.

