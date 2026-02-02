from Bio.PDB import PDBParser
import pandas as pd
import numpy as np


# input data
parser = PDBParser(QUIET=True)
structure = parser.get_structure("X", "260126_nsun2_tRNA_fitted_on640.pdb")
nuxl_table = pd.read_csv("Peak1_NSUN2_tRNA.tsv", sep="\t")
NA_sequence = "UCCUCGUUAGUAUAGUGGUUAGUAUCCCCGCCUGUCACGCGGGAGACXGGGGUUCAAUUCCCCGACGGGGAGCCA"
nuxl_score_threshold = 60
distance_threshold = 15.0

#define chains
model = next(structure.get_models())
chainA = model["A"]
modified_map = {
    "PSU": "U",  # pseudouridine
    "H2U": "U",
    "1MA": "A",
    "7MG": "G",
    "M2G": "G",
    "OMG": "G",
    "OMC": "C",
    "5MC": "C"
    # add others as needed
}

#functions

def keep_chars(s, allowed):
    allowed_set = set(allowed)
    return "".join(c for c in str(s) if c in allowed_set)

#cleanup



allowed = ["A", "C", "G", "U"]

nuxl_table = nuxl_table[nuxl_table["NuXLScore_score"]>nuxl_score_threshold]

nuxl_table["residue_from"] = nuxl_table["NuXL:best_localization_position"] + nuxl_table["start"]

nuxl_table["nucleic_sequence_clean"] = nuxl_table["NuXL:NA"].apply(keep_chars, allowed=allowed)

def closest_motif_chainB_resnum(structure, residue_from, motif, modified_map,
                               chain_protein="A", chain_na="B",
                               anchor_atom_name="CA"):
    """
    Returns (closest_chainB_resnum, best_dist).
    - closest_chainB_resnum is the PDB resseq integer in chain B for the residue that contains the closest atom.
    - best_dist is the shortest distance in Å.
    """

    # --- try to get residue_from in protein chain ---
    try:
        # fast path: exact key with blank insertion code
        res_a = structure[0][chain_protein][(" ", int(residue_from), " ")]
    except Exception:
        # fallback: search by resseq ignoring insertion code
        res_a = None
        try:
            resnum = int(residue_from)
        except Exception:
            print(f"residue number not in structure: {residue_from}")
            return np.nan, np.nan, np.nan

        for r in structure[0][chain_protein]:
            if r.get_id()[0] == " " and r.get_id()[1] == resnum:
                res_a = r
                break

        if res_a is None:
            print(f"residue number not in structure: {residue_from}")
            return np.nan, np.nan, np.nan

    # choose anchor atom on residue A
    if anchor_atom_name in res_a:
        atom_a = res_a[anchor_atom_name]
    else:
        atoms = list(res_a.get_atoms())
        if not atoms:
            return np.nan, np.nan, np.nan
        atom_a = atoms[0]

    # guard motif
    if motif is None:
        return np.nan, np.nan
    motif = str(motif).strip()
    if motif == "":
        return np.nan, np.nan, np.nan

    target_len = len(motif)

    # --- build NA residue list + sequence (your working logic) ---
    na_residues = []
    na_sequence = []

    for residue in structure[0][chain_na]:
        if residue.get_id()[0] != " ":
            continue

        resname = residue.get_resname().strip()
        if resname in ("A", "U", "G", "C"):
            base = resname
        elif resname in ("DA", "DT", "DG", "DC"):
            base = resname[1]
        elif resname in modified_map:
            base = modified_map[resname]
        else:
            continue

        na_residues.append(residue)
        na_sequence.append(base)

    na_sequence = "".join(na_sequence)

    if target_len > len(na_sequence):
        return np.nan, np.nan, np.nan

    best_dist = float("inf")
    best_residue_b = None
    best_atom_b = None

    # --- scan windows (your working logic) ---
    for i in range(len(na_sequence) - target_len + 1):
        window_seq = na_sequence[i : i + target_len]
        if window_seq != motif:
            continue

        window_residues = na_residues[i : i + target_len]

        for residue in window_residues:
            for atom in residue:
                name = atom.get_name()
                # if not name.startswith("C"):
                #     continue
                # if "'" in name:
                #     continue

                d = atom_a - atom
                if d < best_dist:
                    best_dist = d
                    best_residue_b = residue
                    best_atom_b = atom

    if best_residue_b is None or best_dist == float("inf"):
        return np.nan, np.nan, np.nan

    # chain B residue number (resseq)
    closest_chainB_resnum = best_residue_b.get_id()[1]
    closest_atom_b = best_atom_b
    return closest_chainB_resnum, best_dist, best_atom_b


# ---- Apply to your dataframe ----
# assumes you already have:
#   - structure loaded (Bio.PDB)
#   - modified_map defined
#   - nuxl_table has 'residue_from' and 'nucleic_sequence_clean'


def apply_row(row):
    return pd.Series(
        closest_motif_chainB_resnum(
            structure=structure,
            residue_from=row["residue_from"],
            motif=row["nucleic_sequence_clean"],
            modified_map=modified_map,
            chain_protein="A",
            chain_na="B",
            anchor_atom_name="CA",
        ),
        index=["closest_chainB_resnum", "closest_chainB_dist", "NA_atom_to"]
    )

nuxl_table[["closest_chainB_resnum", "closest_chainB_dist", "NA_atom_to"]] = nuxl_table.apply(apply_row, axis=1)

# create pseudobond file

import pandas as pd
import numpy as np

def write_chimerax_pseudobonds_conditional_color(
    nuxl_table: pd.DataFrame,
    out_path: str,
    threshold: float,
    value_col: str = "closest_chainB_dist",   # column used for thresholding
    chain_from: str = "A",
    chain_to: str = "B",
    res_from_col: str = "residue_from",       # RES_FROM
    res_to_col: str = "closest_chainB_resnum",# CLOSEST_CHAINB_RESNUM
    atom_to_col: str = "NA_atom_to",          # NA_ATOM_TO
    atom_from: str = "CA",
    radius: float = 0.4,
    dashes: int = 0,
    color_low: str = "blue",
    color_high: str = "red",
):
    """
    Writes a ChimeraX pseudobond file:

      ; radius = 0.4
      ; dashes = 0
      /A:RES_FROM@CA /B:CLOSEST_CHAINB_RESNUM@NA_ATOM_TO COLOR

    COLOR is chosen per row:
      - color_low if value_col < threshold
      - color_high if value_col > threshold
      - rows with missing data are skipped
      - if value_col == threshold, uses color_low (easy to change)
    """
    with open(out_path, "w") as f:
        f.write(f"; radius = {radius}\n")
        f.write(f"; dashes = {dashes}\n")

        for _, row in nuxl_table.iterrows():
            res_from = row.get(res_from_col)
            res_to = row.get(res_to_col)
            atom_to = row.get(atom_to_col)
            val = row.get(value_col)

            # skip incomplete rows
            if pd.isna(res_from) or pd.isna(res_to) or atom_to is None or pd.isna(val):
                continue

            # normalize residue numbers
            try:
                res_from = int(res_from)
                res_to = int(res_to)
            except Exception:
                continue

            # normalize atom_to: Bio.PDB Atom object -> name, else string
            if hasattr(atom_to, "get_name"):
                atom_to_name = atom_to.get_name()
            else:
                atom_to_name = str(atom_to).strip()

            # pick color
            color = color_low if float(val) <= threshold else color_high

            # write line
            f.write(
                f"/{chain_from}:{res_from}@{atom_from} "
                f"/{chain_to}:{res_to}@{atom_to_name} "
                f"{color}\n"
            )

    return out_path

pb_file = write_chimerax_pseudobonds_conditional_color(
    nuxl_table=nuxl_table,
    out_path="nuxl_links.pb",
    threshold=distance_threshold,                  # Å, for example
    value_col="closest_chainB_dist", # threshold on distance
)
print("Wrote:", pb_file)
