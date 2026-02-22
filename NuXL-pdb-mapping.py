from Bio.PDB import PDBParser
import pandas as pd
import numpy as np


# input data
parser = PDBParser(QUIET=True)
structure = parser.get_structure("X", "260126_nsun2_tRNA_fitted_on640.pdb")
nuxl_table = pd.read_csv("Peak2_NSUN2_tRNA.tsv", sep="\t")
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

def wfy_indices(sequence):
    """
    Return list of indices where sequence contains W, F, or Y.
    """
    if sequence is None:
        return []
    return [i for i, aa in enumerate(str(sequence)) if aa in {"W", "F", "Y"}]

def compute_residue_from_list(row, localization_score_threshold):
    score = row["NuXL:best_localization_score"]
    residue_from = row["residue_from"]
    seq = row["sequence"]
    best_pos = row["NuXL:best_localization_position"]

    # Case 1: confident localization → keep single residue
    if pd.notna(score) and score > localization_score_threshold:
        return residue_from

    # Case 2: ambiguous localization → expand to W/F/Y positions
    indices = wfy_indices(seq)

    if indices:
        # return list of all possible residue_from values
        return [best_pos + i for i in indices]

    # Case 3: no W/F/Y → fallback to original residue
    return residue_from

def map_score_to_radius(score, score_min, score_max, radius_min, radius_max):
    if score_max == score_min:
        return radius_min  # avoid division by zero

    # clamp score
    score = max(score_min, min(score, score_max))

    frac = (score - score_min) / (score_max - score_min)
    return radius_min + frac * (radius_max - radius_min)


#cleanup



allowed = ["A", "C", "G", "U"]

nuxl_table = nuxl_table[nuxl_table["NuXLScore_score"]>nuxl_score_threshold]

nuxl_table["residue_from"] = nuxl_table["NuXL:best_localization_position"] + nuxl_table["start"]

nuxl_table["nucleic_sequence_clean"] = nuxl_table["NuXL:NA"].apply(keep_chars, allowed=allowed)

localization_score_threshold = 0.35  # example

nuxl_table["residue_from_list"] = nuxl_table.apply(
    compute_residue_from_list,
    axis=1,
    localization_score_threshold=localization_score_threshold
)


def closest_motif_chainB_resnum_from_list(
    structure,
    residue_from_list,
    motif,
    modified_map,
    chain_protein="A",
    chain_na="B",
    anchor_atom_name="CA",
):
    """
    residue_from_list: int or list of ints (candidate protein residue numbers)
    motif: string (e.g., 'CC', arbitrarily long)

    Returns:
      (closest_chainB_resnum, best_dist, best_atom_b, residue_from_pdb)

    - closest_chainB_resnum: PDB resseq integer in chain B for the residue that contains the closest atom_to
    - best_dist: shortest distance in Å (global min over all candidates)
    - best_atom_b: Bio.PDB Atom object (atom_to) in chain B that gives the minimum distance
    - residue_from_pdb: the protein residue number (from residue_from_list) that gave the minimum distance
    """

    # normalize residue_from_list to a list
    if residue_from_list is None or (isinstance(residue_from_list, float) and np.isnan(residue_from_list)):
        return np.nan, np.nan, np.nan, np.nan

    if isinstance(residue_from_list, (int, np.integer)):
        candidates = [int(residue_from_list)]
    elif isinstance(residue_from_list, list):
        candidates = []
        for x in residue_from_list:
            try:
                candidates.append(int(x))
            except Exception:
                continue
        if not candidates:
            return np.nan, np.nan, np.nan, np.nan
    else:
        # fallback if something like a string "['1','2']" slipped in
        try:
            candidates = [int(residue_from_list)]
        except Exception:
            return np.nan, np.nan, np.nan, np.nan

    # guard motif
    if motif is None:
        return np.nan, np.nan, np.nan, np.nan
    motif = str(motif).strip()
    if motif == "":
        return np.nan, np.nan, np.nan, np.nan

    target_len = len(motif)

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
        return np.nan, np.nan, np.nan, np.nan

    # --- global best over ALL candidates ---
    best_dist = float("inf")
    best_residue_b = None
    best_atom_b = None
    best_residue_from_pdb = None

    # helper: get residue in protein chain by resseq (ignore insertion code if needed)
    def get_residue_by_resseq(chain, resseq):
        # fast path
        key = (" ", resseq, " ")
        if key in chain:
            return chain[key]
        # fallback scan ignoring insertion code
        for r in chain:
            if r.get_id()[0] == " " and r.get_id()[1] == resseq:
                return r
        return None

    prot_chain = structure[0][chain_protein]

    # precompute motif-hit windows indices (small speed win)
    hit_starts = [
        i for i in range(len(na_sequence) - target_len + 1)
        if na_sequence[i:i+target_len] == motif
    ]
    if not hit_starts:
        return np.nan, np.nan, np.nan, np.nan

    for resnum in candidates:
        res_a = get_residue_by_resseq(prot_chain, resnum)
        if res_a is None:
            print(f"residue number not in structure: {resnum}")
            continue

        # choose anchor atom
        if anchor_atom_name in res_a:
            atom_a = res_a[anchor_atom_name]
        else:
            atoms = list(res_a.get_atoms())
            if not atoms:
                continue
            atom_a = atoms[0]

        # scan motif windows
        for i in hit_starts:
            window_residues = na_residues[i : i + target_len]

            # scan all atoms in the matched window (your current behavior)
            for residue in window_residues:
                for atom in residue:
                    if not atom.get_name().startswith("C"):
                        continue
                    if "'" in atom.get_name():
                         continue
                    d = atom_a - atom
                    if d < best_dist:
                        best_dist = d
                        best_residue_b = residue
                        best_atom_b = atom
                        best_residue_from_pdb = resnum

    if best_residue_b is None or best_atom_b is None or best_dist == float("inf"):
        return np.nan, np.nan, np.nan, np.nan

    closest_chainB_resnum = best_residue_b.get_id()[1]
    atom_to_name = best_atom_b.get_name()
    return closest_chainB_resnum, best_dist, atom_to_name, best_residue_from_pdb



# ---- Apply to your dataframe ----
# assumes you already have:
#   - structure loaded (Bio.PDB)
#   - modified_map defined
#   - nuxl_table has 'residue_from' and 'nucleic_sequence_clean'


def apply_row(row):
    return pd.Series(
        closest_motif_chainB_resnum_from_list(
            structure=structure,
            residue_from_list=row["residue_from_list"],
            motif=row["nucleic_sequence_clean"],
            modified_map=modified_map,
            chain_protein="A",
            chain_na="B",
            anchor_atom_name="CA",
        ),
        index=["closest_chainB_resnum", "closest_chainB_dist", "NA_atom_to", "residue_from_pdb"]
    )

nuxl_table[["closest_chainB_resnum", "closest_chainB_dist", "NA_atom_to", "residue_from_pdb"]] = (
    nuxl_table.apply(apply_row, axis=1)
)


# create pseudobond file


def write_chimerax_pseudobonds_conditional_color(
    nuxl_table: pd.DataFrame,
    out_path: str,
    threshold: float,
    value_col: str = "closest_chainB_dist",
    chain_from: str = "A",
    chain_to: str = "B",
    res_from_col: str = "residue_from_pdb",      # <-- FIXED
    res_to_col: str = "closest_chainB_resnum",
    atom_to_col: str = "NA_atom_to",
    atom_from: str = "CA",
    radius: float = 0.4,
    dashes: int = 0,
    color_low: str = "blue",
    color_high: str = "red",
):
    with open(out_path, "w") as f:
        f.write(f"; radius = {radius}\n")
        f.write(f"; dashes = {dashes}\n")

        for _, row in nuxl_table.iterrows():
            res_from = row.get(res_from_col)
            res_to = row.get(res_to_col)
            atom_to = row.get(atom_to_col)
            val = row.get(value_col)

            # treat NaN atom_to as missing too
            atom_to_missing = (atom_to is None) or (isinstance(atom_to, float) and np.isnan(atom_to))

            if pd.isna(res_from) or pd.isna(res_to) or atom_to_missing or pd.isna(val):
                continue

            try:
                res_from = int(res_from)
                res_to = int(res_to)
            except Exception:
                continue

            # FIXED: always define atom_to_name
            if hasattr(atom_to, "get_name"):
                atom_to_name = atom_to.get_name()
            else:
                atom_to_name = str(atom_to).strip()

            if atom_to_name == "" or atom_to_name.lower() == "nan":
                continue

            color = color_low if float(val) <= threshold else color_high

            f.write(
                f"/{chain_from}:{res_from}@{atom_from} "
                f"/{chain_to}:{res_to}@{atom_to_name} "
                f"{color}\n"
            )

    return out_path


def map_score_to_radius(score, score_min, score_max, radius_min, radius_max):
    if score_max == score_min:
        return radius_min
    score = max(score_min, min(score, score_max))
    frac = (score - score_min) / (score_max - score_min)
    return radius_min + frac * (radius_max - radius_min)

def write_chimerax_pseudobonds_sectioned(
    nuxl_table: pd.DataFrame,
    out_path: str,
    threshold: float,
    # for color
    value_col: str = "closest_chainB_dist",
    color_low: str = "blue",
    color_high: str = "red",
    # for radius mapping
    score_col: str = "NuXLScore_score",
    nuxl_score_min: float = 70.0,
    nuxl_score_max: float = 100.0,
    radius_min: float = 0.2,
    radius_max: float = 1.2,
    # atom specs
    chain_from: str = "A",
    chain_to: str = "B",
    res_from_col: str = "residue_from_pdb",
    res_to_col: str = "closest_chainB_resnum",
    atom_from: str = "CA",
    atom_to_col: str = "NA_atom_to",  # should be string like "C6", "N1", ...
    # section directives
    halfbond: bool = False,
    # optional: rounding radius to avoid overly granular sections (also reduces file size)
    radius_round: int = 3,
    verbose: bool = True,
):
    """
    Writes a ChimeraX pseudobond file split into per-bond sections:

      ; halfbond = false
      ; radius = <computed>
      /A:RES_FROM@CA /B:RES_TO@ATOM_TO COLOR

    COLOR: blue if value_col <= threshold else red
    RADIUS: mapped from score_col into [radius_min, radius_max] using [nuxl_score_min, nuxl_score_max]
    """
    written = 0
    skipped = 0

    hb_str = "true" if halfbond else "false"

    with open(out_path, "w") as f:
        for _, row in nuxl_table.iterrows():
            res_from = row.get(res_from_col)
            res_to = row.get(res_to_col)
            atom_to = row.get(atom_to_col)
            dist_val = row.get(value_col)
            score_val = row.get(score_col)

            # treat NaN atom_to as missing too
            atom_to_missing = (atom_to is None) or (isinstance(atom_to, float) and np.isnan(atom_to))

            if (
                pd.isna(res_from)
                or pd.isna(res_to)
                or atom_to_missing
                or pd.isna(dist_val)
                or pd.isna(score_val)
            ):
                skipped += 1
                continue

            try:
                res_from = int(res_from)
                res_to = int(res_to)
            except Exception:
                skipped += 1
                continue

            atom_to_name = str(atom_to).strip()
            if atom_to_name == "" or atom_to_name.lower() == "nan":
                skipped += 1
                continue

            color = color_low if float(dist_val) <= threshold else color_high

            radius = map_score_to_radius(
                float(score_val),
                nuxl_score_min,
                nuxl_score_max,
                radius_min,
                radius_max,
            )
            radius = round(radius, radius_round)

            # section directives for this bond
            f.write(f"; halfbond = {hb_str}\n")
            f.write(f"; radius = {radius}\n")
            f.write(f"; dashes = 0\n")

            # bond line
            f.write(
                f"/{chain_from}:{res_from}@{atom_from} "
                f"/{chain_to}:{res_to}@{atom_to_name} "
                f"{color}\n"
            )
            written += 1

    if verbose:
        print(f"Wrote {written} pseudobonds to {out_path} (skipped {skipped})")

    return out_path


pb_file = write_chimerax_pseudobonds_conditional_color(
    nuxl_table=nuxl_table,
    out_path="nuxl_links.pb",
    threshold=distance_threshold,                  # Å, for example
    value_col="closest_chainB_dist", # threshold on distance
)

pb_file = write_chimerax_pseudobonds_sectioned(
    nuxl_table=nuxl_table,
    out_path="nuxl_links_sectioned.pb",
    threshold=15.0,
    nuxl_score_min=nuxl_score_threshold + 15,
    nuxl_score_max=100.0,
    radius_min=0.2,
    radius_max=1.2,
    res_from_col="residue_from_pdb",
    radius_round=3
)


print("Wrote:", pb_file)
nuxl_table.to_csv("nuxl_distances.tsv", index=False)
