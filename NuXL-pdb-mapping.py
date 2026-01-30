from Bio.PDB import PDBParser

parser = PDBParser(QUIET=True)
structure = parser.get_structure("X", "260126_nsun2_tRNA_fitted_on640.pdb")

model = next(structure.get_models())
chainA = model["A"]
res_a = chainA[(" ", 123, " ")]          # <-- change
atom_a = res_a["CA"]

motif = "GU"   # arbitrarily long
target_len = len(motif)

best_atom_b = None
best_residue_b = None
best_dist = float("inf")

# 1) extract ordered nucleotide residues from one chain
na_residues = []
na_sequence = []

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

for residue in structure[0]["B"]:
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

# 2) scan sequence windows
for i in range(len(na_sequence) - target_len + 1):
    window_seq = na_sequence[i : i + target_len]

    if window_seq != motif:
        continue

    # 3) residues corresponding to this motif hit
    window_residues = na_residues[i : i + target_len]

    # 4) test all C (non ') atoms in this window
    for residue in window_residues:
        for atom in residue:
            name = atom.get_name()

            if not name.startswith("C"):
                continue
            if "'" in name:
                continue

            d = atom_a - atom
            if d < best_dist:
                best_dist = d
                best_atom_b = atom
                best_residue_b = residue

# result
print("Closest motif hit:", motif)
print("Atom b:", best_atom_b.get_full_id(), best_atom_b.get_name())
print("Residue:", best_residue_b.get_resname(), best_residue_b.get_id())
print("Distance (Å):", best_dist)
