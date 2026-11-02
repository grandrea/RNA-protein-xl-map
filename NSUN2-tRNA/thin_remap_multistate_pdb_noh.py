from __future__ import annotations

import argparse
from pathlib import Path

from Bio.Align import PairwiseAligner
from Bio.PDB.PDBParser import PDBParser


DEFAULT_INPUT = "1000frames.pdb"
DEFAULT_REFERENCE_PDB = Path("..") / "20260529_Model_pXL-EM_AC_EL 1.pdb"

AA_CODES = {
    "ALA": "A",
    "ARG": "R",
    "ASN": "N",
    "ASP": "D",
    "CYS": "C",
    "GLN": "Q",
    "GLU": "E",
    "GLY": "G",
    "HIS": "H",
    "HSD": "H",
    "HSE": "H",
    "HSP": "H",
    "ILE": "I",
    "LEU": "L",
    "LYS": "K",
    "MET": "M",
    "PHE": "F",
    "PRO": "P",
    "SER": "S",
    "THR": "T",
    "TRP": "W",
    "TYR": "Y",
    "VAL": "V",
}

RNA_CODES = {
    "A": "A",
    "ADE": "A",
    "C": "C",
    "CYT": "C",
    "G": "G",
    "GUA": "G",
    "U": "U",
    "URA": "U",
    "5MC": "C",
}


def parse_resseq(line: str) -> int | None:
    try:
        return int(line[22:26])
    except ValueError:
        return None


def is_atom_record(line: str) -> bool:
    return line.startswith(("ATOM  ", "HETATM"))


def is_hydrogen(line: str) -> bool:
    element = line[76:78].strip().upper() if len(line) >= 78 else ""
    atom_name = line[12:16].strip().upper()
    return element == "H" or atom_name.startswith("H")


def residues_from_pdb(
    pdb_path: Path,
    chain_id: str,
    code_map: dict[str, str],
    min_resseq: int | None = None,
    max_resseq: int | None = None,
    first_frame_only: bool = False,
) -> tuple[list[int], str, dict[int, str]]:
    residues: list[int] = []
    residue_names: dict[int, str] = {}

    with pdb_path.open(encoding="utf-8") as handle:
        for line in handle:
            if first_frame_only and line.startswith("END"):
                break
            if not is_atom_record(line):
                continue
            if line[21].strip() != chain_id:
                continue

            resseq = parse_resseq(line)
            if resseq is None:
                continue
            if min_resseq is not None and resseq < min_resseq:
                continue
            if max_resseq is not None and resseq > max_resseq:
                continue

            resname = line[17:20].strip()
            if resname not in code_map:
                continue
            if resseq not in residue_names:
                residue_names[resseq] = resname
                residues.append(resseq)

    sequence = "".join(code_map[residue_names[resseq]] for resseq in residues)
    return residues, sequence, residue_names


def sequence_mapping(
    reference_residues: list[int],
    reference_sequence: str,
    trajectory_residues: list[int],
    trajectory_sequence: str,
    mode: str,
) -> tuple[dict[int, int], set[int]]:
    aligner = PairwiseAligner()
    aligner.mode = mode
    aligner.match_score = 2
    aligner.mismatch_score = -1
    aligner.open_gap_score = -10 if mode == "local" else -5
    aligner.extend_gap_score = -0.5 if mode == "local" else -1

    alignment = aligner.align(reference_sequence, trajectory_sequence)[0]
    ref_to_traj: dict[int, int] = {}
    matched_traj: set[int] = set()

    for ref_block, traj_block in zip(alignment.aligned[0], alignment.aligned[1]):
        for ref_index, traj_index in zip(
            range(ref_block[0], ref_block[1]),
            range(traj_block[0], traj_block[1]),
        ):
            if reference_sequence[ref_index] != trajectory_sequence[traj_index]:
                continue
            ref_resseq = reference_residues[ref_index]
            traj_resseq = trajectory_residues[traj_index]
            ref_to_traj[ref_resseq] = traj_resseq
            matched_traj.add(traj_resseq)

    return ref_to_traj, matched_traj


def invert_mapping(mapping: dict[int, int]) -> dict[int, int]:
    return {trajectory_resseq: reference_resseq for reference_resseq, trajectory_resseq in mapping.items()}


def validate_reference_with_biopython(reference_pdb: Path) -> None:
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("reference", reference_pdb)
    model = next(structure.get_models())
    chains = {chain.id for chain in model.get_chains()}
    missing = {"A", "B"} - chains
    if missing:
        raise SystemExit(
            f"Reference PDB is missing expected chain(s): {', '.join(sorted(missing))}"
        )


def normalize_rna_resname(resname: str) -> str:
    return RNA_CODES.get(resname, resname)


def build_remap(reference_pdb: Path, trajectory_pdb: Path) -> dict[tuple[str, int], tuple[str, int, str]]:
    validate_reference_with_biopython(reference_pdb)

    ref_protein_res, ref_protein_seq, ref_protein_names = residues_from_pdb(
        reference_pdb, chain_id="A", code_map=AA_CODES
    )
    traj_protein_res, traj_protein_seq, _ = residues_from_pdb(
        trajectory_pdb,
        chain_id="X",
        code_map=AA_CODES,
        min_resseq=1,
        max_resseq=586,
        first_frame_only=True,
    )
    protein_ref_to_traj, _ = sequence_mapping(
        ref_protein_res,
        ref_protein_seq,
        traj_protein_res,
        traj_protein_seq,
        mode="local",
    )

    ref_rna_res, ref_rna_seq, ref_rna_names = residues_from_pdb(
        reference_pdb, chain_id="B", code_map=RNA_CODES
    )
    traj_rna_res, traj_rna_seq, traj_rna_names = residues_from_pdb(
        trajectory_pdb,
        chain_id="X",
        code_map=RNA_CODES,
        min_resseq=587,
        max_resseq=661,
        first_frame_only=True,
    )
    rna_ref_to_traj, matched_rna_traj = sequence_mapping(
        ref_rna_res,
        ref_rna_seq,
        traj_rna_res,
        traj_rna_seq,
        mode="global",
    )

    validate_rna_region_around_40(rna_ref_to_traj, ref_rna_names, traj_rna_names)

    remap: dict[tuple[str, int], tuple[str, int, str]] = {}
    for trajectory_resseq, reference_resseq in invert_mapping(protein_ref_to_traj).items():
        remap[("X", trajectory_resseq)] = (
            "A",
            reference_resseq,
            ref_protein_names[reference_resseq],
        )
    for trajectory_resseq, reference_resseq in invert_mapping(rna_ref_to_traj).items():
        remap[("X", trajectory_resseq)] = (
            "B",
            reference_resseq,
            normalize_rna_resname(ref_rna_names[reference_resseq]),
        )

    for trajectory_resseq in sorted(set(traj_rna_res) - matched_rna_traj):
        remap[("X", trajectory_resseq)] = (
            "C",
            trajectory_resseq,
            normalize_rna_resname(traj_rna_names[trajectory_resseq]),
        )
    remap[("X", 662)] = ("C", 662, "SAH")

    missing_protein = [resseq for resseq in ref_protein_res if resseq not in protein_ref_to_traj]
    print(
        "Reference protein residues absent from trajectory: "
        + (", ".join(str(resseq) for resseq in missing_protein) if missing_protein else "none")
    )
    print(
        f"Protein mapping: X:{min(traj_protein_res)}-{max(traj_protein_res)} "
        f"-> A:{min(protein_ref_to_traj)}-{max(protein_ref_to_traj)}"
    )
    print("RNA mapping: X:587-660 -> B:1-74")
    print("Extra trajectory residues kept on chain C: X:661, X:662")

    return remap


def validate_rna_region_around_40(
    rna_ref_to_traj: dict[int, int],
    ref_rna_names: dict[int, str],
    traj_rna_names: dict[int, str],
) -> None:
    print("RNA validation around reference B:32-48")
    errors: list[str] = []
    for ref_resseq in range(32, 49):
        traj_resseq = rna_ref_to_traj.get(ref_resseq)
        ref_name = ref_rna_names.get(ref_resseq)
        traj_name = traj_rna_names.get(traj_resseq) if traj_resseq is not None else None
        print(f"  B:{ref_resseq} {ref_name} -> X:{traj_resseq} {traj_name}")

        if traj_resseq is None:
            errors.append(f"B:{ref_resseq} has no mapped trajectory residue")
            continue
        if RNA_CODES.get(ref_name or "") != RNA_CODES.get(traj_name or ""):
            errors.append(
                f"B:{ref_resseq} {ref_name} maps to X:{traj_resseq} {traj_name}"
            )

    if rna_ref_to_traj.get(40) != 626:
        errors.append(f"Expected B:40 -> X:626, got X:{rna_ref_to_traj.get(40)}")

    if errors:
        raise SystemExit("RNA mapping validation failed:\n" + "\n".join(errors))


def default_output_path(input_path: Path, stride: int) -> Path:
    return input_path.with_name(f"{input_path.stem}_noH_remapped_every{stride}.pdb")


def remap_atom_line(line: str, serial: int, chain_id: str, resseq: int, resname: str) -> str:
    return (
        f"{line[:6]}{serial:5d}{line[11:17]}{resname:>3s}"
        f"{line[20:21]}{chain_id:1s}{resseq:4d}{line[26:]}"
    )


def write_thinned_remapped_pdb(
    input_pdb: Path,
    output_pdb: Path,
    remap: dict[tuple[str, int], tuple[str, int, str]],
    stride: int,
) -> tuple[int, int, int]:
    frames_seen = 0
    frames_written = 0
    atoms_written = 0
    in_frame = False
    write_frame = False
    atom_serial = 1
    cryst1_line: str | None = None

    with input_pdb.open(encoding="utf-8") as source, output_pdb.open("w", encoding="utf-8") as target:
        for line in source:
            if line.startswith("CRYST1"):
                cryst1_line = line
                continue

            if line.startswith("END"):
                if write_frame:
                    target.write("ENDMDL\n")
                in_frame = False
                write_frame = False
                continue

            if not is_atom_record(line):
                continue

            if not in_frame:
                frames_seen += 1
                write_frame = (frames_seen - 1) % stride == 0
                in_frame = True
                if write_frame:
                    frames_written += 1
                    atom_serial = 1
                    target.write(f"MODEL     {frames_written:4d}\n")
                    if cryst1_line is not None:
                        target.write(cryst1_line)

            if not write_frame or is_hydrogen(line):
                continue

            original_chain = line[21].strip()
            original_resseq = parse_resseq(line)
            if original_resseq is None:
                continue

            mapped = remap.get((original_chain, original_resseq))
            if mapped is None:
                raise SystemExit(
                    f"No remapping rule for {original_chain}:{original_resseq} "
                    f"while writing frame {frames_seen}"
                )

            chain_id, resseq, resname = mapped
            target.write(remap_atom_line(line, atom_serial, chain_id, resseq, resname))
            atom_serial += 1
            atoms_written += 1

        if in_frame and write_frame:
            target.write("ENDMDL\n")

    return frames_seen, frames_written, atoms_written


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Thin 1000frames.pdb, remove hydrogens, and remap concatenated chain X "
            "onto reference-style protein/RNA chains."
        )
    )
    parser.add_argument("--input", default=DEFAULT_INPUT, type=Path)
    parser.add_argument("--stride", required=True, type=int)
    parser.add_argument("--reference-pdb", default=DEFAULT_REFERENCE_PDB, type=Path)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.stride < 1:
        raise SystemExit("--stride must be >= 1")

    input_pdb = args.input
    reference_pdb = args.reference_pdb
    output_pdb = args.output or default_output_path(input_pdb, args.stride)

    if not input_pdb.exists():
        raise SystemExit(f"Input PDB not found: {input_pdb}")
    if not reference_pdb.exists():
        raise SystemExit(f"Reference PDB not found: {reference_pdb}")

    remap = build_remap(reference_pdb=reference_pdb, trajectory_pdb=input_pdb)
    frames_seen, frames_written, atoms_written = write_thinned_remapped_pdb(
        input_pdb=input_pdb,
        output_pdb=output_pdb,
        remap=remap,
        stride=args.stride,
    )

    print(f"Wrote {output_pdb}")
    print(f"Frames read: {frames_seen}")
    print(f"Frames written: {frames_written}")
    print(f"Atoms written: {atoms_written}")


if __name__ == "__main__":
    main()
