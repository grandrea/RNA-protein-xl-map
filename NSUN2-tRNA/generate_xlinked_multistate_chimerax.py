from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent

DEFAULT_PDB = "1000frames_noH_remapped_every250.pdb"
DEFAULT_REFERENCE_PDB = Path("..") / "20260529_Model_pXL-EM_AC_EL 1.pdb"
DEFAULT_NUXL_TABLE = "Peak2_tRNA.tsv"
FALLBACK_NUXL_TABLE = "Peak2_NSUN2_tRNA.tsv"
DEFAULT_OUT = "nuxl_xlinked_residues_score80_multistate.cxc"
DEFAULT_SELECTION_NAME = "nuxl_xlinked_protein_residues_score80"
NUXL_SCORE_THRESHOLD = 80
TRAJECTORY_HOLD_STEADY_ATOMS = "#1/A@CA"


def resolve_existing_path(path: str | Path, fallback_names: list[str] | None = None) -> Path:
    requested = Path(path)
    candidates = [
        requested,
        SCRIPT_DIR / requested,
        REPO_ROOT / requested,
    ]

    if fallback_names:
        for fallback_name in fallback_names:
            fallback = Path(fallback_name)
            candidates.extend(
                [
                    fallback,
                    SCRIPT_DIR / fallback,
                    REPO_ROOT / fallback,
                ]
            )

    for candidate in candidates:
        if candidate.exists():
            return candidate

    return requested


def resolve_pdb_path(path: str | Path) -> Path:
    resolved = resolve_existing_path(path)
    if resolved.exists():
        return resolved

    requested = Path(path)
    if requested.name == DEFAULT_PDB:
        matches = sorted(
            SCRIPT_DIR.glob("1000frames_noH_remapped_every*.pdb"),
            key=lambda match: match.stat().st_mtime,
            reverse=True,
        )
        if matches:
            return matches[0]

    return resolved


def protein_residues_from_nuxl_table(table_path: Path, score_threshold: float) -> list[int]:
    nuxl_table = pd.read_csv(table_path, sep="\t")
    nuxl_table = nuxl_table[nuxl_table["NuXLScore_score"] > score_threshold].copy()
    nuxl_table["protein_residue"] = (
        nuxl_table["start"] + nuxl_table["NuXL:best_localization_position"]
    )

    return (
        nuxl_table["protein_residue"]
        .dropna()
        .astype(int)
        .drop_duplicates()
        .sort_values()
        .tolist()
    )


def residues_in_first_model(pdb_path: Path, chain_id: str = "A") -> set[int]:
    residues: set[int] = set()
    in_first_model = False
    saw_model = False

    with pdb_path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.startswith("MODEL"):
                saw_model = True
                in_first_model = True
                continue
            if line.startswith("ENDMDL") and saw_model:
                break
            if saw_model and not in_first_model:
                continue
            if not line.startswith(("ATOM  ", "HETATM")):
                continue
            if line[21].strip() != chain_id:
                continue
            try:
                residues.add(int(line[22:26]))
            except ValueError:
                continue

    return residues


def residue_spec(residues: list[int], chain: str = "A", model: str = "#1") -> str:
    return f"{model}/{chain}:{','.join(str(resnum) for resnum in residues)}"


def chimerax_path(path: Path) -> str:
    return str(path.resolve()).replace("\\", "/").replace('"', '\\"')


def write_chimerax_commands(
    pdb_path: Path,
    residues: list[int],
    out_path: Path,
    selection_name: str,
) -> None:
    residues_target = residue_spec(residues)

    with out_path.open("w", encoding="utf-8") as handle:
        handle.write(f'open "{chimerax_path(pdb_path)}"\n')
        handle.write("set bgColor white\n")
        handle.write("lighting soft\n")
        handle.write("hide #1 target a\n")
        handle.write("hide #1 target r\n")
        handle.write("hide #1 target s\n")
        handle.write("show #1/A target s\n")
        handle.write("color #1/A lightgray target s\n")
        handle.write("show #1/B target r\n")
        handle.write("nucleotides #1/B stubs\n")
        handle.write("color #1/B teal target ar\n")
        handle.write("hide #1/C target a\n")
        handle.write(f"name frozen {selection_name} {residues_target}\n")
        handle.write(f"color {selection_name} red target as\n")
        handle.write("transparency 30 target r\n")
        handle.write(f"select {selection_name}\n")
        handle.write(
            f"coordset #1 1 computeSs true holdSteady {TRAJECTORY_HOLD_STEADY_ATOMS}\n"
        )
        handle.write(
            f"coordset slider #1 computeSs true holdSteady {TRAJECTORY_HOLD_STEADY_ATOMS}\n"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate a ChimeraX command file for score-80 NuXL residues on a "
            "remapped multistate NSUN2-tRNA PDB."
        )
    )
    parser.add_argument("--pdb", default=DEFAULT_PDB)
    parser.add_argument("--reference-pdb", default=DEFAULT_REFERENCE_PDB)
    parser.add_argument("--nuxl-table", default=DEFAULT_NUXL_TABLE)
    parser.add_argument("--out", default=DEFAULT_OUT)
    parser.add_argument("--selection-name", default=DEFAULT_SELECTION_NAME)
    parser.add_argument("--score-threshold", type=float, default=NUXL_SCORE_THRESHOLD)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    pdb_path = resolve_pdb_path(args.pdb)
    reference_pdb_path = resolve_existing_path(args.reference_pdb)
    table_path = resolve_existing_path(
        args.nuxl_table,
        fallback_names=[FALLBACK_NUXL_TABLE]
        if Path(args.nuxl_table).name == DEFAULT_NUXL_TABLE
        else None,
    )
    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = SCRIPT_DIR / out_path

    if not pdb_path.exists():
        raise SystemExit(f"Remapped multistate PDB not found: {pdb_path}")
    if not reference_pdb_path.exists():
        raise SystemExit(f"Reference PDB not found: {reference_pdb_path}")
    if not table_path.exists():
        raise SystemExit(f"NuXL table not found: {table_path}")

    residues = protein_residues_from_nuxl_table(
        table_path=table_path,
        score_threshold=args.score_threshold,
    )
    if not residues:
        raise SystemExit("No protein-side crosslinked residues were found.")

    reference_residues = residues_in_first_model(reference_pdb_path, chain_id="A")
    residues_in_reference = [resnum for resnum in residues if resnum in reference_residues]
    missing_from_reference = [
        resnum for resnum in residues if resnum not in reference_residues
    ]

    pdb_residues = residues_in_first_model(pdb_path, chain_id="A")
    missing_residues = [
        resnum for resnum in residues_in_reference if resnum not in pdb_residues
    ]
    residues_in_model = [
        resnum for resnum in residues_in_reference if resnum in pdb_residues
    ]

    if not residues_in_model:
        raise SystemExit("No selected protein residues are present in chain A of the PDB.")

    write_chimerax_commands(
        pdb_path=pdb_path,
        residues=residues_in_model,
        out_path=out_path,
        selection_name=args.selection_name,
    )

    print(f"Wrote {len(residues_in_model)} residues to {out_path}")
    print(f"Score threshold: {args.score_threshold:g}")
    print(f"Selection name: {args.selection_name}")
    print(f"PDB: {pdb_path}")
    print(f"Reference PDB: {reference_pdb_path}")
    print(f"NuXL table: {table_path}")
    if missing_from_reference:
        print(
            "Ignored score-filtered protein residues absent from the reference "
            f"chain A in {reference_pdb_path}: "
            + ", ".join(str(resnum) for resnum in missing_from_reference)
        )
    if missing_residues:
        print(
            "Warning: these reference-present score-filtered protein residues are absent from "
            f"chain A in {pdb_path}: "
            + ", ".join(str(resnum) for resnum in missing_residues)
        )


if __name__ == "__main__":
    main()
