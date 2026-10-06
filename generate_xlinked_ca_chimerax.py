from pathlib import Path
import argparse

import pandas as pd


DEFAULT_PDB = "20260529_Model_pXL-EM_AC_EL 1.pdb"
DEFAULT_NUXL_TABLE = "Peak2_tRNA.tsv"
FALLBACK_NUXL_TABLE = "Peak2_NSUN2_tRNA.tsv"
DEFAULT_OUT = "nuxl_xlinked_residues_score80.cxc"
DEFAULT_SELECTION_NAME = "nuxl_xlinked_protein_residues"
NUXL_SCORE_THRESHOLD = 80
MAX_CONFIDENCE_SCORE = 85
CONFIDENCE_COLORS = {
    "low": "gold",
    "medium": "orange",
    "high": "red",
}


def resolve_table_path(path):
    requested = Path(path)
    if requested.exists():
        return requested

    fallback = Path(FALLBACK_NUXL_TABLE)
    if requested.name == "Peak2_tRNA.tsv" and fallback.exists():
        return fallback

    return requested


def protein_residue_scores_from_nuxl_table(table_path, score_threshold):
    nuxl_table = pd.read_csv(table_path, sep="\t")
    nuxl_table = nuxl_table[nuxl_table["NuXLScore_score"] > score_threshold].copy()
    nuxl_table = nuxl_table[
        nuxl_table["NuXL:best_localization_position"].notna()
        & (nuxl_table["NuXL:best_localization_position"] >= 0)
    ].copy()
    nuxl_table["protein_residue"] = (
        nuxl_table["start"] + nuxl_table["NuXL:best_localization_position"]
    )

    residue_scores = (
        nuxl_table.dropna(subset=["protein_residue", "NuXLScore_score"])
        .assign(protein_residue=lambda frame: frame["protein_residue"].astype(int))
        .groupby("protein_residue")["NuXLScore_score"]
        .max()
        .sort_index()
        .to_dict()
    )
    return residue_scores


def confidence_bins(residue_scores, score_threshold, max_confidence_score):
    if max_confidence_score <= score_threshold:
        raise ValueError("Maximum-confidence score must exceed the score threshold.")

    midpoint = (score_threshold + max_confidence_score) / 2
    bins = {"low": [], "medium": [], "high": []}
    for residue, score in sorted(residue_scores.items()):
        if score >= max_confidence_score:
            bins["high"].append(residue)
        elif score >= midpoint:
            bins["medium"].append(residue)
        else:
            bins["low"].append(residue)
    return bins, midpoint


def residues_in_pdb(pdb_path, chain_id="A"):
    residues = set()
    with open(pdb_path, encoding="utf-8") as handle:
        for line in handle:
            if not line.startswith(("ATOM  ", "HETATM")):
                continue
            if line[21].strip() != chain_id:
                continue
            try:
                residues.add(int(line[22:26]))
            except ValueError:
                continue
    return residues


def residue_spec(residues, chain="A", model="#1"):
    return f"{model}/{chain}:{','.join(str(resnum) for resnum in residues)}"


def chimerax_path(path):
    return str(Path(path).resolve()).replace("\\", "/").replace('"', '\\"')


def write_chimerax_commands(
    pdb_path,
    bins,
    out_path,
    selection_name,
    score_threshold,
    midpoint,
    max_confidence_score,
):
    all_residues = sorted(residue for residues in bins.values() for residue in residues)
    residues_target = residue_spec(all_residues)

    with open(out_path, "w", encoding="utf-8") as handle:
        handle.write(
            f"# NuXL confidence bins: low >{score_threshold:g} to <{midpoint:g}; "
            f"medium {midpoint:g} to <{max_confidence_score:g}; "
            f"high >={max_confidence_score:g}\n"
        )
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
        handle.write(f"name frozen {selection_name} {residues_target}\n")
        for confidence in ("low", "medium", "high"):
            residues = bins[confidence]
            if not residues:
                continue
            bin_name = f"{selection_name}_{confidence}"
            handle.write(f"name frozen {bin_name} {residue_spec(residues)}\n")
            handle.write(
                f"color {bin_name} {CONFIDENCE_COLORS[confidence]} target as\n"
            )
        handle.write("transparency 30 target r\n")
        handle.write(f"select {selection_name}\n")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate a ChimeraX command file for protein CA atoms crosslinked to RNA."
    )
    parser.add_argument("--pdb", default=DEFAULT_PDB)
    parser.add_argument("--nuxl-table", default=DEFAULT_NUXL_TABLE)
    parser.add_argument("--out", default=DEFAULT_OUT)
    parser.add_argument("--selection-name", default=DEFAULT_SELECTION_NAME)
    parser.add_argument("--score-threshold", type=float, default=NUXL_SCORE_THRESHOLD)
    parser.add_argument(
        "--max-confidence-score",
        type=float,
        default=MAX_CONFIDENCE_SCORE,
        help="Scores at or above this value use the high-confidence color.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    table_path = resolve_table_path(args.nuxl_table)

    residue_scores = protein_residue_scores_from_nuxl_table(
        table_path=table_path,
        score_threshold=args.score_threshold,
    )
    if not residue_scores:
        raise SystemExit("No protein-side crosslinked residues were found.")

    pdb_residues = residues_in_pdb(args.pdb)
    missing_residues = [
        resnum for resnum in residue_scores if resnum not in pdb_residues
    ]
    scores_in_model = {
        resnum: score
        for resnum, score in residue_scores.items()
        if resnum in pdb_residues
    }

    if not scores_in_model:
        raise SystemExit("No selected protein residues are present in the PDB.")

    bins, midpoint = confidence_bins(
        residue_scores=scores_in_model,
        score_threshold=args.score_threshold,
        max_confidence_score=args.max_confidence_score,
    )

    write_chimerax_commands(
        pdb_path=args.pdb,
        bins=bins,
        out_path=args.out,
        selection_name=args.selection_name,
        score_threshold=args.score_threshold,
        midpoint=midpoint,
        max_confidence_score=args.max_confidence_score,
    )

    print(f"Wrote {len(scores_in_model)} residues to {args.out}")
    print(f"Score threshold: {args.score_threshold:g}")
    print(
        "Confidence bins: "
        f"low={len(bins['low'])} (>{args.score_threshold:g} to <{midpoint:g}), "
        f"medium={len(bins['medium'])} ({midpoint:g} to "
        f"<{args.max_confidence_score:g}), "
        f"high={len(bins['high'])} (>={args.max_confidence_score:g})"
    )
    print(f"Selection name: {args.selection_name}")
    if missing_residues:
        print(
            "Warning: these protein residues were found in the TSV but are absent "
            f"from chain A in {args.pdb}: "
            + ", ".join(str(resnum) for resnum in missing_residues)
        )
    if table_path.name != Path(args.nuxl_table).name:
        print(f"Used table: {table_path}")


if __name__ == "__main__":
    main()
