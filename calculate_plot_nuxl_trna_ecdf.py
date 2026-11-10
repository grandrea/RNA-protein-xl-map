from pathlib import Path
import argparse
import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from Bio.PDB import PDBParser
from Bio.PDB.Polypeptide import is_aa
from scipy.stats import ks_2samp

from generate_xlinked_ca_chimerax import (
    protein_residue_scores_from_nuxl_table,
    resolve_table_path,
)


DEFAULT_PDB = "20260529_Model_pXL-EM_AC_EL 1.pdb"
DEFAULT_NUXL_TABLE = "Peak2_NSUN2_tRNA.tsv"
DEFAULT_OUTPUT_PREFIX = "experimental_structure_nuxl_trna_distance"
DEFAULT_OUTPUT_DIR = "experimental_structure_analysis"
CONFIDENCE_THRESHOLD = 60.0
MAX_CONFIDENCE_SCORE = 85.0
PHOSPHATE_ATOM_NAMES = {"P", "OP1", "OP2", "OP3", "O1P", "O2P", "O3P"}
CONFIDENCE_COLORS = {
    "low": "#c89b00",
    "medium": "#f28e2b",
    "high": "#d62728",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Calculate protein-to-tRNA nucleobase distances and plot the NuXL "
            "confidence-stratified ECDF used in the paper."
        )
    )
    parser.add_argument("--pdb", default=DEFAULT_PDB)
    parser.add_argument("--nuxl-table", default=DEFAULT_NUXL_TABLE)
    parser.add_argument("--protein-chain", default="A")
    parser.add_argument("--trna-chain", default="B")
    parser.add_argument("--output-prefix", default=DEFAULT_OUTPUT_PREFIX)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def is_trna_base_heavy_atom(atom):
    atom_name = atom.name.strip()
    return (
        (atom.element or "").upper() != "H"
        and "'" not in atom_name
        and atom_name not in PHOSPHATE_ATOM_NAMES
    )


def minimum_residue_to_trna_distances(
    pdb_path,
    protein_chain_id="A",
    trna_chain_id="B",
):
    structure = PDBParser(QUIET=True).get_structure("nuxl_distance", pdb_path)
    model = next(structure.get_models())
    protein_chain = model[protein_chain_id]
    trna_chain = model[trna_chain_id]

    trna_atoms = []
    trna_metadata = []
    trna_base_atoms = []
    trna_base_metadata = []
    for residue in trna_chain:
        for atom in residue.get_atoms():
            trna_atoms.append(atom.coord)
            trna_metadata.append(
                (residue.id[1], residue.resname.strip(), atom.name.strip())
            )
            if is_trna_base_heavy_atom(atom):
                trna_base_atoms.append(atom.coord)
                trna_base_metadata.append(
                    (residue.id[1], residue.resname.strip(), atom.name.strip())
                )

    if not trna_atoms:
        raise ValueError(f"No atoms found in tRNA chain {trna_chain_id}.")
    if not trna_base_atoms:
        raise ValueError(f"No base heavy atoms found in tRNA chain {trna_chain_id}.")

    trna_coordinates = np.asarray(trna_atoms, dtype=float)
    trna_base_coordinates = np.asarray(trna_base_atoms, dtype=float)
    rows = []
    for residue in protein_chain:
        if not is_aa(residue, standard=True):
            continue

        protein_atoms = list(residue.get_atoms())
        if not protein_atoms:
            continue

        protein_coordinates = np.asarray(
            [atom.coord for atom in protein_atoms], dtype=float
        )
        squared_distances = np.sum(
            (protein_coordinates[:, np.newaxis, :] - trna_coordinates) ** 2,
            axis=2,
        )
        protein_index, trna_index = np.unravel_index(
            np.argmin(squared_distances), squared_distances.shape
        )
        trna_residue, trna_resname, trna_atom = trna_metadata[trna_index]
        base_squared_distances = np.sum(
            (protein_coordinates[:, np.newaxis, :] - trna_base_coordinates) ** 2,
            axis=2,
        )
        protein_base_index, trna_base_index = np.unravel_index(
            np.argmin(base_squared_distances), base_squared_distances.shape
        )
        trna_base_residue, trna_base_resname, trna_base_atom = trna_base_metadata[
            trna_base_index
        ]
        rows.append(
            {
                "Protein_chain": protein_chain_id,
                "Protein_residue": residue.id[1],
                "Protein_resname": residue.resname.strip(),
                "Protein_atom": protein_atoms[protein_index].name.strip(),
                "tRNA_chain": trna_chain_id,
                "tRNA_residue": trna_residue,
                "tRNA_resname": trna_resname,
                "tRNA_atom": trna_atom,
                "Min_atom_distance_A": math.sqrt(
                    squared_distances[protein_index, trna_index]
                ),
                "Protein_atom_to_base": protein_atoms[
                    protein_base_index
                ].name.strip(),
                "tRNA_base_residue": trna_base_residue,
                "tRNA_base_resname": trna_base_resname,
                "tRNA_base_atom": trna_base_atom,
                "Min_base_atom_distance_A": math.sqrt(
                    base_squared_distances[protein_base_index, trna_base_index]
                ),
            }
        )

    return pd.DataFrame(rows).sort_values("Protein_residue").reset_index(drop=True)


def add_nuxl_annotations(distances, table_path):
    annotated = distances.copy()
    scores = protein_residue_scores_from_nuxl_table(
        table_path, CONFIDENCE_THRESHOLD
    )
    suffix = f"score{int(CONFIDENCE_THRESHOLD)}"
    annotated[f"NuXL_max_{suffix}"] = annotated["Protein_residue"].map(scores)
    return annotated


def plot_confidence_cumulative_distribution(data, out_path):
    score_column = f"NuXL_max_score{int(CONFIDENCE_THRESHOLD)}"
    midpoint = (CONFIDENCE_THRESHOLD + MAX_CONFIDENCE_SCORE) / 2
    confidence_groups = {
        "low": data[
            data[score_column].gt(CONFIDENCE_THRESHOLD)
            & data[score_column].lt(midpoint)
        ],
        "medium": data[
            data[score_column].ge(midpoint)
            & data[score_column].lt(MAX_CONFIDENCE_SCORE)
        ],
        "high": data[data[score_column].ge(MAX_CONFIDENCE_SCORE)],
    }
    background = data[data[score_column].isna()]
    labels = {
        "low": f">{CONFIDENCE_THRESHOLD:g} to <{midpoint:g}",
        "medium": f"{midpoint:g} to <{MAX_CONFIDENCE_SCORE:g}",
        "high": f">={MAX_CONFIDENCE_SCORE:g}",
    }
    fig, ax = plt.subplots(figsize=(8.4, 5.4), constrained_layout=True)
    all_distances = np.sort(data["Min_base_atom_distance_A"].to_numpy())
    all_cumulative = np.arange(1, all_distances.size + 1) * 100 / all_distances.size
    all_median = float(np.median(all_distances))
    ax.step(
        all_distances,
        all_cumulative,
        where="post",
        color="#59636e",
        linewidth=2.2,
        label=(
            f"All modeled amino acids "
            f"(n={all_distances.size}, median={all_median:.1f} A)"
        ),
    )
    summaries = {}
    ks_results = []
    ecdf_rows = []
    sorted_all = data.sort_values("Min_base_atom_distance_A")
    for (_, row), cumulative in zip(sorted_all.iterrows(), all_cumulative):
        ecdf_rows.append(
            {
                "Series": "All modeled amino acids",
                "Protein_chain": row["Protein_chain"],
                "Protein_residue": row["Protein_residue"],
                "Protein_resname": row["Protein_resname"],
                "NuXL_score": row[score_column],
                "Confidence_class": "background"
                if pd.isna(row[score_column])
                else "all NuXL scores",
                "Min_base_atom_distance_A": row["Min_base_atom_distance_A"],
                "Cumulative_percent": cumulative,
            }
        )
    annotation_lines = ["One-sided KS: high confidence has shorter distances"]
    for confidence in ("low", "medium", "high"):
        distances = np.sort(
            confidence_groups[confidence]["Min_base_atom_distance_A"].to_numpy()
        )
        if distances.size == 0:
            raise ValueError(f"No modeled residues in {confidence} confidence group.")
        cumulative = np.arange(1, distances.size + 1) * 100 / distances.size
        median = float(np.median(distances))
        summaries[confidence] = {"count": distances.size, "median": median}
        sorted_group = confidence_groups[confidence].sort_values(
            "Min_base_atom_distance_A"
        )
        for (_, row), cumulative_value in zip(
            sorted_group.iterrows(), cumulative
        ):
            ecdf_rows.append(
                {
                    "Series": labels[confidence],
                    "Protein_chain": row["Protein_chain"],
                    "Protein_residue": row["Protein_residue"],
                    "Protein_resname": row["Protein_resname"],
                    "NuXL_score": row[score_column],
                    "Confidence_class": confidence,
                    "Min_base_atom_distance_A": row[
                        "Min_base_atom_distance_A"
                    ],
                    "Cumulative_percent": cumulative_value,
                }
            )
        ax.step(
            distances,
            cumulative,
            where="post",
            color=CONFIDENCE_COLORS[confidence],
            linewidth=2.5,
            label=(
                f"{labels[confidence]} "
                f"(n={distances.size}, median={median:.1f} A)"
            ),
        )

    for first, second in (("low", "medium"), ("low", "high"), ("medium", "high")):
        first_distances = confidence_groups[first][
            "Min_base_atom_distance_A"
        ].to_numpy()
        second_distances = confidence_groups[second][
            "Min_base_atom_distance_A"
        ].to_numpy()
        ks_result = ks_2samp(
            first_distances,
            second_distances,
            alternative="two-sided",
            method="auto",
        )
        ks_results.append(
            {
                "Comparison": f"{first}_vs_{second}",
                "Group_1": first,
                "Group_2": second,
                "N_1": len(first_distances),
                "N_2": len(second_distances),
                "KS_D": ks_result.statistic,
                "P_value": ks_result.pvalue,
            }
        )

    high_distances = confidence_groups["high"][
        "Min_base_atom_distance_A"
    ].to_numpy()
    directional_comparisons = {
        "low": confidence_groups["low"]["Min_base_atom_distance_A"].to_numpy(),
        "non-NuXL background": background["Min_base_atom_distance_A"].to_numpy(),
    }
    for comparison_name, comparison_distances in directional_comparisons.items():
        ks_result = ks_2samp(
            high_distances,
            comparison_distances,
            alternative="greater",
            method="auto",
        )
        comparison_id = comparison_name.replace("-", "_").replace(" ", "_")
        ks_results.append(
            {
                "Comparison": f"high_shorter_vs_{comparison_id}",
                "Group_1": "high",
                "Group_2": comparison_name,
                "N_1": len(high_distances),
                "N_2": len(comparison_distances),
                "KS_D": ks_result.statistic,
                "P_value": ks_result.pvalue,
            }
        )
        annotation_lines.append(
            f"High vs {comparison_name}: "
            f"D={ks_result.statistic:.2f}, p={ks_result.pvalue:.2g}"
        )

    ax.set_xlabel(
        "Minimum distance from any protein-residue atom to any tRNA base atom (A)"
    )
    ax.set_ylabel("Cumulative % crosslinked residues")
    ax.set_xlim(left=0)
    ax.set_ylim(0, 100.5)
    ax.grid(color="#d9dde1", linewidth=0.7, alpha=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(
        frameon=False,
        fontsize=9,
        loc="lower right",
        title="NuXL score",
        title_fontsize=9,
    )
    ax.text(
        0.02,
        0.98,
        "\n".join(annotation_lines),
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=8.5,
        bbox={"facecolor": "white", "edgecolor": "#c7ccd1", "alpha": 0.9},
    )
    fig.savefig(out_path, dpi=300)
    fig.savefig(out_path.with_suffix(".pdf"))
    plt.close(fig)
    summaries["all"] = {"count": all_distances.size, "median": all_median}
    return summaries, pd.DataFrame(ks_results), pd.DataFrame(ecdf_rows)


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    table_path = resolve_table_path(args.nuxl_table)

    distances = minimum_residue_to_trna_distances(
        pdb_path=args.pdb,
        protein_chain_id=args.protein_chain,
        trna_chain_id=args.trna_chain,
    )
    annotated = add_nuxl_annotations(distances, table_path)
    cumulative_path = output_dir / f"{args.output_prefix}_ecdf.png"
    confidence_summary, ks_results, ecdf_data = (
        plot_confidence_cumulative_distribution(
            data=annotated,
            out_path=cumulative_path,
        )
    )
    ks_path = output_dir / f"{args.output_prefix}_ks_tests.csv"
    ks_results.to_csv(ks_path, index=False)
    ecdf_path = output_dir / f"{args.output_prefix}_ecdf_data.csv"
    ecdf_data.to_csv(ecdf_path, index=False)
    print(f"Wrote {cumulative_path}")
    print(f"Wrote {ks_path}")
    print(f"Wrote {ecdf_path}")
    all_summary = confidence_summary["all"]
    print(
        f"All modeled amino acids: n={all_summary['count']}, "
        f"median base-atom distance={all_summary['median']:.2f} A"
    )
    for confidence in ("low", "medium", "high"):
        summary = confidence_summary[confidence]
        print(
            f"{confidence.capitalize()} confidence: n={summary['count']}, "
            f"median base-atom distance={summary['median']:.2f} A"
        )
    for row in ks_results.itertuples(index=False):
        print(
            f"KS {row.Group_1} vs {row.Group_2}: "
            f"D={row.KS_D:.3f}, p={row.P_value:.3g}"
        )

    if table_path.name != Path(args.nuxl_table).name:
        print(f"Used NuXL table: {table_path}")


if __name__ == "__main__":
    main()
