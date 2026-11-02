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
DEFAULT_NUXL_TABLE = "Peak2_tRNA.tsv"
DEFAULT_BIN_WIDTH = 2.0
DEFAULT_MODEL_LABEL = "Original pXL-EM model"
DEFAULT_OUTPUT_PREFIX = "original_model"
SCORE_THRESHOLDS = (60.0, 80.0)
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
            "Compare minimum all-atom protein-to-tRNA distances for NuXL sites "
            "and all modeled amino acids."
        )
    )
    parser.add_argument("--pdb", default=DEFAULT_PDB)
    parser.add_argument("--nuxl-table", default=DEFAULT_NUXL_TABLE)
    parser.add_argument("--protein-chain", default="A")
    parser.add_argument("--trna-chain", default="B")
    parser.add_argument("--bin-width", type=float, default=DEFAULT_BIN_WIDTH)
    parser.add_argument("--model-label", default=DEFAULT_MODEL_LABEL)
    parser.add_argument("--output-prefix", default=DEFAULT_OUTPUT_PREFIX)
    parser.add_argument(
        "--report",
        help="Distance CSV filename; defaults to '<output-prefix>_distances.csv'.",
    )
    parser.add_argument("--output-dir", default=".")
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
    for threshold in SCORE_THRESHOLDS:
        scores = protein_residue_scores_from_nuxl_table(table_path, threshold)
        suffix = f"score{int(threshold)}"
        annotated[f"NuXL_max_{suffix}"] = annotated["Protein_residue"].map(scores)
        annotated[f"Passes_{suffix}"] = annotated["Protein_residue"].isin(scores)
    return annotated


def histogram_bins(distances, bin_width):
    if bin_width <= 0:
        raise ValueError("Histogram bin width must be positive.")
    upper = math.ceil(distances.max() / bin_width) * bin_width
    return np.arange(0, upper + bin_width, bin_width)


def plot_frequency_histogram(data, threshold, bins, out_path, model_label):
    selected_column = f"Passes_score{int(threshold)}"
    all_distances = data["Min_atom_distance_A"].to_numpy()
    selected_distances = data.loc[
        data[selected_column], "Min_atom_distance_A"
    ].to_numpy()
    if selected_distances.size == 0:
        raise ValueError(f"No modeled residues pass score threshold {threshold:g}.")

    all_weights = np.full(all_distances.size, 100.0 / all_distances.size)
    selected_weights = np.full(
        selected_distances.size, 100.0 / selected_distances.size
    )

    fig, ax = plt.subplots(figsize=(8.4, 5.4), constrained_layout=True)
    ax.hist(
        all_distances,
        bins=bins,
        weights=all_weights,
        color="#b8bec5",
        edgecolor="#59636e",
        linewidth=0.7,
        alpha=0.65,
        label=f"All modeled amino acids (n={all_distances.size})",
    )
    ax.hist(
        selected_distances,
        bins=bins,
        weights=selected_weights,
        histtype="step",
        color="#d62728",
        linewidth=2.4,
        label=f"NuXL score > {threshold:g} (n={selected_distances.size})",
    )

    all_median = float(np.median(all_distances))
    selected_median = float(np.median(selected_distances))
    ax.axvline(
        all_median,
        color="#59636e",
        linestyle="--",
        linewidth=1.4,
        label=f"All-AA median: {all_median:.1f} A",
    )
    ax.axvline(
        selected_median,
        color="#d62728",
        linestyle=":",
        linewidth=1.8,
        label=f"NuXL median: {selected_median:.1f} A",
    )

    ax.set_title(
        f"{model_label}\nMinimum all-atom distance to tRNA: "
        f"NuXL score > {threshold:g}"
    )
    ax.set_xlabel("Minimum distance from protein residue to any tRNA atom (A)")
    ax.set_ylabel("Residues per bin (%)")
    ax.set_xlim(bins[0], bins[-1])
    ax.set_ylim(bottom=0)
    ax.grid(axis="y", color="#d9dde1", linewidth=0.7, alpha=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=9)
    fig.savefig(out_path, dpi=300)
    plt.close(fig)

    return {
        "selected_count": selected_distances.size,
        "all_count": all_distances.size,
        "selected_median": selected_median,
        "all_median": all_median,
    }


def plot_confidence_cumulative_distribution(data, out_path, model_label):
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
        "low": f"Low: >{CONFIDENCE_THRESHOLD:g} to <{midpoint:g}",
        "medium": f"Medium: {midpoint:g} to <{MAX_CONFIDENCE_SCORE:g}",
        "high": f"High: >={MAX_CONFIDENCE_SCORE:g}",
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

    ax.set_title(
        f"{model_label}\n"
        "Cumulative minimum distance to tRNA nucleobase heavy atoms"
    )
    ax.set_xlabel(
        "Minimum distance from any protein-residue atom to any tRNA base atom (A)"
    )
    ax.set_ylabel("Cumulative residues (%)")
    ax.set_xlim(left=0)
    ax.set_ylim(0, 100.5)
    ax.grid(color="#d9dde1", linewidth=0.7, alpha=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=9, loc="lower right")
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
    plt.close(fig)
    summaries["all"] = {"count": all_distances.size, "median": all_median}
    return summaries, pd.DataFrame(ks_results)


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
    report_name = args.report or f"{args.output_prefix}_distances.csv"
    report_path = output_dir / report_name
    annotated.to_csv(report_path, index=False)

    bins = histogram_bins(annotated["Min_atom_distance_A"], args.bin_width)
    for threshold in SCORE_THRESHOLDS:
        out_path = output_dir / (
            f"{args.output_prefix}_score{int(threshold)}_"
            "min_atom_to_trna_distance_histogram.png"
        )
        summary = plot_frequency_histogram(
            data=annotated,
            threshold=threshold,
            bins=bins,
            out_path=out_path,
            model_label=args.model_label,
        )
        all_sites = protein_residue_scores_from_nuxl_table(table_path, threshold)
        missing_count = len(set(all_sites) - set(annotated["Protein_residue"]))
        print(
            f"Score > {threshold:g}: plotted {summary['selected_count']} modeled "
            f"NuXL residues; {missing_count} threshold-passing residues absent "
            f"from the PDB. Median distances: NuXL "
            f"{summary['selected_median']:.2f} A, all AA "
            f"{summary['all_median']:.2f} A."
        )
        print(f"Wrote {out_path}")

    cumulative_path = output_dir / (
        f"{args.output_prefix}_confidence_min_base_distance_cumulative.png"
    )
    confidence_summary, ks_results = plot_confidence_cumulative_distribution(
        data=annotated,
        out_path=cumulative_path,
        model_label=args.model_label,
    )
    ks_path = output_dir / f"{args.output_prefix}_confidence_ks_tests.csv"
    ks_results.to_csv(ks_path, index=False)
    print(f"Wrote {cumulative_path}")
    print(f"Wrote {ks_path}")
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

    print(f"Wrote {report_path} with {len(annotated)} modeled amino acids")
    if table_path.name != Path(args.nuxl_table).name:
        print(f"Used NuXL table: {table_path}")


if __name__ == "__main__":
    main()
