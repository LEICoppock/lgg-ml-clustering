
# Code used to create data for Figure 3b

import os
import numpy as np
import pandas as pd


ENRICHMENT_FILE = "enrichment_results/cluster_enrichment_analysis.csv"


PROTEIN_FILE = "esm2_clustering/all_proteins_with_clusters.csv"

# Table with columns: strain, beneficial (1 = beneficial strain, 0 = comparison strain)
STRAIN_METADATA_FILE = "strain_metadata.csv"

OUTPUT_DIR = "prevalence_results"

TOP_N = 10

HYPOTHETICAL_PATTERN = r"\bhypothetical\b|\buncharacteri[sz]ed\b"


# 1. Load inputs
print("Loading files...")

os.makedirs(OUTPUT_DIR, exist_ok=True)

enrichment = pd.read_csv(ENRICHMENT_FILE)
proteins = pd.read_csv(PROTEIN_FILE, usecols=["gene_id", "strain", "cluster"])
metadata = pd.read_csv(STRAIN_METADATA_FILE, usecols=["strain", "beneficial"])

enrichment["annotation"] = enrichment["annotation"].fillna("").astype(str).str.strip()


# 2. Annotation status of each cluster's representative annotation

annotation = enrichment["annotation"]
hypothetical = annotation.str.contains(HYPOTHETICAL_PATTERN, case=False, regex=True)
uninformative = (
    annotation.eq("")
    | annotation.str.contains(r"\bunknown\b", case=False, regex=True)
    | annotation.str.fullmatch(
        r"protein|predicted protein|conserved protein|conserved domain protein|NA|N/A|none|nan",
        case=False)
) & ~hypothetical

# Only clusters with an informative annotation are ranked
enrichment["annotation_status"] = np.select(
    [hypothetical, uninformative], ["Hypothetical", "Uninformative"], default="Annotated")


# 3. Match strain names between the protein table and the metadata

def normalise_strain(name):
    return "_".join(str(name).strip().replace("_", " ").split()).lower()


proteins["strain"] = proteins["strain"].astype(str).str.strip()
metadata["strain"] = metadata["strain"].astype(str).str.strip()
metadata["strain_key"] = metadata["strain"].map(normalise_strain)

protein_strains = pd.Series(sorted(proteins["strain"].unique()))
key_to_strain = dict(zip(protein_strains.map(normalise_strain), protein_strains))

only_metadata = set(metadata["strain_key"]) - set(key_to_strain)
only_proteins = set(key_to_strain) - set(metadata["strain_key"])
if only_metadata or only_proteins:
    raise ValueError(f"Strain names do not match.\nMetadata only: {sorted(only_metadata)}\n"
                     f"Protein file only: {sorted(only_proteins)}")

metadata["protein_strain"] = metadata["strain_key"].map(key_to_strain)
beneficial_strains = metadata.loc[metadata["beneficial"] == 1, "protein_strain"].tolist()
comparison_strains = metadata.loc[metadata["beneficial"] == 0, "protein_strain"].tolist()
n_ben, n_comp = len(beneficial_strains), len(comparison_strains)


# 4. Strain-level prevalence of each cluster

# One row per (cluster, strain): a strain counts once however many proteins it has
presence = proteins.loc[proteins["cluster"] != -1, ["cluster", "strain"]].drop_duplicates()

assert set(presence["cluster"]) == set(enrichment["cluster_id"]), \
    "Protein and enrichment tables have different cluster IDs"

ben_counts = presence[presence["strain"].isin(beneficial_strains)].groupby("cluster").size()
comp_counts = presence[presence["strain"].isin(comparison_strains)].groupby("cluster").size()

enrichment["n_beneficial_strains"] = enrichment["cluster_id"].map(ben_counts).fillna(0).astype(int)
enrichment["n_comparison_strains"] = enrichment["cluster_id"].map(comp_counts).fillna(0).astype(int)

enrichment["prev_ben"] = enrichment["n_beneficial_strains"] / n_ben * 100
enrichment["prev_comp"] = enrichment["n_comparison_strains"] / n_comp * 100
enrichment["prevalence_difference_pp"] = enrichment["prev_ben"] - enrichment["prev_comp"]


# 5. Rank annotated, beneficial-enriched clusters by percentage-point difference

ranked = (
    enrichment[(enrichment["enrichment"] == "beneficial")
               & (enrichment["annotation_status"] == "Annotated")
               & (enrichment["prevalence_difference_pp"] > 0)]
    # Ties: higher beneficial prevalence, then lower comparison prevalence, then cluster ID
    .sort_values(["prevalence_difference_pp", "prev_ben", "prev_comp", "cluster_id"],
                 ascending=[False, False, True, True], kind="mergesort")
    .copy()
)
ranked["rank"] = np.arange(1, len(ranked) + 1)
top = ranked.head(TOP_N)

print(f"Strains: {n_ben} beneficial, {n_comp} comparison")
print(f"Ranked clusters: {len(ranked):,}")
print(top[["rank", "cluster_id", "annotation", "prev_ben", "prev_comp",
           "prevalence_difference_pp"]].to_string(index=False))


# 6. Save outputs
print("Saving outputs...")

enrichment.to_csv(os.path.join(OUTPUT_DIR, "all_clusters_with_prevalence.csv"), index=False)
ranked.to_csv(os.path.join(OUTPUT_DIR, "beneficial_annotated_ranked.csv"), index=False)
top.to_csv(os.path.join(OUTPUT_DIR, "beneficial_top10.csv"), index=False)