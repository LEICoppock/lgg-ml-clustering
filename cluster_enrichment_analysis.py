import os
import numpy as np
import pandas as pd
from scipy.stats import fisher_exact
from statsmodels.stats.multitest import multipletests


# Clustering output from the ESM-2 clustering script
CLUSTERS_FILE = "esm2_clustering/all_proteins_with_clusters.csv"

# Table with columns: strain, beneficial (1 = beneficial strain, 0 = comparison strain)
# Input different metadata matrix for seperate phenotypic analysis
STRAIN_METADATA_FILE = "strain_metadata.csv"

OUTPUT_DIR = "enrichment_results"

LOG2_FOLD_CHANGE_THRESHOLD = 0.5
Q_VALUE_THRESHOLD = 0.05
PSEUDOCOUNT = 1.0

# Strain-name pattern used to find L. rhamnosus GG
# Adjust to analysed chosen strain
LGG_PATTERN = r"rhamnosus.*GG|GG.*rhamnosus"

# Clusters whose most common annotation contains one of these are "hypothetical"
HYPOTHETICAL_PATTERN = r"\bhypothetical\b|\buncharacteri[sz]ed\b"


# 1. Load clusters and add beneficial / non-beneficial labels
print("Loading clusters and metadata...")

os.makedirs(OUTPUT_DIR, exist_ok=True)


def normalise_name(name):
    """Make strain names comparable between the two tables."""
    return str(name).lower().replace(" ", "_").replace("-", "_")


genes = pd.read_csv(CLUSTERS_FILE)
strain_classes = pd.read_csv(STRAIN_METADATA_FILE)

genes["strain_normalised"] = genes["strain"].map(normalise_name)
strain_classes["strain_normalised"] = strain_classes["strain"].map(normalise_name)

genes = genes.merge(strain_classes[["strain_normalised", "beneficial"]],
                    on="strain_normalised", how="left")

# Genes from strains without a label are excluded
genes = genes.dropna(subset=["beneficial"]).copy()
genes["beneficial"] = genes["beneficial"].astype(int)
genes["annotation"] = genes["annotation"].fillna("hypothetical protein")


# 2. Per-cluster counts (HDBSCAN noise, cluster -1, is excluded)

clustered = genes[genes["cluster"] != -1]

cluster_stats = (
    clustered.groupby("cluster")
    .agg(n_genes=("gene_id", "size"),
         n_beneficial=("beneficial", "sum"),
         n_strains=("strain", "nunique"),
         annotation=("annotation", lambda s: s.mode().iloc[0]))  # most common annotation
    .reset_index()
    .rename(columns={"cluster": "cluster_id"})
)
cluster_stats["n_nonbeneficial"] = cluster_stats["n_genes"] - cluster_stats["n_beneficial"]
cluster_stats["is_hypothetical"] = cluster_stats["annotation"].str.contains(
    HYPOTHETICAL_PATTERN, case=False, regex=True, na=False)


# 3. Enrichment: Fisher's exact test and observed/expected fold change
print("Calculating enrichment...")

total_beneficial = int((clustered["beneficial"] == 1).sum())
total_nonbeneficial = int((clustered["beneficial"] == 0).sum())
total_genes = len(clustered)

rows = []
for _, c in cluster_stats.iterrows():
    n_ben, n_non, n_all = int(c["n_beneficial"]), int(c["n_nonbeneficial"]), int(c["n_genes"])

    # 2 x 2 table: rows = beneficial / non-beneficial, columns = in / outside cluster
    odds_ratio, p_value = fisher_exact(
        [[n_ben, total_beneficial - n_ben],
         [n_non, total_nonbeneficial - n_non]],
        alternative="two-sided")

    # Observed / expected count (with pseudocount), for each group separately
    expected_ben = n_all * total_beneficial / total_genes
    expected_non = n_all * total_nonbeneficial / total_genes
    log2_fc_ben = np.log2((n_ben + PSEUDOCOUNT) / (expected_ben + PSEUDOCOUNT))
    log2_fc_non = np.log2((n_non + PSEUDOCOUNT) / (expected_non + PSEUDOCOUNT))

    rows.append({**c.to_dict(),
                 "prop_beneficial": n_ben / n_all,
                 "fold_change_log2": log2_fc_ben,
                 "nonbeneficial_fold_change_log2": log2_fc_non,
                 "odds_ratio": odds_ratio,
                 "p_value": p_value})

enrichment = pd.DataFrame(rows)

# Benjamini-Hochberg false discovery rate across all clusters
enrichment["q_value"] = multipletests(enrichment["p_value"], method="fdr_bh")[1]

# Classification
enrichment["enrichment"] = "neutral"
significant = enrichment["q_value"] < Q_VALUE_THRESHOLD
enrichment.loc[significant & (enrichment["fold_change_log2"] > LOG2_FOLD_CHANGE_THRESHOLD),
               "enrichment"] = "beneficial"
enrichment.loc[significant & (enrichment["nonbeneficial_fold_change_log2"] > LOG2_FOLD_CHANGE_THRESHOLD),
               "enrichment"] = "non-beneficial"

beneficial_enriched = enrichment[enrichment["enrichment"] == "beneficial"].sort_values(
    "fold_change_log2", ascending=False)
beneficial_hypothetical = beneficial_enriched[beneficial_enriched["is_hypothetical"]]

print(enrichment["enrichment"].value_counts().to_string())


# 4. LGG proteins in beneficial-enriched clusters

lgg = genes[genes["strain"].str.contains(LGG_PATTERN, case=False, regex=True, na=False)]
lgg_in_beneficial = lgg[lgg["cluster"].isin(beneficial_enriched["cluster_id"])]

print(f"LGG strains matched: {sorted(lgg['strain'].unique())}")
print(f"LGG proteins in beneficial-enriched clusters: {len(lgg_in_beneficial):,}")


# 5. Save outputs
print("Saving outputs...")

enrichment.to_csv(os.path.join(OUTPUT_DIR, "cluster_enrichment_analysis.csv"), index=False)
beneficial_enriched.to_csv(os.path.join(OUTPUT_DIR, "beneficial_enriched_clusters.csv"), index=False)
beneficial_hypothetical.to_csv(os.path.join(OUTPUT_DIR, "beneficial_hypothetical_clusters.csv"), index=False)
lgg_in_beneficial.to_csv(os.path.join(OUTPUT_DIR, "lgg_beneficial_enriched_proteins.csv"), index=False)