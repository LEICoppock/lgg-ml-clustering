import os
import cupy as cp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from cuml.cluster import HDBSCAN
from cuml.manifold import UMAP
from sklearn.preprocessing import StandardScaler, normalize



MODEL = "esm2"  # adjust here for "esm2" or "nucleotide" embeddings

if MODEL == "esm2":
    EMBEDDINGS_FILE = "esm2_embeddings/esm2_embeddings.npy"
    METADATA_FILE = "esm2_embeddings/gene_ids.csv"
    FASTA_FILE = "all_yes_no_proteins.faa"
else:
    EMBEDDINGS_FILE = "nucleotide_transformer_embeddings/gene_embeddings.npy"
    METADATA_FILE = "nucleotide_transformer_embeddings/gene_metadata.csv"

OUTPUT_DIR = f"{MODEL}_clustering"

OUTPUT_NAME = "all_proteins_with_clusters.csv" if MODEL == "esm2" else "all_genes_with_clusters.csv"

RANDOM_SEED = 42
CLIP_LIMIT = 500
UMAP_N_NEIGHBORS = 50
UMAP_MIN_DIST = 0.0
UMAP_CLUSTERING_DIMENSIONS = 50
MIN_CLUSTER_SIZE = 10
MIN_SAMPLES = 10


# 1. Load embeddings and metadata
print("Loading embeddings and metadata...")

os.makedirs(OUTPUT_DIR, exist_ok=True)
embeddings = np.load(EMBEDDINGS_FILE).astype(np.float32)

if MODEL == "esm2":
    from Bio import SeqIO

    genes = pd.read_csv(METADATA_FILE)  # column: gene_id
    annotations = {}
    for record in SeqIO.parse(FASTA_FILE, "fasta"):
        text = " ".join(record.description.split(" ")[3:]).strip()
        annotations[record.id] = text or "hypothetical protein"
    genes["annotation"] = genes["gene_id"].map(annotations)
    genes["strain"] = (genes["gene_id"].str.split("_")
                       .map(lambda p: "_".join(p[:-2]))
                       .str.replace(" ", "_", regex=False))
else:
    genes = pd.read_csv(METADATA_FILE)  # columns: gene, genome, description
    genes = genes.rename(columns={"gene": "gene_id", "genome": "strain",
                                  "description": "annotation"})

assert embeddings.shape[0] == len(genes), "Embedding rows and metadata rows differ"


# 2. Preprocess: clip, z-score, L2 normalise
print("Preprocessing embeddings...")

embeddings = np.clip(embeddings, -CLIP_LIMIT, CLIP_LIMIT)
embeddings = StandardScaler().fit_transform(embeddings)
embeddings = normalize(embeddings, norm="l2", axis=1)
embeddings_gpu = cp.asarray(embeddings, dtype=cp.float32)


# 3. UMAP (50D for clustering, 2D for plotting)
print("UMAP dimension reduction...")

umap_settings = dict(n_neighbors=UMAP_N_NEIGHBORS, min_dist=UMAP_MIN_DIST,
                     metric="cosine", random_state=RANDOM_SEED, output_type="cupy")

coords_50d = UMAP(n_components=UMAP_CLUSTERING_DIMENSIONS, **umap_settings).fit_transform(embeddings_gpu)
coords_2d = cp.asnumpy(UMAP(n_components=2, **umap_settings).fit_transform(embeddings_gpu))

genes["umap_1"], genes["umap_2"] = coords_2d[:, 0], coords_2d[:, 1]


# 4. HDBSCAN
print("HDBSCAN clustering...")

clusterer = HDBSCAN(min_cluster_size=MIN_CLUSTER_SIZE, min_samples=MIN_SAMPLES,
                    metric="euclidean", cluster_selection_method="eom",
                    allow_single_cluster=False, output_type="cupy")
genes["cluster"] = cp.asnumpy(clusterer.fit_predict(coords_50d)).astype(int)

n_clusters = genes.loc[genes["cluster"] != -1, "cluster"].nunique()
n_noise = int((genes["cluster"] == -1).sum())
print(f"Clusters: {n_clusters:,} | noise: {n_noise:,} ({100 * n_noise / len(genes):.2f}%)")


# 5. Save outputs
print("Saving outputs...")
print("all_proteins_with_clusters.csv/all_genes_with_clusters.csv")

genes.to_csv(os.path.join(OUTPUT_DIR, OUTPUT_NAME), index=False)

(genes.groupby("cluster", as_index=False)
      .agg(n_genes=("gene_id", "size"), n_strains=("strain", "nunique"))
      .to_csv(os.path.join(OUTPUT_DIR, "cluster_statistics.csv"), index=False))

np.save(os.path.join(OUTPUT_DIR, "umap_2d_coordinates.npy"), coords_2d)


# 6. Plot UMAP
print("Plotting UMAP 2D projection...")

noise = genes["cluster"] == -1
fig, ax = plt.subplots(figsize=(12, 9))
ax.scatter(genes.loc[noise, "umap_1"], genes.loc[noise, "umap_2"],
           color="#D0D0D0", s=1, alpha=0.4, linewidths=0, rasterized=True)
ax.scatter(genes.loc[~noise, "umap_1"], genes.loc[~noise, "umap_2"],
           c=genes.loc[~noise, "cluster"].to_numpy() % 20, cmap="tab20",
           vmin=-0.5, vmax=19.5, s=1, alpha=0.6, linewidths=0, rasterized=True)
ax.set(xlabel="UMAP 1", ylabel="UMAP 2",
       title=f"{genes['strain'].nunique():,} strains; {len(genes):,} genes/proteins; {n_clusters:,} clusters")
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, "umap_clusters.png"), dpi=300, bbox_inches="tight")
