# LGG-clustering

This Github repository contains scripts for generating ESM-2 protein and Nucleotide Transformer gene embeddings, UMAP and HDBSCAN clustering, and identifying enriched clusters. In this project, the scripts were applied to bacterial proteins/genes to identify clusters enriched in strains classified as beneficial for skin health based on a literature search.

This workflow requires FASTA files from a Prokka output containing the predicted protein/gene sequences for all bacterial genomes analysed (.faa for proteins and .ffn for genes). Also required is a strain metadata .csv file containing strain classifications: Use `beneficial = 1` for strains considered beneficial for the phenotype being studied and `beneficial = 0` for comparison strains. In this project, classifications were based on literature relating to skin health. The file can be adapted for other organisms or phenotypes.



## Workflow

Run the scripts in this order:
`esm2_embedding.py` or `nucleotide_transformer_embeddings.py`
`cluster_embeddings.py`
`cluster_enrichment_analysis.py`
`ranking_strain_prevalence.py`


#### esm2_embedding.py

Edit with own input file paths. This script will generate ESM-2 embeddings for each protein sequence and saves:

- `esm2_embeddings.npy`
- `gene_ids.csv`

The script can run on CPU or GPU. If CUDA is available, the model is moved to the GPU and converted to half precision to reduce memory use.

#### nucleotide_transformer_embeddings.py

Edit with own input file paths. This script will generate Nucleotide Transformer embeddings for each gene sequence and saves:

- `gene_embeddings.npy`
- `gene_metadata.csv`

The script can run on CPU or GPU. If CUDA is available, the model is moved to the GPU and converted to half precision to reduce memory use.

#### cluster_embeddings.py

Edit with your own input file paths. This script uses the ESM-2/Nucleotide Transformer embeddings created previously, applies UMAP dimensionality reduction, and uses HDBSCAN to assign each protein to a cluster.


#### cluster_enrichment_analysis.py

This script compares protein clusters between strains classified as beneficial for skin health, or as comparison-group enriched. It uses Fisher’s exact tests, calculates enrichment fold changes, applies Benjamini–Hochberg FDR correction, and identifies clusters enriched in beneficial strains.

#### ranking_strain_prevalence.py

This script calculates for each cluster the percentage of beneifical and comparison strain prevalance, and ranks the beneificialy enriched clusters in order of the difference in percentages between the two groups.

## Installation

Each stage uses its own environment, because the stages ran on different machines.

| Stage | Machine | Python | Requirements file |
|---|---|---|---|
| Downstream analysis and figures | Laptop (CPU) | 3.9.6 | `requirements.txt` |
| ESM-2 embeddings | HPC GPU | 3.10.22 | `requirements-esm2-embed.txt` |
| Nucleotide Transformer embeddings | HPC GPU | 3.11.18 | `requirements-nt-embed.txt` |
| UMAP and HDBSCAN clustering | HPC GPU | 3.11.18 | `requirements-clustering.txt` |

Standard `venv` and `pip` work for all but clustering ([uv](https://docs.astral.sh/uv/) also works; use `uv venv` and `uv pip install`).

**Analysis (CPU)**
```bash
python3.9 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

**ESM-2 embeddings (GPU, CUDA 12.1)**
```bash
python3.10 -m venv .venv-esm && source .venv-esm/bin/activate
pip install torch==2.5.1 --index-url [https://download.pytorch.org/whl/cu121](https://download.pytorch.org/whl/cu121)
pip install -r requirements-hpc-esm2.txt
```

**Nucleotide Transformer embeddings (GPU, CUDA 12.6)**
```bash
python3.11 -m venv .venv-nt && source .venv-nt/bin/activate
pip install torch==2.14.1 --index-url [https://download.pytorch.org/whl/cu126](https://download.pytorch.org/whl/cu126)
pip install -r requirements-nt-embed.txt
```
Model: `InstaDeepAI/nucleotide-transformer-2.5b-multi-species`

**Clustering (GPU, RAPIDS)**
```bash
conda env create -f environment-clustering.yml
conda activate rapids_env
```


## References

- Seemann T. Prokka: rapid prokaryotic genome annotation. Bioinformatics. 2014;30(14):2068–2069. https://doi.org/10.1093/bioinformatics/btu153
- Lin Z, Akin H, Rao R, et al. Evolutionary-scale prediction of atomic-level protein structure with a language model. Science. 2023;379(6637):1123–1130. https://doi.org/10.1126/science.ade2574
- Dalla-Torre H, Gonzalez L, Mendoza-Revilla J, et al. Nucleotide Transformer: building and evaluating robust foundation models for human genomics. Nat Methods. 2025;22(2):287–297. https://doi.org/10.1038/s41592-024-02523-z
- McInnes L, Healy J, Melville J. UMAP: Uniform Manifold Approximation and Projection for Dimension Reduction. arXiv. 2018. https://doi.org/10.48550/arXiv.1802.03426
- McInnes L, Healy J, Astels S. hdbscan: Hierarchical density-based clustering. Journal of Open Source Software. 2017;2(11):205. https://doi.org/10.21105/joss.00205
