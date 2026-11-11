import glob
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from Bio import SeqIO
from transformers import AutoTokenizer, AutoModelForMaskedLM


# >>> PROKKA NUCLEOTIDE FILES (.ffn), ONE FOLDER PER GENOME <<<
FFN_GLOB = "prokka_results/*/*.ffn"

# >>> WHERE OUTPUTS SHOULD BE SAVED <<<
OUTPUT_DIR = "nucleotide_transformer_embeddings"

# Model used
MODEL_NAME = "InstaDeepAI/nucleotide-transformer-2.5b-multi-species"

# Batch size for inference (adjust if you hit GPU memory limits).
BATCH_SIZE = 8


# 1. Setup

output_path = Path(OUTPUT_DIR)
output_path.mkdir(exist_ok=True, parents=True)

ffn_files = sorted(glob.glob(FFN_GLOB))
if not ffn_files:
    raise FileNotFoundError(f"No .ffn files found with pattern: {FFN_GLOB}")

print(f"Loading Nucleotide Transformer ({MODEL_NAME})...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
model = AutoModelForMaskedLM.from_pretrained(MODEL_NAME, trust_remote_code=True)
model.eval()
max_len = tokenizer.model_max_length

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
print(f"   ✓ Using device: {device}")

# Use half precision only on GPU for speed; keep full precision on CPU.
if device.type == "cuda":
    model.half()
else:
    print("   WARNING: Running on CPU - this will be very slow.")
    print("   Consider requesting a GPU node on the cluster.")


# 2. Load sequences

print("Loading genes from .ffn files...")
genes = []  # (gene_id, genome, sequence, description)
for ffn in ffn_files:
    genome = os.path.basename(os.path.dirname(ffn))
    for record in SeqIO.parse(ffn, "fasta"):
        genes.append((f"{genome}_{record.id}", genome,
                      str(record.seq).upper(), record.description))

n_genes = len(genes)
print(f"   ✓ Loaded {n_genes:,} genes from {len(ffn_files)} files")


# 3. Generate embeddings

print("Generating mean embeddings...")
all_embeddings = []
truncated = []

start_time = time.time()

for i in range(0, n_genes, BATCH_SIZE):
    batch = genes[i : i + BATCH_SIZE]
    seqs = [g[2] for g in batch]

    # Record genes longer than the model's maximum token length
    truncated += [len(x) > max_len for x in tokenizer(seqs)["input_ids"]]

    enc = tokenizer(seqs, return_tensors="pt", padding=True,
                    truncation=True, max_length=max_len)
    input_ids = enc["input_ids"].to(device)
    attention_mask = input_ids != tokenizer.pad_token_id

    with torch.no_grad():
        results = model(input_ids,
                        attention_mask=attention_mask,
                        encoder_attention_mask=attention_mask,
                        output_hidden_states=True)
        embeddings = results["hidden_states"][-1].mean(dim=1)

    all_embeddings.append(embeddings.float().cpu().numpy())

    processed = min(i + BATCH_SIZE, n_genes)
    if (i // BATCH_SIZE) % max(1, 5000 // BATCH_SIZE) == 0:
        elapsed = time.time() - start_time
        progress = processed / n_genes
        eta = elapsed / progress - elapsed
        print(f"   {processed:,}/{n_genes:,} - {100 * progress:5.1f}% - ETA: {eta/60:4.0f} min")

embeddings_matrix = np.vstack(all_embeddings)
elapsed = time.time() - start_time

print(f"\n   ✓ Generated {len(embeddings_matrix):,} embeddings")
print(f"   ✓ Time: {elapsed/3600:.2f} hours")
print(f"   ✓ Embedding matrix shape: {embeddings_matrix.shape}")
print(f"   ✓ Truncated genes: {sum(truncated):,}")


# 4. Save outputs

print("Saving outputs...")

np.save(output_path / "gene_embeddings.npy", embeddings_matrix)

# Metadata 
pd.DataFrame({
    "gene_id": [g[0] for g in genes],
    "genome": [g[1] for g in genes],
    "description": [g[3] for g in genes],
    "gene_length_bp": [len(g[2]) for g in genes],
    "truncated": truncated,
}).to_csv(output_path / "gene_metadata.csv", index=False)
