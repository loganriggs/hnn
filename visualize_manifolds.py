"""
Visualize manifolds in bilinear transcoder features

Adapted from https://github.com/tdooms/bae (credit: Thomas Dooms)
This script:
1. Finds clusters of hidden dimensions with high cosine similarity
2. Constructs quadratic forms from clustered features
3. Visualizes data points projected onto manifolds
"""

#%% Imports
import torch
import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset, Dataset
from utils.bilinear import Bilinear
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd

#%% Configuration
LAYER_IDX = 2  # Which layer's transcoder to analyze
MODEL_NAME = "EleutherAI/pythia-410m"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# N_BATCHES = 256  # How many batches to sample for manifold
N_BATCHES = 128  # How many batches to sample for manifold
BATCH_SIZE = 32  # Batch size for processing
MAX_LENGTH = 128  # Maximum sequence length

print(f"Using device: {DEVICE}")
print(f"Analyzing layer {LAYER_IDX}")

#%% Load model and tokenizer
print("\nLoading model and tokenizer...")
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token
model.to(DEVICE)
model.eval()

print(f"Model loaded: {MODEL_NAME}")

#%% Load transcoder
from huggingface_hub import hf_hub_download

# Download from HuggingFace
repo_id = "Elriggs/hnn_transcoders"
filename = f"layer_{LAYER_IDX}/transcoder_weights_l{LAYER_IDX}_bilinear_muon_3000b.pt"
print(f"\nDownloading transcoder from HuggingFace: {repo_id}/{filename}")

checkpoint_path = hf_hub_download(repo_id=repo_id, filename=filename)
print(f"Downloaded to: {checkpoint_path}")

transcoder, checkpoint = Bilinear.from_pretrained(checkpoint_path, device=DEVICE)
transcoder.eval()
transcoder.requires_grad_(False)

config = checkpoint['config']
print(f"Transcoder loaded: {config.n_inputs} → {config.n_hidden} → {config.n_outputs}")

#%% Find clusters of similar features
print("\nFinding feature clusters...")
TAU = 0.6  # Cosine similarity threshold for clustering

# Get weight matrices
W = transcoder.head.weight  # [n_hidden, n_outputs] (down projection)
U = W / (W.norm(dim=1, keepdim=True) + 1e-12)  # Normalize to unit vectors

# Align all vectors to point in same general direction (for cleaner clustering)
ref = U.mean(0, keepdim=True)  # [1, n_outputs]
scores = U @ ref.T  # [n_hidden, 1]
U = torch.where(scores < 0, -U, U)  # Flip negatively aligned vectors

# Compute absolute cosine similarity matrix
S = (U @ U.T).abs().cpu().numpy()  # [n_hidden, n_hidden]

# Build adjacency matrix based on threshold
A = sp.csr_matrix(S >= TAU)
n_comp, labels = connected_components(A, directed=False)

print(f"Found {n_comp} clusters with tau={TAU}")

# Get cluster sizes
sizes = np.bincount(labels)
sorted_indices = np.argsort(sizes)[::-1]
top_10_sizes = sizes[sorted_indices[:10]]
top_10_labels = sorted_indices[:10]

print("\nTop 10 cluster sizes:")
for i, (label, size) in enumerate(zip(top_10_labels, top_10_sizes)):
    cluster_members = (labels == label).nonzero()[0]
    print(f"  {i+1}. Cluster {label}: {size} features (indices: {cluster_members[:10].tolist()}...)")

#%% Visualize cluster cosine similarity
CLUSTER_RANK = 0  # Which cluster to visualize (0=largest, 1=second largest, etc.)

print(f"\nVisualizing cluster rank {CLUSTER_RANK} (0=largest)...")

# Select cluster based on rank
if CLUSTER_RANK >= len(top_10_labels):
    print(f"Warning: CLUSTER_RANK={CLUSTER_RANK} is too high, only {len(top_10_labels)} clusters available")
    print(f"Using largest cluster instead")
    CLUSTER_RANK = 1

cluster_label = top_10_labels[CLUSTER_RANK]
cluster_indices = (labels == cluster_label).nonzero()[0]

print(f"Cluster {cluster_label} (rank {CLUSTER_RANK}): {len(cluster_indices)} features")
print(f"Feature indices: {cluster_indices.tolist()}")

# Plot cosine similarity heatmap for this cluster
cluster_sim = S[np.ix_(cluster_indices, cluster_indices)]

plt.figure(figsize=(10, 8))
plt.imshow(cluster_sim, cmap='viridis', aspect='auto')
plt.colorbar(label='|Cosine Similarity|')
# make color bar range from 0 to 1
plt.clim(0, 1)  # Set colorbar range from 0 to 1

# Add text of color bar value to each square in the heatmap
for i in range(len(cluster_indices)):
    for j in range(len(cluster_indices)):
        val = cluster_sim[i, j]
        plt.text(
            j, i, f'{val:.2f}',
            ha='center', va='center',
            color='white' if val < 0.5 else 'black',
            fontsize=20
        )
# Set integer ticks for x and y axes
plt.xticks(
    ticks=np.arange(0, len(cluster_indices)),
    labels=[str(i) for i in range(len(cluster_indices))],
    fontsize=20
)
plt.yticks(
    ticks=np.arange(0, len(cluster_indices)),
    labels=[str(i) for i in range(len(cluster_indices))],
    fontsize=20
)

# color bar got an unexpected keyword "vmin"

# # add text to of color bar value to each square in the heatmap
# for i in range(len(cluster_indices)):
#     for j in range(len(cluster_indices)):
#         plt.text(j, i, f'{cluster_sim[i, j]:.2f}', ha='center', va='center', color='white')
plt.title(f'Cluster {cluster_label} Cosine Similarity\n{len(cluster_indices)} features')

plt.xlabel('Feature index (within cluster)')
plt.ylabel('Feature index (within cluster)')
plt.tight_layout()
plt.savefig(f'figures/cluster_{cluster_label}_similarity_l{LAYER_IDX}.png', dpi=150, bbox_inches='tight')
print(f"Saved to: figures/cluster_{cluster_label}_similarity_l{LAYER_IDX}.png")
plt.show()

#%% Construct quadratic form for selected cluster
print(f"\nConstructing quadratic form for cluster {cluster_label}...")

# Get the features in this cluster
cluster_features = torch.tensor(cluster_indices, device=DEVICE)

# Construct the quadratic form: sum of outer products of left and right vectors
# For each feature i: left[i] ⊗ right[i]
# The manifold is described by: x^T M x where M = sum_i (left[i] ⊗ right[i])

left_vecs = transcoder.left.weight[cluster_features]  # [n_cluster, n_inputs]
right_vecs = transcoder.right.weight[cluster_features]  # [n_cluster, n_inputs]

# Compute quadratic form: M = sum_i (left[i] ⊗ right[i]^T)
# This is equivalent to: M = L^T @ R where L and R are stacked vectors
density = torch.einsum('ci,cj->ij', left_vecs, right_vecs)  # [n_inputs, n_inputs]

# Symmetrize
density = 0.5 * (density + density.T)

print(f"Quadratic form shape: {density.shape}")

# Eigendecomposition to understand dimensionality
eigvals, eigvecs = torch.linalg.eigh(density.float())
eigvals_sorted, sort_idx = torch.sort(eigvals.abs(), descending=True)

print(f"\nTop 10 eigenvalues (by magnitude):")
for i in range(min(10, len(eigvals_sorted))):
    print(f"  {i+1}. {eigvals_sorted[i].item():.6f}")

# Plot eigenvalue spectrum
plt.figure(figsize=(10, 4))
threshold = 1e-3
filtered_eigvals = eigvals[eigvals.abs() > threshold]
plt.plot(filtered_eigvals.cpu().numpy(), marker='o', linestyle='-', markersize=3)
plt.xlabel('Eigenvalue index')
plt.ylabel('Eigenvalue')
plt.title(f'Eigenvalue Spectrum (|λ| > {threshold})\n{len(filtered_eigvals)} non-zero eigenvalues')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f'figures/cluster_{cluster_label}_spectrum_l{LAYER_IDX}.png', dpi=150, bbox_inches='tight')
print(f"Saved to: figures/cluster_{cluster_label}_spectrum_l{LAYER_IDX}.png")
plt.show()

#%% Load dataset for manifold sampling
print("\nLoading dataset for manifold sampling...")
# Load Pile dataset (same as training)
dataset = load_dataset("monology/pile-uncopyrighted", split="train", streaming=True)

# Take a subset
samples = []
for i, example in enumerate(dataset):
    samples.append(example)
    if i >= N_BATCHES * BATCH_SIZE - 1:
        break

dataset = Dataset.from_list(samples)
print(f"Loaded {len(dataset)} samples")

#%% Sample data points and project onto manifold
print("\nSampling data and computing manifold projections...")

# Get top 3 eigenvectors for visualization
top_k_eigvals = 3
top_k_indices = eigvals.abs().topk(k=top_k_eigvals).indices
top_eigvecs = eigvecs[:, top_k_indices]  # [n_inputs, 3]

# Storage for projected points
projected_points = []  # Will store [n_points, 3]
quadratic_values = []  # Will store [n_points] - the value of x^T M x
input_ids_list = []
next_token_preds = []

# Hook to capture MLP input
mlp_inputs = []
def get_mlp_input_hook(module, input, output):
    mlp_inputs.append(output.detach().cpu())

# Register hook
target_layer = model.gpt_neox.layers[LAYER_IDX]
hook = target_layer.post_attention_layernorm.register_forward_hook(get_mlp_input_hook)

# Process batches
for batch_idx in tqdm(range(N_BATCHES), desc="Processing batches"):
    batch_start = batch_idx * BATCH_SIZE
    batch_end = min((batch_idx + 1) * BATCH_SIZE, len(dataset))
    batch_texts = [dataset[i]['text'] for i in range(batch_start, batch_end)]

    # Tokenize
    inputs = tokenizer(
        batch_texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=MAX_LENGTH
    )
    input_ids = inputs['input_ids'].to(DEVICE)
    attention_mask = inputs['attention_mask'].to(DEVICE)

    # Get MLP input and model predictions
    mlp_inputs.clear()
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        next_tokens = logits.argmax(dim=-1)

    mlp_input = mlp_inputs[0]  # [batch, seq, d_model]

    # Normalize activations (L2 norm)
    mlp_input_norm = mlp_input / (mlp_input.pow(2).sum(-1, keepdim=True).sqrt() + 1e-8)

    # Project onto top eigenvectors: [batch, seq, d_model] @ [d_model, 3] -> [batch, seq, 3]
    projected = torch.einsum('bsd,dk->bsk', mlp_input_norm, top_eigvecs.cpu())

    # Compute quadratic form values: x^T M x
    quad_vals = torch.einsum('bsd,de,bse->bs', mlp_input_norm, density.cpu(), mlp_input_norm)

    # Flatten and store
    projected_points.append(projected.reshape(-1, 3))
    quadratic_values.append(quad_vals.reshape(-1))
    input_ids_list.append(input_ids.cpu().reshape(-1))
    next_token_preds.append(next_tokens.cpu().reshape(-1))

# Remove hook
hook.remove()

# Concatenate all results
projected_points = torch.cat(projected_points, dim=0)  # [n_points, 3]
quadratic_values = torch.cat(quadratic_values, dim=0)  # [n_points]
input_ids_all = torch.cat(input_ids_list, dim=0)  # [n_points]
next_token_preds = torch.cat(next_token_preds, dim=0)  # [n_points]

print(f"Sampled {len(projected_points)} data points")

#%% Select top-k points by quadratic value and create dataframe
K_VIS = 250_000  # Number of points to visualize

# Get top-k by absolute quadratic value
top_k_vals, top_k_idx = quadratic_values.abs().topk(k=min(K_VIS, len(quadratic_values)))

# Create dataframe for visualization
inp_tokens = tokenizer.convert_ids_to_tokens(input_ids_all[top_k_idx].cpu())
out_tokens = tokenizer.convert_ids_to_tokens(next_token_preds[top_k_idx].cpu())
token_strings = [
    (i + ' -> ' + o).replace('Ġ', ' ').replace('Ċ', ' ')
    for i, o in zip(inp_tokens, out_tokens)
]

df = pd.DataFrame({
    'x': projected_points[top_k_idx, 0].cpu().numpy(),
    'y': projected_points[top_k_idx, 1].cpu().numpy(),
    'z': projected_points[top_k_idx, 2].cpu().numpy(),
    'value': quadratic_values[top_k_idx].cpu().numpy(),
    'token': token_strings
})

print(f"\nDataframe created with {len(df)} points")
print(df.head())

#%% 3D visualization (optional - requires plotly)
import plotly.express as px

print("\nCreating 3D interactive visualization...")

fig = px.scatter_3d(
    df,
    x='x',
    y='y',
    z='z',
    color='value',
    hover_data={'token': True, 'x': False, 'y': False, 'z': False, 'value': ':.4f'},
    color_continuous_scale='RdBu_r',
    color_continuous_midpoint=0.0,
    title=f'3D Manifold - Cluster {cluster_label} (Layer {LAYER_IDX})',
    height=800,
    width=800
)

fig.update_layout(
    scene=dict(
        xaxis_title=f'Eigenvec {top_k_indices[0].item()}',
        yaxis_title=f'Eigenvec {top_k_indices[1].item()}',
        zaxis_title=f'Eigenvec {top_k_indices[2].item()}'
    )
)

fig.write_html(f'figures/manifold_3d_cluster_{cluster_label}_l{LAYER_IDX}.html')
print(f"Saved to: figures/manifold_3d_cluster_{cluster_label}_l{LAYER_IDX}.html")
fig.show()


#%% Summary statistics
print("\n" + "="*80)
print("SUMMARY")
print("="*80)
print(f"Layer: {LAYER_IDX}")
print(f"Transcoder: {config.n_inputs} → {config.n_hidden} → {config.n_outputs}")
print(f"Largest cluster: {cluster_label}")
print(f"  Features in cluster: {len(cluster_indices)}")
print(f"  Mean |cos sim|: {cluster_sim.mean():.3f}")
print(f"  Non-zero eigenvalues: {len(filtered_eigvals)}")
print(f"  Top eigenvalue: {eigvals_sorted[0].item():.6f}")
print(f"  Effective dimensionality: ~{(eigvals.abs() > 0.01).sum().item()}")
print(f"\nVisualized {len(df)} data points")
print(f"Quadratic value range: [{df['value'].min():.3f}, {df['value'].max():.3f}]")
print("="*80)

#%%
