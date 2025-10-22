"""
Interactive notebook-style script to explore transcoder behavior

This script loads a trained transcoder and compares its output with the actual MLP.
Use with Jupyter, VSCode interactive mode, or similar.
"""

#%% Imports
import torch
import numpy as np
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM, AutoTokenizer
from utils.bilinear import Bilinear
import os

#%% Configuration
LAYER_IDX = 23  # Which layer's transcoder to load
MODEL_NAME = "EleutherAI/pythia-410m"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Using device: {DEVICE}")
print(f"Loading transcoder for layer {LAYER_IDX}")

#%% Load base model and tokenizer
print("\nLoading Pythia-410m model...")
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token  # Set pad token
model.to(DEVICE)
model.eval()

print(f"Model loaded: {MODEL_NAME}")
print(f"Number of layers: {len(model.gpt_neox.layers)}")

#%% Actually load all transcoders and do stats on them
from bae_utils.functions import generalized_effective_dimension, effective_dimension
each_layers_dims_above_threshold = []
THRESHOLD = 1.5
for layer_idx in range(len(model.gpt_neox.layers)):
    checkpoint_path = f"model_weights/transcoder_weights_l{layer_idx}_bilinear_muon_3000b.pt"


    transcoder, checkpoint = Bilinear.from_pretrained(checkpoint_path, device=DEVICE)
    transcoder.eval()
    transcoder.requires_grad_(False)
    d = transcoder.head.weight / transcoder.head.weight.norm(dim=0, keepdim=True)
    g = d.T @ d
    gpr = generalized_effective_dimension(g)
    num_top_dims_above_threshold = (gpr > THRESHOLD).sum().item()
    each_layers_dims_above_threshold.append(num_top_dims_above_threshold)

plt.plot(each_layers_dims_above_threshold)
plt.xlabel('Layer')
plt.title(f'Number of effective dimensions > {THRESHOLD:.2f}')
plt.ylabel('Number of dimensions above threshold for each layer')
plt.show()

#%% Plot the number of dimensions above threshold for each layer

#%% Load transcoder checkpoint
checkpoint_path = f"model_weights/transcoder_weights_l{LAYER_IDX}_bilinear_muon_3000b.pt"

print(f"\nLoading transcoder from: {checkpoint_path}")

# Use from_pretrained method which handles everything
transcoder, checkpoint = Bilinear.from_pretrained(checkpoint_path, device=DEVICE)
transcoder.eval()

config = checkpoint['config']

print(f"Transcoder loaded successfully!")
print(f"  Input dim: {config.n_inputs}")
print(f"  Hidden dim: {config.n_hidden}")
print(f"  Output dim: {config.n_outputs}")
print(f"  Final FVU: {checkpoint['fvu_values'][-1]:.4f}")
print(f"  Final Variance Explained: {checkpoint['variance_explained'][-1]:.3f}")

#%%
# turn off grad
transcoder.requires_grad_(False)
# Let's look at this transcoder
left = transcoder.left.weight
right = transcoder.right.weight
head = transcoder.head.weight
for row in range(10):
    left_row = left[row]
    right_row = right[row]
    head_col = head[:, row]

    # Now let's get the outer product of left_row and right_row
    outer_product = torch.outer(left_row, right_row)
    outer_product.shape

    N = 100
    # Now let's plot the weights for this row
    plt.imshow(outer_product.cpu().numpy()[:N, :N], cmap='viridis')
    plt.title(f'Row {row}')
    plt.colorbar()
    plt.show()
#%%
plt.hist(left_row.cpu().numpy(), bins=100, label='Left row', alpha=0.7)
plt.hist(right_row.cpu().numpy(), bins=100, label='Right row', alpha=0.7)
plt.legend()
plt.show()
plt.plot(left_row.cpu().numpy(), label='Left row', alpha=0.7)
plt.plot(right_row.cpu().numpy(), label='Right row', alpha=0.7)
plt.legend()
plt.show()

#%%
# Let's look at the down weights which have high cos-sim
from bae_utils.functions import generalized_effective_dimension, effective_dimension
d = transcoder.head.weight / transcoder.head.weight.norm(dim=0, keepdim=True)
g = d.T @ d
gpr = generalized_effective_dimension(g)
num_top_dims_above_threshold = (gpr > 1.2).sum().item()
plt.scatter(range(len(gpr.cpu().numpy())), gpr.cpu().numpy(), alpha=0.7)
plt.xlabel('Dimension')
plt.ylabel('Generalized Effective Dimension')
plt.title(f'Generalized Effective Dimension for layer {LAYER_IDX}')
plt.show()

top_ind, top_vals  = gpr.topk(10)
num_top_dims_above_threshold = (gpr > 1.2).sum().item()
print(f"Number of top dimensions above threshold: {num_top_dims_above_threshold}")
print(top_ind)
print(top_vals)
#%%
# surely we can do some sort of clustering on the gram matrix?
W = transcoder.head.weight                     # [N,D], rows = directions
U = W / (W.norm(dim=1, keepdim=True) + 1e-12)  # unit rows

ref = U.mean(0, keepdim=True)                  # [1,D]
scores = U @ ref.T                             # [N,1]
U = torch.where(scores < 0, -U, U)             # broadcast OK: (N,1) -> (N,D)

S = (U @ U.T).abs().cpu().numpy()              # |cosine| Gram
  # |cosine| similarity

import numpy as np, scipy.sparse as sp
from scipy.sparse.csgraph import connected_components

tau = 0.6                           # pick your “high” cos threshold
A = sp.csr_matrix(S >= tau)         # adjacency on |cos|≥tau
n_comp, labels = connected_components(A, directed=False)
# cluster sizes and representatives
sizes = np.bincount(labels)
sorted_indices = np.argsort(sizes)[::-1]
top_sizes = sizes[sorted_indices[:10]]
top_labels = labels[sorted_indices[:10]]
print(top_sizes)
print(top_labels)
cluster_of_interest = (labels == top_labels[0]).nonzero()[0]
print(cluster_of_interest)

#%%
# plot the cluster of interest
plt.imshow(S[cluster_of_interest, :][:, cluster_of_interest], cmap='viridis')
plt.colorbar()
plt.show()
#%% Load some sample data
from datasets import load_dataset

print("\nLoading sample data from Pile...")
dataset = load_dataset("monology/pile-uncopyrighted", split="train", streaming=True)

# Get a few samples
samples = []
for i, example in enumerate(dataset):
    samples.append(example['text'])
    if i >= 63:  # Get 5 samples
        break

print(f"Loaded {len(samples)} text samples")
for i, sample in enumerate(samples):
    preview = sample[:100].replace('\n', ' ')
    print(f"  Sample {i}: {preview}...")

#%% Tokenize samples
print("\nTokenizing samples...")
max_length = 128
inputs = tokenizer(samples, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
input_ids = inputs['input_ids'].to(DEVICE)
attention_mask = inputs['attention_mask'].to(DEVICE)

print(f"Input shape: {input_ids.shape}")

#%% Extract MLP input and output activations
print(f"\nExtracting activations from layer {LAYER_IDX}...")

# Storage for activations
mlp_inputs = []
mlp_outputs = []

def get_mlp_input_hook(module, input, output):
    """Hook to capture input to MLP (after layer norm)"""
    mlp_inputs.append(output.detach().cpu())

def get_mlp_output_hook(module, input, output):
    """Hook to capture output from MLP"""
    mlp_outputs.append(output.detach().cpu())

# Register hooks on the target layer
target_layer = model.gpt_neox.layers[LAYER_IDX]
hook1 = target_layer.post_attention_layernorm.register_forward_hook(get_mlp_input_hook)
hook2 = target_layer.mlp.register_forward_hook(get_mlp_output_hook)

# Forward pass
with torch.no_grad():
    outputs = model(input_ids, attention_mask=attention_mask)

# Remove hooks
hook1.remove()
hook2.remove()

# Get activations
mlp_input = mlp_inputs[0]  # [batch_size, seq_len, d_model]
mlp_output = mlp_outputs[0]  # [batch_size, seq_len, d_model]

print(f"MLP input shape: {mlp_input.shape}")
print(f"MLP output shape: {mlp_output.shape}")

#%% Normalize activations (same as during training)
print("\nNormalizing activations...")

# Flatten to (batch*seq, d_model) for per-dimension normalization
mlp_input_flat = mlp_input.reshape(-1, mlp_input.shape[-1])
mlp_output_flat = mlp_output.reshape(-1, mlp_output.shape[-1])

# Z-score normalization per-dimension (same as training)
# mean(0) computes mean across batch*seq dimension for each feature separately
mlp_input_mean = mlp_input_flat.mean(0, keepdim=True)
mlp_input_std = mlp_input_flat.std(0, keepdim=True)
mlp_input_normalized = (mlp_input_flat - mlp_input_mean) / (mlp_input_std + 1e-6)

mlp_output_mean = mlp_output_flat.mean(0, keepdim=True)
mlp_output_std = mlp_output_flat.std(0, keepdim=True)
mlp_output_normalized = (mlp_output_flat - mlp_output_mean) / (mlp_output_std + 1e-6)

# Reshape back to (batch, seq, d_model)
mlp_input_normalized = mlp_input_normalized.reshape(mlp_input.shape)
mlp_output_normalized = mlp_output_normalized.reshape(mlp_output.shape)

print(f"MLP input - mean: {mlp_input_mean.mean():.4f}, std: {mlp_input_std.mean():.4f}")
print(f"MLP output - mean: {mlp_output_mean.mean():.4f}, std: {mlp_output_std.mean():.4f}")

#%% Run transcoder on normalized MLP input
print("\nRunning transcoder on normalized input...")

with torch.no_grad():
    transcoded_output_normalized = transcoder(mlp_input_normalized.to(DEVICE)).cpu()
# hidden_acts = self.left(x) * self.right(x)
hidden_acts = transcoder.left(mlp_input_normalized.to(DEVICE)) * transcoder.right(mlp_input_normalized.to(DEVICE))

print(f"Transcoded output shape: {transcoded_output_normalized.shape}")

#%%

plt.hist(hidden_acts[:, :, 0].flatten().cpu().numpy(), bins=100, label='Hidden 0', alpha=0.7)
plt.hist(hidden_acts[:, :, 1].flatten().cpu().numpy(), bins=100, label='Hidden 1', alpha=0.7)
plt.yscale('log')
plt.legend()
plt.show()

#%% Compare transcoder output with actual MLP output (normalized)
print("\nComparing transcoder with actual MLP (on normalized data)...")

# Compute error on normalized data
error = mlp_output_normalized - transcoded_output_normalized
mse = (error ** 2).mean().item()
mae = error.abs().mean().item()

# Compute variance explained
total_variance = mlp_output_normalized.var().item()
unexplained_variance = error.var().item()
variance_explained = 1 - (unexplained_variance / total_variance)
fvu = unexplained_variance / total_variance

print(f"  MSE: {mse:.6f}")
print(f"  MAE: {mae:.6f}")
print(f"  Variance Explained: {variance_explained:.3f} ({variance_explained*100:.1f}%)")
print(f"  FVU: {fvu:.4f}")

#%% Visualize comparison for first sample
print("\nVisualizing activations for first sample...")

sample_idx = 0
token_idx = 10  # Look at token 10

# Get normalized activations for specific token
mlp_in = mlp_input_normalized[sample_idx, token_idx].numpy()
mlp_out = mlp_output_normalized[sample_idx, token_idx].numpy()
transcoded_out = transcoded_output_normalized[sample_idx, token_idx].numpy()

# Create visualization
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: MLP output vs Transcoded output (scatter)
axes[0, 0].scatter(mlp_out, transcoded_out, alpha=0.3, s=1)
axes[0, 0].plot([mlp_out.min(), mlp_out.max()], [mlp_out.min(), mlp_out.max()], 'r--', linewidth=2)
axes[0, 0].set_xlabel('Actual MLP Output', fontsize=12)
axes[0, 0].set_ylabel('Transcoder Output', fontsize=12)
axes[0, 0].set_title(f'Layer {LAYER_IDX}: Transcoder vs MLP (Sample {sample_idx}, Token {token_idx})', fontsize=14)
axes[0, 0].grid(True, alpha=0.3)

# Plot 2: Error distribution
error_token = (mlp_out - transcoded_out)
axes[0, 1].hist(error_token, bins=50, alpha=0.7, color='orange', edgecolor='black')
axes[0, 1].axvline(x=0, color='red', linestyle='--', linewidth=2)
axes[0, 1].set_xlabel('Error (MLP - Transcoder)', fontsize=12)
axes[0, 1].set_ylabel('Count', fontsize=12)
axes[0, 1].set_title(f'Error Distribution (MAE: {np.abs(error_token).mean():.4f})', fontsize=14)
axes[0, 1].grid(True, alpha=0.3)

# Plot 3: Output values comparison (line plot)
indices = np.arange(len(mlp_out))
axes[1, 0].plot(indices, mlp_out, label='MLP', linewidth=2, alpha=0.7)
axes[1, 0].plot(indices, transcoded_out, label='Transcoder', linewidth=2, alpha=0.7)
axes[1, 0].set_xlabel('Dimension', fontsize=12)
axes[1, 0].set_ylabel('Activation Value', fontsize=12)
axes[1, 0].set_title('Output Activations by Dimension', fontsize=14)
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# Plot 4: Absolute error by dimension
abs_error = np.abs(error_token)
axes[1, 1].bar(indices, abs_error, alpha=0.7, color='red')
axes[1, 1].set_xlabel('Dimension', fontsize=12)
axes[1, 1].set_ylabel('Absolute Error', fontsize=12)
axes[1, 1].set_title('Absolute Error by Dimension', fontsize=14)
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f'figures/transcoder_comparison_l{LAYER_IDX}_sample{sample_idx}.png', dpi=150, bbox_inches='tight')
print(f"Saved visualization to: figures/transcoder_comparison_l{LAYER_IDX}_sample{sample_idx}.png")
plt.show()

#%% Analyze error across all samples and tokens
print("\nAnalyzing error across all samples...")

# Flatten all normalized activations
mlp_output_flat = mlp_output_normalized.reshape(-1, mlp_output_normalized.shape[-1])
transcoded_output_flat = transcoded_output_normalized.reshape(-1, transcoded_output_normalized.shape[-1])

# Compute per-dimension statistics
dim_mse = ((mlp_output_flat - transcoded_output_flat) ** 2).mean(dim=0).numpy()
dim_mae = (mlp_output_flat - transcoded_output_flat).abs().mean(dim=0).numpy()

# Find most/least accurate dimensions
top_5_accurate = np.argsort(dim_mse)[:5]
top_5_inaccurate = np.argsort(dim_mse)[-5:]

print("\nMost accurate dimensions (lowest MSE):")
for i, dim in enumerate(top_5_accurate):
    print(f"  {i+1}. Dimension {dim}: MSE = {dim_mse[dim]:.6f}")

print("\nLeast accurate dimensions (highest MSE):")
for i, dim in enumerate(top_5_inaccurate):
    print(f"  {i+1}. Dimension {dim}: MSE = {dim_mse[dim]:.6f}")

#%% Compare with random baseline
print("\nComparing with random baseline...")

# Random baseline: random noise with same statistics as normalized MLP output
random_output = torch.randn_like(mlp_output_normalized) * mlp_output_normalized.std() + mlp_output_normalized.mean()
random_error = mlp_output_normalized - random_output
random_mse = (random_error ** 2).mean().item()
random_variance_explained = 1 - (random_error.var().item() / total_variance)

print(f"  Random baseline MSE: {random_mse:.6f}")
print(f"  Random baseline VE: {random_variance_explained:.3f}")
print(f"\nTranscoder is {random_mse/mse:.1f}x better than random!")

#%% Examine specific tokens
print("\nExamining specific tokens from first sample...")

sample_idx = 0
token_ids = input_ids[sample_idx, :20].cpu().numpy()  # First 20 tokens
tokens = [tokenizer.decode([tid]) for tid in token_ids]

# Compute error per token (using normalized data)
token_errors = []
for token_idx in range(20):
    mlp_out_token = mlp_output_normalized[sample_idx, token_idx]
    transcoded_out_token = transcoded_output_normalized[sample_idx, token_idx]
    token_error = ((mlp_out_token - transcoded_out_token) ** 2).mean().item()
    token_errors.append(token_error)

print("\nToken-level MSE:")
for i, (token, error) in enumerate(zip(tokens[:20], token_errors)):
    print(f"  Token {i:2d}: '{token:15s}' MSE = {error:.6f}")

#%% Summary statistics
print("\n" + "="*60)
print("SUMMARY")
print("="*60)
print(f"Layer: {LAYER_IDX}")
print(f"Transcoder architecture: Bilinear ({config.n_inputs} → {config.n_hidden} → {config.n_outputs})")
print(f"Number of samples: {len(samples)}")
print(f"Sequence length: {input_ids.shape[1]}")
print(f"\nPerformance (on normalized data):")
print(f"  MSE: {mse:.6f}")
print(f"  MAE: {mae:.6f}")
print(f"  Variance Explained: {variance_explained:.3f} ({variance_explained*100:.1f}%)")
print(f"  FVU: {fvu:.4f}")
print(f"\nCheckpoint performance (training):")
print(f"  Final VE: {checkpoint['variance_explained'][-1]:.3f}")
print(f"  Final FVU: {checkpoint['fvu_values'][-1]:.4f}")
print("="*60)

#%%
