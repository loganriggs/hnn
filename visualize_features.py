"""
Visualize max-activating examples for bilinear transcoder features

Adapted from https://github.com/tdooms/bae (credit: Thomas Dooms)
This script finds and displays text examples that maximally activate specific
hidden dimensions in our bilinear transcoders.
"""

#%% Imports
import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset, Dataset
from utils.bilinear import Bilinear
from tqdm import tqdm
import os

#%% Configuration
LAYER_IDX = 0  # Which layer's transcoder to visualize
FEATURE_IDX = 100  # Which hidden dimension to visualize
MODEL_NAME = "EleutherAI/pythia-410m"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
N_BATCHES = 16  # How many batches to search through
BATCH_SIZE = 32  # Batch size for processing
MAX_LENGTH = 64  # Maximum sequence length
K_TOP = 5  # How many top examples to show

print(f"Using device: {DEVICE}")
print(f"Visualizing layer {LAYER_IDX}, feature {FEATURE_IDX}")

#%% Load model and tokenizer
print("\nLoading model and tokenizer...")
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token
model.to(DEVICE)
model.eval()

print(f"Model loaded: {MODEL_NAME}")

#%% Load transcoder
checkpoint_path = f"model_weights/transcoder_weights_l{LAYER_IDX}_bilinear_muon_3000b.pt"
print(f"\nLoading transcoder from: {checkpoint_path}")

transcoder, checkpoint = Bilinear.from_pretrained(checkpoint_path, device=DEVICE)
transcoder.eval()
transcoder.requires_grad_(False)

config = checkpoint['config']
print(f"Transcoder loaded: {config.n_inputs} → {config.n_hidden} → {config.n_outputs}")

#%% Load dataset
print("\nLoading dataset...")
dataset = load_dataset("monology/pile-uncopyrighted", split="train", streaming=True)

# Take a subset and convert to non-streaming dataset
samples = []
for i, example in enumerate(dataset):
    samples.append(example)
    if i >= N_BATCHES * BATCH_SIZE - 1:
        break

dataset = Dataset.from_list(samples)
print(f"Loaded {len(dataset)} samples")

#%% Helper functions for colored text output
def to_black_diverging_color(values):
    """
    Simple red-black-blue color map.
    Negative values -> blue, positive values -> red, zero -> black
    """
    r = torch.where(values < 0.5, 0.0, (values - 0.5) * 2.0)
    g = torch.zeros_like(values)
    b = torch.where(values > 0.5, 0.0, (0.5 - values) * 2.0)

    return (torch.stack([r, g, b], dim=-1) * 255).int()

def color_str(text, color):
    """Apply RGB color to text string using ANSI escape codes"""
    r, g, b = color
    # Clean up text - handle BPE tokens properly
    text = text.replace('Ġ', ' ')  # GPT-2 style space token
    text = text.replace('Ċ', '\n')  # Newline token
    text = text.replace('\n', ' ')  # Convert newlines to spaces for display
    # Don't add extra space - the Ġ tokens already have them
    return f"\033[48;2;{int(r)};{int(g)};{int(b)}m{text}\033[0m"

def color_line(tokens, colors, start, end):
    """Color a line of tokens with corresponding colors"""
    return "".join([color_str(tokens[i], colors[i]) for i in range(start, end)])

#%% Extract activations and find max-activating examples
print("\nFinding max-activating examples...")

# Storage for activations
all_activations = []  # Will store [n_samples, seq_len, n_hidden]
all_input_ids = []

# Hook to capture MLP input
mlp_inputs = []
def get_mlp_input_hook(module, input, output):
    mlp_inputs.append(output.detach().cpu())

# Register hook
target_layer = model.gpt_neox.layers[LAYER_IDX]
hook = target_layer.post_attention_layernorm.register_forward_hook(get_mlp_input_hook)

# Process batches
for batch_idx in tqdm(range(N_BATCHES), desc="Processing batches"):
    # Get batch
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

    # Get MLP input activations
    mlp_inputs.clear()
    with torch.no_grad():
        model(input_ids, attention_mask=attention_mask)

    mlp_input = mlp_inputs[0]  # [batch, seq, d_model]

    # Normalize (per-dimension, same as training)
    mlp_input_flat = mlp_input.reshape(-1, mlp_input.shape[-1])
    in_mean = mlp_input_flat.mean(0, keepdim=True)
    in_std = mlp_input_flat.std(0, keepdim=True) + 1e-6
    mlp_input_norm = (mlp_input_flat - in_mean) / in_std
    mlp_input_norm = mlp_input_norm.reshape(mlp_input.shape)

    # Run through transcoder to get hidden activations
    with torch.no_grad():
        # Get hidden activations: left(x) * right(x)
        hidden_acts = (
            transcoder.left(mlp_input_norm.to(DEVICE)) *
            transcoder.right(mlp_input_norm.to(DEVICE))
        ).cpu()

    all_activations.append(hidden_acts)
    all_input_ids.append(input_ids.cpu())

# Remove hook
hook.remove()

# Concatenate all activations
all_activations = torch.cat(all_activations, dim=0)  # [total_samples, seq_len, n_hidden]
all_input_ids = torch.cat(all_input_ids, dim=0)  # [total_samples, seq_len]

print(f"Collected activations shape: {all_activations.shape}")

#%% Find top-k max and min activating positions for the selected feature
print(f"\nFinding top-{K_TOP} activating examples for feature {FEATURE_IDX}...")

feature_acts = all_activations[:, :, FEATURE_IDX]  # [total_samples, seq_len]

# Find max and min per sequence
max_acts_per_seq, max_pos_per_seq = feature_acts.max(dim=1)  # [total_samples]
min_acts_per_seq, min_pos_per_seq = feature_acts.min(dim=1)  # [total_samples]

# Get top-k sequences with highest max activations
top_k_max_vals, top_k_max_seqs = max_acts_per_seq.topk(k=K_TOP)
top_k_min_vals, top_k_min_seqs = min_acts_per_seq.topk(k=K_TOP, largest=False)

print(f"\nTop-{K_TOP} maximum activations: {top_k_max_vals.tolist()}")
print(f"Top-{K_TOP} minimum activations: {top_k_min_vals.tolist()}")

#%% Visualize top activating examples
def visualize_example(seq_idx, feature_idx, largest=True, context_window=20):
    """
    Visualize a single example with colored tokens based on feature activation

    Args:
        seq_idx: Index of the sequence in the dataset
        feature_idx: Which feature to visualize
        largest: If True, find max activation; if False, find min
        context_window: How many tokens to show around the activating token
    """
    # Get activation values for this sequence
    acts = all_activations[seq_idx, :, feature_idx]  # [seq_len]

    # Find the token with max/min activation
    if largest:
        act_val, act_pos = acts.max(dim=0)
    else:
        act_val, act_pos = acts.min(dim=0)

    act_pos = act_pos.item()
    act_val = act_val.item()

    # Get tokens
    token_ids = all_input_ids[seq_idx]
    tokens = tokenizer.convert_ids_to_tokens(token_ids)

    # Normalize activations for coloring (around the activation peak)
    normalized = -(acts / acts.abs().max()) / 2.0 + 0.5
    colors = to_black_diverging_color(normalized)

    # Determine context window
    start = max(0, act_pos - context_window // 2)
    end = min(len(tokens), act_pos + context_window // 2)

    # Print with colors
    colored_text = color_line(tokens, colors, start, end)
    print(f"{act_val:>7.2f}:  {colored_text}")

#%% Display results
print("\n" + "="*80)
print(f"FEATURE {FEATURE_IDX} - TOP {K_TOP} MAX ACTIVATING EXAMPLES")
print("="*80)

for i, seq_idx in enumerate(top_k_max_seqs):
    visualize_example(seq_idx.item(), FEATURE_IDX, largest=True, context_window=30)

print("\n" + "="*80)
print(f"FEATURE {FEATURE_IDX} - TOP {K_TOP} MIN ACTIVATING EXAMPLES")
print("="*80)

for i, seq_idx in enumerate(top_k_min_seqs):
    visualize_example(seq_idx.item(), FEATURE_IDX, largest=False, context_window=30)

print("\n" + "="*80)

#%% Interactive: Change feature and re-visualize
# To explore different features, just change FEATURE_IDX above and re-run the visualization cells

#%%
